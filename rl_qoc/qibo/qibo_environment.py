from time import strftime, gmtime
from typing import Literal

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping as gate_map,
)
from qiskit.exceptions import QiskitError
from qiskit.qasm3 import dumps as qasm3_dumps
from qibo import Circuit as QiboCircuit
import numpy as np
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import DensityMatrix

from rl_qoc import QuantumEnvironment

AVG_GATE = 1.875


class QiboEnvironment(QuantumEnvironment):

    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: List of Action vectors to execute on quantum system
        :return: None
        """
        benchmarking_method: Literal["tomography", "rb"] = (
            self.config.benchmark_config.method
        )
        backend_config = self.config.backend_config
        target = backend_config.physical_qubits[0]
        if self.config.check_on_exp:
            if backend_config.config_type != "qibo":
                raise TypeError("Backend config is not qibo config")
            else:
                path = "rlqoc" + strftime("%d_%b_%H_%M_%S", gmtime())
                with Executor.open(
                    "myexec",
                    path=path,
                    platform=backend_config.platform,
                    targets=[target],
                    update=False,
                    force=True,
                ) as e:
                    e.platform.qubits[target].native_gates.RX.amplitude = float(
                        self.mean_action[0]
                    )
                    if benchmarking_method == "rb":
                        rb_output = e.rb_ondevice(
                            num_of_sequences=1000,
                            max_circuit_depth=1000,
                            delta_clifford=10,
                            n_avg=1,
                            save_sequences=False,
                            apply_inverse=True,
                        )

                        # Calculate infidelity and error
                        # stdevs = np.sqrt(np.diag(np.reshape(rb_output.results.cov[target], (3, 3))))

                        pars = rb_output.results.pars[target]
                        one_minus_p = 1 - pars[2]
                        r_c = one_minus_p * (1 - 1 / 2**1)
                        fidelity = 1 - (r_c / AVG_GATE)
                    elif benchmarking_method == "tomography":
                        qiskit_baseline_circ = from_custom_to_baseline_circuit(qc)
                        qibo_circ = QiboCircuit.from_qasm(
                            qasm3_dumps(qiskit_baseline_circ)
                        )
                        #TODO: test following lines
                        tomography_output = e.state_tomography(
                                circuit = qibo_circ,
                        )
                        rho_real = tomography_output.results.target_density_matrix_real[target]
                        rho_imaginary = tomography_output.results.target_density_matrix_imag[target]
                        rho = np.array(rho_real) + 1j * np.array(rho_imaginary)


                        fidelity = self.target.fidelity(rho, validate=False)
                        # dm = DensityMatrix(rho)
                        # fidelity = self.target.fidelity(dm, validate=False)
                    else:
                        raise ValueError("Benchmarking method not recognized")

            self.circuit_fidelity_history.append(fidelity)
            return fidelity
        return 0


def from_custom_to_baseline_circuit(circ: QuantumCircuit):
    """
    Convert custom circuit to baseline circuit
    :param circ: Custom circuit
    :return: Baseline circuit
    """
    dag = circuit_to_dag(circ)
    op_nodes = dag.op_nodes()
    for node in op_nodes:
        if node.name not in gate_map():
            gate_name: str = node.name.split("_")[0]
            try:
                gate = gate_map()[gate_name.lower()]
            except KeyError:
                raise QiskitError(
                    f"Cannot bind the circuit to the backend because the gate "
                    f"{gate_name} is not found in the standard gate set."
                )
            qc = QuantumCircuit(list(node.qargs))
            qc.append(gate, node.qargs)
            dag.substitute_node_with_dag(node, circuit_to_dag(qc))
    return dag_to_circuit(dag)
