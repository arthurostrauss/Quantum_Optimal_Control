from time import strftime, gmtime
from typing import Literal

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping as gate_map,
)
from qiskit.exceptions import QiskitError
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.quantum_info import DensityMatrix
import qibo
from qibo import Circuit, gates, set_backend
from qibo import Circuit as QiboCircuit
import numpy as np
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import DensityMatrix

from rl_qoc import QuantumEnvironment
from rl_qoc.qibo.utils import resolve_gate_rule

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
        set_backend("qibolab", platform=backend_config.platform)
        target = backend_config.physical_qubits[0]
        if self.config.check_on_exp:
            if backend_config.config_type != "qibo":
                raise TypeError("Backend config is not qibo config")
            else:
                path = "rlqoc_bis" + strftime("%d_%b_%H_%M_%S", gmtime())
                with Executor.open(
                    "myexec",
                    path=path,
                    platform=backend_config.platform,
                    targets=[backend_config.physical_qubits[0]],
                    update=True,
                    force=True,
                ) as e:
                    # Overwrite the rules
                    backend = qibo.get_backend()
                    compiler = backend.compiler
                    gate_rule = resolve_gate_rule(self.config.backend_config.gate_rule)
                    rule = lambda qubits_ids, platform, gate_params: gate_rule[1](
                        qubits_ids, platform, gate_params, params
                    )
                    compiler.register(gate_rule[0])(rule)

                    qiskit_baseline_circ = from_custom_to_baseline_circuit(qc)
                    qibo_circ = QiboCircuit.from_qasm(qasm3_dumps(qiskit_baseline_circ))
                    qibo_circ.draw()
                    tomography_output = e.state_tomography(
                        circuit=qibo_circ,
                        nshots=5000,
                    )
                    rho_real = tomography_output.results.measured_density_matrix_real[
                        target
                    ]
                    rho_imaginary = (
                        tomography_output.results.measured_density_matrix_imag[target]
                    )
                    rho = np.transpose(
                        np.array(rho_real) + 1j * np.array(rho_imaginary)
                    )
                    dm = DensityMatrix(rho)
                    fidelity = self.target.fidelity(dm, validate=False)
                report(e.path, e.history)
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
