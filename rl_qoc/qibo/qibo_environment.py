from qiskit.circuit import QuantumCircuit
import numpy as np
from qibocal.auto.execute import Executor
from qibocal.cli.report import report

from rl_qoc import QuantumEnvironment

class QiboEnvironment(QuantumEnvironment):

    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array ) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: List of Action vectors to execute on quantum system
        :return: None
        """
        backend_config = self.config.backend_config
        if self.config.check_on_exp:
            if backend_config.config_type != "qibo":
                raise TypeError("Backend config is not qibo config")
            else:
                with Executor.open(
                    "myexec",
                    # path=executor_path,
                    platform=backend_config.platform,
                    targets=[backend_config.physical_qubit],
                    update=True,
                    force=True,
                ) as e:
                    e.platform.qubits[target].native_gates.RX.amplitude = self.mean_action[0]
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

                    pars = rb_output.results.pars.get(target)
                    one_minus_p = 1 - pars[2]
                    r_c = one_minus_p * (1 - 1 / 2**1)
                    fidelity = r_c / AVG_GATE
       
            self.circuit_fidelity_history.append(fidelity)
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAA",fidelity)
            return fidelity
        return 0

