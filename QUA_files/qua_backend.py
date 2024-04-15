from __future__ import annotations

from qiskit.providers import BackendV2 as Backend
from qiskit.providers import Provider
from qiskit.providers.models import PulseBackendConfiguration
from qiskit.transpiler import Target

from qiskit import schedule as build_schedule, QuantumCircuit
from qiskit import pulse
from qm.qua import *
from qm import QuantumMachinesManager, QuantumMachine
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.config import QuaConfig
from qiskit.compiler import schedule as build_schedule
from quantumenvironment import QuantumEnvironment


class QuaProvider(Provider):
    def __init__(self, host, port, cluster_name, octave_config):
        """
        Qiskit Provider for the Quantum Orchestration Platform (QOP)
        Args:
            host: The host of the QOP
            port: The port of the QOP
            cluster_name: The name of the cluster
            octave_config: The octave configuration
        """
        self.qmm = QuantumMachinesManager(
            host=host, port=port, cluster_name=cluster_name, octave=octave_config
        )

    def get_backend(self, configuration: Dict | QuaConfig):
        return QuaBackend(self, self.qmm, configuration)

    def backends(self, name=None, filters=None, **kwargs):
        raise NotImplementedError("Not implemented yet")

    def __str__(self):
        pass

    def __repr__(self):
        pass


class QuaBackend(Backend):
    def __init__(
        self,
        provider: QuaProvider,
        qmm: QuantumMachinesManager,
        configuration: Dict | QuaConfig,
    ):
        Backend.__init__(self, provider=provider, name="QUA backend")
        self._target = Target(
            description="QUA target", dt=1e-9, granularity=4, num_qubits=5
        )
        self.populate_target()

        self.qm = qmm.open_qm(configuration)

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        pass

    def populate_target(self):
        """
        Populate the target instructions with the QOP configuration (currently hardcoded)
        """

    def meas_map(self) -> List[List[int]]:
        return self._target.concurrent_measurements

    def drive_channel(self, qubit: int):
        """
        Get the drive channel for a given qubit (should be mapped to a quantum element in configuration)
        """
        return pulse.DriveChannel(qubit)

    def control_channel(self, qubits: Iterable[int]):
        return pulse.ControlChannel(qubits[0])

    def measure_channel(self, qubit: int):
        pass

    def acquire_channel(self, qubit: int):
        pass

    def readout_elements(self):
        return [f"qubit{i}$rr" for i in range(self.num_qubits)]

    def run(self, run_input: Program, **options):
        """
        Run a program on the backend
        Args:
            run_input: The QUA program to run
            **options: The options for the run
        """
        self.qm.execute(run_input)


def rl_qoc_qua_prog(
    qc: QuantumCircuit, qua_backend: QuaBackend, q_env: QuantumEnvironment
):
    """
    Generate a QUA program tailormade for the RL based calibration project
    """
    with program() as rl_qoc:
        # Declare necessary variables
        batchsize = declare(int, value=q_env.batch_size)
        n_shots = declare(int, value=q_env.n_shots)
        input_state_indices = declare(int, size=q_env.n_qubits)
        observables_indices = declare(int, size=q_env.n_qubits)
        I, I_st, Q, Q_st = qua_declaration(q_env.n_qubits)
        params = q_env.parameter_table.declare_variables()
        done = declare(bool, value=False)

    return rl_qoc


def qua_declaration(n_qubits):
    """
    Macro to declare the necessary QUA variables

    :param n_qubits: Number of qubits used in this experiment
    :return:
    """
    I, Q = [[declare(fixed) for _ in range(n_qubits)] for _ in range(2)]
    I_st, Q_st = [[declare_stream() for _ in range(n_qubits)] for _ in range(2)]
    # Workaround to manually assign the results variables to the readout elements
    for i in range(n_qubits):
        assign_variables_to_element(readout_elements[i], I[i], Q[i])
    return I, I_st, Q, Q_st
