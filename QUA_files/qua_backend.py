from __future__ import annotations

from typing import Iterable, List

from qiskit.providers import BackendV2 as Backend
from qiskit.providers import Provider
from qiskit.providers.models import PulseBackendConfiguration
from qiskit.transpiler import Target, InstructionProperties, InstructionDurations

from qiskit import schedule as build_schedule, QuantumCircuit
from qiskit import pulse
from qm.qua import *
from qm import QuantumMachinesManager, QuantumMachine, QmJob, Program
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.config import QuaConfig
from qiskit.compiler import schedule as build_schedule
from quam.components import *
from quam.examples.superconducting_qubits import QuAM, Transmon
from QUA_files.qua_utils import (
    prepare_input_state,
    rand_gauss_moller_box,
    schedule_to_qua_instructions,
)


class QMProvider(Provider):
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

    def get_backend(self, quam: QuAM):
        return QMBackend(self, self.qmm, quam)

    def backends(self, name=None, filters=None, **kwargs):
        raise NotImplementedError("Not implemented yet")

    def __str__(self):
        pass

    def __repr__(self):
        pass


class QMBackend(Backend):
    def __init__(
        self,
        provider: QMProvider,
        qmm: QuantumMachinesManager,
        quam: QuAM,
    ):
        Backend.__init__(self, provider=provider, name="QUA backend")
        self._target = Target(
            description="QUA target",
            dt=1e-9,
            granularity=4,
            num_qubits=len(quam.qubits),
        )
        self.quam = quam
        self.populate_target(quam)
        configuration = quam.generate_config()
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

    def populate_target(self, quam: QuAM):
        """
        Populate the target instructions with the QOP configuration (currently hardcoded)

        """
        pass

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


def rl_qoc_qua_prog(q_env, qc: QuantumCircuit):
    """
    Generate a QUA program tailormade for the RL based calibration project
    """
    if not isinstance(q_env.backend, QMBackend) or q_env.config_type != "qua":
        raise ValueError("The backend should be a QMBackend object")
    trunc_index = q_env.trunc_index
    n_actions = q_env.action_space.shape[-1]
    real_time_parameters = q_env.parameter_table[trunc_index]
    sched_qc = build_schedule(q_env.circuit_truncations[trunc_index], q_env.backend)
    with program() as rl_qoc:
        # Declare necessary variables
        batchsize = declare(int)
        n_shots = declare(int)
        I, I_st, Q, Q_st = qua_declaration(q_env.n_qubits)
        input_state_indices = declare_input_stream(
            int, name="input_state_indices", size=q_env.n_qubits
        )
        input_state_index = declare(int)
        observables_indices = declare_input_stream(
            int, name="observables_indices", size=q_env.n_qubits
        )
        mean_action = declare_input_stream(fixed, name="mean_action", size=n_actions)
        std_action = declare_input_stream(fixed, name="std_action", size=n_actions)
        # param_vars is a Python list of QUA fixed variables
        param_vars = (
            real_time_parameters.declare_variables()
        )  # TODO: Pause is currently happening here
        z1, z2 = declare(fixed, size=n_actions), declare(fixed, size=n_actions)
        batch_r = Random()
        batch_r.set_seed(12321)
        qua_i = declare(int, value=0)
        # Infinite loop to run the training

        with infinite_loop_():
            advance_input_stream(mean_action)
            advance_input_stream(
                std_action
            )  # Load the mean and std of the action (from agent)
            advance_input_stream(
                observables_indices
            )  # Load info about the observables (estimator like)
            advance_input_stream(
                input_state_indices
            )  # Load info about the input states to prepare
            with for_(batchsize, 0, batchsize < q_env.batch_size, batchsize + 1):
                for i in range(n_actions):
                    z1[i], z2[i] = rand_gauss_moller_box(
                        z1[i], z2[i], mean_action[i], std_action[i], batch_r
                    )
                    assign(param_vars[i], z1[i])
                # Sample the input state and the observable
            with for_each_(input_state_index, input_state_indices):
                prepare_input_state(input_state_index, qua_i, q_env.qubit_elements)
                assign(qua_i, qua_i + 1)
            assign(qua_i, 0)

    return rl_qoc


def qua_declaration(n_qubits, readout_elements):
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
