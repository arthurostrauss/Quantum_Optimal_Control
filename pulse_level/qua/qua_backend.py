from __future__ import annotations

from typing import Iterable, List

from qiskit.circuit import ParameterExpression
from qiskit.providers import BackendV2 as Backend
from qiskit.transpiler import Target

from qiskit import pulse
from qm.jobs.running_qm_job import RunningQmJob
from qm.qua import *
from qm import QuantumMachinesManager, Program
from qualang_tools.addons.variables import assign_variables_to_element
from quam.examples.superconducting_qubits import QuAM
from qualang_tools.video_mode import ParameterTable


class QMProvider:
    def __init__(self, host, port, cluster_name, octave_config):
        """
        Qiskit Provider for the Quantum Orchestration Platform (QOP)
        Args:
            host: The host of the QOP
            port: The port of the QOP
            cluster_name: The name of the cluster
            octave_config: The octave configuration
        """
        super().__init__(self)
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

    def run(self, run_input: Program, **options) -> RunningQmJob:
        """
        Run a program on the backend
        Args:
            run_input: The QUA program to run
            **options: The options for the run
        """
        return self.qm.execute(run_input)


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


def get_el_from_channel(channel: pulse.channels.Channel):
    return ""


def get_pulse_from_instruction(
    pulse_instance: pulse.library.Pulse,
    channel: pulse.channels.Channel,
    channel_mapping: dict = None,
    parameter_table: ParameterTable = None,
    pulse_lib: dict = None,
):
    param_statement = {"pulse": pulse_lib[pulse_instance.name]}

    for param_name, param_value in pulse_instance.parameters.items():
        if param_name == "amp" and isinstance(param_value, ParameterExpression):
            amp_ = parameter_table[param_value]
            angle = pulse_instance.parameters.get("angle", None)
            if isinstance(angle, ParameterExpression):
                angle = parameter_table[angle]
            elif angle == 0:
                angle = None

            matrix_elements = (
                [
                    amp_ * Math.cos(angle),
                    -amp_ * Math.sin(angle),
                    amp_ * Math.sin(angle),
                    amp_ * Math.cos(angle),
                ]
                if angle is not None
                else [amp_]
            )
            param_statement["pulse"] *= amp(*matrix_elements)

        elif param_name == "duration" and isinstance(param_value, ParameterExpression):
            param_statement["duration"] = parameter_table[param_value]

    return param_statement


def schedule_to_qua_instructions(
    sched: pulse.Schedule, backend: QMBackend, parameter_table: ParameterTable
):
    """
    Convert a Qiskit pulse schedule to a QUA program
    :param sched: The Qiskit pulse schedule
    :param backend: The QMBackend object
    """

    time_tracker = {channel: 0 for channel in sched.channels}
    for time, instruction in sched.instructions:
        channel = instruction.channels[0]
        if time_tracker[channel] < time:
            wait((time - time_tracker[channel]) // 4, get_el_from_channel(channel))
        time_tracker[channel] = time
        if isinstance(instruction, pulse.Play):
            play(
                **get_pulse_from_instruction(instruction.pulse, channel),
                element=get_el_from_channel(channel),
            )
        elif isinstance(instruction, pulse.ShiftPhase):
            frame_rotation(instruction.phase, get_el_from_channel(channel))
        elif isinstance(instruction, pulse.ShiftFrequency):
            update_frequency(get_el_from_channel(channel), instruction.frequency)
        elif isinstance(instruction, pulse.Delay):
            wait(instruction.duration // 4, get_el_from_channel(channel))
        else:
            raise ValueError(f"Unknown instruction {instruction}")
