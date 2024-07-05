from __future__ import annotations

from typing import Iterable, List, Sequence, Dict, Union, Tuple

from quam.components import Channel as QuAMChannel
from quam.components.pulses import Pulse as QuAMPulse
from quam import quam_dataclass

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.providers import BackendV2 as Backend, QubitProperties
from qiskit.pulse import (
    ScheduleBlock,
    Schedule,
    ControlChannel,
    DriveChannel,
    MeasureChannel,
    AcquireChannel,
    Play,
    ShiftPhase,
    ShiftFrequency,
    Delay,
    UnassignedDurationError,
    PulseError,
)
from qiskit.transpiler import Target
from qiskit.pulse.transforms import block_to_schedule
from qiskit.pulse.channels import Channel as QiskitChannel
from qiskit.pulse.library.pulse import Pulse as QiskitPulse
from qiskit.pulse.library.waveform import Waveform
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.pending_job import QmPendingJob, QmJob
from qm.qua import *
from qm import QuantumMachinesManager, Program
from qualang_tools.addons.variables import assign_variables_to_element
from quam_components.quam import QuAM
from qualang_tools.video_mode import ParameterTable
from oqc import Compiler, HardwareConfig, OperationIdentifier, OperationsMapping

_real_time_parameters = {
    "amp",
    "angle",
    "frequency",
}  # Parameters that can be used in real-time


# TODO: Add duration to the list of real-time parameters (need ScheduleBlock to QUA compiler)
class FluxChannel(QiskitChannel):
    prefix = "f"


@quam_dataclass
class QuAM_from_QiskitPulse(QuAMPulse):
    def __init__(self, pulse: QiskitPulse):
        self.pulse: QiskitPulse = pulse
        super().__init__(
            length=self.pulse.duration if not self.pulse.is_parameterized() else 0,
            id=pulse.name,
        )

    def waveform_function(
        self,
    ) -> Union[
        float,
        complex,
        List[float],
        List[complex],
        Tuple[float, float],
        Tuple[List[float], List[float]],
    ]:
        if isinstance(self.pulse, Waveform):
            return self.pulse.samples
        else:
            try:
                return self.pulse.get_waveform().samples
            except (AttributeError, PulseError) as e:
                raise PulseError(
                    "Pulse waveform could not be retrieved from the given pulse"
                ) from e

    def is_parametrized(self):
        return self.pulse.is_parametrized()


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
        self._pulse_to_quam_channels: Dict[QiskitChannel, QuAMChannel] = {}
        self._quam_to_pulse_channels: Dict[QuAMChannel, QiskitChannel] = {}
        self._operation_mapping_QUA: OperationsMapping = {}
        self.populate_target(quam)
        self.qua_config = quam.generate_config()

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
        Populate the target instructions with the QOP configuration (currently hardcoded for
        Transmon based QuAM architecture)

        """
        for i, qubit in enumerate(quam.qubits.values()):
            self._target.qubit_properties[i] = QubitProperties(
                t1=qubit.T1, t2=qubit.T2ramsey, frequency=qubit.f_01
            )
            self._quam_to_pulse_channels[qubit.xy] = DriveChannel(i)
            self._pulse_to_quam_channels[DriveChannel(i)] = qubit.xy
            self._quam_to_pulse_channels[qubit.z] = FluxChannel(i)
            self._pulse_to_quam_channels[FluxChannel(i)] = qubit.z
            self._quam_to_pulse_channels[qubit.resonator] = MeasureChannel(i)
            self._pulse_to_quam_channels[MeasureChannel(i)] = qubit.resonator
            self._pulse_to_quam_channels[AcquireChannel(i)] = qubit.resonator
            self._quam_to_pulse_channels[qubit.resonator] = AcquireChannel(i)
            # TODO: Add the rest of the channels for QubitPairs (ControlChannels)

            self._target.add_instruction()
            # TODO: Update the instructions both in Qiskit and in the OQC operations mapping
            # TODO: Figure out if pulse calibrations should be added to Target

        self._coupling_map = self._target.build_coupling_map()

    def get_quam_channel(self, channel: QiskitChannel):
        """
        Convert a Qiskit Pulse channel to a QuAM channel

        Args:
            channel: The Qiskit Pulse Channel to convert

        Returns:
            The corresponding QuAM channel
        """
        return self._pulse_to_quam_channels[channel]

    def get_pulse_channel(self, channel: QuAMChannel):
        """
        Convert a QuAM channel to a Qiskit Pulse channel

        Args:
            channel: The QuAM channel to convert

        Returns:
            The corresponding pulse channel
        """
        return self._quam_to_pulse_channels[channel]

    def meas_map(self) -> List[List[int]]:
        return self._target.concurrent_measurements

    def drive_channel(self, qubit: int):
        """
        Get the drive channel for a given qubit (should be mapped to a quantum element in configuration)
        """
        return DriveChannel(qubit)

    def control_channel(self, qubits: Iterable[int]):
        pass

    def measure_channel(self, qubit: int):
        return MeasureChannel(qubit)

    def acquire_channel(self, qubit: int):
        return AcquireChannel(qubit)

    def flux_channel(self, qubit: int):
        return FluxChannel(qubit)

    def run(self, run_input, **options):
        """
        Run a QuantumCircuit on the QOP backend
        Args:
            run_input: The QuantumCircuit to run
            options: The options for the run
        """
        if isinstance(run_input, Sequence):
            qua_progs = []
            for qc in run_input:
                qua_prog = self.qua_prog_from_qc(qc)
                program_id = self.qm.compile(qua_prog)
                qua_progs.append(program_id)

    def schedule_to_qua_macro(self, sched: Schedule):

        def qua_macro(*params):
            param_counter = 0
            time_tracker = {channel: 0 for channel in sched.channels}
            for time, instruction in sched.instructions:
                try:
                    qiskit_channel = instruction.channel
                except AttributeError:
                    raise AttributeError(
                        "Provided instruction not compatible with QUA conversion"
                    )
                if qiskit_channel.is_parametrized():
                    # TODO: Implement parametrized channels
                    raise NotImplementedError(
                        "Parametrized channels are not supported yet"
                    )
                quam_channel = self.get_quam_channel(qiskit_channel)

                if time_tracker[qiskit_channel] < time:
                    quam_channel.wait((time - time_tracker[qiskit_channel]))

                if isinstance(instruction, Play):
                    param_counter = self.qiskit_to_qua_play(
                        quam_channel, instruction, params, param_counter
                    )
                elif isinstance(instruction, ShiftPhase):
                    if instruction.is_parameterized():
                        phase = params[param_counter]
                        param_counter += 1
                    else:
                        phase = instruction.phase
                    quam_channel.frame_rotation(phase)

                elif isinstance(instruction, ShiftFrequency):
                    if instruction.is_parameterized():
                        freq = params[param_counter]
                        param_counter += 1
                    else:
                        freq = instruction.frequency
                    quam_channel.update_frequency(freq)

                elif isinstance(instruction, Delay):
                    quam_channel.wait(instruction.duration)
                else:
                    raise ValueError(f"Unknown instruction {instruction}")

        return qua_macro

    def quantum_circuit_to_qua(self, qc: QuantumCircuit):
        """
        Convert a QuantumCircuit to a QUA program
        """

        basis_gates = self.operation_names.copy()

        if qc.calibrations:
            for gate_name, cal_info in qc.calibrations.items():
                if gate_name not in basis_gates:
                    basis_gates.append(gate_name)
                for (qubits, parameters), schedule in cal_info.items():
                    parametrized_channels_count = 0
                    for channel in schedule.channels:
                        if channel.is_parameterized():
                            parametrized_channels_count += 1
                    if parametrized_channels_count == 0:
                        parametrized_channels_count = None

                    self._operation_mapping_QUA[
                        OperationIdentifier(
                            gate_name,
                            len(parameters),
                            qubits,
                            parametrized_channels_count,
                        )
                    ] = self.schedule_to_qua_macro(schedule)

    def qua_prog_from_qc(self, qc: QuantumCircuit | Schedule | ScheduleBlock | Program):
        """
        Convert given input into a QUA program
        """
        if isinstance(qc, Program):
            return qc
        elif isinstance(qc, QuantumCircuit):
            return self.quantum_circuit_to_qua(qc)
        elif isinstance(qc, ScheduleBlock):  # Convert to Schedule first
            try:
                schedule = block_to_schedule(qc)
            except (UnassignedDurationError, PulseError) as e:
                raise RuntimeError(
                    "ScheduleBlock could not be converted to Schedule (required"
                    "for converting it to QUA program"
                ) from e

            return self.schedule_to_qua_macro(schedule)
        elif isinstance(qc, Schedule):
            return self.schedule_to_qua_macro(qc)
        else:
            raise ValueError(f"Unsupported input {qc}")

    def qiskit_to_qua_play(self, quam_channel, instruction, params, param_counter):
        """
        Convert a Qiskit Play instruction to a QUA Play instruction
        """
        return param_counter


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


def get_el_from_channel(channel: QiskitChannel):
    return ""


def get_pulse_from_instruction(
    pulse_instance: QiskitPulse,
    channel: QiskitChannel,
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
