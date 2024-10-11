from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from functools import partial
from typing import Iterable, List, Sequence, Dict, Union, Tuple, Optional, Callable

from qualang_tools.video_mode.videomode import ParameterValue
from quam.components import Channel as QuAMChannel
from quam.components.pulses import Pulse as QuAMPulse
from quam import quam_dataclass

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.providers import BackendV2 as Backend, QubitProperties
from qiskit.pulse import (
    ScheduleBlock,
    Schedule,
    DriveChannel,
    MeasureChannel,
    AcquireChannel,
    Play,
    ShiftPhase,
    ShiftFrequency,
    SetFrequency,
    SetPhase,
    Delay,
    UnassignedDurationError,
    PulseError,
    Instruction,
    Acquire,
)
from qiskit.pulse.library import Pulse as QiskitPulse
from qiskit.transpiler import Target, InstructionProperties
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.pulse.transforms import block_to_schedule
from qiskit.pulse.channels import Channel as QiskitChannel
from qiskit.pulse.library import SymbolicPulse
from qiskit.pulse.library.waveform import Waveform
from qm.qua import amp as qua_amp, reset_phase, QuaVariableType, assign, switch_, case_
from qm.qua import Math, declare, fixed, declare_stream
from qm import QuantumMachinesManager, Program
from qualang_tools.addons.variables import assign_variables_to_element
from quam_components import QuAM
from qualang_tools.video_mode import ParameterTable
from sympy_to_qua import sympy_to_qua
from oqc import (
    Compiler,
    HardwareConfig,
    OperationIdentifier,
    OperationsMapping,
    QubitsMapping,
)

# TODO: Add duration to the list of real-time parameters (need ScheduleBlock to QUA compiler)
_real_time_parameters = {
    "amp",
    "angle",
    "frequency",
    "duration",
    "phase"
}  # Parameters that can be used in real-time
_ref_amp = 0.1
_ref_phase = 0.0

_qiskit_to_qua_instructions: Dict[Instruction, Dict[str, Callable | List[str]]] = {
    ShiftPhase: {
        "macro": lambda channel, phase: channel.frame_rotation(phase),
        "params": ["phase"],
    },
    ShiftFrequency: {
        "macro": lambda channel, freq: channel.update_frequency(
            freq + channel.intermediate_frequency
        ),
        "params": ["frequency"],
    },
    SetFrequency: {
        "macro": lambda channel, freq: channel.update_frequency(freq),
        "params": ["frequency"],
    },
    SetPhase: {
        "macro": lambda channel, phase: (
            reset_phase(channel.name),
            channel.frame_rotation(phase),
        ),
        "params": ["phase"],
    },
    Delay: {
        "macro": lambda channel, duration: channel.wait(duration),
        "params": ["duration"],
    },
    Play: {
        "macro": lambda pulse, channel,
                        amp=None, angle=None,
                        duration=None: channel.play(pulse.name, amplitude_scale=get_amp_matrix(amp, angle),
                                                    duration=duration),
        "params": ["pulse", "duration", "amp", "angle"]
    }
}


def get_real_time_pulse_parameters(pulse: QiskitPulse):
    """
    Get the real-time parameters of a Qiskit Pulse
    """
    real_time_params = {}
    for param in _real_time_parameters:
        if hasattr(pulse, param) and isinstance(getattr(pulse, param), ParameterExpression):
            real_time_params[param] = getattr(pulse, param)


def _handle_parameterized_instruction(
        instruction: Instruction,
        param_table: ParameterTable,
        param_mapping: Dict[str, QuaVariableType],
):
    """
    Handle the conversion of a parameterized instruction to QUA by creating a dictionary of parameter values
    and assigning them to the corresponding QUA variables
    """
    value_dict = {}
    involved_parameter_values = {}  # Store ParameterValue objects for involved parameters
    for param in instruction.parameters:
        if param.name not in param_table:
            raise ValueError(f"Parameter {param.name} is not in the parameter table")
        if param.name not in param_mapping:
            raise ValueError(
                f"Parameter {param.name} is not in the provided parameters"
            )
        # Assign the QUA variable to the parameter
        assign(param_table[param.name], param_mapping[param.name])
        involved_parameter_values[param.name] = param_table.table[param.name]

    for attribute in _qiskit_to_qua_instructions[type(instruction)]["params"]:
        if isinstance(getattr(instruction, attribute), ParameterExpression):
            value_dict[attribute] = sympy_to_qua(
                getattr(instruction, attribute).sympify(), involved_parameter_values
            )
        elif attribute == "pulse":
            pulse = getattr(instruction, attribute)
            for pulse_param_name, pulse_param in get_real_time_pulse_parameters(pulse).items():
                value_dict[pulse_param_name] = sympy_to_qua(
                    pulse_param.sympify(),
                    involved_parameter_values,
                )
            break
        else:  # TODO: Check if this is necessary
            value_dict[attribute] = getattr(instruction, attribute)  # Assign the value of the attribute

    return value_dict


def register_pulse_features(pulse: SymbolicPulse):
    """
    Register the features of a SymbolicPulse to be used in the QUA compiler
    """
    stored_parameter_expressions = {}
    for param_name, param_value in pulse.parameters.items():
        for feature_name in ["amp", "angle"]:
            if param_name == feature_name and isinstance(
                    param_value, ParameterExpression
            ):
                stored_parameter_expressions[feature_name] = lambda val: sympy_to_qua(
                    param_value.sympify(), val
                )
    return stored_parameter_expressions


class FluxChannel(QiskitChannel):
    prefix = "f"


@quam_dataclass
class QuAMQiskitPulse(QuAMPulse):
    def __init__(self, pulse: SymbolicPulse | Waveform):
        self.pulse: SymbolicPulse | Waveform = pulse
        self._stored_parameter_expressions = {
            "amp": None,
            "angle": None,
            "frequency": None,
        }

        self._stored_parameter_expressions = register_pulse_features(pulse)
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
            return self.pulse.samples.tolist()
        elif isinstance(self.pulse, SymbolicPulse):
            if (
                    self.pulse.is_parametrized()
            ):  # Assign reference values to the parameters
                sched = Schedule(Play(self.pulse, DriveChannel(0)))
                for param_name, param_value in self.pulse.parameters.items():
                    assignment_real_time_params = {}
                    for feature_name, feature_value in zip(
                            ["amp", "angle"], [_ref_amp, _ref_phase]
                    ):
                        if param_name == feature_name and isinstance(
                                param_value, ParameterExpression
                        ):
                            assignment_real_time_params[param_name] = feature_value
                    sched = sched.assign_parameters(assignment_real_time_params)

                    if param_name == "duration":
                        raise NotImplementedError(
                            "Duration parameter cannot be parametrized"
                        )

            try:
                return self.pulse.get_waveform().samples.tolist()
            except (AttributeError, PulseError) as e:
                raise PulseError(
                    "Pulse waveform could not be retrieved from the given pulse"
                ) from e

    def is_parametrized(self):
        return self.pulse.is_parametrized()

    def is_compile_time_parametrized(self):
        """
        Check if the pulse is parametrized with compile-time parameters
        """
        return any(
            [
                isinstance(self.pulse.parameters[param], ParameterExpression)
                and param not in _real_time_parameters
                for param in self.pulse.parameters
            ]
        )

    def is_real_time_parametrized(self):
        """
        Check if the pulse is parametrized with real-time parameters
        """
        return any(
            isinstance(self.pulse.parameters[param], ParameterExpression)
            for param in _real_time_parameters
        )

    @property
    def parameters(self):
        return self.pulse.parameters


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

    def get_backend(self, machine: QuAM):
        return QMBackend(machine)

    def backends(self, name=None, filters=None, **kwargs):
        raise NotImplementedError("Not implemented yet")

    def __str__(self):
        pass

    def __repr__(self):
        pass


def frame_rotation_matrix(angle, amp=1.0):
    return [
        amp * Math.cos(angle),
        -amp * Math.sin(angle),
        amp * Math.sin(angle),
        amp * Math.cos(angle),
    ]


def get_amp_matrix(amp=None, angle=None):
    new_ref = (
            1.0 / _ref_amp
    )  # Recover the reference amplitude from the reference value of 0.1
    if amp is not None and angle is not None:
        amp_matrix = qua_amp(*frame_rotation_matrix(angle, amp * new_ref))
    elif amp is not None:
        amp_matrix = qua_amp(new_ref * amp)
    elif angle is not None:
        amp_matrix = qua_amp(*frame_rotation_matrix(angle))
    else:
        amp_matrix = None
    return amp_matrix


def validate_pulse(pulse: QiskitPulse, channel: QuAMChannel) -> QiskitPulse:
    """
    Validate the pulse on the QuAM channel
    """
    if not pulse.name in channel.operations:
        raise ValueError(f"Pulse {pulse.name} is not in the operations of the QuAM channel")

    return pulse


def validate_instruction(instruction: Instruction) -> Callable:
    """
    Validate the instruction before converting it to QUA and return the corresponding QUA macro
    """
    if type(instruction) in _qiskit_to_qua_instructions:
        return _qiskit_to_qua_instructions[type(instruction)]["macro"]
    elif isinstance(instruction, Acquire):
        raise NotImplementedError("Acquire instructions are not supported yet")
    else:
        raise ValueError(
            f"instruction {instruction} not supported on the QUA backend"
        )


def validate_parameters(params, param_table):
    """
    Validate the parameters of the instruction
    """
    for param in params:
        if param not in param_table:
            raise ValueError(f"Parameter {param} is not in the parameter table")


def _instruction_to_qua(
        instruction: Instruction,
        quam_channel: QuAMChannel,
        param_mapping: Optional[Dict] = None,
        param_table: Optional[ParameterTable] = None,
):
    """
    Convert a Qiskit Pulse instruction to a QUA instruction

    Args:
        instruction: Qiskit Pulse instruction to convert
        quam_channel: QuAM channel on which to apply the instruction
        param_mapping:  Parameters to use for the conversion of parametrized instructions
        param_table:  Parameter table to use for the conversion (contains the reference QUA variables for parameters)
    """

    action = validate_instruction(instruction)

    if isinstance(instruction, Play):
        pulse = validate_pulse(instruction.pulse, quam_channel)
        action = partial(action, pulse=pulse)

    if instruction.is_parameterized():
        assert param_mapping is not None, "Parameters must be provided"
        assert param_table is not None, "Parameter table must be provided"
        values = _handle_parameterized_instruction(
            instruction, param_table, param_mapping
        )

    else:
        values = {}
        for attribute in _qiskit_to_qua_instructions[type(instruction)]["params"]:
            values[attribute] = getattr(instruction, attribute)

    action(quam_channel, **values)


def validate_schedule(schedule: Schedule | ScheduleBlock) -> Schedule:
    if isinstance(schedule, ScheduleBlock):
        if not schedule.is_schedulable():
            raise ValueError(
                "ScheduleBlock is not schedulable (contains unschedulable instructions)"
            )

        schedule = block_to_schedule(schedule)
    if not isinstance(schedule, Schedule):
        raise ValueError("Only Schedule objects are supported")

    return schedule


def handle_parametrized_channel(schedule: Schedule, param_table: ParameterTable) -> ParameterTable:
    """
    Modify type of parameters (-> int) in the Table that refer to channel parameters (they refer to integers)
    """
    for channel in schedule.channels:  # Check if channels are parametrized to change param type (->int)
        if channel.is_parameterized():
            ch_params = list(channel.parameters)
            if len(ch_params) > 1:
                raise NotImplementedError(
                    "Only single parameterized channels are supported"
                )
            ch_param = ch_params[0]
            ch_parameter_value = ParameterValue(ch_param.name, 0,
                                                param_table.table[ch_param.name].index,
                                                int)
            param_table.table[ch_param.name] = ch_parameter_value
    return param_table


class QMBackend(Backend, ABC):
    def __init__(
            self,
            machine: QuAM,
    ):
        Backend.__init__(self, name="QUA backend")
        self.machine = machine

        self._pulse_to_quam_channels: Dict[QiskitChannel, QuAMChannel] = {}
        self._quam_to_pulse_channels: Dict[
            QuAMChannel, QiskitChannel | Sequence[QiskitChannel]
        ] = {}
        self._operation_mapping_QUA: OperationsMapping = {}
        self._target = self._populate_target(machine)
        self._oq3_basis_gates = self.target.operation_names.copy()

    @property
    def target(self):
        return self._target

    @abstractmethod
    @property
    def qubit_mapping(self) -> QubitsMapping:
        """
        Build the qubit to quantum elements mapping for the backend.
        Should be of the form {qubit_index: (quantum_element1, quantum_element2, ...)}
        """
        pass

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        pass

    @abstractmethod
    def _populate_target(self, machine: QuAM) -> Target:
        """
        Populate the target instructions with the QOP configuration (currently hardcoded for
        Transmon based QuAM architecture)

        """
        pass

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
                qua_progs.append(qua_prog)

    def schedule_to_qua_macro(
            self, sched: Schedule, param_table: Optional[ParameterTable] = None
    ) -> Callable:
        """
        Convert a Qiskit Pulse Schedule to a QUA macro

        Args:
            sched: The Qiskit Pulse Schedule to convert
            param_table: The parameter table to use for the conversion of parametrized pulses to QUA variables

        Returns:
            The QUA macro corresponding to the Qiskit Pulse Schedule
        """

        if sched.is_parameterized():
            if param_table is None:
                param_dict = {}
                for channel in sched.channels:
                    if channel.is_parameterized():
                        ch_param = list(channel.parameters)[0]
                        if ch_param.name not in param_dict:
                            param_dict[ch_param.name] = 0
                for param in sched.parameters:
                    if param.name not in param_dict:
                        param_dict[param.name] = 0.0
                param_table = ParameterTable(param_dict)

            else:
                for param in sched.parameters:
                    if param.name not in param_table:
                        raise ValueError(
                            f"Parameter {param.name} is not in the parameter table"
                        )

        def qua_macro(
                *params,
        ):  # Define the QUA macro with parameters

            # Relate passed positional arguments to parameters in ParameterTable
            if not param_table.is_declared:
                param_table.declare_variables(pause_program=False)
            parameter_names = inspect.signature(qua_macro).parameters.keys()
            validate_parameters(parameter_names, param_table)

            param_mapping = {param: arg for param, arg in zip(parameter_names, params)}

            time_tracker = {channel: 0 for channel in sched.channels}
            for time, instruction in sched.instructions:

                if len(instruction.channels) > 1:
                    raise NotImplementedError("Only single channel instructions are supported")
                qiskit_channel = instruction.channels[0]

                if (
                        qiskit_channel.is_parameterized()
                ):  # Basic support for parameterized channels
                    # Filter dictionary of pulses based on provided ChannelType
                    channel_dict = {
                        channel.index: quam_channel
                        for channel, quam_channel in self._pulse_to_quam_channels.items()
                        if isinstance(channel, type(qiskit_channel))
                    }
                    ch_parameter_name = list(qiskit_channel.parameters)[0].name
                    if not param_table.table[ch_parameter_name].type == int:
                        raise ValueError(
                            f"Parameter {ch_parameter_name} must be of type int for switch case"
                        )
                    with switch_(
                            param_mapping[
                                ch_parameter_name
                            ]  # QUA variable corresponding to the parameter
                    ):  # Switch based on the parameter value
                        for i in channel_dict.keys():
                            with case_(i):
                                quam_channel = channel_dict[i]
                                if time_tracker[qiskit_channel] < time:
                                    quam_channel.wait(
                                        (time - time_tracker[qiskit_channel])
                                    )
                                    time_tracker[qiskit_channel] = time
                                _instruction_to_qua(
                                    instruction,
                                    quam_channel,
                                    param_mapping,
                                    param_table,
                                )
                                time_tracker[qiskit_channel] += instruction.duration
                else:
                    quam_channel = self.get_quam_channel(qiskit_channel)
                    if time_tracker[qiskit_channel] < time:
                        quam_channel.wait((time - time_tracker[qiskit_channel]))
                        time_tracker[qiskit_channel] = time
                    _instruction_to_qua(
                        instruction, quam_channel, param_mapping, param_table
                    )
                    time_tracker[qiskit_channel] += instruction.duration

        return qua_macro

    def update_calibrations(
            self, qc: QuantumCircuit
    ):
        """
        This method updates the QuAM with the custom calibrations of the QuantumCircuit (if any)
        and adds the corresponding operations to the QUA operations mapping for the OQC compiler.
        This method should be called before opening the QuantumMachine instance (i.e. before generating the
        configuration through QuAM) as it modifies the QuAM configuration.
        """

        if qc.parameters:
            if "parameter_table" not in qc.metadata:
                qc.metadata["parameter_table"] = ParameterTable(
                    {param.name: 0.0 for param in qc.parameters}
                )

            param_table = qc.metadata["parameter_table"]
        else:
            param_table = None

        if qc.calibrations:  # Check for custom calibrations
            for gate_name, cal_info in qc.calibrations.items():
                if gate_name not in self._oq3_basis_gates:  # Make it a basis gate for OQ compiler
                    self._oq3_basis_gates.append(gate_name)
                for (qubits, parameters), schedule in cal_info.items():
                    schedule = validate_schedule(schedule)  # Check that schedule has fixed duration
                    param_table = handle_parametrized_channel(schedule, param_table)

                    qc.metadata["parameter_table"] = param_table

                    self._operation_mapping_QUA[
                        OperationIdentifier(
                            gate_name,
                            len(parameters),
                            qubits,
                        )
                    ] = self.schedule_to_qua_macro(schedule, param_table)

                    # Update QuAM with additional custom pulses
                    for idx, (time, instruction) in enumerate(
                            schedule.filter(instruction_types=[Play]).instructions
                    ):
                        instruction: Play
                        pulse, channel = instruction.pulse, instruction.channel
                        if not isinstance(pulse, (SymbolicPulse, Waveform)):
                            raise ValueError(
                                "Only SymbolicPulse and Waveform pulses are supported"
                            )

                        pulse_name = pulse.name
                        if (
                                not channel.is_parameterized()
                                and pulse_name in self.get_quam_channel(channel).operations
                        ):
                            pulse_name += str(pulse.id)
                            pulse.name = pulse_name

                        quam_pulse = QuAMQiskitPulse(pulse)
                        if quam_pulse.is_compile_time_parametrized():
                            raise ValueError(
                                "Pulse contains unassigned parameters that cannot be adjusted in real-time"
                            )

                        if (
                                channel.is_parameterized()
                        ):  # Add pulse to each channel of same type
                            for ch in filter(
                                    lambda x: isinstance(x, type(channel)),
                                    self._pulse_to_quam_channels.keys(),
                            ):
                                self.get_quam_channel(ch).operations[pulse.name] = (
                                    QuAMQiskitPulse(pulse)
                                )
                        else:
                            self.get_quam_channel(channel).operations[pulse.name] = (
                                QuAMQiskitPulse(pulse)
                            )

    def quantum_circuit_to_qua(self, qc: QuantumCircuit, param_table: Optional[ParameterTable] = None):
        """
        Convert a QuantumCircuit to a QUA program (can be called within an existing QUA program or to generate a
        program for the circuit)

        Args:
            qc: The QuantumCircuit to convert
            param_table: The parameter table to use for the conversion of parametrized instructions to QUA variables
                        Should be provided if the QuantumCircuit contains parametrized instructions and this function
                        is called within a QUA program

        Returns:
            Compilation result of the QuantumCircuit to QUA
        """
        if qc.parameters:
            if param_table is not None:
                if "parameter_table" not in qc.metadata:
                    qc.metadata["parameter_table"] = param_table
                else:
                    assert qc.metadata["parameter_table"] == param_table, (
                        "Parameter table provided is different from the one in the QuantumCircuit metadata"
                    )
            if param_table is None:
                param_table = qc.metadata.get("parameter_table", None)

        open_qasm_code = qasm3_dumps(qc, includes=(), basis_gates=self._oq3_basis_gates)
        open_qasm_code = "\n".join(
            line
            for line in open_qasm_code.splitlines()
            if not line.strip().startswith(("barrier",))
        )
        result = self.compiler.compile(
            open_qasm_code,
            inputs={param.name: param.var for param in param_table.table} if param_table.is_declared else None,
        )
        return result

    def qua_prog_from_qc(self, qc: QuantumCircuit | Schedule | ScheduleBlock | Program):
        """
        Convert given input into a QUA program
        """
        if isinstance(qc, Program):
            return qc
        else:
            if qc.parameters:  # Initialize the parameter table
                parameter_table = ParameterTable(
                    {param.name: 0.0 for param in qc.parameters}
                )
                qc.metadata["parameter_table"] = parameter_table
            else:
                parameter_table = None
            if isinstance(qc, QuantumCircuit):
                return self.quantum_circuit_to_qua(qc, parameter_table)
            elif isinstance(qc, (ScheduleBlock, Schedule)):  # Convert to Schedule first
                schedule = qc
                if isinstance(qc, ScheduleBlock):
                    if not qc.is_schedulable():
                        # TODO: Build ScheduleBlock to QUA compiler
                        raise ValueError(
                            "ScheduleBlock is not schedulable (contains unschedulable instructions)"
                        )
                    try:
                        schedule = block_to_schedule(qc)
                    except (UnassignedDurationError, PulseError) as e:
                        # TODO: Build ScheduleBlock to QUA compiler
                        raise RuntimeError(
                            "ScheduleBlock could not be converted to Schedule (required"
                            "for converting it to QUA program"
                        ) from e
                return self.schedule_to_qua_macro(schedule, parameter_table)
            else:
                raise ValueError(f"Unsupported input {qc}")

    @property
    def compiler(self) -> Compiler:
        """
        The OpenQASM to QUA compiler.
        """
        return Compiler(
            hardware_config=HardwareConfig(
                quantum_operations_db=self._operation_mapping_QUA,
                physical_qubits=self.qubit_mapping,
            )
        )


class FluxTunableTransmonBackend(QMBackend):

    def __init__(
            self,
            machine: QuAM,
    ):
        super().__init__(machine)

    @property
    def qubit_mapping(self) -> QubitsMapping:
        """
        Retrieve the qubit to quantum elements mapping for the backend.
        """
        return {
            i: (qubit.xy.name, qubit.z.name, qubit.resonator.name)
            for i, qubit in enumerate(self.machine.qubits.values())
        }

    def _populate_target(self, machine: QuAM):
        """
        Populate the target instructions with the QOP configuration (currently hardcoded for
        Transmon based QuAM architecture)

        """
        target = Target("Transmon based QuAM", dt=1e-9, granularity=4, num_qubits=len(machine.qubits),
                        min_length=16, )
        for i, qubit in enumerate(machine.qubits.values()):
            self._target.qubit_properties[i] = QubitProperties(
                t1=qubit.T1, t2=qubit.T2ramsey, frequency=qubit.f_01
            )
            self._quam_to_pulse_channels[qubit.xy] = DriveChannel(i)
            self._pulse_to_quam_channels[DriveChannel(i)] = qubit.xy
            self._quam_to_pulse_channels[qubit.z] = FluxChannel(i)
            self._pulse_to_quam_channels[FluxChannel(i)] = qubit.z
            self._quam_to_pulse_channels[qubit.resonator] = [
                MeasureChannel(i),
                AcquireChannel(i),
            ]
            self._pulse_to_quam_channels[MeasureChannel(i)] = qubit.resonator
            self._pulse_to_quam_channels[AcquireChannel(i)] = qubit.resonator
            # TODO: Add the rest of the channels for QubitPairs (ControlChannels)

            # TODO: Update the instructions both in Qiskit and in the OQC operations mapping
            # TODO: Figure out if pulse calibrations should be added to Target

        self._coupling_map = target.build_coupling_map()

        return target


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
