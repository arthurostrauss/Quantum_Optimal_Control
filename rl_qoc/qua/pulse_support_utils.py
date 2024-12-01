from qiskit.circuit import Parameter
from qiskit.circuit.parametervector import ParameterVectorElement
from qm.qua import Math, amp as qua_amp, reset_phase, QuaVariableType, assign, if_
from qiskit.pulse.library.pulse import Pulse as QiskitPulse
from quam.components import Channel as QuAMChannel
from qiskit.pulse import (
    Play,
    Instruction,
    Schedule,
    ScheduleBlock,
    Acquire,
    ShiftPhase,
    ShiftFrequency,
    SetFrequency,
    SetPhase,
    Delay,
)
from .sympy_to_qua import sympy_to_qua
from qiskit.circuit.parameterexpression import ParameterExpression
from .parameter_table import ParameterTable, ParameterValue
from typing import Dict, List, Optional, Callable, Union, Type
from functools import partial
from qiskit.pulse.transforms import block_to_schedule

# TODO: Add duration to the list of real-time parameters (need ScheduleBlock to QUA compiler)
_real_time_parameters = {
    "amp",
    "angle",
    "frequency",
    "phase",
}  # Parameters that can be used in real-time
_ref_amp = 0.5
_ref_phase = 0.0

_qiskit_to_qua_instructions: Dict[Type, Dict[str, Union[Callable, List[str]]]] = {
    ShiftPhase: {
        "macro": lambda channel, phase: channel.frame_rotation(phase),
        "params": ["phase"],
    },
    ShiftFrequency: {
        "macro": lambda channel, frequency: channel.update_frequency(
            frequency + channel.intermediate_frequency
        ),
        "params": ["frequency"],
    },
    SetFrequency: {
        "macro": lambda channel, frequency: channel.update_frequency(frequency),
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
        "macro": lambda channel, pulse, amp=None, angle=None, duration=None: channel.play(
            pulse.name, amplitude_scale=get_amp_matrix(amp, angle), duration=duration
        ),
        "params": ["pulse"],
    },
}


def get_real_time_pulse_parameters(pulse: QiskitPulse):
    """
    Get the real-time parameters of a Qiskit Pulse
    """
    real_time_params = {}
    for param in _real_time_parameters:
        if hasattr(pulse, param) and isinstance(
            getattr(pulse, param), ParameterExpression
        ):
            real_time_params[param] = getattr(pulse, param)
    return real_time_params


def _handle_parameterized_instruction(
    instruction: Instruction,
    param_table: ParameterTable,
):
    """
    Handle the conversion of a parameterized instruction to QUA by creating a dictionary of parameter values
    and assigning them to the corresponding QUA variables
    """
    value_dict = {}
    involved_parameter_values = {}  # Store ParameterValues for involved parameters
    validate_parameters(instruction.parameters, param_table)
    for param in instruction.parameters:
        involved_parameter_values[param.name] = param_table.get_value(param.name)

    for attribute in _qiskit_to_qua_instructions[type(instruction)]["params"]:
        attribute_value = getattr(instruction, attribute)
        if isinstance(attribute_value, ParameterExpression):
            value_dict[attribute] = sympy_to_qua(
                getattr(instruction, attribute).sympify(), involved_parameter_values
            )
        elif attribute == "pulse":
            pulse = getattr(instruction, attribute)
            for pulse_param_name, pulse_param in get_real_time_pulse_parameters(
                pulse
            ).items():
                value_dict[pulse_param_name] = sympy_to_qua(
                    pulse_param.sympify(),
                    involved_parameter_values,
                )
            break
        # else:  # TODO: Check if this is necessary
        #     value_dict[attribute] = getattr(
        #         instruction, attribute
        #     )  # Assign the value of the attribute

    return value_dict


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
        raise ValueError(
            f"Pulse {pulse.name} is not in the operations of the QuAM channel"
        )

    return pulse


def validate_instruction(instruction: Instruction) -> Callable:
    """
    Validate the instruction before converting it to QUA and return the corresponding QUA macro
    """
    if type(instruction) in _qiskit_to_qua_instructions:
        return _qiskit_to_qua_instructions[type(instruction)]["macro"]
    elif isinstance(instruction, Acquire):
        raise NotImplementedError("Acquire instructions are not yet supported.")
    else:
        raise ValueError(f"Instruction {instruction} not supported on QM backend")


def validate_parameters(
    params, param_table: ParameterTable, param_mapping=None
) -> None:
    """
    Validate the parameters of the instruction by checking them against the parameter table
    and a possible parameter mapping

    Args:
        params: List of parameters to validate (names or Qiskit Parameter objects)
        param_table: Parameter table to check the parameters against
        param_mapping: Mapping of parameters to QUA variables
    """
    if not isinstance(param_table, ParameterTable):
        raise ValueError("Parameter table must be provided")
    for param in params:
        if isinstance(param, ParameterVectorElement):
            param_name = param.vector.name
        elif isinstance(param, Parameter):
            param_name = param.name
        else:
            param_name = param
        if param_name not in param_table:
            raise ValueError(f"Parameter {param_name} is not in the parameter table")
        if param_mapping is not None and param_name not in param_mapping:
            raise ValueError(
                f"Parameter {param_name} is not in the provided parameters mapping"
            )


def _instruction_to_qua(
    instruction: Instruction,
    quam_channel: QuAMChannel,
    param_table: Optional[ParameterTable] = None,
):
    """
    Convert a Qiskit Pulse instruction to a QUA instruction

    Args:
        instruction: Qiskit Pulse instruction to convert
        quam_channel: QuAM channel on which to apply the instruction
        param_table:  Parameter table to use for the conversion (contains the reference QUA variables for parameters)
    """

    action = validate_instruction(instruction)

    if isinstance(instruction, Play):
        pulse = validate_pulse(instruction.pulse, quam_channel)
        action = partial(action, pulse=pulse)

    if instruction.is_parameterized():
        assert param_table is not None, "Parameter table must be provided"
        values = _handle_parameterized_instruction(instruction, param_table)

    else:
        values = {}
        if not isinstance(instruction, Play):
            for attribute in _qiskit_to_qua_instructions[type(instruction)]["params"]:
                values[attribute] = getattr(instruction, attribute)

    action(quam_channel, **values)


def validate_schedule(schedule: Schedule | ScheduleBlock) -> Schedule:
    if isinstance(schedule, ScheduleBlock):
        if not schedule.is_schedulable():
            raise NotImplementedError(
                "ScheduleBlock with parameterized durations are not yet supported"
            )

        schedule = block_to_schedule(schedule)
    if not isinstance(schedule, Schedule):
        raise ValueError("Only Qiskit Pulse Schedule objects are supported")

    return schedule


def handle_parameterized_channel(
    schedule: Schedule, param_table: ParameterTable
) -> ParameterTable:
    """
    Modify type of parameters (-> int) in the Table that refer to channel parameters (they refer to integers)
    """
    for (
        channel
    ) in (
        schedule.channels
    ):  # Check if channels are parametrized to change param type (->int)
        if channel.is_parameterized():
            ch_params = list(channel.parameters)
            if len(ch_params) > 1:
                raise NotImplementedError(
                    "Only single parameterized channels are supported"
                )
            ch_param = ch_params[0]
            ch_parameter_value = ParameterValue(
                ch_param.name, 0, param_table.table[ch_param.name].index, int
            )
            param_table.table[ch_param.name] = ch_parameter_value
    return param_table
