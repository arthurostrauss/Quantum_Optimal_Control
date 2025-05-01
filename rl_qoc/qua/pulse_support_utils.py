from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit.circuit.parametervector import ParameterVectorElement
from qm.qua import Math, amp as qua_amp
from qiskit.pulse.library.pulse import Pulse as QiskitPulse
from quam.components import Channel as QuAMChannel
from qiskit.pulse.channels import Channel as QiskitChannel
from qiskit.pulse import (
    Play,
    Instruction,
    Schedule,
    ScheduleBlock,
    Acquire,
)
from .sympy_to_qua import sympy_to_qua
from qiskit.circuit.parameterexpression import ParameterExpression
from .parameter_table import ParameterTable, Parameter as QuaParameter
from typing import Dict, Optional, Type
from qiskit.pulse.transforms import block_to_schedule
from .pulse_to_qua import *

# TODO: Add duration to the list of real-time parameters (need ScheduleBlock to QUA compiler)
_real_time_parameters = {
    "amp",
    "angle",
    "frequency",
    "phase",
    "duration",
}  # Parameters that can be used in real-time


def get_real_time_pulse_parameters(pulse: QiskitPulse):
    """
    Get the real-time parameters of a Qiskit Pulse, that is parameters of the pulse that are
    not known at compile time and need to be calculated at runtime (i.e., ParameterExpressions)
    """
    real_time_params = {}
    for param in _real_time_parameters:
        if hasattr(pulse, param) and isinstance(getattr(pulse, param), ParameterExpression):
            real_time_params[param] = getattr(pulse, param)
    return real_time_params


def _handle_parameterized_instruction(
    instruction: Instruction,
    param_table: ParameterTable,
    qua_pulse_macro: QuaPulseMacro,
):
    """
    Handle the conversion of a parameterized instruction to QUA by creating a dictionary of parameter values
    and assigning them to the corresponding QUA variables
    """
    if not type(instruction) in qiskit_to_qua_instructions:
        raise ValueError(f"Instruction {instruction} not supported on QM backend")
    value_dict = {}
    involved_parameters = {}  # Store involved Parameters
    validate_parameters(instruction.parameters, param_table)
    for param in instruction.parameters:
        involved_parameters[param.name] = param_table.get_parameter(param.name)

    for attribute in qua_pulse_macro.params:
        attribute_value = getattr(instruction, attribute)
        if isinstance(attribute_value, ParameterExpression):
            value_dict[attribute] = sympy_to_qua(
                getattr(instruction, attribute).sympify(), involved_parameters
            )
        elif attribute == "pulse":
            pulse = getattr(instruction, attribute)
            for pulse_param_name, pulse_param in get_real_time_pulse_parameters(pulse).items():
                value_dict[pulse_param_name] = sympy_to_qua(
                    pulse_param.sympify(),
                    involved_parameters,
                )
            break
        # else:  # TODO: Check if this is necessary
        #     value_dict[attribute] = getattr(
        #         instruction, attribute
        #     )  # Assign the value of the attribute

    return value_dict


def validate_pulse(pulse: QiskitPulse, channel: QuAMChannel) -> QiskitPulse:
    """
    Validate the pulse on the QuAM channel
    """
    if not pulse.name in channel.operations:
        raise ValueError(f"Pulse {pulse.name} is not in the operations of the QuAM channel")

    return pulse


def validate_instruction(instruction: Instruction, quam_channel: QuAMChannel) -> QuaPulseMacro:
    """
    Validate the instruction before converting it to QUA and return the corresponding QUA macro
    """
    kwargs: Dict[str, QuAMChannel | QiskitPulse] = {"channel": quam_channel}
    if isinstance(instruction, Play):
        pulse = instruction.pulse
        kwargs["pulse"] = pulse
    if type(instruction) in qiskit_to_qua_instructions:
        return qiskit_to_qua_instructions[type(instruction)](**kwargs)
    elif isinstance(instruction, Acquire):
        raise NotImplementedError("Acquire instructions are not yet supported.")
    else:
        raise ValueError(f"Instruction {instruction} not supported on QM backend")


def validate_parameters(params, param_table: ParameterTable, param_mapping=None) -> ParameterTable:
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
            raise ValueError(f"Parameter {param_name} is not in the provided parameters mapping")
    return param_table


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

    action = validate_instruction(instruction, quam_channel)

    if instruction.is_parameterized():
        assert param_table is not None, "Parameter table must be provided"
        values = _handle_parameterized_instruction(instruction, param_table, action)

    else:
        values = {}
        if not isinstance(instruction, Play):
            for attribute in action.params:
                values[attribute] = getattr(instruction, attribute)

    action.macro(**values)


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


def handle_parameterized_channel(schedule: Schedule, param_table: ParameterTable) -> ParameterTable:
    """
    Modify type of parameters (-> int) in the Table that refer to channel parameters (they refer to integers)
    """
    for channel in list(filter(lambda ch: ch.is_parameterized(), schedule.channels)):
        ch_params = list(channel.parameters)
        if len(ch_params) > 1:
            raise NotImplementedError("Only single parameterized channels are supported")
        ch_param = ch_params[0]
        if ch_param.name in param_table:
            param_table.get_parameter(ch_param.name).type = int
            param_table.get_parameter(ch_param.name).value = 0
        else:
            ch_parameter_value = QuaParameter(ch_param.name, 0, int)
            param_table.table[ch_param.name] = ch_parameter_value
    return param_table
