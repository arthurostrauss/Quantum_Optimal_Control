"""
This module provides the ParameterValue class, which enables the mapping of a parameter to a QUA variable to be updated.

Author: Arthur Strauss - Quantum Machines
Created: 25/11/2024
"""

from __future__ import annotations

import time
import warnings
from itertools import chain
import sys

from qm.qua._dsl import _ResultSource

from .parameter_pool import ParameterPool
from .input_type import Direction, InputType
from typing import Optional, List, Union, Tuple, Literal, Sequence, TYPE_CHECKING, Dict
import numpy as np
from qm import QuantumMachine
from qm.qua import (
    fixed,
    assign,
    declare,
    pause,
    declare_input_stream,
    save,
    for_,
    advance_input_stream as qua_advance_input_stream,
    declare_stream as qua_declare_stream,
    IO1,
    IO2,
    if_,
    Util,
    declare_struct,
    qua_struct,
    QuaArray,
    declare_external_stream,
    send_to_external_stream as external_stream_send,
    fetch_from_external_stream as external_stream_receive,
)
from qm.qua.type_hints import Scalar, Vector, VectorOfAnyType, ScalarOfAnyType
from qm.jobs.running_qm_job import RunningQmJob
from qualang_tools.results import wait_until_job_is_paused, fetching_tool
from quam.utils.qua_types import QuaVariable

if TYPE_CHECKING:
    from .parameter_table import ParameterTable


def set_type(qua_type: Union[str, type]):
    """
    Set the type of the QUA variable to be declared.
    Args: qua_type: Type of the QUA variable to be declared (int, fixed, bool).
    """

    if qua_type == "fixed" or qua_type == fixed:
        return fixed
    elif qua_type == "bool" or qua_type == bool:
        return bool
    elif qua_type == "int" or qua_type == int:
        return int
    else:
        raise ValueError("Invalid QUA type. Please use 'fixed', 'int' or 'bool'.")


def infer_type(value: Union[int, float, List, np.ndarray] = None):
    """
    Infer automatically the type of the QUA variable to be declared from the type of the initial parameter value.
    """
    if value is None:
        raise ValueError("Initial value must be provided to infer type.")

    elif isinstance(value, float):
        if value.is_integer() and value > 8:
            return int
        else:
            return fixed

    elif isinstance(value, bool):
        return bool

    elif isinstance(value, int):
        return int

    elif isinstance(value, (List, np.ndarray)):
        if isinstance(value, np.ndarray):
            assert value.ndim == 1, "Invalid parameter type, array must be 1D."
            value = value.tolist()
        assert all(
            isinstance(x, type(value[0])) for x in value
        ), "Invalid parameter type, all elements must be of same type."
        if isinstance(value[0], bool):
            return bool
        elif isinstance(value[0], int):
            return int
        elif isinstance(value[0], float):
            return fixed
        else:
            raise ValueError("Invalid parameter type. Please use float, int or bool or list.")
    else:
        raise ValueError("Invalid parameter type. Please use float, int or bool or list.")


class Parameter:
    """
    Class enabling the mapping of a parameter to a QUA variable to be updated. The type of the QUA variable to be
    adjusted can be declared explicitly or either be automatically inferred from the type of provided initial value.
    """

    def __new__(cls, name, value=None, qua_type=None, input_type=None, direction=None, units=""):
        """
        Create a new instance of the Parameter class.
        """
        for obj in ParameterPool.get_all_objs():
            if hasattr(obj, "parameters"):
                for param in obj.parameters:
                    if param.name == name:
                        warnings.warn(
                            f"Parameter with name {name} already exists in parameter table {obj.name}."
                        )
                        return param
            else:
                if obj.name == name:
                    warnings.warn(
                        f"Parameter with name {name} already exists in the parameter pool."
                    )
                    return obj
        # TODO: Handle the case where the parameter is part of a parameter table with indexing
        obj = super().__new__(cls)
        return obj

    def __init__(
        self,
        name: str,
        value: Optional[Union[int, float, List, np.ndarray]] = None,
        qua_type: Optional[Union[str, type]] = None,
        input_type: Optional[Union[Literal["DGX", "INPUT_STREAM", "IO1", "IO2"], InputType]] = None,
        direction: Optional[Union[Literal["INCOMING", "OUTGOING"], Direction]] = None,
        units: str = "",
    ):
        """

        Args:
            name: Name of the parameter.
            value: Initial value of the parameter.
            qua_type: Type of the QUA variable to be declared (int, fixed, bool). Default is None.
            input_type: Input type of the parameter (dgx, input_stream, IO1, IO2). Default is None.
            direction: Direction of the parameter stream (INCOMING, OUTGOING).
                The direction describes in this case the relationship between DGX and OPX in the following manner:
                DGX -> OPX: OUTGOING
                OPX -> DGX: INCOMING
                Default is None. Relevant only if
                          input_type is dgx.
            units: Units of the parameter. Default is "".

        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._name = name
        self.units = units
        self.value = value
        self._index = -1  # Default value for parameters not part of a parameter table
        self._var = None
        self._is_declared = False
        self._stream = None
        self._stream_id = None
        self._type = set_type(qua_type) if qua_type is not None else infer_type(value)
        self._length = 0 if not isinstance(value, (List, np.ndarray)) else len(value)
        self._counter_var = None

        self._external_stream_incoming = None
        self._external_stream_outgoing = None

        if input_type is not None:
            input_type = InputType(input_type) if isinstance(input_type, str) else input_type
        self._input_type: Optional[InputType] = input_type
        self._dgx_struct = None
        if direction is not None:
            direction = Direction(direction) if isinstance(direction, str) else direction
        self._direction = direction
        self._table_indices: Dict[str, int] = {}

        if self._input_type == InputType.DGX and self.direction is None:
            raise ValueError("Direction must be provided for DGX input type.")

        self._initialized = True

    def get_name(self):
        return f"{self.name} [{self.units}]"

    def __repr__(self):
        """
        Returns:
            str: String representation of the parameter.
        """
        return (
            f"Parameter(name={self.name}, value={self.value}, type={self.type}, "
            f"length={self.length}, input_type={self.input_type}, "
            f"direction={self._direction}, units={self.units})"
        )

    def get_index(self, param_table: ParameterTable) -> int:
        """
        Get the index of the parameter in the parameter table.
        Args:
            param_table: ParameterTable object to get the index from.
        Returns:
            int: Index of the parameter in the parameter table.
        """
        if self._index == -1:
            raise ValueError(
                "This parameter is not part of a parameter table. "
                "Please use this method through the parameter table instead."
            )
        if param_table.name not in self._table_indices:
            raise ValueError(
                f"Parameter {self.name} is not part of the parameter table {param_table.name}."
            )
        return self._table_indices[param_table.name]

    def set_index(self, param_table: ParameterTable, index: int):
        """
        Set the index of the parameter in the parameter table.
        Args:
            param_table: ParameterTable object to set the index for.
            index: Index of the parameter in the parameter table.
        """
        if self._index == -1:
            self._index = -2
        if self.input_type == InputType.DGX and not len(self._table_indices) <= 1:
            raise ValueError(
                "This parameter is already part of a parameter table. In DGX mode, "
                "you cannot assign the same parameter to multiple tables."
            )
        if param_table.name not in self._table_indices:
            self._table_indices[param_table.name] = index
        else:
            raise ValueError(
                f"Parameter {self.name} is already part of the parameter table {param_table.name}."
            )

    def assign_value(
        self,
        value: Union["Parameter", ScalarOfAnyType, VectorOfAnyType],
        is_qua_array: bool = False,
        condition=None,
        value_cond: Optional[Union["Parameter", ScalarOfAnyType, VectorOfAnyType]] = None,
    ):
        """
        Assign value to the QUA variable corresponding to the parameter.

        Args:
            value: Value to be assigned to the QUA variable. If the ParameterValue corresponds to a QUA array,
                   the value should be a list or a QUA array of the same length.
            is_qua_array: Boolean indicating if provided value is a QUA array (True) or a list of values (False).
                          Default is False. If True, the value should be a QUA array of the same length as the parameter.
                          When assigning a QUA array, a QUA loop is created to assign each element of the array to the
                          corresponding element of the QUA array. If False, a Python loop is used instead.
            condition: Condition to be met for the value to be assigned to the QUA variable.
            value_cond: Optional value to be assigned to the QUA variable if provided condition is not met.

        Raises:
            ValueError: If the variable is not declared, or if the condition and value_cond are not provided together,
                        or if the value_cond is not of the same type as value, or if the value length does not match
                        the parameter length, or if the input is invalid.
        """
        if not self.is_declared:
            raise ValueError(
                "Variable not declared. Declare the variable first through declare_variable method."
            )
        if (condition is not None) != (value_cond is not None):
            raise ValueError("Both condition and value_cond must be provided together.")

        def assign_with_condition(var, val, cond_val):
            if condition is not None:
                assign(var, Util.cond(condition, val, cond_val))
            else:
                assign(var, val)

        if isinstance(value, Parameter):
            if not value.is_declared:
                raise ValueError(
                    "Variable not declared. Declare the variable first through declare_variable method."
                )
            if value.length != self.length:
                raise ValueError(
                    f"Invalid input. {self.name} should be a list of length {self.length}."
                )
            if value_cond is not None and not isinstance(value_cond, Parameter):
                raise ValueError("Invalid input. value_cond should be of same type as value.")
            if self.is_array:
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    assign_with_condition(
                        self.var[i],
                        value.var[i],
                        value_cond.var[i] if value_cond else None,
                    )
            else:
                assign_with_condition(self.var, value.var, value_cond.var if value_cond else None)
        else:
            if self.is_array:
                if is_qua_array:
                    i = self._counter_var
                    with for_(i, 0, i < self.length, i + 1):
                        assign_with_condition(
                            self.var[i],
                            value[i],
                            value_cond[i] if value_cond is not None else None,
                        )
                else:
                    if len(value) != self.length:
                        raise ValueError(
                            f"Invalid input. {self.name} should be a list of length {self.length}."
                        )
                    for i in range(self.length):
                        assign_with_condition(
                            self.var[i], value[i], value_cond[i] if value_cond else None
                        )
            else:
                if isinstance(value, List):
                    raise ValueError(
                        f"Invalid input. {self.name} should be a single value, not a list."
                    )
                assign_with_condition(self.var, value, value_cond)

    def declare_variable(self, pause_program=False, declare_stream=True):
        """
        Declare the QUA variable associated with the parameter.
        Args: pause_program: Boolean indicating if the program should be paused after declaring the variable.
            Default is False.
        declare_stream: Boolean indicating if an output stream should be declared to save the QUA variable.
        """
        if self.is_declared:
            raise ValueError("Variable already declared. Cannot declare again.")
        if self.input_type == InputType.INPUT_STREAM:
            self._var = declare_input_stream(t=self.type, name=self.name, value=self.value)
        elif self.input_type == InputType.DGX:
            if not self._table_indices:
                # Parameter not part of a parameter table
                dgx_struct = self.dgx_struct
                self._var = declare_struct(dgx_struct)

                if self.direction == Direction.INCOMING:
                    self._external_stream_outgoing = declare_external_stream(
                        dgx_struct, self.stream_id, "OUTGOING"
                    )
                else:
                    self._external_stream_incoming = declare_external_stream(
                        dgx_struct, self.stream_id, "INCOMING"
                    )
            else:
                raise ValueError(
                    "This method should be called from a ParameterTable object "
                    "as this parameter was associated with a bigger packet."
                )

        else:
            self._var = declare(t=self.type, value=self.value)
        if self.is_array:
            self._counter_var = declare(int)
        if declare_stream:
            self._stream = qua_declare_stream()
        if pause_program:
            pause()
        self._is_declared = True
        return self._var

    def declare_stream(self):
        """
        Declare the output stream associated with the parameter.
        """
        if self._stream is None:
            self._stream = qua_declare_stream()
        return self._stream

    @property
    def is_declared(self):
        """Boolean indicating if the QUA variable has been declared."""
        return self._is_declared

    @property
    def name(self):
        """Name of the parameter."""
        return self._name

    @property
    def direction(self):
        if self.input_type != InputType.DGX:
            warnings.warn("This parameter is not associated with a DGX stream.")
            raise ValueError("This parameter is not associated with a DGX stream.")
        return self._direction

    @property
    def dgx_struct(self):
        """
        DGX struct associated with the parameter.
        Can be a reference to a bigger struct describing a ParameterTable (automatically set by the
        ParameterTable class) or a standalone struct created on the fly.
        Returns:

        """
        if self._index < 0:
            if self.input_type != InputType.DGX:
                raise ValueError("Invalid input type for this parameter. Must be dgx.")
            length = 1 if not self.is_array else self.length
            cls_name = f"{self.name}_struct"
            dgxStruct = qua_struct(
                type(
                    cls_name,
                    (object,),
                    {"__annotations__": {self.name: QuaArray[self.type, length]}},
                )
            )

            return dgxStruct
        else:
            return self._dgx_struct

    @dgx_struct.setter
    def dgx_struct(self, value):
        self._dgx_struct = value

    @property
    def stream_id(self) -> int:
        """
        ID of the external stream associated with the parameter (Relevant only for DGX).
        If the Parameter is part of a ParameterTable, returns the stream_id of the table.
        Returns:
            Unique ID integer of the external stream.

        """
        if self._index == -1:  # Not in a ParameterTable
            self._stream_id = ParameterPool.get_id(self)
            self._index = -2  # To avoid reassigning the stream ID

        return self._stream_id

    @stream_id.setter
    def stream_id(self, value: int):
        if self._index != -1:
            raise ValueError(
                "Cannot set stream ID for a parameter that is part of a ParameterTable."
            )
        self._stream_id = value

    @property
    def var(self):
        """
        Returns:
            QUA variable associated with the parameter.
        """
        if not self.is_declared:
            raise ValueError(
                "Variable not declared. Declare the variable first through declare_variable method."
            )
        if self.input_type == InputType.DGX:
            var = getattr(self._var, self.name)
            return var if self.is_array else var[0]

        else:
            return self._var

    @property
    def tables(self) -> List[ParameterTable]:
        """
        Returns:
            List of ParameterTable objects associated with the parameter.
        """
        return list(self._table_indices.keys())

    @property
    def type(self):
        """Type of the associated QUA variable."""
        return self._type

    @type.setter
    def type(self, value: Union[str, type]):
        if self.is_declared:
            raise ValueError("Variable already declared. Cannot change type.")
        self._type = set_type(value)

    @property
    def length(self):
        """Length of the parameter if it refers to a QUA array (
        returns 0 if single value)."""
        return self._length

    @property
    def input_type(self) -> Optional[InputType]:
        """
        Type of input stream associated with the parameter.
        """
        return self._input_type

    @property
    def is_array(self):
        """Boolean indicating if the parameter refers to a QUA array."""
        return self.length > 0

    @property
    def stream(self) -> _ResultSource:
        """Output stream associated with the parameter."""
        if self._stream is None or not self.is_declared:
            raise ValueError("Output stream not declared.")
        return self._stream

    def save_to_stream(self):
        """Save the QUA variable to the output stream."""
        if self.is_declared and self.stream is not None:
            if self.is_array:
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    save(self.var[i], self.stream)
            else:
                save(self.var, self.stream)
        else:
            raise ValueError("Output stream not declared.")

    def stream_processing(self, mode: Literal["save", "save_all"] = "save_all"):
        """
        Process the output stream associated with the parameter.
        """
        if self.stream is not None:
            if mode == "save":
                if self.is_array:
                    self.stream.buffer(self.length).save(self.name)
                else:
                    self.stream.save(self.name)
            elif mode == "save_all":
                if self.is_array:
                    self.stream.buffer(self.length).save_all(self.name)
                else:
                    self.stream.save_all(self.name)
            else:
                raise ValueError("Invalid mode. Must be 'save' or 'save_all'.")
        else:
            raise ValueError("Output stream not declared.")

    def clip(
        self,
        min_val: Optional[Scalar[int], Scalar[float], Vector[int], Vector[float]] = None,
        max_val: Optional[Scalar[int], Scalar[float], Vector[int], Vector[float]] = None,
        is_qua_array: bool = False,
    ):
        """
        Clip the QUA variable to a given range.
        Args: min_val: Minimum value of the range.
            max_val: Maximum value of the range.
            is_array: Boolean indicating if the bounds are QUA arrays.
        """
        if not self.is_declared:
            raise ValueError(
                "Variable not declared. Declare the variable first through declare_variable method."
            )
        if not self.is_array and is_qua_array:
            raise ValueError("Invalid input. Single value cannot be clipped with array bounds.")
        elif (
            isinstance(min_val, (int, float))
            and isinstance(max_val, (int, float))
            and min_val > max_val
        ):
            raise ValueError("Invalid range. Minimum value must be less than maximum value.")

        elif min_val is None and max_val is None:
            warnings.warn("No range specified. No clipping performed.")
            return

        if self.is_array:
            i = self._counter_var
            with for_(i, 0, i < self.length, i + 1):
                if is_qua_array:
                    if min_val is not None:
                        with if_(self.var[i] < min_val[i]):
                            assign(self.var[i], min_val[i])
                    if max_val is not None:
                        with if_(self.var[i] > max_val[i]):
                            assign(self.var[i], max_val[i])
                else:
                    if min_val is not None:
                        with if_(self.var[i] < min_val):
                            assign(self.var[i], min_val)
                    if max_val is not None:
                        with if_(self.var[i] > max_val):
                            assign(self.var[i], max_val)
        else:
            if min_val is not None:
                with if_(self.var < min_val):
                    assign(self.var, min_val)
            if max_val is not None:
                with if_(self.var > max_val):
                    assign(self.var, max_val)

    def load_input_value(self):
        """
        Advance the input stream associated with the parameter.
        The mechanism to advance the input stream depends on the input type.
        For input streams, the stream is advanced.
        For IO1 and IO2, the value is assigned to the QUA variable.
        For dgx, the value is polled.
        """
        if self.input_type is None:
            raise ValueError("No input type specified")
        elif self.input_type == InputType.INPUT_STREAM:
            qua_advance_input_stream(self.var)

        elif self.input_type == InputType.DGX:
            if self._dgx_struct is None:
                external_stream_receive(self._external_stream_incoming, self._var)
            else:
                raise ValueError(
                    "This method should be called from a ParameterTable object "
                    "as this parameter was associated with a bigger packet."
                )

        elif self.input_type in [InputType.IO1, InputType.IO2]:
            io = IO1 if self.input_type == InputType.IO1 else IO2
            if self.is_array:
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    pause()
                    assign(self.var[i], io)
            else:
                pause()
                assign(self.var, io)

        else:
            raise ValueError("Invalid input stream type.")

    def push_to_opx(
        self,
        value: Union[int, float, bool, Sequence[Union[int, float, bool]]],
        job: RunningQmJob,
        qm: Optional[QuantumMachine] = None,
        verbosity: int = 1,
    ):
        """
        To be outside QUA program: pass an input value to the OPX from Python
        Args:
            value: Value to be passed to the OPX.
            job: RunningQmJob object (required if input_type is IO1 or IO2 or input_stream).
            qm: QuantumMachine object (required if input_type is IO1 or IO2).
            verbosity: Verbosity level. Default is 1.
        """

        if self.is_array and len(value) != self.length:
            raise ValueError(
                f"Invalid input. {self.name} should be a list of length {self.length}."
            )
        elif not self.is_array and not isinstance(value, (int, float, bool)):
            raise ValueError(
                f"Invalid input. {self.name} should be a single value (received {type(value)})."
            )
        param_type = self.type
        if param_type == fixed:
            param_type = float
        if self.is_array and not all(isinstance(x, param_type) for x in value):
            try:
                value = [param_type(x) for x in value]
            except ValueError:
                raise ValueError(f"Invalid input. {self.name} should be a list of {param_type}.")
        elif not self.is_array and not isinstance(value, param_type):
            try:
                value = param_type(value)
            except ValueError:
                raise ValueError(
                    f"Invalid input. {self.name} should be a single value of type {param_type}."
                )

        if self.is_array:
            value = list(value)

        if self.input_type in [InputType.IO1, InputType.IO2]:
            io = "set_io1_value" if self.input_type == InputType.IO1 else "set_io2_value"
            if qm is None:
                raise ValueError("QuantumMachine object must be provided.")
            if self.is_array:
                for i in range(self.length):
                    getattr(qm, io)(value[i])
                    wait_until_job_is_paused(job)
                    job.resume()
            else:
                if not isinstance(value, (int, float, bool)):
                    raise ValueError(
                        f"Invalid input. {self.name} should be a single value (received {type(value)})."
                    )
                getattr(qm, io)(value)
                wait_until_job_is_paused(job)
                job.resume()

        elif self.input_type == InputType.INPUT_STREAM:
            job.push_to_input_stream(self.name, value)

        elif self.input_type == InputType.DGX:
            if self.index < 0:
                raise ValueError(
                    "Cannot push value to a standalone parameter,"
                    "Please push through the parameter table instead."
                )
            if self.direction == Direction.INCOMING:
                raise ValueError("Cannot push value to Incoming stream.")
            # Prepare the packet to be sent
            param_dict = {self.name: [value] if not self.is_array else value}
            param_dict = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in param_dict.items()
            }

            if ParameterPool.configured and ParameterPool.patched:
                # from opnic_python.opnic_wrapper import OutgoingPacket, send_packet
                if "opnic_wrapper" not in sys.path:
                    sys.path.append("/home/dpoulos/aps_demo/python-wrapper/wrapper/build/python")
                from opnic_wrapper import OutgoingPacket, send_packet

                flattened_values = list(chain(*param_dict.values()))
                packet = OutgoingPacket(flattened_values)
                for k, v in param_dict.items():
                    setattr(packet, k, v)
                send_packet(self.stream_id, packet)
            else:
                raise ValueError("OPNIC not configured or patched.")

            if verbosity > 1:
                print(f"Sent packet: {packet}")

    def send_to_python(self):
        """
        QUA macro designed to send the value of the parameter to Python.
        This method uses IO variables if input type is IO1 or IO2, and
        external streams if input type is dgx. If specified as an input stream,
        the value is saved to the stream.
        """
        if self.input_type in [InputType.IO1, InputType.IO2]:
            io = IO1 if self.input_type == InputType.IO1 else IO2
            if self.is_array:
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    assign(io, self.var[i])
                    pause()
            else:
                assign(io, self.var)
                pause()
        elif self.input_type == InputType.INPUT_STREAM:
            self.save_to_stream()
        elif self.input_type == InputType.DGX:
            if self.index >= 0:  # Part of a parameter table
                raise ValueError(
                    "Cannot send value to a standalone parameter,"
                    "Please use this method through the parameter table instead."
                )
            if self.direction == Direction.OUTGOING:
                raise ValueError("Cannot send value to outgoing stream.")

            external_stream_send(self._external_stream_outgoing, self._var)

    def fetch_from_opx(
        self,
        job: RunningQmJob,
        qm: Optional[QuantumMachine] = None,
        verbosity: int = 1,
    ):
        """
        To be outside QUA program: fetch an output value from the OPX to Python.
        Returns:
            Value fetched from the OPX.
        """
        if self.input_type in [InputType.IO1, InputType.IO2]:
            io = "get_io1_value" if self.input_type == InputType.IO1 else "get_io2_value"
            if qm is None:
                raise ValueError("QuantumMachine object must be provided.")
            if not self.is_array:
                wait_until_job_is_paused(job)
                value = getattr(qm, io)()
                job.resume()
            else:
                value = []
                for i in range(self.length):
                    wait_until_job_is_paused(job)
                    value.append(getattr(qm, io)())
                    job.resume()
        elif self.input_type == InputType.INPUT_STREAM:
            value = fetching_tool(job, [self.name], mode="live")
            while value.is_processing():
                value = value.fetch_all()
        elif self.input_type == InputType.DGX:
            if self.index < 0:
                raise ValueError(
                    "Cannot fetch value from a standalone parameter,"
                    "Please fetch through the parameter table instead."
                )
            if self.direction == Direction.OUTGOING:
                raise ValueError("Cannot fetch value from outgoing stream.")
            elif not ParameterPool.configured or not ParameterPool.patched:
                raise ValueError("OPNIC not configured or patched.")
            if "opnic_wrapper" not in sys.modules:
                sys.path.append("/home/dpoulos/aps_demo/python-wrapper/wrapper/build/python")
            from opnic_wrapper import wait_for_packets, read_packet

            wait_for_packets(self.stream_id, 1)
            packet = read_packet(self.stream_id, 0)
            value = getattr(packet, self.name)

        else:
            raise ValueError("Invalid input type.")
        if verbosity > 1:
            print(f"Fetched value: {value}")
        return value

    def reset(self):
        """
        Reset the parameter to its initial state.
        """
        self._is_declared = False
        self._var = None
        self._stream = None
        self._stream_id = None
        self._counter_var = None
        self._dgx_struct = None
        self._external_stream_incoming = None
        self._external_stream_outgoing = None
        self._index = -1
