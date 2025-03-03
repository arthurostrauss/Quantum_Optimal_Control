"""
This module provides the ParameterValue class, which enables the mapping of a parameter to a QUA variable to be updated.

Author: Arthur Strauss - Quantum Machines
Created: 25/11/2024
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple, Literal, Iterable, Sequence
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
    QuaVariableType,
    advance_input_stream as qua_advance_input_stream,
    declare_stream as qua_declare_stream,
    IO1,
    IO2,
    poll_dgx,
    Cast,
    stream_dgx,
    if_,
    Util,
)
from qm.jobs.running_qm_job import RunningQmJob
from qualang_tools.results import wait_until_job_is_paused

float_to_int_scaling_factor = 1e6


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
        raise ValueError(
            "Invalid parameter type. Please use float, int or bool or list."
        )


class Parameter:
    """
    Class enabling the mapping of a parameter to a QUA variable to be updated. The type of the QUA variable to be
    adjusted can be declared explicitly or either be automatically inferred from the type of provided initial value.
    """

    def __init__(
        self,
        name: str,
        value: Optional[Union[int, float, List, np.ndarray]] = None,
        qua_type: Optional[Union[str, type]] = None,
        input_type: Optional[Literal["dgx", "input_stream", "IO1", "IO2"]] = None,
        units: str = "",
    ):
        """

        Args:
            name: Name of the parameter.
            value: Initial value of the parameter.
            qua_type: Type of the QUA variable to be declared (int, fixed, bool). Default is None.
            input_type: Input type of the parameter (dgx, input_stream, IO1, IO2). Default is None.

        """
        self._name = name
        self.value = value
        self._index = None
        self.var = None
        self._var_int = None
        self._stream = None
        self._type = set_type(qua_type) if qua_type is not None else infer_type(value)
        self._length = 0 if not isinstance(value, (List, np.ndarray)) else len(value)
        self._input_type = input_type
        self._is_declared = False
        self._counter_var = None
        self.units = units

    def get_name(self):
        return f"{self.name} [{self.units}]"

    def __repr__(self):
        return f"{self.name}: ({self.value}, {self.type}) \n"

    def assign_value(
        self,
        value: Union[
            "Parameter",
            float,
            int,
            bool,
            List[Union[float, int, bool, QuaVariableType]],
            QuaVariableType,
        ],
        is_qua_array: bool = False,
        condition=None,
        value_cond: Optional[Union[float, int, bool, QuaVariableType]] = None,
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
            if self.is_array:
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    assign_with_condition(
                        self.var[i],
                        value.var[i],
                        value_cond.var[i] if value_cond else None,
                    )
            else:
                assign_with_condition(
                    self.var, value.var, value_cond.var if value_cond else None
                )
        else:
            if self.is_array:
                if is_qua_array:
                    i = self._counter_var
                    with for_(i, 0, i < self.length, i + 1):
                        assign_with_condition(
                            self.var[i], value[i], value_cond[i] if value_cond else None
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
        if self.input_type == "input_stream":
            self.var = declare_input_stream(
                t=self.type, name=self.name, value=self.value
            )
        else:
            self.var = declare(t=self.type, value=self.value)
            if self.input_type == "dgx" and self.type != int:
                # if the input type is dgx, add an int variable as
                # it is the only type that can be loaded from the DGX (at the moment)
                self._var_int = declare(
                    int, size=self.length if self.is_array else None
                )
        if declare_stream:
            # if self.is_array:
            #     self._stream = [qua_declare_stream() for _ in range(self.length)]
            # else:
            #     self._stream = qua_declare_stream()
            self._stream = qua_declare_stream()
        if self.is_array:
            self._counter_var = declare(int)
        if pause_program:
            pause()
        self._is_declared = True
        return self.var

    @property
    def is_declared(self):
        """Boolean indicating if the QUA variable has been declared."""
        return self._is_declared

    @property
    def name(self):
        """Name of the parameter."""
        return self._name

    @property
    def index(self):
        """
        Index of the parameter in the parameter table.
        Relevant only if the parameter is part of a parameter table.
        This index can be used to access the parameter in the parameter table
        through the usage of a switch statement.
        """
        return self._index

    @index.setter
    def index(self, value: int):
        if value < 0:
            raise ValueError("Index must be a positive integer.")
        self._index = value

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
    def input_type(self) -> Optional[Literal["dgx", "input_stream", "IO1", "IO2"]]:
        """
        Type of input stream associated with the parameter.
        """
        return self._input_type

    @property
    def is_array(self):
        """Boolean indicating if the parameter refers to a QUA array."""
        return self.length > 0

    @property
    def stream(self):
        """Output stream associated with the parameter."""
        if self._stream is None or not self.is_declared:
            raise ValueError("Output stream not declared.")
        return self._stream

    def save_to_stream(self):
        """Save the QUA variable to the output stream."""
        if self.is_declared and self.stream is not None:
            if self.is_array:
                with for_(
                    self._counter_var,
                    0,
                    self._counter_var < self.length,
                    self._counter_var + 1,
                ):
                    save(self.var[self._counter_var], self.stream)
            else:
                save(self.var, self.stream)
        else:
            raise ValueError("Output stream not declared.")

    def stream_to_dgx(self):
        """
        Stream the output stream to the DGX.
        """
        if not self.input_type == "dgx":
            raise ValueError("Invalid input type for this parameter. Must be dgx.")
        if self.type == int:
            output_var = declare(int, value=[-2, -1])
            assign(output_var[0], self.var)

        elif self.type == fixed:
            if self.is_array:
                output_var = declare(int, value=[-2] * self.length + [-1])
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    assign(
                        output_var[i],
                        Cast.mul_fixed_by_int(float_to_int_scaling_factor, self.var[i]),
                    )
            else:
                output_var = declare(int, value=[-2, -1])
                assign(
                    output_var[0],
                    Cast.mul_fixed_by_int(float_to_int_scaling_factor, self.var),
                )

        else:  # bool
            if self.is_array:
                output_var = declare(int, value=[-2] * self.length + [-1])
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    assign(
                        output_var[i],
                        Cast.to_int(self.var[i]),
                    )
            else:
                output_var = declare(int, value=[-2, -1])
                assign(output_var[0], Cast.to_int(self.var))

        stream_dgx(output_var)

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
        min_val: Optional[Union[int, float, QuaVariableType]] = None,
        max_val: Optional[Union[int, float, QuaVariableType]] = None,
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
            raise ValueError(
                "Invalid input. Single value cannot be clipped with array bounds."
            )
        elif (
            isinstance(min_val, (int, float))
            and isinstance(max_val, (int, float))
            and min_val > max_val
        ):
            raise ValueError(
                "Invalid range. Minimum value must be less than maximum value."
            )

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

    def load_input_value(self, start_with_dummy_packet=False):
        """
        Advance the input stream associated with the parameter.
        The mechanism to advance the input stream depends on the input type.
        For input streams, the stream is advanced.
        For IO1 and IO2, the value is assigned to the QUA variable.
        For dgx, the value is polled.
        Args: start_with_dummy_packet: Boolean indicating if the query should start with a dummy packet
            (relevant only for dgx when called after another parameter loading).

        """
        if self.input_type is None:
            raise ValueError("No input type specified")
        elif self.input_type == "input_stream":
            qua_advance_input_stream(self.var)
        elif self.input_type in ["IO1", "IO2"]:
            io = IO1 if self.input_type == "IO1" else IO2
            if self.is_array:
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    pause()
                    assign(self.var[i], io)
            else:
                pause()
                assign(self.var, io)
        elif self.input_type == "dgx":
            if start_with_dummy_packet:
                dummy_packet = declare(int, [-3, -1])
                stream_dgx(dummy_packet)
            if self.is_array:
                i = self._counter_var
                with for_(i, 0, i < self.length, i + 1):
                    poll_dgx(self._var_int[i])
                    if self.type == fixed:
                        assign(
                            self.var[i],
                            Cast.mul_fixed_by_int(
                                1 / float_to_int_scaling_factor, self._var_int[i]
                            ),
                        )
                    elif self.type == bool:
                        assign(self.var[i], Cast.to_bool(self._var_int[i]))
                    else:
                        assign(self.var[i], self._var_int[i])
                    with if_(i < self.length - 1):
                        dummy_packet = declare(int, [-3, -1])
                        stream_dgx(dummy_packet)
            else:
                poll_dgx(self._var_int)
                if self.type == fixed:
                    assign(
                        self.var,
                        Cast.mul_fixed_by_int(
                            1 / float_to_int_scaling_factor, self._var_int
                        ),
                    )
                elif self.type == bool:
                    assign(self.var, Cast.to_bool(self._var_int))
                else:  # int
                    assign(self.var, self._var_int)

        else:
            raise ValueError("Invalid input stream type.")

    def push_to_opx(
        self,
        value: Union[int, float, bool, Sequence[Union[int, float, bool]]],
        job: RunningQmJob,
        qm: Optional[QuantumMachine] = None,
        dgx_lib=None,
        dgx_stream=None,
        verbosity: int = 0,
        start_with_dummy_packet: bool = False,
    ):
        """
        To be outside QUA program: pass an input value to the OPX from Python
        Args:
            value: Value to be passed to the OPX.
            job: RunningQmJob object (required if input_type is IO1 or IO2 or input_stream).
            qm: QuantumMachine object (required if input_type is IO1 or IO2).
            dgx_lib: DGX library (required if input_type is dgx).
            dgx_stream: DGX stream (required if input_type is dgx).
            verbosity: Verbosity level (0, 1, 2).
            start_with_dummy_packet: Boolean indicating if the query should start with a dummy packet
                (relevant only for dgx when called after another parameter loading).
        """

        if self.is_array and len(value) != self.length:
            raise ValueError(
                f"Invalid input. {self.name} should be a list of length {self.length}."
            )
        elif not self.is_array and not isinstance(value, (int, float, bool)):
            raise ValueError(
                f"Invalid input. {self.name} should be a single value (received {type(value)})."
            )

        if self.is_array:
            value = list(value)

        if self.input_type == "IO1" or self.input_type == "IO2":
            io = "set_io1_value" if self.input_type == "IO1" else "set_io2_value"
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

        elif self.input_type == "input_stream":
            job.push_to_input_stream(self.name, value)

        elif self.input_type == "dgx":
            if start_with_dummy_packet:
                dummy_packet = dgx_lib.DgxStream_pop(dgx_stream)
                if verbosity > 1:
                    print(f"GH <- dummy ")
            if self.is_array:
                if self.type == fixed:
                    for i in range(self.length):
                        if verbosity > 1:
                            print(f"----actions-----")
                            print(f"GH -> {value[i]}")
                        dgx_lib.DgxStream_push(
                            dgx_stream, int(value[i] * float_to_int_scaling_factor)
                        )
                        # If not last element, push a dummy packet
                        if i < self.length - 1:
                            dummy_packet = dgx_lib.DgxStream_pop(dgx_stream)
                            if verbosity > 1:
                                print(f"GH <- dummy ")
                elif self.type == bool:
                    for i in range(self.length):
                        dgx_lib.DgxStream_push(dgx_stream, int(value[i]))
                        # If not last element, push a dummy packet
                        if i < self.length - 1:
                            dummy_packet = dgx_lib.DgxStream_pop(dgx_stream)
                            if verbosity > 1:
                                print(f"GH <- dummy ")
                else:
                    for i in range(self.length):
                        dgx_lib.DgxStream_push(dgx_stream, int(value[i]))
                        # If not last element, push a dummy packet
                        if i < self.length - 1:
                            dummy_packet = dgx_lib.DgxStream_pop(dgx_stream)
                            if verbosity > 1:
                                print(f"GH <- dummy ")
            else:
                if self.type == fixed:
                    dgx_lib.DgxStream_push(
                        dgx_stream, int(value * float_to_int_scaling_factor)
                    )
                else:  # int or bool
                    dgx_lib.DgxStream_push(dgx_stream, int(value))
