"""
Parameter Table: Class enabling the mapping of parameters to be updated to their corresponding 
"to-be-declared" QUA variables.

Author: Arthur Strauss - Quantum Machines
Created: 25/11/2024
"""

from typing import Optional, List, Dict, Union, Tuple, Literal, Callable
import numpy as np
from qiskit.circuit.classical.types import Uint, Bool
from qm.qua import *
from .parameter import Parameter
from qiskit.circuit import QuantumCircuit, Parameter as QiskitParameter
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement


class ParameterTable:
    """
    Class enabling the mapping of parameters to be updated to their corresponding "to-be-declared" QUA variables. The
    type of the QUA variable to be adjusted is automatically inferred from the type of the initial_parameter_value.
    Each parameter in the dictionary should be given a name that the user can then easily access through the table
    with table[parameter_name]. Calling this will return the QUA variable built within the QUA program corresponding
    to the parameter name and its associated Python initial value. Args: parameters_dict: Dictionary of the form {
    "parameter_name": initial_parameter_value }. the QUA program.
    """

    def __init__(
        self,
        parameters_dict: Union[
            Dict[
                str,
                Union[
                    Tuple[
                        Union[float, int, bool, List, np.ndarray],
                        Optional[Union[str, type]],
                        Optional[Literal["input_stream", "dgx", "IO1", "IO2"]],
                    ],
                    Union[float, int, bool, List, np.ndarray],
                ],
            ],
            List[Parameter],
        ],
    ):
        """
        Class enabling the mapping of parameters to be updated to their corresponding "to-be-declared" QUA variables.
        The type of the QUA variable to be adjusted can be specified or either be automatically inferred from the
        type of the initial_parameter_value. Each parameter in the dictionary should be given a name that the user
        can then easily access through the table with table[parameter_name]. Calling this will return the QUA
        variable built within the QUA program corresponding to the parameter name and its associated Python initial
        value.

        Args:
            parameters_dict: Dictionary should be of the form
            { "parameter_name": (initial_value, qua_type, Literal["input_stream"]) }
            where qua_type is the type of the QUA variable to be declared (int, fixed, bool)
             and the last (optional) field indicates if the variable should be declared as an input_stream instead
             of a standard QUA variable.
            Can also be a list of pre-declared ParameterValue objects.


        """
        self.table = {}
        if isinstance(parameters_dict, Dict):
            for index, (parameter_name, parameter_value) in enumerate(
                parameters_dict.items()
            ):
                input_type = None
                if isinstance(parameter_value, Tuple):
                    assert (
                        len(parameter_value) <= 3
                    ), "Invalid format for parameter value."
                    assert isinstance(
                        parameter_value[0], (int, float, bool, List, np.ndarray)
                    ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."
                    if len(parameter_value) >= 2:
                        assert (
                            isinstance(parameter_value[1], (str, type))
                            or parameter_value[1] is None
                            or parameter_value[1] == fixed
                        ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."

                    if len(parameter_value) == 3:
                        assert parameter_value[2] in [
                            "input_stream",
                            "dgx",
                            "IO1",
                            "IO2",
                        ], "Invalid format for input type (choose from 'input_stream', 'dgx', 'IO1', 'IO2')."
                        input_type = parameter_value[2]

                    self.table[parameter_name] = Parameter(
                        parameter_name,
                        parameter_value[0],
                        parameter_value[1],
                        input_type,
                    )
                    self.table[parameter_name].index = index
                else:
                    assert isinstance(
                        parameter_value, (int, float, bool, List, np.ndarray)
                    ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."
                    self.table[parameter_name] = Parameter(
                        parameter_name, parameter_value
                    )
                    self.table[parameter_name].index = index
        elif isinstance(parameters_dict, List):
            for index, parameter_value in enumerate(parameters_dict):
                assert isinstance(
                    parameter_value, Parameter
                ), "Invalid format for parameter value. Please use Parameter object."
                self.table[parameter_value.name] = parameter_value
                parameter_value.index = index

    def declare_variables(
        self, pause_program=False, declare_streams=True
    ) -> Union[QuaVariableType, List[QuaVariableType]]:
        """
        QUA Macro to declare all QUA variables associated with the parameter table.
        Should be called at the beginning of the QUA program.
        Args:
            pause_program: Boolean indicating if the program should pause after declaring the variables.
            declare_streams: Boolean indicating if output streams should be declared for all the parameters.

        """
        for parameter_name, parameter in self.table.items():
            parameter.declare_variable(declare_stream=declare_streams)
        if pause_program:
            pause()
        if len(self.variables) == 1:
            return self.variables[0]
        else:
            return self.variables

    def load_input_values(
        self, filter_function: Optional[Callable[[Parameter], bool]] = None
    ):
        """
        Load all the input values of the parameters in the parameter table.
        Args: filter_func: Optional function to filter the parameters to be loaded.
        """
        if filter_function is not None:
            for parameter in self.parameters:
                if filter_function(parameter):
                    parameter.load_input_value()
        else:
            for i, parameter in enumerate(self.parameters):
                if parameter.input_type is not None:
                    parameter.load_input_value()

    def save_to_stream(self):
        """
        Save all the parameters in the parameter table to their associated output streams.
        """
        for parameter in self.parameters:
            if parameter.is_declared and parameter.stream is not None:
                parameter.save_to_stream()

    def stream_processing(self):
        """
        Process all the streams in the parameter table.
        """
        for parameter in self.parameters:
            if parameter.stream is not None:
                parameter.stream_processing()

    def assign_parameters(
        self,
        values: Dict[
            Union[str, Parameter],
            Union[int, float, bool, List, np.ndarray, QuaExpressionType],
        ],
    ):
        """
        Assign values to the parameters of the parameter table within the QUA program.
        Args: values: Dictionary of the form { "parameter_name": parameter_value }. The parameter value can be either
        a Python value or a QuaExpressionType.
        """
        for parameter_name, parameter_value in values.items():
            if (
                isinstance(parameter_name, str)
                and parameter_name not in self.table.keys()
            ):
                raise KeyError(
                    f"No parameter named {parameter_name} in the parameter table."
                )
            if isinstance(parameter_name, str):
                self.table[parameter_name].assign_value(parameter_value)
            else:
                if not isinstance(parameter_name, Parameter):
                    raise ValueError(
                        "Invalid parameter name. Please use a string or a ParameterValue object."
                    )
                assert (
                    parameter_name in self.parameters
                ), "Provided ParameterValue not in this ParameterTable."
                parameter_name.assign_value(parameter_value)

    def print_parameters(self):
        text = ""
        for parameter_name, parameter in self.table.items():
            text += f"{parameter_name}: {parameter.value}, \n"
        print(text)

    def get_type(self, parameter: Union[str, int]):
        """
        Get the type of a specific parameter in the parameter table (specified by name or index).

        Args: parameter: Name or index (within current table) of the parameter to get the type of.

        Returns: Type of the parameter in the parameter table.
        """
        if isinstance(parameter, str):
            if parameter not in self.table.keys():
                raise KeyError(
                    f"No parameter named {parameter} in the parameter table."
                )
            return self.table[parameter].type
        elif isinstance(parameter, int):
            for param in self.parameters:
                if param.index == parameter:
                    return param.type

    def get_index(self, parameter_name: str):
        """
        Get the index of a specific parameter in the parameter table.
        Args: parameter_name: Name of the parameter to get the index of.
        Returns: Index of the parameter in the parameter table.
        """
        if parameter_name not in self.table.keys():
            raise KeyError(
                f"No parameter named {parameter_name} in the parameter table."
            )
        return self.table[parameter_name].index

    def get_parameter(self, parameter: Union[str, int]) -> Parameter:
        """
        Get the Parameter object of a specific parameter in the parameter table.
        This object contains the QUA variable corresponding to the parameter, its type,
        its index within the current table.

        Args: parameter: Name or index (within current table) of the parameter to be returned.

        Returns: ParameterValue object corresponding to the specified input.
        """
        if isinstance(parameter, str):
            if parameter not in self.table.keys():
                raise KeyError(
                    f"No parameter named {parameter} in the parameter table."
                )
            return self.table[parameter]
        elif isinstance(parameter, int):
            for param in self.parameters:
                if param.index == parameter:
                    return param

            raise IndexError(
                f"No parameter with index {parameter} in the parameter table."
            )
        else:
            raise ValueError("Invalid parameter name. Please use a string or an int.")

    def get_variable(self, parameter: Union[str, int]):
        """
        Get the QUA variable corresponding to the specified parameter name.

        Args: parameter: Name or index (within current table) of the parameter to be returned.
        Returns: QUA variable corresponding to the parameter name.

        """
        if isinstance(parameter, str):
            if parameter not in self.table.keys():
                raise KeyError(
                    f"No parameter named {parameter} in the parameter table."
                )
            return self.table[parameter].var
        elif isinstance(parameter, int):
            for param in self.parameters:
                if param.index == parameter:
                    return param.var

    def add_parameter(self, parameter: Union[Parameter, List[Parameter]]):
        """
        Add a (list of) parameter(s) to the parameter table. The index of the parameter is automatically set to the
        next available index in the table.
        Args: parameter_value: (List of) ParameterValue(s) object(s) to be added to current parameter table.
        """
        if isinstance(parameter, List):
            for parameter in parameter:
                if not isinstance(parameter, Parameter):
                    raise ValueError(
                        "Invalid parameter type. Please use a ParameterValue object."
                    )
                if parameter.name in self.table.keys():
                    raise KeyError(
                        f"Parameter {parameter.name} already exists in the parameter table."
                    )
                max_index = max([param.index for param in self.parameters])
                parameter.index = max_index + 1

                self.table[parameter.name] = parameter
        elif isinstance(parameter, Parameter):
            return self.add_parameter([parameter])

    def remove_parameter(self, parameter_value: Union[str, Parameter]):
        """
        Remove a parameter from the parameter table.
        Args: parameter_value: Name of the parameter to be removed or ParameterValue object to be removed.
        """
        if isinstance(parameter_value, str):
            if parameter_value not in self.table.keys():
                raise KeyError(
                    f"No parameter named {parameter_value} in the parameter table."
                )
            del self.table[parameter_value]
        elif isinstance(parameter_value, Parameter):
            if parameter_value not in self.parameters:
                raise KeyError("Provided ParameterValue not in this ParameterTable.")
            del self.table[parameter_value.name]
        else:
            raise ValueError(
                "Invalid parameter name. Please use a string or a ParameterValue object."
            )

    def add_table(
        self, parameter_table: Union[List["ParameterTable"], "ParameterTable"]
    ) -> "ParameterTable":
        """
        Add a parameter table to the current table.
        Args: parameter_table: ParameterTable object to be merged with the current table.
        """
        if isinstance(parameter_table, ParameterTable):
            return self.add_table([parameter_table])
        elif isinstance(parameter_table, List):
            for table in parameter_table:
                for parameter in table.table.values():
                    if parameter.name in self.table.keys():
                        raise KeyError(
                            f"Parameter {parameter.name} already exists in the parameter table."
                        )
                    self.table[parameter.name] = parameter

        else:
            raise ValueError(
                "Invalid parameter table. Please use a ParameterTable object "
                "or a list of ParameterTable objects."
            )

        return self

    def __contains__(self, item):
        return item in self.table.keys()

    def __iter__(self):
        return iter(self.table.keys())

    def __setitem__(self, key, value):
        """
        Assign values to the parameters of the parameter table within the QUA program.
        Args: key: Name of the parameter to be assigned. value: Value to be assigned to the parameter.
        """
        if key not in self.table.keys():
            raise KeyError(f"No parameter named {key} in the parameter table.")
        self.table[key].assign_value(value)

    def __getitem__(self, item: Union[str, int]):
        """
        Returns the QUA variable corresponding to the specified parameter name or parameter index.
        """
        if isinstance(item, str):
            if item not in self.table.keys():
                raise KeyError(f"No parameter named {item} in the parameter table.")
            if self.table[item].is_declared:
                return self.table[item].var
            else:
                raise ValueError(
                    f"No QUA variable found for parameter {item}. Please use "
                    f"ParameterTable.declare_variables() within QUA program first."
                )
        elif isinstance(item, int):
            for parameter in self.table.values():
                if parameter.index == item:
                    if parameter.is_declared:
                        return parameter.var
                    else:
                        raise ValueError(
                            f"No QUA variable found for parameter {item}. Please use "
                            f"ParameterTable.declare_variables() within QUA program first."
                        )
            raise IndexError(f"No parameter with index {item} in the parameter table.")
        else:
            raise ValueError("Invalid parameter name. Please use a string or an int.")

    def __len__(self):
        return len(self.table)

    def __getattr__(self, item):
        # Get the QUA variable corresponding to the specified parameter name.
        if item in self.table.keys():
            return self.table[item].var
        else:
            raise AttributeError(f"No attribute named {item} in the parameter table.")

    @property
    def variables(self):
        """
        List of the QUA variables corresponding to the parameters in the parameter table.
        """

        return [self[item] for item in self.table.keys()]

    @property
    def variables_dict(self):
        """Dictionary of the QUA variables corresponding to the parameters in the parameter table."""
        if not self.is_declared:
            raise ValueError(
                "Not all parameters have been declared. Please declare all parameters first."
            )
        return {
            parameter_name: parameter.var
            for parameter_name, parameter in self.table.items()
        }

    @property
    def parameters(self):
        """
        List of the parameter values objects in the parameter table.

        Returns: List of ParameterValue objects in the parameter table.
        """
        return list(self.table.values())

    @property
    def is_declared(self):
        """Boolean indicating if all the QUA variables have been declared."""
        return all(parameter.is_declared for parameter in self.parameters)

    def __repr__(self):
        text = ""
        for parameter in self.table.values():
            text += parameter.__repr__()
        return text

    @classmethod
    def from_quantum_circuit(
        cls,
        qc: QuantumCircuit,
        input_type: Literal["input_stream", "dgx", "IO1", "IO2"] = None,
    ) -> "ParameterTable":
        """
        Create a ParameterTable object from a QuantumCircuit object (and stores it in circuit metadata).
        Args: qc: QuantumCircuit object to be converted to a ParameterTable object.
        """
        param_list = []
        for parameter in qc.parameters:
            if isinstance(parameter, QiskitParameter):
                param_list.append(Parameter(parameter.name, 0.0, input_type=input_type))
            elif isinstance(parameter, ParameterVectorElement):
                raise ValueError(
                    "ParameterVectors are not yet supported "
                    "(Reason: Qiskit exporter to OpenQASM3 does not "
                    "support it. Please use individual parameters instead."
                )

        for var in qc.iter_input_vars():
            if var.type.kind == Uint:
                param_list.append(Parameter(var.name, 0, input_type=input_type))
            elif var.type.kind == Bool:
                param_list.append(Parameter(var.name, False, input_type=input_type))
            else:  # Float
                param_list.append(Parameter(var.name, 0.0, input_type=input_type))

        return cls(param_list)
