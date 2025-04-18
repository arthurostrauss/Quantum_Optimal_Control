"""
Parameter Table: Class enabling the mapping of parameters to be updated to their corresponding 
"to-be-declared" QUA variables.

Author: Arthur Strauss - Quantum Machines
Created: 25/11/2024
"""
import warnings
import sys
from itertools import chain
from typing import Optional, List, Dict, Union, Tuple, Literal, Callable
import numpy as np
from qiskit.circuit.classical.types import Uint, Bool
from qm import QuantumMachine
from qm.jobs.running_qm_job import RunningQmJob
from qm.qua import *
from qualang_tools.results import fetching_tool

from qiskit.circuit import QuantumCircuit, Parameter as QiskitParameter
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from .parameter_pool import DGXParameterPool
from .parameter import Parameter
from .input_type import InputType, Direction

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
                        Optional[Union[Literal["INPUT_STREAM", "DGX", "IO1", "IO2"], InputType]],
                        Optional[Union[Literal["INCOMING", "OUTGOING"], Direction]]
                    ],
                    Union[float, int, bool, List, np.ndarray],
                ],
            ],
            List[Parameter],
        ],
            name: Optional[str] = None,
    ):
        """
        Class enabling the mapping of parameters to be updated to their corresponding "to-be-declared" QUA variables.
        The type of the QUA variable to be adjusted can be specified or either be automatically inferred from the
        type of the initial_parameter_value. Each parameter in the dictionary should be given a name that the user
        can then easily access through the table with table[parameter_name]. Calling this will return the QUA
        variable built within the QUA program corresponding to the parameter name and its associated Python initial
        value.
        
        When initialized with a list of Parameter objects, the input type and direction are for all parameters in the
        list should be the same. The input type and direction are inferred from the first parameter in the list.
        

        Args:
            parameters_dict: Dictionary should be of the form
            { "parameter_name": (initial_value, qua_type, Literal["input_stream"]) }
            where qua_type is the type of the QUA variable to be declared (int, fixed, bool)
             and the last (optional) field indicates if the variable should be declared as an input_stream instead
             of a standard QUA variable.
            Can also be a list of pre-declared Parameter objects.
            name: Optional name for the parameter table


        """
        self.table = {}
        if name is not None:
            self.name = name
        else: # Generate a unique name
            self.name = f"ParameterTable_{id(self)}"
        self._input_type = None
        self._id = None
        self._qua_external_stream = None
        self._packet = None
        self._packet_type = None
        self._direction = None
        
        if isinstance(parameters_dict, Dict):
            for index, (parameter_name, parameter) in enumerate(
                parameters_dict.items()
            ):
                input_type = None
                direction = None
                if isinstance(parameter, Tuple):
                    assert (
                        len(parameter) <= 4
                    ), "Invalid format for parameter value."
                    assert isinstance(
                        parameter[0], (int, float, bool, List, np.ndarray)
                    ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."
                    if len(parameter) >= 2:
                        assert (
                            isinstance(parameter[1], (str, type))
                            or parameter[1] is None
                            or parameter[1] == fixed
                        ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."

                    if len(parameter) >= 3:
                        input_type = InputType(parameter[2]) if isinstance(parameter[2], str) else parameter[2]
                        if self._input_type is None:
                            self._input_type = input_type
                        elif self._input_type != input_type:
                            raise ValueError(
                                "All parameters in the table must have the same input type."
                            )
                        if input_type == InputType.DGX:
                            assert len(parameter) == 4, "Direction of the parameter is missing (required for DGX input)."
                            direction = Direction(parameter[3]) if isinstance(parameter[3], str) else parameter[3]
                            if self._direction is None:
                                self._direction = direction
                            elif self._direction != direction:
                                raise ValueError("All parameters in the table must have the same direction.")
                                
                        

                    self.table[parameter_name] = Parameter(
                        parameter_name,
                        parameter[0],
                        parameter[1],
                        input_type,
                        direction
                    )
                    self.table[parameter_name].index = index
                else:
                    assert isinstance(
                        parameter, (int, float, bool, List, np.ndarray)
                    ), "Invalid format for parameter value. Please use (initial_value, qua_type) or initial_value."
                    self.table[parameter_name] = Parameter(
                        parameter_name, parameter
                    )
                    self.table[parameter_name].index = index
        elif isinstance(parameters_dict, List):
            for index, parameter in enumerate(parameters_dict):
                assert isinstance(
                    parameter, Parameter
                ), "Invalid format for parameter value. Please use Parameter object."
                self.table[parameter.name] = parameter
                parameter.index = index
                if self._input_type is None:
                    self._input_type = parameter.input_type
                elif self._input_type != parameter.input_type:
                    raise ValueError(
                        "All parameters in the table must have the same input type."
                    )
                if self._input_type == InputType.DGX:
                    if self._direction is None :
                        self._direction = parameter.direction
                    elif self._direction != parameter.direction:
                        raise ValueError(
                            "All parameters in the table must have the same direction."
                        )
        
        if self.input_type == InputType.DGX:
            attributes = {
                parameter.name: QuaArray[parameter.type, parameter.length if parameter.is_array else 1]
                for parameter in self.parameters
            }
            struct = qua_struct(type("Struct", (object,), {"__annotations__": attributes}))
            self._packet_type = struct
            self._id = DGXParameterPool.get_id(self)
            for parameter in self.parameters:
                parameter.dgx_struct = struct
                parameter.stream_id = self._id
            

    def declare_variables(
        self, pause_program=False, declare_streams=True
    ):
        """
        QUA Macro to declare all QUA variables associated with the parameter table.
        Should be called at the beginning of the QUA program.
        Args:
            pause_program: Boolean indicating if the program should pause after declaring the variables.
            declare_streams: Boolean indicating if output streams should be declared for all the parameters.

        """
        if self.input_type == InputType.DGX:
            qua_direction = QuaStreamDirection.INCOMING if self.direction == Direction.OUTGOING else QuaStreamDirection.OUTGOING
            self._packet = declare_struct(self._packet_type)
            self._qua_external_stream = declare_external_stream(self._packet,
                                                                self._id,
                                                                qua_direction)
            
            for parameter in self.parameters:
                parameter._var = self._packet
                parameter._is_declared = True
                if parameter.is_array:
                    parameter._counter_var = declare(int)
                
                if declare_streams:
                    parameter.declare_stream()
            
            if self._direction == Direction.INCOMING: # OPX -> DGX (Initialize the packet)
                for parameter in self.parameters:
                    if parameter.is_array:
                    
                        for i in range(parameter.length):
                            assign(parameter.var[i], parameter.value[i])
                    else:
                        assign(parameter.var, parameter.value)
            
            if pause_program:
                pause()
                
            return self._packet
            
        else:
            for parameter in self.parameters:
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
        QUA Macro to load all the input values of the parameters in the parameter table.
        This macro is expected to work jointly with the use of push_to_opx method on the 
        Python side.
        Args: filter_func: Optional function to filter the parameters to be loaded.
        """
        if self.input_type == InputType.DGX:
            if filter_function is not None:
                warnings.warn("Filter function is not supported for DGX parameter tables.")
            if self.direction == Direction.INCOMING:
                raise ValueError("Cannot load input values for outgoing DGX parameter tables.")
            elif self.direction == Direction.OUTGOING:
                external_stream_receive(self._qua_external_stream, self._packet)
                
        else:   
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
            Union[int, float, bool, List, np.ndarray, Parameter],
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

    def get_index(self, parameter_name: Union[str, Parameter]):
        """
        Get the index of a specific parameter in the parameter table.
        Args: parameter_name: (Name of the) parameter to get the index of.
        Returns: Index of the parameter in the parameter table.
        """
        if isinstance(parameter_name, Parameter):
            return parameter_name.index if parameter_name in self.parameters else None
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
             
            raise IndexError(f"No parameter with index {parameter} in the parameter table.")
        else:
            raise ValueError(
                "Invalid parameter name. Please use a string or an int.")

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
        if self.is_declared:
            raise ValueError(
                "Cannot add parameters to a parameter table that has already been declared."
            )
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
                if parameter.input_type != self.input_type:
                    raise ValueError(
                        "All parameters in the table must have the same input type."
                    )
                parameter.stream_id = self._id
                parameter.dgx_struct = self._packet_type
                
                self.table[parameter.name] = parameter
        elif isinstance(parameter, Parameter):
            return self.add_parameter([parameter])

    def remove_parameter(self, parameter_value: Union[str, Parameter]):
        """
        Remove a parameter from the parameter table.
        Args: parameter_value: Name of the parameter to be removed or ParameterValue object to be removed.
        """
        if self.is_declared:
            raise ValueError(
                "Cannot remove parameters from a parameter table that has already been declared."
            )
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
                self.add_parameter(table.parameters)

        else:
            raise ValueError(
                "Invalid parameter table. Please use a ParameterTable object "
                "or a list of ParameterTable objects."
            )

        return self

    def __contains__(self, item: str|Parameter):
        if isinstance(item, str):
            return item in self.table.keys()
        elif isinstance(item, Parameter):
            return item in self.parameters
        else:
            raise ValueError("Invalid parameter name. Please use a string or a Parameter object.")

    def __iter__(self):
        return iter(self.table.values())

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
    
    @property
    def input_type(self):
        return self._input_type
    
    @property
    def packet(self):
        if not self.input_type ==InputType.DGX:
            raise ValueError("No packet declared for non-DGX parameter tables.")
        return self._packet
    
    @property
    def stream_id(self)->int:
        """
        Get the stream ID of the parameter table.
        Relevant for DGX parameter tables.
        """
        return self._id
    
    @property
    def direction(self)->Direction:
        """
        Get the direction of the parameter table.
        Relevant for DGX parameter tables.
        "INCOMING": OPX -> DGX
        "OUTGOING": DGX -> OPX
        Returns:

        """
        if self.input_type != InputType.DGX:
            raise ValueError("Direction is only relevant for DGX parameter tables.")
        return self._direction
    
    def push_to_opx(self, param_dict: Dict[Union[str, Parameter], Union[float, int, bool, List, np.ndarray]],
                    job: RunningQmJob,
                    qm: Optional[QuantumMachine] = None,
                    verbosity: int = 1):
        """
        Push the values of the parameters to the OPX.
        Args: param_dict: Dictionary of the form {parameter_name: parameter_value}.
        """
        if self.input_type != InputType.DGX:
            for parameter, value in param_dict.items():
                if isinstance(parameter, str):
                    if parameter not in self.table.keys():
                        raise KeyError(f"No parameter named {parameter} in the parameter table.")
                    self.table[parameter].push_to_opx(value, job, qm, verbosity)
                elif isinstance(parameter, Parameter):
                    if parameter not in self.parameters:
                        raise KeyError("Provided ParameterValue not in this ParameterTable.")
                    parameter.push_to_opx(value, job, qm, verbosity)
                else:
                    raise ValueError("Invalid parameter name. Please use a string or a ParameterValue object.")
        else:
            if self.direction == Direction.INCOMING:
                raise ValueError("Cannot push values to incoming DGX parameter tables.")
            # Check if all parameters are in the dictionary
            for parameter in self.parameters:
                if parameter not in param_dict.keys() and parameter.name not in param_dict.keys():
                    raise KeyError(f"Parameter {parameter.name} not found in the dictionary, all packet must be filled.")
            # Transform all values to fit the packet (convert to list if necessary)
            packet_dict = {parameter.name if isinstance(parameter, Parameter) else parameter: [param_dict[parameter]] if not parameter.is_array else param_dict[parameter] for parameter in self.parameters}
            # Convert potential numpy arrays to lists
            packet_dict = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in packet_dict.items()}
                    
            if DGXParameterPool.configured and DGXParameterPool.patched:
                if "opnic_wrapper" not in sys.modules:
                    sys.path.append('/home/dpoulos/aps_demo/python-wrapper/wrapper/build/python')
                from opnic_wrapper import OutgoingPacket, send_packet
                flattened_values = list(chain(*packet_dict.values()))
                packet = OutgoingPacket(flattened_values)
                send_packet(self.stream_id, packet)
            else:
                raise ValueError("OPNIC wrapper not configured or patched.")
            
            if verbosity > 1:
                print(f"Sent packet: {packet_dict}")
                
    def send_to_python(self):
        """
        This method uses IO variables if input type is IO1 or IO2, and
        external streams if input type is dgx.
        If the input type is input_stream, the values are saved to the streams
        """
        if self.input_type != InputType.DGX:
            for parameter in self.parameters:
                parameter.send_to_python()
        else:
            if self.direction == Direction.OUTGOING:
                raise ValueError("Cannot send values to outgoing DGX parameter tables.")
            external_stream_send(self._qua_external_stream, self._packet)
            
    def fetch_from_opx(self, job: RunningQmJob, qm: Optional[QuantumMachine] = None, verbosity: int = 1):
        """
        Fetch the values of the parameters from the OPX.
        The values are returned in a dictionary of the form {parameter_name: parameter_value}.
        
        Args: job: RunningQmJob object to fetch the values from (input stream).
                qm: QuantumMachine object to fetch the values from (IO variables).
                verbosity: Verbosity level of the fetching process.
        
        Returns: Dictionary of the form {parameter_name: parameter_value}.
        """
        param_dict = {}
        if self.input_type == InputType.IO1 or self.input_type == InputType.IO2:
            for parameter in self.parameters:
                value = parameter.fetch_from_opx(job, qm, verbosity)
                param_dict[parameter.name] = value
        elif self.input_type == InputType.INPUT_STREAM:
            results = fetching_tool(job, [param.name for param in self.parameters], mode='live')
            while results.is_processing():
                results = results.fetch_all()
            for parameter, result in zip(self.parameters, results):
                param_dict[parameter.name] = result
        else: # DGX
            if self.direction == Direction.OUTGOING:
                raise ValueError("Cannot fetch values from outgoing DGX parameter tables.")
            
            if DGXParameterPool.configured and DGXParameterPool.patched:
                if "opnic_wrapper" not in sys.modules:
                    sys.path.append('/home/dpoulos/aps_demo/python-wrapper/wrapper/build/python')
                from opnic_wrapper import read_packet, wait_for_packets
                wait_for_packets(self.stream_id, 1)
                packet = read_packet(self.stream_id, 0)
                for parameter in self.parameters:
                    param_dict[parameter.name] = getattr(packet, parameter.name)
            else:
                raise ValueError("OPNIC wrapper not configured or patched.")
            if verbosity > 1:
                print(f"Received packet: {param_dict}")
        return param_dict
                
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
