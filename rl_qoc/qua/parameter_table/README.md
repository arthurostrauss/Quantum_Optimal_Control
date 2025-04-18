# Parameter and ParameterTable Usage

This document explains how to use the `Parameter` and `ParameterTable` classes provided in the `rl_qoc.qua.parameter_table` module (or similar path). These classes facilitate the management of dynamic parameters within QUA programs, allowing for easier updates and interactions between Python and the OPX, especially for applications like Reinforcement Learning (RL) or Quantum Optimal Control (QOC), including integration with DGX via OPNIC.

We introduce the `Parameter` and `ParameterTable` classes, which are used to facilitate the dynamic parameter update in the QUA program.

## `Parameter` Class

The `Parameter` class represents a single dynamic parameter that maps to a QUA variable. It handles type inference, declaration within QUA, and mechanisms for updating its value from Python or reading its value back.

### Initialization

You create a `Parameter` instance by providing a name and an initial value. The QUA type (`fixed`, `int`, `bool`) can be inferred from the value or specified explicitly. You can also specify how the parameter interacts with the outside world using `input_type` and `direction` (for DGX).


## `ParameterTable` Class

The `ParameterTable` groups multiple `Parameter` objects, simplifying their management, especially when they share the same `input_type` (like `DGX`).

### Initialization

The user can declare a `ParameterTable` instance by passing in a dictionary of parameters intended to be updated dynamically, or a list of pre-defined `Parameter` objects.

**Using a Dictionary:**

The dictionary should be of the form:
`{'parameter_name': (initial_value, qua_type, input_type, direction)}`
or simpler forms where types/inputs are inferred or omitted.


*   `qua_type` can be `bool`, `int`, or `fixed`.
*   It is possible to provide a list or a 1D numpy array as `initial_value` to create a QUA array.
*   The optional `input_type` (`InputType.INPUT_STREAM`, `InputType.IO1`, `InputType.IO2`, `InputType.DGX`) specifies how the parameter interacts externally.
*   The optional `direction` (`Direction.INCOMING`, `Direction.OUTGOING`) is required if `input_type` is `DGX`.
*   If only a value is provided, the type is inferred.

The declaration of the `ParameterTable` can be done as follows:


**Using a List of Parameters:**

This is useful for ensuring consistent `input_type` and `direction`, especially for DGX.

**Important:** When using `InputType.DGX`, all parameters within a table *must* share the same `direction`.

Once this declaration is done, the provided dictionary or list is converted to a `ParameterTable` instance. The `ParameterTable` class serves as an interface between the QUA program and the Python environment, facilitating the declaration and manipulation of QUA variables.

## Usage in QUA Programs

The `Parameter` and `ParameterTable` classes provide methods (QUA macros) to interact with the parameters within a `with program():` block.

### Declaring Variables

Variables must be declared before use.

**For `ParameterTable`:**
Use `declare_variables()` to declare all parameters in the table. This method should ideally be called at the beginning of the QUA program scope.
*   `declare_variables(pause_program=False, declare_streams=True)`:
    *   `pause_program`: If `True`, pauses the QUA program immediately after declaration.
    *   `declare_streams`: If `True` (default), declares a standard QUA `stream` for each parameter for saving results.
    *   **Returns:** A tuple of the declared QUA variables, or the declared QUA struct if it's a DGX table.

**For individual `Parameter`:**
Use `declare_variable()`.
*   `declare_variable(pause_program=False, declare_stream=True)`: Arguments and return value are similar to the table version, but for a single parameter.

### Assigning Values within QUA

You can change parameter values directly within the QUA program.

**For `ParameterTable`:**
Use `assign_parameters()` for multiple assignments or standard dictionary/attribute assignment for single parameters.
*   `assign_parameters(values: Dict[Union[str, Parameter], ...])`: Assigns multiple values specified in the dictionary.

**For individual `Parameter`:**
Use `assign_value()`.
*   `assign_value(value, is_qua_array=False, condition=None, value_cond=None)`:
    *   `value`: The new value (literal, QUA variable/expression, list, or QUA array).
    *   `is_qua_array`: Set to `True` if `value` is a QUA array (for array parameters).
    *   `condition`: A QUA boolean expression. Assignment only happens if `True`.
    *   `value_cond`: The value assigned if `condition` is `False`. Must be provided if `condition` is provided.

### Loading Input Values from Python/External Sources

Use `load_input_values()` (for Table) or `load_input_value()` (for Parameter) to update parameters from their defined input source (`INPUT_STREAM`, `IO1`, `IO2`, `DGX`).
*   `load_input_values()` (Table): Calls `load_input_value()` for each parameter with an `input_type` (unless filtered). For DGX OUTGOING tables, receives the packet.
*   `load_input_value()` (Parameter): Behavior depends on `Parameter.input_type`.

### Sending Values to Python/External Sources

Use `send_to_python()` to make the parameter's current value available externally (via IO, DGX, or standard QUA stream).
*   `send_to_python()` (Table/Parameter): Behavior depends on `Parameter.input_type`. For DGX INCOMING tables, sends the packet.

### Saving to QUA Streams

Explicitly save parameter values to standard QUA streams for later retrieval using result analysis tools.
*   `save_to_stream()` (Table/Parameter): Saves current value(s) to associated stream(s).
*   `stream_processing()` (Table/Parameter): Defines how stream data is handled (e.g., `save`, `save_all`).

### Accessing Parameters and Variables

There are methods to access the underlying `Parameter` objects or their QUA variables. Each `Parameter` within a `ParameterTable` also holds an `index` attribute corresponding to its position in the table (starting from 0). This index is particularly useful within QUA for conditional logic, such as selecting which parameter to update using a `switch_` statement.
*   `get_parameter(parameter: Union[str, int]) -> Parameter`: Returns the `Parameter` object by name or index.
*   `parameter_table[key]` or `parameter_table.attribute`: Accesses the QUA variable directly within QUA (after declaration).
*   `parameter.var`: Accesses the QUA variable of a `Parameter` object within QUA (after declaration).
*   `parameter.index`: Returns the integer index of the `Parameter` within the `ParameterTable`. Useful for `switch_` statements in QUA.

## Interaction from Python (Outside QUA)

Use these methods in your Python script to interact with the running QUA program.

### Pushing Values to OPX (`push_to_opx`)

Send data *from* Python *to* the OPX, corresponding to a `load_input_value()` or `load_input_values()` call in QUA.
*   `push_to_opx(value_or_dict, job, qm=None, verbosity=1)`:
    *   For `Parameter`: `value` is the value/sequence to send.
    *   For `ParameterTable`: `param_dict` maps names to values. For DGX OUTGOING, *all* parameters must be in the dict.
    *   `job`: The `RunningQmJob` object.
    *   `qm`: The `QuantumMachine` object (required for `IO1`/`IO2`).

### Fetching Values from OPX (`fetch_from_opx`)

Retrieve data *from* the OPX *to* Python, corresponding to `send_to_python()` (for IO/DGX) or stream saving.
*   `fetch_from_opx(job, qm=None, verbosity=1)`:
    *   For `Parameter`: Fetches value based on `input_type`.
    *   For `ParameterTable`: Fetches values based on `input_type`. For DGX INCOMING, reads one packet.
    *   **Returns:** The fetched value/sequence (Parameter) or a dictionary (Table).

## DGX Integration (`DGXParameterPool`)

When using `InputType.DGX`, the `Parameter` and `ParameterTable` classes interact with the `DGXParameterPool`. This pool manages unique stream IDs required for OPNIC communication and handles the necessary patching and configuration of the underlying `opnic_wrapper`.

### Typical Workflow

1.  **Define Parameters/Tables:** Create your `Parameter` or `ParameterTable` instances with `input_type=InputType.DGX` and the correct `direction`. This automatically registers them with the `DGXParameterPool`.
2.  **Initialize Streams:** Before running the QUA program, call `DGXParameterPool.initialize_streams()`. This patches the wrapper code and configures the OPNIC streams.
3.  **Execute QUA Job:** Run the QUA program containing `declare_variables()`, `load_input_values()` (for OUTGOING tables), and `send_to_python()` (for INCOMING tables).
4.  **Python Interaction:** Use `table.push_to_opx()` (OUTGOING) or `table.fetch_from_opx()` (INCOMING) in your Python script.
5.  **Close Streams:** After interaction, call `DGXParameterPool.close_streams()`.

These classes provide a structured and flexible way to handle dynamic parameters in QUA, streamlining the process of updating variables from Python and integrating external hardware like DGX. Remember to manage the lifecycle of DGX streams using DGXParameterPool when applicable.