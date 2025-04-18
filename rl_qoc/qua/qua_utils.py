from typing import List, Union, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametervector import (
    ParameterVectorElement,
    Parameter,
    ParameterVector,
)
from qiskit.pulse import Schedule, ScheduleBlock

from qm.qua import *

from .parameter_table import ParameterTable, Parameter as QuaParameter
import numpy as np
from quam.components.quantum_components import Qubit, QubitPair


def validate_parameter_table(qc: QuantumCircuit):
    """
    Validate the parameter table of the circuit.
    This assumes that the parameter table is already added to the circuit metadata.
    """
    if "parameter_table" not in qc.metadata:
        raise ValueError("No parameter table found in the circuit")
    for parameter in qc.parameters:
        if isinstance(parameter, ParameterVectorElement):
            if parameter.name in qc.metadata["parameter_table"].table:
                continue
            if parameter.vector.name not in qc.metadata["parameter_table"].table:
                raise ValueError(
                    f"ParameterVector {parameter.vector.name} not found in the parameter table"
                )
            elif (
                len(parameter.vector)
                != qc.metadata["parameter_table"].table[parameter.vector.name].length
            ):
                raise ValueError(
                    f"ParameterVector {parameter.vector.name} has a different length than the one in the parameter table"
                )
        elif isinstance(parameter, Parameter):
            if parameter.name not in qc.metadata["parameter_table"].table:
                raise ValueError(
                    f"Parameter {parameter.name} not found in the parameter table"
                )
            elif qc.metadata["parameter_table"].table[parameter.name].length != 0:
                raise ValueError(
                    f"Parameter {parameter.name} is not a scalar in the parameter table"
                )


def add_parameter_table_to_circuit(qc: QuantumCircuit):
    """
    Add a parameter table to the circuit
    """

    # Check first if no parameter table is already added
    if "parameter_table" in qc.metadata:
        return qc
    param_dict = {}
    for parameter in qc.parameters:
        if (
            isinstance(parameter, ParameterVectorElement)
            and parameter.vector.name not in param_dict
        ):
            param_dict[parameter.vector.name] = [
                0.0 for _ in range(len(parameter.vector))
            ]
        elif isinstance(parameter, Parameter):
            param_dict[parameter.name] = 0.0

    # Create a ParameterTable object.
    param_table = ParameterTable(param_dict) if param_dict else None
    qc.metadata["parameter_table"] = param_table
    return qc, param_table


def parameter_table_from_qiskit(
    parameter_input: Union[
        ParameterVector, Sequence[Parameter], QuantumCircuit, Schedule, ScheduleBlock
    ]
):
    """
    Create a parameter table from a Qiskit object

    Args:
        parameter_input: The Qiskit object from which to create the parameter table

    Returns:
        The parameter table created from the Qiskit object
    """
    param_dict = {}
    if isinstance(parameter_input, QuantumCircuit):
        qc = parameter_input
        qc, parameter_table = add_parameter_table_to_circuit(qc)
        return parameter_table
    elif isinstance(parameter_input, (Schedule, ScheduleBlock)):
        for channel in list(
            filter(lambda ch: ch.is_parameterized(), parameter_input.channels)
        ):
            ch_params = list(channel.parameters)
            if len(ch_params) > 1:
                raise NotImplementedError(
                    "Only single parameterized channels are supported"
                )
            ch_param = ch_params[0]
            if ch_param.name not in param_dict:
                # Cast to int the parameter index
                param_dict[ch_param.name] = 0
        for param in parameter_input.parameters:
            if param.name not in param_dict:
                param_dict[param.name] = 0.0
    elif isinstance(parameter_input, (ParameterVector, Sequence[Parameter])):
        for parameter in parameter_input:
            if isinstance(parameter, ParameterVectorElement):
                if parameter.vector.name not in param_dict:
                    param_dict[parameter.vector.name] = [
                        0.0 for _ in range(len(parameter.vector))
                    ]
            elif isinstance(parameter, Parameter):
                param_dict[parameter.name] = 0.0
    return ParameterTable(param_dict) if param_dict else None


def clip_qua(param: QuaParameter, min_value, max_value):
    """
    Clip the QUA variable or QUA array between the min_value and the max_value
    :param param: The Parameter object containing the QUA variable or QUA array
    :param min_value: The minimum value
    :param max_value: The maximum value
    :return: The clipped QUA variable or QUA array
    """
    if param.length == 0:
        with if_(param.var < min_value):
            assign(param.var, min_value)
        with elif_(param.var > max_value):
            assign(param.var, max_value)
    else:
        i = declare(int)
        with for_(i, 0, i < param.length, i + 1):
            with if_(param.var[i] < min_value):
                assign(param.var[i], min_value)
            with elif_(param.var[i] > max_value):
                assign(param.var[i], max_value)


def clip_qua_var(param, min_value, max_value):
    """
    Clip the QUA variable between the min_value and the max_value
    :param param: The QUA variable
    :param min_value: The minimum value
    :param max_value: The maximum value
    :return: The clipped QUA variable
    """
    with if_(param < min_value):
        assign(param, min_value)
    with elif_(param > max_value):
        assign(param, max_value)


def clip_qua_array(param, min_value, max_value):
    """
    Clip the QUA array between the min_value and the max_value
    :param param: The QUA array
    :param min_value: The minimum value
    :param max_value: The maximum value
    :return: The clipped QUA array
    """
    i = declare(int)
    with for_(i, 0, i < param.length(), i + 1):
        with if_(param[i] < min_value):
            assign(param[i], min_value)
        with elif_(param[i] > max_value):
            assign(param[i], max_value)


def get_gaussian_sampling_input():
    """
    Get the input for the gaussian sampling function
    """
    n_lookup = 512

    cos_array = declare(
        fixed,
        value=[(np.cos(2 * np.pi * x / n_lookup).tolist()) for x in range(n_lookup)],
    )
    ln_array = declare(
        fixed,
        value=[
            (np.sqrt(-2 * np.log(x / (n_lookup + 1))).tolist())
            for x in range(1, n_lookup + 1)
        ],
    )
    return n_lookup, cos_array, ln_array


def rand_gauss_moller_box(z1, z2, mean, std, rand):
    """
    Return two random numbers using muller box
    """
    n_lookup = 512

    cos_array = declare(
        fixed,
        value=[(np.cos(2 * np.pi * x / n_lookup).tolist()) for x in range(n_lookup)],
    )
    ln_array = declare(
        fixed,
        value=[
            (np.sqrt(-2 * np.log(x / (n_lookup + 1))).tolist())
            for x in range(1, n_lookup + 1)
        ],
    )

    tmp = declare(fixed)
    u1 = declare(int)
    u2 = declare(int)
    assign(tmp, rand.rand_fixed())
    assign(u1, Cast.unsafe_cast_int(tmp >> 19))
    assign(u2, Cast.unsafe_cast_int(tmp) & ((1 << 19) - 1))
    assign(z1, mean + std * ln_array[u1] * cos_array[u2 & (n_lookup - 1)])
    assign(
        z2, mean + std * ln_array[u1] * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)]
    )
    return z1, z2


def prepare_input_state_pauli6(pauli_prep_indices, qubits: List[Qubit]):
    for i in range(len(qubits)):
        qubit = qubits[i]
        with switch_(pauli_prep_indices[i], unsafe=True):
            # TODO: Map this to PauliPrepBasis in Qiskit (for one qubit)
            with case_(0):
                Id(qubit)
            with case_(1):
                X(qubit)
            with case_(2):
                H(qubit)
            with case_(3):
                H(qubit)
                Z(qubit)
            with case_(4):
                H(qubit)
                S(qubit)
            with case_(5):
                H(qubit)
                Sdg(qubit)


def measure_observable(final_state, observable_indices, qubits: List[Qubit]):
    """
    Measure the observable given by the observable indices
    Observable indices is an array composed of integers given by the following mapping:
    0 -> I
    1 -> X
    2 -> Y
    3 -> Z

    For example, the indices [1, 1, 2, 3] would correspond to measuring the observable XXYZ on the 4 qubits in the list
    qubits.

    Each qubit in the list qubits is assumed to have a resonator with a readout operation defined in its operations
    dictionary. The readout operation should have a threshold defined in the dictionary. The threshold is the value that
    the I quadrature must be above in order to be considered in the excited state.



    Args:
        observable_indices: QUA array of integers representing the observable to measure on each qubit
        qubits: List of Transmon objects representing the qubits to measure the observable on

    Returns:

    """
    I, Q = declare(fixed, size=len(qubits)), declare(fixed, size=len(qubits))
    states = declare(bool, size=len(qubits))
    thresholds = [
        qubit.resonator.operations["readout"]["threshold"] for qubit in qubits
    ]
    for i in range(len(qubits)):
        qubit = qubits[i]
        with switch_(observable_indices[i], unsafe=True):
            with case_(0):  # Pauli I
                Id(qubit)
            with case_(1):  # Pauli X
                H(qubit)
            with case_(2):  # Pauli Y
                Sdg(qubit)
                H(qubit)
            with case_(3):  # Pauli Z
                Id(qubit)
        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
        assign(states[i], I[i] > thresholds[i])
        assign(final_state, final_state + 2**i * Cast.to_int(states[i]))
        reset_qubit(
            "active",
            qubit,
            threshold=thresholds[i],
            max_tries=5,
            Ig=I[i],
        )

    return final_state


def Id(qubit: Qubit):
    pass


def X(qubit: Qubit, condition=None):
    pass


def H(qubit: Qubit):
    pass


def Z(qubit: Qubit):
    pass


def S(qubit: Qubit):
    pass


def Sdg(qubit: Qubit):
    pass


def reset_qubit(method: str, qubit: Qubit, **kwargs):
    """
    Macro to reset the qubit state.

    If method is 'cooldown', then the variable cooldown_time (in clock cycles) must be provided as a python integer > 4.

    **Example**: reset_qubit('cooldown', cooldown_times=500)

    If method is 'active', then 3 parameters are available as listed below.

    **Example**: reset_qubit('active', threshold=-0.003, max_tries=3)

    :param method: Method the reset the qubit state. Can be either 'cooldown' or 'active'.
    :param qubit: The qubit to be addressed in QuAM
    :key cooldown_time: qubit relaxation time in clock cycle, needed if method is 'cooldown'. Must be an integer > 4.
    :key threshold: threshold to discriminate between the ground and excited state, needed if method is 'active'.
    :key max_tries: python integer for the maximum number of tries used to perform active reset,
        needed if method is 'active'. Must be an integer > 0 and default value is 1.
    :key Ig: A QUA variable for the information in the `I` quadrature used for active reset. If not given, a new
        variable will be created. Must be of type `Fixed`.
    :key pi_pulse: The pulse to play to get back to the ground state. Default is 'x180'.
    :return:
    """
    if method == "cooldown":
        # Check cooldown_time
        cooldown_time = kwargs.get("cooldown_time", None)
        if (cooldown_time is None) or (cooldown_time < 4):
            raise Exception("'cooldown_time' must be an integer > 4 clock cycles")
        # Reset qubit state
        qubit.xy.wait(cooldown_time)
    elif method == "active":
        # Check threshold
        threshold = kwargs.get("threshold", None)
        if threshold is None:
            raise Exception("'threshold' must be specified for active reset.")
        # Check max_tries
        max_tries = kwargs.get("max_tries", 1)
        if (
            (max_tries is None)
            or (not float(max_tries).is_integer())
            or (max_tries < 1)
        ):
            raise Exception("'max_tries' must be an integer > 0.")
        # Check Ig
        Ig = kwargs.get("Ig", None)
        # Reset qubit state
        return active_reset(threshold, qubit, max_tries=max_tries, Ig=Ig)


# Macro for performing active reset until successful for a given number of tries.
def active_reset(threshold: float, qubit: Qubit, max_tries=1, Ig=None):
    """Macro for performing active reset until successful for a given number of tries.

    :param threshold: threshold for the 'I' quadrature discriminating between ground and excited state.
    :param qubit: The qubit element. Must be defined in the config.
    :param resonator: The resonator element. Must be defined in the config.
    :param max_tries: python integer for the maximum number of tries used to perform active reset. Must >= 1.
    :param Ig: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :param pi_pulse: The pulse to play to get back to the ground state. Default is 'x180'.
    :return: A QUA variable for the information in the `I` quadrature and the number of tries after success.
    """
    if Ig is None:
        Ig = declare(fixed)
    if (max_tries < 1) or (not float(max_tries).is_integer()):
        raise Exception("max_count must be an integer >= 1.")
    # Initialize Ig to be > threshold
    assign(Ig, threshold + 2**-28)
    # Number of tries for active reset
    counter = declare(int)
    # Reset the number of tries
    assign(counter, 0)

    # Perform active feedback
    qubit.xy.align(qubit.resonator.name)
    # Use a while loop and counter for other protocols and tests
    with while_((Ig > threshold) & (counter < max_tries)):
        # Measure the resonator
        qubit.resonator.measure("readout")
        # Play a pi pulse to get back to the ground state
        with if_(Ig > threshold):
            qubit.xy.play("x180")
            X(qubit, condition=(Ig > threshold))
        # Increment the number of tries
        assign(counter, counter + 1)
    return Ig, counter


def binary(n, length):
    """
    Convert an integer to a binary string of a given length
    :param n: Integer to convert
    :param length: Length of the output string
    :return: Binary string corresponding to integer n
    """
    return bin(n)[2:].zfill(length)
