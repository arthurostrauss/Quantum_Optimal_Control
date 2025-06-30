from typing import List, Union, Sequence, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametervector import (
    ParameterVectorElement,
    Parameter,
    ParameterVector,
)

from qm.qua import declare, assign, while_, if_, fixed, Cast, Util, for_, Random

import numpy as np
from qm.qua._expressions import QuaArrayVariable
from quam.components.quantum_components import Qubit, QubitPair
from quam.utils.qua_types import QuaVariableInt, Scalar, ScalarInt


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
                raise ValueError(f"Parameter {parameter.name} not found in the parameter table")
            elif qc.metadata["parameter_table"].table[parameter.name].length != 0:
                raise ValueError(
                    f"Parameter {parameter.name} is not a scalar in the parameter table"
                )


def clip_qua(var: Scalar, min_val: Scalar, max_val: Scalar):
    """
    Clip a QUA variable to a given range.
    :param var: QUA variable to clip
    :param min_val: Minimum value to clip to
    :param max_val: Maximum value to clip to
    :return: Clipped QUA variable
    """
    assign(var, Util.cond(var < min_val, min_val, var))
    assign(var, Util.cond(var > max_val, max_val, var))


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
        value=[(np.sqrt(-2 * np.log(x / (n_lookup + 1))).tolist()) for x in range(1, n_lookup + 1)],
    )
    return n_lookup, cos_array, ln_array


def rand_gauss_moller_box(mean: QuaArrayVariable,
                          std: QuaArrayVariable,
                          rand: Random,
                          z1: QuaArrayVariable,
                          z2: QuaArrayVariable,
                          lower_bound: Optional[QuaArrayVariable]=None,
                          upper_bound: Optional[QuaArrayVariable]=None) -> (QuaArrayVariable, QuaArrayVariable):
    """
    Return two random numbers using muller box
    """
    n_lookup, cos_array, ln_array = get_gaussian_sampling_input()
    i = declare(int)
    u1 = declare(int)
    u2 = declare(int)
    u = declare(fixed)
    with for_(i, 0, i<mean.length(), i + 1):
        assign(u, rand.rand_fixed())
        assign(u1, Cast.unsafe_cast_int(u >> 19))
        assign(u2, Cast.unsafe_cast_int(u) & ((1 << 19) - 1))
        assign(z1[i], mean[i] + std[i] * ln_array[u1] * cos_array[u2 & (n_lookup - 1)])
        assign(z2[i], mean[i] + std[i] * ln_array[u1] * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)])
        if lower_bound is not None and upper_bound is not None:
            clip_qua(z1[i], lower_bound[i], upper_bound[i])
            clip_qua(z2[i], lower_bound[i], upper_bound[i])

    return z1, z2


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
        if (max_tries is None) or (not float(max_tries).is_integer()) or (max_tries < 1):
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
            qubit.xy.play("x180", condition=(Ig > threshold))
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
