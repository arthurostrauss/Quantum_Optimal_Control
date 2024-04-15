import math

from qiskit.circuit import ParameterExpression
from qm.qua._type_hinting import QuaVariableType, QuaArrayType
from qm.qua import *
from videomode import ParameterValue, ParameterTable
from qiskit import pulse


def clip_qua(param: ParameterValue, min_value, max_value):
    """
    Clip the QUA variable or QUA array between the min_value and the max_value
    :param param: The ParameterValue object containing the QUA variable or QUA array
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
    assign(u2, Cast.unsafe_cast_int((tmp & ((1 << 19) - 1)) << 10))
    assign(z1, mean + std * ln_array[u1] * cos_array[u2])
    assign(
        z2, mean + std * ln_array[u1] * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)]
    )
    return z1, z2


def get_el_from_channel(channel: pulse.channels.Channel):
    return ""


def get_pulse_from_instruction(
    pulse_instance: pulse.library.Pulse,
    channel: pulse.channels.Channel,
    channel_mapping: dict = None,
    parameter_table: ParameterTable = None,
    pulse_lib: dict = None,
):
    param_statement = {"pulse": pulse_lib[pulse_instance.name]}

    for param_name, param_value in pulse_instance.parameters.items():
        if param_name == "amp" and isinstance(param_value, ParameterExpression):
            amp_ = parameter_table[param_value]
            angle = pulse_instance.parameters.get("angle", None)
            if isinstance(angle, ParameterExpression):
                angle = parameter_table[angle]
            elif angle == 0:
                angle = None

            matrix_elements = (
                [
                    amp_ * Math.cos(angle),
                    -amp_ * Math.sin(angle),
                    amp_ * Math.sin(angle),
                    amp_ * Math.cos(angle),
                ]
                if angle is not None
                else [amp_]
            )
            param_statement["pulse"] *= amp(*matrix_elements)

        elif param_name == "duration" and isinstance(param_value, ParameterExpression):
            param_statement["duration"] = parameter_table[param_value]

    return param_statement


def schedule_to_qua_instructions(sched: pulse.Schedule):
    """
    Convert a Qiskit pulse schedule to a QUA program
    :param sched: The Qiskit pulse schedule
    """

    time_tracker = {channel: 0 for channel in sched.channels}
    for time, instruction in sched.instructions:
        channel = instruction.channels[0]
        if time_tracker[channel] < time:
            wait((time - time_tracker[channel]) // 4, get_el_from_channel(channel))
        time_tracker[channel] = time
        if isinstance(instruction, pulse.Play):
            play(
                **get_pulse_from_instruction(instruction.pulse, channel),
                element=get_el_from_channel(channel),
            )
        elif isinstance(instruction, pulse.ShiftPhase):
            frame_rotation(instruction.phase, get_el_from_channel(channel))
        elif isinstance(instruction, pulse.ShiftFrequency):
            update_frequency(get_el_from_channel(channel), instruction.frequency)
        elif isinstance(instruction, pulse.Delay):
            wait(instruction.duration // 4, get_el_from_channel(channel))
        else:
            raise ValueError(f"Unknown instruction {instruction}")


def prepare_input_state(pauli_indices, qubit_elements):
    q = declare(int)
    with for_(q, 0, q < pauli_indices.length(), q + 1):
        with switch_(q):
            for i in range(len(qubit_elements)):
                with case_(i):
                    qubit_el = qubit_elements[i]
                    with switch_(pauli_indices[q], unsafe=True):
                        with case_(0):
                            wait(4, qubit_el)
                        with case_(1):
                            X(qubit_el)
                        with case_(2):
                            H(qubit_el)
                        with case_(3):
                            H(qubit_el)
                            Z(qubit_el)
                        with case_(4):
                            H(qubit_el)
                            S(qubit_el)
                        with case_(5):
                            H(qubit_el)
                            Sdg(qubit_el)
