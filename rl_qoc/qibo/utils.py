from typing import Optional

import numpy as np
from qibo import gates, Circuit, set_backend
from qibo.transpiler import Passes
from qibo.transpiler.unroller import Unroller, NativeGates
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence
from qibolab.qubits import QubitId
from qibo.backends import GlobalBackend, Backend

AMP_MAX_LIM = 0.5
TIME_MAX_LIM = 20_000


def checks(pulse_params: dict):
    """Check that the pulse parameters respect the instruments bounds"""
    if np.abs(pulse_params[0]) > AMP_MAX_LIM:
        raise ValueError("Amplitude value too high")
    if pulse_params[1] > TIME_MAX_LIM:
        raise ValueError("Time value too high")


def new_cz_rule(
    gate: gates.Gate,
    platform: Platform,
    pulse_params: list,
    qubit_pair: tuple[QubitId],
):
    """CZ rule returning a custom flux pulse defined by `pulse_params`."""
    old_cz_flux = platform.pairs[qubit_pair].native_gates.CZ.pulses[0]
    pulse_sequence = PulseSequence()
    pulse_sequence.add(
        FluxPulse(
            start=0,
            amplitude=pulse_params[0],
            duration=pulse_params[1],
            shape="Rectangular",
            qubit=old_cz_flux.qubit.name,
        )
    )
    virtual_z_phases = {}
    return pulse_sequence, virtual_z_phases


def execute_action(
    platform: str,
    circuits: list[Circuit],
    hardware_qubit_pair: tuple[QubitId],
    pulse_params: list,
    nshots: int,
):
    """Adds a Controlled-Z (CZ) gate to `circuits` and execute them on the `platform`.

    The CZ gate is applied to the qubits in the quantum hardware specified
    by `hardware_qubit_pair`, where one qubit acts as the control and the
    other as the target.

    The parameters of the CZ gate's flux pulse, such as amplitude and
    duration, are defined, with this order, by the values provided in `pulse_params`.

    The function returns a list of shots according to `nshots`.

    """
    checks(pulse_params)
    set_backend("qibolab", platform=platform)
    backend = GlobalBackend()

    # Transpile circuits
    transpiler = dummy_transpiler(backend)
    qubit_map = hardware_qubit_pair

    # Change CZ pulse parameters
    compiler = backend.compiler
    cz_rule = lambda gate, platform: new_cz_rule(
        gate, platform, pulse_params, hardware_qubit_pair
    )
    compiler.register(gates.CZ)(cz_rule)
    _, results = execute_transpiled_circuits(
        circuits,
        [qubit_map] * len(circuits),
        backend,
        transpiler,
        nshots=nshots,
    )
    return [result.frequencies() for result in results]


def transpile_circuits(
    circuits: list[Circuit],
    qubit_maps: list[list[QubitId]],
    backend: Backend,
    transpiler: Optional[Passes],
) -> list[Circuit]:
    """Transpile and pad `circuits` according to the platform.

    Apply the `transpiler` to `circuits` and pad them in
    circuits with the same number of qubits in the platform.
    Before manipulating the circuits, this function check that the
    `qubit_maps` contain string ids and in the positive case it
    remap them in integers, following the ids order provided by the
    platform.

    .. note::

        In this function we are implicitly assuming that the qubit ids
        are all string or all integers.
    """
    transpiled_circuits = []

    qubits = list(backend.platform.qubits)
    if isinstance(qubit_maps[0][0], str):
        for i, qubit_map in enumerate(qubit_maps):
            qubit_map = map(lambda x: qubits.index(x), qubit_map)
            qubit_maps[i] = list(qubit_map)
    if backend.name == "qibolab":
        platform_nqubits = backend.platform.nqubits
        for circuit, qubit_map in zip(circuits, qubit_maps):
            new_circuit = pad_circuit(platform_nqubits, circuit, qubit_map)
            transpiled_circ, _ = transpiler(new_circuit)
            transpiled_circuits.append(transpiled_circ)
    else:
        transpiled_circuits = circuits
    return transpiled_circuits


def execute_transpiled_circuits(
    circuits: list[Circuit],
    qubit_maps: list[list[QubitId]],
    backend: Backend,
    transpiler: Optional[Passes],
    initial_states=None,
    nshots=1000,
):
    """Transpile `circuits`.

    If the `qibolab` backend is used, this function pads the `circuits` in new
    ones with a number of qubits equal to the one provided by the platform.
    At the end, the circuits are transpiled, executed and the results returned.
    The input `transpiler` is optional, but it should be provided if the backend
    is `qibolab`.
    For the qubit map look :func:`dummy_transpiler`.
    This function returns the list of transpiled circuits and the execution results.
    """
    transpiled_circuits = transpile_circuits(
        circuits,
        qubit_maps,
        backend,
        transpiler,
    )
    return transpiled_circuits, backend.execute_circuits(
        transpiled_circuits, initial_states=initial_states, nshots=nshots
    )


def dummy_transpiler(backend) -> Optional[Passes]:
    """
    If the backend is `qibolab`, a transpiler with just an unroller is returned,
    otherwise None.
    """
    if backend.name == "qibolab":
        unroller = Unroller(NativeGates.default())
        return Passes(connectivity=backend.platform.topology, passes=[unroller])
    return None


def pad_circuit(nqubits, circuit: Circuit, qubit_map: list[int]) -> Circuit:
    """
    Pad `circuit` in a new one with `nqubits` qubits, according to `qubit_map`.
    `qubit_map` is a list `[i, j, k, ...]`, where the i-th physical qubit is mapped
    into the 0th logical qubit and so on.
    """
    new_circuit = Circuit(nqubits)
    new_circuit.add(circuit.on_qubits(*qubit_map))
    return new_circuit
