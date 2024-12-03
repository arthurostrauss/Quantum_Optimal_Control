from typing import Optional, Callable, Tuple

import numpy as np
from qibo import gates, Circuit
from qibo.gates import Gate
from qibo.transpiler import Passes
from qibo.transpiler.unroller import Unroller, NativeGates
from qibolab.platform import Platform
from qibolab import QibolabBackend
from qibolab.pulses import FluxPulse, PulseSequence
from qibolab.qubits import QubitId
from qibo.backends import Backend

AMP_MAX_LIM = 0.5
TIME_MAX_LIM = 20_000


def checks(pulse_params: dict):
    """Check that the pulse parameters respect the instruments bounds"""
    if np.abs(pulse_params[0]) > AMP_MAX_LIM:
        raise ValueError("Amplitude value too high")
    # if pulse_params[1] > TIME_MAX_LIM:
    #     raise ValueError("Time value too high")


def new_cz_rule(
    gate: gates.Gate,
    platform: Platform,
    pulse_params: list,
    targets: tuple[QubitId],
):
    """CZ rule returning a custom flux pulse defined by `pulse_params`."""
    old_cz_flux = platform.pairs[targets].native_gates.CZ.pulses[0]
    pulse_sequence = PulseSequence()
    pulse_sequence.add(
        FluxPulse(
            start=0,
            amplitude=pulse_params[0],
            duration=int(pulse_params[1]),
            shape="Rectangular()",
            qubit=old_cz_flux.qubit.name,
        )
    )
    virtual_z_phases = {}
    return pulse_sequence, virtual_z_phases


def new_rx_rule(
    gate: gates.Gate,
    platform: Platform,
    pulse_params: list,
    targets: QubitId,
    parameters=None,
):
    """RX rule returning a custom flux pulse defined by `pulse_params`."""
    qubit = list(platform.qubits)[gate.target_qubits[0]]
    theta = gate.parameters[0]  # float value by default
    sequence = PulseSequence()
    pulse = platform.create_RX90_pulse(qubit, start=0, relative_phase=theta)
    pulse.amplitude = pulse_params[0]
    sequence.add(pulse)
    virtual_z_phases = {}
    return sequence, virtual_z_phases


def resolve_gate_rule(gate_rule: str | Tuple[str, Callable]):
    if isinstance(gate_rule, Tuple):
        if len(gate_rule) != 2:
            raise ValueError("Invalid gate rule tuple")
        if not callable(gate_rule[1]):
            raise ValueError("Invalid callable in gate rule tuple")
        if not isinstance(gate_rule[0], (Gate, str)):
            raise ValueError("Invalid gate identifier for gate rule")
        if isinstance(gate_rule[0], str):
            for gate_name, gate in zip(
                ["x", "rx", "cz"], [gates.GPI2, gates.GPI2, gates.CZ]
            ):
                if gate_rule[0] == gate_name:
                    return gate, gate_rule[1]
    elif isinstance(gate_rule, str):
        if gate_rule == "cz":
            return gates.CZ, new_cz_rule
        elif gate_rule == "rx" or gate_rule == "sx" or gate_rule == "x":
            return gates.GPI2, new_rx_rule
    else:
        raise ValueError(f"Unknown gate rule: {gate_rule}")


def execute_action(
    platform: str,
    circuits: list[Circuit],
    hardware_targets: tuple[QubitId],
    pulse_params: list,
    nshots: int,
    gate_rule: tuple[Gate, Callable] = (gates.GPI2, new_rx_rule),
):
    """Execute `circuits` on the `platform` according to the rule specified by `gate_rule`.

    The `circuits` are executed on the qubits in the quantum hardware specified
    by `hardware_targets`.

    The parameters of the gate are provided in `pulse_params`.

    The function returns a list of frequencies according to `nshots`.

    """
    backend = QibolabBackend(platform=platform)

    # Transpile circuits
    transpiler = dummy_transpiler(backend)
    qubit_map = hardware_targets

    # Change CZ pulse parameters
    compiler = backend.compiler
    if pulse_params:
        rule = lambda gate, platform: gate_rule[1](
            gate, platform, pulse_params, hardware_targets
        )
        compiler.register(gate_rule[0])(rule)
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
