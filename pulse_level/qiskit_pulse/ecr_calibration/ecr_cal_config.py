from __future__ import annotations

from typing import Optional, List
import os

from qiskit.transpiler import Target, InstructionProperties
from rl_qoc.helpers import (
    to_python_identifier,
    custom_schedule,
    validate_pulse_kwargs,
)
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector, Gate, Parameter
from qiskit.providers import BackendV1, BackendV2
import jax

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
current_dir = os.path.dirname(os.path.realpath(__file__))


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: ParameterVector, tgt_register: QuantumRegister, **kwargs
) -> None:
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param tgt_register: Quantum Register formed of target qubits
    :param kwargs: Additional arguments to be passed to the function (here target dict and backend object)
    :return:
    """

    gate, physical_qubits, backend = validate_pulse_kwargs(**kwargs)
    parametrized_gate_name = f"{gate.name if gate is not None else 'G'}_cal"

    param_vec_name = params.name
    if param_vec_name[-1].isdigit():
        parametrized_gate_name += param_vec_name[-1]

    # Create a custom gate with the same name as the original gate
    # Create new set of parameters that are valid Python identifiers (no brackets from the ParameterVector)
    params2 = [Parameter(to_python_identifier(p.name)) for p in params]
    parametrized_gate = Gate(
        parametrized_gate_name,
        len(physical_qubits),
        params=params2,
    )

    # Create custom schedule with original parameters
    parametrized_schedule = custom_schedule(
        backend=backend,
        physical_qubits=physical_qubits,
        params=params,
    )

    # Add the custom calibration to the circuit with current parameters
    qc.append(parametrized_gate, tgt_register)
    qc.add_calibration(
        parametrized_gate_name, physical_qubits, parametrized_schedule, params.params
    )

    # Add the custom calibration to the backend for enabling transpilation with the custom gate
    if parametrized_gate_name not in backend.operation_names:
        target: Target = backend.target
        properties = InstructionProperties(
            duration=parametrized_schedule.duration * backend.dt,
            calibration=parametrized_schedule.assign_parameters(
                {params: params2}, inplace=False
            ),
        )
        target.add_instruction(parametrized_gate, {tuple(physical_qubits): properties})


def get_circuit_context(
    backend: Optional[BackendV1 | BackendV2] = None,
    initial_layout: Optional[List[int]] = None,
):
    """
    Define here the circuit context to be used for the calibration (relevant only for context-aware calibration)
    :param backend: Backend instance
    :param initial_layout: Initial layout of the qubits
    :return: QuantumCircuit instance
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.barrier()
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.cx(0, 1)

    try:
        circuit = transpile(circuit, backend, initial_layout=initial_layout)
    except Exception as e:
        pass
    print("Circuit context: ")
    print(circuit)

    return circuit
