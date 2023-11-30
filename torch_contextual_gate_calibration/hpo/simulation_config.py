"""
Helper file that allows a user to specify the parameters for the gate calibration simulation.

Author: Lukas Voss
Created on 29/11/2023
"""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2

import torch
import json

from pulse_parametrization_functions_v01 import map_json_inputs
from qconfig import SimulationConfig
"""
-----------------------------------------------------------------------------------------------------
    User Input: Simulation parameters
-----------------------------------------------------------------------------------------------------
"""
# Load the configuration from the JSON file
with open('torch_contextual_gate_calibration/hpo/config.json', 'r') as file:
    config = json.load(file)

config = map_json_inputs(config)

sim_config = SimulationConfig(
    abstraction_level=config['abstraction_level'],
    target_gate=config['target_gate'],
    register=config['register'],
    fake_backend=config['fake_backend'],
    fake_backend_v2=config['fake_backend_v2'],
    n_actions=config['n_actions'],
    sampling_Paulis=config['sampling_Paulis'],
    n_shots=config['n_shots'],
    device=config['device'],
)

# %%
def get_circuit_context(num_total_qubits: int):
    """
    Creates and returns the ``context`` quantum circuit which will then later be transpiled. Within this later transpiled version,
    the target gate will appear whose pulse parameters will then be parametrized and optimized.

    Args:
        - num_total_qubits (int): The total number of qubits in the quantum circuit.

    Returns:
        - QuantumCircuit: A quantum circuit object with the specified number of qubits and a predefined
                        sequence of gates applied to the first qubit.
    """
    target_circuit = QuantumCircuit(num_total_qubits)
    target_circuit.x(0)
    target_circuit.h(0)
    target_circuit.y(0)
    return target_circuit