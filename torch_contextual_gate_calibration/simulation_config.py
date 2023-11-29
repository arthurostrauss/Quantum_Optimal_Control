"""
Helper file that allows a user to specify the parameters for the gate calibration simulation.

Author: Lukas Voss
Created on 29/11/2023
"""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2

import torch

from qconfig import SimulationConfig
"""
-----------------------------------------------------------------------------------------------------
    User Input: Simulation parameters
-----------------------------------------------------------------------------------------------------
"""
sim_config = SimulationConfig(
                              abstraction_level="pulse",
                              target_gate=XGate(),
                              register=[0],
                              fake_backend=FakeJakarta(),
                              fake_backend_v2=FakeJakartaV2(),
                              n_actions=4,
                              sampling_Paulis=50,
                              n_shots=200,
                              device=torch.device("cpu")
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