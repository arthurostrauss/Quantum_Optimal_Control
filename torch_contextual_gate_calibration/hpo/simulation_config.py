"""
Helper file that allows a user to specify the parameters for the gate calibration simulation.

Author: Lukas Voss
Created on 29/11/2023
"""
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector, Gate
from qiskit.circuit.library.standard_gates import XGate, SXGate, YGate, ZGate, HGate, CXGate, SGate, ECRGate
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2, FakeMelbourne, FakeMelbourneV2, FakeRome, FakeRomeV2, FakeSydney, FakeSydneyV2, FakeValencia, FakeValenciaV2, FakeVigo, FakeVigoV2, FakeJakarta, FakeJakartaV2

from pulse_parametrization_functions_v01 import map_json_inputs
from qconfig import SimulationConfig
"""
-----------------------------------------------------------------------------------------------------
    User Input: Simulation parameters
-----------------------------------------------------------------------------------------------------
"""
def get_backend():
    ### 
    # TODO: Set Backend
    ###
    backend = None
    backend = FakeJakarta()
    return backend


def get_target():
    ### 
    # TODO: Set Target Gate and Register
    ###
    target_gate = XGate()
    register = [0]

    return {
        'target_gate': target_gate, 
        'register': register
    }

def get_sim_details():
    ### 
    # TODO: Set Smulation Details
    ###
    abstraction_level = 'pulse'
    n_actions = 4
    sampling_Paulis = 50
    n_shots = 200
    device = 'cpu'

    return {
        'abstraction_level': abstraction_level,
        'n_actions': n_actions,
        'sampling_Paulis': sampling_Paulis,
        'n_shots': n_shots,
        'device': device
    }

# %%
def get_circuit_context():
    """
    Creates and returns the ``context`` quantum circuit which will then later be transpiled. Within this later transpiled version,
    the target gate will appear whose pulse parameters will then be parametrized and optimized.

    TODO: The user can specify a custom context circuit here by hardcoded commands

    Returns:
        - QuantumCircuit: A quantum circuit object with the specified number of qubits and a predefined
                        sequence of gates applied to the first qubit.
    """
    circuit_context = QuantumCircuit(1)
    circuit_context.x(0)
    circuit_context.h(0)
    circuit_context.y(0)
    
    return circuit_context


sim_config = SimulationConfig(
    abstraction_level=get_sim_details()['abstraction_level'],
    target_gate=get_target()['target_gate'],
    register=get_target()['register'],
    backend=get_backend(),
    n_actions=get_sim_details()['n_actions'],
    sampling_Paulis=get_sim_details()['sampling_Paulis'],
    n_shots=get_sim_details()['n_shots'],
    device=get_sim_details()['device'],
)