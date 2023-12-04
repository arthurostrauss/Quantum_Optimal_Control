"""
Helper file that allows a user to specify the parameters for the gate calibration simulation.

Author: Lukas Voss
Created on 29/11/2023
"""
from qiskit import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Options
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import XGate, SXGate, YGate, ZGate, HGate, CXGate, SGate, ECRGate
from qiskit.providers.fake_provider import FakeJakarta, FakeMelbourne, FakeRome, FakeSydney, FakeValencia, FakeVigo

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
    backend = FakeJakarta()


    ### IBM Runtime
    # QiskitRuntimeService.save_account(channel='ibm_quantum', token='MY_IBM_QUANTUM_TOKEN') # Save an IBM Quantum account
    # QiskitRuntimeService.save_account(channel='ibm_cloud', token='MY_IBM_CLOUD_API_KEY', instance='MY_IBM_CLOUD_CRN') # Save an IBM Cloud account
    # service = QiskitRuntimeService()

    # get a real backend from a real provider
    # provider = IBMQ.load_account()
    # backend = provider.get_backend('ibmq_manila')
    # generate a simulator that mimics the real quantum system with the latest calibration results
    # backend_sim = AerSimulator.from_backend(backend)

    return backend

########################################

def set_target():
    ### 
    # TODO: Set Target Gate and Register
    ###
    gate_str = 'x'
    target_gate = XGate()
    register = [0]

    return {
        'gate_str': gate_str,
        'target_gate': target_gate, 
        'register': register
    }

########################################

def get_sim_details():
    ### 
    # TODO: Set Simulation Details
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

########################################

# %%
def get_circuit_context():
    ### 
    # TODO: Set Circuit Context in which the target gate is applied
    ###
    circuit_context = QuantumCircuit(1)
    circuit_context.x(0)
    circuit_context.h(0)
    circuit_context.y(0)
    
    return circuit_context

# Create a configutration object for the simulation
sim_config = SimulationConfig(
    abstraction_level=get_sim_details()['abstraction_level'],
    gate_str=set_target()['gate_str'],
    target_gate=set_target()['target_gate'],
    register=set_target()['register'],
    backend=get_backend(),
    n_actions=get_sim_details()['n_actions'],
    sampling_Paulis=get_sim_details()['sampling_Paulis'],
    n_shots=get_sim_details()['n_shots'],
    device=get_sim_details()['device'],
)