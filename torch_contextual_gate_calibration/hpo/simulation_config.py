"""
Helper file that allows a user to specify the parameters for the gate calibration simulation.

Author: Lukas Voss
Created on 29/11/2023
"""
import json
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector, Gate
from qiskit.circuit.library.standard_gates import XGate, SXGate, YGate, ZGate, HGate, CXGate, SGate, ECRGate
from qiskit.providers import Backend, BackendV1
from qiskit_experiments.calibration_management import Calibrations
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2, FakeMelbourne, FakeMelbourneV2, FakeRome, FakeRomeV2, FakeSydney, FakeSydneyV2, FakeValencia, FakeValenciaV2, FakeVigo, FakeVigoV2, FakeJakarta, FakeJakartaV2


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


### 
# Set Backend
###
backend = None




target_gate = XGate()
register = [0]



def get_estimator_options():

    estimator_options = None

    return estimator_options

{
    "abstraction_level": "pulse",
    "target_gate": "XGate",
    "register": [0],
    "custom_backend_wanted": False,
    "fake_backend": "FakeJakarta",
    "fake_backend_v2": "FakeJakartaV2",
    "n_actions": 4,
    "sampling_Paulis": 50,
    "n_shots": 200,
    "device": "cpu"
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
    abstraction_level=config['abstraction_level'],
    target_gate=config['target_gate'],
    register=config['register'],
    backend=config['backend'],
    n_actions=config['n_actions'],
    sampling_Paulis=config['sampling_Paulis'],
    n_shots=config['n_shots'],
    device=config['device'],
)