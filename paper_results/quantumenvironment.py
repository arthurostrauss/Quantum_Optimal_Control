"""
Class to generate a RL environment suitable for usage with TF-agents, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import DensityMatrix, Statevector, Pauli, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, IBMBackend
from qiskit.primitives import Estimator
import numpy as np
from itertools import product
from typing import Dict, Union, Callable, Optional, Any
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class QuantumEnvironment(PyEnvironment):  # TODO: Build a PyEnvironment out of it

    def __init__(self, n_qubits: int, target_state: Dict[str, Union[str, DensityMatrix]], abstraction_level: str,
                 action_spec: Union[array_spec.ArraySpec, tensor_spec.TensorSpec],
                 Qiskit_setup: Optional[Dict] = None,
                 QUA_setup: Optional[Dict] = None,
                 sampling_Pauli_space: int = 10, n_shots: int = 1, c_factor: float = 0.5):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param n_qubits: Number of qubits in quantum system
        :param target_state: control target of interest
        :param abstraction_level: Circuit or pulse level parametrization of action space
        :param Qiskit_setup: Dictionary containing all info for running a Qiskit program (i.e. service, backend,
        options, parametrized_circuit)
        :param QUA_setup: Dictionary containing all infor for running a QUA program
        :param sampling_Pauli_space: Number of samples to build fidelity estimator for one action
        :param n_shots: Number of shots to sample for one specific computation (action/Pauli expectation sampling)
        :param c_factor: Scaling factor for reward definition
        """

        self.c_factor = c_factor
        assert abstraction_level == 'circuit' or abstraction_level == 'pulse', 'Abstraction layer parameter can be' \
                                                                               'only pulse or circuit'
        self.abstraction_level = abstraction_level
        if abstraction_level == 'circuit':
            assert isinstance(Qiskit_setup, Dict), "Qiskit setup argument not provided whereas circuit abstraction " \
                                                   "was provided"
            self.q_register = QuantumRegister(n_qubits, name="q")
            self.c_register = ClassicalRegister(n_qubits, name="c")
            self.qc = QuantumCircuit(self.q_register, self.c_register)
            self.service = Qiskit_setup["service"]
            self.options = Qiskit_setup.get("options", None)
            self.backend = Qiskit_setup["backend"]
            self.parametrized_circuit_func = Qiskit_setup["parametrized_circuit"]
        else:
            # TODO: Define pulse level (Schedule most likely, cf Qiskit Pulse doc)
            # TODO: Add a QUA program
            self.qua_setup = QUA_setup
        self.Pauli_ops = [{"name": ''.join(s), "matrix": Pauli(''.join(s)).to_matrix()}
                          for s in product(["I", "X", "Y", "Z"], repeat=n_qubits)]
        self.d = 2 ** n_qubits  # Dimension of Hilbert space
        self._state = np.zeros([self.d, self.d], dtype='complex128')
        self.density_matrix = np.zeros([self.d, self.d], dtype='complex128')
        self.sampling_Pauli_space = sampling_Pauli_space
        self.n_shots = n_shots
        self.target_state = self.calculate_chi_target_state(target_state)

        self.time_step = 0
        self._action_spec = action_spec
        self.episode_ended = False

    def observation_spec(self) -> types.NestedArraySpec:
        pass

    def action_spec(self) -> types.NestedArraySpec:
        pass

    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass

    def _step(self, action: types.NestedArray) -> ts.TimeStep:

        angles, batch_size = np.array(action), len(np.array(action))

        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=self.target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, _, pauli_shots = tf.unique_with_counts(k_samples)

        reward_factor = np.round([self.c_factor * self.target_state["Chi"][p] / (self.d * distribution.prob(p))
                                  for p in pauli_index], 5)
        observables = [SparsePauliOp(self.Pauli_ops[p]["name"]) for p in pauli_index]

        # Apply parametrized quantum circuit (action)
        self.parametrized_circuit_func(self.qc)

        # Keep track of state for benchmarking purposes only
        density_matrix = np.zeros([self.d, self.d], dtype='complex128')
        for angle_set in angles:
            qc_2 = self.qc.bind_parameters(angle_set)
            q_state = Statevector.from_instruction(qc_2)
            density_matrix += np.array(q_state.to_operator())
        self._state = DensityMatrix(density_matrix/batch_size)

        total_shots = self.n_shots * pauli_shots
        job_list = []
        result_list = []
        exp_values = np.zeros((len(pauli_index), batch_size))
        with Session(service=self.service, backend=self.backend):
            estimator = Estimator(options=self.options)
            for p in range(len(pauli_index)):
                job = estimator.run(circuits=[self.qc] * batch_size, observables=[observables[p]] * batch_size,
                                    parameter_values=angles, shots=int(total_shots[p]))
                job_list.append(job)
                result_list.append(job.result())
                exp_values[p] = result_list[p].values

        self.qc.clear()

        reward_table = np.mean(reward_factor[:, np.newaxis] * exp_values, axis=0)
        assert len(reward_table) == batch_size
        return reward_table, DensityMatrix(self.density_matrix)  # Shape [batchsize]

    def _reset(self) -> ts.TimeStep:
        if self.abstraction_level == 'circuit':
            self.qc.reset(self.q_register)
            return ts.restart()

    def calculate_chi_target_state(self, target_state: Dict):
        """
        Calculate for all P
        :param target_state: Dictionary containing info on target state (name, density matrix)
        :return: target state, initializes self.target_state argument
        """
        # target_state["Chi"] = np.zeros(self.d ** 2, dtype="complex_")
        assert np.imag([np.array(target_state["dm"].to_operator())
                        @ self.Pauli_ops[k]["matrix"] for k in
                        range(self.d ** 2)]).all() == 0.
        target_state["Chi"] = np.array([np.trace(np.array(target_state["dm"].to_operator())
                                                 @ self.Pauli_ops[k]["matrix"]).real / np.sqrt(self.d) for k in
                                        range(
                                            self.d ** 2)])  # Real part is taken to convert it in a good format, but im
        # is 0 systematically as dm is hermitian and Pauli is traceless
        return target_state

    def perform_action(self, actions: types.NestedTensorOrArray):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :return: Reward table (reward for each run in the batch), observations (measurement outcomes),
        obtained density matrix
        """
        angles, batch_size = np.array(actions), len(np.array(actions))

        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=self.target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, _, pauli_shots = tf.unique_with_counts(k_samples)

        reward_factor = np.round([self.c_factor * self.target_state["Chi"][p] / (self.d * distribution.prob(p))
                                  for p in pauli_index], 5)
        observables = [SparsePauliOp(self.Pauli_ops[p]["name"]) for p in pauli_index]

        # print(type(self.target_state["Chi"]))
        # print("drawn Pauli operators to sample", k_samples)
        # print(pauli_index, pauli_shots)
        # print([distribution.prob(p) for p in pauli_index])
        # print(observables)
        # Perform actions, followed by relevant expectation value sampling for reward calculation

        # Apply parametrized quantum circuit (action)
        self.parametrized_circuit_func(self.qc)

        # Keep track of state for benchmarking purposes only
        self.density_matrix = np.zeros([self.d, self.d], dtype='complex128')
        for angle_set in angles:
            qc_2 = self.qc.bind_parameters(angle_set)
            q_state = Statevector.from_instruction(qc_2)
            self.density_matrix += np.array(q_state.to_operator())
        self.density_matrix /= batch_size

        total_shots = self.n_shots * pauli_shots
        job_list = []
        result_list = []
        exp_values = np.zeros((len(pauli_index), batch_size))
        with Session(service=self.service, backend=self.backend):
            estimator = Estimator(options=self.options)
            for p in range(len(pauli_index)):
                job = estimator.run(circuits=[self.qc] * batch_size, observables=[observables[p]] * batch_size,
                                    parameter_values=angles,
                                    shots=int(total_shots[p]))
                job_list.append(job)
                result_list.append(job.result())
                exp_values[p] = result_list[p].values

        self.qc.clear()

        reward_table = np.mean(reward_factor[:, np.newaxis] * exp_values, axis=0)
        assert len(reward_table) == batch_size
        return reward_table, DensityMatrix(self.density_matrix)  # Shape [batchsize]
