"""
Class to generate a RL environment suitable for usage with TF-agents, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
"""

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, Pauli, SparsePauliOp, state_fidelity, Operator, \
    process_fidelity, average_gate_fidelity
from qiskit.exceptions import QiskitError

# Qiskit dynamics for pulse simulation (benchmarking)
from qiskit_dynamics import DynamicsBackend, Solver, Signal, RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_ibm_provider import IBMBackend
from qiskit_dynamics.backend.backend_string_parser.hamiltonian_string_parser import parse_backend_hamiltonian_dict
from qiskit_dynamics.backend.dynamics_backend import _get_backend_channel_freqs

# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit.primitives import Estimator, BackendEstimator
from qiskit_ibm_runtime import Session, Estimator as runtime_Estimator
from qiskit_aer.primitives import Estimator as aer_Estimator
from qiskit.opflow import Zero
import numpy as np
from itertools import product
from typing import Dict, Union, Optional, Any, List, Tuple

# QUA imports
# from QUA_config_two_sc_qubits import IBMconfig
# from qualang_tools.bakery.bakery import baking
# from qm.qua import *
# import time
# from qm.QuantumMachinesManager import QuantumMachinesManager

# Tensorflow modules
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical
from tf_agents.environments.py_environment import PyEnvironment
# from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# configure jax to use 64 bit mode
import jax

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
# import Array and set default backend

Array.set_default_backend('jax')


def get_solver_and_freq_from_backend(backend: IBMBackend, subsystem_list: Optional[List[int]] = None,
                                     rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
                                     evaluation_mode: str = "dense",
                                     rwa_cutoff_freq: Optional[float] = None,
                                     static_dissipators: Optional[Array] = None,
                                     dissipator_operators: Optional[Array] = None,
                                     dissipator_channels: Optional[List[str]] = None, ) \
        -> Tuple[Dict[str, float], Solver]:
    """
    Method to retrieve solver instance and relevant freq channels information from an IBM
    backend added with potential dissipation operators, inspired from DynamicsBackend.from_backend() method
    :param subsystem_list: The list of qubits in the backend to include in the model.
    :param rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`
    :param evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
    :param rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
            ``"auto"``, allowing this method to pick a rotating frame.
    :param backend: IBMBackend instance from which Hamiltonian parameters are extracted
    :param static_dissipators: static_dissipators: Constant dissipation operators.
    :param dissipator_operators: Dissipation operators with time-dependent coefficients.
    :param dissipator_channels: List of channel names in pulse schedules corresponding to dissipator operators.

    :return: Solver instance carrying Hamiltonian information extracted from the IBMBackend instance
    """
    # get available target, config, and defaults objects
    backend_target = getattr(backend, "target", None)

    if not hasattr(backend, "configuration"):
        raise QiskitError(
            "DynamicsBackend.from_backend requires that the backend argument has a "
            "configuration method."
        )
    backend_config = backend.configuration()

    backend_defaults = None
    if hasattr(backend, "defaults"):
        backend_defaults = backend.defaults()

    # get and parse Hamiltonian string dictionary
    if backend_target is not None:
        backend_num_qubits = backend_target.num_qubits
    else:
        backend_num_qubits = backend_config.n_qubits

    if subsystem_list is not None:
        subsystem_list = sorted(subsystem_list)
        if subsystem_list[-1] >= backend_num_qubits:
            raise QiskitError(
                f"subsystem_list contained {subsystem_list[-1]}, which is out of bounds for "
                f"backend with {backend_num_qubits} qubits."
            )
    else:
        subsystem_list = list(range(backend_num_qubits))

    if backend_config.hamiltonian is None:
        raise QiskitError(
            "get_solver_from_backend requires that backend.configuration() has a "
            "hamiltonian."
        )

    (
        static_hamiltonian,
        hamiltonian_operators,
        hamiltonian_channels,
        subsystem_dims,
    ) = parse_backend_hamiltonian_dict(backend_config.hamiltonian, subsystem_list)

    # construct model frequencies dictionary from backend
    channel_freqs = _get_backend_channel_freqs(
        backend_target=backend_target,
        backend_config=backend_config,
        backend_defaults=backend_defaults,
        channels=hamiltonian_channels,
    )

    # build the solver
    if rotating_frame == "auto":
        if "dense" in evaluation_mode:
            rotating_frame = static_hamiltonian
        else:
            rotating_frame = np.diag(static_hamiltonian)

    # get time step size
    if backend_target is not None and backend_target.dt is not None:
        dt = backend_target.dt
    else:
        # config is guaranteed to have a dt
        dt = backend_config.dt

    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=hamiltonian_operators,
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_freqs,
        dt=dt,
        rotating_frame=rotating_frame,
        evaluation_mode=evaluation_mode,
        rwa_cutoff_freq=rwa_cutoff_freq,
        static_dissipators=static_dissipators,
        dissipator_operators=dissipator_operators,
        dissipator_channels=dissipator_channels
    )

    return channel_freqs, solver


class QuantumEnvironment(PyEnvironment):  # TODO: Build a PyEnvironment out of it

    def __init__(self, n_qubits: int, target: Dict, abstraction_level: str,
                 action_spec: Union[array_spec.ArraySpec, tensor_spec.TensorSpec],
                 observation_spec: Union[array_spec.ArraySpec, tensor_spec.TensorSpec],
                 Qiskit_config: Optional[Dict] = None,
                 QUA_setup: Optional[Dict] = None,
                 sampling_Pauli_space: int = 10, n_shots: int = 1, c_factor: float = 0.5):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param n_qubits: Number of qubits in quantum system
        :param target: control target of interest (can be either a gate to be calibrated or a state to be prepared)
            Target should be a Dict containing target type ('gate' or 'state') as well as necessary fields to initiate the calibration (cf. example)
        :param abstraction_level: Circuit or pulse level parametrization of action space
        :param Qiskit_config: Dictionary containing all info for running Qiskit program (i.e. service, backend,
            options, parametrized_circuit)
        :param QUA_setup: Dictionary containing all infor for running a QUA program
        :param sampling_Pauli_space: Number of samples to build fidelity estimator for one action
        :param n_shots: Number of shots to sample for one specific computation (action/Pauli expectation sampling)
        :param c_factor: Scaling factor for reward definition
        """

        assert abstraction_level == 'circuit' or abstraction_level == 'pulse', 'Abstraction layer parameter can be' \
                                                                               'only pulse or circuit'
        self.abstraction_level = abstraction_level
        if abstraction_level == 'circuit':
            assert isinstance(Qiskit_config, Dict), "Qiskit setup argument not provided whereas circuit abstraction " \
                                                    "was provided"
            self.q_register = QuantumRegister(n_qubits)
            self.c_register = ClassicalRegister(n_qubits)
            self.qc = QuantumCircuit(self.q_register)
            try:
                self.service = Qiskit_config.get("service", None)
                self.estimator_options = Qiskit_config.get("estimator_options", None)
                self.backend = Qiskit_config["backend"]
                self.parametrized_circuit_func = Qiskit_config["parametrized_circuit"]
            except KeyError:
                print("Circuit abstraction on Qiskit uses Runtime, need to provide"
                      "backend (Runtime) and ansatz circuit")
        elif abstraction_level == 'pulse':
            # TODO: Define pulse level (Schedule most likely, cf Qiskit Pulse doc)
            # TODO: Add a QUA program
            if QUA_setup is not None:
                self.qua_setup = QUA_setup
            elif Qiskit_config is not None:
                self.q_register = QuantumRegister(n_qubits)
                self.c_register = ClassicalRegister(n_qubits)
                self.qc = QuantumCircuit(self.q_register)
                self.service = Qiskit_config["service"]
                self.backend = Qiskit_config['backend']

                # For benchmarking the gate at each epoch, set the tools for Pulse level simulator
                self.solver = Qiskit_config['solver']
                self.channel_freq = Qiskit_config['channel_freq']

                if type(self.backend) == IBMBackend:  # Prepare to run the simulation according to dynamics backend
                    # Does not include noise models
                    self.pulse_backend = DynamicsBackend.from_backend(self.backend,
                                                                      subsystem_list=Qiskit_config["target_register"])
                self.parametrized_circuit_func = Qiskit_config['parametrized_circuit']
                self.estimator_options = Qiskit_config['estimator_options']

        self.Pauli_ops = [{"name": ''.join(s), "matrix": Pauli(''.join(s)).to_matrix()}
                          for s in product(["I", "X", "Y", "Z"], repeat=n_qubits)]
        self.c_factor = c_factor
        self._n_qubits = n_qubits
        self.d = 2 ** n_qubits  # Dimension of Hilbert space
        self.density_matrix = np.zeros([self.d, self.d], dtype='complex128')
        self.sampling_Pauli_space = sampling_Pauli_space
        self.n_shots = n_shots

        if target.get("target_type", None) == "state" or target.get("target_type", None) is None:  # Default mode is
            # State preparation if no argument target_type is found
            if 'circuit' in target:
                target["dm"] = DensityMatrix(target["circuit"] @ (Zero ^ self._n_qubits))
            assert 'dm' in target, 'no DensityMatrix or circuit argument provided to target dictionary'
            assert type(target["dm"]) == DensityMatrix, 'Provided dm is not a DensityMatrix object'
            self.target = self.calculate_chi_target_state(target)
            self.target_type = "state"
        elif target.get("target_type", None) == "gate":
            # input_states = [self.calculate_chi_target_state(input_state) for input_state in target["input_states"]]
            self.target = target
            # self.target["input_states"] = input_states
            self.target_type = "gate"
        else:
            raise KeyError('target type not identified, must be either gate or state')

        # Data storage for TF-Agents or plotting
        self.action_history = []
        self._state = np.zeros([self.d, self.d], dtype='complex128')
        self.density_matrix_history = []
        self.reward_history = []
        self.state_fidelity_history = []
        self.process_fidelity_history = []
        self.avg_fidelity_history = []
        self.time_step = 0
        self._action_spec = action_spec
        self._observation_spec = observation_spec
        self.episode_ended = False

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass

    def _step(self, action: types.NestedArray) -> ts.TimeStep:

        pass

    def _reset(self) -> ts.TimeStep:  # TODO:
        if self.abstraction_level == 'circuit':
            self.qc.reset(self.q_register)
            return ts.restart()

    def calculate_chi_target_state(self, target_state: Dict):
        """
        Calculate for all P
        :param target_state: Dictionary containing info on target state (name, density matrix)
        :return: target state
        """
        assert 'dm' in target_state, 'No input data for target state, provide DensityMatrix'
        # assert np.imag([np.array(target_state["dm"].to_operator()) @ self.Pauli_ops[k]["matrix"]
        #                 for k in range(self.d ** 2)]).all() == 0.
        target_state["Chi"] = np.array([np.trace(np.array(target_state["dm"].to_operator())
                                                 @ self.Pauli_ops[k]["matrix"]).real for k in
                                        range(self.d ** 2)])  # Real part is taken to convert it in a good format,
        # but im is 0 systematically as dm is hermitian and Pauli is traceless
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

        assert self.target_type == 'state'
        distribution = Categorical(probs=self.target["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, _, pauli_shots = tf.unique_with_counts(k_samples)

        reward_factor = np.round([self.c_factor * self.target["Chi"][p] / (self.d * distribution.prob(p))
                                  for p in pauli_index], 5)

        # observables = [SparsePauliOp(self.Pauli_ops[p]["name"]) for p in pauli_index]
        observables = SparsePauliOp.from_list([(self.Pauli_ops[p]["name"], reward_factor[i])
                                               for i, p in enumerate(pauli_index)])
        # Apply parametrized quantum circuit (action)
        self.parametrized_circuit_func(self.qc)

        # Keep track of state for benchmarking purposes only
        if self.abstraction_level == 'circuit':
            self.density_matrix = np.zeros([self.d, self.d], dtype='complex128')
            qc_list = [self.qc.bind_parameters(angle_set) for angle_set in angles]
            q_state_list = [Statevector.from_instruction(qc) for qc in qc_list]
            self.density_matrix = DensityMatrix(np.mean([np.array(q_state.to_operator()) for q_state in q_state_list],
                                                        axis=0))
            self.density_matrix_history.append(self.density_matrix)
            self.action_history.append(angles)
            self.state_fidelity_history.append(state_fidelity(self.target["dm"], self.density_matrix))
        else:
            pass
        # total_shots = self.n_shots * pauli_shots
        # job_list, result_list = [], []
        # exp_values = np.zeros((len(pauli_index), batch_size))

        if self.service is not None:
            with Session(service=self.service, backend=self.backend):
                estimator = runtime_Estimator(options=self.estimator_options)
                job = estimator.run(circuits=[self.qc] * batch_size,
                                    observables=[observables] * batch_size,
                                    parameter_values=angles,
                                    shots=self.sampling_Pauli_space)
        else:
            estimator = aer_Estimator()
            job = estimator.run(circuits=[self.qc] * batch_size,
                                observables=[observables] * batch_size,
                                parameter_values=angles,
                                shots=self.sampling_Pauli_space)
        result = job.result()
        reward_table = result.values
        self.qc.clear()

        # reward_table = np.mean(reward_factor[:, np.newaxis] * exp_values, axis=0)
        self.reward_history.append(reward_table)
        assert len(reward_table) == batch_size
        return reward_table  # Shape [batchsize]

    def perform_action_gate_cal(self, actions: types.NestedTensorOrArray):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :return: Reward table (reward for each run in the batch), observations (measurement outcomes),
        obtained density matrix
        """
        angles, batch_size = np.array(actions), len(np.array(actions))
        assert self.target_type == 'gate'

        # Pick a random input state from the possible provided input states (forming a tomographically complete set)
        index = np.random.randint(len(self.target["input_states"]))
        input_state = self.target["input_states"][index]
        # Deduce target state to aim for by applying target operation on it
        target_state = {"target_type": 'state'}
        if 'dm' in input_state:
            target_state["dm"] = Operator(self.target['gate']) @ input_state["dm"]
        elif 'circuit' in input_state:
            target_state_fn = Operator(self.target['gate']) @ input_state["circuit"] @ (Zero ^ self._n_qubits)
            target_state["dm"] = DensityMatrix(target_state_fn)
        target_state = self.calculate_chi_target_state(target_state)

        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, _, pauli_shots = tf.unique_with_counts(k_samples)

        reward_factor = np.round([self.c_factor * target_state["Chi"][p] / (self.d * distribution.prob(p))
                                  for p in pauli_index], 5)

        # Figure out which observables to sample
        # observables = [SparsePauliOp(self.Pauli_ops[p]["name"]) for p in pauli_index]
        observables = SparsePauliOp.from_list([(self.Pauli_ops[p]["name"], reward_factor[i])
                                               for i, p in enumerate(pauli_index)])

        # Prepare input state
        self.qc.append(input_state["circuit"].to_instruction(), input_state["register"])

        # Apply parametrized quantum circuit (action)
        parametrized_circ = QuantumCircuit(self._n_qubits)
        self.parametrized_circuit_func(parametrized_circ)

        # Keep track of process for benchmarking purposes only
        if self.abstraction_level == 'circuit':
            qc_list = [parametrized_circ.bind_parameters(angle_set) for angle_set in angles]
            q_process_list = [Operator(qc) for qc in qc_list]
            prc_fidelity = np.mean([process_fidelity(q_process, Operator(self.target["gate"]))
                                    for q_process in q_process_list])
            avg_fidelity = np.mean([average_gate_fidelity(q_process, Operator(self.target["gate"]))
                                    for q_process in q_process_list])

            self.process_fidelity_history.append(prc_fidelity)  # Avg process fidelity over the action batch
            self.avg_fidelity_history.append(avg_fidelity)  # Avg gate fidelity over the action batch
        else:  # TODO: Qiskit Dynamics
            # job = self.dynamics_backend.run([parametrized_circ.bind_parameters(angle_set) for angle_set in angles])
            # result = job.result()
            # print(result)
            dt = self.backend.configuration().dt
            converter = InstructionToSignals(dt, carriers=self.channel_freq)

        self.action_history.append(angles)
        # Build full quantum circuit: concatenate input state prep and parametrized unitary
        self.qc.append(parametrized_circ.to_instruction(), input_state["register"])
        # total_shots = self.n_shots * pauli_shots
        # self.qc = transpile(self.qc, self.backend)
        estimator = Estimator()
        if self.service is not None:
            with Session(service=self.service, backend=self.backend):
                estimator = runtime_Estimator(options=self.estimator_options)
                job = estimator.run(circuits=[self.qc] * batch_size, observables=[observables] * batch_size,
                                    parameter_values=angles,
                                    shots=self.sampling_Pauli_space)
        elif self.abstraction_level == 'circuit':
            # estimator = Estimator(options=self.options)
            estimator = Estimator()
            if self.noise_model is not None:
                estimator = aer_Estimator()
            # TODO: Add noise model here with Aer estimator

        elif type(self.backend) == DynamicsBackend:
            estimator = BackendEstimator(self.backend, skip_transpilation=True)

        job = estimator.run(circuits=[self.qc] * batch_size, observables=[observables] * batch_size,
                            parameter_values=angles,
                            shots=self.sampling_Pauli_space)
        self.qc.clear()  # Reset the QuantumCircuit instance for next iteration

        reward_table = job.result().values
        print(reward_table)
        self.reward_history.append(reward_table)
        assert len(reward_table) == batch_size
        return reward_table  # Shape [batchsize]
