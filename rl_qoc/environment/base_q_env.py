"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 03/11/2024
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod

# For compatibility for options formatting between Estimators.
import json
import signal
from dataclasses import asdict
from typing import Optional, List, Callable, Any, Tuple

from gymnasium import Env
import numpy as np
from gymnasium.core import ObsType
from gymnasium.spaces import Box
from qiskit import transpile, QiskitError

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
    Parameter,
)

# Qiskit Estimator Primitives: for computing Pauli expectation value sampling easily
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
)

from qiskit.quantum_info import partial_trace

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.quantum_info.operators import (
    SparsePauliOp,
    Operator,
    pauli_basis,
    PauliList,
    Clifford,
)
from qiskit.quantum_info.random import random_clifford

from qiskit.transpiler import (
    Layout,
    PassManager,
)
from qiskit.providers import BackendV2

# Qiskit dynamics for pulse simulation (& benchmarking)
from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.library import InterleavedRB

# Qiskit Experiments for generating reliable baseline for complex gate calibrations / state preparations
from qiskit_experiments.library.tomography.basis import (
    Pauli6PreparationBasis,
)
from qiskit_ibm_runtime import (
    EstimatorV2 as RuntimeEstimatorV2,
)

from .backend_info import QiskitBackendInfo, QiboBackendInfo, BackendInfo
from .target import GateTarget, StateTarget
from ..custom_jax_sim import PulseEstimatorV2, simulate_pulse_level

from ..helpers.helper_functions import (
    retrieve_primitives,
    handle_session,
    retrieve_neighbor_qubits,
    substitute_target_gate,
    get_hardware_runtime_single_circuit,
    has_noise_model,
)
from ..helpers.pulse_utils import (
    get_optimal_z_rotation,
    rotate_unitary,
    projected_state,
    qubit_projection,
    rotate_frame,
)
from .qconfig import (
    QEnvConfig,
    CAFERewardConfig,
    ChannelRewardConfig,
    ORBITRewardConfig,
    XEBRewardConfig,
    QiskitRuntimeConfig,
    StateRewardConfig,
)


class BaseQuantumEnvironment(ABC, Env):

    def __init__(self, training_config: QEnvConfig):
        """
        Initialize the quantum environment
        Args:
            training_config: QEnvConfig object containing the training configuration
        """
        self._training_config = training_config
        self._reward_methods = {
            "channel": self.channel_reward_pubs,
            "state": self.state_reward_pubs,
            "cafe": self.cafe_reward_pubs,
            "xeb": self.xeb_reward_pubs,
            "orbit": self.orbit_reward_pubs,
            "fidelity": self.compute_benchmarks,
        }
        if self.config.reward_method not in self._reward_methods:
            raise ValueError(
                f"Reward method {self.config.reward_method} not implemented. Only "
                f"{list(self._reward_methods.keys())} are supported."
            )

        self.n_shots = training_config.n_shots
        self.n_reps = training_config.n_reps
        self.sampling_Pauli_space = training_config.sampling_paulis
        self.c_factor = training_config.c_factor
        self.batch_size = training_config.batch_size

        self.action_space: Box = training_config.action_space

        self.parametrized_circuit_func: Callable = training_config.parametrized_circuit
        self._func_args = training_config.parametrized_circuit_kwargs
        self.backend = training_config.backend
        if isinstance(self.backend, BackendV2) or self.backend is None:
            self._backend_info = QiskitBackendInfo(
                self.backend,
                training_config.backend_config.instruction_durations,
                pass_manager=training_config.backend_config.pass_manager,
                skip_transpilation=training_config.backend_config.skip_transpilation,
            )
        elif training_config.backend_config.config_type == "qibo":
            self._backend_info = QiboBackendInfo(
                training_config.backend_config.n_qubits,
                training_config.backend_config.coupling_map,
            )
        else:
            raise ValueError("Backend should be a BackendV2 object or a string")
        self._physical_target_qubits = training_config.target.get(
            "physical_qubits", None
        )

        if isinstance(training_config.backend_config, QiskitRuntimeConfig):
            estimator_options = training_config.backend_config.estimator_options
        else:
            estimator_options = None

        self._estimator, self._sampler = retrieve_primitives(
            self.backend, self.config.backend_config, estimator_options
        )

        self._target, self.circuits, self.baseline_circuits = None, None, None

        # self.fidelity_checker = ComputeUncompute(self._sampler)

        self._mean_action = np.zeros(self.action_space.shape[-1])
        self._std_action = np.ones(self.action_space.shape[-1])
        # Data storage
        self._optimal_action = np.zeros(self.action_space.shape[-1])
        self._seed = training_config.seed
        self._session_counts = 0
        self._step_tracker = 0
        self._inside_trunc_tracker = 0
        self._total_shots = []
        self._hardware_runtime = []
        self._max_return = 0
        self._episode_ended = False
        self._episode_tracker = 0
        self.action_history = []
        self.reward_history = []
        self._pubs, self._ideal_pubs = [], []
        self._observables, self._pauli_shots = None, None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.process_fidelity_history = []
        self.avg_fidelity_history = []
        self.circuit_fidelity_history = []
        self.circuit_fidelity_history_nreps = []
        self.avg_fidelity_history_nreps = []

    @abstractmethod
    def define_target_and_circuits(
        self,
    ) -> tuple[
        GateTarget | StateTarget | List[GateTarget | StateTarget],
        List[QuantumCircuit],
        List[QuantumCircuit],
    ]:
        """
        Define the target gate or state and the circuits to be executed on the quantum system.
        This method should be implemented by the user and called at construction of the environment.
        It can typically be called after the initialization of this base class.
        """
        raise NotImplementedError("Define target method not implemented")

    @abstractmethod
    def episode_length(self, global_step: int) -> int:
        """
        Args:
            global_step: Step in the training loop

        Returns: Episode length

        """
        pass

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Benchmark through tomography or through simulation the policy
        Args:
            qc: Quantum circuit to benchmark
            params:

        Returns: Fidelity metric or array of fidelities for all actions in the batch

        """

    def perform_action(self, actions: np.array):
        """
        Send the action batch to the quantum system and retrieve the reward
        :param actions: action vectors to execute on quantum system
        :return: Reward table (reward for each action in the batch)
        """

        trunc_index = self.trunc_index
        qc = self.circuits[trunc_index].copy()
        params, batch_size = np.array(actions), actions.shape[0]
        if batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: {batch_size} != {self.batch_size} ")

        # Get the reward method from the configuration
        reward_method = self.config.reward_method
        if self.do_benchmark():  # Benchmarking or fidelity access
            fids = self.compute_benchmarks(qc, params)
            if reward_method == "fidelity":
                self._total_shots.append(0)
                self._hardware_runtime.append(0.0)
                return fids

        # Check if the reward method exists in the dictionary
        self._pubs, total_shots = self._reward_methods[reward_method](qc, params)
        self._total_shots.append(total_shots)
        if self.backend_info.instruction_durations is not None:
            self._hardware_runtime.append(
                get_hardware_runtime_single_circuit(
                    qc,
                    self.backend_info.instruction_durations.duration_by_name_qubits,
                )
                * self.total_shots[-1]
            )
            print("Hardware runtime taken:", sum(self.hardware_runtime))

        counts = (
            self._session_counts
            if isinstance(self.estimator, RuntimeEstimatorV2)
            else trunc_index
        )
        self.estimator = handle_session(self.estimator, self.backend, counts)
        print(
            f"Sending {'Estimator' if isinstance(self.primitive, BaseEstimatorV2) else 'Sampler'} job..."
        )
        start = time.time()

        job = self.primitive.run(pubs=self._pubs)
        pub_results = job.result()
        print("Time for running", time.time() - start)

        if self.config.dfe:
            reward_table = np.sum(
                [pub_result.data.evs for pub_result in pub_results], axis=0
            ) / len(self._observables)
        else:
            pub_counts = [
                [pub_result.data.meas.get_counts(i) for i in range(self.batch_size)]
                for pub_result in pub_results
            ]
            if self.config.reward_method == "xeb":
                # TODO: Implement XEB reward computation using Sampler
                reward_table = np.zeros(self.batch_size)
                raise NotImplementedError("XEB reward computation not implemented yet")
            else:
                survival_probability = [
                    np.array(
                        [
                            count.get("0" * qc.num_qubits, 0) / self.n_shots
                            for count in counts
                        ]
                    )
                    for counts in pub_counts
                ]
                reward_table = np.mean(survival_probability, axis=0)

        print(
            f"Finished {'Estimator' if isinstance(self.primitive, BaseEstimatorV2) else 'Sampler'} job"
        )
        print("Reward (avg):", np.mean(reward_table), "Std:", np.std(reward_table))

        return reward_table  # Shape [batch size]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment to its initial state
        :param seed: Seed for random number generator
        :param options: Options for the reset
        :return: Initial observation
        """
        super().reset(seed=seed)
        self._episode_tracker += 1
        self._episode_ended = False

        if isinstance(self.estimator, RuntimeEstimatorV2):
            self.estimator.options.update(job_tags=[f"rl_qoc_step{self._step_tracker}"])

        return self._get_obs(), self._get_info()

    def retrieve_observables(
        self,
        target_state: StateTarget,
        qc: QuantumCircuit,
        dfe_tuple: Optional[Tuple[float, float]] = None,
    ):
        """
        Retrieve observables to sample for the DFE protocol (PhysRevLett.106.230501) for given target state

        :param target_state: Target state to prepare
        :param qc: Quantum circuit to be executed on quantum system
        :param dfe_tuple: Optional Tuple (Ɛ, δ) from DFE paper
        :return: Observables to sample, number of shots for each observable
        """
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        probabilities = target_state.Chi**2
        full_basis = pauli_basis(target_state.n_qubits)
        if not np.isclose(np.sum(probabilities), 1, atol=1e-5):
            print("probabilities sum um to", np.sum(probabilities))
            print("probabilities normalized")
            probabilities = probabilities / np.sum(probabilities)

        sample_size = (
            self.sampling_Pauli_space
            if dfe_tuple is None
            else int(np.ceil(1 / (dfe_tuple[0] ** 2 * dfe_tuple[1])))
        )
        k_samples = np.random.choice(
            len(probabilities), size=sample_size, p=probabilities
        )

        pauli_indices, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = self.c_factor / (
            np.sqrt(target_state.dm.dim) * target_state.Chi[pauli_indices]
        )

        if dfe_tuple is not None:
            pauli_shots = np.ceil(
                2
                * np.log(2 / dfe_tuple[1])
                / (
                    target_state.dm.dim
                    * sample_size
                    * dfe_tuple[0] ** 2
                    * target_state.Chi[pauli_indices] ** 2
                )
            )
        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp(
            full_basis[pauli_indices], reward_factor, copy=False
        )

        shots_per_basis = []
        # Group observables by qubit-wise commuting groups to reduce the number of PUBs
        for i, commuting_group in enumerate(
            observables.paulis.group_qubit_wise_commuting()
        ):
            max_pauli_shots = 0
            for pauli in commuting_group:
                pauli_index = list(full_basis).index(pauli)
                ref_index = list(pauli_indices).index(pauli_index)
                max_pauli_shots = max(max_pauli_shots, pauli_shots[ref_index])
            shots_per_basis.append(max_pauli_shots)

        if (
            isinstance(self.target, GateTarget)
            and qc.num_qubits > self.target.causal_cone_size
        ):  # If circuit has more qubits than causal cone, extend to full qubit space
            other_qubit_indices = set(range(qc.num_qubits)) - set(
                self.target.causal_cone_qubits_indices
            )
            observables = observables.apply_layout(None, qc.num_qubits)
            observables = observables.apply_layout(
                self.target.causal_cone_qubits_indices + list(other_qubit_indices)
            )

        return observables, shots_per_basis

    def state_reward_pubs(self, qc: QuantumCircuit, params):
        """
        Compute the PUBs for the state reward method
        """
        if not isinstance(self.config.reward_config, StateRewardConfig):
            raise TypeError("StateConfig object required for state reward method")

        prep_circuit = qc
        target_state = self.target
        if isinstance(self.target, GateTarget):
            # State reward: sample a random input state for target gate
            input_state = np.random.choice(self.target.input_states)

            # Prepend input state to custom circuit with front composition
            prep_circuit = qc.compose(
                input_state.circuit,
                self.target.causal_cone_qubits,
                front=True,
            )
            for _ in range(self.n_reps - 1):  # Repeat circuit for noise amplification
                prep_circuit.compose(qc, inplace=True)

            if (
                qc.num_qubits > self.target.causal_cone_size
            ):  # Add random input state on all qubits (not part of reward calculation)
                other_qubits_indices = set(range(qc.num_qubits)) - set(
                    self.target.causal_cone_qubits_indices
                )
                other_qubits = [qc.qubits[i] for i in other_qubits_indices]
                random_input_context = Pauli6PreparationBasis().circuit(
                    np.random.randint(0, 6, len(other_qubits)).tolist()
                )
                prep_circuit.compose(
                    random_input_context, other_qubits, inplace=True, front=True
                )

            # Modify target state to match input state and target gate
            target_state = input_state.target_state  # (Gate |input>=|target>)

        self._observables, self._pauli_shots = self.retrieve_observables(
            target_state, qc
        )

        prep_circuit = self.backend_info.custom_transpile(
            prep_circuit,
            initial_layout=self.layout,
            scheduling=False,
        )

        pubs = [
            (
                prep_circuit,
                obs.apply_layout(prep_circuit.layout),
                params,
                1 / np.sqrt(self.n_shots * pauli_shots),
            )
            for obs, pauli_shots in zip(
                self._observables.group_commuting(qubit_wise=True),
                self._pauli_shots,
            )
        ]
        total_shots = self.batch_size * np.sum(self._pauli_shots * self.n_shots)

        return pubs, total_shots

    def channel_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ):
        """
        Retrieve observables and input state to sample for the DFE protocol for a target gate

        :param qc: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        :param dfe_precision: Optional Tuple (Ɛ, δ) from DFE paper
        :return: Observables to sample, input state to prepare
        """
        if not isinstance(self.target, GateTarget):
            raise TypeError("Target type should be a gate")
        if not isinstance(self.config.reward_config, ChannelRewardConfig):
            raise TypeError("ChannelConfig object required for channel reward method")
        if self.target.causal_cone_size > 3:
            raise ValueError("Channel reward method is only supported for 1-3 qubits")

        nb_states = self.config.reward_config.num_eigenstates_per_pauli
        if (
            nb_states >= 2**qc.num_qubits
        ):  # TODO: Rethink this condition for causal cone
            raise ValueError(
                f"Number of eigenstates per Pauli should be less than {2 ** qc.num_qubits}"
            )
        n_qubits = self.target.causal_cone_size
        d = 2**n_qubits
        probabilities = self.target.Chi**2 / (d**2)
        non_zero_indices = np.nonzero(probabilities)[0]  # Filter out zero probabilities
        non_zero_probabilities = probabilities[non_zero_indices]

        basis = pauli_basis(num_qubits=n_qubits)

        if dfe_precision is not None:
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            eps, delta = None, None
            pauli_sampling = self.sampling_Pauli_space

        samples, self._pauli_shots = np.unique(
            np.random.choice(
                non_zero_indices, size=pauli_sampling, p=non_zero_probabilities
            ),
            return_counts=True,
        )
        pauli_indices = np.array(
            [np.unravel_index(sample, (d**2, d**2)) for sample in samples], dtype=int
        )

        pauli_prep, pauli_meas = zip(
            *[(basis[p[1]], basis[p[0]]) for p in pauli_indices]
        )
        pauli_prep, pauli_meas = PauliList(pauli_prep), PauliList(pauli_meas)
        reward_factor = [self.c_factor / (d * self.target.Chi[p]) for p in samples]

        observables = SparsePauliOp(pauli_meas, reward_factor, ignore_pauli_phase=True)
        if qc.num_qubits > self.target.causal_cone_size:
            other_qubit_indices = set(range(qc.num_qubits)) - set(
                self.target.causal_cone_qubits_indices
            )
            observables = observables.apply_layout(
                None, qc.num_qubits
            )  # Extend to full qubit space
            # Apply layout to apply non-trivial observables on causal cone qubits
            observables = observables.apply_layout(
                self.target.causal_cone_qubits_indices + list(other_qubit_indices)
            )
        self._observables = observables

        if dfe_precision is not None:
            self._pauli_shots = np.ceil(
                2
                * np.log(2 / delta)
                / (d * pauli_sampling * eps**2 * self.target.Chi[samples] ** 2)
            )

        pubs, total_shots = [], 0
        used_prep_indices, used_indices, idx = (
            [],
            [],
            0,
        )  # Track used input states to reduce number of PUBs

        for prep, obs, shots in zip(pauli_prep, self._observables, self._pauli_shots):

            # Each prep is a Pauli input state, that we need to decompose in its pure eigenbasis
            # Below, we select at random a subset of pure input states to prepare for each prep
            # If nb_states = 1, we prepare all pure input states for each prep (no random selection)
            max_input_states = d // nb_states
            selected_input_states = np.random.choice(
                d, size=max_input_states, replace=False
            )
            for input_state in selected_input_states:
                prep_indices = []
                dedicated_shots = (
                    shots * self.n_shots // max_input_states
                )  # Number of shots per Pauli eigenstate (divided equally)
                if dedicated_shots == 0:
                    continue
                # Convert input state to Pauli6 basis: preparing pure eigenstates of Pauli_prep
                inputs = np.unravel_index(input_state, (2,) * qc.num_qubits)
                parity = (-1) ** np.sum(
                    inputs
                )  # Parity of input state (weighting factor in Pauli prep decomposition)
                for i, pauli_op in enumerate(reversed(prep.to_label())):
                    # Build input state in Pauli6 basis from Pauli prep: look at each qubit individually
                    if pauli_op == "I" or pauli_op == "Z":
                        prep_indices.append(inputs[i])
                    elif pauli_op == "X":
                        prep_indices.append(2 + inputs[i])
                    elif pauli_op == "Y":
                        prep_indices.append(4 + inputs[i])
                if (
                    tuple(prep_indices) not in used_prep_indices
                ):  # If input state not already used, add a PUB
                    used_prep_indices.append(tuple(prep_indices))
                    used_indices.append(idx)
                    idx += 1
                    # Prepare input state in Pauli6 basis (front composition)
                    prep_circuit = qc.compose(
                        Pauli6PreparationBasis().circuit(prep_indices),
                        self.target.causal_cone_qubits,
                        front=True,
                    )
                    # Repeat circuit for noise amplification
                    for _ in range(self.n_reps - 1):
                        prep_circuit.compose(qc, inplace=True)

                    if (
                        qc.num_qubits > self.target.causal_cone_size
                    ):  # Add random input state on other qubits (not part of reward calculation, just for enhanced calibration)
                        other_qubits_indices = set(range(qc.num_qubits)) - set(
                            self.target.causal_cone_qubits_indices
                        )
                        other_qubits = [qc.qubits[i] for i in other_qubits_indices]
                        random_input_context = Pauli6PreparationBasis().circuit(
                            np.random.randint(0, 6, len(other_qubits)).tolist()
                        )
                        prep_circuit.compose(
                            random_input_context, other_qubits, inplace=True, front=True
                        )

                    # Transpile the circuit again to decompose input state preparation
                    prep_circuit = self.backend_info.custom_transpile(
                        prep_circuit,
                        initial_layout=self.layout,
                        scheduling=False,
                        optimization_level=0,
                    )
                    pubs.append(
                        (
                            prep_circuit,
                            parity * obs.apply_layout(prep_circuit.layout),
                            params,
                            1 / np.sqrt(dedicated_shots),
                        )
                    )
                    obs: SparsePauliOp
                    total_shots += (
                        dedicated_shots
                        * self.batch_size
                        * len(obs.group_commuting(qubit_wise=True))
                    )
                else:  # If input state already used, reuse PUB and just update observable and precision
                    pub_ref_index: int = used_indices[
                        used_prep_indices.index(tuple(prep_indices))
                    ]

                    prep_circuit, ref_obs, ref_params, ref_precision = pubs[
                        pub_ref_index
                    ]
                    ref_shots = int(math.ceil(1.0 / ref_precision**2))
                    new_precision = min(ref_precision, 1 / np.sqrt(dedicated_shots))
                    new_shots = int(math.ceil(1.0 / new_precision**2))
                    new_pub = (
                        prep_circuit,
                        ref_obs + parity * obs.apply_layout(prep_circuit.layout),
                        ref_params,
                        new_precision,
                    )
                    pubs[pub_ref_index] = new_pub
                    total_shots -= (
                        ref_shots
                        * self.batch_size
                        * len(ref_obs.group_commuting(qubit_wise=True))
                    )
                    total_shots += (
                        new_shots
                        * self.batch_size
                        * len(new_pub[1].group_commuting(qubit_wise=True))
                    )

        if len(pubs) == 0:  # If nothing was sampled, retry
            pubs, total_shots = self.channel_reward_pubs(qc, params)

        return pubs, total_shots

    def cafe_reward_pubs(self, circuit: QuantumCircuit, params):
        """
        Retrieve PUBs for Context-Aware Fidelity Estimation (CAFE) protocol

        :param circuit: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        """

        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        assert isinstance(
            self.config.reward_config, CAFERewardConfig
        ), "CAFEConfig object required for CAFE reward method"

        pubs = []
        total_shots = 0
        circuit_ref = self.baseline_circuits[self.trunc_index]
        layout = self.layout

        # samples, shots = np.unique(
        #     np.random.choice(len(input_circuits), self.sampling_Pauli_space),
        #     return_counts=True,
        # )
        # for sample, shot in zip(samples, shots):
        for input_state in self.target.input_states:
            run_qc = QuantumCircuit.copy_empty_like(
                circuit, name="cafe_circ"
            )  # Circuit with custom target gate
            ref_qc = QuantumCircuit.copy_empty_like(
                circuit_ref, name="cafe_ref_circ"
            )  # Circuit with reference gate

            for qc, context in zip([run_qc, ref_qc], [circuit, circuit_ref]):
                # Bind input states to the circuits
                qc.compose(input_state.circuit, inplace=True)
                qc.barrier()
                for _ in range(
                    self.n_reps
                ):  # Add custom target gate to run circuit n_reps times
                    qc.compose(context, inplace=True)
                    qc.barrier()

            # Compute inverse unitary for reference circuit
            reverse_unitary = Operator(ref_qc).adjoint()
            reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
            reverse_unitary_qc.unitary(
                reverse_unitary,
                reverse_unitary_qc.qubits,
                label="U_inv",
            )
            reverse_unitary_qc.measure_all()

            reverse_unitary_qc = self.backend_info.custom_transpile(
                reverse_unitary_qc,
                initial_layout=layout,
                scheduling=False,
                optimization_level=3,  # Find smallest circuit implementing inverse unitary
                remove_final_measurements=False,
            )

            # Bind inverse unitary + measurement to run circuit
            for circ, pubs_ in zip([run_qc, ref_qc], [pubs, self._ideal_pubs]):
                transpiled_circuit = self.backend_info.custom_transpile(
                    circ, initial_layout=layout, scheduling=False
                )
                transpiled_circuit.barrier(self.involved_qubits)
                # Add the inverse unitary + measurement to the circuit
                transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                pubs_.append((transpiled_circuit, params, self.n_shots))
            total_shots += self.batch_size * self.n_shots

        return pubs, total_shots

    def xeb_reward_pubs(self, circuit: QuantumCircuit, params):
        """
        Retrieve PUBs for XEB protocol

        :param circuit: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        """
        # TODO: Complete XEB (will be relevant once pattern for parallel XEB is figured out)
        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        assert isinstance(
            self.config.reward_config, XEBRewardConfig
        ), "XEBConfig object required for XEB reward method"
        layout = self.layout
        circuit_ref = self.baseline_circuits[self.trunc_index]
        pubs = []
        total_shots = 0

        for seq in range(self.sampling_Pauli_space):
            ref_qc = QuantumCircuit.copy_empty_like(
                circuit_ref,
                name="xeb_ref_circ",
            )
            run_qc = QuantumCircuit.copy_empty_like(
                circuit,
                name="xeb_run_circ",
            )
            for l in range(self.n_reps):
                pass

        return pubs, total_shots

    def orbit_reward_pubs(self, circuit: QuantumCircuit, params):
        """
        Retrieve PUBs for ORBIT protocol

        :param circuit: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        """
        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        assert isinstance(
            self.config.reward_config, ORBITRewardConfig
        ), "ORBITConfig object required for ORBIT reward method"
        layout = self.layout
        circuit_ref = self.baseline_circuits[self.trunc_index]
        pubs = []
        total_shots = 0
        if self.config.reward_config.use_interleaved:  # Interleaved RB
            try:
                Clifford(circuit_ref)
            except QiskitError as e:
                raise ValueError(
                    "Circuit should be a Clifford circuit for using interleaved RB directly"
                ) from e
            ref_element = circuit_ref.to_gate(label="ref_circ")
            custom_element = circuit.to_gate(label="custom_circ")
            exp = InterleavedRB(
                ref_element,
                self.involved_qubits,
                [self.n_reps],
                self.backend,
                self.sampling_Pauli_space,
                self.seed,
                circuit_order="RRRIII",
            )
            # ref_circuits = exp.circuits()[0: self.n_reps]
            interleaved_circuits = exp.circuits()[self.n_reps :]
            run_circuits = [
                substitute_target_gate(circ, ref_element, custom_element)
                for circ in interleaved_circuits
            ]
            run_circuits = self.backend_info.custom_transpile(
                run_circuits,
                initial_layout=layout,
                scheduling=False,
                remove_final_measurements=False,
            )
            pubs = [(qc, params, self.n_shots) for qc in run_circuits]
            total_shots += self.batch_size * self.n_shots * len(pubs)
            self._ideal_pubs = [
                (qc, params, self.n_shots) for qc in interleaved_circuits
            ]
        else:
            for seq in range(self.sampling_Pauli_space):
                ref_qc = QuantumCircuit.copy_empty_like(
                    circuit_ref,
                    name="orbit_ref_circ",
                )
                run_qc = QuantumCircuit.copy_empty_like(circuit, name="orbit_run_circ")
                for l in range(self.n_reps):
                    r_cliff = random_clifford(circuit.num_qubits)
                    for qc, context in zip([run_qc, ref_qc], [circuit, circuit_ref]):
                        qc.compose(r_cliff.to_circuit(), inplace=True)
                        qc.barrier()
                        qc.compose(context, inplace=True)
                        qc.barrier()

                reverse_unitary = Operator(ref_qc).adjoint()
                reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
                reverse_unitary_qc.unitary(
                    reverse_unitary, reverse_unitary_qc.qubits, label="U_inv"
                )
                reverse_unitary_qc.measure_all()

                reverse_unitary_qc = self.backend_info.custom_transpile(
                    reverse_unitary_qc,
                    initial_layout=layout,
                    scheduling=False,
                    optimization_level=3,
                    remove_final_measurements=False,
                )  # Try to get the smallest possible circuit for the reverse unitary

                for circ, pubs_ in zip([run_qc, ref_qc], [pubs, self._ideal_pubs]):
                    transpiled_circuit = self.backend_info.custom_transpile(
                        circ, initial_layout=layout, scheduling=False
                    )
                    transpiled_circuit.barrier(self.involved_qubits)
                    # Add the inverse unitary + measurement to the circuit
                    transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                    pubs_.append((transpiled_circuit, params, self.n_shots))

                total_shots += self.batch_size * self.n_shots

        return pubs, total_shots

    def simulate_circuit(
        self, qc: QuantumCircuit, params: np.array, update_env_history: bool = True
    ) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        This method should be called only when the abstraction level is "circuit"
        :param qc: QuantumCircuit to execute on quantum system
        :param params: List of Action vectors to execute on quantum system
        :param update_env_history: Boolean to update the environment history
        :return: Fidelity metric or array of fidelities for all actions in the batch
        """
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        if self.abstraction_level != "circuit":
            raise ValueError(
                "This method should only be called when the abstraction level is 'circuit'"
            )

        names = ["qc_channel", "qc_state", "qc_channel_nreps", "qc_state_nreps"]
        qc_channel, qc_state, qc_channel_nreps, qc_state_nreps = [
            qc.copy(name=name) for name in names
        ]
        returned_fidelity_type = (
            "gate"
            if isinstance(self.target, GateTarget) and qc.num_qubits <= 3
            else "state"
        )
        returned_fidelities = []

        for _ in range(self.n_reps - 1):
            qc_channel_nreps.compose(qc, inplace=True)
            qc_state_nreps.compose(qc, inplace=True)
        backend = AerSimulator()
        if self.backend is None or (
            isinstance(self.backend, AerSimulator) and not has_noise_model(self.backend)
        ):  # Ideal simulation

            noise_model = None
            qc_channel.save_unitary()
            qc_channel_nreps.save_unitary()
            qc_state.save_statevector()
            qc_state_nreps.save_statevector()
            channel_output = "unitary"
            state_output = "statevector"

        else:  # Noisy simulation
            if isinstance(self.backend, AerSimulator):
                noise_model = self.backend.options.noise_model
            else:
                noise_model = NoiseModel.from_backend(self.backend)
            qc_channel.save_superop()
            qc_channel_nreps.save_superop()
            qc_state.save_density_matrix()
            qc_state_nreps.save_density_matrix()
            channel_output = "superop"
            state_output = "density_matrix"

        basis_gates = backend.operation_names
        if noise_model is not None:
            basis_gates += noise_model.basis_gates
        qc_channel, qc_channel_nreps, qc_state, qc_state_nreps = transpile(
            [qc_channel, qc_channel_nreps, qc_state, qc_state_nreps],
            backend=backend,
            optimization_level=0,
            basis_gates=basis_gates,
        )
        if isinstance(self.parameters, ParameterVector):
            parameters = [self.parameters]
            n_custom_instructions = 1
        else:  # List of ParameterVectors
            parameters = self.parameters
            n_custom_instructions = self.trunc_index + 1

        parameter_binds = [
            {
                parameters[i][j]: params[:, i * self.n_actions + j]
                for i in range(n_custom_instructions)
                for j in range(self.n_actions)
            }
        ]
        data_length = len(params)

        if isinstance(self.target, GateTarget):
            circuits = [qc_channel, qc_channel_nreps, qc_state, qc_state_nreps]
            methods = [channel_output] * 2 + [state_output] * 2
            fid_arrays = [
                self.avg_fidelity_history,
                self.avg_fidelity_history_nreps,
                self.circuit_fidelity_history,
                self.circuit_fidelity_history_nreps,
            ]
        else:
            circuits = [qc_state, qc_state_nreps]
            fid_arrays = [
                self.circuit_fidelity_history,
                self.circuit_fidelity_history_nreps,
            ]
            methods = [state_output] * 2
        for circ, method, fid_array in zip(circuits, methods, fid_arrays):
            # Avoid channel simulation for more than 3 qubits
            if (method == "superop" or method == "unitary") and circ.num_qubits > 3:
                fidelities = [0.0] * data_length
                n_reps = 1
            else:
                result = backend.run(
                    circ,
                    parameter_binds=parameter_binds,
                    method=method,
                    noise_model=noise_model,
                ).result()
                outputs = [result.data(i)[method] for i in range(data_length)]
                n_reps = self.n_reps if "nreps" in circ.name else 1
                fidelities = [
                    self.target.fidelity(output, n_reps) for output in outputs
                ]
            if (
                (method == "superop" or method == "unitary")
                and returned_fidelity_type == "gate"
                and n_reps == 1
            ):
                returned_fidelities = fidelities
            elif (
                (method == "density_matrix" or method == "statevector")
                and returned_fidelity_type == "state"
                and n_reps == 1
            ):
                returned_fidelities = fidelities
            if update_env_history:
                fid_array.append(np.mean(fidelities))

        return returned_fidelities

    def _observable_to_observation(self):
        """
        Convert the observable to an observation to be given to the agent
        """
        if self.config.reward_method == "state":
            # n_qubits = self.observables.num_qubits
            # d = 2**n_qubits
            # pauli_to_index = {pauli: i for i, pauli in enumerate(pauli_basis(n_qubits))}
            # array_obs = np.zeros(d**2)
            # for pauli in self.observables:
            #     array_obs[pauli_to_index[pauli.paulis[0]]] = pauli.coeffs[0]

            array_obs = []
            return array_obs
        else:
            raise NotImplementedError("Channel estimator not yet implemented")

    def simulate_pulse_circuit(
        self,
        qc: QuantumCircuit,
        params: Optional[np.array] = None,
        update_env_history: bool = True,
    ) -> List[float]:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        This method should be called only when the abstraction level is "pulse"

        :param qc: QuantumCircuit to execute on quantum system
        :param params: List of Action vectors to execute on quantum system
        :param update_env_history: Boolean to update the environment history
        """
        if self.abstraction_level != "pulse":
            raise ValueError(
                "This method should only be called when the abstraction level is 'pulse'"
            )
        if not isinstance(self.backend, DynamicsBackend):
            raise ValueError(
                f"Pulse simulation requires a DynamicsBackend; got {self.backend}"
            )
        returned_fidelity_type = (
            "gate"
            if isinstance(self.target, GateTarget) and qc.num_qubits <= 3
            else "state"
        )  # Fidelity type to return (gate or state fidelity metric)
        returned_fidelities = []
        subsystem_dims = list(
            filter(lambda x: x > 1, self.backend.options.subsystem_dims)
        )  # Filter out qubits with dimension 1 (trivial dimension stated for DynamicsBackend)
        n_benchmarks = 1  # Number of benchmarks to run (1 if no n_reps, 2 if n_reps > 1, to benchmark both qc and qc_nreps)
        qc_nreps = None
        if self.n_reps > 1 and isinstance(
            self.target, GateTarget
        ):  # No need to benchmark n_reps for state targets
            qc_nreps = qc.copy("qc_nreps")
            for _ in range(self.n_reps - 1):
                qc_nreps.compose(qc, inplace=True)
            n_benchmarks *= 2

        y0_gate = Operator(
            np.eye(np.prod(subsystem_dims)),
            input_dims=tuple(subsystem_dims),
            output_dims=tuple(subsystem_dims),
        )
        y0_state = Statevector.from_int(0, dims=subsystem_dims)
        if params is None or isinstance(self.estimator, PulseEstimatorV2):
            circuits = [qc]
            circuits_n_reps = [qc_nreps] if qc_nreps is not None else []
            data_length = 1 if params is None else len(params)
        else:
            if not isinstance(params, np.ndarray):
                params = np.array(params)
            if len(params.shape) == 1:
                params = np.expand_dims(params, axis=0)
            circuits = [qc.assign_parameters(p) for p in params]
            circuits_n_reps = (
                [qc_nreps.assign_parameters(p) for p in params]
                if qc_nreps is not None
                else []
            )
            data_length = len(params)
        circuits_list = circuits + circuits_n_reps

        if isinstance(self.estimator, PulseEstimatorV2):
            # TODO: Handle this case
            sampler_pubs = [(circ, params) for circ in circuits_list]
            y0_list = [y0_state]
            if qc.num_qubits < 3 and isinstance(
                self.target, GateTarget
            ):  # Benchmark channel only for 1-2 qubits
                y0_list += [y0_gate]
                for circ in circuits_list:
                    sampler_pubs.append((circ, params))
                n_benchmarks *= 2

            output_data = []
            for y0, pub in zip(y0_list, sampler_pubs):
                results = simulate_pulse_level(pub, self.backend, y0)
                for result in results:
                    yf = result.y[-1]
                    tf = result.t[-1]
                    rotate_frame(yf, tf, self.backend)
                    output_data.append(yf)

            # Reshape data to isolate benchmarks (Output type can be either State or Channel, and for both qc and qc_nreps)

            output_data = [
                output_data[i * data_length : (i + 1) * data_length]
                for i in range(n_benchmarks)
            ]
            # Reorder data to match the order of the circuits
            qc_data_mapping = {"qc_state": output_data[0], "qc_channel": output_data[1]}
            if qc_nreps is not None:
                qc_data_mapping["qc_state_nreps"] = output_data[2]
                qc_data_mapping["qc_channel_nreps"] = output_data[3]

            circuit_order = [
                "qc_state",
                "qc_state_nreps",
                "qc_channel",
                "qc_channel_nreps",
            ]
            new_output_data = [
                qc_data_mapping.get(name, None)
                for name in circuit_order
                if name in qc_data_mapping
            ]
            output_data = new_output_data

        else:  # Standard Dynamics simulation

            y0_list = (
                [y0_state] * n_benchmarks * data_length
            )  # Initial state for each benchmark

            if qc.num_qubits < 3 and isinstance(
                self.target, GateTarget
            ):  # Benchmark channel only for 1-2 qubits
                y0_list += [y0_gate] * n_benchmarks * data_length
                circuits_list += circuits + circuits_n_reps
                n_benchmarks *= (
                    2  # Double the number of benchmarks to include channel fidelity
                )
            # Simulate all circuits
            output_data = []
            results = self.backend.solve(circuits_list, y0=y0_list)
            for solver_result in results:
                yf = solver_result.y[-1]
                tf = solver_result.t[-1]
                yf = rotate_frame(yf, tf, self.backend)

                output_data.append(yf)

                # Reshape data to isolate benchmarks (Output type can be either State or Channel, and for both qc and qc_nreps)
            output_data = [
                output_data[i * data_length : (i + 1) * data_length]
                for i in range(n_benchmarks)
            ]

        if self.n_reps > 1:  # Benchmark both qc and qc_nreps
            circ_list = [qc, qc_nreps, qc, qc_nreps]
            fid_arrays = [
                self.circuit_fidelity_history,
                self.avg_fidelity_history,
                self.circuit_fidelity_history_nreps,
                self.avg_fidelity_history_nreps,
            ]
        else:  # Benchmark only qc
            circ_list = [qc] * 2
            fid_arrays = [self.circuit_fidelity_history, self.avg_fidelity_history]

        for circ, data, fid_array in zip(circ_list, output_data, fid_arrays):
            n_reps = 1 if "nreps" not in circ.name else self.n_reps

            if isinstance(data[0], (Statevector, DensityMatrix)):
                data = [projected_state(state, subsystem_dims) for state in data]
                if self.target.n_qubits != len(
                    subsystem_dims
                ):  # If state has less qubits than the backend, trace out the rest
                    data = [
                        partial_trace(
                            state,
                            [
                                qubit
                                for qubit in range(state.num_qubits)
                                if qubit not in self.target.physical_qubits
                            ],
                        )
                        for state in data
                    ]
            elif isinstance(data[0], Operator):  # Project channel to qubit subspace
                data = [qubit_projection(op, subsystem_dims) for op in data]

            # Compute fidelities (type of input automatically detected and handled -> state -> state fidelity, channel -> gate fidelity)
            fidelities = [
                (
                    self.target.fidelity(output, n_reps, validate=False)
                    if n_reps > 1
                    else self.target.fidelity(output, validate=False)
                )
                for output in data
            ]

            if (
                returned_fidelity_type == "gate"
            ):  # Optimize gate fidelity by finding optimal Z-rotations before and after gate
                fidelities = self._handle_virtual_rotations(
                    data, fidelities, subsystem_dims, n_reps
                )
                if n_reps == 1:
                    returned_fidelities = fidelities
            elif returned_fidelity_type == "state" and n_reps == 1:
                returned_fidelities = fidelities
            if update_env_history:
                fid_array.append(np.mean(fidelities))

        return returned_fidelities

    def _handle_virtual_rotations(self, operations, fidelities, subsystem_dims, n_reps):
        """
        Optimize gate fidelity by finding optimal Z-rotations before and after gate
        """
        best_op = operations[np.argmax(fidelities)]
        res = get_optimal_z_rotation(
            best_op, self.target.target_operator.power(n_reps), len(subsystem_dims)
        )
        rotated_unitaries = [rotate_unitary(res.x, op) for op in operations]
        fidelities = [self.target.fidelity(op, n_reps) for op in rotated_unitaries]

        return fidelities

    def update_gate_calibration(self, gate_name: Optional[str] = None):
        """
        Update gate calibration parameters

        :param gate_name: Name of custom gate to add to target (if None,
         use target gate and update its attached calibration)
        """
        raise NotImplementedError(
            "Gate calibration not implemented for this environment"
        )

    def modify_environment_params(self, **kwargs):
        """
        Modify environment parameters (can be overridden by subclasses to modify specific parameters)
        """
        pass

    @property
    def config(self):
        return self._training_config

    @property
    def estimator(self) -> BaseEstimatorV2:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimatorV2):
        self._estimator = estimator

    @property
    def sampler(self) -> BaseSamplerV2:
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSamplerV2):
        self._sampler = sampler

    @property
    def primitive(self) -> BaseEstimatorV2 | BaseSamplerV2:
        if (
            self.config.reward_method == "state"
            or self.config.reward_method == "channel"
        ):
            return self.estimator
        else:
            return self.sampler

    @property
    def physical_target_qubits(self):
        return self._physical_target_qubits

    @property
    def physical_neighbor_qubits(self):
        return retrieve_neighbor_qubits(
            self.backend_info.coupling_map, self.physical_target_qubits
        )

    @property
    def physical_next_neighbor_qubits(self):
        return retrieve_neighbor_qubits(
            self.backend_info.coupling_map,
            self.physical_target_qubits + self.physical_neighbor_qubits,
        )

    @property
    @abstractmethod
    def tgt_instruction_counts(self) -> int:
        """
        Number of occurrences of the target instruction in the circuit
        """
        raise NotImplementedError("Target instruction counts not implemented")

    @property
    def fidelity_history(self):
        return (
            self.avg_fidelity_history
            if self.target.target_type == "gate"
            and self.target.target_circuit.num_qubits <= 3
            else self.circuit_fidelity_history
        )

    @property
    def step_tracker(self):
        return self._step_tracker

    @property
    def abstraction_level(self):
        """
        Return the abstraction level of the environment (can be 'circuit' or 'pulse')
        """
        return "pulse" if self.circuits[0].calibrations else "circuit"

    @step_tracker.setter
    def step_tracker(self, step: int):
        assert step >= 0, "step must be positive integer"
        self._step_tracker = step

    def signal_handler(self, signum, frame):
        """Signal handler for SIGTERM and SIGINT signals."""
        print(f"Received signal {signum}, closing environment...")
        self.close()

    def close(self) -> None:
        if hasattr(self.estimator, "session"):
            self.estimator.session.close()

    def clear_history(self):
        """
        Clear all stored data to start new training.
        """
        self._step_tracker = 0
        self._episode_tracker = 0
        self.action_history.clear()
        self.reward_history.clear()
        self._total_shots.clear()
        self._hardware_runtime.clear()
        self.avg_fidelity_history.clear()
        self.process_fidelity_history.clear()
        self.circuit_fidelity_history.clear()

    @property
    def benchmark_cycle(self) -> int:
        """
        Cycle at which fidelity benchmarking is performed
        :return:
        """
        return self.config.benchmark_cycle

    @benchmark_cycle.setter
    def benchmark_cycle(self, step: int) -> None:
        """
        Set cycle at which fidelity benchmarking is performed
        :param step:
        :return:
        """
        assert step >= 0, "Cycle needs to be a positive integer"
        self.config.benchmark_cycle = step

    def do_benchmark(self) -> bool:
        """
        Check if benchmarking should be performed at current step
        :return:
        """
        if self.config.reward_method == "fidelity":
            return True
        elif self.benchmark_cycle == 0:
            return False
        else:
            return self._episode_tracker % self.benchmark_cycle == 0

    def _get_info(self) -> Any:
        step = self._episode_tracker
        if self._episode_ended:
            if self.do_benchmark():
                info = {
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "circuit fidelity": self.fidelity_history[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "max circuit fidelity": np.max(self.fidelity_history),
                    "arg max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "arg max circuit fidelity": np.argmax(self.fidelity_history),
                    "optimal action": self.optimal_action,
                }
            else:
                info = {
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "optimal action": self.optimal_action,
                }
        else:
            info = {
                "reset_stage": self._inside_trunc_tracker == 0,
                "step": step,
                "gate_index": self._inside_trunc_tracker,
                "truncation_index": self.trunc_index,
            }
        return info

    def _ident_str(self):
        """This is a one-line description of the environment with some key parameters."""
        if isinstance(self.target, GateTarget):
            ident_str = f"gate_calibration_{self.target.gate.name}-gate_physical_qubits_{'-'.join(map(str, self.target.physical_qubits))}"
        elif isinstance(self.target, StateTarget):
            ident_str = f"state_preparation_physical_qubits_{'-'.join(map(str, self.target.physical_qubits))}"
        else:
            raise ValueError("Target type not recognized")
        return ident_str

    def __repr__(self):
        string = f"QuantumEnvironment composed of {self.n_qubits} qubits, \n"
        string += (
            f"Defined target: {self.target.target_type} "
            f"({self.target.gate if isinstance(self.target, GateTarget) else self.target.dm})\n"
        )
        string += f"Physical qubits: {self.target.physical_qubits}\n"
        string += f"Backend: {self.backend},\n"
        string += f"Abstraction level: {self.abstraction_level},\n"
        string += f"Run options: N_shots ({self.n_shots}), Sampling_Pauli_space ({self.sampling_Pauli_space}), \n"
        string += f"Batch size: {self.batch_size}, \n"
        return string

    # Properties

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int):
        try:
            assert size > 0 and isinstance(size, int)
            self._batch_size = size
        except AssertionError:
            raise ValueError("Batch size should be positive integer.")

    @property
    def target(self) -> GateTarget | StateTarget:
        """
        Return the target object (GateTarget | StateTarget) of the environment
        """
        return self._target

    @property
    def n_qubits(self):
        return self.target.n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        assert (
            isinstance(n_qubits, int) and n_qubits > 0
        ), "n_qubits must be a positive integer"
        self.target.n_qubits = n_qubits

    @property
    def layout(self):
        return self.target.layout

    @layout.setter
    def layout(self, layout: Layout):
        if not isinstance(layout, Layout):
            raise ValueError("Layout should be a Qiskit Layout object")
        self.target.layout = layout

    @property
    @abstractmethod
    def parameters(
        self,
    ) -> List[ParameterVector | List[Parameter]] | ParameterVector | List[Parameter]:
        """
        Return the Qiskit Parameter(s) instance(s) defining the abstract actions applied on the environment
        """
        raise NotImplementedError("Parameters not implemented")

    @property
    def involved_qubits(self):
        """
        Return the qubits involved in the calibration task
        """
        return list(self.layout.get_physical_bits().keys())

    @property
    def backend_info(self) -> BackendInfo:
        """
        Return the backend information object
        """
        return self._backend_info

    @property
    def pass_manager(self) -> Optional[PassManager]:
        """
        Return the custom pass manager for transpilation (if specified)
        """
        return self.backend_info.pass_manager

    @property
    def observables(self) -> SparsePauliOp:
        """
        Return set of observables sampled for current epoch of training (relevant only for
        direct fidelity estimation methods, e.g. 'channel' or 'state')
        """
        return self._observables

    @property
    def total_shots(self):
        return self._total_shots

    @property
    def hardware_runtime(self):
        return self._hardware_runtime

    @property
    def n_actions(self):
        return self.action_space.shape[-1]

    @property
    def optimal_action(self):
        return self._optimal_action

    @property
    def mean_action(self):
        return self._mean_action

    @mean_action.setter
    def mean_action(self, value):
        self._mean_action = np.array(value)

    @property
    def std_action(self):
        return self._std_action

    @std_action.setter
    def std_action(self, value):
        self._std_action = np.array(value)

    @property
    def ident_str(self):
        return self._ident_str()

    @property
    @abstractmethod
    def trunc_index(self) -> int:
        """
        Index of the truncation to be applied
        """
        raise NotImplementedError("Truncation index not implemented")

    @property
    def metadata(self):
        """
        Return metadata of the environment
        """
        return self.config.env_metadata

    def to_json(self):
        return json.dumps(
            {
                "n_qubits": self.n_qubits,
                "config": asdict(self.config),
                "abstraction_level": self.abstraction_level,
                "sampling_Pauli_space": self.sampling_Pauli_space,
                "n_shots": self.n_shots,
                "target_type": self.target.target_type,
                "target": self.target,
                "c_factor": self.c_factor,
                "reward_history": self.reward_history,
                "action_history": self.action_history,
                "fidelity_history": (
                    self.avg_fidelity_history
                    if self.target.target_type == "gate"
                    else self.circuit_fidelity_history
                ),
            }
        )
