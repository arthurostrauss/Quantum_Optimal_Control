from typing import List, Tuple, Optional

from qiskit import ClassicalRegister
from qiskit.circuit.classical.types import Uint
from qiskit.primitives import BaseEstimatorV2

from ..base_reward import Reward
from dataclasses import dataclass, field
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, pauli_basis
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit_experiments.library.tomography.basis import (
    Pauli6PreparationBasis,
    PauliMeasurementBasis,
)
import numpy as np

from ..real_time_utils import handle_real_time_n_reps
from ...environment.configuration.qconfig import QEnvConfig
from ...environment.target import GateTarget
from ...helpers import get_single_qubit_input_states
from ...helpers.circuit_utils import (
    handle_n_reps,
    extend_observables,
    extend_input_state_prep,
    observables_to_indices,
    pauli_input_to_indices,
    precision_to_shots,
    shots_to_precision,
)
from ...helpers.helper_functions import validate_circuit_and_target
from .channel_reward_data import ChannelRewardData, ChannelRewardDataList


@dataclass
class ChannelReward(Reward):
    """
    Configuration for computing the reward based on channel fidelity estimation
    """

    num_eigenstates_per_pauli: int = 1
    fiducials_seed: int = 2000
    input_states_seed: int = 2001
    fiducials_rng: np.random.Generator = field(init=False)
    input_states_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.fiducials_rng = np.random.default_rng(self.fiducials_seed)
        self.input_states_rng = np.random.default_rng(self.input_states_seed)

    def set_reward_seed(self, seed: int):
        """
        Set the seed for the random number generators used in the reward computation
        """
        self.fiducials_seed = seed + 30
        self.input_states_seed = seed + 31
        self.fiducials_rng = np.random.default_rng(self.fiducials_seed)
        self.input_states_rng = np.random.default_rng(self.input_states_seed)

    @property
    def reward_args(self):
        return {"num_eigenstates_per_pauli": self.num_eigenstates_per_pauli}

    @property
    def reward_method(self):
        return "channel"

    @property
    def observables(self) -> SparsePauliOp:
        """
        Pauli observables to sample
        """
        return self._observables

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        env_config: QEnvConfig,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ) -> ChannelRewardDataList:
        """
        Compute reward data related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            env_config: QEnvConfig containing the backend information and execution configuration
            dfe_precision: Tuple (Ɛ, δ) from DFE paper

        Returns:
            List of pubs related to the reward method
        """
        if not isinstance(target, GateTarget):
            raise ValueError("Channel reward can only be computed for a target gate")

        if target.causal_cone_size > 3:
            raise ValueError(
                "Channel reward can only be computed for a target gate with causal cone size <= 3"
            )
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info
        n_qubits = target.causal_cone_size
        dim = 2**n_qubits
        nb_states = self.num_eigenstates_per_pauli
        if nb_states >= dim:
            raise ValueError(
                f"Number of eigenstates per Pauli should be less than or equal to {dim}"
            )

        n_reps = execution_config.current_n_reps
        n_shots = execution_config.n_shots
        control_flow = execution_config.control_flow_enabled
        c_factor = execution_config.c_factor

        Chi = target.Chi(n_reps)  # Characteristic function for cycle circuit repeated n_reps times

        # Build repeated circuit
        repeated_circuit = handle_n_reps(qc, n_reps, backend_info.backend, control_flow)
        prep_basis = Pauli6PreparationBasis()

        probabilities = Chi**2 / (dim**2)
        cutoff = 1e-8
        non_zero_indices = np.nonzero(probabilities > cutoff)[0]
        non_zero_probabilities = probabilities[non_zero_indices]
        non_zero_probabilities /= np.sum(non_zero_probabilities)

        basis = pauli_basis(num_qubits=n_qubits)  # all n_fold tensor-product single qubit Paulis

        if dfe_precision is not None:
            # DFE precision guarantee, ϵ additive error, δ failure probability
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            # User-defined hyperparameter
            pauli_sampling = execution_config.sampling_paulis

        # Sample a list of input/observable pairs
        # If one pair is sampled repeatedly, it increases number of shots used to estimate its expectation value

        self.total_counts = pauli_sampling

        samples, counts = np.unique(
            self.fiducials_rng.choice(
                non_zero_indices, size=pauli_sampling, p=non_zero_probabilities
            ),
            return_counts=True,
        )

        # Convert samples to a pair of indices in the Pauli basis
        pauli_indices = np.array(
            [np.unravel_index(sample, (dim**2, dim**2)) for sample in samples],
            dtype=int,
        )

        # Filter out case where identity ('I'*n_qubits) is sampled (trivial case)
        identity_terms = np.where(pauli_indices[:, 1] == 0)[0]
        id_coeff = (
            c_factor * dim * np.sum(counts[identity_terms] / (dim * Chi[samples[identity_terms]]))
        )
        # Additional dim factor to account for all eigenstates of identity input state
        self.id_coeff = (
            c_factor * dim * np.sum(counts[identity_terms] / (dim * Chi[samples[identity_terms]]))
        )  # Additional dim factor to account for all eigenstates of identity input state
        self._id_count = len(identity_terms)

        pauli_indices = np.delete(pauli_indices, identity_terms, axis=0)
        samples = np.delete(samples, identity_terms)
        counts = np.delete(counts, identity_terms)

        reward_factor = c_factor * counts / (dim * Chi[samples])  # Based on DFE estimator
        fiducials_list = [
            (basis[p[0]], SparsePauliOp(basis[p[1]], r))
            for p, r in zip(pauli_indices, reward_factor)
        ]

        self._observables = SparsePauliOp("I" * n_qubits, self.id_coeff) + sum(
            [obs for _, obs in fiducials_list]
        )

        obs_dict = {}
        # Regroup observables for same input Pauli
        for i, (prep, obs) in enumerate(fiducials_list):
            label = prep.to_label()
            if label not in obs_dict:
                obs_dict[label] = (prep, obs, counts[i])
            else:
                _, ref_obs, ref_count = obs_dict[label]
                obs_dict[label] = (
                    prep,
                    (ref_obs + obs).simplify(),
                    max(ref_count, counts[i]),
                )

        filtered_fiducials_list = [(prep, obs) for prep, obs, _ in obs_dict.values()]
        filtered_pauli_shots = [count for _, _, count in obs_dict.values()]

        fiducials = filtered_fiducials_list
        pauli_shots = filtered_pauli_shots

        used_prep_indices = {}

        for (prep, obs_list), shots in zip(fiducials, pauli_shots):
            # Each prep is a Pauli input state, that we need to decompose in its pure eigenbasis.
            # Below, we select at random a subset of pure input states to prepare for each prep.
            # If nb_states = 1, we prepare all pure input states for each Pauli prep (no random selection)

            # self._fiducials_indices.append(([], observables_to_indices(obs_list)))
            # self._full_fiducials.append(([], obs_list))
            max_input_states = dim // nb_states
            selected_input_states: List[int] = self.input_states_rng.choice(
                dim, size=max_input_states, replace=False
            )
            prep_label = prep.to_label()
            dedicated_shots = (
                shots * n_shots // max_input_states
            )  # Number of shots per Pauli eigenstate (divided equally)
            if dedicated_shots == 0:
                continue

            for pure_eig_state in selected_input_states:
                # Convert input state to Pauli6 basis:
                # preparing pure eigenstates of Pauli_prep

                inputs = np.unravel_index(pure_eig_state, (2,) * n_qubits)
                parity = np.prod(
                    [
                        (-1) ** inputs[q_idx]
                        for q_idx, term in enumerate(reversed(prep_label))
                        if term != "I"
                    ]
                )

                prep_indices = pauli_input_to_indices(prep, inputs)

                # self._fiducials_indices[-1][0].append(tuple(prep_indices))

                # Create input state preparation circuit
                input_circuit = qc.copy_empty_like().compose(
                    prep_basis.circuit(prep_indices),
                    target.causal_cone_qubits,
                    inplace=False,
                )
                input_circuit, extended_prep_indices = extend_input_state_prep(
                    input_circuit, qc, target, prep_indices
                )  # Add random input state on other qubits
                prep_indices = tuple(prep_indices)
                input_circuit.metadata["indices"] = extended_prep_indices
                # Prepend input state to custom circuit with front composition
                prep_circuit = repeated_circuit.compose(
                    input_circuit,
                    front=True,
                    inplace=False,
                )
                # Transpile circuit to decompose input state preparation
                prep_circuit = backend_info.custom_transpile(
                    prep_circuit,
                    initial_layout=target.layout,
                    scheduling=False,
                    optimization_level=0,
                )

                obs_ = parity * obs_list
                pub_obs = extend_observables(obs_, prep_circuit, target.causal_cone_qubits_indices)
                # pub_obs = pub_obs.apply_layout(prep_circuit.layout)
                if prep_indices not in used_prep_indices:  # Add new PUB
                    # Add PUB
                    pub = (
                        prep_circuit,
                        pub_obs,
                        params,
                        shots_to_precision(dedicated_shots),
                    )
                    used_prep_indices[prep_indices] = ChannelRewardData(
                        pub,
                        input_circuit,
                        obs_,
                        n_reps,
                        target.causal_cone_qubits_indices,
                        prep,
                        extended_prep_indices,
                        observables_to_indices(obs_),
                    )

                else:  # Update PUB (regroup observables for same input circuit, redundant for I/Z terms)
                    ref_prep = used_prep_indices[prep_indices].pub.circuit
                    ref_obs = used_prep_indices[prep_indices].observables
                    ref_precision = used_prep_indices[prep_indices].precision

                    ref_shots = precision_to_shots(ref_precision)
                    new_precision = shots_to_precision(dedicated_shots)
                    new_obs = (ref_obs + obs_).simplify()

                    used_prep_indices[prep_indices].observables = new_obs
                    used_prep_indices[prep_indices].pub = EstimatorPub.coerce(
                        (
                            ref_prep,
                            extend_observables(
                                new_obs, prep_circuit, target.causal_cone_qubits_indices
                            ),
                            params,
                            min(ref_precision, new_precision),
                        ),
                    )

                    used_prep_indices[prep_indices].shots = max(dedicated_shots, ref_shots)
                    used_prep_indices[prep_indices].observables_indices = observables_to_indices(
                        new_obs
                    )

        reward_data = []
        for data in used_prep_indices.values():
            reward_data.append(data)

        reward_data = ChannelRewardDataList(reward_data, pauli_sampling, id_coeff)
        return reward_data

    def get_reward_with_primitive(
        self,
        reward_data: ChannelRewardDataList,
        estimator: BaseEstimatorV2,
    ) -> np.array:
        """
        Retrieve the reward from the PUBs and the primitive
        """
        job = estimator.run(reward_data.pubs)
        pub_results = job.result()
        reward = np.sum([pub_result.data.evs for pub_result in pub_results], axis=0)
        reward += reward_data.id_coeff
        reward /= reward_data.pauli_sampling
        dim = 2**reward_data.num_qubits
        reward = (dim * reward + 1) / (dim + 1)

        return reward

    def get_shot_budget(self, pubs: List[EstimatorPub]) -> int:
        """
        Retrieve number of shots associated to the input pub list
        """
        total_shots = 0
        for pub in pubs:
            observables = []
            for obs_dict in pub.observables.ravel().tolist():
                for pauli_string, coeff in obs_dict.items():
                    observables.append((pauli_string, coeff))
            observables = SparsePauliOp.from_list(observables).simplify()

            total_shots += (
                pub.parameter_values.shape[0]
                * len(observables.group_commuting(qubit_wise=True))
                * precision_to_shots(pub.precision)
            )
        return total_shots

    def get_real_time_circuit(
        self,
        circuits: QuantumCircuit | List[QuantumCircuit],
        target: GateTarget,
        env_config: QEnvConfig,
        skip_transpilation: bool = False,
        *args,
    ) -> QuantumCircuit:

        n_reps = env_config.current_n_reps
        all_n_reps = env_config.n_reps

        prep_circuits = [circuits] if isinstance(circuits, QuantumCircuit) else circuits
        qubits = [qc.qubits for qc in prep_circuits]
        if not all(qc.qubits == qubits[0] for qc in prep_circuits):
            raise ValueError("All circuits must have the same qubits")

        qc = prep_circuits[0].copy_empty_like("real_time_channel_qc")
        qc.reset(qc.qubits)
        num_qubits = qc.num_qubits

        n_reps_var = qc.add_input("n_reps", Uint(8)) if len(all_n_reps) > 1 else n_reps

        if not qc.clbits:
            meas = ClassicalRegister(target.causal_cone_size, name="meas")
            qc.add_register(meas)
        else:
            meas = qc.cregs[0]
            if meas.size != target.causal_cone_size:
                raise ValueError("Classical register size must match the target causal cone size")

        input_state_vars = [qc.add_input(f"input_state_{i}", Uint(8)) for i in range(num_qubits)]
        observables_vars = [
            qc.add_input(f"observable_{i}", Uint(4)) for i in range(target.causal_cone_size)
        ]
        input_circuits = [circ.decompose() for circ in get_single_qubit_input_states("pauli6")]

        for q, qubit in enumerate(qc.qubits):
            # Input state prep (over all qubits of the circuit context)
            with qc.switch(input_state_vars[q]) as case_input_state:
                for i, input_circuit in enumerate(input_circuits):
                    with case_input_state(i):
                        qc.compose(input_circuit, qubit, inplace=True)

        if len(prep_circuits) > 1:  # Switch over possible contexts
            circuit_choice = qc.add_input("circuit_choice", Uint(8))
            with qc.switch(circuit_choice) as case_circuit:
                for i, prep_circuit in enumerate(prep_circuits):
                    with case_circuit(i):
                        handle_real_time_n_reps(all_n_reps, n_reps_var, prep_circuit, qc)
        else:
            handle_real_time_n_reps(all_n_reps, n_reps_var, prep_circuits[0], qc)

        # Local Basis rotation handling
        meas_basis = PauliMeasurementBasis()
        for q, qubit in enumerate(target.causal_cone_qubits):
            with qc.switch(observables_vars[q]) as case_observable:
                for i in range(3):
                    with case_observable(i):
                        qc.compose(
                            meas_basis.circuit([i]).decompose().remove_final_measurements(False),
                            qubit,
                            inplace=True,
                        )

        # Measurement
        qc.measure(target.causal_cone_qubits, meas)

        if skip_transpilation:
            return qc

        return env_config.backend_info.custom_transpile(
            qc,
            optimization_level=1,
            initial_layout=target.layout,
            scheduling=False,
            remove_final_measurements=False,
        )

    def compute_expectation_values(
        self,
        qc: QuantumCircuit,
        target: GateTarget,
        env_config: QEnvConfig,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ) -> List[float]:
        """
        Compute the expectation values of the desired observables on the given circuit

        Args:
            qc: Quantum circuit to be executed on quantum system
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
            dfe_precision: Tuple (Ɛ, δ) from DFE paper

        Returns:
            List of expectation values of the desired observables
        """
        from qiskit.quantum_info import Operator, Statevector

        if not isinstance(target, GateTarget):
            raise ValueError("Channel reward can only be computed for a target gate")

        if target.causal_cone_size > 3:
            raise ValueError(
                "Channel reward can only be computed for a target gate with causal cone size <= 3"
            )
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info
        n_qubits = target.causal_cone_size
        n_reps = execution_config.current_n_reps
        control_flow = execution_config.control_flow_enabled
        c_factor = execution_config.c_factor
        dim = 2**n_qubits
        Chi = target.Chi(n_reps)  # Characteristic function for cycle circuit repeated n_reps times

        # Reset storage variables
        self._fiducials_indices = []
        self._full_fiducials = []
        self.id_coeff = 0.0
        self.id_count = 0

        # Build repeated circuit
        repeated_circuit = handle_n_reps(qc, n_reps, backend_info.backend, control_flow)

        nb_states = (
            self.num_eigenstates_per_pauli
        )  # Number of eigenstates per Pauli input to sample
        if nb_states >= dim:
            raise ValueError(
                f"Number of eigenstates per Pauli should be less than or equal to {dim}"
            )

        probabilities = Chi**2 / (dim**2)
        non_zero_indices = np.nonzero(probabilities)[0]  # Filter out zero probabilities
        non_zero_probabilities = probabilities[non_zero_indices]

        basis = pauli_basis(num_qubits=n_qubits)  # all n_fold tensor-product single qubit Paulis

        if (
            dfe_precision is not None
        ):  # Choose if we want to sample Paulis based on DFE precision guarantee
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            pauli_sampling = execution_config.sampling_paulis

        # Sample a list of input/observable pairs
        # If one pair is sampled repeatedly, it increases number of shots used to estimate its expectation value

        self.total_counts = pauli_sampling

        samples, counts = np.unique(
            self.fiducials_rng.choice(
                non_zero_indices, size=pauli_sampling, p=non_zero_probabilities
            ),
            return_counts=True,
        )

        # Convert samples to a pair of indices in the Pauli basis
        pauli_indices = np.array(
            [np.unravel_index(sample, (dim**2, dim**2)) for sample in samples],
            dtype=int,
        )

        # Filter the case where the identity observable is sampled (trivial case as it serves as offset coefficient)
        filtered_pauli_indices, filtered_samples, filtered_counts = [], [], []
        for i, (indices, sample) in enumerate(zip(pauli_indices, samples)):
            if indices[1] == 0:  # 0 is the index corresponding to 'I'*n_qubits
                self.id_coeff += c_factor * counts[i] / (dim * Chi[sample])
                self.id_count += 1
            else:
                filtered_pauli_indices.append(indices)
                filtered_samples.append(sample)
                filtered_counts.append(counts[i])

        pauli_indices = filtered_pauli_indices
        samples = filtered_samples
        counts = filtered_counts

        reward_factor = [
            c_factor * count / (dim * Chi[p]) for count, p in zip(counts, samples)
        ]  # Based on DFE estimator
        fiducials_list = [
            (basis[p[0]], extend_observables(SparsePauliOp(basis[p[1]], r), qc, target))
            for p, r in zip(pauli_indices, reward_factor)
        ]

        self._observables = SparsePauliOp("I" * n_qubits, self.id_coeff) + sum(
            [obs for _, obs in fiducials_list]
        )
        filtered_fiducials_list, filtered_pauli_shots, used_prep_paulis = [], [], []
        for i, (prep, obs) in enumerate(fiducials_list):
            # Regroup observables for same input Pauli
            label = prep.to_label()
            should_add_pauli = label not in used_prep_paulis
            if should_add_pauli:
                used_prep_paulis.append(label)
                filtered_fiducials_list.append((prep, obs))
                filtered_pauli_shots.append(counts[i])

            else:
                index = used_prep_paulis.index(label)
                _, ref_obs = filtered_fiducials_list[index]
                filtered_fiducials_list[index] = (
                    prep,
                    (ref_obs + obs).simplify(),
                )
                filtered_pauli_shots[index] = max(filtered_pauli_shots[index], counts[i])

        for i, (prep, obs) in enumerate(filtered_fiducials_list):
            filtered_fiducials_list[i] = (prep, obs.group_commuting(qubit_wise=True))

        self._fiducials = filtered_fiducials_list
        self._pauli_shots = filtered_pauli_shots

        noisy_expectation_values = []
        expectation_values = []
        repeated_unitary = Operator(repeated_circuit)

        for (prep, obs_list), shots in zip(self._fiducials, self._pauli_shots):
            prep: Pauli
            obs_list: List[SparsePauliOp]
            self._fiducials_indices.append(([], observables_to_indices(obs_list)))
            self._full_fiducials.append(([], obs_list))
            max_input_states = dim // nb_states
            selected_input_states: List[int] = self.input_states_rng.choice(
                dim, size=max_input_states, replace=False
            )

            for pure_eig_state in selected_input_states:
                # Convert input state to Pauli6 basis: preparing pure eigenstates of Pauli_prep
                prep_label = prep.to_label()
                inputs = np.unravel_index(pure_eig_state, (2,) * n_qubits)
                parity = 1
                for q_idx, symbol in enumerate(reversed(prep_label)):
                    if symbol != "I":
                        parity *= (-1) ** inputs[q_idx]

                # Create input state preparation circuit
                input_circuit = qc.copy_empty_like()
                input_circuit.compose(
                    Pauli6PreparationBasis().circuit(inputs),
                    target.causal_cone_qubits,
                    inplace=True,
                )  # Apply input state on causal cone qubits
                input_circuit, prep_indices = extend_input_state_prep(
                    input_circuit, qc, target, inputs
                )  # Add random input state on other qubits

                prep_circuit = repeated_circuit.compose(
                    input_circuit,
                    front=True,
                    inplace=False,
                )
                prep_circuit.save_expectation_value(parity * sum(obs_list), qc.qubits)
                backend = backend_info.backend
                prep_circuit = backend_info.custom_transpile(
                    prep_circuit,
                    initial_layout=target.layout,
                    scheduling=False,
                    optimization_level=0,
                )
                prep_circuit.metadata["indices"] = prep_indices
                self._full_fiducials[-1][0].append(prep_circuit)
                job = backend.run(prep_circuit, shots=shots)
                result = job.result()
                noisy_exp_value = result.data(0).get("expectation_value")
                noisy_expectation_values.append(noisy_exp_value)

                # Prepare the input state vector
                input_unitary = Operator(input_circuit)
                output_state = Statevector.from_int(0, (2,) * n_qubits)
                output_state = output_state.evolve(input_unitary).evolve(repeated_unitary)

                ideal_exp_value = output_state.expectation_value(parity * sum(obs_list))
                expectation_values.append(ideal_exp_value)

        return noisy_expectation_values, expectation_values
