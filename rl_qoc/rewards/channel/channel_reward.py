from itertools import product
from typing import List, Tuple, Optional, Literal

from qiskit import ClassicalRegister
from qiskit.circuit.classical.types import Uint
from qiskit.primitives import BaseEstimatorV2

from ..base_reward import Reward
from dataclasses import dataclass, field
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, pauli_basis, PauliList
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit_experiments.library.tomography.basis import (
    Pauli6PreparationBasis,
    PauliMeasurementBasis,
)
import numpy as np

from ..real_time_utils import handle_real_time_n_reps
from ...environment.configuration.qconfig import QEnvConfig
from ...environment.target import GateTarget
from ...helpers import get_single_qubit_input_states, group_input_paulis_by_qwc
from ...helpers.circuit_utils import (
    group_pauli_pairs_by_qwc,
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

    sorting: Literal["default", "sorted"] = "default"
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
        return {}

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
        params: np.ndarray,
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

        n_reps = execution_config.current_n_reps
        n_shots = execution_config.n_shots
        control_flow = execution_config.control_flow_enabled
        c_factor = execution_config.c_factor

        Chi = target.Chi(n_reps)  # Characteristic function for cycle circuit repeated n_reps times
        if Chi is None:
            raise ValueError("Chi is not computed for more than 3 qubits")
        cutoff = 1e-4  # Cutoff for negligible probabilities
        non_zero_indices = np.nonzero(np.abs(Chi) > cutoff)[0]
        # Remove index 0 from non_zero_indices as it corresponds to the identity Pauli
        non_zero_indices = non_zero_indices[non_zero_indices != 0]
        sorted_indices = np.argsort(np.abs(Chi[non_zero_indices]))[
            ::-1
        ]  # Sort indices by Chi value

        basis = pauli_basis(num_qubits=n_qubits)
        basis_to_indices = {pauli: i for i, pauli in enumerate(basis)}

        id_coeff = c_factor * (Chi[0]/dim**2)  # Coefficient for the identity Pauli
        # id_coeff = 0
        non_zero_indices = non_zero_indices[sorted_indices]
        pair_indices = [
            np.unravel_index(sorted_index, (dim**2, dim**2)) for sorted_index in non_zero_indices
        ]
        pauli_pairs = [(basis[i], basis[j]) for (i, j) in pair_indices]
        # Build dictionary of Pauli pairs with their respective Chi values
        Chi_dict = {
            pair: {
                "chi": Chi[non_zero_indices[i]],
                "indices": pair_indices[i],
                "index": non_zero_indices[i],
            }
            for i, pair in enumerate(pauli_pairs)
        }
        grouped_pauli_pairs = group_pauli_pairs_by_qwc(pauli_pairs)

        grouped_chi = [
            np.array(
                [Chi_dict[(pauli_in, pauli_obs)]["chi"] for pauli_in, pauli_obs in zip(*group)]
            )
            for group in grouped_pauli_pairs
        ]
        grouped_probabilities = [np.sum([chi**2 for chi in Chi]) / (dim**2) for Chi in grouped_chi]
        grouped_probabilities = np.array(grouped_probabilities) / np.sum(grouped_probabilities)

        # Build repeated circuit
        repeated_circuit = handle_n_reps(qc, n_reps, backend_info.backend, control_flow)
        prep_basis = Pauli6PreparationBasis()  # Pauli6 basis for input state preparation

        if dfe_precision is not None:
            # DFE precision guarantee, ϵ additive error, δ failure probability
            print("DFE precision guarantee")
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            # User-defined hyperparameter
            pauli_sampling = execution_config.sampling_paulis

        # Sample a list of input/observable pairs
        # If one pair is sampled repeatedly, it increases number of shots used to estimate its expectation value

        sampled_group_indices, counts = np.unique(
            self.fiducials_rng.choice(
                len(grouped_pauli_pairs), size=pauli_sampling, p=grouped_probabilities
            ),
            return_counts=True,
        )

        # Calculate reward factors for each group
        reward_data = []
        sampled_grouped_pauli_pairs = []
        for c, idx in zip(counts, sampled_group_indices):
            chi_group = grouped_chi[idx]
            chi_squared_sum = np.sum([chi**2 for chi in chi_group])
            reward_factor = c_factor * c * chi_group / (dim * chi_squared_sum)
            prep, obs_list = grouped_pauli_pairs[idx]
            sampled_grouped_pauli_pairs.append((prep, SparsePauliOp(obs_list, reward_factor)))

        for g, group_info in enumerate(zip(sampled_grouped_pauli_pairs, counts)):
            # Each prep is a Pauli that we need to decompose in its pure eigenbasis.
            # Below, we select at random a subset of pure input states to prepare for each prep
            # If nb_states = dim, we prepare all pure input states for each Pauli prep (no random selection)
            (prep_group, obs_group), c = group_info
            selected_input_states, counts_input_states = np.unique(
                self.input_states_rng.choice(dim, size=c), return_counts=True
            )
            # Build representative of qubit-wise commuting Pauli group (the one with the highest weight)
            pauli_rep = Pauli(
                (np.logical_or.reduce(prep_group.z), np.logical_or.reduce(prep_group.x))
            )
            prep_labels = [pauli.to_label() for pauli in prep_group]
            if dfe_precision is None:
                # Number of shots per Pauli eigenstate (divided equally)
                dedicated_shots = counts_input_states * n_shots
            else: # DFE precision guarantee
                dedicated_shots = 2 * np.log(2 / delta) / (pauli_sampling * eps**2)
                dedicated_shots *= np.sum(grouped_chi[sampled_group_indices[g]]) ** 2
                dedicated_shots /= np.sum(grouped_chi[sampled_group_indices[g]] ** 2) ** 2 
                dedicated_shots *= counts_input_states # Convert into an array for each sampled input state

            for pure_eig_state, shots in zip(selected_input_states, dedicated_shots):
                # Convert input state to Pauli6 basis:
                # preparing pure eigenstates of Pauli_prep
                if shots == 0:
                    continue
                inputs = np.unravel_index(pure_eig_state, (2,) * n_qubits)
                inputs = tuple(int(i) for i in inputs)  # Convert to tuple for indexing
                parity = [
                    np.prod(
                        [
                            (-1) ** inputs[q_idx]
                            for q_idx, term in enumerate(reversed(label))
                            if term != "I"
                        ]
                    )
                    for label in prep_labels
                ]

                prep_indices = pauli_input_to_indices(pauli_rep, inputs)

                # Create input state preparation circuit
                input_circuit = qc.copy_empty_like().compose(
                    prep_basis.circuit(prep_indices),
                    target.causal_cone_qubits,
                    inplace=False,
                )
                input_circuit, extended_prep_indices = extend_input_state_prep(
                    input_circuit, qc, target, prep_indices
                )  # Add random input state on other qubits
                input_circuit.metadata["indices"] = extended_prep_indices
                # Prepend input state to custom circuit with front composition
                prep_circuit = repeated_circuit.compose(
                    input_circuit,
                    front=True,
                    inplace=False,
                )
                # Transpile circuit to decompose input state preparation
                prep_circuit: QuantumCircuit = backend_info.custom_transpile(
                    prep_circuit,
                    initial_layout=target.layout,
                    scheduling=False,
                    optimization_level=0,
                )

                obs_ = [p * o for p, o in zip(parity, obs_group)]
                pub_obs = extend_observables(
                    SparsePauliOp.sum(obs_).simplify(),
                    prep_circuit,
                    target.causal_cone_qubits_indices,
                )
                # Check if coeff array is all 0. If so, skip this pub
                if np.all(pub_obs.coeffs <= 1e-6):
                    continue
                # pub_obs = pub_obs.apply_layout(prep_circuit.layout)
                # Add PUB
                pub = (
                    prep_circuit,
                    pub_obs,
                    params,
                    shots_to_precision(shots),
                )

                reward_data.append(
                    ChannelRewardData(
                        pub,
                        input_circuit,
                        obs_,
                        n_reps,
                        target.causal_cone_qubits_indices,
                        prep_group,
                        extended_prep_indices,
                        observables_to_indices(SparsePauliOp.sum(obs_).simplify()),
                    )
                )

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
        dim = 2**reward_data.num_qubits
        pub_results = job.result()
        reward = np.sum([pub_result.data.evs for pub_result in pub_results], axis=0)
        reward *= (dim**2 - 1) / dim**2
        reward /= reward_data.pauli_sampling
        reward += reward_data.id_coeff

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

        if not isinstance(target, GateTarget):
            raise ValueError("Channel reward can only be computed for a target gate")
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
                        if input_circuit.data:
                            qc.compose(input_circuit, qubit, inplace=True)
                        else:
                            qc.delay(16, qubit)

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
                    basis_rot_circuit = (
                        meas_basis.circuit([i]).decompose().remove_final_measurements(False)
                    )
                    with case_observable(i):
                        if basis_rot_circuit.data:
                            # Apply the measurement basis circuit to the qubit
                            qc.compose(basis_rot_circuit, qubit, inplace=True)
                        else:
                            qc.delay(16, qubit)

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
