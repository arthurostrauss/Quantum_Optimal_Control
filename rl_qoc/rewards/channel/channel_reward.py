from __future__ import annotations
from typing import List, Tuple, Optional, TYPE_CHECKING

from qiskit import ClassicalRegister
from qiskit.circuit.classical.types import Uint
from qiskit.primitives import BaseEstimatorV2, BitArray

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

if TYPE_CHECKING:
    from qiskit_qm_provider.parameter_table import ParameterTable, Parameter as QuaParameter
    from ...qua.circuit_params import CircuitParams
    from qm import Program


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
        backend_info = env_config.backend_config
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
    ) -> np.ndarray:
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

        return env_config.backend_config.custom_transpile(
            qc,
            optimization_level=1,
            initial_layout=target.layout,
            scheduling=False,
            remove_final_measurements=False,
        )

    def qm_step(
        self,
        reward_data: ChannelRewardDataList,
        fetching_index: int,
        fetching_size: int,
        circuit_params: CircuitParams,
        reward: QuaParameter,
        config: QEnvConfig,
        **push_args,
    ):
        """
        This function is used to compute the reward for the channel reward.
        It is used in the QMEnvironment.step() function.

        Args:
            reward_data: Reward data to be used to compute the reward (can be used to send inputs to the QUA program and also to post-process measurement outcomes/counts coming out of the QUA program)
            fetching_index: Index of the first measurement outcome to be fetched in stream processing / DGX Quantum stream
            fetching_size: Number of measurement outcomes to be fetched
            circuit_params: Parameters defining the quantum program to be executed, those are entrypoints towards streaming values to control-flow that define the program adaptively (e.g. input state, number of repetitions, observable, etc.)
            reward: Reward parameter to be used to fetch measurement outcomes from the QUA program and compute the reward
            config: Environment configuration
            **push_args: Additional arguments to pass necessary entrypoints to communicate with the OPX (e.g. job, qm, verbosity, etc.)

        Returns:
            Reward array of shape (batch_size,)
        """
        from ...qua.qm_config import QMConfig
        from ..real_time_utils import push_circuit_context

        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")
        if not isinstance(config.target, GateTarget):
            raise ValueError("Channel reward is only supported for GateTarget")
        reward_array = np.zeros(shape=(config.batch_size,))
        num_qubits = config.target.causal_cone_size
        dim = 2**num_qubits
        binary = lambda n, l: bin(n)[2:].zfill(l)
        input_indices = reward_data.input_indices
        max_input_state = len(input_indices)
        num_obs_per_input_state = tuple(
            len(reward_data.observables_indices[i]) for i in range(max_input_state)
        )

        push_circuit_context(circuit_params, config.target, **push_args)
        if circuit_params.n_reps_var is not None:
            # Number of repetitions of cycle circuit to vary
            circuit_params.n_reps_var.push_to_opx(config.current_n_reps, **push_args)

        circuit_params.max_input_state.push_to_opx(max_input_state, **push_args)
        for i, input_state in enumerate(input_indices):
            input_state_dict = {
                input_var: input_state_val
                for input_var, input_state_val in zip(
                    circuit_params.input_state_vars.parameters, input_state
                )
            }
            circuit_params.input_state_vars.push_to_opx(input_state_dict, **push_args)
            observables = reward_data.observables_indices[i]
            num_obs = len(observables)

            circuit_params.max_observables.push_to_opx(num_obs, **push_args)
            circuit_params.n_shots.push_to_opx(reward_data.shots[i], **push_args)
            for j, observable in enumerate(observables):
                observable_dict = {
                    observable_var: observable_val
                    for observable_var, observable_val in zip(
                        circuit_params.observable_vars.parameters, observable
                    )
                }
                circuit_params.observable_vars.push_to_opx(observable_dict, **push_args)
        collected_counts = reward.fetch_from_opx(
            push_args["job"],
            fetching_index=fetching_index,
            fetching_size=fetching_size,
            verbosity=config.backend_config.verbosity,
            time_out=config.backend_config.timeout,
        )
        counts = []
        formatted_counts = []
        count_idx = 0
        # Format the counts
        for i_idx in range(max_input_state):
            formatted_counts.append([])
            num_obs = num_obs_per_input_state[i_idx]
            for o_idx in range(num_obs):
                formatted_counts[i_idx].append([])
                counts_array = np.array(collected_counts[count_idx], dtype=int)
                formatted_counts[i_idx][o_idx] = counts_array
                count_idx += 1
        # Reshape the counts
        for batch_idx in range(config.batch_size):
            counts.append([])
            for i_idx in range(max_input_state):
                counts[batch_idx].append([])
                for o_idx in range(num_obs_per_input_state[i_idx]):
                    counts[batch_idx][i_idx].append(formatted_counts[i_idx][o_idx][batch_idx])
        # Compute the expectation values
        for batch_idx in range(config.batch_size):
            exp_value = 0.0
            for i_idx in range(max_input_state):
                obs_group = reward_data[i_idx].observables.group_commuting(True)
                for o_idx, obs in enumerate(obs_group):
                    counts_dict = {
                        binary(i, num_qubits): int(counts[batch_idx][i_idx][o_idx][i])
                        for i in range(dim)
                    }
                    bit_array = BitArray.from_counts(counts_dict, num_bits=num_qubits)
                    diag_obs = SparsePauliOp("I" * num_qubits, 0.0)
                    for obs_, coeff in zip(obs.paulis, obs.coeffs):
                        diag_obs_label = ""
                        for char in obs_.to_label():
                            diag_obs_label += char if char == "I" else "Z"
                        diag_obs += SparsePauliOp(diag_obs_label, coeff)
                    exp_value += bit_array.expectation_values(diag_obs.simplify())
            exp_value += reward_data.id_coeff
            exp_value /= reward_data.pauli_sampling
            reward_array[batch_idx] = exp_value
        return reward_array

    def rl_qoc_training_qua_prog(
        self,
        qc: QuantumCircuit,
        policy: ParameterTable,
        reward: QuaParameter,
        circuit_params: CircuitParams,
        config: QEnvConfig,
        num_updates: int = 1000,
        test: bool = False,
    ) -> Program:
        from qm.qua import (
            program,
            declare,
            Random,
            for_,
            stream_processing,
            assign,
            fixed,
        )
        from qiskit_qm_provider import QMBackend
        from qiskit_qm_provider.backend import get_measurement_outcomes
        from ...qua.qua_utils import rand_gauss_moller_box, rescale_and_clip_wrapper
        from ...qua.qm_config import QMConfig
        from ..real_time_utils import load_circuit_context

        if not isinstance(config.backend, QMBackend):
            raise ValueError("Backend must be a QMBackend")
        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")

        if circuit_params.max_input_state is None:
            raise ValueError("max_input_state should be set for Channel reward")
        if circuit_params.input_state_vars is None:
            raise ValueError("input_state_vars should be set for Channel reward")
        if circuit_params.n_shots is None:
            raise ValueError("n_shots should be set for Channel reward")
        if circuit_params.max_observables is None:
            raise ValueError("max_observables should be set for Channel reward")
        if circuit_params.observable_vars is None:
            raise ValueError("observable_vars should be set for Channel reward")
        if not isinstance(config.target, GateTarget):
            raise ValueError("Channel reward is only supported for GateTarget")
        policy.reset()
        reward.reset()
        circuit_params.reset()

        num_qubits = config.target.causal_cone_size
        dim = int(2**num_qubits)
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) >= 2:
                raise ValueError("Overlapping classical registers are not supported")
        with program() as rl_qoc_training_prog:
            # Declare the necessary variables (all are counters variables to loop over the corresponding hyperparameters)
            circuit_params.declare_variables()
            policy.declare_variables()
            reward.declare_variable()
            reward.declare_stream()

            n_u = declare(int)
            shots = declare(int)
            i_idx = declare(int)
            o_idx = declare(int)
            b = declare(int)
            j = declare(int)
            tmp1 = declare(fixed, size=config.n_actions)
            tmp2 = declare(fixed, size=config.n_actions)

            mu = policy.get_variable("mu")
            sigma = policy.get_variable("sigma")
            counts = reward.var
            batch_r = Random(config.seed)

            if config.backend.init_macro is not None:
                config.backend.init_macro()

            with for_(n_u, 0, n_u < num_updates, n_u + 1):
                policy.load_input_values()

                # Load context
                load_circuit_context(circuit_params)

                n_reps_var = circuit_params.n_reps_var
                if n_reps_var is not None and n_reps_var.input_type is not None:
                    n_reps_var.load_input_value()

                circuit_params.max_input_state.load_input_value()
                with for_(i_idx, 0, i_idx < circuit_params.max_input_state.var, i_idx + 1):
                    circuit_params.input_state_vars.load_input_values()
                    circuit_params.max_observables.load_input_value()
                    circuit_params.n_shots.load_input_value()

                    with for_(o_idx, 0, o_idx < circuit_params.max_observables.var, o_idx + 1):
                        circuit_params.observable_vars.load_input_values()
                        batch_r.set_seed(config.seed + n_u)
                        with for_(b, 0, b < config.batch_size, b + 2):
                            # Sample from a multivariate Gaussian distribution (Muller-Box method)

                            tmp1, tmp2 = rand_gauss_moller_box(
                                mu,
                                sigma,
                                batch_r,
                                tmp1,
                                tmp2,
                            )
                            if (
                                config.backend_config.wrapper_data.get("rescale_and_clip", None)
                                is not None
                            ):
                                new_box = config.backend_config.wrapper_data["rescale_and_clip"]
                                tmp1, tmp2 = rescale_and_clip_wrapper(
                                    [tmp1, tmp2],
                                    config.action_space,
                                    new_box,
                                )
                            # Assign 1 or 2 depending on batch_size being even or odd (only relevant at last iteration)
                            with for_(j, 0, j < 2, j + 1):
                                # Assign the sampled actions to the action batch
                                for i, parameter in enumerate(
                                    circuit_params.real_time_circuit_parameters.parameters
                                ):
                                    parameter.assign(
                                        tmp1[i], condition=(j == 0), value_cond=tmp2[i]
                                    )

                                with for_(shots, 0, shots < circuit_params.n_shots.var, shots + 1):
                                    result = config.backend.quantum_circuit_to_qua(
                                        qc, circuit_params.circuit_variables
                                    )
                                    state_int = get_measurement_outcomes(qc, result)[qc.cregs[0].name]["state_int"]
                                    assign(counts[state_int], counts[state_int] + 1)

                                reward.stream_back(reset=True)

            with stream_processing():
                buffer = (config.batch_size, dim)
                reward.stream_processing(buffer=buffer)

        return rl_qoc_training_prog

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
        backend_info = env_config.backend_config
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
