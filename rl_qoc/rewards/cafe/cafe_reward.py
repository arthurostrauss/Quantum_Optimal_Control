from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal, TYPE_CHECKING
import numpy as np
from qiskit import ClassicalRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical.types import Uint
from qiskit.primitives import BaseSamplerV2, BitArray
from .cafe_reward_data import CAFERewardData, CAFERewardDataList
from ..real_time_utils import handle_real_time_n_reps
from ...environment.target import GateTarget
from ...environment.configuration.qconfig import QEnvConfig
from ..base_reward import Reward
from ...helpers import validate_circuit_and_target
from ...helpers.circuit_utils import (
    handle_n_reps,
    causal_cone_circuit,
    get_input_states_cardinality_per_qubit,
    get_single_qubit_input_states,
)
from qiskit_aer import AerSimulator

if TYPE_CHECKING:
    from ...qua.circuit_params import CircuitParams
    from qiskit_qm_provider.parameter_table import ParameterTable, Parameter as QuaParameter
    from qm import Program


@dataclass
class CAFEReward(Reward):
    """
    Configuration for computing the reward based on Context-Aware Fidelity Estimation (CAFE)
    """

    input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"
    input_states_seed: int = 2000
    input_states_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.input_states_rng = np.random.default_rng(self.input_states_seed)

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice}

    @property
    def reward_method(self):
        return "cafe"

    def set_reward_seed(self, seed: int):
        """
        Set the seed for the random number generator
        """
        self.input_states_seed = seed + 357
        self.input_states_rng = np.random.default_rng(self.input_states_seed)

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        env_config: QEnvConfig,
        baseline_circuit: Optional[QuantumCircuit] = None,
    ) -> CAFERewardDataList:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            env_config: QEnvConfig containing the backend information and execution configuration
            baseline_circuit: Ideal circuit that qc should implement
        """
        if not isinstance(target, GateTarget):
            raise ValueError("CAFE reward can only be computed for a target gate")
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info

        if baseline_circuit is not None:
            circuit_ref = baseline_circuit.copy()
        else:
            circuit_ref = qc.metadata["baseline_circuit"].copy()
        layout = target.layout
        num_qubits = target.causal_cone_size

        input_states_samples = np.unique(
            self.input_states_rng.choice(
                len(target.input_states(self.input_states_choice)),
                env_config.sampling_paulis,
                replace=True,
            ),
        )
        reward_data = []
        for sample in input_states_samples:
            # for input_state in target.input_states:
            input_state_indices = np.unravel_index(
                sample,
                (get_input_states_cardinality_per_qubit(self.input_states_choice),) * num_qubits,
            )
            input_state = target.input_states(self.input_states_choice)[sample]
            run_qc = qc.copy_empty_like(name="cafe_circ")  # Circuit with custom target gate
            ref_qc = circuit_ref.copy_empty_like(
                name="cafe_ref_circ"
            )  # Circuit with reference gate

            for circuit, context, control_flow in zip(
                [run_qc, ref_qc],
                [qc, circuit_ref],
                [execution_config.control_flow_enabled, False],
            ):
                # Bind input states to the circuits
                circuit.compose(
                    input_state.circuit,
                    qubits=target.causal_cone_qubits,
                    inplace=True,
                )
                circuit.barrier()
                cycle_circuit = handle_n_reps(
                    context,
                    execution_config.current_n_reps,
                    backend_info.backend,
                    control_flow=control_flow,
                )
                circuit.compose(cycle_circuit, inplace=True)

            # Compute inverse unitary for reference circuit
            sim_qc = causal_cone_circuit(ref_qc.decompose(), target.causal_cone_qubits)[0]
            sim_qc.save_unitary()
            backend = (
                backend_info.backend
                if isinstance(backend_info.backend, AerSimulator)
                else AerSimulator()
            )
            sim_unitary = (
                backend.run(sim_qc, noise_model=None, method="unitary").result().get_unitary()
            )

            reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
            reverse_unitary_qc.unitary(
                sim_unitary.adjoint(),  # Inverse unitary
                target.causal_cone_qubits,
                label="U_inv",
            )
            reverse_unitary_qc = backend_info.custom_transpile(
                reverse_unitary_qc,
                initial_layout=layout,
                scheduling=False,
                optimization_level=3,  # Find smallest circuit implementing inverse unitary
                remove_final_measurements=False,
            )

            # Bind inverse unitary + measurement to run circuit
            transpiled_circuit = backend_info.custom_transpile(
                run_qc, initial_layout=layout, scheduling=False
            )
            transpiled_circuit.barrier()
            # Add the inverse unitary and measurement to the circuit
            transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
            creg = ClassicalRegister("meas", target.causal_cone_size)
            transpiled_circuit.add_register(creg)
            transpiled_circuit.measure(target.causal_cone_qubits, creg)

            pub = (transpiled_circuit, params, execution_config.n_shots)
            reward_data.append(
                CAFERewardData(
                    pub,
                    input_state.circuit,
                    execution_config.current_n_reps,
                    input_state_indices,
                    (
                        causal_cone_circuit(reverse_unitary_qc, target.physical_qubits)[0]
                        if reverse_unitary_qc.num_qubits != target.causal_cone_size
                        else reverse_unitary_qc
                    ),
                    target.causal_cone_qubits_indices,
                )
            )
            # for circ, pubs_ in zip([run_qc, ref_qc], [pubs, ideal_pubs]):
            #     transpiled_circuit = backend_info.custom_transpile(
            #         circ, initial_layout=layout, scheduling=False
            #     )
            #     transpiled_circuit.barrier()
            #     # Add the inverse unitary + measurement to the circuit
            #     transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
            #     transpiled_circuit.measure_all()
            #     pubs_.append((transpiled_circuit, params, execution_config.n_shots))

        return CAFERewardDataList(reward_data)

    def get_reward_with_primitive(
        self,
        reward_data: CAFERewardDataList,
        primitive: BaseSamplerV2,
    ) -> np.array:
        """
        Compute the reward based on the input pubs
        """
        job = primitive.run(reward_data.pubs)
        causal_cone_qubits_indices = reward_data.causal_cone_qubits_indices
        causal_cone_size = reward_data.causal_cone_size
        pub_results = job.result()
        batch_size = reward_data.pubs[0].parameter_values.shape[0]
        n_shots = reward_data.pubs[0].shots
        assert all(
            [pub.shots == n_shots for pub in reward_data.pubs]
        ), "All pubs should have the same number of shots"
        assert all(
            [pub.parameter_values.shape[0] == batch_size for pub in reward_data.pubs]
        ), "All pubs should have the same batch size"
        num_bits = pub_results[0].data.meas[0].num_bits
        if num_bits == causal_cone_size:
            # No post-selection based on causal cone
            pub_data = [
                [pub_result.data.meas[i] for i in range(batch_size)] for pub_result in pub_results
            ]
            survival_probability = [
                [bit_array.get_int_counts().get(0, 0) / n_shots for bit_array in bit_arrays]
                for bit_arrays in pub_data
            ]
        else:
            # Post-select based on causal cone qubits
            pub_data = [
                [
                    pub_result.data.meas[i].postselect(
                        causal_cone_qubits_indices,
                        [0] * causal_cone_size,
                    )
                    for i in range(batch_size)
                ]
                for pub_result in pub_results
            ]
            survival_probability = [
                [bit_array.num_shots / n_shots for bit_array in bit_arrays]
                for bit_arrays in pub_data
            ]
        reward = np.mean(survival_probability, axis=0)
        return reward

    def get_real_time_circuit(
        self,
        circuits: QuantumCircuit | List[QuantumCircuit],
        target: GateTarget,
        env_config: QEnvConfig,
        skip_transpilation: bool = False,
        *args,
    ) -> QuantumCircuit:
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info
        prep_circuits = [circuits] if isinstance(circuits, QuantumCircuit) else circuits

        qubits = [qc.qubits for qc in prep_circuits]
        if not all(q == qubits[0] for q in qubits):
            raise ValueError("All circuits must have the same qubits")

        qc = prep_circuits[0].copy_empty_like(name="real_time_cafe_qc")
        qc.reset(qc.qubits)
        num_qubits = qc.num_qubits
        all_n_reps = execution_config.n_reps
        n_reps = execution_config.current_n_reps

        n_reps_var = qc.add_input("n_reps", Uint(8)) if len(all_n_reps) > 1 else n_reps
        if not qc.clbits:
            meas = ClassicalRegister(target.causal_cone_size, name="meas")
            qc.add_register(meas)
        else:
            meas = qc.cregs[0]
            if meas.size != target.causal_cone_size:
                raise ValueError("Classical register size must match the target causal cone size")

        input_state_vars = [qc.add_input(f"input_state_{i}", Uint(8)) for i in range(num_qubits)]

        input_choice = self.input_states_choice
        input_circuits = [
            circ.decompose() if input_choice in ["pauli4", "pauli6"] else circ
            for circ in get_single_qubit_input_states(input_choice)
        ]

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

        # Inversion step: compute the ideal reverse unitaries
        ref_circuits = [circ.metadata.get("baseline_circuit", None) for circ in prep_circuits]
        cycle_circuit_inverses = [[] for _ in range(len(ref_circuits))]
        input_state_inverses = [input_circuit.inverse() for input_circuit in input_circuits]
        for i, ref_circ in enumerate(ref_circuits):
            if ref_circ is None:
                raise ValueError("Baseline circuit not found in metadata")
            for n in all_n_reps:
                cycle_circuit, _ = causal_cone_circuit(
                    ref_circ.repeat(n).decompose(), target.causal_cone_qubits
                )
                cycle_circuit.save_unitary()
                sim_unitary = (
                    AerSimulator(method="unitary").run(cycle_circuit).result().get_unitary()
                )
                inverse_circuit = ref_circ.copy_empty_like(name="inverse_circuit")
                inverse_circuit.unitary(
                    sim_unitary.adjoint(),  # Inverse unitary
                    target.causal_cone_qubits,
                    label="U_inv",
                )
                inverse_circuit = backend_info.custom_transpile(
                    inverse_circuit,
                    initial_layout=target.layout,
                    scheduling=False,
                    optimization_level=3,  # Find smallest circuit implementing inverse unitary
                    remove_final_measurements=False,
                )
                inverse_circuit, _ = causal_cone_circuit(inverse_circuit, target.physical_qubits)
                cycle_circuit_inverses[i].append(inverse_circuit)

            # Add the inverse unitary that matches the combo of circuit choice and n_reps
            if len(prep_circuits) > 1:
                with qc.switch(circuit_choice) as case_circuit:
                    for i, inverse_circuit in enumerate(cycle_circuit_inverses):
                        with case_circuit(i):
                            if len(all_n_reps) > 1:
                                with qc.switch(n_reps_var) as case_n_reps:
                                    for j, n in enumerate(all_n_reps):
                                        with case_n_reps(n):
                                            if inverse_circuit[j].data:
                                                qc.compose(
                                                    inverse_circuit[j],
                                                    target.causal_cone_qubits,
                                                    inplace=True,
                                                )
                                            else:
                                                qc.delay(16, target.causal_cone_qubits)
                            else:
                                if inverse_circuit[0].data:
                                    qc.compose(
                                        inverse_circuit[0],
                                        target.causal_cone_qubits,
                                        inplace=True,
                                    )
                                else:
                                    qc.delay(16, target.causal_cone_qubits)
            else:
                if len(all_n_reps) > 1:
                    with qc.switch(n_reps_var) as case_n_reps:
                        for j, n in enumerate(all_n_reps):
                            with case_n_reps(n):
                                if cycle_circuit_inverses[0][j].data:
                                    qc.compose(
                                        cycle_circuit_inverses[0][j],
                                        target.causal_cone_qubits,
                                        inplace=True,
                                    )
                                else:
                                    qc.delay(16, target.causal_cone_qubits)
                else:
                    if cycle_circuit_inverses[0][0].data:
                        qc.compose(
                            cycle_circuit_inverses[0][0],
                            target.causal_cone_qubits,
                            inplace=True,
                        )
                    else:
                        qc.delay(16, target.causal_cone_qubits)

            # Revert the input state prep
            for q, qubit in enumerate(target.causal_cone_qubits):
                with qc.switch(input_state_vars[q]) as case_input_state:
                    for i, input_circuit in enumerate(input_state_inverses):
                        with case_input_state(i):
                            if input_circuit.data:
                                qc.compose(input_circuit, qubit, inplace=True)
                            else:
                                qc.delay(16, qubit)
        # Measure the causal cone qubits
        qc.measure(target.causal_cone_qubits, meas)

        if skip_transpilation:
            return qc

        return backend_info.custom_transpile(
            qc,
            optimization_level=1,
            initial_layout=target.layout,
            scheduling=False,
            remove_final_measurements=False,
        )

    def qm_step(
        self,
        reward_data: CAFERewardDataList,
        fetching_index: int,
        fetching_size: int,
        circuit_params: CircuitParams,
        reward: QuaParameter,
        config: QEnvConfig,
        **push_args,
    ):
        from ...qua.qm_config import QMConfig

        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")
        if not isinstance(config.target, GateTarget):
            raise ValueError("CAFE reward is only supported for GateTarget")
        reward_array = np.zeros(shape=(config.batch_size,))
        num_qubits = config.target.causal_cone_size
        dim = 2**num_qubits
        binary = lambda n, l: bin(n)[2:].zfill(l)
        input_indices = reward_data.input_indices
        max_input_state = len(input_indices)
        if circuit_params.circuit_choice_var is not None and isinstance(config.target, GateTarget):
            circuit_params.circuit_choice_var.push_to_opx(config.target.circuit_choice, **push_args)

        if circuit_params.n_reps_var is not None:
            circuit_params.n_reps_var.push_to_opx(config.current_n_reps, **push_args)

        circuit_params.max_input_state.push_to_opx(max_input_state, **push_args)
        for input_state in input_indices:
            input_state_dict = {
                input_var: input_state_val
                for input_var, input_state_val in zip(
                    circuit_params.input_state_vars.parameters, input_state
                )
            }
            circuit_params.input_state_vars.push_to_opx(input_state_dict, **push_args)

        collected_counts = reward.fetch_from_opx(
            push_args["job"],
            fetching_index=fetching_index,
            fetching_size=fetching_size,
            verbosity=config.backend_config.verbosity,
            time_out=config.backend_config.timeout,
        )

        shape = (max_input_state, config.batch_size, dim)
        transpose_shape = (1, 0, 2)
        counts = np.transpose(np.array(collected_counts).reshape(shape), transpose_shape)

        for batch_idx in range(config.batch_size):
            survival_probability = np.zeros(shape=(max_input_state,))
            for i_idx in range(max_input_state):
                counts_dict = {
                    binary(i, num_qubits): counts[batch_idx][i_idx][i] for i in range(dim)
                }
                bit_array = BitArray.from_counts(counts_dict, num_bits=num_qubits)
                survival_probability[i_idx] = bit_array.get_int_counts().get(0, 0) / config.n_shots
            reward_array[batch_idx] = np.mean(survival_probability)

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
            Util,
            stream_processing,
            assign,
            save,
            fixed,
        )
        from qiskit_qm_provider import QMBackend, Parameter as QuaParameter, ParameterTable
        from ...qua.qua_utils import rand_gauss_moller_box, get_state_int, rescale_and_clip_wrapper
        from ...qua.qm_config import QMConfig

        if not isinstance(config.backend, QMBackend):
            raise ValueError("Backend must be a QMBackend")
        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")
        if circuit_params.max_input_state is None:
            raise ValueError("max_input_state should be set for CAFE reward")
        if circuit_params.input_state_vars is None:
            raise ValueError("input_state_vars should be set for CAFE reward")
        if circuit_params.n_shots is None:
            raise ValueError("n_shots should be set for CAFE reward")
        if not isinstance(config.target, GateTarget):
            raise ValueError("CAFE reward is only supported for GateTarget")
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
            counts = reward.var

            n_u = declare(int)
            shots = declare(int)
            i_idx = declare(int)
            state_int = declare(int, value=0)
            batch_r = Random(config.seed)
            b = declare(int)
            j = declare(int)
            tmp1 = declare(fixed, size=config.n_actions)
            tmp2 = declare(fixed, size=config.n_actions)
            mu = policy.get_variable("mu")
            sigma = policy.get_variable("sigma")

            if config.backend.init_macro is not None:
                config.backend.init_macro()

            with for_(n_u, 0, n_u < num_updates, n_u + 1):
                policy.load_input_values()
                for var in [
                    circuit_params.circuit_choice_var,
                    circuit_params.n_reps_var,
                    circuit_params.context_parameters,
                ]:
                    if var is not None and var.input_type is not None:
                        if isinstance(var, QuaParameter):
                            var.load_input_value()
                        elif isinstance(var, ParameterTable):
                            var.load_input_values()

                circuit_params.max_input_state.load_input_value()

                with for_(i_idx, 0, i_idx < circuit_params.max_input_state.var, i_idx + 1):
                    circuit_params.input_state_vars.load_input_values()

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
                            tmp1 = rescale_and_clip_wrapper(
                                tmp1,
                                config.action_space,
                                new_box,
                            )
                            tmp2 = rescale_and_clip_wrapper(
                                tmp2,
                                config.action_space,
                                new_box,
                            )

                        # Assign 1 or 2 depending on batch_size being even or odd (only relevant at last iteration)
                        with for_(j, 0, j < 2, j + 1):
                            # Assign the sampled actions to the action batch
                            for i, parameter in enumerate(
                                circuit_params.real_time_circuit_parameters.parameters
                            ):
                                parameter.assign(tmp1[i], condition=(j == 0), value_cond=tmp2[i])

                            with for_(shots, 0, shots < circuit_params.n_shots.var, shots + 1):
                                result = config.backend.quantum_circuit_to_qua(
                                    qc, circuit_params.circuit_variables
                                )
                                state_int = get_state_int(qc, result, state_int)
                                assign(counts[state_int], counts[state_int] + 1)
                                assign(state_int, 0)  # Reset state_int for the next shot

                            reward.stream_back()
                            reward.assign([0.0] * dim)

            with stream_processing():
                buffer = (config.batch_size, dim)
                reward.stream_processing(buffer=buffer)
        return rl_qoc_training_prog
