from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np
from qiskit import QuantumCircuit
from ...environment.target import Target, StateTarget, GateTarget
from ...environment.configuration.qconfig import QEnvConfig
from ...rewards.base_reward import Reward
from .pi_pulse_reward_data import PiPulseRewardData, PiPulseRewardDataList
from qiskit.primitives import BaseSamplerV2, BitArray

if TYPE_CHECKING:
    from qiskit_qm_provider.parameter_table import ParameterTable, Parameter as QuaParameter
    from rl_qoc.qua.circuit_params import CircuitParams
    from qm import Program


class PiPulseReward(Reward):
    def __init__(self):
        super().__init__()

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.ndarray,
        target: Target,
        env_config: QEnvConfig,
        *args,
        **kwargs,
    ) -> PiPulseRewardDataList:
        return PiPulseRewardDataList([PiPulseRewardData((qc, params, env_config.n_shots))])

    def get_reward_with_primitive(
        self, reward_data: PiPulseRewardDataList, primitive: BaseSamplerV2, *args, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError("This reward method does not support primitive sampling")

    def get_real_time_circuit(
        self,
        circuits: QuantumCircuit | List[QuantumCircuit],
        target: StateTarget | GateTarget | List[StateTarget | GateTarget],
        env_config: QEnvConfig,
        skip_transpilation: bool = False,
        *args,
    ) -> QuantumCircuit:
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        if len(circuits) > 1:
            raise ValueError("PiPulseReward only supports single-circuit circuits")
        circuit = circuits[0].copy()

        # Prepend reset to circuit
        reset_circuit = QuantumCircuit(circuit.num_qubits)
        reset_circuit.reset(range(circuit.num_qubits))
        circuit.compose(reset_circuit, inplace=True, front=True)

        # Add measurement
        circuit.measure_all(inplace=True)

        return circuit

    @property
    def reward_method(self) -> str:
        return "pi_pulse"

    def qm_step(
        self,
        reward_data: PiPulseRewardDataList,
        fetching_index: int,
        fetching_size: int,
        circuit_params: CircuitParams,
        reward: QuaParameter,
        config: QEnvConfig,
        **push_args,
    ):
        """
        This function is used to compute the reward for the pi pulse reward.
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

        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")

        reward_array = np.zeros(shape=(config.batch_size,))
        num_qubits = config.target.n_qubits
        dim = 2**num_qubits
        binary = lambda n, l: bin(n)[2:].zfill(l)

        collected_counts = reward.fetch_from_opx(
            push_args["job"],
            fetching_index=fetching_index,
            fetching_size=fetching_size,
            verbosity=config.backend_config.verbosity,
            time_out=config.backend_config.timeout,
        )[0]
        for b in range(config.batch_size):
            counts_dict = {binary(i, num_qubits): collected_counts[b][i] for i in range(dim)}
            bit_array = BitArray.from_counts(counts_dict, num_bits=num_qubits)
            reward_array[b] = bit_array.get_int_counts().get(1, 0) / config.n_shots

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
            save,
            fixed,
        )
        from qiskit_qm_provider import QMBackend
        from ..qua_utils import rand_gauss_moller_box, rescale_and_clip_wrapper
        from qiskit_qm_provider.backend import get_measurement_outcomes
        from ..qm_config import QMConfig

        if not isinstance(config.backend, QMBackend):
            raise ValueError("Backend must be a QMBackend")
        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")

        policy.reset()
        reward.reset()
        circuit_params.reset()

        dim = int(2**config.target.n_qubits)
        if config.backend_config.wrapper_data.get("rescale_and_clip", None) is not None:
            new_box = config.backend_config.wrapper_data["rescale_and_clip"]

        with program() as rl_qoc_training_prog:
            # Declare the necessary variables (all are counters variables to loop over the corresponding hyperparameters)
            circuit_params.declare_variables()
            policy.declare_variables()
            reward.declare_variable()
            reward.declare_stream()

            n_u = declare(int)
            batch_r = Random(config.seed)
            shots = declare(int)
            j = declare(int)
            b = declare(int)
            tmp1 = declare(fixed, size=config.n_actions)
            tmp2 = declare(fixed, size=config.n_actions)

            mu = policy.get_variable("mu")
            sigma = policy.get_variable("sigma")
            counts = reward.var

            if config.backend.init_macro is not None:
                config.backend.init_macro()

            with for_(n_u, 0, n_u < num_updates, n_u + 1):
                policy.load_input_values()
                if test:
                    policy.save_to_stream()
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
                    if config.backend_config.wrapper_data.get("rescale_and_clip", None) is not None:
                        tmp1, tmp2 = rescale_and_clip_wrapper(
                            [tmp1, tmp2],
                            config.action_space,
                            new_box,
                        )
                    # Assign the sampled actions to the action batch
                    with for_(j, 0, j < 2, j + 1):
                        for i, parameter in enumerate(
                            circuit_params.real_time_circuit_parameters.parameters
                        ):

                            parameter.assign(tmp1[i], condition=(j == 0), value_cond=tmp2[i])

                        with for_(shots, 0, shots < circuit_params.n_shots.var, shots + 1):
                            result = config.backend.quantum_circuit_to_qua(
                                qc, circuit_params.circuit_variables
                            )
                            state_int = get_measurement_outcomes(qc, result)[qc.cregs[0].name]["state_int"]
                            if test:
                                save(state_int, "state_int")
                            assign(counts[state_int], counts[state_int] + 1)

                        reward.stream_back(reset=True)

            with stream_processing():
                buffer = (config.batch_size, dim)
                reward.stream_processing(buffer=buffer)
                if test:
                    circuit_params.stream_processing()
                    policy.stream_processing()

        return rl_qoc_training_prog
