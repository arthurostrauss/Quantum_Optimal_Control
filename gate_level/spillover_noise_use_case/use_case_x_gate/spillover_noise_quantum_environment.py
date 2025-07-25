import numpy as np
import math
import sys
from typing import List, Tuple

# Qiskit imports
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RXGate, IGate
from qiskit.quantum_info.operators.measures import average_gate_fidelity, state_fidelity
from qiskit_aer import noise, AerSimulator

from rl_qoc import QEnvConfig, ContextAwareQuantumEnvironment

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


class SpilloverNoiseQuantumEnvironment(ContextAwareQuantumEnvironment):
    """
    Quantum environment for the spillover noise use case.

    This class is a subclass of ContextAwareQuantumEnvironmentV2 and provides one additional method:
        - get_baseline_fid_from_phi_gamma: Calculate the average gate fidelity of a noisy circuit with respect to an ideal circuit.
        Based on this baseline fidelity, the number of repetitions required to reach a target fidelity will be calculated and set as the number of repetitions for the environment.
    """

    def __init__(
        self,
        training_config: QEnvConfig,
        circuit_context: QuantumCircuit,
        phi_gamma_tuple: Tuple[float, float],
        training_steps_per_gate: List[int] | int = 1500,
        intermediate_rewards: bool = False,
    ):
        super().__init__(
            training_config,
            circuit_context,
            training_steps_per_gate,
            intermediate_rewards,
        )
        self.phi_gamma_tuple = phi_gamma_tuple

    def set_env_params(self, **kwargs):
        """
        Set environment parameters according to the provided kwargs
        """
        # TODO: Let user pass a list of target fidelities and return and set n_reps to the miniumum number required to push the baseline fidelity below the smallest target fidelity
        if "target_fidelities" in kwargs:
            self.n_reps = self.get_n_reps(kwargs["target_fidelities"])
            logging.warning(
                f"n_reps set to {self.n_reps} to start calibration below target fidelities {kwargs['target_fidelities']}"
            )

    def get_n_reps(self, target_fidelities):
        baseline_fidelity = self.get_baseline_fid_from_phi_gamma()
        # To which integer power do I need to raise baseline_fidelity to be below the lowest target fidelity?
        smallest_N_reps = math.ceil(math.log(min(target_fidelities), baseline_fidelity))
        if smallest_N_reps == 150:
            logging.warning(
                "WARNING: n_reps set to max value of 150. Consider increasing noise strength."
            )
        return min([smallest_N_reps, 150])

    def get_baseline_fid_from_phi_gamma(self):
        ideal_circ = self._get_ideal_circ()
        noisy_circ = self._build_noisy_circ()
        backend = self._bind_noise_get_backend()
        process_results = backend.run(noisy_circ).result()
        q_process_list = [
            process_results.data(0)["superop"],
            # process_results.data(0)['density_matrix'],
        ]
        avg_fidelity = np.mean(
            [
                average_gate_fidelity(q_process, Operator(ideal_circ))
                for q_process in q_process_list
                # state_fidelity(q_process, Statevector(ideal_circ))
                # for q_process in q_process_list
            ]
        )
        return avg_fidelity

    def get_unitary_from_phi_gamma(self):
        noisy_circ = self._build_noisy_circ()
        backend = self._bind_noise_get_backend()
        process_results = backend.run(noisy_circ).result()
        q_process_list = [
            process_results.data(0)["superop"],
            # process_results.data(0)['density_matrix'],
        ]
        return q_process_list[0]

    def _get_noisy_circuit(self):
        circuit = QuantumCircuit(1)

        rx_phi_op = Operator(RXGate(self.phi))
        circuit.unitary(rx_phi_op, [0], label="RX(phi)")

        rx_phi_gamma_op = Operator(RXGate(self.gamma * self.phi))
        circuit.unitary(rx_phi_gamma_op, [0], label="RX(gamma*phi)")

        # Model custom CX gate
        optimal_params_noise_free = [np.pi]
        circuit.rx(optimal_params_noise_free[0], 0)

        return circuit

    def _bind_noise_get_backend(self):

        custom_rx_gate_label = "custom_rx"
        rx_phi_gamma_op = Operator(RXGate(self.gamma * self.phi))

        noise_model = noise.NoiseModel(["unitary", "u", "h", "x", "s", "z"])
        coherent_rx_noise = noise.coherent_unitary_error(rx_phi_gamma_op)
        noise_model.add_quantum_error(coherent_rx_noise, [custom_rx_gate_label], [0])

        backend = AerSimulator(noise_model=noise_model)
        return backend

    def _build_noisy_circ(self):
        custom_rx_gate_label = "custom_rx"
        circuit = QuantumCircuit(1)
        subcircuit = QuantumCircuit(1)
        subcircuit.rx(self.phi, 0)
        circuit.unitary(Operator(subcircuit), [0], label=custom_rx_gate_label)

        # Model custom X gate
        optimal_params = [np.pi]
        circuit.rx(optimal_params[0], 0)
        circuit.save_superop()
        # circuit.save_density_matrix()
        return circuit

    def _get_ideal_circ(self):
        ideal_circ = QuantumCircuit(1, name=f"custom_x")
        ideal_circ.rx(self.phi, 0)
        ideal_circ.x(
            0,
        )
        return ideal_circ

    def __str__(self):
        """This is a one-line description of the environment with some key parameters."""
        base_ident_str = super().__str__()
        return f"SpilloverNoise_phi-{self.phi / np.pi}pi_gamma-{self.gamma}_{base_ident_str}"

    def __repr__(self):
        string = ContextAwareQuantumEnvironment.__repr__(self)
        string += f"Custom Spillover Noise Use Case with noisy RX(phi={self.phi / np.pi}pi) rotation on qubit 0\n"
        string += f"Spillover Noise leads to the unwanted noise effect on qubit 1 in form of RX({self.gamma} * {self.phi / np.pi}pi) with gamma as the noise strength parameter.\n"
        return string

    @property
    def phi(self):
        return self.phi_gamma_tuple[0]

    @property
    def gamma(self):
        return self.phi_gamma_tuple[1]
