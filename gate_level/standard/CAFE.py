from __future__ import annotations
from typing import Callable, Optional
import os
import sys
import numpy as np
from itertools import product
from gymnasium.spaces import Box
import cma

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit import QuantumCircuit, Gate, ParameterVector
from qiskit.primitives import BaseSamplerV2
from qiskit_experiments.library.tomography.basis import PauliPreparationBasis
from qiskit.quantum_info import Operator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
module_path = os.path.abspath(
    os.path.join(
        "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control"
    )
)
if module_path not in sys.path:
    sys.path.append(module_path)

from rl_qoc import BaseQuantumEnvironment


def bounds_from_action_space(action_space: Box):
    """
    Convert the action space of the agent to bounds for the optimizer
    """
    return [(low, high) for low, high in zip(action_space.low, action_space.high)]


def substitute_target_gate(
    circuit: QuantumCircuit,
    target_gate: Gate,
    parametrized_circuit_func: Callable,
    params: ParameterVector,
    **kwargs,
):
    """
    Substitute a target gate in a circuit with a parametrized version of the gate.
    """
    qc = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for instruction in circuit.data:
        if instruction.operation != target_gate:
            qc.append(instruction)
        else:
            parametrized_circuit_func(qc, params, qc.qregs[0], **kwargs)
    return qc


class CAFE:
    """
    Class to implement a modified version of CAFE (Context-Aware Fidelity Estimation) as introduced in https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043202.

    Attributes:
        q_env (QuantumEnvironment): The quantum environment containing necessary details like circuits, backend, etc.
        n_reps (int): Number of repetitions of the custom target gate in the circuit. Default is 3.
        circuits_run (list): List of quantum circuits to be run.
        circuits_ref (list): List of reference quantum circuits.
        fidelities (list): List to store fidelities of the circuits.

    Methods:
        __init__(q_env: QuantumEnvironment, n_reps: int = 3): Initializes the CAFE class with the provided quantum environment and repetitions.
        get_cafe_circuits(): Generates and returns the circuits to be run and the reference circuits.
        run_cafe_circuits(parameter_values): Runs the generated circuits with the provided parameter values and calculates the fidelity.
    """

    def __init__(self, q_env: BaseQuantumEnvironment, n_reps: int = 3):
        """
        Initializes the CAFE class with the provided quantum environment and repetitions.

        Parameters:
        q_env (QuantumEnvironment): The quantum environment containing necessary details like circuits, backend, etc.
        n_reps (int): Number of repetitions of the custom target gate in the circuit. Default is 3.
        """
        self.q_env = q_env
        self.n_reps: int = n_reps
        self.circuits_run, self.circuits_ref = self.get_cafe_circuits()
        self.fidelities = []

    def get_cafe_circuits(self):
        """
        Generates and returns the circuits to be run and the reference circuits.

        Returns:
            tuple: A tuple containing two lists:
                - circuits_run: List of quantum circuits to be run with the custom target gate.
                - circuits_ref: List of reference quantum circuits with the ideal gate.
        """

        circuits_run, circuits_ref = [], []
        circuit_run = self.q_env.circuits[0]
        circuit_ref = self.q_env.baseline_circuits[0]
        n_qubits = self.q_env.n_qubits

        input_circuits = [
            PauliPreparationBasis().circuit(s)
            for s in product(range(4), repeat=n_qubits)
        ]

        for input_circ in input_circuits:
            run_qc = QuantumCircuit(n_qubits)  # Circuit with the custom target gate
            ref_qc = QuantumCircuit(
                n_qubits
            )  # Circuit with the ideal gate for reference

            # Bind input states to the circuits
            for qc in [run_qc, ref_qc]:
                qc.compose(input_circ, inplace=True)
                qc.barrier()

            # Add the custom target gate to the run circuit n_reps times
            for qc, context in zip([run_qc, ref_qc], [circuit_run, circuit_ref]):
                for _ in range(1, self.n_reps + 1):
                    qc.compose(context, inplace=True)
                qc.barrier()
            run_qc = substitute_target_gate(
                ref_qc,
                self.q_env.target.gate,
                self.q_env.parametrized_circuit_func,
                self.q_env.parameters[0],
                **self.q_env._func_args,
            )

            reverse_unitary = Operator(ref_qc).adjoint().to_instruction()
            reverse_unitary_qc = QuantumCircuit(n_qubits)
            reverse_unitary_qc.unitary(
                reverse_unitary,
                reverse_unitary_qc.qubits,
                label="reverse circuit unitary",
            )
            reverse_unitary = transpile(
                reverse_unitary_qc, self.q_env.backend, optimization_level=3
            )  # Try to get the smallest possible circuit for the reverse unitary

            for qc, context in zip([run_qc, ref_qc], [circuit_run, circuit_ref]):
                qc = transpile(qc, self.q_env.backend, optimization_level=0)

            # Add the inverse unitary to the circuits
            for qc in [run_qc, ref_qc]:
                qc.compose(reverse_unitary_qc, inplace=True)
                # qc.unitary(reverse_unitary, qc.qubits, label="reverse circuit unitary")
                qc.measure_all()

            circuits_run.append(run_qc)
            circuits_ref.append(ref_qc)

        return circuits_run, circuits_ref

    def run_cafe_circuits(self, parameter_values):
        sampler = self.q_env.sampler
        if isinstance(sampler, BaseSamplerV2):
            pubs = []
            for qc in self.circuits_run:
                pubs.append((qc, parameter_values))
            results = sampler.run(pubs, shots=self.q_env.n_shots).result()
            counts = [result.data.meas.get_counts() for result in results]
            for count in counts:
                for key in [
                    bin(i)[2:].zfill(self.q_env.n_qubits)
                    for i in range(2**self.q_env.n_qubits)
                ]:
                    if key not in count.keys():
                        count[key] = 0
            fidelity = np.mean(
                [
                    count["0" * self.q_env.n_qubits] / self.q_env.n_shots
                    for count in counts
                ]
            )

        else:
            results = sampler.run(
                self.circuits_run,
                [parameter_values] * len(self.circuits_run),
                shots=self.q_env.n_shots,
            ).result()
            counts = [results.quasi_dists[i] for i in range(len(self.circuits_run))]

            # Count the number of dictionaries where 0 is a key
            # No matter the number of qubits, the zero state will always be "0" since the keys represent the sum over all qubits (so, so state 011 would be mapped to sum(011)=2; 000 to 0; etc.)
            zero_state_key = int(0)
            values_for_key_0 = [
                count[zero_state_key] for count in counts if zero_state_key in count
            ]
            # Calculate the mean value for key 0
            fidelity = sum(values_for_key_0) / len(counts) if values_for_key_0 else 0.0

            return fidelity

    def optimize_CMA(self, initial_params):
        """
        Optimize the parameter values for the circuit to maximize the fidelity using CMA-ES optimizer.

        Parameters:
        initial_params (list or np.array): Initial guess for the parameters.
        """

        # Define the objective function
        def objective_function(params):
            fidelity = self.run_cafe_circuits(params)
            self.fidelities.append(fidelity)
            return 1.0 - fidelity

        # Run the optimizer
        es = cma.CMAEvolutionStrategy(
            initial_params,
            0.5,
            {
                "maxiter": 1000,
                "tolx": 1e-6,
                "bounds": [-np.pi, np.pi],
            },
        )
        es.optimize(objective_function)

        # Get the best parameters
        optimal_params = es.result.xbest
        minimized_value = es.result.fbest

        return optimal_params, minimized_value
