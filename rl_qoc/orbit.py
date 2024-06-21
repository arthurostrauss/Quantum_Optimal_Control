from qiskit import QuantumCircuit, transpile
from qiskit_experiments.library.randomized_benchmarking import InterleavedRB
from qiskit.circuit import Gate, ParameterVector
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import random_clifford, Operator
from rl_qoc.quantumenvironment import QuantumEnvironment, GateTarget
from gymnasium.spaces import Box
import numpy as np
from typing import Callable
from scipy import optimize


def substitute_target_gate(
    circuit: QuantumCircuit,
    target_gate: Gate,
    parametrized_circuit_func: Callable,
    params: ParameterVector,
    **kwargs
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


def bounds_from_action_space(action_space: Box):
    """
    Convert the action space of the agent to bounds for the optimizer
    """
    return [(low, high) for low, high in zip(action_space.low, action_space.high)]


class ORBIT:
    def __init__(
        self,
        rb_length: int,
        num_sequences: int,
        q_env: QuantumEnvironment,
        use_interleaved: bool = False,
    ):
        self.rb_length = rb_length
        self.num_sequences = num_sequences
        self.seed = q_env.seed
        self.q_env = q_env
        assert isinstance(
            q_env.target, GateTarget
        ), "Target should be a GateTarget object"

        if use_interleaved:
            self.exp = InterleavedRB(
                q_env.target.gate,
                q_env.physical_target_qubits,
                lengths=[rb_length],
                seed=self.seed,
                num_samples=num_sequences,
                circuit_order="RRRIII",
                backend=q_env.backend,
            )
            self.ref_circuits = self.exp.circuits()[0:num_sequences]
            self.ref_interleaved_circuits = self.exp.circuits()[num_sequences:]
            self.run_circuits = [
                substitute_target_gate(
                    circuit,
                    q_env.target.gate,
                    q_env.parametrized_circuit_func,
                    q_env.parameters[0],
                    **q_env._func_args
                )
                for circuit in self.ref_interleaved_circuits
            ]
        else:
            self.run_circuits, self.ref_circuits = self.orbit_circuits()
        self.transpiled_circuits = [
            transpile(circuit, q_env.backend, optimization_level=0)
            for circuit in self.run_circuits
        ]
        self.fidelities = []

    def orbit_circuits(self):

        circuits, ref_circuits = [], []
        circuit = self.q_env.circuits[self.q_env.trunc_index]
        circuit_ref = self.q_env.baseline_circuits[self.q_env.trunc_index]

        for seq in range(self.num_sequences):
            run_qc = QuantumCircuit(*circuit.qregs)
            ref_qc = QuantumCircuit(*circuit_ref.qregs)
            for l in range(self.rb_length):
                r_cliff = random_clifford(self.q_env.n_qubits, self.q_env.seed)
                for qc, context in zip([run_qc, ref_qc], [circuit, circuit_ref]):
                    qc.compose(r_cliff.to_circuit(), inplace=True)
                    qc.barrier()
                    qc.compose(context, inplace=True)
                    qc.barrier()

            final_unitary = Operator(ref_qc)
            reverse_unitary = final_unitary.adjoint()
            for qc in [run_qc, ref_qc]:
                qc.append(reverse_unitary, qc.qregs[0])
                qc.measure_all()

            circuits.append(run_qc)
            ref_circuits.append(ref_qc)

        return circuits, ref_circuits

    def run_orbit_circuits(self, parameter_values):
        sampler = self.q_env.sampler
        if isinstance(sampler, BaseSamplerV2):
            pubs = []
            for qc in self.transpiled_circuits:
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
                self.transpiled_circuits,
                [parameter_values] * len(self.transpiled_circuits),
                shots=self.q_env.n_shots,
            ).result()
            counts = [
                results.quasi_dists[i] for i in range(len(self.transpiled_circuits))
            ]
            fidelity = np.mean([count[0] for count in counts])

        return fidelity

    def optimize(self, parameter_values):
        """
        Optimize the parameter values for the circuit to maximize the fidelity
        """

        def objective_function(params):
            return 1 - self.run_orbit_circuits(params)

        def callback(x):
            self.fidelities.append(1 - objective_function(x))

        result = optimize.minimize(
            objective_function,
            parameter_values,
            method="Nelder-Mead",
            bounds=bounds_from_action_space(self.q_env.action_space),
            callback=callback,
        )

        return result

    def optimize_CMA(self, initial_params):
        """
        Optimize the parameter values for the circuit to maximize the fidelity using CMA-ES optimizer.

        Parameters:
        initial_params (list or np.array): Initial guess for the parameters.
        """

        # Define the objective function
        import cma

        def objective_function(params):
            return 1 - self.run_orbit_circuits(params)

        # Run the optimizer
        es = cma.CMAEvolutionStrategy(
            initial_params,
            0.5,
            {
                "maxiter": 1000,
                "tolx": 1e-6,
                "bounds": bounds_from_action_space(self.q_env.action_space),
            },
        )
        es.optimize(objective_function)

        # Get the best parameters
        optimal_params = es.result.xbest
        minimized_value = es.result.fbest

        return optimal_params, minimized_value
