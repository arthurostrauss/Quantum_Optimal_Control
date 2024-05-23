"""
Helper functions for the pulse spillover noise use case

Created: 23/05/2024
"""


import numpy as np
import sympy as sp
import sympy as sp
from sympy.physics.quantum import TensorProduct
from sympy import Matrix, eye, simplify
from qiskit.quantum_info import DensityMatrix, Statevector, Operator


def validate_target_state(
        input_state_index: int, phi_val: float, n_reps: int = 1
    ):
        # TODO: Extend function to ECR (currently only CX supported)

        # Define the symbols for the angles
        phi, gamma = sp.symbols("phi gamma")
        a = sp.symbols("a:7")  # a[0] to a[6]

        I = eye(2)
        H = 1 / np.sqrt(2) * Matrix([[1, 1], [1, -1]])
        S = Matrix([[1, 0], [0, 1j]])
        X = Matrix([[0, 1], [1, 0]])

        def get_input_rotation_gates(index: int):
            """
            Will return the input rotation gates for the given index (0-15) for the input states that form a tomographically complete set
            """
            input_rotation_circuits = {
                0: TensorProduct(I, I),
                1: TensorProduct(X, I),
                2: TensorProduct(H, I),
                3: TensorProduct(S * H, I),
                4: TensorProduct(I, X),
                5: TensorProduct(X, X),
                6: TensorProduct(H, X),
                7: TensorProduct(S * H, X),
                8: TensorProduct(I, H),
                9: TensorProduct(X, H),
                10: TensorProduct(H, H),
                11: TensorProduct(S * H, H),
                12: TensorProduct(I, S * H),
                13: TensorProduct(X, S * H),
                14: TensorProduct(H, S * H),
                15: TensorProduct(S * H, S * H),
            }
            return input_rotation_circuits[index]

        # Define RX gate in symbolic form
        def RX(theta):
            return sp.Matrix(
                [
                    [sp.cos(theta / 2), -1j * sp.sin(theta / 2)],
                    [-1j * sp.sin(theta / 2), sp.cos(theta / 2)],
                ]
            )

        # Define U gate in symbolic form
        def U_gate(theta, phi, lambd):
            return sp.Matrix(
                [
                    [sp.cos(theta / 2), -sp.exp(1j * lambd) * sp.sin(theta / 2)],
                    [
                        sp.exp(1j * phi) * sp.sin(theta / 2),
                        sp.exp(1j * lambd + 1j * phi) * sp.cos(theta / 2),
                    ],
                ]
            )

        # Define RZX gate in symbolic form
        def RZX(theta):
            cos = sp.cos(theta / 2)
            sin = sp.sin(theta / 2)
            i = 1j
            return sp.Matrix(
                [
                    [cos, 0, -i * sin, 0],
                    [0, cos, 0, i * sin],
                    [-i * sin, 0, cos, 0],
                    [0, i * sin, 0, cos],
                ]
            )

        # Create the circuit matrix step by step
        # RX gates
        RX_q0 = TensorProduct(
            sp.eye(2), RX(phi)
        )  # RX(phi) with phi=0 applied to qubit 0
        RX_q1 = TensorProduct(
            RX(phi * gamma), sp.eye(2)
        )  # RX(gamma*phi) with gamma*phi=0 applied to qubit 1

        # U gates
        U_q0 = TensorProduct(
            sp.eye(2), U_gate(a[0], a[1], a[2])
        )  # U gate applied to qubit 0
        U_q1 = TensorProduct(
            U_gate(a[3], a[4], a[5]), sp.eye(2)
        )  # U gate applied to qubit 1

        # RZX gate
        RZX_q12 = RZX(a[6])  # RZX gate applied between qubits 0 and 1

        def get_complete_circuit(
            parametrized_cnot_circuit, input_state_index: int, n_reps: int = 1
        ):
            """Returns the full parametrized CNOT-gate circuit for the given input state index"""
            input_rotation_gates = get_input_rotation_gates(input_state_index)
            full_circuit = parametrized_cnot_circuit**n_reps * input_rotation_gates

            return simplify(full_circuit)

        gate_parameters = {
            phi: phi_val,
            gamma: 0.0,
            a[0]: 0,
            a[1]: 0,
            a[2]: sp.pi / 2,
            a[3]: sp.pi / 2,
            a[4]: -sp.pi / 2,
            a[5]: sp.pi / 2,
            a[6]: -sp.pi / 2,
        }

        parametrized_cnot_circuit = RZX_q12 * U_q1 * U_q0 * RX_q1 * RX_q0
        circuit_evaluated = parametrized_cnot_circuit.subs(gate_parameters).evalf(
            chop=True
        )
        CompleteCircuit_evaluated = get_complete_circuit(
            circuit_evaluated, input_state_index=input_state_index, n_reps=n_reps
        )

        CompleteCircuit_evaluated = sp.nsimplify(
            CompleteCircuit_evaluated, tolerance=1e-10
        )
        CompleteCircuit_evaluated_float = CompleteCircuit_evaluated.evalf(chop=True)
        circuit_operator = Operator(
            np.array(CompleteCircuit_evaluated_float).astype(np.complex128)
        )

        output_state = Statevector.from_label("00").evolve(circuit_operator)

        return DensityMatrix(output_state)