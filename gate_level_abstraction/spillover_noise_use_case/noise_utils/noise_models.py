import numpy as np
from scipy.linalg import sqrtm

from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
from qiskit_aer.noise.errors import QuantumError
from qiskit.quantum_info.operators.channel import Kraus


def generate_random_cptp_map(dim: int, num_ops: int, epsilon: float):
    """
    Generate a random CPTP map represented by Kraus operators, with a specified noise strength.

    Parameters:
    - dim: The dimension of the Hilbert space.
    - num_ops: The number of Kraus operators to generate.
    - epsilon: A parameter controlling the strength of the noise (0 <= epsilon <= 1).

    Returns:
    A list of Kraus operators that define a CPTP map.
    """
    # Step 1: Generate random matrices
    kraus_ops_tilde = [
        np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        for _ in range(num_ops - 1)
    ]

    # Step 2: Impose the completeness condition using the generated matrices
    S = sum([E_tilde.conj().T @ E_tilde for E_tilde in kraus_ops_tilde])
    E_0 = sqrtm(np.eye(dim) - epsilon * S)

    # Step 3: Scale the non-primary Kraus operators by sqrt(epsilon) to introduce noise strength
    scaled_kraus_ops = [np.sqrt(epsilon) * K for K in kraus_ops_tilde]

    # Combine the primary Kraus operator with the scaled ones
    kraus_operators = [E_0] + scaled_kraus_ops

    assert np.allclose(
        sum([K.conj().T @ K for K in kraus_operators]), np.eye(dim), atol=1e-6
    ) and np.all(
        np.linalg.eigvals(kraus_operators[0]) >= 0
    ), """
        The Kraus operators do not form a valid CPTP map.
        This can sometimes happen due to the randomized noise. Trying again can resolve this issue.
        If it persists, consider adjusting the noise strength parameter epsilon.
        """

    return Kraus(kraus_operators)


def gate_overrotation_error(gate: Gate, rotation_rad: float, error_prob: float):
    """
    Compute the quantum error representing the overrotation of an arbitrary single-qubit gate.

    This function takes a Qiskit gate object, an overrotation angle in radians, and an error probability.
    It returns a QuantumError object that models the gate being applied correctly with probability (1 - error_prob)
    and being overrotated by `rotation_rad` radians with probability `error_prob`.

    Parameters:
    - gate (Gate): A single-qubit Qiskit gate object. The gate for which the overrotation error is to be calculated.
    - rotation_rad (float): The overrotation angle in radians. This angle is used to calculate the overrotated version of the gate.
    - error_prob (float): The probability of the gate being overrotated. This is used in the construction of the QuantumError object,
      with the original gate being applied with probability (1 - error_prob) and the overrotated gate with probability `error_prob`.

    Returns:
    - QuantumError: A quantum error object representing the original gate applied with probability (1 - error_prob) and the overrotated
      version of the gate applied with probability `error_prob`.
    """

    # Convert the Qiskit gate to its matrix representation
    gate_matrix = gate.to_matrix()

    # Define the overrotation angle in radians
    # Calculate the overrotated gate's matrix
    rotation_matrix = np.array(
        [
            [np.cos(rotation_rad / 2), -1j * np.sin(rotation_rad / 2)],
            [-1j * np.sin(rotation_rad / 2), np.cos(rotation_rad / 2)],
        ]
    )

    overrotated_gate_matrix = rotation_matrix @ gate_matrix

    # Create a quantum error for the overrotated gate
    gate_overrotation_error = QuantumError(
        [
            (Operator(overrotated_gate_matrix), error_prob),
            (Operator(gate_matrix), 1 - error_prob),
        ]
    )
    return gate_overrotation_error
