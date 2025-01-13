from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp, pauli_basis
from qiskit_experiments.library.tomography.basis import Pauli6PreparationBasis
from .qconfig import ExecutionConfig
from .target import GateTarget, StateTarget
from .backend_info import BackendInfo


from typing import Optional, Tuple
import numpy as np


def extend_observables(observables: SparsePauliOp, qc: QuantumCircuit,
                        target: GateTarget) -> SparsePauliOp:
    """
    Extend the observables to all qubits in the quantum circuit if necessary

    Args:
        observables: Pauli observables to sample
        qc: Quantum circuit to be executed on quantum system
        target: Target gate to prepare (possibly within a wider circuit context)

    Returns:
        Extended Pauli observables
    """

    if qc.num_qubits > target.causal_cone_size:
        other_qubits_indices = set(range(qc.num_qubits)) - set(
            target.causal_cone_qubits_indices
        )
        observables = observables.apply_layout(
            None, qc.num_qubits
        ).apply_layout(target.causal_cone_qubits_indices + list(other_qubits_indices))

    return observables


def extend_input_state_prep(input_circuit, qc, target: GateTarget):
    if qc.num_qubits > target.causal_cone_size:  # Add random input state on all qubits (not part of reward calculation)
        other_qubits_indices = set(range(qc.num_qubits)) - set(
            target.causal_cone_qubits_indices
        )
        other_qubits = [qc.qubits[i] for i in other_qubits_indices]
        random_input_context = Pauli6PreparationBasis().circuit(
            np.random.randint(0, 6, len(other_qubits)).tolist()
        )
        return input_circuit.compose(
            random_input_context, other_qubits, front=True
        )
    return input_circuit


def handle_n_reps(qc: QuantumCircuit, n_reps: int = 1, backend=None):
    # Repeat the circuit n_reps times and prepend the input state preparation
    if isinstance(backend, BackendV2) and "for_loop" in backend.operation_names:
        prep_circuit = qc.copy_empty_like()

        with prep_circuit.for_loop(range(n_reps)) as i:
            prep_circuit.compose(qc, inplace=True)
    else:
        prep_circuit = qc.repeat(n_reps)
    return prep_circuit


def retrieve_observables(
        target_state: StateTarget,
        dfe_tuple: Optional[Tuple[float, float]] = None,
        sampling_paulis: int = 100,
        c_factor: float = 1.0,
):
    """
    Retrieve observables to sample for the DFE protocol (PhysRevLett.106.230501) for given target state

    :param target_state: Target state to prepare
    :param dfe_tuple: Optional Tuple (Ɛ, δ) from DFE paper
    :param sampling_paulis: Number of Pauli observables to sample
    :param c_factor: Constant factor for reward calculation
    :return: Observables to sample, number of shots for each observable
    """
    # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
    probabilities = target_state.Chi ** 2
    full_basis = pauli_basis(target_state.n_qubits)
    if not np.isclose(np.sum(probabilities), 1, atol=1e-5):
        print("probabilities sum um to", np.sum(probabilities))
        print("probabilities normalized")
        probabilities = probabilities / np.sum(probabilities)

    sample_size = (
        sampling_paulis
        if dfe_tuple is None
        else int(np.ceil(1 / (dfe_tuple[0] ** 2 * dfe_tuple[1])))
    )
    k_samples = np.random.choice(
        len(probabilities), size=sample_size, p=probabilities
    )

    pauli_indices, pauli_shots = np.unique(k_samples, return_counts=True)
    reward_factor = c_factor / (
            np.sqrt(target_state.dm.dim) * target_state.Chi[pauli_indices]
    )

    if dfe_tuple is not None:
        pauli_shots = np.ceil(
            2
            * np.log(2 / dfe_tuple[1])
            / (
                    target_state.dm.dim
                    * sample_size
                    * dfe_tuple[0] ** 2
                    * target_state.Chi[pauli_indices] ** 2
            )
        )
    # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
    observables = SparsePauliOp(
        full_basis[pauli_indices], reward_factor, copy=False
    )

    shots_per_basis = []
    # Group observables by qubit-wise commuting groups to reduce the number of PUBs
    for i, commuting_group in enumerate(
            observables.paulis.group_qubit_wise_commuting()
    ):
        max_pauli_shots = 0
        for pauli in commuting_group:
            pauli_index = list(full_basis).index(pauli)
            ref_index = list(pauli_indices).index(pauli_index)
            max_pauli_shots = max(max_pauli_shots, pauli_shots[ref_index])
        shots_per_basis.append(max_pauli_shots)

    return observables, shots_per_basis