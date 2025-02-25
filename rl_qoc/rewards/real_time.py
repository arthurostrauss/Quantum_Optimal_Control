from typing import List, Optional, Tuple, Literal
from ..environment.target import GateTarget, StateTarget
from ..environment.configuration.execution_config import ExecutionConfig
from ..environment.backend_info import BackendInfo
from ..helpers.circuit_utils import get_single_qubit_input_states, causal_cone_circuit
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.classical.types import Uint
from qiskit.circuit.classical.expr import Var
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis
from qiskit_aer import AerSimulator


def get_real_time_reward_circuit(
    circuits: QuantumCircuit | List[QuantumCircuit],
    target: List[GateTarget] | GateTarget | StateTarget,
    backend_info: BackendInfo,
    execution_config: ExecutionConfig,
    reward_method: Literal["channel", "state", "cafe"] = "state",
    dfe_precision: Optional[Tuple[float, float]] = None,
) -> QuantumCircuit:
    """
    Compute the quantum circuit for real-time reward computation

    Args:
        circuits: Quantum circuit to be executed on quantum system
        params: Parameters to feed the parametrized circuit
        target: List of target gates
        backend_info: Backend information
        execution_config: Execution configuration
        reward_method: Method to compute the reward (channel, state or cafe)
        dfe_precision: Tuple (Ɛ, δ) from DFE paper
    """

    prep_circuits = [circuits] if isinstance(circuits, QuantumCircuit) else circuits
    target_instances = (
        [target] if isinstance(target, (GateTarget, StateTarget)) else target
    )
    if len(prep_circuits) != len(target_instances):
        raise ValueError("Number of circuits and targets must be the same")
    if not all(
        isinstance(target_instance, (GateTarget, StateTarget))
        for target_instance in target_instances
    ):
        raise ValueError("All targets must be gate targets")

    target_instance = target
    is_gate_target = isinstance(target_instance, GateTarget)
    target_state = target_instance if isinstance(target_instance, StateTarget) else None

    # Compare qubits of each circuit between each other and ensure they are the same
    qubits = [qc.qubits for qc in prep_circuits]
    if len(qubits) > 1 and not all(
        qubits[0] == qubits[i] for i in range(1, len(qubits))
    ):
        raise ValueError("All circuits must have the same qubits")

    qc = prep_circuits[0].copy_empty_like(name="real_time_qc")
    num_qubits = qc.num_qubits  # all qubits (even those outside of causal cone)
    n_reps = execution_config.current_n_reps

    if len(execution_config.n_reps) > 1:  # Switch over possible number of repetitions
        n_reps_var = qc.add_input("n_reps", Uint(8))
    else:
        n_reps_var = n_reps

    if is_gate_target:
        causal_cone_size = target_instance.causal_cone_size
        causal_cone_qubits = target_instance.causal_cone_qubits
    else:
        causal_cone_size = num_qubits
        causal_cone_qubits = qc.qubits

    # Add classical register for measurements
    if not qc.clbits:
        meas = ClassicalRegister(causal_cone_size, name="meas")
        qc.add_register(meas)
    else:
        meas = qc.clbits

    if is_gate_target:  # Declare input states variables
        input_state_vars = [
            qc.add_input(f"input_state_{i}", Uint(4)) for i in range(num_qubits)
        ]
    else:
        input_state_vars = None

    observables_vars = [
        qc.add_input(f"observable_{i}", Uint(4)) for i in range(causal_cone_size)
    ]

    if is_gate_target:
        input_circuits = get_single_qubit_input_states(
            target_instance.input_states_choice
        )

        for q_idx, qubit in enumerate(qc.qubits):  # Input state preparation
            with qc.switch(input_state_vars[q_idx]) as case_input_state:
                for i, input_circuit in enumerate(input_circuits):
                    with case_input_state(i):
                        qc.compose(input_circuit, [qubit], inplace=True)

    if len(prep_circuits) > 1:  # Switch over possible circuit contexts
        circuit_choice = qc.add_input("circuit_choice", Uint(8))
        with qc.switch(circuit_choice) as circuit_case:
            for i, prep_circuit in enumerate(prep_circuits):
                with circuit_case(i):
                    handle_real_time_n_reps(
                        execution_config.n_reps, n_reps_var, prep_circuit, qc
                    )
    else:
        prep_circuit = prep_circuits[0]
        handle_real_time_n_reps(execution_config.n_reps, n_reps_var, prep_circuit, qc)

    if reward_method in ["state", "channel"]:
        for q_idx, qubit in enumerate(causal_cone_qubits):
            with qc.switch(observables_vars[q_idx]) as case_observable:
                for i in range(3):
                    with case_observable(i):
                        qc.compose(
                            PauliMeasurementBasis()
                            .circuit([i])
                            .remove_final_measurements(False)
                            .decompose(),
                            [qubit],
                            inplace=True,
                        )
        for qubit, clbit in zip(causal_cone_qubits, meas):
            qc.measure(qubit, clbit)
        qc.reset(qc.qubits)

    elif reward_method == "cafe":
        for circ in prep_circuits:
            if circ.metadata.get("baseline_circuit") is None:
                raise ValueError("Baseline circuit not found in metadata")
        ref_circuits: List[QuantumCircuit] = [
            circ.metadata["baseline_circuit"].copy() for circ in prep_circuits
        ]
        layout = target_instance.layout
        cycle_circuit_inverses = [[] for _ in range(len(ref_circuits))]
        for i, ref_circ in enumerate(ref_circuits):
            for n in execution_config.n_reps:
                cycle_circuit, _ = causal_cone_circuit(
                    ref_circ.repeat(n), causal_cone_qubits
                )
                cycle_circuit.save_unitary()
                sim_unitary = (
                    AerSimulator(method="unitary")
                    .run(cycle_circuit)
                    .result()
                    .get_unitary()
                )
                inverse_circuit = ref_circ.copy_empty_like()
                inverse_circuit.unitary(
                    sim_unitary.adjoint(), causal_cone_qubits, label="U_inv"
                )
                inverse_circuit.measure(causal_cone_qubits, meas)
                cycle_circuit_inverses[i].append(inverse_circuit)

    qc = backend_info.custom_transpile(
        qc,
        initial_layout=target_instance.layout,
        scheduling=False,
        remove_final_measurements=False,
    )

    return qc


def apply_real_time_n_reps(
    n_reps_int: int, qc: QuantumCircuit, prep_circuit: QuantumCircuit
):
    """
    Apply the number of repetitions of the circuit in the real-time reward computation

    Args:
        n_reps_var: Variable for the number of repetitions
        qc: Quantum circuit to add the repetitions to
        prep_circuit: Circuit to be repeated
    """
    if n_reps_int > 1:
        with qc.for_loop(range(n_reps_int)):
            qc.compose(prep_circuit, inplace=True)
    else:
        qc.compose(prep_circuit, inplace=True)


def handle_real_time_n_reps(
    n_reps: List[int],
    n_reps_var: int | Var,
    prep_circuit: QuantumCircuit,
    qc: QuantumCircuit,
):
    """
    Handle the number of repetitions of the circuit in the real-time reward computation

    Args:
        n_reps: List of possible number of repetitions
        n_reps_var: Variable for the number of repetitions
        prep_circuit: Circuit to be repeated
        qc: Quantum circuit to add the repetitions to
    """
    if isinstance(n_reps_var, int):
        apply_real_time_n_reps(n_reps_var, qc, prep_circuit)
    elif isinstance(n_reps_var, Var):
        # TODO: When Qiskit will support variable range in for loop, replace
        # this with a for loop with appropriate range object
        with qc.switch(n_reps_var) as case_reps:
            for n in n_reps:
                with case_reps(n):
                    apply_real_time_n_reps(n, qc, prep_circuit)
        # with qc.for_loop([n_reps_var]):
        #     qc.compose(prep_circuit, inplace=True)
