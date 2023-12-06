"""
Class to generate a RL environment suitable for usage with TF-agents, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
"""

# For compatibility for options formatting between Estimators.
import json
from itertools import product
from typing import Dict, Union, Optional, List, Callable

import numpy as np

# Qiskit imports
from qiskit import pulse, schedule
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate, CircuitInstruction

# Qiskit Estimator Primitives: for computing Pauli expectation value sampling easily
from qiskit.primitives import BaseEstimator, Estimator, BackendEstimator

# Qiskit backends
from qiskit.providers import Options, BackendV2, BackendV1


# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.operators.measures import (
    average_gate_fidelity,
    state_fidelity,
    process_fidelity,
)
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.transpiler import Layout

from qiskit_aer.primitives import Estimator as AerEstimator

# Qiskit dynamics for pulse simulation (& benchmarking)
from qiskit_dynamics import DynamicsBackend
from qiskit_dynamics.array import Array
from qiskit_ibm_provider import IBMBackend

# Qiskit Experiments for generating reliable baseline for complex gate calibrations / state preparations
from qiskit_experiments.framework import BatchExperiment
from qiskit_experiments.library import ProcessTomography
from qiskit_experiments.library.tomography.basis import (
    PauliPreparationBasis,
)  # , Pauli6PreparationBasis
from qiskit_ibm_runtime import (
    Estimator as RuntimeEstimator,
    Options as RuntimeOptions,
)

# Tensorflow modules
from tensorflow_probability.python.distributions import Categorical

from helper_functions import retrieve_estimator
from qconfig import QiskitConfig

# QUA imports
# from qualang_tools.bakery.bakery import baking
# from qm.qua import *
# from qm.QuantumMachinesManager import QuantumMachinesManager

Estimator_type = Union[AerEstimator, RuntimeEstimator, Estimator, BackendEstimator]
Backend_type = Union[BackendV1, BackendV2]


def _calculate_chi_target_state(target_state: Dict, n_qubits: int):
    """
    Calculate for all P
    :param target_state: Dictionary containing info on target state (name, density matrix)
    :param n_qubits: Number of qubits
    :return: Target state supplemented with appropriate "Chi" key
    """
    assert "dm" in target_state, "No input data for target state, provide DensityMatrix"
    d = 2**n_qubits
    Pauli_basis = pauli_basis(num_qubits=n_qubits)
    target_state["Chi"] = np.array(
        [
            np.trace(
                np.array(target_state["dm"].to_operator()) @ Pauli_basis[k].to_matrix()
            ).real
            for k in range(d**2)
        ]
    )
    # Real part is taken to convert it in good format,
    # but imaginary part is always 0. as dm is hermitian and Pauli is traceless
    return target_state


def _define_target(target: Dict):
    tgt_register = target.get("register", None)
    q_register = None
    layout = None
    if tgt_register is not None:
        if isinstance(tgt_register, List):
            q_register = QuantumRegister(len(tgt_register))
            layout = Layout(
                {q_register[i]: tgt_register[i] for i in range(len(tgt_register))}
            )
        elif isinstance(tgt_register, QuantumRegister):  # QuantumRegister or None
            q_register = tgt_register
        else:
            raise TypeError("Register should be of type List[int] or QuantumRegister")

    if "gate" not in target and "circuit" not in target and "dm" not in target:
        raise KeyError(
            "No target provided, need to have one of the following: 'gate' for gate calibration,"
            " 'circuit' or 'dm' for state preparation"
        )
    elif ("gate" in target and "circuit" in target) or (
        "gate" in target and "dm" in target
    ):
        raise KeyError("Cannot have simultaneously a gate target and a state target")
    if "circuit" in target or "dm" in target:  # State preparation task
        target["target_type"] = "state"
        if "circuit" in target:
            assert isinstance(target["circuit"], QuantumCircuit), (
                "Provided circuit is not a qiskit.QuantumCircuit " "object"
            )
            target["dm"] = DensityMatrix(target["circuit"])

        assert (
            "dm" in target
        ), "no DensityMatrix or circuit argument provided to target dictionary"
        assert isinstance(
            target["dm"], DensityMatrix
        ), "Provided dm is not a DensityMatrix object"
        dm: DensityMatrix = target["dm"]
        n_qubits = dm.num_qubits

        if q_register is None:
            q_register = QuantumRegister(n_qubits)

        if layout is None:
            layout = Layout.generate_trivial_layout(q_register)

        return (
            _calculate_chi_target_state(target, n_qubits),
            "state",
            q_register,
            n_qubits,
            layout,
        )

    elif "gate" in target:  # Gate calibration task
        target["target_type"] = "gate"
        assert isinstance(
            target["gate"], Gate
        ), "Provided gate is not a qiskit.circuit.Gate operation"
        gate: Gate = target["gate"]
        n_qubits = gate.num_qubits
        if q_register is None:
            q_register = QuantumRegister(n_qubits)
        if layout is None:
            layout = Layout.generate_trivial_layout(q_register)

        assert gate.num_qubits == len(q_register), (
            f"Target gate number of qubits ({gate.num_qubits}) "
            f"incompatible with indicated 'register' ({len(q_register)})"
        )
        if "input_states" not in target:
            target["input_states"] = [
                {"circuit": PauliPreparationBasis().circuit(s).decompose()}
                for s in product(range(4), repeat=len(tgt_register))
            ]

            # target['input_states'] = [{"dm": Pauli6PreparationBasis().matrix(s),
            #                            "circuit": CircuitOp(Pauli6PreparationBasis().circuit(s).decompose())}
            #                           for s in product(range(6), repeat=len(tgt_register))]

        for i, input_state in enumerate(target["input_states"]):
            if "circuit" not in input_state:
                raise KeyError("'circuit' key missing in input_state")
            assert isinstance(input_state["circuit"], QuantumCircuit), (
                "Provided circuit is not a" "qiskit.QuantumCircuit object"
            )

            input_circuit: QuantumCircuit = input_state["circuit"]
            input_state["dm"] = DensityMatrix(input_circuit)

            state_target_circuit = QuantumCircuit(q_register)
            state_target_circuit.append(input_circuit.to_instruction(), q_register)
            state_target_circuit.append(CircuitInstruction(gate, q_register))

            input_state["target_state"] = {
                "dm": DensityMatrix(state_target_circuit),
                "circuit": state_target_circuit,
                "target_type": "state",
            }
            input_state["target_state"] = _calculate_chi_target_state(
                input_state["target_state"], n_qubits
            )
        return target, "gate", q_register, n_qubits, layout
    else:
        raise KeyError("target type not identified, must be either gate or state")


class QuantumEnvironment:
    def __init__(
        self,
        target: Dict,
        abstraction_level: str = "circuit",
        Qiskit_config: Optional[QiskitConfig] = None,
        QUA_config: Optional[Dict] = None,
        sampling_Pauli_space: int = 10,
        n_shots: int = 1,
        c_factor: float = 0.5,
    ):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param target: control target of interest (can be either a gate to be calibrated or a state to be prepared)
            Target should be a Dict containing target type ('gate' or 'state') as well as necessary fields to initiate
            the calibration (cf. example)
        :param abstraction_level: Circuit or pulse level parametrization of action space
        :param Qiskit_config: Dictionary containing all info for running Qiskit program (i.e. service, backend,
            options, parametrized_circuit)
        :param QUA_config: Dictionary containing all infor for running a QUA program
        :param sampling_Pauli_space: Number of samples to build fidelity estimator for one action
        :param n_shots: Number of shots to sample for one specific computation (action/Pauli expectation sampling)
        :param c_factor: Scaling factor for reward normalization
        """

        assert abstraction_level == "circuit" or abstraction_level == "pulse", (
            "Abstraction layer can be either pulse" " or circuit"
        )
        self.abstraction_level: str = abstraction_level
        if Qiskit_config is None and QUA_config is None:
            raise AttributeError(
                "QuantumEnvironment requires one hardware configuration (can be Qiskit or QUA based)"
            )
        if Qiskit_config is not None and QUA_config is not None:
            raise AttributeError(
                "Cannot provide simultaneously a QUA setup and a Qiskit config "
            )
        elif Qiskit_config is not None:
            self._config_type = "Qiskit"
            self._config: Qiskit_config = Qiskit_config
            (
                self.target,
                self.target_type,
                self.tgt_register,
                self._n_qubits,
                self._layout,
            ) = _define_target(target)

            self._d = 2**self.n_qubits
            self.c_factor = c_factor
            self.sampling_Pauli_space = sampling_Pauli_space
            self.n_shots = n_shots

            self.backend: Optional[Backend_type] = Qiskit_config.backend
            if self.backend is not None:
                if self.n_qubits > self.backend.num_qubits:
                    raise ValueError(
                        f"Target contains more qubits ({self._n_qubits}) than backend ({self.backend.num_qubits})"
                    )
            self.parametrized_circuit_func: Callable = (
                Qiskit_config.parametrized_circuit
            )
            estimator_options: Optional[
                Union[Options, Dict, RuntimeOptions]
            ] = Qiskit_config.estimator_options

            self._estimator = retrieve_estimator(
                self.backend,
                self.layout,
                self.config,
                self.abstraction_level,
                estimator_options,
            )

        elif QUA_config is not None:
            raise AttributeError("QUA compatibility not yet implemented")
            # self._config_type = "QUA"
            # self._config = QUA_config

            # TODO: Add a QUA program

        # Data storage for TF-Agents or plotting
        self._session_counts = 0
        self._step_tracker = 0
        self.action_history = []
        self.density_matrix_history = []
        self.reward_history = []
        self.qc_history = []
        if self.target_type == "gate":
            self.process_fidelity_history = []
            self.avg_fidelity_history = []
            self.built_unitaries = []
        else:
            self.state_fidelity_history = []

    def perform_action(self, actions: np.array, do_benchmark: bool = True):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :param do_benchmark: Indicates if actual fidelity computation should be done on top of reward computation
        :return: Reward table (reward for each run in the batch)
        """

        qc = QuantumCircuit(
            self.tgt_register
        )  # Reset the QuantumCircuit instance for next iteration
        parametrized_circ = QuantumCircuit(self.tgt_register)
        angles, batch_size = np.array(actions), len(np.array(actions))
        self.action_history.append(angles)

        if self.target_type == "gate":
            # Pick random input state from the list of possible input states (forming a tomographically complete set)
            index = np.random.randint(len(self.target["input_states"]))
            input_state = self.target["input_states"][index]

            target_state = input_state[
                "target_state"
            ]  # Ideal output state associated to input (Gate |input>=|output>)
            # Append input state circuit to full quantum circuit for gate calibration
            qc.append(input_state["circuit"].to_instruction(), self.tgt_register)
        else:  # State preparation task
            target_state = self.target
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = np.round(
            [
                self.c_factor
                * target_state["Chi"][p]
                / (self._d * distribution.prob(p))
                for p in pauli_index
            ],
            5,
        )

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp.from_list(
            [
                (pauli_basis(num_qubits=self._n_qubits)[p].to_label(), reward_factor[i])
                for i, p in enumerate(pauli_index)
            ]
        )
        # Benchmarking block: not part of reward calculation
        # Apply parametrized quantum circuit (action), for benchmarking only

        self.parametrized_circuit_func(parametrized_circ)
        qc_list = [
            parametrized_circ.assign_parameters(angle_set) for angle_set in angles
        ]
        self.qc_history.append(qc_list)
        if do_benchmark:
            self.store_benchmarks(qc_list)

        # Build full quantum circuit: concatenate input state prep and parametrized unitary
        self.parametrized_circuit_func(qc)
        if isinstance(self.estimator, RuntimeEstimator):
            job = self.estimator.run(
                circuits=[qc] * batch_size,
                observables=[observables] * batch_size,
                parameter_values=angles,
                shots=self.sampling_Pauli_space * self.n_shots,
                job_tags=[f"rl_qoc_step{self._step_tracker}"],
            )
        else:
            job = self.estimator.run(
                circuits=[qc] * batch_size,
                observables=[observables] * batch_size,
                parameter_values=angles,
                shots=self.sampling_Pauli_space * self.n_shots,
            )

        self._step_tracker += 1
        reward_table = job.result().values
        self.reward_history.append(reward_table)
        assert len(reward_table) == batch_size
        return reward_table  # Shape [batchsize]

    def store_benchmarks(self, qc_list: List[QuantumCircuit]):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        """

        # Circuit list for each action of the batch
        if self.abstraction_level == "circuit":
            q_state_list = [Statevector.from_instruction(qc) for qc in qc_list]
            density_matrix = DensityMatrix(
                np.mean(
                    [q_state.to_operator().to_matrix() for q_state in q_state_list],
                    axis=0,
                )
            )
            self.density_matrix_history.append(density_matrix)

            if self.target_type == "state":
                self.state_fidelity_history.append(
                    state_fidelity(self.target["dm"], density_matrix)
                )
            else:  # Gate calibration task
                q_process_list = [Operator(qc) for qc in qc_list]

                prc_fidelity = np.mean(
                    [
                        process_fidelity(q_process, Operator(self.target["gate"]))
                        for q_process in q_process_list
                    ]
                )
                avg_fidelity = np.mean(
                    [
                        average_gate_fidelity(q_process, Operator(self.target["gate"]))
                        for q_process in q_process_list
                    ]
                )
                self.built_unitaries.append(q_process_list)
                self.process_fidelity_history.append(
                    prc_fidelity
                )  # Avg process fidelity over the action batch
                self.avg_fidelity_history.append(
                    avg_fidelity
                )  # Avg gate fidelity over the action batch
                # for i, input_state in enumerate(self.target["input_states"]):
                #     output_states = [DensityMatrix(Operator(qc) @ input_state["dm"] @ Operator(qc).adjoint())
                #                      for qc in qc_list]
                #     self.input_output_state_fidelity_history[i].append(
                #         np.mean([state_fidelity(input_state["target_state"]["dm"],
                #                                 output_state) for output_state in output_states]))
        elif self.abstraction_level == "pulse":
            # Pulse simulation
            schedule_list = [
                schedule(qc, backend=self.backend, dt=self.backend.target.dt)
                for qc in qc_list
            ]
            unitaries = self._simulate_pulse_schedules(schedule_list)
            # TODO: Line below yields an error if simulation is not done over a set of qubit (fails if third level of
            # TODO: transmon is simulated), adapt the target gate operator accordingly.
            unitaries = [Operator(np.array(unitary.y[0])) for unitary in unitaries]

            if self.target_type == "state":
                density_matrix = DensityMatrix(
                    np.mean(
                        [
                            Statevector.from_int(0, dims=self._d).evolve(unitary)
                            for unitary in unitaries
                        ]
                    )
                )
                self.state_fidelity_history.append(
                    state_fidelity(self.target["dm"], density_matrix)
                )
            else:
                self.process_fidelity_history.append(
                    np.mean(
                        [
                            process_fidelity(unitary, self.target["gate"])
                            for unitary in unitaries
                        ]
                    )
                )
                self.avg_fidelity_history.append(
                    np.mean(
                        [
                            average_gate_fidelity(unitary, self.target["gate"])
                            for unitary in unitaries
                        ]
                    )
                )
            self.built_unitaries.append(unitaries)

    def gate_fidelity_from_process_tomography(self, qc_list: List[QuantumCircuit]):
        """
        Extract average gate and process fidelities from batch of Quantum Circuit for target gate
        """
        # Process tomography
        assert self.target_type == "gate", "Target must be of type gate"
        batch_size = len(qc_list)

        exps = BatchExperiment(
            [
                ProcessTomography(
                    qc, backend=self.backend, physical_qubits=self.tgt_register
                )
                for qc in qc_list
            ]
        )

        results = exps.run().block_for_results()
        process_results = [
            results.child_data(i).analysis_results(0) for i in range(batch_size)
        ]
        Choi_matrices = [matrix.value for matrix in process_results]
        avg_gate_fidelity = np.mean(
            [
                average_gate_fidelity(Choi_matrix, Operator(self.target["gate"]))
                for Choi_matrix in Choi_matrices
            ]
        )
        prc_fidelity = np.mean(
            [
                process_fidelity(Choi_matrix, Operator(self.target["gate"]))
                for Choi_matrix in Choi_matrices
            ]
        )

        self.process_fidelity_history.append(prc_fidelity)
        self.avg_fidelity_history.append(avg_gate_fidelity)
        return avg_gate_fidelity, prc_fidelity

    def _simulate_pulse_schedules(
        self, schedule_list: List[Union[pulse.Schedule, pulse.ScheduleBlock]]
    ):
        """
        Method used to simulate pulse schedules, jit compatible
        """
        time_f = self.backend.target.dt * schedule_list[0].duration
        if not isinstance(self.backend, DynamicsBackend):
            raise TypeError(
                "Backend should be of type DynamicsBackend for this benchmark"
            )
        unitaries = self.backend.options.solver.solve(
            t_span=Array([0.0, time_f]),
            y0=self.backend.options.initial_state
            if self.backend.options.initial_state != "ground_state"
            else Statevector(self.backend._dressed_states[:, 0]),
            t_eval=[time_f],
            signals=schedule_list,
            method="jax_odeint",
        )

        return unitaries

    def clear_history(self):
        self.qc_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        if self.target_type == "gate":
            self.avg_fidelity_history.clear()
            self.process_fidelity_history.clear()
            self.built_unitaries.clear()

        else:
            self.state_fidelity_history.clear()
            self.density_matrix_history.clear()

    def close(self) -> None:
        if isinstance(self.estimator, RuntimeEstimator):
            self.estimator.session.close()
        elif isinstance(self.backend, IBMBackend):
            self.backend.cancel_session()

    def __repr__(self):
        string = f"QuantumEnvironment composed of {self._n_qubits} qubits, \n"
        string += (
            f"Defined target: {self.target_type} "
            f"({self.target.get('gate', None) if not None else self.target['dm']})\n"
        )
        string += f"Physical qubits: {self.target['register']}\n"
        string += f"Backend: {self.backend},\n"
        string += f"Abstraction level: {self.abstraction_level},\n"
        string += f"Run options: N_shots ({self.n_shots}), Sampling_Pauli_space ({self.sampling_Pauli_space}), \n"
        return string

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        assert (
            isinstance(n_qubits, int) and n_qubits > 0
        ), "n_qubits must be a positive integer"
        self._n_qubits = n_qubits

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout: Layout):
        self._layout = layout

    @property
    def config_type(self):
        return self._config_type

    @property
    def config(self):
        return self._config

    @property
    def estimator(self) -> Estimator_type:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator):
        self._estimator = estimator

    @property
    def step_tracker(self):
        return self._step_tracker

    @step_tracker.setter
    def step_tracker(self, step: int):
        assert step >= 0, "step must be positive integer"
        self._step_tracker = step

    def to_json(self):
        return json.dumps(
            {
                "n_qubits": self.n_qubits,
                "config": self._config,
                "abstraction_level": self.abstraction_level,
                "sampling_Pauli_space": self.sampling_Pauli_space,
                "n_shots": self.n_shots,
                "target_type": self.target_type,
                "target": self.target,
                "c_factor": self.c_factor,
                "reward_history": self.reward_history,
                "action_history": self.action_history,
                "fidelity_history": self.avg_fidelity_history
                if self.target_type == "gate"
                else self.state_fidelity_history,
            }
        )

    @classmethod
    def from_json(cls, json_str):
        """Return a MyCustomClass instance based on the input JSON string."""

        class_info = json.loads(json_str)
        abstraction_level = class_info["abstraction_level"]
        target = class_info["target"]
        n_shots = class_info["n_shots"]
        c_factor = class_info["c_factor"]
        sampling_Pauli_space = class_info["sampling_Pauli_space"]
        config = class_info["config"]
        q_env = cls(
            target,
            abstraction_level,
            config,
            None,
            sampling_Pauli_space,
            n_shots,
            c_factor,
        )
        q_env.reward_history = class_info["reward_history"]
        q_env.action_history = class_info["action_history"]
        if class_info["target_type"] == "gate":
            q_env.avg_fidelity_history = class_info["fidelity_history"]
        else:
            q_env.state_fidelity_history = class_info["fidelity_history"]
        return q_env
