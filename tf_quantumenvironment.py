"""
Class to generate a RL environment suitable for usage with TF-agents, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
"""
from abc import ABC

# Qiskit imports
from qiskit import pulse, schedule
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.transpiler import InstructionProperties
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister, Reset, Gate
from qiskit.circuit.library import XGate, SXGate, RZGate, HGate, IGate, ZGate, SGate, SdgGate, TGate, TdgGate, CXGate
from qiskit.providers import BackendV1, BackendV2
from quantumenvironment import QuantumEnvironment

from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.operators.measures import average_gate_fidelity, state_fidelity, process_fidelity
# Qiskit dynamics for pulse simulation (benchmarking)
from qiskit_dynamics import DynamicsBackend, Solver
from qiskit_dynamics.array import Array, wrap
from qiskit_dynamics.models import HamiltonianModel

# Qiskit Experiments for generating reliable baseline for more complex gate calibrations / state preparations
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.library.calibration import RoughXSXAmplitudeCal, RoughDragCal
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.library import ProcessTomography
from qiskit_experiments.library.tomography.basis import PauliPreparationBasis, Pauli6PreparationBasis

# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit.primitives import Estimator, BackendEstimator
from qiskit_ibm_runtime import Estimator as Runtime_Estimator, IBMBackend as Runtime_Backend
from qiskit_aer.primitives import Estimator as Aer_Estimator
from qiskit_aer.backends.aerbackend import AerBackend

import numpy as np
from itertools import product
from typing import Dict, Union, Optional, List, Callable
from copy import deepcopy

# QUA imports
# from qualang_tools.bakery.bakery import baking
# from qm.qua import *
# from qm.QuantumMachinesManager import QuantumMachinesManager

# Tensorflow modules
from tensorflow_probability.python.distributions import Categorical
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.environments import PyEnvironment

import jax

jit = wrap(jax.jit, decorator=True)


class TFQuantumEnvironment(QuantumEnvironment, PyEnvironment):

    def observation_spec(self) -> types.NestedArraySpec:
        pass

    def action_spec(self) -> types.NestedArraySpec:
        pass

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        pass

    def _reset(self) -> ts.TimeStep:
        pass

    def __init__(self, q_env: QuantumEnvironment, circuit_context: QuantumCircuit,
                 action_spec: types.ArraySpec,
                 observation_spec: types.ArraySpec,
                 batch_size: int = 100):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param q_env: Existing QuantumEnvironment object, containing a target and a configuration
        :param batch_size: Number of trajectories to compute average return
        """
        if q_env.config_type == "Qiskit":
            super().__init__(q_env.target, q_env.abstraction_level, q_env.config, None,
                             q_env.sampling_Pauli_space, q_env.n_shots, q_env.c_factor)
        else:
            super().__init__(q_env.target, q_env.abstraction_level, None, q_env.config,
                             q_env.sampling_Pauli_space, q_env.n_shots, q_env.c_factor)

        self._batch_size = batch_size

    def batched(self) -> bool:
        if self.config == 'Qiskit':
            return True
        else:
            return False

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        try:
            assert size > 0 and isinstance(size, int)
            self._batch_size = size
        except AssertionError:
            raise ValueError('Batch size should be positive integer.')

    def perform_action(self, actions: types.NestedTensorOrArray, do_benchmark: bool = True):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :param do_benchmark: Indicates if actual fidelity computation should be done on top of reward computation
        :return: Reward table (reward for each run in the batch)
        """
        qc = QuantumCircuit(self.q_register)  # Reset the QuantumCircuit instance for next iteration
        angles, batch_size = np.array(actions), len(np.array(actions))
        self.action_history.append(angles)

        if self.target_type == 'gate':
            # Pick random input state from the list of possible input states (forming a tomographically complete set)
            index = np.random.randint(len(self.target["input_states"]))
            input_state = self.target["input_states"][index]
            target_state = input_state["target_state"]  # Ideal output state associated to input (Gate |input>=|output>)
            # Append input state circuit to full quantum circuit for gate calibration
            qc.append(input_state["circuit"].to_instruction(), self.tgt_register)

        else:  # State preparation task
            target_state = self.target
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = np.round([self.c_factor * target_state["Chi"][p] / (self.d * distribution.prob(p))
                                  for p in pauli_index], 5)

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp.from_list([(self.Pauli_ops[p].to_label(), reward_factor[i])
                                               for i, p in enumerate(pauli_index)])

        # Benchmarking block: not part of reward calculation
        # Apply parametrized quantum circuit (action), for benchmarking only
        parametrized_circ = QuantumCircuit(self._n_qubits)
        self.parametrized_circuit_func(parametrized_circ)
        qc_list = [parametrized_circ.bind_parameters(angle_set) for angle_set in angles]
        self.qc_history.append(qc_list)
        if do_benchmark:
            self._store_benchmarks(qc_list)

        # Build full quantum circuit: concatenate input state prep and parametrized unitary
        self.parametrized_circuit_func(qc)
        job = self.estimator.run(circuits=[qc] * batch_size, observables=[observables] * batch_size,
                                 parameter_values=angles, shots=self.sampling_Pauli_space * self.n_shots)
        reward_table = job.result().values
        self.reward_history.append(reward_table)
        assert len(reward_table) == batch_size
        return reward_table  # Shape [batchsize]

    def _store_benchmarks(self, qc_list: List[QuantumCircuit]):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        """

        # Circuit list for each action of the batch
        if self.abstraction_level == 'circuit':
            q_state_list = [Statevector.from_instruction(qc) for qc in qc_list]
            density_matrix = DensityMatrix(np.mean([q_state.to_operator() for q_state in q_state_list], axis=0))
            self.density_matrix_history.append(density_matrix)

            if self.target_type == 'state':
                self.state_fidelity_history.append(state_fidelity(self.target["dm"], density_matrix))
            else:  # Gate calibration task
                q_process_list = [Operator(qc) for qc in qc_list]

                prc_fidelity = np.mean([process_fidelity(q_process, Operator(self.target["gate"]))
                                        for q_process in q_process_list])
                avg_fidelity = np.mean([average_gate_fidelity(q_process, Operator(self.target["gate"]))
                                        for q_process in q_process_list])
                self.built_unitaries.append(q_process_list)
                self.process_fidelity_history.append(prc_fidelity)  # Avg process fidelity over the action batch
                self.avg_fidelity_history.append(avg_fidelity)  # Avg gate fidelity over the action batch
                # for i, input_state in enumerate(self.target["input_states"]):
                #     output_states = [DensityMatrix(Operator(qc) @ input_state["dm"] @ Operator(qc).adjoint())
                #                      for qc in qc_list]
                #     self.input_output_state_fidelity_history[i].append(
                #         np.mean([state_fidelity(input_state["target_state"]["dm"],
                #                                 output_state) for output_state in output_states]))
        elif self.abstraction_level == 'pulse':
            # Pulse simulation
            schedule_list = [schedule(qc, backend=self.backend, dt=self.backend.target.dt) for qc in qc_list]
            unitaries = self._simulate_pulse_schedules(schedule_list)
            # TODO: Line below yields an error if simulation is not done over a set of qubit (fails if third level of
            # TODO: transmon is simulated), adapt the target gate operator accordingly.
            unitaries = [Operator(np.array(unitary.y[0])) for unitary in unitaries]

            if self.target_type == 'state':
                density_matrix = DensityMatrix(np.mean([Statevector.from_int(0, dims=self.d).evolve(unitary)
                                                        for unitary in unitaries]))
                self.state_fidelity_history.append(state_fidelity(self.target["dm"], density_matrix))
            else:
                self.process_fidelity_history.append(np.mean([process_fidelity(unitary, self.target["gate"])
                                                              for unitary in unitaries]))
                self.avg_fidelity_history.append(np.mean([average_gate_fidelity(unitary, self.target["gate"])
                                                          for unitary in unitaries]))
            self.built_unitaries.append(unitaries)
