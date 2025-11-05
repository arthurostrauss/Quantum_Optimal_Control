"""
Linear demonstration script for running x_cal on DGX-Quantum (DGX-Q) using rl_qoc.

This mirrors the style of the sync-hook main.py: build environment inline, then
at the end switch to the DGX workflow:
  1) Start the OPX QUA program with local QM manager/machine
  2) Generate the DGX-side Python program
  3) Deploy and run that program on the DGX over SSH with streaming
  4) Close the environment when done
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
import json
import os

from rl_qoc.agent.ppo_config import TotalUpdates
from rl_qoc.environment.wrappers.custom_wrappers import RescaleAndClipAction
from rl_qoc.qua.dgx_q.generate_dgx_program import generate_dgx_program
from rl_qoc.qua.dgx_q.ssh_ops import deploy_and_run_script
from rl_qoc.qua.qm_environment import QMEnvironment
from rl_qoc import QEnvConfig, ExecutionConfig, BenchmarkConfig, GateTarget, StateTarget
from rl_qoc.qua import QMConfig
from rl_qoc import ChannelReward
from qiskit_qm_provider import (
    FluxTunableTransmonBackend,
    InputType,
)
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, Gate
from typing import List
from qiskit_qm_provider.backend.backend_utils import add_basic_macros_to_machine
from rl_qoc.qua.iqcc import get_machine_from_iqcc
import numpy as np
from gymnasium.spaces import Box
from rl_qoc.helpers import load_from_yaml_file
from iqcc_calibration_tools.quam_config.components import Transmon
from qiskit_qm_provider import QMInstructionProperties
from rl_qoc.helpers import add_custom_gate

# Set your quantum computer backend
iqcc_token_path = Path.home() / "iqcc_token.json"
gh_token_path = Path.home() / "dgx_suite_config.json"
with open(iqcc_token_path, "r") as f:
    iqcc_config = json.load(f)

with open(gh_token_path, "r") as f:
    gh_config = json.load(f)

gh_username = gh_config["GH_USER"]
gh_password = gh_config["GH_SP"]
gh_host = gh_config["GH_HOST"]

backend_name = "gilboa"

machine, iqcc = get_machine_from_iqcc(backend_name, iqcc_config[backend_name])

add_basic_macros_to_machine(machine)
backend = FluxTunableTransmonBackend(machine)
path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
ppo_config = load_from_yaml_file(path)

path_to_python_wrapper = "path_to_python_wrapper.py"


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: List[Parameter], q_reg: QuantumRegister, **kwargs
):

    physical_qubits: List[int] = kwargs["physical_qubits"]
    backend: FluxTunableTransmonBackend = kwargs["backend"]

    # TODO: Enter your custom parametric QUA macro here
    def qua_macro(amp):
        qubit: Transmon = backend.get_qubit(physical_qubits[0])
        qubit.xy.play("x180", amplitude_scale=amp)

    # Create a custom gate with the QUA macro
    custom_x = Gate("x_cal", 1, params)
    instruction_prop = QMInstructionProperties(qua_pulse_macro=qua_macro)
    qc = add_custom_gate(qc, custom_x, q_reg, params, physical_qubits, backend, instruction_prop)
    return qc


physical_qubits = (0,)

target_name = "x"
target = GateTarget(gate=target_name, physical_qubits=physical_qubits)
reward = ChannelReward()


# Action space specification
param_bounds = [(-1.98, 2.0)]  # Can be any number of bounds

# Environment execution parameters
seed = 1203  # Master seed to make training reproducible
batch_size = 32  # Number of actions to evaluate per policy evaluation
n_shots = 50  # Minimum number of shots per fiducial evaluation
pauli_sampling = 100  # Number of fiducials to compute for fidelity estimation (DFE only)
n_reps = 1  # Number of repetitions of the cycle circuit
num_updates = TotalUpdates(50)
input_type = InputType.INPUT_STREAM


def create_action_space(param_bounds):
    param_bounds = np.array(param_bounds, dtype=np.float32)
    lower_bound, upper_bound = param_bounds.T
    return Box(low=lower_bound, high=upper_bound, shape=(len(param_bounds),), dtype=np.float32)


action_space = create_action_space(param_bounds)

backend_config = QMConfig(
    parametrized_circuit=apply_parametrized_circuit,
    backend=backend,
    input_type=input_type,
    verbosity=2,
    parametrized_circuit_kwargs={"physical_qubits": physical_qubits, "backend": backend},
    num_updates=num_updates.total_updates,
)
execution_config = ExecutionConfig(
    batch_size=batch_size,
    sampling_paulis=pauli_sampling,
    n_shots=n_shots,
    n_reps=[n_reps],
    seed=seed,
)
q_env_config = QEnvConfig(
    target=target,
    backend_config=backend_config,
    action_space=action_space,
    execution_config=execution_config,
    reward=reward,
    benchmark_config=BenchmarkConfig(0),
)  # No benchmark for now

q_env = QMEnvironment(training_config=q_env_config)
rescaled_env = RescaleAndClipAction(q_env, -1.0, 1.0)


###############################
# Start OPX job and run DGX    #
###############################

# 1) Start the OPX-side QUA program
# qm_job = q_env.start_program()

# 2) Generate DGX-side program locally
local_prog_path = generate_dgx_program(
    env=q_env,
    ppo_config=ppo_config,
    path_to_python_wrapper=path_to_python_wrapper,
)
print(local_prog_path)