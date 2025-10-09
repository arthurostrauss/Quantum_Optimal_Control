from __future__ import annotations

from typing import List, Optional, Union, Dict, Any as _Any
from dataclasses import asdict
import numpy as np
import os

from rl_qoc import GateTarget, StateTarget
from rl_qoc.qua import QMConfig

from ..iqcc.sync_hook_specs import (
    WrapperSpec,
    RewardSpec,
    coerce_wrappers,
    coerce_reward,
)


def _infer_specs_from_env(env: _Any) -> tuple[list[dict[str, _Any]], dict[str, _Any]]:
    wrappers_list: list[dict[str, _Any]] = []

    # Reward spec: default to reward class and grab reward_args when available
    try:
        reward_obj = env.unwrapped.config.reward
        reward_spec: dict[str, _Any] = {
            "name": reward_obj.__class__.__name__,
            "module": getattr(reward_obj.__class__, "__module__", None),
        }
        reward_args = getattr(reward_obj, "reward_args", None)
        if isinstance(reward_args, dict) and reward_args:
            reward_spec["kwargs"] = reward_args
    except Exception:
        reward_spec = {"name": "FidelityReward"}

    # Walk wrapper chain (generic handling using recorded constructor args)
    cur = env
    visited = set()
    while hasattr(cur, "env") and id(cur) not in visited:
        visited.add(id(cur))
        cls_obj = cur.__class__
        cls_name = cls_obj.__name__
        cls_module = getattr(cls_obj, "__module__", None)
        saved_kwargs = getattr(cur, "_saved_kwargs", None)

        wrapper_spec: Dict[str, _Any] = {"name": cls_name}
        if cls_module:
            wrapper_spec["module"] = cls_module
        if isinstance(saved_kwargs, dict) and saved_kwargs:
            wrapper_spec["kwargs"] = saved_kwargs

        wrappers_list.append(wrapper_spec)
        cur = cur.env

    return wrappers_list, reward_spec


def generate_dgx_program(
    env: _Any,
    ppo_config: Dict[str, _Any],
    path_to_python_wrapper: str,
    output_dir: Optional[str] = None,
    wrappers: Optional[Union[List[WrapperSpec], List[Dict[str, _Any]]]] = None,
    reward_spec: Optional[Union[RewardSpec, Dict[str, _Any]]] = None,
) -> str:
    """Generate a DGX program (sync-like) that runs with local QM backend (no cloud job).

    Differences from cloud sync:
    - No iqcc cloud runtime; no external job object
    - QMConfig receives path_to_python_wrapper to enable OPNIC streams
    - Wrapper and reward resolution support external/gym modules and reward_args
    """
    if env is None:
        raise ValueError("env must be provided to infer configuration")

    try:
        qenv = env.unwrapped
    except Exception:
        qenv = env

    # Target
    effective_target = qenv.config.target

    target_initialization_code = ""
    if isinstance(effective_target, GateTarget):
        physical_qubits = tuple(effective_target.physical_qubits)
        gate_name = effective_target.gate.name
        target_initialization_code = f"""physical_qubits = {physical_qubits}
target_name = \"{gate_name}\"
target = GateTarget(gate=target_name, physical_qubits=physical_qubits)
"""
    elif isinstance(effective_target, StateTarget):
        physical_qubits = tuple(effective_target.physical_qubits)
        state_str = np.array2string(
            effective_target.dm.data, precision=8, separator=", ", suppress_small=True
        )
        target_initialization_code = f"""
physical_qubits = {physical_qubits}
target_state = np.array({state_str}, dtype=complex)
target = StateTarget(state=target_state, physical_qubits=physical_qubits)
"""
    else:
        raise ValueError("Unsupported target type from env")

    # Action bounds
    aspace = getattr(qenv, "action_space", None) or getattr(qenv.config, "action_space", None)
    if aspace is None:
        raise ValueError("Unable to infer action bounds from env.action_space")
    low = np.array(aspace.low).flatten()
    high = np.array(aspace.high).flatten()
    effective_param_bounds = list(map(lambda x: (float(x[0]), float(x[1])), zip(low, high)))

    # Execution settings
    exec_cfg = qenv.config.execution_config
    effective_seed = exec_cfg.seed
    effective_batch_size = exec_cfg.batch_size
    effective_n_shots = exec_cfg.n_shots
    effective_pauli_sampling = exec_cfg.sampling_paulis
    effective_n_reps = exec_cfg.n_reps

    # Backend settings
    bcfg = qenv.config.backend_config
    effective_input_type = bcfg.input_type
    verbosity_val = getattr(bcfg, "verbosity", 1)
    timeout_val = getattr(bcfg, "timeout", 60)
    total_updates_value = int(getattr(bcfg, "num_updates", 0))
    if total_updates_value <= 0:
        raise ValueError("backend_config.num_updates must be > 0")

    # Reward and wrappers
    reward_spec_dict = coerce_reward(reward_spec) if reward_spec is not None else None
    if reward_spec_dict is None:
        reward_obj = qenv.config.reward
        reward_kwargs = getattr(reward_obj, "reward_args", {}) or {}
        reward_spec_dict = {
            "name": reward_obj.__class__.__name__,
            "module": getattr(reward_obj.__class__, "__module__", None),
        }
        if isinstance(reward_kwargs, dict) and reward_kwargs:
            reward_spec_dict["kwargs"] = reward_kwargs

    wrappers_list = coerce_wrappers(wrappers) if wrappers is not None else None
    if wrappers_list is None:
        wrappers_list, _ = _infer_specs_from_env(env)

    wrappers_literal = repr(wrappers_list)
    reward_spec_literal = repr(reward_spec_dict)
    input_type_str = getattr(effective_input_type, "name", str(effective_input_type))

    # Assemble DGX program content
    script_code = f"""# AUTO-GENERATED DGX PROGRAM - DO NOT EDIT MANUALLY
# Generated by generate_dgx_program

import json
import importlib
import rl_qoc.environment.wrappers as _custom_wrappers
import gymnasium.wrappers as _gym_wrappers
import rl_qoc.rewards as _rewards_mod
from rl_qoc.agent import PPOConfig, TrainingConfig, TrainFunctionSettings, TotalUpdates
from rl_qoc.qua import QMEnvironment, QMConfig
from rl_qoc import (
    GateTarget,
    StateTarget,
    ExecutionConfig,
    QEnvConfig,
    BenchmarkConfig,
)
import numpy as np
from gymnasium.spaces import Box

# Embedded specs
_WRAPPERS_SPEC = {wrappers_literal}
_REWARD_SPEC = {reward_spec_literal}
_PATH_TO_PY_WRAPPER = {path_to_python_wrapper!r}


def _resolve_wrapper(name: str, module: str | None = None):
    if isinstance(module, str) and module:
        try:
            mod = importlib.import_module(module)
            if hasattr(mod, name):
                return getattr(mod, name)
        except Exception:
            pass
    if hasattr(_custom_wrappers, name):
        return getattr(_custom_wrappers, name)
    if hasattr(_gym_wrappers, name):
        return getattr(_gym_wrappers, name)
    raise ValueError("Unknown wrapper '%s' (module=%s)" % (name, module))


def _instantiate_reward_from_spec(spec):
    name = spec.get("name")
    kwargs = spec.get("kwargs")
    module = spec.get("module")
    cls = None
    if isinstance(module, str) and module:
        try:
            mod = importlib.import_module(module)
            cls = getattr(mod, name, None)
        except Exception:
            cls = None
    if cls is None and hasattr(_rewards_mod, name):
        cls = getattr(_rewards_mod, name)
    if cls is None:
        reward_dict = getattr(_rewards_mod, "reward_dict", {{}})
        if name in reward_dict:
            cls = reward_dict[name]
        else:
            lname = str(name).lower()
            cls = reward_dict.get(lname)
    if cls is None:
        raise ValueError("Unknown reward '%s' (module=%s)" % (name, module))
    if isinstance(kwargs, dict) and kwargs:
        return cls(**kwargs)
    return cls()


{target_initialization_code}
reward = _instantiate_reward_from_spec(_REWARD_SPEC)

# Action space
param_bounds = {effective_param_bounds}

# Execution parameters
seed = {effective_seed}
batch_size = {effective_batch_size}
n_shots = {effective_n_shots}
pauli_sampling = {effective_pauli_sampling}
n_reps = {effective_n_reps}
num_updates = TotalUpdates({total_updates_value})

backend_config = QMConfig(
    num_updates=num_updates.total_updates,
    input_type="{input_type_str}",
    verbosity={verbosity_val},
    timeout={timeout_val},
    path_to_python_wrapper=_PATH_TO_PY_WRAPPER,
)

execution_config = ExecutionConfig(
    batch_size=batch_size,
    sampling_paulis=pauli_sampling,
    n_shots=n_shots,
    n_reps=n_reps,
    seed=seed,
)

def create_action_space(param_bounds):
    param_bounds = np.array(param_bounds, dtype=np.float32)
    lower_bound, upper_bound = param_bounds.T
    return Box(low=lower_bound, high=upper_bound, shape=(len(param_bounds),), dtype=np.float32)


action_space = create_action_space(param_bounds)
q_env_config = QEnvConfig(
    target=target,
    backend_config=backend_config,
    action_space=action_space,
    execution_config=execution_config,
    reward=reward,
    benchmark_config=BenchmarkConfig(0),
)
q_env = QMEnvironment(q_env_config)

# Apply wrappers (outermost to innermost order as provided)
def _apply_wrappers(env, wrappers_spec):
    '''Apply an ordered list of wrappers to env.

    For wrappers subclassing ContextSamplingWrapper, a 'config' dict is accepted
    and converted to ContextSamplingWrapperConfig via from_dict. For all other
    wrappers, 'kwargs' are passed positionally after 'env' as standard keyword
    arguments.
    '''

    for spec in wrappers_spec[::-1]:
        name = spec.get("name")
        kwargs = spec.get("kwargs", {{}}) or {{}}
        module = spec.get("module", None)
        cls = _resolve_wrapper(name, module)
        env = cls(env, **kwargs)
    return env

env = _apply_wrappers(q_env, _WRAPPERS_SPEC)
ppo_config = PPOConfig.from_dict({ppo_config})

from rl_qoc.qua import CustomQMPPO
ppo_agent = CustomQMPPO(ppo_config, env)

ppo_training = TrainingConfig(num_updates)
ppo_settings = TrainFunctionSettings(plot_real_time=True, print_debug=True)

results = ppo_agent.train(ppo_training, ppo_settings)
results['hardware_runtime'] = q_env.hardware_runtime

print(json.dumps(results))
"""

    target_output_dir = output_dir if output_dir is not None else os.path.dirname(__file__)
    program_path = os.path.join(target_output_dir, "dgx_program.py")
    with open(program_path, "w") as f:
        f.write(script_code)
    return script_code


