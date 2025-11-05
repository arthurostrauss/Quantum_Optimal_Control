from typing import List, Optional, Union, Dict, Any, Tuple, cast
from dataclasses import asdict
import numpy as np
import os

from rl_qoc.qua import QMConfig
from rl_qoc import GateTarget, StateTarget

from .sync_hook_specs import (
    WrapperSpec,
    RewardSpec,
    coerce_wrappers,
    coerce_reward,
)


def _infer_specs_from_env(env: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Infer wrapper and reward specs from an already created environment.

    This walks the wrapper chain (outermost to innermost) to build an ordered
    wrappers list, and extracts the reward class name. For ContextSamplingWrapper
    instances, it serializes the context_config dataclass. For RescaleAndClipAction,
    it tries to recover min_action/max_action from env.unwrapped.config.backend_config
    (wrapper_data['rescale_and_clip'].low/high).

    Returns (wrappers_list, reward_spec_dict)
    """
    wrappers_list: list[dict[str, Any]] = []

    # Reward: use class name, kwargs unknown (default to empty)
    try:
        reward_obj = env.unwrapped.config.reward
        reward_spec = {"name": reward_obj.__class__.__name__, "kwargs": {}}
    except Exception:
        reward_spec = {"name": "FidelityReward", "kwargs": {}}  # safe default

    # Walk wrapper chain
    cur = env
    # Access backend wrapper data from the base env if available
    backend_wrapper_box = None
    try:
        backend_wrapper_box = env.unwrapped.config.backend_config.wrapper_data.get("rescale_and_clip", None)
    except Exception:
        backend_wrapper_box = None

    visited = set()
    while hasattr(cur, "env") and id(cur) not in visited:
        visited.add(id(cur))
        cls_obj = cur.__class__
        cls_name = cls_obj.__name__
        cls_module = getattr(cls_obj, "__module__", None)
        saved_kwargs = getattr(cur, "_saved_kwargs", None)
        wrapper_spec_else: Dict[str, Any] = {"name": cls_name}
        if cls_module:
            wrapper_spec_else["module"] = cls_module
        if isinstance(saved_kwargs, dict) and saved_kwargs:
            wrapper_spec_else["kwargs"] = cast(Dict[str, Any], saved_kwargs)
        wrappers_list.append(wrapper_spec_else)
        cur = cur.env

    return wrappers_list, reward_spec


def generate_sync_hook(
    env: Any,
    ppo_config: Dict[str, Any],
    output_dir: Optional[str] = None,
    wrappers: Optional[Union[List[WrapperSpec], List[Dict[str, Any]]]] = None,
    reward_spec: Optional[Union[RewardSpec, Dict[str, Any]]] = None,
) -> str:
    """
    Generate a sync_hook.py that constructs and trains a PPO agent on a QUA-based
    environment. This function writes a self-contained Python file that will run
    in the cloud runtime.

    Minimal usage:
    - generate_sync_hook(env=q_env, ppo_config=ppo_cfg)

    Optional overrides:
    - wrappers: ordered list of wrapper specs; if omitted, inferred from env.
    - reward_spec: reward instantiation; if omitted, inferred from env.
    - output_dir: where to write sync_hook.py; defaults to the iqcc folder.

    All other environment parameters (target, action bounds, execution settings,
    backend settings, input type, total updates) are inferred from `env`.
    """
    if env is None:
        raise ValueError("env must be provided to infer configuration")

    # Unwrap env for config access
    try:
        qenv = env.unwrapped
    except Exception:
        qenv = env

    # Target
    try:
        effective_target = qenv.config.target
    except Exception as e:
        raise ValueError("Unable to infer target from env.config.target") from e

    # Target initialization code
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
    try:
        aspace = getattr(qenv, "action_space", None) or getattr(qenv.config, "action_space", None)
        if aspace is None:
            raise ValueError("action_space is None on env or env.config")
        low = np.array(aspace.low).flatten()
        high = np.array(aspace.high).flatten()
        effective_param_bounds = list(map(lambda x: (float(x[0]), float(x[1])), zip(low, high)))
    except Exception as e:
        raise ValueError("Unable to infer action bounds from env.action_space") from e

    # Execution settings
    try:
        exec_cfg = qenv.config.execution_config
        effective_seed = exec_cfg.seed
        effective_batch_size = exec_cfg.batch_size
        effective_n_shots = exec_cfg.n_shots
        effective_pauli_sampling = exec_cfg.sampling_paulis
        effective_n_reps = exec_cfg.n_reps
    except Exception as e:
        raise ValueError("Unable to infer execution settings from env.config.execution_config") from e

    # Backend settings and total updates
    try:
        bcfg = qenv.config.backend_config
        effective_input_type = bcfg.input_type
        verbosity_val = getattr(bcfg, "verbosity", 1)
        timeout_val = getattr(bcfg, "timeout", 60)
        total_updates_value = int(getattr(bcfg, "num_updates", 0))
        if total_updates_value <= 0:
            raise ValueError
    except Exception as e:
        raise ValueError("Unable to infer backend settings or num_updates from env.config.backend_config") from e

    # Reward spec (capture module and kwargs via reward.reward_args when available)
    reward_spec_dict = coerce_reward(reward_spec) if reward_spec is not None else None
    if reward_spec_dict is None:
        try:
            reward_obj = qenv.config.reward
            reward_cls = reward_obj.__class__
            reward_kwargs = getattr(reward_obj, "reward_args", {}) or {}
            reward_spec_dict = {
                "name": reward_cls.__name__,
                "module": getattr(reward_cls, "__module__", None),
            }
            if isinstance(reward_kwargs, dict) and reward_kwargs:
                reward_spec_dict["kwargs"] = reward_kwargs
        except Exception as e:
            raise ValueError("Unable to infer reward from env.config.reward; pass reward_spec explicitly") from e

    # Wrappers: infer if not provided
    wrappers_list = coerce_wrappers(wrappers) if wrappers is not None else None
    if wrappers_list is None:
        try:
            wrappers_list, _ = _infer_specs_from_env(env)
        except Exception as e:
            wrappers_list = []  # fallback to default RescaleAndClipAction in generated code

    # Prepare embedded literals
    wrappers_literal = repr(wrappers_list)
    reward_spec_literal = repr(reward_spec_dict)
    input_type_str = getattr(effective_input_type, "name", str(effective_input_type))

    # Build generated sync hook content
    sync_hook_code = f"""# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
# This file was automatically generated by generate_sync_hook

import json
import importlib
import rl_qoc.environment.wrappers as _custom_wrappers
import gymnasium.wrappers as _gym_wrappers
import rl_qoc.rewards as _rewards_mod
from rl_qoc.agent import PPOConfig, TrainingConfig, TrainFunctionSettings, TotalUpdates
from rl_qoc.qua import QMEnvironment, CustomQMPPO, QMConfig
from iqcc_cloud_client.runtime import get_qm_job
from rl_qoc import (
    RescaleAndClipAction,
    GateTarget,
    StateTarget,
    ExecutionConfig,
    QEnvConfig,
    BenchmarkConfig,
    ContextSamplingWrapper,
    ContextSamplingWrapperConfig,
)
import numpy as np
from gymnasium.spaces import Box

# Embedded construction specs
_WRAPPERS_SPEC = {wrappers_literal}
_REWARD_SPEC = {reward_spec_literal}


def _resolve_wrapper(name: str, module: str | None = None):
    '''Resolve a wrapper class by name.
    Order: explicit module > rl_qoc.environment.wrappers > gymnasium.wrappers.'''
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
    # Resolve class from explicit module > rl_qoc.rewards > reward_dict
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
    # Only pass kwargs if present and non-empty; some rewards have no-arg ctors
    if isinstance(kwargs, dict) and kwargs:
        return cls(**kwargs)
    return cls()


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


job = get_qm_job()
{target_initialization_code}

# Instantiate reward from embedded spec. The spec contains the class name,
# an optional module path, and the constructor kwargs captured from the original
# environment via reward.reward_args when available.
reward = _instantiate_reward_from_spec(_REWARD_SPEC)

# Action space specification
param_bounds = {effective_param_bounds}  # Can be any number of bounds

# Environment execution parameters
seed = {effective_seed}  # Master seed to make training reproducible
batch_size = {effective_batch_size}  # Number of actions to evaluate per policy evaluation
n_shots = {effective_n_shots}  # Minimum number of shots per fiducial evaluation
pauli_sampling = {effective_pauli_sampling}  # Number of fiducials to compute for fidelity estimation (DFE only)
n_reps = {effective_n_reps}  # Number of repetitions of the cycle circuit
num_updates = TotalUpdates({total_updates_value})
backend_config = QMConfig(
    num_updates=num_updates.total_updates,
    input_type="{input_type_str}",
    verbosity={verbosity_val},
    timeout={timeout_val}
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
    benchmark_config=BenchmarkConfig(0),  # No benchmark for now)
)
q_env = QMEnvironment(q_env_config, job=job)
# Apply wrapper stack (or default if empty)
env = _apply_wrappers(q_env, _WRAPPERS_SPEC)
ppo_config = PPOConfig.from_dict({ppo_config})

ppo_agent = CustomQMPPO(ppo_config, env)

ppo_training = TrainingConfig(num_updates)
ppo_settings = TrainFunctionSettings(plot_real_time=True, print_debug=True)

results = ppo_agent.train(ppo_training, ppo_settings)
results['hardware_runtime'] = q_env.hardware_runtime

print(json.dumps(results))
"""

    target_output_dir = output_dir if output_dir is not None else os.path.dirname(__file__)
    sync_hook_path = os.path.join(target_output_dir, "sync_hook.py")
    with open(sync_hook_path, "w") as f:
        f.write(sync_hook_code)

    return sync_hook_path

