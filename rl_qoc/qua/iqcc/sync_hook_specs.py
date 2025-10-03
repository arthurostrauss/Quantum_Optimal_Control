from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union


@dataclass
class WrapperSpec:
    """Schema describing a single Gymnasium wrapper to layer onto the base env.

    Fields
    - name: Class name of the wrapper. This may exist in either
      rl_qoc.environment.wrappers or gymnasium.wrappers, or another module if
      accompanied by the optional 'module' field.
      Examples: "RescaleAndClipAction", "ParametricGateContextWrapper",
      "TransformObservation".
    - kwargs: Optional constructor keyword arguments besides the 'env' argument.
      For RescaleAndClipAction, use {"min_action": float, "max_action": float}.
    - config: Optional configuration dictionary for wrappers that expect a single
      config object as the second positional argument. Any subclass of
      ContextSamplingWrapper (e.g., ParametricGateContextWrapper) supports this
      via ContextSamplingWrapperConfig.from_dict.
    - module: Optional string module path for the wrapper (e.g.,
      "gymnasium.wrappers"). If omitted, the generator will try to resolve the
      name in rl_qoc.environment.wrappers first, then gymnasium.wrappers.

    Usage
    - Provide an ordered list of WrapperSpec to `generate_sync_hook` to define a
      wrapper stack, applied in the provided order. If no wrappers are provided,
      a default RescaleAndClipAction(-1, 1) is used for backward compatibility.
    """

    name: str
    kwargs: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    module: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class RewardSpec:
    """Schema describing how to instantiate a reward.

    Fields
    - name: Class name of the reward. This may exist in rl_qoc.rewards or in an
      external module if accompanied by the optional 'module' field.
    - kwargs: Optional constructor keyword arguments for the reward.
    - module: Optional string module path for the reward (e.g.,
      "my_project.rewards"). If omitted, the generator will try to resolve the
      name in rl_qoc.rewards first.

    Usage
    - Pass a RewardSpec to `generate_sync_hook` to instantiate a non-default
      reward (or a reward that requires constructor arguments).
    - If omitted, the generator will use the provided `reward` instance's class
      name with a no-argument constructor inside the generated file.
    """

    name: str
    kwargs: Optional[Dict[str, Any]] = None
    module: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def coerce_wrappers(
    wrappers: Optional[Union[List[WrapperSpec], List[Dict[str, Any]]]]
) -> Optional[List[Dict[str, Any]]]:
    """Normalize wrapper specifications into plain dictionaries.

    Accepts either WrapperSpec dataclasses or plain dict entries. Validates that
    each entry has a string 'name' and strips out keys with None values for compactness.
    Returns None if the input is None.
    """
    if wrappers is None:
        return None
    out: List[Dict[str, Any]] = []
    for w in wrappers:
        if isinstance(w, WrapperSpec):
            out.append(w.to_dict())
        elif isinstance(w, dict):
            if "name" not in w or not isinstance(w["name"], str):
                raise ValueError("Each wrapper spec must include a string 'name' field")
            out.append({k: v for k, v in w.items() if v is not None})
        else:
            raise TypeError("Wrapper specs must be WrapperSpec or dict")
    return out


def coerce_reward(
    reward_spec: Optional[Union[RewardSpec, Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Normalize reward specification into a plain dictionary or None.

    Accepts either RewardSpec dataclasses or plain dict entries. Validates that
    the 'name' field exists and is a string. Returns None if the input is None.
    """
    if reward_spec is None:
        return None
    if isinstance(reward_spec, RewardSpec):
        return reward_spec.to_dict()
    if isinstance(reward_spec, dict):
        if "name" not in reward_spec or not isinstance(reward_spec["name"], str):
            raise ValueError("Reward spec must include a string 'name' field")
        return {k: v for k, v in reward_spec.items() if v is not None}
    raise TypeError("Reward spec must be RewardSpec or dict")
