from gymnasium.wrappers import TransformAction
from gymnasium.utils import RecordConstructorArgs
from gymnasium.spaces import Box
from gymnasium import Wrapper, Env
from gymnasium.core import ActType, ObsType, WrapperActType
import numpy as np
from gymnasium.wrappers.utils import rescale_box


class RescaleAndClipAction(
    TransformAction[ObsType, WrapperActType, ActType], RecordConstructorArgs
):
    """
    A wrapper that rescales and clips the actions from the agent to the range `[min_action, max_action]`
    and clips them to the bounds of the environment's original action space.

    The base environment must have an action space of type `spaces.Box`.
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        min_action: np.floating | np.integer | np.ndarray | float,
        max_action: np.floating | np.integer | np.ndarray | float,
    ):
        """
        Initializes the RescaleAndClipAction wrapper.

        Args:
            env: The environment to wrap.
            min_action: The minimum action value in the environment.
            max_action: The maximum action value in the environment.
        """
        assert isinstance(env.action_space, Box)
        RecordConstructorArgs.__init__(
            self, min_action=min_action, max_action=max_action
        )

        def rescale_and_clip(action: np.ndarray) -> np.ndarray:
            # Rescale the action to the desired range
            _, _, rescale_func = rescale_box(env.action_space, min_action, max_action)
            action = rescale_func(action)
            # Clip the action to the bounds of the original action space
            return np.clip(action, env.action_space.low, env.action_space.high)

        TransformAction.__init__(
            self, env=env, func=rescale_and_clip, action_space=env.action_space
        )
