from .base_reward import Reward
from .cafe import CAFEReward, CAFERewardData, CAFERewardDataList
from .channel import ChannelReward, ChannelRewardData, ChannelRewardDataList
from .fidelity import FidelityReward, FidelityRewardData, FidelityRewardDataList
from .orbit import ORBITReward, ORBITRewardData, ORBITRewardDataList
from .state import StateReward, StateRewardData, StateRewardDataList
from .xeb import XEBReward, XEBRewardData, XEBRewardDataList
from typing import Dict, Literal, Type

reward_dict: Dict[str, Type[Reward]] = {
    "fidelity": FidelityReward,
    "channel": ChannelReward,
    "state": StateReward,
    "xeb": XEBReward,
    "cafe": CAFEReward,
    "orbit": ORBITReward,
    "shadow": FidelityReward,  # Placeholder for shadow tomography reward
}
REWARD_STRINGS = Literal["cafe", "channel", "orbit", "state", "xeb", "fidelity", "shadow"]
