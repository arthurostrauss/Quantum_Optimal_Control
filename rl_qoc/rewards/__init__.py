from .base_reward import Reward
from .cafe_reward import CAFEReward
from .channel_reward import ChannelReward
from .fidelity_reward import FidelityReward
from .orbit_reward import ORBITReward
from .state_reward import StateReward
from .xeb_reward import XEBReward

reward_dict = {
    "fidelity": FidelityReward,
    "channel": ChannelReward,
    "state": StateReward,
    "xeb": XEBReward,
    "cafe": CAFEReward,
    "orbit": ORBITReward,
}
