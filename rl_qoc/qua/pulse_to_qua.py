from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Callable, Literal, Sequence, Optional, Dict, Type

from qiskit.circuit import ParameterExpression
from qiskit.pulse import (
    ShiftPhase,
    ShiftFrequency,
    SetFrequency,
    SetPhase,
    Delay,
    Play,
    Instruction,
)
from qm.qua import reset_phase, Math, amp as qua_amp
from quam.components import Channel
from qiskit.pulse.library.pulse import Pulse
from qm.qua.type_hints import Scalar
from quam.utils.qua_types import ScalarFloat, ScalarInt
from inspect import Parameter, Signature

__all__ = [
    "QuaPlayMacro",
    "QuaShiftPhaseMacro",
    "QuaShiftFrequencyMacro",
    "QuaSetFrequencyMacro",
    "QuaSetPhaseMacro",
    "QuaDelayMacro",
    "QuaPulseMacro",
    "get_amp_matrix",
    "frame_rotation_matrix",
    "qiskit_to_qua_instructions",
]

_ref_amp = 0.1  # Reference amplitude for the QUA compiler
_ref_phase = 0.0  # Reference phase for the QUA compiler

real_time_parameters_dict = {"amp": _ref_amp, "angle": _ref_phase}


def frame_rotation_matrix(angle, amp=1.0):
    return [
        amp * Math.cos(angle),
        -amp * Math.sin(angle),
        amp * Math.sin(angle),
        amp * Math.cos(angle),
    ]


def get_amp_matrix(amp=None, angle=None):
    new_ref = (
        1.0 / _ref_amp
    )  # Recover the reference amplitude from the reference value of 0.1
    if amp is not None and angle is not None:
        amp_matrix = qua_amp(*frame_rotation_matrix(angle, amp * new_ref))
    elif amp is not None:
        amp_matrix = qua_amp(new_ref * amp)
    elif angle is not None:
        amp_matrix = qua_amp(*frame_rotation_matrix(angle))
    else:
        amp_matrix = None
    return amp_matrix


@dataclass
class QuaPulseMacro(ABC):
    """
    Abstract class for QUA macros implementing Qiskit pulse instructions.
    """

    channel: Channel

    def apply(self, *args, **kwargs):
        """
        Apply the QUA macro to the channel.
        """

    @property
    @abstractmethod
    def macro(self) -> Callable:
        """
        Get the QUA macro function.
        """
        pass

    @property
    @abstractmethod
    def params(self):
        pass


@dataclass
class QuaPlayMacro(QuaPulseMacro):
    """
    QUA macro for Play instruction.
    """

    pulse: Pulse

    @property
    def macro(self) -> Callable:
        """
        Get the QUA macro function.
        """

        def play_pulse_macro(
            amp: Optional[ScalarInt | ScalarFloat] = None,
            angle: Optional[ScalarFloat] = None,
            duration: Optional[ScalarInt] = None,
            **kwargs
        ):
            self.channel.play(
                self.pulse.name,
                amplitude_scale=get_amp_matrix(amp, angle),
                duration=duration,
            )

        return play_pulse_macro

    @property
    def params(self):
        return ["pulse"]


@dataclass
class QuaShiftPhaseMacro(QuaPulseMacro):

    @property
    def macro(self) -> Callable:
        """
        Get the QUA macro function.
        """

        def shift_phase_macro(phase: ScalarFloat):
            self.channel.frame_rotation(phase)

        return shift_phase_macro

    @property
    def params(self):
        return ["phase"]


@dataclass
class QuaShiftFrequencyMacro(QuaPulseMacro):

    @property
    def macro(self) -> Callable:
        """
        Get the QUA macro function.
        """

        def shift_frequency_macro(frequency: ScalarFloat):
            self.channel.update_frequency(
                frequency + self.channel.intermediate_frequency
            )

        return shift_frequency_macro

    @property
    def params(self):
        return ["frequency"]


@dataclass
class QuaSetFrequencyMacro(QuaPulseMacro):

    @property
    def macro(self) -> Callable:
        """
        Get the QUA macro function.
        """

        def set_frequency_macro(frequency: ScalarFloat):
            self.channel.update_frequency(frequency)

        return set_frequency_macro

    @property
    def params(self):
        return ["frequency"]


@dataclass
class QuaSetPhaseMacro(QuaPulseMacro):

    @property
    def macro(self) -> Callable:
        """
        Get the QUA macro function.
        """

        def set_phase_macro(phase: ScalarFloat):
            reset_phase(self.channel.name)
            self.channel.reset_if_phase()
            self.channel.frame_rotation(phase)

        return set_phase_macro

    @property
    def params(self):
        return ["phase"]


@dataclass
class QuaDelayMacro(QuaPulseMacro):

    @property
    def macro(self) -> Callable:
        """
        Get the QUA macro function.
        """

        def delay_macro(duration: ScalarInt):
            self.channel.wait(duration)

        return delay_macro

    @property
    def params(self):
        return ["duration"]


qiskit_to_qua_instructions: Dict[Type[Instruction], Type[QuaPulseMacro]] = {
    ShiftPhase: QuaShiftPhaseMacro,
    ShiftFrequency: QuaShiftFrequencyMacro,
    SetFrequency: QuaSetFrequencyMacro,
    SetPhase: QuaSetPhaseMacro,
    Delay: QuaDelayMacro,
    Play: QuaPlayMacro,
}
