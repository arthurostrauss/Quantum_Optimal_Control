from typing import Union, Any, Tuple, Optional
from quam.components.channels import IQChannel
from quam.components.macro import QubitMacro, QubitPairMacro, PulseMacro
from quam.components.pulses import ReadoutPulse, Pulse
from quam.core import quam_dataclass
from quam.utils import string_reference as str_ref
from rl_qoc.qua.quam_components import Transmon, ReadoutResonator
from qm.qua import declare, assign, while_, Cast, broadcast, fixed
from quam.utils.qua_types import QuaVariableBool, QuaVariableFloat, QuaVariableInt

__all__ = ["MeasureMacro", "ResetMacro", "VirtualZMacro"]


def get_pulse_name(pulse: Pulse) -> str:
    """
    Get the name of the pulse. If the pulse has an id, return it.
    """
    if pulse.id is not None:
        return pulse.id
    elif pulse.parent is not None:
        return pulse.parent.get_attr_name(pulse)
    else:
        raise AttributeError(
            f"Cannot infer id of {pulse} because it is not attached to a parent"
        )


@quam_dataclass
class MeasureMacro(QubitMacro):
    pulse: Union[ReadoutPulse, str] = "readout"

    def apply(self, **kwargs) -> Any:
        state = kwargs.get("state", declare(int))
        qua_vars = kwargs.get("qua_vars", (declare(fixed), declare(fixed)))
        pulse: ReadoutPulse = (
            self.pulse
            if isinstance(self.pulse, Pulse)
            else self.qubit.get_pulse(self.pulse)
        )

        resonator: ReadoutResonator = self.qubit.resonator
        resonator.measure(get_pulse_name(pulse), qua_vars=qua_vars)
        I, Q = qua_vars
        assign(state, Cast.to_int(I > pulse.threshold))
        return state


@quam_dataclass
class ResetMacro(QubitMacro):
    pi_pulse: Union[Pulse, str] = "x"
    readout_pulse: Union[ReadoutPulse, str] = "measure"
    max_attempts: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.max_attempts > 0, "max_attempts must be greater than 0"

    def apply(self, **kwargs) -> Any:
        I, Q = declare(fixed), declare(fixed)
        state = declare(bool)
        attempts = declare(int, value=1)
        assign(attempts, 1)
        resonator: ReadoutResonator = self.qubit.resonator
        xy: IQChannel = self.qubit.xy
        self.qubit.align()
        readout_pulse: ReadoutPulse = (
            self.readout_pulse
            if isinstance(self.readout_pulse, ReadoutPulse)
            else self.qubit.get_pulse(self.readout_pulse)
        )
        pi_pulse: Pulse = (
            self.pi_pulse
            if isinstance(self.pi_pulse, Pulse)
            else self.qubit.get_pulse(self.pi_pulse)
        )
        resonator.measure(get_pulse_name(readout_pulse), qua_vars=(I, Q))
        assign(state, I > readout_pulse.threshold)
        resonator.wait(resonator.depletion_time // 4)
        self.qubit.align()
        with while_(
            broadcast.and_(
                (I > readout_pulse.rus_exit_threshold),
                (attempts < self.max_attempts),
            )
        ):
            self.qubit.align()
            resonator.measure(get_pulse_name(readout_pulse), qua_vars=(I, Q))
            assign(state, I > readout_pulse.threshold)
            resonator.wait(resonator.depletion_time // 4)
            self.qubit.align()
            xy.play(get_pulse_name(pi_pulse), condition=state)
            self.qubit.align()
            assign(attempts, attempts + 1)

        self.qubit.align()


@quam_dataclass
class VirtualZMacro(QubitMacro):
    def apply(self, angle: float):
        self.qubit.xy.frame_rotation_2pi(angle)


@quam_dataclass
class CZMacro(QubitPairMacro):
    flux_pulse_control: Union[Pulse, str]
    coupler_flux_pulse: Pulse = None

    pre_wait: int = 4

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0

    @property
    def flux_pulse_control_label(self) -> str:
        pulse = (
            self.qubit_control.get_pulse(self.flux_pulse_control)
            if isinstance(self.flux_pulse_control, str)
            else self.flux_pulse_control
        )
        return get_pulse_name(pulse)

    @property
    def coupler_flux_pulse_label(self) -> str:
        pulse = (
            self.coupler.get_pulse(self.coupler_flux_pulse)
            if isinstance(self.coupler_flux_pulse, str)
            else self.coupler_flux_pulse
        )
        return get_pulse_name(pulse)

    def apply(
        self,
        *,
        amplitude_scale=None,
        phase_shift_control=None,
        phase_shift_target=None,
        **kwargs,
    ):
        self.qubit_control.z.play(
            self.flux_pulse_control_label,
            validate=False,
            amplitude_scale=amplitude_scale,
        )

        if self.coupler_flux_pulse is not None:
            self.coupler.play(self.coupler_flux_pulse_label, validate=False)

        self.transmon_pair.align()
        self.qubit_control.xy.frame_rotation_2pi(
            phase_shift_control
            if phase_shift_control is not None
            else self.phase_shift_control
        )
        self.qubit_target.xy.frame_rotation_2pi(
            phase_shift_target
            if phase_shift_target is not None
            else self.phase_shift_target
        )
        self.qubit_control.xy.play("x180", amplitude_scale=0.0, duration=4)
        self.qubit_target.xy.play("x180", amplitude_scale=0.0, duration=4)
        self.transmon_pair.align()
