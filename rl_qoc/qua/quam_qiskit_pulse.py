from copy import deepcopy

import numpy as np
from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import PulseChannel as QiskitChannel
from qiskit.pulse import SymbolicPulse, Waveform, Play, Schedule, PulseError, Constant
from typing import List, Tuple, Union, Optional
from .pulse_to_qua import real_time_parameters_dict
from .sympy_to_qua import sympy_to_qua
from quam.components.pulses import Pulse as QuAMPulse
from quam import quam_dataclass
from .pulse_support_utils import _real_time_parameters


def register_pulse_features(pulse: SymbolicPulse):
    """
    Register the features of a SymbolicPulse to be used in the QUA compiler
    """
    stored_parameter_expressions = {}
    for param_name, param_value in pulse.parameters.items():
        if param_name in _real_time_parameters and isinstance(
            param_value, ParameterExpression
        ):
            stored_parameter_expressions[param_name] = lambda val: sympy_to_qua(
                param_value.sympify(), val
            )
    return stored_parameter_expressions


def return_samples_output(pulse: SymbolicPulse):
    """
    Return the samples of a SymbolicPulse to be used in the QUA compiler
    """
    if np.abs(pulse.angle) < 1e-10:
        return (
            pulse.amp
            if isinstance(pulse, Constant)
            else pulse.get_waveform().samples.real
        )
    else:
        return (
            pulse.amp * np.exp(1j * pulse.angle)
            if isinstance(pulse, Constant)
            else pulse.get_waveform().samples
        )


class FluxChannel(QiskitChannel):
    prefix = "f"


@quam_dataclass
class QuAMQiskitPulse(QuAMPulse):
    # pulse: Union[SymbolicPulse, Waveform]
    # length: Optional[int] = None
    # id: Optional[str] = None

    def __init__(self, pulse: Union[SymbolicPulse, Waveform]):
        self.pulse = pulse
        self._stored_parameter_expressions = {
            "amp": None,
            "angle": None,
            "frequency": None,
        }

        self._stored_parameter_expressions = register_pulse_features(pulse)

        super().__init__(
            length=(
                self.pulse.duration
                if not isinstance(self.pulse.duration, ParameterExpression)
                else 0
            ),
            id=pulse.name,
        )

    # def __post_init__(self):
    #     self._stored_parameter_expressions = register_pulse_features(self.pulse)
    #     self.length = (
    #         self.pulse.duration if not isinstance(self.pulse.duration, ParameterExpression) else 0
    #     )
    #     self.id = self.pulse.name
    #     super().__post__init()

    def waveform_function(
        self,
    ) -> Union[
        float,
        complex,
        List[float],
        List[complex],
        Tuple[float, float],
        Tuple[List[float], List[float]],
    ]:
        if isinstance(self.pulse, Waveform):
            return self.pulse.samples.tolist()

        if isinstance(self.pulse, SymbolicPulse) and self.pulse.is_parameterized():
            new_pulse_parameters = {
                param_name: (
                    real_time_parameters_dict[param_name]
                    if param_name in real_time_parameters_dict
                    and isinstance(param_value, ParameterExpression)
                    else param_value
                )
                for param_name, param_value in self.pulse.parameters.items()
            }

            if "duration" in new_pulse_parameters and isinstance(
                new_pulse_parameters["duration"], ParameterExpression
            ):
                raise NotImplementedError(
                    "Duration parameter cannot be parametrized (currently not supported)"
                )

            pulse = deepcopy(self.pulse)
            pulse._params.update(new_pulse_parameters)

            return return_samples_output(pulse)

        try:
            return_samples_output(self.pulse)
        except (AttributeError, PulseError) as e:
            raise PulseError(
                "Pulse waveform could not be retrieved from the given pulse"
            ) from e

    def is_parameterized(self):
        return self.pulse.is_parameterized()

    def is_compile_time_parameterized(self):
        """
        Check if the pulse is parametrized with compile-time parameters
        """
        return any(
            [
                isinstance(self.pulse.parameters[param], ParameterExpression)
                and param not in _real_time_parameters
                for param in self.pulse.parameters
            ]
        )

    def is_real_time_parametrized(self):
        """
        Check if the pulse is parametrized with real-time parameters
        """
        return any(
            isinstance(self.pulse.parameters[param], ParameterExpression)
            for param in _real_time_parameters
        )

    @property
    def parameters(self):
        return self.pulse.parameters

    @property
    def stored_parameter_expressions(self):
        return self._stored_parameter_expressions
