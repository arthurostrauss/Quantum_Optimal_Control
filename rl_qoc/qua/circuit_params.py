from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_qm_provider import ParameterTable, Parameter as QuaParameter, InputType, Direction

from rl_qoc import QEnvConfig


@dataclass
class CircuitParams:
    input_state_vars: Optional[ParameterTable] = None
    observable_vars: Optional[ParameterTable] = None
    n_reps_var: Optional[QuaParameter] = None
    circuit_choice_var: Optional[QuaParameter] = None
    n_shots: Optional[QuaParameter] = None
    max_input_state: Optional[QuaParameter] = None
    max_observables: Optional[QuaParameter] = None
    real_time_circuit_parameters: Optional[ParameterTable] = None

    @classmethod
    def from_circuit(
        cls, qc: QuantumCircuit, input_type: InputType, config: QEnvConfig
    ) -> "CircuitParams":
        input_state_vars = ParameterTable.from_qiskit(
            qc,
            input_type=input_type,
            filter_function=lambda x: "input" in x.name,
            name="input_state_vars",
        )
        observable_vars = ParameterTable.from_qiskit(
            qc,
            input_type=input_type,
            filter_function=lambda x: "observable" in x.name,
            name="observable_vars",
        )
        n_reps_var = (
            QuaParameter(
                "n_reps",
                config.current_n_reps,
                input_type=input_type,
                direction=Direction.OUTGOING,
            )
            if qc.has_var("n_reps")
            else None
        )
        circuit_choice_var = (
            QuaParameter(
                "circuit_choice",
                0,
                input_type=input_type,
                direction=Direction.OUTGOING,
            )
            if qc.has_var("circuit_choice")
            else None
        )
        n_shots = QuaParameter(
            "pauli_shots",
            config.n_shots,
            input_type=input_type if config.dfe else None,
            direction=Direction.OUTGOING if config.dfe else None,
        )
        max_input_state = (
            QuaParameter(
                "max_input_state",
                1,
                input_type=input_type,
                direction=Direction.OUTGOING,
            )
            if config.reward_method in ["channel", "cafe"]
            else None
        )
        max_observables = (
            QuaParameter(
                "max_observables",
                0,
                input_type=input_type,
                direction=Direction.OUTGOING,
            )
            if config.dfe
            else None
        )
        real_time_circuit_parameters = ParameterTable.from_qiskit(
            qc,
            input_type=None,
            filter_function=lambda x: isinstance(x, Parameter),
            name="real_time_circuit_parameters",
        )
        return cls(
            input_state_vars=input_state_vars,
            observable_vars=observable_vars,
            n_reps_var=n_reps_var,
            circuit_choice_var=circuit_choice_var,
            n_shots=n_shots,
            max_input_state=max_input_state,
            max_observables=max_observables,
            real_time_circuit_parameters=real_time_circuit_parameters,
        )

    def reset(self):
        for param in self.all_parameters:
            param.reset()

    def declare_variables(self):
        for param in self.all_parameters:
            if isinstance(param, QuaParameter):
                param.declare_variable()
            elif isinstance(param, ParameterTable):
                param.declare_variables()

    def declare_streams(self):
        for param in self.all_parameters:
            if isinstance(param, QuaParameter):
                param.declare_stream()
            elif isinstance(param, ParameterTable):
                param.declare_streams()

    def stream_processing(self):
        for param in self.all_parameters:
            if isinstance(param, QuaParameter) and param.stream is not None:
                param.stream_processing()
            elif isinstance(param, ParameterTable):
                param.stream_processing()

    @property
    def all_parameters(self) -> List[QuaParameter | ParameterTable]:
        """
        Return all the parameters of the circuit params that are not None
        """
        return [
            param
            for param in [
                self.input_state_vars,
                self.observable_vars,
                self.n_reps_var,
                self.circuit_choice_var,
                self.n_shots,
                self.max_input_state,
                self.max_observables,
                self.real_time_circuit_parameters,
            ]
            if param is not None
        ]

    @property
    def circuit_variables(self) -> List[QuaParameter | ParameterTable]:
        """
        Return all the parameters of the circuit embedded variables and parameters that are not None
        """
        return [
            param
            for param in [
                self.input_state_vars,
                self.observable_vars,
                self.real_time_circuit_parameters,
                self.n_reps_var,
                self.circuit_choice_var,
            ]
            if param is not None
        ]
