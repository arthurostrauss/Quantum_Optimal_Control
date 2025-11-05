from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, List

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
    context_parameters: Optional[List[Optional[ParameterTable]]] = None
    benchmark_cycle_var: Optional[QuaParameter] = None

    @classmethod
    def from_circuit(
        cls,
        qc: QuantumCircuit,
        input_type: InputType,
        config: QEnvConfig,
        context_parameters: Sequence[Sequence[Parameter]] = (),
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
        benchmark_cycle_var = (
            QuaParameter(
                "benchmark_cycle",
                False,
                input_type=input_type,
                direction=Direction.OUTGOING,
            )
            if config.benchmark_cycle > 0 and config.check_on_exp
            else None
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
            filter_function=lambda x: isinstance(x, Parameter) and x not in context_parameters,
            name="real_time_circuit_parameters",
        )
        context_parameters_table = [
            ParameterTable.from_qiskit(
                qc,
                input_type=None,
                filter_function=lambda x: isinstance(x, Parameter) and x in context_parameters[i],
                name=f"context_parameters_{i}",
            )
            for i in range(len(context_parameters))
        ]

        return cls(
            input_state_vars=input_state_vars,
            observable_vars=observable_vars,
            n_reps_var=n_reps_var,
            circuit_choice_var=circuit_choice_var,
            n_shots=n_shots,
            max_input_state=max_input_state,
            max_observables=max_observables,
            real_time_circuit_parameters=real_time_circuit_parameters,
            context_parameters=context_parameters_table,
            benchmark_cycle_var=benchmark_cycle_var,
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

    def stream_processing(
        self,
        mode: Literal["save", "save_all"] = "save_all",
        buffer: int | Tuple[int, ...] | None | Literal["default"] = "default",
    ):
        for param in self.all_parameters:
            if isinstance(param, QuaParameter) and param.stream is not None:
                param.stream_processing(mode=mode, buffer=buffer)
            elif isinstance(param, ParameterTable):
                param.stream_processing(mode=mode, buffering=buffer)

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

    @property
    def fiducials_variables(self) -> List[QuaParameter]:
        """
        Return all the parameters that are not directly used in the circuit but which structure the QUA program
        execution.
        """
        return [
            param
            for param in [
                self.n_shots,
                self.max_input_state,
                self.max_observables,
            ]
            if param is not None
        ]

    def get_by_name(self, name: str) -> QuaParameter | ParameterTable:
        """
        Return the parameter with the given name
        """
        for param in self.all_parameters:
            if param.name == name:
                return param
        raise ValueError(f"Parameter with name {name} not found")
