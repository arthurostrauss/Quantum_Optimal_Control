from __future__ import annotations

from typing import Iterable, List, Dict, Optional, Callable, Union, Tuple, Any

import numpy as np
from quam.components import Channel as QuAMChannel, Qubit, QubitPair
from quam.components import BasicQuam as QuAM

from qiskit.circuit import (
    QuantumCircuit,
    SwitchCaseOp,
    ForLoopOp,
    IfElseOp,
    WhileLoopOp,
    Gate,
)
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping,
)
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.providers import BackendV2 as Backend, QubitProperties, Options
from qiskit.pulse import (
    ScheduleBlock,
    Schedule,
    DriveChannel,
    MeasureChannel,
    AcquireChannel,
    Play,
)
from qiskit.pulse.library.pulse import Pulse as QiskitPulse

from qiskit.transpiler import Target, InstructionProperties
from qiskit.qasm3 import Exporter
from qiskit.pulse.channels import Channel as QiskitChannel, ControlChannel
from qiskit.pulse.library import SymbolicPulse
from qiskit.pulse.library.waveform import Waveform
from qm.qua import switch_, case_, program
from qm.qua import declare, fixed, declare_stream
from qm import QuantumMachinesManager, Program, DictQuaConfig
from qualang_tools.addons.variables import assign_variables_to_element

from .parameter_table import ParameterTable, InputType

from .pulse_support_utils import (
    _instruction_to_qua,
    validate_parameters,
    validate_schedule,
    handle_parameterized_channel,
)
from .quam_qiskit_pulse import QuAMQiskitPulse, FluxChannel
from inspect import Signature, Parameter as sigParam
from oqc import (
    Compiler,
    HardwareConfig,
    OperationIdentifier,
    QubitsMapping,
    CompilationResult,
)

__all__ = [
    "QMBackend",
    "QMProvider",
    "QMInstructionProperties",
    "validate_machine",
    "look_for_standard_op",
]
RunInput = Union[QuantumCircuit, Schedule, ScheduleBlock]


class QMInstructionProperties(InstructionProperties):
    def __init__(
        self,
        duration: float | None = None,
        error: float | None = None,
        qua_pulse_macro: Callable | None = None,
    ):
        super().__init__(duration=duration, error=error)
        self._qua_pulse_macro = qua_pulse_macro

    @property
    def qua_pulse_macro(self) -> Callable | None:
        return self._qua_pulse_macro

    @qua_pulse_macro.setter
    def qua_pulse_macro(self, value: Callable | None):
        self._qua_pulse_macro = value

    def __repr__(self):
        return (
            f"QMInstructionProperties(duration={self.duration}, "
            f"error={self.error}, "
            f"qua_pulse_macro={self.qua_pulse_macro})"
        )

    def __getstate__(self):
        return (super().__getstate__(), self.qua_pulse_macro)

    def __setstate__(self, state: tuple):
        super().__setstate__(state[0])
        self.qua_pulse_macro = state[1]


class QMProvider:
    def __init__(self, qmm: QuantumMachinesManager):
        """
        Qiskit Provider for the Quantum Orchestration Platform (QOP)
        Args:
            host: The host of the QOP
            port: The port of the QOP
            cluster_name: The name of the cluster
            octave_config: The octave configuration
        """
        super().__init__(self)
        self.qmm = qmm

    def get_backend(
        self, machine: QuAM, channel_mapping: Optional[Dict[QiskitChannel, QuAMChannel]]
    ):
        return QMBackend(machine, channel_mapping)

    def backends(self, name=None, filters=None, **kwargs):
        raise NotImplementedError("Not implemented yet")

    def __str__(self):
        return f"QMProvider({self.qmm})"

    def __repr__(self):
        return f"QMProvider({self.qmm})"


def validate_machine(machine) -> QuAM:
    if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
        raise ValueError(
            "Invalid QuAM instance provided, should have qubits and qubit_pairs attributes"
        )
    if not all(isinstance(qubit, Qubit) for qubit in machine.qubits.values()):
        raise ValueError("All qubits should be of type Qubit")
    if not all(isinstance(qubit_pair, QubitPair) for qubit_pair in machine.qubit_pairs.values()):
        raise ValueError("All qubit pairs should be of type QubitPair")

    return machine


def look_for_standard_op(op: str):
    op = op.lower()
    if op == "cphase":
        return "cz"
    elif op == "cnot":
        return "cx"
    elif op == "x/2" or op == "x90":
        return "sx"
    elif op == "x180":
        return "x"
    elif op == "y180":
        return "y"
    elif op == "y90":
        return "sy"
    return op


class QMBackend(Backend):
    def __init__(
        self,
        machine: QuAM,
        channel_mapping: Optional[Dict[QiskitChannel, QuAMChannel]] = None,
        init_macro: Optional[Callable] = None,
    ):
        """
        Initialize the QM backend
        Args:
            machine: The QuAM instance
            channel_mapping: Optional mapping of Qiskit Pulse Channels to QuAM Channels.
                             This mapping enables the conversion of Qiskit schedules into parametric QUA macros.
            init_macro: Optional macro to be called at the beginning of the QUA program

        """

        Backend.__init__(self, name="QM backend")

        self.machine = validate_machine(machine)
        self.channel_mapping: Dict[QiskitChannel, QuAMChannel] = channel_mapping
        self.reverse_channel_mapping: Dict[QuAMChannel, QiskitChannel] = (
            {v: k for k, v in channel_mapping.items()} if channel_mapping is not None else {}
        )
        self._qubit_dict = {qubit.name: i for i, qubit in enumerate(machine.active_qubits)}
        self._target, self._operation_mapping_QUA = self._populate_target(machine)
        self._oq3_custom_gates = []
        self._init_macro = init_macro

    @property
    def target(self):
        return self._target

    @property
    def qubit_dict(self):
        """
        Get the qubit dictionary for the backend
        """
        return self._qubit_dict

    @property
    def qubit_mapping(self) -> QubitsMapping:
        """
        Build the qubit to quantum elements mapping for the backend.
        Should be of the form {qubit_index: (quantum_element1, quantum_element2, ...)}
        """
        return {
            i: (channel.name for channel in qubit.channels)
            for i, qubit in enumerate(self.machine.active_qubits)
        }

    @property
    def qubit_index_dict(self):
        """
        Returns a dictionary mapping qubit indices (Qiskit numbering) to corresponding Qubit objects (based on
        active_qubits attribute of QuAM instance)
        """
        return {i: qubit for i, qubit in enumerate(self.machine.active_qubits)}

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024,
        )

    def _populate_target(self, machine: QuAM) -> Tuple[Target, Dict[OperationIdentifier, Callable]]:
        """
        Populate the target instructions with the QOP configuration
        """
        gate_map = get_standard_gate_name_mapping()

        class SYGate(Gate):
            def __init__(self, label=None):
                super().__init__("sy", 1, [], label=label)

            def _define(self):
                qc = QuantumCircuit(1)
                qc.ry(np.pi / 2, 0)
                self.definition = qc

        gate_map["sy"] = SYGate()
        target = Target(
            "Transmon based QuAM",
            dt=1e-9,
            granularity=4,
            num_qubits=len(machine.active_qubits),
            min_length=16,
            qubit_properties=[
                QubitProperties(t1=qubit.T1, t2=qubit.T2ramsey, frequency=qubit.f_01)
                for qubit in machine.active_qubits
            ],
        )

        operations_dict = {}
        operations_qua_dict = {}

        # Add single qubit instructions
        for q, qubit in enumerate(machine.active_qubits):
            for op, func in qubit.macros.items():
                op_ = look_for_standard_op(op)

                if op_ in gate_map:
                    gate_op = gate_map[op_]
                    num_params = len(gate_op.params)

                    operations_dict.setdefault(op_, {})[(q,)] = None
                    operations_qua_dict[OperationIdentifier(op_, num_params, (q,))] = func.apply
        for qubit_pair in machine.active_qubit_pairs:
            q_ctrl = self.qubit_dict[qubit_pair.qubit_control.name]
            q_tgt = self.qubit_dict[qubit_pair.qubit_target.name]
            for op, func in qubit_pair.macros.items():
                op_ = look_for_standard_op(op)
                if op_ in gate_map:
                    gate_op = gate_map[op_]
                    num_params = len(gate_op.params)
                    operations_dict.setdefault(op_, {})[(q_ctrl, q_tgt)] = None
                    operations_qua_dict[OperationIdentifier(op_, num_params, (q_ctrl, q_tgt))] = (
                        func.apply
                    )

        for op, properties in operations_dict.items():
            target.add_instruction(gate_map[op], properties=properties)

        for control_flow_op, control_op_name in zip(
            [SwitchCaseOp, ForLoopOp, IfElseOp, WhileLoopOp],
            ["switch_case", "for_loop", "if_else", "while_loop"],
        ):
            target.add_instruction(control_flow_op, name=control_op_name)

        return target, operations_qua_dict

    def get_quam_channel(self, channel: QiskitChannel):
        """
        Convert a Qiskit Pulse channel to a QuAM channel

        Args:
            channel: The Qiskit Pulse Channel to convert

        Returns:
            The corresponding QuAM channel
        """
        try:
            return self.channel_mapping[channel]
        except KeyError:
            raise ValueError(f"Channel {channel} not in the channel mapping")

    def get_pulse_channel(self, channel: QuAMChannel):
        """
        Convert a QuAM channel to a Qiskit Pulse channel

        Args:
            channel: The QuAM channel to convert

        Returns:
            The corresponding pulse channel
        """
        return self.reverse_channel_mapping[channel]

    def meas_map(self) -> List[List[int]]:
        return self._target.concurrent_measurements

    def drive_channel(self, qubit: int):
        """
        Get the drive channel for a given qubit (should be mapped to a quantum element in configuration)
        """
        return DriveChannel(qubit)

    def measure_channel(self, qubit: int):
        return MeasureChannel(qubit)

    def acquire_channel(self, qubit: int):
        return AcquireChannel(qubit)

    def control_channel(self, qubits: Iterable[int]):
        """Return the secondary drive channel for the given qubit

        This is typically used for controlling multiqubit interactions.
        This channel is derived from other channels.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Args:
            qubits: Tuple or list of qubits of the form
                ``(control_qubit, target_qubit)``.

        Returns:
            List[ControlChannel]: The multi qubit control line.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        channels = []
        qubits = list(qubits)
        if len(qubits) != 2:
            raise ValueError("Control channel should be defined for a qubit pair")
        if self.channel_mapping is None:
            raise ValueError("Channel mapping not defined")
        for channel, element in self.channel_mapping.items():
            if isinstance(channel, ControlChannel):
                qubit_pair: QubitPair = element.parent
                qubit_control = qubit_pair.qubit_control
                qubit_target = qubit_pair.qubit_target
                q_ctrl_idx = self.qubit_dict[qubit_control.name]
                q_tgt_idx = self.qubit_dict[qubit_target.name]
                if (q_ctrl_idx, q_tgt_idx) == tuple(qubits):
                    channels.append(channel)
        if len(channels) == 0:
            raise ValueError(
                f"Control channel not found for qubit pair {qubits} in the channel mapping"
            )
        return channels

    def run(self, run_input: RunInput | List[RunInput], **options):
        """
        Run a QuantumCircuit on the QOP backend (currently not supported)
        Args:
            run_input: The QuantumCircuit (or list thereof) to run on the backend. Can
            also be a Qiskit Pulse Schedule or ScheduleBlock
            options: The options for the run
        """
        num_shots = options.get("shots", self.options.shots)
        raise NotImplementedError("Running on the QOP backend is not supported yet")

    def schedule_to_qua_macro(
        self, sched: Schedule, param_table: Optional[ParameterTable] = None
    ) -> Callable:
        """
        Convert a Qiskit Pulse Schedule to a QUA macro

        Args:
            sched: The Qiskit Pulse Schedule to convert
            param_table: The parameter table to use for the conversion of parameterized pulses to QUA variables

        Returns:
            The QUA macro corresponding to the Qiskit Pulse Schedule
        """
        sig = Signature()
        if sched.is_parameterized():
            if param_table is None:
                param_dict = {}
                for channel in list(filter(lambda ch: ch.is_parameterized(), sched.channels)):
                    ch_params = list(channel.parameters)
                    if len(ch_params) > 1:
                        raise NotImplementedError(
                            "Only single parameterized channels are supported"
                        )
                    ch_param = ch_params[0]
                    if ch_param.name not in param_dict:
                        param_dict[ch_param.name] = 0
                for param in sched.parameters:
                    if param.name not in param_dict:
                        param_dict[param.name] = 0.0
                param_table = ParameterTable(param_dict)

            else:
                validate_parameters(sched.parameters, param_table)

            involved_parameters = [value.name for value in sched.parameters]
            params = [
                sigParam(param, sigParam.POSITIONAL_OR_KEYWORD) for param in involved_parameters
            ]
            sig = Signature(params)

        def qua_macro(*args, **kwargs):  # Define the QUA macro with parameters

            # Relate passed positional arguments to parameters in ParameterTable
            bound_params = sig.bind(*args, **kwargs)
            bound_params.apply_defaults()
            if param_table is not None:
                for param_name, value in bound_params.arguments.items():
                    if not param_table.get_parameter(param_name).is_declared:
                        param_table.get_parameter(param_name).declare_variable()
                    param_table.get_parameter(param_name).assign_value(value)

            time_tracker = {channel: 0 for channel in sched.channels}

            for time, instruction in sched.instructions:
                if len(instruction.channels) > 1:
                    raise NotImplementedError("Only single channel instructions are supported")
                qiskit_channel = instruction.channels[0]

                if qiskit_channel.is_parameterized():  # Basic support for parameterized channels
                    # Filter dictionary of pulses based on provided ChannelType
                    channel_dict = {
                        channel.index: quam_channel
                        for channel, quam_channel in self.channel_mapping.items()
                        if isinstance(channel, type(qiskit_channel))
                    }
                    ch_parameter_name = list(qiskit_channel.parameters)[0].name
                    if not param_table.get_parameter(ch_parameter_name).type == int:
                        raise ValueError(
                            f"Parameter {ch_parameter_name} must be of type int for switch case"
                        )

                    # QUA variable corresponding to the channel parameter
                    with switch_(param_table[ch_parameter_name]):
                        for i, quam_channel in channel_dict.items():
                            with case_(i):
                                qiskit_channel = self.get_pulse_channel(quam_channel)
                                if time_tracker[qiskit_channel] < time:
                                    quam_channel.wait((time - time_tracker[qiskit_channel]))
                                    time_tracker[qiskit_channel] = time
                                _instruction_to_qua(
                                    instruction,
                                    quam_channel,
                                    param_table,
                                )
                                time_tracker[qiskit_channel] += instruction.duration
                else:
                    quam_channel = self.get_quam_channel(qiskit_channel)
                    if time_tracker[qiskit_channel] < time:
                        quam_channel.wait((time - time_tracker[qiskit_channel]))
                        time_tracker[qiskit_channel] = time
                    _instruction_to_qua(instruction, quam_channel, param_table)
                    time_tracker[qiskit_channel] += instruction.duration

        qua_macro.__name__ = sched.name if sched.name else "macro" + str(id(sched))
        qua_macro.__signature__ = sig
        return qua_macro

    def add_pulse_operations(
        self,
        pulse_input: Union[Schedule, ScheduleBlock, QiskitPulse],
        name: Optional[str] = None,
    ):
        """
        Add pulse operations created in Qiskit to QuAM operations mapping

        Args:
            pulse_input: The pulse input to add to the QuAM operations mapping (can be a Schedule, ScheduleBlock or Pulse)
            name: An optional name to refer to the pulse operations to be added to the QuAM operations mapping. If
            a Schedule or ScheduleBlock is provided, all pulse operations are named as "{name}_{i}" where i is the number
            of the pulse operation in the schedule. If a Pulse is provided, it is named as "{name}".
        """
        if isinstance(pulse_input, QiskitPulse):
            pulse_input = Schedule(Play(pulse_input, DriveChannel(0)))

        pulse_input = validate_schedule(pulse_input)

        # Update QuAM with additional custom pulses
        for idx, (time, instruction) in enumerate(
            pulse_input.filter(instruction_types=[Play]).instructions
        ):
            instruction: Play
            pulse, channel = instruction.pulse, instruction.channel
            if not isinstance(pulse, (SymbolicPulse, Waveform)):
                raise ValueError("Only SymbolicPulse and Waveform pulses are supported")

            pulse_name = pulse.name
            if (
                not channel.is_parameterized()
                and pulse_name in self.get_quam_channel(channel).operations
            ):
                pulse_name += str(pulse.id)
                pulse.name = pulse_name

            # Check if pulse fits QOP constraints
            if pulse.duration < 16:
                raise ValueError("Pulse duration must be at least 16 ns")
            elif pulse.duration % 4 != 0:
                raise ValueError("Pulse duration must be a multiple of 4 ns")
            if pulse.name is None:
                if name is not None:
                    pulse.name = f"{name}_{idx}"
                else:
                    pulse.name = f"qiskit_pulse_{id(pulse)}"
            quam_pulse = QuAMQiskitPulse(pulse)
            if quam_pulse.is_compile_time_parameterized():
                raise ValueError(
                    "Pulse contains unassigned parameters that cannot be adjusted in real-time"
                )

            if channel.is_parameterized():  # Add pulse to each channel of same type
                for ch in filter(
                    lambda x: isinstance(x, type(channel)),
                    self.channel_mapping.keys(),
                ):
                    self.get_quam_channel(ch).operations[pulse.name] = QuAMQiskitPulse(pulse)
            else:
                self.get_quam_channel(channel).operations[pulse.name] = QuAMQiskitPulse(pulse)

    def update_calibrations(
        self,
        qc: Optional[QuantumCircuit] = None,
        input_type: Optional[InputType] = None,
    ):
        """
        This method updates the QuAM with the custom calibrations of the QuantumCircuit (if any)
        and adds the corresponding operations to the QUA operations mapping for the OQC compiler.
        This method should be called before opening the QuantumMachine instance (i.e. before generating the
        configuration through QuAM) as it modifies the QuAM configuration.
        It also looks at the Target object and checks if new operations are added to the target. If
        so, it adds them to the QUA operations mapping for the OQC compiler.
        """
        # Check the target object for new operations
        for op_name, op_properties in self.target.items():
            gate_set = list(set(key.name for key in self._operation_mapping_QUA.keys())) + list(
                CONTROL_FLOW_OP_NAMES
            )
            if op_name not in gate_set:
                for qubits, properties in op_properties.items():
                    if properties is None:
                        raise ValueError(
                            f"Operation {op_name} has no properties defined in the target,"
                            f"hence cannot be added to the QUA operations mapping"
                        )
                    elif isinstance(properties, QMInstructionProperties):
                        if properties.qua_pulse_macro is None:
                            raise ValueError(
                                f"Operation {op_name} has no QUA macro defined in the target,"
                                f"hence cannot be added to the QUA operations mapping"
                            )
                        sched = properties.qua_pulse_macro
                        num_params = len(sched.__signature__.parameters)
                        self._operation_mapping_QUA[
                            OperationIdentifier(
                                op_name,
                                num_params,
                                qubits,
                            )
                        ] = sched
                    elif isinstance(properties, InstructionProperties):
                        if properties.calibration is None:
                            raise ValueError(
                                f"Operation {op_name} has no calibration defined in the target,"
                                f"hence cannot be added to the QUA operations mapping"
                            )
                        sched = validate_schedule(properties.calibration)
                        num_params = len(sched.parameters)
                        if num_params > 0:
                            param_table = ParameterTable.from_qiskit(
                                sched,
                                input_type=input_type,
                            )
                            sched.metadata["parameter_table"] = param_table
                        else:
                            param_table = None
                        self._operation_mapping_QUA[
                            OperationIdentifier(
                                op_name,
                                num_params,
                                qubits,
                            )
                        ] = self.schedule_to_qua_macro(sched, param_table)

        if qc is not None:
            if not isinstance(qc, QuantumCircuit):
                raise ValueError("qc should be a QuantumCircuit")
            if not hasattr(qc, "calibrations"):
                raise ValueError("qc should have calibrations")
            if qc.parameters or qc.iter_input_vars():
                param_table = qc.metadata.get(
                    "parameter_table",
                    ParameterTable.from_qiskit(qc, input_type=input_type),
                )
            else:
                param_table = None

            if hasattr(qc, "calibrations") and qc.calibrations:  # Check for custom calibrations
                for gate_name, cal_info in qc.calibrations.items():
                    if (
                        gate_name not in self._oq3_custom_gates
                    ):  # Make it a basis gate for OQ compiler
                        self._oq3_custom_gates.append(gate_name)
                    for (qubits, parameters), schedule in cal_info.items():
                        schedule = validate_schedule(
                            schedule
                        )  # Check that schedule has fixed duration

                        # Convert type of parameters to int if required (for switch case over channels)
                        if param_table is not None:
                            param_table = handle_parameterized_channel(schedule, param_table)
                            qc.metadata["parameter_table"] = param_table

                        self._operation_mapping_QUA[
                            OperationIdentifier(
                                gate_name,
                                len(parameters),
                                qubits,
                            )
                        ] = self.schedule_to_qua_macro(schedule, param_table)

                        self.add_pulse_operations(schedule, name=schedule.name)

    def quantum_circuit_to_qua(
        self,
        qc: QuantumCircuit,
        param_table: Optional[ParameterTable | List[ParameterTable | Parameter]] = None,
    ):
        """
        Convert a QuantumCircuit to a QUA program (can be called within an existing QUA program or to generate a
        program for the circuit)

        Args:
            qc: The QuantumCircuit to convert
            param_table: The parameter table to use for the conversion of parameterized instructions to QUA variables
                        Should be provided if the QuantumCircuit contains real-time variables or symbolic Parameters
                         to be cast as real-time parameters (typically amp, phase, frequency or duration parameters)
                         and this function is called within a QUA program

        Returns:
            Compilation result of the QuantumCircuit to QUA
        """
        # if qc.parameters and param_table is None:
        #     raise ValueError(
        #         "QuantumCircuit contains parameters but no parameter table provided"
        #     )
        basis_gates = [gate for gate in self._oq3_custom_gates if gate not in ["measure", "reset"]]
        basis_gates += [
            gate
            for gate in self.target.operation_names
            if gate not in basis_gates and gate not in ["measure", "reset"]
        ]
        # Check if all custom calibrations are in the oq3 basis gates
        for gate_name in qc.calibrations.keys():
            if gate_name not in basis_gates:
                raise ValueError(
                    f"Custom calibration {gate_name} not in basis gates {basis_gates}",
                    f"Run update_calibrations() before compiling the circuit",
                )
        exporter = Exporter(includes=(), basis_gates=basis_gates, disable_constants=True)
        open_qasm_code = exporter.dumps(qc)
        open_qasm_code = "\n".join(
            line
            for line in open_qasm_code.splitlines()
            if not line.strip().startswith(("barrier",))
        )
        inputs = None
        if param_table is not None:
            inputs = {}
            if isinstance(param_table, (ParameterTable, Parameter)):
                param_table = [param_table]
            for table in param_table:
                if not table.is_declared:
                    if isinstance(table, ParameterTable):
                        table.declare_variables(pause_program=False)
                    else:
                        table.declare_variable(pause_program=False)
                variables = (
                    table.variables_dict
                    if isinstance(table, ParameterTable)
                    else {table.name: table.var}
                )
                inputs.update(variables)

        result = self.compiler.compile(
            open_qasm_code,
            compilation_name=f"{qc.name}_qua",
            inputs=inputs,
        )
        return result

    def qiskit_to_qua_macro(
        self,
        qc: RunInput,
        input_type: Optional[InputType] = None,
    ) -> CompilationResult | Program | Callable[..., Any]:
        """
        Convert given input into a QUA program
        """

        if qc.parameters:  # Initialize the parameter table
            parameter_table = ParameterTable.from_qiskit(qc, input_type=input_type)
            qc.metadata["parameter_table"] = parameter_table
        else:
            parameter_table = None
        if isinstance(qc, QuantumCircuit):
            return self.quantum_circuit_to_qua(qc, parameter_table)
        elif isinstance(qc, (ScheduleBlock, Schedule)):  # Convert to Schedule first
            schedule = validate_schedule(qc)

            return self.schedule_to_qua_macro(schedule, parameter_table)
        else:
            raise ValueError(f"Unsupported input {qc}")

    @property
    def compiler(self) -> Compiler:
        """
        The OpenQASM to QUA compiler.
        """
        return Compiler(
            hardware_config=HardwareConfig(
                quantum_operations_db=self._operation_mapping_QUA,
                physical_qubits=self.qubit_mapping,
            )
        )

    def connect(self) -> QuantumMachinesManager:
        """
        Connect to the Quantum Machines Manager
        """
        return self.machine.connect()

    def generate_config(self) -> DictQuaConfig:
        """
        Generate the configuration for the Quantum Machine
        """
        return self.machine.generate_config()

    @property
    def init_macro(self) -> Optional[Callable]:
        """
        The macro to be called at the beginning of the QUA program
        """
        return self._init_macro


class FluxTunableTransmonBackend(QMBackend):

    def __init__(
        self,
        machine: QuAM,
    ):
        """
        Initialize the QM backend for the Flux-Tunable Transmon based QuAM

        Args:
            machine: The QuAM instance
            channel_mapping: Optional mapping of Qiskit Pulse Channels to QuAM Channels.
                             This mapping enables the conversion of Qiskit schedules into parametric QUA macros.
        """
        if not hasattr(machine, "qubits") or not hasattr(machine, "qubit_pairs"):
            raise ValueError(
                "Invalid QuAM instance provided, should have qubits and qubit_pairs attributes"
            )
        drive_channel_mapping = {
            DriveChannel(i): qubit.xy for i, qubit in enumerate(machine.qubits.values())
        }
        flux_channel_mapping = {
            FluxChannel(i): qubit.z for i, qubit in enumerate(machine.qubits.values())
        }
        readout_channel_mapping = {
            MeasureChannel(i): qubit.resonator for i, qubit in enumerate(machine.qubits.values())
        }
        control_channel_mapping = {
            ControlChannel(i): qubit_pair.coupler
            for i, qubit_pair in enumerate(machine.qubit_pairs.values())
        }
        channel_mapping = {
            **drive_channel_mapping,
            **flux_channel_mapping,
            **control_channel_mapping,
            **readout_channel_mapping,
        }
        super().__init__(
            machine,
            channel_mapping=channel_mapping,
            init_macro=machine.apply_all_flux_to_joint_idle,
        )

    @property
    def qubit_mapping(self) -> QubitsMapping:
        """
        Retrieve the qubit to quantum elements mapping for the backend.
        """
        return {
            i: (qubit.xy.name, qubit.z.name, qubit.resonator.name)
            for i, qubit in enumerate(self.machine.qubits.values())
        }

    @property
    def meas_map(self) -> List[List[int]]:
        """
        Retrieve the measurement map for the backend.
        """
        return [[i] for i in range(len(self.machine.qubits))]

    def flux_channel(self, qubit: int):
        """
        Retrieve the flux channel for the given qubit.
        """
        return FluxChannel(qubit)


def qua_declaration(n_qubits, readout_elements):
    """
    Macro to declare the necessary QUA variables

    :param n_qubits: Number of qubits used in this experiment
    :return:
    """
    I, Q = [[declare(fixed) for _ in range(n_qubits)] for _ in range(2)]
    I_st, Q_st = [[declare_stream() for _ in range(n_qubits)] for _ in range(2)]
    # Workaround to manually assign the results variables to the readout elements
    for i in range(n_qubits):
        assign_variables_to_element(readout_elements[i], I[i], Q[i])
    return I, I_st, Q, Q_st
