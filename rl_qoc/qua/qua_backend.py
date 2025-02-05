from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence, Dict, Optional, Callable, Union, Tuple

from quam.components import Channel as QuAMChannel, Qubit, QubitPair
from quam.components import BasicQuAM as QuAM

from qiskit.circuit import (
    QuantumCircuit,
    SwitchCaseOp,
    ForLoopOp,
    IfElseOp,
    WhileLoopOp,
)
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping,
)
from qiskit.providers import BackendV2 as Backend, QubitProperties
from qiskit.pulse import (
    ScheduleBlock,
    Schedule,
    DriveChannel,
    MeasureChannel,
    AcquireChannel,
    Play,
    UnassignedDurationError,
    PulseError,
)
from qiskit.pulse.library.pulse import Pulse as QiskitPulse

from qiskit.transpiler import Target, InstructionProperties, CouplingMap
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.pulse.transforms import block_to_schedule
from qiskit.pulse.channels import Channel as QiskitChannel, ControlChannel
from qiskit.pulse.library import SymbolicPulse
from qiskit.pulse.library.waveform import Waveform
from qm.qua import switch_, case_
from qm.qua import declare, fixed, declare_stream
from qm import QuantumMachinesManager, Program
from qualang_tools.addons.variables import assign_variables_to_element
from .parameter_table import ParameterTable

from .pulse_support_utils import (
    _instruction_to_qua,
    validate_parameters,
    validate_schedule,
    handle_parameterized_channel,
)
from .qua_utils import add_parameter_table_to_circuit
from .quam_qiskit_pulse import QuAMQiskitPulse, FluxChannel

from oqc import (
    Compiler,
    HardwareConfig,
    OperationIdentifier,
    OperationsMapping,
    QubitsMapping,
)


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
    if not all(
        isinstance(qubit_pair, QubitPair) for qubit_pair in machine.qubit_pairs.values()
    ):
        raise ValueError("All qubit pairs should be of type QubitPair")

    return machine


def look_for_standard_op(op: str):
    op = op.lower()
    if op == "cphase":
        return "cz"
    elif op == "cnot":
        return "cx"

    return op


class QMBackend(Backend):
    def __init__(
        self,
        machine: QuAM,
        channel_mapping: Optional[Dict[QiskitChannel, QuAMChannel]] = None,
    ):
        """
        Initialize the QM backend
        Args:
            machine: The QuAM instance
            channel_mapping: Optional mapping of Qiskit Pulse Channels to QuAM Channels.
                             This mapping enables the conversion of Qiskit schedules into parametric QUA macros.

        """

        Backend.__init__(self, name="QM backend")

        self.machine = validate_machine(machine)
        self.channel_mapping: Dict[QiskitChannel, QuAMChannel] = channel_mapping
        self.reverse_channel_mapping: Dict[QuAMChannel, QiskitChannel] = (
            {v: k for k, v in channel_mapping.items()}
            if channel_mapping is not None
            else {}
        )
        self._qubit_dict = {
            qubit.name: i for i, qubit in enumerate(machine.active_qubits)
        }
        self._target, self._operation_mapping_QUA = self._populate_target(machine)
        self._oq3_basis_gates = list(self.target.operation_names)

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
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        pass

    def _populate_target(
        self, machine: QuAM
    ) -> Tuple[Target, Dict[OperationIdentifier, Callable]]:
        """
        Populate the target instructions with the QOP configuration
        """
        gate_map = get_standard_gate_name_mapping()
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
                    operations_qua_dict[OperationIdentifier(op_, num_params, (q,))] = (
                        func.apply
                    )
        for qubit_pair in machine.active_qubit_pairs:
            q_ctrl = self.qubit_dict[qubit_pair.qubit_control.name]
            q_tgt = self.qubit_dict[qubit_pair.qubit_target.name]
            for op, func in qubit_pair.macros.items():
                op_ = look_for_standard_op(op)
                if op_ in gate_map:
                    gate_op = gate_map[op_]
                    num_params = len(gate_op.params)
                    operations_dict.setdefault(op_, {})[(q_ctrl, q_tgt)] = None
                    operations_qua_dict[
                        OperationIdentifier(op_, num_params, (q_ctrl, q_tgt))
                    ] = func.apply

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

    def control_channel(self, qubits: Iterable[int]):
        pass

    def measure_channel(self, qubit: int):
        return MeasureChannel(qubit)

    def acquire_channel(self, qubit: int):
        return AcquireChannel(qubit)

    def run(self, run_input, **options):
        """
        Run a QuantumCircuit on the QOP backend (currently not supported)
        Args:
            run_input: The QuantumCircuit (or list thereof) to run on the backend
            options: The options for the run
        """
        raise NotImplementedError(
            "Run method not supported for QOP backend as deprecated by Qiskit"
        )

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
        if sched.is_parameterized():
            if param_table is None:
                param_dict = {}
                for channel in list(
                    filter(lambda ch: ch.is_parameterized(), sched.channels)
                ):
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

            involved_parameters = [value.name for value in param_table.parameters]

        def qua_macro(
            *params,
        ):  # Define the QUA macro with parameters

            # Relate passed positional arguments to parameters in ParameterTable
            if not param_table.is_declared:
                if not params:
                    raise ValueError(
                        "Parameter table not declared and no parameters provided"
                    )
                param_table.declare_variables(pause_program=False)
                for param, value in zip(param_table.parameters, params):
                    param.assign_value(value)

            time_tracker = {channel: 0 for channel in sched.channels}

            for time, instruction in sched.instructions:
                if len(instruction.channels) > 1:
                    raise NotImplementedError(
                        "Only single channel instructions are supported"
                    )
                qiskit_channel = instruction.channels[0]

                if (
                    qiskit_channel.is_parameterized()
                ):  # Basic support for parameterized channels
                    # Filter dictionary of pulses based on provided ChannelType
                    channel_dict = {
                        channel.index: quam_channel
                        for channel, quam_channel in self.channel_mapping.items()
                        if isinstance(channel, type(qiskit_channel))
                    }
                    ch_parameter_name = list(qiskit_channel.parameters)[0].name
                    if not param_table.table[ch_parameter_name].type == int:
                        raise ValueError(
                            f"Parameter {ch_parameter_name} must be of type int for switch case"
                        )
                    with switch_(
                        param_table[
                            ch_parameter_name
                        ]  # QUA variable corresponding to the channel parameter
                    ):  # Switch based on the parameter value
                        for i in channel_dict.keys():
                            with case_(i):
                                quam_channel = channel_dict[i]
                                qiskit_channel = self.get_pulse_channel(quam_channel)
                                if time_tracker[qiskit_channel] < time:
                                    quam_channel.wait(
                                        (time - time_tracker[qiskit_channel])
                                    )
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

        return qua_macro

    def add_pulse_operations(
        self, pulse_input: Union[Schedule, ScheduleBlock, QiskitPulse]
    ):
        """
        Add pulse operations created in Qiskit to QuAM operations mapping

        Args:
            pulse_input: The pulse input to add to the QuAM operations mapping (can be a Schedule, ScheduleBlock or Pulse)
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
                    self.get_quam_channel(ch).operations[pulse.name] = QuAMQiskitPulse(
                        pulse
                    )
            else:
                self.get_quam_channel(channel).operations[pulse.name] = QuAMQiskitPulse(
                    pulse
                )

    def update_calibrations(self, qc: QuantumCircuit):
        """
        This method updates the QuAM with the custom calibrations of the QuantumCircuit (if any)
        and adds the corresponding operations to the QUA operations mapping for the OQC compiler.
        This method should be called before opening the QuantumMachine instance (i.e. before generating the
        configuration through QuAM) as it modifies the QuAM configuration.
        """

        qc, param_table = add_parameter_table_to_circuit(qc)

        if (
            hasattr(qc, "calibrations") and qc.calibrations
        ):  # Check for custom calibrations
            for gate_name, cal_info in qc.calibrations.items():
                if (
                    gate_name not in self._oq3_basis_gates
                ):  # Make it a basis gate for OQ compiler
                    self._oq3_basis_gates.append(gate_name)
                for (qubits, parameters), schedule in cal_info.items():
                    schedule = validate_schedule(
                        schedule
                    )  # Check that schedule has fixed duration

                    # Convert type of parameters to int if required (for switch case over channels)
                    param_table = handle_parameterized_channel(schedule, param_table)

                    qc.metadata["parameter_table"] = param_table

                    self._operation_mapping_QUA[
                        OperationIdentifier(
                            gate_name,
                            len(parameters),
                            qubits,
                        )
                    ] = self.schedule_to_qua_macro(schedule, param_table)

                    self.add_pulse_operations(schedule)

    def quantum_circuit_to_qua(
        self, qc: QuantumCircuit, param_table: Optional[ParameterTable] = None
    ):
        """
        Convert a QuantumCircuit to a QUA program (can be called within an existing QUA program or to generate a
        program for the circuit)

        Args:
            qc: The QuantumCircuit to convert
            param_table: The parameter table to use for the conversion of parameterized instructions to QUA variables
                        Should be provided if the QuantumCircuit contains parameterized instructions and this function
                        is called within a QUA program

        Returns:
            Compilation result of the QuantumCircuit to QUA
        """
        if qc.parameters:
            if param_table is not None:
                validate_parameters(qc.parameters, param_table)
                if "parameter_table" not in qc.metadata:
                    qc.metadata["parameter_table"] = param_table
                else:
                    assert (
                        qc.metadata["parameter_table"] == param_table
                    ), "Parameter table provided is different from the one in the QuantumCircuit metadata"
            else:
                param_table = qc.metadata.get("parameter_table", None)

        open_qasm_code = qasm3_dumps(qc, includes=(), basis_gates=self._oq3_basis_gates)
        open_qasm_code = "\n".join(
            line
            for line in open_qasm_code.splitlines()
            if not line.strip().startswith(("barrier",))
        )
        result = self.compiler.compile(
            open_qasm_code,
            inputs=(param_table.variables_dict if param_table.is_declared else None),
        )
        return result

    def qua_prog_from_qc(self, qc: QuantumCircuit | Schedule | ScheduleBlock | Program):
        """
        Convert given input into a QUA program
        """
        if isinstance(qc, Program):
            return qc
        else:
            if qc.parameters:  # Initialize the parameter table
                parameter_table = ParameterTable(
                    {param.name: 0.0 for param in qc.parameters}
                )
                qc.metadata["parameter_table"] = parameter_table
            else:
                parameter_table = None
            if isinstance(qc, QuantumCircuit):
                return self.quantum_circuit_to_qua(qc, parameter_table)
            elif isinstance(qc, (ScheduleBlock, Schedule)):  # Convert to Schedule first
                schedule = qc
                if isinstance(qc, ScheduleBlock):
                    if not qc.is_schedulable():
                        # TODO: Build ScheduleBlock to QUA compiler
                        raise ValueError(
                            "ScheduleBlock is not schedulable (contains unschedulable instructions)"
                        )
                    try:
                        schedule = block_to_schedule(qc)
                    except (UnassignedDurationError, PulseError) as e:
                        # TODO: Build ScheduleBlock to QUA compiler
                        raise RuntimeError(
                            "ScheduleBlock could not be converted to Schedule (required"
                            "for converting it to QUA program"
                        ) from e
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


class FluxTunableTransmonBackend(QMBackend):

    def __init__(
        self,
        machine: QuAM,
    ):
        """
        Initialize the QM backend for the Flux Tunable Transmon based QuAM

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
            MeasureChannel(i): qubit.resonator
            for i, qubit in enumerate(machine.qubits.values())
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
        super().__init__(machine, channel_mapping=channel_mapping)

    @property
    def qubit_mapping(self) -> QubitsMapping:
        """
        Retrieve the qubit to quantum elements mapping for the backend.
        """
        return {
            i: (qubit.xy.name, qubit.z.name, qubit.resonator.name)
            for i, qubit in enumerate(self.machine.qubits.values())
        }

    # def _populate_target(self, machine: QuAM):
    #     """
    #     Populate the target instructions with the QOP configuration (currently hardcoded for
    #     Transmon based QuAM architecture)
    #
    #     """
    #     gates = gate_map()
    #     target = Target(
    #         "Transmon based QuAM",
    #         dt=1e-9,
    #         granularity=4,
    #         num_qubits=len(machine.qubits),
    #         min_length=16,
    #         qubit_properties=[
    #             QubitProperties(t1=qubit.T1, t2=qubit.T2ramsey, frequency=qubit.f_01)
    #             for qubit in machine.qubits.values()
    #         ],
    #     )
    #     # Create CouplingMap from QuAM qubit pairs
    #     qubit_dict = {qubit.name: i for i, qubit in enumerate(machine.qubits.values())}
    #     coupling_map = CouplingMap()
    #     for qubit in range(len(machine.qubits)):
    #         coupling_map.add_physical_qubit(qubit)
    #     for qubit_pair in machine.qubit_pairs.values():
    #         coupling_map.add_edge(
    #             qubit_dict[qubit_pair.qubit_control.name],
    #             qubit_dict[qubit_pair.qubit_target.name],
    #         )
    #
    #     target.add_instruction(
    #         gates["x"], properties={(i,): None for i in range(len(machine.qubits))}
    #     )
    #     target.add_instruction(
    #         gates["sx"], properties={(i,): None for i in range(len(machine.qubits))}
    #     )
    #     target.add_instruction(
    #         gates["cz"], properties={(i, j): None for i, j in coupling_map.get_edges()}
    #     )
    #     # TODO: Add the rest of the channels for QubitPairs (ControlChannels)
    #
    #     # TODO: Update the instructions both in Qiskit and in the OQC operations mapping
    #     # TODO: Figure out if pulse calibrations should be added to Target
    #
    #     self._coupling_map = target.build_coupling_map()
    #
    #     return target, ()


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
