from __future__ import annotations

import os
import sys
import pickle
import gzip
import warnings

from qiskit import pulse, QuantumRegister
from qiskit.circuit import (
    QuantumCircuit,
    Gate,
    Parameter,
    CircuitInstruction,
    ParameterVector,
    Delay,
    Qubit,
)
from qiskit.circuit.library import get_standard_gate_name_mapping as gate_map, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.primitives import (
    BackendEstimator,
    Estimator,
    Sampler,
    BackendSampler,
    StatevectorEstimator,
    StatevectorSampler,
    BaseEstimatorV1,
    BaseEstimatorV2,
    BackendSamplerV2,
    BackendEstimatorV2,
)

from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import (
    Operator,
    Statevector,
    DensityMatrix,
)
from qiskit.transpiler import (
    CouplingMap,
    PassManager,
    InstructionDurations,
)

from qiskit.providers import (
    BackendV1,
    Backend,
    BackendV2,
    Options as AerOptions,
    QiskitBackendNotFoundError,
)

from qiskit_ibm_runtime.fake_provider import FakeProvider, FakeProviderForBackendV2
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2, FakeBackend
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_ibm_runtime import (
    Session,
    IBMBackend as RuntimeBackend,
    EstimatorV2 as RuntimeEstimatorV2,
    Options as RuntimeOptions,
    EstimatorOptions as RuntimeEstimatorOptions,
    SamplerV2 as RuntimeSamplerV2,
    QiskitRuntimeService,
)

from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.framework import BatchExperiment, BaseAnalysis
from qiskit_experiments.library import (
    StateTomography,
    ProcessTomography,
)

from itertools import chain
from typing import Optional, Tuple, List, Union, Dict, Sequence, Callable, Any
import yaml

import numpy as np

from gymnasium.spaces import Box
import optuna

import keyword
import re

from .pulse_utils import perform_standard_calibrations
from .transpiler_passes import CustomGateReplacementPass
from ..custom_jax_sim import PulseEstimatorV2
from ..environment.qconfig import (
    BackendConfig,
    ExecutionConfig,
    BenchmarkConfig,
    StateRewardConfig,
    QiskitRuntimeConfig,
    DynamicsConfig,
    QEnvConfig,
    CAFERewardConfig,
    ChannelRewardConfig,
    XEBRewardConfig,
    ORBITRewardConfig,
    FidelityConfig,
)

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

Estimator_type = Union[
    RuntimeEstimatorV2,
    Estimator,
    BackendEstimator,
    BackendEstimatorV2,
    StatevectorEstimator,
]
Sampler_type = Union[
    RuntimeSamplerV2,
    Sampler,
    BackendSampler,
    BackendSamplerV2,
    StatevectorSampler,
]
Backend_type = Optional[Union[BackendV1, BackendV2]]
QuantumTarget = Union[Statevector, Operator, DensityMatrix]
QuantumInput = Union[QuantumCircuit, pulse.Schedule, pulse.ScheduleBlock]
PulseInput = Union[pulse.Schedule, pulse.ScheduleBlock]

reward_configs = {
    "channel": ChannelRewardConfig,
    "xeb": XEBRewardConfig,
    "orbit": ORBITRewardConfig,
    "state": StateRewardConfig,
    "cafe": CAFERewardConfig,
    "fidelity": FidelityConfig,
}


def to_python_identifier(s):
    # Prepend underscore if the string starts with a digit
    if s[0].isdigit():
        s = "_" + s

    # Replace non-alphanumeric characters with underscore
    s = re.sub("\W|^(?=\d)", "_", s)

    # Append underscore if the string is a Python keyword
    if keyword.iskeyword(s):
        s += "_"

    return s


def count_gates(qc: QuantumCircuit):
    """
    Count number of gates in a Quantum Circuit
    """
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            if not isinstance(gate.operation, Delay):
                gate_count[qubit] += 1
    return gate_count


def remove_unused_wires(qc: QuantumCircuit):
    """
    Remove unused wires from a Quantum Circuit
    """
    gate_count = count_gates(qc)
    for qubit, count in gate_count.items():
        if count == 0:
            for instr in qc.data:
                if qubit in instr.qubits:
                    qc.data.remove(instr)
            qc.qubits.remove(qubit)
    return qc


def get_instruction_timings(circuit: QuantumCircuit):
    # Initialize the timings for each qubit
    qubit_timings = {i: 0 for i in range(circuit.num_qubits)}

    # Initialize the list of start times
    start_times = []

    # Loop over each instruction in the circuit
    for instruction in circuit.data:
        qubits = instruction.qubits
        qubit_indices = [circuit.find_bit(qubit).index for qubit in qubits]
        # Find the maximum time among the qubits involved in the instruction
        start_time = max(qubit_timings[i] for i in qubit_indices)

        # Add the start time to the list of start times
        start_times.append(start_time)

        # Update the time for each qubit involved in the instruction
        for i in qubit_indices:
            qubit_timings[i] = start_time + 1

    return start_times


def density_matrix_to_statevector(density_matrix: DensityMatrix):
    """
    Convert a density matrix to a statevector (if the density matrix represents a pure state)

    Args:
        density_matrix: DensityMatrix object representing the pure state

    Returns:
        Statevector: Statevector object representing the pure state

    Raises:
        ValueError: If the density matrix does not represent a pure state
    """
    # Check if the state is pure by examining if Tr(rho^2) is 1
    if np.isclose(density_matrix.purity(), 1):
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix.data)

        # Find the eigenvector corresponding to the eigenvalue 1 (pure state)
        # The statevector is the eigenvector corresponding to the maximum eigenvalue
        max_eigenvalue_index = np.argmax(eigenvalues)
        statevector = eigenvectors[:, max_eigenvalue_index]

        # Return the Statevector object
        return Statevector(statevector)
    else:
        raise ValueError("The density matrix does not represent a pure state.")


def causal_cone_circuit(
    circuit: QuantumCircuit,
    qubits: Sequence[int | Qubit] | QuantumRegister,
) -> Tuple[QuantumCircuit, List[Qubit]]:
    """
    Get the causal cone circuit of the specified qubits as well as the qubits involved in the causal cone

    Args:
        circuit: Quantum Circuit
        qubits: Qubits of interest
    """

    dag = circuit_to_dag(circuit)
    if isinstance(qubits, List) and all(isinstance(q, int) for q in qubits):
        qubits = [dag.qubits[q] for q in qubits]
    involved_qubits = [dag.quantum_causal_cone(q) for q in qubits]
    involved_qubits = list(set([q for sublist in involved_qubits for q in sublist]))
    filtered_dag = dag.copy_empty_like()
    for node in dag.topological_op_nodes():
        if all(q in involved_qubits for q in node.qargs):
            filtered_dag.apply_operation_back(node.op, node.qargs)

    filtered_dag.remove_qubits(
        *[q for q in filtered_dag.qubits if q not in involved_qubits]
    )
    return dag_to_circuit(filtered_dag, False), involved_qubits


def fidelity_from_tomography(
    qc_input: List[QuantumCircuit] | QuantumCircuit,
    backend: Optional[Backend],
    physical_qubits: Optional[Sequence[int]],
    target: Optional[QuantumTarget | List[QuantumTarget]] = None,
    analysis: Union[BaseAnalysis, None, str] = "default",
    **run_options,
):
    """
    Extract average state or gate fidelity from batch of Quantum Circuit for target state or gate

    Args:
        qc_input: Quantum Circuit input to benchmark (Note that we handle removing final measurements if any)
        backend: Backend instance
        physical_qubits: Physical qubits on which state or process tomography is to be performed
        analysis: Analysis instance
        target: Target state or gate for fidelity calculation (must be either Operator or QuantumState)
    Returns:
        avg_fidelity: Average state or gate fidelity (over the batch of Quantum Circuits)
    """
    if isinstance(qc_input, QuantumCircuit):
        qc_input = [qc_input.remove_final_measurements(False)]
    else:
        qc_input = [qc.remove_final_measurements(False) for qc in qc_input]
    if isinstance(target, QuantumTarget):
        target = [target] * len(qc_input)
    elif target is not None:
        if len(target) != len(qc_input):
            raise ValueError(
                "Number of target states/gates does not match the number of input circuits"
            )
    else:
        target = [Statevector(qc) for qc in qc_input]

    exps = []
    fids = []
    for qc, tgt in zip(qc_input, target):
        if isinstance(tgt, Operator):
            exps.append(
                ProcessTomography(
                    qc, physical_qubits=physical_qubits, analysis=analysis, target=tgt
                )
            )
            fids.append("process_fidelity")
        elif isinstance(tgt, QuantumState):
            exps.append(
                StateTomography(
                    qc, physical_qubits=physical_qubits, analysis=analysis, target=tgt
                )
            )
            fids.append("state_fidelity")
        else:
            raise TypeError("Target must be either Operator or QuantumState")
    batch_exp = BatchExperiment(exps, backend=backend, flatten_results=False)

    exp_data = batch_exp.run(backend_run=True, **run_options).block_for_results()
    results = []
    for fid, tgt, child_data in zip(fids, target, exp_data.child_data()):
        result = child_data.analysis_results(fid).value
        if fid == "process_fidelity" and tgt.is_unitary():
            # Convert to average gate fidelity metric
            dim, _ = tgt.dim
            result = dim * result / (dim + 1)
        results.append(result)

    return results if len(results) > 1 else results[0]


def get_gate(gate: Gate | str):
    """
    Get gate from gate_map
    """
    if isinstance(gate, str):
        if gate.lower() == "cnot":
            gate = "cx"
        elif gate.lower() == "cphase":
            gate = "cz"
        elif gate.lower() == "x/2":
            gate = "sx"
        try:
            gate = gate_map()[gate.lower()]
        except KeyError:
            raise ValueError("Invalid target gate name")
    return gate


def retrieve_primitives(
    backend: Backend_type,
    config: BackendConfig,
    estimator_options: Optional[
        Dict | AerOptions | RuntimeOptions | RuntimeEstimatorOptions
    ] = None,
) -> Tuple[Estimator_type, Sampler_type]:
    """
    Retrieve appropriate Qiskit primitives (estimator and sampler) from backend and layout

    Args:
        backend: Backend instance
        config: Configuration dictionary
        estimator_options: Estimator options
    """
    if isinstance(backend, DynamicsBackend):
        assert isinstance(
            config, DynamicsConfig
        ), "Configuration must be a DynamicsConfig"
        dummy_param = Parameter("dummy")
        if hasattr(dummy_param, "jax_compat"):
            estimator = PulseEstimatorV2(backend=backend, options=estimator_options)
        else:
            estimator = BackendEstimatorV2(backend=backend, options=estimator_options)
        sampler = BackendSamplerV2(backend=backend)

        if config.do_calibrations and not backend.target.has_calibration("x", (0,)):
            calibration_files = config.calibration_files
            _, _ = perform_standard_calibrations(backend, calibration_files)
    elif config.config_type == "qibo":
        from ..qibo import QiboEstimatorV2

        estimator = QiboEstimatorV2(
            platform=config.platform,
            options={"qubits": config.physical_qubits, "gate_rule": config.gate_rule},
        )
        sampler = StatevectorSampler()  # Dummy sampler
    elif isinstance(backend, (FakeBackend, FakeBackendV2, AerBackend)):
        from qiskit_aer.primitives import (
            EstimatorV2 as AerEstimatorV2,
            SamplerV2 as AerSamplerV2,
        )

        estimator = AerEstimatorV2.from_backend(backend=backend)
        sampler = AerSamplerV2.from_backend(backend=backend)
    elif backend is None:  # No backend specified, ideal state-vector simulation
        sampler = StatevectorSampler()
        estimator = StatevectorEstimator()

    elif isinstance(backend, RuntimeBackend):
        estimator = RuntimeEstimatorV2(
            mode=Session(backend=backend),
            options=estimator_options,
        )
        sampler = RuntimeSamplerV2(mode=estimator.mode)

    # elif config.config_type == 'qua':
    #     estimator = QMEstimator(backend=backend, options=estimator_options)
    #     sampler = QMSampler(backend=backend)
    else:
        estimator = BackendEstimatorV2(backend=backend, options=estimator_options)
        sampler = BackendSamplerV2(backend=backend)

    return estimator, sampler


def substitute_target_gate(
    circuit: QuantumCircuit,
    target_gate: Gate | str,
    custom_gate: Gate | str,
    qubits: Optional[Sequence[int]] = None,
    parameters: ParameterVector | List[Parameter] | List[float] = None,
):
    """
    Substitute target gate in Quantum Circuit with a parametrized version of the gate.
    The parametrized_circuit function signature should match the expected one for a QiskitConfig instance.

    Args:
        circuit: Quantum Circuit instance
        target_gate: Target gate to be substituted
        custom_gate: Custom gate to be substituted with
        qubits: Physical qubits on which the gate is to be applied (if None, all qubits of input circuit are considered)
    """

    if isinstance(custom_gate, str):
        try:
            custom_gate2 = gate_map()[custom_gate]
        except KeyError:
            raise ValueError(f"Custom gate {custom_gate} not found in gate map")
        if custom_gate2.params and parameters is not None:
            assert len(custom_gate2.params) == len(parameters), (
                f"Number of parameters ({len(parameters)}) does not match number of parameters "
                f"required by the custom gate ({len(custom_gate2.params)})"
            )

    pass_manager = PassManager(
        [
            CustomGateReplacementPass(
                (target_gate, qubits), custom_gate, parameters=parameters
            )
        ]
    )

    return pass_manager.run(circuit)


def handle_session(
    estimator: BaseEstimatorV1 | BaseEstimatorV2,
    backend: Backend_type,
    counter: Optional[int] = None,
):
    """
    Handle session reopening for RuntimeEstimator or load necessary data for custom DynamicsBackendEstimator
    Args:
        estimator: Estimator instance
        backend: Backend instance
        counter: Optional session counter (for RuntimeEstimator) or circuit macro counter (for DynamicsBackendEstimator)
        qc: Optional QuantumCircuit instance (for DynamicsBackendEstimator)
        input_state_circ: Optional input state QuantumCircuit instance (for DynamicsBackendEstimator)

    Returns:
        Updated Estimator instance
    """
    if (
        isinstance(estimator, RuntimeEstimatorV2)
        and estimator.mode.status() == "Closed"
    ):
        old_session = estimator.mode
        counter += 1
        print(f"New Session opened (#{counter})")
        session, options = (
            Session(old_session.service, backend),
            estimator.options,
        )
        estimator = type(estimator)(mode=session, options=options)

    return estimator


def select_backend(
    real_backend: Optional[bool] = None,
    channel: Optional[str] = None,
    instance: Optional[str] = None,
    backend_name: Optional[str] = None,
    use_dynamics: Optional[bool] = None,
    physical_qubits: Optional[List[int]] = None,
    solver_options: Optional[Dict] = None,
    calibration_files: Optional[str] = None,
) -> Optional[BackendV2]:
    """
    Select backend to use for training among real backend or fake backend (Aer Simulator)

    Args:
        real_backend: Boolean indicating if real backend should be used
        channel: Channel to use for Runtime Service
        instance: Instance to use for Runtime Service
        backend_name: Name of the backend to use for training
        use_dynamics: Boolean indicating if DynamicsBackend should be used
        physical_qubits: Physical qubits on which DynamicsBackend should be used
        solver_options: Solver options for DynamicsBackend
        calibration_files: Calibration files for DynamicsBackend

    Returns:
        backend: Backend instance
    """

    backend = None
    if real_backend is not None:
        if real_backend:
            service = QiskitRuntimeService(channel=channel, instance=instance)
            if backend_name is None:
                backend = service.least_busy(
                    min_num_qubits=2, simulator=False, operational=True, open_pulse=True
                )
            else:
                backend = service.get_backend(backend_name)

            # Specify options below if needed
            # backend.set_options(**options)
        else:
            # Fake backend initialization (Aer Simulator)
            try:
                if not use_dynamics:
                    backend = FakeProviderForBackendV2().backend(
                        backend_name if backend_name is not None else "fake_jakarta"
                    )
                else:
                    backend = FakeProvider().get_backend(
                        backend_name if backend_name is not None else "fake_jakarta"
                    )
            except QiskitBackendNotFoundError:
                raise QiskitError(
                    "Backend not found. Please check the backend name and try again."
                )

    if backend is not None:
        if use_dynamics:
            solver_options = convert_solver_options(solver_options)
            assert isinstance(
                backend, BackendV1
            ), "DynamicsBackend can only be used with BackendV1 instances"
            backend = DynamicsBackend.from_backend(
                backend,
                subsystem_list=list(physical_qubits),
                solver_options=solver_options,
            )
            _, _ = perform_standard_calibrations(
                backend, calibration_files=calibration_files
            )

    if backend is None:
        warnings.warn(
            "No backend selected. Training will be performed on Statevector simulator"
        )
    return backend


def convert_solver_options(
    solver_options: Optional[Dict], dt: Optional[float | int] = None
) -> Optional[Dict]:
    """
    Convert solver options passed from YAML to correct format
    """
    if solver_options["hmax"] == "auto" and dt is not None:
        solver_options["hmax"] = dt
    if solver_options["hmax"] == "auto" and dt is None:
        raise ValueError("dt must be specified for hmax='auto'")
    for key in ["atol", "rtol"]:
        solver_options[key] = float(solver_options[key])
    return solver_options


def has_noise_model(backend: AerBackend):
    """
    Check if Aer backend has noise model or not

    Args:
        backend: AerBackend instance
    """
    if (
        backend.options.noise_model is None
        or backend.options.noise_model.to_dict() == {}
        or len(backend.options.noise_model.to_dict()["errors"]) == 0
    ):
        return False
    else:
        return True


def load_q_env_from_yaml_file(file_path: str):
    """
    Load Qiskit Quantum Environment from yaml file

    Args:
        file_path: File path
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    if "ENV" not in config:
        raise KeyError("ENV section must be present in the configuration")
    elif "BACKEND" not in config:
        raise KeyError("BACKEND section must be present in the configuration")
    elif "TARGET" not in config:
        raise KeyError("TARGET section must be present in the configuration")

    env_config = config["ENV"]

    if "ACTION_SPACE" not in env_config:
        raise KeyError("ACTION_SPACE section must be present in the configuration")
    action_space_config = env_config["ACTION_SPACE"]
    if "LOW" not in action_space_config or "HIGH" not in action_space_config:
        raise KeyError("LOW and HIGH must be present in the ACTION_SPACE section")
    if not all(isinstance(val, (int, float)) for val in action_space_config["LOW"]):
        try:
            action_space_config["LOW"] = [
                float(val) for val in action_space_config["LOW"]
            ]
        except ValueError:
            raise ValueError("LOW values in action space must be numeric")
    if not all(isinstance(val, (int, float)) for val in action_space_config["HIGH"]):
        try:
            action_space_config["HIGH"] = [
                float(val) for val in action_space_config["HIGH"]
            ]
        except ValueError:
            raise ValueError("HIGH values in action space must be numeric")
    low = np.array(action_space_config["LOW"], dtype=np.float32)
    high = np.array(action_space_config["HIGH"], dtype=np.float32)
    if low.shape != high.shape:
        raise ValueError(
            "Low and high arrays in action space should have the same shape"
        )
    action_shape = low.shape

    try:
        execution_config = env_config["EXECUTION"]
    except KeyError:
        raise KeyError("EXECUTION section must be present in the configuration")

    params = {
        "action_space": Box(low=low, high=high, shape=action_shape, dtype=np.float32),
        "execution_config": ExecutionConfig(**get_lower_keys_dict(execution_config)),
        "benchmark_config": BenchmarkConfig(
            **(
                get_lower_keys_dict(env_config["BENCHMARKING"])
                if "BENCHMARKING" in env_config
                else {}
            )
        ),
        "reward_config": reward_configs[env_config["REWARD"]["REWARD_METHOD"]](
            **remove_none_values(
                get_lower_keys_dict(
                    env_config.get(
                        "REWARD_PARAMS", env_config["REWARD"].get("REWARD_PARAMS", {})
                    )
                )
            )
        ),
        "training_with_cal": env_config.get("TRAINING_WITH_CAL", False),
        "target": {
            "physical_qubits": config["TARGET"]["PHYSICAL_QUBITS"],
        },
    }
    if "GATE" in config["TARGET"] and config["TARGET"]["GATE"] is not None:
        try:
            params["target"]["gate"] = gate_map()[config["TARGET"]["GATE"].lower()]
        except KeyError:
            raise KeyError("Specified gate not found in standard gate set of Qiskit")
    else:
        try:
            params["target"]["state"] = Statevector.from_label(
                config["TARGET"]["STATE"]
            )
        except KeyError:
            raise KeyError(
                "Target gate or state must be specified in the configuration"
            )

    backend_config = config.get("BACKEND", {})
    dynamics_config = backend_config.get(
        "DYNAMICS",
        {
            "USE_DYNAMICS": None,
            "PHYSICAL_QUBITS": None,
            "SOLVER_OPTIONS": {
                "hmax": "auto",
                "atol": 1e-6,
                "rtol": 1e-8,
                "method": "jax_odeint",
            },
            "CALIBRATION_FILES": None,
        },
    )
    service_config = config.get("SERVICE", {"CHANNEL": None, "INSTANCE": None})
    backend_params = {
        "real_backend": backend_config.get("REAL_BACKEND", None),
        "backend_name": backend_config.get("NAME", None),
        "use_dynamics": dynamics_config["USE_DYNAMICS"],
        "physical_qubits": dynamics_config["PHYSICAL_QUBITS"],
        "channel": service_config["CHANNEL"],
        "instance": service_config["INSTANCE"],
        "solver_options": dynamics_config["SOLVER_OPTIONS"],
        "calibration_files": dynamics_config["CALIBRATION_FILES"],
    }

    runtime_options = config.get("RUNTIME_OPTIONS", {})

    if backend_params["real_backend"]:
        print("Runtime Options:", runtime_options)

    return (
        params,
        backend_params,
        remove_none_values(runtime_options),
    )


def get_lower_keys_dict(dictionary: Dict[str, Any]):
    """
    Get dictionary with lower keys

    Args:
        dictionary: Dictionary
    """
    return {key.lower(): value for key, value in dictionary.items()}


def get_q_env_config(
    config_file_path: str,
    parametrized_circ_func: Callable[
        [QuantumCircuit, ParameterVector, QuantumRegister, Dict[str, Any]], None
    ],
    backend: Optional[Backend_type | Callable[[Any], Backend_type]] = None,
    pass_manager: Optional[PassManager] = PassManager(),
    instruction_durations: Optional[InstructionDurations] = None,
    **backend_callable_args,
):
    """
    Get Qiskit Quantum Environment configuration from yaml file

    Args:
        config_file_path: Configuration file path (yaml, should contain at least ENV and TARGET section)
        parametrized_circ_func: Function to applying parametrized gate (should be defined in your Python config)
        backend: Optional custom backend instance
            (if None, backend will be selected based on configuration set in yaml file)

        backend_callable_args: Additional arguments for backend if it was passed as a callable

    """
    params, backend_params, runtime_options = load_q_env_from_yaml_file(
        config_file_path
    )
    if isinstance(backend, Callable):
        backend = backend(**backend_callable_args)
    elif backend is None:
        backend = select_backend(**backend_params)

    if isinstance(backend, DynamicsBackend):
        backend_config = DynamicsConfig(
            parametrized_circ_func,
            backend,
            pass_manager=pass_manager,
            instruction_durations=instruction_durations,
        )
    else:
        backend_config = QiskitRuntimeConfig(
            parametrized_circ_func,
            backend,
            pass_manager=pass_manager,
            instruction_durations=instruction_durations,
            estimator_options=(
                runtime_options if isinstance(backend, RuntimeBackend) else None
            ),
        )

    q_env_config = QEnvConfig(backend_config=backend_config, **params)
    return q_env_config


def remove_none_values(dictionary: Dict):
    """
    Remove None values from dictionary

    Args:
        dictionary: Dictionary
    """
    new_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            v = remove_none_values(v)
        if v is not None:
            new_dict[k] = v
    return new_dict


def load_from_yaml_file(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_from_pickle(file_path: str):
    """Load data from a pickle or gzip file."""
    try:
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rb") as file:
                data = pickle.load(file)
        else:
            with open(file_path, "rb") as file:
                data = pickle.load(file)
    except Exception as e:
        logging.warning(f"Failed to open file {file_path}")
        logging.warning(f"Error Message: {e}")
        return None
    return data


def save_to_pickle(data, file_path: str) -> None:
    """Save data as a pickle or gzip file."""
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    try:
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "wb") as file:
                pickle.dump(data, file)
        else:
            with open(file_path, "wb") as file:
                pickle.dump(data, file)
    except Exception as e:
        logging.warning(f"Failed to save file {file_path}")
        logging.warning(f"Error Message: {e}")


def create_hpo_agent_config(
    trial: optuna.trial.Trial, hpo_config: Dict, path_to_agent_config: str
):
    hyper_params = {}
    hyperparams_in_scope = []

    # Loop through hpo_config and decide whether to optimize or use the provided value
    for param, values in hpo_config.items():
        if isinstance(values, list):
            if len(values) == 2:  # If values is a list of length 2, optimize
                if isinstance(values[0], int):
                    hyper_params[param] = trial.suggest_int(param, values[0], values[1])
                elif isinstance(values[0], float):
                    if param == "LR":  # If learning rate, suggest in log scale
                        hyper_params[param] = trial.suggest_float(
                            param, values[0], values[1], log=True
                        )
                    else:
                        hyper_params[param] = trial.suggest_float(
                            param, values[0], values[1]
                        )
                hyperparams_in_scope.append(param)
            elif (
                len(values) > 2
            ):  # If values is a list of more than 2, choose from list and optimize
                hyper_params[param] = trial.suggest_categorical(param, values)
                hyperparams_in_scope.append(param)
        else:
            hyper_params[param] = values

    # Dynamically calculate batchsize from minibatch_size and num_minibatches
    print("MINIBATCH_SIZE", hyper_params["MINIBATCH_SIZE"])
    print("NUM_MINIBATCHES", hyper_params["NUM_MINIBATCHES"])
    hyper_params["BATCHSIZE"] = (
        hyper_params["MINIBATCH_SIZE"] * hyper_params["NUM_MINIBATCHES"]
    )

    # Print hyperparameters considered for HPO
    print("Hyperparameters considered for HPO:", hyperparams_in_scope)

    # Print hyperparameters NOT considered for HPO
    hyperparams_not_in_scope = [
        param for param in hpo_config if param not in hyperparams_in_scope
    ]
    print("Hyperparameters NOT in scope of HPO:", hyperparams_not_in_scope)

    # Take over attributes from agent_config and populate hyper_params
    agent_config = load_from_yaml_file(path_to_agent_config)
    final_config = hyper_params.copy()
    final_config.update(agent_config)
    final_config.update(hyper_params)

    return final_config, hyperparams_in_scope


def get_hardware_runtime_single_circuit(
    qc: QuantumCircuit,
    instruction_durations_dict: Dict[Tuple[str, Tuple[int, ...]], Tuple[float, str]],
):
    total_time_per_qubit = {qubit: 0.0 for qubit in qc.qubits}

    for instruction in qc.data:
        qubits_involved = instruction.qubits
        gate_name: str = (
            instruction.operation.name
            if not instruction.operation.label
            else instruction.operation.label
        )

        if len(qubits_involved) == 1:
            qbit1 = qubits_involved[0]
            qbit1_index = qc.find_bit(qbit1)[0]
            key = (gate_name, (qbit1_index,))
            if key in instruction_durations_dict:
                gate_time = instruction_durations_dict[key][0]
                total_time_per_qubit[qbit1] += gate_time

        elif len(qubits_involved) == 2:
            qbit1, qbit2 = qubits_involved
            qbit1_index = qc.find_bit(qbit1)[0]
            qbit2_index = qc.find_bit(qbit2)[0]
            key = (gate_name, (qbit1_index, qbit2_index))
            if key in instruction_durations_dict:
                gate_time = instruction_durations_dict[key][0]
                for qbit in qubits_involved:
                    total_time_per_qubit[qbit] += gate_time

        else:
            raise NotImplementedError(
                "Hardware runtimes of 3-qubit gates are not implemented currently."
            )

    # Find the maximum execution time among all qubits
    total_execution_time = (
        max(total_time_per_qubit.values())
        + instruction_durations_dict[("reset", (0,))][
            0
        ]  # Reset time is the same for all qubits
        + instruction_durations_dict[("measure", (0,))][
            0
        ]  # Reset time is the same for all qubits
    )

    return total_execution_time


def get_hardware_runtime_cumsum(
    qc: QuantumCircuit, circuit_gate_times: Dict, total_shots: List[int]
) -> np.array:
    return np.cumsum(
        get_hardware_runtime_single_circuit(qc, circuit_gate_times)
        * np.array(total_shots)
    )


def retrieve_neighbor_qubits(coupling_map: CouplingMap, target_qubits: List):
    """
    Retrieve neighbor qubits of target qubits

    Args:
        coupling_map: Coupling map
        target_qubits: Target qubits

    Returns:
        neighbor_qubits: List of neighbor qubits indices for specified target qubits
    """
    neighbors = set()
    for target_qubit in target_qubits:
        neighbors.update(coupling_map.neighbors(target_qubit))
    return list(neighbors - set(target_qubits))


def retrieve_tgt_instruction_count(qc: QuantumCircuit, target: Dict):
    """
    Retrieve count of target instruction in Quantum Circuit

    Args:
        qc: Quantum Circuit (ideally already transpiled)
        target: Target in form of {"gate": "X", "physical_qubits": [0, 1]}
    """
    tgt_instruction = CircuitInstruction(
        target["gate"], [qc.qubits[i] for i in target["physical_qubits"]]
    )
    return qc.data.count(tgt_instruction)


def generate_default_instruction_durations_dict(
    n_qubits: int,
    single_qubit_gate_time: float,
    two_qubit_gate_time: float,
    circuit_gate_times: Dict,
    virtual_gates: Optional[List] = None,
):
    """
    Generates a dictionary of default instruction durations for each gate and qubit combination. This allows for calculating the total execution time of a quantum circuit.
    In particular, the metric of hardware runtime becomes relevant to benchmark the performance of different methods for the same calibration task.

    Args:
        n_qubits (int): The number of qubits in the quantum circuit.
        single_qubit_gate_time (float): The duration of a single-qubit gate.
        two_qubit_gate_time (float): The duration of a two-qubit gate.
        circuit_gate_times (dict): A dictionary mapping gate names to their respective durations.
        virtual_gates (list): A list of gates that are performed by software and have zero duration.

    Returns:
        dict: A dictionary where the keys are tuples of the form (gate, qubits) and the values are tuples of the form (duration, unit).
              The duration is the default duration for the gate and qubit combination, and the unit is the time unit (e.g., 's' for seconds).

    """
    default_instruction_durations_dict = {}

    # Identify single-qubit and two-qubit gates
    single_qubit_gates = []
    two_qubit_gates = []

    for gate in circuit_gate_times:
        if virtual_gates is not None and gate in virtual_gates:
            continue
        if gate == "measure" or gate == "reset":
            continue
        if circuit_gate_times[gate] == single_qubit_gate_time:
            single_qubit_gates.append(gate)
        elif circuit_gate_times[gate] == two_qubit_gate_time:
            two_qubit_gates.append(gate)

    # Single qubit gates
    for gate in single_qubit_gates:
        for qubit in range(n_qubits):
            default_instruction_durations_dict[(gate, (qubit,))] = (
                circuit_gate_times[gate],
                "s",
            )

    # Two qubit gates (assuming all-to-all connectivity)
    for gate in two_qubit_gates:
        for qubit1 in range(n_qubits):
            for qubit2 in range(n_qubits):
                if qubit1 != qubit2:
                    default_instruction_durations_dict[(gate, (qubit1, qubit2))] = (
                        two_qubit_gate_time,
                        "s",
                    )

    # Reset and Measure operations
    for qubit in range(n_qubits):
        default_instruction_durations_dict[("measure", (qubit,))] = (
            circuit_gate_times["measure"],
            "s",
        )
        default_instruction_durations_dict[("reset", (qubit,))] = (
            circuit_gate_times["reset"],
            "s",
        )

    # Gates done by software
    if virtual_gates is not None:
        for gate in virtual_gates:
            for qubit in range(n_qubits):
                default_instruction_durations_dict[(gate, (qubit,))] = (0.0, "s")

    return default_instruction_durations_dict
