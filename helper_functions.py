import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, List, Union, Dict
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import Model
from qiskit.providers import BackendV1
from qiskit.quantum_info import Operator
from qiskit_dynamics import Solver, DynamicsBackend, RotatingFrame
from qiskit_dynamics.array import Array
from qiskit.exceptions import QiskitError
from qiskit_dynamics.backend.backend_string_parser.hamiltonian_string_parser import parse_backend_hamiltonian_dict
from qiskit_dynamics.backend.dynamics_backend import _get_backend_channel_freqs


def constrain_mean_value(mu_var):
    return [tf.clip_by_value(m, -1., 1.) for m in mu_var]


def constrain_std_value(std_var):
    return [tf.clip_by_value(std, 1e-3, 3) for std in std_var]


def get_control_channel_map(backend: BackendV1, qubit_tgt_register: List[int]):
    """
    Get reduced control_channel_map from Backend configuration (needs to be of type BackendV1)
    :param backend: IBM Backend instance, must carry a configuration method
    :param qubit_tgt_register: Subsystem of interest from which to build control_channel_map
    """
    control_channel_map = {}
    control_channel_map_backend = {
        **{qubits: backend.configuration().control_channels[qubits][0].index for qubits in
           backend.configuration().control_channels}}
    for qubits in control_channel_map_backend:
        if qubits[0] in qubit_tgt_register and qubits[1] in qubit_tgt_register:
            control_channel_map[qubits] = control_channel_map_backend[qubits]
    return control_channel_map


def get_solver_and_freq_from_backend(backend: BackendV1,
                                     subsystem_list: Optional[List[int]] = None,
                                     rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
                                     evaluation_mode: str = "dense",
                                     rwa_cutoff_freq: Optional[float] = None,
                                     static_dissipators: Optional[Array] = None,
                                     dissipator_operators: Optional[Array] = None,
                                     dissipator_channels: Optional[List[str]] = None, ) \
        -> Tuple[Dict[str, float], Solver]:
    """
    Method to retrieve solver instance and relevant freq channels information from an IBM
    backend added with potential dissipation operators, inspired from DynamicsBackend.from_backend() method
    :param subsystem_list: The list of qubits in the backend to include in the model.
    :param rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`
    :param evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
    :param rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
            ``"auto"``, allowing this method to pick a rotating frame.
    :param backend: IBMBackend instance from which Hamiltonian parameters are extracted
    :param static_dissipators: static_dissipators: Constant dissipation operators.
    :param dissipator_operators: Dissipation operators with time-dependent coefficients.
    :param dissipator_channels: List of channel names in pulse schedules corresponding to dissipator operators.

    :return: Solver instance carrying Hamiltonian information extracted from the IBMBackend instance
    """
    # get available target, config, and defaults objects
    backend_target = getattr(backend, "target", None)

    if not hasattr(backend, "configuration"):
        raise QiskitError(
            "DynamicsBackend.from_backend requires that the backend argument has a "
            "configuration method."
        )
    backend_config = backend.configuration()

    backend_defaults = None
    if hasattr(backend, "defaults"):
        backend_defaults = backend.defaults()

    # get and parse Hamiltonian string dictionary
    if backend_target is not None:
        backend_num_qubits = backend_target.num_qubits
    else:
        backend_num_qubits = backend_config.n_qubits

    if subsystem_list is not None:
        subsystem_list = sorted(subsystem_list)
        if subsystem_list[-1] >= backend_num_qubits:
            raise QiskitError(
                f"subsystem_list contained {subsystem_list[-1]}, which is out of bounds for "
                f"backend with {backend_num_qubits} qubits."
            )
    else:
        subsystem_list = list(range(backend_num_qubits))

    if backend_config.hamiltonian is None:
        raise QiskitError(
            "get_solver_from_backend requires that backend.configuration() has a "
            "hamiltonian."
        )

    (
        static_hamiltonian,
        hamiltonian_operators,
        hamiltonian_channels,
        subsystem_dims,
    ) = parse_backend_hamiltonian_dict(backend_config.hamiltonian, subsystem_list)

    # construct model frequencies dictionary from backend
    channel_freqs = _get_backend_channel_freqs(
        backend_target=backend_target,
        backend_config=backend_config,
        backend_defaults=backend_defaults,
        channels=hamiltonian_channels,
    )

    # build the solver
    if rotating_frame == "auto":
        if "dense" in evaluation_mode:
            rotating_frame = static_hamiltonian
        else:
            rotating_frame = np.diag(static_hamiltonian)

    # get time step size
    if backend_target is not None and backend_target.dt is not None:
        dt = backend_target.dt
    else:
        # config is guaranteed to have a dt
        dt = backend_config.dt

    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=hamiltonian_operators,
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_freqs,
        dt=dt,
        rotating_frame=rotating_frame,
        evaluation_mode=evaluation_mode,
        rwa_cutoff_freq=rwa_cutoff_freq,
        static_dissipators=static_dissipators,
        dissipator_operators=dissipator_operators,
        dissipator_channels=dissipator_channels
    )

    return channel_freqs, solver


def select_optimizer(lr: float, optimizer: str = "Adam", grad_clip: Optional[float] = None,
                     concurrent_optimization: bool = True, lr2: Optional[float] = None):
    if concurrent_optimization:
        if optimizer == 'Adam':
            return tf.optimizers.Adam(learning_rate=lr, clipvalue=grad_clip)
        elif optimizer == 'SGD':
            return tf.optimizers.SGD(learning_rate=lr, clipvalue=grad_clip)
    else:
        if optimizer == 'Adam':
            return tf.optimizers.Adam(learning_rate=lr), tf.optimizers.Adam(learning_rate=lr2, clipvalue=grad_clip)
        elif optimizer == 'SGD':
            return tf.optimizers.SGD(learning_rate=lr), tf.optimizers.SGD(learning_rate=lr2, clipvalue=grad_clip)


def generate_model(input_shape: Tuple, hidden_units: Union[List, Tuple], n_actions: int,
                   actor_critic_together: bool = True,
                   hidden_units_critic: Optional[Union[List, Tuple]] = None):
    """
    Helper function to generate fully connected NN
    :param input_shape: Input shape of the NN
    :param hidden_units: List containing number of neurons per hidden layer
    :param n_actions: Output shape of the NN on the actor part, i.e. dimension of action space
    :param actor_critic_together: Decide if actor and critic network should be distinct or should be sharing layers
    :param hidden_units_critic: If actor_critic_together set to False, List containing number of neurons per hidden
           layer for critic network
    :return: Model or Tuple of two Models for actor critic network
    """
    input_layer = Input(shape=input_shape)
    Net = Dense(hidden_units[0], activation='relu', input_shape=input_shape,
                kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{0}")(input_layer)
    for i in range(1, len(hidden_units)):
        Net = Dense(hidden_units[i], activation='relu', kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                    bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{i}")(Net)

    mean_param = Dense(n_actions, activation='tanh', name='mean_vec')(Net)  # Mean vector output
    sigma_param = Dense(n_actions, activation="softplus", name="sigma_vec")(Net)  # Diagonal elements of cov matrix
    # output

    if actor_critic_together:
        critic_output = Dense(1, activation='linear', name="critic_output")(Net)
        return Model(inputs=input_layer, outputs=[mean_param, sigma_param, critic_output])
    else:
        assert hidden_units_critic is not None, "Network structure for critic network not provided"
        input_critic = Input(shape=input_shape)
        Critic_Net = Dense(hidden_units_critic[0], activation='relu', input_shape=input_shape,
                           kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{0}")(input_critic)
        for i in range(1, len(hidden_units)):
            Critic_Net = Dense(hidden_units[i], activation='relu',
                               kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                               bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{i}")(
                Critic_Net)
            critic_output = Dense(1, activation='linear', name="critic_output")(Critic_Net)
            return Model(inputs=input_layer, outputs=[mean_param, sigma_param]), \
                Model(inputs=input_critic, outputs=critic_output)
