from typing import Optional, Tuple, Union, List
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.initializers.initializers_v2 import RandomNormal


def select_optimizer(
    lr: float,
    optimizer: str = "Adam",
    grad_clip: Optional[float] = None,
    concurrent_optimization: bool = True,
    lr2: Optional[float] = None,
):
    """
    Selects an optimizer for the actor and critic networks.

    Args:
        lr: The learning rate for the optimizer.
        optimizer: The name of the optimizer to use.
        grad_clip: The value to clip the gradients to.
        concurrent_optimization: Whether to use the same optimizer for the actor and critic.
        lr2: The learning rate for the critic optimizer if `concurrent_optimization` is False.

    Returns:
        The selected optimizer(s).
    """
    if concurrent_optimization:
        if optimizer == "Adam":
            return Adam(learning_rate=lr, clipvalue=grad_clip)
        elif optimizer == "SGD":
            return SGD(learning_rate=lr, clipvalue=grad_clip)
    else:
        if optimizer == "Adam":
            return Adam(learning_rate=lr), Adam(learning_rate=lr2, clipvalue=grad_clip)
        elif optimizer == "SGD":
            return SGD(learning_rate=lr), SGD(learning_rate=lr2, clipvalue=grad_clip)


def constrain_mean_value(mu_var):
    """
    Constrains the mean value of a variable to be between -1 and 1.

    Args:
        mu_var: The variable to constrain.

    Returns:
        The constrained variable.
    """
    return [clip_by_value(m, -1.0, 1.0) for m in mu_var]


def constrain_std_value(std_var):
    """
    Constrains the standard deviation of a variable to be between 1e-3 and 3.

    Args:
        std_var: The variable to constrain.

    Returns:
        The constrained variable.
    """
    return [clip_by_value(std, 1e-3, 3) for std in std_var]


def generate_model(
    input_shape: Tuple,
    hidden_units: Union[List, Tuple],
    n_actions: int,
    actor_critic_together: bool = True,
    hidden_units_critic: Optional[Union[List, Tuple]] = None,
):
    """
    Generates a fully connected neural network model.

    Args:
        input_shape: The input shape of the model.
        hidden_units: A list of the number of units in each hidden layer.
        n_actions: The number of actions in the output layer.
        actor_critic_together: Whether to use the same model for the actor and critic.
        hidden_units_critic: A list of the number of units in each hidden layer of the critic network if
            `actor_critic_together` is False.

    Returns:
        The generated model(s).
    """
    input_layer = Input(shape=input_shape)
    Net = Dense(
        hidden_units[0],
        activation="relu",
        input_shape=input_shape,
        kernel_initializer=RandomNormal(stddev=0.1),
        bias_initializer=RandomNormal(stddev=0.5),
        name=f"hidden_{0}",
    )(input_layer)
    for i in range(1, len(hidden_units)):
        Net = Dense(
            hidden_units[i],
            activation="relu",
            kernel_initializer=RandomNormal(stddev=0.1),
            bias_initializer=RandomNormal(stddev=0.5),
            name=f"hidden_{i}",
        )(Net)

    mean_param = Dense(n_actions, activation="tanh", name="mean_vec")(Net)
    sigma_param = Dense(n_actions, activation="softplus", name="sigma_vec")(
        Net
    )

    if actor_critic_together:
        critic_output = Dense(1, activation="linear", name="critic_output")(Net)
        return Model(inputs=input_layer, outputs=[mean_param, sigma_param, critic_output])
    else:
        assert hidden_units_critic is not None, "Network structure for critic network not provided"
        input_critic = Input(shape=input_shape)
        Critic_Net = Dense(
            hidden_units_critic[0],
            activation="relu",
            input_shape=input_shape,
            kernel_initializer=RandomNormal(stddev=0.1),
            bias_initializer=RandomNormal(stddev=0.5),
            name=f"hidden_{0}",
        )(input_critic)
        for i in range(1, len(hidden_units)):
            Critic_Net = Dense(
                hidden_units[i],
                activation="relu",
                kernel_initializer=RandomNormal(stddev=0.1),
                bias_initializer=RandomNormal(stddev=0.5),
                name=f"hidden_{i}",
            )(Critic_Net)
            critic_output = Dense(1, activation="linear", name="critic_output")(Critic_Net)
            return Model(inputs=input_layer, outputs=[mean_param, sigma_param]), Model(
                inputs=input_critic, outputs=critic_output
            )
