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
    return [clip_by_value(m, -1.0, 1.0) for m in mu_var]


def constrain_std_value(std_var):
    return [clip_by_value(std, 1e-3, 3) for std in std_var]


def generate_model(
    input_shape: Tuple,
    hidden_units: Union[List, Tuple],
    n_actions: int,
    actor_critic_together: bool = True,
    hidden_units_critic: Optional[Union[List, Tuple]] = None,
):
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

    mean_param = Dense(n_actions, activation="tanh", name="mean_vec")(Net)  # Mean vector output
    sigma_param = Dense(n_actions, activation="softplus", name="sigma_vec")(
        Net
    )  # Diagonal elements of cov matrix
    # output

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
