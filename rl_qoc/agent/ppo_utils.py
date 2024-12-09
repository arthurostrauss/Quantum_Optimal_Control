import torch.nn as nn
import torch.optim as optim

module_dict = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
    "none": nn.ReLU,
    "softmax": nn.Softmax,
    "log_softmax": nn.LogSoftmax,
    "gelu": nn.GELU,
    "identity": nn.Identity,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "hardshrink": nn.Hardshrink,
    "tanhshrink": nn.Tanhshrink,
    "softshrink": nn.Softshrink,
    "hardtanh": nn.Hardtanh,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "celu": nn.CELU,
    "linear": nn.Linear,
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "conv3d": nn.Conv3d,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
    "rnn": nn.RNN,
    "lstm_cell": nn.LSTMCell,
    "gru_cell": nn.GRUCell,
    "rnn_cell": nn.RNNCell,
}

reverse_module_dict = {v: k for k, v in module_dict.items()}

optim_dict = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
    "sgd": optim.SGD,
}

reverse_optim_dict = {v: k for k, v in optim_dict.items()}


def get_module(module: str | nn.Module) -> nn.Module:
    """
    Converts a string representation of a module to the corresponding PyTorch module class.

    Args:
        module (str): The string representation of the module.

    Returns:
        torch.nn.Module: The PyTorch module class corresponding to the input string.

    Raises:
        ValueError: If the input string does not match any of the supported module names.
    """

    if isinstance(module, str) and module not in module_dict:
        raise ValueError(
            f"Agent Config `ACTIVATION` needs to be one of {module_dict.keys()}"
        )
    elif not isinstance(module, (nn.Module, str)):
        raise ValueError("Activation function must be a string or a torch.nn.Module")

    return module_dict[module]() if isinstance(module, str) else module


def get_optimizer(optimizer: str):
    """
    Returns the optimizer class corresponding to the given optimizer string.

    Args:
        optimizer (str): The optimizer string.

    Returns:
        torch.optim.Optimizer: The optimizer class.

    Raises:
        ValueError: If the optimizer string is not valid.
    """

    if isinstance(optimizer, str) and optimizer not in optim_dict:
        raise ValueError(
            f"Agent Config `OPTIMIZER` needs to be one of {optim_dict.keys()}"
        )

    return optim_dict[optimizer]
