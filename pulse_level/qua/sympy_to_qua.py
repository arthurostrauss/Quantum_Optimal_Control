import functools
from typing import List, Dict

import qm.qua as qua
import sympy as sp
import symengine as se
from functools import partial

from qm.qua import fixed, declare, assign
from qualang_tools.video_mode.videomode import ParameterValue
from sympy import Symbol

sympy_to_qua_dict = {
    sp.Float: qua.fixed,
    sp.Integer: int,
    sp.Abs: qua.Math.abs,
    sp.cos: qua.Math.cos,
    sp.sin: qua.Math.sin,
    sp.exp: qua.Math.exp,
    sp.ln: qua.Math.ln,
    sp.log: qua.Math.log,
    partial(sp.log, 10): qua.Math.log10,
    partial(sp.log, 2): qua.Math.log2,
    partial(sp.Pow, -1): qua.Math.inv,
    partial(sp.Pow, 0.5): qua.Math.sqrt,
    sp.Pow: qua.Math.pow,
    sp.sqrt: qua.Math.sqrt,
}


def match_expr(expr: sp.Function):
    """
    Match a sympy expression to a Qua function
    """
    for key, value in sympy_to_qua_dict.items():
        if isinstance(key, partial):
            if expr.func == key.func and expr.args[1] == key.args[0]:
                return value
        elif isinstance(expr, key):
            return value
    raise ValueError(f"Unsupported sympy expression: {expr}")


def sympy_to_qua(
    sympy_expr: sp.Basic, parameter_vals: Dict[str, ParameterValue]
) -> qua.QuaVariableType:
    """
    Convert a Sympy expression to a QuaVariableType

    Args:
        sympy_expr: Sympy expression to convert (could contain multiple Parameters)
        parameter_vals: Dictionary of parameter values (contain name and QUA variable)
    Returns:
        QuaVariableType: The equivalent QUA variable transformed from the sympy expression
    """
    # Convert sympy_expr that could be from symengine to actual sympy
    if any([isinstance(param.type, fixed) for param in parameter_vals.values()]):
        new_val = declare(fixed)
    else:
        new_val = declare(int)

    sympy_to_qua_dict = {}
    for symbol in sympy_expr.free_symbols:
        assert (
            symbol.name in parameter_vals
        ), f"Parameter {symbol.name} not found in parameter_vals"
        sympy_to_qua_dict[symbol] = parameter_vals[symbol.name].var

    if isinstance(sympy_expr, se.Basic):
        sympy_expr = sp.sympify(str(sympy_expr))
    if isinstance(sympy_expr, sp.Symbol):
        result = sympy_to_qua_dict[sympy_expr]

    elif isinstance(sympy_expr, sp.Number):
        result = sympy_expr.evalf()

    elif isinstance(sympy_expr, sp.Mul):
        # handle multiplication for arbitrary number of terms
        result = functools.reduce(
            lambda x, y: x * y,
            [sympy_to_qua(term, parameter_vals) for term in sympy_expr.args],
        )

    elif isinstance(sympy_expr, sp.Add):
        # handle addition for arbitrary number of terms
        result = functools.reduce(
            lambda x, y: x + y,
            [sympy_to_qua(term, parameter_vals) for term in sympy_expr.args],
        )
    elif isinstance(sympy_expr, sp.Pow):
        result = sympy_to_qua_dict[type(sympy_expr)](
            sympy_to_qua(sympy_expr.args[0], parameter_vals),
            sympy_to_qua(sympy_expr.args[1], parameter_vals),
        )
    elif isinstance(sympy_expr, sp.Function):
        qua_func = match_expr(sympy_expr)
        if qua_func:
            result = qua_func(sympy_to_qua(sympy_expr.args[0], parameter_vals))
    else:
        raise ValueError(f"Unsupported sympy expression: {sympy_expr}")
    assign(new_val, result)
    return new_val
