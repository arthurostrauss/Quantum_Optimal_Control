import functools
import qm.qua as qua
import sympy as sp
import symengine as se
from functools import partial
from qualang_tools.video_mode.videomode import ParameterValue

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
    sympy_expr: sp.Basic, parameter_val: ParameterValue
) -> qua.QuaVariableType:
    """
    Convert a Sympy expression to a QuaVariableType

    Args:
        sympy_expr: Sympy expression to convert
        parameter_val: ParameterValue instance
            (contains the name and QUA variable for the parameter)

    Returns:
        QuaVariableType: The equivalent QUA variable transformed from the sympy expression
    """
    # Convert sympy_expr that could be from symengine to actual sympy
    if isinstance(sympy_expr, se.Basic):
        sympy_expr = sp.sympify(str(sympy_expr))
    if isinstance(sympy_expr, sp.Symbol):
        return parameter_val.var
    if isinstance(sympy_expr, sp.Number):
        return sympy_expr.evalf()
    if isinstance(sympy_expr, sp.Mul):
        # handle multiplication for arbitrary number of terms
        return functools.reduce(
            lambda x, y: x * y,
            [sympy_to_qua(term, parameter_val) for term in sympy_expr.args],
        )
    if isinstance(sympy_expr, sp.Add):
        # handle addition for arbitrary number of terms
        return functools.reduce(
            lambda x, y: x + y,
            [sympy_to_qua(term, parameter_val) for term in sympy_expr.args],
        )
    if isinstance(sympy_expr, sp.Pow):
        return sympy_to_qua_dict[type(sympy_expr)](
            sympy_to_qua(sympy_expr.args[0], parameter_val),
            sympy_to_qua(sympy_expr.args[1], parameter_val),
        )
    if isinstance(sympy_expr, sp.Function):
        qua_func = match_expr(sympy_expr)
        if qua_func:
            return qua_func(sympy_to_qua(sympy_expr.args[0], parameter_val))

    raise ValueError(f"Unsupported sympy expression: {sympy_expr}")
