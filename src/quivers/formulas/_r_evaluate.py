"""R expressions evaluated on the columns of a data frame.

brms evaluates the arguments of addition terms in the data, so
``se(sqrt(1 / (N - 3)))`` is the standard error computed from the column
``N``. The expression is read by the shared R expression reader and applied
to the columns through the compiled QVR value expression.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from quivers.dsl._value_expressions import _parse_value_expression
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.dsl.let_expr_traversal import free_let_names
from quivers.formulas._r_expression import translate_r_expression


def evaluate_r_expression(
    text: str, frame: pd.DataFrame, *, context: str
) -> npt.NDArray[np.float64]:
    """Evaluate an R expression over numeric columns, one value per row.

    Parameters
    ----------
    text : str
        R expression over column names, numbers, arithmetic and the
        functions of the shared reader, such as ``sqrt(1 / (N - 3))``.
    frame : pandas.DataFrame
        Rows whose columns the expression names.
    context : str
        Description of the argument holding the expression, named in errors.

    Returns
    -------
    numpy.ndarray
        Float64 values, one per row. A missing value in a column gives a
        missing value in the result.

    Raises
    ------
    ValueError
        If the expression is malformed, names a column that does not exist
        or is not numeric, or does not evaluate to one real number per row.

    Notes
    -----
    Cost is linear in the number of rows.
    """
    columns = frozenset(str(name) for name in frame.columns)
    # R reads numeric data as doubles, so every column enters as Float64.
    translated = translate_r_expression(text, context=context, variables=columns)
    unknown = sorted(set(translated.names) - columns)
    if unknown:
        raise ValueError(
            f"{context}: {text!r} names {unknown!r}, which are not dataframe columns"
        )
    node = _parse_value_expression(translated.expression)
    environment: dict[str, torch.Tensor] = {}
    for name in free_let_names(node):
        values = np.asarray(frame[name])
        if values.dtype.kind not in "biuf":
            raise ValueError(f"{context}: column {name!r} of {text!r} is not numeric")
        environment[name] = torch.as_tensor(
            values.astype(np.float64), dtype=torch.float64
        )
    result = torch.as_tensor(
        _ProgramsMixin._compile_let_expr(node)(environment), dtype=torch.float64
    )
    if result.ndim == 0:
        result = result.expand(len(frame))
    if tuple(result.shape) != (len(frame),):
        raise ValueError(f"{context}: {text!r} must give one value per row")
    return result.numpy().copy()


__all__: list[str] = []
