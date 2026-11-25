"""Named predictor dependencies over the existing formula and QVR expression IRs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from graphlib import CycleError, TopologicalSorter
import re
from typing import Literal

import didactic.api as dx
from formulae.matrices import DesignMatrices
import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import LetExprNode, LetExprVar
from quivers.dsl.let_expr_traversal import free_let_names, substitute_let_expr
from quivers.formulas.formula import (
    Formula,
    FixedColumn,
    _qvr_name,
)
from quivers.formulas._r_expression import translate_r_expression


class NonlinearExpression(ValueModel):
    """A deterministic QVR value expression, rather than an additive design.

    Parameters
    ----------
    expression : str or Mapping[str or int, str]
        Scalar expression or explicit component-label mapping using named
        nonlinear parameters and numeric data columns.
        Distributional expressions describe the family's link scale; inverse
        links are applied once, after evaluation. Intermediate nonlinear
        parameters have the identity link.
    syntax : {"qvr", "r"}
        Language of the expression text. ``"r"`` reads an R expression over
        Stan's functions, as brms ``nlf()`` takes, with numeric data read as
        doubles.
    """

    expression: str | dict[str | int, str] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    syntax: Literal["qvr", "r"] = "qvr"


class ComponentExpression(ValueModel):
    """An explicitly labelled vector value whose coordinates keep nominal order.

    Parameters
    ----------
    components : tuple[str, ...]
        Authored coordinate labels, never inferred from extent equality.
    values : tuple[LetExprNode, ...]
        Scalar expressions aligned to those coordinate labels.
    """

    components: tuple[str, ...]
    values: tuple[LetExprNode, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class PredictorGraph(ValueModel):
    """Retain additive parameter designs and a checked deterministic dependency DAG.

    Parameters
    ----------
    leaves : dict[str, Formula]
        Nonlinear parameters whose coefficients use ordinary formula emission.
    expressions : dict[str, LetExprNode | ComponentExpression]
        Deterministic intermediate nodes in dependency order.
    roots : dict[str, LetExprNode | ComponentExpression]
        Family link-scale outputs, keyed by distributional parameter.
    inputs : dict[str, FixedColumn]
        Numeric raw columns with their exact fitted carriers and row axis.
    """

    leaves: dict[str, Formula] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    expressions: dict[str, LetExprNode | ComponentExpression] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    roots: dict[str, LetExprNode | ComponentExpression] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    inputs: dict[str, FixedColumn] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


def nonlinear_response_source(formula: str, nonlinear: Mapping | None) -> str:
    """Preserve response modifiers while deferring a nonlinear RHS to value parsing.

    Parameters
    ----------
    formula : str
        Authored response formula.
    nonlinear : Mapping or None
        Explicit activation of value-expression semantics, including empty maps.

    Returns
    -------
    str
        Existing response/layout parser input without spurious coefficients.

    Raises
    ------
    ValueError
        If an activated nonlinear formula has no tilde.
    """
    if nonlinear is None:
        return formula
    if "~" not in formula:
        raise ValueError("nonlinear response formula requires '~'")
    return formula.split("~", 1)[0] + " ~ 0"


def _parse_expression(text: str) -> LetExprNode:
    """Parse exactly one expression through the canonical QVR parser.

    Parameters
    ----------
    text : str
        Authored deterministic expression.

    Returns
    -------
    LetExprNode
        Canonical existing expression AST.

    Raises
    ------
    ValueError
        If the text is empty or contains statement boundaries.
    """
    from quivers.dsl._value_expressions import _parse_value_expression

    return _parse_value_expression(text)


def _value_expression(
    value: NonlinearExpression,
    name: str,
    integer_names: frozenset[str],
    variables: frozenset[str],
):
    """Parse scalar or explicitly labelled expressions through the shared parser.

    Parameters
    ----------
    value : NonlinearExpression
        Authored scalar expression or component mapping.
    name : str
        Parameter the expression defines, named in errors.
    integer_names : frozenset[str]
        Data columns with an integer or Boolean carrier.
    variables : frozenset[str]
        Data columns and nonlinear parameters.

    Returns
    -------
    LetExprNode or ComponentExpression
        Canonical scalar AST or labelled vector ASTs.

    Raises
    ------
    ValueError
        If coordinate labels are empty or collide after string normalization.
    """

    def parse(item: str) -> LetExprNode:
        """Parse one expression in the value's declared syntax.

        Parameters
        ----------
        item : str
            Expression text.

        Returns
        -------
        LetExprNode
            Canonical QVR expression AST.
        """
        if value.syntax == "qvr":
            return _parse_expression(item)
        return _parse_expression(
            translate_r_expression(
                item,
                context=f"nonlinear formula of {name!r}",
                integer_names=integer_names,
                variables=variables,
            ).expression
        )

    text = value.expression
    if isinstance(text, str):
        return parse(text)
    labels = tuple(str(label) for label in text)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("nonlinear component labels must be nonempty and unique")
    return ComponentExpression(
        components=labels,
        values=tuple(parse(item) for item in text.values()),
    )


def _value_names(value):
    """Collect dependencies from every coordinate of one typed graph value.

    Parameters
    ----------
    value : LetExprNode or ComponentExpression
        Scalar or labelled expression.

    Returns
    -------
    set[str]
        Free value names, excluding function callees.
    """
    values = value.values if isinstance(value, ComponentExpression) else (value,)
    return set().union(*(set(free_let_names(item)) for item in values))


def _substitute_value(value, substitution):
    """Apply the same capture-safe QVR substitution to every labelled coordinate.

    Parameters
    ----------
    value : LetExprNode or ComponentExpression
        Scalar or explicitly labelled vector expression.
    substitution : Mapping
        Canonical parameter and covariate bindings.

    Returns
    -------
    LetExprNode or ComponentExpression
        Value retaining component labels and authored ordering.
    """
    if isinstance(value, ComponentExpression):
        return value.with_(
            values=tuple(
                substitute_let_expr(item, substitution) for item in value.values
            )
        )
    return substitute_let_expr(value, substitution)


def build_predictor_graph(
    formula: str,
    nonlinear: Mapping[str, str | NonlinearExpression] | None,
    distributional: Mapping[str, str | NonlinearExpression] | None,
    data: IntoDataFrame,
    response: str,
    *,
    family_parameters: frozenset[str],
    evaluate: Callable[[str], tuple[Formula, DesignMatrices]],
    designs: dict[str, DesignMatrices] | None = None,
) -> PredictorGraph | None:
    """Resolve parameter dependencies once and retain their existing design evaluators.

    Parameters
    ----------
    formula : str
        Authored response formula, whose RHS is nonlinear when activated.
    nonlinear : Mapping or None
        Additive parameter formulas or explicit derived value expressions.
    distributional : Mapping or None
        Auxiliary additive formulas or explicit link-scale expressions.
    data : IntoDataFrame
        Included training rows, after response subset selection.
    response : str
        Response name used by the existing formula parser.
    family_parameters : frozenset[str]
        Legal family roots, including mu.
    evaluate : Callable
        Evaluates one additive formula.
    designs : dict[str, DesignMatrices] or None
        Optional destination for retained additive evaluators.

    Returns
    -------
    PredictorGraph or None
        Canonical graph, or None for an ordinary additive model.

    Raises
    ------
    ValueError
        If names, dependencies, data carriers, or root declarations are invalid.
    TypeError
        If a parameter specification is neither a formula nor expression.
    """
    for name, specification in (distributional or {}).items():
        if not isinstance(specification, (str, NonlinearExpression)):
            raise TypeError(
                f"distributional parameter {name!r} requires a formula or NonlinearExpression"
            )
    frame = nw.from_native(data, eager_only=True).to_pandas()
    integer_names = frozenset(
        str(name) for name in frame.columns if frame[name].dtype.kind in "biu"
    )
    variables = frozenset(map(str, frame.columns)) | frozenset(nonlinear or ())
    roots = {
        name: _value_expression(value, name, integer_names, variables)
        for name, value in (distributional or {}).items()
        if isinstance(value, NonlinearExpression)
    }
    if nonlinear is not None and "mu" not in roots:
        # brms writes the nonlinear predictor as an R expression over Stan's
        # functions, with numeric data read as doubles.
        roots["mu"] = _parse_expression(
            translate_r_expression(
                formula.split("~", 1)[1],
                context=f"nonlinear formula {formula.strip()!r}",
                integer_names=integer_names,
                variables=variables,
            ).expression
        )
    if (
        "mu" in roots
        and isinstance((distributional or {}).get("mu"), NonlinearExpression)
        and formula.split("~", 1)[1].strip() != "0"
    ):
        raise ValueError(
            "an explicit nonlinear mu root requires a response formula with '~0'"
        )
    if not roots and nonlinear is None:
        return None
    unknown_roots = roots.keys() - family_parameters
    if unknown_roots:
        raise ValueError(f"unknown nonlinear family parameters {sorted(unknown_roots)}")
    declarations = dict(nonlinear or {})
    for name in declarations:
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", name
        ):
            raise ValueError(f"nonlinear parameter {name!r} requires a QVR identifier")
        if name in family_parameters or name.startswith(("dpar_", "nlpar_", "nlcov_")):
            raise ValueError(f"nonlinear parameter name {name!r} is reserved")
    leaves: dict[str, Formula] = {}
    expressions: dict[str, LetExprNode | ComponentExpression] = {}
    for name, value in declarations.items():
        if isinstance(value, NonlinearExpression):
            expressions[name] = _value_expression(value, name, integer_names, variables)
        elif isinstance(value, str):
            if "~" not in value:
                raise ValueError(f"nonlinear formula for {name!r} requires '~'")
            lhs, rhs = value.split("~", 1)
            if lhs.strip() not in ("", name):
                raise ValueError(
                    f"nonlinear formula {name!r} has mismatched left-hand side"
                )
            parsed, design = evaluate(f"{response} ~ {rhs}")
            leaves[name] = parsed
            if designs is not None:
                designs[name] = design
        else:
            raise TypeError(
                f"nonlinear parameter {name!r} requires a formula or NonlinearExpression"
            )
    dependencies = {
        name: _value_names(value) & declarations.keys()
        for name, value in expressions.items()
    }
    dependencies.update({name: set() for name in leaves})
    try:
        order = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as exc:
        raise ValueError(
            f"cyclic nonlinear parameter dependencies: {exc.args[1]}"
        ) from exc
    reachable = set().union(
        *(_value_names(value) & declarations.keys() for value in roots.values())
    )
    pending = list(reachable)
    while pending:
        for dependency in dependencies[pending.pop()]:
            if dependency not in reachable:
                reachable.add(dependency)
                pending.append(dependency)
    unused = declarations.keys() - reachable
    if unused:
        raise ValueError(f"unused nonlinear parameters {sorted(unused)}")
    raw_names = (
        set().union(
            *(_value_names(value) for value in (*expressions.values(), *roots.values()))
        )
        - declarations.keys()
    )
    forbidden = raw_names & family_parameters
    if forbidden:
        raise ValueError(
            f"family outputs cannot be nonlinear dependencies: {sorted(forbidden)}"
        )
    unknown = raw_names - set(frame.columns)
    if unknown:
        raise ValueError(
            f"unknown nonlinear symbols {sorted(unknown)}; declare parameters or supply numeric columns"
        )
    inputs = {}
    for name in sorted(raw_names):
        values = np.asarray(frame[name])
        if (
            values.ndim != 1
            or values.dtype.kind not in "biuf"
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"nonlinear covariate {name!r} requires finite numeric scalar rows"
            )
        qname = "nlcov_" + _qvr_name(name)
        if qname in {column.qvr_name for column in inputs.values()}:
            raise ValueError(f"nonlinear covariate {name!r} has a colliding QVR name")
        inputs[name] = FixedColumn(term=name, name=name, qvr_name=qname, data=values)
    substitution = {name: LetExprVar(name=f"nlpar_{name}") for name in declarations}
    substitution.update(
        {name: LetExprVar(name=column.qvr_name) for name, column in inputs.items()}
    )
    return PredictorGraph(
        leaves=leaves,
        expressions={
            name: _substitute_value(expressions[name], substitution)
            for name in order
            if name in expressions
        },
        roots={
            name: _substitute_value(value, substitution)
            for name, value in roots.items()
        },
        inputs=inputs,
    )


def graph_input_bindings(graph: PredictorGraph, data):
    """Validate fresh expression covariates against their fitted numeric carriers.

    Parameters
    ----------
    graph : PredictorGraph
        Fitted covariate contracts and expressions.
    data : pandas.DataFrame
        Fresh included predictor rows.

    Returns
    -------
    dict[str, torch.Tensor]
        Typed row bindings consumed by the same generated QVR program.

    Raises
    ------
    ValueError
        If a required column is absent, nonfinite, or changes numeric carrier.
    """
    import torch

    bindings = {}
    for name, column in graph.inputs.items():
        if name not in data:
            raise ValueError(f"new_data is missing nonlinear covariate {name!r}")
        value = np.asarray(data[name])
        kind = column.data.dtype.kind
        if (
            value.ndim != 1
            or value.dtype.kind not in "biuf"
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"nonlinear covariate {name!r} requires finite numeric scalar rows"
            )
        if (kind == "b" and value.dtype.kind != "b") or (
            kind in "iu" and value.dtype.kind not in "iu"
        ):
            raise ValueError(
                f"nonlinear covariate {name!r} changes its fitted numeric carrier"
            )
        if value.dtype.kind == "u" and (value > np.iinfo(np.int64).max).any():
            raise ValueError(
                f"nonlinear covariate {name!r} exceeds signed Int64 capacity"
            )
        bindings[column.qvr_name] = torch.as_tensor(
            value.copy(),
            dtype=torch.bool
            if kind == "b"
            else torch.int64
            if kind in "iu"
            else torch.float64,
        )
    return bindings


def predictor_items(
    formula: Formula,
    distributional: Mapping[str, Formula],
    graph: PredictorGraph | None,
):
    """Enumerate every additive design using one stable coefficient namespace.

    Parameters
    ----------
    formula : Formula
        Response metadata and ordinary location design, when present.
    distributional : Mapping[str, Formula]
        Ordinary auxiliary designs.
    graph : PredictorGraph or None
        Additional nonlinear parameter designs.

    Returns
    -------
    tuple[tuple[str, str, Formula], ...]
        Public parameter label, QVR coefficient prefix, and evaluated design.
    """
    return (
        ("mu", "", formula),
        *((name, f"dpar_{name}_", value) for name, value in distributional.items()),
        *(
            ((name, f"nlpar_{name}_", value) for name, value in graph.leaves.items())
            if graph is not None
            else ()
        ),
    )


__all__ = [
    "NonlinearExpression",
    "PredictorGraph",
    "ComponentExpression",
    "build_predictor_graph",
    "nonlinear_response_source",
    "predictor_items",
    "graph_input_bindings",
]
