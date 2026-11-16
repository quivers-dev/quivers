"""Checked QVR expressions evaluated jointly over retained sample axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics._coordinates import event_dimensions
from quivers.diagnostics._numeric import real_samples
from quivers.diagnostics.hypothesis import HypothesisSummary, hypothesis_summary
from quivers.diagnostics.density import PointNullSpec
from quivers.dsl._value_expressions import _parse_value_expression
from quivers.dsl.ast_nodes import LetExprVar, LetStep, ProgramDecl
from quivers.dsl.let_expr_traversal import free_let_names, substitute_let_expr


class ContrastExpression(ValueModel):
    """One QVR value expression over jointly aligned retained variables.

    Parameters
    ----------
    expression : str
        Existing QVR arithmetic, functions, and event-axis indexing. Chain and
        draw axes are protected from indexing and reductions. Original Int
        and Bool carriers retain QVR integer arithmetic and predicate typing;
        conversion to diagnostic Float64 values follows expression evaluation.
    bindings : dict[str, str]
        Optional expression aliases mapped to original variable labels. This
        permits labels such as ``condition[a]`` without changing their spelling.
        Unused aliases and unknown variables are errors.
    """

    expression: str
    bindings: dict[str, str] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )


@dataclass(frozen=True)
class _ContrastPlan:
    """One checked callable and its retained event-coordinate contract.

    Parameters
    ----------
    names : tuple[str, ...]
        Original input variable labels in callable argument order.
    evaluate : Callable
        Existing native QVR expression callable, applied once per joint draw.
    dimensions : tuple[str, ...]
        Nominal result event axes from QIEC checking.
    shapes : tuple[tuple[int, ...], ...]
        Original input event shapes, excluding sample axes.
    input_carriers : tuple[str, ...]
        Canonical QVR scalar carriers of each retained input.
    carrier : str
        Canonical scalar carrier of the checked expression result.
    """

    names: tuple[str, ...]
    evaluate: Callable
    dimensions: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    input_carriers: tuple[str, ...]
    carrier: str


def _carrier(value: torch.Tensor) -> str:
    """Read one retained tensor's canonical QVR scalar carrier.

    Parameters
    ----------
    value : torch.Tensor
        Original retained values, before diagnostic floating conversion.

    Returns
    -------
    str
        Real, Int, or Bool as used by the canonical QVR type checker.

    Raises
    ------
    TypeError
        If the input is not a real tensor.
    """
    if not isinstance(value, torch.Tensor) or value.is_complex():
        raise TypeError("a contrast requires real retained tensors")
    return (
        "Bool"
        if value.dtype == torch.bool
        else "Real"
        if value.is_floating_point()
        else "Int"
    )


def _contrast_plan(
    expression: ContrastExpression,
    samples: Mapping[str, torch.Tensor],
    dims: Mapping[str, Sequence[str]] | None,
    coords: Mapping[str, Sequence[object]] | None,
) -> _ContrastPlan:
    """Check nominal event axes with the canonical QVR/QIEC type checker.

    Parameters
    ----------
    expression : ContrastExpression
        Authored expression and exact variable aliases.
    samples : Mapping[str, torch.Tensor]
        Retained tensors with explicit chain and draw axes.
    dims, coords : Mapping or None
        The shared retained-analysis dimension and coordinate declarations.

    Returns
    -------
    _ContrastPlan
        One checked native callable, variable ordering, and result dimensions.

    Raises
    ------
    ValueError, TypeError
        If bindings, nominal types, or expression capabilities disagree.
    """
    from quivers.dsl import parse
    from quivers.dsl.compiler.programs import _ProgramsMixin
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
    from quivers.qiec import INT, REAL
    from quivers.qiec.canonical import tensor_shape
    from quivers.qiec.types import IndexVariable

    node = _parse_value_expression(expression.expression)
    variables = tuple(sorted(free_let_names(node)))
    unused = expression.bindings.keys() - set(variables)
    if unused:
        raise ValueError(f"unused contrast aliases: {sorted(unused)}")
    names = tuple(expression.bindings.get(variable, variable) for variable in variables)
    unknown = set(names) - samples.keys()
    if unknown:
        raise ValueError(f"unknown contrast variables: {sorted(unknown)}")
    shapes = tuple(tuple(samples[name].shape[2:]) for name in names)
    dimensions = event_dimensions(
        dict(zip(names, shapes, strict=True)), dims=dims, coords=coords
    )
    axes = tuple(dict.fromkeys(axis for name in names for axis in dimensions[name]))
    nominal = {axis: f"ContrastAxis{i}" for i, axis in enumerate(axes)}
    arguments = []
    input_carriers = tuple(_carrier(samples[name]) for name in names)
    substitution = {}
    for i, (variable, name) in enumerate(zip(variables, names, strict=True)):
        alias = f"contrast_input_{i}"
        substitution[variable] = LetExprVar(name=alias)
        shape = dimensions[name]
        scalar = input_carriers[i]
        type_ = (
            scalar
            if not shape
            else f"Tensor[{scalar}](["
            + ",".join(nominal[axis] for axis in shape)
            + "])"
        )
        arguments.append(f"{alias}:{type_}")
    static = (
        "[" + ",".join(f"{name}:Nat" for name in nominal.values()) + "]" if axes else ""
    )
    # Only generated identifiers enter this header. The authored expression is
    # parsed once, substituted structurally, and checked as its original AST.
    module = parse(
        "object ContrastOut:Real 1\n"
        f"program retained_contrast{static}({','.join(arguments)}):ContrastOut->ContrastOut\n"
        "    let contrast_value=0.0\n"
        "    return contrast_value\n"
    )
    node = substitute_let_expr(node, substitution)
    module = module.with_(
        statements=tuple(
            statement.with_(draws=(LetStep(name="contrast_value", value=node),))
            if isinstance(statement, ProgramDecl)
            else statement
            for statement in module.statements
        )
    )
    checked = lower_qvr_to_qiec(module)
    computation = next(
        item for item in checked.computations if item.name == "retained_contrast"
    )
    type_ = computation.type.result
    result_shape = tensor_shape(type_)
    if (type_ if result_shape is None else result_shape[0]) not in (REAL, INT):
        raise TypeError(
            "a contrast must return real or integer values, not a predicate or structured value"
        )
    inverse = {value: key for key, value in nominal.items()}
    result_dims = tuple(
        inverse[axis.name]
        if isinstance(axis, IndexVariable) and axis.name in inverse
        else f"contrast_result_dim_{i}"
        for i, axis in enumerate(result_shape[1] if result_shape is not None else ())
    )
    compiled = _ProgramsMixin._compile_let_expr(node)

    def evaluate(*values):
        """Apply the compiled expression to ordered per-draw tensor arguments.

        Parameters
        ----------
        *values : torch.Tensor
            Original arguments without chain or draw axes.

        Returns
        -------
        torch.Tensor
            One checked real-valued contrast event.
        """
        return compiled(
            {f"contrast_input_{i}": value for i, value in enumerate(values)}
        )

    scalar_result = type_ if result_shape is None else result_shape[0]
    return _ContrastPlan(
        names,
        evaluate,
        result_dims,
        shapes,
        input_carriers,
        "Int" if scalar_result == INT else "Real",
    )


def _evaluate_contrast(
    plan: _ContrastPlan, samples: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Evaluate one checked callable with both sample axes protected by vmap.

    Parameters
    ----------
    plan : _ContrastPlan
        Checked expression and nominal event layout.
    samples : Mapping[str, torch.Tensor]
        Joint posterior or prior draws, with aligned sample axes.

    Returns
    -------
    torch.Tensor
        Finite Float64 contrast draws on the original device.

    Raises
    ------
    ValueError
        If sample alignment, event shapes, or evaluated values are invalid.
    """
    if not samples:
        raise ValueError("a retained contrast requires a nonempty draw mapping")
    anchor = next(iter(samples.values()))
    if anchor.ndim < 2 or not all(anchor.shape[:2]):
        raise ValueError("contrast samples require nonempty chain and draw axes")
    values = []
    for name, shape, carrier in zip(
        plan.names, plan.shapes, plan.input_carriers, strict=True
    ):
        original = samples[name]
        checked = real_samples(original, label=f"contrast variable {name!r}")
        if _carrier(original) != carrier:
            raise TypeError(
                f"contrast variable {name!r} requires the same {carrier} carrier in both streams"
            )
        value = (
            checked
            if carrier == "Real"
            else original.detach().to(
                dtype=torch.bool if carrier == "Bool" else torch.int64
            )
        )
        if value.shape[:2] != anchor.shape[:2] or tuple(value.shape[2:]) != shape:
            raise ValueError(
                "contrast variables require aligned sample axes and matching event shapes"
            )
        if value.device != anchor.device:
            raise ValueError("contrast variables must use the same device")
        values.append(value)
    if not values:
        # A constant is still a value at every retained draw. The dummy binding
        # supplies the sample layout and device without inspecting unused data.
        constant = plan.evaluate()
        return real_samples(
            torch.as_tensor(constant, device=anchor.device).expand(*anchor.shape[:2]),
            label="contrast",
        )
    result = torch.vmap(torch.vmap(plan.evaluate))(*values)
    return real_samples(result, label="contrast")


def _contrast_hypothesis(
    samples: Mapping[str, torch.Tensor],
    expression: str | ContrastExpression,
    *,
    prior: Mapping[str, torch.Tensor] | None = None,
    null: float | int | torch.Tensor = 0.0,
    probs: Sequence[float] = (0.025, 0.975),
    robust: bool = False,
    point: PointNullSpec | None = None,
    name: str = "contrast",
    kind: Literal["mcmc", "variational", "prior", "unspecified"] = "unspecified",
    dims: Mapping[str, Sequence[str]] | None = None,
    coords: Mapping[str, Sequence[object]] | None = None,
) -> HypothesisSummary:
    """Use one checked expression plan for jointly aligned posterior/prior streams.

    Parameters
    ----------
    samples, prior : Mapping[str, torch.Tensor], Mapping or None
        Joint retained posterior and optional prior samples.
    expression : str or ContrastExpression
        Existing QVR expression with optional original-label aliases.
    null : float, int, or torch.Tensor
        Comparison value on event axes only.
    probs : Sequence[float]
        Requested quantile probabilities in output order.
    robust : bool
        Use median and scaled MAD instead of mean and sample SD.
    point : PointNullSpec or None
        Optional marginal point-density request on a checked Real contrast.
    name : str
        Output contrast label.
    kind : str
        Retained sample provenance.
    dims, coords : Mapping or None
        Input nominal event axes and their original labels.

    Returns
    -------
    HypothesisSummary
        Strict-region summaries and optional distinct point-density evidence.

    Raises
    ------
    ValueError, TypeError
        If expression typing, coordinates, or retained numeric values disagree.
    """
    expression = (
        ContrastExpression(expression=expression)
        if isinstance(expression, str)
        else expression
    )
    if not isinstance(expression, ContrastExpression):
        raise TypeError("a contrast requires text or ContrastExpression")
    plan = _contrast_plan(expression, samples, dims, coords)
    if point is not None and plan.carrier != "Real":
        raise TypeError("point-density evidence requires a checked Real contrast")
    return hypothesis_summary(
        _evaluate_contrast(plan, samples),
        prior=_evaluate_contrast(plan, prior) if prior is not None else None,
        null=null,
        probs=probs,
        robust=robust,
        point=point,
        name=name,
        kind=kind,
        dims=plan.dimensions,
        coords=coords,
    )


__all__ = ["ContrastExpression"]
