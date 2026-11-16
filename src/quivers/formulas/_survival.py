"""Retained spline structures and shared AST lowering for proportional hazards."""

from __future__ import annotations

import didactic.api as dx
import numpy as np

from quivers._splines import MSplineBasis
from quivers._spline_design import cox_spline_basis
from quivers.formulas._ast import (
    _factor,
    _var,
    _index,
    _mul,
    _call,
    _real_tensor_literal,
)
from quivers.dsl.ast_nodes import (
    DiscreteConstructor,
    DrawArgNamed,
    DrawArgList,
    DrawArgScalar,
    LetStep,
    ObjectDecl,
    SampleStep,
    TypeFromExpr,
    TypeName,
    AxisSpec,
    OptionEntry,
    OptionName,
)

from quivers.formulas.formula import Formula


class BaselineSpec(dx.Model):
    """Retain spline structure independently of fitted baseline weights.

    Parameters
    ----------
    df : int
        Number of normalized M-spline components when selecting knots.
    degree : int
        Polynomial degree when selecting knots.
    boundary_knots : tuple[float, float] or None
        Explicit finite boundary span, starting at zero. None selects knots
        once from training events and follow-up times.
    interior_knots : tuple[float, ...]
        Explicit interior knots when boundary_knots is supplied.
    by : str or None
        Grouping factor for separate normalized profiles on this common basis.
    """

    df: int = 5
    degree: int = 3
    boundary_knots: tuple[float, ...] | None = None
    interior_knots: tuple[float, ...] = ()
    by: str | None = None

    def resolve(self, formula: Formula) -> MSplineBasis:
        """Resolve the baseline once from retained training responses.

        Parameters
        ----------
        formula : Formula
            Evaluated positive scalar response and modifier columns.

        Returns
        -------
        MSplineBasis
            Explicit basis or event-quantile training basis.

        Raises
        ------
        ValueError
            If responses are not positive finite scalar event times or the
            requested automatic knot structure is not identified by the data.
        """
        values = np.asarray(formula.response_values)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(
                "cox responses require finite strictly positive event times"
            )
        if self.boundary_knots is not None:
            if len(self.boundary_knots) != 2:
                raise ValueError("cox baseline requires two boundary knots")
            if self.boundary_knots[0] != 0:
                raise ValueError("cox baseline must start at zero")
            return MSplineBasis(
                (self.boundary_knots[0], self.boundary_knots[1]),
                self.interior_knots,
                self.degree,
            )
        if self.interior_knots:
            raise ValueError("explicit interior knots require boundary knots")
        censoring = formula.modifiers.censoring
        events = values if censoring is None else values[censoring.status.data == 0]
        followup = values
        if censoring is not None and censoring.upper is not None:
            followup = np.concatenate(
                (followup, censoring.upper.data[censoring.status.data == 2])
            )
        return cox_spline_basis(
            events.tolist(), followup.tolist(), df=self.df, degree=self.degree
        )


def baseline_steps(
    basis: MSplineBasis, *, prior: SampleStep | None = None, group: str | None = None
):
    """Emit one small retained table and a simplex contraction, without row copies.

    Parameters
    ----------
    basis : MSplineBasis
        Checked immutable training basis.
    prior : SampleStep or None
        Parsed prior, using the existing auxiliary-prior lowering. A scalar
        Dirichlet concentration expands over basis coordinates; the default is uniform.
    group : str or None
        QVR grouping object. Profiles are normalized separately at each level;
        contraction is shared by rows through the existing group index input.

    Returns
    -------
    tuple[list[ObjectDecl], list[LetStep | SampleStep]]
        Structural finite objects and source steps producing baseline_breaks,
        sbhaz, and baseline_controls. Only sbhaz is latent.

    Raises
    ------
    ValueError
        If the prior is not a finite positive scalar Dirichlet concentration.
    """
    if prior is None:
        prior = SampleStep(
            vars=("sbhaz",),
            morphism="Dirichlet",
            args=(
                DrawArgNamed(parameter="concentration", value=DrawArgScalar(value=1.0)),
            ),
        )
    if prior.morphism == "Dirichlet":
        arguments = []
        for argument in prior.args or ():
            if (
                isinstance(argument, DrawArgNamed)
                and argument.parameter == "concentration"
                and isinstance(argument.value, DrawArgScalar)
            ):
                if not np.isfinite(argument.value.value) or argument.value.value <= 0:
                    raise ValueError("sbhaz concentration must be finite and positive")
                argument = argument.with_(
                    value=DrawArgList(items=(argument.value,) * basis.size)
                )
            arguments.append(argument)
        prior = prior.with_(args=tuple(arguments))
    names = ("FormulaBaselineSegment", "FormulaBaselineBasis", "FormulaBaselineControl")
    sizes = (len(basis.breaks) - 1, basis.size, basis.degree + 1)
    objects = [
        ObjectDecl(
            names=(name,),
            init=TypeFromExpr(
                expr=DiscreteConstructor(constructor="FinSet", args=(str(size),))
            ),
        )
        for name, size in zip(names, sizes, strict=True)
    ]

    weighted = _mul(
        _index(
            "sbhaz",
            *((_var("baseline_group"),) if group is not None else ()),
            _var("baseline_component"),
        ),
        _index(
            "baseline_coefficients",
            _var("baseline_segment"),
            _var("baseline_component"),
            _var("baseline_control"),
        ),
    )
    contraction = _factor(
        (
            *(((("baseline_group", group),)) if group is not None else ()),
            ("baseline_segment", names[0]),
            ("baseline_control", names[2]),
        ),
        body=_call("sum", _factor((("baseline_component", names[1]),), body=weighted)),
    )
    steps = [
        LetStep(name="baseline_breaks", value=_real_tensor_literal(basis.breaks)),
        LetStep(
            name="baseline_coefficients",
            value=_real_tensor_literal(basis.m_coefficients),
        ),
        prior.with_(
            index=TypeName(name=names[1] if group is None else group),
            axes=prior.axes
            if group is None
            else AxisSpec(over=(names[1],), iid_over=(group,)),
            options=prior.options
            if group is None
            else (
                *prior.options,
                OptionEntry(key="over", value=OptionName(value=names[1])),
                OptionEntry(key="iid_over", value=OptionName(value=group)),
            ),
        ),
        LetStep(
            name="baseline_controls" if group is None else "baseline_profiles",
            value=contraction,
        ),
    ]
    if group is not None:
        steps.append(
            LetStep(
                name="baseline_controls",
                value=_index("baseline_profiles", _var(f"{group}_idx")),
            )
        )
    return objects, steps


__all__ = ["BaselineSpec"]
