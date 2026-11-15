"""Sugar table for the compositional measure algebra.

Brms- and Stan-style named families that desugar to canonical
operator-form expressions at compile time when all arguments are
constants. The user-facing surface keeps the ergonomic names:

    observe y <- TruncatedNormal(0.0, 1.0, 0.0, 1.0)
    observe y <- HalfNormal(2.0)
    observe y <- HalfCauchy(2.5)
    observe y <- HalfStudentT(4.0, 1.5)

while the compiler internally walks the canonical desugared form:

    observe y <- Restrict(Normal(0.0, 1.0), 0.0, 1.0)
    observe y <- Restrict(Normal(0.0, 2.0), 0.0)
    observe y <- Restrict(Cauchy(0.0, 2.5), 0.0)
    observe y <- Restrict(StudentT(4.0, 0.0, 1.5), 0.0)

The sugar desugaring runs only when every argument is a literal
(`DrawArgScalar`); sugar calls with free-variable arguments continue
to route through their dedicated inline family entries
(`ZeroInflatedPoisson`, `HurdlePoisson`, `MixtureNormal` etc.),
which internally compose the same `Mixture` / `Restrict` / `PointMass`
operators. The dual surface is the "two-way recognition" the design
note specifies: source can be either form, the compiler canonicalises
to the operator form internally when it can, and the pretty-printer
re-sugars on the way out.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgList,
    DrawArgNamed,
    DrawArgScalar,
    ObserveStep,
    SampleStep,
)
from quivers.dsl.draw_args import bind_family_arguments


def _named(parameter: str, value: DrawArg) -> DrawArgNamed:
    return DrawArgNamed(parameter=parameter, value=value)


def _dist(family: str, **arguments: DrawArg) -> DrawArgDist:
    return DrawArgDist(
        family=family,
        args=tuple(_named(parameter, value) for parameter, value in arguments.items()),
    )


def _scalar(value: int | float) -> DrawArgScalar:
    return DrawArgScalar(value=value)


def _literal_values(
    family: str,
    args: tuple[DrawArg, ...],
) -> tuple[DrawArgScalar, ...] | None:
    """Return keyword-bound scalar values in the family's canonical order."""
    try:
        bound = bind_family_arguments(family, args, SUGAR_PARAMETERS[family])
    except TypeError:
        # The normal validation pass owns diagnostics for incomplete or
        # otherwise malformed family calls. Sugar only canonicalizes calls
        # that are already structurally complete.
        return None
    values = tuple(value for _, value in bound)
    if not all(isinstance(value, DrawArgScalar) for value in values):
        return None
    return cast(tuple[DrawArgScalar, ...], values)


def _desugar_truncated_normal(
    args: tuple[DrawArg, ...],
) -> tuple[str, tuple[DrawArg, ...]]:
    if len(args) != 4:
        raise ValueError(
            f"TruncatedNormal: expected (mu, sigma, low, high), got {len(args)} args"
        )
    mu, sigma, low, high = args
    return "Restrict", (
        _named("base", _dist("Normal", loc=mu, scale=sigma)),
        _named("low", low),
        _named("high", high),
    )


def _desugar_half(base_family: str, default_loc: float = 0.0):
    """Build a `Half{base_family}(scale)` -> `Restrict(base(0, scale), 0)`
    rewriter. Used for Normal, Cauchy, Laplace, ...
    """

    def _impl(args: tuple[DrawArg, ...]) -> tuple[str, tuple[DrawArg, ...]]:
        if len(args) != 1:
            raise ValueError(
                f"Half{base_family}: expected (scale,), got {len(args)} args"
            )
        (scale,) = args
        return "Restrict", (
            _named(
                "base",
                _dist(base_family, loc=_scalar(default_loc), scale=scale),
            ),
            _named("low", _scalar(0.0)),
        )

    return _impl


def _desugar_half_student_t(
    args: tuple[DrawArg, ...],
) -> tuple[str, tuple[DrawArg, ...]]:
    if len(args) != 2:
        raise ValueError(f"HalfStudentT: expected (nu, scale), got {len(args)} args")
    nu, scale = args
    return "Restrict", (
        _named(
            "base",
            _dist("StudentT", df=nu, loc=_scalar(0.0), scale=scale),
        ),
        _named("low", _scalar(0.0)),
    )


# Sugar entries that produce constant-arg operator-form expressions.
# Each rewriter consumes the original draw args (already validated
# all-scalar by `_all_scalar`) and emits a (operator_family, new_args)
# pair the compiler dispatches on.
SUGAR_TABLE: dict[
    str, Callable[[tuple[DrawArg, ...]], tuple[str, tuple[DrawArg, ...]]]
] = {
    "TruncatedNormal": _desugar_truncated_normal,
    "HalfNormal": _desugar_half("Normal"),
    "HalfCauchy": _desugar_half("Cauchy"),
    "HalfLaplace": _desugar_half("Laplace"),
    "HalfStudentT": _desugar_half_student_t,
}


# Canonical parameters consumed by each surface family before it is
# rewritten into the operator algebra. Binding through the same shared
# helper as ordinary compilation makes source order irrelevant here too.
SUGAR_PARAMETERS: dict[str, tuple[str, ...]] = {
    "TruncatedNormal": ("mu", "sigma", "low", "high"),
    "HalfNormal": ("scale",),
    "HalfCauchy": ("scale",),
    "HalfLaplace": ("scale",),
    "HalfStudentT": ("nu", "scale"),
}


def desugar_step(step: SampleStep | ObserveStep) -> SampleStep | ObserveStep:
    """Rewrite a step whose morphism is a sugar family into the
    canonical operator-algebra form. Steps whose morphism is not in
    the sugar table, or whose args contain free variables, pass
    through unchanged.

    Recurses on the args so nested sugar (e.g. `Mixture([0.3, 0.7],
    [PointMass(0), HalfNormal(1.0)])` with all literals) is fully
    desugared in one pass.
    """
    args = step.args
    if args is not None:
        args = tuple(_desugar_arg(a) for a in args)
    morphism = step.morphism
    if morphism in SUGAR_TABLE and args is not None:
        values = _literal_values(morphism, args)
        if values is not None:
            morphism, args = SUGAR_TABLE[morphism](values)
    if morphism == step.morphism and args == step.args:
        return step
    return step.with_(morphism=morphism, args=args)


def _desugar_arg(arg: DrawArg) -> DrawArg:
    """Recursively desugar a draw arg: `DrawArgDist`s whose family
    name is in the sugar table and whose args are all scalar are
    rewritten; lists are mapped element-wise; everything else passes
    through.
    """
    if isinstance(arg, DrawArgNamed):
        return arg.with_(value=_desugar_arg(arg.value))
    if isinstance(arg, DrawArgDist):
        inner_args = tuple(_desugar_arg(a) for a in arg.args)
        if arg.family in SUGAR_TABLE:
            values = _literal_values(arg.family, inner_args)
            if values is not None:
                new_family, new_args = SUGAR_TABLE[arg.family](values)
                return DrawArgDist(family=new_family, args=new_args)
        return DrawArgDist(family=arg.family, args=inner_args)
    if isinstance(arg, DrawArgList):
        return DrawArgList(items=tuple(_desugar_arg(item) for item in arg.items))
    return arg


__all__ = []
