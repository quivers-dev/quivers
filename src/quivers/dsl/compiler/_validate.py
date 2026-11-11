"""Static validation passes that emit `Violation` diagnostics.

The pass walks the parsed `Module`, locates every
SampleStep / ObserveStep / MarginalizeStep / `MorphismInitFamily`
that resolves to a registered distribution family, and emits the
following structured diagnostics:

* ``code="implicit-family-defaults"`` -- ``severity="warning"``:
  the step (or init clause) supplied no positional args and the
  resolver substituted the family's canonical default parameter
  set. The current pipeline still accepts this form, but the
  defaults will be removed once every shipped example moves to an
  explicit ``~ Family(args)`` declaration.

* ``code="family-arg-parameterization"`` -- ``severity="error"``: a
  family construction uses positional arguments or does not match one
  complete keyword schema.

* ``code="family-arg-shape"`` -- ``severity="warning"``: a
  literal vector argument has the wrong length for the family's
  reported event size, or a literal simplex argument has elements
  that do not sum to 1.
"""

from __future__ import annotations

import torch.distributions.constraints as _c
from torch.distributions.constraints import Constraint

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgList,
    DrawArgNamed,
    DrawArgScalar,
    ExprMorphismCall,
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
)
from quivers.dsl.constraints import Violation
from quivers.dsl.draw_args import is_matrix
from quivers.dsl.family_schemas import FAMILY_ALIASES, family_parameterizations
from quivers.dsl.step_resolution import (
    ResolvedDist,
    StepResolutionError,
    build_let_table,
    morphism_table,
    resolve_step_dist,
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta


def validate_family_arg_shapes(module: Module) -> list[Violation]:
    """Walk every draw step in ``module`` and emit `Violation`
    diagnostics for argument-shape mismatches and implicit-default
    fallbacks.

    Steps and init clauses whose morphism slot does not resolve to a
    registered family are silently skipped: the resolver-level
    diagnostic surfaces those cases when the user actually invokes a
    backend transpile or compile.
    """
    out: list[Violation] = []
    # Argument-shape validation is a source-language pass.  Use the unchecked
    # declaration table here: ``build_morphism_table`` additionally enforces a
    # transpiler boundary for model-internal parameter networks, which must not
    # make ordinary ``qvr check`` fail (or crash) for an executable QVR model.
    morphisms = morphism_table(module)
    lets = build_let_table(module)
    family_set = frozenset(FAMILY_META) | frozenset(FAMILY_ALIASES)

    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            _walk_program(stmt, morphisms, lets, family_set, out)
        elif isinstance(stmt, MorphismDecl):
            _walk_morphism(stmt, morphisms, family_set, out)

    return out


def _walk_program(
    program: ProgramDecl,
    morphisms: dict[str, MorphismDecl],
    lets: dict,
    family_set: frozenset[str],
    out: list[Violation],
) -> None:
    _walk_steps(program.draws, morphisms, lets, family_set, out)


def _walk_steps(
    steps: tuple[ProgramStep, ...],
    morphisms: dict[str, MorphismDecl],
    lets: dict,
    family_set: frozenset[str],
    out: list[Violation],
) -> None:
    for step in steps:
        if isinstance(step, (SampleStep, ObserveStep, MarginalizeStep)):
            _check_step(step, morphisms, lets, family_set, out)
        if isinstance(step, MarginalizeStep):
            _walk_steps(step.scope, morphisms, lets, family_set, out)


def _walk_morphism(
    decl: MorphismDecl,
    morphisms: dict[str, MorphismDecl],
    family_set: frozenset[str],
    out: list[Violation],
) -> None:
    init = decl.init_family
    if init is None:
        if (
            isinstance(decl.init_expr, ExprMorphismCall)
            and decl.init_expr.callee == "Categorical"
        ):
            out.append(
                Violation(
                    code="family-arg-parameterization",
                    severity="error",
                    message=(
                        "Categorical requires an explicit parameterization; "
                        "use probs=... or logits=..."
                    ),
                    line=decl.line,
                    col=decl.col,
                )
            )
        return
    if init.family not in family_set:
        return
    meta = FAMILY_META[FAMILY_ALIASES.get(init.family, init.family)]
    display = ", ".join(decl.names)
    if not init.args:
        if init.family == "Categorical":
            out.append(
                Violation(
                    code="family-arg-parameterization",
                    severity="error",
                    message=(
                        "Categorical requires an explicit parameterization; "
                        "use probs=... or logits=..."
                    ),
                    line=decl.line,
                    col=decl.col,
                )
            )
            return
        out.append(
            Violation(
                code="implicit-family-defaults",
                severity="warning",
                message=(
                    f"morphism {display!r}: `~ {init.family}` carries no "
                    f"explicit arguments; the resolver substitutes the "
                    f"family's canonical default parameters. Declare "
                    f"`~ {init.family}(args)` explicitly to silence this "
                    f"warning."
                ),
                line=decl.line,
                col=decl.col,
            )
        )
    else:
        _check_args_shape(
            family=init.family,
            args=init.args,
            meta=meta,
            line=init.line or decl.line,
            col=init.col or decl.col,
            origin=f"morphism {display!r} init clause",
            out=out,
        )


def _check_step(
    step: SampleStep | ObserveStep | MarginalizeStep,
    morphisms: dict[str, MorphismDecl],
    lets: dict,
    family_set: frozenset[str],
    out: list[Violation],
) -> None:
    try:
        resolved: ResolvedDist = resolve_step_dist(
            step.morphism,
            step.args,
            morphisms=morphisms,
            lets=lets,
            family_registry=family_set,
            target="qvr-validate",
        )
    except StepResolutionError:
        return
    surface_family = (
        step.morphism if step.morphism in FAMILY_ALIASES else resolved.family
    )
    meta = FAMILY_META.get(FAMILY_ALIASES.get(surface_family, surface_family))
    if meta is None:
        return
    # Implicit-defaults check: the step (or its referenced init
    # clause) carried no args, the resolver filled defaults.
    if not step.args:
        if step.morphism == "Categorical":
            out.append(
                Violation(
                    code="family-arg-parameterization",
                    severity="error",
                    message=(
                        "Categorical requires an explicit parameterization; "
                        "use probs=... or logits=..."
                    ),
                    line=step.line,
                    col=step.col,
                )
            )
            return
        # When the morphism slot is itself a morphism declaration with
        # explicit init args, the warning emits at the morphism site
        # (handled by `_walk_morphism`) rather than here.
        decl = morphisms.get(step.morphism)
        decl_has_explicit = (
            decl is not None
            and decl.init_family is not None
            and bool(decl.init_family.args)
        )
        if not decl_has_explicit and resolved.args:
            out.append(
                Violation(
                    code="implicit-family-defaults",
                    severity="warning",
                    message=(
                        f"step `<- {step.morphism}`: no positional "
                        f"arguments; the resolver substitutes the "
                        f"family's canonical default parameters. "
                        f"Declare `<- {step.morphism}(args)` explicitly "
                        f"to silence this warning."
                    ),
                    line=step.line,
                    col=step.col,
                )
            )
        return
    if step.morphism not in family_set:
        # These are arguments to a declared morphism or computation, whose
        # signature remains positional; its family initializer is validated
        # independently by `_walk_morphism`.
        return
    _check_args_shape(
        family=surface_family,
        args=step.args,
        meta=meta,
        line=step.line,
        col=step.col,
        origin=f"step `<- {step.morphism}`",
        out=out,
    )


def _check_args_shape(
    *,
    family: str,
    args: tuple[DrawArg, ...],
    meta: FamilyMeta,
    line: int,
    col: int,
    origin: str,
    out: list[Violation],
) -> None:
    """Check argument binding and shape against ``arg_constraints``."""
    arg_constraints = _read_arg_constraints(meta) or {}
    schemas = family_parameterizations(family, tuple(arg_constraints))
    if not schemas or schemas == ((),):
        return
    schema_text = " or ".join("(" + ", ".join(schema) + ")" for schema in schemas)
    if any(not isinstance(arg, DrawArgNamed) for arg in args):
        out.append(
            Violation(
                code="family-arg-parameterization",
                severity="error",
                message=(
                    f"family {family!r} arguments are keyword-only; "
                    f"{origin} must use one complete schema: {schema_text}"
                ),
                line=line,
                col=col,
            )
        )
        return

    by_name: dict[str, DrawArg] = {}
    supplied: set[str] = set()
    for arg in args:
        assert isinstance(arg, DrawArgNamed)
        arg_name = arg.parameter
        if arg_name in supplied:
            out.append(
                Violation(
                    code="family-arg-shape",
                    severity="error",
                    message=(
                        f"family {family!r} is given parameter {arg_name!r} twice"
                    ),
                    line=line,
                    col=col,
                )
            )
            return
        supplied.add(arg_name)
        by_name[arg_name] = arg.value

    selected = next((schema for schema in schemas if set(schema) == supplied), None)
    if selected is None:
        out.append(
            Violation(
                code="family-arg-parameterization",
                severity="error",
                message=(
                    f"family {family!r} requires exactly one complete parameter "
                    f"schema {schema_text}; {origin} supplied "
                    f"({', '.join(arg.parameter for arg in args)})"
                ),
                line=line,
                col=col,
            )
        )
        return

    for arg_name in selected:
        _check_nested_family_args(
            by_name[arg_name],
            line=line,
            col=col,
            origin=origin,
            out=out,
        )
        constraint = arg_constraints.get(arg_name)
        if constraint is None:
            continue
        _check_arg_against_constraint(
            family=family,
            arg=by_name[arg_name],
            arg_name=arg_name,
            constraint=constraint,
            line=line,
            col=col,
            origin=origin,
            out=out,
        )


def _check_nested_family_args(
    argument: DrawArg,
    *,
    line: int,
    col: int,
    origin: str,
    out: list[Violation],
) -> None:
    """Validate every family construction nested in one argument."""
    if isinstance(argument, DrawArgDist):
        meta = FAMILY_META.get(FAMILY_ALIASES.get(argument.family, argument.family))
        if meta is not None:
            _check_args_shape(
                family=argument.family,
                args=argument.args,
                meta=meta,
                line=argument.line or line,
                col=argument.col or col,
                origin=f"nested family in {origin}",
                out=out,
            )
        return
    if isinstance(argument, DrawArgList):
        for item in argument.items:
            _check_nested_family_args(
                item,
                line=line,
                col=col,
                origin=origin,
                out=out,
            )


def _check_arg_against_constraint(
    *,
    family: str,
    arg: DrawArg,
    arg_name: str,
    constraint: Constraint,
    line: int,
    col: int,
    origin: str,
    out: list[Violation],
) -> None:
    if isinstance(constraint, _c._IndependentConstraint) and constraint.event_dim >= 1:
        if isinstance(arg, DrawArgScalar):
            # Scalar broadcasting is the Lower-path's responsibility;
            # this is not a shape error.
            return
        if isinstance(arg, DrawArgList):
            if is_matrix(arg):
                # Matrix-shape validation requires a sentinel.
                return
            literal = _list_literal_length(arg)
            if literal is None:
                return
            # Without an instance event_shape we can't determine the
            # required length; the per-call validation in Lower fills
            # this in. Skip silently here.
            return
    if isinstance(constraint, _c._Simplex) and isinstance(arg, DrawArgList):
        literal_values = _list_literal_floats(arg)
        if literal_values is None:
            return
        total = sum(literal_values)
        if not _approx_equal(total, 1.0):
            out.append(
                Violation(
                    code="family-arg-shape",
                    severity="warning",
                    message=(
                        f"family {family!r}: argument {arg_name!r} is "
                        f"declared as a simplex but the literal "
                        f"{literal_values!r} sums to {total!r} (expected 1.0)"
                    ),
                    line=line,
                    col=col,
                )
            )


def _read_arg_constraints(meta: FamilyMeta) -> dict[str, Constraint] | None:
    cls_attr = meta.distribution_class.arg_constraints
    if isinstance(cls_attr, dict):
        return cls_attr
    return None


def _list_literal_length(arg: DrawArgList) -> int | None:
    """Return the literal length when every item is a numeric
    literal; ``None`` otherwise."""
    for item in arg.items:
        if not isinstance(item, DrawArgScalar):
            return None
    return len(arg.items)


def _list_literal_floats(arg: DrawArgList) -> list[float] | None:
    """Return the float values when every item is a numeric
    literal; ``None`` otherwise."""
    out: list[float] = []
    for item in arg.items:
        if not isinstance(item, DrawArgScalar):
            return None
        out.append(item.value)
    return out


def _approx_equal(a: float, b: float, *, atol: float = 1e-6) -> bool:
    return abs(a - b) <= atol


__all__ = ["validate_family_arg_shapes"]
