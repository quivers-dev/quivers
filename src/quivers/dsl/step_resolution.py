"""Resolve the distribution references of program steps.

A draw may name a family, a declared morphism, or a let alias. This
module resolves those forms to canonical family names and arguments,
for the QIEC elaboration and the transpilers alike. It also rejects
consumed parameter-source morphisms because their learned weights are
absent from the QVR wire representation.
"""

from __future__ import annotations

import dataclasses
import inspect as _inspect

import torch.distributions as _td

from quivers.dsl.ast_nodes import (
    CallStep,
    BindStep,
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    DrawStep,
    Expr,
    ExprCap,
    ExprChangeBase,
    ExprChartFold,
    ExprCompose,
    ExprCup,
    ExprCurry,
    ExprDagger,
    ExprFan,
    ExprFreeze,
    ExprFromData,
    ExprIdent,
    ExprIdentity,
    ExprMarginalize,
    ExprMorphismCall,
    ExprParser,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    ExprTrace,
    ExprTransCompose,
    DefineDecl,
    GroupedBodyObserveStep,
    GroupedLatentInitStep,
    GroupedMarginalizeStep,
    LetExprNode,
    LetStep,
    MarginalizeStep,
    Module,
    MorphismDecl,
    MorphismInitFamily,
    ObserveStep,
    PlateDrawStep,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScoreStep,
    VectorisedObserveStep,
)
from quivers.dsl.let_expr_traversal import free_let_names
from quivers.dsl.ast_nodes._shared import (
    OptionCall,
    OptionEntry,
    OptionName,
    OptionNumber,
    OptionString,
)
from quivers.dsl.draw_args import (
    encode_index,
    is_matrix,
    list_atoms,
    matrix_rows,
)


class StepResolutionError(ValueError):
    """A step's distribution reference could not be resolved.

    Parameters
    ----------
    target : str
        The consumer the resolution ran for, such as a transpile target.
    kinds : list[str]
        Structured identifiers of what could not be resolved.
    """

    def __init__(self, target: str, kinds: list[str]) -> None:
        super().__init__(f"{target}: " + "; ".join(kinds))
        self.target = target
        self.kinds = list(kinds)


@dataclasses.dataclass(frozen=True)
class ResolvedDist:
    """The (family, args) pair a sample / observe step resolves to.

    ``family`` is the canonical QVR family name, exactly as it would
    appear if the user had written ``sample x <- family(args)``.
    ``args`` is the tuple of positional arguments after resolution,
    held in the legacy wire form ``str | float`` for backends that
    have not migrated to the structural `DrawArg` representation.

    The ``original_morphism_name`` and ``via`` fields are diagnostic:
    when a sample resolves through a morphism declaration, the
    transpiled output can reference both the underlying family and the
    original morphism name (some backends prefer one or the other).
    """

    family: str
    args: tuple[str | float, ...]
    original_morphism_name: str
    via: tuple[str, ...] = ()


def _draw_arg_to_wire(arg: DrawArg) -> str | float:
    """Lower a `DrawArg` atomic variant to its wire form. Compound
    variants encode positionally so the legacy backend pipeline
    receives the structural literal as a parseable string surrogate."""
    if isinstance(arg, DrawArgNamed):
        return _draw_arg_to_wire(arg.value)
    if isinstance(arg, DrawArgScalar):
        return arg.value
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgIndex):
        return encode_index(arg)
    if isinstance(arg, DrawArgDist):
        raise StepResolutionError(
            "qvr-transpile",
            [
                f"nested-distribution-arg:{arg.family}: a "
                "distribution-valued argument has no wire form"
            ],
        )
    if isinstance(arg, DrawArgList):
        if is_matrix(arg):
            rows = ", ".join(
                "[" + ", ".join(_atom_to_text(e) for e in row) + "]"
                for row in matrix_rows(arg)
            )
            return f"[{rows}]"
        return "[" + ", ".join(_atom_to_text(e) for e in list_atoms(arg)) + "]"
    raise TypeError(f"_draw_arg_to_wire: unsupported arg variant {type(arg).__name__}")


def _atom_to_text(value: str | float) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_number(float(value))
    return str(value)


def _format_number(value: float) -> str:
    if value == int(value):
        return f"{int(value)}"
    return repr(value)


def _role_of(decl: MorphismDecl) -> str | None:
    """The role a morphism declaration's option block names.

    Parameters
    ----------
    decl
        The declaration.

    Returns
    -------
    str | None
        The ``role=`` option's name, or ``None`` when absent.
    """
    for entry in decl.options:
        if entry.key == "role" and isinstance(entry.value, OptionName):
            return entry.value.value
    return None


def _weights_latent(decl: MorphismDecl) -> bool:
    """Whether a morphism declares ``[weights=latent]``.

    Parameters
    ----------
    decl
        The declaration.

    Returns
    -------
    bool
        ``True`` when its network's weights are latent sites.
    """
    return any(
        entry.key == "weights"
        and isinstance(entry.value, OptionName)
        and entry.value.value == "latent"
        for entry in decl.options
    )


def param_source_kind(decl: MorphismDecl, *, target: str) -> str | None:
    """Return the parameter-source kind a morphism declaration names
    in its ``[param_source=...]`` option, or ``None`` when it carries
    no such option.

    Two surface shapes name a kind. The bare identifier
    (``[param_source=mlp]``) parses to an
    [`OptionName`][quivers.dsl.ast_nodes._shared.OptionName] whose
    ``value`` is the kind; the parameterised call
    (``[param_source=mlp(16, 8)]``) parses to an
    [`OptionCall`][quivers.dsl.ast_nodes._shared.OptionCall] whose
    ``func`` is the kind and whose ``args`` are the hidden widths.
    Both denote the same source architecture, so both resolve to the
    same kind string and hit the same transpile boundary.

    Any other option shape (a bare flag, a number, a list) names no
    architecture, so there is nothing for the message to report and
    nothing a target could be told it is missing. That is a malformed
    declaration rather than an unsupported one, and it is rejected
    here by name instead of reaching a target.
    """
    for entry in decl.options:
        if entry.key != "param_source":
            continue
        value = entry.value
        if isinstance(value, OptionName):
            return value.value
        if isinstance(value, OptionCall):
            return value.func
        raise StepResolutionError(
            target,
            [
                f"param-source:unnamed:{decl.names[0]}",
            ],
        )
    return None


def _let_expr_value_names(expr: LetExprNode) -> frozenset[str]:
    """Return every free name a let / score expression reads.

    A name in a let expression is a *value*: the emitted target binds
    it to whatever the host supplies under that name. The set is what
    ``_reject_param_source_consumed``
    tests against the module's parameter-source morphisms.

    Binders subtract their own names: a lambda parameter and a factor
    binder's index variable shadow anything declared outside, so a
    morphism whose name they reuse is not read at that occurrence.
    """
    # A callee is a value in this native-resolution pass: a morphism
    # without a distribution initializer is wired under that name.
    return frozenset(free_let_names(expr, include_callees=True))


def _draw_arg_value_names(arg: DrawArg) -> frozenset[str]:
    """Return every name a draw-step argument reads as a value."""
    if isinstance(arg, DrawArgNamed):
        return _draw_arg_value_names(arg.value)
    if isinstance(arg, DrawArgName):
        return frozenset({arg.text})
    if isinstance(arg, DrawArgIndex):
        return frozenset({arg.name})
    if isinstance(arg, DrawArgScalar):
        return frozenset()
    if isinstance(arg, DrawArgDist):
        return frozenset().union(
            *(_draw_arg_value_names(a) for a in arg.args),
            frozenset(),
        )
    if isinstance(arg, DrawArgList):
        return frozenset().union(
            *(_draw_arg_value_names(i) for i in arg.items),
            frozenset(),
        )
    raise TypeError(
        f"_draw_arg_value_names: unsupported arg variant {type(arg).__name__}"
    )


def _expr_ident_names(expr: Expr) -> frozenset[str]:
    """Return every morphism / binding name a composition expression
    references.

    This is the composite counterpart of
    [`_let_expr_value_names`][quivers.dsl.step_resolution._let_expr_value_names]:
    it walks a ``define``d chain (or a morphism's ``~ <expr>`` init)
    and collects the leaves, so a parameter-source morphism buried
    under ``scan(...)``, ``stack(...)`` or a ``>>`` chain is visible to
    the module-level check even when the pre-lower expansion pass
    never flattens the composite into program steps.

    Object names (``identity(A)``, ``cup(A)``) and category / goal
    names are not morphism references and contribute nothing.
    """
    if isinstance(expr, ExprIdent):
        return frozenset({expr.name})
    if isinstance(expr, (ExprIdentity, ExprCup, ExprCap, ExprFromData)):
        # Object-indexed or data-indexed primitives: the payload is an
        # object name or a data key, never a morphism.
        return frozenset()
    if isinstance(
        expr,
        (ExprFreeze, ExprDagger, ExprTrace, ExprMarginalize, ExprCurry),
    ):
        return _expr_ident_names(expr.inner)
    if isinstance(expr, ExprChangeBase):
        return _expr_ident_names(expr.inner) | _expr_ident_names(expr.phi)
    if isinstance(expr, (ExprCompose, ExprTensorProduct, ExprTransCompose)):
        return _expr_ident_names(expr.left) | _expr_ident_names(expr.right)
    if isinstance(expr, ExprFan):
        return frozenset().union(
            *(_expr_ident_names(e) for e in expr.exprs),
            frozenset(),
        )
    if isinstance(expr, (ExprRepeat, ExprStack, ExprScan)):
        return _expr_ident_names(expr.expr)
    if isinstance(expr, ExprParser):
        # `rules` names rule declarations and `terminal` the lexicon
        # morphism; `categories` and `start` name categories.
        return frozenset(expr.rules) | (
            frozenset({expr.terminal}) if expr.terminal is not None else frozenset()
        )
    if isinstance(expr, ExprChartFold):
        return frozenset().union(
            _expr_ident_names(expr.lex),
            *(_expr_ident_names(e) for e in (expr.binary, expr.unary) if e is not None),
            frozenset(),
        )
    if isinstance(expr, ExprMorphismCall):
        return frozenset({expr.callee, *expr.args})
    raise TypeError(
        f"_expr_ident_names: unsupported expression variant {type(expr).__name__}"
    )


def _step_site_names(step: ProgramStep) -> frozenset[str]:
    """Return every name ``step`` consumes in a *site* position: the
    morphism slot of a draw step, which is the slot that has to denote
    a distribution.

    A marginalize / bind step contributes its own slot and those of
    every step in its nested scope.
    """
    if isinstance(step, (SampleStep, ObserveStep)):
        return frozenset({step.morphism})
    if isinstance(step, MarginalizeStep):
        return frozenset({step.morphism}).union(
            *(_step_site_names(s) for s in step.scope),
            frozenset(),
        )
    if isinstance(step, BindStep):
        return frozenset({step.morphism}).union(
            *(_step_site_names(s) for s in step.scope or ()),
            frozenset(),
        )
    if isinstance(
        step,
        (
            DrawStep,
            PlateDrawStep,
            VectorisedObserveStep,
            GroupedBodyObserveStep,
        ),
    ):
        return frozenset({step.morphism})
    if isinstance(
        step,
        (
            LetStep,
            CallStep,
            ScoreStep,
            ReturnStep,
            GroupedMarginalizeStep,
            GroupedLatentInitStep,
        ),
    ):
        # None of these carries a morphism slot: a let / score binds an
        # expression, a call names a computation, a return names bound
        # program variables, and the two grouped-marginalize IR steps
        # carry env slots and a class size rather than a distribution
        # reference.
        return frozenset()
    raise TypeError(f"_step_site_names: unsupported step variant {type(step).__name__}")


def _step_value_names(step: ProgramStep) -> frozenset[str]:
    """Return every name ``step`` reads in a value position.

    The morphism slot of a draw step is deliberately excluded: that is
    the *site* position,
    [`_step_site_names`][quivers.dsl.step_resolution._step_site_names]
    collects it separately, and the two positions produce different
    diagnostics. What remains is the argument list, the right-hand
    side of a let / score binding, and (for a step with a nested
    scope) the same over that scope.

    A `ReturnStep`'s ``vars`` are excluded on the same grounds: the
    grammar admits only bound program variables there, so a morphism
    name in that slot is impossible and reading it would only invent
    false positives against a shadowing binding.
    """
    if isinstance(step, (LetStep, ScoreStep)):
        return _let_expr_value_names(step.value)
    if isinstance(step, CallStep):
        names: frozenset[str] = frozenset()
        for argument in step.call.arguments:
            names |= _let_expr_value_names(argument)
        return names
    if isinstance(
        step,
        (
            SampleStep,
            ObserveStep,
            DrawStep,
            PlateDrawStep,
            VectorisedObserveStep,
            GroupedBodyObserveStep,
        ),
    ):
        return frozenset().union(
            *(_draw_arg_value_names(a) for a in step.args or ()),
            frozenset(),
        )
    if isinstance(step, MarginalizeStep):
        return frozenset().union(
            *(_draw_arg_value_names(a) for a in step.args or ()),
            *(_step_value_names(s) for s in step.scope),
            frozenset(),
        )
    if isinstance(step, BindStep):
        return frozenset().union(
            *(_draw_arg_value_names(a) for a in step.args or ()),
            *(_step_value_names(s) for s in step.scope or ()),
            frozenset(),
        )
    if isinstance(
        step,
        (ReturnStep, GroupedMarginalizeStep, GroupedLatentInitStep),
    ):
        return frozenset()
    raise TypeError(
        f"_step_value_names: unsupported step variant {type(step).__name__}"
    )


_SITE_POSITION = "site"
_VALUE_POSITION = "value"
_COMPOSITE_POSITION = "composite"

_PARAM_SOURCE_REMEDY = (
    "Express the network as explicit sampled weights and a "
    "deterministic forward pass, or write the step as a `sample` / "
    "`observe` against a closed-form family."
)


def _param_source_message(
    *, name: str, kind: str, position: str, line: int, via: str
) -> str:
    """Compose the diagnostic for a parameter-source morphism consumed
    at ``position``. The three positions fail for different reasons
    and a target given the program would fail differently, so each
    says what the emitted program would otherwise have done."""
    if position == _SITE_POSITION:
        return (
            f"morphism {name!r} draws its parameters from a {kind!r} "
            f"network. The network's weights are model-internal and "
            f"appear in neither the wire form nor the sample sites, so "
            f"no backend can reconstruct the mean the morphism "
            f"computes at line {line}. " + _PARAM_SOURCE_REMEDY
        )
    if position == _VALUE_POSITION:
        return (
            f"morphism {name!r} draws its parameters from a {kind!r} "
            f"network, and the program reads it as a value at line "
            f"{line} instead of drawing from it. The morphism is "
            f"consumed inside a composite (a `scan(...)` cell, or a "
            f"link of a `define`d chain), and flattening that "
            f"composite leaves its name in an ordinary expression: the "
            f"declaration's own family contributes no density and the "
            f"network's weights, which are model-internal, have "
            f"nowhere to be emitted. A target given that program would "
            f"read {name!r} as a free input and score a different "
            f"measure. " + _PARAM_SOURCE_REMEDY
        )
    return (
        f"morphism {name!r} draws its parameters from a {kind!r} "
        f"network, and the step at line {line} consumes it through the "
        f"composition {via}. The composite never puts {name!r} in a "
        f"slot a target can emit: its declared family contributes no "
        f"density and the network's weights, which are "
        f"model-internal, have nowhere to go, so a target given that "
        f"program would score a different measure. " + _PARAM_SOURCE_REMEDY
    )


def _reject_param_source_consumed(
    module: Module, morphisms: dict[str, MorphismDecl]
) -> None:
    """Reject a module that consumes a parameter-source morphism.

    Parameter-source morphisms depend on model weights that are absent from
    emitted source and sample sites. This check follows their use from program
    steps through composite definitions and morphism initializers. It rejects
    uses as a draw distribution, value expression, argument, or reachable
    composite leaf. Unused declarations and definitions unreachable from a
    program step are permitted. Traversal follows program order with sorted
    closures, making the reported morphism deterministic.
    """
    lets = build_let_table(module)
    for stmt in module.statements:
        if not isinstance(stmt, ProgramDecl):
            continue
        for step in stmt.draws:
            seeds = [
                (name, _SITE_POSITION) for name in sorted(_step_site_names(step))
            ] + [(name, _VALUE_POSITION) for name in sorted(_step_value_names(step))]
            line = _step_line(step)
            for seed, seed_position in seeds:
                _reject_reachable_param_source(
                    seed=seed,
                    seed_position=seed_position,
                    line=line,
                    morphisms=morphisms,
                    lets=lets,
                )


def _step_line(step: ProgramStep) -> int:
    """Return a step's source line. Every concrete `ProgramStep`
    variant declares ``line: int``, but the tagged-union base does
    not, so the value is narrowed here rather than assumed."""
    line = step.line
    if not isinstance(line, int):
        raise TypeError(
            f"_step_line: {type(step).__name__}.line is {type(line).__name__}, not int"
        )
    return line


def _reject_reachable_param_source(
    *,
    seed: str,
    seed_position: str,
    line: int,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
) -> None:
    """Raise if ``seed``, or any name reachable from it through the
    ``define`` table or a morphism's ``~ <expr>`` init, is a
    parameter-source morphism.

    The frontier is a breadth-first walk in sorted order, and each
    entry carries the chain of names it was reached through so the
    diagnostic can name the composition that consumed the morphism.
    """
    frontier: list[tuple[str, tuple[str, ...]]] = [(seed, ())]
    seen: set[str] = {seed}
    while frontier:
        name, chain = frontier.pop(0)
        decl = morphisms.get(name)
        if decl is not None and _weights_latent(decl):
            kind = param_source_kind(decl, target="qvr-transpile") or "linear"
            raise StepResolutionError(
                "qvr-transpile",
                [
                    f"param-source:{kind}:latent",
                    f"the network of {name!r} (line {line}, through "
                    f"{' -> '.join((*chain, name))}) draws its weights as latent "
                    f"sites, but no target has a form for the network's affine "
                    f"layers applied to drawn weights. Write the layers as "
                    f"`factor` comprehensions over sampled weight matrices to "
                    f"transpile the model.",
                ],
            )
        if decl is not None and _role_of(decl) == "function":
            kind = param_source_kind(decl, target="qvr-transpile") or "linear"
            if kind != "identity":
                raise StepResolutionError(
                    "qvr-transpile",
                    [
                        f"param-source:{kind}",
                        f"the learned function {name!r} (line {line}, through "
                        f"{' -> '.join((*chain, name))}) applies a {kind!r} "
                        f"network whose weights are model-internal: they are "
                        f"neither sites the program declares nor data a target "
                        f"is given, so no target can compute the value it "
                        f"returns. Only a `param_source=identity` function, "
                        f"which has no weights, transpiles.",
                    ],
                )
        elif decl is not None:
            kind = param_source_kind(decl, target="qvr-transpile")
            if kind is not None:
                position = seed_position if not chain else _COMPOSITE_POSITION
                suffix = (
                    "" if position == _SITE_POSITION else f":{position}-position:{name}"
                )
                raise StepResolutionError(
                    "qvr-transpile",
                    [
                        f"param-source:{kind}{suffix}",
                        _param_source_message(
                            name=name,
                            kind=kind,
                            position=position,
                            line=line,
                            via=" -> ".join((*chain, name)),
                        ),
                    ],
                )
        reached: frozenset[str] = frozenset()
        if name in lets:
            reached |= _expr_ident_names(lets[name])
        if decl is not None and decl.init_expr is not None:
            reached |= _expr_ident_names(decl.init_expr)
        for nxt in sorted(reached):
            if nxt in seen:
                continue
            seen.add(nxt)
            frontier.append((nxt, (*chain, name)))


def morphism_table(module: Module) -> dict[str, MorphismDecl]:
    """Return name → MorphismDecl for every morphism declaration in
    ``module``, without the parameter-source boundary check.

    A plural-name declaration contributes one entry per name (each
    name is an independent morphism with the same signature and
    init). Duplicate names are an error.

    Parameters
    ----------
    module
        The parsed module.

    Returns
    -------
    dict[str, MorphismDecl]
        The declaration each morphism name denotes.

    Raises
    ------
    StepResolutionError
        If a morphism name is declared twice.
    """
    out: dict[str, MorphismDecl] = {}
    for stmt in module.statements:
        if isinstance(stmt, MorphismDecl):
            for name in stmt.names:
                if name in out:
                    msg = (
                        f"duplicate morphism declaration {name!r}: "
                        f"first at line {out[name].line}, again at "
                        f"line {stmt.line}"
                    )
                    raise StepResolutionError("qvr-transpile", [msg])
                out[name] = stmt
    return out


def build_morphism_table(module: Module) -> dict[str, MorphismDecl]:
    """Return name → MorphismDecl for every morphism declaration in
    ``module``, checked for the transpile boundary.

    The table is [`morphism_table`][quivers.dsl.step_resolution.morphism_table]'s;
    building it is also where a module is checked for parameter-source
    morphisms it consumes, in a site, value or composite position; see
    ``_reject_param_source_consumed``. The table is the resolver's
    authority on what each name denotes, so it is the one place that
    sees every declaration at once, which is what deciding a
    module-wide boundary needs.

    Parameters
    ----------
    module
        The parsed module.

    Returns
    -------
    dict[str, MorphismDecl]
        The declaration each morphism name denotes.

    Raises
    ------
    StepResolutionError
        If a morphism name is declared twice, or a parameter-source
        morphism is consumed.
    """
    out = morphism_table(module)
    _reject_param_source_consumed(module, out)
    return out


def build_let_table(module: Module) -> dict[str, Expr]:
    """Return name → expr for every top-level ``let_decl``."""
    out: dict[str, Expr] = {}
    for stmt in module.statements:
        if isinstance(stmt, DefineDecl):
            out[stmt.name] = stmt.expr
    return out


def resolve_step_dist(
    morphism_name: str,
    raw_args: tuple[DrawArg, ...] | None,
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    family_registry: frozenset[str],
    target: str,
    _seen: tuple[str, ...] = (),
) -> ResolvedDist:
    """Resolve a sample / observe step's morphism slot to a
    ``(family, args)`` pair.

    Parameters
    ----------
    morphism_name
        The string in ``SampleStep.morphism`` /
        ``ObserveStep.morphism``. Either a family name, a declared
        morphism's name, or a let-binding name.
    raw_args
        Positional arguments on the step (``None`` is treated as
        empty). When the resolver unfolds a morphism with its own
        ``~ Family(args)`` init clause, the step-supplied ``raw_args``
        take precedence over the declaration's defaults; in current
        practice the step does not supply args when referring to a
        morphism (the morphism's args carry the parameter set).
    morphisms
        Name → MorphismDecl table from
        [`build_morphism_table`][quivers.dsl.step_resolution.build_morphism_table].
    lets
        Name → Expr table from
        [`build_let_table`][quivers.dsl.step_resolution.build_let_table].
    family_registry
        Frozen set of canonical QVR family names. When
        ``morphism_name`` is in this set, the resolver returns
        immediately with the step's own args.
    target
        Backend name (for error messages).
    _seen
        Internal: the chain of names visited during resolution. The
        resolver detects cycles (``let a = b; let b = a``) by
        membership in this tuple.
    """
    if morphism_name in family_registry:
        wire: tuple[str | float, ...]
        if raw_args:
            wire = tuple(_draw_arg_to_wire(a) for a in raw_args)
        else:
            wire = _FAMILY_DEFAULT_ARGS.get(morphism_name, ())
        return ResolvedDist(
            family=morphism_name,
            args=wire,
            original_morphism_name=morphism_name,
        )

    if morphism_name in _seen:
        msg = (
            f"morphism / let cycle while resolving {morphism_name!r}; "
            f"chain: {' -> '.join((*_seen, morphism_name))}"
        )
        raise StepResolutionError(target, [msg])

    chain = (*_seen, morphism_name)

    if morphism_name in morphisms:
        decl = morphisms[morphism_name]
        param_source = param_source_kind(decl, target=target)
        if param_source is not None:
            msg = (
                f"morphism {morphism_name!r} draws its parameters from a "
                f"{param_source!r} network. The network's weights are "
                f"model-internal and appear in neither the wire form nor the "
                f"sample sites, so no backend can reconstruct the mean the "
                f"morphism computes. Express the network as explicit sampled "
                f"weights and a deterministic forward pass, or observe against "
                f"a closed-form family."
            )
            raise StepResolutionError(target, [f"param-source:{param_source}", msg])
        if decl.init_family is not None:
            return _from_init_family(
                morphism_name=morphism_name,
                init=decl.init_family,
                step_args=raw_args,
                chain=chain,
                morphism_options=decl.options,
            )
        if decl.init_expr is not None:
            # When the init expression is a bare family identifier
            # (`morphism foo : T -> T [scale=0.1] ~ Normal`) the
            # parser models it as an `init_expr=ExprIdent("Normal")`
            # rather than `init_family=Normal`. Route this case
            # through the same option-aware family-merge that
            # `_from_init_family` uses so the morphism's option
            # block (`[scale=0.1]`) populates the matching family
            # parameter slot.
            if (
                isinstance(decl.init_expr, ExprIdent)
                and decl.init_expr.name in family_registry
            ):
                return _from_init_family(
                    morphism_name=morphism_name,
                    init=MorphismInitFamily(
                        family=decl.init_expr.name,
                        args=(),
                    ),
                    step_args=raw_args,
                    chain=chain,
                    morphism_options=decl.options,
                )
            return _resolve_expr(
                morphism_name=morphism_name,
                expr=decl.init_expr,
                raw_args=raw_args,
                morphisms=morphisms,
                lets=lets,
                family_registry=family_registry,
                target=target,
                chain=chain,
            )
        if _role_of(decl) == "embed":
            msg = (
                f"morphism {morphism_name!r} is an embedding, which places a "
                f"Gaussian kernel at each element's learned centre; the "
                f"centres and scales are model-internal tables that appear "
                f"in neither the wire form nor the sample sites, so no "
                f"backend can reconstruct the kernel."
            )
            raise StepResolutionError(target, [f"embed:{morphism_name}", msg])
        msg = (
            f"morphism {morphism_name!r} has neither `~ Family(...)` "
            f"nor `~ <expr>` init clause; transpile cannot derive a "
            f"distribution"
        )
        raise StepResolutionError(target, [msg])

    if morphism_name in lets:
        return _resolve_expr(
            morphism_name=morphism_name,
            expr=lets[morphism_name],
            raw_args=raw_args,
            morphisms=morphisms,
            lets=lets,
            family_registry=family_registry,
            target=target,
            chain=chain,
        )

    msg = (
        f"sample / observe step references {morphism_name!r} which "
        f"is neither a family in the registry, a declared morphism, "
        f"nor a let-bound name"
    )
    raise StepResolutionError(target, [f"family:{morphism_name}", msg])


def _from_init_family(
    *,
    morphism_name: str,
    init: MorphismInitFamily,
    step_args: tuple[DrawArg, ...] | None,
    chain: tuple[str, ...],
    morphism_options: tuple[OptionEntry, ...] = (),
) -> ResolvedDist:
    """Unfold a ``~ Family(args)`` init clause.

    ``init.args`` and the step's own args are structural `DrawArg`
    variants and are lowered to wire form here. When the declaration
    carries explicit init args and the step also supplies args, the
    step args take precedence; otherwise the declaration's init args
    are used.

    When the init clause is the bare ``~ Family`` form (no
    parentheses), arg slots are filled in canonical order from
    three sources:

    1. Step args (``emission(s_new)``) populate the leading
       positional slots.
    2. Remaining slots whose canonical name matches a morphism
       option (`[scale=0.1]` -> the `scale` slot of `Normal`) take
       the option value.
    3. Anything still empty falls back to the family's canonical
       default (`_FAMILY_DEFAULT_ARGS`).
    """
    if init.args:
        wire: tuple[str | float, ...]
        if step_args:
            wire = tuple(_draw_arg_to_wire(a) for a in step_args)
        else:
            wire = tuple(_draw_arg_to_wire(argument) for argument in init.args)
        return ResolvedDist(
            family=init.family,
            args=wire,
            original_morphism_name=morphism_name,
            via=chain[:-1] if chain else (),
        )
    defaults = _FAMILY_DEFAULT_ARGS.get(init.family, ())
    arg_names = _FAMILY_ARG_NAMES.get(init.family, ())
    option_map = _options_to_map(morphism_options)
    step_wire = tuple(_draw_arg_to_wire(a) for a in step_args) if step_args else ()
    wire_list: list[str | float] = list(step_wire)
    for i in range(len(step_wire), len(defaults)):
        name = arg_names[i] if i < len(arg_names) else None
        if name is not None and name in option_map:
            wire_list.append(option_map[name])
        else:
            wire_list.append(defaults[i])
    return ResolvedDist(
        family=init.family,
        args=tuple(wire_list),
        original_morphism_name=morphism_name,
        via=chain[:-1] if chain else (),
    )


def _options_to_map(
    options: tuple[OptionEntry, ...],
) -> dict[str, str | float]:
    """Reduce a tuple of `OptionEntry` to a `key -> wire-value` map
    of numeric / string-valued options. Identifier / list / call
    option shapes are ignored (they describe declaration metadata
    like `role=kernel` rather than family arguments)."""
    out: dict[str, str | float] = {}
    for entry in options:
        value = entry.value
        if isinstance(value, OptionNumber):
            out[entry.key] = value.value
        elif isinstance(value, OptionString):
            out[entry.key] = value.value
    return out


# Per-family ordered parameter names. Mirrors the positional layout
# used by `_FAMILY_DEFAULT_ARGS` so an option-block entry
# (`scale=0.1`) can be routed to the right slot by name.
#
# IMPORTANT: the names AND their order must match
# `torch.distributions.<Family>.arg_constraints.keys()` for every
# entry where torch has a Distribution of that name. The
# `_assert_family_arg_names_match_torch` startup check below
# enforces this so a typo here cannot silently transpose the loc
# and scale slots (which would route `[scale=0.1]` to `loc=0.1`).
_FAMILY_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "Normal": ("loc", "scale"),
    "HalfNormal": ("scale",),
    "Cauchy": ("loc", "scale"),
    "HalfCauchy": ("scale",),
    "Laplace": ("loc", "scale"),
    "LogNormal": ("loc", "scale"),
    "Beta": ("concentration1", "concentration0"),
    "Bernoulli": ("probs",),
    "Gamma": ("concentration", "rate"),
    "InverseGamma": ("concentration", "rate"),
    "Exponential": ("rate",),
    "Uniform": ("low", "high"),
    "StudentT": ("df", "loc", "scale"),
    "Pareto": ("scale", "alpha"),
    "Weibull": ("scale", "concentration"),
}


def _assert_family_arg_names_match_torch() -> None:
    """Startup invariant: every entry in `_FAMILY_ARG_NAMES` for a
    family that torch ships must match that distribution's positional
    constructor parameters in name AND order.

    The constructor signature, not ``arg_constraints``, is the
    positional contract a QVR call site writes against: ``~ Pareto(a,
    b)`` binds ``a`` and ``b`` the way ``torch.distributions.Pareto``
    binds them, and the conditional family classes in
    [quivers.continuous.families][] declare their parameters in the
    same order. For most families the two agree, but they can diverge
    (``Pareto.arg_constraints`` is keyed ``alpha, scale`` while the
    constructor takes ``scale, alpha``), and following the wrong one
    transposes the slots and silently changes the density.

    Raises `AssertionError` at module-import time if the table drifts
    from torch's ground truth. The cost of running this once per
    process is negligible; the cost of letting a transposition bug
    ship is a silent wrong-density that no syntactic test catches.

    Families that are not present in torch (the custom shim families
    defined in `family_meta`) are skipped: the table's entry is the
    source of truth for those.
    """
    for family, declared in _FAMILY_ARG_NAMES.items():
        cls = getattr(_td, family, None)
        if cls is None:
            continue
        try:
            params = _inspect.signature(cls.__init__).parameters
        except TypeError, ValueError:
            continue
        # Alternative parameterisations trail the canonical ones
        # (``Bernoulli(probs, logits)``), so compare against the
        # leading prefix; transposition is still caught.
        ctor_names = tuple(
            name for name in params if name not in ("self", "validate_args")
        )
        prefix = ctor_names[: len(declared)]
        if declared != prefix:
            raise AssertionError(
                f"_FAMILY_ARG_NAMES[{family!r}] = {declared!r} does "
                f"not match the leading positional parameters of "
                f"torch.distributions.{family}.__init__ "
                f"({ctor_names!r}). The option-block routing depends "
                "on this ordering matching torch's positional "
                "parameter contract; fix the table to agree with the "
                "constructor signature."
            )


_assert_family_arg_names_match_torch()


# Canonical default args for kernel morphisms declared `~ Family` with
# no explicit init args. These let the resolver produce a target call
# with the right arity for backends whose family signature is
# positional (Stan's `normal(0, 1)`, WebPPL's `Gaussian({mu, sigma})`).
# Mirror of `_expand_composites._FAMILY_DEFAULT_ARGS` for
# resolver-side use (the expansion pass does in-chain substitution,
# this fallback is for standalone kernels).
_FAMILY_DEFAULT_ARGS: dict[str, tuple[str | float, ...]] = {
    "Normal": (0.0, 1.0),
    "HalfNormal": (1.0,),
    "Cauchy": (0.0, 1.0),
    "HalfCauchy": (1.0,),
    "Laplace": (0.0, 1.0),
    "LogNormal": (0.0, 1.0),
    "Beta": (1.0, 1.0),
    "Bernoulli": (0.5,),
    "Gamma": (1.0, 1.0),
    "InverseGamma": (1.0, 1.0),
    "Exponential": (1.0,),
    "Uniform": (0.0, 1.0),
    "StudentT": (1.0, 0.0, 1.0),
    "Pareto": (1.0, 1.0),
    "Weibull": (1.0, 1.0),
    # Multivariate-shape families (`MultivariateNormal`,
    # `MatrixNormal`, `GP`) have no entry here. They opt in to
    # structured lowering via the `structured_lowering` field on their
    # [`FamilyMeta`][quivers.transpile.family_meta.FamilyMeta]; the
    # plan then reads the bare ``~ Family`` form's structured
    # arguments off the checked construction and
    # synthesises the data-input names with the right matrix / vector
    # / cov-matrix shape from the morphism's `[over=...]` axes. A
    # placeholder string here would survive into the data block as a
    # free identifier and produce invalid syntax in every backend.
}


def _resolve_expr(
    *,
    morphism_name: str,
    expr: Expr,
    raw_args: tuple[DrawArg, ...] | None,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    family_registry: frozenset[str],
    target: str,
    chain: tuple[str, ...],
) -> ResolvedDist:
    """Unfold a let / morphism init expression. Pure aliases
    (``let a = b``) recurse into the aliased name; composite
    expressions (``a >> b``, ``a @ b``, ``scan(cell)``, ``fan(a, b)``)
    are normally pre-expanded into atomic sample / let chains by
    [`expand_composite_lets`][quivers.dsl.composite_lets.expand_composite_lets]
    before lowering reaches the resolver. Reaching this branch means
    the pre-expansion pass bailed (an unrecognised leaf shape inside
    the composition) and the surface composite has flowed through to
    the resolver intact; raise a precise unsupported-construct error
    naming the offending kind."""
    if isinstance(expr, ExprIdent):
        return resolve_step_dist(
            expr.name,
            raw_args,
            morphisms=morphisms,
            lets=lets,
            family_registry=family_registry,
            target=target,
            _seen=chain,
        )
    expr_kind = str(getattr(expr, "kind", type(expr).__name__))
    raise StepResolutionError(
        target,
        [
            f"let:composite_expression:{expr_kind}",
        ],
    )


__all__ = []
