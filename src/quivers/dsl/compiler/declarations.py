"""Compiler mixin: declaration-level statements.

Handles the consolidated Statement family: composition, category,
rule, schema, bundle, type, morphism. The mixin dispatches on the
`TypeInitializer` variant inside a `ObjectDecl` and on
the ``role`` option inside a `MorphismDecl` to pick the
runtime construction.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import cast

import torch

from quivers.analysis.init_spec import _algebra_init_spec
from quivers.core.algebras import (
    Algebra,
    BilinearForm,
    BooleanAlgebra,
    CompositionRule,
    CustomAlgebra,
    CustomBilinearForm,
    CustomSemigroupoid,
    GodelAlgebra,
    LukasiewiczAlgebra,
    ProbabilityAlgebra,
    ProductFuzzyAlgebra,
    Semigroupoid,
)
from quivers.continuous.boundaries import Discretize, Embed
from quivers.continuous.deterministic import (
    FUNCTION_OUTPUTS,
    FunctionOutput,
    LearnedFunction,
)
from quivers.continuous.flows import ConditionalFlow
from quivers.continuous.inline import make_inline_distribution
from quivers.continuous.param_source import FunctionActivation, function_source
from quivers.continuous.spaces import ContinuousSpace, ProductSpace
from quivers.core.morphisms import (
    ObservedMorphism,
    morphism as make_latent,
)
from quivers.core.objects import (
    EnumSet,
    FinSet,
    FreeMonoid,
    FreeResiduated,
    SetObject,
)
from quivers.dsl.ast_nodes import (
    AxisSpec,
    BundleDecl,
    CategoryDecl,
    CompositionDecl,
    CompositionRuleEntry,
    DrawArg,
    DrawArgNamed,
    ExprIdent,
    MorphismDecl,
    RuleDecl,
    SchemaDecl,
    Module,
    ObjectDecl,
    TypeEnumSet,
    TypeName,
    ObjectCoproduct,
    ObjectEffectApply,
    ObjectExpr,
    ObjectSlash,
    TypeFreeMonoid,
    TypeFreeResiduated,
    TypeFromExpr,
    ObjectProduct,
)
from quivers.dsl.compiler._options import (
    check_option_keys,
    find_option,
    get_option_call_text,
    get_option_float,
    get_option_int_list,
    get_option_int,
    get_option_name,
    get_option_name_list,
)
from quivers.dsl.compiler._prelude import (
    _ALGEBRA_REGISTRY,
    CompileError,
    _available_axes_for,
    _get_family_registry,
    _shape_size,
    _validate_axis_spec,
    _wrap_join_dim,
)
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.dsl.draw_args import kernel_argument_refusal
from quivers.stochastic import StochasticMorphism
from quivers.stochastic.categories import (
    BUILTIN_CONSTRUCTOR_NAMES,
    constructor_name_error,
)
from quivers.stochastic.schema import (
    SCHEMA_REGISTRY,
    PatternBinarySchema,
    PatternUnarySchema,
)


_VALID_ROLES: frozenset[str] = frozenset(
    {"latent", "observed", "kernel", "embed", "discretize", "let", "function"}
)


def _is_residuated_expr(expr: ObjectExpr, alias_names: set[str]) -> bool:
    """Whether an object expression lives in the residuated stratum.

    The residuated formers (``/``, ``\\``, and effect application)
    denote in a residuated-monoidal universe rather than in
    **FinSet** or in the continuous-space stratum, so a declaration
    whose right-hand side contains one is recorded as a pattern
    alias instead of being resolved to a concrete object. A bare
    name that already stands for such a pattern carries the same
    reading, and a product or coproduct inherits it from any
    component.
    """
    if isinstance(expr, (ObjectSlash, ObjectEffectApply)):
        return True
    if isinstance(expr, TypeName):
        return expr.name in alias_names
    if isinstance(expr, (ObjectProduct, ObjectCoproduct)):
        return any(_is_residuated_expr(c, alias_names) for c in expr.components)
    return False


def _type_names(expr: ObjectExpr) -> list[TypeName]:
    """Collect every bare name an object expression mentions."""
    if isinstance(expr, TypeName):
        return [expr]
    if isinstance(expr, (ObjectProduct, ObjectCoproduct)):
        return [n for c in expr.components for n in _type_names(c)]
    if isinstance(expr, ObjectSlash):
        return _type_names(expr.result) + _type_names(expr.argument)
    if isinstance(expr, ObjectEffectApply):
        return [n for a in expr.args for n in _type_names(a)]
    return []


def _bare_message(exc: CompileError) -> str:
    """Return a `CompileError`'s message without its location prefix.

    The prefix is re-attached by the enclosing `CompileError` that
    adds the declaration's name, so carrying it through twice would
    print the position inside the sentence as well as in front of it.
    """
    text = str(exc)
    prefix = f"line {exc.line}, col {exc.col}: "
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


# Closed option-key set for ``morphism`` declarations. Every key the
# compiler reads off a `MorphismDecl`'s option block appears here:
#
# * ``role`` / ``replicate`` pick the lowering and its multiplicity;
# * ``scale`` / ``init`` configure the latent lowering;
# * ``bins`` configures the discretize lowering;
# * ``over`` / ``iid`` carry the axis-role clause for family inits
#   (``over`` doubles as the MatrixNormal rows/cols selector);
# * ``n_layers`` / ``hidden_dim`` / ``param_source`` / ``rank`` /
#   ``temperature`` configure family-backed kernel construction.
#
# Family construction is NOT open-ended: `_make_continuous_morphism`
# threads exactly these keys into the family constructors, so the set
# is complete and strict checking is safe for every family.
_MORPHISM_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "role",
        "replicate",
        "scale",
        "init",
        "bins",
        "over",
        "iid",
        "n_layers",
        "hidden_dim",
        "param_source",
        "rank",
        "temperature",
        "activation",
        "output",
    }
)

# The union above spans every role, so checking against it alone would
# accept a key the chosen lowering never reads and drop it in silence.
# Each role therefore declares the keys it actually consumes, and a key
# outside its set is rejected rather than ignored: `scale` configures a
# latent morphism's init and means nothing to a family-backed kernel,
# whose parameters come from its `param_source`.
#
# ``role`` and ``replicate`` pick the lowering and its multiplicity, so
# every role reads them.
_COMMON_MORPHISM_OPTION_KEYS: frozenset[str] = frozenset({"role", "replicate"})

_ROLE_OPTION_KEYS: dict[str, frozenset[str]] = {
    # `_compile_latent_role` reads scale / init; a `~ Family(...)` init
    # carries its axis-role clause through `_validate_family_axes`.
    "latent": frozenset({"scale", "init", "over", "iid"}),
    "observed": frozenset(),
    # `_make_continuous_morphism` threads these into the family.
    "kernel": frozenset(
        {
            "n_layers",
            "hidden_dim",
            "param_source",
            "rank",
            "temperature",
            "over",
            "iid",
        }
    ),
    "embed": frozenset(),
    "discretize": frozenset({"bins"}),
    "let": frozenset({"over", "iid"}),
    # `_compile_function_role` builds the network and its output map.
    "function": frozenset({"param_source", "activation", "output"}),
}


def _draw_arg_location(arg: DrawArg, decl: MorphismDecl) -> tuple[int, int]:
    """The source position to blame for one of a declaration's arguments.

    Parameters
    ----------
    arg : DrawArg
        An argument of the declaration's ``~ Family(...)`` initializer.
    decl : MorphismDecl
        The declaration.

    Returns
    -------
    tuple[int, int]
        The argument's line and column when the parser recorded them,
        else the initializer's, else the declaration's.
    """
    if isinstance(arg, DrawArgNamed) and arg.line:
        return arg.line, arg.col
    init = decl.init_family
    if init is not None and init.line:
        return init.line, init.col
    return decl.line, decl.col


def _signature_params(cls) -> frozenset[str]:
    """The keyword names a family's ``__init__`` accepts."""
    return frozenset(inspect.signature(cls.__init__).parameters)


def _apply_auto_init(morph, domain, codomain, algebra) -> None:
    """Apply the algebra's saturation-free init recipe to a freshly
    constructed `LatentMorphism`.

    The recipe is computed at depth 1 (a top-level latent declaration
    is, in isolation, a one-step morphism; downstream composition is
    out of scope for the static recipe) with the larger of the
    morphism's resolved domain / codomain numel as the intermediate
    axis size. The recipe is in value space; the raw parameter feeds
    through `LatentMorphism`'s sigmoid bijector, so for the
    sigmoid case we invert via ``logit`` before sampling; for
    algebras whose latent representation does not pass through a
    bijector (Markov, log-prob, real, max-plus, tropical) the recipe
    is applied to the raw parameter directly.
    """

    def _numel(obj) -> int:
        shape = getattr(obj, "shape", ())
        n = 1
        for s in shape:
            n *= int(s)
        return max(1, n)

    intermediate_size = max(_numel(domain), _numel(codomain))
    spec = _algebra_init_spec(
        algebra,
        depth=1,
        intermediate_size=intermediate_size,
    )
    raw = morph.raw
    is_sigmoid_bijected = isinstance(
        algebra,
        (
            ProductFuzzyAlgebra,
            BooleanAlgebra,
            GodelAlgebra,
            LukasiewiczAlgebra,
            ProbabilityAlgebra,
        ),
    )

    def _logit(p: float) -> float:
        p = max(min(p, 1.0 - 1e-6), 1e-6)
        return math.log(p / (1.0 - p))

    with torch.no_grad():
        if spec.distribution == "constant":
            target = _logit(spec.mean) if is_sigmoid_bijected else spec.mean
            raw.data.fill_(target)
        elif spec.distribution == "uniform":
            if is_sigmoid_bijected:
                lo = _logit(spec.lower)
                hi = _logit(spec.upper)
            else:
                lo = spec.lower
                hi = spec.upper
            raw.data.uniform_(lo, hi)
        else:
            mean = _logit(spec.mean) if is_sigmoid_bijected else spec.mean
            raw.data.normal_(mean=mean, std=max(spec.std, 1e-6))


class _DeclarationsMixin:
    """Mixin: declaration-level compilation methods.

    The compiler base supplies every environment slot below; the
    annotations let the type checker verify each access from a
    mixin method.
    """

    _algebra: CompositionRule
    _categories: list[str]
    _rules: dict
    _bundles: dict[str, tuple[str, ...]]
    _aliases: dict[str, ObjectExpr]
    _alias_names: set[str]
    _objects: dict[str, SetObject]
    _spaces: dict[str, ContinuousSpace]
    _morphisms: dict
    _groups: dict[str, list[str]]
    _module: Module

    # ``_resolve_type``, ``_resolve_any_space``, ``_compile_expr``
    # come from `_ResolutionMixin` and
    # `_ExpressionsMixin` via the ``Compiler`` MRO.

    # ------------------------------------------------------------------
    # composition
    # ------------------------------------------------------------------

    def _compile_composition(self, decl: CompositionDecl) -> None:
        """Install ``decl`` as the active composition rule for this module.

        ``decl.level`` advertises the algebraic level at which the rule
        is offered (``algebra`` / ``semigroupoid`` / ``bilinear_form``
        / ``rule``). Two body forms:

        * empty body: ``decl.name`` references a rule in the registry;
          the compiler confirms its concrete type matches the
          declared level.
        * non-empty body: ``decl.body`` defines the rule's operations
          inline; the compiler synthesises a ``CustomAlgebra`` /
          ``CustomSemigroupoid`` / ``CustomBilinearForm`` of the
          declared level.

        The declaration's only option key is ``level``; the parser
        collapses the option block into ``decl.level`` (a typed
        Literal), so there is no raw option tuple left to key-check
        here.
        """
        level = decl.level or "rule"
        if decl.body:
            rule = self._build_custom_composition_rule(decl, level)
        else:
            key = decl.name.lower()
            if key not in _ALGEBRA_REGISTRY:
                raise CompileError(
                    f"unknown {level} {decl.name!r}; available: "
                    f"{', '.join(sorted(_ALGEBRA_REGISTRY))}",
                    decl.line,
                    decl.col,
                )
            rule = _ALGEBRA_REGISTRY[key]
        self._verify_composition_rule_level(rule, decl, level)
        self._algebra = rule  # type: ignore[assignment]

    def _verify_composition_rule_level(
        self,
        rule: "CompositionRule",
        decl: CompositionDecl,
        level: str,
    ) -> None:
        required = {
            "algebra": Algebra,
            "semigroupoid": Semigroupoid,
            "bilinear_form": BilinearForm,
            "rule": CompositionRule,
        }
        required_class = required.get(level, CompositionRule)
        if not isinstance(rule, required_class):
            actual = (
                "Algebra"
                if isinstance(rule, Algebra)
                else "Semigroupoid"
                if isinstance(rule, Semigroupoid)
                else "BilinearForm"
                if isinstance(rule, BilinearForm)
                else "CompositionRule"
            )
            raise CompileError(
                f"composition {decl.name!r}: registered rule is a "
                f"{actual}, which is not at level {level!r}. Declare "
                f"with the matching ``[level=...]`` option or register "
                f"a rule at the right level.",
                decl.line,
                decl.col,
            )

    def _build_custom_composition_rule(
        self,
        decl: CompositionDecl,
        level: str,
    ) -> "CompositionRule":
        entries: dict[str, CompositionRuleEntry] = {}
        for entry in decl.body:
            if entry.key in entries:
                raise CompileError(
                    f"composition {decl.name!r}: duplicate entry {entry.key!r}",
                    entry.line,
                    entry.col,
                )
            entries[entry.key] = entry
        required_keys = {
            "algebra": {"tensor_op", "join", "unit", "zero"},
            "semigroupoid": {"tensor_op", "join"},
            "bilinear_form": {"tensor_op", "join"},
            "rule": {"tensor_op", "join"},
        }[level]
        missing = required_keys - set(entries)
        if missing:
            raise CompileError(
                f"composition {decl.name!r}: missing required entries "
                f"{sorted(missing)}",
                decl.line,
                decl.col,
            )
        compiled: dict[str, "Callable[..., torch.Tensor] | float"] = {}
        for key, entry in entries.items():
            compiled[key] = self._compile_composition_rule_entry(entry, decl)
        tensor_op = compiled["tensor_op"]
        join = compiled["join"]
        if not callable(tensor_op):
            raise CompileError(
                f"composition {decl.name!r}: ``tensor_op`` must be a "
                f"function entry like ``tensor_op(a, b) = …``",
                decl.line,
                decl.col,
            )
        if not callable(join):
            raise CompileError(
                f"composition {decl.name!r}: ``join`` must be a "
                f"function entry like ``join(t) = …``",
                decl.line,
                decl.col,
            )
        join_wrapped = _wrap_join_dim(join)
        if level == "algebra":
            unit = compiled["unit"]
            zero = compiled["zero"]
            if callable(unit) or callable(zero):
                raise CompileError(
                    f"composition {decl.name!r}: ``unit`` and ``zero`` "
                    f"must be value entries (no parens)",
                    decl.line,
                    decl.col,
                )
            negate_fn = compiled.get("negation")
            if negate_fn is not None and not callable(negate_fn):
                raise CompileError(
                    f"composition {decl.name!r}: ``negation`` must be "
                    f"a function entry like ``negation(a) = …``",
                    decl.line,
                    decl.col,
                )
            return CustomAlgebra(
                name=decl.name,
                tensor_op=tensor_op,
                join=join_wrapped,
                unit=float(unit),
                zero=float(zero),
                negate=negate_fn,
                verify=False,
            )
        if level == "semigroupoid":
            return CustomSemigroupoid(
                name=decl.name,
                tensor_op=tensor_op,
                join=join_wrapped,
                verify_associative=False,
            )
        return CustomBilinearForm(
            name=decl.name,
            tensor_op=tensor_op,
            join=join_wrapped,
        )

    def _compile_composition_rule_entry(
        self,
        entry: CompositionRuleEntry,
        decl: CompositionDecl,
    ) -> "Callable[..., torch.Tensor] | float":
        body_fn = _ProgramsMixin._compile_let_expr(entry.body)
        if not entry.params:
            try:
                return body_fn({})
            except Exception as exc:
                raise CompileError(
                    f"composition {decl.name!r}: value entry "
                    f"{entry.key!r} could not be evaluated: {exc}",
                    entry.line,
                    entry.col,
                ) from exc
        param_names = entry.params

        def _callable(*args):
            if len(args) != len(param_names):
                raise TypeError(
                    f"{entry.key}(): expected {len(param_names)} "
                    f"arguments, got {len(args)}"
                )
            env = dict(zip(param_names, args))
            return body_fn(env)

        return _callable

    # ------------------------------------------------------------------
    # category
    # ------------------------------------------------------------------

    def _compile_category(self, decl: CategoryDecl) -> None:
        """Register one or more category-atom generators.

        ``decl.names`` is a tuple to accommodate the comma-separated
        surface form ``category A, B, C``; each name is appended to
        the running category list.
        """
        for name in decl.names:
            if name in self._categories:
                raise CompileError(
                    f"category {name!r} already declared",
                    decl.line,
                    decl.col,
                )
            self._categories.append(name)

    # ------------------------------------------------------------------
    # rule / schema
    # ------------------------------------------------------------------

    def _compile_rule(self, decl: RuleDecl) -> None:
        """Compile an inference rule into a pattern schema."""
        if decl.name in self._rules:
            raise CompileError(
                f"rule {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"rule {decl.name!r} shadows a built-in schema; "
                f"choose a different name",
                decl.line,
                decl.col,
            )
        variables = frozenset(decl.variables)
        n_premises = len(decl.premises)
        if n_premises == 2:
            schema = PatternBinarySchema(
                left_pattern=decl.premises[0],
                right_pattern=decl.premises[1],
                conclusion_pattern=decl.conclusion,
                variables=variables,
                name=decl.name,
            )
        elif n_premises == 1:
            schema = PatternUnarySchema(
                premise_pattern=decl.premises[0],
                conclusion_pattern=decl.conclusion,
                variables=variables,
                name=decl.name,
            )
        else:
            raise CompileError(
                f"rule {decl.name!r} has {n_premises} premises; only "
                f"unary (1) and binary (2) rules are supported",
                decl.line,
                decl.col,
            )
        self._rules[decl.name] = schema

    def _compile_schema(self, decl: SchemaDecl) -> None:
        """Compile a pattern-polymorphic schema declaration."""
        if decl.name in self._rules:
            raise CompileError(
                f"schema {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"schema {decl.name!r} shadows a built-in schema; "
                f"choose a different name",
                decl.line,
                decl.col,
            )
        variables: frozenset[str] = frozenset(
            n for group in decl.parameters for n in group.names
        )
        if isinstance(decl.domain, ObjectProduct) and len(decl.domain.components) == 2:
            left, right = decl.domain.components
            schema = PatternBinarySchema(
                left_pattern=left,
                right_pattern=right,
                conclusion_pattern=decl.codomain,
                variables=variables,
                name=decl.name,
            )
        else:
            schema = PatternUnarySchema(
                premise_pattern=decl.domain,
                conclusion_pattern=decl.codomain,
                variables=variables,
                name=decl.name,
            )
        self._rules[decl.name] = schema

    # ------------------------------------------------------------------
    # bundle
    # ------------------------------------------------------------------

    def _compile_bundle(self, decl: BundleDecl) -> None:
        """Register a rule bundle for later expansion at use site."""
        if decl.name in self._bundles:
            raise CompileError(
                f"bundle {decl.name!r} already declared",
                decl.line,
                decl.col,
            )
        if decl.name in self._rules or decl.name in SCHEMA_REGISTRY:
            raise CompileError(
                f"bundle {decl.name!r} shadows a rule / built-in schema",
                decl.line,
                decl.col,
            )
        self._bundles[decl.name] = tuple(decl.rules)

    # ------------------------------------------------------------------
    # type
    # ------------------------------------------------------------------

    def _compile_type(self, decl: ObjectDecl) -> None:
        """Compile a ``type NAME, ... : VALUE`` declaration.

        Each name in ``decl.names`` registers an independent object
        constructed from the same VALUE. The init's tagged-union
        variant picks the construction:

        * `TypeEnumSet` -> `EnumSet`
        * `TypeFreeResiduated` -> `FreeResiduated`
        * `TypeFreeMonoid` -> `FreeMonoid`
        * `TypeFromExpr` -> the inner type expression is
          resolved via the unified resolver; the resulting object is
          either a `SetObject` (discrete) or a
          `ContinuousSpace`, bound under the declared name in the
          appropriate environment.
        """
        for name in decl.names:
            self._compile_type_named(decl, name)

    def _compile_type_named(self, decl: ObjectDecl, name: str) -> None:
        if name in self._objects or name in self._spaces:
            raise CompileError(
                f"type {name!r} already declared",
                decl.line,
                decl.col,
            )
        init = decl.init
        if isinstance(init, TypeEnumSet):
            self._objects[name] = EnumSet(
                name=name,
                elements=init.elements,
            )
            return
        if isinstance(init, TypeFreeMonoid):
            gen = self._objects.get(init.generators)
            if not isinstance(gen, FinSet):
                raise CompileError(
                    f"FreeMonoid generators {init.generators!r} must "
                    f"reference a previously-declared FinSet (got "
                    f"{type(gen).__name__ if gen else 'undefined'})",
                    decl.line,
                    decl.col,
                )
            self._objects[name] = FreeMonoid(
                generators=gen,
                max_length=init.max_length,
            )
            return
        if isinstance(init, TypeFreeResiduated):
            gen = self._objects.get(init.generators)
            if not isinstance(gen, EnumSet):
                raise CompileError(
                    f"FreeResiduated generators {init.generators!r} "
                    f"must reference a previously-declared EnumSet "
                    f"(got "
                    f"{type(gen).__name__ if gen else 'undefined'})",
                    decl.line,
                    decl.col,
                )
            if not init.ops:
                raise CompileError(
                    f"FreeResiduated {name!r}: ops=[] closes under no "
                    f"connective, leaving only the generator atoms; "
                    f"name at least one of "
                    f"{sorted(BUILTIN_CONSTRUCTOR_NAMES)}",
                    decl.line,
                    decl.col,
                )
            for op in init.ops:
                if op not in BUILTIN_CONSTRUCTOR_NAMES:
                    raise CompileError(
                        f"FreeResiduated {name!r}: {constructor_name_error(op)}",
                        decl.line,
                        decl.col,
                    )
            if init.depth < 0:
                raise CompileError(
                    f"FreeResiduated {name!r}: depth must be "
                    f"non-negative, got {init.depth}",
                    decl.line,
                    decl.col,
                )
            self._objects[name] = FreeResiduated(
                generators=gen,
                depth=init.depth,
                ops=init.ops,
            )
            return
        if isinstance(init, TypeFromExpr):
            expr = init.expr
            if _is_residuated_expr(expr, self._alias_names):
                # Residuated patterns and effect-typed RHS have no
                # denotation in the bare object/space stratum; record
                # the alias for use-site substitution inside schema
                # patterns.
                if name in self._alias_names:
                    raise CompileError(
                        f"alias {name!r} already declared",
                        decl.line,
                        decl.col,
                    )
                self._alias_names.add(name)
                self._aliases[name] = expr
                return
            try:
                resolved = self._resolve_any_space(expr)
            except CompileError as exc:
                raise CompileError(
                    f"object {name!r}: {_bare_message(exc)}"
                    f"{self._forward_reference_hint(expr)}",
                    exc.line if exc.line else decl.line,
                    exc.col if exc.line else decl.col,
                ) from exc
            if isinstance(resolved, ContinuousSpace):
                self._spaces[name] = resolved
            else:
                self._objects[name] = resolved
            return
        raise CompileError(
            f"unrecognized type initializer for {name!r}: {type(init).__name__}",
            decl.line,
            decl.col,
        )

    def _forward_reference_hint(self, expr: ObjectExpr) -> str:
        """Name any object the expression uses ahead of its declaration.

        Declarations are compiled in source order, so a name is in
        scope only from the line after it is introduced. Without this
        hint a forward reference reads as a missing declaration, which
        sends the author looking for a name that is already
        there, further down the file.
        """
        pending: list[str] = []
        for ref in _type_names(expr):
            if ref.name.isdigit():
                continue
            if (
                ref.name in self._objects
                or ref.name in self._spaces
                or ref.name in self._alias_names
            ):
                continue
            for stmt in self._module.statements:
                if isinstance(stmt, ObjectDecl) and ref.name in stmt.names:
                    pending.append(
                        f"{ref.name!r} is declared later, on line {stmt.line}"
                    )
                    break
        if not pending:
            return ""
        joined = "; ".join(pending)
        return f" ({joined}; declarations come into scope in source order)"

    # ------------------------------------------------------------------
    # morphism
    # ------------------------------------------------------------------

    def _compile_morphism(self, decl: MorphismDecl) -> None:
        """Compile a ``morphism NAME, ... : DOM -> COD [options] [~ init]``.

        Each name in ``decl.names`` compiles as an independent
        declaration (fresh parameters per name) sharing the same
        signature, options, and initializer. The ``role`` option
        picks the runtime construction:

        * ``role=latent``  : learnable algebraic morphism on finite
          sets, optionally re-initialised by the algebra's auto
          recipe when ``init=auto`` is present.
        * ``role=observed``: fixed structural morphism whose tensor
          comes from the ``~ <expr>`` initializer; the expression's
          domain/codomain may be anonymous and rebinds to the
          declared types when their numel matches.
        * ``role=kernel``  : Markov kernel. Without ``~ Family``, a
          lookup-table `StochasticMorphism` on finite sets;
          with ``~ Family(...)``, a parametric continuous kernel.
        * ``role=embed``   : `Embed` boundary, finite-set to
          continuous space.
        * ``role=discretize``: `Discretize` boundary,
          continuous space to finite set; ``[bins=N]`` is required.
        * ``role=let``     : deterministic morphism whose value is
          ``~ <expr>`` (composition pipeline, contraction call,
          transformation invocation, etc.).
        * ``role=function``: learned deterministic map between
          continuous spaces, a `LearnedFunction` whose network the
          ``param_source`` option selects.

        When ``role`` is absent the morphism is a kernel: programs
        draw from kernels, so the parametric-Markov-kernel lowering
        is the only sound default. Every other role (latent /
        observed / embed / discretize / let) is always explicit.
        """
        display = ", ".join(decl.names)
        check_option_keys(
            decl.options,
            _MORPHISM_OPTION_KEYS,
            owner=f"morphism {display!r}",
            line=decl.line,
            col=decl.col,
        )
        for name in decl.names:
            self._compile_morphism_named(decl, name)

    def _compile_morphism_named(self, decl: MorphismDecl, name: str) -> None:
        """Compile one name of a (possibly plural) morphism declaration."""
        if name in self._morphisms:
            raise CompileError(
                f"morphism {name!r} already declared",
                decl.line,
                decl.col,
            )
        role = self._resolve_morphism_role(decl, name)
        self._check_role_options(decl, name, role)
        replicate = get_option_int(
            decl.options,
            "replicate",
            line=decl.line,
            col=decl.col,
        )
        names = (
            [f"{name}_{i}" for i in range(int(replicate))]
            if replicate is not None
            else [name]
        )
        if role == "latent":
            self._compile_latent_role(decl, name, names)
        elif role == "observed":
            self._compile_observed_role(decl, name, names)
        elif role == "kernel":
            self._compile_kernel_role(decl, name, names)
        elif role == "embed":
            self._compile_embed_role(decl, name, names)
        elif role == "discretize":
            self._compile_discretize_role(decl, name, names)
        elif role == "function":
            self._compile_function_role(decl, name, names)
        else:
            self._compile_let_role(decl, name, names)
        if replicate is not None:
            self._groups[name] = names

    def _check_role_options(self, decl: MorphismDecl, name: str, role: str) -> None:
        """Reject an option the resolved role's lowering never reads.

        `check_option_keys` has already rejected keys outside the
        union, so anything reaching here is a real morphism option
        applied to a role that ignores it. Ignoring it silently is how
        ``[scale=0.5] ~ Normal`` came to read as a configured init
        while doing nothing at all.
        """
        allowed = _COMMON_MORPHISM_OPTION_KEYS | _ROLE_OPTION_KEYS[role]
        for entry in decl.options:
            if entry.key in allowed:
                continue
            owners = sorted(
                r for r, keys in _ROLE_OPTION_KEYS.items() if entry.key in keys
            )
            where = (
                f"it configures {', '.join(f'role={r}' for r in owners)}"
                if owners
                else "no role reads it"
            )
            raise CompileError(
                f"morphism {name!r}: option {entry.key!r} is not read by "
                f"role={role}; {where}. Remove it, or declare the "
                f"morphism under a role that reads it.",
                entry.line or decl.line,
                entry.col or decl.col,
            )

    def _resolve_morphism_role(self, decl: MorphismDecl, name: str) -> str:
        """Resolve a morphism's role: explicit ``role=`` wins; absent
        role defaults to ``kernel``.

        A kernel is what programs draw from (a parametric Markov
        kernel with a family prior), so it is the only sound default;
        ``latent`` (learnable point estimate), ``observed`` (fixed
        structural input), and the boundary / binding roles are
        always explicit.
        """
        role = get_option_name(
            decl.options,
            "role",
            line=decl.line,
            col=decl.col,
        )
        if role is None:
            return "kernel"
        if role not in _VALID_ROLES:
            entry = find_option(decl.options, "role")
            ln = entry.line if entry is not None else decl.line
            cl = entry.col if entry is not None else decl.col
            raise CompileError(
                f"morphism {name!r}: unknown role {role!r}; "
                f"expected one of {sorted(_VALID_ROLES)}",
                ln,
                cl,
            )
        return role

    # role-specific lowerings ------------------------------------------

    def _compile_latent_role(
        self,
        decl: MorphismDecl,
        name: str,
        names: list[str],
    ) -> None:
        if decl.init_expr is not None:
            # `Expr` is a tagged-union root, so its ``line`` / ``col``
            # type as the generic field-value union; narrow to int
            # before using them as a source location.
            init_line = decl.init_expr.line
            init_col = decl.init_expr.col
            if isinstance(init_line, int) and isinstance(init_col, int) and init_line:
                ln, cl = init_line, init_col
            else:
                ln, cl = decl.line, decl.col
            raise CompileError(
                f"latent morphism {name!r}: ``~ <expression>`` "
                f"init is reserved for ``role=let`` and ``role="
                f"observed``; latent priors take a ``~ Family(...)`` "
                f"form instead",
                ln,
                cl,
            )
        if decl.init_family is not None:
            self._validate_family_axes(decl, decl.init_family.family)
        domain = self._resolve_type(decl.domain)
        codomain = self._resolve_type(decl.codomain)
        scale = get_option_float(
            decl.options,
            "scale",
            line=decl.line,
            col=decl.col,
            default=0.5,
        )
        init_mode = get_option_name(
            decl.options,
            "init",
            line=decl.line,
            col=decl.col,
        )
        for member in names:
            morph = make_latent(
                domain,
                codomain,
                init_scale=float(scale),
                algebra=self._algebra,
            )
            if init_mode == "auto":
                _apply_auto_init(morph, domain, codomain, self._algebra)
            self._morphisms[member] = morph

    def _compile_observed_role(
        self,
        decl: MorphismDecl,
        name: str,
        names: list[str],
    ) -> None:
        if decl.init_expr is None and decl.init_family is None:
            raise CompileError(
                f"observed morphism {name!r} requires an "
                f"initializer (e.g. ``~ identity({decl.domain})``)",
                decl.line,
                decl.col,
            )
        if decl.init_family is not None:
            ln = decl.init_family.line or decl.line
            cl = decl.init_family.col or decl.col
            raise CompileError(
                f"observed morphism {name!r}: ``~ Family(...)`` "
                f"is a stochastic-kernel prior; use ``role=kernel`` "
                f"or supply a deterministic ``~ <expression>`` "
                f"initializer instead",
                ln,
                cl,
            )
        domain = self._resolve_type(decl.domain)
        codomain = self._resolve_type(decl.codomain)
        for member in names:
            morph = self._compile_expr(decl.init_expr)
            morph = self._coerce_observed_shape(morph, domain, codomain, decl, name)
            self._morphisms[member] = morph

    def _coerce_observed_shape(
        self,
        morph,
        domain,
        codomain,
        decl: MorphismDecl,
        name: str,
    ):
        """Rebind ``morph``'s declared domain/codomain when its
        underlying tensor's numel matches the declared types.

        Categorically, ``B = B1 * ... * Bk`` and a flat ``B'`` whose
        cardinality is ``|B1| * ... * |Bk|`` are isomorphic objects;
        the storage is identical up to reshape. The compiler accepts
        an init expression whose anonymous shape matches that
        invariant and rebinds the morphism to the declared factored
        shape.
        """
        if morph.domain == domain and morph.codomain == codomain:
            return morph

        def _numel(shape) -> int:
            n = 1
            for s in shape:
                n *= int(s)
            return n

        init_d = _numel(morph.domain.shape)
        init_c = _numel(morph.codomain.shape)
        decl_d = _numel(domain.shape)
        decl_c = _numel(codomain.shape)
        if init_d == decl_d and init_c == decl_c:
            target_shape = tuple(domain.shape) + tuple(codomain.shape)
            reshaped = morph.tensor.reshape(target_shape)
            return ObservedMorphism(
                domain,
                codomain,
                reshaped,
                algebra=morph.algebra,
            )
        raise CompileError(
            f"morphism {name!r} init expression has type "
            f"{morph.domain!r} -> {morph.codomain!r} (numel {init_d} "
            f"-> {init_c}), expected {domain!r} -> {codomain!r} "
            f"(numel {decl_d} -> {decl_c})",
            decl.line,
            decl.col,
        )

    def _compile_kernel_role(
        self,
        decl: MorphismDecl,
        name: str,
        names: list[str],
    ) -> None:
        family = decl.init_family.family if decl.init_family is not None else None
        # ``~ Normal`` (bare identifier, no parens) parses as
        # ``init_expr`` rather than ``init_family``. If the bare
        # initializer names a registered family, promote it.
        if (
            family is None
            and decl.init_expr is not None
            and isinstance(decl.init_expr, ExprIdent)
            and decl.init_expr.name in _get_family_registry()
        ):
            family = decl.init_expr.name
        if family is None and decl.init_expr is not None:
            ln = decl.init_expr.line or decl.line
            cl = decl.init_expr.col or decl.col
            raise CompileError(
                f"kernel morphism {name!r}: a deterministic "
                f"``~ <expression>`` initializer is not a kernel prior; "
                f"use ``role=observed`` for a fixed morphism or "
                f"``role=let`` for a deterministic one, or supply a "
                f"distribution family (``~ Family(...)``)",
                ln,
                cl,
            )
        if family is not None:
            self._validate_family_axes(decl, family)
        if family is None:
            domain = self._resolve_type(decl.domain)
            codomain = self._resolve_type(decl.codomain)
            for member in names:
                self._morphisms[member] = StochasticMorphism(domain, codomain)
            return
        domain = self._resolve_any_space(decl.domain)
        codomain = self._resolve_any_space(decl.codomain)
        explicit_args = decl.init_family.args if decl.init_family is not None else ()
        refusal = kernel_argument_refusal(name, family, explicit_args)
        if refusal is not None:
            offending, message = refusal
            ln, cl = _draw_arg_location(offending, decl)
            raise CompileError(message, ln, cl)
        if explicit_args:
            network_options = {
                entry.key
                for entry in decl.options
                if entry.key
                in {"hidden_dim", "n_layers", "param_source", "rank", "temperature"}
            }
            if network_options:
                key = sorted(network_options)[0]
                entry = find_option(decl.options, key)
                raise CompileError(
                    f"kernel morphism {name!r}: option {key!r} configures a "
                    f"learned parameter source, but explicit literal family "
                    f"arguments declare a fixed {family}",
                    entry.line if entry is not None and entry.line else decl.line,
                    entry.col if entry is not None and entry.col else decl.col,
                )
            try:
                fixed, inputs = make_inline_distribution(
                    family,
                    explicit_args,
                    codomain,
                )
            except (TypeError, ValueError) as exc:
                raise CompileError(
                    f"kernel morphism {name!r}: cannot construct fixed "
                    f"{family} family from its literal arguments: {exc}",
                    *_draw_arg_location(explicit_args[0], decl),
                ) from exc
            if inputs is not None:
                raise CompileError(
                    f"kernel morphism {name!r}: fixed {family} unexpectedly "
                    f"retained parameter inputs",
                    decl.line,
                    decl.col,
                )
            fixed = fixed.with_domain(domain)
            for member in names:
                self._morphisms[member] = fixed
            return
        for member in names:
            morph = self._make_continuous_morphism(
                domain,
                codomain,
                family,
                decl,
            )
            self._morphisms[member] = morph

    def _compile_embed_role(
        self,
        decl: MorphismDecl,
        name: str,
        names: list[str],
    ) -> None:
        domain = self._resolve_type(decl.domain)
        if not isinstance(domain, FinSet):
            raise CompileError(
                f"embed morphism {name!r}: domain must be a "
                f"FinSet, got {type(domain).__name__}",
                decl.line,
                decl.col,
            )
        codomain = self._resolve_any_space(decl.codomain)
        if not isinstance(codomain, ContinuousSpace):
            raise CompileError(
                f"embed morphism {name!r}: codomain must be a "
                f"ContinuousSpace, got {type(codomain).__name__}",
                decl.line,
                decl.col,
            )
        for member in names:
            self._morphisms[member] = Embed(domain, codomain)

    def _compile_discretize_role(
        self,
        decl: MorphismDecl,
        name: str,
        names: list[str],
    ) -> None:
        space = self._resolve_any_space(decl.domain)
        if not isinstance(space, ContinuousSpace):
            raise CompileError(
                f"discretize morphism {name!r}: domain must be a "
                f"ContinuousSpace, got {type(space).__name__}",
                decl.line,
                decl.col,
            )
        bins = get_option_int(
            decl.options,
            "bins",
            line=decl.line,
            col=decl.col,
        )
        if bins is None:
            raise CompileError(
                f"discretize morphism {name!r}: required option ``bins`` is missing",
                decl.line,
                decl.col,
            )
        for member in names:
            self._morphisms[member] = Discretize(space, n_bins=bins)

    def _compile_let_role(
        self,
        decl: MorphismDecl,
        name: str,
        names: list[str],
    ) -> None:
        if decl.init_expr is None:
            raise CompileError(
                f"let morphism {name!r}: ``role=let`` requires an "
                f"``~ <expression>`` initializer",
                decl.line,
                decl.col,
            )
        for member in names:
            self._morphisms[member] = self._compile_expr(decl.init_expr)

    def _compile_function_role(
        self,
        decl: MorphismDecl,
        name: str,
        names: list[str],
    ) -> None:
        """Compile ``[role=function]``: a learned deterministic map.

        Parameters
        ----------
        decl : MorphismDecl
            The declaration.
        name : str
            The declared name.
        names : list[str]
            The members to bind, one per replica.

        Raises
        ------
        CompileError
            If the declaration has an initializer, a side is not a
            continuous space, or the options name no learned-function
            network.
        """
        if decl.init_family is not None or decl.init_expr is not None:
            init = decl.init_family if decl.init_family is not None else decl.init_expr
            line = getattr(init, "line", 0)
            col = getattr(init, "col", 0)
            raise CompileError(
                f"function morphism {name!r}: a learned function takes no "
                f"``~`` initializer; its network comes from ``param_source`` "
                f"and its output from ``output``",
                line if isinstance(line, int) and line else decl.line,
                col if isinstance(col, int) and line else decl.col,
            )
        domain = self._resolve_any_space(decl.domain)
        codomain = self._resolve_any_space(decl.codomain)
        for side, space in (("domain", domain), ("codomain", codomain)):
            factors = (
                space.components if isinstance(space, ProductSpace) else (space,)
            )
            if not all(isinstance(factor, ContinuousSpace) for factor in factors):
                raise CompileError(
                    f"function morphism {name!r}: its {side} {space!s} is not a "
                    f"continuous space or a product of them; a learned function "
                    f"maps real coordinates (a finite domain is read through "
                    f"``role=embed``)",
                    decl.line,
                    decl.col,
                )
        if isinstance(codomain, ProductSpace):
            raise CompileError(
                f"function morphism {name!r}: its codomain {codomain!s} is a "
                f"product; a learned function writes one space",
                decl.line,
                decl.col,
            )
        source_text = get_option_call_text(
            decl.options, "param_source", line=decl.line, col=decl.col
        )
        if (
            source_text is not None
            and source_text.strip() == "identity"
            and isinstance(domain, ProductSpace)
        ):
            entry = find_option(decl.options, "param_source")
            raise CompileError(
                f"function morphism {name!r}: an identity function passes one "
                f"input through, but its domain {domain!s} is a product; select "
                f"a learned source (linear or mlp) to read several inputs",
                entry.line if entry is not None and entry.line else decl.line,
                entry.col if entry is not None and entry.line else decl.col,
            )
        activation = get_option_name(
            decl.options, "activation", line=decl.line, col=decl.col
        )
        output = get_option_name(
            decl.options, "output", line=decl.line, col=decl.col, default="identity"
        )
        if output not in FUNCTION_OUTPUTS:
            entry = find_option(decl.options, "output")
            raise CompileError(
                f"function morphism {name!r}: unknown output {output!r}; "
                f"expected one of {sorted(FUNCTION_OUTPUTS)}",
                entry.line if entry is not None and entry.line else decl.line,
                entry.col if entry is not None and entry.line else decl.col,
            )
        width = _shape_size(codomain)
        for member in names:
            try:
                source = function_source(
                    domain,
                    width,
                    source_text,
                    activation=cast(FunctionActivation | None, activation),
                )
            except ValueError as error:
                key = "activation" if activation is not None else "param_source"
                entry = find_option(decl.options, key)
                raise CompileError(
                    f"function morphism {name!r}: {error}",
                    entry.line if entry is not None and entry.line else decl.line,
                    entry.col if entry is not None and entry.line else decl.col,
                ) from error
            self._morphisms[member] = LearnedFunction(
                domain, codomain, source, output=cast(FunctionOutput, output)
            )

    # family-construction plumbing ------------------------------------

    def _validate_family_axes(
        self,
        decl: MorphismDecl,
        family: str,
    ) -> None:
        """Check the option block's ``over``/``iid`` axis lists against
        the family's declared event/batch decomposition."""
        over = get_option_name_list(
            decl.options,
            "over",
            line=decl.line,
            col=decl.col,
        )
        iid = get_option_name_list(
            decl.options,
            "iid",
            line=decl.line,
            col=decl.col,
        )
        if not over and not iid:
            return
        # Point axis-role diagnostics at the option entry that
        # carries the clause rather than the declaration header.
        entry = find_option(decl.options, "over") or find_option(decl.options, "iid")
        ln = entry.line if entry is not None and entry.line else decl.line
        cl = entry.col if entry is not None and entry.line else decl.col
        axes_spec = AxisSpec(
            over=over,
            iid_over=iid,
            line=ln,
            col=cl,
        )
        _validate_axis_spec(
            axes_spec,
            family,
            _available_axes_for(decl.domain, decl.codomain),
            ln,
            cl,
        )

    def _make_continuous_morphism(
        self,
        domain,
        codomain,
        family_name: str,
        decl: MorphismDecl,
    ):
        if family_name == "Flow":
            n_layers = get_option_int(
                decl.options,
                "n_layers",
                line=decl.line,
                col=decl.col,
                default=4,
            )
            hidden_dim = get_option_int(
                decl.options,
                "hidden_dim",
                line=decl.line,
                col=decl.col,
                default=64,
            )
            return ConditionalFlow(
                domain,
                codomain,
                n_layers=int(n_layers),
                hidden_dim=int(hidden_dim),
            )
        registry = _get_family_registry()
        if family_name not in registry:
            raise CompileError(
                f"unknown distribution family {family_name!r}; "
                f"available: {', '.join(sorted(registry))}",
                decl.line,
                decl.col,
            )
        cls = registry[family_name]
        hidden_dim = get_option_int_list(
            decl.options,
            "hidden_dim",
            line=decl.line,
            col=decl.col,
        )
        kwargs: dict = {}
        if hidden_dim:
            kwargs["hidden_dim"] = hidden_dim[0] if len(hidden_dim) == 1 else hidden_dim
        # Optional `[param_source=<kind>]` / `[param_source=<kind>(...)]`
        # DSL surface for picking the parameter-source architecture
        # (linear, MLP, attention, identity). The default is linear.
        # Both keys are threaded to the family's `__init__`, which hands
        # them to `_make_source`: the option's text carries any
        # parenthesised widths, so the call form has to reach it as
        # surface text rather than as a bare name.
        param_source_opt = get_option_call_text(
            decl.options,
            "param_source",
            line=decl.line,
            col=decl.col,
        )
        if param_source_opt is not None:
            kwargs["param_source_option"] = param_source_opt
        # A family whose parameters do not come from a source has
        # nothing to point these at. Saying so beats a TypeError out of
        # a constructor with no line or column on it.
        unread = sorted(k for k in kwargs if k not in _signature_params(cls))
        if unread:
            raise CompileError(
                f"morphism {decl.names[0]!r}: {family_name} does not take "
                f"{', '.join(repr(k) for k in unread)}; its parameters do "
                f"not come from a ``param_source``",
                decl.line,
                decl.col,
            )
        rank = get_option_int(
            decl.options,
            "rank",
            line=decl.line,
            col=decl.col,
        )
        if rank is not None:
            kwargs["rank"] = int(rank)
        temperature = get_option_float(
            decl.options,
            "temperature",
            line=decl.line,
            col=decl.col,
        )
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        over = get_option_name_list(
            decl.options,
            "over",
            line=decl.line,
            col=decl.col,
        )
        if family_name == "MatrixNormal" and over:
            if len(over) != 2:
                entry = find_option(decl.options, "over")
                ln = entry.line if entry is not None and entry.line else decl.line
                cl = entry.col if entry is not None and entry.line else decl.col
                raise CompileError(
                    f"MatrixNormal requires ``over=[rows_axis, "
                    f"cols_axis]``; got over={list(over)!r}",
                    ln,
                    cl,
                )
            rows_axis, cols_axis = over
            kwargs["rows"] = self._axis_dim(decl, rows_axis)
            kwargs["cols"] = self._axis_dim(decl, cols_axis)
        # Bespoke ``Conditional<F>`` classes (Horseshoe, GP, ...) have
        # parameter sets that don't include ``hidden_dim`` / ``rank`` /
        # ``temperature``. The kwargs are accumulated for every family
        # above; here we drop any that the target class doesn't accept,
        # so the call site stays uniform across the registry.
        kwargs = _filter_kwargs_for(cls, kwargs)
        return cls(domain, codomain, **kwargs)

    def _axis_dim(self, decl: MorphismDecl, axis_name: str) -> int:
        if axis_name == "dom":
            return _shape_size(self._resolve_any_space(decl.domain))
        if axis_name == "cod":
            return _shape_size(self._resolve_any_space(decl.codomain))
        if axis_name in self._objects:
            return int(self._objects[axis_name].cardinality)
        if axis_name in self._spaces:
            return _shape_size(self._spaces[axis_name])
        raise CompileError(
            f"axis {axis_name!r}: not a declared object/space and "
            f"not a ``dom``/``cod`` shortcut",
            decl.line,
            decl.col,
        )


def _filter_kwargs_for(cls: type, kwargs: dict) -> dict:
    """Drop kwargs the target class's ``__init__`` does not accept.

    The ``Conditional<F>`` classes in
    [`quivers.continuous.families`][quivers.continuous.families] are a
    mix of the standard parametric template (which accepts
    ``hidden_dim``) and hand-written bespoke priors (`Horseshoe`, `GP`,
    `RelaxedOneHotCategorical`) whose ``__init__`` signatures diverge.
    The morphism-kernel compile path accumulates every option-derived
    kwarg uniformly and then filters here, so callers don't have to
    know which families honour which options.

    A class whose ``__init__`` accepts ``**kwargs`` is treated as
    accepting every supplied kwarg (no filtering).
    """
    sig = inspect.signature(cls.__init__)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return kwargs
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in accepted}


__all__ = []
