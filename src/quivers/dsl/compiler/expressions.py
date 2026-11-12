"""Compiler mixin: expression compilation.

Handles expression compilation, transformation expressions,
composition, parsers, chart folds, exports, and data binding.
"""

from __future__ import annotations
import torch
from quivers.core.morphisms import identity as make_identity
from quivers.core.morphisms import cap as _make_cap, cup as _make_cup
from quivers.core.algebras import (
    Algebra,
    BilinearForm,
    CompositionRule,
    Semigroupoid,
)
from quivers.core.objects import SetObject
from quivers.continuous.scan import ScanMorphism
from quivers.continuous.spaces import ContinuousSpace
from quivers.core.trans import TransSeq
from quivers.dsl.ast_nodes import (
    ExportDecl,
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
)
from quivers.dsl.compiler._prelude import (
    CompileError,
    _ChartHandlerComposite,
    _ALGEBRA_REGISTRY,
)


class _ExpressionsMixin:
    """Mixin: expression compilation methods.

    The compiler base supplies every environment slot below; the
    annotations let the type checker verify each access from a
    mixin method.
    """

    _algebra: CompositionRule
    _morphisms: dict
    _objects: dict[str, SetObject]
    _spaces: dict[str, ContinuousSpace]
    _transformations: dict
    _trans_singletons: dict
    _trans_constructors: dict
    _output_expr: Expr | None
    _exports: list[Expr]
    _contractions: dict
    _rules: dict
    _bundles: dict

    def _compile_export(self, decl: ExportDecl) -> None:
        """Record an exported expression.

        A module may declare any number of ``export`` statements,
        each selecting a top-level binding for the module's public
        surface. The compiled output runner picks the first export;
        further exports become additional accessible morphisms on
        the compiled object.
        """
        if not hasattr(self, "_exports"):
            self._exports = []
        self._exports.append(decl.expr)
        # The first export wins for the single-output entry point
        # that downstream helpers (``compile_env``,
        # ``Program(root_morphism)``) consult.
        if self._output_expr is None:
            self._output_expr = decl.expr

    def bind_data(self, data: dict) -> None:
        """Bind a runtime data dictionary for ``from_data("KEY")``
        initializers.

        Each key in ``data`` maps to a tensor (or tensor-like
        object) that supplies the morphism tensor for any
        ``from_data("KEY")`` expression that references it. The
        bindings are consulted at compile time; supply them BEFORE
        calling `compile`.
        """
        self._data_bindings = dict(data)

    def _require_algebra(self, op_name: str, line: int, col: int) -> None:
        """Raise ``CompileError`` if the module's composition rule
        isn't an `Algebra`.

        Operations that need a unit, zero, dagger, or compact-closed
        structure (``identity``, ``cup``, ``cap``, ``.dagger``,
        ``.trace``) call this before constructing their result.
        The error names the operation and the active rule so the
        user can decide whether to declare an algebra-level
        composition (``composition NAME [level=algebra]``) or drop the
        operation.
        """
        rule = self._algebra
        if isinstance(rule, Algebra):
            return
        kind = (
            "semigroupoid"
            if isinstance(rule, Semigroupoid)
            else (
                "bilinear_form"
                if isinstance(rule, BilinearForm)
                else "composition_rule"
            )
        )
        raise CompileError(
            f"{op_name}: requires an algebra-level composition rule (one "
            f"with an identity, a unit, and a zero); the module declares "
            f"the {kind} {rule.name!r}, which has none. Declare an algebra, "
            f"as in `composition product_fuzzy [level=algebra]`, or drop "
            f"the {op_name} operation.",
            line,
            col,
        )

    def _compile_trans_expr(self, expr):
        """Compile an expression that evaluates to a transformation.

        Recognised expression shapes:

        * `ExprIdent` — a bare name resolved against
          `_transformations` (user define-bindings) and then
          `_trans_singletons` (built-in singletons).
        * `ExprMorphismCall` — a constructor call whose
          callee names an entry of `_trans_constructors`;
          each argument is resolved against the surrounding scope
          (objects, morphisms, algebras).
        * `ExprTransCompose` — ``t1 >>> t2``; recursively
          compiles each side, flattens nested sequences, and
          type-checks each adjacent ``target == source`` boundary,
          returning a `TransSeq` (or, for two single-step
          values, an unboxed TransSeq of length 2).

        Returns a value that the runtime understands: a
        `AlgebraHomomorphism`, a
        `MorphismTransformation`, or a `TransSeq`.
        Raises `CompileError` on any unresolved reference
        or type mismatch.
        """
        if isinstance(expr, ExprIdent):
            if expr.name in self._transformations:
                return self._transformations[expr.name]
            if expr.name in self._trans_singletons:
                return self._trans_singletons[expr.name]
            if expr.name in self._trans_constructors:
                raise CompileError(
                    f"change_base: {expr.name!r} is a transformation "
                    f"constructor and needs arguments; call it like "
                    f"`{expr.name}(axis_object)`",
                    expr.line,
                    expr.col,
                )
            raise CompileError(
                f"change_base: undefined transformation "
                f"{expr.name!r}; available singletons: "
                f"{sorted(self._trans_singletons)}; constructors: "
                f"{sorted(self._trans_constructors)}; define-bound: "
                f"{sorted(self._transformations)}",
                expr.line,
                expr.col,
            )
        if isinstance(expr, ExprMorphismCall):
            factory = self._trans_constructors.get(expr.callee)
            if factory is None:
                raise CompileError(
                    f"change_base: undefined transformation "
                    f"constructor {expr.callee!r}; available: "
                    f"{sorted(self._trans_constructors)}",
                    expr.line,
                    expr.col,
                )
            resolved: list = []
            for arg_name in expr.args:
                value = self._resolve_trans_constructor_argument(
                    arg_name,
                    expr.line,
                    expr.col,
                    constructor_name=expr.callee,
                )
                resolved.append(value)
            try:
                return factory(*resolved)
            except (TypeError, ValueError) as e:
                raise CompileError(
                    f"change_base: constructor {expr.callee!r} "
                    f"rejected arguments {expr.args!r}: {e}",
                    expr.line,
                    expr.col,
                ) from e
        if isinstance(expr, ExprTransCompose):
            left_val = self._compile_trans_expr(expr.left)
            right_val = self._compile_trans_expr(expr.right)
            left_steps = (
                left_val.steps if isinstance(left_val, TransSeq) else (left_val,)
            )
            right_steps = (
                right_val.steps if isinstance(right_val, TransSeq) else (right_val,)
            )
            steps = left_steps + right_steps
            # Type-check the seam between each adjacent step.
            for i in range(len(steps) - 1):
                tgt = steps[i].target
                src = steps[i + 1].source
                if type(tgt) is not type(src):
                    raise CompileError(
                        f">>>: target of step {i} ({tgt.name!r}) "
                        f"does not match source of step {i + 1} "
                        f"({src.name!r})",
                        expr.line,
                        expr.col,
                    )
            return TransSeq(steps)
        raise CompileError(
            f"change_base: expression of kind "
            f"{type(expr).__name__!r} does not denote a "
            f"transformation",
            expr.line,
            expr.col,
        )

    def _resolve_trans_constructor_argument(
        self,
        name: str,
        line: int,
        col: int,
        *,
        constructor_name: str,
    ):
        """Resolve a named constructor argument to its compiled
        value.  Tries the value sorts that show up in
        transformation constructors (objects, morphisms,
        continuous morphisms, algebras) and surfaces a clear
        error if the name doesn't resolve.
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._morphisms:
            return self._morphisms[name]
        if (
            hasattr(self, "_continuous_morphisms")
            and name in self._continuous_morphisms
        ):
            return self._continuous_morphisms[name]
        if name in _ALGEBRA_REGISTRY:
            return _ALGEBRA_REGISTRY[name]
        raise CompileError(
            f"change_base: constructor {constructor_name!r} "
            f"argument {name!r} unresolved (must name an object, "
            f"morphism, or algebra)",
            line,
            col,
        )

    def _apply_trans(self, inner_morph, phi):
        """Apply a transformation value (a singleton, constructor
        result, or `TransSeq`) to a morphism via the
        morphism's own ``change_base``.

        Sequences are unfolded into iterated change_base calls;
        each intermediate result feeds the next step.
        """
        if isinstance(phi, TransSeq):
            current = inner_morph
            for step in phi.steps:
                current = current.change_base(step)
            return current
        return inner_morph.change_base(phi)

    def _compose_with_op(self, left, right, op: str):
        """Compose two morphisms sequentially.

        The surface exposes one sequential-composition operator
        family: ``>>`` composes forward and ``<<`` composes the same
        pipeline written right-to-left, so ``f << g`` is ``g >> f``.
        Composition uses the operands' shared algebra via the
        `Morphism.__rshift__` path, which raises ``incompatible
        algebras`` when they differ; algebra-tagged composition is
        expressed via ``.change_base(...)`` or a ``composition``
        declaration.
        """
        if op == ">>":
            return left >> right
        if op == "<<":
            return right >> left
        raise CompileError(f"unknown composition operator {op!r}", 0, 0)

    def _compile_expr(self, expr: Expr):
        """Compile a value expression into a morphism.

        Parameters
        ----------
        expr : Expr
            The expression to compile.

        Returns
        -------
        Morphism or ContinuousMorphism
            The compiled morphism (possibly a DAG of compositions).
        """
        if isinstance(expr, ExprIdent):
            if expr.name not in self._morphisms:
                raise CompileError(
                    f"undefined morphism {expr.name!r}", expr.line, expr.col
                )
            return self._morphisms[expr.name]
        elif isinstance(expr, ExprMorphismCall):
            return self._compile_morphism_call(expr)
        elif isinstance(expr, ExprIdentity):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"undefined object {expr.object_name!r}", expr.line, expr.col
                )
            self._require_algebra("identity", expr.line, expr.col)
            obj = self._objects[expr.object_name]
            return make_identity(obj, algebra=self._algebra)
        elif isinstance(expr, ExprDagger):
            inner = self._compile_expr(expr.inner)
            self._require_algebra("dagger", expr.line, expr.col)
            return inner.dagger
        elif isinstance(expr, ExprTrace):
            inner = self._compile_expr(expr.inner)
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"trace: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_algebra("trace", expr.line, expr.col)
            try:
                return inner.trace(self._objects[expr.object_name])
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprChangeBase):
            inner = self._compile_expr(expr.inner)
            phi = self._compile_trans_expr(expr.phi)
            try:
                return self._apply_trans(inner, phi)
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprCup):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"cup: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_algebra("cup", expr.line, expr.col)
            return _make_cup(self._objects[expr.object_name], algebra=self._algebra)
        elif isinstance(expr, ExprCap):
            if expr.object_name not in self._objects:
                raise CompileError(
                    f"cap: undefined object {expr.object_name!r}",
                    expr.line,
                    expr.col,
                )
            self._require_algebra("cap", expr.line, expr.col)
            return _make_cap(self._objects[expr.object_name], algebra=self._algebra)
        elif isinstance(expr, ExprFreeze):
            inner = self._compile_expr(expr.inner)
            # Materialise the inner morphism's tensor with detach()
            # and wrap as an ObservedMorphism. Gradient flow stops
            # at this boundary; the constituent morphisms' parameters
            # are not part of the result's parameter set.
            from quivers.core.morphisms import (
                Morphism as _CatMorph,
                ObservedMorphism as _Obs,
            )

            if not isinstance(inner, _CatMorph):
                raise CompileError(
                    f"freeze: inner expression compiled to "
                    f"{type(inner).__name__}; expected a Morphism",
                    expr.line,
                    expr.col,
                )
            frozen = _Obs(
                inner.domain,
                inner.codomain,
                inner.tensor.detach().clone(),
                algebra=inner.algebra,
            )
            return frozen
        elif isinstance(expr, ExprFromData):
            # ``from_data("KEY")`` cannot resolve the tensor at
            # compile time because the runtime data dict is only
            # available at fit time. We must therefore emit a
            # late-binding ObservedMorphism whose tensor is filled
            # in via a fit-time hook on the program's data
            # dictionary. For the standalone ``let``-expression
            # case this is materialised as a deferred binding the
            # compiler can resolve once the data dict is known.
            #
            # The data dictionary is stored on the Compiler under
            # ``self._data_bindings``; users supply it via
            # `Compiler.bind_data` BEFORE calling compile()
            # (or via the high-level ``loads()`` ``data=`` kwarg).
            data_dict = getattr(self, "_data_bindings", None)
            if data_dict is None or expr.key not in data_dict:
                available = sorted(data_dict.keys()) if data_dict else []
                raise CompileError(
                    f"from_data: unknown data key {expr.key!r}; "
                    f"available: {available}. Bind the data dict via "
                    f"``Compiler.bind_data(...)`` or the ``data=`` "
                    f"keyword on ``loads()`` before compiling.",
                    expr.line,
                    expr.col,
                )
            from quivers.core.morphisms import ObservedMorphism as _Obs

            tensor = data_dict[expr.key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            # The user-supplied domain/codomain on the morphism
            # declaration is in the parent ``observed`` decl; here
            # we synthesize a one-shot ObservedMorphism whose
            # domain/codomain are inferred from the tensor's shape
            # split halfway. For the common ``observed f : A -> B =
            # from_data("KEY")`` pattern the parent decl supplies
            # the correct domain/codomain post-substitution; this
            # let-expression path is for ``let h = from_data(...)``
            # which the surrounding context constrains.
            from quivers.core.objects import FinSet as _FS

            if tensor.dim() < 1:
                raise CompileError(
                    f"from_data: tensor at key {expr.key!r} has "
                    f"rank 0; cannot infer a morphism",
                    expr.line,
                    expr.col,
                )
            # Synthetic anonymous domain/codomain for the
            # let-expression path. The morphism's downstream
            # usage typically discards this synthetic typing
            # because it is reassigned to a properly typed
            # ``observed`` slot, OR used as a tensor source.
            dom = _FS(name=f"_data_dom_{expr.key}", cardinality=tensor.shape[0])
            if tensor.dim() == 1:
                cod = _FS(name="_data_unit", cardinality=1)
                tensor = tensor.unsqueeze(-1)
            else:
                cod_size = int(torch.tensor(tensor.shape[1:]).prod().item())
                cod = _FS(name=f"_data_cod_{expr.key}", cardinality=cod_size)
                if tensor.dim() > 2:
                    tensor = tensor.reshape(tensor.shape[0], -1)
            return _Obs(dom, cod, tensor, algebra=self._algebra)
        elif isinstance(expr, ExprCompose):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            try:
                return self._compose_with_op(left, right, expr.op)
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprTensorProduct):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            return left @ right
        elif isinstance(expr, ExprMarginalize):
            inner = self._compile_expr(expr.inner)
            sets = []
            for name in expr.names:
                if name not in self._objects:
                    raise CompileError(
                        f"undefined object {name!r} in marginalize", expr.line, expr.col
                    )
                sets.append(self._objects[name])
            try:
                return inner.marginalize(*sets)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprCurry):
            from quivers.core.morphisms import CurriedMorphism

            inner = self._compile_expr(expr.inner)
            try:
                return CurriedMorphism(inner, direction=expr.direction)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprChartFold):
            return self._compile_chart_fold(expr)
        elif isinstance(expr, ExprFan):
            from quivers.continuous.morphisms import FanOutMorphism

            components = []
            for sub_expr in expr.exprs:
                if isinstance(sub_expr, ExprIdent) and sub_expr.name in self._groups:
                    for member_name in self._groups[sub_expr.name]:
                        components.append(self._morphisms[member_name])
                else:
                    morph = self._compile_expr(sub_expr)
                    components.append(morph)
            try:
                return FanOutMorphism(components)
            except (TypeError, ValueError) as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprRepeat):
            morph = self._compile_expr(expr.expr)
            if expr.count is None:
                from quivers.core.morphisms import RepeatMorphism

                try:
                    return RepeatMorphism(morph, n=1)
                except (TypeError, ValueError) as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            result = morph
            for _ in range(expr.count - 1):
                try:
                    result = result >> morph
                except TypeError as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            return result
        elif isinstance(expr, ExprStack):
            import copy

            morph = self._compile_expr(expr.expr)
            result = copy.deepcopy(morph)
            for _ in range(expr.count - 1):
                clone = copy.deepcopy(morph)
                try:
                    result = result >> clone
                except TypeError as e:
                    raise CompileError(str(e), expr.line, expr.col) from e
            return result
        elif isinstance(expr, ExprScan):
            cell = self._compile_expr(expr.expr)
            try:
                return ScanMorphism(
                    cell,
                    init=expr.init,
                    outputs=expr.outputs,
                    reverse=expr.reverse,
                )
            except TypeError as e:
                raise CompileError(str(e), expr.line, expr.col) from e
        elif isinstance(expr, ExprParser):
            from quivers.stochastic.schema import SCHEMA_REGISTRY

            schemas: list = []
            morphisms: list = []

            def _expand(name: str, seen: frozenset[str]) -> list[str]:
                """Recursively expand a bundle reference into rule names.

                Cycle detection: if ``name`` already appears in ``seen``,
                raises CompileError.
                """
                if name not in self._bundles:
                    return [name]
                if name in seen:
                    raise CompileError(
                        f"bundle cycle through {name!r}",
                        expr.line,
                        expr.col,
                    )
                expanded: list[str] = []
                for member in self._bundles[name]:
                    expanded.extend(_expand(member, seen | {name}))
                return expanded

            resolved_rules: list[str] = []
            for rule_name in expr.rules:
                resolved_rules.extend(_expand(rule_name, frozenset()))

            for rule_name in resolved_rules:
                if rule_name in self._rules:
                    schemas.append(self._rules[rule_name])
                elif (schema_obj := SCHEMA_REGISTRY.get(rule_name)) is not None:
                    schemas.append(schema_obj)
                elif rule_name in self._morphisms:
                    morphisms.append(self._morphisms[rule_name])
                else:
                    raise CompileError(
                        f"unknown rule {rule_name!r}; not a declared rule, schema primitive ({', '.join(sorted(SCHEMA_REGISTRY))}), or a declared morphism",
                        expr.line,
                        expr.col,
                    )
            if schemas and morphisms:
                raise CompileError(
                    "parser() rules must be all schema primitives or all morphism references, not a mix",
                    expr.line,
                    expr.col,
                )
            if morphisms:
                return self._compile_parser_morphisms(morphisms, expr)
            if not schemas:
                raise CompileError(
                    "parser() requires at least one rule", expr.line, expr.col
                )
            return self._compile_parser_schemas(schemas, expr)
        else:
            raise CompileError(f"unknown expression type: {type(expr).__name__}")

    def _compile_parser_morphisms(self, morphisms: list, expr: ExprParser):
        """Compile parser from user-declared morphisms via type inspection.

        Classifies each morphism by its type signature:

        - ``N → N ⊗ N`` (codomain is a product of the domain with
          itself) contributes binary deductions.
        - ``N → T`` (codomain differs from domain) contributes
          lexical axioms.

        Parameters
        ----------
        morphisms : list
            Compiled morphism objects.
        expr : ExprParser
            The AST node (for error reporting).
        """
        from quivers.core.objects import ProductSet
        from quivers.stochastic.inside import InsideAlgorithm

        binary = None
        lexical = None
        for morph in morphisms:
            cod = morph.codomain
            if (
                isinstance(cod, ProductSet)
                and len(cod.components) == 2
                and all((c == morph.domain for c in cod.components))
            ):
                if binary is not None:
                    raise CompileError(
                        "parser() received multiple binary morphisms (codomain = domain ⊗ domain); expected one",
                        expr.line,
                        expr.col,
                    )
                binary = morph
            else:
                if lexical is not None:
                    raise CompileError(
                        "parser() received multiple lexical morphisms; expected one",
                        expr.line,
                        expr.col,
                    )
                lexical = morph
        if binary is None:
            raise CompileError(
                "parser() requires a binary morphism (type N → N ⊗ N) among its rules",
                expr.line,
                expr.col,
            )
        if lexical is None:
            raise CompileError(
                "parser() requires a lexical morphism (type N → T) among its rules",
                expr.line,
                expr.col,
            )
        try:
            start = expr.start if isinstance(expr.start, int) else 0
            return InsideAlgorithm(binary, lexical, start=start)
        except TypeError as e:
            raise CompileError(str(e), expr.line, expr.col) from e

    def _compile_chart_fold(self, expr):
        """Compile a ``chart_fold(...)`` primitive expression.

        ``parser(rules=...)`` is surface syntax for this primitive. A lexical
        morphism, a binary morphism, and an optional unary morphism define the
        `InsideAlgorithm` chart parser.

        When ``effect_depth`` is positive, the category universe is extended
        with bounded effect stacks. The supplied binary and unary morphisms
        must act on that enlarged universe. Each declared handler is then
        post-composed with the parser's denotation in declaration order.
        """
        from quivers.stochastic.inside import InsideAlgorithm

        lex = self._compile_expr(expr.lex)
        if expr.binary is None:
            raise CompileError(
                "chart_fold(...) requires a binary= argument (a morphism "
                "Cat * Cat -> Cat representing the union of binary rule "
                "schemas)",
                expr.line,
                expr.col,
            )
        binary = self._compile_expr(expr.binary)

        unary = self._compile_expr(expr.unary) if expr.unary is not None else None

        handlers_morphisms: list = []
        for h_expr in getattr(expr, "handlers", ()) or ():
            handlers_morphisms.append(self._compile_expr(h_expr))

        try:
            start = expr.start if isinstance(expr.start, int) else 0
            parser = InsideAlgorithm(binary, lex, start=start, unary=unary)
        except (TypeError, ValueError) as e:
            raise CompileError(str(e), expr.line, expr.col) from e

        # Compose handlers as post-applications on the parser's output.
        # Each handler is a morphism Cat → Cat (or a more refined effect
        # reduction); composition is right-to-left in declaration order.
        result = parser
        for handler in handlers_morphisms:
            result = _ChartHandlerComposite(result, handler)
        return result

    def _compile_parser_schemas(self, schemas: list, expr: ExprParser):
        """Compile parser from schema functors over a category system.

        Parameters
        ----------
        schemas : list
            Schema objects from ``SCHEMA_REGISTRY``.
        expr : ExprParser
            The AST node.
        """
        from quivers.stochastic.categories import CategorySystem
        from quivers.stochastic.parsers import ChartParser

        from quivers.core.objects import FreeResiduated

        if expr.categories:
            categories = list(expr.categories)
        elif self._categories:
            categories = list(self._categories)
        else:
            # Look for a FreeResiduated object in scope and use its
            # generators' atom names. If exactly one residuated universe
            # is declared, this avoids the user having to spell out
            # `categories=[NP, S, VP, ...]` redundantly.
            residuated = [
                obj for obj in self._objects.values() if isinstance(obj, FreeResiduated)
            ]
            if len(residuated) == 1:
                categories = list(residuated[0].generators.elements)
            elif len(residuated) > 1:
                raise CompileError(
                    "parser() with schema rules: multiple FreeResiduated "
                    "objects in scope; pass categories=[...] explicitly to "
                    "select the atom set",
                    expr.line,
                    expr.col,
                )
            else:
                raise CompileError(
                    "parser() with schema rules requires category atoms — declare them via `object Atoms = {NP, S, VP, ...}` plus `object Cat = FreeResiduated(Atoms, ...)`, or pass categories=[NP, S, VP, ...] inline",
                    expr.line,
                    expr.col,
                )
        if expr.constructors is not None:
            cs = CategorySystem.from_generators(
                atoms=categories,
                constructors=list(expr.constructors),
                max_depth=expr.depth,
            )
        else:
            cs = CategorySystem.from_atoms_and_slash_depth(
                categories, max_depth=expr.depth
            )
        schema = schemas[0]
        for piece in schemas[1:]:
            schema = schema | piece
        if expr.terminal is None:
            raise CompileError(
                "parser() with schema rules requires terminal=<object> — the declared object serving as the terminal vocabulary",
                expr.line,
                expr.col,
            )
        if expr.terminal not in self._objects:
            raise CompileError(
                f"terminal={expr.terminal!r} does not refer to a declared object",
                expr.line,
                expr.col,
            )
        n_term = self._objects[expr.terminal].size
        try:
            start = expr.start if isinstance(expr.start, str) else "S"
            return ChartParser.from_schema(schema, cs, n_terminals=n_term, start=start)
        except (TypeError, ValueError) as e:
            raise CompileError(str(e), expr.line, expr.col) from e


__all__ = []
