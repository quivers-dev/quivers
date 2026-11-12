"""Normalize program ASTs before IR lowering.

The pass expands sampled composite bindings into atomic sample and let
steps. It preserves marginalize blocks for the lowering stage while
expanding composite references inside their scopes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    CallStep,
    DrawArg,
    DrawArgName,
    Expr,
    ExprCompose,
    ExprFan,
    ExprIdent,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    DefineDecl,
    LetStep,
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObjectExpr,
    ObjectProduct,
    ObserveStep,
    OptionName,
    OptionNumber,
    ProgramDecl,
    ProgramStep,
    QiecCallComputation,
    SampleStep,
    TypeName,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprList,
    LetExprNode,
    LetExprString,
    LetExprVar,
)
from quivers.dsl.markov_chains import markov_cell, returns_marginalized_latent
from quivers.dsl.sequence_functions import sequence_source_kind
from quivers.dsl.step_resolution import StepResolutionError, _step_value_names
from quivers.dsl.ast_nodes import atom_to_draw_arg


class _ChainElem(dx.TaggedUnion, discriminator="kind"):
    """One leaf in a flattened composite-let chain.

    Variants encode the surface shape needed to emit the right
    step kind for each leaf.
    """


class _StochasticLeaf(_ChainElem):
    """A morphism with `~ Family(args)` init; emits a `SampleStep`
    whose family slot is the morphism name."""

    name: str
    kind: Literal["stochastic"] = "stochastic"


class _DeterministicLeaf(_ChainElem):
    """A morphism with no init clause (an embedding lookup, a
    learnable linear layer, etc.) or an undeclared free name;
    emits a `LetStep` whose RHS is `name(prev)`. The morphism
    name flows through as a free identifier the host supplies."""

    name: str
    kind: Literal["deterministic"] = "deterministic"


class _ScanLeaf(_ChainElem):
    """A `scan(cell)` higher-order combinator; emits a `LetStep`
    whose RHS is `scan(step, input)` over a step program built from
    the cell, with the initial state's name as a third argument when
    the scan declares one and its `outputs` and `reverse` options as
    trailing flags (see `scan_let_call`)."""

    name: str
    init: str = "zeros"
    outputs: Literal["last", "all"] = "last"
    reverse: bool = False
    kind: Literal["scan"] = "scan"


class _ParallelLeaf(_ChainElem):
    """One branch-bundle of an `ExprFan` / `ExprTensorProduct`
    construction; emits each branch's elements with the same
    upstream input, then a final `LetStep` aggregates the branch
    tails into a list. ``branches`` carries the per-branch
    element sequences."""

    branches: tuple[tuple[_ChainElem, ...], ...] = ()
    product: bool = False
    kind: Literal["parallel"] = "parallel"


def expand_composite_lets(
    module: Module,
    *,
    target: str | None = None,
) -> Module:
    """Rewrite `program_decl` bodies so composite-let sample steps
    become equivalent chains of atomic sample steps.

    A composite let is a `DefineDecl` whose `.expr` is an `ExprCompose`
    (the `prior >> likelihood` form). Each `SampleStep` /
    `ObserveStep` whose `morphism` slot names such a let is rewritten
    into a sequence of fresh `SampleStep`s, one per element of the
    composition chain, with the trailing step keeping the original
    step's bound variable name.

    The returned `Module` shares vertex identity with the input for
    every non-rewritten statement; only the `program_decl`s with
    composite-let references are rebuilt.

    The rewrite is target-independent: every backend consumes the same
    expanded AST, and the choice of whether to enumerate or to sample a
    marginalize latent is each renderer's, taken against the lowered
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]. `target` is
    accepted so callers can name the backend they are compiling for
    without the pass branching on it.
    """
    morphism_table: dict[str, MorphismDecl] = {
        name: s
        for s in module.statements
        if isinstance(s, MorphismDecl)
        for name in s.names
    }
    let_table: dict[str, Expr] = {
        s.name: s.expr for s in module.statements if isinstance(s, DefineDecl)
    }
    program_table: dict[str, ProgramDecl] = {
        s.name: s for s in module.statements if isinstance(s, ProgramDecl)
    }

    del target
    declared = set(morphism_table)
    declared_programs = set(program_table)
    new_statements: list = []
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            program = _Program(
                morphism_table,
                let_table,
                program_table,
                _domain_names(stmt),
                _domain_factors(stmt.domain),
            )
            _refuse_cell_uses(stmt.draws, program_table)
            new_draws = _expand_draws(stmt.draws, program)
            refuse_marginal_chain_reads(tuple(new_draws), stmt.return_vars)
            # A scan's step program is declared before the program that
            # scans with it, so a consumer reading in order sees it.
            for name, declaration in program.programs.items():
                if name not in declared_programs:
                    declared_programs.add(name)
                    new_statements.append(declaration)
            if new_draws is stmt.draws:
                new_statements.append(stmt)
            else:
                new_statements.append(stmt.with_(draws=tuple(new_draws)))
        else:
            new_statements.append(stmt)
    # A replica of a `[replicate=k]` morphism and a copy a `stack`
    # makes are morphisms of their own, with their own parameters;
    # the expanded module declares them so every consumer sees them.
    synthesized = [
        decl for name, decl in morphism_table.items() if name not in declared
    ]
    if synthesized:
        last = max(
            index
            for index, stmt in enumerate(new_statements)
            if isinstance(stmt, MorphismDecl)
        )
        new_statements[last + 1 : last + 1] = synthesized
    return module.with_(statements=tuple(new_statements))


def _domain_names(program: ProgramDecl) -> tuple[str, ...]:
    """The names a program's domain factors are read through.

    Parameters
    ----------
    program : ProgramDecl
        The program.

    Returns
    -------
    tuple[str, ...]
        The declared parameter names when the program names them,
        else each factor's object name in lowercase.
    """
    if program.params:
        return tuple(program.params)
    return tuple(
        factor.name.lower() if isinstance(factor, TypeName) else ""
        for factor in _domain_factors(program.domain)
    )


def _domain_factors(domain: ObjectExpr) -> tuple[ObjectExpr, ...]:
    """A domain's product factors, left to right.

    Parameters
    ----------
    domain : ObjectExpr
        The domain expression.

    Returns
    -------
    tuple[ObjectExpr, ...]
        The factors of a product, or the expression itself.
    """
    if isinstance(domain, ObjectProduct):
        return tuple(
            factor
            for component in domain.components
            for factor in _domain_factors(component)
        )
    return (domain,)


def _replicas(name: str, morphisms: dict[str, MorphismDecl]) -> tuple[str, ...] | None:
    """The replica names of a ``[replicate=k]`` morphism.

    Each replica is declared in ``morphisms`` as a single-name copy of
    the declaration without the ``replicate`` option.

    Parameters
    ----------
    name : str
        The morphism.
    morphisms : dict[str, MorphismDecl]
        The morphism table, extended with the replicas.

    Returns
    -------
    tuple[str, ...] | None
        ``name_0`` through ``name_{k-1}``, or ``None`` when the
        morphism is not replicated.
    """
    decl = morphisms.get(name)
    if decl is None:
        return None
    count = next(
        (
            int(entry.value.value)
            for entry in decl.options
            if entry.key == "replicate" and isinstance(entry.value, OptionNumber)
        ),
        None,
    )
    if count is None:
        return None
    names = tuple(f"{name}_{index}" for index in range(count))
    for member in names:
        if member not in morphisms:
            morphisms[member] = decl.with_(
                names=(member,),
                options=tuple(
                    entry for entry in decl.options if entry.key != "replicate"
                ),
            )
    return names


def _copied(
    chain: tuple[_ChainElem, ...],
    suffix: str,
    morphisms: dict[str, MorphismDecl],
) -> tuple[_ChainElem, ...]:
    """A chain with every declared morphism replaced by a fresh copy.

    A ``stack`` composes copies of its morphism that share nothing: the
    copy's parameters are its own, so each copy is a morphism of its
    own, declared in ``morphisms`` under the suffixed name.

    Parameters
    ----------
    chain : tuple[_ChainElem, ...]
        The chain to copy.
    suffix : str
        The suffix the copies' names carry.
    morphisms : dict[str, MorphismDecl]
        The morphism table, extended with the copies.

    Returns
    -------
    tuple[_ChainElem, ...]
        The chain over the copies; names that are not declared
        morphisms are left as they are.
    """
    out: list[_ChainElem] = []
    for elem in chain:
        if isinstance(elem, _StochasticLeaf | _DeterministicLeaf):
            decl = morphisms.get(elem.name)
            if decl is None:
                out.append(elem)
                continue
            copy_name = f"{elem.name}{suffix}"
            if copy_name not in morphisms:
                morphisms[copy_name] = decl.with_(names=(copy_name,))
            out.append(elem.with_(name=copy_name))
        elif isinstance(elem, _ParallelLeaf):
            out.append(
                elem.with_(
                    branches=tuple(
                        _copied(branch, suffix, morphisms) for branch in elem.branches
                    )
                )
            )
        else:
            out.append(elem)
    return tuple(out)


def expand_program_scans(
    declaration: ProgramDecl, module: Module
) -> tuple[tuple[ProgramStep, ...], dict[str, ProgramDecl]]:
    """Expand the draws of one program that run a scan.

    A ``sample`` or ``observe`` step whose morphism resolves to a chain
    with a ``scan`` is rewritten exactly as
    [`expand_composite_lets`][quivers.dsl.composite_lets.expand_composite_lets]
    rewrites it: a ``let h = scan(step, xs)`` over a step program, with
    the chain's other elements as steps of the step program or of the
    program. Every other step is kept as written. The expansion draws
    its fresh names from the serial the module-wide expansion uses, so
    an expanded step binds the names the checked elaboration binds.

    Parameters
    ----------
    declaration : ProgramDecl
        The program.
    module : Module
        The module declaring it, its morphisms, defines, and programs.

    Returns
    -------
    tuple[tuple[ProgramStep, ...], dict[str, ProgramDecl]]
        The program's draws, the input itself when no step runs a scan;
        and the step programs the expansion declared, by
        name, each after the step programs it scans with.
    """
    morphisms: dict[str, MorphismDecl] = {
        name: statement
        for statement in module.statements
        if isinstance(statement, MorphismDecl)
        for name in statement.names
    }
    lets: dict[str, Expr] = {
        statement.name: statement.expr
        for statement in module.statements
        if isinstance(statement, DefineDecl)
    }
    programs: dict[str, ProgramDecl] = {
        statement.name: statement
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
    }
    declared = set(programs)
    _refuse_cell_uses(declaration.draws, programs)
    if not _selects(declaration.draws, morphisms, lets, _scans):
        return declaration.draws, {}
    program = _Program(
        morphisms,
        lets,
        programs,
        _domain_names(declaration),
        _domain_factors(declaration.domain),
    )
    draws = _expand_draws(
        declaration.draws,
        program,
        select=_scans,
    )
    refuse_marginal_chain_reads(draws, declaration.return_vars)
    steps = {
        name: decl for name, decl in program.programs.items() if name not in declared
    }
    return draws, steps


def _selects(
    draws: tuple[ProgramStep, ...],
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    select: Callable[[tuple[_ChainElem, ...]], bool],
) -> bool:
    """Whether some draw of a body resolves to a chain a predicate selects.

    Parameters
    ----------
    draws : tuple[ProgramStep, ...]
        The steps, searched through marginalization scopes.
    morphisms : dict[str, MorphismDecl]
        The morphism table.
    lets : dict[str, Expr]
        The define table.
    select : Callable[[tuple[_ChainElem, ...]], bool]
        The predicate.

    Returns
    -------
    bool
        ``True`` when a ``sample`` or ``observe`` step's chain is
        selected.
    """
    for step in draws:
        if isinstance(step, MarginalizeStep):
            if _selects(step.scope, morphisms, lets, select):
                return True
        elif isinstance(step, (SampleStep, ObserveStep)):
            chain = _resolve_to_chain(
                step.morphism, morphisms=dict(morphisms), lets=lets
            )
            if chain is not None and select(chain):
                return True
    return False


def _scans(chain: tuple[_ChainElem, ...]) -> bool:
    """Whether a chain runs a scan.

    Parameters
    ----------
    chain : tuple[_ChainElem, ...]
        The chain.

    Returns
    -------
    bool
        ``True`` when a scan element is in the chain or in a branch of
        a parallel element.
    """
    for elem in chain:
        if isinstance(elem, _ScanLeaf):
            return True
        if isinstance(elem, _ParallelLeaf) and any(
            _scans(branch) for branch in elem.branches
        ):
            return True
    return False


def _expand_draws(
    draws: tuple[ProgramStep, ...],
    program: _Program,
    select: Callable[[tuple[_ChainElem, ...]], bool] | None = None,
) -> tuple[ProgramStep, ...]:
    """Expand every SampleStep / ObserveStep whose morphism slot
    resolves to a composite-let chain.

    A `MarginalizeStep` passes through with its axis roles intact:
    its `[over=...]` names the grouping (batch) axes while its `: T`
    index names either the enumerated support or a replication axis,
    a distinction `SampleStep` has no slot for. Only the step's scope
    is rewritten, so composite-let references nested under a
    marginalize still expand.

    Parameters
    ----------
    draws
        The steps.
    program
        The enclosing program's expansion.
    select
        Which chains' expansions replace their steps; ``None`` selects
        every chain. A chain not selected is still expanded, so the
        names later expansions draw are the ones a full expansion
        draws, and its step is kept as written.

    Returns
    -------
    tuple[ProgramStep, ...]
        The steps with every selected composite draw expanded; the
        input itself when nothing expanded.
    """
    any_changed = False
    out: list[ProgramStep] = []
    for step in draws:
        if isinstance(step, MarginalizeStep):
            scope_expanded = _expand_draws(step.scope, program, select)
            if scope_expanded is step.scope:
                out.append(step)
            else:
                any_changed = True
                out.append(step.with_(scope=scope_expanded))
            continue
        if isinstance(step, (SampleStep, ObserveStep)):
            chain = _resolve_to_chain(
                step.morphism,
                morphisms=program.morphisms,
                lets=program.lets,
            )
            if chain is not None:
                expanded = _expand_step(step, chain, program)
                if select is None or select(chain):
                    any_changed = True
                    out.extend(expanded)
                    continue
        out.append(step)
    return tuple(out) if any_changed else draws


def _resolve_to_chain(
    name: str,
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    _seen: tuple[str, ...] = (),
) -> tuple[_ChainElem, ...] | None:
    """If `name` is a composite-let binding, return the ordered tuple
    of chain elements. Otherwise return None.

    Resolves through alias chains: `let a = b; let b = c >> d` returns
    the chain elements for `(c, d)`.
    """
    if name in _seen:
        return None
    if name not in lets:
        return None
    expr = lets[name]
    if isinstance(expr, ExprIdent):
        return _resolve_to_chain(
            expr.name,
            morphisms=morphisms,
            lets=lets,
            _seen=(*_seen, name),
        )
    if isinstance(
        expr,
        (ExprCompose, ExprTensorProduct, ExprFan, ExprScan, ExprStack, ExprRepeat),
    ):
        return _flatten_compose(
            expr,
            morphisms=morphisms,
            lets=lets,
            seen=(*_seen, name),
        )
    return None


def _flatten_compose(
    expr: Expr,
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    seen: tuple[str, ...] = (),
) -> tuple[_ChainElem, ...] | None:
    """Flatten a (possibly nested) composition expression into a
    tuple of `_ChainElem` records in left-to-right order.

    Handles every leaf shape the gallery uses:

    * `ExprIdent` -> the named morphism, classified as stochastic
      (morphism with `~ Family(args)` init) or deterministic
      (no init clause). Composite-let aliases resolve transitively
      into their chain.
    * `ExprStack(name, n)` / `ExprRepeat(name, n)` -> n copies of
      the inner leaf (sequential repetition; canonical Kleisli
      iteration).
    * `ExprFan(exprs)` / `ExprTensorProduct(left, right)` ->
      a single parallel `_ChainElem` whose `branches` field
      carries the per-branch chains. The expansion emits each
      branch's steps sequentially against the same upstream
      input, then a final list-bundling `LetStep` aggregates
      the branch tails.
    * `ExprScan(cell)` -> a scan leaf naming the cell morphism;
      emits as a deterministic let-call against the host-supplied
      `scan` combinator.

    Returns None when any leaf is a shape the expansion cannot
    classify (e.g. an embedded function call, marginalization
    expression). The caller falls back to leaving the original
    step intact, which the resolver then rejects with a precise
    `UnsupportedConstruct`.
    """
    out: list[_ChainElem] = []

    def walk(e: Expr) -> bool:
        if isinstance(e, ExprCompose):
            return walk(e.left) and walk(e.right)
        if isinstance(e, ExprIdent):
            return _resolve_ident_leaf(
                e.name,
                morphisms=morphisms,
                lets=lets,
                seen=seen,
                out=out,
            )
        if isinstance(e, ExprStack):
            inner = _expr_to_name(e.expr)
            if inner is None or e.count is None or e.count <= 0:
                return False
            first: list[_ChainElem] = []
            if not _resolve_ident_leaf(
                inner,
                morphisms=morphisms,
                lets=lets,
                seen=seen,
                out=first,
            ):
                return False
            out.extend(first)
            for index in range(1, e.count):
                out.extend(_copied(tuple(first), f"_copy{index}", morphisms))
            return True
        if isinstance(e, ExprRepeat):
            inner = _expr_to_name(e.expr)
            if inner is None or e.count is None or e.count <= 0:
                return False
            for _ in range(e.count):
                if not _resolve_ident_leaf(
                    inner,
                    morphisms=morphisms,
                    lets=lets,
                    seen=seen,
                    out=out,
                ):
                    return False
            return True
        if isinstance(e, ExprFan):
            branches: list[tuple[_ChainElem, ...]] = []
            for sub in e.exprs:
                replicas = (
                    _replicas(sub.name, morphisms)
                    if isinstance(sub, ExprIdent)
                    else None
                )
                if replicas is not None:
                    for member in replicas:
                        member_out: list[_ChainElem] = []
                        _resolve_ident_leaf(
                            member,
                            morphisms=morphisms,
                            lets=lets,
                            seen=seen,
                            out=member_out,
                        )
                        branches.append(tuple(member_out))
                    continue
                sub_out: list[_ChainElem] = []
                if not _walk_into(
                    sub,
                    sub_out,
                    morphisms=morphisms,
                    lets=lets,
                    seen=seen,
                ):
                    return False
                branches.append(tuple(sub_out))
            out.append(_ParallelLeaf(branches=tuple(branches)))
            return True
        if isinstance(e, ExprTensorProduct):
            left_out: list[_ChainElem] = []
            right_out: list[_ChainElem] = []
            if not _walk_into(
                e.left,
                left_out,
                morphisms=morphisms,
                lets=lets,
                seen=seen,
            ):
                return False
            if not _walk_into(
                e.right,
                right_out,
                morphisms=morphisms,
                lets=lets,
                seen=seen,
            ):
                return False
            out.append(
                _ParallelLeaf(
                    branches=(tuple(left_out), tuple(right_out)),
                    product=True,
                )
            )
            return True
        if isinstance(e, ExprScan):
            cell_name = _expr_to_name(e.expr)
            if cell_name is None:
                return False
            out.append(
                _ScanLeaf(
                    name=cell_name,
                    init=e.init,
                    outputs=e.outputs,
                    reverse=e.reverse,
                )
            )
            return True
        return False

    if not walk(expr):
        return None
    return tuple(out)


def _walk_into(
    e: Expr,
    out: list[_ChainElem],
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    seen: tuple[str, ...],
) -> bool:
    """Run `_flatten_compose` against `e` and append its elements to
    `out`. Returns True on success.

    A bare `ExprIdent` to a morphism / let-bound name is treated as
    a single-element chain so parallel branches like
    `fan(forward_path, backward_path)` work when each branch is
    itself a let-bound morphism.
    """
    sub = _flatten_compose(e, morphisms=morphisms, lets=lets, seen=seen)
    if sub is not None:
        out.extend(sub)
        return True
    if isinstance(e, ExprIdent):
        return _resolve_ident_leaf(
            e.name,
            morphisms=morphisms,
            lets=lets,
            seen=seen,
            out=out,
        )
    return False


def _resolve_ident_leaf(
    name: str,
    *,
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    seen: tuple[str, ...],
    out: list[_ChainElem],
) -> bool:
    """Resolve a bare-identifier leaf to one or more chain elements.

    A morphism identifier classifies as stochastic (has an init
    clause naming a family) or deterministic (no init clause).
    A let-bound identifier whose RHS is itself a composite expands
    transitively into that let's chain. An identifier that resolves
    to nothing recognised is treated as deterministic so the host
    wires it.
    """
    if name in morphisms:
        decl = morphisms[name]
        if _morphism_is_stochastic(decl):
            out.append(_StochasticLeaf(name=name))
        else:
            out.append(_DeterministicLeaf(name=name))
        return True
    if name in lets:
        sub = _resolve_to_chain(
            name,
            morphisms=morphisms,
            lets=lets,
            _seen=seen,
        )
        if sub is not None:
            out.extend(sub)
            return True
        inner = lets[name]
        if isinstance(inner, ExprIdent):
            return _resolve_ident_leaf(
                inner.name,
                morphisms=morphisms,
                lets=lets,
                seen=(*seen, name),
                out=out,
            )
        return False
    out.append(_DeterministicLeaf(name=name))
    return True


def _morphism_is_stochastic(decl: MorphismDecl) -> bool:
    """True iff `decl` carries either a `~ Family(args)` init clause
    or a `~ <bare-family-identifier>` init clause, or is an embedding,
    which places a Gaussian kernel at each element's centre."""
    if decl.init_family is not None:
        return True
    if isinstance(decl.init_expr, ExprIdent):
        return True
    return any(
        entry.key == "role"
        and isinstance(entry.value, OptionName)
        and entry.value.value == "embed"
        for entry in decl.options
    )


def _expr_to_name(e: Expr) -> str | None:
    """If `e` is an `ExprIdent`, return its name; otherwise None."""
    if isinstance(e, ExprIdent):
        return e.name
    return None


class _Program:
    """What one program's expansion carries along its steps.

    Parameters
    ----------
    morphisms
        The morphism table, extended with replicas and copies.
    lets
        The define table.
    programs
        The module's program declarations by name, extended with the
        step programs of scans.
    domain_names
        The names the program's domain factors are read through.
    domain_factors
        The program's domain factors, in order.
    """

    def __init__(
        self,
        morphisms: dict[str, MorphismDecl],
        lets: dict[str, Expr],
        programs: dict[str, ProgramDecl],
        domain_names: tuple[str, ...],
        domain_factors: tuple[ObjectExpr, ...],
    ) -> None:
        self.morphisms = morphisms
        self.lets = lets
        self.programs = programs
        self.domain_names = domain_names
        self.domain_factors = domain_factors
        self.counter = 0

    def fresh(self, stem: str) -> str:
        """A name no earlier step of the expansion bound.

        Parameters
        ----------
        stem : str
            The name's stem.

        Returns
        -------
        str
            The stem with the next serial.
        """
        self.counter += 1
        return f"{stem}_{self.counter}"


def _refuse_program_leaves(
    chain: tuple[_ChainElem, ...],
    program: _Program,
    step: SampleStep | ObserveStep,
) -> None:
    """Refuse a chain that composes a program other than as a scan's cell.

    A program composes with ``sample y <- p(x)``, or runs at every
    position as the cell of ``scan(p)``. Composed with ``>>``, repeated
    with ``repeat``, stacked with ``stack``, or bundled with ``fan`` or
    ``@``, it would be read as a host function of its input, which it
    is not.

    Parameters
    ----------
    chain : tuple[_ChainElem, ...]
        The chain, searched through parallel branches.
    program : _Program
        The enclosing program's expansion.
    step : SampleStep | ObserveStep
        The step being expanded.

    Raises
    ------
    StepResolutionError
        If a non-scan element names a program.
    """
    for elem in chain:
        if isinstance(elem, _ParallelLeaf):
            for branch in elem.branches:
                _refuse_program_leaves(branch, program, step)
        elif (
            isinstance(elem, _StochasticLeaf | _DeterministicLeaf)
            and elem.name in program.programs
        ):
            raise StepResolutionError(
                "qvr-expand",
                [f"chain:program:{elem.name}"],
                explanation=(
                    f"the morphism {step.morphism!r} drawn at line {step.line} "
                    f"composes the program {elem.name!r} with `>>`, `repeat`, "
                    f"`stack`, `fan`, or `@`; a program composes by drawing "
                    f"from it, `sample y <- {elem.name}(x)`, or runs at every "
                    f"position as the cell of `scan({elem.name})`"
                ),
            )


def _refuse_sequence_function_leaves(
    chain: tuple[_ChainElem, ...],
    program: _Program,
    step: SampleStep | ObserveStep,
) -> None:
    """Refuse a chain that composes a sequence function.

    A learned function over whole sequences (``param_source`` of
    ``self_attention`` or ``positions``) reads every position of a
    sequence at once. A chain applies its links one value at a time, a
    scan's cell and the links before it at every position, so neither
    can apply it; a program applies it in a ``let`` to the sequence a
    scan with ``outputs=all`` binds.

    Parameters
    ----------
    chain : tuple[_ChainElem, ...]
        The chain, searched through parallel branches.
    program : _Program
        The enclosing program's expansion.
    step : SampleStep | ObserveStep
        The step being expanded.

    Raises
    ------
    StepResolutionError
        If a link or a scan's cell names a sequence function.
    """
    for elem in chain:
        if isinstance(elem, _ParallelLeaf):
            for branch in elem.branches:
                _refuse_sequence_function_leaves(branch, program, step)
            continue
        name = getattr(elem, "name", None)
        decl = program.morphisms.get(name) if isinstance(name, str) else None
        if decl is None or sequence_source_kind(decl) is None:
            continue
        raise StepResolutionError(
            "qvr-expand",
            [f"chain:sequence-function:{name}"],
            explanation=(
                f"the morphism {step.morphism!r} drawn at line {step.line} "
                f"composes {name!r}, which reads a whole sequence at once, in "
                f"a chain or a scan that applies it one position at a time; "
                f"bind the sequence and apply it in a let, as in "
                f"`sample xs <- states` then `let ys = {name}(xs)`"
            ),
        )


def _expand_step(
    step: SampleStep | ObserveStep,
    chain: tuple[_ChainElem, ...],
    program: _Program,
) -> list[ProgramStep]:
    """Convert a single sample / observe step that references a
    composite-let chain into N atomic steps.

    The first N-1 steps are fresh latent samples / let bindings
    named `<base>_chain_<counter>`; the final step keeps the
    original step's bound variable name.

    Each leaf's step shape comes from its `_ChainElem.kind`:

    * stochastic -> `SampleStep` drawing from the morphism's
      family, with the previous chain output threaded into the
      first positional slot.
    * deterministic -> `LetStep` whose RHS is `name(prev)`.
    * scan -> `LetStep` whose RHS is `scan(step, input)`, the step
      program built from the chain's elements before the scan and
      the scan's cell, applied at every position of the input.
    * parallel -> one chain per branch sharing the same upstream
      input, or, for a tensor product, each branch on its own
      factor of the input, followed by a final `LetStep`
      aggregating the branch tails into a list literal.

    Parameters
    ----------
    step
        The step.
    chain
        The chain its morphism slot resolves to.
    program
        The enclosing program's expansion.

    Returns
    -------
    list[ProgramStep]
        The atomic steps.
    """
    if len(chain) < 1:
        return [step]
    _refuse_program_leaves(chain, program, step)
    if len(chain) > 1 or not isinstance(chain[0], _DeterministicLeaf):
        _refuse_sequence_function_leaves(chain, program, step)
    base_name = step.vars[0] if step.vars else "tmp"
    if len(chain) == 1 and isinstance(chain[0], _StochasticLeaf):
        elem = chain[0]
        if elem.name == step.morphism:
            return [step]
        if isinstance(step, ObserveStep):
            return [
                ObserveStep(
                    vars=(base_name,),
                    morphism=elem.name,
                    args=step.args,
                    index=step.index,
                    axes=step.axes,
                    via=step.via,
                    via_axes=step.via_axes,
                    options=step.options,
                    line=step.line,
                    col=step.col,
                )
            ]
        return [
            SampleStep(
                vars=(base_name,),
                morphism=elem.name,
                args=step.args,
                index=step.index,
                axes=step.axes,
                options=step.options,
                line=step.line,
                col=step.col,
            )
        ]
    return _emit_chain(
        chain,
        base_name=base_name,
        terminal_var=base_name,
        prev_var=None,
        head_args=step.args,
        original=step,
        program=program,
        keep_last=True,
    )


def _emit_chain(
    chain: tuple[_ChainElem, ...],
    *,
    base_name: str,
    terminal_var: str,
    prev_var: str | None,
    head_args: tuple[DrawArg, ...] | None,
    original: SampleStep | ObserveStep,
    program: _Program,
    keep_last: bool,
    prev_sequence: bool = False,
) -> list[ProgramStep]:
    """Emit the steps of a chain, threading each output into the next.

    A scan in the chain absorbs the elements before it: they and the
    scan's cell form a step program applied at every position of the
    chain's input, and the chain resumes from the scan's value, its
    final state or, for ``outputs=all``, its every state. A ``fan`` or
    ``@`` whose branches scan is not absorbed: it runs first, and the
    scan after it runs over the sequence its branches pair.

    Parameters
    ----------
    chain
        The chain.
    base_name
        The stem intermediate names are built from.
    terminal_var
        The name the chain's output binds.
    prev_var
        The upstream output, or ``None`` when the chain reads the
        step's row or the program's input.
    head_args
        The row the chain's head conditions on.
    original
        The step being expanded.
    program
        The enclosing program's expansion.
    keep_last
        Whether the last element keeps the original step's shape (an
        observation, its annotations); a branch of a parallel element
        never does.
    prev_sequence
        Whether ``prev_var`` holds a sequence, one entry per position,
        as a scan with ``outputs=all`` leaves.

    Returns
    -------
    list[ProgramStep]
        The emitted steps.

    Raises
    ------
    StepResolutionError
        If a scan runs over a value that is not a sequence, or an
        element other than a scan or a ``fan`` of scans reads a
        sequence.
    """
    out: list[ProgramStep] = []
    elements = list(chain)

    def emit_plain(segment: list[_ChainElem], ends_chain: bool) -> None:
        """Emit elements that are not scans, one after another.

        Parameters
        ----------
        segment : list[_ChainElem]
            The elements.
        ends_chain : bool
            Whether the segment's last element is the chain's.
        """
        nonlocal prev_var, prev_sequence
        last_index = len(segment) - 1
        for index, elem in enumerate(segment):
            if prev_var is not None and prev_sequence:
                _refuse_sequence_reader(elem, original)
            is_last = ends_chain and index == last_index
            var = terminal_var if is_last else program.fresh(f"{base_name}_chain")
            out.extend(
                _emit_chain_elem(
                    elem=elem,
                    terminal_var=var,
                    prev_var=prev_var,
                    is_last=is_last and keep_last,
                    original=original,
                    program=program,
                    head_args=head_args,
                    prev_sequence=prev_sequence,
                )
            )
            prev_sequence = _sequence_valued((elem,), prev_sequence)
            prev_var = var

    while elements:
        scan_index = next(
            (
                index
                for index, elem in enumerate(elements)
                if isinstance(elem, _ScanLeaf)
            ),
            None,
        )
        if scan_index is None:
            emit_plain(elements, ends_chain=True)
            break
        # A parallel element whose branches scan runs before the scan
        # rather than inside its step: its branches each run along the
        # whole sequence, and the scan reads what they pair.
        hoisted = max(
            (
                index
                for index, elem in enumerate(elements[:scan_index])
                if isinstance(elem, _ParallelLeaf) and _scans((elem,))
            ),
            default=None,
        )
        if hoisted is not None:
            emit_plain(elements[: hoisted + 1], ends_chain=False)
            head_args = None
            elements = elements[hoisted + 1 :]
            continue
        scan = elements[scan_index]
        assert isinstance(scan, _ScanLeaf)
        prefix = tuple(elements[:scan_index])
        rest = elements[scan_index + 1 :]
        if prev_var is not None and not prev_sequence:
            raise StepResolutionError(
                "qvr-expand",
                [f"scan:input:{scan.name}"],
                explanation=(
                    f"the scan over {scan.name!r} drawn at line {original.line} "
                    f"runs over {prev_var!r}, one value rather than a sequence; "
                    f"a scan before it returns every state with `outputs=all`"
                ),
            )
        source = _scan_input(prev_var, head_args, program, original)
        var = terminal_var if not rest else program.fresh(f"{base_name}_chain")
        cell_program = program.programs.get(scan.name)
        if cell_program is not None and returns_marginalized_latent(cell_program):
            out.append(
                _marginal_scan(
                    scan, cell_program, source, var, bool(prefix), bool(rest), original
                )
            )
            break
        step_name = _scan_step_program(
            scan.name, prefix, program, original, _scan_label(var, original)
        )
        out.append(
            LetStep(
                name=var,
                value=scan_let_expression(
                    ScanCall(
                        step=step_name,
                        sequence=source,
                        init=(
                            None
                            if scan.init == "zeros"
                            else f"{scan.name}{LEARNED_SCAN_INIT_SUFFIX}"
                            if scan.init == "learned"
                            # Any other name is a binding of the
                            # scanning program the scan starts from.
                            else scan.init
                        ),
                        outputs=scan.outputs,
                        reverse=scan.reverse,
                    )
                ),
                line=original.line,
                col=original.col,
            )
        )
        prev_var = var
        prev_sequence = scan.outputs == "all"
        head_args = None
        elements = rest
    return out


def _marginal_scan(
    scan: _ScanLeaf,
    cell: ProgramDecl,
    source: str,
    var: str,
    prefixed: bool,
    followed: bool,
    original: SampleStep | ObserveStep,
) -> LetStep:
    """Bind a scan whose cell marginalizes and returns its state.

    The scan sums the chain's states out, so it is the whole chain a
    ``sample`` step draws: it reads its sequence directly, names the
    distribution of the state before the first position with ``init``,
    and its value is that of no state. The cell is checked here, for
    every consumer of the expansion.

    Parameters
    ----------
    scan : _ScanLeaf
        The scan.
    cell : ProgramDecl
        Its cell, which returns the latent it marginalizes.
    source : str
        The sequence the scan runs over.
    var : str
        The name the scan binds.
    prefixed : bool
        Whether the chain applies other elements before the scan.
    followed : bool
        Whether the chain applies other elements after the scan.
    original : SampleStep | ObserveStep
        The step being expanded.

    Returns
    -------
    LetStep
        ``let var = scan(cell, xs, init, ..., "marginal")``.

    Raises
    ------
    StepResolutionError
        If the cell is not a chain step the forward algorithm sums out,
        or the scan is composed with other elements, observed, started
        from no distribution, or asked for every state.
    """
    markov_cell(cell)
    name = cell.name
    if prefixed or followed:
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:marginal:chain:{name}"],
            explanation=(
                f"the marginalized chain over {name!r} drawn at line "
                f"{original.line} is composed with other elements; it reads its "
                f"sequence directly and is the whole of what the step draws"
            ),
        )
    if isinstance(original, ObserveStep):
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:marginal:observe:{name}"],
            explanation=(
                f"the marginalized chain over {name!r} is observed at line "
                f"{original.line}; its states are summed out and its "
                f"observations are the cell's, so draw it with `sample`"
            ),
        )
    if scan.init in ("zeros", "learned"):
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:marginal:init:{name}"],
            explanation=(
                f"the marginalized chain over {name!r} drawn at line "
                f"{original.line} names no distribution of its first state; "
                f"write `scan({name}, init=p)` with `p` a probability vector "
                f"over the states bound before the scan"
            ),
        )
    if scan.outputs == "all":
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:marginal:outputs:{name}"],
            explanation=(
                f"the marginalized chain over {name!r} drawn at line "
                f"{original.line} sums its states out, so it has no state per "
                f"position to return with `outputs=all`"
            ),
        )
    return LetStep(
        name=var,
        value=scan_let_expression(
            ScanCall(
                step=name,
                sequence=source,
                init=scan.init,
                reverse=scan.reverse,
                marginal=True,
            )
        ),
        line=original.line,
        col=original.col,
    )


def refuse_marginal_chain_reads(
    draws: tuple[ProgramStep, ...], returned: tuple[str, ...]
) -> None:
    """Refuse a read of the name a marginalized chain binds.

    The chain sums its states out, so the name its ``sample`` step binds
    holds no state: no later step reads it and the program does not
    return it. A marginalized chain inside a ``marginalize`` block is
    refused too, since the block's weight is a single number per atom.

    Parameters
    ----------
    draws : tuple[ProgramStep, ...]
        A program's expanded steps.
    returned : tuple[str, ...]
        The names it returns.

    Raises
    ------
    StepResolutionError
        If a later step or the return reads such a name, or such a
        chain lies inside a ``marginalize`` block.
    """
    for index, step in enumerate(draws):
        if isinstance(step, MarginalizeStep):
            for inner in step.scope:
                scan = (
                    scan_let_call(inner.value) if isinstance(inner, LetStep) else None
                )
                if scan is not None and scan.marginal:
                    raise StepResolutionError(
                        "qvr-expand",
                        [f"scan:marginal:scope:{scan.step}"],
                        explanation=(
                            f"the marginalized chain over {scan.step!r} at line "
                            f"{inner.line} lies inside the block over "
                            f"{step.var!r}; draw it in the program's body"
                        ),
                    )
            continue
        if not isinstance(step, LetStep):
            continue
        scan = scan_let_call(step.value)
        if scan is None or not scan.marginal:
            continue
        readers = [
            later
            for later in draws[index + 1 :]
            if step.name in _step_value_names(later)
        ]
        if readers or step.name in returned:
            where = f"line {readers[0].line}" if readers else "the program's return"
            raise StepResolutionError(
                "qvr-expand",
                [f"scan:marginal:read:{step.name}"],
                explanation=(
                    f"{step.name!r} binds the marginalized chain over "
                    f"{scan.step!r}, whose states are summed out, so it holds "
                    f"no state for {where} to read; the chain contributes its "
                    f"log marginal likelihood to the joint"
                ),
            )


def _refuse_cell_uses(
    draws: tuple[ProgramStep, ...], programs: dict[str, ProgramDecl]
) -> None:
    """Refuse every draw or call of a cell that returns a marginalized latent.

    Parameters
    ----------
    draws : tuple[ProgramStep, ...]
        A program body, searched through marginalization scopes.
    programs : dict[str, ProgramDecl]
        The module's programs by name.

    Raises
    ------
    StepResolutionError
        If a step draws from or calls such a cell other than as the cell
        of a scan.
    """
    for step in draws:
        if isinstance(step, MarginalizeStep):
            _refuse_cell_uses(step.scope, programs)
        else:
            _refuse_cell_use(step, programs)


def _refuse_cell_use(step: ProgramStep, programs: dict[str, ProgramDecl]) -> None:
    """Refuse a draw or call of a cell that returns a marginalized latent.

    Parameters
    ----------
    step : ProgramStep
        A step of a program body.
    programs : dict[str, ProgramDecl]
        The module's programs by name.

    Raises
    ------
    StepResolutionError
        If the step draws from or calls such a cell other than as the
        cell of a scan.
    """
    name: str | None = None
    if isinstance(step, SampleStep | ObserveStep):
        name = step.morphism
    elif isinstance(step, CallStep):
        name = step.call.callee
    cell = programs.get(name) if name is not None else None
    if cell is not None and returns_marginalized_latent(cell):
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:marginal:entry:{cell.name}"],
            explanation=(
                f"the step at line {step.line} runs {cell.name!r}, which returns "
                f"the latent it marginalizes and so has no value of its own; it "
                f"runs only as the cell of `scan({cell.name}, init=p)`"
            ),
        )


def _sequence_valued(chain: tuple[_ChainElem, ...], incoming: bool) -> bool:
    """Whether a chain's value is a sequence, one entry per position.

    Parameters
    ----------
    chain : tuple[_ChainElem, ...]
        The chain.
    incoming : bool
        Whether the value the chain reads is a sequence.

    Returns
    -------
    bool
        ``True`` when the chain ends in a scan with ``outputs=all`` or
        in a parallel element all of whose branches are sequences; the
        incoming answer for an empty chain.
    """
    valued = incoming
    for elem in chain:
        if isinstance(elem, _ScanLeaf):
            valued = elem.outputs == "all"
        elif isinstance(elem, _ParallelLeaf):
            branches = [branch for branch in elem.branches if branch]
            valued = bool(branches) and all(
                _sequence_valued(branch, valued) for branch in branches
            )
        else:
            valued = False
    return valued


def _refuse_sequence_reader(
    elem: _ChainElem, original: SampleStep | ObserveStep
) -> None:
    """Refuse an element that would read a whole sequence as one value.

    A scan with ``outputs=all`` leaves one state per position. What
    follows it in a chain runs at every position only inside another
    scan's step, so a morphism after it, outside a scan, would read
    the sequence as one value.

    Parameters
    ----------
    elem : _ChainElem
        The element after the sequence.
    original : SampleStep | ObserveStep
        The step being expanded.

    Raises
    ------
    StepResolutionError
        Unless the element is a parallel element, whose branches are
        checked in turn.
    """
    if isinstance(elem, _ParallelLeaf):
        return
    name = getattr(elem, "name", type(elem).__name__)
    raise StepResolutionError(
        "qvr-expand",
        [f"scan:outputs:{name}"],
        explanation=(
            f"the morphism {original.morphism!r} drawn at line {original.line} "
            f"applies {name!r} to the every-state sequence of a scan with "
            f"`outputs=all`; a morphism reads each state only inside a later "
            f"scan, as in `scan(cell, outputs=all) >> {name} >> scan(next)`"
        ),
    )


def _scan_input(
    prev_var: str | None,
    head_args: tuple[DrawArg, ...] | None,
    program: _Program,
    original: SampleStep | ObserveStep,
) -> str:
    """The name of the sequence a scan runs over.

    Parameters
    ----------
    prev_var
        The upstream output, when the scan is not at the chain's head.
    head_args
        The row the chain's head conditions on.
    program
        The enclosing program's expansion.
    original
        The step being expanded.

    Returns
    -------
    str
        The upstream output, else the step's one named argument, else
        the program's one domain input.

    Raises
    ------
    StepResolutionError
        If the scan's input is not one named sequence.
    """
    if prev_var is not None:
        return prev_var
    if head_args:
        if len(head_args) == 1 and isinstance(head_args[0], DrawArgName):
            return head_args[0].text
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:input:{original.morphism}"],
            explanation=(
                f"the scan drawn at line {original.line} runs over one sequence, "
                f"but the step conditions on {len(head_args)} arguments"
            ),
        )
    if len(program.domain_names) == 1 and program.domain_names[0]:
        return program.domain_names[0]
    raise StepResolutionError(
        "qvr-expand",
        [f"scan:input:{original.morphism}"],
        explanation=(
            f"the scan drawn at line {original.line} runs over the program's "
            f"input, which is not one named factor"
        ),
    )


def _scan_label(var: str, original: SampleStep | ObserveStep) -> str:
    """The label a scan's per-position sites are named under.

    Parameters
    ----------
    var : str
        The name the scan's final state binds.
    original : SampleStep | ObserveStep
        The step the scan was expanded from.

    Returns
    -------
    str
        ``var`` when it is the name the source binds; otherwise, for a
        scan inside a longer chain or a parallel branch, ``var``
        qualified by that name as ``"<binding>$<var>"``. Every label
        derives from a binding the source wrote, so two scans of one
        program never share one, and the ``$`` the source cannot write
        keeps a qualified label apart from every source name.
    """
    if var in original.vars:
        return var
    return f"{original.vars[0]}${var}"


def _scan_step_program(
    cell: str,
    prefix: tuple[_ChainElem, ...],
    program: _Program,
    original: SampleStep | ObserveStep,
    label: str,
) -> str:
    """Declare the program a scan applies at every position.

    The step program takes the position's input and the previous
    state, draws the chain's elements before the scan at the input,
    and applies the cell to the result and the state. A cell that is a
    program is called, and its sites keep their own labels; one that is
    a stochastic morphism is drawn at the site ``label``; one that is a
    deterministic morphism is applied, and contributes no site. The
    chain's elements before the scan draw at sites qualified by
    ``label``, ``"<label>$x_in"``, and the step's parameters are named
    the same way, so no name the step program binds is one the source
    could write.

    Parameters
    ----------
    cell
        The scan's cell, a program or a morphism.
    prefix
        The chain's elements before the scan.
    program
        The enclosing program's expansion, which gains the step program.
    original
        The step being expanded.
    label
        The label of the scan's per-position sites, from
        [`_scan_label`][quivers.dsl.composite_lets._scan_label].

    Returns
    -------
    str
        The step program's name.

    Raises
    ------
    StepResolutionError
        If the cell is neither a program nor a morphism, or the input
        object cannot be read off the chain.
    """
    name = f"{cell}__scan_step"
    cell_program = program.programs.get(cell)
    cell_morphism = program.morphisms.get(cell)
    if cell_program is not None:
        state_object = cell_program.codomain
        cell_domain = _domain_factors(cell_program.domain)
    elif cell_morphism is not None:
        state_object = cell_morphism.codomain
        cell_domain = _domain_factors(cell_morphism.domain)
    else:
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:cell:{cell}"],
            explanation=(f"`scan({cell})` names neither a program nor a morphism"),
        )
    if len(cell_domain) != 2:
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:cell:{cell}"],
            explanation=(
                f"the cell {cell!r} of a scan takes an input and a state; its "
                f"domain has {len(cell_domain)} factors"
            ),
        )
    input_object: ObjectExpr | None = None
    if prefix:
        head = prefix[0]
        if isinstance(head, _StochasticLeaf | _DeterministicLeaf):
            decl = program.morphisms.get(head.name)
            if decl is not None:
                input_object = decl.domain
    else:
        input_object = cell_domain[0]
    if input_object is None:
        raise StepResolutionError(
            "qvr-expand",
            [f"scan:input:{cell}"],
            explanation=(
                f"the input object of the scan over {cell!r} cannot be read off "
                f"its chain"
            ),
        )
    steps: list[ProgramStep] = []
    entry = f"{label}$x_t"
    state = f"{label}$h_prev"
    fed = entry
    if prefix:
        fed = f"{label}$x_in"
        steps.extend(
            _emit_chain(
                prefix,
                base_name=fed,
                terminal_var=fed,
                prev_var=None,
                head_args=(DrawArgName(text=entry),),
                original=original,
                program=program,
                keep_last=False,
            )
        )
    if cell_program is not None:
        steps.append(
            CallStep(
                name=label,
                call=QiecCallComputation(
                    callee=cell,
                    arguments=(LetExprVar(name=fed), LetExprVar(name=state)),
                ),
                line=original.line,
                col=original.col,
            )
        )
    elif cell_morphism is not None and not _morphism_is_stochastic(cell_morphism):
        steps.append(
            LetStep(
                name=label,
                value=LetExprCall(
                    func=cell, args=(LetExprVar(name=fed), LetExprVar(name=state))
                ),
                line=original.line,
                col=original.col,
            )
        )
    else:
        steps.append(
            SampleStep(
                vars=(label,),
                morphism=cell,
                args=(DrawArgName(text=fed), DrawArgName(text=state)),
                line=original.line,
                col=original.col,
            )
        )
    declaration = ProgramDecl(
        name=name,
        params=(entry, state),
        domain=ObjectProduct(components=(input_object, state_object)),
        codomain=state_object,
        draws=tuple(steps),
        return_vars=(label,),
        line=original.line,
        col=original.col,
    )
    # The same cell scanned after another chain is another step
    # program, named apart by its serial.
    while name in program.programs and not _same_program(
        program.programs[name], declaration
    ):
        name = program.fresh(f"{cell}__scan_step")
        declaration = declaration.with_(name=name)
    program.programs[name] = declaration
    return name


def _same_program(left: ProgramDecl, right: ProgramDecl) -> bool:
    """Whether two program declarations agree apart from their position.

    Parameters
    ----------
    left : ProgramDecl
        One declaration.
    right : ProgramDecl
        The other.

    Returns
    -------
    bool
        ``True`` when their names, parameters, objects, steps, and
        returns agree.
    """
    return (
        left.name == right.name
        and left.params == right.params
        and left.domain == right.domain
        and left.codomain == right.codomain
        and left.draws == right.draws
        and left.return_vars == right.return_vars
    )


def _emit_chain_elem(
    *,
    elem: _ChainElem,
    terminal_var: str,
    prev_var: str | None,
    is_last: bool,
    original: SampleStep | ObserveStep,
    program: _Program,
    head_args: tuple[DrawArg, ...] | None,
    prev_sequence: bool = False,
) -> list[ProgramStep]:
    """Emit the program steps for one chain element.

    Parameters
    ----------
    elem
        The element.
    terminal_var
        The name its output binds.
    prev_var
        The upstream chain output, or ``None`` at the head.
    is_last
        Whether the element is the chain's last and keeps the original
        step's shape.
    original
        The step being expanded.
    program
        The enclosing program's expansion.
    head_args
        The row the chain's head conditions on: the original step's
        arguments, or a branch's factor of them.
    prev_sequence
        Whether ``prev_var`` holds a sequence, one entry per position.

    Returns
    -------
    list[ProgramStep]
        The emitted steps.

    Raises
    ------
    AssertionError
        If the element is of no known variant.
    """
    if isinstance(elem, _StochasticLeaf):
        args = _derive_chain_args(
            morphism_name=elem.name,
            prev_var=prev_var,
            head_args=head_args,
            morphisms=program.morphisms,
        )
        if is_last and isinstance(original, ObserveStep):
            return [
                ObserveStep(
                    vars=(terminal_var,),
                    morphism=elem.name,
                    args=args,
                    index=original.index,
                    axes=original.axes,
                    via=original.via,
                    via_axes=original.via_axes,
                    options=original.options,
                    line=original.line,
                    col=original.col,
                )
            ]
        sample_options = (
            original.options if (is_last and isinstance(original, SampleStep)) else ()
        )
        sample_axes = (
            original.axes if (is_last and isinstance(original, SampleStep)) else None
        )
        sample_index = (
            original.index if (is_last and isinstance(original, SampleStep)) else None
        )
        return [
            SampleStep(
                vars=(terminal_var,),
                morphism=elem.name,
                args=args,
                index=sample_index,
                axes=sample_axes,
                options=sample_options,
                line=original.line,
                col=original.col,
            )
        ]
    if isinstance(elem, _DeterministicLeaf):
        rhs = _function_call_expr(elem.name, prev_var)
        return [
            LetStep(
                name=terminal_var,
                value=rhs,
                line=original.line,
                col=original.col,
            )
        ]
    if isinstance(elem, _ParallelLeaf):
        return _emit_parallel(
            elem=elem,
            terminal_var=terminal_var,
            prev_var=prev_var,
            original=original,
            program=program,
            head_args=head_args,
            prev_sequence=prev_sequence,
        )
    raise AssertionError(
        f"_emit_chain_elem: unhandled chain-elem variant {type(elem).__name__!r}"
    )


def _emit_parallel(
    *,
    elem: _ParallelLeaf,
    terminal_var: str,
    prev_var: str | None,
    original: SampleStep | ObserveStep,
    program: _Program,
    head_args: tuple[DrawArg, ...] | None,
    prev_sequence: bool = False,
) -> list[ProgramStep]:
    """Emit each parallel branch's steps against the same upstream
    input, then bundle the branch tails into a list literal bound to
    `terminal_var`.

    Branches that each end in a scan with ``outputs=all`` are paired
    position by position instead: the bundle is the sequence whose
    entry at a position lays the branches' entries there end to end,
    ``positions(a, b)`` (see `POSITIONS_BUNDLE`).

    A fan's branches all read the upstream input; a tensor product's
    each read their own factor of it, which at the head of a chain is
    the original step's argument at their position or, absent
    arguments, the program's domain factor at their position.

    A branch's steps run in declaration order, with each branch's
    own fresh chain-position vars. The aggregated list literal lets
    a downstream `combine`-style morphism index into the merged
    result; even when no downstream consumer reads the parallel
    element, the list materialises so the program's return value
    has a deterministic shape.

    Parameters
    ----------
    elem
        The parallel element.
    terminal_var
        The name the bundle binds.
    prev_var
        The upstream chain output, or ``None`` at the head.
    original
        The step being expanded.
    program
        The enclosing program's expansion.
    head_args
        The row the chain's head conditions on.
    prev_sequence
        Whether ``prev_var`` holds a sequence, one entry per position.

    Returns
    -------
    list[ProgramStep]
        The emitted steps.

    Raises
    ------
    StepResolutionError
        If some branches end in a sequence and others in one value.
    """
    out: list[ProgramStep] = []
    branch_tails: list[str] = []
    valued = [
        _sequence_valued(branch, prev_sequence) for branch in elem.branches if branch
    ]
    paired = bool(valued) and all(valued)
    if any(valued) and not paired:
        raise StepResolutionError(
            "qvr-expand",
            ["scan:outputs:fan"],
            explanation=(
                f"the morphism {original.morphism!r} drawn at line "
                f"{original.line} pairs branches of which some return every "
                f"state of a scan (`outputs=all`) and others one value; pair "
                f"sequences with sequences, position by position"
            ),
        )
    for branch_idx, branch in enumerate(elem.branches):
        branch_head = head_args
        if elem.product and prev_var is None:
            if head_args and len(head_args) == len(elem.branches):
                branch_head = (head_args[branch_idx],)
            elif (
                not head_args
                and len(program.domain_names) == len(elem.branches)
                and program.domain_names[branch_idx]
            ):
                branch_head = (DrawArgName(text=program.domain_names[branch_idx]),)
        if not branch:
            continue
        tail = program.fresh(f"{terminal_var}_par_{branch_idx}")
        out.extend(
            _emit_chain(
                branch,
                base_name=f"{terminal_var}_par_{branch_idx}",
                terminal_var=tail,
                prev_var=prev_var,
                head_args=branch_head,
                original=original,
                program=program,
                keep_last=False,
                prev_sequence=prev_sequence,
            )
        )
        branch_tails.append(tail)
    if not branch_tails:
        out.append(
            LetStep(
                name=terminal_var,
                value=LetExprCall(func="tuple", args=()),
                line=original.line,
                col=original.col,
            )
        )
        return out
    items = tuple(LetExprVar(name=t) for t in branch_tails)
    out.append(
        LetStep(
            name=terminal_var,
            value=(
                LetExprCall(func=POSITIONS_BUNDLE, args=items)
                if paired
                else LetExprList(items=items)
            ),
            line=original.line,
            col=original.col,
        )
    )
    return out


def _function_call_expr(func_name: str, prev_var: str | None) -> LetExprNode:
    """Build a let-expression for a deterministic morphism leaf.

    With an upstream input, emit `morphism_name + prev`: a sum
    over the morphism's parameter table and the previous chain
    output. The sum operator carries no inference content (the
    morphism is treated as a host-supplied frozen parameter), but
    it surfaces both `morphism_name` and `prev` as free
    variables, which the lower pass declares as data inputs that
    the host wires. Every target language accepts elementwise
    addition on plated arrays of matching shape, so the rendered
    code is syntactically valid in each backend.

    With no upstream input (the chain head), emit a bare
    `LetExprVar(morphism_name)`: the unindexed reference yields
    the whole parameter table.
    """
    if prev_var is None:
        return LetExprVar(name=func_name)
    return LetExprBinOp(
        op="+",
        left=LetExprVar(name=func_name),
        right=LetExprVar(name=prev_var),
    )


def _derive_chain_args(
    *,
    morphism_name: str,
    prev_var: str | None,
    head_args: tuple[DrawArg, ...] | None,
    morphisms: dict[str, MorphismDecl],
) -> tuple[DrawArg, ...] | None:
    """Compute the chain-position args for a kernel morphism.

    The argument list of a draw whose head names a declared morphism
    is a *conditioning row*, not a list of family parameters: the
    morphism's own parameter map computes the family's parameters
    from that row (``docs/semantics/morphisms.md`` §2.1), and the row
    is what the runtime assembles from the list before applying the
    morphism.
    A chained kernel thus conditions on the upstream step's
    output; the chain's head conditions on the row the original step
    wrote, and an absent list conditions on the chain's own input,
    which is exactly the pair of draws
    [`SampledComposition`][quivers.continuous.morphisms.SampledComposition]
    makes: ``y ~ f(x, .)`` then ``z ~ g(y, .)``.

    A declaration that writes its own parameters (``~ Normal(0, 1)``)
    means them: it denotes a constant kernel, so the chain position
    keeps the declared arguments and reads nothing upstream.

    Parameters
    ----------
    morphism_name
        The kernel at this chain position.
    prev_var
        The upstream chain output, or ``None`` at the head.
    head_args
        The original step's argument list, which the head conditions on.
    morphisms
        The module's morphism table.

    Returns
    -------
    tuple[DrawArg, ...] | None
        The position's argument list, or ``None`` when the position
        conditions on the chain's own input.
    """
    decl = morphisms.get(morphism_name)
    if decl is not None and decl.init_family is not None:
        explicit_args = decl.init_family.args
        if explicit_args:
            return tuple(atom_to_draw_arg(a) for a in explicit_args)
    if prev_var is None:
        return head_args if head_args else None
    return (atom_to_draw_arg(prev_var),)


LEARNED_SCAN_INIT_SUFFIX = "_scan_init"
"""The suffix of the input a scan with ``init=learned`` starts from."""


def is_learned_scan_init(name: str) -> bool:
    """Whether a scan's initial-state name is a learned initial state.

    Parameters
    ----------
    name : str
        The third argument of ``scan(step, xs, init)``.

    Returns
    -------
    bool
        True for the ``<cell>_scan_init`` input ``init=learned``
        introduces; false for a binding of the scanning program.
    """
    return name.endswith(LEARNED_SCAN_INIT_SUFFIX)


POSITIONS_BUNDLE = "$positions"
"""The let function pairing sequences position by position.

``$positions(a, b)`` over sequences of one length is the sequence whose
entry at each position lays the entries of ``a`` and ``b`` there end to
end. The expansion of a ``fan`` or ``@`` whose branches each return
every state of a scan binds it; the ``$`` keeps it apart from every
function a source can call.
"""


class ScanCall(dx.Model):
    """A scan as the expansion binds it: ``let h = scan(step, xs, ...)``.

    Parameters
    ----------
    step : str
        The step program's name.
    sequence : str
        The name of the sequence the scan runs over.
    init : str | None
        The initial state's name: a binding of the scanning program,
        or the ``<cell>_scan_init`` input of a learned initial state;
        ``None`` to start from zero.
    outputs : Literal["last", "all"]
        Whether the scan's value is its final state or every state.
    reverse : bool
        Whether the scan visits the positions from the last to the
        first.
    marginal : bool
        Whether the scan sums out a finite state its cell marginalizes
        and returns (a marginalized chain); ``step``
        then names the cell itself and ``init`` the distribution of the
        state before the first position.
    """

    step: str
    sequence: str
    init: str | None = None
    outputs: Literal["last", "all"] = "last"
    reverse: bool = False
    marginal: bool = False


_SCAN_ALL_FLAG = "outputs=all"
_SCAN_REVERSE_FLAG = "reverse"
_SCAN_MARGINAL_FLAG = "marginal"
_SCAN_FLAGS = (_SCAN_ALL_FLAG, _SCAN_REVERSE_FLAG, _SCAN_MARGINAL_FLAG)


def scan_let_expression(call: ScanCall) -> LetExprCall:
    """The let expression binding a scan.

    Parameters
    ----------
    call : ScanCall
        The scan.

    Returns
    -------
    LetExprCall
        ``scan(step, xs)``, with the initial state's name as a third
        argument when there is one, then the string flags
        ``"outputs=all"``, ``"reverse"``, and ``"marginal"`` for the
        options that differ from their defaults.
    """
    arguments: list[LetExprNode] = [
        LetExprVar(name=call.step),
        LetExprVar(name=call.sequence),
    ]
    if call.init is not None:
        arguments.append(LetExprVar(name=call.init))
    if call.outputs == "all":
        arguments.append(LetExprString(value=_SCAN_ALL_FLAG))
    if call.reverse:
        arguments.append(LetExprString(value=_SCAN_REVERSE_FLAG))
    if call.marginal:
        arguments.append(LetExprString(value=_SCAN_MARGINAL_FLAG))
    return LetExprCall(func="scan", args=tuple(arguments))


def scan_let_call(expr: LetExprNode) -> ScanCall | None:
    """Read a scan off a let expression `scan_let_expression` built.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    ScanCall | None
        The scan; ``None`` when the expression is not a scan of names.
    """
    if not isinstance(expr, LetExprCall) or expr.func != "scan":
        return None
    names = [
        argument.name for argument in expr.args if isinstance(argument, LetExprVar)
    ]
    flags = [
        argument.value for argument in expr.args if isinstance(argument, LetExprString)
    ]
    if (
        len(names) not in (2, 3)
        or len(names) + len(flags) != len(expr.args)
        or not all(
            isinstance(argument, LetExprVar) for argument in expr.args[: len(names)]
        )
        or any(flag not in _SCAN_FLAGS for flag in flags)
        or len(set(flags)) != len(flags)
    ):
        return None
    return ScanCall(
        step=names[0],
        sequence=names[1],
        init=names[2] if len(names) == 3 else None,
        outputs="all" if _SCAN_ALL_FLAG in flags else "last",
        reverse=_SCAN_REVERSE_FLAG in flags,
        marginal=_SCAN_MARGINAL_FLAG in flags,
    )


def positions_let_call(expr: LetExprNode) -> tuple[str, ...] | None:
    """Read ``$positions(a, b, ...)`` off a let expression.

    Parameters
    ----------
    expr : LetExprNode
        The expression.

    Returns
    -------
    tuple[str, ...] | None
        The paired sequences' names, in order; ``None`` when the
        expression is not such a call of names.
    """
    if not isinstance(expr, LetExprCall) or expr.func != POSITIONS_BUNDLE:
        return None
    if not expr.args or not all(isinstance(a, LetExprVar) for a in expr.args):
        return None
    return tuple(a.name for a in expr.args if isinstance(a, LetExprVar))


__all__ = []
