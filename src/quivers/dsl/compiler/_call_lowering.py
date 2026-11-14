"""Lowering a program's calls of computations to compiled runtime steps.

A program step ``let y <- f(args)`` names a computation of the checked
module. The compiled runtime runs a fixed chain of steps, so a call is
fit only when the callee's body can be stated as such a chain at the
call. Three forms are:

* a pure callee, whose body (with every pure computation it calls, and
  every ``if`` between pure branches as ``select``) is one value: its
  bindings become let steps, by the analysis the transpile planner
  shares (`inline_call`);
* a callee whose only requests are the program's own ``Random`` and
  ``Score``: it runs in place, as a draw from a program does, with each
  site it draws named by the call's binder, ``y$label``, and each score
  it adds a score step;
* ``traverse(xs, x -> f(x, ...))``: a pure ``f`` maps over ``xs`` as a
  comprehension, and an ``f`` that draws becomes one plated draw per
  site, over the axis ``xs`` runs along, named ``y$label``.

Any other call is refused with a located diagnostic that says why: the
program still runs on the reference machine, but it cannot be fit.
"""

from __future__ import annotations

from collections.abc import Mapping

from quivers.dsl.ast_nodes import (
    CallStep,
    DrawArg,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprNode,
    LetExprString,
    LetExprVar,
    LetFactorBinder,
    LetStep,
    MarginalizeStep,
    ProgramStep,
    SampleStep,
    ScoreStep,
    TypeName,
)
from quivers.dsl.compiler._prelude import CompileError
from quivers.dsl.let_expr_traversal import substitute_let_expr
from quivers.qiec.canonical import tensor_shape, tensor_type
from quivers.qiec.kinds import NAT
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    Comprehension,
    Computation,
    DistributionValue,
    Gather,
    Handle,
    If,
    LiteralValue,
    Local,
    LogDensity,
    NewInstance,
    Perform,
    Resume,
    Return,
    SiteValue,
    Value,
    Var,
    WeightSum,
)
from quivers.qiec.types import INT, REAL, IndexLiteral, TypeExpr
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pure_calls import (
    LOG_PROB_CALL,
    CallNotLowered,
    _expression,
    _LetFloater,
    computation_value,
    free_local_names,
    handled_reason,
    inline_call,
    origin_line,
    recursion_reason,
    replace_locals,
)
from quivers.transpile._builtin_spellings import RUNTIME_TARGET


def _fresh(base: str, occupied: set[str]) -> str:
    """A name no step of the program uses, which joins the occupied set.

    Parameters
    ----------
    base : str
        The preferred name.
    occupied : set[str]
        The names in use.

    Returns
    -------
    str
        ``base``, or ``base`` suffixed until it is unused.
    """
    stem = base.lstrip("_") or "arg"
    candidate = stem
    suffix = 0
    while candidate in occupied:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    occupied.add(candidate)
    return candidate


class _Plate:
    """The axis a traverse's iterations run along.

    Parameters
    ----------
    axis : str
        The object the plated draws are indexed by.
    extent : int
        Its cardinality, the number of iterations.
    index : Local
        The comprehension binder standing for the iteration.
    """

    def __init__(self, axis: str, extent: int, index: Local) -> None:
        self.axis = axis
        self.extent = extent
        self.index = index

    @property
    def size(self) -> IndexLiteral:
        """The extent as an index literal."""
        return IndexLiteral(value=self.extent, sort=NAT)

    def lifted_type(self, element: TypeExpr) -> TypeExpr:
        """The type of one value per iteration.

        Parameters
        ----------
        element : TypeExpr
            The type of one iteration's value.

        Returns
        -------
        TypeExpr
            A tensor whose leading axis runs over the iterations.
        """
        split = tensor_shape(element)
        if split is None:
            return tensor_type(element, (self.size,))
        return tensor_type(split[0], (self.size, *split[1]))


class _Lowering:
    """The steps one call contributes to the program, in order.

    Parameters
    ----------
    computations : Mapping[str, NamedComputation]
        The module's computations, by name.
    cards : dict[str, int]
        The cardinality of every finite object, which names an axis.
    occupied : set[str]
        Every name the program uses; each name the lowering binds joins it.
    program : str
        The calling program, for diagnostics.
    step : CallStep
        The call, whose location every emitted step carries.
    plate : _Plate | None
        The traverse's axis, when the call runs once per item.
    """

    def __init__(
        self,
        computations: Mapping[str, NamedComputation],
        cards: dict[str, int],
        occupied: set[str],
        program: str,
        step: CallStep,
        plate: _Plate | None,
    ) -> None:
        self.computations = computations
        self.cards = cards
        self.occupied = occupied
        self.program = program
        self.step = step
        self.plate = plate
        self.floater = _LetFloater(occupied)
        self.steps: list[ProgramStep] = []
        self.sites: set[str] = set()

    def expression(self, value: Value) -> LetExprNode:
        """A value as the let expression the compiled runtime evaluates.

        Parameters
        ----------
        value : Value
            A value reading program names only.

        Returns
        -------
        LetExprNode
            The expression.

        Raises
        ------
        CallNotLowered
            If the value has no expression form.
        """
        try:
            return _expression(value, self.cards, RUNTIME_TARGET, self.program)
        except UnsupportedConstruct as error:
            raise CallNotLowered(
                "binds a value the compiled runtime's let expressions have no "
                f"form for ({', '.join(error.kinds)})"
            ) from error

    def let(self, name: str, value: Value) -> None:
        """Bind a value's hoisted bindings and the value itself as let steps.

        Parameters
        ----------
        name : str
            The name the value binds, already reserved.
        value : Value
            The value, reading program names only.
        """
        hoisted, result = self.floater.value(value, {})
        for binding in hoisted:
            self.steps.append(self._let_step(binding.binder.name, binding.value))
        self.steps.append(self._let_step(name, result))

    def _let_step(self, name: str, value: Value) -> LetStep:
        """One let step at the call's location.

        Parameters
        ----------
        name : str
            The name bound.
        value : Value
            The value, with no binding form left in it.

        Returns
        -------
        LetStep
            The step.
        """
        return LetStep(
            name=name,
            value=self.expression(value),
            line=self.step.line,
            col=self.step.col,
        )

    def per_item(self, value: Value) -> bool:
        """Whether a value varies with the traverse's iteration.

        Parameters
        ----------
        value : Value
            The value.

        Returns
        -------
        bool
            True on a plated lowering when the value reads the
            iteration's index.
        """
        return self.plate is not None and self.plate.index.name in free_local_names(
            value
        )

    def bound(self, hint: str, value: Value, type_: TypeExpr) -> Value:
        """Bind a value to a fresh program name and read it back.

        A variable or literal is read in place. On a plated lowering a
        value that varies with the iteration binds one entry per
        iteration, and the read is that tensor at the index.

        Parameters
        ----------
        hint : str
            The preferred name.
        value : Value
            The value, reading program names and the iteration index.
        type_ : TypeExpr
            The type of one iteration's value.

        Returns
        -------
        Value
            The read of the bound value.
        """
        if isinstance(value, Var | LiteralValue) and not self.per_item(value):
            return value
        if self.plate is None or not self.per_item(value):
            name = self.floater.fresh(hint)
            self.let(name, value)
            return Var(local=Local(name=name, type=type_))
        plate = self.plate
        lifted = plate.lifted_type(type_)
        name = self.floater.fresh(hint)
        self.let(
            name,
            Comprehension(
                binder=plate.index, extent=plate.size, body=value, result_type=lifted
            ),
        )
        return Gather(
            value=Var(local=Local(name=name, type=lifted)),
            index=Var(local=plate.index),
            result_type=type_,
        )

    def call(
        self,
        callee: NamedComputation,
        arguments: tuple[Value, ...],
        result_type: TypeExpr,
        prefix: str,
        active: tuple[str, ...],
    ) -> Value:
        """Lower a call of a computation, its sites named under a prefix.

        Parameters
        ----------
        callee : NamedComputation
            The computation called.
        arguments : tuple[Value, ...]
            The call's arguments, reading program names (and, on a
            plated lowering, the iteration index).
        result_type : TypeExpr
            The type of the call's result.
        prefix : str
            The name the callee's sites are named under, ``prefix$label``.
        active : tuple[str, ...]
            The computations whose bodies are being lowered, outermost
            first, which a call may not reenter.

        Returns
        -------
        Value
            The call's result, reading program names.

        Raises
        ------
        CallNotLowered
            If the callee cannot run as steps of the program.
        """
        try:
            return inline_call(
                callee, arguments, result_type, self.computations, active
            )
        except CallNotLowered as error:
            if not error.effects:
                raise
        if callee.telescope:
            raise CallNotLowered(
                f"passes static arguments to `{callee.name}`, which the "
                f"compiled runtime does not instantiate in a call"
            )
        if len(arguments) != len(callee.parameters):
            raise CallNotLowered(
                f"passes {len(arguments)} argument(s) to `{callee.name}`, "
                f"which takes {len(callee.parameters)}"
            )
        env: dict[Local, Value] = dict(zip(callee.parameters, arguments, strict=True))
        inner = (*active, callee.name)
        node: Computation = callee.body
        while isinstance(node, Bind):
            first = node.first
            if isinstance(first, Bind):
                # A nested sequence associates into the enclosing one; its
                # binder is fresh to the continuation.
                node = Bind(
                    binder=first.binder,
                    first=first.first,
                    then=Bind(binder=node.binder, first=first.then, then=node.then),
                )
                continue
            env[node.binder] = self.statement(
                first, node.binder, callee, env, prefix, inner
            )
            node = node.then
        return self.statement(
            node, Local(name=prefix, type=result_type), callee, env, prefix, inner
        )

    def statement(
        self,
        term: Computation,
        binder: Local,
        owner: NamedComputation,
        env: Mapping[Local, Value],
        prefix: str,
        active: tuple[str, ...],
    ) -> Value:
        """Lower one computation a callee's body sequences.

        Parameters
        ----------
        term : Computation
            The computation.
        binder : Local
            The callee's local its result binds, which names it.
        owner : NamedComputation
            The callee whose body holds the computation.
        env : Mapping[Local, Value]
            The read of each of the callee's locals bound so far.
        prefix : str
            The name the callee's sites are named under.
        active : tuple[str, ...]
            The computations being lowered, the owner last.

        Returns
        -------
        Value
            The read of the computation's result.

        Raises
        ------
        CallNotLowered
            If the computation cannot run as steps of the program.
        """
        if isinstance(term, Return):
            value = replace_locals(term.value, env)
            return self.bound(binder.name, value, binder.type)
        if isinstance(term, Perform):
            return self.request(term, binder, owner, env, prefix)
        if isinstance(term, Call):
            callee = self.computations.get(term.name)
            if callee is None:
                raise CallNotLowered(
                    f"calls `{term.name}` from `{owner.name}`, which is not a "
                    f"computation of the module"
                )
            if term.name in active:
                raise CallNotLowered(recursion_reason((*active, term.name)))
            arguments = tuple(replace_locals(item, env) for item in term.arguments)
            result = self.call(
                callee,
                arguments,
                binder.type,
                f"{prefix}${binder.name}",
                active,
            )
            return self.bound(binder.name, result, binder.type)
        if isinstance(term, If):
            try:
                value = computation_value(
                    term, env, binder.type, owner, self.computations, active
                )
            except CallNotLowered as error:
                if not error.effects:
                    raise
                raise CallNotLowered(
                    f"runs an `if` in `{owner.name}`{origin_line(term)} whose "
                    f"branches draw or score, so the set of sites the program "
                    f"has would depend on the branch taken, while the compiled "
                    f"runtime fits a fixed set of sites; draw outside the "
                    f"conditional and choose between values with `select`"
                ) from error
            return self.bound(binder.name, value, binder.type)
        if isinstance(term, Case):
            raise CallNotLowered(
                f"branches on a family value with `case` in `{owner.name}`"
                f"{origin_line(term)}; the compiled runtime lowers only an "
                f"`if` whose branches are pure, as `select`"
            )
        if isinstance(term, Handle | NewInstance | Resume):
            raise CallNotLowered(handled_reason(owner, term))
        raise CallNotLowered(
            f"runs a {type(term).__name__} in `{owner.name}`, which the "
            f"compiled runtime has no step for"
        )

    def request(
        self,
        term: Perform,
        binder: Local,
        owner: NamedComputation,
        env: Mapping[Local, Value],
        prefix: str,
    ) -> Value:
        """Lower a ``Random.sample`` or ``Score.add`` request to a step.

        Parameters
        ----------
        term : Perform
            The request.
        binder : Local
            The local its answer binds.
        owner : NamedComputation
            The callee making it.
        env : Mapping[Local, Value]
            The read of each of the callee's locals bound so far.
        prefix : str
            The name the callee's sites are named under.

        Returns
        -------
        Value
            The read of the drawn value; a unit for a score.

        Raises
        ------
        CallNotLowered
            If the request is on another effect, or its site or
            distribution has no step form.
        """
        request = term.request
        effect = request.effect.name
        if effect == "Random":
            return self.draw(term, binder, owner, env, prefix)
        if effect == "Score":
            (weight,) = request.arguments
            self.score(replace_locals(weight, env), prefix)
            return LiteralValue(value=None, type=binder.type)
        raise CallNotLowered(
            f"performs `{effect}` in `{owner.name}`{origin_line(term)}, which "
            f"the compiled runtime runs only through the program's own "
            f"`Random` and `Score`"
        )

    def draw(
        self,
        term: Perform,
        binder: Local,
        owner: NamedComputation,
        env: Mapping[Local, Value],
        prefix: str,
    ) -> Value:
        """Lower a draw to a sample step named ``prefix$label``.

        Parameters
        ----------
        term : Perform
            The ``Random.sample`` request.
        binder : Local
            The local the drawn value binds.
        owner : NamedComputation
            The callee drawing.
        env : Mapping[Local, Value]
            The read of each of the callee's locals bound so far.
        prefix : str
            The name the callee's sites are named under.

        Returns
        -------
        Value
            The read of the drawn value.

        Raises
        ------
        CallNotLowered
            If the site is not a literal label, the label is drawn
            twice in one call, or the distribution has no step form.
        """
        arguments = term.request.arguments
        site = arguments[0] if len(arguments) == 2 else None
        distribution = arguments[1] if len(arguments) == 2 else None
        if not (
            isinstance(site, SiteValue) and isinstance(distribution, DistributionValue)
        ):
            raise CallNotLowered(
                f"draws in `{owner.name}`{origin_line(term)} from a value that "
                f"is not a family construction at a named site"
            )
        name = f"{prefix}${site.label}"
        if name in self.sites or (name in self.occupied and name not in self.sites):
            raise CallNotLowered(
                f"draws the site {site.label!r} in `{owner.name}` more than once "
                f"in one call, so its draws would share the name {name!r}"
            )
        self.sites.add(name)
        self.occupied.add(name)
        index: TypeName | None = None
        if distribution.plate.batch:
            if self.plate is not None or len(distribution.plate.batch) != 1:
                raise CallNotLowered(
                    f"draws {site.label!r} in `{owner.name}` over several plate "
                    f"axes, which a call lowers only one of"
                )
            index = TypeName(name=self._axis(distribution.plate.batch[0].size, owner))
        elif self.plate is not None:
            index = TypeName(name=self.plate.axis)
        named: list[DrawArg] = []
        for parameter, value in distribution.arguments:
            read = replace_locals(value, env)
            named.append(
                DrawArgNamed(
                    parameter=parameter,
                    value=self.draw_argument(f"{name}_{parameter}", read),
                )
            )
        self.steps.append(
            SampleStep(
                vars=(name,),
                morphism=distribution.name,
                args=tuple(named),
                index=index,
                line=self.step.line,
                col=self.step.col,
            )
        )
        if self.plate is None:
            return Var(local=Local(name=name, type=binder.type))
        lifted = self.plate.lifted_type(binder.type)
        return Gather(
            value=Var(local=Local(name=name, type=lifted)),
            index=Var(local=self.plate.index),
            result_type=binder.type,
        )

    def _axis(self, size: object, owner: NamedComputation) -> str:
        """The object a plate axis of a draw runs over.

        Parameters
        ----------
        size : object
            The axis's extent, an index term.
        owner : NamedComputation
            The callee drawing, for the diagnostic.

        Returns
        -------
        str
            The first finite object of that cardinality.

        Raises
        ------
        CallNotLowered
            If the extent is not a literal, or no object has it.
        """
        if isinstance(size, IndexLiteral) and isinstance(size.value, int):
            for name, cardinality in self.cards.items():
                if cardinality == size.value:
                    return name
        raise CallNotLowered(
            f"draws in `{owner.name}` over an axis of extent {size!r}, which no "
            f"object of the module has, so the draw has no axis to run over"
        )

    def draw_argument(self, hint: str, value: Value) -> DrawArg:
        """A family argument as a draw argument, bound first unless atomic.

        Parameters
        ----------
        hint : str
            The preferred name of the binding.
        value : Value
            The argument, reading program names.

        Returns
        -------
        DrawArg
            A literal number, or the name holding the argument; on a
            plated lowering a per-item argument binds one entry per
            iteration.
        """
        if isinstance(value, LiteralValue) and isinstance(value.value, int | float):
            if not isinstance(value.value, bool):
                return DrawArgScalar(value=float(value.value))
        if isinstance(value, Var) and not self.per_item(value):
            return DrawArgName(text=value.local.name)
        if self.plate is not None and self.per_item(value):
            plate = self.plate
            element = _value_type(value)
            lifted = plate.lifted_type(element)
            name = self.floater.fresh(hint)
            self.let(
                name,
                Comprehension(
                    binder=plate.index,
                    extent=plate.size,
                    body=value,
                    result_type=lifted,
                ),
            )
            return DrawArgName(text=name)
        name = self.floater.fresh(hint)
        self.let(name, value)
        return DrawArgName(text=name)

    def score(self, weight: Value, prefix: str) -> None:
        """Lower a ``Score.add`` to a score step.

        A log density of a family at a value is the internal let call
        ``$log_prob(family, value, parameter, argument, ...)``; any other
        weight is its value. On a plated lowering the step adds the sum
        of every iteration's weight.

        Parameters
        ----------
        weight : Value
            The weight added, reading program names.
        prefix : str
            The name the callee's sites are named under.

        Raises
        ------
        CallNotLowered
            If the weight has no expression form.
        """
        name = self.floater.fresh(f"{prefix}$score")
        density = weight.value if isinstance(weight, WeightSum) else weight
        if isinstance(density, LogDensity):
            expression = self.log_density(density, name)
        elif isinstance(weight, WeightSum):
            expression = LetExprCall(
                func="sum", args=(self.expression(self.lifted(weight.value, name)),)
            )
        else:
            expression = self.expression(self.lifted(weight, name))
        if self.plate is not None:
            expression = LetExprCall(func="sum", args=(expression,))
        self.steps.append(
            ScoreStep(
                name=name, value=expression, line=self.step.line, col=self.step.col
            )
        )

    def lifted(self, value: Value, hint: str) -> Value:
        """A value read once per iteration on a plated lowering.

        Parameters
        ----------
        value : Value
            The value.
        hint : str
            The preferred name of a binding the value needs.

        Returns
        -------
        Value
            The value itself, or on a plated lowering the bound tensor
            of its values over the iterations.
        """
        if self.plate is None:
            return value
        element = _value_type(value)
        lifted = self.plate.lifted_type(element)
        name = self.floater.fresh(hint)
        self.let(
            name,
            Comprehension(
                binder=self.plate.index,
                extent=self.plate.size,
                body=value,
                result_type=lifted,
            ),
        )
        return Var(local=Local(name=name, type=lifted))

    def log_density(self, density: LogDensity, hint: str) -> LetExprNode:
        """A log density of a family at a value, as an internal let call.

        Parameters
        ----------
        density : LogDensity
            The log density.
        hint : str
            The preferred name of the bindings its arguments need.

        Returns
        -------
        LetExprNode
            ``$log_prob("Family", value, "parameter", argument, ...)``,
            entrywise over the iterations on a plated lowering.

        Raises
        ------
        CallNotLowered
            If the density is not of a family construction.
        """
        sampleable = density.sampleable
        if not isinstance(sampleable, DistributionValue):
            raise CallNotLowered(
                "scores the log density of a value that is not a family construction"
            )
        if sampleable.plate.batch and self.plate is not None:
            raise CallNotLowered(
                f"scores a plated {sampleable.name} density once per item, "
                f"which a call lowers over one axis only"
            )
        arguments: list[LetExprNode] = [
            LetExprString(value=sampleable.name),
            self.expression(self.lifted(density.value, f"{hint}_value")),
        ]
        for parameter, value in sampleable.arguments:
            arguments.append(LetExprString(value=parameter))
            arguments.append(self.expression(self.lifted(value, f"{hint}_{parameter}")))
        call = LetExprCall(func=LOG_PROB_CALL, args=tuple(arguments))
        if sampleable.plate.batch or sampleable.plate.event:
            return LetExprCall(func="sum", args=(call,))
        return call


def _value_type(value: Value) -> TypeExpr:
    """The type a value has, as far as a plated lowering needs it.

    Parameters
    ----------
    value : Value
        The value.

    Returns
    -------
    TypeExpr
        The value's recorded type; ``Real`` for a value that records
        none, which the lifting then treats as one entry per item.
    """
    for field in ("result_type", "type"):
        recorded = getattr(value, field, None)
        if recorded is not None and not isinstance(recorded, str):
            return recorded
    if isinstance(value, Var):
        return value.local.type
    return REAL


def _binder_type(body: Computation | None, name: str) -> TypeExpr | None:
    """The type a program's checked body gives one of its bindings.

    Parameters
    ----------
    body : Computation | None
        The body of the program's checked computation.
    name : str
        The binding's name.

    Returns
    -------
    TypeExpr | None
        The type of the first local of that name the body binds, or
        ``None`` when the body or the binding is absent.
    """
    if body is None:
        return None
    pending: list[Computation] = [body]
    while pending:
        node = pending.pop()
        if isinstance(node, Bind):
            if node.binder.name == name:
                return node.binder.type
            pending.append(node.first)
            pending.append(node.then)
        elif isinstance(node, If):
            pending.append(node.then)
            pending.append(node.otherwise)
        elif isinstance(node, Handle | NewInstance):
            pending.append(node.computation if isinstance(node, Handle) else node.body)
    return None


class CallLowering:
    """Lower the calls of one program body.

    Parameters
    ----------
    program : str
        The program's name.
    module : QiecModule
        The checked module.
    cards : dict[str, int]
        The cardinality of every finite object.
    occupied : set[str]
        Every name the body uses.
    """

    def __init__(
        self,
        program: str,
        module: QiecModule,
        cards: dict[str, int],
        occupied: set[str],
    ) -> None:
        self.program = program
        self.module = module
        self.cards = cards
        self.occupied = occupied
        self.computations = {item.name: item for item in module.computations}
        self.entries = {item.name: item for item in module.entries}
        entry = self.entries.get(program)
        self.checked_body: Computation | None = next(
            (
                item.body
                for item in module.computations
                if entry is not None and item.id == entry.computation
            ),
            None,
        )
        self.refusal: CompileError | None = None
        self.lambdas: dict[str, LetExprLambda] = {}

    def body(self, steps: tuple[ProgramStep, ...]) -> tuple[ProgramStep, ...]:
        """Lower every call a body makes, searching marginalization scopes.

        A call that cannot be lowered is kept, and the first such
        call's refusal is recorded.

        Parameters
        ----------
        steps : tuple[ProgramStep, ...]
            The body.

        Returns
        -------
        tuple[ProgramStep, ...]
            The body with every lowerable call replaced by its steps.
        """
        out: list[ProgramStep] = []
        for step in steps:
            if isinstance(step, MarginalizeStep) and step.scope:
                out.append(step.with_(scope=self.body(step.scope)))
                continue
            if isinstance(step, LetStep) and isinstance(step.value, LetExprLambda):
                self.lambdas[step.name] = step.value
            if not isinstance(step, CallStep):
                out.append(step)
                continue
            try:
                out.extend(self.call(step))
            except CallNotLowered as error:
                if self.refusal is None:
                    self.refusal = CompileError(
                        f"program {self.program!r} cannot be fit on the compiled "
                        f"runtime: `let {step.name} <- {step.call.callee}(...)` "
                        f"{error.reason}. The program still runs on the "
                        f"reference machine",
                        step.line,
                        step.col,
                    )
                out.append(step)
        return tuple(out)

    def call(self, step: CallStep) -> tuple[ProgramStep, ...]:
        """The steps one call lowers to.

        Parameters
        ----------
        step : CallStep
            The call.

        Returns
        -------
        tuple[ProgramStep, ...]
            The steps, the last binding the call's name.

        Raises
        ------
        CallNotLowered
            If the call cannot be lowered.
        """
        if step.call.callee == "traverse" and "traverse" not in self.computations:
            return self.traverse(step)
        if step.call.callee in self.entries:
            raise CallNotLowered(
                f"calls the program `{step.call.callee}` as a computation; draw "
                f"from it with `sample {step.name} <- {step.call.callee}(...)`, "
                f"which runs it in place"
            )
        callee = self.computations.get(step.call.callee)
        if callee is None:
            raise CallNotLowered(
                f"names `{step.call.callee}`, which is not a computation of the module"
            )
        if step.call.static_arguments or callee.telescope:
            raise CallNotLowered(
                f"passes static arguments to `{callee.name}`, which the "
                f"compiled runtime does not instantiate in a call"
            )
        if len(step.call.arguments) != len(callee.parameters):
            raise CallNotLowered(
                f"passes {len(step.call.arguments)} argument(s) to "
                f"`{callee.name}`, which takes {len(callee.parameters)}"
            )
        lowering = _Lowering(
            self.computations, self.cards, self.occupied, self.program, step, None
        )
        arguments: list[Value] = []
        for parameter, argument in zip(
            callee.parameters, step.call.arguments, strict=True
        ):
            if isinstance(argument, LetExprVar):
                name = argument.name
            else:
                name = _fresh(f"{step.name}_{parameter.name}", self.occupied)
                lowering.steps.append(
                    LetStep(name=name, value=argument, line=step.line, col=step.col)
                )
            arguments.append(Var(local=Local(name=name, type=parameter.type)))
        result = lowering.call(
            callee, tuple(arguments), callee.type.result, step.name, ()
        )
        lowering.let(step.name, result)
        return tuple(lowering.steps)

    def traverse(self, step: CallStep) -> tuple[ProgramStep, ...]:
        """The steps ``let ys <- traverse(xs, x -> f(x, ...))`` lowers to.

        The callee runs once per item of ``xs``, each run reading only its
        item and names bound before the traverse, so the runs are
        independent: a pure ``f`` maps over the items as a comprehension,
        and each site ``f`` draws is one plated draw over the axis ``xs``
        runs along, named ``ys$label``.

        Parameters
        ----------
        step : CallStep
            The traverse.

        Returns
        -------
        tuple[ProgramStep, ...]
            The steps, the last binding ``ys`` to the tensor of results.

        Raises
        ------
        CallNotLowered
            If the lambda does not call a computation, the traverse's
            extent is not an object of the module, or the callee cannot
            be lowered.
        """
        source, function = step.call.arguments
        if isinstance(function, LetExprVar):
            resolved = self.lambdas.get(function.name)
            if resolved is None:
                raise CallNotLowered(
                    f"traverses with `{function.name}`, which is not a lambda "
                    f"bound earlier in the program"
                )
            function = resolved
        if not isinstance(function, LetExprLambda) or not isinstance(
            function.body, LetExprCall
        ):
            raise CallNotLowered(
                "traverses with a function that does not call a computation"
            )
        callee = self.computations.get(function.body.func)
        if callee is None:
            raise CallNotLowered(
                f"traverses with `{function.body.func}`, which is not a "
                f"computation of the module"
            )
        if callee.telescope:
            raise CallNotLowered(
                f"passes static arguments to `{callee.name}`, which the "
                f"compiled runtime does not instantiate in a call"
            )
        result_type = _binder_type(self.checked_body, step.name)
        shape = tensor_shape(result_type) if result_type is not None else None
        if shape is None or not shape[1] or not isinstance(shape[1][0], IndexLiteral):
            raise CallNotLowered(
                "traverses a tensor whose leading extent the checked program "
                "does not fix"
            )
        extent = int(shape[1][0].value)
        axis = next((name for name, size in self.cards.items() if size == extent), None)
        if axis is None:
            raise CallNotLowered(
                f"traverses {extent} items, and no object of the module has "
                f"{extent} elements for the iterations to run along"
            )
        index = Local(name=_fresh(f"{step.name}_item", self.occupied), type=INT)
        plate = _Plate(axis, extent, index)
        lowering = _Lowering(
            self.computations, self.cards, self.occupied, self.program, step, plate
        )
        if isinstance(source, LetExprVar):
            source_name = source.name
        else:
            source_name = _fresh(f"{step.name}_source", self.occupied)
            lowering.steps.append(
                LetStep(name=source_name, value=source, line=step.line, col=step.col)
            )
        item = LetExprIndex(
            array=LetExprVar(name=source_name),
            indices=(LetExprVar(name=index.name),),
        )
        if len(function.body.args) != len(callee.parameters):
            raise CallNotLowered(
                f"passes {len(function.body.args)} argument(s) to "
                f"`{callee.name}`, which takes {len(callee.parameters)}"
            )
        arguments: list[Value] = []
        for parameter, argument in zip(
            callee.parameters, function.body.args, strict=True
        ):
            applied = substitute_let_expr(argument, {function.param: item})
            lifted = plate.lifted_type(parameter.type)
            name = _fresh(f"{step.name}_{parameter.name}", self.occupied)
            lowering.steps.append(
                LetStep(
                    name=name,
                    value=LetExprFactor(
                        binders=(
                            LetFactorBinder(var=index.name, index=TypeName(name=axis)),
                        ),
                        body=applied,
                    ),
                    line=step.line,
                    col=step.col,
                )
            )
            arguments.append(
                Gather(
                    value=Var(local=Local(name=name, type=lifted)),
                    index=Var(local=index),
                    result_type=parameter.type,
                )
            )
        result = lowering.call(
            callee, tuple(arguments), callee.type.result, step.name, ()
        )
        assert result_type is not None
        lowering.let(
            step.name,
            Comprehension(
                binder=index, extent=plate.size, body=result, result_type=result_type
            ),
        )
        return tuple(lowering.steps)


def lower_calls(
    program: str,
    steps: tuple[ProgramStep, ...],
    module: QiecModule,
    cards: dict[str, int],
    occupied: set[str],
) -> tuple[tuple[ProgramStep, ...], CompileError | None]:
    """Lower every call a program body makes to compiled runtime steps.

    Parameters
    ----------
    program : str
        The program's name.
    steps : tuple[ProgramStep, ...]
        Its body.
    module : QiecModule
        The checked module holding its callees.
    cards : dict[str, int]
        The cardinality of every finite object.
    occupied : set[str]
        Every name the body uses; each name the lowering binds joins it.

    Returns
    -------
    tuple[tuple[ProgramStep, ...], CompileError | None]
        The body with every lowerable call replaced by its steps, and
        the located refusal of the first call that cannot be lowered,
        which is kept in the body; ``None`` when every call lowers.
    """
    lowering = CallLowering(program, module, cards, occupied)
    lowered = lowering.body(steps)
    return lowered, lowering.refusal


__all__: list[str] = []
