"""Compile nonlexical deductions to typed weighted-chart fixed points.

A round computes ``axioms + rules(previous)`` in the declared semiring:
every sum and product is the deduction's search and collector, which
carry the semiring as data, so nothing here depends on which semiring it
is. Keeping the previous chart's keys, with zero weights before
accumulation, gives a stable order for comparing the entire chart.

Beside its weight each entry counts the derivations of nonzero weight of
height at most the round, a structural number in the counting semiring.
The iteration stops when two rounds agree in every weight, exactly or
within the declared tolerance, or in every count. Equal counts mean no
derivation of the next height, hence none higher (a derivation's
subderivations have nonzero weight when it has, since the zero
annihilates), so the chart sums every derivation: a certified stop.
Agreeing weights certify only in exact arithmetic, where a fixed point
reached from below is the least one; in rounded arithmetic they are a
stop by a control, which the settlement refuses or reports. No
proof-height cutoff is involved; nonconvergent systems exhaust the
caller's execution budget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from quivers.qiec.canonical import tensor_type
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.identifiers import ConstructorId, FamilyId, StaticScopeId
from quivers.qiec.terms import (
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    Comprehension,
    ConstructorValue,
    Gather,
    Reduction,
    If,
    LiteralValue,
    Local,
    Return,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, INT, REAL, TypeApplication, TypeExpr

if TYPE_CHECKING:
    from quivers.dsl.deduction_elaboration import _Context, _System
    from quivers.dsl.qiec_lowering import _Elaborator
    from quivers.qiec.module import NamedComputation


def declare_chart(elaborator: _Elaborator, system: _System) -> None:
    """Declare a recursive list of item-weight entries.

    Parameters
    ----------
    elaborator : _Elaborator
        The module builder and declaration registry.
    system : _System
        The deduction whose item and weight types label entries.
    """
    name = system.name("Chart")
    identity = FamilyId.derive(elaborator.source.module_name, "family", name)
    ids = tuple(ConstructorId.derive(identity, tag) for tag in ("nil", "cons"))
    family = FamilyDecl(
        id=identity, name=name, parameters=(), indices=(), constructors=ids
    )
    chart = TypeApplication(constructor=family.type_constructor, arguments=())
    elaborator.families[name] = family
    elaborator.registry.register_family(family)
    for tag, constructor_id, fields in zip(
        ("nil", "cons"),
        ids,
        (
            (),
            (
                FieldDef(name="item", type=system.item),
                FieldDef(name="weight", type=system.carrier),
                FieldDef(name="count", type=INT),
                FieldDef(name="tail", type=chart),
            ),
        ),
        strict=True,
    ):
        constructor = ConstructorDecl(
            id=constructor_id,
            family=identity,
            name=system.name(f"chart_{tag}"),
            telescope=(),
            fields=fields,
            result_indices=(),
        )
        elaborator.constructors[constructor.name] = constructor
        elaborator.registry.register_constructor(constructor)


class _Chart:
    """Term builders for a single nonlexical weighted chart.

    Parameters
    ----------
    context : _Context
        The deduction's checked term builders and registered signatures.
    """

    def __init__(self, context: _Context) -> None:
        self.c = context
        self.s = context.system
        self.type = TypeApplication(
            constructor=context.elaborator.families[
                self.s.name("Chart")
            ].type_constructor,
            arguments=(),
        )
        self.nil = context.elaborator.constructors[self.s.name("chart_nil")]
        self.cons = context.elaborator.constructors[self.s.name("chart_cons")]

    def integer(self, value: int) -> LiteralValue:
        """Build an integer literal."""
        return LiteralValue(value=value, type=INT)

    def boolean(self, value: bool) -> LiteralValue:
        """Build a Boolean literal."""
        return LiteralValue(value=value, type=BOOL)

    def empty(self) -> ConstructorValue:
        """Build an empty chart."""
        return ConstructorValue(
            constructor=self.nil.id,
            static_arguments=(),
            fields=(),
            result_type=self.type,
        )

    def node(
        self, item: Value, weight: Value, count: Value, tail: Value
    ) -> ConstructorValue:
        """Build one item-weight-count entry followed by a chart."""
        return ConstructorValue(
            constructor=self.cons.id,
            static_arguments=(),
            fields=(item, weight, count, tail),
            result_type=self.type,
        )

    def call(self, suffix: str, *args: Value) -> Computation:
        """Call a registered chart helper."""
        return self.c.call(self.s.name(suffix), args)

    def zero(self) -> Value:
        """Build the semiring's additive identity."""
        return self.c.element(self.s.semiring.zero())

    def unit(self) -> Value:
        """Build the semiring's multiplicative identity."""
        return self.c.element(self.s.semiring.unit())

    def is_zero(self, weight: Value) -> Value:
        """Whether a weight is the semiring's zero."""
        return self.c.same_weight(weight, self.zero())

    def times(
        self, left: Value, right: Value, then: Callable[[Value], Computation]
    ) -> Computation:
        """Multiply two semiring weights and continue with the product."""
        return self.c.bind(
            "product",
            self.s.carrier,
            self.call("chart_times", left, right),
            lambda product: then(Var(local=product)),
        )

    def case(
        self,
        value: Value,
        result: TypeExpr,
        empty: Computation,
        node: Callable[[Value, Value, Value, Value], Computation],
    ) -> Case:
        """Eliminate an empty chart or one item-weight-count entry."""
        fields = tuple(
            self.s.local(field.name, field.type) for field in self.cons.fields
        )
        bodies = (
            (self.nil, (), empty),
            (self.cons, fields, node(*(Var(local=f) for f in fields))),
        )
        return Case(
            scrutinee=value,
            motive=CaseMotive(indices=(), result_type=result),
            branches=tuple(
                CaseBranch(
                    constructor=con.id,
                    static_arguments=(),
                    fields=fs,
                    body=body,
                    scope=StaticScopeId.derive(
                        self.c.module,
                        self.c.path,
                        "chart",
                        *(f.name for f in fields),
                        con.id,
                    ),
                )
                for con, fs, body in bodies
            ),
        )

    def build(
        self, suffix: str, names: tuple[str, ...], body: Callable[..., Computation]
    ) -> NamedComputation:
        """Check one helper against its predeclared signature."""
        signature = self.c.signatures[self.s.name(suffix)]
        locals_ = tuple(
            Local(name=name, type=type_)
            for name, type_ in zip(names, self.c.inputs(signature), strict=True)
        )
        return self.c.computation(
            signature, locals_, body(*(Var(local=x) for x in locals_))
        )

    def plus(self, left: Value, right: Value) -> Computation:
        """Add two weights by the deduction's search."""
        return self.c.semiring_sum((left, right))

    def product(self, left: Value, right: Value) -> Computation:
        """Multiply two weights by the deduction's collector."""
        return self.c.semiring_product((left, right))

    def add(
        self, chart: Value, item: Value, weight: Value, count: Value
    ) -> Computation:
        """Merge one contribution into the entry for its item."""
        c = self.c

        def node(
            head: Value, current: Value, counted: Value, tail: Value
        ) -> Computation:
            return c.bind(
                "same",
                BOOL,
                self.call("eq", head, item),
                lambda same: If(
                    condition=Var(local=same),
                    then=c.bind(
                        "sum",
                        self.s.carrier,
                        self.call("chart_plus", current, weight),
                        lambda total: Return(
                            value=self.node(
                                head,
                                Var(local=total),
                                c.primitive("add_int", counted, count),
                                tail,
                            )
                        ),
                    ),
                    otherwise=c.bind(
                        "rest",
                        self.type,
                        self.call("chart_add", tail, item, weight, count),
                        lambda rest: Return(
                            value=self.node(head, current, counted, Var(local=rest))
                        ),
                    ),
                ),
            )

        return If(
            condition=self.is_zero(weight),
            then=Return(value=chart),
            otherwise=self.case(
                chart,
                self.type,
                Return(value=self.node(item, weight, count, self.empty())),
                node,
            ),
        )

    def clear(self, chart: Value) -> Computation:
        """Keep keys in order while clearing the round's contributions."""
        return self.case(
            chart,
            self.type,
            Return(value=self.empty()),
            lambda item, weight, count, tail: self.c.bind(
                "rest",
                self.type,
                self.call("chart_zero", tail),
                lambda rest: Return(
                    value=self.node(item, self.zero(), self.integer(0), Var(local=rest))
                ),
            ),
        )

    def compare(
        self,
        suffix: str,
        left: Value,
        right: Value,
        same: Callable[[Value, Value, Value, Value], Value],
    ) -> Computation:
        """Whether two charts have one support and every entry agrees.

        ``suffix`` names the helper being built, which recurses on the
        tails; ``same`` receives the two weights and the two counts of an
        item.
        """

        def node(item: Value, weight: Value, count: Value, tail: Value) -> Computation:
            return self.case(
                right,
                BOOL,
                Return(value=self.boolean(False)),
                lambda other, value, counted, rest: self.c.bind(
                    "same",
                    BOOL,
                    self.call("eq", item, other),
                    lambda same_item: If(
                        condition=self.c.primitive(
                            "and",
                            Var(local=same_item),
                            same(weight, value, count, counted),
                        ),
                        then=self.call(suffix, tail, rest),
                        otherwise=Return(value=self.boolean(False)),
                    ),
                ),
            )

        return self.case(
            left,
            BOOL,
            self.case(
                right,
                BOOL,
                Return(value=self.boolean(True)),
                lambda item, weight, count, tail: Return(value=self.boolean(False)),
            ),
            node,
        )

    def weights_agree(
        self, weight: Value, value: Value, _count: Value, _counted: Value
    ) -> Value:
        """Whether two weights agree, exactly or within the declared tolerance."""
        return self.c.within_tolerance(weight, value)

    def counts_agree(
        self, _weight: Value, _value: Value, count: Value, counted: Value
    ) -> Value:
        """Whether two counts of derivations are equal."""
        return self.c.primitive("eq_int", count, counted)

    def equal(self, left: Value, right: Value) -> Computation:
        """Compare support and all weights, not just the goal weight."""
        return self.compare("chart_equal", left, right, self.weights_agree)

    def settled(self, left: Value, right: Value) -> Computation:
        """Compare support and every count of derivations."""
        return self.compare("chart_settled", left, right, self.counts_agree)

    def residual(self, left: Value, right: Value) -> Computation:
        """The largest distance of two weights of one item in two charts."""
        c = self.c
        return self.case(
            left,
            REAL,
            Return(value=LiteralValue(value=0.0, type=REAL)),
            lambda item, weight, count, tail: self.case(
                right,
                REAL,
                Return(value=LiteralValue(value=0.0, type=REAL)),
                lambda other, value, counted, rest: c.bind(
                    "rest",
                    REAL,
                    self.call("chart_residual", tail, rest),
                    lambda farthest: Return(
                        value=c.primitive(
                            "max_real", c.residual(weight, value), Var(local=farthest)
                        )
                    ),
                ),
            ),
        )

    def goal(self, chart: Value) -> Computation:
        """Sum the weights of all items accepted by the start pattern."""
        return self.case(
            chart,
            self.s.carrier,
            Return(value=self.zero()),
            lambda item, weight, count, tail: self.c.bind(
                "rest",
                self.s.carrier,
                self.call("chart_goal", tail),
                lambda rest: self.c.bind(
                    "accept",
                    BOOL,
                    self.call("goal", item),
                    lambda accept: If(
                        condition=Var(local=accept),
                        then=self.call("chart_plus", weight, Var(local=rest)),
                        otherwise=Return(value=Var(local=rest)),
                    ),
                ),
            ),
        )

    def axioms(
        self, position: Value, chart: Value, axioms: Value, weights: Value
    ) -> Computation:
        """Accumulate each input axiom exactly once per round, as one derivation."""
        length = self.length()
        return If(
            condition=self.c.primitive("lt_int", position, length),
            then=self.c.bind(
                "next",
                self.type,
                self.call(
                    "chart_add",
                    chart,
                    Gather(value=axioms, index=position, result_type=self.s.item),
                    Gather(value=weights, index=position, result_type=self.s.carrier),
                    self.integer(1),
                ),
                lambda next_: self.call(
                    "chart_axioms",
                    self.c.primitive("add_int", position, self.integer(1)),
                    Var(local=next_),
                    axioms,
                    weights,
                ),
            ),
            otherwise=Return(value=chart),
        )

    def length(self) -> Value:
        """Reify the input tensor extent as a scalar integer."""
        return Reduction(
            operator="sum",
            value=Comprehension(
                binder=self.s.local("k", INT),
                extent=self.c.extent,
                body=self.integer(1),
                result_type=tensor_type(INT, (self.c.extent,)),
            ),
            result_type=INT,
        )

    def rule(
        self,
        index: int,
        position: int,
        old: Value,
        cursor: Value,
        accumulated: Value,
        product: Value,
        count: Value,
        *premises: Value,
    ) -> Computation:
        """Fold the old chart over each ordered premise tuple of a rule.

        ``product`` is the product of the chosen premises' weights and
        ``count`` the product of their counts, the number of derivations
        of the premise tuple.
        """
        c, rule = self.c, self.s.rules[index]
        if position < len(rule.premises):
            return self.case(
                cursor,
                self.type,
                Return(value=accumulated),
                lambda item, weight, counted, tail: If(
                    condition=self.is_zero(weight),
                    then=self.call(
                        f"chart_rule{index}_{position}",
                        old,
                        tail,
                        accumulated,
                        product,
                        count,
                        *premises,
                    ),
                    otherwise=self.times(
                        product,
                        weight,
                        lambda extended: c.bind(
                            "next",
                            self.type,
                            self.call(
                                f"chart_rule{index}_{position + 1}",
                                old,
                                old,
                                accumulated,
                                extended,
                                c.primitive("mul_int", count, counted),
                                *premises,
                                item,
                            ),
                            lambda next_: self.call(
                                f"chart_rule{index}_{position}",
                                old,
                                tail,
                                Var(local=next_),
                                product,
                                count,
                                *premises,
                            ),
                        ),
                    ),
                ),
            )

        def matched(bindings: dict[str, Local]) -> Computation:
            conclusion = c.instantiate(rule.conclusion, bindings)

            def weighted(weight: Value) -> Computation:
                return self.times(
                    product,
                    weight,
                    lambda total: self.call(
                        "chart_add", accumulated, conclusion, total, count
                    ),
                )

            if self.s.normalized_cell[0]:
                result = c.production_weight(("rule", index), weighted)
            elif self.s.semiring.log_weights and (
                rule.learnable or rule.parent is not None
            ):
                result = c.rule_weight(rule, index, bindings, weighted)
            else:
                result = weighted(c.weight_of(None))
            bound = self.s.depth_cell[0]
            if bound is None:
                return result
            return c.bind(
                "depth",
                INT,
                self.call("depth", conclusion),
                lambda depth: If(
                    condition=c.primitive(
                        "le_int", Var(local=depth), self.integer(bound)
                    ),
                    then=result,
                    otherwise=Return(value=accumulated),
                ),
            )

        def match(position_: int, bindings: dict[str, Local]) -> Computation:
            if position_ == len(premises):
                return matched(bindings)
            return c.bind(
                "premise",
                self.s.item,
                Return(value=premises[position_]),
                lambda local: c.match(
                    local,
                    rule.premises[position_],
                    bindings,
                    self.type,
                    lambda extended: match(position_ + 1, extended),
                    "chart",
                    index,
                    position_,
                    failed=lambda: Return(value=accumulated),
                ),
            )

        return match(0, {})

    def step(self, old: Value, axioms: Value, weights: Value) -> Computation:
        """Compute axioms plus the rule contributions from the old chart."""

        def rules(index: int, accumulated: Value) -> Computation:
            if index == len(self.s.rules):
                return Return(value=accumulated)
            return self.c.bind(
                "next",
                self.type,
                self.call(
                    f"chart_rule{index}_0",
                    old,
                    old,
                    accumulated,
                    self.unit(),
                    self.integer(1),
                ),
                lambda next_: rules(index + 1, Var(local=next_)),
            )

        return self.c.bind(
            "cleared",
            self.type,
            self.call("chart_zero", old),
            lambda cleared: self.c.bind(
                "base",
                self.type,
                self.call(
                    "chart_axioms", self.integer(0), Var(local=cleared), axioms, weights
                ),
                lambda base: rules(0, Var(local=base)),
            ),
        )

    def derive(self, old: Value, axioms: Value, weights: Value) -> Computation:
        """Iterate until two rounds agree, and stop at the goal's weight."""
        c = self.c

        def judged(current: Value, agree: Value, settled: Value) -> Computation:
            return c.bind(
                "goal",
                self.s.carrier,
                self.call("chart_goal", current),
                lambda goal: c.bind(
                    "residual",
                    REAL,
                    self.call("chart_residual", old, current),
                    lambda residual: c.stop(
                        Var(local=goal),
                        c.certified(agree, settled),
                        Var(local=residual),
                    ),
                ),
            )

        return c.bind(
            "current",
            self.type,
            self.call("chart_step", old, axioms, weights),
            lambda current: c.bind(
                "agree",
                BOOL,
                self.call("chart_equal", old, Var(local=current)),
                lambda agree: c.bind(
                    "settled",
                    BOOL,
                    self.call("chart_settled", old, Var(local=current)),
                    lambda settled: If(
                        condition=c.primitive(
                            "or", Var(local=agree), Var(local=settled)
                        ),
                        then=judged(
                            Var(local=current), Var(local=agree), Var(local=settled)
                        ),
                        otherwise=self.call(
                            "derive", Var(local=current), axioms, weights
                        ),
                    ),
                ),
            ),
        )

    def entry(self, axioms: Value, weights: Value) -> Computation:
        """Settle the goal weight of the iterated chart."""
        handlers = self.s.handlers_cell[0]
        assert handlers is not None
        return self.c.settled(
            self.call("derive", self.empty(), axioms, weights), handlers
        )


def elaborate_chart(context: _Context) -> list[NamedComputation]:
    """Build and check the chart helpers, fixed point, and public entry.

    Parameters
    ----------
    context : _Context
        The deduction's term builders.

    Returns
    -------
    list[NamedComputation]
        Checked computations using only ordinary kernel terms.
    """
    chart = _Chart(context)
    result = [
        chart.build("chart_plus", ("left", "right"), chart.plus),
        chart.build("chart_times", ("left", "right"), chart.product),
        chart.build("chart_add", ("chart", "item", "weight", "count"), chart.add),
        chart.build("chart_zero", ("chart",), chart.clear),
        chart.build("chart_equal", ("left", "right"), chart.equal),
        chart.build("chart_settled", ("left", "right"), chart.settled),
        chart.build("chart_residual", ("left", "right"), chart.residual),
        chart.build("chart_goal", ("chart",), chart.goal),
        chart.build(
            "chart_axioms", ("position", "chart", "axioms", "weights"), chart.axioms
        ),
        chart.build("chart_step", ("old", "axioms", "weights"), chart.step),
        chart.build("derive", ("old", "axioms", "weights"), chart.derive),
        chart.build("run", ("axioms", "weights"), chart.entry),
    ]
    if context.table is not None:
        result = result[-3:]
    for index, rule in enumerate(context.system.rules):
        for position in range(len(rule.premises) + 1):
            result.append(
                chart.build(
                    f"chart_rule{index}_{position}",
                    (
                        "old",
                        "cursor",
                        "accumulated",
                        "product",
                        "count",
                        *(f"premise{k}" for k in range(position)),
                    ),
                    lambda *args, i=index, p=position: chart.rule(i, p, *args),
                )
            )
    return result


__all__ = []
