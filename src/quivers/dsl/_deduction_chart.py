"""Compile nonlexical deductions to typed weighted-chart fixed points.

A round computes ``axioms + rules(previous)``. Keeping the previous
chart's keys, with zero weights before accumulation, gives a stable order
for comparing the entire chart. No proof-height cutoff is involved;
nonconvergent systems exhaust the caller's execution budget.
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
    TensorValue,
    Value,
    Var,
)
from quivers.qiec.kinds import NAT
from quivers.qiec.types import BOOL, INT, REAL, IndexLiteral, TypeApplication, TypeExpr

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

    def node(self, item: Value, weight: Value, tail: Value) -> ConstructorValue:
        """Build one item-weight entry followed by a chart."""
        return ConstructorValue(
            constructor=self.cons.id,
            static_arguments=(),
            fields=(item, weight, tail),
            result_type=self.type,
        )

    def call(self, suffix: str, *args: Value) -> Computation:
        """Call a registered chart helper."""
        return self.c.call(self.s.name(suffix), args)

    def zero(self) -> Value:
        """Build the semiring's additive identity."""
        if self.s.semiring == "Boolean":
            return self.boolean(False)
        if self.s.semiring == "Counting":
            return self.integer(0)
        return self.c.weight_of(
            Reduction(
                operator="logsumexp",
                value=TensorValue(
                    items=(),
                    result_type=tensor_type(REAL, (IndexLiteral(value=0, sort=NAT),)),
                ),
                result_type=REAL,
            )
        )

    def equal_weight(
        self, left: Value, right: Value, *, tolerance: bool = False
    ) -> Value:
        """Compare two weights, optionally using the declared tolerance."""
        c = self.c
        if self.s.semiring in ("Boolean", "Counting"):
            return c.primitive(
                "eq_bool" if self.s.semiring == "Boolean" else "eq_int", left, right
            )
        a, b = c.primitive("weight_value", left), c.primitive("weight_value", right)
        equal = c.primitive("eq_real", a, b)
        if not tolerance or not self.s.tolerance_cell[0]:
            return equal
        return c.primitive(
            "or",
            equal,
            c.primitive(
                "le_real",
                c.primitive("abs_real", c.primitive("sub_real", a, b)),
                LiteralValue(value=self.s.tolerance_cell[0], type=REAL),
            ),
        )

    def times(self, left: Value, right: Value) -> Value:
        """Multiply two semiring weights."""
        return self.c.primitive(
            {"Boolean": "and", "Counting": "mul_int"}.get(
                self.s.semiring, "add_weight"
            ),
            left,
            right,
        )

    def case(
        self,
        value: Value,
        result: TypeExpr,
        empty: Computation,
        node: Callable[[Value, Value, Value], Computation],
    ) -> Case:
        """Eliminate an empty chart or one item-weight entry."""
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
        """Add weights without evaluating log-sum-exp at negative infinity."""
        c = self.c
        if self.s.semiring in ("Boolean", "Counting"):
            return Return(
                value=c.primitive(
                    "or" if self.s.semiring == "Boolean" else "add_int", left, right
                )
            )
        a, b = c.primitive("weight_value", left), c.primitive("weight_value", right)
        combined = (
            c.primitive("max_real", a, b)
            if self.s.semiring == "Viterbi"
            else c.log_sum_exp([a, b])
        )
        return If(
            condition=self.equal_weight(left, self.zero()),
            then=Return(value=right),
            otherwise=If(
                condition=self.equal_weight(right, self.zero()),
                then=Return(value=left),
                otherwise=Return(value=c.weight_of(combined)),
            ),
        )

    def add(self, chart: Value, item: Value, weight: Value) -> Computation:
        """Merge one contribution into the entry for its item."""
        c = self.c

        def node(head: Value, current: Value, tail: Value) -> Computation:
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
                            value=self.node(head, Var(local=total), tail)
                        ),
                    ),
                    otherwise=c.bind(
                        "rest",
                        self.type,
                        self.call("chart_add", tail, item, weight),
                        lambda rest: Return(
                            value=self.node(head, current, Var(local=rest))
                        ),
                    ),
                ),
            )

        return If(
            condition=self.equal_weight(weight, self.zero()),
            then=Return(value=chart),
            otherwise=self.case(
                chart,
                self.type,
                Return(value=self.node(item, weight, self.empty())),
                node,
            ),
        )

    def clear(self, chart: Value) -> Computation:
        """Keep keys in order while clearing the round's contributions."""
        return self.case(
            chart,
            self.type,
            Return(value=self.empty()),
            lambda item, weight, tail: self.c.bind(
                "rest",
                self.type,
                self.call("chart_zero", tail),
                lambda rest: Return(
                    value=self.node(item, self.zero(), Var(local=rest))
                ),
            ),
        )

    def equal(self, left: Value, right: Value) -> Computation:
        """Compare support and all weights, not just the goal weight."""

        def node(item: Value, weight: Value, tail: Value) -> Computation:
            return self.case(
                right,
                BOOL,
                Return(value=self.boolean(False)),
                lambda other, value, rest: self.c.bind(
                    "same",
                    BOOL,
                    self.call("eq", item, other),
                    lambda same: If(
                        condition=self.c.primitive(
                            "and",
                            Var(local=same),
                            self.equal_weight(weight, value, tolerance=True),
                        ),
                        then=self.call("chart_equal", tail, rest),
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
                lambda item, weight, tail: Return(value=self.boolean(False)),
            ),
            node,
        )

    def goal(self, chart: Value) -> Computation:
        """Sum the weights of all items accepted by the start pattern."""
        return self.case(
            chart,
            self.s.carrier,
            Return(value=self.zero()),
            lambda item, weight, tail: self.c.bind(
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
        """Accumulate each input axiom exactly once per round."""
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
        *premises: Value,
    ) -> Computation:
        """Fold the old chart over each ordered premise tuple of a rule."""
        c, rule = self.c, self.s.rules[index]
        if position < len(rule.premises):
            return self.case(
                cursor,
                self.type,
                Return(value=accumulated),
                lambda item, weight, tail: If(
                    condition=self.equal_weight(weight, self.zero()),
                    then=self.call(
                        f"chart_rule{index}_{position}",
                        old,
                        tail,
                        accumulated,
                        product,
                        *premises,
                    ),
                    otherwise=c.bind(
                        "next",
                        self.type,
                        self.call(
                            f"chart_rule{index}_{position + 1}",
                            old,
                            old,
                            accumulated,
                            self.times(product, weight),
                            *premises,
                            item,
                        ),
                        lambda next_: self.call(
                            f"chart_rule{index}_{position}",
                            old,
                            tail,
                            Var(local=next_),
                            product,
                            *premises,
                        ),
                    ),
                ),
            )

        def matched(bindings: dict[str, Local]) -> Computation:
            conclusion = c.instantiate(rule.conclusion, bindings)

            def weighted(weight: Value) -> Computation:
                return self.call(
                    "chart_add", accumulated, conclusion, self.times(product, weight)
                )

            if self.s.normalized_cell[0]:
                result = c.production_weight(("rule", index), weighted)
            elif self.s.semiring in ("LogProb", "Viterbi") and (
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
                    self.c.weight_of(None),
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
        """Iterate until every reachable item's weight converges."""
        return self.c.bind(
            "current",
            self.type,
            self.call("chart_step", old, axioms, weights),
            lambda current: self.c.bind(
                "same",
                BOOL,
                self.call("chart_equal", old, Var(local=current)),
                lambda same: If(
                    condition=Var(local=same),
                    then=Return(value=Var(local=current)),
                    otherwise=self.call("derive", Var(local=current), axioms, weights),
                ),
            ),
        )

    def entry(self, axioms: Value, weights: Value) -> Computation:
        """Read the goal weight from the converged chart."""
        return self.c.bind(
            "chart",
            self.type,
            self.call("derive", self.empty(), axioms, weights),
            lambda chart: self.call("chart_goal", Var(local=chart)),
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
        chart.build("chart_add", ("chart", "item", "weight"), chart.add),
        chart.build("chart_zero", ("chart",), chart.clear),
        chart.build("chart_equal", ("left", "right"), chart.equal),
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
                        *(f"premise{k}" for k in range(position)),
                    ),
                    lambda *args, i=index, p=position: chart.rule(i, p, *args),
                )
            )
    return result


__all__ = []
