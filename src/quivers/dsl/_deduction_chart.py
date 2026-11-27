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

from quivers.dsl.compiler._deduction_positions import SPAN_SYMBOL
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
from quivers.qiec.types import BOOL, INT, REAL, STRING, TypeApplication, TypeExpr

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
        self.lexical = bool(self.s.entries)
        self.reader = "chart_lexicon" if self.lexical else "chart_axioms"
        self.inputs: tuple[str, ...] = (
            ("tokens",) if self.lexical else ("axioms", "weights")
        )

    def read(self, position: Value, chart: Value, *inputs: Value) -> Computation:
        """Add the axioms from a position on to a round's chart.

        Parameters
        ----------
        position : Value
            The first input not yet added.
        chart : Value
            The chart so far.
        *inputs : Value
            The axioms and their weights, or the sentence.

        Returns
        -------
        Computation
            The chart with every remaining axiom added.
        """
        if self.lexical:
            return self.lexicon(position, chart, *inputs)
        return self.axioms(position, chart, *inputs)

    def lexicon(self, position: Value, chart: Value, tokens: Value) -> Computation:
        """Add the lexical axioms of the tokens from a position on.

        Each entry whose word is the token at a position is one axiom: the
        span it yields there, with the entry's weight and the count one.

        Parameters
        ----------
        position : Value
            The first token not yet read.
        chart : Value
            The chart so far.
        tokens : Value
            The sentence.

        Returns
        -------
        Computation
            The chart with the axioms of every remaining token added.
        """
        c = self.c
        token = Gather(value=tokens, index=position, result_type=STRING)

        def entries(index: int, current: Value) -> Computation:
            """Add the entries from one index on that match the token.

            Parameters
            ----------
            index : int
                The entry tested here.
            current : Value
                The chart so far.

            Returns
            -------
            Computation
                The chart with the matching entries added, then the next
                token read.
            """
            if index == len(self.s.entries):
                return self.call(
                    "chart_lexicon",
                    c.primitive("add_int", position, self.integer(1)),
                    current,
                    tokens,
                )
            entry = self.s.entries[index]

            def added(weight: Value) -> Computation:
                """Add the entry's span at its weight, then go on.

                Parameters
                ----------
                weight : Value
                    The entry's weight.

                Returns
                -------
                Computation
                    The rest of the reading.
                """
                return c.bind(
                    "added",
                    self.type,
                    self.call(
                        "chart_add",
                        current,
                        c.lexical_item(index, position),
                        weight,
                        self.integer(1),
                    ),
                    lambda next_: entries(index + 1, Var(local=next_)),
                )

            return If(
                condition=c.primitive(
                    "eq_string", token, LiteralValue(value=entry.word, type=STRING)
                ),
                then=c.entry_weight(index, entry, added),
                otherwise=entries(index + 1, current),
            )

        return If(
            condition=c.primitive("lt_int", position, self.length()),
            then=entries(0, chart),
            otherwise=Return(value=chart),
        )

    def covers(self, item: Value) -> Computation:
        """Whether a goal item covers the whole input.

        A span of a lexical system has to cover the sentence; any other
        item, and any item of a system without a lexicon, covers it.

        Parameters
        ----------
        item : Value
            An item the start pattern accepts.

        Returns
        -------
        Computation
            A ``Bool``.
        """
        if not self.lexical:
            return Return(value=self.boolean(True))
        c = self.c
        return c.case(
            item,
            BOOL,
            {
                SPAN_SYMBOL: lambda fields: Return(
                    value=c.primitive(
                        "and",
                        c.primitive("eq_int", Var(local=fields[0]), self.integer(0)),
                        c.primitive("eq_int", Var(local=fields[1]), self.length()),
                    )
                )
            },
            lambda: Return(value=self.boolean(True)),
            "chart",
            "goal",
            "extent",
        )

    def integer(self, value: int) -> LiteralValue:
        """Build an integer literal.

        Parameters
        ----------
        value : int
            The value.

        Returns
        -------
        LiteralValue
            The literal.
        """
        return LiteralValue(value=value, type=INT)

    def boolean(self, value: bool) -> LiteralValue:
        """Build a Boolean literal.

        Parameters
        ----------
        value : bool
            The value.

        Returns
        -------
        LiteralValue
            The literal.
        """
        return LiteralValue(value=value, type=BOOL)

    def empty(self) -> ConstructorValue:
        """Build an empty chart.

        Returns
        -------
        ConstructorValue
            The chart with no entry.
        """
        return ConstructorValue(
            constructor=self.nil.id,
            static_arguments=(),
            fields=(),
            result_type=self.type,
        )

    def node(
        self, item: Value, weight: Value, count: Value, tail: Value
    ) -> ConstructorValue:
        """Build one item-weight-count entry followed by a chart.

        Parameters
        ----------
        item : Value
            An item.
        weight : Value
            A weight of the carrier.
        count : Value
            A count of derivations.
        tail : Value
            The rest of the chart.

        Returns
        -------
        ConstructorValue
            The chart.
        """
        return ConstructorValue(
            constructor=self.cons.id,
            static_arguments=(),
            fields=(item, weight, count, tail),
            result_type=self.type,
        )

    def call(self, suffix: str, *args: Value) -> Computation:
        """Call a registered chart helper.

        Parameters
        ----------
        suffix : str
            The helper's name after the deduction's.
        *args : Value
            Its arguments.

        Returns
        -------
        Computation
            The call.
        """
        return self.c.call(self.s.name(suffix), args)

    def zero(self) -> Value:
        """Build the semiring's additive identity.

        Returns
        -------
        Value
            The zero.
        """
        return self.c.element(self.s.semiring.zero())

    def unit(self) -> Value:
        """Build the semiring's multiplicative identity.

        Returns
        -------
        Value
            The unit.
        """
        return self.c.element(self.s.semiring.unit())

    def is_zero(self, weight: Value) -> Value:
        """Whether a weight is the semiring's zero.

        Parameters
        ----------
        weight : Value
            A weight of the carrier.

        Returns
        -------
        Value
            A ``Bool``.
        """
        return self.c.same_weight(weight, self.zero())

    def times(
        self, left: Value, right: Value, then: Callable[[Value], Computation]
    ) -> Computation:
        """Multiply two semiring weights and continue with the product.

        Parameters
        ----------
        left : Value
            The left operand.
        right : Value
            The right operand.
        then : Callable[[Value], Computation]
            What to run with the result.

        Returns
        -------
        Computation
            The product, then the continuation.
        """
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
        """Eliminate an empty chart or one item-weight-count entry.

        Parameters
        ----------
        value : Value
            The chart eliminated.
        result : TypeExpr
            The type the elimination produces.
        empty : Computation
            The body for the empty chart.
        node : Callable[[Value, Value, Value, Value], Computation]
            The body for an entry, given its fields.

        Returns
        -------
        Case
            The elimination.
        """
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
        """Check one helper against its predeclared signature.

        Parameters
        ----------
        suffix : str
            The helper's name after the deduction's.
        names : tuple[str, ...]
            The parameter names.
        body : Callable[..., Computation]
            Builds the body from the parameters.

        Returns
        -------
        NamedComputation
            The checked helper.
        """
        signature = self.c.signatures[self.s.name(suffix)]
        locals_ = tuple(
            Local(name=name, type=type_)
            for name, type_ in zip(names, self.c.inputs(signature), strict=True)
        )
        return self.c.computation(
            signature, locals_, body(*(Var(local=x) for x in locals_))
        )

    def plus(self, left: Value, right: Value) -> Computation:
        """Add two weights by the deduction's search.

        Parameters
        ----------
        left : Value
            The left operand.
        right : Value
            The right operand.

        Returns
        -------
        Computation
            The sum.
        """
        return self.c.semiring_sum((left, right))

    def product(self, left: Value, right: Value) -> Computation:
        """Multiply two weights by the deduction's collector.

        Parameters
        ----------
        left : Value
            The left operand.
        right : Value
            The right operand.

        Returns
        -------
        Computation
            The product.
        """
        return self.c.semiring_product((left, right))

    def add(
        self, chart: Value, item: Value, weight: Value, count: Value
    ) -> Computation:
        """Merge one contribution into the entry for its item.

        Parameters
        ----------
        chart : Value
            A chart.
        item : Value
            An item.
        weight : Value
            A weight of the carrier.
        count : Value
            A count of derivations.

        Returns
        -------
        Computation
            The chart with the contribution merged; unchanged by a zero weight.
        """
        c = self.c

        def node(
            head: Value, current: Value, counted: Value, tail: Value
        ) -> Computation:
            """Merge into or rebuild one entry.

            Parameters
            ----------
            head : Value
                The entry's item.
            current : Value
                The entry's weight.
            counted : Value
                The entry's count.
            tail : Value
                The rest of the chart.

            Returns
            -------
            ConstructorValue
                The chart.
            """
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
        """Keep keys in order while clearing the round's contributions.

        Parameters
        ----------
        chart : Value
            A chart.

        Returns
        -------
        Computation
            The chart with zero weights and counts.
        """
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

        Parameters
        ----------
        suffix : str
            The helper's name after the deduction's.
        left : Value
            The left operand.
        right : Value
            The right operand.
        same : Callable[[Value, Value, Value, Value], Value]
            Whether two entries agree, given their weights and counts.

        Returns
        -------
        Computation
            A ``Bool``.
        """

        def node(item: Value, weight: Value, count: Value, tail: Value) -> Computation:
            """Merge into or rebuild one entry.

            Parameters
            ----------
            item : Value
                An item.
            weight : Value
                A weight of the carrier.
            count : Value
                A count of derivations.
            tail : Value
                The rest of the chart.

            Returns
            -------
            ConstructorValue
                The chart.
            """
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
        """Whether two weights agree, exactly or within the declared tolerance.

        Parameters
        ----------
        weight : Value
            The first weight.
        value : Value
            The second weight.
        _count : Value
            The first count, unused.
        _counted : Value
            The second count, unused.

        Returns
        -------
        Value
            A ``Bool``.
        """
        return self.c.within_tolerance(weight, value)

    def counts_agree(
        self, _weight: Value, _value: Value, count: Value, counted: Value
    ) -> Value:
        """Whether two counts of derivations are equal.

        Parameters
        ----------
        _weight : Value
            The first weight, unused.
        _value : Value
            The second weight, unused.
        count : Value
            A count of derivations.
        counted : Value
            The entry's count.

        Returns
        -------
        Value
            A ``Bool``.
        """
        return self.c.primitive("eq_int", count, counted)

    def equal(self, left: Value, right: Value) -> Computation:
        """Compare support and all weights, not just the goal weight.

        Parameters
        ----------
        left : Value
            The left operand.
        right : Value
            The right operand.

        Returns
        -------
        Computation
            A ``Bool``.
        """
        return self.compare("chart_equal", left, right, self.weights_agree)

    def settled(self, left: Value, right: Value) -> Computation:
        """Compare support and every count of derivations.

        Parameters
        ----------
        left : Value
            The left operand.
        right : Value
            The right operand.

        Returns
        -------
        Computation
            A ``Bool``.
        """
        return self.compare("chart_settled", left, right, self.counts_agree)

    def residual(self, left: Value, right: Value) -> Computation:
        """The largest distance of two weights of one item in two charts.

        Parameters
        ----------
        left : Value
            The left operand.
        right : Value
            The right operand.

        Returns
        -------
        Computation
            A ``Real``.
        """
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
        """Sum the weights of the items the start pattern accepts that cover the input.

        Parameters
        ----------
        chart : Value
            A chart.

        Returns
        -------
        Computation
            The goal's weight.
        """
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
                    lambda accept: self.c.bind(
                        "covers",
                        BOOL,
                        self.covers(item),
                        lambda covers: If(
                            condition=self.c.primitive(
                                "and", Var(local=accept), Var(local=covers)
                            ),
                            then=self.call("chart_plus", weight, Var(local=rest)),
                            otherwise=Return(value=Var(local=rest)),
                        ),
                    ),
                ),
            ),
        )

    def axioms(
        self, position: Value, chart: Value, axioms: Value, weights: Value
    ) -> Computation:
        """Accumulate each input axiom exactly once per round, as one derivation.

        Parameters
        ----------
        position : Value
            The index of the next axiom.
        chart : Value
            A chart.
        axioms : Value
            The input axioms.
        weights : Value
            Their weights.

        Returns
        -------
        Computation
            The chart with the remaining axioms added.
        """
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
        """Reify the input tensor extent as a scalar integer.

        Returns
        -------
        Value
            The number of axioms, an ``Int``.
        """
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

        Parameters
        ----------
        index : int
            The rule's index.
        position : int
            The premise being chosen.
        old : Value
            The chart of the previous round.
        cursor : Value
            The entries still to try at this premise.
        accumulated : Value
            The chart of this round so far.
        product : Value
            The product of the chosen premises' weights.
        count : Value
            A count of derivations.
        *premises : Value
            The chosen premise items.

        Returns
        -------
        Computation
            The round's chart with the rule's contributions added.
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
            """Add the conclusion of a matched premise tuple.

            Parameters
            ----------
            bindings : dict[str, Local]
                The pattern variables bound so far.

            Returns
            -------
            Computation
                The chart with the conclusion added.
            """
            conclusion = c.instantiate(rule.conclusion, bindings)

            def weighted(weight: Value) -> Computation:
                """Add the conclusion at a rule weight.

                Parameters
                ----------
                weight : Value
                    A weight of the carrier.

                Returns
                -------
                Computation
                    The chart with the conclusion added.
                """
                return self.times(
                    product,
                    weight,
                    lambda total: self.call(
                        "chart_add", accumulated, conclusion, total, count
                    ),
                )

            if self.s.normalized_cell[0]:
                result = c.production_weight(("rule", rule.presented.index), weighted)
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
            """Match the premises from left to right.

            Parameters
            ----------
            position_ : int
                The next premise.
            bindings : dict[str, Local]
                The pattern variables bound so far.

            Returns
            -------
            Computation
                The matches, innermost the conclusion.
            """
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

    def step(self, old: Value, *inputs: Value) -> Computation:
        """Compute axioms plus the rule contributions from the old chart.

        Parameters
        ----------
        old : Value
            The chart of the previous round.
        *inputs : Value
            The axioms and their weights, or the sentence.

        Returns
        -------
        Computation
            The next round's chart.
        """

        def rules(index: int, accumulated: Value) -> Computation:
            """Fire the rules from one index on.

            Parameters
            ----------
            index : int
                The rule's index.
            accumulated : Value
                The chart of this round so far.

            Returns
            -------
            Computation
                The chart with every rule's contributions.
            """
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
                self.call(self.reader, self.integer(0), Var(local=cleared), *inputs),
                lambda base: rules(0, Var(local=base)),
            ),
        )

    def derive(self, old: Value, round_: Value, *inputs: Value) -> Computation:
        """Iterate until two rounds agree, and stop at the goal's weight.

        Equal counts stop the iteration with a certificate. Agreeing
        weights stop it in exact arithmetic, where they certify; in
        rounded arithmetic they stop it only once a cycle is shown: a
        count that changes in round ``k`` belongs to a derivation of height
        ``k``, whose items are ``k`` distinct items of the chart when no
        item derives itself, so a change after more rounds than the chart
        has items shows a cycle, under which counts never settle. Before
        that the iteration goes on, so an acyclic system always settles.

        Parameters
        ----------
        old : Value
            The chart of the previous round.
        round_ : Value
            The number of the round about to be computed, from one.
        *inputs : Value
            The axioms and their weights, or the sentence.

        Returns
        -------
        Computation
            The stop at the goal's weight.
        """
        c = self.c

        def judged(current: Value, agree: Value, settled: Value) -> Computation:
            """Stop at the current chart's goal weight.

            Parameters
            ----------
            current : Value
                The entry's weight.
            agree : Value
                Whether the rounds agree in every weight.
            settled : Value
                Whether the rounds agree in every count.

            Returns
            -------
            Computation
                The stop.
            """
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

        def stops(current: Value, agree: Value, settled: Value) -> Computation:
            """Whether the iteration stops after this round.

            Parameters
            ----------
            current : Value
                This round's chart.
            agree : Value
                Whether the rounds agree in every weight.
            settled : Value
                Whether the rounds agree in every count.

            Returns
            -------
            Computation
                A ``Bool``.
            """
            if self.s.semiring.exact:
                return Return(value=c.primitive("or", agree, settled))
            return c.bind(
                "size",
                INT,
                self.call("chart_size", current),
                lambda size: Return(
                    value=c.primitive(
                        "or",
                        settled,
                        c.primitive(
                            "and",
                            agree,
                            c.primitive("gt_int", round_, Var(local=size)),
                        ),
                    )
                ),
            )

        return c.bind(
            "current",
            self.type,
            self.call("chart_step", old, *inputs),
            lambda current: c.bind(
                "agree",
                BOOL,
                self.call("chart_equal", old, Var(local=current)),
                lambda agree: c.bind(
                    "settled",
                    BOOL,
                    self.call("chart_settled", old, Var(local=current)),
                    lambda settled: c.bind(
                        "stop",
                        BOOL,
                        stops(Var(local=current), Var(local=agree), Var(local=settled)),
                        lambda stop: If(
                            condition=Var(local=stop),
                            then=judged(
                                Var(local=current),
                                Var(local=agree),
                                Var(local=settled),
                            ),
                            otherwise=self.call(
                                "derive",
                                Var(local=current),
                                c.primitive("add_int", round_, self.integer(1)),
                                *inputs,
                            ),
                        ),
                    ),
                ),
            ),
        )

    def size(self, chart: Value) -> Computation:
        """The number of items of a chart.

        Parameters
        ----------
        chart : Value
            A chart.

        Returns
        -------
        Computation
            An ``Int``.
        """
        return self.case(
            chart,
            INT,
            Return(value=self.integer(0)),
            lambda item, weight, count, tail: self.c.bind(
                "rest",
                INT,
                self.call("chart_size", tail),
                lambda rest: Return(
                    value=self.c.primitive("add_int", Var(local=rest), self.integer(1))
                ),
            ),
        )

    def entry(self, *inputs: Value) -> Computation:
        """Settle the goal weight of the iterated chart.

        Parameters
        ----------
        *inputs : Value
            The axioms and their weights, or the sentence.

        Returns
        -------
        Computation
            The settled iteration.
        """
        handlers = self.s.handlers_cell[0]
        assert handlers is not None
        return self.c.settled(
            self.call("derive", self.empty(), self.integer(1), *inputs), handlers
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
        chart.build("chart_size", ("chart",), chart.size),
        chart.build("chart_goal", ("chart",), chart.goal),
        chart.build(chart.reader, ("position", "chart", *chart.inputs), chart.read),
        chart.build("chart_step", ("old", *chart.inputs), chart.step),
        chart.build("derive", ("old", "round", *chart.inputs), chart.derive),
        chart.build("run", chart.inputs, chart.entry),
    ]
    if context.table is not None:
        # The weighted variant elaborates again what reads production
        # weights: the round, the iteration and the entry, and the
        # lexicon's axioms.
        result = result[-4:] if chart.lexical else result[-3:]
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
