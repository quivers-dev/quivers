"""Tests for the agenda-based weighted-deduction engine.

Exercises every public abstraction (Agenda variants, Chart, rule
matching, AgendaResult) on hand-built deduction systems covering
the canonical algorithmic specializations.
"""

from __future__ import annotations

import torch

from quivers.stochastic.agenda import (
    AgendaResult,
    ChartView,
    DeductionSystem,
    FIFOAgenda,
    HashChart,
    InferenceRule,
    LIFOAgenda,
    PriorityQueueAgenda,
    Wildcard,
    cky_agenda,
    depth_first_agenda,
    earley_agenda,
    instantiate,
    match,
    run_agenda,
    semi_naive_agenda,
    viterbi_agenda,
)
from quivers.stochastic.semiring import (
    BOOLEAN,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


def test_rules_accept_bare_wildcard_premises_and_conclusions() -> None:
    """Whole-item variables preserve the weights of a two-step derivation."""
    item = Wildcard(name="item")
    rules = (
        InferenceRule(name="unwrap", premises=(("source", item),), conclusion=item),
        InferenceRule(
            name="finish",
            premises=(item,),
            conclusion=("done",),
            side_condition=lambda bindings: bindings["item"] == ("answer",),
        ),
    )
    result = run_agenda(
        axioms=[(("source", ("answer",)), torch.tensor(-0.75))],
        rules=rules,
        semiring=LOG_PROB,
    )
    weights = dict(result.chart.items())
    assert set(weights) == {("source", ("answer",)), ("answer",), ("done",)}
    for weight in weights.values():
        torch.testing.assert_close(weight, torch.tensor(-0.75))
    assert match(item, ("answer",), {"item": ("different",)}) is None


class TestMatch:
    def test_constant_match(self):
        assert match(("span", "A", 0, 3), ("span", "A", 0, 3)) == {}

    def test_constant_mismatch(self):
        assert match(("span", "A", 0, 3), ("span", "B", 0, 3)) is None

    def test_arity_mismatch(self):
        assert match(("span", "A"), ("span", "A", 0, 3)) is None

    def test_wildcard_bind(self):
        b = match(("span", Wildcard(name="X"), 0, 3), ("span", "A", 0, 3))
        assert b == {"X": "A"}

    def test_wildcard_shared(self):
        # Same wildcard name must bind to equal values.
        ok = match(("rule", Wildcard(name="X"), Wildcard(name="X")), ("rule", "A", "A"))
        assert ok == {"X": "A"}
        bad = match(
            ("rule", Wildcard(name="X"), Wildcard(name="X")), ("rule", "A", "B")
        )
        assert bad is None

    def test_nested_pattern(self):
        pat = ("step", ("span", Wildcard(name="A"), 0, Wildcard(name="j")))
        item = ("step", ("span", "NP", 0, 3))
        assert match(pat, item) == {"A": "NP", "j": 3}

    def test_instantiate_substitutes(self):
        out = instantiate(
            ("span", Wildcard(name="X"), 0, Wildcard(name="j")), {"X": "VP", "j": 4}
        )
        assert out == ("span", "VP", 0, 4)


# ---------------------------------------------------------------------------
# Hash chart insert / aggregate / lookup
# ---------------------------------------------------------------------------


class TestHashChart:
    def test_insert_new(self):
        chart = HashChart()
        ok = chart.insert_or_aggregate(("a",), torch.tensor(1.0), LOG_PROB)
        assert ok
        assert chart.get(("a",)) is not None

    def test_aggregate_via_semiring(self):
        chart = HashChart()
        chart.insert_or_aggregate(("a",), torch.tensor(0.0), LOG_PROB)
        # Same item, larger weight under log-prob (logsumexp) — should
        # increase the aggregate.
        chart.insert_or_aggregate(("a",), torch.tensor(1.0), LOG_PROB)
        stored = chart.get(("a",))
        # logsumexp(0, 1) > 1
        assert stored is not None
        assert float(stored) > 1.0 - 1e-6

    def test_lookup_with_pattern(self):
        chart = HashChart()
        chart.insert_or_aggregate(("span", "A", 0, 3), torch.tensor(0.0), LOG_PROB)
        chart.insert_or_aggregate(("span", "B", 0, 3), torch.tensor(0.0), LOG_PROB)
        chart.insert_or_aggregate(("token", "A"), torch.tensor(0.0), LOG_PROB)
        out = list(chart.lookup(("span", Wildcard(name="X"), 0, 3)))
        assert len(out) == 2
        names = sorted(item[1] for item, _ in out)
        assert names == ["A", "B"]


# ---------------------------------------------------------------------------
# Agenda disciplines — FIFO, LIFO, priority queue
# ---------------------------------------------------------------------------


class TestAgendas:
    def test_fifo_order(self):
        a = FIFOAgenda()
        for i, x in enumerate(["a", "b", "c"]):
            a.push((x,), torch.tensor(float(i)))
        popped = []
        while not a.empty():
            item, _ = a.pop()
            popped.append(item[0])
        assert popped == ["a", "b", "c"]

    def test_lifo_order(self):
        a = LIFOAgenda()
        for x in ["a", "b", "c"]:
            a.push((x,), torch.tensor(0.0))
        popped = []
        while not a.empty():
            item, _ = a.pop()
            popped.append(item[0])
        assert popped == ["c", "b", "a"]

    def test_priority_queue_max_first(self):
        a = PriorityQueueAgenda(lambda item, w: float(w))
        a.push(("low",), torch.tensor(1.0))
        a.push(("high",), torch.tensor(10.0))
        a.push(("mid",), torch.tensor(5.0))
        item, _ = a.pop()
        assert item == ("high",)


# ---------------------------------------------------------------------------
# End-to-end: a tiny CFG run through the agenda
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """A 2-rule context-free deduction over a hand-built lexicon."""

    @staticmethod
    def grammar() -> tuple[list[InferenceRule], ChartSemiring]:
        # Items: ('span', cat, i, j) — category over positions [i, j).
        # Rules:
        #   span(NP, i, k) + span(VP, k, j) |- span(S, i, j)   (concat)
        rule_s = InferenceRule(
            name="combine",
            premises=(
                ("span", "NP", Wildcard(name="i"), Wildcard(name="k")),
                ("span", "VP", Wildcard(name="k"), Wildcard(name="j")),
            ),
            conclusion=("span", "S", Wildcard(name="i"), Wildcard(name="j")),
        )
        return [rule_s], LOG_PROB

    def test_cky_finds_S_span(self):
        rules, semiring = self.grammar()
        axioms = [
            (("span", "NP", 0, 1), torch.tensor(0.0)),
            (("span", "VP", 1, 3), torch.tensor(0.0)),
        ]
        result = run_agenda(
            axioms=axioms,
            rules=rules,
            semiring=semiring,
            agenda=cky_agenda(),
            goal=lambda item: item == ("span", "S", 0, 3),
        )
        assert isinstance(result, AgendaResult)
        assert len(result.goal_items) == 1
        item, w = result.goal_items[0]
        assert item == ("span", "S", 0, 3)
        # Two log(1) = 0 premises × LOG_PROB.times = 0 + 0 = 0
        assert float(w) == 0.0

    def test_strategy_independence_acyclic(self):
        """On an acyclic grammar the chart is the sum over derivations,
        whatever the agenda discipline. Run CKY, FIFO/semi-naive, LIFO,
        and Earley on the same grammar and axioms and assert the goal
        weights agree.
        """
        rules, semiring = self.grammar()
        axioms = [
            (("span", "NP", 0, 1), torch.tensor(0.0)),
            (("span", "VP", 1, 3), torch.tensor(0.0)),
            (("span", "NP", 0, 2), torch.tensor(-1.0)),
            (("span", "VP", 2, 5), torch.tensor(-0.5)),
        ]
        weights = []
        for agenda_factory in (
            cky_agenda,
            semi_naive_agenda,
            depth_first_agenda,
            earley_agenda,
        ):
            r = run_agenda(
                axioms=axioms,
                rules=rules,
                semiring=semiring,
                agenda=agenda_factory(),
                goal=lambda item: (
                    isinstance(item, tuple) and item[0] == "span" and item[1] == "S"
                ),
            )
            # Collect weights as a dict for comparison.
            weights.append({i: float(w) for i, w in r.goal_items})
        # All strategies agree on the chart's goal-item set.
        keys = [tuple(sorted(d)) for d in weights]
        assert all(k == keys[0] for k in keys), f"goal-item sets differ: {keys}"
        # All strategies agree on each goal's weight, up to the
        # floating-point order of the logsumexp reductions.
        for k in weights[0]:
            vals = [d[k] for d in weights]
            assert all(abs(v - vals[0]) < 1e-5 for v in vals), (
                f"weight at {k} disagrees across strategies: {vals}"
            )

    def test_viterbi_picks_best(self):
        """Under the Viterbi semiring (max, +) (log-Viterbi), two
        paths to the same item produce the max of the two
        log-weight sums, not a sum.
        """
        rule = InferenceRule(
            name="combine",
            premises=(
                ("span", "NP", Wildcard(name="i"), Wildcard(name="k")),
                ("span", "VP", Wildcard(name="k"), Wildcard(name="j")),
            ),
            conclusion=("span", "S", Wildcard(name="i"), Wildcard(name="j")),
        )
        axioms = [
            (("span", "NP", 0, 1), torch.tensor(-0.6)),
            (("span", "VP", 1, 3), torch.tensor(-0.3)),
            (("span", "NP", 0, 2), torch.tensor(-0.1)),
            (("span", "VP", 2, 3), torch.tensor(-1.2)),
        ]
        r = run_agenda(
            axioms=axioms,
            rules=[rule],
            semiring=VITERBI,
            agenda=viterbi_agenda(lambda i, w: float(w)),
            goal=lambda item: item == ("span", "S", 0, 3),
        )
        assert len(r.goal_items) == 1
        # Two paths in log-space (Viterbi = (max, +)):
        #   NP[0,1] + VP[1,3] = -0.6 + -0.3 = -0.9
        #   NP[0,2] + VP[2,3] = -0.1 + -1.2 = -1.3
        # Viterbi picks max → -0.9.
        _, w = r.goal_items[0]
        assert abs(float(w) - (-0.9)) < 1e-5

    def test_boolean_semiring_recognition(self):
        """Boolean semiring: any derivation suffices; the chart's
        goal weight is 1 if the goal is derivable, 0 otherwise.
        """
        rule = InferenceRule(
            name="combine",
            premises=(
                ("span", "NP", Wildcard(name="i"), Wildcard(name="k")),
                ("span", "VP", Wildcard(name="k"), Wildcard(name="j")),
            ),
            conclusion=("span", "S", Wildcard(name="i"), Wildcard(name="j")),
        )
        # Derivable case.
        r_ok = run_agenda(
            axioms=[
                (("span", "NP", 0, 1), torch.tensor(1.0)),
                (("span", "VP", 1, 3), torch.tensor(1.0)),
            ],
            rules=[rule],
            semiring=BOOLEAN,
            agenda=cky_agenda(),
            goal=lambda item: item == ("span", "S", 0, 3),
        )
        assert len(r_ok.goal_items) == 1

        # Non-derivable case: NP and VP don't share a midpoint.
        r_no = run_agenda(
            axioms=[
                (("span", "NP", 0, 1), torch.tensor(1.0)),
                (("span", "VP", 2, 3), torch.tensor(1.0)),
            ],
            rules=[rule],
            semiring=BOOLEAN,
            agenda=cky_agenda(),
            goal=lambda item: item == ("span", "S", 0, 3),
        )
        assert len(r_no.goal_items) == 0


# ---------------------------------------------------------------------------
# Deduction-system wrapper
# ---------------------------------------------------------------------------


class TestChartView:
    """Tests for the user-facing ChartView (charts as values)."""

    @staticmethod
    def _simple_system() -> DeductionSystem:
        rule = InferenceRule(
            name="combine",
            premises=(
                ("span", "NP", Wildcard(name="i"), Wildcard(name="k")),
                ("span", "VP", Wildcard(name="k"), Wildcard(name="j")),
            ),
            conclusion=("span", "S", Wildcard(name="i"), Wildcard(name="j")),
        )
        return DeductionSystem(
            rules=(rule,),
            semiring=LOG_PROB,
            axiom_injector=lambda inp: inp,
            goal=lambda item: (
                isinstance(item, tuple)
                and item[0] == "span"
                and item[1] == "S"
                and item[2] == 0
                and item[3] == 3
            ),
        )

    def test_call_returns_chartview(self):
        sys = self._simple_system()
        axioms = [
            (("span", "NP", 0, 1), torch.tensor(-0.3)),
            (("span", "VP", 1, 3), torch.tensor(-0.4)),
        ]
        view = sys(axioms)
        assert isinstance(view, ChartView)
        assert view.semiring is LOG_PROB

    def test_weight_query(self):
        sys = self._simple_system()
        axioms = [
            (("span", "NP", 0, 1), torch.tensor(-0.3)),
            (("span", "VP", 1, 3), torch.tensor(-0.4)),
        ]
        view = sys(axioms)
        w = view.weight(("span", "S", 0, 3))
        # log-prob times = sum: -0.3 + -0.4 = -0.7.
        assert abs(float(w) - (-0.7)) < 1e-5

    def test_try_weight_default(self):
        sys = self._simple_system()
        view = sys([(("span", "NP", 0, 1), torch.tensor(0.0))])
        w = view.try_weight(("span", "S", 0, 3), default=torch.tensor(-99.0))
        assert float(w) == -99.0

    def test_enumerate_pattern(self):
        sys = self._simple_system()
        axioms = [
            (("span", "NP", 0, 1), torch.tensor(0.0)),
            (("span", "VP", 1, 3), torch.tensor(0.0)),
            (("span", "NP", 0, 2), torch.tensor(0.0)),
            (("span", "VP", 2, 4), torch.tensor(0.0)),
        ]
        view = sys(axioms)
        nps = view.enumerate(("span", "NP", Wildcard(name="i"), Wildcard(name="j")))
        assert len(nps) == 2

    def test_goal_weight(self):
        sys = self._simple_system()
        axioms = [
            (("span", "NP", 0, 1), torch.tensor(-0.5)),
            (("span", "VP", 1, 3), torch.tensor(-0.2)),
        ]
        view = sys(axioms)
        gw = view.goal_weight()
        assert abs(float(gw) - (-0.7)) < 1e-5

    def test_gradient_flows_through_chart(self):
        """Differentiability: rule-weight gradients propagate through
        the agenda fixed-point. Standard chain-rule check via
        torch.autograd.grad.
        """
        # Use learnable axiom weights as a stand-in for rule weights.
        np_w = torch.tensor(-0.3, requires_grad=True)
        vp_w = torch.tensor(-0.4, requires_grad=True)
        sys = self._simple_system()
        axioms = [
            (("span", "NP", 0, 1), np_w),
            (("span", "VP", 1, 3), vp_w),
        ]
        view = sys(axioms)
        loss = view.weight(("span", "S", 0, 3))
        grads = torch.autograd.grad(loss, [np_w, vp_w])
        # d(np_w + vp_w)/d(np_w) = 1, similarly for vp_w.
        assert abs(float(grads[0]) - 1.0) < 1e-5
        assert abs(float(grads[1]) - 1.0) < 1e-5


class TestDeductionSystem:
    def test_system_wraps_run(self):
        rule = InferenceRule(
            name="step",
            premises=(("a",),),
            conclusion=("b",),
        )

        def axiom_injector(input_value: int) -> list:
            return [(("a",), torch.tensor(float(input_value)))]

        sys = DeductionSystem(
            rules=(rule,),
            semiring=LOG_PROB,
            axiom_injector=axiom_injector,
            goal=lambda item: item == ("b",),
        )
        r = sys.run(input_value=2)
        assert len(r.goal_items) == 1
        item, _ = r.goal_items[0]
        assert item == ("b",)
