"""The agenda engine against brute-force enumeration of derivations.

A weighted deductive system assigns an item the semiring sum, over its
derivation trees, of the semiring product of the weights in the tree
(Goodman 1999, semiring parsing). The oracle here builds every
derivation tree explicitly, with its own pattern matcher, and folds the
trees with plain Python arithmetic; the engine's chart must agree with
it at every item, under every agenda discipline and several semirings.

The grammars include rules whose premises one item can fill more than
once (``NP, NP |- S`` with a single ``NP``), an item that fills such a
rule after several separate contributions, and a three-premise rule over
one item, which are the cases a per-premise firing over-counts.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import pytest
import torch

from quivers.stochastic import agenda as agenda_module
from quivers.stochastic.agenda import (
    Agenda,
    FIFOAgenda,
    InferenceRule,
    LIFOAgenda,
    Wildcard,
    knuth_agenda,
    run_agenda,
)
from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)


# ---------------------------------------------------------------------------
# The oracle: an independent matcher, explicit trees, Python arithmetic
# ---------------------------------------------------------------------------


type Term = tuple[str | int, ...]
type Pat = tuple[str | int | Wildcard, ...]
type Env = dict[str, str | int]


class FloatSemiring:
    """A semiring on Python floats, written independently of the engine.

    Parameters
    ----------
    plus : Callable[[float, float], float]
        The semiring sum.
    times : Callable[[float, float], float]
        The semiring product.
    """

    def __init__(
        self,
        plus: Callable[[float, float], float],
        times: Callable[[float, float], float],
    ) -> None:
        self.plus = plus
        self.times = times


def _logaddexp(a: float, b: float) -> float:
    """Log of the sum of the exponentials of two floats.

    Parameters
    ----------
    a, b : float
        Log-domain values.

    Returns
    -------
    float
        ``log(exp(a) + exp(b))``.
    """
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


ORACLE: dict[str, FloatSemiring] = {
    "LogProb": FloatSemiring(_logaddexp, lambda a, b: a + b),
    "Viterbi": FloatSemiring(max, lambda a, b: a + b),
    "Counting": FloatSemiring(lambda a, b: a + b, lambda a, b: a * b),
    "Boolean": FloatSemiring(max, lambda a, b: a * b),
}

ENGINE: dict[str, ChartSemiring] = {
    "LogProb": LOG_PROB,
    "Viterbi": VITERBI,
    "Counting": COUNTING,
    "Boolean": BOOLEAN,
}


def _bind(pattern: Pat, term: Term, env: Env) -> Env | None:
    """Match a flat pattern against a flat term (the oracle's own matcher).

    Parameters
    ----------
    pattern : Term
        A tuple whose slots are constants or `Wildcard` variables.
    term : Term
        A ground tuple.
    env : Env
        Variables already bound.

    Returns
    -------
    Env or None
        The extended bindings, or ``None`` on a mismatch.
    """
    if len(pattern) != len(term):
        return None
    out = dict(env)
    for p, v in zip(pattern, term):
        if isinstance(p, Wildcard):
            if p.name in out and out[p.name] != v:
                return None
            out[p.name] = v
        elif p != v:
            return None
    return out


def _ground(pattern: Pat, env: Env) -> Term:
    """Substitute bound variables into a flat pattern.

    Parameters
    ----------
    pattern : Term
        The pattern.
    env : Env
        The bindings.

    Returns
    -------
    Term
        The ground tuple.
    """
    return tuple(env[p.name] if isinstance(p, Wildcard) else p for p in pattern)


class OracleRule:
    """A rule as the oracle sees it: premises, conclusion, constant weight.

    Parameters
    ----------
    name : str
        The rule's name.
    premises : tuple of Term
        Premise patterns.
    conclusion : Term
        Conclusion pattern.
    weight : float or None
        A constant factor the rule multiplies in, or ``None`` for none.
    side : Callable[[Env], bool] or None
        A side condition on the bindings.
    """

    def __init__(
        self,
        name: str,
        premises: tuple[Pat, ...],
        conclusion: Pat,
        weight: float | None = None,
        side: Callable[[Env], bool] | None = None,
    ) -> None:
        self.name = name
        self.premises = premises
        self.conclusion = conclusion
        self.weight = weight
        self.side = side

    def engine_rule(self) -> InferenceRule:
        """The same rule for the engine.

        Returns
        -------
        InferenceRule
            The rule, with a ``weight_fn`` multiplying in the constant
            weight when there is one.
        """
        constant = self.weight

        def times_constant(
            _bindings: dict[str, str | int],
            premise_weights: tuple[torch.Tensor, ...],
            semiring: ChartSemiring,
        ) -> torch.Tensor:
            acc = torch.tensor(constant)
            for w in premise_weights:
                acc = semiring.times(acc, w)
            return acc

        side = self.side
        return InferenceRule(
            name=self.name,
            premises=self.premises,
            conclusion=self.conclusion,
            weight_fn=times_constant if constant is not None else None,
            side_condition=(lambda b: side(b)) if side is not None else None,
        )


def _derivable(axioms: list[tuple[Term, float]], rules: list[OracleRule]) -> set[Term]:
    """Every item the rules derive from the axioms (unweighted closure).

    Parameters
    ----------
    axioms : list of (Term, float)
        The axioms.
    rules : list of OracleRule
        The rules.

    Returns
    -------
    set of Term
        The derivable items.
    """
    items = {a for a, _ in axioms}
    while True:
        new = set()
        for rule in rules:
            for tup in itertools.product(
                sorted(items, key=repr), repeat=len(rule.premises)
            ):
                e: Env | None = {}
                for pat, it in zip(rule.premises, tup):
                    e = _bind(pat, it, e) if e is not None else None
                if e is None or (rule.side is not None and not rule.side(e)):
                    continue
                c = _ground(rule.conclusion, e)
                if c not in items:
                    new.add(c)
        if not new:
            return items
        items |= new


def oracle_values(
    axioms: list[tuple[Term, float]],
    rules: list[OracleRule],
    sr: FloatSemiring,
) -> dict[Term, float]:
    """Each derivable item's value as a fold over its derivation trees.

    The trees are enumerated explicitly (an axiom occurrence is a leaf;
    a rule application over a tuple of antecedent items combines one
    tree per antecedent), each tree's weight is the product of its leaf
    and rule weights, and the item's value is the sum of its trees'
    weights. Requires an acyclic system.

    Parameters
    ----------
    axioms : list of (Term, float)
        The axioms; an item listed twice has two leaf derivations.
    rules : list of OracleRule
        The rules.
    sr : FloatSemiring
        The semiring.

    Returns
    -------
    dict of Term to float
        The value of every derivable item.
    """
    items = _derivable(axioms, rules)
    ordered = sorted(items, key=repr)

    def trees(item: Term, depth: int = 0) -> list[float]:
        assert depth < 50, "the oracle needs an acyclic system"
        out = [w for a, w in axioms if a == item]
        for rule in rules:
            for tup in itertools.product(ordered, repeat=len(rule.premises)):
                env: Env | None = {}
                for pat, it in zip(rule.premises, tup):
                    env = _bind(pat, it, env) if env is not None else None
                if env is None or (rule.side is not None and not rule.side(env)):
                    continue
                if _ground(rule.conclusion, env) != item:
                    continue
                for children in itertools.product(*(trees(c, depth + 1) for c in tup)):
                    acc = rule.weight
                    for child in children:
                        acc = child if acc is None else sr.times(acc, child)
                    assert acc is not None
                    out.append(acc)
        return out

    values: dict[Term, float] = {}
    for item in items:
        ws = trees(item)
        total = ws[0]
        for w in ws[1:]:
            total = sr.plus(total, w)
        values[item] = total
    return values


# ---------------------------------------------------------------------------
# Grammars
# ---------------------------------------------------------------------------


I, J, K, X, Y = (Wildcard(n) for n in ("i", "j", "k", "x", "y"))


def g_shared_premise() -> tuple[list[tuple[Term, float]], list[OracleRule]]:
    """``NP, NP |- S`` with one ``NP`` axiom."""
    return [(("NP",), -0.7)], [OracleRule("app", (("NP",), ("NP",)), ("S",))]


def g_shared_premise_two_contributions() -> tuple[
    list[tuple[Term, float]], list[OracleRule]
]:
    """``NP`` derived three ways, then ``NP, NP |- S`` and ``NP, NP, NP |- T``."""
    axioms: list[tuple[Term, float]] = [
        (("NP",), -0.4),
        (("NP",), -1.3),
        (("N",), -0.2),
    ]
    rules = [
        OracleRule("lift", (("N",),), ("NP",), weight=-0.9),
        OracleRule("app", (("NP",), ("NP",)), ("S",), weight=-0.1),
        OracleRule("tri", (("NP",), ("NP",), ("NP",)), ("T",)),
        OracleRule("mix", (("S",), ("NP",)), ("U",)),
    ]
    return axioms, rules


def g_ambiguous_cky() -> tuple[list[tuple[Term, float]], list[OracleRule]]:
    """``X -> X X`` over four words: Catalan-many bracketings."""
    words = [-0.3, -0.5, -0.2, -0.9]
    axioms: list[tuple[Term, float]] = [
        (("span", "X", i, i + 1), w) for i, w in enumerate(words)
    ]
    axioms.append((("span", "Y", 1, 3), -1.1))
    rules = [
        OracleRule(
            "bin",
            (("span", "X", I, K), ("span", "X", K, J)),
            ("span", "X", I, J),
            weight=-0.25,
        ),
        OracleRule("unary", (("span", "Y", I, J),), ("span", "X", I, J), weight=-0.6),
    ]
    return axioms, rules


def g_variables() -> tuple[list[tuple[Term, float]], list[OracleRule]]:
    """Pairs over a set, with a diagonal rule and a side condition."""
    axioms: list[tuple[Term, float]] = [
        (("A", "a"), -0.2),
        (("A", "b"), -0.8),
        (("A", "a"), -1.5),
    ]
    rules = [
        OracleRule("pair", (("A", X), ("A", Y)), ("B", X, Y)),
        OracleRule("diag", (("A", X), ("A", X)), ("C", X), weight=-0.3),
        OracleRule(
            "lt",
            (("B", X, Y), ("C", Y)),
            ("D", X),
            side=lambda e: str(e["x"]) <= str(e["y"]),
        ),
    ]
    return axioms, rules


GRAMMARS = {
    "shared_premise": g_shared_premise,
    "shared_premise_two_contributions": g_shared_premise_two_contributions,
    "ambiguous_cky": g_ambiguous_cky,
    "variables": g_variables,
}

AGENDAS: dict[str, Callable[[], Agenda]] = {
    "fifo": FIFOAgenda,
    "lifo": LIFOAgenda,
    "knuth": knuth_agenda,
}


def _engine_axioms(
    axioms: list[tuple[Term, float]], name: str
) -> list[tuple[Term, torch.Tensor]]:
    """The axioms in the engine's encoding of the semiring.

    Boolean and Counting read every axiom as a one (one derivation, or
    true); the log semirings read the log weights as given.

    Parameters
    ----------
    axioms : list of (Term, float)
        The axioms with log weights.
    name : str
        The semiring's name.

    Returns
    -------
    list of (Term, torch.Tensor)
        The engine's axioms.
    """
    return [(a, torch.tensor(_value(w, name), dtype=torch.float64)) for a, w in axioms]


def _value(w: float, name: str) -> float:
    """A log weight in the representation the named semiring uses.

    Parameters
    ----------
    w : float
        A log weight.
    name : str
        The semiring's name.

    Returns
    -------
    float
        ``w`` for the log semirings; ``1.0`` for Boolean; ``exp(w)``
        for Counting, which then sums products of probabilities.
    """
    if name in ("LogProb", "Viterbi"):
        return w
    if name == "Boolean":
        return 1.0
    return math.exp(w)


def _oracle_rules(rules: list[OracleRule], name: str) -> list[OracleRule]:
    """The rules with their constant weights in the named representation.

    Parameters
    ----------
    rules : list of OracleRule
        Rules with log weights.
    name : str
        The semiring's name.

    Returns
    -------
    list of OracleRule
        The same rules re-weighted.
    """
    return [
        OracleRule(
            r.name,
            r.premises,
            r.conclusion,
            None if r.weight is None else _value(r.weight, name),
            r.side,
        )
        for r in rules
    ]


@pytest.mark.parametrize("agenda_name", sorted(AGENDAS))
@pytest.mark.parametrize("semiring_name", sorted(ENGINE))
@pytest.mark.parametrize("grammar_name", sorted(GRAMMARS))
def test_chart_equals_sum_over_derivations(
    grammar_name: str, semiring_name: str, agenda_name: str
) -> None:
    """Every chart item equals the fold of its explicitly enumerated trees."""
    torch.set_default_dtype(torch.float64)
    try:
        axioms, rules = GRAMMARS[grammar_name]()
        rules = _oracle_rules(rules, semiring_name)
        expected = oracle_values(
            [(a, _value(w, semiring_name)) for a, w in axioms],
            rules,
            ORACLE[semiring_name],
        )
        result = run_agenda(
            axioms=_engine_axioms(axioms, semiring_name),
            rules=[r.engine_rule() for r in rules],
            semiring=ENGINE[semiring_name],
            agenda=AGENDAS[agenda_name](),
        )
        got = {item: float(w) for item, w in result.chart.items()}
    finally:
        torch.set_default_dtype(torch.float32)
    assert set(got) == set(expected)
    for item, value in expected.items():
        assert got[item] == pytest.approx(value, rel=1e-12, abs=1e-12), item


def test_one_item_filling_two_premises_is_one_derivation() -> None:
    """``NP, NP |- S`` over one ``NP`` of probability one scores log 1 = 0."""
    rule = InferenceRule(name="app", premises=(("NP",), ("NP",)), conclusion=("S",))
    for make in AGENDAS.values():
        r = run_agenda([(("NP",), torch.tensor(0.0))], [rule], LOG_PROB, make())
        assert float(r.chart.get(("S",))) == 0.0  # type: ignore[arg-type]
        c = run_agenda([(("NP",), torch.tensor(1.0))], [rule], COUNTING, make())
        assert float(c.chart.get(("S",))) == 1.0  # type: ignore[arg-type]


def test_catalan_counts() -> None:
    """``X -> X X`` over n words has Catalan(n - 1) derivations."""
    rule = InferenceRule(
        name="bin",
        premises=(("span", I, K), ("span", K, J)),
        conclusion=("span", I, J),
    )
    for n, catalan in [(2, 1), (3, 2), (4, 5), (5, 14), (6, 42)]:
        axioms = [(("span", i, i + 1), torch.tensor(1.0)) for i in range(n)]
        r = run_agenda(axioms, [rule], COUNTING, LIFOAgenda())
        assert float(r.chart.get(("span", 0, n))) == catalan  # type: ignore[arg-type]


class _UpdateForgettingOld(agenda_module._Update):
    """A mutant that treats an earlier copy of the popped item as absent."""

    def __init__(
        self, *, item: tuple, delta: torch.Tensor, old: torch.Tensor | None
    ) -> None:
        super().__init__(item=item, delta=delta, old=None)


class _UpdateReadingDeltaAsOld(agenda_module._Update):
    """A mutant that reads the increment where the old weight belongs."""

    def __init__(
        self, *, item: tuple, delta: torch.Tensor, old: torch.Tensor | None
    ) -> None:
        super().__init__(item=item, delta=delta, old=delta)


@pytest.mark.parametrize("mutant", [_UpdateForgettingOld, _UpdateReadingDeltaAsOld])
def test_oracle_rejects_mutated_update(
    mutant: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong old-weight factor disagrees with the derivation oracle."""
    monkeypatch.setattr(agenda_module, "_Update", mutant)
    axioms, rules = g_shared_premise_two_contributions()
    expected = oracle_values(axioms, rules, ORACLE["LogProb"])
    result = run_agenda(
        axioms=_engine_axioms(axioms, "LogProb"),
        rules=[r.engine_rule() for r in rules],
        semiring=LOG_PROB,
        agenda=LIFOAgenda(),
    )
    got = {item: float(w) for item, w in result.chart.items()}
    assert any(
        abs(got.get(item, -math.inf) - value) > 1e-6 for item, value in expected.items()
    )


def test_rule_callback_runs_once_per_application() -> None:
    """The callback sees each rule application once, with final weights."""
    seen: list[tuple[str, tuple, tuple, float]] = []

    def callback(
        name: str,
        antecedents: list[tuple[tuple, torch.Tensor]],
        conclusion: tuple,
        weight: torch.Tensor,
    ) -> None:
        seen.append((name, tuple(a for a, _ in antecedents), conclusion, float(weight)))

    rule = InferenceRule(name="app", premises=(("NP",), ("NP",)), conclusion=("S",))
    run_agenda(
        [(("NP",), torch.tensor(-0.4)), (("NP",), torch.tensor(-1.3))],
        [rule],
        LOG_PROB,
        LIFOAgenda(),
        rule_callback=callback,
    )
    assert len(seen) == 1
    name, antecedents, conclusion, weight = seen[0]
    np_weight = _logaddexp(-0.4, -1.3)
    assert (name, antecedents, conclusion) == ("app", (("NP",), ("NP",)), ("S",))
    assert weight == pytest.approx(2 * np_weight, abs=1e-6)
