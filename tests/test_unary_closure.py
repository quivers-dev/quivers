"""Unary-rule closure in the chart parsers against enumeration of chains and trees.

A chain of unary rules derives a category over a span from another over
the same span. The weight such chains add to a cell is the cell times
the Kleene star ``U* = I ⊕ U ⊕ U² ⊕ ...`` of the rule matrix, a sum with
one term per chain (Stolcke 1995, definition 6; Goodman 1999). The
oracle enumerates the chains, or the parse trees, one by one and folds
them with Python arithmetic; a closure that has no finite value must
raise.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
import torch

from quivers.core.morphisms import observed
from quivers.core.objects import FinSet, ProductSet
from quivers.dsl import load, loads
from quivers.qiec.program_runtime import run_deduction
from quivers.stochastic._kleene import DivergentClosureError, kleene_star
from quivers.stochastic._rule_system import RuleSystem
from quivers.stochastic.inside import InsideAlgorithm
from quivers.stochastic.parsers import ChartParser
from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)


NEG_INF = float("-inf")


def _chains(
    rules: dict[tuple[int, int], float], n: int, max_len: int
) -> dict[tuple[int, int], list[list[float]]]:
    """Every chain of at most ``max_len`` rules, by its endpoints.

    Parameters
    ----------
    rules : dict of (b, a) to float
        Rule weights: the rule derives ``a`` from ``b``.
    n : int
        Number of categories.
    max_len : int
        Longest chain to list.

    Returns
    -------
    dict of (b, a) to list of list of float
        For each pair, the weight sequence of each chain from ``b`` to
        ``a`` (the empty chain from ``b`` to ``b`` is ``[]``).
    """
    out: dict[tuple[int, int], list[list[float]]] = {
        (b, a): [] for b in range(n) for a in range(n)
    }
    frontier = [(b, b, []) for b in range(n)]
    for _ in range(max_len + 1):
        nxt = []
        for start, end, ws in frontier:
            out[(start, end)].append(ws)
            for (b, a), w in rules.items():
                if b == end:
                    nxt.append((start, a, [*ws, w]))
        frontier = nxt
    return out


def _fold_log(chains: list[list[float]], best: bool) -> float:
    """Fold log-weight chains: product is a sum; sum is logsumexp or max.

    Parameters
    ----------
    chains : list of list of float
        Log weights of each chain's rules.
    best : bool
        Take the maximum (Viterbi) instead of the log of the sum.

    Returns
    -------
    float
        The folded value, ``-inf`` for no chains.
    """
    totals = [sum(ws) for ws in chains]
    if not totals:
        return NEG_INF
    if best:
        return max(totals)
    hi = max(totals)
    return hi + math.log(sum(math.exp(t - hi) for t in totals))


def _matrix(rules: dict[tuple[int, int], float], n: int, zero: float) -> torch.Tensor:
    """The rule matrix ``U[b, a]``.

    Parameters
    ----------
    rules : dict of (b, a) to float
        Rule weights.
    n : int
        Number of categories.
    zero : float
        The semiring's zero.

    Returns
    -------
    torch.Tensor
        ``(n, n)`` float64 matrix.
    """
    m = torch.full((n, n), zero, dtype=torch.float64)
    for (b, a), w in rules.items():
        m[b, a] = w
    return m


ACYCLIC = {(0, 1): -0.4, (1, 2): -1.1, (0, 2): -2.0, (2, 3): -0.3, (1, 3): -0.7}


@pytest.mark.parametrize("semiring_name", ["LogProb", "Viterbi"])
def test_acyclic_star_equals_sum_over_chains(semiring_name: str) -> None:
    """With no cycle every chain is listed; the star must match exactly."""
    semiring: ChartSemiring = LOG_PROB if semiring_name == "LogProb" else VITERBI
    star = kleene_star(_matrix(ACYCLIC, 4, NEG_INF), semiring)
    chains = _chains(ACYCLIC, 4, max_len=4)
    for (b, a), cs in chains.items():
        expected = _fold_log(cs, best=semiring_name == "Viterbi")
        assert float(star[b, a]) == pytest.approx(expected, abs=1e-12), (b, a)


def test_acyclic_star_counts_chains() -> None:
    """Under Counting with unit weights the star counts the chains."""
    rules = {k: 1.0 for k in ACYCLIC}
    star = kleene_star(_matrix(rules, 4, 0.0), COUNTING)
    chains = _chains(rules, 4, max_len=4)
    for (b, a), cs in chains.items():
        assert float(star[b, a]) == pytest.approx(len(cs), abs=1e-12), (b, a)


def test_boolean_star_is_reachability() -> None:
    """Under Boolean the star says whether any chain exists, cycles included."""
    rules = {(0, 1): 1.0, (1, 0): 1.0, (1, 2): 1.0}
    star = kleene_star(_matrix(rules, 4, 0.0), BOOLEAN)
    chains = _chains(rules, 4, max_len=6)
    for (b, a), cs in chains.items():
        assert float(star[b, a]) == (1.0 if cs else 0.0), (b, a)


CYCLIC = {(0, 1): math.log(0.5), (1, 0): math.log(0.6), (1, 2): math.log(0.3)}


def test_convergent_cycle_matches_truncated_enumeration() -> None:
    """A contractive cycle: the chain sum converges geometrically.

    Every chain longer than ``L`` passes the cycle (product 0.3) at least
    ``(L - 1) // 2`` times, so the omitted tail is below
    ``0.3 ** ((L - 1) // 2) / (1 - 0.3)`` times the largest single-pass
    weight; at ``L = 40`` this is under 1e-9, the tolerance used.
    """
    star = kleene_star(_matrix(CYCLIC, 3, NEG_INF), LOG_PROB).exp()
    chains = _chains(CYCLIC, 3, max_len=40)
    for (b, a), cs in chains.items():
        expected = sum(math.exp(sum(ws)) for ws in cs)
        assert float(star[b, a]) == pytest.approx(expected, abs=1e-9), (b, a)
    # The self-loop case reported against the inside algorithm: one
    # rule of probability 1/2 from a category to itself sums to 2.
    loop = kleene_star(torch.tensor([[math.log(0.5)]], dtype=torch.float64), LOG_PROB)
    assert float(loop.exp()) == pytest.approx(2.0, abs=1e-12)


def test_viterbi_negative_cycle_takes_best_simple_chain() -> None:
    """Under Viterbi a cycle of negative log weight never helps."""
    star = kleene_star(_matrix(CYCLIC, 3, NEG_INF), VITERBI)
    chains = _chains(CYCLIC, 3, max_len=12)
    for (b, a), cs in chains.items():
        assert float(star[b, a]) == pytest.approx(_fold_log(cs, best=True)), (b, a)


@pytest.mark.parametrize(
    ("semiring", "rules", "zero"),
    [
        (LOG_PROB, {(0, 1): 0.0, (1, 0): 0.0}, NEG_INF),
        (LOG_PROB, {(0, 0): math.log(1.2)}, NEG_INF),
        (LOG_PROB, {(0, 1): math.log(0.9), (1, 0): math.log(1.2)}, NEG_INF),
        (COUNTING, {(0, 1): 1.0, (1, 0): 1.0}, 0.0),
        (VITERBI, {(0, 1): 0.5, (1, 0): -0.2}, NEG_INF),
    ],
)
def test_divergent_closure_raises(
    semiring: ChartSemiring, rules: dict[tuple[int, int], float], zero: float
) -> None:
    """A cycle of weight at least one (positive log weight for Viterbi) raises."""
    with pytest.raises(DivergentClosureError):
        kleene_star(_matrix(rules, 2, zero), semiring)


def test_iterated_accumulation_is_rejected() -> None:
    """The accumulating iteration ``cell ← cell ⊕ cell·U`` overstates the star.

    Eight rounds of it on one rule ``A -> B`` of probability 1/2 give
    ``(I + U)^8``, whose ``[A, B]`` entry is 8 · 1/2 = 4, where the star
    gives 1/2. The oracle check of the star must tell the two apart.
    """
    u = torch.tensor([[0.0, 0.5], [0.0, 0.0]], dtype=torch.float64)
    iterated = torch.linalg.matrix_power(torch.eye(2, dtype=torch.float64) + u, 8)
    star = kleene_star(u.clamp(min=1e-300).log().where(u > 0, NEG_INF), LOG_PROB)
    chains = _chains({(0, 1): math.log(0.5)}, 2, max_len=3)
    expected = math.exp(_fold_log(chains[(0, 1)], best=False))
    assert float(star[0, 1].exp()) == pytest.approx(expected, abs=1e-12)
    assert float(iterated[0, 1]) != pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Whole parsers against parse-tree enumeration
# ---------------------------------------------------------------------------


def _tree_weights(
    cat: int,
    i: int,
    j: int,
    tokens: list[int],
    lexical: dict[tuple[int, int], float],
    binary: dict[tuple[int, int, int], float],
    unary: dict[tuple[int, int], float],
    n_cat: int,
) -> list[float]:
    """Log weights of every parse tree of ``tokens[i:j]`` rooted at ``cat``.

    Parameters
    ----------
    cat : int
        The root category.
    i, j : int
        The span.
    tokens : list of int
        The sentence.
    lexical : dict of (category, token) to float
        Lexical log weights.
    binary : dict of (parent, left, right) to float
        Binary rule log weights.
    unary : dict of (child, parent) to float
        Unary rule log weights, acyclic.
    n_cat : int
        Number of categories.

    Returns
    -------
    list of float
        One total log weight per tree.
    """
    out: list[float] = []
    if j == i + 1 and (cat, tokens[i]) in lexical:
        out.append(lexical[(cat, tokens[i])])
    for k in range(i + 1, j):
        for left, right in itertools.product(range(n_cat), repeat=2):
            w = binary.get((cat, left, right))
            if w is None:
                continue
            for wl in _tree_weights(left, i, k, tokens, lexical, binary, unary, n_cat):
                for wr in _tree_weights(
                    right, k, j, tokens, lexical, binary, unary, n_cat
                ):
                    out.append(w + wl + wr)
    for (child, parent), w in unary.items():
        if parent == cat:
            for wc in _tree_weights(child, i, j, tokens, lexical, binary, unary, n_cat):
                out.append(w + wc)
    return out


def test_inside_algorithm_with_unary_chains_matches_tree_enumeration() -> None:
    """The inside weight is the sum over parse trees, unary chains included."""
    n_cat, n_tok = 4, 2
    rng = torch.Generator().manual_seed(0)
    bin_t = torch.rand(n_cat, n_cat, n_cat, generator=rng, dtype=torch.float64) * 0.2
    lex_t = torch.rand(n_cat, n_tok, generator=rng, dtype=torch.float64)
    # unary[b, a]: derive a from b. The chain 3 -> 2 -> 1 -> 0 and 3 -> 0.
    un_t = torch.zeros(n_cat, n_cat, dtype=torch.float64)
    for (b, a), p in {(3, 2): 0.5, (2, 1): 0.4, (1, 0): 0.3, (3, 0): 0.2}.items():
        un_t[b, a] = p
    N = FinSet(name="N", cardinality=n_cat)
    T = FinSet(name="T", cardinality=n_tok)
    inside = InsideAlgorithm(
        observed(N, ProductSet(components=(N, N)), bin_t),
        observed(N, T, lex_t),
        start=0,
        unary=observed(N, N, un_t),
    )
    lexical = {
        (c, t): math.log(float(lex_t[c, t])) for c in range(n_cat) for t in range(n_tok)
    }
    binary = {
        (a, b, c): math.log(float(bin_t[a, b, c]))
        for a, b, c in itertools.product(range(n_cat), repeat=3)
    }
    unary = {
        (b, a): math.log(float(un_t[b, a])) for b, a in zip(*un_t.nonzero().T.tolist())
    }
    for tokens in ([0], [1, 0], [0, 1, 1]):
        trees = _tree_weights(0, 0, len(tokens), tokens, lexical, binary, unary, n_cat)
        expected = math.log(sum(math.exp(t) for t in trees))
        got = float(inside(torch.tensor(tokens)))
        assert got == pytest.approx(expected, abs=1e-6), tokens


def test_inside_algorithm_single_unary_rule() -> None:
    """One rule ``A -> B`` of probability 1/2 gives ``β(A) = 1/2 · β(B)``."""
    N = FinSet(name="N", cardinality=2)
    T = FinSet(name="T", cardinality=1)
    binary = torch.zeros(2, 2, 2, dtype=torch.float64)
    lexical = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    unary = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)
    inside = InsideAlgorithm(
        observed(N, ProductSet(components=(N, N)), binary),
        observed(N, T, lexical),
        start=0,
        unary=observed(N, N, unary),
    )
    assert float(inside(torch.tensor([0])).exp()) == pytest.approx(0.5, abs=1e-12)


def test_inside_algorithm_divergent_unary_cycle_raises() -> None:
    """A unary cycle of probability one has no finite inside weight."""
    N = FinSet(name="N", cardinality=2)
    T = FinSet(name="T", cardinality=1)
    inside = InsideAlgorithm(
        observed(N, ProductSet(components=(N, N)), torch.zeros(2, 2, 2)),
        observed(N, T, torch.ones(2, 1)),
        unary=observed(N, N, torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
    )
    with pytest.raises(DivergentClosureError):
        inside(torch.tensor([0]))


@pytest.mark.parametrize("semiring_name", ["LogProb", "Viterbi"])
def test_chart_parser_with_unary_chains_matches_tree_enumeration(
    semiring_name: str,
) -> None:
    """The span parser's goal weight folds every parse tree once."""
    torch.manual_seed(0)
    semiring: ChartSemiring = LOG_PROB if semiring_name == "LogProb" else VITERBI
    unary_rules = ((2, 3), (1, 2), (0, 1), (0, 3))  # (result, input)
    unary_w = (-0.7, -0.9, -1.2, -1.6)
    binary_rules = ((0, 1, 3), (3, 3, 3), (1, 2, 0))  # (result, left, right)
    binary_w = (-0.5, -0.8, -0.4)
    system = RuleSystem(
        binary_rules=binary_rules,
        unary_rules=unary_rules,
        n_categories=4,
        binary_weights=binary_w,
        unary_weights=unary_w,
    )
    parser = ChartParser(system, n_terminals=2, start_idx=0, semiring=semiring)
    log_lex = parser.axiom.log_lexicon.detach().double()  # type: ignore[union-attr]
    lexical = {(c, t): float(log_lex[t, c]) for c in range(4) for t in range(2)}
    binary = {
        (r, left, right): w for (r, left, right), w in zip(binary_rules, binary_w)
    }
    unary = {(inp, res): w for (res, inp), w in zip(unary_rules, unary_w)}
    for tokens in ([0], [1, 0], [0, 1, 1]):
        trees = _tree_weights(0, 0, len(tokens), tokens, lexical, binary, unary, 4)
        if semiring_name == "LogProb":
            expected = math.log(sum(math.exp(t) for t in trees))
        else:
            expected = max(trees)
        got = float(parser(torch.tensor([tokens])).detach())
        assert got == pytest.approx(expected, abs=1e-5), tokens


def test_chart_parser_divergent_unary_cycle_raises() -> None:
    """Unary rules ``A -> B`` and ``B -> A`` of log weight 0 diverge under LogProb."""
    system = RuleSystem(
        binary_rules=((0, 0, 0),),
        unary_rules=((0, 1), (1, 0)),
        n_categories=2,
    )
    parser = ChartParser(system, n_terminals=1, start_idx=0)
    with pytest.raises(DivergentClosureError):
        parser(torch.tensor([[0]]))


_EXAMPLE = Path("docs/examples/source/schema_chart_parser.qvr")


def test_schema_parser_runtime_and_reference_count_each_tree_once() -> None:
    """The classic chart, the reference machine, and tree enumeration agree.

    The example's ``perm`` schema adds a unary rule per slash category,
    so trees with and without a unary step both reach ``S``.
    """
    torch.manual_seed(0)
    program = load(str(_EXAMPLE))
    classic = program.morphism
    assert isinstance(classic, ChartParser)
    assert program.qiec is not None
    with torch.no_grad():
        classic.axiom.lexicon_logits.zero_()  # type: ignore[union-attr]
    system = classic.rule_system
    n_cat = system.n_categories
    lexical = {(c, t): -math.log(n_cat) for c in range(n_cat) for t in range(4)}
    binary = {(r, left, right): 0.0 for r, left, right in system.binary_rules}
    unary = {(inp, res): 0.0 for res, inp in system.unary_rules}
    words = ["the", "dog", "cat", "sleeps"]
    parameters = {f"lp_parser.lex.{i}": -math.log(n_cat) for i in range(n_cat * 4)}
    for tokens in ([1, 3], [0, 1, 3]):
        trees = _tree_weights(
            classic._start_idx, 0, len(tokens), tokens, lexical, binary, unary, n_cat
        )
        expected = math.log(sum(math.exp(t) for t in trees))
        assert float(classic(torch.tensor(tokens)).detach()) == pytest.approx(
            expected, abs=1e-5
        )
        reference = run_deduction(
            program.qiec,
            "lp_parser",
            tokens=[words[t] for t in tokens],
            parameters=parameters,
            fuel=1_000_000,
        )
        assert reference.weight == pytest.approx(expected, abs=1e-9)


def test_schema_parser_with_cyclic_unary_rules() -> None:
    """Cyclic unary rules: no reference entry, and a divergent chart raises."""
    source = _EXAMPLE.read_text()
    source = source.replace(
        "schema perm (X, Y : Cat) : X / Y -> X \\ Y",
        "schema perm (X, Y : Cat) : X / Y -> X \\ Y\n"
        "schema unperm (X, Y : Cat) : X \\ Y -> X / Y",
    ).replace(
        "bundle lp_rules : [fwd_app, bwd_app, perm]",
        "bundle lp_rules : [fwd_app, bwd_app, perm, unperm]",
    )
    program = loads(source)
    assert program.qiec is not None
    assert "lp_parser" not in [entry.name for entry in program.entry_points()]
    assert "unary rules that cycle" in (program.qiec.gap or "")
    parser = program.morphism
    assert isinstance(parser, ChartParser)
    with pytest.raises(DivergentClosureError):
        parser(torch.tensor([1, 3]))
