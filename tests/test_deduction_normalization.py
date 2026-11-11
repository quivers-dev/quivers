"""Deduction weights normalized per left-hand side (``normalize=lhs``).

Under ``normalize=lhs`` the rules and lexicon entries that rewrite one
left-hand side form a probability distribution: each group's free
log-weights pass through a ``log_softmax``, a production that is not
learnable entering with free log-weight zero. The deduction is then a
probabilistic context-free grammar. These tests check the inside weight
of both evaluators against a CKY recursion written here, the per-group
sums, maximum-likelihood recovery of a generating grammar from its
yields, the partition function, the refusals of every deduction whose
grouping is not fixed, and that the unnormalized mode is unchanged.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence

import pytest
import torch

from quivers.dsl import Compiler, loads, parse
from quivers.dsl.ast_nodes import DeductionDecl
from quivers.dsl.compiler import CompileError
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_deduction
from quivers.stochastic import DeductionSystem
from quivers.stochastic.deduction import adam_fit_deduction

# An ambiguous recursive grammar: prepositional phrases attach to noun or
# verb phrases, and "saw" is a noun or a verb. The unary ``vp_v`` and the
# entry for "with" are not learnable, so they enter their groups with free
# log-weight zero.
AMBIGUOUS = """
object Term : FinSet 16
deduction G : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, VP, PP, V, P, Det, N, span, the, a, dog, cat, telescope, saw, runs, with
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S) #[learnable]
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP) #[learnable]
    rule np_np_pp : span(I, K, NP), span(K, J, PP) |- span(I, J, NP) #[learnable]
    rule vp_v_np : span(I, K, V), span(K, J, NP) |- span(I, J, VP) #[learnable]
    rule vp_vp_pp : span(I, K, VP), span(K, J, PP) |- span(I, J, VP) #[learnable]
    rule vp_v : span(I, J, V) |- span(I, J, VP)
    rule pp_p_np : span(I, K, P), span(K, J, NP) |- span(I, J, PP) #[learnable]
    lexicon
        "the"       : Det = the       #[learnable]
        "a"         : Det = a         #[learnable]
        "dog"       : N   = dog       #[learnable]
        "cat"       : N   = cat       #[learnable]
        "telescope" : N   = telescope #[learnable]
        "saw"       : N   = saw       #[learnable]
        "saw"       : V   = saw       #[learnable]
        "runs"      : V   = runs      #[learnable]
        "with"      : P   = with
"""

# The grouping the oracle uses, written out by hand: each left-hand side
# with its productions, rules by name and entries by (word, category).
AMBIGUOUS_GROUPS: dict[str, list[str]] = {
    "S": ["s_np_vp"],
    "NP": ["np_det_n", "np_np_pp"],
    "VP": ["vp_v_np", "vp_vp_pp", "vp_v"],
    "PP": ["pp_p_np"],
    "Det": ["the/Det", "a/Det"],
    "N": ["dog/N", "cat/N", "telescope/N", "saw/N"],
    "V": ["saw/V", "runs/V"],
    "P": ["with/P"],
}
AMBIGUOUS_BINARY = {
    "s_np_vp": ("S", "NP", "VP"),
    "np_det_n": ("NP", "Det", "N"),
    "np_np_pp": ("NP", "NP", "PP"),
    "vp_v_np": ("VP", "V", "NP"),
    "vp_vp_pp": ("VP", "VP", "PP"),
    "pp_p_np": ("PP", "P", "NP"),
}
AMBIGUOUS_UNARY = {"vp_v": ("VP", "V")}
# The expanded lexicon in declaration order: the agenda's ``lex_weight_<i>``.
AMBIGUOUS_LEXICON = [
    "the/Det",
    "a/Det",
    "dog/N",
    "cat/N",
    "telescope/N",
    "saw/N",
    "saw/V",
    "runs/V",
    "with/P",
]
SENTENCES = [
    "the dog runs".split(),
    "the dog saw a cat".split(),
    "the dog runs with a cat".split(),
    "the dog saw the cat with a telescope".split(),
    "a saw saw the saw with the dog".split(),
    "the cat runs with a dog with the telescope".split(),
]

# A grammar whose every yield has exactly one derivation, with right
# recursion through ``an_adj_an``; used to sample a corpus and refit it.
UNAMBIGUOUS = """
object Term : FinSet 16
deduction U : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, VP, AN, Det, Adj, N, V, span, the, a, big, old, dog, cat, runs, sees
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S) #[learnable]
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP) #[learnable]
    rule np_det_an : span(I, K, Det), span(K, J, AN) |- span(I, J, NP) #[learnable]
    rule an_adj_n : span(I, K, Adj), span(K, J, N) |- span(I, J, AN) #[learnable]
    rule an_adj_an : span(I, K, Adj), span(K, J, AN) |- span(I, J, AN) #[learnable]
    rule vp_v : span(I, J, V) |- span(I, J, VP) #[learnable]
    rule vp_v_np : span(I, K, V), span(K, J, NP) |- span(I, J, VP) #[learnable]
    lexicon
        "the"  : Det = the  #[learnable]
        "a"    : Det = a    #[learnable]
        "big"  : Adj = big  #[learnable]
        "old"  : Adj = old  #[learnable]
        "dog"  : N   = dog  #[learnable]
        "cat"  : N   = cat  #[learnable]
        "runs" : V   = runs #[learnable]
        "sees" : V   = sees #[learnable]
"""

# The generating grammar: each left-hand side's productions, each with its
# right-hand side (categories, or one word) and probability.
GENERATOR: dict[str, list[tuple[str, tuple[str, ...], float]]] = {
    "S": [("s_np_vp", ("NP", "VP"), 1.0)],
    "NP": [("np_det_n", ("Det", "N"), 0.7), ("np_det_an", ("Det", "AN"), 0.3)],
    "AN": [("an_adj_n", ("Adj", "N"), 0.6), ("an_adj_an", ("Adj", "AN"), 0.4)],
    "VP": [("vp_v", ("V",), 0.35), ("vp_v_np", ("V", "NP"), 0.65)],
    "Det": [('"the"', ("the",), 0.55), ('"a"', ("a",), 0.45)],
    "Adj": [('"big"', ("big",), 0.5), ('"old"', ("old",), 0.5)],
    "N": [('"dog"', ("dog",), 0.65), ('"cat"', ("cat",), 0.35)],
    "V": [('"runs"', ("runs",), 0.3), ('"sees"', ("sees",), 0.7)],
}


def _parameter_names(system: DeductionSystem, deduction: str) -> dict[str, str]:
    """Map each agenda parameter to the name the checked search reads.

    Parameters
    ----------
    system : DeductionSystem
        The compiled deduction.
    deduction : str
        Its name.

    Returns
    -------
    dict[str, str]
        For ``_rule_module.rule_<r>.BASE``, ``<D>.rule.<r>:``; for
        ``_axiom_module.lex_weight_<i>``, ``<D>.lex.<i>``.
    """
    names: dict[str, str] = {}
    for name, _ in system.named_parameters():
        if name.startswith("_rule_module.rule_"):
            rule = name.removeprefix("_rule_module.rule_").removesuffix(".BASE")
            names[name] = f"{deduction}.rule.{rule}:"
        else:
            names[name] = f"{deduction}.lex.{name.rsplit('_', 1)[1]}"
    return names


def _randomize(system: DeductionSystem, seed: int) -> dict[str, float]:
    """Fill every parameter with a standard Normal draw.

    Parameters
    ----------
    system : DeductionSystem
        The compiled deduction.
    seed : int
        The seed.

    Returns
    -------
    dict[str, float]
        The values, by agenda parameter name.
    """
    generator = torch.Generator().manual_seed(seed)
    values: dict[str, float] = {}
    with torch.no_grad():
        for name, parameter in system.named_parameters():
            parameter.copy_(torch.randn((), generator=generator))
            values[name] = float(parameter)
    return values


def _free_weights(values: Mapping[str, float]) -> dict[str, float]:
    """The ambiguous grammar's free log-weights by production name.

    Parameters
    ----------
    values : Mapping[str, float]
        The parameter values by agenda name.

    Returns
    -------
    dict[str, float]
        Each production's free log-weight, zero when not learnable.
    """
    free = {name: 0.0 for group in AMBIGUOUS_GROUPS.values() for name in group}
    for name, value in values.items():
        if name.startswith("_rule_module.rule_"):
            free[name.removeprefix("_rule_module.rule_").removesuffix(".BASE")] = value
        else:
            free[AMBIGUOUS_LEXICON[int(name.rsplit("_", 1)[1])]] = value
    return free


def _log_probs(free: Mapping[str, float], groups: Mapping[str, list[str]]) -> dict:
    """Normalize free log-weights over the given groups, in float64.

    Parameters
    ----------
    free : Mapping[str, float]
        Each production's free log-weight.
    groups : Mapping[str, list[str]]
        The grouping to normalize over.

    Returns
    -------
    dict
        Each production's log-probability within its group.
    """
    out: dict[str, float] = {}
    for members in groups.values():
        peak = max(free[name] for name in members)
        total = peak + math.log(sum(math.exp(free[name] - peak) for name in members))
        for name in members:
            out[name] = free[name] - total
    return out


def _logaddexp(left: float, right: float) -> float:
    """``log(exp(left) + exp(right))``, with ``-inf`` the identity.

    Parameters
    ----------
    left : float
        A log weight.
    right : float
        A log weight.

    Returns
    -------
    float
        Their log-sum.
    """
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    peak = max(left, right)
    return peak + math.log(math.exp(left - peak) + math.exp(right - peak))


def _cky(
    tokens: Sequence[str],
    log_prob: Mapping[str, float],
    binary: Mapping[str, tuple[str, str, str]],
    unary: Mapping[str, tuple[str, str]],
    lexicon: Sequence[str],
    start: str,
) -> float:
    """The log inside probability of the start symbol by CKY.

    Parameters
    ----------
    tokens : Sequence[str]
        The sentence.
    log_prob : Mapping[str, float]
        Each production's log-probability.
    binary : Mapping[str, tuple[str, str, str]]
        Each binary rule's left-hand side and two right-hand categories.
    unary : Mapping[str, tuple[str, str]]
        Each unary rule's left-hand side and right-hand category; the
        unary rules form no cycle.
    lexicon : Sequence[str]
        The lexical productions as ``word/category``.
    start : str
        The start symbol.

    Returns
    -------
    float
        ``log P(start =>* tokens)``.
    """
    n = len(tokens)
    chart: dict[tuple[int, int], dict[str, float]] = {}

    def close(cell: dict[str, float]) -> dict[str, float]:
        """Add every unary derivation over a cell, to a fixed point."""
        closed = dict(cell)
        for _ in range(len(unary) + 1):
            updated = dict(cell)
            for name, (parent, child) in unary.items():
                if child in closed:
                    updated[parent] = _logaddexp(
                        updated.get(parent, -math.inf), log_prob[name] + closed[child]
                    )
            closed = updated
        return closed

    for position, token in enumerate(tokens):
        cell: dict[str, float] = {}
        for production in lexicon:
            word, category = production.split("/")
            if word == token:
                cell[category] = _logaddexp(
                    cell.get(category, -math.inf), log_prob[production]
                )
        chart[(position, position + 1)] = close(cell)
    for width in range(2, n + 1):
        for left in range(n - width + 1):
            right = left + width
            cell = {}
            for split in range(left + 1, right):
                for name, (parent, first, second) in binary.items():
                    one = chart[(left, split)].get(first)
                    two = chart[(split, right)].get(second)
                    if one is None or two is None:
                        continue
                    cell[parent] = _logaddexp(
                        cell.get(parent, -math.inf), log_prob[name] + one + two
                    )
            chart[(left, right)] = close(cell)
    return chart[(0, n)].get(start, -math.inf)


def _agenda(system: DeductionSystem, tokens: Sequence[str]) -> float:
    """The agenda chart's goal weight.

    Parameters
    ----------
    system : DeductionSystem
        The compiled deduction.
    tokens : Sequence[str]
        The sentence.

    Returns
    -------
    float
        The weight.
    """
    return float(system(list(tokens)).goal_weight().detach())


# ---------------------------------------------------------------------------
# Inside weights against CKY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_both_evaluators_compute_the_cky_inside_probability(seed: int) -> None:
    program = loads(AMBIGUOUS)
    system = program.deductions["G"]
    values = _randomize(system, seed)
    log_prob = _log_probs(_free_weights(values), AMBIGUOUS_GROUPS)
    checked_names = _parameter_names(system, "G")
    parameters = {checked_names[name]: value for name, value in values.items()}
    for tokens in SENTENCES:
        expected = _cky(
            tokens, log_prob, AMBIGUOUS_BINARY, AMBIGUOUS_UNARY, AMBIGUOUS_LEXICON, "S"
        )
        assert math.isfinite(expected)
        # The agenda runs in float32; the checked search, like the oracle,
        # in float64 on the same parameter values.
        assert _agenda(system, tokens) == pytest.approx(expected, abs=2e-5)
        # The checked search enumerates derivations without sharing
        # subresults, so it runs on the sentences of up to six tokens.
        if len(tokens) <= 6:
            checked = run_deduction(
                program.qiec, "G", tokens=tokens, parameters=parameters
            )
            assert checked.weight == pytest.approx(expected, abs=1e-9)


def test_normalizing_over_the_wrong_grouping_fails_the_cky_comparison() -> None:
    program = loads(AMBIGUOUS)
    system = program.deductions["G"]
    values = _randomize(system, 0)
    free = _free_weights(values)
    everything = {
        "all": [name for group in AMBIGUOUS_GROUPS.values() for name in group]
    }
    by_kind = {
        "rules": [name for name in free if "/" not in name],
        "words": [name for name in free if "/" in name],
    }
    for wrong in (everything, by_kind):
        log_prob = _log_probs(free, wrong)
        mismatches = [
            abs(
                _agenda(system, tokens)
                - _cky(
                    tokens,
                    log_prob,
                    AMBIGUOUS_BINARY,
                    AMBIGUOUS_UNARY,
                    AMBIGUOUS_LEXICON,
                    "S",
                )
            )
            for tokens in SENTENCES
        ]
        assert min(mismatches) > 0.1


def test_each_left_hand_sides_probabilities_sum_to_one() -> None:
    system = loads(AMBIGUOUS).deductions["G"]
    groups = system.production_groups()
    assert [
        (group.label, [production.label for production in group.productions])
        for group in groups
    ] == [
        ("S", ["s_np_vp"]),
        ("NP", ["np_det_n", "np_np_pp"]),
        ("VP", ["vp_v_np", "vp_vp_pp", "vp_v"]),
        ("PP", ["pp_p_np"]),
        ("Det", ['"the"', '"a"']),
        ("N", ['"dog"', '"cat"', '"telescope"', '"saw"']),
        ("V", ['"saw"', '"runs"']),
        ("P", ['"with"']),
    ]
    for seed in range(5):
        values = _randomize(system, seed)
        expected = _log_probs(_free_weights(values), AMBIGUOUS_GROUPS)
        table = system.production_log_probs()
        for (label, members), group in zip(
            AMBIGUOUS_GROUPS.items(), groups, strict=True
        ):
            probabilities = torch.stack(list(table[label].values())).double().exp()
            assert float(probabilities.sum().detach()) == pytest.approx(1.0, abs=1e-6)
            for name, production in zip(members, group.productions, strict=True):
                assert float(table[label][production.label].detach()) == pytest.approx(
                    expected[name], abs=1e-6
                )


def test_a_wildcard_entry_joins_the_group_of_each_candidate_category() -> None:
    source = """
object Term : FinSet 8
deduction W : Term -> Term [semiring=LogProb, start=S, depth=2, normalize=lhs]
    atoms S, A, B, span, x, y
    rule s_a : span(I, K, A), span(K, J, B) |- span(I, J, S) #[learnable]
    rule s_b : span(I, K, B), span(K, J, A) |- span(I, J, S) #[learnable]
    lexicon
        "x" : {A, B} = x
        "y" : B = y #[learnable]
"""
    program = loads(source)
    system = program.deductions["W"]
    assert [
        (group.label, [production.label for production in group.productions])
        for group in system.production_groups()
    ] == [("S", ["s_a", "s_b"]), ("A", ['"x"']), ("B", ['"x"', '"y"'])]
    values = _randomize(system, 3)
    lex = [values[f"_axiom_module.lex_weight_{index}"] for index in range(3)]
    rules = [values["_rule_module.rule_s_a.BASE"], values["_rule_module.rule_s_b.BASE"]]
    rule_norm = math.log(sum(math.exp(value) for value in rules))
    b_norm = math.log(math.exp(lex[1]) + math.exp(lex[2]))
    # "x y": S -> A B with x as A (alone in its group) and y as B.
    expected_xy = rules[0] - rule_norm + 0.0 + lex[2] - b_norm
    # "x x": S -> A B or S -> B A, one x an A and the other a B.
    expected_xx = math.log(
        math.exp(rules[0] - rule_norm + lex[1] - b_norm)
        + math.exp(rules[1] - rule_norm + lex[1] - b_norm)
    )
    parameters = {
        _parameter_names(system, "W")[name]: value for name, value in values.items()
    }
    for tokens, expected in ((["x", "y"], expected_xy), (["x", "x"], expected_xx)):
        assert _agenda(system, tokens) == pytest.approx(expected, abs=2e-5)
        checked = run_deduction(program.qiec, "W", tokens=tokens, parameters=parameters)
        assert checked.weight == pytest.approx(expected, abs=1e-9)


def test_structured_categories_and_items_beyond_spans_group_alike() -> None:
    # A categorial lexicon with ground application rules, and a multiple
    # context-free grammar whose ``p`` items are no spans.
    categorial = """
object Term : FinSet 8
deduction C : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, cat, runs
    rule np_app : span(I, K, Fwd(NP, N)), span(K, J, N) |- span(I, J, NP) #[learnable]
    rule s_app : span(I, K, NP), span(K, J, Bwd(S, NP)) |- span(I, J, S) #[learnable]
    lexicon
        "the"  : Fwd(NP, N) = the  #[learnable]
        "dog"  : N          = dog  #[learnable]
        "cat"  : N          = cat  #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]
"""
    copy = """
object Term : FinSet 8
deduction Copy : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, A, B, p, span, a, b
    rule pair_a : span(I1, J1, A), span(I2, J2, A) |- p(I1, J1, I2, J2) #[learnable]
    rule pair_b : span(I2, J2, B), span(I1, J1, B) |- p(I1, J1, I2, J2) #[learnable]
    rule grow_a : span(J1, K1, A), p(I1, J1, I2, J2), span(J2, K2, A) |- p(I1, K1, I2, K2) #[learnable]
    rule grow_b : p(I1, J1, I2, J2), span(J2, K2, B), span(J1, K1, B) |- p(I1, K1, I2, K2) #[learnable]
    rule copy : p(I, J, J, K) |- span(I, K, S) #[learnable]
    lexicon
        "a" : A = a
        "b" : B = b
"""
    cases = [
        (
            categorial,
            "C",
            [("NP", 1), ("S", 1), ("Fwd(NP, N)", 1), ("N", 2), ("Bwd(S, NP)", 1)],
            [["the", "dog", "runs"], ["the", "cat", "runs"]],
        ),
        (
            copy,
            "Copy",
            [("p(_, _, _, _)", 4), ("S", 1), ("A", 1), ("B", 1)],
            [["a", "a"], ["a", "b", "a", "b"], ["b", "a", "a", "b", "a", "a"]],
        ),
    ]
    for source, name, shape, sentences in cases:
        program = loads(source)
        system = program.deductions[name]
        assert [
            (group.label, len(group.productions))
            for group in system.production_groups()
        ] == shape
        values = _randomize(system, 5)
        parameters = {
            _parameter_names(system, name)[key]: value for key, value in values.items()
        }
        for tokens in sentences:
            agenda = _agenda(system, tokens)
            checked = run_deduction(
                program.qiec, name, tokens=tokens, parameters=parameters
            ).weight
            assert math.isfinite(agenda)
            assert checked == pytest.approx(agenda, abs=2e-5)
    # The categorial grammar's one derivation of "the dog runs" multiplies
    # one production per group: N picks "dog" among two nouns.
    system = loads(categorial).deductions["C"]
    values = _randomize(system, 5)
    nouns = [values["_axiom_module.lex_weight_1"], values["_axiom_module.lex_weight_2"]]
    expected = nouns[0] - math.log(sum(math.exp(value) for value in nouns))
    assert _agenda(system, ["the", "dog", "runs"]) == pytest.approx(expected, abs=2e-5)


# ---------------------------------------------------------------------------
# Maximum likelihood on yields sampled from a known grammar
# ---------------------------------------------------------------------------


def _sample(
    rng: random.Random, symbol: str, counts: Counter[tuple[str, str]]
) -> list[str]:
    """Expand a symbol top-down under `GENERATOR`, counting productions.

    Parameters
    ----------
    rng : random.Random
        The source of randomness.
    symbol : str
        The category to expand.
    counts : Counter[tuple[str, str]]
        The ``(left-hand side, production)`` uses, updated in place.

    Returns
    -------
    list[str]
        The yield.
    """
    productions = GENERATOR[symbol]
    name, rhs, _ = rng.choices(productions, weights=[p for *_, p in productions])[0]
    counts[(symbol, name)] += 1
    if name.startswith('"'):
        return list(rhs)
    return [word for child in rhs for word in _sample(rng, child, counts)]


def test_maximum_likelihood_recovers_the_generating_grammar() -> None:
    rng = random.Random(20261111)
    counts: Counter[tuple[str, str]] = Counter()
    corpus = [_sample(rng, "S", counts) for _ in range(250)]
    uses = Counter[str]()
    for (symbol, _), count in counts.items():
        uses[symbol] += count
    system = loads(UNAMBIGUOUS).deductions["U"]
    # Every yield has one derivation, so the log-likelihood is a sum of
    # per-left-hand-side multinomial log-likelihoods of the production
    # counts that derivation fixes: it is bounded above by its value at
    # the relative frequencies, which are the unique maximizer for every
    # left-hand side the corpus uses. That bound is the identifiability
    # the model has: each group used at least once is identified.
    relative = {
        (symbol, name): counts[(symbol, name)] / uses[symbol]
        for symbol, productions in GENERATOR.items()
        for name, _, _ in productions
    }
    bound = -sum(
        count * math.log(relative[key]) for key, count in counts.items() if count
    )
    history = adam_fit_deduction(system, corpus, steps=250, lr=0.1)
    # The loss is bounded below by the negative log-likelihood at the
    # relative frequencies and settles at it.
    assert min(history) >= bound - 1e-2
    assert history[-1] == pytest.approx(bound, abs=2e-2)
    assert max(history[-20:]) - min(history[-20:]) < 1e-2
    table = system.production_log_probs()
    for symbol, productions in GENERATOR.items():
        for name, _, truth in productions:
            fitted = math.exp(float(table[symbol][name]))
            assert fitted == pytest.approx(relative[(symbol, name)], abs=5e-3)
            # Four binomial standard errors at the true probability over
            # the left-hand side's uses in the sampled derivations.
            tolerance = 4 * math.sqrt(truth * (1 - truth) / uses[symbol])
            assert abs(fitted - truth) <= tolerance


def test_unnormalized_fitting_runs_away_where_normalized_fitting_settles() -> None:
    rng = random.Random(7)
    counts: Counter[tuple[str, str]] = Counter()
    corpus = [_sample(rng, "S", counts) for _ in range(30)]
    free = loads(UNAMBIGUOUS.replace(", normalize=lhs", "")).deductions["U"]
    history = adam_fit_deduction(free, corpus, steps=150, lr=0.1)
    # Without normalization the loss falls below zero, which no
    # probability model of the corpus can reach.
    assert history[-1] < -10.0
    assert history[-1] < history[-50] - 5.0


# ---------------------------------------------------------------------------
# The partition function of a consistent grammar
# ---------------------------------------------------------------------------

RECURSIVE = """
object Term : FinSet 4
deduction R : Term -> Term [semiring=LogProb, start=S, depth=2, normalize=lhs]
    atoms S, X, span, a, b
    rule more : span(I, K, X), span(K, J, S) |- span(I, J, S) #[learnable]
    rule last : span(I, J, X) |- span(I, J, S) #[learnable]
    lexicon
        "a" : X = a #[learnable]
        "b" : X = b #[learnable]
"""


def test_the_partition_function_of_a_consistent_grammar_is_at_most_one() -> None:
    system = loads(RECURSIVE).deductions["R"]
    stop = 0.6
    with torch.no_grad():
        for name, parameter in system.named_parameters():
            if name.endswith("rule_more.BASE"):
                parameter.fill_(math.log(1 - stop))
            elif name.endswith("rule_last.BASE"):
                parameter.fill_(math.log(stop))
            else:
                parameter.fill_(0.0)
    # Every nonempty string over {a, b} has one derivation; strings of
    # length n carry total probability (1 - stop)^(n - 1) * stop, so the
    # strings up to length L carry 1 - (1 - stop)^L.
    total = 0.0
    for length in range(1, 8):
        mass = sum(
            math.exp(_agenda(system, list(tokens)))
            for tokens in itertools.product("ab", repeat=length)
        )
        assert mass == pytest.approx((1 - stop) ** (length - 1) * stop, rel=1e-4)
        total += mass
        assert total <= 1.0 + 1e-6
    assert total == pytest.approx(1 - (1 - stop) ** 7, rel=1e-4)


def test_a_finite_language_carries_probability_one_and_free_weights_do_not() -> None:
    finite = """
object Term : FinSet 8
deduction F : Term -> Term [semiring=LogProb, start=S, depth=2, normalize=lhs]
    atoms S, NP, Det, N, V, span, the, a, cat, dog, runs, sleeps
    rule s : span(I, K, NP), span(K, J, V) |- span(I, J, S) #[learnable]
    rule np : span(I, K, Det), span(K, J, N) |- span(I, J, NP) #[learnable]
    lexicon
        "the" : Det = the #[learnable]
        "a" : Det = a #[learnable]
        "cat" : N = cat #[learnable]
        "dog" : N = dog #[learnable]
        "runs" : V = runs #[learnable]
        "sleeps" : V = sleeps #[learnable]
"""
    sentences = [
        list(words)
        for words in itertools.product(["the", "a"], ["cat", "dog"], ["runs", "sleeps"])
    ]
    for source, expected in (
        (finite, 1.0),
        (finite.replace(", normalize=lhs", ""), 8.0),
    ):
        program = loads(source)
        system = program.deductions["F"]
        values = _randomize(system, 11)
        parameters = {
            _parameter_names(system, "F")[name]: value for name, value in values.items()
        }
        agenda = sum(math.exp(_agenda(system, tokens)) for tokens in sentences)
        checked = sum(
            math.exp(
                run_deduction(
                    program.qiec, "F", tokens=tokens, parameters=parameters
                ).weight
            )
            for tokens in sentences
        )
        if expected == 1.0:
            assert agenda == pytest.approx(1.0, abs=1e-5)
            assert checked == pytest.approx(1.0, abs=1e-12)
        else:
            # The free weights' partition is the product of each group's
            # summed exponentiated weights, here far from one.
            assert abs(agenda - 1.0) > 0.5
            assert checked == pytest.approx(agenda, rel=1e-5)


# ---------------------------------------------------------------------------
# The unnormalized mode is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("option", ["", ", normalize=none"])
def test_the_unnormalized_mode_keeps_free_weights(option: str) -> None:
    source = AMBIGUOUS.replace(", normalize=lhs", option)
    program = loads(source)
    system = program.deductions["G"]
    values = _randomize(system, 4)
    free = _free_weights(values)
    tokens = "the dog runs".split()
    # One derivation; each production contributes its free log-weight.
    expected = (
        free["s_np_vp"] + free["np_det_n"] + free["the/Det"] + free["dog/N"]
    ) + (free["vp_v"] + free["runs/V"])
    assert _agenda(system, tokens) == pytest.approx(expected, abs=2e-5)
    parameters = {
        _parameter_names(system, "G")[name]: value for name, value in values.items()
    }
    checked = run_deduction(program.qiec, "G", tokens=tokens, parameters=parameters)
    assert checked.weight == pytest.approx(expected, abs=1e-9)
    assert [group.label for group in system.production_groups()][:2] == ["S", "NP"]
    with pytest.raises(ValueError, match="normalize=lhs"):
        system.production_log_probs()


def test_the_unnormalized_mode_keeps_category_keyed_weights() -> None:
    categorial = """
object Term : FinSet 16
deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, big, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the"  : Fwd(NP, N) = the
        "big"  : Fwd(N, N)  = big
        "dog"  : N          = dog
        "runs" : Bwd(S, NP) = runs
"""
    system = loads(categorial).deductions["AB"]
    system("the big dog runs".split())
    assert sorted(name for name, _ in system.named_parameters()) == [
        "_rule_module.rule_bwd_app.X_TSatom_SSE_Y_TSatom_SNPE",
        "_rule_module.rule_fwd_app.X_TSatom_SNE_Y_TSatom_SNE",
        "_rule_module.rule_fwd_app.X_TSatom_SNPE_Y_TSatom_SNE",
    ]
    with pytest.raises(ValueError, match="binds 'X', 'Y'"):
        system.production_groups()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

_HEAD = "object Term : FinSet 8\n"


def _agenda_compile(source: str) -> None:
    """Compile only the source's deduction with the agenda compiler.

    Parameters
    ----------
    source : str
        The QVR text, declaring one deduction.
    """
    module = parse(source)
    (decl,) = (item for item in module.statements if isinstance(item, DeductionDecl))
    Compiler(module)._compile_deduction(decl)


REFUSED = [
    (
        "category variables",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, NP, Fwd, span, the, dog
    rule fwd : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    lexicon
        "the" : Fwd(S, NP) = the
        "dog" : NP = dog
""",
        3,
        "binds 'X', 'Y' outside the chart positions",
    ),
    (
        "a premise variable under a ground conclusion",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, NP, span, dog
    rule any : span(I, K, X), span(K, J, X) |- span(I, J, S) #[learnable]
    lexicon
        "dog" : NP = dog
""",
        3,
        "binds 'X'",
    ),
    (
        "logical forms",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, NP, span, dog
    rule up : span(I, J, NP, F) |- span(I, J, S, F) #[learnable]
    lexicon
        "dog" : NP = dog
""",
        1,
        "carry logical forms",
    ),
    (
        "a bounded rule",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, NP, span, dog
    rule up : span(I, J, NP) |- span(I, J, S) #[learnable, bounded]
    lexicon
        "dog" : NP = dog
""",
        3,
        "declares 'bounded'",
    ),
    (
        "a parented rule",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, NP, span, dog
    rule up : span(I, J, NP) |- span(I, J, S) #[learnable]
    rule again : span(I, J, NP) |- span(I, J, S) #[parent=up]
    lexicon
        "dog" : NP = dog
""",
        4,
        "declares 'parent'",
    ),
    (
        "axioms given as input",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, NP, span
    rule up : span(I, J, NP) |- span(I, J, S) #[learnable]
""",
        1,
        "takes its axioms as input",
    ),
    (
        "a semiring that reads no weights",
        """deduction D : Term -> Term [semiring=Boolean, start=S, normalize=lhs]
    atoms S, NP, span, dog
    rule up : span(I, J, NP) |- span(I, J, S)
    lexicon
        "dog" : NP = dog
""",
        1,
        "Boolean semiring reads no production weights",
    ),
    (
        "a lexicon category that is not ground",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, span, dog
    rule up : span(I, K, S), span(K, J, S) |- span(I, J, S) #[learnable]
    lexicon
        "dog" : NP = dog
""",
        5,
        "mentions 'NP', not a declared atom",
    ),
    (
        "a duplicated entry",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms S, NP, span, dog
    rule up : span(I, J, NP) |- span(I, J, S) #[learnable]
    lexicon
        "dog" : NP = dog
        "dog" : NP = dog
""",
        6,
        "gives 'dog' category NP twice",
    ),
    (
        "an unknown mode",
        """deduction D : Term -> Term [semiring=LogProb, start=S, normalize=rhs]
    atoms S, NP, span, dog
    rule up : span(I, J, NP) |- span(I, J, S) #[learnable]
    lexicon
        "dog" : NP = dog
""",
        1,
        "normalize=rhs is not one of lhs, none",
    ),
]


@pytest.mark.parametrize(
    ("body", "line", "message"),
    [(body, line, message) for _, body, line, message in REFUSED],
    ids=[label for label, *_ in REFUSED],
)
def test_a_deduction_whose_grouping_is_not_fixed_is_refused_on_both_paths(
    body: str, line: int, message: str
) -> None:
    source = _HEAD + body
    with pytest.raises(CompileError) as agenda:
        _agenda_compile(source)
    assert message in str(agenda.value)
    assert agenda.value.line == line + 1
    with pytest.raises(QiecDiagnosticError) as checked:
        lower_qvr_to_qiec(parse(source), file_path="normalized.qvr")
    assert checked.value.code == "qiec-program"
    assert message in checked.value.message
    assert checked.value.line == line + 1
