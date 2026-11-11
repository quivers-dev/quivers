"""Production probabilities a program supplies to a normalized grammar.

A program writes ``parse(D, corpus, Plate, ("A", p_A), ...)``: the
deduction, normalized per left-hand side, reads each left-hand side's
production probabilities from the program's values instead of its
learned weights, so a prior on them (a Dirichlet per left-hand side) has
a posterior that MCMC infers. The corpus is one input of one run, one
sentence per object of the plate. These tests check the joint against a
hand computation on both execution paths, the gradient the compiled path
sends to the sites, the conjugate posterior under NUTS, the Python
interface, the located refusals, and the refusal of every transpile
target.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest
import torch

from quivers.dsl import loads, parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.inference import MCMC, NUTSKernel
from quivers.inference.trace import trace
from quivers.transpile import UnsupportedConstruct, available_targets, transpile

# A grammar of three-word sentences, and an adverb category no rule uses,
# whose probabilities the corpus thus says nothing about.
GRAMMAR = """
object Term : FinSet 16
object Weight : Real 1

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, VP, Det, N, V, Adv, span, the, a, cat, dog, sleeps, runs, fast, slowly
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S)
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP)
    rule vp_v : span(I, J, V) |- span(I, J, VP)
    lexicon
        "the"    : Det = the
        "a"      : Det = a
        "cat"    : N   = cat
        "dog"    : N   = dog
        "sleeps" : V   = sleeps
        "runs"   : V   = runs
        "fast"   : Adv = fast
        "slowly" : Adv = slowly
"""

PRIOR = {"Det": [2.0, 3.0], "N": [1.0, 1.0], "V": [1.5, 0.5], "Adv": [2.0, 1.0]}
WORDS = {"Det": ["the", "a"], "N": ["cat", "dog"], "V": ["sleeps", "runs"]}


def _program(rows: int, prior: dict[str, list[float]] = PRIOR) -> str:
    """A Bayesian PCFG over a corpus of ``rows`` sentences.

    Parameters
    ----------
    rows : int
        The corpus size.
    prior : dict[str, list[float]]
        Each left-hand side's Dirichlet concentration.

    Returns
    -------
    str
        The QVR source.
    """
    draws = "\n".join(
        f"    sample p_{label.lower()} <- Dirichlet(concentration={values})"
        for label, values in prior.items()
    )
    bindings = ", ".join(f'("{label}", p_{label.lower()})' for label in prior)
    return (
        GRAMMAR
        + f"""
object Sentence : FinSet {rows}

program bayes : Term -> Weight
{draws}
    let chart = parse(PCFG, corpus, Sentence, {bindings})
    score evidence = chart.goal_weight()
    return evidence

export bayes
"""
    )


def _corpus(rows: int, truth: dict[str, list[float]], seed: int) -> list[list[str]]:
    """Sample sentences ``Det N V`` from known word probabilities.

    Parameters
    ----------
    rows : int
        The number of sentences.
    truth : dict[str, list[float]]
        Each category's word probabilities, in `WORDS` order.
    seed : int
        The seed.

    Returns
    -------
    list[list[str]]
        The sentences.
    """
    rng = random.Random(seed)
    return [
        [rng.choices(WORDS[label], weights=truth[label])[0] for label in WORDS]
        for _ in range(rows)
    ]


def _joint(corpus: list[list[str]], values: dict[str, list[float]]) -> float:
    """The joint by hand: Dirichlet log densities plus the corpus likelihood.

    Parameters
    ----------
    corpus : list[list[str]]
        The sentences.
    values : dict[str, list[float]]
        Each left-hand side's probabilities.

    Returns
    -------
    float
        ``sum log Dir(p_A; alpha_A) + sum_s log P(s)``, where each
        sentence's one derivation multiplies one word probability per
        category (the rules are alone in their groups).
    """
    prior = sum(
        float(
            torch.distributions.Dirichlet(
                torch.tensor(PRIOR[label], dtype=torch.float64)
            ).log_prob(torch.tensor(values[label], dtype=torch.float64))
        )
        for label in PRIOR
    )
    likelihood = sum(
        math.log(values[label][WORDS[label].index(word)])
        for sentence in corpus
        for label, word in zip(WORDS, sentence, strict=True)
    )
    return prior + likelihood


VALUES = {"Det": [0.7, 0.3], "N": [0.45, 0.55], "V": [0.2, 0.8], "Adv": [0.6, 0.4]}


def test_the_joint_is_the_prior_density_plus_the_log_inside_probability() -> None:
    corpus = _corpus(5, {"Det": [0.5, 0.5], "N": [0.5, 0.5], "V": [0.5, 0.5]}, 3)
    program = loads(_program(5))
    expected = _joint(corpus, VALUES)
    sites = {f"p_{label.lower()}": value for label, value in VALUES.items()}
    run = program.run(
        "bayes",
        data={"corpus": tuple(tuple(sentence) for sentence in corpus)},
        sites={name: tuple(value) for name, value in sites.items()},
        seed=0,
    )
    assert run.log_joint == pytest.approx(expected, abs=1e-9)
    recorded = trace(
        program.morphism,
        torch.zeros(1, 1),
        observations={
            "corpus": corpus,
            **{name: torch.tensor(value) for name, value in sites.items()},
        },
    )
    # The compiled path runs in float32.
    assert float(recorded.log_joint.detach()) == pytest.approx(expected, abs=1e-4)


def test_the_compiled_path_sends_the_likelihoods_gradient_to_the_sites() -> None:
    corpus = _corpus(6, {"Det": [0.5, 0.5], "N": [0.5, 0.5], "V": [0.5, 0.5]}, 4)
    program = loads(_program(6))
    sites = {
        f"p_{label.lower()}": torch.tensor(value, requires_grad=True)
        for label, value in VALUES.items()
    }
    joint = program.morphism.log_joint(
        torch.zeros(1, 1), {"corpus": corpus, **sites}
    ).sum()
    joint.backward()
    for label, words in WORDS.items():
        counts = Counter(sentence[list(WORDS).index(label)] for sentence in corpus)
        for position, word in enumerate(words):
            value = VALUES[label][position]
            # d/dp of n log p + (alpha - 1) log p, plus the derivative of
            # the renormalization log(sum p), which is -N_A at a point on
            # the simplex.
            alpha = PRIOR[label][position]
            total = sum(counts.values())
            expected = (counts[word] + alpha - 1) / value - total
            grad = sites[f"p_{label.lower()}"].grad
            assert grad is not None
            assert float(grad[position]) == pytest.approx(expected, rel=1e-4)


@pytest.mark.slow
def test_nuts_recovers_the_conjugate_dirichlet_posterior() -> None:
    truth = {"Det": [0.75, 0.25], "N": [0.35, 0.65], "V": [0.6, 0.4]}
    rows = 40
    corpus = _corpus(rows, truth, 20261111)
    program = loads(_program(rows))
    torch.manual_seed(0)
    result = MCMC(
        NUTSKernel(step_size=0.3, max_tree_depth=4),
        num_warmup=200,
        num_samples=300,
        num_chains=2,
    ).run(program.morphism, torch.zeros(1, 1), {"corpus": corpus})
    for label in PRIOR:
        counts = (
            Counter(sentence[list(WORDS).index(label)] for sentence in corpus)
            if label in WORDS
            else Counter()
        )
        words = WORDS.get(label, ["fast", "slowly"])
        posterior = torch.tensor(
            [PRIOR[label][i] + counts[word] for i, word in enumerate(words)],
            dtype=torch.float64,
        )
        exact_mean = posterior / posterior.sum()
        total = float(posterior.sum())
        exact_sd = torch.sqrt(exact_mean * (1 - exact_mean) / (total + 1))
        samples = result.samples[f"p_{label.lower()}"].reshape(-1, 2).double()
        ess = result.ess[f"p_{label.lower()}"]
        assert float(ess.min()) >= 100
        # The Monte Carlo standard error of the mean is at most
        # sd / sqrt(100) given at least 100 effective draws.
        tolerance = 4 * exact_sd / math.sqrt(100)
        assert torch.all((samples.mean(dim=0) - exact_mean).abs() <= tolerance)
        if label in truth:
            # The posterior concentrates near the generating probabilities:
            # within four posterior standard deviations.
            assert torch.all(
                (exact_mean - torch.tensor(truth[label], dtype=torch.float64)).abs()
                <= 4 * exact_sd
            )
        else:
            # The corpus never uses an adverb: its posterior is its prior.
            prior = torch.tensor(PRIOR[label], dtype=torch.float64)
            assert torch.allclose(exact_mean, prior / prior.sum())


def test_the_python_interface_takes_the_same_probabilities() -> None:
    system = loads(GRAMMAR).deductions["PCFG"]
    probabilities = {label: torch.tensor(value) for label, value in VALUES.items()}
    assert [group.label for group in system.production_groups()] == [
        "S",
        "NP",
        "VP",
        "Det",
        "N",
        "V",
        "Adv",
    ]
    corpus = [["the", "cat", "runs"], ["a", "dog", "sleeps"]]
    chart = system.corpus(corpus, probabilities)
    expected = math.log(0.7 * 0.45 * 0.8) + math.log(0.3 * 0.55 * 0.2)
    assert float(chart.goal_weight().detach()) == pytest.approx(expected, abs=1e-5)
    table = system.production_log_probs(probabilities)
    assert float(table["V"]['"runs"'].detach()) == pytest.approx(math.log(0.8))
    assert float(table["S"]["s_np_vp"].detach()) == 0.0
    with pytest.raises(ValueError, match=r"missing \['Adv'\]"):
        system(
            ["the", "cat", "runs"],
            {k: v for k, v in probabilities.items() if k != "Adv"},
        )
    with pytest.raises(ValueError, match="one production, whose probability is one"):
        system(["the", "cat", "runs"], {**probabilities, "S": torch.tensor([1.0])})
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        system(["the", "cat", "runs"], {**probabilities, "V": torch.ones(3) / 3})
    with pytest.raises(ValueError, match="negative entry"):
        system(
            ["the", "cat", "runs"], {**probabilities, "V": torch.tensor([1.2, -0.2])}
        )
    free = loads(GRAMMAR.replace(", normalize=lhs", "")).deductions["PCFG"]
    with pytest.raises(ValueError, match="normalize=lhs"):
        free(["the", "cat", "runs"], probabilities)


def test_a_sentence_and_a_corpus_agree() -> None:
    sentence_program = (
        GRAMMAR
        + """
program one : Term -> Weight
    sample p_det <- Dirichlet(concentration=[1.0, 1.0])
    sample p_n <- Dirichlet(concentration=[1.0, 1.0])
    sample p_v <- Dirichlet(concentration=[1.0, 1.0])
    sample p_adv <- Dirichlet(concentration=[1.0, 1.0])
    let chart = parse(PCFG, sentence, ("Det", p_det), ("N", p_n), ("V", p_v), ("Adv", p_adv))
    score evidence = chart.goal_weight()
    return evidence

export one
"""
    )
    program = loads(sentence_program)
    sites = {f"p_{label.lower()}": tuple(value) for label, value in VALUES.items()}
    run = program.run(
        "one", data={"sentence": ("a", "dog", "runs")}, sites=sites, seed=0
    )
    assert run.value == pytest.approx(math.log(0.3 * 0.55 * 0.8), abs=1e-12)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

_BINDINGS = '("Det", p_det), ("N", p_n), ("V", p_v), ("Adv", p_adv)'
_DRAWS = """    sample p_det <- Dirichlet(concentration=[1.0, 1.0])
    sample p_n <- Dirichlet(concentration=[1.0, 1.0])
    sample p_v <- Dirichlet(concentration=[1.0, 1.0])
    sample p_adv <- Dirichlet(concentration=[1.0, 1.0])
"""


def _refused(grammar: str, draws: str, call: str) -> tuple[QiecDiagnosticError, int]:
    """Lower a program calling ``parse`` and return the refusal.

    Parameters
    ----------
    grammar : str
        The grammar's source.
    draws : str
        The program's sample steps.
    call : str
        The ``parse`` call.

    Returns
    -------
    tuple[QiecDiagnosticError, int]
        The diagnostic and the line of the ``parse`` call.
    """
    head = (
        grammar
        + "\nobject Sentence : FinSet 2\n\nprogram bayes : Term -> Weight\n"
        + draws
    )
    source = (
        head
        + f"    let chart = {call}\n"
        + "    score evidence = chart.goal_weight()\n    return evidence\n\nexport bayes\n"
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        lower_qvr_to_qiec(parse(source), file_path="priors.qvr")
    return caught.value, head.count("\n") + 1


@pytest.mark.parametrize(
    ("grammar", "draws", "call", "message"),
    [
        (
            GRAMMAR.replace(", normalize=lhs", ""),
            _DRAWS,
            f"parse(PCFG, corpus, Sentence, {_BINDINGS})",
            "whose weights are free log-weights",
        ),
        (
            GRAMMAR,
            _DRAWS,
            'parse(PCFG, corpus, Sentence, ("Det", p_det), ("N", p_n), ("V", p_v))',
            "binds no probabilities for Adv",
        ),
        (
            GRAMMAR,
            _DRAWS,
            f'parse(PCFG, corpus, Sentence, {_BINDINGS}, ("Pro", p_v))',
            "'Pro', which is no left-hand side of 'PCFG'",
        ),
        (
            GRAMMAR,
            _DRAWS,
            f'parse(PCFG, corpus, Sentence, {_BINDINGS}, ("S", p_v))',
            "'S', which has one production in 'PCFG', whose probability is one",
        ),
        (
            GRAMMAR,
            _DRAWS,
            f'parse(PCFG, corpus, Sentence, {_BINDINGS}, ("V", p_n))',
            "binds 'V' twice",
        ),
        (
            GRAMMAR,
            _DRAWS.replace(
                "p_v <- Dirichlet(concentration=[1.0, 1.0])",
                "p_v <- Dirichlet(concentration=[1.0, 1.0, 1.0])",
            ),
            f"parse(PCFG, corpus, Sentence, {_BINDINGS})",
            '\'V\' has 2 productions ("sleeps", "runs")',
        ),
        (
            GRAMMAR,
            _DRAWS.replace(
                "p_v <- Dirichlet(concentration=[1.0, 1.0])",
                "p_v : Rule2 <- Normal(loc=0.0, scale=1.0)",
            ),
            f"parse(PCFG, corpus, Sentence, {_BINDINGS})",
            "drawn from a family whose support is real",
        ),
        (
            GRAMMAR,
            _DRAWS,
            "parse(PCFG, corpus, Sentence, p_det)",
            "pairs after its input and optional plate",
        ),
    ],
    ids=[
        "unnormalized",
        "missing",
        "unknown",
        "singleton",
        "twice",
        "size",
        "support",
        "malformed",
    ],
)
def test_bindings_that_do_not_fit_the_grammar_are_located_errors(
    grammar: str, draws: str, call: str, message: str
) -> None:
    error, line = _refused(grammar + "object Rule2 : FinSet 2\n", draws, call)
    assert error.code == "qiec-program"
    assert message in error.message
    assert error.line == line


def test_every_transpile_target_refuses_the_search_by_name() -> None:
    module = parse(_program(4))
    for target in available_targets():
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(module, target=target)
        assert caught.value.kinds == ["qiec:capability:search:PCFG__weighted_run"], (
            target,
            caught.value.kinds,
        )
