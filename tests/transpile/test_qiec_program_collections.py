"""Program-plan regressions for checked collection computations."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from quivers.dsl import parse
from quivers.transpile import transpile


_DYNAMIC_TARGETS = (
    "pyro",
    "numpyro",
    "pymc",
    "edward2",
    "turing",
    "gen",
    "webppl",
    "church",
)

_ROOT = Path(__file__).resolve().parents[2]

_TRAVERSE_PROGRAM = """\
define shift(value : Real, offset : Real) : Real !{} =
    return value + offset

object Row : FinSet 3
object Value : Real 1

program shifted : Row -> Value
    sample offset <- Normal(loc=0.0, scale=1.0)
    let locations <- traverse(
        [1.0, 2.0, 3.0],
        value -> shift(value, offset),
    )
    observe y : Row <- Normal(loc=locations, scale=1.0)
    return offset

export shifted
"""

_INT_PROGRAM = """\
object Row : FinSet 1
object Value : Real 1

program cast_choice : Row -> Value
    let choice = int(observed_choice)
    return observed_choice

export cast_choice
"""


@pytest.mark.parametrize("target", _DYNAMIC_TARGETS)
def test_program_traverse_flattens_into_the_dynamic_target_plan(target: str) -> None:
    """A fixed traversal is ordinary sequenced QIEC, including in a program."""
    output = transpile(parse(_TRAVERSE_PROGRAM), target=target)
    assert output


@pytest.mark.parametrize(
    ("target", "spelling"),
    (
        ("pyro", "torch.as_tensor(observed_choice).long()"),
        ("numpyro", "jnp.int32(observed_choice)"),
        ("pymc", "pymc.pytensorf.pt.cast(observed_choice"),
        ("edward2", "tf.cast(observed_choice,tf.int32)"),
    ),
)
def test_python_targets_render_real_to_int(target: str, spelling: str) -> None:
    """The checked ``real_to_int`` primitive keeps its integer result type."""
    output = transpile(parse(_INT_PROGRAM), target=target).decode()
    ast.parse(output)
    assert spelling in output


@pytest.mark.parametrize("target", ("pyro", "numpyro"))
def test_lexical_uncertainty_rsa_transpiles_to_python_targets(target: str) -> None:
    """The released collection/effect example crosses the complete boundary."""
    source = (_ROOT / "docs/examples/qiec/lexical-uncertainty-rsa.qvr").read_text()
    output = transpile(parse(source), target=target).decode()
    ast.parse(output)
    stacked = {"pyro": "torch.stack(", "numpyro": "jnp.stack("}[target]
    assert f"states={stacked}" in output.replace(" ", "")
    if target == "pyro":
        import pyro
        import torch

        namespace = {"pyro": pyro, "torch": torch}
        exec(output, namespace)
        model = pyro.condition(
            namespace["model"], data={"raw_alpha": torch.tensor(0.0)}
        )
        trace = pyro.poutine.trace(model).get_trace(
            listener_choice=torch.tensor(0.0),
            heard_some=torch.tensor([1.0]),
        )
        trace.compute_log_prob()
        assert float(trace.log_prob_sum()) == pytest.approx(-2.89819356, abs=1e-6)
        posterior = trace.nodes["_RETURN"]["value"]
        assert tuple(float(value) for value in posterior) == pytest.approx(
            (0.29586356, 0.70413644), abs=1e-6
        )


def _lexical_uncertainty_reference(
    alpha: float, epsilon: float
) -> tuple[list[list[float]], list[float]]:
    """Equations (34) to (37) of Bergen, Levy and Goodman (2016), Figure 5.

    Written from the paper, independently of the QVR source: uniform
    world prior, costs (1, 1, 5) for ("some", "all", null), the three
    enrichments of "some", and lexica marginalized only at L1. A false
    cell holds ``epsilon``; ``epsilon = 0`` is the paper's exact model.

    Parameters
    ----------
    alpha : float
        The speaker rationality (the paper's lambda).
    epsilon : float
        The value of a false lexicon cell.

    Returns
    -------
    tuple[list[list[float]], list[float]]
        ``S1("some" | w, L)`` indexed ``[lexicon][world]``, and
        ``L1(w | "some")``.
    """
    prior = [0.5, 0.5]
    costs = [1.0, 1.0, 5.0]
    some_meanings = [[1.0, 1.0], [epsilon, 1.0], [1.0, epsilon]]
    all_meaning = [1.0, epsilon]
    speakers: list[list[float]] = []
    for some_meaning in some_meanings:
        lexicon = [some_meaning, all_meaning, [1.0, 1.0]]
        literal = []
        for meaning in lexicon:
            masses = [meaning[w] * prior[w] for w in range(2)]
            literal.append([mass / sum(masses) for mass in masses])
        row = []
        for world in range(2):
            scores = [
                math.exp(alpha * (math.log(literal[u][world]) - costs[u]))
                if literal[u][world] > 0.0
                else 0.0
                for u in range(3)
            ]
            row.append(scores[0] / sum(scores))
        speakers.append(row)
    masses = [prior[w] * sum(s[w] for s in speakers) / 3.0 for w in range(2)]
    return speakers, [mass / sum(masses) for mass in masses]


@pytest.mark.parametrize("alpha", (0.25, 0.5, 1.0, 2.0, 4.0))
def test_lexical_uncertainty_rsa_matches_the_paper_recursion(alpha: float) -> None:
    """The checked recursion equals an independent transcription of the
    paper's equations, and the 1e-6 relaxation stays close to exact zeros."""
    from quivers.dsl import load

    program = load(str(_ROOT / "docs/examples/qiec/lexical-uncertainty-rsa.qvr"))
    speakers = program.run("lexical_speakers", alpha, 1e-6).value
    listener = program.run("lexical_listener", alpha, 1e-6).value
    reference_speakers, reference_listener = _lexical_uncertainty_reference(
        alpha, 1e-6
    )
    for row, reference_row in zip(speakers, reference_speakers, strict=True):
        assert tuple(row) == pytest.approx(tuple(reference_row), rel=1e-9, abs=1e-12)
    assert tuple(listener) == pytest.approx(tuple(reference_listener), abs=1e-12)
    # A false cell of 1e-6 enters S1 as (1e-6)^alpha, so the gap to the
    # exact model shrinks with alpha: about 8e-4 at alpha = 0.5, 1.1e-5
    # at alpha = 1, and under 1e-7 from alpha = 2.
    _, exact_listener = _lexical_uncertainty_reference(alpha, 0.0)
    assert tuple(listener) == pytest.approx(tuple(exact_listener), abs=1e-3)
    if alpha >= 1.0:
        assert tuple(listener) == pytest.approx(tuple(exact_listener), abs=2e-5)
    # The specificity implicature of Figure 5: "some" favours
    # some-but-not-all at L1.
    assert exact_listener[1] > exact_listener[0]


_LIST_LITERAL_PROGRAM = """\
object Step : FinSet 3
object Val : Real 1

program p : Step -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    let ys = [1.0, 2.0, 3.0]
    let doubled = map(ys, y -> 2.0 * y)
    let shifted = ys * mu + [0.5, 0.0, -0.5]
    score s = sum(doubled) + sum(shifted)
    return mu

export p
"""


@pytest.mark.parametrize("target", ("pyro", "numpyro"))
def test_a_list_literal_is_one_tensor_for_arithmetic_and_for_map(target: str) -> None:
    """A list literal's checked type is a tensor, so the Python targets
    build one array from it, which elementwise arithmetic and a mapped
    collection both read; Pyro scores the program's joint."""
    output = transpile(parse(_LIST_LITERAL_PROGRAM), target=target).decode()
    ast.parse(output)
    stacked = {"pyro": "torch.stack(", "numpyro": "jnp.stack("}[target]
    compact = output.replace(" ", "")
    assert f"ys={stacked}" in compact
    assert f"shifted=(ys*mu)+{stacked}" in compact
    if target == "pyro":
        import pyro
        import torch

        namespace = {"pyro": pyro, "torch": torch}
        exec(output, namespace)
        mu = 0.7
        model = pyro.condition(namespace["model"], data={"mu": torch.tensor(mu)})
        trace = pyro.poutine.trace(model).get_trace()
        trace.compute_log_prob()
        prior = float(torch.distributions.Normal(0.0, 1.0).log_prob(torch.tensor(mu)))
        score = 2.0 * (1.0 + 2.0 + 3.0) + mu * (1.0 + 2.0 + 3.0) + (0.5 + 0.0 - 0.5)
        assert float(trace.log_prob_sum()) == pytest.approx(prior + score, abs=1e-5)
