"""Bernoulli takes its log-odds directly as `logits`.

A logistic likelihood written `Bernoulli(logits=eta)` is scored from the
log-odds on both execution paths, so it keeps its precision where the
probability saturates: the compiled runtime through
`torch.distributions.Bernoulli(logits=...)` and the reference machine
through ``log p = -softplus(-eta)`` and ``log (1 - p) = -softplus(eta)``.
Both are pinned to the float64 `torch.distributions` density. The same
likelihood written through a single-precision `sigmoid` loses a tenth of
a nat at a log-odds of 15, which the compiled tolerance here rejects.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.compiler import CompileError

_LOGITS = """\
object Resp : FinSet 1
object Val : Real 1

program p : Resp -> Val
    sample eta <- Normal(loc=0.0, scale=10.0)
    observe y : Resp <- Bernoulli(logits=eta)
    return eta

export p
"""

_SIGMOID = """\
object Resp : FinSet 1
object Val : Real 1

program p : Resp -> Val
    sample eta <- Normal(loc=0.0, scale=10.0)
    let prob = sigmoid(eta)
    observe y : Resp <- Bernoulli(probs=prob)
    return eta

export p
"""

_LATENT = """\
object One : FinSet 1

program q : One -> One
    sample z <- Bernoulli(logits=2.0)
    return z

export q
"""

_POINTS = [(eta, y) for eta in (-30.0, -15.0, 0.3, 15.0, 30.0) for y in (0.0, 1.0)]


def _exact(eta: float, y: float) -> float:
    """The float64 joint at one point.

    Parameters
    ----------
    eta : float
        The log-odds.
    y : float
        The outcome.

    Returns
    -------
    float
        The prior on ``eta`` plus the Bernoulli log mass of ``y``.
    """
    value = torch.tensor(eta, dtype=torch.float64)
    prior = D.Normal(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(10.0, dtype=torch.float64)
    )
    return float(
        prior.log_prob(value)
        + D.Bernoulli(logits=value).log_prob(torch.tensor(y, dtype=torch.float64))
    )


def _compiled(source: str, eta: float, y: float) -> float:
    """The compiled runtime's joint at one point.

    Parameters
    ----------
    source : str
        The program.
    eta : float
        The log-odds.
    y : float
        The outcome.

    Returns
    -------
    float
        The log joint.
    """
    program = loads(source)
    return float(
        program.morphism.log_joint(
            torch.zeros(1, 1), {"eta": torch.tensor([[eta]]), "y": torch.tensor([y])}
        )
    )


@pytest.mark.parametrize(("eta", "y"), _POINTS)
def test_compiled_runtime_scores_the_logits(eta: float, y: float) -> None:
    assert _compiled(_LOGITS, eta, y) == pytest.approx(_exact(eta, y), rel=1e-5)


@pytest.mark.parametrize(("eta", "y"), _POINTS)
def test_reference_machine_scores_the_logits(eta: float, y: float) -> None:
    run = loads(_LOGITS).run("p", data={"y": (bool(y),)}, sites={"eta": eta})
    assert run.log_joint == pytest.approx(_exact(eta, y), rel=1e-9)


def test_the_sigmoid_route_fails_the_compiled_tolerance() -> None:
    # What the compiled check rejects: the same likelihood scored through a
    # float32 probability is off by more than a tenth of a nat at eta = 15.
    exact = _exact(15.0, 0.0)
    assert abs(_compiled(_SIGMOID, 15.0, 0.0) - exact) > 0.1
    assert _compiled(_SIGMOID, 15.0, 0.0) != pytest.approx(exact, rel=1e-5)


def test_logits_read_as_probabilities_would_score_differently() -> None:
    # A path that read the log-odds 0.3 as the probability 0.3 misses the
    # oracle by more than the compiled tolerance.
    misread = float(
        D.Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(10.0, dtype=torch.float64),
        ).log_prob(torch.tensor(0.3, dtype=torch.float64))
        + math.log(0.3)
    )
    assert misread != pytest.approx(_exact(0.3, 1.0), rel=1e-5)


def test_naming_probs_and_logits_together_is_refused() -> None:
    with pytest.raises(CompileError, match="exactly one complete parameter schema"):
        loads(_LOGITS.replace("logits=eta", "probs=0.5, logits=eta"))


def test_a_latent_drawn_from_logits_has_the_logistic_frequency() -> None:
    # 4000 seeded draws from Bernoulli(logits=2): the success frequency
    # sits within four standard errors of sigmoid(2).
    program = loads(_LATENT)
    draws = [bool(program.run("q", seed=seed).value) for seed in range(4000)]
    expected = 1.0 / (1.0 + math.exp(-2.0))
    error = math.sqrt(expected * (1.0 - expected) / len(draws))
    assert abs(sum(draws) / len(draws) - expected) < 4.0 * error
    run = program.run("q", sites={"z": True})
    assert run.log_joint == pytest.approx(-math.log1p(math.exp(-2.0)), rel=1e-12)
