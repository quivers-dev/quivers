"""A marginalized Bernoulli-family latent weighted from its logits scores exactly.

A zero-inflated count is a two-component mixture whose weights are the
logistic function of a linear predictor ``eta`` and its complement. Written
as ``ContinuousBernoulli(probs=sigmoid(eta))`` at a ``marginalize`` head, the
compiled runtime forms ``log1p(-sigmoid(eta))`` in single precision, which
cancels once ``eta`` passes about 10. Written as
``ContinuousBernoulli(logits=eta)`` (or ``Bernoulli(logits=eta)``), both
execution paths weigh the atoms by ``log sigmoid(-eta)`` and
``log sigmoid(eta)``, computed in log space.

Each joint is checked against an exact double-precision computation of the
zero-inflated Poisson likelihood at moderate and extreme predictors. The
compiled runtime's tolerance is fixed in advance from single-precision
resolution: 64 units in the last place of a float32 at the joint's
magnitude, ``2**-18`` relative (the joint is a sum of a few dozen float32
terms, each correct to an ulp or two). The reference machine computes in
double precision and is held to ``1e-12`` relative. The probability-scale
spelling of the same model is shown to fall outside the compiled tolerance
at a large predictor, so the tolerance detects the loss it guards against.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads

#: The compiled runtime's relative tolerance: 64 float32 ulps.
_FLOAT32_TOLERANCE = 2.0**-18

#: The reference machine's relative tolerance.
_FLOAT64_TOLERANCE = 1e-12

_RATE = 3.0

#: Counts: zeros reachable from both components and positive counts
#: reachable only from the active one. The positive counts are large
#: enough that the inactive component's ``Poisson(0)`` contributes nothing
#: within the tolerance however the runtime scores a zero rate.
_COUNTS = (0, 3, 0, 5)

_ETAS = (-30.0, -15.0, 0.0, 15.0, 30.0)


def _program(head: str, lets: str = "") -> str:
    """A zero-inflated Poisson program with one shared predictor.

    Parameters
    ----------
    head : str
        The marginalized latent's family call.
    lets : str
        Bindings placed before the block.

    Returns
    -------
    str
        The QVR source.
    """
    return (
        "object Resp : FinSet 4\n"
        "object Val : Real 1\n"
        "program p : Resp -> Val\n"
        "    sample eta <- Normal(loc=0.0, scale=10.0)\n"
        f"{lets}"
        f"    marginalize z : Resp <- {head}\n"
        f"        let gated_rate = z * {_RATE}\n"
        "        observe y : Resp <- Poisson(rate=gated_rate)\n"
        "    return eta\n"
        "export p\n"
    )


def _f64(value: float) -> torch.Tensor:
    """A number as a double-precision tensor.

    Parameters
    ----------
    value : float
        The number.

    Returns
    -------
    torch.Tensor
        The number, in double precision.
    """
    return torch.tensor(value, dtype=torch.float64)


def _exact(eta: float) -> float:
    """The joint in double precision, from the mixture's closed form.

    Parameters
    ----------
    eta : float
        The predictor: the logit of the active component's weight.

    Returns
    -------
    float
        The prior's log density plus each row's log marginal.
    """
    logit = _f64(eta)
    counts = torch.tensor(_COUNTS, dtype=torch.float64)
    active = torch.nn.functional.logsigmoid(logit) + D.Poisson(_f64(_RATE)).log_prob(
        counts
    )
    inactive = torch.where(
        counts == 0,
        torch.nn.functional.logsigmoid(-logit).expand(counts.shape),
        torch.tensor(-math.inf, dtype=torch.float64),
    )
    prior = D.Normal(_f64(0.0), _f64(10.0)).log_prob(logit)
    return float(prior + torch.logaddexp(active, inactive).sum())


def _compiled(source: str, eta: float) -> float:
    """The compiled runtime's joint.

    Parameters
    ----------
    source : str
        The program.
    eta : float
        The predictor.

    Returns
    -------
    float
        The joint, in the runtime's single precision.
    """
    program = loads(source)
    return float(
        program.morphism.log_joint(
            torch.zeros(1, 1),
            {
                "eta": torch.tensor([[eta]]),
                "y": torch.tensor(_COUNTS, dtype=torch.float32),
            },
        )
    )


def _reference(source: str, eta: float) -> float:
    """The reference machine's joint.

    Parameters
    ----------
    source : str
        The program.
    eta : float
        The predictor.

    Returns
    -------
    float
        The joint, in double precision.
    """
    return loads(source).run("p", data={"y": _COUNTS}, sites={"eta": eta}).log_joint


_LOGIT_HEADS = {
    "continuous-bernoulli": "ContinuousBernoulli(logits=eta)",
}


@pytest.mark.parametrize("eta", _ETAS)
@pytest.mark.parametrize("head", sorted(_LOGIT_HEADS))
def test_compiled_runtime_scores_the_exact_mixture_from_logits(
    head: str, eta: float
) -> None:
    expected = _exact(eta)
    compiled = _compiled(_program(_LOGIT_HEADS[head]), eta)
    assert compiled == pytest.approx(expected, rel=_FLOAT32_TOLERANCE)


@pytest.mark.parametrize("eta", _ETAS)
@pytest.mark.parametrize("head", sorted(_LOGIT_HEADS))
def test_reference_machine_scores_the_exact_mixture_from_logits(
    head: str, eta: float
) -> None:
    expected = _exact(eta)
    reference = _reference(_program(_LOGIT_HEADS[head]), eta)
    assert reference == pytest.approx(expected, rel=_FLOAT64_TOLERANCE)


@pytest.mark.parametrize("eta", _ETAS)
def test_complement_logits_weigh_the_other_atom(eta: float) -> None:
    # The weight `1 - sigmoid(eta)` is `sigmoid(-eta)`: written as logits
    # `-eta`, the block scores the mixture at the negated predictor, under a
    # prior symmetric about zero.
    source = _program("ContinuousBernoulli(logits=neg)", "    let neg = 0.0 - eta\n")
    expected = _exact(-eta)
    assert _compiled(source, eta) == pytest.approx(expected, rel=_FLOAT32_TOLERANCE)
    assert _reference(source, eta) == pytest.approx(expected, rel=_FLOAT64_TOLERANCE)


def test_logits_read_as_probabilities_would_score_another_density() -> None:
    # A renderer or runtime that read the logits as the probability would
    # clamp them into the unit interval and score a different mixture.
    misread = _exact(0.0)
    compiled = _compiled(_program("ContinuousBernoulli(logits=eta)"), 0.7)
    assert abs(compiled - _exact(0.7)) < abs(compiled - misread)
    probability = _compiled(_program("ContinuousBernoulli(probs=eta)"), 0.7)
    assert abs(compiled - probability) > 1e-2


def test_the_probability_scale_weight_falls_outside_the_tolerance() -> None:
    # The issue's spelling: an active weight `1 - sigmoid(eta)` formed on the
    # probability scale. At eta = 15 the float32 `sigmoid(15)` is within an
    # ulp of one, so the complement keeps almost none of its digits, and the
    # tolerance above rejects the joint the compiled runtime scores; the
    # logit spelling of the same weight passes it.
    eta = 15.0
    source = _program(
        "ContinuousBernoulli(probs=pi_z)", "    let pi_z = 1.0 - sigmoid(eta)\n"
    )
    expected = _exact(-eta)
    compiled = _compiled(source, eta)
    assert abs(compiled - expected) > _FLOAT32_TOLERANCE * abs(expected)
