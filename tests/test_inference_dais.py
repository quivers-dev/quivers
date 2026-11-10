"""Tests for the `AutoDAIS` objective path.

Each DAIS draw carries the extended-space log-density of the trajectory
that produced it, so ``log p(z) - log_density`` is that trajectory's
log importance weight ``log w`` (Zhang et al. 2021, Eq. 15), with
``E[w] = p(y)``. The tests pin:

* the pairing: ``ELBO`` scores each draw with its own trajectory's
  density, and the guide refuses to score arbitrary sites;
* the change of variables: with vanishing step sizes the density is
  the base guide's density at the draw, Jacobian included, on a
  positive-support latent;
* validity: ``E[w] = 1`` for a normalized model without observations,
  so ``E[log w] <= 0``;
* differentiation through the leapfrog map: the autograd derivative of
  a fixed-seed estimate equals its central finite difference.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import pytest
import torch

from quivers.continuous import Draw, Euclidean, MonadicProgram, make_fixed_normal
from quivers.continuous.inline import FixedDistribution, make_fixed_halfnormal
from quivers.core import Unit
from quivers.inference import (
    ELBO,
    RWS,
    AutoDAIS,
    AutoNormalGuide,
    IWAEBound,
    Objective,
    Reparameterized,
    ScoreFunction,
    StickingTheLanding,
)

_REAL = Euclidean(name="R", dim=1)
_X = torch.zeros(1, 1)


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _program(prior: FixedDistribution) -> MonadicProgram:
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[Draw(names=("z",), morphism=prior)],
        return_vars=(),
    )


def _dais(
    model: MonadicProgram,
    loc: float,
    scale: float,
    *,
    num_steps: int = 3,
    step_size: float = 0.3,
) -> AutoDAIS:
    base = AutoNormalGuide(model, observed_names=set())
    with torch.no_grad():
        getattr(base, "loc_z").fill_(loc)
        getattr(base, "log_scale_z").fill_(math.log(scale))
    return AutoDAIS(
        base, model, observations={}, num_steps=num_steps, init_step_size=step_size
    )


def _log_weights(model: MonadicProgram, guide: AutoDAIS, n: int) -> torch.Tensor:
    """``log w = log p(z) - log_density`` for ``n`` independent draws."""
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n):
            (draw,) = guide.rsample_strata(_X)
            log_p = model.log_joint(_X, draw.sites)
            values.append((log_p - draw.log_density).reshape(()))
    return torch.stack(values)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_log_prob_refuses_arbitrary_sites() -> None:
    model = _program(make_fixed_normal(0.0, 1.0, _REAL))
    guide = _dais(model, 0.5, 0.8)
    with pytest.raises(ValueError, match="no density at arbitrary sites"):
        guide.log_prob(_X, {"z": torch.tensor([0.0])})


def test_elbo_scores_each_draw_with_its_own_trajectory() -> None:
    model = _program(make_fixed_halfnormal(1.0, _REAL))
    guide = _dais(model, -0.3, 0.6)
    torch.manual_seed(7)
    loss = float(ELBO()(model, guide, _X, {}).detach())
    torch.manual_seed(7)
    (draw,) = guide.rsample_strata(_X)
    log_w = model.log_joint(_X, draw.sites) - draw.log_density
    assert math.isclose(loss, -float(log_w.detach()), rel_tol=1e-12, abs_tol=1e-12)


def test_density_at_vanishing_step_size_is_the_base_density() -> None:
    """With step sizes near zero the chain does not move, so the draw's
    density is the base guide's (constrained-space) density at the
    draw: a mismatch in the bijector's log-Jacobian would show here."""
    model = _program(make_fixed_halfnormal(1.0, _REAL))
    guide = _dais(model, -0.3, 0.6, step_size=1e-9)
    torch.manual_seed(3)
    with torch.no_grad():
        for _ in range(5):
            (draw,) = guide.rsample_strata(_X)
            base_density = guide.base.log_prob(_X, draw.sites)
            assert torch.allclose(draw.log_density, base_density, atol=1e-6)


@pytest.mark.parametrize(
    "make",
    [
        lambda: ELBO(estimator=StickingTheLanding()),
        lambda: ELBO(estimator=ScoreFunction()),
        lambda: IWAEBound(num_particles=2),
        lambda: RWS(num_particles=2),
    ],
    ids=["stl", "score", "iwae-dreg", "rws"],
)
def test_estimators_needing_a_density_at_fixed_points_raise(
    make: Callable[[], Objective],
) -> None:
    objective = make()
    model = _program(make_fixed_normal(0.0, 1.0, _REAL))
    guide = _dais(model, 0.5, 0.8)
    with pytest.raises(ValueError, match="Reparameterized"):
        objective(model, guide, _X, {})


def test_importance_weighted_bound_with_reparameterized_estimator() -> None:
    torch.manual_seed(0)
    model = _program(make_fixed_normal(0.0, 1.0, _REAL))
    guide = _dais(model, 0.5, 0.8)
    with torch.no_grad():
        bound = -float(
            IWAEBound(num_particles=4, estimator=Reparameterized())(
                model, guide, _X, {}
            )
        )
    assert math.isfinite(bound)


# ---------------------------------------------------------------------------
# Validity of the bound
# ---------------------------------------------------------------------------


def _check_unbiased_weight(n: int, tolerance_in_se: float) -> None:
    for prior, loc, scale in [
        (make_fixed_normal(0.0, 1.0, _REAL), 0.5, 0.8),
        (make_fixed_halfnormal(1.0, _REAL), -0.3, 0.6),
    ]:
        model = _program(prior)
        guide = _dais(model, loc, scale)
        torch.manual_seed(0)
        log_w = _log_weights(model, guide, n)
        w = log_w.exp()
        se = float(w.std()) / math.sqrt(n)
        assert abs(float(w.mean()) - 1.0) <= tolerance_in_se * se, (
            float(w.mean()),
            se,
        )
        # E[log w] <= log E[w] = 0.
        assert float(log_w.mean()) <= 4.0 * float(log_w.std()) / math.sqrt(n)


def test_importance_weight_is_unbiased_for_the_evidence() -> None:
    """``E[w] = p(y) = 1`` for a normalized prior without observations."""
    _check_unbiased_weight(n=300, tolerance_in_se=4.0)


@pytest.mark.slow
def test_importance_weight_is_unbiased_for_the_evidence_full() -> None:
    _check_unbiased_weight(n=4000, tolerance_in_se=4.0)


# ---------------------------------------------------------------------------
# Differentiation through the trajectory
# ---------------------------------------------------------------------------


def test_gradient_through_the_trajectory_matches_finite_differences() -> None:
    """With the seed fixed, the ELBO estimate is a smooth function of
    the step size, the base location, and the schedule; autograd must
    match its central difference, which requires differentiating the
    bridge gradients inside each leapfrog step."""
    model = _program(make_fixed_halfnormal(1.0, _REAL))
    guide = _dais(model, -0.3, 0.6, step_size=0.4)
    targets = [
        (guide.log_step_size, 1),
        (getattr(guide.base, "loc_z"), 0),
        (guide.beta_increments_raw, 0),
    ]

    def estimate() -> torch.Tensor:
        torch.manual_seed(11)
        return -ELBO()(model, guide, _X, {})

    guide.zero_grad()
    estimate().backward()
    step = 1e-6
    for parameter, index in targets:
        assert parameter.grad is not None
        analytic = float(parameter.grad.reshape(-1)[index])
        values: list[float] = []
        for sign in (1.0, -1.0):
            with torch.no_grad():
                parameter.view(-1)[index] += sign * step
            values.append(float(estimate().detach()))
            with torch.no_grad():
                parameter.view(-1)[index] -= sign * step
        numeric = (values[0] - values[1]) / (2 * step)
        assert math.isclose(analytic, numeric, rel_tol=1e-5, abs_tol=1e-7), (
            analytic,
            numeric,
        )
