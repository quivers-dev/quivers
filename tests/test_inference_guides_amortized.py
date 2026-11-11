"""Tests for `AmortizedGuide`.

Cover:

1. On a linear-Gaussian model, whose exact posterior mean is a linear
   function of the data and whose posterior variance is constant, a
   linear Normal encoder fitted by SVI recovers the mean map and the
   variance within tolerances fixed in advance, and its posterior means
   on held-out data match the exact ones (amortization).
2. The ELBO gradient with respect to the encoder's weights matches
   central finite differences of the same loss at fixed noise.
3. The guide's log-density matches a hand computation, the bijector's
   Jacobian included, for a positive site under a real encoder; plated
   local latents take one encoder row per plate row; draws per run
   follow the input's rows.
4. Every binding error is refused: an unknown site, a support or event
   size that does not fit, a missing or misaligned datum, and an
   encoder whose density is a path approximation.
5. The guide composes with `AutoGuideList`, and every objective and
   estimator takes it.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Normal

from quivers.continuous import (
    Draw,
    Euclidean,
    MonadicProgram,
    PlateDraw,
    make_fixed_halfnormal,
    make_fixed_normal,
)
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.program_steps import Observe
from quivers.core import Unit
from quivers.dsl import loads
from quivers.inference import (
    ELBO,
    RWS,
    SVI,
    AmortizedGuide,
    AutoGuideList,
    AutoNormalGuide,
    ChiVI,
    DReGsBound,
    IWAEBound,
    RenyiBound,
    VRIWAEBound,
)
from quivers.inference.estimators import (
    DoublyReparameterized,
    ScoreFunction,
    StickingTheLanding,
)
from quivers.inference.objectives import Objective


# ---------------------------------------------------------------------------
# The linear-Gaussian model
# ---------------------------------------------------------------------------

_W = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
_SIGMA = 0.5
# z ~ N(0, I), y | z ~ N(W z, sigma^2 I): the posterior precision is
# I + W^T W / sigma^2 = diag(5, 17), diagonal because W's columns are
# orthogonal, and the posterior mean is Sigma W^T y / sigma^2.
_POST_VAR = torch.tensor([1.0 / 5.0, 1.0 / 17.0])
_MEAN_MAP = torch.diag(_POST_VAR) @ _W.t() / _SIGMA**2


def _decoder() -> ConditionalNormal:
    """The fixed linear-Gaussian likelihood ``y ~ N(W z, sigma^2 I)``."""
    decoder = ConditionalNormal(Euclidean(name="Z", dim=2), Euclidean(name="Y", dim=3))
    with torch.no_grad():
        linear = decoder.param_source.linear
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[:3] = _W
        linear.bias[3:] = math.log(_SIGMA)
    decoder.requires_grad_(False)
    return decoder


def _linear_gaussian() -> MonadicProgram:
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="Z", dim=2)),
            ),
            Observe(names=("y",), morphism=_decoder(), args=("z",)),
        ],
        return_vars=(),
    )


def _data(n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 2, generator=generator)
    return z @ _W.t() + _SIGMA * torch.randn(n, 3, generator=generator)


def _encoder() -> ConditionalNormal:
    """A linear Normal encoder ``Y -> Z``: it represents the exact
    posterior with mean ``M y`` and constant log-scales."""
    return ConditionalNormal(Euclidean(name="Y", dim=3), Euclidean(name="Z", dim=2))


def _encoder_params(
    encoder: ConditionalNormal, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return encoder._get_params(y)


def test_fitted_encoder_recovers_the_exact_posterior_map() -> None:
    torch.manual_seed(0)
    model = _linear_gaussian()
    n = 200
    y = _data(n, seed=1)
    x = torch.zeros(n, 1)
    encoder = _encoder()
    guide = AmortizedGuide(
        model, {"y"}, {"z": (encoder, "y")}, x=x, observations={"y": y}
    )
    optim = torch.optim.Adam(guide.parameters(), lr=0.02)
    svi = SVI(model, guide, optim, ELBO(num_particles=4))
    for _ in range(1500):
        svi.step(x, {"y": y})
    for group in optim.param_groups:
        group["lr"] = 0.002
    for _ in range(1000):
        svi.step(x, {"y": y})

    weight = encoder.param_source.linear.weight.detach()
    bias = encoder.param_source.linear.bias.detach()
    # Tolerances fixed before fitting: 0.05 on the mean map's entries
    # and offset, 5% on the posterior standard deviations.
    assert (weight[:2] - _MEAN_MAP).abs().max() < 0.05, weight[:2]
    assert bias[:2].abs().max() < 0.05
    assert weight[2:].abs().max() < 0.05
    sd = bias[2:].exp()
    assert ((sd / _POST_VAR.sqrt()) - 1.0).abs().max() < 0.05, sd

    # Amortization: on held-out data the encoder's posterior means are
    # the exact ones.
    held_out = _data(500, seed=2)
    mean, scale = _encoder_params(encoder, held_out)
    assert (mean.detach() - held_out @ _MEAN_MAP.t()).abs().max() < 0.15
    assert ((scale.detach() / _POST_VAR.sqrt()) - 1.0).abs().max() < 0.05
    # Draws on held-out data follow its rows.
    guide.observe({"y": held_out})
    draws = guide.rsample(torch.zeros(500, 1))
    assert draws["z"].shape == (500, 2)


def test_elbo_gradient_matches_finite_differences() -> None:
    model = _linear_gaussian()
    y = _data(6, seed=3)
    x = torch.zeros(6, 1)
    encoder = _encoder()
    guide = AmortizedGuide(
        model, {"y"}, {"z": (encoder, "y")}, x=x, observations={"y": y}
    )
    objective = ELBO(num_particles=2)

    def loss() -> torch.Tensor:
        torch.manual_seed(7)
        return objective(model, guide, x, {"y": y})

    weight = encoder.param_source.linear.weight
    value = loss()
    (grad,) = torch.autograd.grad(value, weight)
    h = 1e-2
    for row, col in ((0, 0), (1, 2), (3, 1), (2, 0)):
        with torch.no_grad():
            weight[row, col] += h
            plus = float(loss())
            weight[row, col] -= 2 * h
            minus = float(loss())
            weight[row, col] += h
        numeric = (plus - minus) / (2 * h)
        assert numeric == pytest.approx(float(grad[row, col]), rel=2e-2, abs=2e-2)


# ---------------------------------------------------------------------------
# Densities and shapes
# ---------------------------------------------------------------------------


def _scale_program() -> MonadicProgram:
    """A positive latent ``tau ~ HalfNormal(1)`` per run."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("tau",),
                morphism=make_fixed_halfnormal(1.0, Euclidean(name="R", dim=1)),
            )
        ],
        return_vars=(),
    )


def test_log_density_includes_the_bijector_jacobian() -> None:
    model = _scale_program()
    encoder = ConditionalNormal(Euclidean(name="Y", dim=3), Euclidean(name="U", dim=1))
    y = torch.randn(4, 3)
    x = torch.zeros(4, 1)
    guide = AmortizedGuide(model, set(), {"tau": (encoder, "y")}, observations={"y": y})
    tau = guide.rsample(x)["tau"]
    assert tau.shape == (4,)
    assert bool((tau > 0).all())
    mean, scale = encoder._get_params(y)
    # tau = exp(u) with u ~ N(mean, scale): log q(tau) = log N(log tau)
    # - log tau.
    expected = Normal(mean[:, 0], scale[:, 0]).log_prob(tau.log()) - tau.log()
    assert torch.allclose(guide.log_prob(x, {"tau": tau}), expected, atol=1e-5)


def _plate_program() -> MonadicProgram:
    """Four plated local latents ``z_i ~ N(0, 1)`` in one run."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=PlateDraw(
                    4, make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1))
                ),
            )
        ],
        return_vars=(),
    )


def test_plated_local_latents_take_one_row_each() -> None:
    model = _plate_program()
    encoder = ConditionalNormal(Euclidean(name="Y", dim=3), Euclidean(name="U", dim=1))
    y = torch.randn(4, 3)
    x = torch.zeros(1, 1)
    guide = AmortizedGuide(model, set(), {"z": (encoder, "y")}, observations={"y": y})
    z = guide.rsample(x)["z"]
    assert z.shape == (4,)
    mean, scale = encoder._get_params(y)
    expected = Normal(mean[:, 0], scale[:, 0]).log_prob(z).sum().reshape(1)
    assert torch.allclose(guide.log_prob(x, {"z": z}), expected, atol=1e-5)
    # Two runs, each with its own plate of data.
    y_runs = torch.randn(2, 4, 3)
    guide.observe({"y": y_runs})
    z_runs = guide.rsample(torch.zeros(2, 1))["z"]
    assert z_runs.shape == (2, 4)
    mean, scale = encoder._get_params(y_runs.reshape(8, 3))
    per_run = Normal(mean[:, 0], scale[:, 0]).log_prob(z_runs.reshape(8)).reshape(2, 4)
    assert torch.allclose(
        guide.log_prob(torch.zeros(2, 1), {"z": z_runs}), per_run.sum(-1), atol=1e-5
    )


# ---------------------------------------------------------------------------
# Binding errors
# ---------------------------------------------------------------------------


def test_bindings_are_checked() -> None:
    model = _linear_gaussian()
    y = _data(5, seed=0)
    x = torch.zeros(5, 1)
    with pytest.raises(ValueError, match=r"\['w'\] are not latent sites"):
        AmortizedGuide(
            model, {"y"}, {"w": (_encoder(), "y")}, x=x, observations={"y": y}
        )
    with pytest.raises(
        ValueError, match="draws 3 coordinates per row; the site takes 2"
    ):
        wide = ConditionalNormal(Euclidean(name="Y", dim=3), Euclidean(name="W", dim=3))
        AmortizedGuide(model, {"y"}, {"z": (wide, "y")}, x=x, observations={"y": y})
    halfnormal = loads(
        "object Obs : Real 3\nobject Pos : Real 2\n"
        "morphism hn : Obs -> Pos ~ HalfNormal\nexport hn\n"
    ).morphism
    with pytest.raises(
        ValueError, match="must have the site's support, or be real-valued"
    ):
        AmortizedGuide(
            model, {"y"}, {"z": (halfnormal, "y")}, x=x, observations={"y": y}
        )
    with pytest.raises(
        ValueError, match=r"read \['y'\], which the observations do not"
    ):
        AmortizedGuide(
            model, {"y"}, {"z": (_encoder(), "y")}, x=x, observations={"v": y}
        )
    guide = AmortizedGuide(
        model, {"y"}, {"z": (_encoder(), "y")}, x=x, observations={"y": y}
    )
    with pytest.raises(ValueError, match="takes \\(3,\\)"):
        guide.rsample(torch.zeros(3, 1))
    unobserved = AmortizedGuide(model, {"y"}, {"z": (_encoder(), "y")}, x=x)
    with pytest.raises(RuntimeError, match="no data observed"):
        unobserved.rsample(x)


def test_an_encoder_with_a_stochastic_hidden_layer_is_refused() -> None:
    source = (
        "object Obs : Real 3\nobject H : Real 4\nobject Latent : Real 2\n"
        "morphism hid : Obs -> H ~ Normal\nmorphism head : H -> Latent ~ Normal\n"
        "define deep = hid >> head\nexport deep\n"
    )
    deep = loads(source).morphism
    model = _linear_gaussian()
    y = _data(5, seed=0)
    with pytest.raises(ValueError, match="scores a reference path"):
        AmortizedGuide(
            model, {"y"}, {"z": (deep, "y")}, x=torch.zeros(5, 1), observations={"y": y}
        )


# ---------------------------------------------------------------------------
# Composition and objectives
# ---------------------------------------------------------------------------


def _hierarchical() -> MonadicProgram:
    """A global ``mu`` and per-run local ``z`` with ``y ~ N(W z + mu, ...)``."""
    decoder = _decoder()
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("mu",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="M", dim=1)),
            ),
            Draw(
                names=("z",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="Z", dim=2)),
            ),
            Observe(names=("y",), morphism=decoder, args=("z",)),
        ],
        return_vars=(),
    )


def test_composes_with_auto_guide_list() -> None:
    model = _hierarchical()
    y = _data(5, seed=4)
    x = torch.zeros(5, 1)
    amortized = AmortizedGuide(
        model, {"y"}, {"z": (_encoder(), "y")}, x=x, observations={"y": y}
    )
    assert amortized.uncovered_sites == ("mu",)
    # Read off the model's steps, the registry leaves out the names
    # marked observed, so this guide covers ``mu`` alone.
    rest = AutoNormalGuide(model, {"y", "z"})
    guide = AutoGuideList({"local": amortized, "global": rest})
    other = _data(5, seed=5)
    loss = ELBO(num_particles=2)(model, guide, x, {"y": other})
    loss.backward()
    # The objective handed the step's data to the amortized part.
    assert amortized._data is not None and torch.equal(amortized._data["y"], other)
    assert all(p.grad is not None for p in amortized.parameters())
    assert set(guide.rsample(x)) == {"mu", "z"}


@pytest.mark.parametrize(
    "objective",
    [
        ELBO(num_particles=2),
        ELBO(num_particles=2, estimator=StickingTheLanding()),
        ELBO(num_particles=2, estimator=ScoreFunction()),
        IWAEBound(num_particles=3),
        IWAEBound(num_particles=3, estimator=DoublyReparameterized()),
        RenyiBound(alpha=0.5, num_particles=3),
        VRIWAEBound(alpha=0.3, num_particles=3),
        ChiVI(n=2.0, num_particles=3),
        DReGsBound(num_particles=3),
        RWS(num_particles=3),
    ],
    ids=lambda o: f"{type(o).__name__}-{type(getattr(o, 'estimator', None)).__name__}",
)
def test_every_objective_trains_the_encoder(objective: Objective) -> None:
    model = _linear_gaussian()
    y = _data(5, seed=6)
    x = torch.zeros(5, 1)
    encoder = _encoder()
    # Start near the exact posterior, so that every bound's importance
    # weights are well conditioned.
    with torch.no_grad():
        linear = encoder.param_source.linear
        linear.weight.zero_()
        linear.weight[:2] = _MEAN_MAP
        linear.bias.zero_()
        linear.bias[2:] = 0.5 * _POST_VAR.log() + 0.1
    guide = AmortizedGuide(
        model, {"y"}, {"z": (encoder, "y")}, x=x, observations={"y": y}
    )
    loss = objective(model, guide, x, {"y": y})
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in encoder.parameters()]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads if g is not None)
