"""Subsampled plates give an unbiased, site-scoped estimate of the log joint.

Under ``subsample(plate, index)`` every site in the plate is scored at the
index's rows and multiplied by the plate's size over the index's size;
every other site is scored in full. Drawing the index uniformly among the
subsets of a fixed size, the expectation of that estimate is a finite
average over every subset, so the tests enumerate the subsets and check
the average against the full-data value exactly (to float32 round-off):
for the log joint, for the ELBO at fixed guide draws, and for both
gradients. Two wrong variants, one that leaves the plate's factor out and
one that also scales the global site, fail the same check. The factor is
checked per plate on a model with sibling and nested plates subsampled at
different fractions, against a hand computation. Minibatch SVI is checked
against the analytic posterior that full-batch SVI reaches, and the
consumers of the exact posterior and the nonlinear bounds refuse a
subsampled plate.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.distributions as D

from quivers.continuous import MonadicProgram
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.effects import ScaleHandler, SubsampleHandler, subsample
from quivers.inference import (
    ELBO,
    AmortizedGuide,
    MCMC,
    RWS,
    SVI,
    AutoDeltaGuide,
    AutoNormalGuide,
    IWAEBound,
    LatentRegistry,
)
from quivers.inference.guides import (
    AutoLaplaceApproximation,
    AutoMultivariateNormalGuide,
)
from quivers.inference.mcmc import NUTSKernel, PotentialFn
from quivers.inference.trace import trace

_X = torch.zeros(1, 1)

_HIERARCHY = """\
object Item : FinSet 6

program hierarchy : Item -> Item
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample z : Item <- Normal(loc=mu, scale=1.0)
    observe y : Item <- Normal(loc=z, scale=0.5)
    return mu

export hierarchy
"""

_Y = torch.tensor([0.1, 0.2, 0.3, 0.4, -1.0, 2.0])
_MU = torch.tensor([[0.3]])
_Z = torch.tensor([0.0, 0.5, -0.2, 0.1, 0.3, 1.0])

#: Float32 round-off of a log joint of about twenty nats summed over at
#: most twenty subsets; the factor errors the variants make are of order
#: one nat.
_EXACT = 1e-4


def _program():  # type: ignore[no-untyped-def]
    return loads(_HIERARCHY).morphism


def _subsets(size: int) -> list[torch.Tensor]:
    return [torch.tensor(rows) for rows in itertools.combinations(range(6), size)]


def _site_scores(program, values, index):  # type: ignore[no-untyped-def]
    """The per-site contributions of one run under a subset."""
    with subsample("Item", index):
        recorded = trace(program, _X, values)
    return {name: site.log_prob.sum() for name, site in recorded.sites.items()}


def _values() -> dict[str, torch.Tensor]:
    return {"y": _Y, "mu": _MU, "z": _Z}


# ---------------------------------------------------------------------------
# the estimate is unbiased, by enumeration


@pytest.mark.parametrize("size", [1, 2, 4])
def test_the_subset_average_of_the_log_joint_is_the_full_log_joint(size: int) -> None:
    program = _program()
    full = float(program.log_joint(_X, _values()).sum())
    estimates = []
    for index in _subsets(size):
        with subsample("Item", index):
            estimates.append(float(program.log_joint(_X, _values()).sum()))
    assert sum(estimates) / len(estimates) == pytest.approx(full, abs=_EXACT)
    # One subset alone is not the full value: the estimate varies.
    assert max(estimates) - min(estimates) > 1.0


def test_the_subset_average_of_the_gradient_is_the_full_gradient() -> None:
    program = _program()

    def gradient(index: torch.Tensor | None) -> torch.Tensor:
        mu = _MU.clone().requires_grad_()
        z = _Z.clone().requires_grad_()
        values = {"y": _Y, "mu": mu, "z": z}
        if index is None:
            joint = program.log_joint(_X, values).sum()
        else:
            with subsample("Item", index):
                joint = program.log_joint(_X, values).sum()
        mu_grad, z_grad = torch.autograd.grad(joint, (mu, z))
        return torch.cat([mu_grad.reshape(-1), z_grad.reshape(-1)])

    full = gradient(None)
    subsets = _subsets(2)
    average = sum(gradient(index) for index in subsets) / len(subsets)
    assert torch.allclose(average, full, atol=_EXACT)


def test_observations_given_at_the_subset_size_score_as_the_full_rows() -> None:
    program = _program()
    for index in _subsets(3):
        with subsample("Item", index):
            from_full = program.log_joint(_X, _values())
            from_subset = program.log_joint(_X, {"y": _Y[index], "mu": _MU, "z": _Z})
        assert torch.allclose(from_full, from_subset)


def test_the_check_rejects_an_omitted_factor_and_a_scaled_global_site() -> None:
    """Both wrong variants are computed from the same per-site scores: one
    divides the plate sites back by their factor, the other multiplies
    the global site by it. Their subset averages miss the full value."""
    program = _program()
    full = float(program.log_joint(_X, _values()).sum())
    factor = 6 / 2
    omitted, global_scaled, correct = [], [], []
    for index in _subsets(2):
        scores = _site_scores(program, _values(), index)
        plate = scores["z"] + scores["y"]
        correct.append(float(scores["mu"] + plate))
        omitted.append(float(scores["mu"] + plate / factor))
        global_scaled.append(float(factor * scores["mu"] + plate))
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    assert mean(correct) == pytest.approx(full, abs=_EXACT)
    assert abs(mean(omitted) - full) > 1.0
    assert abs(mean(global_scaled) - full) > 1.0


def test_the_elbo_and_its_gradients_are_unbiased_at_fixed_guide_draws() -> None:
    """With the guide's draws held fixed by the seed, the ELBO estimate
    under each subset averages to the full-data estimate at the same
    draws, and so do its gradients in the guide's parameters, the local
    latents' among them."""
    program = _program()
    torch.manual_seed(0)
    guide = AutoNormalGuide(program, {"y"})
    elbo = ELBO(num_particles=3)

    def loss_and_grad(index: torch.Tensor | None) -> tuple[float, torch.Tensor]:
        guide.zero_grad()
        torch.manual_seed(42)
        if index is None:
            loss = elbo(program, guide, _X, {"y": _Y})
        else:
            with subsample("Item", index):
                loss = elbo(program, guide, _X, {"y": _Y[index]})
        loss.backward()
        grads = torch.cat(
            [parameter.grad.reshape(-1) for parameter in guide.parameters()]  # type: ignore[union-attr]
        )
        return float(loss), grads

    full_loss, full_grad = loss_and_grad(None)
    results = [loss_and_grad(index) for index in _subsets(2)]
    average_loss = sum(loss for loss, _ in results) / len(results)
    average_grad = sum(grad for _, grad in results) / len(results)
    assert average_loss == pytest.approx(full_loss, abs=_EXACT * 3)
    assert torch.allclose(average_grad, full_grad, atol=_EXACT * 3)


def test_random_subsets_estimate_the_elbo_within_monte_carlo_error() -> None:
    """Over random subsets (and fresh guide draws) the estimates' mean sits
    within five standard errors of the full-data ELBO's mean over the
    same number of fresh guide draws."""
    program = _program()
    torch.manual_seed(3)
    guide = AutoNormalGuide(program, {"y"})
    elbo = ELBO(num_particles=1)
    draws = 2000
    full, sub = [], []
    with torch.no_grad():
        for _ in range(draws):
            full.append(float(elbo(program, guide, _X, {"y": _Y})))
            index = torch.randperm(6)[:2].sort().values
            with subsample("Item", index):
                sub.append(float(elbo(program, guide, _X, {"y": _Y[index]})))
    full_t, sub_t = torch.tensor(full), torch.tensor(sub)
    error = math.sqrt(float(full_t.var()) / draws + float(sub_t.var()) / draws)
    assert abs(float(sub_t.mean()) - float(full_t.mean())) < 5 * error


# ---------------------------------------------------------------------------
# scaling is per plate

_PLATES = """\
object Row : FinSet 3
object Col : FinSet 2
object One : FinSet 1

program plates : Row -> Row
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample w : Row <- Normal(loc=mu, scale=1.0) [over=[Col, One]]
    sample v : Col <- Normal(loc=mu, scale=2.0)
    sample u : Row <- Normal(loc=mu, scale=3.0)
    return mu

export plates
"""


def test_each_site_takes_its_own_plates_factor_against_a_hand_computation() -> None:
    program = loads(_PLATES).morphism
    torch.manual_seed(5)
    values = {name: site.value for name, site in trace(program, _X).sites.items()}
    rows, cols = torch.tensor([0, 2]), torch.tensor([1])
    with subsample("Row", rows), subsample("Col", cols):
        recorded = trace(program, _X, values)
    mu = values["mu"].reshape(())
    w = values["w"].reshape(3, 2, 1)
    expected = {
        "mu": D.Normal(0.0, 1.0).log_prob(mu),
        # Nested in both plates: 3/2 * 2/1 over the kept block.
        "w": (3 / 2) * (2 / 1) * D.Normal(mu, 1.0).log_prob(w[rows][:, cols]).sum(),
        # Siblings: each in its own plate's factor alone.
        "v": (2 / 1) * D.Normal(mu, 2.0).log_prob(values["v"][cols]).sum(),
        "u": (3 / 2) * D.Normal(mu, 3.0).log_prob(values["u"][rows]).sum(),
    }
    scales = {"mu": None, "w": 3.0, "v": 2.0, "u": 1.5}
    for name, site in recorded.sites.items():
        assert float(site.log_prob.sum()) == pytest.approx(
            float(expected[name]), rel=1e-6
        ), name
        assert site.metadata.get("subsample_scale") == scales[name], name


def test_a_handler_for_one_plate_leaves_the_others_unscaled() -> None:
    program = loads(_PLATES).morphism
    torch.manual_seed(6)
    values = {name: site.value for name, site in trace(program, _X).sites.items()}
    full = trace(program, _X, values)
    with subsample("Col", torch.tensor([0])):
        recorded = trace(program, _X, values)
    for name in ("mu", "u"):
        assert torch.allclose(recorded.sites[name].log_prob, full.sites[name].log_prob)
    assert not torch.allclose(recorded.sites["v"].log_prob, full.sites["v"].log_prob)


# ---------------------------------------------------------------------------
# minibatch SVI reaches the full-data posterior

_CONJUGATE_N = 40

_CONJUGATE = f"""\
object Item : FinSet {_CONJUGATE_N}

program mean : Item -> Item
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Item <- Normal(loc=mu, scale=1.0)
    return mu

export mean
"""


def _fit(
    program,  # type: ignore[no-untyped-def]
    y: torch.Tensor,
    batch: int | None,
    *,
    steps: int,
    scale_globals: bool = False,
) -> tuple[float, float]:
    """Fit a mean-field guide by SVI and average the last half of its iterates."""
    torch.manual_seed(1)
    guide = AutoNormalGuide(program, {"y"})
    optimizer = torch.optim.Adam(guide.parameters(), lr=0.02)
    svi = SVI(program, guide, optimizer, ELBO(num_particles=1))
    locs, scales = [], []
    size = int(y.shape[0])
    for step in range(steps):
        if batch is None:
            svi.step(_X, {"y": y})
        else:
            index = torch.randperm(size)[:batch].sort().values
            with subsample("Item", index):
                if scale_globals:
                    with ScaleHandler(batch / size):
                        svi.step(_X, {"y": y[index]})
                else:
                    svi.step(_X, {"y": y[index]})
        if step >= steps // 2:
            with torch.no_grad():
                locs.append(float(guide.loc_mu.reshape(())))  # type: ignore[union-attr]
                scales.append(float(guide._scale("mu").reshape(())))
    return sum(locs) / len(locs), sum(scales) / len(scales)


def test_minibatch_svi_reaches_the_analytic_posterior() -> None:
    """``mu | y`` is ``Normal(sum(y) / (N + 1), 1 / sqrt(N + 1))``. The
    averaged iterates of one-particle SVI sit within a fifth of a
    posterior standard deviation of the mean and ten percent of the
    standard deviation, with full batches and with batches of eight."""
    program = loads(_CONJUGATE).morphism
    torch.manual_seed(0)
    y = torch.randn(_CONJUGATE_N) + 0.7
    mean = float(y.sum()) / (_CONJUGATE_N + 1)
    sd = 1.0 / math.sqrt(_CONJUGATE_N + 1)
    for batch in (None, 8):
        loc, scale = _fit(program, y, batch, steps=1500)
        assert abs(loc - mean) < 0.2 * sd, (batch, loc, mean)
        assert abs(scale - sd) < 0.1 * sd, (batch, scale, sd)


def test_the_svi_check_rejects_a_factor_on_the_global_site() -> None:
    """Scaling the whole run by the batch fraction, the global prior
    included, fits a posterior far wider than the analytic one."""
    program = loads(_CONJUGATE).morphism
    torch.manual_seed(0)
    y = torch.randn(_CONJUGATE_N) + 0.7
    sd = 1.0 / math.sqrt(_CONJUGATE_N + 1)
    _, scale = _fit(program, y, 8, steps=1500, scale_globals=True)
    assert abs(scale - sd) > 0.1 * sd


def test_minibatch_svi_fits_local_latents_in_the_subsampled_plate() -> None:
    """``z_n ~ Normal(mu, 1)``, ``y_n ~ Normal(z_n, 1)``: the posterior
    means are ``E[mu] = (sum(y) / 2) / (1 + N / 2)`` and ``E[z_n] =
    (E[mu] + y_n) / 2``, which a mean-field Gaussian guide recovers
    exactly. Batches of five of twenty rows reach them."""
    size = 20
    program = loads(
        f"object Item : FinSet {size}\n\n"
        "program local : Item -> Item\n"
        "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
        "    sample z : Item <- Normal(loc=mu, scale=1.0)\n"
        "    observe y : Item <- Normal(loc=z, scale=1.0)\n"
        "    return mu\n\nexport local\n"
    ).morphism
    torch.manual_seed(0)
    y = torch.randn(size) * 1.5 + 0.7
    mean = float(y.sum() / 2) / (1 + size / 2)
    sd = 1.0 / math.sqrt(1 + size / 2)
    torch.manual_seed(1)
    guide = AutoNormalGuide(program, {"y"})
    optimizer = torch.optim.Adam(guide.parameters(), lr=0.02)
    svi = SVI(program, guide, optimizer, ELBO(num_particles=1))
    steps = 3000
    locs, local_locs = [], []
    for step in range(steps):
        index = torch.randperm(size)[:5].sort().values
        with subsample("Item", index):
            svi.step(_X, {"y": y[index]})
        if step >= steps // 2:
            with torch.no_grad():
                locs.append(float(guide.loc_mu.reshape(())))  # type: ignore[union-attr]
                local_locs.append(guide.loc_z.reshape(-1).clone())  # type: ignore[union-attr]
    loc = sum(locs) / len(locs)
    local = torch.stack(local_locs).mean(0)
    assert abs(loc - mean) < 0.2 * sd
    # Each local posterior has standard deviation 1 / sqrt(2).
    assert float((local - (mean + y) / 2).abs().max()) < 0.2 / math.sqrt(2)


# ---------------------------------------------------------------------------
# refusals


def test_the_index_must_be_distinct_increasing_rows_of_a_known_plate() -> None:
    program = _program()
    with pytest.raises(ValueError, match="increasing order"):
        SubsampleHandler("Item", torch.tensor([2, 1]))
    with pytest.raises(ValueError, match="integer rows"):
        SubsampleHandler("Item", torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="nonempty"):
        SubsampleHandler("Item", torch.tensor([], dtype=torch.long))
    with (
        subsample("Group", torch.tensor([0])),
        pytest.raises(ValueError, match="no site of the program ranges over"),
    ):
        program.log_joint(_X, _values())
    with (
        subsample("Item", torch.tensor([0, 9])),
        pytest.raises(ValueError, match="beyond the plate's 6 rows"),
    ):
        program.log_joint(_X, _values())
    with (
        subsample("Item", torch.tensor([0])),
        subsample("Item", torch.tensor([1])),
        pytest.raises(ValueError, match="active twice"),
    ):
        program.log_joint(_X, _values())


def test_exact_posterior_consumers_refuse_a_subsampled_plate() -> None:
    program = _program()
    observations = {"y": _Y}
    registry = LatentRegistry.from_trace(program, trace(program, _X, observations))
    potential = PotentialFn(program, registry, _X, observations)
    with subsample("Item", torch.tensor([0, 1])):
        with pytest.raises(ValueError, match="PotentialFn targets the exact"):
            PotentialFn(program, registry, _X, observations)
        with pytest.raises(ValueError, match="PotentialFn targets the exact"):
            potential.log_density(torch.zeros(registry.total_unconstrained_dim))
        with pytest.raises(ValueError, match="MCMC.run targets the exact"):
            MCMC(kernel=NUTSKernel(), num_samples=2, num_warmup=2).run(
                program, _X, observations
            )
        laplace = AutoLaplaceApproximation(program, {"y"})
        with pytest.raises(ValueError, match="targets the exact posterior"):
            laplace.fit_hessian(program, _X, observations)


def test_nonlinear_bounds_refuse_a_subsampled_plate() -> None:
    program = _program()
    guide = AutoNormalGuide(program, {"y"})
    with subsample("Item", torch.tensor([0, 1])):
        with pytest.raises(ValueError, match="only an objective linear"):
            IWAEBound(num_particles=4)(program, guide, _X, {"y": _Y[:2]})
        with pytest.raises(ValueError, match="only an objective linear"):
            RWS(num_particles=4)(program, guide, _X, {"y": _Y[:2]})


def test_a_guide_whose_density_does_not_split_refuses_local_latents() -> None:
    program = _program()
    guide = AutoMultivariateNormalGuide(program, {"y"})
    with subsample("Item", torch.tensor([0, 1])):
        with pytest.raises(ValueError, match="cannot score latent 'z'"):
            ELBO()(program, guide, _X, {"y": _Y[:2]})
    # A plate holding only observations leaves such a guide usable.
    conjugate = loads(_CONJUGATE).morphism
    global_guide = AutoMultivariateNormalGuide(conjugate, {"y"})
    with subsample("Item", torch.tensor([0, 1])):
        loss = ELBO()(conjugate, global_guide, _X, {"y": torch.zeros(2)})
    assert torch.isfinite(loss)


def test_the_delta_guide_scores_subsampled_latents() -> None:
    program = _program()
    guide = AutoDeltaGuide(program, {"y"})
    with subsample("Item", torch.tensor([0, 3])):
        loss = ELBO()(program, guide, _X, {"y": _Y[[0, 3]]})
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# an amortized guide over a subsampled plate

_LOCAL = """\
object Item : FinSet 6

program local : Item -> Item
    sample z : Item <- Normal(loc=0.0, scale=1.0)
    observe y : Item <- Normal(loc=z, scale=0.5)
    return z

export local
"""

#: The observations, and the same rows as the encoder reads them, one
#: coordinate each. A plate latent is drawn at every row of its plate, so
#: the encoder reads the whole plate's data under a subset too.
_LOCAL_DATA = {"y": _Y, "y_rows": _Y.unsqueeze(-1)}

#: The encoder ``q(z_i | y_i) = Normal(a y_i + b, s)``.
_A, _B, _S = 0.7, 0.1, 0.4


def _local_guide() -> tuple[MonadicProgram, AmortizedGuide]:
    """The local-latent model and an amortized guide with a fixed linear
    Normal encoder.

    Returns
    -------
    tuple[MonadicProgram, AmortizedGuide]
        The model and the guide.
    """
    program = loads(_LOCAL).morphism
    encoder = ConditionalNormal(Euclidean(name="Y", dim=1), Euclidean(name="Z", dim=1))
    with torch.no_grad():
        linear = encoder.param_source.linear
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[0, 0] = _A
        linear.bias[0] = _B
        linear.bias[1] = math.log(_S)
    guide = AmortizedGuide(
        program, {"y"}, {"z": (encoder, "y_rows")}, x=_X, observations=_LOCAL_DATA
    )
    return program, guide


def _row_elbo(y: float) -> float:
    """One row's exact ELBO term under the encoder, in closed form.

    ``E_q[log N(z; 0, 1) + log N(y; z, 0.5) - log q(z | y)]`` for
    ``q = Normal(m, s)`` with ``m = a y + b``.

    Parameters
    ----------
    y : float
        The row's observation.

    Returns
    -------
    float
        The row's term.
    """
    m = _A * y + _B
    log_two_pi = math.log(2 * math.pi)
    prior = -0.5 * log_two_pi - 0.5 * (m * m + _S * _S)
    likelihood = (
        -0.5 * log_two_pi - math.log(0.5) - ((y - m) ** 2 + _S * _S) / (2 * 0.25)
    )
    entropy = 0.5 * log_two_pi + math.log(_S) + 0.5
    return prior + likelihood + entropy


def test_the_amortized_guide_scores_a_subsampled_plate_at_the_subset_rows() -> None:
    # log q of the plate site under the subset is the subset rows' encoder
    # log densities times N / B, against the Normal densities by hand.
    _, guide = _local_guide()
    z = torch.tensor([0.3, -0.2, 0.9, 0.0, -1.1, 0.5])
    index = torch.tensor([1, 4])
    with subsample("Item", index):
        scored = float(guide.log_prob(_X, {"z": z})[0])
    rows = D.Normal(_A * _Y + _B, _S).log_prob(z)
    expected = 3.0 * float(rows[index].sum())
    assert scored == pytest.approx(expected, abs=1e-5)
    # Neither the unscaled subset nor the whole plate is that density.
    assert abs(scored - float(rows[index].sum())) > 1.0
    assert abs(scored - float(rows.sum())) > 1.0


def test_the_amortized_elbo_under_a_subset_estimates_the_scaled_rows() -> None:
    # Over fresh guide draws the subset's ELBO estimate averages to N / B
    # times the subset rows' closed-form terms, within five standard errors.
    program, guide = _local_guide()
    elbo = ELBO(num_particles=1)
    index = torch.tensor([0, 3])
    torch.manual_seed(5)
    draws = 3000
    with torch.no_grad(), subsample("Item", index):
        estimates = torch.tensor(
            [-float(elbo(program, guide, _X, _LOCAL_DATA)) for _ in range(draws)]
        )
    expected = 3.0 * sum(_row_elbo(float(_Y[i])) for i in index.tolist())
    error = math.sqrt(float(estimates.var()) / draws)
    assert abs(float(estimates.mean()) - expected) < 5 * error
    # The full-plate ELBO is another number, outside that band.
    full = sum(_row_elbo(float(value)) for value in _Y.tolist())
    assert abs(full - expected) > 10 * error


def test_the_amortized_elbo_is_unbiased_over_subsets_at_fixed_draws() -> None:
    program, guide = _local_guide()
    elbo = ELBO(num_particles=3)

    def loss_and_grad(index: torch.Tensor | None) -> tuple[float, torch.Tensor]:
        guide.zero_grad()
        torch.manual_seed(42)
        if index is None:
            loss = elbo(program, guide, _X, _LOCAL_DATA)
        else:
            with subsample("Item", index):
                loss = elbo(program, guide, _X, _LOCAL_DATA)
        loss.backward()
        grads = torch.cat(
            [parameter.grad.reshape(-1) for parameter in guide.parameters()]  # type: ignore[union-attr]
        )
        return float(loss), grads

    full_loss, full_grad = loss_and_grad(None)
    results = [loss_and_grad(index) for index in _subsets(2)]
    average_loss = sum(loss for loss, _ in results) / len(results)
    average_grad = sum(grad for _, grad in results) / len(results)
    assert average_loss == pytest.approx(full_loss, abs=_EXACT * 3)
    assert torch.allclose(average_grad, full_grad, atol=_EXACT * 3)
