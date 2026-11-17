"""Direct empirical density evidence agrees with independent scalar density oracles."""

from __future__ import annotations

import copy
import math
import pickle

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, norm
import torch

from quivers.diagnostics import (
    DensityOrdinate,
    DensityRatio,
    PointNullAssumptions,
    density_ratio,
    log_density_ordinate,
)


def _bandwidth(values, rule, adjust=1.0):
    sd = np.std(values, ddof=1)
    if rule == "scott":
        return sd * len(values) ** (-0.2) * adjust
    iqr = np.diff(np.quantile(values, [0.25, 0.75]))[0] / 1.34
    scale = min(sd, iqr) or sd
    return 0.9 * scale * len(values) ** (-0.2) * adjust


@pytest.mark.parametrize("rule", ["nrd0", "scott"])
@pytest.mark.parametrize("budget", [1, 5, 262144])
def test_event_ordinates_bandwidths_and_local_weights_match_scipy(rule, budget):
    raw = np.random.default_rng(23).normal(size=(2, 19, 2, 3))
    tensor = torch.tensor(raw)
    point = torch.tensor([-0.3, 0.2, 0.8], dtype=torch.float64)
    result = log_density_ordinate(
        tensor, at=point, bandwidth=rule, adjust=1.3, max_working_elements=budget
    )
    for index in np.ndindex(2, 3):
        sample = raw[(slice(None), slice(None), *index)].reshape(-1)
        at = point[index[1]].item()
        width = _bandwidth(sample, rule, 1.3)
        oracle = gaussian_kde(sample, bw_method=width / np.std(sample, ddof=1))
        assert result.log_density[index].item() == pytest.approx(
            oracle.logpdf(at)[0], abs=4e-14
        )
        assert result.bandwidth[index].item() == pytest.approx(width, rel=3e-15)
        weights = np.exp(-0.5 * ((sample - at) / width) ** 2)
        expected = weights.sum() ** 2 / np.sum(weights**2)
        assert result.kernel_effective_count[index].item() == pytest.approx(
            expected, rel=5e-15
        )
    assert result.sample_shape == (2, 19) and result.num_samples == 38
    assert result.max_working_elements == budget and result.bandwidth_rule == rule


def test_explicit_per_coordinate_bandwidth_and_independent_gaussian_mixture():
    raw = np.array([[-2.0, 1.0], [0.0, 2.0], [3.0, 5.0], [4.0, 7.0]])
    point = np.array([0.2, 2.3])
    widths = np.array([0.7, 1.3])
    result = log_density_ordinate(
        torch.tensor(raw)[None],
        at=torch.tensor(point),
        bandwidth=torch.tensor(widths),
        adjust=0.5,
    )
    expected = logsumexp(
        norm.logpdf(point, loc=raw, scale=widths * 0.5), axis=0
    ) - math.log(4)
    np.testing.assert_allclose(result.log_density, expected, atol=2e-14)
    np.testing.assert_allclose(result.bandwidth, widths * 0.5, rtol=2e-15)
    assert result.bandwidth_rule == "explicit"


def test_nrd0_zero_iqr_falls_back_to_sd_without_jitter():
    raw = np.array([0.0] * 19 + [1.0])
    result = log_density_ordinate(torch.tensor(raw).reshape(1, -1))
    assert result.bandwidth.item() == pytest.approx(_bandwidth(raw, "nrd0"), rel=3e-15)
    assert math.isfinite(result.log_density.item())


@pytest.mark.parametrize(
    "declarations", [None, (False, False), (True, False), (False, True), (True, True)]
)
def test_density_ratio_is_not_bayes_factor_without_both_explicit_declarations(
    declarations,
):
    rng = np.random.default_rng(28)
    post, prior = rng.normal(0.2, 0.7, 43), rng.normal(-0.1, 1.3, 67)
    assumptions = (
        None
        if declarations is None
        else PointNullAssumptions(
            proper_continuous_prior=declarations[0],
            compatible_null_prior=declarations[1],
        )
    )
    result = density_ratio(
        torch.tensor(post)[None],
        torch.tensor(prior)[None],
        null=0.1,
        bandwidth="scott",
        assumptions=assumptions,
    )
    expected = gaussian_kde(post).logpdf(0.1)[0] - gaussian_kde(prior).logpdf(0.1)[0]
    assert result.log_ratio.item() == pytest.approx(expected, abs=3e-14)
    assert result.posterior.num_samples == 43 and result.prior.num_samples == 67
    assert result.assumptions == assumptions
    if declarations == (True, True):
        assert result.log_bayes_factor_01 is result.log_ratio
    else:
        assert result.log_bayes_factor_01 is None
    assert not hasattr(result, "posterior_model_probability")


def test_normal_conjugate_point_null_ratio_matches_analytic_model_evidence():
    # Deterministic Normal quantiles avoid a random Monte Carlo tolerance.
    n = 32768
    q = norm.ppf((np.arange(n) + 0.5) / n)
    y, noise, prior_sd = 0.6, 0.5, 1.0
    variance = 1 / (1 / prior_sd**2 + 1 / noise**2)
    mean = variance * y / noise**2
    post = q * math.sqrt(variance) + mean
    prior = q * prior_sd
    assumptions = PointNullAssumptions(
        proper_continuous_prior=True, compatible_null_prior=True
    )
    result = density_ratio(
        torch.tensor(post)[None],
        torch.tensor(prior)[None],
        bandwidth=0.02,
        prior_bandwidth=0.02,
        assumptions=assumptions,
        max_working_elements=512,
    )
    analytic = norm.logpdf(y, 0, noise) - norm.logpdf(y, 0, math.hypot(noise, prior_sd))
    assert result.log_bayes_factor_01.item() == pytest.approx(analytic, abs=0.003)
    # Also compare the explicitly smoothed Normal ordinate, exposing KDE bias.
    smoothed = norm.logpdf(0, mean, math.sqrt(variance + 0.02**2)) - norm.logpdf(
        0, 0, math.sqrt(prior_sd**2 + 0.02**2)
    )
    assert result.log_ratio.item() == pytest.approx(smoothed, abs=3e-6)


@pytest.mark.parametrize("scale", [3.0, -4.0, 1e150, 1e-150])
def test_affine_jacobian_cancels_from_density_ratio(scale):
    rng = np.random.default_rng(61)
    post, prior = (
        torch.tensor(rng.normal(size=(2, 31))),
        torch.tensor(rng.normal(1.0, 2.0, size=(1, 43))),
    )
    baseline = density_ratio(post, prior, null=0.2, bandwidth="scott")
    shift = 7 * scale
    transformed = density_ratio(
        post * scale + shift,
        prior * scale + shift,
        null=0.2 * scale + shift,
        bandwidth="scott",
    )
    assert transformed.log_ratio.item() == pytest.approx(
        baseline.log_ratio.item(), abs=3e-12
    )
    assert transformed.posterior.log_density.item() == pytest.approx(
        baseline.posterior.log_density.item() - math.log(abs(scale)), abs=3e-12
    )


def test_nonlinear_jacobian_agreement_is_approximate_not_exact_kde_invariance():
    # Positive lognormal draws have a nonlinear change of variable x=exp(z).
    n = 8192
    q = norm.ppf((np.arange(n) + 0.5) / n)
    z1, z0 = q * 0.6 + 0.2, q * 0.9 - 0.1
    at = 0.3
    source = density_ratio(
        torch.tensor(z1)[None], torch.tensor(z0)[None], null=at, bandwidth=0.04
    )
    transformed = density_ratio(
        torch.tensor(np.exp(z1))[None],
        torch.tensor(np.exp(z0))[None],
        null=math.exp(at),
        bandwidth=0.04 * math.exp(at),
    )
    analytic = norm.logpdf(at, 0.2, 0.6) - norm.logpdf(at, -0.1, 0.9)
    assert source.log_ratio.item() == pytest.approx(analytic, abs=0.004)
    assert transformed.log_ratio.item() == pytest.approx(analytic, abs=0.004)
    assert transformed.posterior.log_density.item() + at == pytest.approx(
        source.posterior.log_density.item(), abs=0.005
    )
    assert transformed.log_ratio.item() != source.log_ratio.item()


def test_joint_hierarchical_prior_contrast_is_not_independently_permuted():
    rng = np.random.default_rng(71)
    theta = rng.normal(0, 3, 512)
    a = theta + rng.normal(0, 0.2, 512)
    b = theta + rng.normal(0, 0.2, 512)
    contrast = torch.tensor(a - b).reshape(2, 256)
    before = contrast.clone()
    result = density_ratio(contrast * 0.8, contrast, bandwidth="scott")
    oracle = gaussian_kde(a - b).logpdf(0)[0]
    permuted = gaussian_kde(a - np.roll(b, 123)).logpdf(0)[0]
    assert result.prior.log_density.item() == pytest.approx(oracle, abs=2e-14)
    assert result.prior.log_density.item() - permuted > 2
    assert torch.equal(contrast, before)


@pytest.mark.parametrize("tail", [-1e150, 1e150])
def test_extreme_tail_local_count_and_ratio_retain_small_center_differences(tail):
    post = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
    result = density_ratio(
        post, post - 1, null=tail, bandwidth=1.0, max_working_elements=1
    )
    # Each sum is dominated by its largest retained value. Direct subtraction
    # of the two rounded log densities loses this representable log ratio.
    assert result.posterior.log_density.item() == result.prior.log_density.item()
    assert result.log_ratio.item() == pytest.approx(tail, rel=8e-14)
    assert result.posterior.kernel_effective_count.item() == 1
    assert result.prior.kernel_effective_count.item() == 1


@pytest.mark.parametrize("magnitude", [1.7e308, 1e-320])
def test_extreme_bandwidth_scale_remains_available_in_log_space(magnitude):
    raw = torch.tensor([[-magnitude, magnitude]], dtype=torch.float64)
    result = log_density_ordinate(raw, bandwidth="scott")
    width_normalized = math.sqrt(2) * 2 ** (-0.2)
    log_width = math.log(magnitude) + math.log(width_normalized)
    expected = -0.5 / width_normalized**2 - log_width - 0.5 * math.log(2 * math.pi)
    assert result.log_bandwidth.item() == pytest.approx(log_width, abs=3e-13)
    assert result.log_density.item() == pytest.approx(expected, abs=5e-13)
    assert result.kernel_effective_count.item() == pytest.approx(2, abs=1e-14)
    if magnitude > 1e300:
        assert bool(torch.isinf(result.bandwidth))


def test_high_offset_adjacent_values_keep_their_original_spread():
    first = 1e308
    second = math.nextafter(first, math.inf)
    delta = second - first
    samples = torch.tensor([[first, second]], dtype=torch.float64)
    result = log_density_ordinate(samples, at=first, bandwidth="scott")
    oracle = gaussian_kde([0.0, 1.0]).logpdf(0)[0] - math.log(delta)
    assert result.log_density.item() == pytest.approx(oracle, abs=4e-13)


def test_named_axes_and_nested_opaque_records_survive_copy_and_pickle():
    raw = torch.arange(48.0, dtype=torch.float64).reshape(2, 4, 2, 3)
    assumptions = PointNullAssumptions(
        proper_continuous_prior=True, compatible_null_prior=True
    )
    result = density_ratio(
        raw,
        raw[:1] + 1,
        name="exact[name]",
        dims=["row", "component"],
        coords={"row": ["a", 2], "component": ["x", "y", "z"]},
        assumptions=assumptions,
        max_working_elements=2,
    )
    assert isinstance(result, DensityRatio) and isinstance(
        result.posterior, DensityOrdinate
    )
    assert result.posterior.dimension_names == ("row", "component")
    assert result.posterior.coordinate_labels == (("a", 2), ("x", "y", "z"))
    assert result.prior.sample_shape == (1, 4)
    for restored in (
        copy.copy(result),
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
    ):
        torch.testing.assert_close(
            restored.posterior.log_density, result.posterior.log_density
        )
        torch.testing.assert_close(
            restored.prior.kernel_effective_count, result.prior.kernel_effective_count
        )
        torch.testing.assert_close(restored.log_bayes_factor_01, result.log_ratio)
        assert restored.assumptions == assumptions


@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int64, torch.uint64, torch.complex128]
)
def test_known_discrete_or_complex_carriers_are_not_continuous_point_evidence(dtype):
    with pytest.raises(TypeError, match="floating continuous"):
        log_density_ordinate(torch.ones(1, 4, dtype=dtype))


@pytest.mark.parametrize(
    "samples",
    [
        torch.ones(1, 1),
        torch.empty(0, 4),
        torch.ones(4),
        torch.ones(1, 4),
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[0.0, float("inf")]]),
    ],
)
def test_empty_tiny_constant_or_nonfinite_data_are_not_repaired(samples):
    with pytest.raises(ValueError):
        log_density_ordinate(samples)
    with pytest.raises(ValueError):
        log_density_ordinate(samples, bandwidth=1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_working_elements": True},
        {"max_working_elements": 0},
        {"max_working_elements": 1.5},
        {"adjust": True},
        {"adjust": 0.0},
        {"adjust": float("inf")},
        {"adjust": 10**1000},
        {"bandwidth": "bad"},
        {"bandwidth": 0.0},
        {"bandwidth": -1.0},
        {"bandwidth": True},
        {"bandwidth": torch.tensor(True)},
        {"bandwidth": float("nan")},
        {"at": True},
        {"at": float("inf")},
        {"at": 2**53 + 1},
        {"at": torch.zeros(2, 4)},
        {"dims": ["chain"]},
        {"coords": {"contrast_dim_0": [1]}},
        {"name": ""},
    ],
)
def test_invalid_settings_or_event_metadata_refuse(kwargs):
    with pytest.raises((ValueError, TypeError)):
        log_density_ordinate(
            torch.arange(16.0, dtype=torch.float64).reshape(2, 4, 2), **kwargs
        )


def test_prior_alignment_and_assumption_carrier_are_checked():
    samples = torch.arange(8.0, dtype=torch.float64).reshape(2, 4)
    with pytest.raises(ValueError, match="same event shape"):
        density_ratio(samples, samples[..., None])
    with pytest.raises(TypeError, match="PointNullAssumptions"):
        density_ratio(samples, samples, assumptions=True)


def test_two_distinct_draws_are_valid_and_no_jitter_or_rng_is_used():
    samples = torch.tensor([[0.0, 1.0]], requires_grad=True)
    state = torch.get_rng_state().clone()
    result = log_density_ordinate(samples, bandwidth="scott")
    assert bool(torch.isfinite(result.log_density)) and result.num_samples == 2
    assert not result.log_density.requires_grad
    assert torch.equal(state, torch.get_rng_state())


def test_log_score_capacity_refuses_instead_of_returning_false_zero_density():
    with pytest.raises(ValueError, match="Float64 capacity"):
        log_density_ordinate(
            torch.tensor([[0.0, 1.0]], dtype=torch.float64), at=1e308, bandwidth=1e-300
        )


def test_cuda_device_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA Float64 device unavailable")
    samples = torch.arange(8.0, device="cuda").reshape(2, 4)
    result = density_ratio(samples, samples + 1, bandwidth="scott")
    assert result.log_ratio.device == samples.device
    with pytest.raises(ValueError, match="same device"):
        density_ratio(samples, samples.cpu())


def test_kernel_tiles_respect_budget_even_when_event_width_exceeds_it(monkeypatch):
    import quivers.diagnostics.density as module

    original = module._log_distance
    sizes = []

    def checked(left, right):
        size = math.prod(torch.broadcast_shapes(left.shape, right.shape))
        sizes.append(size)
        assert size <= 4
        return original(left, right)

    monkeypatch.setattr(module, "_log_distance", checked)
    result = log_density_ordinate(
        torch.arange(161.0, dtype=torch.float64).reshape(1, 23, 7),
        bandwidth="scott",
        max_working_elements=4,
    )
    assert result.log_density.shape == (7,)
    assert sizes and max(sizes) == 4


def test_density_helpers_import_without_optional_statistics_or_plotting_packages():
    import os
    from pathlib import Path
    import subprocess
    import sys

    code = """import sys
import torch
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','arviz','xarray'}:
            raise ModuleNotFoundError('optional statistics blocked')
sys.meta_path.insert(0,BlockOptional())
from quivers.diagnostics import density_ratio
samples=torch.tensor([[-1.,0.,1.,2.]])
result=density_ratio(samples,samples+1)
assert torch.isfinite(result.log_ratio)
assert not any(name in sys.modules for name in ('scipy','arviz','xarray'))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_nearly_equal_bandwidth_extreme_tail_ratio_matches_decimal_oracle():
    from decimal import Decimal, localcontext

    point = 1e150
    width = math.nextafter(1.0, math.inf)
    post = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
    result = density_ratio(
        post, post - 1, null=point, bandwidth=1.0, prior_bandwidth=width
    )
    with localcontext() as context:
        context.prec = 400
        x = Decimal.from_float(point)
        h = Decimal.from_float(width)
        expected = -((x - 2) ** 2 - (x - 1) ** 2 / h**2) / 2 + h.ln()
    assert result.log_ratio.item() == pytest.approx(float(expected), rel=2e-13)
