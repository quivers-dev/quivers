"""Point-null facades reuse one typed joint contrast and the density estimator."""

import copy
import math
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from quivers.diagnostics import (
    ContrastExpression,
    PointNullAssumptions,
    PointNullSpec,
    PosteriorAnalysis,
    density_ratio,
    hypothesis_summary,
)
from quivers.diagnostics.contrast import _contrast_plan, _evaluate_contrast
from quivers.formulas import BayesianFit, PriorSpec, R2D2, RegularizedHorseshoe, fit


def _analysis(samples, kind="mcmc"):
    return PosteriorAnalysis(samples=samples, log_likelihood={}, kind=kind)


def _log_ordinate(samples, at, width):
    raw = samples.flatten(0, 1)
    return (
        torch.logsumexp(-0.5 * ((raw - at) / width).square(), 0)
        - math.log(raw.shape[0])
        - torch.as_tensor(width, dtype=samples.dtype, device=samples.device).log()
        - 0.5 * math.log(2 * math.pi)
    )


@pytest.mark.parametrize(
    "declarations", [None, (True, False), (False, True), (True, True)]
)
def test_joint_typed_contrast_point_evidence_matches_independent_kernel_sum(
    declarations,
):
    a = torch.linspace(-0.5, 1.5, 22, dtype=torch.double).reshape(2, 11)
    b = a.square() / 2
    prior_a = torch.linspace(-1.2, 1.1, 31, dtype=torch.double).reshape(1, 31)
    prior_b = 0.2 * prior_a.square() + 0.1
    assumptions = (
        None
        if declarations is None
        else PointNullAssumptions(
            proper_continuous_prior=declarations[0],
            compatible_null_prior=declarations[1],
        )
    )
    spec = PointNullSpec(
        bandwidth=0.3,
        prior_bandwidth=0.7,
        max_working_elements=3,
        assumptions=assumptions,
    )
    posterior, prior = (
        _analysis({"a": a, "b": b}),
        _analysis({"a": prior_a, "b": prior_b}, "prior"),
    )
    with (
        patch(
            "quivers.diagnostics.contrast._contrast_plan", wraps=_contrast_plan
        ) as plans,
        patch(
            "quivers.diagnostics.contrast._evaluate_contrast", wraps=_evaluate_contrast
        ) as evaluate,
        patch(
            "quivers.diagnostics.hypothesis.density_ratio", wraps=density_ratio
        ) as density,
    ):
        result = posterior.hypothesis("exp(a)-b", prior=prior, null=0.2, point=spec)
    assert (
        plans.call_count == 1 and evaluate.call_count == 2 and density.call_count == 1
    )
    post_value, prior_value = a.exp() - b, prior_a.exp() - prior_b
    expected = _log_ordinate(post_value, 0.2, 0.3) - _log_ordinate(
        prior_value, 0.2, 0.7
    )
    torch.testing.assert_close(
        result.point_evidence.log_ratio, expected, atol=2e-14, rtol=2e-14
    )
    assert result.point_evidence.posterior.max_working_elements == 3
    assert result.prior_sample_shape == (1, 31) and result.sample_shape == (2, 11)
    plain = posterior.hypothesis("exp(a)-b", prior=prior, null=0.2)
    for name in (
        "estimate",
        "quantiles",
        "equal_probability",
        "prior_equal_probability",
    ):
        torch.testing.assert_close(getattr(result, name), getattr(plain, name))
    torch.testing.assert_close(
        result.above.log_bayes_factor, plain.above.log_bayes_factor, equal_nan=True
    )
    assert plain.point_evidence is None
    if declarations == (True, True):
        torch.testing.assert_close(result.point_evidence.log_bayes_factor_01, expected)
    else:
        assert result.point_evidence.log_bayes_factor_01 is None


def test_point_result_retains_nominal_event_coordinates_and_configuration():
    values = torch.linspace(-1, 1, 30, dtype=torch.double).reshape(2, 5, 3)
    prior = torch.linspace(-2, 1.5, 21, dtype=torch.double).reshape(1, 7, 3)
    widths = torch.tensor([0.2, 0.3, 0.4], dtype=torch.double)
    spec = PointNullSpec(bandwidth=widths)
    torch.testing.assert_close(copy.deepcopy(spec).bandwidth, widths)
    result = _analysis({"effect original": values}).hypothesis(
        ContrastExpression(expression="effect", bindings={"effect": "effect original"}),
        prior=_analysis({"effect original": prior}, "prior"),
        dims={"effect original": ("component",)},
        coords={"component": ("left", 7, "right"), "chain": ("a", "b")},
        point=spec,
    )
    assert result.point_evidence.posterior.dimension_names == ("component",)
    assert result.point_evidence.posterior.coordinate_labels == (("left", 7, "right"),)
    torch.testing.assert_close(
        copy.deepcopy(result).point_evidence.log_ratio, result.point_evidence.log_ratio
    )


def test_facade_preserves_authoritative_factored_extreme_tail_ratio():
    posterior = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.double)
    prior = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.double)
    result = _analysis({"a": posterior}).hypothesis(
        "a",
        prior=_analysis({"a": prior}, "prior"),
        null=1e150,
        point=PointNullSpec(bandwidth=1.0),
    )
    expected = density_ratio(posterior, prior, null=1e150, bandwidth=1.0)
    torch.testing.assert_close(result.point_evidence.log_ratio, expected.log_ratio)
    assert result.point_evidence.log_ratio > 1e149
    assert (
        result.point_evidence.posterior.log_density
        - result.point_evidence.prior.log_density
        == 0
    )


@pytest.mark.parametrize("entry", ["direct", "typed"])
def test_point_evidence_refuses_integer_contrasts_before_diagnostic_float_conversion(
    entry,
):
    samples = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    with pytest.raises((TypeError, ValueError), match="Real|floating|continuous"):
        if entry == "direct":
            hypothesis_summary(samples, prior=samples, point=PointNullSpec())
        else:
            _analysis({"count": samples}, "prior").hypothesis(
                "count",
                prior=_analysis({"count": samples}, "prior"),
                point=PointNullSpec(),
            )


def test_missing_prior_or_invalid_request_is_explicit():
    values = torch.tensor([[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="prior"):
        hypothesis_summary(values, point=PointNullSpec())
    with pytest.raises(TypeError, match="PointNullSpec"):
        hypothesis_summary(values, prior=values, point=True)
    with pytest.raises(ValueError, match="bandwidth"):
        hypothesis_summary(values, prior=values, point=PointNullSpec(bandwidth=-1.0))


@pytest.mark.parametrize("hierarchy", [RegularizedHorseshoe(), R2D2()])
def test_formula_point_contrasts_reuse_joint_hierarchical_prior_streams(hierarchy):
    frame = pd.DataFrame(
        {
            "y": [0.2, 0.4, -0.1, 0.5, -0.2, 0.3],
            "x": [-1.0, 0.0, 1.0, -0.3, 0.2, 0.7],
            "z": [0.5, -0.3, 0.2, 0.6, -0.7, 0.1],
        }
    )
    result = fit(
        "y~x+z",
        data=frame,
        method="svi",
        num_steps=1,
        priors=PriorSpec(distribution=hierarchy, lower=-0.5, upper=0.8),
        sample_prior="yes",
        num_prior_samples=11,
    )
    draws = result.draws(num_samples=7, seed=8)
    prior = result.prior_view().draws()
    oracle = density_ratio(
        draws["beta_x"].double() - draws["beta_z"].double(),
        prior["beta_x"].double() - prior["beta_z"].double(),
        bandwidth=0.3,
    )
    state = torch.random.get_rng_state()
    with patch.object(
        BayesianFit, "draws", autospec=True, side_effect=BayesianFit.draws
    ) as collect:
        actual = result.hypothesis(
            "x-z", num_samples=7, seed=8, prior=True, point=PointNullSpec(bandwidth=0.3)
        )
    assert collect.call_count == 2
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(actual.point_evidence.log_ratio, oracle.log_ratio)
    with patch.object(
        BayesianFit,
        "draws",
        side_effect=AssertionError("must validate prior request before drawing"),
    ):
        with pytest.raises(ValueError, match="prior=True"):
            result.hypothesis("x", point=PointNullSpec())


@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
def test_group_point_evidence_matches_native_bindings_on_each_joint_stream(coordinates):
    from quivers.inference import trace

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        frame = pd.DataFrame(
            {
                "y": [0.2, 0.4, -0.1, 0.5, -0.2, 0.3],
                "x": [-1.0, 0.0, 1.0, -0.3, 0.2, 0.7],
                "g": ["a", "a", "b", "b", "c", "c"],
            }
        )
        result = fit(
            "y~x+(1+x|g)",
            data=frame,
            method="svi",
            num_steps=1,
            reparameterize=coordinates,
            sample_prior="yes",
            num_prior_samples=9,
        )
        posterior, prior = (
            result.draws(num_samples=7, seed=9),
            result.prior_view().draws(),
        )

        intercept_site = next(
            target.site
            for target in result.prediction_plan.prior_targets
            if target.class_ == "Intercept"
        )

        def native_contrast(stream):
            values = []
            for index in range(next(iter(stream.values())).shape[1]):
                recorded = trace(
                    result.program,
                    torch.zeros(1, 1),
                    result.observations
                    | {name: value[0, index] for name, value in stream.items()},
                )
                values.append(
                    recorded.sites["beta_g_x"].value.reshape(3)
                    + recorded.sites["beta_x"].value.squeeze()
                    - recorded.sites["alpha_g"].value.reshape(3)
                    - recorded.sites[intercept_site].value.squeeze()
                )
            return torch.stack(values).unsqueeze(0)

        expected = density_ratio(
            native_contrast(posterior), native_contrast(prior), bandwidth=0.5
        )
        actual = result.hypothesis(
            "x-Intercept",
            scope="coef",
            group="g",
            num_samples=7,
            seed=9,
            prior=True,
            point=PointNullSpec(bandwidth=0.5),
        )
        torch.testing.assert_close(
            actual.point_evidence.log_ratio, expected.log_ratio, atol=3e-12, rtol=3e-12
        )
        assert actual.point_evidence.posterior.coordinate_labels == (("a", "b", "c"),)
    finally:
        torch.set_default_dtype(previous)


def test_categorical_point_evidence_preserves_free_component_coordinates():
    frame = pd.DataFrame(
        {
            "y": pd.Categorical(
                ["blue", "red", "green", "blue", "red", "green"],
                categories=["green", "red", "blue"],
            ),
            "x": [-1.0, 0.0, 1.0, -0.3, 0.2, 0.7],
        }
    )
    result = fit(
        "y~x",
        data=frame,
        family="categorical",
        method="svi",
        num_steps=1,
        sample_prior="yes",
        num_prior_samples=9,
    )
    posterior, prior = result.draws(num_samples=7, seed=8), result.prior_view().draws()
    expected = density_ratio(
        posterior["beta_x"].double(), prior["beta_x"].double(), bandwidth=0.5
    )
    actual = result.hypothesis(
        "x", num_samples=7, seed=8, prior=True, point=PointNullSpec(bandwidth=0.5)
    )
    torch.testing.assert_close(actual.point_evidence.log_ratio, expected.log_ratio)
    assert actual.point_evidence.posterior.coordinate_labels == (
        result.coefficient_labels,
    )
    assert len(result.coefficient_labels) == 2
