"""Original observation identities and latent coordinates remain distinct."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm

from quivers.formulas._missing import bind_latent_input, bind_latent_response
from quivers.formulas.formula import RowSelection


def test_missing_outcomes_use_only_missing_latents_and_reassemble_original_order():
    raw = np.array([2.0, np.nan, -1.0, np.nan])
    plan = bind_latent_response("x", raw)
    assert plan.observed_positions == (0, 2)
    assert plan.latent_positions == (1, 3)
    assert plan.observed_rows == (0, 2)
    assert plan.full_to_combined == (0, 2, 1, 3)
    np.testing.assert_array_equal(plan.observed.data, [2.0, -1.0])
    truth = torch.tensor([0.3, -0.8], dtype=torch.float64, requires_grad=True)
    combined = torch.cat((truth.new_tensor(plan.observed.data), truth))
    actual = combined[list(plan.full_to_combined)]
    torch.testing.assert_close(actual, truth.new_tensor([2.0, 0.3, -1.0, -0.8]))
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), truth)[0], 2 * truth
    )
    raw[0] = 99.0
    assert plan.observed.data[0] == 2.0
    assert not plan.observed.data.flags.writeable


def test_known_measurement_error_has_all_true_values_latent_and_no_fake_events():
    plan = bind_latent_response(
        "x", [1.0, None, 3.0, 4.0], standard_error=[0.2, np.nan, np.inf, 0.4]
    )
    assert plan.latent_positions == (0, 1, 2, 3)
    assert plan.observed_positions == (0, 3)
    assert plan.full_to_combined == (0, 1, 2, 3)
    np.testing.assert_array_equal(plan.observed.data, [1.0, 4.0])
    np.testing.assert_array_equal(plan.standard_error.data, [0.2, 0.4])
    assert np.isfinite(plan.observed.data).all()
    assert np.isfinite(plan.standard_error.data).all()


def test_measurement_plan_matches_analytic_normal_posterior_every_latent_gradient():
    values = np.array([0.4, -0.7, 0.1])
    errors = np.array([0.2, 0.5, 0.3])
    plan = bind_latent_response("x", values, standard_error=errors)
    latent = torch.tensor([0.1, -0.3, 0.2], dtype=torch.float64, requires_grad=True)
    mean, sd = 0.25, 0.8
    prior = (
        torch.distributions.Normal(latent.new_tensor(mean), latent.new_tensor(sd))
        .log_prob(latent)
        .sum()
    )
    measurement = (
        torch.distributions.Normal(
            latent[list(plan.observed_positions)],
            latent.new_tensor(plan.standard_error.data),
        )
        .log_prob(latent.new_tensor(plan.observed.data))
        .sum()
    )
    actual = prior + measurement
    expected = (
        norm.logpdf(latent.detach().numpy(), mean, sd).sum()
        + norm.logpdf(values, latent.detach().numpy(), errors).sum()
    )
    assert actual.item() == pytest.approx(expected, abs=1e-13)
    expected_gradient = (
        -(latent.detach().numpy() - mean) / sd**2
        + (values - latent.detach().numpy()) / errors**2
    )
    np.testing.assert_allclose(
        torch.autograd.grad(actual, latent)[0].numpy(), expected_gradient, rtol=1e-14
    )
    posterior_variance = 1 / (sd**-2 + errors**-2)
    posterior_mean = posterior_variance * (mean / sd**2 + values / errors**2)
    theta = torch.tensor(posterior_mean, requires_grad=True)
    stationary = (
        torch.distributions.Normal(theta.new_tensor(mean), theta.new_tensor(sd))
        .log_prob(theta)
        .sum()
        + torch.distributions.Normal(theta, theta.new_tensor(errors))
        .log_prob(theta.new_tensor(values))
        .sum()
    )
    torch.testing.assert_close(
        torch.autograd.grad(stationary, theta)[0],
        torch.zeros_like(theta),
        atol=2e-14,
        rtol=0,
    )


def test_shared_noisy_truth_reuses_canonical_group_codes_and_scores_once():
    frame = pd.DataFrame({"id": ["b", "a", "b", "a"]})
    plan = bind_latent_response(
        "x",
        [2.0, 1.0, 2.0, 1.0],
        standard_error=[0.4, 0.2, 0.4, 0.2],
        group="id",
        frame=frame,
    )
    assert plan.levels == ("a", "b")
    assert plan.truth_size == 2
    assert plan.latent_positions == (0, 1)
    assert plan.observed_positions == (0, 1)
    assert plan.observed_rows == (1, 0)
    assert plan.full_to_combined == (1, 0, 1, 0)
    np.testing.assert_array_equal(plan.observed.data, [1.0, 2.0])
    np.testing.assert_array_equal(plan.standard_error.data, [0.2, 0.4])


def test_subsets_preserve_original_event_units_and_explicit_cross_row_indices():
    source_rows = RowSelection(size=8, positions=(0, 3, 6))
    source = bind_latent_response("x", [1.0, np.nan, 2.0], rows=source_rows)
    assert source.observed_rows == (0, 6)
    target = RowSelection(size=8, positions=(6, 3))
    result = bind_latent_input(source, target)
    assert result.positions == (2, 1)
    assert result.response == "x"
    result = bind_latent_input(
        source,
        RowSelection(size=2, positions=(0, 1)),
        source_index=("A", "B", "C"),
        target_index=("C", "A"),
        index="idx",
    )
    assert result.positions == (2, 0)
    assert result.index == "idx"
    with pytest.raises(ValueError, match="no modeled source"):
        bind_latent_input(source, RowSelection(size=8, positions=(1, 4, 7)))
    with pytest.raises(ValueError, match="same original scope"):
        bind_latent_input(source, RowSelection(size=3, positions=(0, 1, 2)))


@pytest.mark.parametrize(
    "values, errors",
    [
        ([1.0, np.inf], None),
        ([True, 2.0], None),
        ([1.0, 1j], None),
        (["1", None], None),
        ([1.0, 2.0], [0.1, 0.0]),
        ([1.0, 2.0], [0.1, -0.2]),
        ([1.0, 2.0], [0.1, np.nan]),
        ([1.0, 2.0], [True, 0.2]),
        ([1.0, 2.0], [0.1, 0.2, 0.3]),
        ([[1.0, 2.0]], None),
    ],
)
def test_invalid_observation_and_error_contracts_are_rejected(values, errors):
    with pytest.raises(ValueError):
        bind_latent_response("x", values, standard_error=errors)


@pytest.mark.parametrize("values", [[], [None, np.nan], [1.0, 2.0]])
def test_zero_all_or_no_missing_coordinates_remain_explicit(values):
    plan = bind_latent_response("x", values)
    assert len(plan.observed_positions) + len(plan.latent_positions) == len(values)
    assert len(plan.full_to_combined) == len(values)
    restored = pickle.loads(pickle.dumps(plan))
    assert restored == plan
    assert hash(restored) == hash(plan)


@pytest.mark.parametrize(
    "source_index, target_index",
    [
        (("a", "a"), ("a",)),
        (("a", "b"), ("unknown",)),
        ((True, 2), (True,)),
        ((1.0, 2.0), (1.0,)),
    ],
)
def test_index_joins_reject_ambiguous_or_invalid_nominal_identities(
    source_index, target_index
):
    source = bind_latent_response("x", [1.0, None])
    with pytest.raises(ValueError):
        bind_latent_input(
            source,
            RowSelection(size=1, positions=(0,)),
            source_index=source_index,
            target_index=target_index,
            index="idx",
        )


def test_group_repeats_cannot_silently_average_conflicting_values_or_errors():
    frame = pd.DataFrame({"g": ["a", "a"]})
    for data, error in (([1.0, 2.0], [0.1, 0.1]), ([1.0, 1.0], [0.1, 0.2])):
        with pytest.raises(ValueError, match="identical"):
            bind_latent_response(
                "x", data, standard_error=error, group="g", frame=frame
            )


@pytest.mark.parametrize(
    "positions,size", [((0, 0), 2), ((0, 2), 2), ((True,), 2), ((0,), -1)]
)
def test_original_row_domains_reject_duplicate_or_invalid_identities(positions, size):
    from didactic.fields import ValidationError

    source = bind_latent_response("x", [1.0, 2.0])
    with pytest.raises((ValueError, TypeError, ValidationError)):
        bind_latent_input(source, RowSelection(size=size, positions=positions))


def test_private_plan_records_reject_forged_layouts_and_immutable_roundtrip():
    from didactic.fields import ValidationError

    plan = bind_latent_response("x", [1.0, None, 2.0])
    for change in (
        {"observed_positions": (0, 0)},
        {"latent_positions": ()},
        {"observed_rows": (2, 0)},
        {"full_to_combined": (0, 1, 2)},
    ):
        with pytest.raises(ValidationError):
            plan.with_(**change)
    noisy = bind_latent_response("x", [1.0, 2.0], standard_error=0.2)
    with pytest.raises(ValidationError):
        noisy.with_(latent_positions=(0,))
    reference = bind_latent_input(plan, RowSelection(size=3, positions=(0, 2)))
    with pytest.raises(ValidationError):
        reference.with_(positions=(0, 3))
    repeated = bind_latent_input(
        plan,
        RowSelection(size=2, positions=(0, 1)),
        source_index=("a", "b", "c"),
        target_index=("b", "b"),
        index="id",
    )
    assert repeated.positions == (1, 1)
