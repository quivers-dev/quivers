"""Independent checks of atom reference carriers and selective native sampling."""

import pytest
import torch
from torch.distributions import Beta, Exponential, Gamma, Independent, Normal

from quivers.continuous._atoms import atom_reference, log_atom_mass, safe_log_prob
from quivers.continuous._batch_selection import sample_active, select_batch
from quivers.continuous._closed_probability import ClosedBinomial, ClosedPoisson


def test_selective_poisson_never_samples_inactive_unrepresentable_rate():
    law = ClosedPoisson(torch.tensor([2.0, 1e20], dtype=torch.float64))
    values = sample_active(law, torch.Size([1000]), torch.tensor([True, False]))
    assert values.shape == (1000, 2)
    assert values.dtype == law.sample((0,)).dtype
    assert bool((values[:, 1] == 0).all())
    assert abs(values[:, 0].double().mean().item() - 2) < 0.2


def test_selective_probability_endpoint_and_finite_logits_are_preserved():
    for law in [
        ClosedBinomial(torch.tensor([3, 5]), probs=torch.tensor([0.0, 1.0])),
        ClosedBinomial(torch.tensor([3, 5]), logits=torch.tensor([-1000.0, 1000.0])),
    ]:
        # Evaluate both lazy representations before selection to catch cache-dependent choices.
        _ = law.probs, law.logits
        selected = select_batch(law, torch.tensor([False, True]))
        torch.testing.assert_close(
            selected.log_prob(torch.tensor([5.0])), torch.zeros(1, dtype=torch.float64)
        )
        assert bool((selected.sample((10,)) == 5).all())


def test_selection_preserves_independent_event_axes_and_parameter_gradients():
    loc = torch.arange(12.0, dtype=torch.float64).reshape(3, 4).requires_grad_()
    law = Independent(Normal(loc, torch.ones_like(loc)), 1)
    selected = select_batch(law, torch.tensor([True, False, True]))
    assert selected.batch_shape == (2,)
    assert selected.event_shape == (4,)
    score = selected.log_prob(torch.zeros(2, 4, dtype=torch.float64)).sum()
    (grad,) = torch.autograd.grad(score, loc)
    torch.testing.assert_close(
        grad, torch.stack([-loc[0], torch.zeros(4, dtype=torch.float64), -loc[2]])
    )


@pytest.mark.parametrize("law", [Normal(0.0, 1.0), Beta(2.0, 3.0), Gamma(2.0, 3.0)])
def test_atomless_family_queries_need_no_cdf(monkeypatch, law):
    def forbidden(*args, **kwargs):
        raise AssertionError("atomless proof should avoid evaluating a CDF")

    monkeypatch.setattr(type(law), "cdf", forbidden)
    point = torch.tensor([0.0, 0.2, 1.0])
    assert not bool(atom_reference(law, point).any())
    assert bool(torch.isneginf(log_atom_mass(law, point)).all())


def test_safe_component_density_masks_support_without_changing_valid_scores():
    point = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    score = safe_log_prob(Exponential(torch.tensor(2.0, dtype=torch.float64)), point)
    torch.testing.assert_close(
        score,
        torch.tensor(
            [
                -torch.inf,
                torch.log(torch.tensor(2.0, dtype=torch.float64)).item(),
                torch.log(torch.tensor(2.0, dtype=torch.float64)).item() - 2,
            ],
            dtype=torch.float64,
        ),
    )


def test_masked_vector_count_selection_preserves_event_axes_and_exact_trials():
    from quivers.continuous._multinomial import Multinomial

    law = Multinomial(
        torch.tensor([3, 33554433, 0]),
        probs=torch.tensor(
            [[0.2, 0.3, 0.5], [0.5, 0.2, 0.3], [0.1, 0.8, 0.1]], dtype=torch.float64
        ),
    )
    selected = select_batch(law, torch.tensor([True, True, False]))
    assert selected.event_shape == (3,)
    assert selected.batch_shape == (2,)
    torch.testing.assert_close(selected.total_count, law.total_count[:2])
    torch.testing.assert_close(
        selected.log_prob(
            torch.tensor(
                [[1.0, 1.0, 1.0], [16777216.0, 6710887.0, 10066330.0]],
                dtype=torch.float64,
            )
        ),
        law.log_prob(
            torch.tensor(
                [[1.0, 1.0, 1.0], [16777216.0, 6710887.0, 10066330.0], [0.0, 0.0, 0.0]],
                dtype=torch.float64,
            )
        )[:2],
    )


def test_censoring_and_mixed_carriers_do_not_disappear_at_zero_weights():
    from quivers.continuous._censored import Censored
    from quivers.continuous._zero_inflated_asymmetric_laplace import (
        ZeroInflatedAsymmetricLaplace,
    )
    from quivers.continuous._zip_hurdle import ZeroOneInflatedBeta

    point = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    beta = ZeroOneInflatedBeta(
        *(torch.tensor(v, dtype=torch.float64) for v in [0.3, 5.0, 0.0, 0.2])
    )
    torch.testing.assert_close(
        atom_reference(beta, point), torch.tensor([True, False, True])
    )
    assert bool(torch.isneginf(log_atom_mass(beta, point)).all())
    ald = ZeroInflatedAsymmetricLaplace(0.0, 0.2, 1.0, 0.4)
    torch.testing.assert_close(
        atom_reference(ald, point), torch.tensor([True, False, False])
    )
    assert bool(torch.isneginf(log_atom_mass(ald, point)).all())
    censored = Censored(
        Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ),
        low=0.0,
        high=1.0,
    )
    torch.testing.assert_close(
        atom_reference(censored, point), torch.tensor([True, False, True])
    )
    expected = torch.stack(
        [
            torch.tensor(0.5, dtype=torch.float64).log(),
            torch.tensor(-torch.inf, dtype=torch.float64),
            torch.special.log_ndtr(torch.tensor(-1.0, dtype=torch.float64)),
        ]
    )
    torch.testing.assert_close(log_atom_mass(censored, point), expected)


def test_global_integer_carrier_includes_zero_mass_cells():
    law = ClosedBinomial(1, probs=torch.tensor(0.3, dtype=torch.float64))
    point = torch.tensor([-2.0, 0.5, 2.0], dtype=torch.float64)
    torch.testing.assert_close(
        atom_reference(law, point), torch.tensor([True, False, True])
    )
    assert bool(torch.isneginf(log_atom_mass(law, point)).all())


def test_native_reference_fixed_carriers_and_mass_agree():
    from quivers.qiec._atoms import (
        atom_reference as reference_carrier,
        log_atom_mass as reference_mass,
    )
    from quivers.qiec.distributions import HostLaw
    from quivers.continuous._zero_inflated_asymmetric_laplace import (
        ZeroInflatedAsymmetricLaplace,
    )

    native = ZeroInflatedAsymmetricLaplace(0.3, 0.2, 1.0, 0.4)
    reference = HostLaw(
        family="ZeroInflatedAsymmetricLaplace",
        arguments={"zero_prob": 0.3, "loc": 0.2, "scale": 1.0, "quantile": 0.4},
    )
    for value in [0.0, 0.3, -2.0]:
        point = torch.tensor(value)
        assert bool(atom_reference(native, point)) == reference_carrier(
            reference, value
        )
        assert log_atom_mass(native, point).item() == pytest.approx(
            reference_mass(reference, value)
        )


def test_censored_atom_gradients_are_tail_gradients():
    from quivers.continuous._censored import Censored

    loc = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    law = Censored(Normal(loc, torch.tensor(0.7, dtype=torch.float64)), 0.0, 1.0)
    score = log_atom_mass(law, torch.tensor(0.0, dtype=torch.float64))
    expected = torch.special.log_ndtr(-loc / 0.7)
    (actual_grad,) = torch.autograd.grad(score, loc, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, loc)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_inactive_boundary_density_is_substituted_before_autograd():
    shape = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    score = safe_log_prob(
        Beta(shape, torch.tensor(2.0, dtype=torch.float64)),
        torch.tensor([0.0, 0.4], dtype=torch.float64),
        active=torch.tensor([False, True]),
    )
    (gradient,) = torch.autograd.grad(score[1], shape)
    (oracle,) = torch.autograd.grad(
        Beta(shape, torch.tensor(2.0, dtype=torch.float64)).log_prob(
            torch.tensor(0.4, dtype=torch.float64)
        ),
        shape,
    )
    torch.testing.assert_close(gradient, oracle)
    assert torch.isneginf(score[0])


def test_selection_retains_resource_policy_and_zero_rate_parameterization():
    from quivers.continuous._com_poisson import COMPoisson

    rate = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    law = COMPoisson(rate=rate, dispersion=2.0, max_terms=77, max_proposals=13)
    selected = select_batch(law, torch.tensor([True, False]))
    assert selected.max_terms == 77
    assert selected.max_proposals == 13
    (gradient,) = torch.autograd.grad(
        selected.log_prob(torch.tensor([0.0])).sum(), rate
    )
    torch.testing.assert_close(gradient, torch.tensor([-1.0, 0.0], dtype=torch.float64))


def test_safe_density_preserves_floating_observation_precision_and_score_dtype():
    law = Normal(
        torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)
    )
    point = torch.tensor([0.12345678901234], dtype=torch.float64)
    torch.testing.assert_close(safe_log_prob(law, point), law.log_prob(point))
    assert safe_log_prob(law, point).dtype == torch.float64


def test_censored_selective_sampling_ignores_inactive_huge_count_rate():
    from quivers.continuous._censored import Censored

    law = Censored(
        ClosedPoisson(torch.tensor([1.0, 1e20], dtype=torch.float64)), 0.0, 5.0
    )
    values = sample_active(law, torch.Size([20]), torch.tensor([True, False]))
    assert values.shape == (20, 2)
    assert bool((values[:, 1] == 0).all())
    assert bool(((values[:, 0] >= 0) & (values[:, 0] <= 5)).all())


def test_selected_extended_beta_retains_class_moment_budgets_and_affine_metadata():
    from quivers.continuous._extended_beta import ExtendedBeta
    from quivers.continuous._affine_beta import _AffineBeta

    law = ExtendedBeta(
        torch.tensor([0.2, 0.6]),
        torch.tensor([3.0, 7.0]),
        torch.tensor([0.1, 0.2]),
        moment_max_panels=17,
    )
    selected = select_batch(law, torch.tensor([False, True]))
    assert type(selected) is ExtendedBeta
    assert selected.moment_max_panels == 17
    torch.testing.assert_close(selected.mu, law.mu[1:])
    base = select_batch(law.base, torch.tensor([False, True]))
    assert type(base) is _AffineBeta
    torch.testing.assert_close(base.left, law.base.left[1:])
    values = sample_active(law, torch.Size([20]), torch.tensor([False, True]))
    assert bool((values[:, 0] == 0).all())
    assert bool(((values[:, 1] >= 0) & (values[:, 1] <= 1)).all())


def test_transformed_selection_preserves_sigmoid_affine_and_nested_event_axes():
    from torch.distributions import TransformedDistribution, transforms

    base = Normal(
        torch.arange(6.0, dtype=torch.float64).reshape(3, 2),
        torch.ones(3, 2, dtype=torch.float64),
    )
    sigmoid = TransformedDistribution(base, [transforms.SigmoidTransform()])
    event = Independent(sigmoid, 1)
    selected = select_batch(event, torch.tensor([False, True, False]))
    assert selected.batch_shape == (1,)
    assert selected.event_shape == (2,)
    torch.testing.assert_close(
        selected.log_prob(torch.tensor([[0.4, 0.6]], dtype=torch.float64)),
        event.log_prob(torch.tensor([[0.4, 0.6]], dtype=torch.float64).expand(3, 2))[
            1:2
        ],
    )
    affine = TransformedDistribution(
        Independent(base, 1),
        transforms.ComposeTransform(
            [
                transforms.AffineTransform(
                    torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64
                    ),
                    torch.tensor(2.0),
                    event_dim=1,
                ),
                transforms.ExpTransform(),
            ]
        ),
    )
    selected = select_batch(affine, torch.tensor([True, False, True]))
    point = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)
    allpoint = torch.stack([point[0], point[0], point[1]])
    torch.testing.assert_close(
        selected.log_prob(point), affine.log_prob(allpoint)[[0, 2]]
    )


def test_censored_nested_independent_selection_preserves_reinterpreted_axes():
    from quivers.continuous._censored import Censored

    law = Independent(
        Censored(
            Normal(
                torch.arange(6.0, dtype=torch.float64).reshape(3, 2),
                torch.ones(3, 2, dtype=torch.float64),
            ),
            0.0,
            10.0,
        ),
        1,
    )
    selected = select_batch(law, torch.tensor([False, True, False]))
    assert selected.batch_shape == (1,)
    assert selected.event_shape == (2,)
    assert selected.sample((4,)).shape == (4, 1, 2)


def test_generic_transformed_atom_does_not_inherit_logitnormal_adapter_proof():
    from torch.distributions import TransformedDistribution, transforms
    from quivers.continuous.measure import PointMass
    from quivers.continuous._atoms import atomicity

    law = TransformedDistribution(
        PointMass(torch.tensor(0.0, dtype=torch.float64)), transforms.ExpTransform()
    )
    assert atomicity(law) == "discrete"
    point = torch.tensor([1.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(atom_reference(law, point), torch.tensor([True, False]))
    torch.testing.assert_close(
        log_atom_mass(law, point), torch.tensor([0.0, -torch.inf], dtype=torch.float64)
    )


def test_constructor_selection_and_expansion_preserve_sample_validation_policy():
    law = ClosedBinomial(3, probs=torch.tensor(0.3), validate_args=True)
    for transformed in [select_batch(law, torch.tensor(True)), law.expand((2,))]:
        with pytest.raises(ValueError):
            transformed.log_prob(torch.tensor(0.5))


def test_mixture_same_family_selection_retains_component_and_event_axes():
    from torch.distributions import Categorical, MixtureSameFamily

    loc = torch.arange(12.0, dtype=torch.float64).reshape(3, 2, 2)
    law = MixtureSameFamily(
        Categorical(
            probs=torch.tensor(
                [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]], dtype=torch.float64
            )
        ),
        Independent(Normal(loc, torch.ones_like(loc)), 1),
    )
    selected = select_batch(law, torch.tensor([True, False, True]))
    assert selected.batch_shape == (2,)
    assert selected.event_shape == (2,)
    value = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)
    torch.testing.assert_close(
        selected.log_prob(value),
        law.log_prob(torch.stack([value[0], value[0], value[1]]))[[0, 2]],
    )
    assert selected.sample((4,)).shape == (4, 2, 2)
