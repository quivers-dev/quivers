"""Recording events share native trace, plate, weighting, and prediction paths."""

import math

import pytest
import torch

from quivers.continuous._observation_events import event_log_prob
from quivers.effects import subsample
from quivers.modeling import Model


SOURCE = """object Row : FinSet 4
object Out : Real 1
program regression(y : Tensor[Real]([Row]), c : Tensor[Int]([Row]), u : Tensor[Real]([Row]), w : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    observe y : Row <- Normal(loc=b, scale=1.0) [cens=c, cens_upper=u, weights=w]
    return b
export regression
"""


def _data():
    return {
        "y": torch.tensor([0.2, 0.5, -0.4, 0.1], dtype=torch.float64),
        "c": torch.tensor([0, 1, -1, 2]),
        "u": torch.tensor([math.nan, math.nan, math.nan, 0.8], dtype=torch.float64),
        "w": torch.tensor([0.5, 2.0, 1.0, 0.0], dtype=torch.float64),
    }


def _normal(value):
    return -0.5 * (value.square() + math.log(2 * math.pi))


def test_source_literal_event_endpoints_keep_double_precision_through_plate_layout():
    source = """object Row : FinSet 2
program p(y:Tensor[Real]([Row]),c:Tensor[Int]([Row])) : Row -> Row
    observe y:Row <- Normal(loc=0.0,scale=1.0) [cens=c,cens_upper=1.3,trunc_lower=-0.7,trunc_upper=1.7]
    return y
export p
"""
    y = torch.tensor([0.3, 0.5], dtype=torch.float64)
    recorded = Model(source).bind({"y": y, "c": torch.tensor([2, 2])}).trace({})
    cdf = torch.special.ndtr
    expected = (cdf(torch.tensor(1.3, dtype=torch.float64)) - cdf(y)).log() - (
        cdf(torch.tensor(1.7, dtype=torch.float64))
        - cdf(torch.tensor(-0.7, dtype=torch.float64))
    ).log()
    torch.testing.assert_close(
        recorded.sites["y"].sampleable.raw_log_prob_pointwise(y),
        expected,
        rtol=2e-12,
        atol=2e-12,
    )


def test_recorded_event_trace_keeps_unweighted_pointwise_likelihood():
    data = _data()
    model = Model(SOURCE).bind(data)
    b = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    trace = model.trace({"b": b})
    raw = event_log_prob(
        torch.distributions.Normal(b, 1.0),
        data["y"],
        status=data["c"],
        interval_upper=data["u"],
    )
    expected = _normal(b).sum() + (data["w"] * raw).sum()
    torch.testing.assert_close(trace.log_joint.sum(), expected)
    torch.testing.assert_close(
        torch.autograd.grad(trace.log_joint.sum(), b)[0],
        torch.autograd.grad(expected, b)[0],
    )
    site = trace.sites["y"]
    torch.testing.assert_close(site.sampleable.raw_log_prob_pointwise(data["y"]), raw)


def test_subsampled_recorded_events_reuse_the_weighted_plate_reduction():
    data = _data()
    model = Model(SOURCE).bind(data)
    b = torch.tensor([0.3], dtype=torch.float64)
    raw = event_log_prob(
        torch.distributions.Normal(b, 1.0),
        data["y"],
        status=data["c"],
        interval_upper=data["u"],
    )
    kept = torch.tensor([0, 2])
    with subsample("Row", kept):
        trace = model.trace({"b": b})
    expected = _normal(b).sum() + 2 * (data["w"][kept] * raw[kept]).sum()
    torch.testing.assert_close(trace.log_joint.sum(), expected)


def test_truncated_prediction_draws_responses_instead_of_censoring_endpoints():
    source = SOURCE.replace(
        "weights=w]", "weights=w, trunc_lower=-0.5, trunc_upper=1.0]"
    )
    data = _data()
    model = Model(source).bind(data)
    trace = model.trace({"b": torch.tensor([0.1], dtype=torch.float64)})
    sampleable = trace.sites["y"].sampleable
    with torch.random.fork_rng():
        torch.manual_seed(2026)
        draws = sampleable.raw_morphism.rsample(sampleable.input, torch.Size((128,)))
    assert bool(((draws >= -0.5) & (draws <= 1.0)).all())
    assert draws.unique().numel() > 100


@pytest.mark.parametrize("status", [-2, 3])
def test_invalid_recording_codes_are_rejected_at_native_execution(status):
    data = _data()
    data["c"][0] = status
    with pytest.raises(ValueError, match="censoring status"):
        Model(SOURCE).bind(data).trace({"b": torch.tensor([0.1])})


def test_parameter_handlers_override_event_scores_and_sampling_laws():
    from quivers.continuous.families import ConditionalNormal
    from quivers.continuous.plate import VectorisedObserve
    from quivers.continuous.spaces import Euclidean
    from quivers.core.objects import FinSet
    from quivers.effects._observation_events import RecordedObservation
    from quivers.effects.interpreter import _ParameterizedMorphism

    data = _data()
    family = ConditionalNormal(
        FinSet(name="Row", cardinality=4), Euclidean(name="Out", dim=1)
    ).double()
    plate = VectorisedObserve(family, data["y"])
    overrides = {
        name: torch.zeros_like(value, requires_grad=True)
        for name, value in plate.named_parameters()
    }
    assert overrides
    parameterized = _ParameterizedMorphism(plate, overrides)
    event = {"status": data["c"], "interval_upper": data["u"]}
    wrapped = RecordedObservation(parameterized, plate, event)
    x = torch.arange(4)
    actual = wrapped.log_prob_pointwise(x, data["y"])
    expected = event_log_prob(
        torch.distributions.Normal(torch.zeros(4, dtype=torch.float64), 1.0),
        data["y"],
        status=data["c"],
        interval_upper=data["u"],
    )
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual.sum(), tuple(overrides.values()))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    law = wrapped.sampling_distribution(x)
    torch.testing.assert_close(law.loc, torch.zeros((4, 1), dtype=torch.float64))
    torch.testing.assert_close(law.scale, torch.ones((4, 1), dtype=torch.float64))


def test_independent_array_event_metadata_follows_values_and_selected_rows():
    from quivers.continuous.families import ConditionalNormal
    from quivers.continuous.plate import VectorisedObserve
    from quivers.continuous.spaces import Euclidean
    from quivers.core.objects import FinSet
    from quivers.effects._observation_events import RecordedObservation
    from quivers.effects.interpreter import _ParameterizedMorphism

    family = ConditionalNormal(
        FinSet(name="Row", cardinality=4), Euclidean(name="Out", dim=1)
    ).double()
    y = torch.tensor(
        [[0.1, 0.2], [0.4, 0.5], [-0.1, -0.2], [0.0, 0.3]], dtype=torch.float64
    )
    # Each array entry has its own recording role; the same indexing rules
    # apply to outcomes, event data, and shared family parameters.
    status = torch.tensor([[0, 1], [-1, 0], [0, -1], [1, 0]])
    plate = VectorisedObserve(family, y, array_shape=(2,))
    inner = _ParameterizedMorphism(
        plate, {name: torch.zeros_like(p) for name, p in plate.named_parameters()}
    )
    wrapped = RecordedObservation(inner, plate, {"status": status})
    x = torch.arange(4)
    expected = event_log_prob(
        torch.distributions.Normal(torch.zeros_like(y), 1.0), y, status=status
    ).sum(-1)
    torch.testing.assert_close(wrapped.log_prob_pointwise(x, y), expected)
    kept = torch.tensor([3, 1])
    torch.testing.assert_close(
        wrapped.log_prob_pointwise_rows(x, y, kept, torch.arange(2)), expected[kept]
    )
