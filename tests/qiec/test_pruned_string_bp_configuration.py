"""Portable finite pruned BP settings and measured inference diagnostics."""

from math import exp, log

import pytest

from didactic.api import ValidationError
from quivers.core.relations import RelationLimitError
from quivers.qiec import (
    PrunedStringBPConfig,
    SerializationError,
    from_data,
    to_data,
)
from tests.qiec.test_string_solver_configuration import _model, _parameters, _roundtrip


def _config():
    return PrunedStringBPConfig(
        supports=(("x", ((1,), (0,))),),
        beam=2,
        max_sweeps=4,
        max_entries=32,
        max_states=128,
        max_work=256,
    )


def test_decoded_pruned_bp_config_keeps_graph_separate_and_derives_message_status_and_mass():
    graph = _roundtrip(_model())
    config = _roundtrip(_config())
    parameters = _parameters()
    assert config.objective == "pruned_string_bp"
    assert config.supports == (("x", ((1,), (0,))),)
    estimate = config.infer(graph, parameters)
    assert estimate.status == "converged" and estimate.exact is False
    assert not estimate.truncated and estimate.log_marginals is not None
    weight0 = exp(-7 / 8) + exp(-1)
    weight1 = exp(-5 / 8)
    assert estimate.log_marginals["x"].exp().tolist() == pytest.approx(
        [weight1 / (weight0 + weight1), weight0 / (weight0 + weight1)]
    )
    assert estimate.proposals[0][0].log_total_mass.item() == pytest.approx(
        log(exp(-1 / 8) + exp(-1 / 4) + exp(-5 / 8))
    )
    assert estimate.proposals[0][0].log_omitted_mass.item() == -float("inf")

    small = _roundtrip(config.with_(beam=1))
    result = small.infer(graph, parameters)
    assert result.status == "empty_support" and result.exact is False
    assert result.truncated and result.log_marginals is None
    assert [group[0].words for group in result.proposals] == [((0,),), ((1,),)]
    assert result.proposals[0][0].log_omitted_mass.item() == pytest.approx(-5 / 8)
    assert result.proposals[1][0].log_omitted_mass.item() == pytest.approx(-3 / 4)

    for name, value in (
        ("status", "converged"),
        ("log_omitted_mass", 0.0),
        ("log_total_mass", 100.0),
        ("retained_mass", 100.0),
        ("exact", True),
    ):
        data = to_data(config)
        data["root"]["fields"].append({"name": name, "value": to_data(value)["root"]})
        with pytest.raises(SerializationError, match="field"):
            from_data(data)


@pytest.mark.parametrize(
    "change, diagnostic",
    (
        ({"beam": 0}, "beam must be a positive integer"),
        ({"max_sweeps": 0}, "at least one sweep"),
        ({"max_entries": -1}, "max_entries"),
        ({"supports": (("x", ((0,), (0,))),)}, "distinct declared words"),
        ({"supports": (("x", ((-1,),)),)}, "nonnegative integers"),
        ({"supports": (("x", ((0,),)), ("x", ((1,),)))}, "distinct nonempty"),
        ({"supports": ()}, "declared finite supports"),
    ),
)
def test_pruned_config_validates_source_support_and_schedule(change, diagnostic):
    with pytest.raises(ValidationError, match=diagnostic):
        _config().with_(**change)


def test_binding_checks_graph_alphabet_and_preserves_plan_resource_failures():
    graph, parameters = _model(), _parameters()
    with pytest.raises(ValueError, match="invalid symbol"):
        _config().with_(supports=(("x", ((2,),)),)).infer(graph, parameters)
    with pytest.raises(ValueError, match="every graph variable exactly once"):
        _config().with_(supports=(("other", ((0,),)),)).infer(graph, parameters)
    with pytest.raises(RelationLimitError, match="max_entries"):
        _config().with_(max_entries=3).infer(graph, parameters)
