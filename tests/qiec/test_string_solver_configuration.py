"""Separate checked solver and graph records, with independent objectives."""

from fractions import Fraction
from math import exp

import pytest
import torch
from didactic.api import ValidationError

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import Arc, Boundary, RelationLimitError, RelationTopology
from quivers.qiec import (
    JointAlignmentMapConfig,
    ParameterId,
    RationalRelationSpec,
    RegularStringEPConfig,
    SerializationError,
    StringFactorGraphSpec,
    dumps,
    from_data,
    loads,
    to_data,
)


_TAPE = StringSet(alphabet=FinSet(name="Letter", cardinality=2))
_UNIT = ParameterId.derive("solver-config", "unit")
_SCORES = tuple(ParameterId.derive("solver-config", f"score-{i}") for i in range(4))


def _relation(alphabet, occurrences, *, algebra=LOG_PROB):
    topology = RelationTopology(
        (alphabet,),
        0,
        2,
        tuple(
            Arc(0, 1, (symbol,), slot)
            for slot, (symbol, _) in enumerate(occurrences, 1)
        ),
        (Boundary(0, 0),),
        (Boundary(1, 0),),
        ("boundary", *(f"arc-{i}" for i in range(len(occurrences)))),
    )
    return RationalRelationSpec.of(
        topology, algebra, (_UNIT, *(parameter for _, parameter in occurrences))
    )


def _map_config():
    return JointAlignmentMapConfig(
        max_iterations=40,
        max_ngram_order=3,
        max_features=8,
        max_states=256,
        max_work=2048,
        step_ratio=(1, 4),
    )


def _ep_config(base=None):
    if base is None:
        base = _relation(_TAPE, ((0, _UNIT), (1, _UNIT)))
    return RegularStringEPConfig(
        bases=(("x", base),),
        features=(("x", ((0,),)),),
        penalty=0.0,
        max_sweeps=8,
        max_states=256,
        max_work=2048,
        projection_tolerance=1e-10,
    )


def _model():
    left = _relation(_TAPE, ((0, _SCORES[0]), (0, _SCORES[1]), (1, _SCORES[2])))
    right = _relation(_TAPE, ((0, _SCORES[3]), (1, _UNIT)))
    return StringFactorGraphSpec(
        variables=(("x", "Letter", 2, None),),
        relations=(("left", left), ("right", right)),
        factors=(("left", ("x",)), ("right", ("x",))),
    )


def _parameters():
    return {
        key: torch.tensor(value, dtype=torch.float64)
        for key, value in zip(
            (_UNIT, *_SCORES), (0, -0.125, -0.25, -0.625, -0.75), strict=True
        )
    }


def _roundtrip(original):
    data = dumps(original)
    value = loads(data)
    assert value == original and dumps(value) == data
    return value


def test_one_decoded_graph_supports_distinct_map_and_ep_objectives():
    model = _roundtrip(_model())
    parameter = _parameters()
    map_config = _roundtrip(_map_config())
    ep_config = _roundtrip(_ep_config())
    assert map_config.objective == "joint_alignment_map"
    assert ep_config.objective == "regular_string_ep"

    best = map_config.solve(model, parameter)
    point_scores = {0: (Fraction(-7, 8), Fraction(-1)), 1: (Fraction(-5, 8),)}
    expected_word = max(point_scores, key=lambda key: max(point_scores[key]))
    assert best.status == "optimal" and best.certificate is not None
    assert best.assignment == (("x", (expected_word,)),)
    assert best.score == best.upper_bound == max(point_scores[expected_word])
    assert all(
        alignment.words == ((expected_word,),)
        for alignment in best.certificate.alignments
    )

    inference = ep_config.infer(model, parameter)
    assert inference.status == "converged"
    point_masses = {word: sum(map(exp, paths)) for word, paths in point_scores.items()}
    posterior = point_masses[0] / sum(point_masses.values())
    assert posterior > 0.5
    assert inference.marginals["x"].log_prob((0,)).exp().item() == pytest.approx(
        posterior
    )
    assert all(site.log_scale.ndim == 0 for site in inference.sites)


def test_map_roundtrip_cannot_turn_disagreement_or_budget_into_certificate():
    model = _model()
    config = _roundtrip(_map_config().with_(max_iterations=1, max_features=0))
    result = config.solve(model, _parameters())
    assert result.status == "limit" and result.certificate is None
    assert result.assignment == (("x", (0,)),)
    assert result.score == Fraction(-7, 8)
    assert result.upper_bound == Fraction(-1, 8)
    with pytest.raises(RelationLimitError, match="max_states"):
        config.with_(max_states=0).solve(model, _parameters())
    payload = to_data(config)
    payload["root"]["fields"].append(
        {"name": "certified_optimal", "value": to_data(True)["root"]}
    )
    with pytest.raises(SerializationError, match="field"):
        from_data(payload)
    with pytest.raises(ValidationError, match="step numerator"):
        config.with_(step_ratio=(1, 0))


def test_ep_retains_reference_weights_and_separates_failure_modes():
    model = _model()
    parameter = _parameters()
    base = _relation(_TAPE, ((0, _SCORES[3]), (1, _UNIT)))
    config = _roundtrip(_ep_config(base))
    unupdated = config.with_(max_sweeps=0).infer(model, parameter)
    assert unupdated.status == "limit"
    probability = exp(-0.75) / (exp(-0.75) + 1)
    assert unupdated.marginals["x"].log_prob((0,)).exp().item() == pytest.approx(
        probability
    )
    assert (
        config.with_(projection_max_iterations=0).infer(model, parameter).status
        == "projection_failed"
    )
    with pytest.raises(RelationLimitError, match="max_states"):
        config.with_(max_states=0).infer(model, parameter)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"features": (("other", ((0,),)),)}, "one base and vocabulary"),
        ({"features": (("x", ((0,), (0,))),)}, "distinct nonempty"),
        ({"features": (("x", ((2,),)),)}, "reference alphabet"),
        ({"penalty": -0.5}, "nonnegative finite"),
        ({"damping": 1.0}, "tolerance or damping"),
    ],
)
def test_ep_configuration_checks_support_and_settings_at_construction(change, message):
    with pytest.raises(ValidationError, match=message):
        _ep_config().with_(**change)


def test_ep_configuration_checks_model_alphabet_and_base_algebra():
    base = _relation(_TAPE, ((0, _UNIT), (1, _UNIT)), algebra=MARKOV)
    with pytest.raises(ValidationError, match="one-tape LogProb"):
        _ep_config(base)
    wrong = StringSet(alphabet=FinSet(name="DifferentLetter", cardinality=2))
    base = _relation(wrong, ((0, _UNIT), (1, _UNIT)))
    with pytest.raises(TypeError, match="mistyped LogProb base"):
        _ep_config(base).infer(_model(), _parameters())
