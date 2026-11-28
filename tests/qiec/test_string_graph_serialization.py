"""Check graph wire scope, shared parameter identities, and exact inference."""

import pytest
import torch
from didactic.api import ValidationError
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV, REAL
from quivers.core.relations import Arc, Boundary, RelationTopology
from quivers.qiec import (
    ParameterId,
    RationalRelationSpec,
    SerializationError,
    StringFactorGraphSpec,
    dumps,
    from_data,
    loads,
    to_data,
)

_TIED = ParameterId.derive("string-graph", "tied")
_UNIT = ParameterId.derive("string-graph", "unit")
_OTHER = ParameterId.derive("string-graph", "other")
_A = StringSet(alphabet=FinSet(name="A", cardinality=2))
_B = StringSet(alphabet=FinSet(name="B", cardinality=2))


def _prior(algebra, tied=_TIED, unit=_UNIT):
    return RationalRelationSpec.of(
        RelationTopology(
            (_A,),
            0,
            2,
            (Arc(0, 1, (0,), 0), Arc(0, 1, (1,), 1)),
            (Boundary(0, 1),),
            (Boundary(1, 1),),
            ("zero", "unit"),
        ),
        algebra,
        (tied, unit),
    )


def _coupling(algebra):
    repeated = Arc(0, 1, (0, 1), 0)
    return RationalRelationSpec.of(
        RelationTopology(
            (_A, _B),
            1,
            2,
            (repeated, repeated, Arc(0, 1, (1, 1), 1)),
            (Boundary(0, 1),),
            (Boundary(1, 1),),
            ("tied", "unit"),
        ),
        algebra,
        (_TIED, _UNIT),
    )


def _graph(algebra):
    return StringFactorGraphSpec(
        variables=(("x", "A", 2, None), ("z", "A", 2, None), ("y", "B", 2, (1,))),
        relations=(
            ("prior", _prior(algebra)),
            ("coupling", _coupling(algebra)),
            ("unused", _prior(algebra, _OTHER)),
        ),
        factors=(("prior", ("x",)), ("coupling", ("x", "y")), ("prior", ("z",))),
    )


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_round_trip_preserves_relation_reuse_parameter_sharing_and_observations(
    algebra, monkeypatch
):
    spec = _graph(algebra)
    encoded = dumps(spec)
    decoded = loads(encoded)
    assert isinstance(decoded, StringFactorGraphSpec)
    assert decoded == spec
    assert dumps(decoded) == encoded

    bound_specs = []
    bind = RationalRelationSpec.bind

    def track(rel, parameters):
        bound_specs.append(rel)
        return bind(rel, parameters)

    monkeypatch.setattr(RationalRelationSpec, "bind", track)
    parameter = nn.Parameter(torch.empty((), dtype=torch.float64))
    unit = parameter.new_tensor(0.0 if algebra is LOG_PROB else 1.0)
    with torch.no_grad():
        parameter.fill_(-1.0 if algebra is LOG_PROB else 0.5)
    graph = decoded.bind({_TIED: parameter, _UNIT: unit})
    assert bound_specs == [spec.relations[0][1], spec.relations[1][1]]
    query = graph.tree_marginal("x", max_states=512, max_work=2048)
    for value in (0.25, 0.8):
        with torch.no_grad():
            p = parameter.new_tensor(value)
            parameter.copy_(p.log() if algebra is LOG_PROB else p)
        raw = parameter.exp() if algebra is LOG_PROB else parameter
        expected_mass = (2 * raw.square() + 1) * (raw + 1)
        intended = expected_mass.log() if algebra is LOG_PROB else expected_mass
        actual = query.partition()
        torch.testing.assert_close(actual, intended)
        torch.testing.assert_close(
            query.log_prob((0,)).exp(), 2 * raw.square() / (2 * raw.square() + 1)
        )
        (actual_gradient,) = torch.autograd.grad(actual, parameter)
        (oracle_gradient,) = torch.autograd.grad(intended, parameter)
        torch.testing.assert_close(actual_gradient, oracle_gradient)


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("factors", (("unknown", ("x",)),), "unknown relation"),
        ("factors", (("coupling", ("x", "x")),), "one distinct variable"),
        ("factors", (("coupling", ("y", "x")),), "mistyped variable"),
        ("variables", (("x", "A", 2, None), ("x", "A", 2, None)), "must be distinct"),
        ("variables", (("x", "A", 2, (2,)),), "invalid observed symbol"),
        ("relations", (("prior", 0),), "RationalRelationSpec"),
        ("variables", (("x", "A", True, None),), "got bool"),
    ],
)
def test_decode_checks_nominal_scope_and_observations(field, value, diagnostic):
    data = to_data(_graph(MARKOV))
    fields = {entry["name"]: entry for entry in data["root"]["fields"]}
    fields[field]["value"] = to_data(value)["root"]
    with pytest.raises(SerializationError, match=diagnostic):
        from_data(data)


def test_model_checks_algebra_only_for_referenced_relations():
    spec = _graph(MARKOV)
    with pytest.raises(ValidationError, match="one shared algebra"):
        StringFactorGraphSpec(
            variables=spec.variables,
            relations=(("prior", _prior(MARKOV)), ("coupling", _coupling(LOG_PROB))),
            factors=spec.factors,
        )
    with pytest.raises(ValidationError, match="Markov or LogProb"):
        _graph(REAL)
    with pytest.raises(
        ValidationError, match="relation names must be nonempty and distinct"
    ):
        StringFactorGraphSpec(
            variables=spec.variables,
            relations=(*spec.relations, spec.relations[0]),
            factors=spec.factors,
        )


def test_binding_checks_parameter_identity_and_dtype_across_relation_boundaries():
    spec = StringFactorGraphSpec(
        variables=(("x", "A", 2, None), ("z", "A", 2, None)),
        relations=(
            ("a", _prior(MARKOV, _TIED, _TIED)),
            ("b", _prior(MARKOV, _OTHER, _OTHER)),
        ),
        factors=(("a", ("x",)), ("b", ("z",))),
    )
    x = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="missing string graph parameter"):
        spec.bind({_TIED: x})
    with pytest.raises(ValueError, match="same parameter identity"):
        spec.bind({_TIED: x, _OTHER: x})
    with pytest.raises(ValueError, match="common dtype and device"):
        spec.bind({_TIED: x, _OTHER: x.float().detach()})
