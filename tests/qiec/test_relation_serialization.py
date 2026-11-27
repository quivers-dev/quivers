"""QIEC round trips for sparse relations and their live parameter identities."""

from fractions import Fraction
import json
from math import log

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import COUNTING, LOG_PROB, REAL, RealAlgebra
from quivers.core.laws import ClaimSubject
from quivers.core.relations import EPSILON, Arc, Boundary, RelationTopology
from quivers.qiec import (
    ParameterId,
    RationalRelationSpec,
    SerializationError,
    dumps,
    from_data,
    loads,
    to_data,
)
from tests.evidence.compare import Accuracy, compare_gradients, compare_number


_TIED = ParameterId.derive("codec", "tied")
_BOUNDARY = ParameterId.derive("codec", "boundary")
_OTHER = ParameterId.derive("codec", "other")


def _topology():
    input_tape = StringSet(alphabet=FinSet(name="source", cardinality=2))
    output_tape = StringSet(alphabet=FinSet(name="target", cardinality=2))
    left = Arc(0, 1, (0, EPSILON), 0)
    return RelationTopology(
        tapes=(input_tape, output_tape),
        input_tapes=1,
        n_states=3,
        arcs=(left, left, Arc(1, 2, (EPSILON, 1), 1)),
        initial=(Boundary(0, 2),),
        final=(Boundary(2, 2),),
        weight_names=("prefix", "suffix", "boundary"),
    )


@pytest.mark.parametrize("algebra", [REAL, LOG_PROB])
def test_round_trip_keeps_tape_types_occurrences_and_live_parameter_identity(algebra):
    topology = _topology()
    spec = RationalRelationSpec.of(topology, algebra, (_TIED, _TIED, _BOUNDARY))
    text = dumps(spec)
    restored = loads(text)
    assert isinstance(restored, RationalRelationSpec)
    assert restored == spec
    assert dumps(restored) == text
    assert restored.topology == topology
    assert restored.topology.tapes[0] != restored.topology.tapes[1]
    assert restored.topology.arcs[0] == restored.topology.arcs[1]
    assert restored.topology.arcs[0].labels == (0, EPSILON)
    assert restored.topology.arcs[2].labels == (EPSILON, 1)
    assert restored.weight_parameters == (_TIED, _TIED, _BOUNDARY)

    is_log = algebra is LOG_PROB
    parameter = nn.Parameter(torch.empty((), dtype=torch.float64))
    with torch.no_grad():
        parameter.fill_(log(0.5) if is_log else 0.5)
    boundary = torch.tensor(0.0 if is_log else 1.0, dtype=torch.float64)
    store = {_TIED: parameter, _BOUNDARY: boundary, _OTHER: torch.zeros(())}
    relation = restored.bind(store)
    other_relation = spec.bind(store)
    original = list(
        nn.ModuleList((relation.module(), other_relation.module())).parameters()
    )
    assert len(original) == 1 and original[0] is parameter
    buffers = list(relation.module().buffers())
    assert len(buffers) == 1 and buffers[0] is boundary
    assert relation.domain == topology.tapes[0]
    assert relation.codomain == topology.tapes[1]
    assert relation.algebra is algebra
    assert not hasattr(relation, "tensor")
    query = relation.query(((0,), (1,)))

    accuracy = Accuracy(domain="linear", operations=40)
    for value in (Fraction(1, 2), Fraction(3, 4)):
        with torch.no_grad():
            parameter.fill_(log(value) if is_log else float(value))
        score = query.weight()
        (derivative,) = torch.autograd.grad(score, parameter)
        # Two paths, each with two occurrences of the same parameter.
        compare_number(
            (score.exp() if is_log else score).item(),
            2 * value**2,
            accuracy,
            "codec path mass",
        )
        compare_gradients(
            {"tied": derivative},
            {"tied": Fraction(2) if is_log else 4 * value},
            accuracy,
            "codec path derivative",
        )
    assert dumps(restored) == text
    with pytest.raises(SerializationError, match="runtime or unregistered"):
        dumps(relation)
    with pytest.raises(SerializationError, match="host callables"):
        dumps(relation.module())


def test_codec_rejects_unknown_claims_and_lookalike_algebras():
    class Lookalike(RealAlgebra):
        @property
        def subject(self):
            return REAL.subject

    ids = (_TIED, _TIED, _BOUNDARY)
    with pytest.raises(TypeError, match="shipped scalar semiring"):
        RationalRelationSpec.of(_topology(), Lookalike(), ids)
    spec = RationalRelationSpec.of(_topology(), REAL, ids)
    data = json.loads(dumps(spec))
    field = next(
        entry for entry in data["root"]["fields"] if entry["name"] == "algebra"
    )
    field["value"] = to_data(ClaimSubject(kind="operation", name=REAL.subject.name))[
        "root"
    ]
    with pytest.raises(SerializationError, match="trusted scalar semiring claim"):
        from_data(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("n_states", 1, "invalid state ID"),
        ("input_tapes", True, "input_tapes"),
        ("weight_parameters", {"$tuple": []}, "one parameter identity"),
    ),
)
def test_codec_rejects_malformed_topology_and_parameter_layout(field, value, message):
    spec = RationalRelationSpec.of(_topology(), REAL, (_TIED, _TIED, _BOUNDARY))
    data = json.loads(dumps(spec))
    fields = {entry["name"]: entry for entry in data["root"]["fields"]}
    fields[field]["value"] = value
    with pytest.raises(SerializationError, match=message):
        from_data(data)


def test_binding_refuses_unsupported_providers_and_ambiguous_parameters():
    spec = RationalRelationSpec.of(_topology(), REAL, (_TIED, _TIED, _BOUNDARY))
    value = nn.Parameter(torch.tensor(0.25))
    unit = torch.tensor(1.0)
    with pytest.raises(TypeError, match="explicit parameter mapping"):
        spec.bind(nn.Identity())
    with pytest.raises(ValueError, match="missing relation parameter"):
        spec.bind({_TIED: value})
    with pytest.raises(TypeError, match="dense scalar leaf tensor"):
        spec.bind({_TIED: nn.Parameter(torch.tensor([0.25])), _BOUNDARY: unit})
    with pytest.raises(TypeError, match="dense scalar leaf tensor"):
        spec.bind({_TIED: value.square(), _BOUNDARY: unit})
    with pytest.raises(TypeError, match="requires torch.nn.Parameter"):
        spec.bind({_TIED: torch.tensor(0.25, requires_grad=True), _BOUNDARY: unit})
    with pytest.raises(ValueError, match="common dtype and device"):
        spec.bind({_TIED: value, _BOUNDARY: unit.double()})
    separate = RationalRelationSpec.of(_topology(), REAL, (_TIED, _OTHER, _BOUNDARY))
    with pytest.raises(ValueError, match="same parameter identity"):
        separate.bind({_TIED: value, _OTHER: value, _BOUNDARY: unit})


def test_binding_validates_carrier_at_binding_and_after_updates():
    ids = (_TIED, _TIED, _BOUNDARY)
    spec = RationalRelationSpec.of(_topology(), COUNTING, ids)
    with pytest.raises(ValueError, match="outside the Counting carrier"):
        spec.bind({_TIED: torch.tensor(-1), _BOUNDARY: torch.tensor(1)})

    log_spec = RationalRelationSpec.of(_topology(), LOG_PROB, ids)
    original = nn.Parameter(torch.tensor(-float("inf")))
    relation = log_spec.bind({_TIED: original, _BOUNDARY: torch.tensor(0.0)})
    assert relation.weight(((0,), (1,))).item() == -float("inf")
    with torch.no_grad():
        original.fill_(float("nan"))
    with pytest.raises(ValueError, match="outside the .* carrier"):
        relation.weight(((0,), (1,)))
