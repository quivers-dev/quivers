"""Measure queries retain batches and reduce independent event products."""

import math

import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec, QiecDiagnosticError
from quivers.qiec import (
    REAL,
    LOG_WEIGHT,
    DistributionValue,
    Evaluator,
    KernelError,
    KernelRegistry,
    LiteralValue,
    LogMass,
    Return,
    RuntimeDistribution,
    dumps,
    loads,
    family,
    infer_value,
    run_named,
    sampleable_type,
    tensor_type,
)
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.kinds import NAT
from quivers.qiec.terms import PlateAxis, PlateShape
from quivers.qiec.types import IndexLiteral
from quivers.qiec.types import IndexVariable
from quivers.qiec.substitution import StaticSubstitution, substitute_value


ORIGIN = SourceOrigin(
    module="mass", structural_path=("test",), role="mass", source_protocol="test"
)


def _normal(plate=PlateShape()):
    shape = tuple(axis.size for axis in (*plate.batch, *plate.event))
    element = tensor_type(REAL, shape) if shape else REAL
    return DistributionValue(
        family=family("Normal").id,
        name="Normal",
        arguments=(
            ("loc", LiteralValue(value=0.0, type=REAL)),
            ("scale", LiteralValue(value=1.0, type=REAL)),
        ),
        result_type=sampleable_type(element),
        origin=ORIGIN,
        plate=plate,
    )


def test_total_and_batched_queries_check_and_roundtrip():
    row = PlateAxis(name="row", size=IndexLiteral(value=2, sort=NAT))
    event = PlateAxis(name="event", size=IndexLiteral(value=3, sort=NAT))
    distribution = _normal(PlateShape(batch=(row,), event=(event,)))
    total = LogMass(sampleable=distribution, origin=ORIGIN)
    batched = total.with_(batch=(row,))
    assert infer_value(total, KernelRegistry()) == LOG_WEIGHT
    assert infer_value(batched, KernelRegistry()) == tensor_type(
        LOG_WEIGHT, (row.size,)
    )
    for term, expected in ((total, 0.0), (batched, (0.0, 0.0))):
        assert loads(dumps(Return(value=term))) == Return(value=term)
        assert Evaluator().evaluate(Return(value=term)) == expected
    with pytest.raises(KernelError, match="batch axes"):
        infer_value(total.with_(batch=(event,)), KernelRegistry())
    with pytest.raises(KernelError, match="takes a Sampleable"):
        infer_value(
            LogMass(sampleable=LiteralValue(value=1.0, type=REAL), origin=ORIGIN),
            KernelRegistry(),
        )


def test_mass_substitution_preserves_origin_and_specializes_both_axis_records():
    size = IndexVariable(name="n", sort=NAT)
    row = PlateAxis(name="row", size=size)
    term = LogMass(
        sampleable=_normal(PlateShape(batch=(row,))), origin=ORIGIN, batch=(row,)
    )
    actual = substitute_value(
        term, StaticSubstitution(indices=(("n", IndexLiteral(value=2, sort=NAT)),))
    )
    assert actual.origin == ORIGIN
    assert actual.batch == actual.sampleable.plate.batch
    assert actual.batch[0].size.value == 2
    assert Evaluator().evaluate(Return(value=actual)) == (0.0, 0.0)


@pytest.mark.parametrize("retained,expected_shape", [(0, ()), (1, (2,)), (2, (2, 2))])
def test_reference_mass_preserves_requested_batch_prefix_and_sums_events(
    retained, expected_shape
):
    base = RuntimeDistribution(family="Uniform", arguments={"low": 0.0, "high": 1.0})
    high = (((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)), ((0.7, 0.8, 0.9), (0.2, 0.4, 0.8)))
    measure = RuntimeDistribution(
        family="Restrict",
        arguments={"base": base, "low": 0.0, "high": high},
        batch=(2, 2),
        event=(3,),
        ranks={"base": 0, "low": 0, "high": 0},
    )
    row = tuple(
        tuple(sum(math.log(p) for p in event) for event in batch) for batch in high
    )
    expected = (
        sum(sum(batch) for batch in row)
        if retained == 0
        else (tuple(sum(batch) for batch in row) if retained == 1 else row)
    )
    actual = measure.log_mass(batch_axes=retained)

    def flatten(value):
        return (
            [value]
            if isinstance(value, float)
            else [x for item in value for x in flatten(item)]
        )

    assert flatten(actual) == pytest.approx(flatten(expected), abs=2e-14)
    shape = ()
    while isinstance(actual, tuple):
        shape += (len(actual),)
        actual = actual[0]
    assert shape == expected_shape


@pytest.mark.parametrize("rank", [-1, 3, True, 1.5])
def test_reference_mass_rejects_invalid_retained_rank(rank):
    measure = RuntimeDistribution(
        family="Normal", arguments={"loc": 0.0, "scale": 1.0}, batch=(2,)
    )
    with pytest.raises(ValueError, match="batch rank"):
        measure.log_mass(batch_axes=rank)


def test_source_mass_is_a_pure_structural_query_with_shared_normalization():
    source = """define mass() : LogWeight !{} =
    let base = Uniform(0.0, 1.0)
    return log_mass(Restrict(base, 0.2, 0.7))
define normalized() : LogWeight !{} =
    return log_mass(Normalize(Restrict(Uniform(0.0, 1.0), 0.2, 0.7)))
"""
    module = lower_qvr_to_qiec(parse(source), file_path="mass.qvr")
    assert run_named(module, "mass").value == pytest.approx(math.log(0.5))
    assert run_named(module, "normalized").value == 0.0


@pytest.mark.parametrize(
    "expression", ["log_mass()", "log_mass(1.0)", "log_mass(Normal(0.0, 1.0), 0.0)"]
)
def test_source_mass_rejects_arity_and_carrier(expression):
    source = f"define invalid() : LogWeight !{{}} =\n    return {expression}\n"
    with pytest.raises(QiecDiagnosticError, match="log_mass"):
        lower_qvr_to_qiec(parse(source), file_path="invalid.qvr")
