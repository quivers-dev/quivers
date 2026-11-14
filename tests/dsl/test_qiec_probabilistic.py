"""Distributions, sites, and log densities from QVR source to the reference machine."""

from __future__ import annotations

import math

import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec import (
    REAL,
    DistributionValue,
    LogDensity,
    RuntimeConfiguration,
    RuntimeSelection,
    SiteValue,
    run_named,
    sampleable_type,
    seed_reference_rng,
    site_type,
    tensor_type,
)
from quivers.qiec.builtins import RANDOM, SCORE
from quivers.qiec.kinds import NatSort
from quivers.qiec.types import IndexLiteral

_MODEL = """\
instance random : Random
instance score : Score

handler draw for Random : Real -> Real [coverage=total, implementation=foreign]
    sample[A : Type] resumes 1

handler accumulate for Score : Real -> (Real * LogWeight) [coverage=total, implementation=foreign]
    add resumes 1

define model(mu : Real) : Real !{random, score} =
    let x <- perform random.sample[Real](site("x"), Normal(mu, 1.0))
    perform score.add(log_prob(Normal(0.0, 2.0), x))
    return x

define run(mu : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle random with draw in
            model(mu)
"""

_RUNTIME = RuntimeConfiguration(
    selections=(
        RuntimeSelection(
            provider="core",
            options={
                "handlers": {"draw": {"kind": "draw"}, "accumulate": {"kind": "score"}}
            },
        ),
    )
)


def test_prelude_interfaces_resolve_without_declaration() -> None:
    """``Random`` and ``Score`` join the module under the prelude's identities."""
    module = lower_qvr_to_qiec(parse(_MODEL), file_path="model.qvr")
    names = {effect.ref.name for effect in module.effects}
    assert names == {"Random", "Score"}
    identities = {effect.ref.id for effect in module.effects}
    assert identities == {RANDOM.id, SCORE.id}


def test_family_application_site_and_log_density_lower_to_typed_values() -> None:
    module = lower_qvr_to_qiec(parse(_MODEL), file_path="model.qvr")
    model = next(item for item in module.computations if item.name == "model")
    request = model.body.first.request  # type: ignore[union-attr]
    site, distribution = request.arguments
    assert site == SiteValue(label="x", result_type=site_type(REAL))
    assert isinstance(distribution, DistributionValue)
    assert distribution.name == "Normal"
    assert [name for name, _ in distribution.arguments] == ["loc", "scale"]
    assert distribution.result_type == sampleable_type(REAL)
    score = model.body.then.first.request  # type: ignore[union-attr]
    (weight,) = score.arguments
    assert isinstance(weight, LogDensity)


def test_a_sampled_and_scored_model_runs_on_the_reference_machine() -> None:
    """The draw is seeded and the accumulated weight is the declared density."""
    module = lower_qvr_to_qiec(parse(_MODEL), file_path="model.qvr")
    seed_reference_rng(7)
    first = run_named(module, "run", (1.5,), runtime=_RUNTIME)
    seed_reference_rng(7)
    second = run_named(module, "run", (1.5,), runtime=_RUNTIME)
    assert first.value == second.value
    x, weight = first.value  # type: ignore[misc]
    expected = -0.5 * (x / 2.0) ** 2 - math.log(2.0) - 0.5 * math.log(2 * math.pi)
    assert weight == pytest.approx(expected)
    assert first.to_data()["result_type"] == "Product2[Real, LogWeight]"
    events = [event.event for event in first.trace]
    assert events.count("operation.handled") == 2


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("    return Gaussian(0.0, 1.0)\n", "unknown builtin"),
        ("    return Normal(0.0, 1.0, 2.0)\n", "takes at most 2 parameters"),
        ("    return Normal(0.0, true)\n", "takes a Real"),
        ("    return log_prob(1.0, 2.0)\n", "takes a Sampleable"),
        ("    return log_prob(Normal(0.0, 1.0), true)\n", "at a value of type"),
        ('    return site("x")\n', "fixes its type"),
        ("    return Bernoulli(0.3)\n", "Sampleable"),
    ],
)
def test_ill_formed_distribution_values_are_source_located(
    body: str, fragment: str
) -> None:
    """Each malformed construction names its defect at the source line.

    The last case is well formed on its own: ``Bernoulli`` samples a
    Boolean, so it is the declared ``Sampleable[Real]`` result that
    rejects it.
    """
    source = "define probe() : Sampleable[Real] !{} =\n" + body
    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parse(source), file_path="probe.qvr")
    assert fragment in str(captured.value), str(captured.value)
    assert captured.value.line == (1 if fragment == "Sampleable" else 2)


def test_an_annotated_binding_types_a_site() -> None:
    source = (
        "define probe() : Site[Int] !{} =\n"
        '    let s : Site[Int] = site("count")\n'
        "    return s\n"
    )
    module = lower_qvr_to_qiec(parse(source), file_path="probe.qvr")
    assert run_named(module, "probe").value == "count"


def test_list_literals_lower_to_tensors_and_fix_event_shapes() -> None:
    source = """\
define probs() : Sampleable[Int] !{} =
    return Categorical([0.2, 0.8])

define simplex() : Sampleable[Tensor[Real]([3])] !{} =
    return Dirichlet([1.0, 2.0, 3])

define matrix() : Tensor[Real]([2, 2]) !{} =
    return [[1.0, 2.0], [3.0, 4.0]]

define counts() : Tensor[Int]([3]) !{} =
    return [1, 2, 3]

define mixed() : Tensor[Real]([2]) !{} =
    return [1, 2.5]

define scored() : LogWeight !{} =
    let d = Categorical([0.2, 0.8])
    return log_prob(d, 1)

define correlation() : Sampleable[Tensor[Real]([3, 3])] !{} =
    return LKJCholesky(2.0)
"""
    module = lower_qvr_to_qiec(parse(source), file_path="tensors.qvr")
    assert run_named(module, "matrix").value == ((1.0, 2.0), (3.0, 4.0))
    assert run_named(module, "counts").value == (1, 2, 3)
    assert run_named(module, "mixed").value == (1.0, 2.5)
    assert run_named(module, "scored").value == pytest.approx(math.log(0.8))
    simplex = next(item for item in module.computations if item.name == "simplex")
    assert simplex.type.result == sampleable_type(
        tensor_type(REAL, (IndexLiteral(value=3, sort=NatSort()),))
    )
    correlation = next(
        item for item in module.computations if item.name == "correlation"
    )
    assert correlation.type.result == sampleable_type(
        tensor_type(
            REAL,
            (
                IndexLiteral(value=3, sort=NatSort()),
                IndexLiteral(value=3, sort=NatSort()),
            ),
        )
    )


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("    return [1.0, true]\n", "share one type"),
        ("    return []\n", "0 entries"),
        ("    return Dirichlet([1.0, [2.0]])\n", "share one type"),
    ],
)
def test_ill_formed_tensor_literals_are_source_located(
    body: str, fragment: str
) -> None:
    source = "define probe() : Tensor[Real]([2]) !{} =\n" + body
    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parse(source), file_path="probe.qvr")
    assert fragment in str(captured.value), str(captured.value)


def test_a_tensor_literal_against_the_wrong_expected_shape_is_rejected() -> None:
    source = "define probe() : Tensor[Real]([3]) !{} =\n    return [1.0, 2.0]\n"
    with pytest.raises(QiecDiagnosticError, match="2 entries"):
        lower_qvr_to_qiec(parse(source), file_path="probe.qvr")
    source = "define probe() : Sampleable[Tensor[Real]([2, 2])] !{} =\n    return LKJCholesky(2.0)\n"
    module = lower_qvr_to_qiec(parse(source), file_path="probe.qvr")
    assert module.computations[0].type.result == sampleable_type(
        tensor_type(
            REAL,
            (
                IndexLiteral(value=2, sort=NatSort()),
                IndexLiteral(value=2, sort=NatSort()),
            ),
        )
    )
    source = (
        "define probe() : Real !{} =\n    let d = LKJCholesky(2.0)\n    return 1.0\n"
    )
    with pytest.raises(QiecDiagnosticError, match="no parameter carries"):
        lower_qvr_to_qiec(parse(source), file_path="probe.qvr")
