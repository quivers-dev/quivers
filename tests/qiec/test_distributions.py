"""Distribution values in the kernel and the reference backend behind them."""

from __future__ import annotations

import math
import random

import pytest

from quivers.qiec import (
    BOOL,
    INT,
    REAL,
    DistributionValue,
    Evaluator,
    KernelError,
    KernelRegistry,
    LiteralValue,
    LogDensity,
    ReferenceBackend,
    Return,
    RuntimeDistribution,
    SiteValue,
    TensorValue,
    TupleValue,
    distribution_backend,
    dumps,
    family,
    infer_value,
    install_distribution_backend,
    loads,
    sampleable_type,
    site_type,
    tensor_type,
)
from quivers.qiec.distributions import DistributionError
from quivers.qiec.identifiers import DistributionId, SourceOrigin
from quivers.qiec.types import IndexLiteral, product_type
from quivers.qiec.kinds import NatSort

ORIGIN = SourceOrigin("distributions", ("test",), "distribution", "test")


def _normal(loc: float = 0.0, scale: float = 1.0) -> DistributionValue:
    """A well-typed ``Normal`` construction.

    Parameters
    ----------
    loc : float
        The mean.
    scale : float
        The standard deviation.

    Returns
    -------
    DistributionValue
        ``Normal(loc, scale)`` at ``Sampleable[Real]``.
    """
    return DistributionValue(
        family("Normal").id,
        "Normal",
        (("loc", LiteralValue(loc, REAL)), ("scale", LiteralValue(scale, REAL))),
        sampleable_type(REAL),
        ORIGIN,
    )


def test_a_scalar_family_infers_its_sampleable_type() -> None:
    assert infer_value(_normal(), KernelRegistry()) == sampleable_type(REAL)
    value = Evaluator().evaluate(Return(_normal(1.0, 2.0)))
    assert isinstance(value, RuntimeDistribution)
    assert value.family == "Normal"
    assert dict(value.arguments) == {"loc": 1.0, "scale": 2.0}


def test_a_vector_family_takes_its_event_shape_from_its_source_parameter() -> None:
    three = tensor_type(REAL, (IndexLiteral(3, NatSort()),))
    concentration = TupleValue(
        tuple(LiteralValue(1.0, REAL) for _ in range(3)), product_type(REAL, REAL, REAL)
    )
    # A product is not a tensor, so the argument is rejected on its type.
    with pytest.raises(KernelError, match="Tensor\\[Real\\] of rank 1"):
        infer_value(
            DistributionValue(
                family("Dirichlet").id,
                "Dirichlet",
                (("concentration", concentration),),
                sampleable_type(three),
                ORIGIN,
            ),
            KernelRegistry(),
        )


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (
            lambda: DistributionValue(
                DistributionId.derive("builtin", "Gaussian"),
                "Gaussian",
                (("loc", LiteralValue(0.0, REAL)),),
                sampleable_type(REAL),
                ORIGIN,
            ),
            "unknown distribution family",
        ),
        (
            lambda: DistributionValue(
                family("Beta").id,
                "Normal",
                (("loc", LiteralValue(0.0, REAL)),),
                sampleable_type(REAL),
                ORIGIN,
            ),
            "identity of another",
        ),
        (
            lambda: DistributionValue(
                family("Normal").id,
                "Normal",
                (("loc", LiteralValue(0.0, REAL)), ("loc", LiteralValue(1.0, REAL))),
                sampleable_type(REAL),
                ORIGIN,
            ),
            "twice",
        ),
        (
            lambda: DistributionValue(
                family("Normal").id,
                "Normal",
                (("mean", LiteralValue(0.0, REAL)),),
                sampleable_type(REAL),
                ORIGIN,
            ),
            "has no parameter 'mean'",
        ),
        (
            lambda: DistributionValue(
                family("Normal").id,
                "Normal",
                (("loc", LiteralValue(0, INT)),),
                sampleable_type(REAL),
                ORIGIN,
            ),
            "takes a Real",
        ),
        (
            lambda: DistributionValue(
                family("Normal").id, "Normal", (), sampleable_type(REAL), ORIGIN
            ),
            "no parameters",
        ),
        (
            lambda: DistributionValue(
                family("Normal").id,
                "Normal",
                (("loc", LiteralValue(0.0, REAL)),),
                sampleable_type(INT),
                ORIGIN,
            ),
            "produces",
        ),
        (
            lambda: DistributionValue(
                family("Bernoulli").id,
                "Bernoulli",
                (("probs", LiteralValue(0.5, REAL)),),
                sampleable_type(REAL),
                ORIGIN,
            ),
            "produces",
        ),
    ],
)
def test_malformed_constructions_are_rejected(build, fragment: str) -> None:
    with pytest.raises(KernelError, match=fragment):
        infer_value(build(), KernelRegistry())


def test_log_density_and_site_are_typed() -> None:
    registry = KernelRegistry()
    assert (
        infer_value(
            LogDensity(_normal(), LiteralValue(0.5, REAL), ORIGIN), registry
        ).constructor.name
        == "LogWeight"
    )  # type: ignore[union-attr]
    with pytest.raises(KernelError, match="takes a Sampleable"):
        infer_value(
            LogDensity(LiteralValue(1.0, REAL), LiteralValue(0.5, REAL), ORIGIN),
            registry,
        )
    with pytest.raises(KernelError, match="at a value of type"):
        infer_value(LogDensity(_normal(), LiteralValue(True, BOOL), ORIGIN), registry)
    assert infer_value(SiteValue("x", site_type(REAL)), registry) == site_type(REAL)
    with pytest.raises(KernelError, match="not a Site"):
        infer_value(SiteValue("x", REAL), registry)
    with pytest.raises(KernelError, match="empty"):
        infer_value(SiteValue("", site_type(REAL)), registry)
    assert Evaluator().evaluate(Return(SiteValue("x", site_type(REAL)))) == "x"


def test_values_round_trip_through_the_wire_format() -> None:
    term = Return(
        TupleValue(
            (
                _normal(),
                LogDensity(_normal(), LiteralValue(0.5, REAL), ORIGIN),
                SiteValue("x", site_type(REAL)),
            ),
            product_type(sampleable_type(REAL), REAL, site_type(REAL)),
        )
    )
    assert loads(dumps(term)) == term


def test_log_density_evaluates_through_the_reference_backend() -> None:
    value = Evaluator().evaluate(
        Return(LogDensity(_normal(0.0, 2.0), LiteralValue(1.0, REAL), ORIGIN))
    )
    assert value == pytest.approx(-0.125 - math.log(2.0) - 0.5 * math.log(2 * math.pi))


@pytest.mark.parametrize(
    ("name", "arguments", "point", "expected"),
    [
        ("Normal", {"loc": 0.0, "scale": 1.0}, 0.0, -0.5 * math.log(2 * math.pi)),
        ("Bernoulli", {"probs": 0.25}, True, math.log(0.25)),
        ("Bernoulli", {"logits": 0.0}, False, math.log(0.5)),
        ("Categorical", {"probs": (0.2, 0.3, 0.5)}, 2, math.log(0.5)),
        ("Categorical", {"logits": (0.0, 0.0)}, 1, math.log(0.5)),
        ("Beta", {"concentration1": 2.0, "concentration0": 2.0}, 0.5, math.log(1.5)),
        ("Gamma", {"concentration": 1.0, "rate": 2.0}, 1.0, math.log(2.0) - 2.0),
        ("Exponential", {"rate": 3.0}, 0.0, math.log(3.0)),
        ("Uniform", {"low": 0.0, "high": 4.0}, 1.0, -math.log(4.0)),
        ("Poisson", {"rate": 2.0}, 0, -2.0),
        ("Binomial", {"total_count": 2, "probs": 0.5}, 1, math.log(0.5)),
        ("Geometric", {"probs": 0.5}, 1, math.log(0.25)),
        ("Laplace", {"loc": 0.0, "scale": 1.0}, 0.0, -math.log(2.0)),
        ("Cauchy", {"loc": 0.0, "scale": 1.0}, 0.0, -math.log(math.pi)),
        ("LogNormal", {"loc": 0.0, "scale": 1.0}, 1.0, -0.5 * math.log(2 * math.pi)),
        (
            "HalfNormal",
            {"scale": 1.0},
            0.0,
            math.log(2.0) - 0.5 * math.log(2 * math.pi),
        ),
        ("Dirichlet", {"concentration": (1.0, 1.0)}, (0.5, 0.5), 0.0),
        (
            "StudentT",
            {"df": 1.0, "loc": 0.0, "scale": 1.0},
            0.0,
            -math.log(math.pi),
        ),
    ],
)
def test_reference_densities_match_closed_forms(
    name: str, arguments: dict[str, object], point: object, expected: float
) -> None:
    assert RuntimeDistribution(name, arguments).log_prob(point) == pytest.approx(
        expected
    )


def test_reference_samples_land_in_the_support_and_are_seeded() -> None:
    rng = random.Random(3)
    draws = [
        RuntimeDistribution(
            "Beta", {"concentration1": 2.0, "concentration0": 3.0}
        ).sample(rng)
        for _ in range(20)
    ]
    assert all(0.0 < draw < 1.0 for draw in draws)
    assert RuntimeDistribution("Categorical", {"probs": (0.0, 1.0)}).sample(rng) == 1
    assert RuntimeDistribution("Bernoulli", {"probs": 1.0}).sample(rng) is True
    simplex = RuntimeDistribution(
        "Dirichlet", {"concentration": (1.0, 2.0, 3.0)}
    ).sample(rng)
    assert sum(simplex) == pytest.approx(1.0)  # type: ignore[arg-type]
    first = RuntimeDistribution("Normal", {"loc": 0.0, "scale": 1.0}).sample(
        random.Random(9)
    )
    second = RuntimeDistribution("Normal", {"loc": 0.0, "scale": 1.0}).sample(
        random.Random(9)
    )
    assert first == second


def test_unknown_families_and_bad_parameters_are_reported() -> None:
    with pytest.raises(DistributionError, match="cannot sample Unheard"):
        RuntimeDistribution("Unheard", {"df": 3.0}).sample(random.Random(0))
    with pytest.raises(DistributionError, match="cannot score Unheard"):
        RuntimeDistribution("Unheard", {"df": 3.0}).log_prob(1.0)
    with pytest.raises(
        DistributionError,
        match="Wishart needs covariance_matrix, precision_matrix, or scale_tril",
    ):
        RuntimeDistribution("Wishart", {"df": 3.0}).log_prob(((1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(DistributionError, match="needs parameter 'scale'"):
        RuntimeDistribution("Normal", {"loc": 0.0}).log_prob(0.0)
    with pytest.raises(DistributionError, match="probs or logits"):
        RuntimeDistribution("Bernoulli", {}).log_prob(True)


def test_a_backend_can_be_installed_and_restored() -> None:
    class Constant:
        def sample(self, family: str, arguments, rng) -> object:
            return 42.0

        def log_prob(self, family: str, arguments, value: object) -> float:
            return -1.0

    previous = install_distribution_backend(Constant())
    try:
        assert isinstance(previous, ReferenceBackend)
        assert distribution_backend().__class__ is Constant
        assert RuntimeDistribution("Wishart", {"df": 3.0}).sample() == 42.0
        assert RuntimeDistribution("Wishart", {"df": 3.0}).log_prob(0.0) == -1.0
    finally:
        install_distribution_backend(previous)
    assert distribution_backend() is previous


def _vector(*entries: float) -> TensorValue:
    """A real vector literal.

    Parameters
    ----------
    *entries : float
        The entries.

    Returns
    -------
    TensorValue
        The vector at ``Tensor[Real]([n])``.
    """
    return TensorValue(
        tuple(LiteralValue(entry, REAL) for entry in entries),
        tensor_type(REAL, (IndexLiteral(len(entries), NatSort()),)),
    )


def test_tensor_literals_type_check_evaluate_and_serialize() -> None:
    registry = KernelRegistry()
    vector = _vector(1.0, 2.0, 3.0)
    assert infer_value(vector, registry) == vector.result_type
    matrix = TensorValue(
        (_vector(1.0, 2.0), _vector(3.0, 4.0)),
        tensor_type(REAL, (IndexLiteral(2, NatSort()), IndexLiteral(2, NatSort()))),
    )
    assert infer_value(matrix, registry) == matrix.result_type
    assert Evaluator().evaluate(Return(matrix)) == ((1.0, 2.0), (3.0, 4.0))
    assert loads(dumps(Return(matrix))) == Return(matrix)
    dirichlet = DistributionValue(
        family("Dirichlet").id,
        "Dirichlet",
        (("concentration", vector),),
        sampleable_type(vector.result_type),
        ORIGIN,
    )
    assert infer_value(dirichlet, registry) == sampleable_type(vector.result_type)
    density = Evaluator().evaluate(
        Return(LogDensity(dirichlet, _vector(0.2, 0.3, 0.5), ORIGIN))
    )
    assert density == pytest.approx(
        math.log(120.0) - math.log(2.0) + math.log(0.3) + 2 * math.log(0.5)
    )


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (lambda: TensorValue((LiteralValue(1.0, REAL),), REAL), "non-tensor type"),
        (
            lambda: TensorValue(
                (LiteralValue(1.0, REAL),),
                tensor_type(REAL, (IndexLiteral(2, NatSort()),)),
            ),
            "1 entries along a leading dimension of 2",
        ),
        (
            lambda: TensorValue(
                (LiteralValue(1, INT), LiteralValue(2.0, REAL)),
                tensor_type(REAL, (IndexLiteral(2, NatSort()),)),
            ),
            "entry 0 has type",
        ),
        (
            lambda: TensorValue(
                (_vector(1.0), _vector(2.0)),
                tensor_type(REAL, (IndexLiteral(2, NatSort()),)),
            ),
            "not the slice type",
        ),
    ],
)
def test_malformed_tensor_literals_are_rejected(build, fragment: str) -> None:
    with pytest.raises(KernelError, match=fragment):
        infer_value(build(), KernelRegistry())
