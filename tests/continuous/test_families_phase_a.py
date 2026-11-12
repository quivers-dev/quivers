"""Verification tests for the 13 Phase A distribution families.

Each test asserts:

* the `Conditional<Name>` class exists in `quivers.continuous.families`
  and subclasses
  [`ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism];
* the class exposes class-level `arg_constraints` and `support`
  attributes the transpile lower pipeline reads;
* the instance-level `.support` returns a
  [`torch.distributions.constraints.Constraint`][torch.distributions.constraints.Constraint];
* `log_prob` and `rsample` work for in-support values;
* the FAMILY_META lookup resolves to the right class.
"""

from __future__ import annotations


import pytest
import torch
from torch.distributions import constraints as c
from torch.distributions import transforms as T

from quivers.continuous.families import (
    ConditionalBinomial,
    ConditionalGeometric,
    ConditionalIndependent,
    ConditionalLKJCholesky,
    ConditionalLogisticNormal,
    ConditionalMixture,
    ConditionalNegativeBinomial,
    ConditionalOneHotCategorical,
    ConditionalPoisson,
    ConditionalTransformed,
    ConditionalVonMises,
    LKJCorrelationFactor,
    Truncated,
)
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
from quivers.transpile.family_meta import FAMILY_META


@pytest.fixture
def domain() -> FinSet:
    return FinSet(name="A", cardinality=3)


@pytest.fixture
def codomain_1d() -> Euclidean:
    return Euclidean(name="E", dim=1)


@pytest.fixture
def codomain_3d() -> Euclidean:
    return Euclidean(name="E", dim=3)


@pytest.fixture
def x_batch() -> torch.Tensor:
    return torch.tensor([0, 1, 2])


# ---------------------------------------------------------------------------
# Subclass / interface sanity for every Phase A family.
# ---------------------------------------------------------------------------


_PHASE_A_CLASSES = [
    ConditionalPoisson,
    ConditionalNegativeBinomial,
    ConditionalGeometric,
    ConditionalBinomial,
    ConditionalVonMises,
    ConditionalLogisticNormal,
    ConditionalOneHotCategorical,
    ConditionalLKJCholesky,
    ConditionalMixture,
    ConditionalIndependent,
    ConditionalTransformed,
    Truncated,
    LKJCorrelationFactor,
]


@pytest.mark.parametrize("cls", _PHASE_A_CLASSES, ids=lambda c: c.__name__)
def test_subclass_of_continuous_morphism(cls: type) -> None:
    assert issubclass(cls, ContinuousMorphism)


@pytest.mark.parametrize("cls", _PHASE_A_CLASSES, ids=lambda c: c.__name__)
def test_class_level_arg_constraints_is_dict(cls: type) -> None:
    assert isinstance(cls.arg_constraints, dict)
    for key, val in cls.arg_constraints.items():
        assert isinstance(key, str)
        assert isinstance(val, c.Constraint)


@pytest.mark.parametrize("cls", _PHASE_A_CLASSES, ids=lambda c: c.__name__)
def test_class_level_support_is_constraint(cls: type) -> None:
    assert isinstance(cls.support, c.Constraint)


# ---------------------------------------------------------------------------
# Per-family construction + support / log_prob / rsample.
# ---------------------------------------------------------------------------


def test_poisson_support_and_sampling(
    domain: FinSet, codomain_1d: Euclidean, x_batch: torch.Tensor
) -> None:
    m = ConditionalPoisson(domain, codomain_1d)
    assert m.support == c.nonnegative_integer
    y = torch.tensor([[0.0], [1.0], [2.0]])
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()
    s = m.rsample(x_batch)
    assert s.shape == (3, 1)


def test_negative_binomial_support_and_sampling(
    domain: FinSet, codomain_1d: Euclidean, x_batch: torch.Tensor
) -> None:
    m = ConditionalNegativeBinomial(domain, codomain_1d)
    assert m.support == c.nonnegative_integer
    y = torch.tensor([[0.0], [1.0], [2.0]])
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()
    s = m.rsample(x_batch)
    assert s.shape == (3, 1)


def test_geometric_support_and_sampling(
    domain: FinSet, codomain_1d: Euclidean, x_batch: torch.Tensor
) -> None:
    m = ConditionalGeometric(domain, codomain_1d)
    assert m.support == c.nonnegative_integer
    y = torch.tensor([[0.0], [1.0], [2.0]])
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()


def test_binomial_support_includes_upper_bound(
    domain: FinSet, codomain_1d: Euclidean, x_batch: torch.Tensor
) -> None:
    m = ConditionalBinomial(domain, codomain_1d, total_count=10)
    # Class-level `support` is the broadest (nonnegative_integer).
    assert ConditionalBinomial.support == c.nonnegative_integer
    # The instance-state holds the per-call upper bound.
    assert m._total_count == 10
    y = torch.tensor([[0.0], [3.0], [10.0]])
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()


def test_vonmises_support(
    domain: FinSet, codomain_1d: Euclidean, x_batch: torch.Tensor
) -> None:
    m = ConditionalVonMises(domain, codomain_1d)
    assert m.support == c.real
    y = torch.tensor([[0.1], [0.2], [0.3]])
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()


def test_logistic_normal_support_is_simplex(
    domain: FinSet, codomain_3d: Euclidean, x_batch: torch.Tensor
) -> None:
    m = ConditionalLogisticNormal(domain, codomain_3d)
    assert m.support == c.simplex
    y = torch.tensor(
        [
            [0.33, 0.33, 0.34],
            [0.5, 0.3, 0.2],
            [0.1, 0.1, 0.8],
        ]
    )
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()


def test_one_hot_categorical_support(
    domain: FinSet, codomain_3d: Euclidean, x_batch: torch.Tensor
) -> None:
    m = ConditionalOneHotCategorical(domain, codomain_3d)
    # torch's OneHotCategorical.support is OneHot().
    from torch.distributions import OneHotCategorical

    assert m.support is OneHotCategorical.support
    y = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()


def test_lkj_cholesky_support(domain: FinSet, x_batch: torch.Tensor) -> None:
    cod = Euclidean(name="L", dim=3)
    m = ConditionalLKJCholesky(domain, cod)
    assert m.support == c.corr_cholesky
    s = m.rsample(x_batch)
    lp = m.log_prob(x_batch, s)
    assert torch.isfinite(lp).all()


def test_mixture_supports_component_support(
    domain: FinSet, codomain_1d: Euclidean, x_batch: torch.Tensor
) -> None:
    m = ConditionalMixture(domain, codomain_1d, ConditionalPoisson, num_components=2)
    # Class-level default is `real`; per-instance still resolves to `real`.
    assert ConditionalMixture.support == c.real
    y = torch.tensor([[0.0], [1.0], [2.0]])
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()


def test_independent_wraps_base(
    domain: FinSet, codomain_3d: Euclidean, x_batch: torch.Tensor
) -> None:
    base = ConditionalPoisson(domain, codomain_3d)
    m = ConditionalIndependent(base)
    y = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 3.0],
        ]
    )
    lp = m.log_prob(x_batch, y)
    assert torch.isfinite(lp).all()


def test_transformed_wraps_base(
    domain: FinSet, codomain_1d: Euclidean, x_batch: torch.Tensor
) -> None:
    base = ConditionalPoisson(domain, codomain_1d)
    m = ConditionalTransformed(base, [T.ExpTransform()])
    assert isinstance(m.support, c.Constraint)
    s = m.rsample(x_batch)
    assert torch.isfinite(s).all()


def test_truncated_accepts_base_and_bounds(
    domain: FinSet, codomain_1d: Euclidean, x_batch: torch.Tensor
) -> None:
    base = ConditionalPoisson(domain, codomain_1d)
    m = Truncated(base, lower=0.0, upper=5.0)
    assert m._lower == 0.0
    assert m._upper == 5.0
    # In-bounds.
    y_in = torch.tensor([[1.0], [2.0], [3.0]])
    lp_in = m.log_prob(x_batch, y_in)
    assert torch.isfinite(lp_in).all()
    # Out-of-bounds is -inf.
    y_out = torch.tensor([[10.0], [20.0], [30.0]])
    lp_out = m.log_prob(x_batch, y_out)
    assert (lp_out == float("-inf")).all()


def test_lkj_correlation_factor_support(domain: FinSet, x_batch: torch.Tensor) -> None:
    m = LKJCorrelationFactor(dim=3, eta=1.5, domain=domain)
    assert m.support == c.corr_cholesky
    s = m.rsample(x_batch)
    lp = m.log_prob(x_batch, s)
    assert torch.isfinite(lp).all()


# ---------------------------------------------------------------------------
# FAMILY_META resolution.
# ---------------------------------------------------------------------------


_FAMILY_META_LOOKUP = [
    ("Poisson", ConditionalPoisson),
    ("NegativeBinomial", ConditionalNegativeBinomial),
    ("Geometric", ConditionalGeometric),
    ("Binomial", ConditionalBinomial),
    ("VonMises", ConditionalVonMises),
    ("LogisticNormal", ConditionalLogisticNormal),
    ("OneHotCategorical", ConditionalOneHotCategorical),
    ("LKJCholesky", ConditionalLKJCholesky),
    ("Mixture", ConditionalMixture),
    ("Independent", ConditionalIndependent),
    ("Transformed", ConditionalTransformed),
    ("Truncated", Truncated),
    ("LKJCorrelationFactor", LKJCorrelationFactor),
]


@pytest.mark.parametrize(
    "qvr_name,expected_cls",
    _FAMILY_META_LOOKUP,
    ids=[n for n, _ in _FAMILY_META_LOOKUP],
)
def test_family_meta_resolves_to_real_class(qvr_name: str, expected_cls: type) -> None:
    meta = FAMILY_META[qvr_name]
    # `distribution_class` carries the torch (or torch-shim) class
    # the transpile lower pipeline reads `arg_constraints` and
    # `.support` from; the `Conditional*` ContinuousMorphism the
    # inference layer instantiates lives on the `quivers_class`
    # slot.
    assert meta.quivers_class is expected_cls
    assert meta.quivers_class.__module__ == "quivers.continuous.families"


def test_family_meta_carries_all_thirteen_phase_a_entries() -> None:
    for qvr_name, _ in _FAMILY_META_LOOKUP:
        assert qvr_name in FAMILY_META


def test_family_meta_total_count_unchanged() -> None:
    # A registered family is one a target can be asked to emit, so
    # the count is pinned: a family arriving without its per-target
    # names, or leaving, changes what `transpile` accepts, and this
    # is where that shows up.
    assert len(FAMILY_META) == 60
