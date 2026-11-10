"""The semantic family registry agrees with the host-side family metadata.

The registry is the meaning of a family; the transpile metadata and the
``torch.distributions`` classes are host views of it. These tests hold the
views to the registry, so a family cannot gain a parameter, change its
support, or be spelled differently on a target without the semantic
record changing first.
"""

from __future__ import annotations

import inspect
from typing import cast

import torch.distributions.constraints as c

from quivers.qiec import BOOL, INT, REAL
from quivers.qiec.families import FAMILIES, DistributionFamily, family
from quivers.qiec.identifiers import DistributionId
from quivers.transpile.family_meta import FAMILY_META

_MANUAL_PARAMETERS = {
    "Uniform": ("low", "high"),
    "Wishart": ("df", "covariance_matrix", "precision_matrix", "scale_tril"),
}

#: Families whose torch classes constrain only some of their parameters:
#: the compositional ones declare only the tensor parameters, and the
#: relaxed ones take their temperature unconstrained, so the registry
#: names the rest by hand.
_COMPOSITIONAL_EXTRAS = {
    "RelaxedBernoulli": ("temperature",),
    "RelaxedOneHotCategorical": ("temperature",),
    "Mixture": ("weights", "component"),
    "Independent": ("base", "reinterpreted_batch_ndims"),
    "Transformed": ("base", "transforms"),
    "Truncated": ("base",),
    "Restrict": ("base", "low", "high"),
    "Normalize": ("base",),
    "PointMass": ("value",),
}


#: Parameters whose registry rank exceeds the torch shim's: a Gaussian
#: mixture's location and scale are one entry per component, which the
#: shim declares scalar because each component is.
_RANK_OVERRIDES: dict[str, dict[str, int]] = {"MixtureNormal": {"loc": 1, "scale": 1}}


def _symbolic(constraint: object) -> tuple[str, int]:
    """Read a torch constraint as the registry's symbolic vocabulary.

    Parameters
    ----------
    constraint : object
        A ``torch.distributions.constraints`` constraint.

    Returns
    -------
    tuple[str, int]
        The symbolic constraint and the tensor rank it applies to.
    """
    if isinstance(constraint, c.independent):
        base, rank = _symbolic(constraint.base_constraint)
        return base, rank + constraint.reinterpreted_batch_ndims
    rank = getattr(constraint, "event_dim", 0) or 0
    if isinstance(constraint, c._Real):
        return "real", rank
    if isinstance(constraint, c._Boolean):
        return "boolean", rank
    if isinstance(constraint, c._Simplex):
        return "simplex", rank
    if isinstance(constraint, c._OneHot):
        return "one_hot", rank
    if isinstance(constraint, c._LowerCholesky):
        return "lower_cholesky", rank
    if isinstance(constraint, c._CorrCholesky):
        return "corr_cholesky", rank
    if isinstance(constraint, c._PositiveDefinite):
        return "positive_definite", rank
    if isinstance(constraint, c._IntegerGreaterThan):
        return "nonnegative_integer", rank
    if isinstance(constraint, c._GreaterThanEq):
        return "nonnegative", rank
    if isinstance(constraint, c._GreaterThan):
        return "positive", rank
    if isinstance(constraint, c._HalfOpenInterval):
        return "unit_interval_half_open", rank
    if isinstance(constraint, c._Interval):
        return "unit_interval", rank
    return "dependent", rank


def test_every_transpile_family_has_a_semantic_record() -> None:
    assert set(FAMILY_META) == set(FAMILIES)


def test_registry_identities_are_distinct_and_stable() -> None:
    identities = {item.id for item in FAMILIES.values()}
    assert len(identities) == len(FAMILIES)
    assert family("Normal").id == DistributionId.derive("builtin", "Normal")


def _constructor_order(cls: type, constrained: tuple[str, ...]) -> tuple[str, ...]:
    """The constrained parameters in the class's positional order.

    Parameters
    ----------
    cls : type
        The torch distribution class.
    constrained : tuple[str, ...]
        The names ``arg_constraints`` lists.

    Returns
    -------
    tuple[str, ...]
        The constructor's parameters that carry a constraint, in the
        order a positional call binds them, followed by any constrained
        parameter the constructor does not name.
    """
    try:
        positional = tuple(inspect.signature(cls.__init__).parameters)[1:]
    except TypeError, ValueError:
        positional = ()
    leading = tuple(item for item in positional if item in constrained)
    return (*leading, *(item for item in constrained if item not in leading))


def test_parameters_constraints_and_ranks_match_the_torch_classes() -> None:
    """Each family's parameters are the torch class's, bound positionally
    as its constructor binds them."""
    for name, meta in FAMILY_META.items():
        record = family(name)
        constraints = meta.distribution_class.arg_constraints
        if isinstance(constraints, property):
            assert record.parameter_names == _MANUAL_PARAMETERS[name], name
            continue
        constraints = cast(dict[str, object], constraints)
        extras = _COMPOSITIONAL_EXTRAS.get(name, ())
        declared = tuple(item for item in record.parameter_names if item not in extras)
        expected = _constructor_order(meta.distribution_class, tuple(constraints))
        assert declared == expected, name
        assert set(extras) <= set(record.parameter_names), name
        for parameter in record.parameters:
            if parameter.name in extras:
                continue
            symbolic, rank = _symbolic(constraints[parameter.name])
            rank = _RANK_OVERRIDES.get(name, {}).get(parameter.name, rank)
            assert (parameter.constraint, parameter.rank) == (symbolic, rank), (
                name,
                parameter.name,
            )


def test_support_event_rank_and_targets_match_the_transpile_metadata() -> None:
    for name, meta in FAMILY_META.items():
        record = family(name)
        assert record.event_rank == meta.event_rank, name
        assert dict(record.targets) == dict(meta.target_names), name
        support = meta.distribution_class.support
        expected = (
            "dependent" if isinstance(support, property) else _symbolic(support)[0]
        )
        assert record.support == expected, name
        has_rsample = getattr(meta.distribution_class, "has_rsample", False)
        if isinstance(has_rsample, bool):
            assert record.reparameterizable == has_rsample, name


def test_sample_elements_follow_discreteness() -> None:
    for record in FAMILIES.values():
        if record.name == "Bernoulli":
            assert record.element == BOOL
            assert record.finite_support == (False, True)
        elif record.discrete:
            assert record.element == INT, record.name
        else:
            assert record.element == REAL, record.name
        for source in record.event_sources:
            assert record.parameter(source).rank == record.event_rank


def test_lookup_rejects_unknown_families() -> None:
    try:
        family("Gaussian")
    except KeyError as error:
        assert "Gaussian" in str(error)
    else:  # pragma: no cover - the assertion is the test
        raise AssertionError("unknown family accepted")
    assert isinstance(family("Normal"), DistributionFamily)
