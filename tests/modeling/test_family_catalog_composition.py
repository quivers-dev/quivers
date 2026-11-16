"""Joint native catalogs retain contracts and executable/support identities."""

import pytest
import torch
from torch.distributions import constraints

from quivers import FamilyCatalog
from quivers.qiec.families import FAMILIES


def _normal(parameters):
    return torch.distributions.Normal(parameters["loc"], parameters["scale"])


def _record(name):
    return FAMILIES["Normal"].with_(name=name, targets={})


def test_catalog_composition_shares_native_implementations_without_global_mutation():
    first = FamilyCatalog().with_family(_record("FirstUserNormal"), _normal)
    second = FamilyCatalog().with_family(_record("SecondUserNormal"), _normal)
    merged = first.with_catalog(second)
    assert first.with_catalog(first) is first
    assert not first.is_custom("SecondUserNormal")
    assert not second.is_custom("FirstUserNormal")
    assert not FamilyCatalog().is_custom("FirstUserNormal")
    loc = torch.tensor(0.2, requires_grad=True)
    distribution = merged.build(
        "SecondUserNormal", {"loc": loc, "scale": torch.tensor(0.7)}
    )
    torch.testing.assert_close(
        distribution.log_prob(torch.tensor(0.4)),
        _normal({"loc": loc, "scale": torch.tensor(0.7)}).log_prob(torch.tensor(0.4)),
    )
    assert torch.autograd.grad(distribution.log_prob(torch.tensor(0.4)), loc)[
        0
    ] == pytest.approx(0.2 / 0.49)
    assert merged.support("SecondUserNormal") is second.support("SecondUserNormal")


def test_same_named_semantics_cannot_replace_another_factory_or_support_identity():
    record = _record("UserNormal")
    first = FamilyCatalog().with_family(record, _normal)
    assert first.with_catalog(FamilyCatalog().with_family(record, _normal)) is first
    competing = FamilyCatalog().with_family(
        record, lambda parameters: _normal(parameters)
    )
    with pytest.raises(ValueError, match="conflicting catalog definitions"):
        first.with_catalog(competing)
    other_support = FamilyCatalog().with_family(
        record, _normal, support=constraints.interval(-torch.inf, torch.inf)
    )
    with pytest.raises(ValueError, match="conflicting catalog definitions"):
        first.with_catalog(other_support)
    conflicting_contract = FamilyCatalog().with_family(
        record.with_(atom_locations=(0.0,)), _normal
    )
    with pytest.raises(ValueError, match="conflicting catalog definitions"):
        first.with_catalog(conflicting_contract)
    with pytest.raises(TypeError, match="FamilyCatalog"):
        first.with_catalog({})
