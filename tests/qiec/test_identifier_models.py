"""Stable identifier models retain their namespace and wire contracts."""

from __future__ import annotations

from hashlib import sha256
import json

import didactic.api as dx
import pytest

from quivers.qiec.identifiers import EffectId, OperationId, QIEC_ABI


def test_identifier_digest_matches_the_canonical_wire_payload() -> None:
    """Derivation agrees with an independent JSON and SHA-256 calculation."""
    payload = [QIEC_ABI, "effect", "example", {"b": 2, "a": [True, "x"]}]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    expected = sha256(encoded).hexdigest()
    actual = EffectId.derive("example", {"a": [True, "x"], "b": 2})
    assert isinstance(actual, dx.Model)
    assert actual.digest == expected
    assert actual.to_data() == f"qiec:effect:{expected}"
    assert EffectId.parse(actual.to_data()) == actual
    assert OperationId.derive("example", {"a": [True, "x"], "b": 2}).digest != expected


def test_identifier_equality_hash_and_order_preserve_namespaces() -> None:
    """Equal digests compare only within the same concrete identifier class."""
    low = EffectId(digest="0" * 64)
    high = EffectId(digest="f" * 64)
    equal = EffectId(digest="0" * 64)
    other_namespace = OperationId(digest="0" * 64)
    assert low == equal
    assert hash(low) == hash(equal)
    assert low != other_namespace
    assert sorted((high, low, equal)) == [low, equal, high]
    assert low <= equal < high
    assert high > low and high >= high
    with pytest.raises(TypeError):
        low < other_namespace
    assert {low: "value"}[equal] == "value"


@pytest.mark.parametrize("digest", ["", "0" * 63, "0" * 65, "A" * 64, "g" * 64])
def test_identifier_construction_and_updates_validate_the_digest(digest: str) -> None:
    """Neither construction nor immutable updates admit malformed digests.

    Parameters
    ----------
    digest : str
        A malformed digest.
    """
    with pytest.raises(dx.ValidationError, match="invalid effect digest"):
        EffectId(digest=digest)
    original = EffectId(digest="0" * 64)
    with pytest.raises(dx.ValidationError, match="invalid effect digest"):
        original.with_(digest=digest)
    with pytest.raises(ValueError, match="invalid effect digest"):
        EffectId.parse(f"qiec:effect:{digest}")
    assert original.digest == "0" * 64
    assert original.with_(digest="f" * 64) == EffectId(digest="f" * 64)
