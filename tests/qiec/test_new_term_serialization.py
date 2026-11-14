"""Serialization of the call, resumption, and allocation terms.

Two properties matter here and they pull in opposite directions. A term
must survive a deterministic round trip, or a checked module cannot be
handed to a backend. And the decoder must refuse anything it does not
recognise, or the stable boundary stops being a boundary.

The rejection half is the one worth testing hardest. A decoder that
accepts an extra field, or a node type it has never heard of, will
happily read a payload written by a different version of the kernel and
produce a module that type-checks against the wrong contract.
"""

from __future__ import annotations

import copy
import json

import pytest

from quivers.qiec import (
    EMPTY_ROW,
    INT,
    ComputationId,
    ComputationType,
    EffectId,
    EffectRef,
    QiecModule,
)
from quivers.qiec.identifiers import EffectInstanceId, SourceOrigin
from quivers.qiec.module import NamedComputation
from quivers.qiec.serialization import SerializationError, dumps, loads
from quivers.qiec.terms import (
    Call,
    Computation,
    LiteralValue,
    NewInstance,
    Resume,
    Return,
)


_ORIGIN = SourceOrigin(
    module="example",
    structural_path=("path",),
    role="call",
    source_protocol="qvr-source/v0.19",
)


def _bodies() -> dict[str, Computation]:
    """One module body per new term.

    Returns
    -------
    dict[str, Computation]
        Label to the computation exercising that term.
    """
    return {
        "call": Call(
            callee=ComputationId.derive("example", "helper"),
            name="helper",
            static_arguments=(),
            arguments=(),
            result_type=INT,
            effects=EMPTY_ROW,
            origin=_ORIGIN,
        ),
        "resume": Resume(value=LiteralValue(value=1, type=INT), origin=_ORIGIN),
        "new_instance": NewInstance(
            instance=EffectInstanceId.derive("example", "cell"),
            effect=EffectRef(id=EffectId.derive("example", "State"), name="State"),
            body=Return(value=LiteralValue(value=0, type=INT)),
            origin=_ORIGIN,
        ),
    }


def _module(label: str, body: Computation) -> QiecModule:
    """Wrap a body in a module so it can be serialized.

    Parameters
    ----------
    label : str
        Name for the computation, also its derived identity.
    body : Computation
        The body under test.

    Returns
    -------
    QiecModule
        A module holding exactly that computation.
    """
    computation = NamedComputation(
        id=ComputationId.derive("example", label),
        name=label,
        telescope=(),
        parameters=(),
        body=body,
        type=ComputationType(effects=EMPTY_ROW, result=INT),
        origin=_ORIGIN,
    )
    return QiecModule(
        module="example",
        source_protocol="qvr-source/v0.19",
        computations=(computation,),
    )


@pytest.mark.parametrize("label", sorted(_bodies()))
def test_each_new_term_survives_a_round_trip(label: str) -> None:
    """Dump and load returns an equal module.

    A term the encoder can write but the decoder cannot read would make
    a checked module unusable by any out-of-process consumer.
    """
    module = _module(label, _bodies()[label])
    assert loads(dumps(module)) == module


@pytest.mark.parametrize("label", sorted(_bodies()))
def test_each_new_term_encodes_deterministically(label: str) -> None:
    """The same module encodes to the same bytes.

    Identities downstream are content-addressed, so an encoder that
    ordered a field differently between runs would make equal modules
    hash apart.
    """
    module = _module(label, _bodies()[label])
    assert dumps(module) == dumps(module)


def _encoded_call() -> dict:
    """A serialized module whose body is a call.

    Returns
    -------
    dict
        The decoded JSON, ready to mutate.
    """
    return json.loads(dumps(_module("call", _bodies()["call"])))


def _find(node: object, tag: str) -> dict | None:
    """The first node in a decoded payload carrying a `$type` tag.

    Parameters
    ----------
    node : object
        Decoded JSON to search.
    tag : str
        The `$type` value to look for.

    Returns
    -------
    dict or None
        The matching node, or None when the payload has none.
    """
    if isinstance(node, dict):
        if node.get("$type") == tag:
            return node
        for value in node.values():
            found = _find(value, tag)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find(value, tag)
            if found is not None:
                return found
    return None


def test_the_payload_actually_contains_the_term_under_test() -> None:
    """Guards the mutation tests below.

    If the call were encoded under some other tag, every rejection test
    would be mutating a node that is not there and passing for the wrong
    reason.
    """
    assert _find(_encoded_call(), "terms.Call") is not None


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("an extra field", lambda n: n["fields"].append({"name": "x", "value": 1})),
        ("a missing field", lambda n: n["fields"].pop(1)),
        ("an unknown node type", lambda n: n.__setitem__("$type", "terms.Bogus")),
        (
            "a malformed identifier",
            lambda n: n["fields"][0].__setitem__("value", {"$id": "not-an-id"}),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_the_decoder_rejects_a_damaged_call(label: str, mutate) -> None:
    """Each way of damaging an encoded call is refused.

    Accepting any of these would let a payload from a different kernel
    version decode into a module that then checks against the wrong
    contract, which is worse than failing to decode at all.
    """
    payload = _encoded_call()
    node = _find(payload, "terms.Call")
    assert node is not None
    mutate(node)
    with pytest.raises(SerializationError):
        loads(json.dumps(payload))


def test_an_undamaged_payload_still_loads() -> None:
    """The control for the rejection tests.

    Without it, a decoder that rejected everything would pass every test
    above.
    """
    payload = _encoded_call()
    assert loads(json.dumps(copy.deepcopy(payload))) == _module(
        "call", _bodies()["call"]
    )
