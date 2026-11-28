"""Portable signed-delay tape equality with exact original-path oracles."""

from fractions import Fraction
from hashlib import sha256
from itertools import product

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import RelationLimitError
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    FixedRelationRuntimeProvider,
    FixedTransformedRelationRequest,
    QiecModule,
    RelationCascadeSpec,
    RelationCascadeStep,
    RuntimeConfiguration,
    UnaryRelationTransformSpec,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "checked_relation_tape_equality"
_NATIVE = """
instance observed : Compute[Unit, Real]
handler observed_weight for Compute[Unit, Real] : Real -> Real [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define score : Real !{} =
    handle observed with observed_weight in
        let weight <- perform observed.apply(unit)
        return weight
"""
_SOURCE = """object Letters : FinSet 2
object WrongLetters : FinSet 2
object Marker : FinSet 2

relation original [algebra=markov, states=5]
    input Letters
    output Letters
    output Marker
    initial 0 [weight=unit]
    arc 0 -> 0 [labels=(0, eps, eps), weight=p]
    arc 0 -> 0 [labels=(0, eps, eps), weight=p]
    arc 0 -> 0 [labels=(1, eps, eps), weight=q]
    arc 0 -> 1 [labels=(eps, 0, 1), weight=unit]
    arc 1 -> 2 [labels=(eps, 0, 1), weight=unit]
    arc 2 -> 3 [labels=(eps, 0, 1), weight=unit]
    arc 3 -> 4 [labels=(eps, 0, 1), weight=unit]
    final 4 [weight=unit]
"""


def _fixture(source=_SOURCE):
    compiler = Compiler(parse(source), module_name=_MODULE)
    primitive = compiler.relation_specs["original"]
    names = primitive.topology.weight_names
    values = {"unit": 1.0, "p": 0.125, "q": 0.25}
    parameters = {
        owner: torch.nn.Parameter(torch.tensor(values[name], dtype=torch.float64))
        for name, owner in zip(names, primitive.weight_parameters, strict=True)
    }
    ids = dict(zip(names, primitive.weight_parameters, strict=True))
    return primitive, parameters, ids


def _equality(primitive):
    return UnaryRelationTransformSpec(
        source=primitive, operation="auto_intersect", positions=(0, 1)
    )


def _cascade(primitive):
    return RelationCascadeSpec(
        steps=(
            RelationCascadeStep(name="original", primitive=primitive),
            RelationCascadeStep(
                name="equal",
                operation="auto_intersect",
                operands=(0,),
                positions=(0, 1),
            ),
            RelationCascadeStep(
                name="observed", operation="project", operands=(1,), positions=(2,)
            ),
        )
    )


def _legacy_requests(primitive):
    transform = UnaryRelationTransformSpec(
        source=primitive, operation="project", positions=(2,)
    )
    step = RelationCascadeStep(name="original", primitive=primitive)
    projection = RelationCascadeStep(
        name="projected", operation="project", operands=(0,), positions=(2,)
    )
    cascade = RelationCascadeSpec(steps=(step, projection))
    request = FixedTransformedRelationRequest(
        relation=transform, words=((1,) * 4,), max_states=400, max_work=4000
    )
    return transform, cascade, request


def _restricted_original_powers(p, q):
    total, slope = Fraction(0), Fraction(0)
    # Each productive original walk takes a finite sequence of loop
    # occurrences followed by the four forced labelled exits. The second
    # tape is fixed to four zeroes. Consequently its equality class
    # needs exactly four loop occurrences, all from the first two slots.
    outputs = (0, 0, 1)
    weights = (p, p, q)
    for count in range(5):
        for choices in product(range(len(outputs)), repeat=count):
            if tuple(outputs[arc] for arc in choices) != (0,) * 4:
                continue
            path = Fraction(1)
            for arc in choices:
                path *= weights[arc]
            total += path
            slope += 4 * p**3
    return total, slope


def test_existing_unary_cascade_and_fixed_query_wire_bytes_are_unchanged():
    primitive, _, _ = _fixture()
    old_hashes = (
        "d419ebb51c6f4431260ebc4be43efc69473eb5daca25c1353ea2f97c4cc96ddc",
        "34dc2085e9a3573fa3ef364c3cd019cb761a64194965b101eb62c6405f364c94",
        "fb46d266a5a8666283cda5b4cf7d33cddb1ad1cc69a1fb3fcb7eac61a40f4c9c",
    )
    for request, digest in zip(_legacy_requests(primitive), old_hashes, strict=True):
        encoded = dumps(request)
        assert sha256(encoded.encode()).hexdigest() == digest
        assert dumps(loads(encoded)) == encoded


def test_portable_tape_equality_can_erase_synchronized_tapes_and_keep_live_weights():
    primitive, parameters, ids = _fixture()
    request = FixedTransformedRelationRequest(
        relation=_cascade(primitive), words=((1,) * 4,), max_states=400, max_work=4000
    )
    encoded = dumps(request)
    restored = loads(encoded)
    assert isinstance(restored, FixedTransformedRelationRequest)
    assert dumps(restored) == encoded
    plan = restored.plan(parameters)
    module = Compiler(parse(_SOURCE + _NATIVE), module_name=_MODULE).qiec_module
    module = loads(dumps(module))
    assert isinstance(module, QiecModule)
    provider = FixedRelationRuntimeProvider(
        restored, parameters, handler_name="observed_weight"
    )
    runtime = RuntimeConfiguration(providers=(provider,))
    p, q = parameters[ids["p"]], parameters[ids["q"]]
    for value in (Fraction(1, 8), Fraction(1, 4)):
        with torch.no_grad():
            p.fill_(float(value))
        expected, slope = _restricted_original_powers(value, Fraction(1, 4))
        assert expected == (2 * value) ** 4
        assert slope == 8 * (2 * value) ** 3
        for actual in (
            plan.weight(),
            run_named(module, "score", runtime=runtime).value,
        ):
            dp, dq = torch.autograd.grad(actual, (p, q))
            assert (actual.item(), dp.item(), dq.item()) == (
                float(expected),
                float(slope),
                0.0,
            )
    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "score" and "observed_weight" in item.reason
        for item in target.obstacles
    )


def test_standalone_equality_preserves_type_and_checks_word_identity():
    primitive, parameters, _ = _fixture()
    recipe = _equality(primitive)
    restored = loads(dumps(recipe))
    assert isinstance(restored, UnaryRelationTransformSpec) and restored == recipe
    equal = restored.bind(parameters, max_states=400, max_work=4000)
    original = primitive.bind(parameters)
    assert equal.topology.tapes == original.topology.tapes
    assert equal.topology.input_tapes == original.topology.input_tapes
    for prefix in ((0,) * 4, (0, 0, 0), (0, 0, 0, 1)):
        words = prefix, (0,) * 4, (1,) * 4
        assert original.weight(words).item() > 0
        expected = Fraction(1, 4) ** 4 if prefix == (0,) * 4 else 0
        assert equal.weight(words).item() == expected


def test_nominal_arity_signed_delay_and_sparse_limits_refuse():
    primitive, parameters, _ = _fixture()
    recipe = _equality(primitive)
    for positions in ((), (0,), (0, 0), (0, 1, 2)):
        with pytest.raises(dx.ValidationError, match="two distinct tape positions"):
            recipe.with_(positions=positions)
    with pytest.raises(dx.ValidationError, match="same nominal tape"):
        recipe.with_(positions=(0, 2))
    opposite, owners, ids = _fixture(
        _SOURCE.replace(
            "    final 4",
            "    arc 4 -> 4 [labels=(eps, 0, eps), weight=q]\n    final 4",
        )
    )
    with torch.no_grad():
        owners[ids["q"]].zero_()
    with pytest.raises(ValueError, match="positive and negative cycle delays"):
        _equality(opposite).bind(owners, max_states=400, max_work=4000)
    with pytest.raises(
        RelationLimitError, match="auto-intersection exceeds max_states"
    ):
        recipe.bind(parameters, max_states=1, max_work=4000)
    with pytest.raises(RelationLimitError, match="cycle analysis exceeds max_work"):
        recipe.bind(parameters, max_states=400, max_work=0)
