"""Source-to-wire and topology identity checks for authored sparse relations."""

from math import log

import pytest
import torch
from torch import nn

from quivers.dsl import Compiler, module_to_source, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_tooling import qiec_bindings
from quivers.qiec import (
    REAL,
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedRelationCostRequest,
    FixedRelationRequest,
    FixedRelationRuntimeProvider,
    HandlerId,
    ParameterId,
    QiecModule,
    RationalRelationSpec,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "authored_relations_test"
_SOURCE = """object Letters : FinSet 2
object Output : Letters

relation channel [algebra=markov, states=3]
    input Letters
    output Output
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0, eps), weight=shared]
    arc 0 -> 1 [labels=(0, eps), weight=shared]
    arc 1 -> 2 [labels=(eps, 0), weight=tail]
    arc 0 -> 2 [labels=(0, 0), weight=direct]
    final 2 [weight=one]

relation_score pair : Real [operation=pair_weight, relation=channel]
relation_score entropy : Real [operation=path_entropy, relation=channel]
relation_score risk : Real [operation=expected_path_cost, relation=channel]
relation_score conditional : LogWeight [operation=conditional_log_prob, relation=channel]
relation_score unbound : Real [operation=pair_weight]
"""


def _compile(source: str = _SOURCE) -> tuple[Compiler, QiecModule]:
    compiler = Compiler(parse(source), module_name=_MODULE, file_path="authored.qvr")
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    return compiler, module


def _values(spec: RationalRelationSpec) -> dict[ParameterId, torch.Tensor]:
    values = {
        "one": torch.tensor(1.0, dtype=torch.float64),
        "shared": nn.Parameter(torch.tensor(0.125, dtype=torch.float64)),
        "tail": nn.Parameter(torch.tensor(0.5, dtype=torch.float64)),
        "direct": nn.Parameter(torch.tensor(0.125, dtype=torch.float64)),
    }
    return {
        id_: values[name]
        for name, id_ in zip(spec.weight_names, spec.weight_parameters, strict=True)
    }


def _execute(
    name: str,
    request: FixedRelationRequest | FixedRelationCostRequest,
    parameters: dict[ParameterId, torch.Tensor],
    module: QiecModule,
) -> torch.Tensor:
    attachment = FixedRelationRuntimeProvider.for_score(name, request, parameters)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), attachment)
    )
    result = run_named(module, name, runtime=runtime)
    assert result.result_type == request.result_type
    assert isinstance(result.value, torch.Tensor)
    return result.value


def test_source_to_wire_preserves_tapes_occurrences_and_scores():
    compiler, module = _compile()
    spec = compiler.relation_specs["channel"]
    assert spec.tapes == (("Letters", 2), ("Letters", 2))
    assert spec.input_tapes == 1 and spec.n_states == 3
    assert spec.arcs[:2] == ((0, 1, (0, None), 1),) * 2
    assert spec.weight_names == ("one", "shared", "tail", "direct")
    original_ids = tuple(
        ParameterId.derive(_MODULE, "relation_weights", name)
        for name in spec.weight_names
    )
    assert spec.weight_parameters == original_ids
    assert loads(dumps(spec)) == spec
    emitted_compiler, _ = _compile(module_to_source(parse(_SOURCE)))
    assert emitted_compiler.relation_specs["channel"] == spec
    assert any(
        binding.name == "channel"
        and binding.kind == "relation"
        and binding.semantic_kind == "type"
        for binding in qiec_bindings(parse(_SOURCE))
    )
    wire = loads(dumps(module))
    assert isinstance(wire, QiecModule)
    parameters = _values(spec)
    pair = FixedRelationRequest(
        relation=spec,
        operation="pair_weight",
        words=((0,), (0,)),
        max_states=24,
        max_work=40,
    )
    mass = _execute("pair", pair, parameters, wire)
    assert mass.item() == pytest.approx(0.25)
    observed = torch.autograd.grad(
        mass, tuple(parameters[id_] for id_ in original_ids[1:])
    )
    assert [grad.item() for grad in observed] == pytest.approx([1.0, 0.25, 1.0])
    entropy = _execute(
        "entropy", pair.with_(operation="path_entropy"), parameters, wire
    )
    assert entropy.item() == pytest.approx(1.5 * log(2))
    risk = FixedRelationCostRequest(
        relation=spec,
        words=pair.words,
        max_states=24,
        max_work=40,
        arc_costs=(-3.0, -1.0, -2.0, 0.0),
    )
    assert _execute("risk", risk, parameters, wire).item() == pytest.approx(-2.0)
    assert _execute(
        "conditional", pair.with_(operation="conditional_log_prob"), parameters, wire
    ).item() == pytest.approx(0.0)


def test_same_typed_substitute_topology_is_rejected_and_unbound_identity_is_stable():
    compiler, module = _compile()
    source = compiler.relation_specs["channel"]
    substitute = source.with_(arcs=source.arcs[1:])
    parameters = _values(source)
    request = FixedRelationRequest(
        relation=substitute,
        operation="pair_weight",
        words=((0,), (0,)),
        max_states=24,
        max_work=40,
    )
    assert request.result_type == REAL
    with pytest.raises(ExecutionFailure, match="authored topology"):
        _execute("pair", request, parameters, module)
    declaration = next(
        item for item in module.handlers if item.name == "__relation_score_unbound"
    )
    assert declaration.id == HandlerId.derive(
        _MODULE,
        "relation_scores",
        "unbound",
        "pair_weight",
        "handler",
    )
    assert _execute("unbound", request, parameters, module).item() == pytest.approx(
        3 / 16
    )
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        projected = project_for_target(module, discharged, target)
        obstacle = next(
            item for item in projected.obstacles if item.computation == "pair"
        )
        assert "__relation_score_pair" in obstacle.reason


@pytest.mark.parametrize(
    "old, new, diagnostic",
    (
        ("output Output", "output Missing", "requires a declared finite alphabet"),
        ("labels=(0, eps)", "labels=(2, eps)", "invalid symbol"),
        ("arc 1 -> 2", "arc 1 -> 3", "invalid state"),
        (
            "    input Letters\n    output Output",
            "    output Output\n    input Letters",
            "input relation tapes must precede output",
        ),
        ("relation=channel]", "relation=Missing]", "unknown authored relation"),
        ("entropy : Real", "entropy : LogWeight", "path statistics require a Real"),
        ("pair : Real", "pair : LogWeight", "pair weight requires Real"),
    ),
)
def test_authored_tape_and_carrier_mismatches_are_checked(old, new, diagnostic):
    with pytest.raises(QiecDiagnosticError, match=diagnostic):
        _compile(_SOURCE.replace(old, new, 1))


def test_authored_relations_share_named_scalar_ids_without_sharing_occurrences():
    second = """
relation alternate [algebra=markov, states=2]
    output Letters
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(1), weight=shared]
    final 1 [weight=one]
"""
    compiler, _ = _compile(_SOURCE + second)
    relation_a, relation_b = (
        compiler.relation_specs[name] for name in ("channel", "alternate")
    )
    assert relation_b.tapes == (("Letters", 2),) and relation_b.input_tapes == 0
    ids = dict(zip(relation_a.weight_names, relation_a.weight_parameters, strict=True))
    assert relation_b.weight_parameters == (ids["one"], ids["shared"])
    assert relation_a.topology.arcs[0].labels == (0, None)
    assert relation_b.topology.arcs[0].labels == (1,)
