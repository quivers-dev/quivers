"""Authored graph topology, ordered wire identity, and checked tree scores."""

from fractions import Fraction
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
    HandlerId,
    ParameterId,
    QiecModule,
    RuntimeConfiguration,
    StringFactorGraphSpec,
    TreeStringGraphRequest,
    TreeStringGraphRuntimeProvider,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "authored_string_graph"
_SOURCE = """object Letters : FinSet 2
object Other : FinSet 2

#! The observed right word is fixed by the topology declaration.
string_graph chain
    variable left : Letters
    variable middle : Letters
    variable right : Letters [observed=(1)]
    variable empty : Letters [observed=()]
    factor edit(left, middle)
    factor edit(middle, right)

string_graph_score partition : Real [operation=tree_partition, graph=chain]
string_graph_score normalized : LogWeight [operation=tree_log_prob, graph=chain]
string_graph_score generic : Real [operation=tree_partition]

relation edit [algebra=markov, states=2]
    input Letters
    output Letters
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0, 0), weight=same]
    arc 0 -> 1 [labels=(0, 1), weight=different]
    arc 0 -> 1 [labels=(1, 0), weight=different]
    arc 0 -> 1 [labels=(1, 1), weight=same]
    final 1 [weight=one]
"""


def _compile(source: str = _SOURCE) -> tuple[Compiler, QiecModule]:
    compiler = Compiler(parse(source), module_name=_MODULE, file_path="graph.qvr")
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    return compiler, module


def _evaluate(module, name, request, parameters):
    provider = TreeStringGraphRuntimeProvider.for_score(name, request, parameters)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    result = run_named(module, name, runtime=runtime)
    assert isinstance(result.value, torch.Tensor)
    assert result.result_type == request.result_type
    return result.value


def test_authored_graph_round_trips_and_reuses_original_relation_and_parameters():
    compiler, module = _compile()
    graph = compiler.string_graph_specs["chain"]
    relation = compiler.relation_specs["edit"]
    assert graph.variables == (
        ("left", "Letters", 2, None),
        ("middle", "Letters", 2, None),
        ("right", "Letters", 2, (1,)),
        ("empty", "Letters", 2, ()),
    )
    assert graph.relations == (("edit", relation),)
    assert graph.factors == (
        ("edit", ("left", "middle")),
        ("edit", ("middle", "right")),
    )
    emitted = module_to_source(parse(_SOURCE))
    assert "#! The observed right word is fixed by the topology declaration." in emitted
    recompiled, remodule = _compile(emitted)
    assert recompiled.string_graph_specs == compiler.string_graph_specs
    assert tuple(item.id for item in remodule.handlers) == tuple(
        item.id for item in module.handlers
    )
    assert any(
        binding.name == "chain" and binding.kind == "string-graph"
        for binding in qiec_bindings(parse(_SOURCE))
    )
    decoded = loads(dumps(graph))
    assert isinstance(decoded, StringFactorGraphSpec) and decoded == graph
    wire = loads(dumps(module))
    assert isinstance(wire, QiecModule)
    same = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    different = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    parameters = {
        ParameterId.derive(_MODULE, "relation_weights", "one"): same.new_tensor(1),
        ParameterId.derive(_MODULE, "relation_weights", "same"): same,
        ParameterId.derive(_MODULE, "relation_weights", "different"): different,
    }
    bound = decoded.bind(parameters)
    assert bound._factors[0].relation is bound._factors[1].relation
    request = TreeStringGraphRequest(
        graph=decoded,
        variable="left",
        max_states=60,
        max_work=180,
    )
    wire_request = loads(dumps(request))
    assert isinstance(wire_request, TreeStringGraphRequest)
    partition = _evaluate(wire, "partition", wire_request, parameters)
    expected_partition = Fraction(3, 4) ** 2
    assert partition.item() == float(expected_partition)
    derivatives = torch.autograd.grad(partition, (same, different))
    assert tuple(value.item() for value in derivatives) == (1.5, 1.5)
    posterior = _evaluate(wire, "normalized", wire_request.with_(word=(0,)), parameters)
    assert posterior.item() == pytest.approx(
        log(float(Fraction(1, 4) / expected_partition))
    )
    derivatives = torch.autograd.grad(posterior, (same, different))
    assert [value.item() for value in derivatives] == pytest.approx([-2 / 3, 4 / 3])

    substitute = decoded.with_(factors=tuple(reversed(decoded.factors)))
    replacement = wire_request.with_(graph=substitute)
    assert replacement.result_type == wire_request.result_type == REAL
    with pytest.raises(ExecutionFailure, match="authored graph"):
        _evaluate(wire, "partition", replacement, parameters)
    # The legacy declaration without a graph still has its original identity.
    handler_name = "__string_graph_score_generic"
    unbound = next(item for item in wire.handlers if item.name == handler_name)
    assert unbound.id == HandlerId.derive(
        _MODULE, "string_graph", handler_name, "partition"
    )
    assert _evaluate(wire, "generic", replacement, parameters).item() == float(
        expected_partition
    )

    discharged = discharge_module(wire)
    for target in ("stan", "bugs", "jags"):
        obstacle = next(
            x
            for x in project_for_target(wire, discharged, target).obstacles
            if x.computation == "partition"
        )
        assert "__string_graph_score_partition" in obstacle.reason


@pytest.mark.parametrize(
    "source,replacement,diagnostic",
    (
        (
            "factor edit(left, middle)",
            "factor missing(left, middle)",
            "unknown authored factor relation",
        ),
        ("variable right : Letters", "variable right : Other", "mistyped variable"),
        ("observed=(1)", "observed=(2)", "invalid observed symbol"),
        (
            "factor edit(left, middle)",
            "factor edit(left, left)",
            "one distinct variable",
        ),
        ("graph=chain]", "graph=absent]", "unknown authored string graph"),
        ("partition : Real", "partition : LogWeight", "partition requires Real"),
    ),
)
def test_authored_graph_rejects_unknown_and_mistyped_occurrences(
    source, replacement, diagnostic
):
    with pytest.raises(QiecDiagnosticError, match=diagnostic):
        _compile(_SOURCE.replace(source, replacement, 1))
