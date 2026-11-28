"""Derived factor scopes preserve sparse paths and original weight identities."""

from fractions import Fraction
from hashlib import sha256
from itertools import product
from math import exp, fsum, log

import pytest
import torch
from torch import nn

from quivers.core.relations import RelationLimitError
from quivers.dsl import Compiler, module_to_source, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import (
    ExecutionFailure,
    HandlerId,
    JointAlignmentMapConfig,
    ParameterId,
    PrunedStringBPConfig,
    PrunedStringGraphRequest,
    PrunedStringGraphRuntimeProvider,
    QiecModule,
    RegularStringEPConfig,
    RelationCascadeSpec,
    StringFactorGraphSpec,
    TreeStringGraphRequest,
    TreeStringGraphRuntimeProvider,
    dumps,
    loads,
)
from tests.dsl.test_authored_string_graphs import _compile
from tests.qiec.test_authored_string_inference_objectives import _run
from tests.qiec.test_pruned_string_graph_runtime import _with_pruned_query

_MODULE = "authored_derived_string_factors"
_WORDS = ((1,), (0,))
_SOURCE = """object Letters : FinSet 2
object Middle : FinSet 2
object Output : FinSet 2

string_graph model
    variable x : Letters
    factor visible(x)
    factor visible(x)

string_graph_score partition : LogWeight [operation=tree_partition, graph=model]

relation visible = project(joined, (0))
relation joined = compose(channel, emit)
relation direct = project(channel, (0))

relation channel [algebra=logprob, states=2]
    input Letters
    output Middle
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0, 0), weight=p]
    arc 0 -> 1 [labels=(0, 0), weight=p]
    arc 0 -> 1 [labels=(1, 1), weight=unit]
    final 1 [weight=unit]

relation emit [algebra=logprob, states=2]
    input Middle
    output Output
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0, 0), weight=unit]
    arc 0 -> 1 [labels=(1, 1), weight=unit]
    final 1 [weight=unit]

relation reference [algebra=logprob, states=2]
    input Letters
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=unit]
    arc 0 -> 1 [labels=(1), weight=unit]
    final 1 [weight=unit]
"""


def _compile_derived(source=_SOURCE):
    compiler = Compiler(parse(source), module_name=_MODULE)
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler, compiler.qiec_module


def _parameters():
    parameter = nn.Parameter(torch.tensor(-0.25, dtype=torch.float64))
    values = {
        ParameterId.derive(_MODULE, "relation_weights", "unit"): parameter.new_zeros(
            ()
        ),
        ParameterId.derive(_MODULE, "relation_weights", "p"): parameter,
    }
    return values, parameter


def _configurations(compiler):
    kwargs = dict(max_states=120, max_work=300)
    return (
        PrunedStringBPConfig(
            supports=(("x", _WORDS),), beam=2, max_sweeps=8, max_entries=4, **kwargs
        ),
        JointAlignmentMapConfig(
            max_iterations=8, max_ngram_order=2, max_features=4, **kwargs
        ),
        RegularStringEPConfig(
            bases=(("x", compiler.relation_specs["reference"]),),
            features=(("x", ((0,), (1,))),),
            penalty=0.0,
            max_sweeps=8,
            projection_tolerance=1e-10,
            **kwargs,
        ),
    )


def test_repeated_authored_derived_factor_keeps_paths_and_live_weight_owners():
    compiler, authored = _compile_derived()
    declared = compiler.string_graph_specs["model"]
    assert declared.factors == (("visible", ("x",)),) * 2
    graph = loads(dumps(declared))
    assert isinstance(graph, StringFactorGraphSpec)
    assert (
        graph
        == declared
        == _compile_derived(module_to_source(parse(_SOURCE)))[0].string_graph_specs[
            "model"
        ]
    )
    ((name, cascade),) = graph.relations
    assert name == "visible" and isinstance(cascade, RelationCascadeSpec)
    assert tuple(step.name for step in cascade.steps) == (
        "channel",
        "emit",
        "joined",
        "visible",
    )
    assert cascade.steps[0].primitive == compiler.relation_specs["channel"]

    parameters, learned = _parameters()
    bound = graph.bind(parameters, max_states=120, max_work=300)
    head, tail = bound._factors
    assert head.relation is tail.relation
    assert tuple(head.relation.module().parameters()) == (learned,)

    pruned, best_path, projected = _configurations(compiler)
    belief = PrunedStringGraphRequest(graph=graph, config=pruned, variable="x")
    module = loads(dumps(_with_pruned_query(authored, belief)))
    assert isinstance(module, QiecModule)
    tree = TreeStringGraphRequest(
        graph=graph, variable="x", max_states=120, max_work=300
    )
    exact_provider = TreeStringGraphRuntimeProvider.for_score(
        "partition", tree, parameters
    )
    pruned_provider = PrunedStringGraphRuntimeProvider(belief, parameters)
    for value in (Fraction(-1, 4), Fraction(1, 4)):
        with torch.no_grad():
            learned.fill_(float(value))
        # Each x=0 path has exactly one compatible middle and output
        # path. The two original channel occurrences therefore give four
        # distinct joint paths across the two factors.
        scores = {
            (0,): tuple(a + b for a, b in product((value, value), repeat=2)),
            (1,): (Fraction(0),),
        }
        mass = {word: fsum(map(exp, paths)) for word, paths in scores.items()}
        total = fsum(mass.values())
        posterior = mass[(0,)] / total
        evidence = _run(module, "partition", exact_provider)
        assert evidence.item() == pytest.approx(log(total), abs=1e-12, rel=0.0)
        assert torch.autograd.grad(evidence, learned)[0].item() == pytest.approx(
            2 * posterior
        )

        report = _run(module, "pruned_belief", pruned_provider)
        assert report[0] == "converged" and not report[4].item()
        assert report[5].exp().tolist() == pytest.approx([1 - posterior, posterior])
        assert torch.autograd.grad(report[5][1], learned)[0].item() == pytest.approx(
            2 * (1 - posterior)
        )

        best = best_path.solve(graph, parameters)
        optimum = max(scores, key=lambda word: max(scores[word]))
        assert best.status == "optimal" and best.score == best.upper_bound == max(
            scores[optimum]
        )
        assert best.assignment == (("x", optimum),)
        assert best.certificate is not None
        assert len(best.certificate.alignments) == 2
        result = projected.infer(graph, parameters)
        assert result.status == "converged"
        assert result.marginals["x"].log_prob((0,)).exp().item() == pytest.approx(
            posterior, abs=8e-8
        )

    left = cascade.steps[0]
    replacement = left.with_(
        primitive=left.primitive.with_(arcs=left.primitive.arcs[1:])
    )
    substitute = graph.with_(
        relations=((name, cascade.with_(steps=(replacement, *cascade.steps[1:]))),)
    )
    changed = tree.with_(graph=substitute)
    assert changed.result_type == tree.result_type
    wrong = TreeStringGraphRuntimeProvider.for_score("partition", changed, parameters)
    with pytest.raises(ExecutionFailure, match="authored graph"):
        _run(module, "partition", wrong)


def test_graph_factor_limits_reach_all_routes_and_nominal_tapes_are_checked():
    compiler, native = _compile_derived()
    graph = compiler.string_graph_specs["model"]
    parameters, _ = _parameters()
    pruned, best_path, projected = _configurations(compiler)
    tree = TreeStringGraphRequest(
        graph=graph, variable="x", max_states=120, max_work=300
    )
    with pytest.raises(ValueError, match="explicit construction limits"):
        graph.bind(parameters)
    for limits, reason in (
        ({"max_states": 2}, "composition exceeds max_states"),
        ({"max_work": 3}, "cascade step count exceeds max_work"),
    ):
        with pytest.raises(RelationLimitError, match=reason):
            tree.with_(**limits).bind(parameters)
        with pytest.raises(RelationLimitError, match=reason):
            best_path.with_(**limits).solve(graph, parameters)
        with pytest.raises(RelationLimitError, match=reason):
            projected.with_(**limits).infer(graph, parameters)
        limited = PrunedStringGraphRequest(
            graph=graph, config=pruned.with_(**limits), variable="x"
        )
        safe = _with_pruned_query(native, limited)
        report = _run(
            safe, "pruned_belief", PrunedStringGraphRuntimeProvider(limited, parameters)
        )
        assert report[0] == "resource_limit" and reason in report[1]

    with pytest.raises(QiecDiagnosticError, match="mistyped variable"):
        _compile_derived(_SOURCE.replace("variable x : Letters", "variable x : Output"))
    direct = _compile_derived(_SOURCE.replace("factor visible(x)", "factor direct(x)"))[
        0
    ]
    ((name, normalized),) = direct.string_graph_specs["model"].relations
    assert name == "direct" and isinstance(normalized, RelationCascadeSpec)
    assert tuple(step.name for step in normalized.steps) == ("channel", "direct")
    assert direct.string_graph_specs["model"].bind(
        parameters, max_states=120, max_work=300
    )


def test_primitive_graph_and_exact_request_keep_legacy_wire_and_owner_digest():
    compiler, module = _compile()
    graph = compiler.string_graph_specs["chain"]
    request = TreeStringGraphRequest(
        graph=graph, variable="left", max_states=60, max_work=180
    )
    graph_digest = "3fe4c2fd6d6fc6d2d5cec75c4a440b28b62045568fd696edd76b63d0aa1c2319"
    request_digest = "aad1829aeae8b4a5b49ed34e282d2255dde8015a68e1c3e2d1d4f58aaadd8765"
    for value, expected in ((graph, graph_digest), (request, request_digest)):
        wire = dumps(value)
        assert sha256(wire.encode()).hexdigest() == expected
        assert dumps(loads(wire)) == wire
    handler = next(
        item
        for item in module.handlers
        if item.name == "__string_graph_score_partition"
    )
    assert handler.id == HandlerId.derive(
        module.module,
        "string_graph",
        handler.name,
        "partition",
        "authored_graph",
        graph_digest,
    )
