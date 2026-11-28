"""Compare separate checked objectives on one authored factor graph."""

from fractions import Fraction
from itertools import product
from math import exp, fsum, log

import pytest
import torch
from torch import nn

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    CoreRuntimeProvider,
    JointAlignmentMapConfig,
    ParameterId,
    PrunedStringBPConfig,
    PrunedStringGraphRequest,
    PrunedStringGraphRuntimeProvider,
    QiecModule,
    RegularStringEPConfig,
    RuntimeConfiguration,
    TreeStringGraphRequest,
    TreeStringGraphRuntimeProvider,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_pruned_string_graph_runtime import _with_pruned_query

_MODULE = "authored_inference_objectives"
_WORDS = ((1,), (0,))
_SOURCE = """object Letter : FinSet 2

string_graph choice
    variable x : Letter
    factor left(x)
    factor right(x)

string_graph_score partition : LogWeight [operation=tree_partition, graph=choice]
string_graph_score posterior : LogWeight [operation=tree_log_prob, graph=choice]

relation left [algebra=logprob, states=3]
    output Letter
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(eps), weight=unit]
    arc 1 -> 2 [labels=(0), weight=left_zero]
    arc 1 -> 2 [labels=(0), weight=left_duplicate]
    arc 0 -> 2 [labels=(1), weight=left_one]
    final 2 [weight=unit]

relation right [algebra=logprob, states=2]
    output Letter
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=right_zero]
    arc 0 -> 1 [labels=(1), weight=unit]
    final 1 [weight=unit]

relation reference [algebra=logprob, states=2]
    output Letter
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=unit]
    arc 0 -> 1 [labels=(1), weight=unit]
    final 1 [weight=unit]
"""


def _owner(name):
    return ParameterId.derive(_MODULE, "relation_weights", name)


def _fixed_paths(spec, word, values):
    """Enumerate the original arcs without using a relation plan."""
    finals = dict(spec.final)
    slots = tuple(values[owner] for owner in spec.weight_parameters)

    def visit(state, offset, arcs, score):
        if state in finals and offset == len(word):
            yield arcs, score + slots[finals[state]]
        for occurrence, (source, target, labels, slot) in enumerate(spec.arcs):
            if source != state:
                continue
            (symbol,) = labels
            if symbol is None or (offset < len(word) and word[offset] == symbol):
                yield from visit(
                    target,
                    offset + (symbol is not None),
                    (*arcs, occurrence),
                    score + slots[slot],
                )

    for state, slot in spec.initial:
        yield from visit(state, 0, (), slots[slot])


def _oracle(graph, values):
    relations = dict(graph.relations)
    local = tuple(
        {word: tuple(_fixed_paths(relations[name], word, values)) for word in _WORDS}
        for name, _ in graph.factors
    )
    scores = {}
    for word in _WORDS:
        pairs = product(*(factor[word] for factor in local))
        scores[word] = {
            tuple(arcs for arcs, _ in pair): sum(score for _, score in pair)
            for pair in pairs
        }
    masses = {
        word: fsum(exp(score) for score in scores[word].values()) for word in _WORDS
    }
    upper = sum(
        max(score for paths in factor.values() for _, score in paths)
        for factor in local
    )
    return local, scores, masses, upper


def _run(module, name, provider):
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    return run_named(module, name, runtime=runtime).value


def test_same_authored_graph_separates_path_map_tree_pruning_and_regular_ep():
    compiler = Compiler(parse(_SOURCE), module_name=_MODULE)
    compiler.compile()
    authored = compiler.qiec_module
    graph = loads(dumps(compiler.string_graph_specs["choice"]))
    pruned = PrunedStringGraphRequest(
        graph=graph,
        variable="x",
        config=PrunedStringBPConfig(
            supports=(("x", _WORDS),),
            beam=2,
            max_sweeps=8,
            max_entries=4,
            max_states=80,
            max_work=100,
        ),
    )
    native = loads(dumps(_with_pruned_query(authored, pruned)))
    assert isinstance(native, QiecModule)
    pruned = loads(dumps(pruned))
    map_config = loads(
        dumps(
            JointAlignmentMapConfig(
                max_iterations=32,
                max_ngram_order=2,
                max_features=8,
                max_states=70,
                max_work=100,
                step_ratio=(1, 4),
            )
        )
    )
    ep_config = loads(
        dumps(
            RegularStringEPConfig(
                bases=(("x", compiler.relation_specs["reference"]),),
                features=(("x", ((0,), (1,))),),
                penalty=0.0,
                max_sweeps=8,
                max_states=160,
                max_work=400,
                projection_tolerance=1e-10,
            )
        )
    )

    values = {
        _owner(name): value
        for name, value in (
            ("unit", Fraction(0)),
            ("left_zero", Fraction(-1, 8)),
            ("left_duplicate", Fraction(-1, 4)),
            ("left_one", Fraction(-5, 8)),
            ("right_zero", Fraction(-3, 4)),
        )
    }
    owners = {
        name: nn.Parameter(torch.tensor(float(value), dtype=torch.float64))
        for name, value in values.items()
    }
    changed = _owner("right_zero")
    tree = TreeStringGraphRequest(
        graph=graph, variable="x", max_states=80, max_work=100
    )
    partition = TreeStringGraphRuntimeProvider.for_score("partition", tree, owners)
    posterior = TreeStringGraphRuntimeProvider.for_score(
        "posterior", tree.with_(word=(0,)), owners
    )
    beam = PrunedStringGraphRuntimeProvider(pruned, owners)
    narrow = pruned.with_(config=pruned.config.with_(beam=1))
    assert narrow.handler_name == pruned.handler_name
    narrow_beam = PrunedStringGraphRuntimeProvider(narrow, owners)
    for revision, right_score in enumerate((Fraction(-3, 4), Fraction(1, 8))):
        values[changed] = right_score
        with torch.no_grad():
            owners[changed].fill_(float(right_score))
        local, scores, masses, uncoordinated_bound = _oracle(graph, values)
        z = fsum(masses.values())
        exact = [masses[word] / z for word in _WORDS]
        assert max(masses, key=masses.get) == (0,)

        total = _run(native, "partition", partition)
        marginal = _run(native, "posterior", posterior)
        assert total.item() == pytest.approx(log(z), abs=1e-12, rel=0.0)
        assert marginal.exp().item() == pytest.approx(exact[1], abs=1e-12, rel=0.0)
        (derivative,) = torch.autograd.grad(total, owners[changed])
        assert derivative.item() == pytest.approx(exact[1], abs=1e-12, rel=0.0)

        status, reason, sweeps, residual, cut, belief, present, raw = _run(
            native, "pruned_belief", beam
        )
        assert status == "converged" and not reason and sweeps.item() > 0
        assert residual.item() < 1e-9 and present.item() and not cut.item()
        assert belief.exp().tolist() == pytest.approx(exact, abs=1e-12, rel=0.0)
        for factor, row in zip(local, raw.exp(), strict=True):
            mass = fsum(exp(score) for paths in factor.values() for _, score in paths)
            assert row.tolist() == pytest.approx([mass, mass, 0.0], abs=1e-12, rel=0.0)
        (derivative,) = torch.autograd.grad(belief[1], owners[changed])
        assert derivative.item() == pytest.approx(1 - exact[1], abs=1e-12, rel=0.0)

        sparse_beam = _run(native, "pruned_belief", narrow_beam)
        assert sparse_beam[4].item() and sparse_beam[6].item()
        if revision == 0:
            assert sparse_beam[0] == "empty_support"
            assert torch.isneginf(sparse_beam[5]).all()
        else:
            assert sparse_beam[0] == "converged"
            assert sparse_beam[5].exp().tolist() == [0.0, 1.0]

        best = map_config.solve(graph, owners)
        word = (1,) if revision == 0 else (0,)
        assert best.status == "optimal" and best.certificate is not None
        assert best.assignment == (("x", word),)
        optimum = max(score for rows in scores.values() for score in rows.values())
        assert best.score == best.upper_bound == optimum
        chosen = tuple(item.arc_ids for item in best.certificate.alignments)
        assert scores[word][chosen] == optimum
        assert all(item.words == (word,) for item in best.certificate.alignments)
        if revision == 0:
            limited = map_config.with_(max_iterations=1, max_features=0).solve(
                graph, owners
            )
            assert limited.status == "limit" and limited.certificate is None
            assert limited.score < optimum < limited.upper_bound == uncoordinated_bound

        projected = ep_config.infer(graph, owners)
        assert projected.status == "converged"
        assert projected.message_residual <= ep_config.tolerance
        projected_belief = tuple(
            projected.marginals["x"].log_prob(word) for word in _WORDS
        )
        assert not any(item.requires_grad for item in projected_belief)
        assert [item.exp().item() for item in projected_belief] == pytest.approx(
            exact, abs=5e-8
        )

    forced = ep_config.with_(max_sweeps=0).infer(graph, owners)
    failed = ep_config.with_(projection_max_iterations=0).infer(graph, owners)
    assert forced.status == "limit" and failed.status == "projection_failed"

    discharged = discharge_module(native)
    for target in ("stan", "bugs", "jags"):
        reasons = {
            item.computation: item.reason
            for item in project_for_target(native, discharged, target).obstacles
        }
        for name, handler in (
            ("partition", "__string_graph_score_partition"),
            ("posterior", "__string_graph_score_posterior"),
            ("pruned_belief", pruned.handler_name),
        ):
            assert f"foreign handler `{handler}`" in reasons[name]
