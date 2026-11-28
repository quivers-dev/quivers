"""Exhaustive original occurrence oracles for native deterministic path ranking."""

from fractions import Fraction
from math import prod

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import ProductiveEpsilonCycleError, RelationLimitError
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExecutionFailure,
    FixedRelationPathRankRequest,
    FixedRelationPathRankRuntimeProvider,
    QiecModule,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "checked_ranked_paths"
_SOURCE = """object Letters : FinSet 2
object Wrong : FinSet 2

relation edits [algebra=$ALGEBRA, states=6]
    input Letters
    output Letters
    initial 1 [weight=unit]
    initial 0 [weight=unit]
    arc 1 -> 2 [labels=(0, 0), weight=p]
    arc 1 -> 2 [labels=(0, 0), weight=p]
    arc 1 -> 3 [labels=(eps, eps), weight=unit]
    arc 3 -> 2 [labels=(0, 0), weight=p]
    arc 0 -> 4 [labels=(0, 0), weight=q]
    arc 2 -> 5 [labels=(eps, eps), weight=unit]
    final 0 [weight=unit]
    final 1 [weight=unit]
    final 2 [weight=unit]
    final 4 [weight=unit]
    final 5 [weight=unit]
"""


def _fixture(log_domain=False, *, cyclic=False):
    source = _SOURCE.replace("$ALGEBRA", "logprob" if log_domain else "markov")
    if cyclic:
        source = source.replace(
            "    final 0",
            "    arc 2 -> 2 [labels=(eps, eps), weight=p]\n    final 0",
        )
    compiler = Compiler(parse(source), module_name=_MODULE)
    spec = compiler.relation_specs["edits"]
    names = spec.topology.weight_names
    values = (
        {"unit": 0.0, "p": -1000.0, "q": -999.0}
        if log_domain
        else {"unit": 1.0, "p": 0.25, "q": 0.5}
    )
    parameters = {
        owner: torch.nn.Parameter(torch.tensor(values[name], dtype=torch.float64))
        for name, owner in zip(names, spec.weight_parameters, strict=True)
    }
    request = FixedRelationPathRankRequest(
        relation=spec,
        words=((0,), (0,)),
        k=9,
        trace_capacity=3,
        max_states=32,
        max_work=64,
        max_rank_entries=128,
        max_rank_work=256,
        max_result_cells=64,
    )
    p = spec.weight_parameters[names.index("p")]
    q = spec.weight_parameters[names.index("q")]
    return source, request, parameters, p, q


def _module(source, request, log_domain=False):
    k, capacity = request.k, request.trace_capacity
    score = "LogWeight" if log_domain else "Real"
    type_ = (
        f"(Int * Tensor[Int]([{k}, 2]) * Tensor[Int]([{k}, {capacity}]) "
        f"* Tensor[{score}]([{k}]))"
    )
    name = request.handler_name
    source += f"""
instance ordering : Compute[Unit, {type_}]
handler {name} for Compute[Unit, {type_}] : {type_} -> {type_} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define ranked : {type_} !{{}} =
    handle ordering with {name} in
        let paths <- perform ordering.apply(unit)
        return paths
"""
    module = Compiler(parse(source), module_name=_MODULE).qiec_module
    loaded = loads(dumps(module))
    assert isinstance(loaded, QiecModule)
    return loaded


def _run(module, provider):
    return run_named(
        module, "ranked", runtime=RuntimeConfiguration(providers=(provider,))
    )


def _enumerate(request, parameters, log_domain):
    topology = request.relation.topology
    weights = tuple(parameters[owner] for owner in request.relation.weight_parameters)
    final_slots = {mark.state: mark.weight for mark in topology.final}
    arcs = sorted(
        enumerate(topology.arcs),
        key=lambda record: next(
            (tape for tape, label in enumerate(record[1].labels) if label is not None),
            -1,
        ),
    )
    candidates = []
    zero = -torch.inf if log_domain else 0.0

    def visit(start, state, ids, words, slots):
        if state in final_slots and words == request.words:
            owners = [weights[slot] for slot in (*slots, final_slots[state])]
            raw = [item.item() for item in owners]
            if zero not in raw:
                exact = tuple(map(Fraction, raw))
                score = sum(exact) if log_domain else prod(exact)
                values = torch.stack(owners)
                tensor = values.sum() if log_domain else values.prod()
                candidates.append((score, start, state, ids, tensor))
        for arc_id, arc in arcs:
            if arc.source != state:
                continue
            emitted = tuple(
                prefix if label is None else (*prefix, label)
                for prefix, label in zip(words, arc.labels, strict=True)
            )
            if all(
                word[: len(prefix)] == prefix
                for prefix, word in zip(emitted, request.words, strict=True)
            ):
                visit(start, arc.target, (*ids, arc_id), emitted, (*slots, arc.weight))

    for mark in topology.initial:
        visit(
            mark.state, mark.state, (), tuple(() for _ in request.words), (mark.weight,)
        )
    return sorted(candidates, key=lambda item: item[0], reverse=True)[: request.k]


def _check_run(run, request, oracle, log_domain):
    assert run.result_type == request.result_type
    count, boundaries, arcs, scores = run.value
    assert count.item() == len(oracle)
    observed = list(zip(boundaries.tolist(), arcs.tolist(), strict=True))
    expected = [
        ([start, stop], list(ids) + [-1] * (request.trace_capacity - len(ids)))
        for _, start, stop, ids, _ in oracle
    ]
    absent = ([-1, -1], [-1] * request.trace_capacity)
    assert observed == expected + [absent] * (request.k - len(oracle))
    assert scores.tolist() == [float(path[0]) for path in oracle] + [
        -torch.inf if log_domain else 0.0
    ] * (request.k - len(oracle))
    return scores[: len(oracle)]


@pytest.mark.parametrize("log_domain", [False, True])
def test_native_rank_keeps_parallel_paths_order_and_live_gradient(log_domain):
    source, request, parameters, p, q = _fixture(log_domain)
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, FixedRelationPathRankRequest) and dumps(request) == wire
    module = _module(source, request, log_domain)
    provider = FixedRelationPathRankRuntimeProvider(request, parameters)
    for change in (False, True):
        if change:
            with torch.no_grad():
                parameters[p].copy_(parameters[q])
        oracle = _enumerate(request, parameters, log_domain)
        assert len(oracle) == 7
        ids = {entry[3] for entry in oracle}
        assert (0,) in ids and (1,) in ids
        assert oracle[0][3] == ((2, 3) if change else (4,))
        scores = _check_run(_run(module, provider), request, oracle, log_domain)
        index_weights = scores.new_tensor(range(1, len(oracle) + 1))
        reference = torch.stack([path[-1] for path in oracle])
        actual_gradient = torch.autograd.grad(
            scores @ index_weights, tuple(parameters.values())
        )
        expected_gradient = torch.autograd.grad(
            reference @ index_weights, tuple(parameters.values())
        )
        assert all(
            torch.equal(a, b)
            for a, b in zip(actual_gradient, expected_gradient, strict=True)
        )

    best = request.with_(k=1)
    oracle = _enumerate(best, parameters, log_domain)
    scores = _check_run(
        _run(
            _module(source, best, log_domain),
            FixedRelationPathRankRuntimeProvider(best, parameters),
        ),
        best,
        oracle,
        log_domain,
    )
    p_gradient, q_gradient = torch.autograd.grad(
        scores.sum(), (parameters[p], parameters[q])
    )
    assert p_gradient.item() == 1.0 and q_gradient.item() == 0.0


def test_empty_walks_keep_boundaries_and_absent_rows_look_different():
    source, request, parameters, _, _ = _fixture()
    for words, expected in ((((), ()), 2), (((1,), (1,)), 0)):
        request = request.with_(words=words, trace_capacity=0)
        oracle = _enumerate(request, parameters, False)
        assert len(oracle) == expected
        assert all(not path[3] for path in oracle)
        run = _run(
            _module(source, request),
            FixedRelationPathRankRuntimeProvider(request, parameters),
        )
        _check_run(run, request, oracle, False)
        assert run.value[2].shape == (request.k, 0)


def test_exhaustive_oracle_detects_omitted_parallel_occurrence(monkeypatch):
    source, request, parameters, _, _ = _fixture()
    module = _module(source, request)
    provider = FixedRelationPathRankRuntimeProvider(request, parameters)
    oracle = _enumerate(request, parameters, False)
    _check_run(_run(module, provider), request, oracle, False)
    rows = provider._walks._arcs
    without_parallel = [tuple(edge for edge in row if edge[1] != 1) for row in rows]
    monkeypatch.setattr(provider._walks, "_arcs", without_parallel)
    mutated = _run(module, provider)
    assert mutated.value[0].item() == len(oracle) - 2
    with pytest.raises(AssertionError):
        _check_run(mutated, request, oracle, False)


def test_structural_budgets_live_selected_capacity_and_cyclic_refusal():
    source, request, parameters, p, q = _fixture()
    markers = (2, 3, 1, 1, 2, 1, 1)
    rank_work = 3 + sum(n + request.k * (1 + n.bit_length()) for n in markers)
    rank_entries = len(markers) * request.k
    result_cells = 1 + request.k * (request.trace_capacity + 3)
    request = request.with_(
        max_rank_entries=rank_entries,
        max_rank_work=rank_work,
        max_result_cells=result_cells,
    )
    assert rank_work == 167 and rank_entries == 63 and result_cells == 55
    FixedRelationPathRankRuntimeProvider(request, parameters)
    for name, cap in (("max_rank_entries", rank_entries), ("max_rank_work", rank_work)):
        with pytest.raises(RelationLimitError, match=name):
            FixedRelationPathRankRuntimeProvider(
                request.with_(**{name: cap - 1}), parameters
            )
    with pytest.raises(dx.ValidationError, match="max_result_cells"):
        request.with_(max_result_cells=result_cells - 1)
    with pytest.raises(RelationLimitError, match="max_work"):
        FixedRelationPathRankRuntimeProvider(request.with_(max_work=0), parameters)

    narrow = request.with_(k=1, trace_capacity=1)
    module = _module(source, narrow)
    provider = FixedRelationPathRankRuntimeProvider(narrow, parameters)
    assert _run(module, provider).value[2].tolist() == [[4]]
    with torch.no_grad():
        parameters[p].copy_(parameters[q])
    with pytest.raises(ExecutionFailure, match="selected path exceeds trace_capacity"):
        _run(module, provider)

    _, cyclic, parameters, p, _ = _fixture(cyclic=True)
    with torch.no_grad():
        parameters[p].zero_()
    with pytest.raises(ProductiveEpsilonCycleError, match="acyclic productive graph"):
        FixedRelationPathRankRuntimeProvider(cyclic, parameters)


def test_same_shape_substitution_and_foreign_targets_refuse():
    source, request, parameters, _, _ = _fixture()
    module = _module(source, request)
    for modified in (
        request.with_(words=((1,), (1,))),
        request.with_(
            relation=request.relation.with_(tapes=(("Wrong", 2), ("Letters", 2)))
        ),
    ):
        assert (
            modified.result_type == request.result_type
            and modified.handler_name != request.handler_name
        )
        with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
            _run(module, FixedRelationPathRankRuntimeProvider(modified, parameters))
    report = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "ranked" and request.handler_name in item.reason
        for item in report.obstacles
    )
