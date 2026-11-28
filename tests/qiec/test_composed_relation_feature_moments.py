"""Original accepted path-pair moments after hiding a cyclic middle tape."""

from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import log

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import ProductiveEpsilonCycleError, RelationLimitError
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ComposedRelationSpec,
    FixedComposedArcFeatureMomentRequest,
    FixedComposedRelationRequest,
    FixedRelationRuntimeProvider,
    ParameterId,
    discharge_module,
    dumps,
    loads,
    run_program,
)
from quivers.transpile import project_for_target
from tests.qiec.test_relation_composition_runtime import _operands
from tests.qiec.test_relation_feature_moments import _module, _wire_logs

_MODULE = "composed_cyclic_features"
_SOURCE = """object Letters : FinSet 2
object Middle : FinSet 2
object Output : FinSet 2

relation projected_channel = compose(left, right)

relation left [algebra=markov, states=1]
    input Letters
    output Middle
    initial 0 [weight=unit]
    arc 0 -> 0 [labels=(0, 0), weight=branch]
    arc 0 -> 0 [labels=(0, 1), weight=branch]
    arc 0 -> 0 [labels=(eps, eps), weight=loop]
    final 0 [weight=unit]

relation right [algebra=markov, states=1]
    input Middle
    output Output
    initial 0 [weight=unit]
    arc 0 -> 0 [labels=(0, 0), weight=b0]
    arc 0 -> 0 [labels=(1, 0), weight=b1]
    arc 0 -> 0 [labels=(eps, eps), weight=loop]
    final 0 [weight=unit]
"""
_P = Fraction(1, 8)
_A = Fraction(1, 2)
_B0, _B1 = Fraction(1, 2), Fraction(1, 4)


def _owner(name):
    return ParameterId.derive(_MODULE, "relation_weights", name)


def _fixture(length, algebra=MARKOV):
    source = _SOURCE if algebra is MARKOV else _SOURCE.replace("markov", "logprob")
    compiler = Compiler(parse(source), module_name=_MODULE)
    compiler.compile()
    relation = compiler.composed_relation_specs["projected_channel"]
    scalar = FixedComposedRelationRequest(
        relation=relation,
        words=((0,) * length, (0,) * length),
        max_states=1200,
        max_work=4300,
        execution="materialized",
        max_closure_size=8,
    )
    request = FixedComposedArcFeatureMomentRequest(
        request=scalar,
        feature_extent=5,
        left_arc_logs=_wire_logs(((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0))),
        right_arc_logs=_wire_logs(((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 1, 0))),
        max_feature_cells=12_800,
    )
    weights = {
        _owner(name): nn.Parameter(
            torch.tensor(
                float(value) if algebra is MARKOV else log(value), dtype=torch.float64
            )
        )
        for name, value in (
            ("unit", Fraction(1)),
            ("branch", _A),
            ("loop", _P),
            ("b0", _B0),
            ("b1", _B1),
        )
    }
    return request, weights


def _oracle(length, loop):
    # Pick the common hidden word and independently pick the number of
    # epsilon transitions on each operand in each gap. Each choice is
    # exactly one accepted original pair, even when both sides advance.
    per_symbol = _A * (_B0 + _B1)
    total = per_symbol**length / (1 - loop) ** (2 * (length + 1))
    eps_visits = (length + 1) * loop / (1 - loop)
    moment = (
        total * length * _B0 / (_B0 + _B1),
        total * length * _B1 / (_B0 + _B1),
        total * eps_visits,
        total * eps_visits,
        Fraction(0),
    )
    shuffling = per_symbol**length / (1 - 2 * loop) ** (length + 1)
    return (total, *moment), shuffling


@pytest.mark.parametrize("length,algebra", [(0, MARKOV), (5, LOG_PROB), (64, MARKOV)])
def test_long_projected_composition_features_match_accepted_path_pairs(length, algebra):
    request, weights = _fixture(length, algebra)
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, FixedComposedArcFeatureMomentRequest)
    assert dumps(request) == wire
    left, right = request.request.relation.left, request.request.relation.right
    assert (
        left.weight_parameters[left.arcs[0][-1]]
        == left.weight_parameters[left.arcs[1][-1]]
    )
    assert (
        left.weight_parameters[left.arcs[2][-1]]
        == right.weight_parameters[right.arcs[2][-1]]
    )
    native = loads(dumps(_module(request, extent=6)))
    provider = FixedRelationRuntimeProvider(
        request, weights, handler_name=request.handler_name
    )
    loop_owner = weights[_owner("loop")]
    for p in (_P, Fraction(1, 4)):
        with torch.no_grad():
            loop_owner.fill_(float(p) if algebra is MARKOV else log(p))
        reference, shuffled = _oracle(length, p)
        actual = run_program(
            native, "features", data={}, sites={}, providers=(provider,)
        ).value
        assert actual.numel() == 6
        for value, expected in zip(actual, reference, strict=True):
            if expected:
                assert value.item() == pytest.approx(log(expected), abs=1e-10, rel=0.0)
            else:
                assert value.item() == -torch.inf
        assert actual[0].item() < log(shuffled)
        d_log_total = 2 * (length + 1) / (1 - p)
        for output, derivative in (
            (actual[0], d_log_total),
            (actual[3], d_log_total + 1 / p + 1 / (1 - p)),
            (actual[4], d_log_total + 1 / p + 1 / (1 - p)),
        ):
            (gradient,) = torch.autograd.grad(output, loop_owner, retain_graph=True)
            if algebra is LOG_PROB:
                derivative *= p
            assert gradient.item() == pytest.approx(
                float(derivative), abs=2e-9, rel=0.0
            )

    if length == 5:
        discharged = discharge_module(native)
        for target in ("stan", "bugs", "jags"):
            assert any(
                item.computation == "evaluate"
                and f"foreign handler `{request.handler_name}`" in item.reason
                for item in project_for_target(native, discharged, target).obstacles
            )


def test_feature_budget_counts_the_materialized_split_edges_and_visited_chart():
    length = 5
    request, weights = _fixture(length)
    scalar = request.request
    composition = scalar.relation.bind(weights)
    machine, origins = composition._materialize(
        max_states=scalar.max_states, max_work=scalar.max_work, capture_origins=True
    )
    topology = machine.topology
    # Three filter states, six split emitting matches, one split paired
    # epsilon, four one-sided epsilons, and three stopping markers.
    assert (
        topology.n_states,
        len(topology.arcs),
        len(topology.initial),
        len(topology.final),
    ) == (10, 18, 1, 3)
    assert Counter(origins) == {(side, arc): 3 for side in range(2) for arc in range(3)}
    reused, reused_origins = composition._materialize(
        max_states=scalar.max_states, max_work=scalar.max_work, capture_origins=True
    )
    assert reused.topology is topology and reused_origins is origins
    query = scalar.plan(weights)
    assert query.work.active_states == 10 * length + 4
    assert query.work.active_arcs == 18 * length + 6
    cells = 6 * (6 + 10 + 18 + 1 + 3 + (10 * length + 4) + (18 * length + 6))
    assert request.with_(max_feature_cells=cells).plan(weights)().shape == (6,)
    with pytest.raises(RelationLimitError, match="operand feature cells"):
        request.with_(max_feature_cells=35).plan({})
    with pytest.raises(RelationLimitError, match="materialized feature cells"):
        request.with_(max_feature_cells=6 * (6 + 10 + 18 + 1 + 3) - 1).plan(weights)
    with pytest.raises(RelationLimitError, match="composed observation feature cells"):
        request.with_(max_feature_cells=cells - 1).plan(weights)
    with pytest.raises(RelationLimitError, match="composition exceeds max_states"):
        request.with_(request=scalar.with_(max_states=9)).plan(weights)
    with pytest.raises(ProductiveEpsilonCycleError, match="productive epsilon"):
        request.with_(request=scalar.with_(max_closure_size=None)).plan(weights)
    with pytest.raises(dx.ValidationError, match="materialized execution"):
        request.with_(request=scalar.with_(execution="lazy", max_closure_size=None))
    reordered = request.with_(left_arc_logs=tuple(reversed(request.left_arc_logs)))
    assert reordered.result_type == request.result_type
    with pytest.raises(ValueError, match="exact observation handler"):
        FixedRelationRuntimeProvider(reordered, {}, handler_name=request.handler_name)


def test_legacy_scalar_composition_remains_byte_stable():
    baselines = (
        (
            MARKOV,
            "lazy",
            "5713689c7b57a0d5b2fe098be5a295f42ef7510b0a50ba2f158b0a7edabd4c8a",
        ),
        (
            MARKOV,
            "materialized",
            "f2e57b9c117aca81cd4873d8a281c599c63b81c0594a56760ada4fa3b05eb9a5",
        ),
        (
            LOG_PROB,
            "lazy",
            "3493ad142ae8f761ce900ef1e9ef96178aaaa9b6c0e7f8ba01f1af2ac06a3801",
        ),
        (
            LOG_PROB,
            "materialized",
            "13ed4b7ed3e57c2a645adda80e3b42b45797e7413185c49edf0573a5b7cc8d74",
        ),
    )
    for algebra, execution, digest in baselines:
        left, right = _operands(algebra)
        request = FixedComposedRelationRequest(
            relation=ComposedRelationSpec(left=left, right=right),
            words=((0,), (0,)),
            max_states=32,
            max_work=64,
            execution=execution,
        )
        wire = dumps(request)
        assert sha256(wire.encode()).hexdigest() == digest
        assert dumps(loads(wire)) == wire
