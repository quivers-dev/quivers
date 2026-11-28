"""An authored graph through the typed approximate Compute boundary."""

from fractions import Fraction
from math import exp, inf, log1p

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.qiec import (
    CoreRuntimeProvider,
    ExecutionFailure,
    ParameterId,
    PrunedStringBPConfig,
    PrunedStringGraphRequest,
    PrunedStringGraphRuntimeProvider,
    QiecModule,
    RelationWeightComplement,
    RuntimeConfiguration,
    TreeStringGraphRequest,
    discharge_module,
    dumps,
    loads,
    pruned_string_graph_handler,
    run_named,
)
from quivers.qiec.identifiers import ComputationId, EffectInstanceId
from quivers.transpile import project_for_target
from tests.dsl.test_authored_string_graphs import _MODULE, _compile, _evaluate
from tests.test_pruned_string_beliefs import _SCOPES, _WORDS, _oracle

_CYCLIC_SOURCE = """object Letters : FinSet 2

string_graph cycle
    variable x : Letters
    variable y : Letters
    variable z : Letters
    factor edge(x, y)
    factor edge(y, z)
    factor edge(x, z)

string_graph_score partition : Real [operation=tree_partition, graph=cycle]

relation edge [algebra=markov, states=2]
    input Letters
    output Letters
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(1, 1), weight=w0]
    arc 0 -> 1 [labels=(1, eps), weight=w1]
    arc 0 -> 1 [labels=(1, 0), weight=w2]
    arc 0 -> 1 [labels=(eps, 1), weight=w3]
    arc 0 -> 1 [labels=(eps, eps), weight=w4]
    arc 0 -> 1 [labels=(eps, 0), weight=w5]
    arc 0 -> 1 [labels=(0, 1), weight=w6]
    arc 0 -> 1 [labels=(0, eps), weight=w7]
    arc 0 -> 1 [labels=(0, 0), weight=w8]
    final 1 [weight=one]
"""


def _with_pruned_query(module, request):
    template = next(comp for comp in module.computations if comp.name == "partition")
    body = template.body
    stored = next(
        instance
        for instance in module.instances
        if instance.entry.instance == body.instance
    )
    identity = EffectInstanceId.derive(module.module, "pruned", "belief")
    handler = pruned_string_graph_handler(module.module, request)
    entry = stored.entry.with_(instance=identity, effect=handler.effect)
    instance = stored.with_(name="pruned_belief_instance", entry=entry)
    operation = body.computation.request.with_(
        instance=identity, effect=handler.effect, result_type=request.result_type
    )
    computation = template.with_(
        name="pruned_belief",
        id=ComputationId.derive(module.module, "pruned", "belief"),
        type=template.type.with_(result=request.result_type),
        body=body.with_(
            instance=identity,
            handler=handler.id,
            computation=body.computation.with_(request=operation),
        ),
    )
    return module.with_(
        computations=(*module.computations, computation),
        handlers=(*module.handlers, handler),
        instances=(*module.instances, instance),
    )


def _run(module, request, parameters, *, provider=None):
    if provider is None:
        provider = PrunedStringGraphRuntimeProvider(request, parameters)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    checked = run_named(module, "pruned_belief", runtime=runtime)
    assert checked.result_type == request.result_type
    return checked.value


def _owner(name):
    return ParameterId.derive(_MODULE, "relation_weights", name)


def test_ordinary_qvr_pruned_handler_preserves_graph_owner_and_live_complements():
    compiler, _ = _compile(_CYCLIC_SOURCE)
    graph = compiler.string_graph_specs["cycle"]
    config = PrunedStringBPConfig(
        supports=tuple((name, _WORDS) for name in ("x", "y", "z")),
        beam=2,
        max_sweeps=2,
        max_entries=70,
        max_states=400,
        max_work=500,
        damping=1 / 3,
        tolerance=None,
    )
    request = PrunedStringGraphRequest(graph=graph, config=config, variable="x")
    report = "(String * String * Int * Real * Bool * Tensor[LogWeight]([3]) * Bool * Tensor[LogWeight]([6, 3]))"
    name = request.handler_name
    native = (
        _CYCLIC_SOURCE
        + f"""
instance beliefs : Compute[Unit, {report}]
handler {name} for Compute[Unit, {report}] : {report} -> {report} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define pruned_belief : {report} !{{}} =
    handle beliefs with {name} in
        let result <- perform beliefs.apply(unit)
        return result
"""
    )
    source, compiled = _compile(native)
    module = loads(dumps(compiled))
    assert isinstance(module, QiecModule)
    assert source.string_graph_specs["cycle"] == request.graph
    assert pruned_string_graph_handler(module.module, request) not in module.handlers

    constants = tuple(map(Fraction, (0.25, 0.5, 0.75, 0.125, 0.25, 0.125, 0.0625)))
    p = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    parameters = {
        _owner(f"w{i}"): p.new_tensor(float(value))
        for i, value in enumerate(constants, 2)
    }
    parameters.update(
        {
            _owner("w0"): p,
            _owner("w1"): RelationWeightComplement(owner=_owner("w0")),
            _owner("one"): p.new_ones(()),
        }
    )
    provider = PrunedStringGraphRuntimeProvider(request, parameters)
    for value in (Fraction(1, 4), Fraction(3, 8)):
        with torch.no_grad():
            p.fill_(float(value))
        status, reason, sweeps, _, _, belief, *_ = _run(
            module, request, parameters, provider=provider
        )
        assert (status, reason, sweeps.item()) == ("limit", "", 2)
        expected = _oracle((value, 1 - value, *constants), Fraction(1, 3), 2)[2][0]
        assert belief.exp().tolist() == pytest.approx(list(map(float, expected)))
        step = Fraction(1, 2**18)
        plus = _oracle((value + step, 1 - value - step, *constants), Fraction(1, 3), 2)[
            2
        ][0][0]
        minus = _oracle(
            (value - step, 1 - value + step, *constants), Fraction(1, 3), 2
        )[2][0][0]
        (gradient,) = torch.autograd.grad(belief[0].exp(), p)
        assert gradient.item() == pytest.approx(
            float((plus - minus) / (2 * step)), abs=1e-8
        )

    other_variable = request.with_(variable="y")
    assert other_variable.result_type == request.result_type
    with pytest.raises(ExecutionFailure, match="incompatible pruned string handler"):
        _run(module, other_variable, parameters)
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            item.computation == "pruned_belief" and name in item.reason
            for item in project_for_target(module, discharged, target).obstacles
        )


def test_checked_loopy_belief_and_ordered_proposal_scales_match_fraction_oracle():
    compiler, authored = _compile(_CYCLIC_SOURCE)
    graph = compiler.string_graph_specs["cycle"]
    config = PrunedStringBPConfig(
        supports=tuple((name, _WORDS) for name in ("x", "y", "z")),
        beam=2,
        max_sweeps=2,
        max_entries=70,
        max_states=400,
        max_work=500,
        damping=1 / 3,
        tolerance=None,
    )
    request = PrunedStringGraphRequest(graph=graph, config=config, variable="x")
    assert request.candidate_words == ((1,), (), (0,))
    module = loads(dumps(_with_pruned_query(authored, request)))
    assert isinstance(module, QiecModule)
    request = loads(dumps(request))
    assert isinstance(request, PrunedStringGraphRequest)
    assert request.graph == compiler.string_graph_specs["cycle"]
    assert request.config == config
    exact = tuple(
        map(Fraction, (0.875, 0.625, 0.25, 0.5, 0.75, 0.125, 0.25, 0.125, 0.0625))
    )
    parameters = {
        _owner(f"w{index}"): nn.Parameter(
            torch.tensor(float(value), dtype=torch.float64)
        )
        for index, value in enumerate(exact)
    }
    parameters[_owner("one")] = torch.tensor(1.0, dtype=torch.float64)
    provider = PrunedStringGraphRuntimeProvider(request, parameters)
    learned = parameters[_owner("w0")]
    for weights in (exact, (Fraction(7, 8) + Fraction(1, 64), *exact[1:])):
        with torch.no_grad():
            learned.fill_(float(weights[0]))
        report = _run(module, request, parameters, provider=provider)
        status, reason, sweeps, residual, truncated, belief, proposals, masses = report
        _, raw, marginals, expected_residual = _oracle(weights, Fraction(1, 3), 2)
        assert (status, reason, sweeps.item()) == ("limit", "", 2)
        assert truncated.item() and proposals.item()
        assert residual.item() == pytest.approx(float(expected_residual))
        assert belief.exp().tolist() == pytest.approx(list(map(float, marginals[0])))
        rows = []
        for f, scope in enumerate(_SCOPES):
            for variable in scope:
                kept, original, omitted = raw[f, variable]
                rows.append(
                    [
                        float(sum(original)),
                        float(sum(original[index] for index in kept)),
                        float(omitted),
                    ]
                )
        torch.testing.assert_close(
            masses.exp(), torch.tensor(rows, dtype=torch.float64)
        )
        (gradient,) = torch.autograd.grad(belief[0].exp(), learned)
        step = Fraction(1, 2**18)
        plus = (weights[0] + step, *weights[1:])
        minus = (weights[0] - step, *weights[1:])
        numerical = (
            _oracle(plus, Fraction(1, 3), 2)[2][0][0]
            - _oracle(minus, Fraction(1, 3), 2)[2][0][0]
        ) / (2 * step)
        assert gradient.item() == pytest.approx(float(numerical), abs=1e-8)

    reordered = request.with_(
        config=request.config.with_(
            supports=(("x", tuple(reversed(_WORDS))), *request.config.supports[1:])
        )
    )
    alternate_graph = request.graph.with_(
        factors=tuple(reversed(request.graph.factors))
    )
    for alternate in (reordered, request.with_(graph=alternate_graph)):
        assert alternate.result_type == request.result_type
        assert alternate.handler_name != request.handler_name
        with pytest.raises(
            ExecutionFailure, match="incompatible pruned string handler"
        ):
            _run(module, alternate, parameters)

    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            record.computation == "pruned_belief"
            and f"foreign handler `{request.handler_name}`" in record.reason
            for record in project_for_target(module, discharged, target).obstacles
        )


def test_exact_tree_and_pruned_modes_keep_the_same_authored_graph_and_report_refusals():
    compiler, authored = _compile()
    graph = compiler.string_graph_specs["chain"]
    config = PrunedStringBPConfig(
        supports=(
            ("left", ((1,), (0,))),
            ("middle", ((1,), (0,))),
            ("right", ((0,), (1,))),
            ("empty", ((),)),
        ),
        beam=2,
        max_sweeps=8,
        max_entries=14,
        max_states=160,
        max_work=180,
    )
    request = PrunedStringGraphRequest(graph=graph, config=config, variable="left")
    assert request.with_(variable="right").candidate_words == ((1,),)
    parameters = {
        _owner("same"): nn.Parameter(torch.tensor(0.5, dtype=torch.float64)),
        _owner("different"): nn.Parameter(torch.tensor(0.25, dtype=torch.float64)),
        _owner("one"): torch.tensor(1.0, dtype=torch.float64),
    }
    native = _with_pruned_query(authored, request)
    exact = TreeStringGraphRequest(
        graph=graph, variable="left", max_states=80, max_work=180
    )
    partition = _evaluate(native, "partition", exact, parameters)
    assert partition.item() == float(Fraction(9, 16))

    converged = _run(native, request, parameters)
    assert converged[0] == "converged"
    assert not converged[4].item() and converged[6].item()
    assert converged[5].exp().tolist() == pytest.approx([5 / 9, 4 / 9])
    torch.testing.assert_close(
        converged[7].exp(),
        torch.tensor(
            [[0.75, 0.75, 0.0]] * 3 + [[0.375, 0.375, 0.0]], dtype=torch.float64
        ),
    )
    observed = request.with_(variable="right")
    assert _run(_with_pruned_query(authored, observed), observed, parameters)[
        5
    ].tolist() == [0.0]

    conflict = request.with_(config=config.with_(beam=1))
    limited = request.with_(config=config.with_(max_entries=5))
    assert request.handler_name == conflict.handler_name == limited.handler_name
    empty = _run(native, conflict, parameters)
    assert empty[0] == "empty_support" and empty[1]
    assert empty[2].item() == 1 and empty[3].item() == -1
    assert empty[4].item() and empty[6].item()
    assert empty[5].tolist() == [-torch.inf, -torch.inf]
    assert empty[7][2].exp().tolist() == pytest.approx([0.75, 0.5, 0.25])
    exhausted = _run(native, limited, parameters)
    assert exhausted[0] == "resource_limit" and "max_entries" in exhausted[1]
    assert exhausted[2].item() == 0 and exhausted[3].item() == -1
    assert not exhausted[4].item() and not exhausted[6].item()
    assert torch.isneginf(exhausted[5]).all() and torch.isneginf(exhausted[7]).all()

    with pytest.raises(dx.ValidationError, match="invalid symbol"):
        request.with_(
            config=config.with_(supports=(("left", ((2,),)), *config.supports[1:]))
        )
    with pytest.raises(dx.ValidationError, match="excludes its observation"):
        request.with_(
            config=config.with_(
                supports=(
                    *config.supports[:2],
                    ("right", ((0,),)),
                    config.supports[3],
                )
            )
        )


_GAUGE_SOURCE = """object Letters : FinSet 2

relation weights [algebra=logprob, states=2]
    output Letters
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=w0]
    arc 0 -> 1 [labels=(1), weight=w1]
    final 1 [weight=unit]

string_graph centered
    variable x : Letters
    factor weights(x)
"""


def _pruned_gauge_fixture():
    compiler, _ = _compile(_GAUGE_SOURCE)
    config = PrunedStringBPConfig(
        supports=(("x", ((0,), (1,))),),
        beam=2,
        max_sweeps=1,
        tolerance=1e-9,
        damping=0.5,
        max_entries=2,
        max_states=10,
        max_work=20,
    )
    request = PrunedStringGraphRequest(
        graph=compiler.string_graph_specs["centered"], config=config, variable="x"
    )
    report = "(String * String * Int * Real * Bool * Tensor[LogWeight]([2]) * Bool * Tensor[LogWeight]([1, 3]))"
    name = request.handler_name
    _, parsed = _compile(
        _GAUGE_SOURCE
        + f"""
instance gauge : Compute[Unit, {report}]
handler {name} for Compute[Unit, {report}] : {report} -> {report} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define pruned_belief : {report} !{{}} =
    handle gauge with {name} in
        let result <- perform gauge.apply(unit)
        return result
"""
    )
    module = loads(dumps(parsed))
    assert isinstance(module, QiecModule)
    owners = tuple(
        nn.Parameter(torch.tensor(0.0, dtype=torch.float64)) for _ in range(2)
    )
    weights = {_owner("unit"): owners[0].new_zeros(())}
    weights.update(zip((_owner("w0"), _owner("w1")), owners, strict=True))
    return request, module, weights, owners


def test_checked_damped_pruning_normalizes_in_the_candidate_gauge():
    request, module, weights, owners = _pruned_gauge_fixture()
    provider = PrunedStringGraphRuntimeProvider(request, weights)
    fixtures = ((0, 0), (-1e20, -1e20), (1e20, 1e20), (0, 2), (1e6, 1e6 + 2))
    for left, right in fixtures:
        with torch.no_grad():
            owners[0].fill_(left)
            owners[1].fill_(right)
        status, reason, sweeps, residual, truncated, belief, present, masses = _run(
            module, request, weights, provider=provider
        )
        candidate = 1 / (1 + exp(right - left))
        # One unary factor has unit cavities. The starting message is
        # uniform, so one damped update mixes it with the exact candidate.
        fitted = 0.25 + 0.5 * candidate
        distance = abs(fitted - candidate)
        assert status == ("converged" if distance == 0 else "limit")
        assert (reason, sweeps.item(), truncated.item(), present.item()) == (
            "",
            1,
            False,
            True,
        )
        assert residual.item() == pytest.approx(distance, abs=1e-15)
        assert belief.exp().tolist() == pytest.approx([fitted, 1 - fitted], abs=1e-15)
        expected_derivative = 0.5 * candidate * (1 - candidate)
        derivative = torch.autograd.grad(belief[0].exp(), owners)
        assert [grad.item() for grad in derivative] == pytest.approx(
            [expected_derivative, -expected_derivative], abs=1e-15
        )
        raw_mass = max(left, right) + log1p(exp(-abs(left - right)))
        assert masses.tolist() == [[raw_mass, raw_mass, -inf]]


def test_checked_damped_pruning_keeps_log_zero_and_refuses_range_overflow():
    request, module, weights, owners = _pruned_gauge_fixture()
    provider = PrunedStringGraphRuntimeProvider(request, weights)
    with torch.no_grad():
        owners[1].fill_(-inf)
    status, _, _, residual, _, belief, *_ = _run(
        module, request, weights, provider=provider
    )
    assert status == "converged" and residual.item() == 0
    assert belief.tolist() == [0.0, -inf]

    limit = torch.finfo(torch.float64).max
    with torch.no_grad():
        owners[0].fill_(-limit)
        owners[1].fill_(limit)
    with pytest.raises(
        ExecutionFailure, match="candidate.*outside the evaluation precision"
    ) as rejected:
        _run(module, request, weights, provider=provider)
    assert rejected.value.diagnostic.code == "qiec-run-evaluation"
