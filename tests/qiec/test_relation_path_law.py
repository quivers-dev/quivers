"""Independent original-trace probabilities and checked Random execution."""

from fractions import Fraction
from math import log

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import ProductiveEpsilonCycleError
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExecutionFailure,
    FixedRelationPathSampleRequest,
    ParameterId,
    RuntimeConfiguration,
    RuntimeSelection,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_relation_runtime import _request


def _module(capacity=4):
    trace = f"(Int * Int * Tensor[Int]([{capacity}]))"
    source = f"""
instance random : Random

handler draw for Random : {trace} -> {trace} [coverage=total, implementation=foreign]
    sample[A : Type] resumes 1

handler replay for Random : {trace} -> {trace} [coverage=partial, forwards=unknown, implementation=foreign]
    sample[A : Type] resumes 1

define probe(law : Sampleable[{trace}]) : {trace} !{{random}} =
    let value <- perform random.sample[{trace}](site("path"), law)
    return value

define drawn(law : Sampleable[{trace}]) : {trace} !{{}} =
    handle random with draw in
        probe(law)

define replayed(law : Sampleable[{trace}]) : {trace} !{{}} =
    handle random with draw in
        handle random with replay in
            probe(law)

define density(law : Sampleable[{trace}], value : {trace}) : LogWeight !{{}} =
    return log_prob(law, value)

define mass(law : Sampleable[{trace}]) : LogWeight !{{}} =
    return log_mass(law)
"""
    compiler = Compiler(parse(source), module_name="typed-path-law")
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    return module


def _sample_request(algebra, capacity=4):
    base, parameters, ids = _request("pair_weight", algebra)
    zero_id = ParameterId.derive("relation-runtime", "zero")
    a, b, one = (parameters[identity] for identity in ids)
    relation = base.relation.with_(
        n_states=4,
        arcs=(
            (0, 1, (None, None), 0),
            (0, 1, (None, None), 2),
            (1, 2, (None, None), 1),
            (2, 3, (None, None), 4),
        ),
        final=((0, 3), (1, 3), (2, 3), (3, 3)),
        weight_names=(*base.relation.weight_names, "zero"),
        weight_parameters=(*base.relation.weight_parameters, zero_id),
    )
    with torch.no_grad():
        a.fill_(0.25 if algebra is MARKOV else log(0.25))
        b.fill_(0.5 if algebra is MARKOV else log(0.5))
    parameters[zero_id] = nn.Parameter(
        one.new_tensor(0.0 if algebra is MARKOV else -torch.inf)
    )
    request = FixedRelationPathSampleRequest(
        relation=relation,
        words=((), ()),
        max_states=20,
        max_work=40,
        trace_capacity=capacity,
    )
    return request, parameters, (a, b)


def _trace_key(value):
    initial, final, arcs = value
    return initial.item(), final.item(), tuple(arcs.tolist())


def _runtime(replay=None):
    handlers = {"draw": {"kind": "draw"}}
    if replay is not None:
        handlers["replay"] = {
            "kind": "replay",
            "policy": "clamp-no-score",
            "values": {"path": replay},
        }
    return RuntimeConfiguration(
        selections=(RuntimeSelection(provider="core", options={"handlers": handlers}),)
    )


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_typed_trace_log_density_matches_all_original_occurrences(algebra):
    request, parameters, (a, b) = _sample_request(algebra)
    decoded = loads(dumps(request))
    assert isinstance(decoded, FixedRelationPathSampleRequest)
    assert decoded == request and dumps(decoded) == dumps(request)
    law = decoded.bind(parameters, generator=torch.Generator().manual_seed(471))
    module = _module()

    def density(value) -> torch.Tensor:
        result = run_named(module, "density", (law, value)).value
        assert isinstance(result, torch.Tensor)
        return result

    a_exact, b_exact = Fraction(1, 4), Fraction(1, 2)
    paths = {
        (0, 0, (-1, -1, -1, -1)): Fraction(1),
        (0, 1, (0, -1, -1, -1)): a_exact,
        (0, 1, (1, -1, -1, -1)): b_exact,
        (0, 2, (0, 2, -1, -1)): a_exact**2,
        (0, 2, (1, 2, -1, -1)): b_exact * a_exact,
    }
    normalizer = sum(paths.values())
    masses = [density(path).exp() for path in paths]
    assert [mass.item() for mass in masses] == pytest.approx(
        [float(mass / normalizer) for mass in paths.values()]
    )
    assert torch.stack(masses).sum().item() == pytest.approx(1)
    total = run_named(module, "mass", (law,)).value
    assert isinstance(total, torch.Tensor) and total.item() == 0
    assert density((0, 3, (0, 2, 3, -1))).item() == -torch.inf
    assert density((0, 2, (0, -1, 2, -1))).item() == -torch.inf
    assert density((0, 0, (0, -1, -1, -1))).item() == -torch.inf
    assert density((1, 1, (-1, -1, -1, -1))).item() == -torch.inf

    score = density((0, 2, (0, 2, -1, -1)))
    derivative = torch.autograd.grad(score, (a, b))
    expected_a = 2 / a_exact - (1 + 2 * a_exact + b_exact) / normalizer
    expected_b = -(1 + a_exact) / normalizer
    if algebra is LOG_PROB:
        expected_a *= a_exact
        expected_b *= b_exact
    assert [value.item() for value in derivative] == pytest.approx(
        [float(expected_a), float(expected_b)]
    )
    with torch.no_grad():
        parameters[request.relation.weight_parameters[-1]].fill_(
            0.125 if algebra is MARKOV else log(0.125)
        )
    assert torch.isfinite(density((0, 3, (0, 2, 3, -1))))


def test_checked_random_uses_a_live_seeded_draw_and_replays_the_same_trace():
    request, parameters, _ = _sample_request(MARKOV)
    generator = torch.Generator().manual_seed(125)
    law = request.bind(parameters, generator=generator)
    module = _module()
    start = generator.get_state().clone()
    direct = _trace_key(law.sample())
    generator.set_state(start)
    drawn = run_named(module, "drawn", (law,), runtime=_runtime()).value
    assert _trace_key(drawn) == direct
    after_draw = generator.get_state().clone()
    assert not torch.equal(after_draw, start)
    replayed = run_named(module, "replayed", (law,), runtime=_runtime(direct)).value
    assert _trace_key(replayed) == direct
    assert torch.equal(generator.get_state(), after_draw)

    other = request.with_(trace_capacity=3).bind(parameters, generator=generator)
    assert other.element != law.element
    with pytest.raises(ExecutionFailure) as error:
        run_named(module, "mass", (other,))
    assert error.value.diagnostic.code == "qiec-run-argument"
    assert torch.equal(generator.get_state(), after_draw)

    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        obstacles = project_for_target(module, discharged, target).obstacles
        assert any(
            item.computation == "drawn" and "foreign handler `draw`" in item.reason
            for item in obstacles
        )


def test_trace_capacity_is_checked_before_any_draw_even_for_dormant_arcs():
    request, parameters, _ = _sample_request(MARKOV, capacity=2)
    generator = torch.Generator().manual_seed(52)
    initial_state = generator.get_state().clone()
    with pytest.raises(ValueError, match="smaller than a structurally accepting path"):
        request.bind(parameters, generator=generator)
    assert torch.equal(generator.get_state(), initial_state)
    cycle = request.with_(
        relation=request.relation.with_(
            arcs=(*request.relation.arcs, (0, 0, (None, None), 0))
        )
    )
    with pytest.raises(ProductiveEpsilonCycleError, match="productive"):
        cycle.bind(parameters, generator=generator)
    assert torch.equal(generator.get_state(), initial_state)


def test_zero_arc_capacity_handles_the_empty_path_without_padding():
    request, parameters, _ = _sample_request(MARKOV, capacity=0)
    relation = request.relation.with_(n_states=1, arcs=(), final=((0, 3),))
    law = request.with_(relation=relation).bind(
        parameters, generator=torch.Generator().manual_seed(31)
    )
    module = _module(capacity=0)
    drawn = run_named(module, "drawn", (law,), runtime=_runtime()).value
    assert _trace_key(drawn) == (0, 0, ())
    score = run_named(module, "density", (law, drawn)).value
    assert isinstance(score, torch.Tensor) and score.item() == 0


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_layered_posterior_handles_exponentially_many_underflowing_paths(algebra):
    base, parameters, (a_id, _, _) = _request("pair_weight", algebra)
    steps = 40
    relation = base.relation.with_(
        n_states=steps + 1,
        arcs=tuple(
            (state, state + 1, (None, None), slot)
            for state in range(steps)
            for slot in (0, 1)
        ),
        final=((steps, 3),),
    )
    with torch.no_grad():
        parameters[a_id].fill_(2.0**-100 if algebra is MARKOV else -1000.0)
    request = FixedRelationPathSampleRequest(
        relation=relation,
        words=((), ()),
        max_states=steps + 1,
        max_work=2 * steps,
        trace_capacity=steps,
    )
    law = request.bind(parameters, generator=torch.Generator().manual_seed(4))
    chosen = law.sample()
    initial, final, ids = _trace_key(chosen)
    assert initial == 0 and final == steps
    assert tuple(arc_id // 2 for arc_id in ids) == tuple(range(steps))
    score = law.log_prob(chosen)
    assert score.item() == pytest.approx(-steps * log(2), abs=1e-9, rel=0)
    (derivative,) = torch.autograd.grad(score, parameters[a_id])
    if algebra is MARKOV:
        derivative = derivative * parameters[a_id]
    assert derivative.item() == pytest.approx(0, abs=5e-9)


def test_connected_trace_with_wrong_tape_words_has_zero_posterior_mass():
    base, parameters, _ = _request("pair_weight")
    request = FixedRelationPathSampleRequest(
        relation=base.relation,
        words=base.words,
        max_states=12,
        max_work=16,
        trace_capacity=1,
    )
    law = request.bind(parameters, generator=torch.Generator().manual_seed(12))
    module = _module(capacity=1)
    for arc_id, expected in ((0, -log(2)), (1, -log(2)), (2, -torch.inf)):
        value = run_named(module, "density", (law, (0, 1, (arc_id,)))).value
        assert isinstance(value, torch.Tensor)
        assert value.item() == pytest.approx(expected)
