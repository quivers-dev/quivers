"""Independent geometric oracles for the tagged native path pushforward."""

from fractions import Fraction
from hashlib import sha256
from math import log

import pytest
import torch

from quivers.core.algebras import (
    LOG_PROB,
    MARKOV,
    DivergentClosureError,
    UndecidedClosureError,
)
from quivers.core.relations import RelationLimitError
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    CappedRelationPathLaw,
    ExecutionFailure,
    FixedCappedCyclicRelationPathSampleRequest,
    ParameterId,
    QiecModule,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.qiec.builtins import _support
from quivers.qiec.evaluator import InvalidHandlerError
from quivers.transpile import project_for_target
from tests.qiec.test_cyclic_relation_path_law import _fixture, _loop_owner
from tests.qiec.test_relation_path_law import _runtime

_P = Fraction(1, 8)


def _request(algebra=MARKOV, capacity=1):
    original, parameters = _fixture(algebra, capacity)
    with torch.no_grad():
        parameters[_loop_owner()].fill_(float(_P) if algebra is MARKOV else log(_P))
    return FixedCappedCyclicRelationPathSampleRequest(
        request=original,
        max_closure_cells=4,
        max_prefix_work=128,
        max_prefix_cells=64,
    ), parameters


def _module(capacity):
    outcome = f"(Bool * Int * Int * Tensor[Int]([{capacity}]))"
    source = f"""
instance random : Random

handler draw for Random : {outcome} -> {outcome} [coverage=total, implementation=foreign]
    sample[A : Type] resumes 1

handler replay for Random : {outcome} -> {outcome} [coverage=partial, forwards=unknown, implementation=foreign]
    sample[A : Type] resumes 1

define probe(law : Sampleable[{outcome}]) : {outcome} !{{random}} =
    let value <- perform random.sample[{outcome}](site("path"), law)
    return value

define drawn(law : Sampleable[{outcome}]) : {outcome} !{{}} =
    handle random with draw in
        probe(law)

define replayed(law : Sampleable[{outcome}]) : {outcome} !{{}} =
    handle random with draw in
        handle random with replay in
            probe(law)

define density(law : Sampleable[{outcome}], value : {outcome}) : LogWeight !{{}} =
    return log_prob(law, value)

define mass(law : Sampleable[{outcome}]) : LogWeight !{{}} =
    return log_mass(law)
"""
    module = Compiler(
        parse(module_to_source(parse(source))), module_name="checked-capped-path-law"
    ).qiec_module
    decoded = loads(dumps(module))
    assert isinstance(decoded, QiecModule)
    return decoded


def _key(value):
    tail, initial, final, ids = value
    return tail.item(), initial.item(), final.item(), tuple(ids.tolist())


def _tail(capacity):
    return True, -1, -1, (-1,) * capacity


def _trace(ids=(), capacity=1):
    return False, 0, 0, (*ids, *((-1,) * (capacity - len(ids))))


def _density(module, law, value):
    result = run_named(module, "density", (law, value)).value
    assert isinstance(result, torch.Tensor)
    return result


@pytest.mark.parametrize("algebra", (MARKOV, LOG_PROB))
def test_checked_probability_and_gradients_for_a_partition_of_geometric_paths(algebra):
    request, parameters = _request(algebra)
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, FixedCappedCyclicRelationPathSampleRequest)
    assert (
        dumps(request) == wire and request.element_type != request.request.element_type
    )
    generator = torch.Generator().manual_seed(173)
    law = request.bind(parameters, generator=generator)
    assert (
        isinstance(law, CappedRelationPathLaw) and law.element == request.element_type
    )
    before = generator.get_state().clone()
    module = _module(1)
    outcomes = (_trace(), _trace((0,)), _trace((1,)), _tail(1))
    weight = parameters[_loop_owner()]
    for p in (_P, Fraction(1, 4)):
        with torch.no_grad():
            weight.fill_(float(p) if algebra is MARKOV else log(p))
        stop = 1 - 2 * p
        probs = (stop, stop * p, stop * p, (2 * p) ** 2)
        deriv = (-2 / stop, 1 / p - 2 / stop, 1 / p - 2 / stop, 2 / p)
        assert sum(probs) == 1
        scores = [_density(module, law, value) for value in outcomes]
        assert [score.item() for score in scores] == pytest.approx(
            [log(prob) for prob in probs]
        )
        grads = [torch.autograd.grad(score, weight)[0].item() for score in scores]
        assert grads == pytest.approx(
            [float(d if algebra is MARKOV else p * d) for d in deriv]
        )
        assert sum(score.exp().item() for score in scores) == pytest.approx(1.0)
        assert run_named(module, "mass", (law,)).value.item() == 0
    with pytest.raises(InvalidHandlerError, match="implicit"):
        _support(law)
    assert torch.equal(before, generator.get_state())


def _geometric_draw(seed, capacity):
    # This models the elementary Markov experiment, independent of the
    # path and closure implementations. It returns the RNG state too.
    generator = torch.Generator().manual_seed(seed)
    torch.multinomial(torch.ones(1, dtype=torch.float64), 1, generator=generator)
    weights = torch.tensor(
        (float(_P), float(_P), float(1 - 2 * _P)), dtype=torch.float64
    )
    arcs = []
    while True:
        choice = int(torch.multinomial(weights, 1, generator=generator))
        if choice == 2:
            return _trace(tuple(arcs), capacity), generator.get_state()
        if len(arcs) == capacity:
            return _tail(capacity), generator.get_state()
        arcs.append(choice)


def test_checked_draw_replay_and_cap_zero_distinguish_tail_from_empty():
    scenarios = (
        (0, _tail(0)),
        (0, _trace(capacity=0)),
        (1, _tail(1)),
        (1, _trace((0,))),
        (1, _trace((1,))),
        (2, _trace((0, 0), 2)),
    )
    for capacity, expected in scenarios:
        seed = next(
            seed
            for seed in range(2048)
            if _geometric_draw(seed, capacity)[0] == expected
        )
        _, oracle_state = _geometric_draw(seed, capacity)
        request, parameters = _request(capacity=capacity)
        generator = torch.Generator().manual_seed(seed)
        law = request.bind(parameters, generator=generator)
        module = _module(capacity)
        draw = run_named(module, "drawn", (law,), runtime=_runtime()).value
        assert _key(draw) == expected and torch.equal(
            generator.get_state(), oracle_state
        )
        replay = run_named(module, "replayed", (law,), runtime=_runtime(expected)).value
        assert _key(replay) == expected and torch.equal(
            generator.get_state(), oracle_state
        )
        if capacity == 0:
            mass = 2 * _P if expected[0] else 1 - 2 * _P
            assert _density(module, law, expected).item() == pytest.approx(log(mass))

    original, parameters = _fixture(MARKOV, 0)
    old_law = original.bind(parameters, generator=torch.Generator().manual_seed(73))
    with pytest.raises(ExecutionFailure) as error:
        run_named(_module(0), "mass", (old_law,))
    assert error.value.diagnostic.code == "qiec-run-argument"

    module = _module(1)
    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "drawn" and "foreign handler `draw`" in item.reason
        for item in target.obstacles
    )


def test_canonical_records_and_impossible_tail():
    request, parameters = _request(capacity=2)
    generator = torch.Generator().manual_seed(194)
    law = request.bind(parameters, generator=generator)
    module = _module(2)
    for invalid in (
        (True, 0, -1, (-1, -1)),
        (True, -1, -1, (0, -1)),
        (False, -1, -1, (-1, -1)),
        (False, 0, 0, (-1, 1)),
        (False, 0, 0, (7, -1)),
    ):
        assert _density(module, law, invalid).item() == -torch.inf
    with torch.no_grad():
        parameters[_loop_owner()].fill_(0.0)
    assert _density(module, law, _tail(2)).item() == -torch.inf
    assert _density(module, law, _trace(capacity=2)).item() == 0
    assert run_named(module, "mass", (law,)).value.item() == 0
    assert _key(run_named(module, "drawn", (law,), runtime=_runtime()).value) == _trace(
        capacity=2
    )


def test_extreme_log_tail_and_closure_refusals_precede_any_draw():
    request, parameters = _request(LOG_PROB)
    generator = torch.Generator().manual_seed(73)
    before = generator.get_state().clone()
    for changed, name in (
        (request.with_(max_prefix_cells=2), "max_prefix_cells"),
        (request.with_(max_prefix_work=18), "max_prefix_work"),
        (request.with_(max_closure_cells=3), "max_closure_cells"),
    ):
        with pytest.raises(RelationLimitError, match=name):
            changed.bind(parameters, generator=generator)

    admissible = request.with_(max_prefix_cells=3, max_prefix_work=19)
    with torch.no_grad():
        parameters[_loop_owner()].fill_(-1000)
    law = admissible.bind(parameters, generator=generator)
    score = _density(_module(1), law, _tail(1))
    assert score.item() == pytest.approx(2 * (-1000 + log(2)), abs=1e-12)
    assert torch.autograd.grad(score, parameters[_loop_owner()])[
        0
    ].item() == pytest.approx(2)

    with torch.no_grad():
        parameters[_loop_owner()].fill_(-log(2))
    with pytest.raises(UndecidedClosureError, match="rounding"):
        law.sample()
    with torch.no_grad():
        parameters[_loop_owner()].fill_(0.0)
    with pytest.raises(DivergentClosureError):
        law.sample()
    assert torch.equal(before, generator.get_state())


def test_positive_retained_markov_path_can_have_underflowing_raw_product():
    request, parameters = _request(capacity=2)
    cyclic = request.request
    original = cyclic.request
    zero = ParameterId.derive("capped-path", "zero")
    relation = original.relation
    names = (*relation.weight_names, "zero")
    # Keep a structurally productive dormant loop on the unique positive
    # path. Each of its two actual steps has probability 2**-700.
    tiny, unit, zero_slot = map(names.index, ("loop", "unit", "zero"))
    spec = relation.with_(
        n_states=3,
        arcs=((0, 1, (0,), tiny), (1, 1, (None,), zero_slot), (1, 2, (0,), tiny)),
        initial=((0, unit),),
        final=((2, unit),),
        weight_names=names,
        weight_parameters=(*relation.weight_parameters, zero),
    )
    parameters[zero] = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        parameters[_loop_owner()].fill_(2.0**-700)
    original = original.with_(relation=spec, words=((0, 0),))
    law = request.with_(request=cyclic.with_(request=original)).bind(
        parameters, generator=torch.Generator().manual_seed(45)
    )
    module = _module(2)
    only = (False, 0, 2, (0, 2))
    assert _key(run_named(module, "drawn", (law,), runtime=_runtime()).value) == only
    assert _density(module, law, only).item() == 0
    assert _density(module, law, _tail(2)).item() == -torch.inf


def test_multistate_tail_uses_the_suffix_for_each_prefix_endpoint():
    source = """object Alphabet : FinSet 1
relation branching [algebra=markov, states=2]
    output Alphabet
    initial 0 [weight=unit]
    arc 0 -> 0 [labels=(eps), weight=q]
    arc 0 -> 1 [labels=(eps), weight=p]
    arc 1 -> 1 [labels=(eps), weight=r]
    final 0 [weight=unit]
    final 1 [weight=finish]
"""
    spec = Compiler(parse(source), module_name="capped-branching").relation_specs[
        "branching"
    ]
    request, _ = _request()
    cyclic = request.request
    original = cyclic.request.with_(relation=spec)
    request = request.with_(request=cyclic.with_(request=original))
    p, q, r, finish = Fraction(1, 4), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)
    owners = dict(zip(spec.weight_names, spec.weight_parameters, strict=True))
    values = {"unit": 1, "p": p, "q": q, "r": r, "finish": finish}
    parameters = {
        owner: torch.nn.Parameter(
            torch.tensor(float(values[name]), dtype=torch.float64)
        )
        for name, owner in owners.items()
    }
    law = request.bind(parameters, generator=torch.Generator().manual_seed(5))
    # Sum the two geometric series analytically, then enumerate the two
    # possible destinations of a length-two prefix.
    h1 = finish / (1 - r)
    h0 = (1 + p * h1) / (1 - q)
    h0_prime = h1 / (1 - q)
    tail = q**2 + p * (q + r) * h1 / h0
    derivative = (q + r) * h1 * (h0 - p * h0_prime) / h0**2
    score = _density(_module(1), law, _tail(1))
    assert torch.autograd.grad(score, parameters[owners["p"]])[
        0
    ].item() == pytest.approx(float(derivative / tail))
    assert score.item() == pytest.approx(log(tail))


def test_original_cyclic_path_wire_is_frozen():
    request, _ = _request()
    original = request.request
    assert (
        sha256(dumps(original).encode()).hexdigest()
        == "85a912f860d1e9a8d5a8acaef96e5b768161c4ca1c7ed632a3b53601ab89ca47"
    )
    assert original.result_type == original.request.result_type
