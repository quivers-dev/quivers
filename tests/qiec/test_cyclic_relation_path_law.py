"""Exact full-posterior scores and checked bounded attempts on cyclic walks."""

from fractions import Fraction
from hashlib import sha256
from itertools import product
from math import log

import pytest
import torch
from torch import nn

from quivers.core.algebras import DivergentClosureError, LOG_PROB, MARKOV
from quivers.core.relations import ProductiveEpsilonCycleError, RelationLimitError
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    Enumeration,
    ExecutionFailure,
    FixedCyclicRelationPathSampleRequest,
    FixedRelationPathSampleRequest,
    ParameterId,
    RuntimeConfiguration,
    RuntimeSelection,
    dumps,
    loads,
    run_named,
)
from quivers.qiec.builtins import _support
from quivers.qiec.evaluator import InvalidHandlerError
from tests.qiec.test_relation_path_law import (
    _module,
    _runtime,
    _sample_request,
    _trace_key,
)

_MODEL = "cyclic_path_observation_test"


def _loop_owner():
    return ParameterId.derive(_MODEL, "relation_weights", "loop")


def _fixture(algebra, capacity=4):
    corpus = f"""object Alphabet : FinSet 1
relation loop_machine [algebra={"markov" if algebra is MARKOV else "logprob"}, states=1]
    output Alphabet
    initial 0 [weight=unit]
    arc 0 -> 0 [labels=(eps,), weight=loop]
    arc 0 -> 0 [labels=(eps,), weight=loop]
    final 0 [weight=unit]
"""
    compiler = Compiler(parse(corpus), module_name=_MODEL)
    compiler.compile()
    spec = compiler.relation_specs["loop_machine"]
    source = FixedRelationPathSampleRequest(
        relation=spec, words=((),), trace_capacity=capacity, max_states=4, max_work=8
    )
    unit = ParameterId.derive(_MODEL, "relation_weights", "unit")
    values = {
        unit: torch.tensor(1.0 if algebra is MARKOV else 0.0, dtype=torch.float64),
        _loop_owner(): nn.Parameter(
            torch.tensor(0.25 if algebra is MARKOV else log(0.25), dtype=torch.float64)
        ),
    }
    return FixedCyclicRelationPathSampleRequest(
        request=source, max_closure_size=8
    ), values


def _score(module, law, original_arcs, capacity):
    trace = (0, 0, (*original_arcs, *(-1,) * (capacity - len(original_arcs))))
    value = run_named(module, "density", (law, trace)).value
    assert isinstance(value, torch.Tensor)
    return value


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_live_original_cyclic_scores_use_full_posterior_and_retained_multiplicity(
    algebra,
):
    source, parameters = _fixture(algebra)
    wire = dumps(source)
    assert isinstance(source := loads(wire), FixedCyclicRelationPathSampleRequest)
    assert dumps(source) == wire
    assert source.result_type == source.request.result_type
    assert source.element_type == source.request.element_type
    assert source.request.relation.arcs[0] == source.request.relation.arcs[1]
    generator = torch.Generator().manual_seed(11)
    law = source.bind(parameters, generator=generator)
    assert law.has_productive_cycle and law.element == source.element_type
    module = loads(dumps(_module()))
    loop = parameters[_loop_owner()]
    for p in (Fraction(1, 4), Fraction(1, 8)):
        with torch.no_grad():
            loop.fill_(float(p) if algebra is MARKOV else log(p))
        seen = Fraction(0)
        scored = 0.0
        for length in range(source.request.trace_capacity + 1):
            for arcs in product(range(2), repeat=length):
                probability = (1 - 2 * p) * p**length
                actual = _score(module, law, arcs, source.request.trace_capacity)
                assert actual.item() == pytest.approx(log(probability))
                seen += probability
                scored += actual.exp().item()
        assert seen == 1 - (2 * p) ** (source.request.trace_capacity + 1)
        assert scored == pytest.approx(float(seen))
        score = _score(module, law, (0, 1, 0), source.request.trace_capacity)
        observed = torch.autograd.grad(score, loop)[0]
        derivative = 3 / p - 2 / (1 - 2 * p)
        if algebra is LOG_PROB:
            derivative *= p
        assert observed.item() == pytest.approx(float(derivative))
        invalid = (0, 0, (1, -1, 0, -1))
        assert run_named(module, "density", (law, invalid)).value.item() == -torch.inf


def _oracle_sample(seed, max_steps):
    """Use the rational one-state posterior choices, with no query planner."""
    generator = torch.Generator().manual_seed(seed)
    one = torch.tensor((1.0,), dtype=torch.float64)
    torch.multinomial(one, 1, generator=generator)
    conditional = torch.tensor((0.25, 0.25, 0.5), dtype=torch.float64)
    trace = []
    while True:
        choice = int(torch.multinomial(conditional, 1, generator=generator))
        if choice == 2:
            return tuple(trace)
        if len(trace) == max_steps:
            return None
        trace.append(choice)


def test_native_random_draw_and_replay_preserve_parallel_occurrence_ids():
    source, parameters = _fixture(MARKOV)
    module = _module()
    output = []
    for arc in (0, 1):
        seed = next(seed for seed in range(128) if _oracle_sample(seed, 4) == (arc,))
        generator = torch.Generator().manual_seed(seed)
        law = source.bind(parameters, generator=generator)
        start = generator.get_state().clone()
        density = _score(module, law, (arc,), 4)
        assert density.item() == pytest.approx(log(Fraction(1, 8)))
        drawn = _trace_key(run_named(module, "drawn", (law,), runtime=_runtime()).value)
        assert drawn == (0, 0, (arc, -1, -1, -1))
        after = generator.get_state().clone()
        assert not torch.equal(start, after)
        replay = _trace_key(
            run_named(module, "replayed", (law,), runtime=_runtime(drawn)).value
        )
        assert replay == drawn and torch.equal(after, generator.get_state())
        output.append(drawn)
    # Both complete sample paths emit the empty word and have the same
    # boundary states. The original arc occurrence still distinguishes them.
    assert output[0][:2] == output[1][:2] and output[0][2] != output[1][2]


def _enumeration_module(capacity):
    trace = f"(Int * Int * Tensor[Int]([{capacity}]))"
    source = f"""instance random : Random
handler exact for Random : LogWeight -> LogWeight [coverage=total, implementation=foreign]
    sample resumes omega

define expectation(law : Sampleable[{trace}]) : LogWeight !{{}} =
    handle random with exact in
        let value <- perform random.sample[{trace}](site("path"), law)
        return log_prob(law, value)
"""
    compiler = Compiler(parse(source), module_name="cyclic_path_enumeration_refusal")
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    return module.with_(
        handlers=(
            module.handlers[0].with_(foreign=Enumeration(reduction="log_expectation")),
        )
    )


def test_cycle_refuses_whole_carrier_mass_and_exact_enumeration_without_a_draw():
    source, parameters = _fixture(MARKOV)
    generator = torch.Generator().manual_seed(4)
    law = source.bind(parameters, generator=generator)
    before = generator.get_state().clone()
    with pytest.raises(
        ExecutionFailure, match="log_mass of a cyclic path law is unavailable"
    ):
        run_named(_module(), "mass", (law,))
    runtime = RuntimeConfiguration(
        selections=(
            RuntimeSelection(
                provider="core", options={"handlers": {"exact": {"kind": "declared"}}}
            ),
        )
    )
    with pytest.raises(ExecutionFailure, match="nonduplicable local 'law'"):
        run_named(_enumeration_module(4), "expectation", (law,), runtime=runtime)
    with pytest.raises(
        InvalidHandlerError, match="cyclic path law has no finite support to enumerate"
    ):
        _support(law)
    assert torch.equal(before, generator.get_state())


def test_cyclic_bound_raises_in_native_draw_without_retry_or_partial_walk():
    source, parameters = _fixture(MARKOV, capacity=0)
    seed = next(seed for seed in range(128) if _oracle_sample(seed, 0) is None)
    generator = torch.Generator().manual_seed(seed)
    law = source.bind(parameters, generator=generator)
    module = _module(capacity=0)
    initial = generator.get_state().clone()
    with pytest.raises(RelationLimitError, match="sampled path exceeds max_steps"):
        law.sample()
    attempted = generator.get_state().clone()
    assert not torch.equal(initial, attempted)
    generator.set_state(initial)
    with pytest.raises(ExecutionFailure, match="sampled path exceeds max_steps"):
        run_named(module, "drawn", (law,), runtime=_runtime())
    assert torch.equal(attempted, generator.get_state())
    assert _score(module, law, (), 0).item() == pytest.approx(log(Fraction(1, 2)))
    stop_seed = next(seed for seed in range(128) if _oracle_sample(seed, 0) == ())
    generator.manual_seed(stop_seed)
    returned = run_named(module, "drawn", (law,), runtime=_runtime()).value
    assert _trace_key(returned) == (0, 0, ())


def test_legacy_path_bytes_closure_budget_and_divergence():
    for algebra, expected in (
        (MARKOV, "2a6cad567698308b112ec49d6a4a9a75fc3cd98fea9f4cb10fdf8eadf55bd72b"),
        (LOG_PROB, "4c9230c08687d7c86b6553fdfdf7369053f21fe2ab6f0e9bca06c050cfb2af88"),
    ):
        legacy, _, _ = _sample_request(algebra)
        assert sha256(dumps(legacy).encode()).hexdigest() == expected
    source, parameters = _fixture(MARKOV)
    generator = torch.Generator().manual_seed(11)
    before = generator.get_state().clone()
    with pytest.raises(ProductiveEpsilonCycleError):
        source.request.bind(parameters, generator=generator)
    with pytest.raises(RelationLimitError, match="max_closure_size"):
        source.with_(max_closure_size=0).bind(parameters, generator=generator)
    with torch.no_grad():
        parameters[_loop_owner()].fill_(0.75)
    with pytest.raises(DivergentClosureError):
        source.bind(parameters, generator=generator)
    assert torch.equal(before, generator.get_state())
