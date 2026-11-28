"""Finite term-valued laws use the checked Random and exact marginal handlers."""

from __future__ import annotations

import math
import random

import didactic.api as dx
import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.effects import replay
from quivers.inference import LatentRegistry, prior_draws, trace
from quivers.inference._likelihood import collect_log_likelihood, row_log_likelihood
from quivers.inference._posterior import collect_site_draws
from quivers.inference._response import response_distribution, response_sample
from quivers.inference.registry import _trace_latent_values
from quivers.qiec.builtins import RANDOM, RANDOM_SAMPLE, TraceRecorder, trace_handler
from quivers.qiec.canonical import sampleable_type
from quivers.qiec import distributions as distribution_runtime
from quivers.qiec.distributions import (
    DistributionError,
    ReferenceBackend,
    RuntimeDistribution,
)
from quivers.qiec.evaluator import (
    RuntimeAttachments,
    RuntimeConstructor,
    RuntimeValidator,
)
from quivers.qiec.execution import (
    ExecutionFailure,
    RuntimeConfiguration,
    RuntimeSelection,
    run_named,
)
from quivers.qiec.foreign import Enumeration
from quivers.qiec.identifiers import AttachmentId
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.program_runtime import run_program, sample_program
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.structured_sampleables import FiniteSampleable
from quivers.qiec.types import STRING, TypeApplication, TypeExpr


SOURCE = """\
object Token : FinSet 1
object Result : Real 1

family Meaning : Type
    constructor word : Meaning
    constructor intensify : Meaning -> Meaning
    constructor impossible : Meaning

instance random : Random

define encode(meaning : Meaning) : Real !{} =
    case meaning motive => Real
        word =>
            return 0.0
        intensify(child) =>
            let code <- encode(child)
            return code + 2.0
        impossible =>
            return -10.0

define encode_sample(law : Sampleable[Meaning]) : Real !{random} =
    let meaning <- perform random.sample[Meaning](site("meaning"), law)
    encode(meaning)

program inspect(law : Sampleable[Meaning]) : Token -> Result
    let code <- encode_sample(law)
    observe response <- Normal(loc=code, scale=1.0)
    return code

handler remember for Random : Real -> Real [coverage=partial, forwards=unknown, implementation=foreign]
    sample resumes aff

define record_sample(law : Sampleable[Meaning]) : Real !{random} =
    handle random with remember in
        encode_sample(law)

program recorded(law : Sampleable[Meaning]) : Token -> Result
    let code <- record_sample(law)
    observe response <- Normal(loc=code, scale=1.0)
    return code

handler exact for Random : (Real * LogWeight) -> LogWeight [coverage=total, implementation=foreign]
    sample resumes omega

define integrate(law : Sampleable[Meaning], response : Real) : LogWeight !{} =
    handle random with exact in
        let code <- encode_sample(law)
        return (code, log_prob(Normal(code, 1.0), response))

define density(law : Sampleable[Meaning], term : Meaning) : LogWeight !{} =
    return log_prob(law, term)

define mass(law : Sampleable[Meaning]) : LogWeight !{} =
    return log_mass(law)

export inspect
export recorded
"""


class _RecordingProvider:
    name = "finite-recording"

    def __init__(self, recorder: TraceRecorder) -> None:
        self.recorder = recorder

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        definition = next(item for item in module.handlers if item.name == "remember")
        runtime = trace_handler(
            RANDOM,
            (RANDOM_SAMPLE,),
            self.recorder,
            validators={
                RANDOM_SAMPLE: lambda value: isinstance(value, RuntimeConstructor)
            },
            answer_type=definition.input_type,
        )
        attachments.bind_handler(runtime.with_(definition=definition))

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        return None


class _LawProvider:
    name = "finite-law"

    def __init__(self, law: FiniteSampleable) -> None:
        self.law = law

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        attachments.bind_value(
            AttachmentId.derive(module.module, self.name),
            self.law,
            sampleable_type(self.law.element),
            duplicable=True,
        )

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        return None


_RUNTIME = RuntimeConfiguration(
    selections=(
        RuntimeSelection(
            provider="core", options={"handlers": {"exact": {"kind": "declared"}}}
        ),
    ),
)


def _module() -> QiecModule:
    module = load_source(SOURCE).qiec
    assert isinstance(module, QiecModule)
    module = module.with_(
        handlers=tuple(
            handler.with_(foreign=Enumeration()) if handler.name == "exact" else handler
            for handler in module.handlers
        )
    )
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    return restored


def _law(
    module: QiecModule, weights: torch.Tensor
) -> tuple[FiniteSampleable, RuntimeConstructor, RuntimeConstructor]:
    family = next(item for item in module.families if item.name == "Meaning")
    element = TypeApplication(constructor=family.type_constructor, arguments=())
    leaf, parent, absent = family.constructors
    word = RuntimeConstructor(
        constructor=leaf, static_arguments=(), fields=(), result_type=element
    )
    nested = RuntimeConstructor(
        constructor=parent, static_arguments=(), fields=(word,), result_type=element
    )
    duplicate = nested.with_(fields=(word.with_(fields=()),))
    zero = RuntimeConstructor(
        constructor=absent, static_arguments=(), fields=(), result_type=element
    )
    law = FiniteSampleable(
        element=element,
        values=(word, nested, duplicate, zero),
        log_weights=weights,
        registry=validate_module(module),
    )
    return law, nested, zero


def test_structured_measure_draws_encodes_conditions_and_replays_in_one_program() -> (
    None
):
    module = _module()
    weights = torch.tensor(
        [0.2, -0.3, 1.2, -torch.inf], dtype=torch.float64, requires_grad=True
    )
    law, nested, zero = _law(module, weights)
    y = 1.7

    def normal(mean: float) -> float:
        return -math.log(2 * math.pi) / 2 - (y - mean) ** 2 / 2

    data = {"law": law, "response": y}

    recorder = TraceRecorder()
    recorded = run_program(
        module,
        "recorded",
        data=data,
        sites={"meaning": nested},
        providers=(_RecordingProvider(recorder),),
    )
    (event,) = recorder.events
    assert event.operation == RANDOM_SAMPLE
    assert event.result is nested
    assert event.arguments == ("meaning", law)
    assert recorded.value == 2.0

    actual = run_program(module, "inspect", data=data, sites={"meaning": nested})

    conditioned = sample_program(
        module, "inspect", data=data, sites={"meaning": nested}
    )
    expected = weights[1:3].logsumexp(0) - weights[:3].logsumexp(0) + normal(2.0)
    torch.testing.assert_close(actual.log_joint, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(conditioned.log_joint, expected, atol=1e-12, rtol=1e-12)
    assert actual.value == conditioned.value == 2.0
    actual_grad = torch.autograd.grad(actual.log_joint, weights, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, weights)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)

    def addresses(run):
        return tuple(
            event.detail["address"]
            for event in run.result.trace
            if event.event == "operation.requested"
            and event.detail["operation"] == str(RANDOM_SAMPLE)
        )

    draw = sample_program(module, "inspect", data=data)
    assert draw.value in (0.0, 2.0)
    assert addresses(actual) == addresses(conditioned) == addresses(draw)
    assert len(addresses(draw)) == 1
    assert torch.isneginf(law.log_prob(zero))
    with pytest.raises(ExecutionFailure, match="does not inhabit"):
        run_program(module, "inspect", data=data, sites={"meaning": torch.tensor(2)})


def test_exact_structural_enumeration_accumulates_support_and_gradients_each_run() -> (
    None
):
    module = _module()
    weights = torch.tensor(
        [-0.3, 0.2, 1.1, -torch.inf], dtype=torch.float64, requires_grad=True
    )
    law, nested, zero = _law(module, weights)
    y = 0.2
    log_likelihoods = weights.new_tensor(
        [-math.log(2 * math.pi) / 2 - (y - mean) ** 2 / 2 for mean in (0, 2, 2)]
    )
    with pytest.raises(ExecutionFailure, match="captures nonduplicable local 'law'"):
        run_named(module, "integrate", arguments=(law, y), runtime=_RUNTIME)
    runtime = _RUNTIME.with_(providers=(_LawProvider(law),))
    for _ in range(3):
        support = law.weighted_support()
        assert len(support) == 2 and support[1][0] == nested
        assert all(value != zero for value, _ in support)
        actual = run_named(module, "integrate", arguments=(law, y), runtime=runtime)
        expected = (weights[:3] + log_likelihoods).logsumexp(0) - weights[:3].logsumexp(
            0
        )
        torch.testing.assert_close(actual.value, expected, atol=1e-12, rtol=1e-12)
        gradient = torch.autograd.grad(actual.value, weights)[0]
        expected_gradient = torch.autograd.grad(expected, weights)[0]
        torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
        with torch.no_grad():
            weights.add_(0.25 * gradient)


def test_finite_law_validates_support_and_centers_before_normalizing() -> None:
    module = _module()
    weights = torch.tensor([1e308, 1e308, -torch.inf, -torch.inf], dtype=torch.float64)
    law, nested, absent = _law(module, weights)

    expected = weights.new_tensor(-math.log(2.0))
    torch.testing.assert_close(law.log_prob(nested), expected)
    torch.testing.assert_close(
        run_named(module, "density", (law, nested)).value, expected
    )
    torch.testing.assert_close(
        run_named(module, "mass", (law,)).value, weights.new_zeros(())
    )
    torch.testing.assert_close(
        torch.stack([weight for _, weight in law.weighted_support()]).exp().sum(),
        weights.new_tensor(1.0),
    )
    with pytest.raises(
        dx.ValidationError, match="invalid finite sampleable constructor point"
    ):
        law.with_(values=(nested.with_(fields=()), *law.values[1:]))
    forged = nested.with_(fields=(weights.new_tensor(2.0),))
    with pytest.raises(DistributionError, match="invalid finite sampleable point"):
        law.log_prob(forged)
    with pytest.raises(dx.ValidationError, match="without positive mass"):
        law.with_(log_weights=weights.new_full((4,), -torch.inf))
    with pytest.raises(dx.ValidationError, match="NaN"):
        law.with_(log_weights=weights.new_tensor([1.0, 0.0, 2.0, torch.nan]))
    with pytest.raises(DistributionError, match="out of range"):
        law.with_(log_weights=weights.new_tensor([1e308, -1e308, 0.0, 0.0])).log_prob(
            absent
        )


def test_public_host_entry_rejects_law_at_the_wrong_nominal_carrier() -> None:
    module = _module()
    other = FiniteSampleable(
        element=STRING,
        values=("word",),
        log_weights=torch.tensor([0.0]),
        registry=validate_module(module),
    )
    with pytest.raises(ExecutionFailure) as errors:
        run_named(module, "mass", arguments=(other,))
    assert errors.value.diagnostic.code == "qiec-run-argument"


def test_common_trace_preserves_typed_sites_through_condition_and_replay() -> None:
    compiled = load_source(SOURCE.replace("export recorded\n", ""))
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    weights = torch.tensor(
        [0.2, -0.3, 1.2, -torch.inf], dtype=torch.float64, requires_grad=True
    )
    law, nested, _zero = _law(module, weights)
    x = weights.new_zeros((1, 1))
    observed = weights.new_tensor(1.7)
    data = {"law": law, "response": observed}

    sampled = trace(compiled.morphism, x, data)
    point = sampled.sites["meaning"]
    assert point.is_structured and not point.is_observed
    assert point.sampleable is law and point.morphism is None
    assert point.value in (law.values[0], nested)
    assert point.log_prob.shape == (1,)
    assert sampled.sites["response"].is_observed
    assert not sampled.sites["response"].is_structured
    with pytest.raises(
        ValueError, match="structured latent 'meaning'.*continuous latent registry"
    ):
        LatentRegistry.from_trace(compiled.morphism, sampled)
    with pytest.raises(
        ValueError, match="tensor draws cannot retain structured site 'meaning'"
    ):
        _trace_latent_values(sampled)
    unflattened = LatentRegistry.from_trace(
        compiled.morphism, sampled, exclude={"meaning"}
    )
    assert unflattened.names == ()
    assert unflattened.observed_names == {"response"}

    clamped = trace(compiled.morphism, x, {**data, "meaning": nested})
    assert clamped.sites["meaning"].is_observed
    assert clamped.sites["meaning"].value is nested
    assert clamped.sites["meaning"].address == point.address
    registry = LatentRegistry.from_trace(compiled.morphism, clamped)
    assert registry.names == ()
    assert registry.observed_names == {"meaning", "response"}
    assert isinstance(clamped.log_joint, torch.Tensor)
    expected = (
        weights[1:3].logsumexp(0)
        - weights[:3].logsumexp(0)
        - (observed - 2.0).square() / 2
        - math.log(2 * math.pi) / 2
    )
    torch.testing.assert_close(
        clamped.log_joint, expected.reshape(1), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        torch.autograd.grad(clamped.log_joint.sum(), weights)[0],
        torch.autograd.grad(expected, weights)[0],
        atol=1e-12,
        rtol=1e-12,
    )

    new_weights = weights.new_tensor([-0.5, 1.4, 0.7, -torch.inf])
    new_law = law.with_(log_weights=new_weights)
    with replay(clamped):
        replayed = trace(compiled.morphism, x, {**data, "law": new_law})
    site = replayed.sites["meaning"]
    assert site.value is nested and site.sampleable is new_law
    assert site.address == point.address
    expected_replay = new_weights[1:3].logsumexp(0) - new_weights[:3].logsumexp(0)
    torch.testing.assert_close(site.log_prob, expected_replay.reshape(1))


def test_structured_draws_are_not_vectorized_into_one_shared_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = load_source(SOURCE.replace("export recorded\n", ""))
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    weights = torch.tensor([0.2, -0.3, 1.2, -torch.inf], dtype=torch.float64)
    law, nested, _zero = _law(module, weights)
    x = weights.new_zeros((2, 1))
    data = {"law": law, "response": weights.new_tensor(1.7)}
    with pytest.raises(ValueError, match="structured trace site.*one independent run"):
        trace(compiled.morphism, x, data)

    values = iter((law.values[0], nested))
    monkeypatch.setattr(FiniteSampleable, "sample", lambda _law, **_: next(values))
    actual = compiled.morphism.rsample(x, observations=data)
    torch.testing.assert_close(actual, weights.new_tensor([0.0, 2.0]))
    assert compiled.morphism.work.iterations == 2
    with pytest.raises(StopIteration):
        next(values)


def test_seeded_public_program_draws_use_reference_rng_and_replay_does_not_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random()
    monkeypatch.setattr(distribution_runtime, "_RNG", rng)
    compiled = load_source(SOURCE.replace("export recorded\n", ""))
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    weights = torch.tensor([0.2, -0.3, 1.2, -torch.inf], dtype=torch.float64)
    law, nested, _zero = _law(module, weights)
    data = {"law": law, "response": 1.7}
    assert compiled.run("inspect", data=data, seed=1).value == 0.0
    assert compiled.run("inspect", data=data, seed=2).value == 2.0
    state = rng.getstate()
    assert compiled.run("inspect", data=data, sites={"meaning": nested}).value == 2.0
    assert rng.getstate() == state

    class SelectRight(ReferenceBackend):
        def sample(self, law: RuntimeDistribution, rng: random.Random) -> object:
            del rng
            assert law.family == "Categorical"
            return torch.tensor(1, dtype=torch.int64, device=law.numbers.device)

    overridden = law.with_(backend=SelectRight())
    assert (
        compiled.run("inspect", data={**data, "law": overridden}, seed=1).value == 2.0
    )


def test_seeded_public_prior_uses_torch_and_refuses_structured_tensor_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = load_source(SOURCE.replace("export recorded\n", ""))
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    weights = torch.tensor([0.2, -0.3, 1.2, -torch.inf], dtype=torch.float64)
    law, nested, _zero = _law(module, weights)
    x = weights.new_zeros((1, 1))
    data = {"law": law, "response": weights.new_tensor(1.7)}
    real_sample = FiniteSampleable.sample
    draws = []

    def sample(law, *, backend=None):
        assert isinstance(backend, TensorBackend)
        result = real_sample(law, backend=backend)
        draws.append(result)
        return result

    monkeypatch.setattr(FiniteSampleable, "sample", sample)
    cpu_state = torch.random.get_rng_state().clone()
    for _ in range(2):
        with pytest.raises(
            ValueError, match="tensor draws cannot retain structured site 'meaning'"
        ):
            prior_draws(compiled.morphism, x, observations=data, num_samples=1, seed=2)
        assert torch.equal(torch.random.get_rng_state(), cpu_state)
    assert len(draws) == 2 and draws[0] is draws[1]
    assert draws[0] in (law.values[0], nested)
    clamped = trace(compiled.morphism, x, {**data, "meaning": nested})
    with replay(clamped):
        replayed = trace(compiled.morphism, x, data)
    assert replayed.sites["meaning"].value is nested
    assert len(draws) == 2
    assert torch.equal(torch.random.get_rng_state(), cpu_state)


def test_structured_observation_likelihood_is_raw_and_tensor_outputs_refuse() -> None:
    compiled = load_source(SOURCE.replace("export recorded\n", ""))
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    weights = torch.tensor(
        [0.2, -0.3, 1.2, -torch.inf], dtype=torch.float64, requires_grad=True
    )
    law, nested, _zero = _law(module, weights)
    recorded = trace(
        compiled.morphism,
        weights.new_zeros((1, 1)),
        {"law": law, "meaning": nested, "response": weights.new_tensor(1.7)},
    )
    site = recorded.sites["meaning"]
    # A score handler may rescale the recorded contribution. The raw
    # likelihood still reads the finite law at the actual typed observation.
    transformed = recorded.with_(
        sites={**recorded.sites, "meaning": site.with_(log_prob=2 * site.log_prob)}
    )
    raw = row_log_likelihood(transformed, {"meaning"})["meaning"]
    expected = weights[1:3].logsumexp(0) - weights[:3].logsumexp(0)
    assert raw.shape == (1,)
    torch.testing.assert_close(raw, expected.reshape(1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(raw.sum(), weights)[0],
        torch.autograd.grad(expected, weights)[0],
        atol=1e-12,
        rtol=1e-12,
    )
    joint = collect_log_likelihood([transformed], {"meaning"}, (1, 1))["meaning"]
    torch.testing.assert_close(joint, expected.reshape(1, 1, 1))
    with pytest.raises(
        TypeError, match="posterior site 'meaning'.*no tensor draw layout"
    ):
        collect_site_draws(({"meaning": site.value},), (1, 1))
    with pytest.raises(
        TypeError, match="structured response 'meaning'.*no tensor sample layout"
    ):
        response_sample(recorded, "meaning")
    with pytest.raises(
        TypeError, match="structured response 'meaning'.*no tensor distribution"
    ):
        response_distribution(recorded, "meaning")
