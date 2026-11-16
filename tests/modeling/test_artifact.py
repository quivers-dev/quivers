"""Fitted artifacts preserve checked models without fitting or sampling on read."""

import hashlib
import json

import numpy as np
import pytest
import torch

from quivers.inference import (
    AutoNormalGuide,
    MCMC,
    MCMCResult,
    NUTSKernel,
    PriorDraws,
    SVI,
)
from quivers.modeling import (
    ArtifactFactory,
    MCMCMethod,
    Model,
    ModelFit,
    VariationalMethod,
    load_fit,
    save_fit,
)
from quivers.modeling._artifact_codec import _Codec, _module_state, _restore_module

SOURCE = """object Row : FinSet _
object Out : Real 1
program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""


def _data():
    return {"x": torch.tensor([0.5, -1.0, 2.0]), "y": torch.tensor([-0.25, 0.5, 1.5])}


def _record(samples, count=3):
    return MCMCResult(
        samples=samples,
        log_densities=torch.arange(2 * count, dtype=torch.float64).reshape(2, count),
        acceptance_rates=torch.tensor([0.8, 0.9]),
        divergence_counts=torch.tensor([1, 0]),
        num_warmup=7,
        num_samples=count,
        sample_stats={"accepted": torch.ones(2, count)},
        stopped_by=("draws", "evaluations"),
        num_draws=torch.tensor([count, count + 1]),
    )


def _forbid(*args, **kwargs):
    raise AssertionError("artifact restoration invoked stochastic fitting or sampling")


def _no_inference(monkeypatch):
    from quivers.effects.sites import TorchSampleable

    monkeypatch.setattr(MCMC, "run", _forbid)
    monkeypatch.setattr(SVI, "run", _forbid)
    monkeypatch.setattr(AutoNormalGuide, "rsample", _forbid)
    monkeypatch.setattr(TorchSampleable, "sample", _forbid)
    monkeypatch.setattr(TorchSampleable, "rsample", _forbid)


def test_normal_guide_restores_exact_state_joint_and_fresh_scores_without_draws(
    tmp_path, monkeypatch
):
    fitted = Model(SOURCE).fit(_data(), method=VariationalMethod(num_steps=0, seed=11))
    with torch.no_grad():
        fitted.posterior.loc_b.fill_(0.375)
        fitted.posterior.log_scale_b.fill_(-0.8)
    path = tmp_path / "fit.pt"
    before = torch.random.get_rng_state().clone()
    save_fit(fitted, path)
    _no_inference(monkeypatch)
    restored = load_fit(path)
    assert torch.equal(before, torch.random.get_rng_state())
    assert type(restored.posterior) is AutoNormalGuide
    assert restored.posterior is not fitted.posterior
    assert restored.posterior.registry.names == ("b",)
    for name, value in fitted.posterior.state_dict().items():
        assert torch.equal(value, restored.posterior.state_dict()[name])
    b = torch.tensor([0.4], requires_grad=True)
    score = restored.model.trace({"b": b}).log_joint
    expected = (
        torch.distributions.Normal(0.0, 1.0).log_prob(b)
        + torch.distributions.Normal(b * _data()["x"], 1.0).log_prob(_data()["y"]).sum()
    )
    torch.testing.assert_close(score, expected)
    torch.testing.assert_close(
        torch.autograd.grad(score.sum(), b)[0],
        torch.autograd.grad(expected.sum(), b)[0],
    )
    assert restored.refit_recipe.method.num_steps == 0
    assert restored.refit_recipe.method.seed == 11


def test_mcmc_preserves_all_retained_tensors_counts_and_raw_likelihood(
    tmp_path, monkeypatch
):
    b = torch.tensor([[-0.5, 0.0, 0.7], [0.25, -0.1, 1.0]], dtype=torch.float64)
    bound = Model(SOURCE).bind(_data())
    record = _record({"b": b})
    fitted = ModelFit(model=bound, posterior=record, losses=(2.0, 1.0))
    path = tmp_path / "mcmc.pt"
    save_fit(fitted, path)
    _no_inference(monkeypatch)
    restored = load_fit(path)
    for name in record.__field_specs__:
        old, new = getattr(record, name), getattr(restored.posterior, name)
        if isinstance(old, torch.Tensor):
            assert old.dtype == new.dtype and torch.equal(old, new)
        elif isinstance(old, dict):
            assert old.keys() == new.keys()
            assert all(torch.equal(value, new[key]) for key, value in old.items())
        else:
            assert old == new
    expected = torch.distributions.Normal(b[..., None] * _data()["x"], 1.0).log_prob(
        _data()["y"]
    )
    torch.testing.assert_close(restored.log_likelihood()["y"], expected)
    assert restored.losses == (2.0, 1.0) and restored.refit_recipe is None


def test_prior_draws_keep_boolean_and_integer_carriers_without_float_conversion(
    tmp_path, monkeypatch
):
    source = """object One:FinSet 1
program prior(y:Real,n:Int):One->One
    sample flag <- Bernoulli(probs=0.3)
    sample count <- Poisson(rate=2.0)
    observe y <- Normal(loc=0.0,scale=1.0)
    return y
export prior
"""
    bound = Model(source).bind({"y": torch.tensor(0.5), "n": 2**63 - 1})
    prior = PriorDraws(
        samples={
            "flag": torch.tensor([[False, True]]),
            "count": torch.tensor([[0, 2**53]], dtype=torch.int64),
        },
        num_samples=2,
    )
    fitted = ModelFit(model=bound, posterior=prior, prior=prior, sample_prior="only")
    path = tmp_path / "prior.pt"
    save_fit(fitted, path)
    _no_inference(monkeypatch)
    restored = load_fit(path)
    assert restored.model.data["n"] == 2**63 - 1
    for name, value in prior.samples.items():
        assert value.dtype == restored.posterior.samples[name].dtype
        assert torch.equal(value, restored.posterior.samples[name])
        assert torch.equal(value, restored.prior.samples[name])
    assert restored.sample_prior == "only"


def test_scan_guide_reconstructs_every_dynamic_address_without_sampling(
    tmp_path, monkeypatch
):
    from tests.dsl.test_scan_program_cells import STATE_SPACE

    bound = Model(STATE_SPACE).bind(
        {"o": torch.zeros(3, 2)}, input=torch.zeros(1, 3, 2), seed=3
    )
    guide = AutoNormalGuide(
        bound.require_tensor_program(),
        observed_names={"o"},
        x=bound.input,
        observations=bound.tensor_data(),
    )
    with torch.no_grad():
        for index, parameter in enumerate(guide.parameters()):
            parameter.fill_(0.03 * index)
    fitted = ModelFit(model=bound, posterior=guide)
    path = tmp_path / "scan.pt"
    save_fit(fitted, path)
    _no_inference(monkeypatch)
    restored = load_fit(path)
    assert restored.posterior.registry.names == guide.registry.names
    assert set(guide.registry.names) == {"s@0", "s@1", "s@2"}
    for name, site in guide.registry.sites.items():
        replacement = restored.posterior.registry.sites[name]
        assert site.constrained_shape == replacement.constrained_shape
        assert site.event_shape == replacement.event_shape
    assert all(
        torch.equal(value, restored.posterior.state_dict()[name])
        for name, value in guide.state_dict().items()
    )
    anchors = {name: torch.zeros(1, 3) for name in guide.registry.names}
    torch.testing.assert_close(
        restored.model.trace(anchors).log_joint, fitted.model.trace(anchors).log_joint
    )


def test_deterministic_module_states_restore_all_heads_dtype_aliases_and_gradients(
    tmp_path, monkeypatch
):
    source = """object Unit:Real 1
object Response:Real 2
morphism emission:Unit->Response ~ Normal
program model:Unit->Response
    observe y <- emission
    return y
export model
"""
    bound = Model(source).bind(
        {"y": torch.tensor([0.5, -0.2], dtype=torch.float64)},
        input=torch.tensor([[0.25]], dtype=torch.float64),
    )
    program = bound.require_tensor_program().double()
    with torch.no_grad():
        for index, parameter in enumerate(program.parameters()):
            parameter.fill_(0.01 * (index + 1))
    program.eval()
    fitted = ModelFit(model=bound, posterior=PriorDraws(samples={}, num_samples=3))
    path = tmp_path / "heads.pt"
    save_fit(fitted, path)
    _no_inference(monkeypatch)
    restored = load_fit(path)
    replacement = restored.model.require_tensor_program()
    assert not replacement.training
    assert all(
        torch.equal(value, replacement.state_dict()[name])
        for name, value in program.state_dict().items()
    )
    assert all(
        parameter.dtype == torch.float64 for parameter in replacement.parameters()
    )
    first = bound.trace().log_joint.sum()
    second = restored.model.trace().log_joint.sum()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    old_grads = torch.autograd.grad(first, tuple(program.parameters()))
    new_grads = torch.autograd.grad(second, tuple(replacement.parameters()))
    for old, new in zip(old_grads, new_grads, strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_module_state_retains_shared_parameter_and_buffer_identity():
    module = torch.nn.Module()
    module.left = torch.nn.Linear(2, 2)
    module.right = torch.nn.Linear(2, 2)
    module.right.weight = module.left.weight
    module.register_buffer("a", torch.ones(2))
    module.register_buffer("b", module.a)
    state = _module_state(module)
    clone = torch.nn.Module()
    clone.left = torch.nn.Linear(2, 2)
    clone.right = torch.nn.Linear(2, 2)
    clone.right.weight = clone.left.weight
    clone.register_buffer("a", torch.zeros(2))
    clone.register_buffer("b", clone.a)
    _restore_module(clone, state)
    assert clone.left.weight is clone.right.weight
    assert clone.a is clone.b


def test_saved_recipe_retains_original_data_policy_method_and_prior_seeds(tmp_path):
    fitted = Model(SOURCE).fit(
        _data(),
        method=VariationalMethod(num_steps=0),
        sample_prior="yes",
        num_prior_samples=2,
        prior_seed=17,
        snapshot_policy="reference",
    )
    path = tmp_path / "refit.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert restored.refit_recipe.snapshot_policy == "reference"
    assert restored.refit_recipe.provenance["prior_seed"] == 17
    assert restored.refit_recipe.provenance["num_prior_samples"] == 2
    updated = restored.update(method=VariationalMethod(num_steps=0, seed=7))
    assert updated.prior.num_samples == 2
    assert updated.refit_recipe.provenance["prior_seed"] == 17
    expected = fitted.update(method=VariationalMethod(num_steps=0, seed=7))
    assert all(
        torch.equal(value, expected.prior.samples[name])
        for name, value in updated.prior.samples.items()
    )


def test_mcmc_refit_configuration_roundtrips_without_kernel_execution(
    tmp_path, monkeypatch
):
    from quivers.modeling._refit import source_recipe

    bound = Model(SOURCE).bind(_data())
    method = MCMCMethod(
        sampler=MCMC(
            NUTSKernel(step_size=0.25, max_tree_depth=3, adaptation="windowed"),
            num_warmup=5,
            num_samples=3,
            num_chains=2,
            seed=31,
            initial_values={"b": 0.1},
        )
    )
    fitted = ModelFit(
        model=bound,
        posterior=_record({"b": torch.zeros(2, 3)}),
        refit_recipe=source_recipe(
            bound,
            method,
            snapshot_policy="clone",
            sample_prior="no",
            num_prior_samples=4,
            prior_seed=5,
        ),
    )
    path = tmp_path / "method.pt"
    save_fit(fitted, path)
    _no_inference(monkeypatch)
    restored = load_fit(path)
    sampler = restored.refit_recipe.method.sampler
    assert sampler.kernel.configuration() == method.sampler.kernel.configuration()
    assert sampler.seed == 31 and sampler.num_chains == 2
    assert torch.equal(sampler.initial_values["b"], method.sampler.initial_values["b"])


@pytest.mark.parametrize(
    "values",
    [
        np.array([2**53 + 1, 2**63 - 1], dtype=np.int64),
        np.array([True, False]),
        np.array(["a", "long"], dtype="U4"),
        np.float64(0.125),
    ],
)
def test_safe_codec_preserves_numpy_dtype_and_exact_values(values):
    codec = _Codec()
    restored = codec.decode(codec.encode(values))
    assert values.dtype == restored.dtype
    np.testing.assert_array_equal(values, restored)


class _CustomGuide(AutoNormalGuide):
    pass


def test_unregistered_custom_guides_refuse_without_replacing_existing_artifact(
    tmp_path,
):
    bound = Model(SOURCE).bind(_data())
    guide = _CustomGuide(bound.require_tensor_program(), {"y"})
    path = tmp_path / "fit.pt"
    path.write_bytes(b"existing artifact")
    with pytest.raises(TypeError, match="ArtifactFactory"):
        save_fit(ModelFit(model=bound, posterior=guide), path)
    assert path.read_bytes() == b"existing artifact"


def test_custom_factory_is_explicit_and_gets_checked_context(tmp_path, monkeypatch):
    bound = Model(SOURCE).bind(_data())
    guide = _CustomGuide(bound.require_tensor_program(), {"y"})
    seen = []

    def reconstruct(config, context):
        seen.append(context.program)
        return _CustomGuide(context.program, {"y"}, init_scale=config["scale"])

    factory = ArtifactFactory(_CustomGuide, lambda value: {"scale": 0.2}, reconstruct)
    path = tmp_path / "custom.pt"
    save_fit(
        ModelFit(model=bound, posterior=guide), path, factories={"my-guide/v1": factory}
    )
    with pytest.raises(ValueError, match="my-guide/v1"):
        load_fit(path)
    _no_inference(monkeypatch)
    restored = load_fit(path, factories={"my-guide/v1": factory})
    assert seen == [restored.model.require_tensor_program()]
    assert type(restored.posterior) is _CustomGuide
    assert all(
        torch.equal(value, restored.posterior.state_dict()[name])
        for name, value in guide.state_dict().items()
    )


def _rewrite(path, change):
    payload = torch.load(path, weights_only=True)
    manifest = json.loads(payload["manifest"])
    change(manifest, payload)
    payload["manifest"] = json.dumps(manifest, sort_keys=True)
    payload["manifest_sha256"] = hashlib.sha256(
        payload["manifest"].encode()
    ).hexdigest()
    torch.save(payload, path)


@pytest.mark.parametrize(
    "corruption", ["version", "tensor", "manifest", "latent-layout", "missing-anchor"]
)
def test_malformed_or_incompatible_checkpoint_refuses_before_any_sampling(
    tmp_path, monkeypatch, corruption
):
    fitted = Model(SOURCE).fit(_data(), method=VariationalMethod(num_steps=0))
    path = tmp_path / "bad.pt"
    save_fit(fitted, path)
    if corruption == "version":
        _rewrite(path, lambda manifest, payload: manifest.update(format="future/v999"))
    elif corruption == "tensor":
        _rewrite(path, lambda manifest, payload: payload["tensors"]["0"].add_(1))
    elif corruption == "manifest":
        payload = torch.load(path, weights_only=True)
        payload["manifest"] += " "
        torch.save(payload, path)
    elif corruption == "latent-layout":
        _rewrite(
            path, lambda manifest, payload: manifest["posterior"].update(observed=[])
        )
    else:
        _rewrite(
            path,
            lambda manifest, payload: manifest["posterior"].update(
                anchors={"kind": "mapping", "items": []}
            ),
        )
    _no_inference(monkeypatch)
    before = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        load_fit(path)
    assert torch.equal(before, torch.random.get_rng_state())


def test_loaded_fit_reuses_existing_cross_validation_callback_with_identical_scores(
    tmp_path,
):
    from quivers.diagnostics.cross_validation import FoldPlan

    fitted = Model(SOURCE).fit(_data(), method=VariationalMethod(num_steps=0, seed=3))
    path = tmp_path / "cv.pt"
    save_fit(fitted, path)
    restored = load_fit(path)

    def factory(method, seed):
        return method.with_(seed=seed)

    original_callback = fitted.refit_callback(
        method_factory=factory, row_axis="Row", num_samples=4
    )
    restored_callback = restored.refit_callback(
        method_factory=factory, row_axis="Row", num_samples=4
    )
    split = FoldPlan(assignments=(0, 1, 0), seed=7).split(0)
    first, second = original_callback(split), restored_callback(split)
    assert first.log_likelihood.keys() == second.log_likelihood.keys()
    for name, value in first.log_likelihood.items():
        torch.testing.assert_close(value, second.log_likelihood[name], rtol=0, atol=0)
    assert second.fit is not restored


@pytest.mark.parametrize("count", [1, 3])
def test_static_extent_specialization_and_parameter_free_guide_survive(tmp_path, count):
    source = """program fixed[n:Nat](y:Tensor[Real]([n])):n->n
    observe y:n <- Normal(loc=0.2,scale=0.8)
    return y
export fixed
"""
    bound = Model(source).bind({"y": torch.zeros(count)}, parameters={"n": count})
    guide = AutoNormalGuide(bound.require_tensor_program(), {"y"})
    path = tmp_path / "static.pt"
    save_fit(ModelFit(model=bound, posterior=guide), path)
    restored = load_fit(path)
    assert restored.model.template_arguments == {"n": count}
    assert not restored.posterior.registry.names
    assert restored.model.data["y"].shape == (count,)
    torch.testing.assert_close(
        restored.log_likelihood(num_samples=2)["y"],
        torch.distributions.Normal(0.2, 0.8)
        .log_prob(torch.zeros(count))
        .expand(1, 2, count),
    )


def test_safe_reader_rejects_custom_pickle_globals_without_executing_them(tmp_path):
    class Payload:
        def __reduce__(self):
            return eval, ("1 + 2",)

    path = tmp_path / "unsafe.pt"
    torch.save({"payload": Payload()}, path)
    with pytest.raises(Exception, match="Weights only load failed|Unsupported global"):
        load_fit(path)


def test_zero_shape_only_extent_preserves_empty_data_without_inventing_an_axis(
    tmp_path,
):
    source = """object One:FinSet 1
program fixed[n:Nat](x:Tensor[Real]([n]),y:Real):One->One
    observe y <- Normal(loc=0.0,scale=1.0)
    return y
export fixed
"""
    bound = Model(source).bind(
        {"x": torch.empty(0), "y": torch.tensor(0.2)}, parameters={"n": 0}
    )
    fit = ModelFit(model=bound, posterior=PriorDraws(samples={}, num_samples=2))
    save_fit(fit, tmp_path / "zero.pt")
    restored = load_fit(tmp_path / "zero.pt")
    assert restored.model.data["x"].shape == (0,)
    assert restored.model.template_arguments == {"n": 0}


@pytest.mark.parametrize(
    "samples", [{}, {"b": torch.zeros(2, 3, 2)}, {"stray": torch.zeros(2, 3)}]
)
def test_recorded_posterior_site_or_geometry_mismatch_refuses_before_sampling(
    tmp_path, monkeypatch, samples
):
    fitted = ModelFit(model=Model(SOURCE).bind(_data()), posterior=_record(samples))
    save_fit(fitted, tmp_path / "invalid.pt")
    _no_inference(monkeypatch)
    with pytest.raises(ValueError, match="sites|shape"):
        load_fit(tmp_path / "invalid.pt")


def test_restored_mcmc_predicts_fresh_row_count_without_response_or_refitting(
    tmp_path, monkeypatch
):
    b = torch.tensor([[-0.5, 0.0, 0.7], [0.25, -0.1, 1.0]], dtype=torch.float64)
    fitted = ModelFit(model=Model(SOURCE).bind(_data()), posterior=_record({"b": b}))
    path = tmp_path / "predict.pt"
    with monkeypatch.context() as during_restore:
        _no_inference(during_restore)
        save_fit(fitted, path)
        restored = load_fit(path)
    x = torch.tensor([-2.0, -0.25, 0.0, 0.5, 2.0])
    actual = restored.posterior_epred({"x": x})["y"]
    torch.testing.assert_close(actual, b[..., None] * x)


def test_artifact_captures_values_at_write_time_without_sharing_live_training_storage(
    tmp_path,
):
    fitted = Model(SOURCE).fit(
        _data(), method=VariationalMethod(num_steps=0), snapshot_policy="reference"
    )
    expected = fitted.model.data["x"].clone()
    save_fit(fitted, tmp_path / "snapshot.pt")
    fitted.model.data["x"].fill_(123)
    restored = load_fit(tmp_path / "snapshot.pt")
    assert torch.equal(restored.model.data["x"], expected)
    assert restored.refit_recipe.snapshot_policy == "reference"


def test_custom_guide_factory_cannot_substitute_an_unrelated_model(tmp_path):
    bound = Model(SOURCE).bind(_data())
    guide = _CustomGuide(bound.require_tensor_program(), {"y"})
    factory = ArtifactFactory(
        _CustomGuide, lambda value: None, lambda config, context: guide
    )
    path = tmp_path / "wrong-model.pt"
    save_fit(ModelFit(model=bound, posterior=guide), path, factories={"guide": factory})
    with pytest.raises(ValueError, match="factory changed.*model"):
        load_fit(path, factories={"guide": factory})


def test_module_state_preserves_torch_local_version_metadata():
    class VersionedLinear(torch.nn.Linear):
        _version = 17

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args):
            assert local_metadata["version"] == 17
            super()._load_from_state_dict(state_dict, prefix, local_metadata, *args)

    original = VersionedLinear(2, 3).double()
    reconstructed = VersionedLinear(2, 3)
    _restore_module(reconstructed, _module_state(original))
    assert reconstructed.weight.dtype == torch.float64
    assert torch.equal(original.weight, reconstructed.weight)


def test_shared_parameter_buffer_alias_retains_its_registration_kind():
    module = torch.nn.Linear(2, 3)
    module.register_buffer("alias", module.weight)
    reconstructed = torch.nn.Linear(2, 3)
    reconstructed.register_buffer("alias", reconstructed.weight)
    _restore_module(reconstructed, _module_state(module))
    assert reconstructed.alias is reconstructed.weight
    assert (
        "alias" in reconstructed._buffers and "alias" not in reconstructed._parameters
    )


@pytest.mark.parametrize(
    "factories",
    [
        {"": ArtifactFactory(object, lambda x: None, lambda x, c: object())},
        {"bad": None},
        {1: ArtifactFactory(object, lambda x: None, lambda x, c: object())},
    ],
)
def test_factory_registration_requires_an_explicit_named_contract(factories):
    with pytest.raises(TypeError, match="factories require"):
        _Codec(factories)


def test_module_buffer_autograd_ownership_survives_state_restoration():
    original, reconstructed = torch.nn.Module(), torch.nn.Module()
    original.register_buffer(
        "value", torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    )
    reconstructed.register_buffer("value", torch.zeros(1))
    _restore_module(reconstructed, _module_state(original))
    assert (
        reconstructed.value.requires_grad and reconstructed.value.dtype == torch.float64
    )
    assert (
        torch.autograd.grad(reconstructed.value.square().sum(), reconstructed.value)[
            0
        ].item()
        == 1.0
    )


@pytest.mark.parametrize("policy", ["clone", "reference"])
@pytest.mark.parametrize("as_lists", [False, True])
def test_restored_source_recipe_retains_actual_binding_alias_policy(
    tmp_path, policy, as_lists
):
    data = _data()
    if as_lists:
        data = {name: value.tolist() for name, value in data.items()}
    fitted = Model(SOURCE).fit(
        data, method=VariationalMethod(num_steps=0), snapshot_policy=policy
    )
    save_fit(fitted, tmp_path / "ownership.pt")
    restored = load_fit(tmp_path / "ownership.pt")
    restored.model.data["x"][0] = 9.0
    updated = restored.update()
    assert float(updated.model.data["x"][0]) == (9.0 if policy == "reference" else 0.5)


def test_source_artifact_uses_recipe_original_source_after_public_model_mutation(
    tmp_path,
):
    specification = Model(SOURCE)
    fitted = specification.fit(_data(), method=VariationalMethod(num_steps=0))
    specification.source = SOURCE.replace("scale=1.0", "scale=2.0")
    save_fit(fitted, tmp_path / "source.pt")
    restored = load_fit(tmp_path / "source.pt")
    assert restored.model.specification.source == SOURCE


def test_noncanonical_live_recipe_requires_explicit_factory(tmp_path):
    from quivers.modeling import RefitRecipe

    bound = Model(SOURCE).bind(_data())
    recipe = RefitRecipe(
        _data(), "arbitrary operation", lambda data, method, seed: None
    )
    fitted = ModelFit(
        model=bound, posterior=_record({"b": torch.zeros(2, 3)}), refit_recipe=recipe
    )
    with pytest.raises(TypeError, match="ArtifactFactory"):
        save_fit(fitted, tmp_path / "operation.pt")
