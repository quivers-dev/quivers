"""Refits reuse checked preparation and explicit partitions without stale state."""

from dataclasses import FrozenInstanceError
import math
import pickle

import pytest
import torch

from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.inference import AutoNormalGuide, HMCKernel, MCMC
from quivers.modeling import (
    MCMCMethod,
    Model,
    ModelFit,
    RefitRecipe,
    VariationalMethod,
    refit_callback,
)


_SOURCE = """
object Row : FinSet _
object Other : FinSet 4
object One : FinSet 1
program regression(y:Tensor[Real]([Row]), constants:Tensor[Real]([Other])) : One -> One
    sample mu <- Normal(loc=0.0,scale=1.0)
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return mu
export regression
"""


def _data():
    return {"y": torch.tensor([-0.6, 0.2, 0.8, -0.3]), "constants": torch.arange(4.0)}


class _ConjugateNormal:
    """An exact conjugate fitting plugin, independent of numerical optimization."""

    def __init__(self):
        self.calls = []

    def run(self, bound):
        self.calls.append(bound)
        y = bound.tensor_data()["y"]
        variance = 1 / (1 + y.numel())
        guide = AutoNormalGuide(bound.require_tensor_program(), {"y"})
        with torch.no_grad():
            guide._loc("mu").fill_(variance * y.sum())
            guide._log_scale("mu").fill_(math.log(math.sqrt(variance)))
        return ModelFit(model=bound, posterior=guide)


def test_recipe_clones_nested_nonleaf_tensors_and_exposes_isolated_data():
    value = torch.tensor([1.0], requires_grad=True) * 2
    data = {"nested": [value]}
    recipe = RefitRecipe(
        data, "retained", lambda data, method, seed: (data, method, seed)
    )
    value.detach().add_(10)
    assert recipe.data["nested"][0].item() == 2
    assert not recipe.data["nested"][0].requires_grad
    exposed = recipe.data
    exposed["nested"][0].add_(5)
    assert recipe.data["nested"][0].item() == 2
    assert recipe.run()[1:] == ("retained", None)
    assert recipe.run(method="new", seed=7)[1:] == ("new", 7)
    with pytest.raises(FrozenInstanceError):
        recipe.method = "changed"


def test_recipe_reference_policy_is_explicit_and_observes_mutations():
    data = {"x": torch.tensor([1.0])}
    recipe = RefitRecipe(
        data, 1, lambda data, method, seed: data, snapshot_policy="reference"
    )
    data["x"].add_(3)
    assert recipe.data is data and recipe.run() is data
    assert recipe.run()["x"].item() == 4
    with pytest.raises(ValueError, match="snapshot_policy"):
        RefitRecipe(data, 1, lambda data, method, seed: data, snapshot_policy="unknown")


def test_live_recipe_has_explicit_serialization_boundary():
    recipe = RefitRecipe([1, 2], 1, lambda data, method, seed: data)
    with pytest.raises(TypeError, match="not a durable"):
        pickle.dumps(recipe)


def test_callback_snapshots_full_data_once_not_once_per_fold():
    copies = []

    def snapshot(data):
        copies.append(len(data))
        return tuple(data)

    recipe = RefitRecipe(
        tuple(range(8)), 1, lambda data, method, seed: data, snapshot=snapshot
    )
    callback = refit_callback(
        recipe,
        lambda data, rows: tuple(data[i] for i in rows),
        lambda fit, data, seed: {"y": torch.zeros(1, 2, len(data))},
        method_factory=lambda method, seed: method,
    )
    cross_validate(FoldPlan(assignments=(0, 1, 2, 3, 0, 1, 2, 3)), callback)
    assert copies == [8, 8]


def test_source_update_preserves_snapshot_and_exact_posterior_without_reparsing(
    monkeypatch,
):
    import quivers.modeling.model as module

    original_parse = module.parse
    parsed = []
    monkeypatch.setattr(
        module,
        "parse",
        lambda source: (parsed.append(source), original_parse(source))[1],
    )
    method = _ConjugateNormal()
    data = _data()
    fitted = Model(_SOURCE).fit(data, method=method, seed=31)
    original_program, original_posterior = fitted.model.program, fitted.posterior
    data["y"].add_(100)
    updated = fitted.update()
    assert parsed == [_SOURCE]
    assert updated.model.specification is fitted.model.specification
    assert updated.model.program is not original_program
    assert updated.posterior is not original_posterior
    assert fitted.posterior is original_posterior
    torch.testing.assert_close(updated.posterior._loc("mu"), torch.tensor([0.02]))
    assert updated.model.compilation_seed == 31
    replacement = {**_data(), "y": torch.tensor([1.0, 2.0])}
    second = updated.update(replacement, seed=47)
    torch.testing.assert_close(second.posterior._loc("mu"), torch.tensor([1.0]))
    torch.testing.assert_close(
        second.posterior._scale("mu"), torch.tensor([math.sqrt(1 / 3)])
    )
    assert second.model.compilation_seed == 47 and len(parsed) == 1
    assert second.refit_recipe.method is method


def test_retained_source_does_not_follow_public_model_source_mutation():
    model = Model(_SOURCE)
    fitted = model.fit(_data(), method=_ConjugateNormal())
    model.source = _SOURCE.replace("scale=1.0)", "scale=7.0)")
    updated = fitted.update()
    assert updated.model.specification.source == _SOURCE
    assert updated.model.specification is not model


def test_source_reference_policy_and_explicit_method_override():
    data = _data()
    fitted = Model(_SOURCE).fit(
        data, method=_ConjugateNormal(), snapshot_policy="reference"
    )
    data["y"].add_(1)
    updated = fitted.update(method=VariationalMethod(num_steps=0, seed=7))
    torch.testing.assert_close(updated.posterior._loc("mu"), torch.zeros(1))
    torch.testing.assert_close(updated.model.data["y"], data["y"])
    assert isinstance(updated.refit_recipe.method, VariationalMethod)


def test_update_is_a_fresh_fit_and_does_not_weaken_prediction_rebind():
    source = _SOURCE.replace(
        "sample mu <-",
        "sample local:Row <- Normal(loc=0.0,scale=1.0)\n    sample mu <-",
    )
    fitted = Model(source).fit(_data(), method=VariationalMethod(num_steps=0))
    replacement = {**_data(), "y": torch.zeros(2)}
    with pytest.raises(ValueError, match="latent"):
        fitted.model.rebind(replacement)
    updated = fitted.update(replacement)
    assert updated.posterior._registry["local"].constrained_shape == (2,)
    assert fitted.posterior._registry["local"].constrained_shape == (4,)


def test_source_fold_adapter_partitions_nominal_axis_and_scores_raw_likelihood():
    method = _ConjugateNormal()
    fitted = Model(_SOURCE).fit(_data(), method=method)
    seen_seeds = []

    def factory(original, seed):
        assert original is method
        seen_seeds.append(seed)
        return original

    callback = fitted.refit_callback(
        row_axis="Row", method_factory=factory, num_samples=1024
    )
    plan = FoldPlan(assignments=(0, 0, 1, 1), seed=12)
    result = cross_validate(plan, callback, save_fits=True)
    assert result.complete and seen_seeds == [13, 14]
    for fold, refitted in result.fits.items():
        split = plan.split(fold)
        torch.testing.assert_close(
            refitted.model.data["y"], _data()["y"][list(split.train)]
        )
        torch.testing.assert_close(refitted.model.data["constants"], torch.arange(4.0))
        variance = 1 / 3
        mean = variance * _data()["y"][list(split.train)].sum()
        expected = torch.distributions.Normal(mean, math.sqrt(1 + variance)).log_prob(
            _data()["y"][list(split.test)]
        )
        actual = result.log_predictive_density[
            [i for i, unit in enumerate(result.units) if unit[0] == fold]
        ]
        torch.testing.assert_close(actual, expected.double(), atol=0.04, rtol=0)
    assert len(method.calls) == 3


def test_fold_invalid_axis_and_missing_recipe_fail_before_inference():
    method = _ConjugateNormal()
    fitted = Model(_SOURCE).fit(_data(), method=method)
    with pytest.raises(ValueError, match="exactly one"):
        fitted.refit_callback(method_factory=lambda m, s: m)
    callback = fitted.refit_callback(row_axis="Wrong", method_factory=lambda m, s: m)
    with pytest.raises(ValueError, match="nominal"):
        callback(FoldPlan(assignments=(0, 0, 1, 1)).split(0))
    assert len(method.calls) == 1
    manual = fitted.with_(refit_recipe=None)
    with pytest.raises(ValueError, match="no retained"):
        manual.update()


def test_generic_callback_reuses_declared_partition_and_seed_policy():
    calls = []
    recipe = RefitRecipe(
        tuple(range(4)),
        3,
        lambda data, method, seed: (calls.append((data, method, seed)), data)[1],
    )
    callback = refit_callback(
        recipe,
        lambda data, rows: tuple(data[i] for i in rows),
        lambda fitted, test, seed: {"y": torch.zeros(1, 2, len(test))},
        method_factory=lambda method, seed: method + seed,
    )
    split = FoldPlan(assignments=(0, 0, 1, 1), seed=9).split(0)
    actual = callback(split)
    assert calls == [((2, 3), 13, 10)] and actual.fit == (2, 3)


def test_source_mcmc_override_retains_real_chains_and_likelihood_semantics():
    fitted = Model(_SOURCE).fit(_data(), method=VariationalMethod(num_steps=0))
    sampler = MCMC(
        HMCKernel(step_size=0.15, num_steps=3),
        num_warmup=5,
        num_samples=8,
        num_chains=1,
        seed=19,
    )
    updated = fitted.update(method=MCMCMethod(sampler=sampler))
    draws = updated.draws()["mu"]
    scores = updated.log_likelihood()["y"]
    expected = torch.distributions.Normal(draws[..., None], 1.0).log_prob(_data()["y"])
    torch.testing.assert_close(scores, expected)
    assert draws.shape == (1, 8)


def test_custom_nondraw_result_is_returned_without_recipe_conversion():
    class Method:
        def run(self, bound):
            return {"answer": 4}

    assert Model(_SOURCE).fit(_data(), method=Method()) == {"answer": 4}


@pytest.mark.parametrize("mode", ["yes", "only"])
def test_update_retains_prior_policy_stream_and_caller_rng(mode):
    fitted = Model(_SOURCE).fit(
        _data(),
        method=VariationalMethod(num_steps=0, seed=9),
        sample_prior=mode,
        num_prior_samples=3,
        prior_seed=17,
    )
    state = torch.get_rng_state().clone()
    updated = fitted.update()
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert updated.sample_prior == mode
    if mode == "yes":
        assert updated.prior.num_samples == 3
        torch.testing.assert_close(
            updated.prior.samples["mu"], fitted.prior.samples["mu"]
        )


def test_update_retains_explicit_static_arguments_and_compile_seed():
    source = """
object Row : FinSet _
object Group : FinSet 2
object One : FinSet 1
program grouped[G:FinSet](idx:Tensor[Int]([Row]),y:Tensor[Real]([Row])) : One -> One
    sample a:G <- Normal(loc=0.0,scale=1.0)
    let mu = a[idx]
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return a
export grouped
"""
    fitted = Model(source).fit(
        {"idx": torch.tensor([0, 1, 0]), "y": torch.zeros(3)},
        parameters={"G": "Group"},
        method=VariationalMethod(num_steps=0),
        seed=23,
    )
    updated = fitted.update({"idx": torch.tensor([1, 0]), "y": torch.ones(2)})
    assert updated.model.template_arguments == {"G": "Group"}
    assert updated.posterior._registry["a"].constrained_shape == (2,)
    assert updated.model.compilation_seed == 23
