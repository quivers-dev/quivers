"""Durable formula fits rebuild designs and bindings without rerunning inference."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import NonlinearExpression, PriorSpec, bf, families, fit
from quivers.dsl.emit import module_to_source
from quivers.inference import AutoNormalGuide, MCMC, MCMCResult, SVI
from quivers.modeling import ArtifactFactory, load_fit, save_fit
from quivers.modeling._artifact_codec import _Codec


def _forbid(*args, **kwargs):
    raise AssertionError("restoration must not fit, sample, or execute a predictor")


def _record(fitted):
    """Keep a small deterministic retained grid in its existing posterior schema."""
    values = fitted.draws(num_samples=4, seed=17)
    return fitted.with_(
        posterior=MCMCResult(
            samples=values,
            log_densities=torch.zeros(1, 4),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=2,
            num_samples=4,
        )
    )


def _restore(fitted, tmp_path, monkeypatch, *, factories=None):
    path = tmp_path / "formula.pt"
    save_fit(fitted, path, factories=factories)
    before = torch.random.get_rng_state().clone()
    with monkeypatch.context() as patch:
        from quivers.effects.sites import TorchSampleable
        import quivers.formulas._fit as api

        patch.setattr(api, "_run_inference", _forbid)
        patch.setattr(MCMC, "run", _forbid)
        patch.setattr(SVI, "run", _forbid)
        patch.setattr(AutoNormalGuide, "rsample", _forbid)
        patch.setattr(TorchSampleable, "sample", _forbid)
        patch.setattr(TorchSampleable, "rsample", _forbid)
        restored = load_fit(path, factories=factories)
    assert torch.equal(before, torch.random.get_rng_state())
    return restored


@pytest.mark.parametrize("policy", ["quivers", "brms"])
def test_joint_artifact_restores_the_fitted_prior_policy(policy, tmp_path, monkeypatch):
    """Rebuild the checked program with the policy used during fitting."""
    frame = pd.DataFrame(
        {"y": [-2.0, 0.0, 2.0, 4.0], "z": [-3.0, 3.0, 6.0, 9.0], "x": range(4)}
    )
    fitted = fit(
        bf("y ~ x") + bf("z ~ x"),
        data=frame,
        method="svi",
        num_steps=3,
        seed=2,
        default_priors=policy,
    )
    restored = _restore(fitted, tmp_path, monkeypatch)
    assert restored.qvr_source == fitted.qvr_source
    assert restored.specification == fitted.specification
    draws = restored.draws(num_samples=3, seed=11)
    for name, value in fitted.draws(num_samples=3, seed=11).items():
        torch.testing.assert_close(draws[name], value, rtol=0, atol=0)


def _assert_replay(fitted, restored, *, fresh=None):
    assert type(restored.posterior) is type(fitted.posterior)
    assert module_to_source(restored.prediction_plan.module) == module_to_source(
        fitted.prediction_plan.module
    )
    assert restored.losses == fitted.losses
    assert restored.category_labels == fitted.category_labels
    assert restored.coefficient_labels == fitted.coefficient_labels
    assert restored.observations.keys() == fitted.observations.keys()
    for name, value in fitted.observations.items():
        assert value.dtype == restored.observations[name].dtype
        torch.testing.assert_close(value, restored.observations[name], rtol=0, atol=0)
    for name, value in fitted.draws(num_samples=3, seed=11).items():
        torch.testing.assert_close(
            value, restored.draws(num_samples=3, seed=11)[name], rtol=0, atol=0
        )
    for name, value in fitted.log_likelihood(num_samples=3, seed=11).items():
        torch.testing.assert_close(
            value, restored.log_likelihood(num_samples=3, seed=11)[name], rtol=0, atol=0
        )
    for method in ("posterior_linpred", "posterior_predict"):
        a = getattr(fitted, method)(num_samples=3, seed=12, new_data=fresh)
        b = getattr(restored, method)(num_samples=3, seed=12, new_data=fresh)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("recorded", [False, True])
def test_gaussian_joint_and_every_coefficient_gradient_match_independent_normal(
    tmp_path, monkeypatch, recorded
):
    data = pd.DataFrame({"y": [0.2, -0.4, 0.7], "x": [-1.0, 0.5, 2.0]})
    fitted = fit(
        "y ~ x",
        data=data,
        method="svi",
        num_steps=2,
        seed=9,
        fixed_parameters={"sigma": 0.7},
        fixed_prior="Normal(0.0, 2.0)",
        priors={"intercept": "Normal(0.0, 2.0)", "beta_x": "Normal(0.0, 2.0)"},
    )
    if recorded:
        fitted = _record(fitted)
    restored = _restore(fitted, tmp_path, monkeypatch)
    _assert_replay(fitted, restored, fresh=pd.DataFrame({"x": [3.0, -2.0]}))
    from quivers.effects.clamp import ClampHandler
    from quivers.inference import trace

    parameters = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    sites = {"intercept": parameters[0], "beta_x": parameters[1]}
    with ClampHandler(sites):
        actual = trace(
            restored.program, torch.zeros(1, 1), dict(restored.observations)
        ).log_joint
    x = torch.tensor(data.x.to_numpy(), dtype=torch.float64)
    y = torch.tensor(data.y.to_numpy(), dtype=torch.float64)
    expected = (
        torch.distributions.Normal(0.0, 2.0).log_prob(parameters).sum()
        + torch.distributions.Normal(parameters[0] + parameters[1] * x, 0.7)
        .log_prob(y)
        .sum()
    )
    torch.testing.assert_close(actual.reshape(()), expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameters, retain_graph=True)[0],
        torch.autograd.grad(expected, parameters)[0],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_group_covariance_labels_and_new_group_policy_survive(
    tmp_path, monkeypatch, parameterization
):
    frame = pd.DataFrame(
        {
            "y": [0.2, -0.3, 0.6, 0.1],
            "x": [0.0, 1.0, 2.0, -1.0],
            "g": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "unused"]),
        }
    )
    fitted = fit(
        "y ~ x + (1 + x | id | g)",
        data=frame,
        method="svi",
        num_steps=0,
        reparameterize=parameterization,
        fixed_parameters={"sigma": 0.8},
    )
    restored = _restore(fitted, tmp_path, monkeypatch)
    _assert_replay(
        fitted, restored, fresh=pd.DataFrame({"x": [1.0, -2.0], "g": ["b", "a"]})
    )
    new = pd.DataFrame({"x": [1.0, -2.0], "g": ["fresh", "fresh"]})
    torch.testing.assert_close(
        fitted.posterior_linpred(
            new_data=new,
            allow_new_levels=True,
            sample_new_levels="old_levels",
            num_samples=3,
            seed=7,
        ),
        restored.posterior_linpred(
            new_data=new,
            allow_new_levels=True,
            sample_new_levels="old_levels",
            num_samples=3,
            seed=7,
        ),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "kind", ["categorical", "multinomial", "dirichlet", "logistic_normal"]
)
def test_factor_and_vector_coordinates_retain_reference_and_trials(
    tmp_path, monkeypatch, kind
):
    family = families[kind].with_reference("B" if kind != "categorical" else "b")
    if kind == "categorical":
        frame = pd.DataFrame(
            {
                "y": pd.Categorical(["b", "a", "c", "a"], categories=["c", "b", "a"]),
                "x": [0.0, 1.0, -1.0, 0.5],
            }
        )
        formula = "y ~ x"
    else:
        values = [[2, 1, 1], [0, 3, 2], [1, 1, 2], [3, 0, 1]]
        if kind in {"dirichlet", "logistic_normal"}:
            values = [
                [0.2, 0.3, 0.5],
                [0.1, 0.7, 0.2],
                [0.4, 0.3, 0.3],
                [0.6, 0.2, 0.2],
            ]
        frame = pd.DataFrame(values, columns=["A", "B", "C"]).assign(
            x=[0.0, 1.0, -1.0, 0.5]
        )
        formula = "cbind(A,B,C) ~ x"
    fitted = fit(formula, data=frame, family=family, method="svi", num_steps=0)
    restored = _restore(fitted, tmp_path, monkeypatch)
    _assert_replay(fitted, restored)
    if kind == "multinomial":
        new = pd.DataFrame({"x": [0.2, -0.1]})
        torch.testing.assert_close(
            fitted.posterior_predict(new_data=new, trials=7, num_samples=2, seed=3),
            restored.posterior_predict(new_data=new, trials=7, num_samples=2, seed=3),
            rtol=0,
            atol=0,
        )


def test_subset_censoring_truncation_weights_and_original_refit_rows(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        {
            "y": [0.2, 0.4, 0.6, 0.8],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "status": [0, 1, 2, 0],
            "upper": [np.nan, np.nan, 0.9, np.nan],
            "keep": [True, False, True, True],
            "w": [1.0, 10.0, 0.5, 2.0],
        }
    )
    fitted = fit(
        "y | subset(keep) + cens(status,upper) + trunc(lb=0,ub=1) + weights(w) ~ x",
        data=frame,
        method="svi",
        num_steps=0,
        fixed_parameters={"sigma": 0.4},
    )
    restored = _restore(fitted, tmp_path, monkeypatch)
    _assert_replay(fitted, restored, fresh=pd.DataFrame({"x": [0.5, -0.2]}))
    pd.testing.assert_frame_equal(frame, restored.refit_recipe.data)
    assert restored.formula.modifiers.subset.positions == (0, 2, 3)


def test_nonlinear_graph_transforms_typed_priors_and_fresh_rows(tmp_path, monkeypatch):
    frame = pd.DataFrame({"y": [0.1, 0.3, 0.5, 0.8], "x": [-2.0, -1.0, 1.0, 2.0]})
    fitted = fit(
        "y ~ 0",
        data=frame,
        method="svi",
        num_steps=0,
        distributional={"mu": NonlinearExpression(expression="a * x")},
        nonlinear={"a": "~ 1"},
        priors=[
            PriorSpec(distribution="Normal(0.0, 1.5)", nlpar="a", class_="Intercept")
        ],
        fixed_parameters={"sigma": 0.5},
    )
    restored = _restore(fitted, tmp_path, monkeypatch)
    _assert_replay(fitted, restored, fresh=pd.DataFrame({"x": [3.0, -0.5]}))
    assert restored.predictor_graph == fitted.predictor_graph
    assert restored.priors == fitted.priors


def test_prior_retention_is_restored_without_independent_redraw(tmp_path, monkeypatch):
    fitted = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0.2, -0.1]}),
        method="svi",
        num_steps=0,
        sample_prior="yes",
        num_prior_samples=3,
        prior_seed=31,
    )
    restored = _restore(fitted, tmp_path, monkeypatch)
    assert restored.sample_prior == "yes"
    assert restored.prior is not fitted.prior
    for name, value in fitted.prior.samples.items():
        torch.testing.assert_close(value, restored.prior.samples[name], rtol=0, atol=0)


class _Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def test_neural_trained_and_initial_recipe_states_are_distinct_no_forward_on_load(
    tmp_path, monkeypatch
):
    predictor = _Predictor()
    data = torch.tensor([[1.0, 0.5], [0.0, 1.0], [-1.0, 2.0]], dtype=torch.float64)
    factory = {
        "linear-head/v1": ArtifactFactory(
            _Predictor, lambda value: {}, lambda config, context: _Predictor()
        )
    }
    fitted = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0.2, 0.5, -0.3]}),
        predictor=predictor,
        predictor_data=data,
        method="svi",
        num_steps=3,
        seed=8,
        fixed_parameters={"sigma": 0.8},
    )
    with monkeypatch.context() as patch:
        patch.setattr(_Predictor, "forward", _forbid)
        restored = _restore(fitted, tmp_path, patch, factories=factory)
    _assert_replay(fitted, restored, fresh=None)
    for name, tensor in predictor.state_dict().items():
        torch.testing.assert_close(
            tensor, restored.predictor.state_dict()[name], rtol=0, atol=0
        )
        initial = restored.refit_recipe.method["predictor"].state_dict()[name]
        assert not torch.equal(tensor, initial)
    fresh = pd.DataFrame({"unused": [1.0, 2.0]})
    inputs = data[:2]
    torch.testing.assert_close(
        fitted.posterior_predict(
            new_data=fresh, predictor_data=inputs, num_samples=2, seed=4
        ),
        restored.posterior_predict(
            new_data=fresh, predictor_data=inputs, num_samples=2, seed=4
        ),
        rtol=0,
        atol=0,
    )


def test_dataframe_safe_codec_preserves_ints_categories_nullable_and_index():
    frame = pd.DataFrame(
        {
            "count": pd.Series([2**53 + 1, 2**63 - 1], dtype="Int64"),
            "flag": pd.Series([True, None], dtype="boolean"),
            "label": pd.Categorical(
                ["b", None], categories=["c", "b", "a"], ordered=True
            ),
        }
    )
    frame.index = pd.Index(["r2", "r1"], name="row")
    codec = _Codec(formula=True)
    restored = codec.decode(codec.encode(frame))
    pd.testing.assert_frame_equal(restored, frame)
    with pytest.raises(TypeError, match="artifact contract"):
        codec.encode(pd.DataFrame({"unsafe": [object()]}))


def test_formula_artifact_requires_reconstruction_inputs(tmp_path):
    fitted = fit(
        "y ~ 1", data=pd.DataFrame({"y": [0.0, 1.0]}), method="svi", num_steps=0
    )
    with pytest.raises(ValueError, match="canonical original-data"):
        save_fit(fitted.with_(refit_recipe=None), tmp_path / "bad.pt")


def test_changed_preparation_and_observation_geometry_refuse(tmp_path):
    fitted = fit(
        "y ~ 1", data=pd.DataFrame({"y": [0.0, 1.0]}), method="svi", num_steps=0
    )
    path = tmp_path / "bad.pt"
    save_fit(fitted, path)
    payload = torch.load(path, weights_only=True)
    manifest = json.loads(payload["manifest"])
    manifest["source"] += "\n# changed\n"
    payload["manifest"] = json.dumps(manifest, sort_keys=True, allow_nan=False)
    payload["manifest_sha256"] = hashlib.sha256(
        payload["manifest"].encode()
    ).hexdigest()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="changed the checked source"):
        load_fit(path)


def test_source_artifact_public_types_do_not_import_formula_extras():
    script = """
import sys, importlib.abc, typing
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'formulae','pandas','polars','narwhals'}:
            raise ImportError('optional formula dependency blocked')
sys.meta_path.insert(0, Block())
from quivers.modeling import save_fit, load_fit
assert typing.get_type_hints(save_fit)
assert typing.get_type_hints(load_fit)
assert 'quivers.formulas' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("ownership", ["clone", "reference"])
def test_original_training_snapshot_survives_reference_recipe_mutations(
    tmp_path, monkeypatch, ownership
):
    frame = pd.DataFrame({"y": [0.2, 0.4, -0.1], "x": [-1.0, 0.0, 2.0]})
    fitted = fit(
        "y ~ center(x)",
        data=frame,
        method="svi",
        num_steps=0,
        snapshot_policy=ownership,
        fixed_parameters={"sigma": 0.6},
    )
    original = frame.copy(deep=True)
    plan = fitted.prediction_plan
    if ownership == "clone":
        assert plan.preparation_data is fitted.refit_recipe._data
    else:
        assert plan.preparation_data is not frame
    frame.loc[:, "x"] += 20
    restored = _restore(fitted, tmp_path, monkeypatch)
    _assert_replay(fitted, restored, fresh=pd.DataFrame({"x": [1.0, -2.0]}))
    pd.testing.assert_frame_equal(restored.prediction_plan.preparation_data, original)
    pd.testing.assert_frame_equal(
        restored.refit_recipe.data, frame if ownership == "reference" else original
    )


@pytest.mark.parametrize("family", ["cumulative", "hurdle_cumulative", "cox"])
def test_retained_threshold_domain_and_survival_basis(tmp_path, monkeypatch, family):
    if family == "cox":
        frame = pd.DataFrame(
            {"y": [0.2, 0.4, 0.7, 1.2, 1.5, 2.0], "x": [-1.0, 0.0, 0.5, 1.0, -0.5, 2.0]}
        )
        descriptor = families[family].with_baseline(df=3, degree=1)
        formula = "y ~ x"
    else:
        frame = pd.DataFrame(
            {
                "y": [0, 1, 2, 3, 1, 2],
                "x": [-1.0, 0.0, 0.5, 1.0, -0.5, 2.0],
                "z": [-0.3, 0.2, 0.1, -0.2, 0.5, 0.4],
            }
        )
        descriptor = families[family].with_thresholds("equidistant")
        formula = "y ~ x + cs(z)"
    fitted = fit(formula, data=frame, family=descriptor, method="svi", num_steps=0)
    restored = _restore(fitted, tmp_path, monkeypatch)
    _assert_replay(fitted, restored)
    if family != "cox":
        torch.testing.assert_close(
            fitted.posterior_thresholds(num_samples=3, seed=5),
            restored.posterior_thresholds(num_samples=3, seed=5),
            rtol=0,
            atol=0,
        )


def _catalog(*, constraint="real"):
    from quivers.family_catalog import FamilyCatalog
    from quivers.qiec import DistributionFamily, FamilyParameter, REAL

    semantic = DistributionFamily(
        name="PortableNormal",
        parameters=(FamilyParameter(name="loc", constraint=constraint, rank=0),),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
    )
    return FamilyCatalog().with_family(
        semantic, lambda parameters: torch.distributions.Normal(parameters["loc"], 1.0)
    )


def _catalog_factories(*, changed=False):
    from quivers.family_catalog import FamilyCatalog

    return {
        "portable-normal/v1": ArtifactFactory(
            FamilyCatalog,
            lambda catalog: {},
            lambda config, context: _catalog(
                constraint="positive" if changed else "real"
            ),
        )
    }


@pytest.mark.parametrize("frontend", ["source", "formula"])
def test_explicit_catalog_factory_retains_contract_and_refuses_semantic_drift(
    tmp_path, monkeypatch, frontend
):
    from quivers.modeling import Model, VariationalMethod
    from quivers.formulas import custom_family

    catalog = _catalog()
    if frontend == "source":
        source = """object Out : Real 1
program p(y:Real) : Out -> Out
    sample b <- Normal(loc=0.0,scale=1.0)
    observe y <- PortableNormal(loc=b)
    return b
export p
"""
        fitted = Model(source, families=catalog).fit(
            {"y": 0.3}, input=torch.zeros(1, 1), method=VariationalMethod(num_steps=0)
        )
    else:
        family = custom_family("PortableNormal", catalog=catalog, location="loc")
        fitted = fit(
            "y ~ x",
            data=pd.DataFrame({"y": [0.2, -0.1], "x": [0.0, 1.0]}),
            family=family,
            method="svi",
            num_steps=0,
        )
    path = tmp_path / "custom.pt"
    with pytest.raises(TypeError, match="ArtifactFactory"):
        save_fit(fitted, path)
    save_fit(fitted, path, factories=_catalog_factories())
    restored = load_fit(path, factories=_catalog_factories())
    if frontend == "formula":
        _assert_replay(fitted, restored, fresh=pd.DataFrame({"x": [2.0]}))
    else:
        assert restored.model.specification.families is not catalog
        torch.testing.assert_close(
            fitted.posterior.state_dict()["loc_b"],
            restored.posterior.state_dict()["loc_b"],
        )
    with pytest.raises(ValueError, match="changed its checked semantic"):
        load_fit(path, factories=_catalog_factories(changed=True))
    with pytest.raises(ValueError, match="explicitly registered factory"):
        load_fit(path)


def test_restored_update_and_cross_validation_reuse_original_fitter_and_raw_scores(
    tmp_path, monkeypatch
):
    from quivers.diagnostics import FoldPlan

    frame = pd.DataFrame({"y": [0.2, -0.1, 0.6, 0.4], "x": [-1.0, 0.0, 1.0, 2.0]})
    fitted = fit(
        "y ~ x",
        data=frame,
        method="svi",
        num_steps=0,
        seed=13,
        fixed_parameters={"sigma": 0.7},
    )
    restored = _restore(fitted, tmp_path, monkeypatch)
    original_update = fitted.update(seed=31, num_steps=1)
    loaded_update = restored.update(seed=31, num_steps=1)
    for name, tensor in original_update.posterior.state_dict().items():
        torch.testing.assert_close(
            tensor, loaded_update.posterior.state_dict()[name], rtol=0, atol=0
        )
    split = FoldPlan(assignments=(0, 0, 1, 1), seed=19).split(0)

    def method(settings, seed):
        return {**settings, "seed": seed}

    a = fitted.refit_callback(method_factory=method, num_samples=3)(split)
    b = restored.refit_callback(method_factory=method, num_samples=3)(split)
    torch.testing.assert_close(
        a.log_likelihood["y"], b.log_likelihood["y"], rtol=0, atol=0
    )
    values = b.fit.draws(num_samples=3, seed=split.seed)
    held = frame.iloc[list(split.test)]
    mean = values["intercept"][..., None] + values["beta_x"][..., None] * values[
        "intercept"
    ].new_tensor(held.x.to_numpy())
    oracle = torch.distributions.Normal(mean, 0.7).log_prob(
        mean.new_tensor(held.y.to_numpy())
    )
    torch.testing.assert_close(b.log_likelihood["y"], oracle, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "index",
    [
        pd.CategoricalIndex(
            ["b", "a"], categories=["a", "b", "unused"], ordered=True, name="category"
        ),
        pd.MultiIndex(
            levels=[["unused", "a", "b"], [3, 2, 1]],
            codes=[[1, 2], [2, 1]],
            names=["g", "row"],
        ),
    ],
)
def test_frame_index_retains_unused_levels_order_and_codes(index):
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=index)
    codec = _Codec(formula=True)
    pd.testing.assert_frame_equal(frame, codec.decode(codec.encode(frame)))
