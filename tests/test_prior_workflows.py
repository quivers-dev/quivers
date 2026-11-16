"""Independent prior laws and the shared source/formula inference interfaces."""

import math
import didactic.api as dx

import pytest
import torch

from quivers.dsl import loads
from quivers.inference import prior_draws, PriorDraws


def test_ancestral_joint_draws_preserve_hierarchical_moments_and_rng():
    source = """object R:Real 1
program p:R->R
    sample a <- Normal(loc=1.0,scale=2.0)
    sample b <- Normal(loc=a,scale=0.5)
    observe y <- Normal(loc=b,scale=1.0)
    return b
export p
"""
    model = loads(source).morphism
    torch.manual_seed(71)
    state = torch.random.get_rng_state().clone()
    result = prior_draws(
        model, torch.zeros(1, 1), {"y": torch.tensor(999.0)}, num_samples=1000, seed=4
    )
    assert isinstance(result, PriorDraws)
    assert set(result.samples) == {"a", "b"}
    assert torch.equal(state, torch.random.get_rng_state())
    a, b = (result.samples[name].flatten() for name in ("a", "b"))
    assert abs(float(a.mean()) - 1) < 0.2
    assert abs(float(a.var()) - 4) < 0.5
    assert abs(float((b - a).var()) - 0.25) < 0.04
    assert abs(float(torch.cov(torch.stack((a, b)))[0, 1]) - 4) < 0.5
    again = prior_draws(model, torch.zeros(1, 1), num_samples=4, seed=4)
    for name in result.samples:
        torch.testing.assert_close(again.samples[name], result.samples[name][:, :4])


@pytest.mark.parametrize(
    "family,arguments,kind",
    [
        ("Gamma", "concentration=2.0,rate=1.0", "positive"),
        ("Beta", "concentration1=2.0,concentration0=3.0", "interval"),
        ("Poisson", "rate=3.0", "integer"),
    ],
)
def test_independent_draws_use_declared_support_without_a_continuous_bijector(
    family, arguments, kind
):
    source = f"object R:Real 1\nprogram p:R->R\n    sample z <- {family}({arguments})\n    return z\nexport p\n"
    values = prior_draws(
        loads(source).morphism, torch.zeros(1, 1), num_samples=20, seed=11
    ).samples["z"]
    assert values.shape == (1, 20)
    if kind == "positive":
        assert bool((values > 0).all())
    elif kind == "interval":
        assert bool(((values > 0) & (values < 1)).all())
    else:
        assert bool(((values >= 0) & (values == values.floor())).all())


def test_independent_simplex_plate_axes_and_registry_share_geometry():
    from quivers.inference import LatentRegistry

    source = """object Row:FinSet 2
object Category:FinSet 3
object R:Real 1
program p:Row->R
    sample pi:Row <- Dirichlet(concentration=1.0) [over=Category,iid_over=Row]
    return pi
export p
"""
    program = loads(source).morphism
    result = prior_draws(program, torch.zeros(1, 1), num_samples=7, seed=2)
    assert result.samples["pi"].shape == (1, 7, 2, 3)
    assert LatentRegistry.from_model(program, set())["pi"].constrained_shape == (2, 3)
    torch.testing.assert_close(result.samples["pi"].sum(-1), torch.ones(1, 7, 2))


def test_ancestral_refuses_global_potentials_but_prior_only_mcmc_retains_them():
    from quivers.inference import MCMC, NUTSKernel, MCMCResult
    from tests.test_prior_sampling import SOURCE

    source = SOURCE.replace(
        "    observe y:Row <- Normal(loc=b,scale=1.0) [weights=w]\n", ""
    )
    program = loads(source).morphism
    with pytest.raises(ValueError, match="score potentials.*MCMC"):
        prior_draws(program, torch.zeros(1, 1), num_samples=2)
    driver = MCMC(NUTSKernel(), num_warmup=4, num_samples=5, num_chains=1, seed=8)
    result = prior_draws(program, torch.zeros(1, 1), method=driver)
    assert isinstance(result, MCMCResult)
    b = result.samples["b"]
    expected = -b.square() - 0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(result.log_densities, expected)


def test_formula_prior_modes_reuse_predictions_raw_likelihoods_and_separate_rng():
    import pandas as pd
    from quivers.formulas import fit

    data = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [20.0, 21.0, 22.0]})
    ordinary = fit("y~x", data=data, method="svi", num_steps=2, seed=23)
    retained = fit(
        "y~x",
        data=data,
        method="svi",
        num_steps=2,
        seed=23,
        sample_prior="yes",
        num_prior_samples=4,
        prior_seed=19,
    )
    for name, value in ordinary.draws(num_samples=3, seed=42).items():
        torch.testing.assert_close(value, retained.draws(num_samples=3, seed=42)[name])
    assert retained.losses == ordinary.losses and isinstance(retained.prior, PriorDraws)
    view = retained.prior_view()
    assert view.program is retained.program and view.posterior is retained.prior
    assert view.posterior_epred().shape == (1, 4, 3)
    assert view.posterior_predict(seed=18).shape == (1, 4, 3)
    assert view.log_likelihood()[view.formula.response_name].shape == (1, 4, 3)
    analysis = view.analysis()
    assert analysis.kind == "prior" and analysis.sample_stats == {}
    assert all(summary.convergence is None for summary in analysis.summary().values())
    only = fit(
        "y~x", data=data, method="svi", num_steps=2, seed=23, sample_prior="only"
    )
    assert only.sample_prior == "only" and only.losses != ordinary.losses


def test_source_model_prior_view_uses_existing_posterior_collectors():
    from quivers.modeling import Model, VariationalMethod
    from tests.modeling.test_model import SOURCE as source

    bound = Model(source).bind({"x": torch.ones(3), "y": torch.zeros(3)})
    retained = bound.infer(
        VariationalMethod(num_steps=2, seed=4),
        sample_prior="yes",
        num_prior_samples=3,
        prior_seed=5,
    )
    assert isinstance(retained.prior, PriorDraws)
    view = retained.prior_view()
    assert view.draws()["b"].shape == (1, 3)
    assert view.predict(seed=3)["y"].shape[-1] == 3
    assert view.analysis().kind == "prior"


@pytest.mark.parametrize("mode", ["bad", "YES", True, 1])
def test_bad_prior_modes_are_rejected(mode):
    from quivers.modeling import Model, VariationalMethod
    from tests.modeling.test_model import SOURCE as source

    bound = Model(source).bind({"x": torch.ones(3), "y": torch.zeros(3)})
    with pytest.raises(ValueError, match="sample_prior"):
        bound.infer(VariationalMethod(num_steps=1), sample_prior=mode)


def test_prior_projection_preserves_scan_sites_and_feedback_without_its_downstream_output():
    from quivers.continuous.program_steps import Draw, Let, Scan, reading
    from quivers.continuous.programs import MonadicProgram
    from quivers.continuous.spaces import Euclidean, ProductSpace

    real = Euclidean(name="R", dim=1)
    normal = (
        loads(
            "object R:Real 1\nprogram p:R->R\n    sample e <- Normal(loc=0.0,scale=1.0)\n    return e\nexport p\n"
        )
        .morphism.steps[0]
        .morphism
    )
    cell = MonadicProgram(
        ProductSpace(components=(real, real)),
        real,
        (
            Draw(names=("noise",), morphism=normal, args=("entry",)),
            Let(
                name="next",
                value=reading(
                    lambda env: env["entry"] + env["state"] + env["noise"],
                    {"entry", "state", "noise"},
                ),
            ),
        ),
        ("next",),
        params=("entry", "state"),
    )
    program = MonadicProgram(
        real,
        real,
        (Scan(name="trajectory", step=cell, source=None, initial=torch.zeros(1)),),
        ("trajectory",),
        params=None,
    )
    result = prior_draws(program, torch.ones(1, 3, 1), num_samples=7, seed=3)
    assert set(result.samples) == {"noise@0", "noise@1", "noise@2"}
    assert all(value.shape == (1, 7) for value in result.samples.values())


def test_prior_container_validates_draw_axes_and_discrete_summaries_do_not_claim_chains():
    from quivers.diagnostics.posterior import PosteriorAnalysis

    result = PriorDraws(samples={"z": torch.tensor([[0, 1, 2, 3]])}, num_samples=4)
    summary = PosteriorAnalysis(
        samples=result.samples, log_likelihood={}, kind="prior"
    ).summary()["z"]
    assert summary.mean.item() == 1.5 and summary.convergence is None


@pytest.mark.parametrize(
    "shape,count", [((2, 3), 3), ((1, 2), 3), ((3,), 3), ((1, 0), 0)]
)
def test_prior_container_rejects_inconsistent_axes(shape, count):
    with pytest.raises(dx.ValidationError):
        PriorDraws(samples={"x": torch.zeros(shape)}, num_samples=count)


def test_normalized_measure_prior_remains_a_proper_conditional_kernel():
    source = """object R:Real 1
program p:R->R
    sample z <- Normalize(base=Restrict(base=Normal(loc=0.0,scale=1.0),low=-1.0,high=1.0))
    return z
export p
"""
    result = prior_draws(
        loads(source).morphism, torch.zeros(1, 1), num_samples=25, seed=5
    )
    assert bool((result.samples["z"].abs() < 1).all())


def test_prediction_law_does_not_score_a_censored_count_dummy_but_raw_likelihood_does():
    import pandas as pd
    from quivers.formulas import fit

    data = pd.DataFrame({"y": [-1, 1, 2], "status": [1, 0, 0], "x": [-1.0, 0.0, 1.0]})
    result = fit(
        "y | cens(status) ~ x",
        data=data,
        family="poisson",
        method="svi",
        num_steps=1,
        seed=14,
    )
    fresh = pd.DataFrame({"x": [0.5, -0.5]})
    assert result.posterior_epred(new_data=fresh, num_samples=2).shape == (1, 2, 2)
    assert result.posterior_predict(new_data=fresh, num_samples=2).shape == (1, 2, 2)
    ll = result.log_likelihood(num_samples=2)["y"]
    assert ll.shape == (1, 2, 3) and bool(torch.isfinite(ll).all())
    torch.testing.assert_close(ll[..., 0], torch.zeros_like(ll[..., 0]))


def test_data_dependent_prior_keeps_its_inputs_and_reuses_exact_random_stream():
    source = """object R:Real 1
program p(h:Real,y:Real):R->R
    sample b <- Normal(loc=h,scale=1.0)
    observe y <- Normal(loc=b,scale=1.0)
    return b
export p
"""
    program = loads(source).morphism
    zero = prior_draws(
        program,
        torch.zeros(1, 1),
        {"h": torch.tensor(0.0), "y": torch.tensor(5.0)},
        num_samples=5,
        seed=31,
    )
    shifted = prior_draws(
        program,
        torch.zeros(1, 1),
        {"h": torch.tensor(3.0), "y": torch.tensor(-100.0)},
        num_samples=5,
        seed=31,
    )
    torch.testing.assert_close(
        shifted.samples["b"] - zero.samples["b"], torch.full((1, 5), 3.0)
    )


def test_shared_geometry_preserves_registered_intervention_extraction():
    from quivers.effects import do
    from quivers.inference import LatentRegistry, trace

    program = loads(
        "object R:Real 1\nprogram p:R->R\n    sample z <- Normal(loc=0.0,scale=1.0)\n    return z\nexport p\n"
    ).morphism
    registry = LatentRegistry.from_model(program, set())
    with do({"z": torch.tensor([2.0])}):
        recorded = trace(program, torch.zeros(1, 1))
    assert registry.trace_draw(recorded)["z"].item() == 2.0
