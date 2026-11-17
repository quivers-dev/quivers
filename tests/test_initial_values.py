"""Named initial values reach inference in the declared latent support."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.dsl import loads
from quivers.formulas import fit
from quivers.inference import MCMC, HMCKernel, NUTSKernel, trace
from quivers.inference.guides.normal import AutoNormalGuide
from quivers.inference.mcmc import MCMCCheckpoint


_SOURCE = """
object Obs : FinSet 3
object Comp : FinSet 3
program p : Obs -> Obs
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=1.0)
    sample fraction <- Uniform(low=-2.0, high=3.0)
    sample mix : Comp <- Dirichlet(concentration=1.0)
    sample by_row : Obs <- Normal(loc=0.0, scale=1.0)
    observe y : Obs <- Normal(loc=mu, scale=sigma)
    return mu
export p
"""
_X = torch.zeros(1, 1)
_DATA = {"y": torch.tensor([-0.2, 0.1, 0.4])}


@pytest.fixture(scope="module")
def model() -> MonadicProgram:
    result = loads(_SOURCE).morphism
    assert isinstance(result, MonadicProgram)
    return result


def _driver(**kwargs):
    kwargs.setdefault("num_chains", 1)
    return MCMC(
        HMCKernel(step_size=0.03, num_steps=2),
        num_warmup=0,
        num_samples=2,
        seed=83,
        **kwargs,
    )


def _guide(model, values):
    return AutoNormalGuide(
        model,
        {"y"},
        x=_X,
        observations=_DATA,
        initial_values=values,
    )


def _record_kernel_starts(
    monkeypatch, kernel_type: type[HMCKernel] | type[NUTSKernel] = HMCKernel
):
    starts = []
    original = kernel_type.init

    def recorded(self, registry, model, x, observations, initial_position):
        values, _ = registry.to_constrained(
            registry.unflatten_unconstrained(initial_position)
        )
        starts.append({name: value.detach().clone() for name, value in values.items()})
        return original(self, registry, model, x, observations, initial_position)

    monkeypatch.setattr(kernel_type, "init", recorded)
    return starts


def test_guide_centers_match_analytic_constraint_coordinates(model):
    guide = _guide(
        model,
        {
            "mu": 0.7,
            "sigma": 1.2,
            "fraction": 0.3,
            "mix": torch.tensor([0.2, 0.3, 0.5]),
            "by_row": torch.tensor([1.0, -0.5, 2.0]),
        },
    )
    expected = {
        "mu": torch.tensor([0.7]),
        "sigma": torch.tensor([math.log(1.2)]),
        "fraction": torch.tensor([math.log(0.46 / 0.54)]),
        # The first stick has the dimension-dependent log(2) offset.
        "mix": torch.tensor([math.log(0.2 / 0.8) + math.log(2), math.log(0.3 / 0.5)]),
        "by_row": torch.tensor([[1.0], [-0.5], [2.0]]),
    }
    for name, value in expected.items():
        torch.testing.assert_close(guide._loc(name), value)
        torch.testing.assert_close(guide._scale(name), torch.full_like(value, 0.1))


@pytest.mark.parametrize("strategy", ["zero", "jitter", "prior"])
def test_named_values_reach_first_kernel_after_strategy(model, monkeypatch, strategy):
    starts = _record_kernel_starts(monkeypatch)
    values = {
        "mu": 0.7,
        "sigma": 1.2,
        "fraction": 0.3,
        "mix": torch.tensor([0.2, 0.3, 0.5]),
        "by_row": 1.5,
    }
    result = _driver(init_strategy=strategy, initial_values=values, num_chains=2).run(
        model, _X, _DATA
    )
    assert len(starts) == 2
    assert starts[0]["by_row"].shape == (3,)
    for start in starts:
        for name, value in values.items():
            torch.testing.assert_close(
                start[name], torch.as_tensor(value).expand_as(start[name])
            )
    assert all(bool(torch.isfinite(value).all()) for value in result.samples.values())


@pytest.mark.parametrize("strategy", ["zero", "jitter", "prior"])
def test_partial_values_leave_other_seeded_strategy_coordinates_unchanged(
    model, monkeypatch, strategy
):
    starts = _record_kernel_starts(monkeypatch)
    _driver(init_strategy=strategy).run(model, _X, _DATA)
    _driver(init_strategy=strategy, initial_values={"mu": 2.0}).run(model, _X, _DATA)
    for name in starts[0]:
        if name != "mu":
            assert torch.equal(starts[0][name], starts[1][name])
    assert starts[1]["mu"].item() == pytest.approx(2.0)


def test_named_values_override_seeded_guide_draw_without_replacing_other_sites(
    model, monkeypatch
):
    starts = _record_kernel_starts(monkeypatch)
    guide = _guide(model, {"mu": 0.7, "sigma": 1.2})
    _driver(init_strategy="guide").run(model, _X, _DATA, guide=guide)
    _driver(init_strategy="guide", initial_values={"mu": 2.0}).run(
        model, _X, _DATA, guide=guide
    )
    assert starts[1]["mu"].item() == pytest.approx(2.0)
    for name in starts[0]:
        if name != "mu":
            assert torch.equal(starts[0][name], starts[1][name])


def test_scalar_broadcast_and_exact_plate_values_are_distinct(model):
    scalar = _guide(model, {"by_row": 1.5})
    torch.testing.assert_close(scalar._loc("by_row"), torch.full((3, 1), 1.5))
    vector = _guide(model, {"by_row": torch.tensor([1.0, -2.0, 3.0])})
    torch.testing.assert_close(
        vector._loc("by_row"), torch.tensor([[1.0], [-2.0], [3.0]])
    )
    simplex = _guide(model, {"mix": 1 / 3})
    torch.testing.assert_close(simplex._loc("mix"), torch.zeros(2), atol=2e-7, rtol=0)
    # An omitted positive site's center remains zero in log coordinates.
    assert torch.equal(scalar._loc("sigma"), torch.zeros(1))


_INVALID = [
    ("missing", 1.0),
    ("y", 0.0),
    ("mu", True),
    ("mu", float("nan")),
    ("mu", float("inf")),
    ("mu", float("-inf")),
    ("mu", torch.tensor(1 + 2j)),
    ("mu", torch.tensor([0.1])),
    ("sigma", 0.0),
    ("sigma", -1.0),
    ("fraction", -2.0),
    ("fraction", 3.0),
    ("fraction", 3.1),
    ("by_row", torch.tensor([True, False, True])),
    ("by_row", torch.ones(2)),
    ("by_row", torch.ones(3, 1)),
    ("by_row", torch.tensor([0.0, float("nan"), 1.0])),
    ("mix", torch.tensor([0.0, 0.5, 0.5])),
    ("mix", torch.tensor([1.0, 0.0, 0.0])),
    ("mix", torch.tensor([-0.1, 0.5, 0.6])),
    ("mix", torch.tensor([0.2, 0.3, 0.4])),
    ("mix", torch.tensor([0.5, 0.5])),
]


@pytest.mark.parametrize("engine", ["guide", "mcmc"])
@pytest.mark.parametrize("name,value", _INVALID)
def test_invalid_values_fail_before_inference(model, monkeypatch, engine, name, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid initial values reached a kernel transition")

    monkeypatch.setattr(HMCKernel, "init", forbidden)
    with pytest.raises(ValueError, match="initial|unknown"):
        if engine == "guide":
            _guide(model, {name: value})
        else:
            _driver(initial_values={name: value}).run(model, _X, _DATA)


def test_initial_values_own_copies_without_inheriting_autograd(model, monkeypatch):
    starts = _record_kernel_starts(monkeypatch)
    value = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    values = {"by_row": value}
    driver = _driver(initial_values=values)
    guide = _guide(model, values)
    with torch.no_grad():
        value.fill_(8.0)
    values["by_row"] = torch.full((3,), -9.0)
    driver.run(model, _X, _DATA)
    expected = torch.tensor([0.1, 0.2, 0.3])
    torch.testing.assert_close(starts[0]["by_row"], expected)
    torch.testing.assert_close(guide._loc("by_row").reshape(-1), expected)
    assert guide._loc("by_row").is_leaf
    guide._loc("by_row").sum().backward()
    assert value.grad is None
    assert guide._loc("by_row").grad is not None


def test_named_initialization_is_seeded_without_changing_global_rng(model):
    driver = _driver(initial_values={"sigma": 1.2, "mu": 0.7})
    rng = torch.get_rng_state().clone()
    first = driver.run(model, _X, _DATA)
    second = driver.run(model, _X, _DATA)
    assert torch.equal(torch.get_rng_state(), rng)
    for name in first.samples:
        assert torch.equal(first.samples[name], second.samples[name])


def test_checkpoint_requires_matching_named_initial_values(model, tmp_path):
    path = tmp_path / "checkpoint"
    values = {"sigma": 1.2, "mu": 0.7}
    expected = _driver(
        initial_values=values, checkpoint=MCMCCheckpoint(directory=str(path), every=1)
    ).run(model, _X, _DATA)
    # Dict insertion order is not part of the initialization contract.
    resumed = _driver(initial_values={"mu": 0.7, "sigma": 1.2}).resume(
        path, model, _X, _DATA
    )
    for name in expected.samples:
        assert torch.equal(expected.samples[name], resumed.samples[name])
    for changed in ({"sigma": 1.2, "mu": 0.8}, {"sigma": 1.2}, None):
        with pytest.raises(ValueError, match="initialization|settings"):
            _driver(initial_values=changed).resume(path, model, _X, _DATA)


def test_interrupted_named_initialization_resumes_bitwise(model, tmp_path):
    values = {"sigma": 1.2, "mu": 0.7}
    expected = _driver(initial_values=values).run(model, _X, _DATA)

    class Interrupted(Exception):
        pass

    def interrupt(chain, iteration, state, phase):
        if iteration == 1:
            raise Interrupted

    with pytest.raises(Interrupted):
        _driver(
            initial_values=values,
            checkpoint=MCMCCheckpoint(directory=str(tmp_path), every=1),
            callback=interrupt,
        ).run(model, _X, _DATA)
    resumed = _driver(initial_values=values).resume(tmp_path, model, _X, _DATA)
    for name in expected.samples:
        assert torch.equal(expected.samples[name], resumed.samples[name])
    assert torch.equal(expected.log_densities, resumed.log_densities)
    for name in expected.sample_stats:
        assert torch.equal(expected.sample_stats[name], resumed.sample_stats[name])


def test_formula_user_initial_values_reach_unoptimized_guide():
    frame = pd.DataFrame({"y": [0.1, 0.4, 0.8], "x": [-1.0, 0.0, 1.0]})
    result = fit(
        "y ~ x",
        data=frame,
        method="svi",
        num_steps=0,
        initial_values={"intercept": 0.3, "beta_x": -0.2, "sigma": 0.7},
    )
    assert isinstance(result.posterior, AutoNormalGuide)
    torch.testing.assert_close(result.posterior._loc("intercept"), torch.tensor([0.3]))
    torch.testing.assert_close(result.posterior._loc("beta_x"), torch.tensor([-0.2]))
    torch.testing.assert_close(
        result.posterior._loc("sigma"), torch.tensor([math.log(0.7)])
    )


@pytest.mark.parametrize("method", ["hmc", "nuts"])
def test_formula_user_initial_values_reach_first_kernel(monkeypatch, method):
    starts = _record_kernel_starts(
        monkeypatch, HMCKernel if method == "hmc" else NUTSKernel
    )
    result = fit(
        "y ~ x",
        data=pd.DataFrame({"y": [0.1, 0.4, 0.8], "x": [-1.0, 0.0, 1.0]}),
        method=method,
        num_warmup=0,
        num_samples=1,
        num_chains=1,
        initial_values={"intercept": 0.3, "beta_x": -0.2, "sigma": 0.7},
    )
    assert len(starts) == 1
    for name, expected in {"intercept": 0.3, "beta_x": -0.2, "sigma": 0.7}.items():
        assert starts[0][name].item() == pytest.approx(expected)
    assert set(result.draws()) == {"intercept", "beta_x", "sigma"}


def test_regressed_shift_initial_center_is_inside_observation_support():
    frame = pd.DataFrame(
        {
            "y": [0.01, 0.04, 0.02, 0.08],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [2.0, -1.0, 3.0, 0.0],
        }
    )
    result = fit(
        "y ~ x",
        data=frame,
        family="shifted_lognormal",
        distributional={"ndt": "~ z"},
        fixed_parameters={"sigma": 0.5},
        method="svi",
        num_steps=0,
    )
    assert isinstance(result.posterior, AutoNormalGuide)
    guide = result.posterior
    centers = {}
    for name in guide._registry.names:
        site = guide._registry[name]
        center = site.bijector(guide._loc(name))
        assert isinstance(center, torch.Tensor)
        centers[name] = center.reshape(site.constrained_shape)
    intercept = centers["dpar_ndt_intercept"]
    slope = centers["dpar_ndt_beta_z"]
    shifts = torch.exp(
        intercept + slope * torch.tensor(frame.z.to_numpy(), dtype=torch.float32)
    )
    assert bool((shifts > 0).all())
    assert bool((shifts < torch.tensor(frame.y.to_numpy())).all())
    recorded = trace(result.program, _X, dict(result.observations) | centers)
    assert bool(torch.isfinite(recorded.sites["y"].log_prob).all())


@pytest.mark.parametrize("method", ["hmc", "nuts"])
def test_regressed_shift_automatic_initialization_reaches_supported_kernel_state(
    monkeypatch, method
):
    starts = _record_kernel_starts(
        monkeypatch, HMCKernel if method == "hmc" else NUTSKernel
    )
    frame = pd.DataFrame(
        {
            "y": [0.01, 0.04, 0.02, 0.08],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [2.0, -1.0, 3.0, 0.0],
        }
    )
    result = fit(
        "y ~ x",
        data=frame,
        family="shifted_lognormal",
        distributional={"ndt": "~ z"},
        fixed_parameters={"sigma": 0.5},
        method=method,
        num_warmup=0,
        num_samples=1,
        num_chains=1,
    )
    assert len(starts) == 1
    center = starts[0]
    shifts = torch.exp(
        center["dpar_ndt_intercept"]
        + center["dpar_ndt_beta_z"]
        * torch.tensor(frame.z.to_numpy(), dtype=torch.float32)
    )
    assert bool((shifts > 0).all())
    assert bool((shifts < torch.tensor(frame.y.to_numpy())).all())
    recorded = trace(result.program, _X, dict(result.observations) | center)
    assert bool(torch.isfinite(recorded.sites["y"].log_prob).all())


def test_formula_explicit_shift_initial_center_overrides_automatic_default():
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0.01, 0.02, 0.04], "z": [-1.0, 0.0, 1.0]}),
        family="shifted_lognormal",
        distributional={"ndt": "~ z"},
        fixed_parameters={"sigma": 0.5},
        method="svi",
        num_steps=0,
        initial_values={"dpar_ndt_intercept": -9.0, "dpar_ndt_beta_z": 0.1},
    )
    assert isinstance(result.posterior, AutoNormalGuide)
    torch.testing.assert_close(
        result.posterior._loc("dpar_ndt_intercept"), torch.tensor([-9.0])
    )
    torch.testing.assert_close(
        result.posterior._loc("dpar_ndt_beta_z"), torch.tensor([0.1])
    )
