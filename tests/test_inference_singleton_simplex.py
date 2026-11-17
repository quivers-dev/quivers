"""Singleton simplices retain event axes without artificial free coordinates."""

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, loads, parse
from quivers.formulas import families, fit
from quivers.inference import (
    AutoNormal,
    AutoDelta,
    AutoMultivariateNormal,
    AutoLowRankMVN,
    AutoIAFNormal,
    AutoNeuralSplineGuide,
    AutoStructured,
    AutoLaplace,
)
from quivers.inference.registry import LatentRegistry, LatentSite
from quivers.qiec.program_runtime import run_program

SOURCE = """object K:FinSet 1
object Group:FinSet 2
object Out:Real 1
program model:Out->Out
    sample p:K <- Dirichlet(concentration=[2.0])
    sample profiles:Group <- Dirichlet(concentration=[3.0]) [over=K,iid_over=Group]
    sample a <- Normal(loc=0.0,scale=1.0)
    sample b <- Normal(loc=0.0,scale=1.0)
    return p
export model
"""


def test_singleton_simplex_registry_and_source_have_exact_zero_coordinate_layout():
    program = loads(SOURCE).morphism
    registry = LatentRegistry.from_model(program, set())
    assert registry["p"].constrained_shape == (1,)
    assert registry["p"].unconstrained_shape == (0,)
    assert registry["profiles"].constrained_shape == (2, 1)
    assert registry["profiles"].unconstrained_shape == (2, 0)
    assert not registry["p"].is_scalar and not registry["profiles"].is_scalar
    assert registry["a"].is_scalar and registry["b"].is_scalar
    assert registry.total_unconstrained_dim == 2
    z = registry.randn_unconstrained()
    values, jacobians = registry.to_constrained(z)
    torch.testing.assert_close(values["p"], torch.ones(1))
    torch.testing.assert_close(values["profiles"], torch.ones(2, 1))
    assert jacobians["p"].sum() == jacobians["profiles"].sum() == 0
    inverse, _ = registry.to_unconstrained(values)
    for key in z:
        torch.testing.assert_close(inverse[key], z[key])
    flat = registry.flatten_unconstrained(z)
    assert flat.shape == (2,)
    restored = registry.unflatten_unconstrained(flat)
    for key in z:
        torch.testing.assert_close(restored[key], z[key])
    checked = Compiler(parse(SOURCE)).qiec_module
    result = run_program(
        checked,
        "model",
        data={"out": (0.0,)},
        sites={"p": (1.0,), "profiles": ((1.0,), (1.0,)), "a": 0.0, "b": 0.0},
    )
    assert result.value == (1.0,)
    assert result.log_joint == pytest.approx(
        -torch.log(torch.tensor(2 * torch.pi)).item()
    )


@pytest.mark.parametrize(
    "guide_type",
    [
        AutoNormal,
        AutoDelta,
        AutoMultivariateNormal,
        AutoLowRankMVN,
        AutoIAFNormal,
        AutoNeuralSplineGuide,
        AutoStructured,
        AutoLaplace,
    ],
)
def test_guides_keep_singleton_event_axes_and_finite_zero_coordinate_density(
    guide_type,
):
    program = loads(SOURCE).morphism
    options = (
        {"rank": 1}
        if guide_type is AutoLowRankMVN
        else {"conditionals": "normal"}
        if guide_type is AutoStructured
        else {}
    )
    guide = guide_type(program, observed_names=set(), **options)
    x = torch.zeros(1, 1)
    values = guide.rsample(x)
    assert values["p"].shape == (1, 1)
    assert values["profiles"].shape == (2, 1)
    torch.testing.assert_close(values["p"], torch.ones(1, 1))
    torch.testing.assert_close(values["profiles"], torch.ones(2, 1))
    assert torch.isfinite(guide.log_prob(x, values)).all()
    assert torch.isfinite(program.log_joint(x, values)).all()


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("grouped", [False, True])
def test_singleton_baseline_inference_retains_labels_and_no_free_simplex_parameter(
    method, grouped
):
    family = families["cox"].with_baseline(df=1, degree=0, by="g" if grouped else None)
    options = (
        {"num_steps": 1}
        if method == "svi"
        else {"num_warmup": 1, "num_samples": 2, "num_chains": 1}
    )
    result = fit(
        "y~1",
        data=pd.DataFrame({"y": [0.2, 0.7, 1.0], "g": ["a", "a", "b"]}),
        family=family,
        method=method,
        seed=7,
        **options,
    )
    registry = LatentRegistry.from_model(result.program, set(result.observations))
    assert registry["sbhaz"].flat_length == 0
    assert registry.total_unconstrained_dim == 1
    weights = result.draws(num_samples=2, seed=3)["sbhaz"]
    assert weights.shape == ((1, 2, 2, 1) if grouped else (1, 2, 1))
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_scalar_proof_preserves_singleton_matrix_event_axes():
    from torch.distributions import constraints

    from quivers.continuous._matrix import PositiveDefiniteTransform

    program = loads(SOURCE).morphism
    simplex = LatentRegistry.from_model(program, set())["p"]
    matrix = LatentSite(
        name="matrix",
        morphism=simplex.morphism,
        support=constraints.positive_definite,
        bijector=PositiveDefiniteTransform(1),
        constrained_dim=1,
        unconstrained_dim=1,
        is_plate=False,
        plate_index_size=0,
        spec_index=0,
        flat_offset=0,
        flat_length=1,
    )
    assert matrix.constrained_shape == (1, 1)
    assert not matrix.is_scalar
    registry = LatentRegistry({"matrix": matrix}, frozenset(), program)
    values, _ = registry.to_constrained({"matrix": torch.zeros(1)})
    assert values["matrix"].shape == (1, 1)
    torch.testing.assert_close(values["matrix"], torch.ones(1, 1))
    inverse, _ = registry.to_unconstrained(values)
    torch.testing.assert_close(inverse["matrix"], torch.zeros(1))
