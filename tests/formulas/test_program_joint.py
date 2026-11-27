"""A program block beside a family block in one joint model.

Oracle: the joint program's log density at fixed latents is the sum of the
blocks' log densities, each written out in torch; the shared group block
``(1 | p | g)`` draws one correlated pair of effects for the location of the
program and the location of the family. Readers keyed by response read each
block's own observation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quivers.formulas import MultivariateFit, bf, fit, formula_to_qvr, set_rescor
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry

from tests.formulas._program_cases import GAUSS, frame

SHARED = "~ x + (1 | p | g)"


def _joint():
    return (
        bf(program=GAUSS, parameters={"mu": SHARED}, response="y")
        + bf("z ~ x + (1 | p | g)")
        + set_rescor(False)
    )


def test_the_blocks_share_one_group_block() -> None:
    source = formula_to_qvr(_joint(), data=frame())
    assert source.count("LKJCholesky") == 1
    assert "response_y__intercept" in source
    assert "response_z__intercept" in source
    # The program's own scale keeps its law under its block's namespace.
    assert "sample response_y__sigma <- HalfNormal(scale=1.0)" in source


def test_readers_key_each_block_by_its_response() -> None:
    program = fit(_joint(), data=frame(), method="svi", num_steps=1, seed=0)
    formula = fit(
        bf("y ~ x + (1 | p | g)") + bf("z ~ x + (1 | p | g)") + set_rescor(False),
        data=frame(),
        priors={"response_y__sigma": "HalfNormal(1.0)"},
        method="svi",
        num_steps=1,
        seed=0,
    )
    assert isinstance(program, MultivariateFit)
    assert isinstance(formula, MultivariateFit)
    names = LatentRegistry.from_model(
        program.program, observed_names=set(program.observations)
    ).names
    assert set(names) == set(
        LatentRegistry.from_model(
            formula.program, observed_names=set(formula.observations)
        ).names
    )
    generator = torch.Generator().manual_seed(4)
    draws = {
        name: torch.rand(
            (1, 3, *program.draws(num_samples=1)[name].shape[2:]),
            generator=generator,
            dtype=torch.float64,
        )
        + 0.3
        for name in names
    }
    draws["cor_joint_shared_p"] = torch.eye(2, dtype=torch.float64).expand(1, 3, 2, 2)

    def pinned(fitted):
        return fitted.with_(
            posterior=MCMCResult(
                samples=draws,
                log_densities=torch.zeros(1, 3),
                acceptance_rates=torch.ones(1),
                divergence_counts=torch.zeros(1, dtype=torch.long),
                num_warmup=0,
                num_samples=3,
            )
        )

    for resp in (None, "y", "z"):
        scores = pinned(program).log_likelihood(resp=resp)
        expected = pinned(formula).log_likelihood(resp=resp)
        assert set(scores) == set(expected)
        for key, value in scores.items():
            torch.testing.assert_close(
                value.double(), expected[key].double(), rtol=1e-10, atol=0.0
            )
    only = pinned(program).log_likelihood(resp="y")
    assert set(only) == {"y"}


@pytest.mark.parametrize("student", [False, True])
def test_an_authored_observation_can_share_a_vector_residual(student: bool) -> None:
    source = GAUSS.replace("mu", "center").replace("sigma", "spread")
    if student:
        source = source.replace(
            "observe y : Trial <- Normal(loc=center, scale=spread)",
            "observe y : Trial <- StudentT(df=5.0, loc=center, scale=spread)",
        )
    other = (
        bf("z ~ 1", family="student_t", fixed_parameters={"nu": 5.0})
        if student
        else bf("z ~ 1")
    )
    model = bf(program=source, parameters={}, response="y") + other + set_rescor(True)
    rows = frame().assign(y=np.arange(-6, 6) / 8, z=np.arange(12)[::-1] / 8)
    fitted = fit(model, data=rows, method="svi", num_steps=1, seed=0)
    assert isinstance(fitted, MultivariateFit)
    tensor = torch.tensor
    values = {
        "response_y__center": tensor([[0.375]], dtype=torch.float64),
        "response_y__spread": tensor([[0.75]], dtype=torch.float64),
        "response_z__intercept": tensor([[0.125]], dtype=torch.float64),
        "response_z__sigma": tensor([[1.25]], dtype=torch.float64),
        "rescor": tensor([[[[1.0, 0.0], [0.6, 0.8]]]], dtype=torch.float64),
    }
    names = LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names
    assert set(names) == set(values)
    fitted = fitted.with_(
        posterior=MCMCResult(
            samples=values,
            log_densities=torch.zeros(1, 1),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=1,
        )
    )
    means = tensor([0.375, 0.125], dtype=torch.float64)
    scales = tensor([0.75, 1.25], dtype=torch.float64).diag()
    cholesky = scales @ values["rescor"][0, 0]
    observed = tensor(rows[["y", "z"]].to_numpy())

    def score(scale: torch.Tensor) -> torch.Tensor:
        if not student:
            return torch.distributions.MultivariateNormal(
                means, scale_tril=scale
            ).log_prob(observed)
        degrees = tensor(5.0, dtype=torch.float64)
        whitened = torch.linalg.solve_triangular(
            scale, (observed - means).T, upper=False
        )
        log_constant = (
            torch.lgamma((degrees + 2) / 2)
            - torch.lgamma(degrees / 2)
            - torch.log(degrees * torch.pi)
            - scale.diag().log().sum()
        )
        return log_constant - (degrees + 2) / 2 * torch.log1p(
            whitened.square().sum(0) / degrees
        )

    actual = next(iter(fitted.log_likelihood().values())).reshape(-1).double()
    torch.testing.assert_close(actual, score(cholesky), rtol=1e-10, atol=1e-10)
    assert float((actual - score(scales)).abs().max()) > 1e-2


def test_correlated_residuals_need_elliptical_families() -> None:
    with pytest.raises(ValueError, match="rescor requires"):
        formula_to_qvr(
            bf(program=GAUSS, parameters={"mu": "~ x"}, response="y")
            + bf("accept ~ x", family="bernoulli")
            + set_rescor(True),
            data=frame(),
        )
