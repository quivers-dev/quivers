"""A family, a nonlinear formula and a mixture are kernels: their programs agree.

Oracles: the formula fits of the same models. At shared latent values the
program with formulas has the log density of the formula fit (the Gaussian
program against ``y ~ x + (1 | g)`` with ``sigma ~ z``; the decay program
against the nonlinear formula ``y ~ a * exp(0 - b * x)``), and the two-class
program has the pointwise log likelihood of ``mixture(gaussian, gaussian)``
row by row. Each comparison is paired with a model that differs in one
respect and must not agree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quivers.formulas import fit, mixture

from tests.formulas._program_cases import (
    DECAY,
    GAUSS,
    MIX2,
    frame,
    latent_names,
    pin,
    run,
)

TOLERANCE = 1e-10


def _svi(*arguments, **keywords):
    return fit(*arguments, data=frame(), method="svi", num_steps=1, seed=0, **keywords)


def _values(names, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.rand(
            (3,) if name.startswith("z_") else (),
            generator=generator,
            dtype=torch.float64,
        )
        + 0.2
        for name in names
    }


def test_the_gaussian_program_is_the_gaussian_family() -> None:
    program = _svi(
        program=GAUSS,
        parameters={"mu": "~ x + (1 | g)", "sigma": "~ z"},
        response="y",
    )
    family = _svi("y ~ x + (1 | g)", distributional={"sigma": "~ z"})
    names = latent_names(program)
    assert set(names) == set(latent_names(family))
    values = _values(names, seed=1)
    joint = float(run(program, values).log_joint.sum())
    assert joint == pytest.approx(
        float(run(family, values).log_joint.sum()), rel=TOLERANCE
    )
    other = _svi("y ~ x + (1 | g)", distributional={"sigma": "~ x"})
    assert abs(joint - float(run(other, values).log_joint.sum())) > 1e-3


def test_the_decay_program_is_the_nonlinear_formula() -> None:
    program = _svi(program=DECAY, parameters={"a": "~ 1", "b": "~ 1"}, response="y")
    formula = _svi(
        "y ~ a * exp(0 - b * x)",
        nonlinear={"a": "~ 1", "b": "~ 1"},
        priors={"sigma": "HalfNormal(1.0)"},
    )
    pairs = {
        "dpar_a_intercept": "nlpar_a_intercept",
        "dpar_b_intercept": "nlpar_b_intercept",
        "sigma": "sigma",
    }
    assert set(latent_names(program)) == set(pairs)
    assert set(latent_names(formula)) == set(pairs.values())
    values = {"dpar_a_intercept": 1.3, "dpar_b_intercept": 0.4, "sigma": 0.8}
    tensors = {
        name: torch.tensor(value, dtype=torch.float64) for name, value in values.items()
    }
    joint = float(run(program, tensors).log_joint.sum())
    renamed = {pairs[name]: value for name, value in tensors.items()}
    assert joint == pytest.approx(
        float(run(formula, renamed).log_joint.sum()), rel=TOLERANCE
    )
    growth = _svi(
        "y ~ a * exp(b * x)",
        nonlinear={"a": "~ 1", "b": "~ 1"},
        priors={"sigma": "HalfNormal(1.0)"},
    )
    assert abs(joint - float(run(growth, renamed).log_joint.sum())) > 1e-3


def test_the_two_class_program_is_the_gaussian_mixture_row_by_row() -> None:
    program = _svi(program=MIX2, parameters={}, response="y")
    family = _svi("y ~ 1", family=mixture("gaussian", "gaussian", order="none"))
    logits = np.array([[0.3, -0.4]])
    centers = np.array([[-0.5, 1.7]])
    sigma = np.array([0.9])
    weights = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    scores = pin(
        program, {"logits": logits, "centers": centers, "sigma": sigma}
    ).log_likelihood()["y"]
    names = set(latent_names(family))
    assert names == {
        "dpar_mu1_intercept",
        "dpar_mu2_intercept",
        "sigma1",
        "sigma2",
        "theta",
    }
    mixed = pin(
        family,
        {
            "dpar_mu1_intercept": centers[:, 0],
            "dpar_mu2_intercept": centers[:, 1],
            "sigma1": sigma,
            "sigma2": sigma,
            "theta": weights,
        },
    ).log_likelihood()["y"]
    torch.testing.assert_close(scores.double(), mixed.double(), rtol=1e-6, atol=1e-6)
    wider = pin(
        family,
        {
            "dpar_mu1_intercept": centers[:, 0],
            "dpar_mu2_intercept": centers[:, 1],
            "sigma1": sigma,
            "sigma2": 2.0 * sigma,
            "theta": weights,
        },
    ).log_likelihood()["y"]
    assert float((scores - wider).abs().max()) > 1e-3
