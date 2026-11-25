"""Every compiled program knows the laws its terms draw from."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.continuous.family_spec import FAMILY_REGISTRY
from quivers.dsl.ast_nodes import ProgramDecl, SampleStep
from quivers.family_catalog import FamilyCatalog
from quivers.formulas import (
    BayesianFit,
    MultivariateFit,
    PriorSpec,
    bf,
    fit,
    mixture,
    set_rescor,
)
from quivers.formulas._latent_effects import (
    FieldTerm,
    LatentTerm,
    car_catalog,
    car_graph,
)
from quivers.formulas._predictor import OffsetTerm
from quivers.formulas._term import join_catalogs
from quivers.inference.trace import trace
from quivers.qiec.families import FAMILIES

ROWS = 24
FRAME = pd.DataFrame(
    {
        "y1": np.random.default_rng(1).poisson(3.0, size=ROWS),
        "y2": np.random.default_rng(2).poisson(2.0, size=ROWS),
        "y3": np.random.default_rng(3).normal(size=ROWS).round(3),
        "x": np.random.default_rng(4).normal(size=ROWS).round(3),
        "t": list(range(12)) * 2,
        "ser": ["u"] * 12 + ["v"] * 12,
        "four": [f"l{i % 4}" for i in range(ROWS)],
        "five": [f"k{i % 5}" for i in range(ROWS)],
    }
)


def _cycle(prefix: str, count: int) -> pd.DataFrame:
    matrix = np.zeros((count, count))
    for index in range(count):
        matrix[index, (index + 1) % count] = matrix[(index + 1) % count, index] = 1.0
    labels = [f"{prefix}{index}" for index in range(count)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


DATA2 = {"W4": _cycle("l", 4), "W5": _cycle("k", 5)}


def _proper(adjacency: np.ndarray, scale: float, alpha: float) -> torch.Tensor:
    precision = (np.diag(adjacency.sum(1)) - alpha * adjacency) / scale**2
    return torch.tensor(precision, dtype=torch.float64)


def _log_joint(
    fitted: BayesianFit | MultivariateFit, latents: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    value = trace(
        fitted.program, torch.zeros(1, 1), dict(fitted.observations) | dict(latents)
    ).log_joint
    assert value is not None
    return value.sum().double()


def test_joint_responses_draw_from_their_own_graphs():
    """The joint density is the product of dense CAR and Poisson densities."""
    fitted = fit(
        bf("y1 ~ 1 + car(W4, gr = four)", family="poisson")
        + bf("y2 ~ 1 + car(W5, gr = five)", family="poisson")
        + set_rescor(False),
        data=FRAME,
        data2=DATA2,
        method="svi",
        num_steps=1,
        seed=0,
    )
    generator = torch.Generator().manual_seed(7)
    expected = torch.zeros((), dtype=torch.float64)
    latents: dict[str, torch.Tensor] = {}
    for response, matrix, column, (scale, alpha, intercept) in (
        ("y1", "W4", "four", (0.7, 0.4, 0.3)),
        ("y2", "W5", "five", (1.3, 0.8, -0.2)),
    ):
        adjacency = DATA2[matrix].to_numpy()
        field = torch.randn(len(adjacency), generator=generator, dtype=torch.float64)
        prefix = f"response_{response}__"
        latents |= {
            prefix + "intercept": torch.tensor(intercept, dtype=torch.float64),
            prefix + "sdcar": torch.tensor(scale, dtype=torch.float64),
            prefix + "car": torch.tensor(alpha, dtype=torch.float64),
            prefix + "rcar": field,
        }
        codes = [int(label[1:]) for label in FRAME[column]]
        rate = torch.exp(intercept + field[codes])
        counts = torch.tensor(FRAME[response].to_numpy(), dtype=torch.float64)
        expected = (
            expected
            + torch.distributions.Normal(0.0, 5.0).log_prob(torch.tensor(intercept))
            + torch.distributions.HalfNormal(1.0).log_prob(torch.tensor(scale))
            + torch.distributions.MultivariateNormal(
                torch.zeros(len(adjacency), dtype=torch.float64),
                precision_matrix=_proper(adjacency, scale, alpha),
            ).log_prob(field)
            + torch.distributions.Poisson(rate).log_prob(counts).sum()
        )
    torch.testing.assert_close(
        _log_joint(fitted, latents), expected, rtol=1e-5, atol=1e-4
    )


@pytest.mark.parametrize(
    "term",
    [
        "car(W4, gr = four, type = 'esicar')",
        "car(W4, gr = four, type = 'icar')",
        "car(W4, gr = four, type = 'bym2')",
        "ar(time = t, gr = ser)",
        "cosy(time = t, gr = ser)",
        "unstr(time = t, gr = ser)",
    ],
)
def test_joint_fields_have_the_density_of_the_single_response_model(term):
    """A response's factor in the joint density is its own model's density."""
    options = {"data": FRAME, "data2": DATA2, "method": "svi", "num_steps": 1}
    single = fit(f"y1 ~ x + {term}", family="poisson", seed=0, **options)
    joint = fit(
        bf(f"y1 ~ x + {term}", family="poisson") + bf("y3 ~ 1") + set_rescor(False),
        seed=0,
        **options,
    )
    draws = {
        name: value[0, 0]
        for name, value in single.draws(num_samples=1, seed=3).items()
        if name in _sampled(single)
    }
    alone = _log_joint(single, draws)
    other = {
        "response_y3__intercept": torch.tensor(0.2, dtype=torch.float64),
        "response_y3__sigma": torch.tensor(0.9, dtype=torch.float64),
    }
    both = _log_joint(
        joint,
        other | {f"response_y1__{name}": value for name, value in draws.items()},
    )
    response = torch.tensor(FRAME["y3"].to_numpy(), dtype=torch.float64)
    rest = (
        torch.distributions.Normal(0.0, 5.0).log_prob(torch.tensor(0.2))
        + torch.distributions.HalfCauchy(2.0).log_prob(torch.tensor(0.9))
        + torch.distributions.Normal(0.2, 0.9).log_prob(response).sum()
    )
    torch.testing.assert_close(both, alone + rest.double(), rtol=1e-5, atol=1e-4)


def _sampled(fitted: BayesianFit) -> set[str]:
    assert fitted.prediction_plan is not None
    program = next(
        item
        for item in fitted.prediction_plan.module.statements
        if isinstance(item, ProgramDecl)
    )
    return {step.vars[0] for step in program.draws if isinstance(step, SampleStep)}


def test_an_exact_process_predicts_under_configured_priors():
    """Checking the specialized program finds the law of its fresh inputs."""
    fitted = fit(
        "y3 ~ gp(x)",
        data=FRAME,
        method="svi",
        num_steps=1,
        seed=0,
        priors=PriorSpec(distribution="HalfNormal(0.5)", class_="sdgp"),
    )
    fresh = pd.DataFrame({"x": [0.1, 0.2, 2.0]})
    assert fitted.posterior_epred(new_data=fresh, num_samples=2).shape == (1, 2, 3)


def test_an_exact_process_of_a_mixture_component_predicts_new_rows():
    """Each component draws its function from its own conditional law."""
    fitted = fit(
        "y3 ~ gp(x)",
        data=FRAME,
        family=mixture("gaussian", "gaussian"),
        method="svi",
        num_steps=1,
        seed=0,
    )
    fresh = pd.DataFrame({"x": [0.1, 0.2, 2.0]})
    assert fitted.posterior_epred(new_data=fresh, num_samples=2).shape == (1, 2, 3)


def _car(count: int) -> FamilyCatalog:
    adjacency = _cycle("l", count)
    return car_catalog(car_graph(adjacency.to_numpy(), tuple(adjacency.index), "icar"))


def test_renamed_catalogs_share_their_laws():
    """Renaming changes the name of a law and nothing else."""
    original = _car(4)
    renamed = original.renamed({"SparseICAR": "mine"})
    assert {record.name for record in renamed.definitions} == {"SparseCAR", "mine"}
    assert {record.name for record in original.definitions} == {
        "SparseCAR",
        "SparseICAR",
    }
    arguments: Mapping[str, torch.Tensor] = {
        "scale": torch.tensor(0.8),
        "anchor": torch.zeros(4),
    }
    value = torch.tensor([0.5, -0.5, 1.0, -1.0])
    torch.testing.assert_close(
        renamed.build("mine", arguments).log_prob(value),
        original.build("SparseICAR", arguments).log_prob(value),
    )
    assert renamed.support("mine") is original.support("SparseICAR")
    with pytest.raises(ValueError, match="already defined"):
        original.renamed({"SparseICAR": "SparseCAR"})


def test_joined_catalogs_hold_every_law_once():
    """A term without laws of its own leaves the response catalog unchanged."""

    lawless = OffsetTerm(prefix="", columns=())
    adjacency = _cycle("l", 4)
    labels = tuple(adjacency.index)
    spatial = FieldTerm(
        term=LatentTerm(
            name="car",
            kind="car",
            car_type="icar",
            graph=car_graph(adjacency.to_numpy(), labels, "icar"),
        ),
        scale_prior="HalfNormal(1.0)",
        correlation_prior="LKJCholesky(1.0)",
    )
    base = _car(5).renamed({"SparseCAR": "a", "SparseICAR": "b"})
    assert join_catalogs(base, [lawless]) is base
    assert join_catalogs(None, [lawless]) is None
    joined = join_catalogs(base, [lawless, spatial])
    assert joined is not None
    assert {record.name for record in joined.definitions} == {
        "a",
        "b",
        "SparseCAR",
        "SparseICAR",
    }


def test_renamed_catalogs_rename_native_records():
    """A native record keeps its law under another name."""
    gamma = FAMILY_REGISTRY["Gamma"].with_(name="Mine")
    catalog = FamilyCatalog().with_family(
        FAMILIES["Gamma"].with_(name="Mine", targets={}), gamma
    )
    renamed = catalog.renamed({"Mine": "Yours"})
    arguments = {
        name: torch.tensor(value)
        for name, value in zip(gamma.param_names, (2.0, 3.0), strict=True)
    }
    value = torch.tensor([0.4, 1.2])
    torch.testing.assert_close(
        renamed.build("Yours", arguments).log_prob(value),
        torch.distributions.Gamma(2.0, 3.0).log_prob(value),
    )
