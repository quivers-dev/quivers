"""Group-block laws shared across the responses of a joint model.

A block of coefficients that share an identifier across responses has one
law: one level covariance, one ``by`` partition, one Student-t mixing
multiplier per level. Each density test compares the emitted joint program
with a density written directly in torch at a pinned point, checks the
gradient, and checks that dropping the factor under test moves the density.

Tolerance, fixed before comparing: the programs run in double precision and
each density is a sum of a few dozen terms of order one, so the comparison
uses torch's default double tolerance (relative 1e-7, absolute 1e-7); the
factor under test must move the density by more than 1e-3.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.dsl.ast_nodes import ObserveStep, ProgramDecl, SampleStep
from quivers.formulas import GroupCovariance, bf, fit, set_rescor
from quivers.formulas._fit import _prepare_formula
from quivers.formulas._group_law import (
    ByLKJCorrelation,
    LKJCorrelation,
    group_coefficients,
)
from quivers.formulas._joint_compile import emit_joint
from quivers.formulas._response import response_tensor
from quivers.inference import trace

LEVELS = ("a", "b", "c")
FULL = GroupCovariance(
    levels=(*LEVELS, "d"),
    matrix=(
        (1.0, 0.3, 0.1, 0.2),
        (0.3, 1.0, 0.2, 0.1),
        (0.1, 0.2, 1.0, 0.4),
        (0.2, 0.1, 0.4, 1.0),
    ),
)
DEFICIENT = GroupCovariance(
    levels=LEVELS,
    matrix=((1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    rank=2,
)
CHOLESKY = ((1.0, 0.0), (0.3, math.sqrt(0.91)))
OTHER_CHOLESKY = ((1.0, 0.0), (-0.5, math.sqrt(0.75)))


@pytest.fixture(autouse=True)
def double_literals():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _frame(rows: int = 12) -> pd.DataFrame:
    generator = torch.Generator().manual_seed(11)
    groups = [LEVELS[index % 3] for index in range(rows)]
    return pd.DataFrame(
        {
            "y": torch.randn(rows, generator=generator).tolist(),
            "z": torch.randn(rows, generator=generator).tolist(),
            "w": torch.randn(rows, generator=generator).tolist(),
            "g": groups,
            # by is constant within a level of g.
            "fb": [{"a": "m", "b": "n", "c": "m"}[level] for level in groups],
        }
    )


def _observations(emission, prepared):
    values = dict(emission.fixed_inputs)
    observed = {
        step.vars[0]
        for statement in emission.module.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(step, ObserveStep)
    }
    for name, response in emission.responses.items():
        local = response.lens.fixed_column_observations(response.formula)
        if response.names[response.formula.response_name] in observed:
            local[response.formula.response_name] = response_tensor(
                response.formula, response.lens._family
            )
        local.update(
            {
                group + "_idx": torch.tensor(codes)
                for group, codes in response.formula.group_indices.items()
            }
        )
        values.update(
            {response.names.get(key, key): value for key, value in local.items()}
        )
    return values


def _build(specs, frame, **options):
    prepared = {
        name: _prepare_formula(formula, data=frame, family="gaussian", **options)
        for name, formula in specs.items()
    }
    emission = emit_joint(
        {name: (part.lens, part.formula) for name, part in prepared.items()}
    )
    program = Compiler(emission.module).compile()
    return emission, program, _observations(emission, prepared)


def _df_log_prior(df: torch.Tensor) -> torch.Tensor:
    """Gamma(2, 0.1) restricted to df > 1, normalized on that interval."""
    two, rate = df.new_tensor(2.0), df.new_tensor(0.1)
    return torch.distributions.Gamma(two, rate).log_prob(df) - torch.log1p(
        -torch.special.gammainc(two, rate)
    )


def _response_density(observations, names, columns, intercepts, sigmas, effects, rows):
    total = (
        torch.distributions.Normal(0.0, 5.0).log_prob(intercepts).sum()
        + torch.distributions.HalfCauchy(2.0).log_prob(sigmas).sum()
    )
    for index, (name, column) in enumerate(zip(names, columns, strict=True)):
        total = (
            total
            + torch.distributions.Normal(
                intercepts[index] + effects[rows, index], sigmas[index]
            )
            .log_prob(observations[f"response_{name}__{column}"].to(torch.double))
            .sum()
        )
    return total


@pytest.mark.parametrize(
    "covariance", [None, FULL, DEFICIENT], ids=["identity", "known", "deficient"]
)
def test_shared_student_block_has_one_mixing_multiplier_per_level(covariance):
    """B = diag(sqrt(df * udf)) . F Z . (diag(sd) L)^T across two responses."""
    frame = _frame()
    options = {} if covariance is None else {"group_covariances": {"g": covariance}}
    emission, program, observations = _build(
        {
            "first": "y ~ 1 + (1 | p | gr(g, dist = 'student'))",
            "second": "z ~ 1 + (1 | p | gr(g, dist = 'student'))",
        },
        frame,
        **options,
    )
    (law,) = emission.group_laws
    mixing = law.level.mixing
    assert mixing is not None
    assert (mixing.df_site, mixing.mixing_site) == ("df_g", "udf_g")
    assert isinstance(law.correlation, LKJCorrelation)
    assert [member.prefix for member in law.members] == [
        "response_first__",
        "response_second__",
    ]
    basis = law.level.basis
    factor = (
        basis.tensors()[2] if basis is not None else torch.eye(3, dtype=torch.double)
    )
    generator = torch.Generator().manual_seed(3)
    carrier = torch.randn(factor.shape[1], 2, generator=generator)
    values = {
        law.correlation.site: torch.tensor(CHOLESKY),
        "scales": torch.tensor([0.7, 0.4]),
        "df_g": torch.tensor(3.5),
        "udf_g": torch.tensor([0.4, 0.7, 1.1]),
        "carrier": carrier,
        "intercepts": torch.tensor([0.1, -0.2]),
        "sigmas": torch.tensor([0.8, 1.1]),
    }
    values = {name: value.requires_grad_() for name, value in values.items()}
    sites = {
        law.correlation.site: values[law.correlation.site],
        "df_g": values["df_g"],
        "udf_g": values["udf_g"],
    }
    for index, (member, name) in enumerate(
        zip(law.members, ("first", "second"), strict=True)
    ):
        sites[member.scale.site] = values["scales"][index]
        sites[f"response_{name}__intercept"] = values["intercepts"][index]
        sites[f"response_{name}__sigma"] = values["sigmas"][index]
    raw_sites = [member.raw_site for member in law.members]
    if basis is not None:
        assert len(set(raw_sites)) == 1 and raw_sites[0] is not None
        sites[raw_sites[0]] = values["carrier"]
    else:
        for index, site in enumerate(raw_sites):
            assert site is not None
            sites[site] = values["carrier"][:, index]
    rows = torch.tensor(emission.responses["first"].formula.group_indices["g"])

    def oracle(mixed: bool) -> torch.Tensor:
        cholesky, scales = values[law.correlation.site], values["scales"]
        effects = factor @ values["carrier"] @ (scales[:, None] * cholesky).T
        if mixed:
            effects = torch.sqrt(values["df_g"] * values["udf_g"])[:, None] * effects
        return (
            torch.distributions.LKJCholesky(2, torch.tensor(1.0)).log_prob(cholesky)
            + torch.distributions.HalfNormal(1.0).log_prob(scales).sum()
            + _df_log_prior(values["df_g"])
            + torch.distributions.InverseGamma(values["df_g"] / 2, torch.tensor(0.5))
            .log_prob(values["udf_g"])
            .sum()
            + torch.distributions.Normal(0.0, 1.0).log_prob(values["carrier"]).sum()
            + _response_density(
                observations,
                ("first", "second"),
                ("y", "z"),
                values["intercepts"],
                values["sigmas"],
                effects,
                rows,
            )
        )

    result = trace(program.morphism, torch.zeros(1, 1), observations | sites)
    actual = result.log_joint.sum()
    torch.testing.assert_close(actual, oracle(True))
    assert abs(float((oracle(False) - actual).detach())) > 1e-3
    leaves = tuple(values.values())
    for got, wanted in zip(
        torch.autograd.grad(actual, leaves, retain_graph=True),
        torch.autograd.grad(oracle(True), leaves),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted)


def test_shared_by_block_reads_each_level_scale_and_correlation():
    """Row j of B is diag(sd[by_j]) L[by_j] z_j, for both responses."""
    frame = _frame()
    emission, program, observations = _build(
        {
            "first": "y ~ 1 + (1 | p | gr(g, by = fb))",
            "second": "z ~ 1 + (1 | p | gr(g, by = fb))",
        },
        frame,
    )
    (law,) = emission.group_laws
    by = law.level.by
    assert by is not None and by.levels == ("m", "n")
    assert isinstance(law.correlation, ByLKJCorrelation)
    codes = torch.tensor(by.codes)
    torch.testing.assert_close(observations[by.input], codes)
    values = {
        "cholesky": torch.tensor([CHOLESKY, OTHER_CHOLESKY]),
        "scales": torch.tensor([[0.7, 0.4], [1.3, 0.9]]),
        "innovations": torch.tensor([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.8]]),
        "intercepts": torch.tensor([0.1, -0.2]),
        "sigmas": torch.tensor([0.8, 1.1]),
    }
    values = {name: value.requires_grad_() for name, value in values.items()}
    sites = {}
    for index, site in enumerate(law.correlation.by_sites):
        sites[site] = values["cholesky"][index]
    for index, (member, name) in enumerate(
        zip(law.members, ("first", "second"), strict=True)
    ):
        assert member.raw_site is not None
        # One scale per by level for each response's coefficient.
        sites[member.scale.site] = values["scales"][:, index]
        sites[member.raw_site] = values["innovations"][:, index]
        sites[f"response_{name}__intercept"] = values["intercepts"][index]
        sites[f"response_{name}__sigma"] = values["sigmas"][index]
    rows = torch.tensor(emission.responses["first"].formula.group_indices["g"])

    def oracle(partitioned: bool) -> torch.Tensor:
        level = codes if partitioned else torch.zeros_like(codes)
        cholesky = values["cholesky"][level]
        correlated = (cholesky @ values["innovations"][:, :, None]).squeeze(-1)
        effects = values["scales"][level] * correlated
        return (
            torch.distributions.LKJCholesky(2, torch.tensor(1.0))
            .log_prob(values["cholesky"])
            .sum()
            + torch.distributions.HalfNormal(1.0).log_prob(values["scales"]).sum()
            + torch.distributions.Normal(0.0, 1.0).log_prob(values["innovations"]).sum()
            + _response_density(
                observations,
                ("first", "second"),
                ("y", "z"),
                values["intercepts"],
                values["sigmas"],
                effects,
                rows,
            )
        )

    result = trace(program.morphism, torch.zeros(1, 1), observations | sites)
    actual = result.log_joint.sum()
    torch.testing.assert_close(actual, oracle(True))
    assert abs(float((oracle(False) - actual).detach())) > 1e-3
    leaves = tuple(values.values())
    for got, wanted in zip(
        torch.autograd.grad(actual, leaves, retain_graph=True),
        torch.autograd.grad(oracle(True), leaves),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted)
    # Closed form: unit innovations e_k at every level give column k of each
    # level's factor A_j, and A_j A_j^T = diag(sd[by_j]) R[by_j] diag(sd[by_j]).
    columns = []
    for coordinate in range(2):
        draws = {
            name: value.detach().reshape(1, 1, *value.shape)
            for name, value in sites.items()
        }
        for index, member in enumerate(law.members):
            assert member.raw_site is not None
            draws[member.raw_site] = torch.full((1, 1, 3), float(index == coordinate))
        rebuilt = group_coefficients(draws, emission.group_laws, {"g": LEVELS})
        columns.append(
            torch.stack(
                [
                    rebuilt[(member.prefix, "g", "Intercept")].reshape(3)
                    for member in law.members
                ],
                dim=-1,
            )
        )
    factors = torch.stack(columns, dim=-1)
    for level, code in enumerate(by.codes):
        scale = values["scales"][code].detach()
        cholesky = values["cholesky"][code].detach()
        torch.testing.assert_close(
            factors[level] @ factors[level].T,
            scale[:, None] * (cholesky @ cholesky.T) * scale[None, :],
        )


def test_response_blocks_read_the_mixing_the_shared_block_draws():
    frame = _frame().assign(x=[0.1 * index for index in range(12)])
    emission, program, _ = _build(
        {
            "first": "y ~ 1 + (1 | p | gr(g, dist = 'student'))"
            " + (0 + x | gr(g, dist = 'student'))",
            "second": "z ~ 1 + (1 | p | gr(g, dist = 'student'))",
        },
        frame,
    )
    sampled = [
        step.vars[0]
        for statement in emission.module.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(step, SampleStep)
    ]
    # One degrees-of-freedom site and one mixing site for the level, read by
    # the shared block and by the first response's own slope block.
    assert [name for name in sampled if name.endswith("df_g")] == ["df_g", "udf_g"]
    slope = next(
        law
        for law in emission.group_laws
        if [member.slope for member in law.members] == ["x"]
    )
    assert slope.level.mixing is not None
    assert slope.level.mixing.multiplier == "dfm_g"
    assert program.morphism is not None


def test_block_options_come_from_its_members_not_the_first_response():
    frame = _frame()
    emission, _, _ = _build(
        {
            "plain": "y ~ 1 + (1 | g)",
            "first": "z ~ 1 + (1 | p | gr(g, dist = 'student'))",
            "second": "w ~ 1 + (1 | p | gr(g, dist = 'student'))",
        },
        frame,
    )
    shared = [law for law in emission.group_laws if len(law.members) == 2]
    assert len(shared) == 1 and shared[0].level.mixing is not None
    alone = [law for law in emission.group_laws if len(law.members) == 1]
    assert len(alone) == 1 and alone[0].level.mixing is None


@pytest.mark.parametrize(
    "first,second",
    [
        ("(1 | p | g)", "(1 | p | gr(g, dist = 'student'))"),
        ("(1 | p | gr(g, dist = 'student'))", "(1 | p | g)"),
        ("(1 | p | gr(g, by = fb))", "(1 | p | g)"),
        ("(1 | p | g)", "(1 | p | gr(g, by = fb))"),
    ],
)
def test_members_declaring_different_level_factors_are_refused(first, second):
    with pytest.raises(ValueError, match="one law"):
        _build({"first": f"y ~ 1 + {first}", "second": f"z ~ 1 + {second}"}, _frame())


def test_responses_assigning_one_level_two_by_values_are_refused():
    frame = _frame()
    frame["other"] = [{"a": "m", "b": "m", "c": "n"}[level] for level in frame["g"]]
    with pytest.raises(ValueError, match="by"):
        _build(
            {
                "first": "y ~ 1 + (1 | p | gr(g, by = fb))",
                "second": "z ~ 1 + (1 | p | gr(g, by = other))",
            },
            frame,
        )


@pytest.mark.parametrize(
    "term,options",
    [
        ("(1 | p | gr(g, dist = 'student'))", {}),
        ("(1 | p | gr(g, dist = 'student'))", {"group_covariances": {"g": FULL}}),
        ("(1 | p | gr(g, by = fb))", {}),
        ("(1 | p | gr(g, by = fb))", {"group_covariances": {"g": FULL}}),
    ],
    ids=["student", "student-known", "by", "by-known"],
)
def test_shared_blocks_fit_and_predict_a_new_level(term, options):
    frame = _frame(24)
    fitted = fit(
        bf(f"y ~ 1 + {term}") + bf(f"z ~ 1 + {term}") + set_rescor(False),
        data=frame,
        method="svi",
        num_steps=3,
        seed=2,
        **options,
    )
    fresh = frame.iloc[:3].assign(g=["a", "d", "c"], fb=["m", "n", "m"])
    draws = fitted.posterior_predict(
        new_data=fresh, num_samples=2, seed=1, allow_new_levels=True
    )
    assert set(draws) == {"y", "z"}
    for value in draws.values():
        assert value.shape[-1] == 3 and torch.isfinite(value).all()


def test_joint_prediction_codes_a_new_level_by_its_fresh_by_level():
    frame = _frame(24)
    term = "(1 | p | gr(g, by = fb))"
    fitted = fit(
        bf(f"y ~ 1 + {term}") + bf(f"z ~ 1 + {term}") + set_rescor(False),
        data=frame,
        method="svi",
        num_steps=3,
        seed=2,
    )
    fresh = frame.iloc[:3].assign(g=["a", "d", "c"], fb=["m", "unfitted", "m"])
    with pytest.raises(ValueError, match="needs one fitted level"):
        fitted.posterior_predict(
            new_data=fresh, num_samples=2, seed=1, allow_new_levels=True
        )
