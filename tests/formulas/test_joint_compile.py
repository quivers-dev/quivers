"""Joint formula emissions reuse canonical likelihood and Gaussian block laws."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.dsl.ast_nodes import ProgramDecl, ObserveStep
from quivers.formulas import bf, formula_to_qvr, set_rescor
from quivers.formulas._fit import _prepare_formula
from quivers.formulas._joint_compile import emit_joint
from quivers.formulas._response import response_tensor
from quivers.formulas._priors import PriorSpec, RegularizedHorseshoe
from quivers.inference import trace


@pytest.fixture(autouse=True)
def double_literals():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _host(value):
    return value.item() if value.ndim == 0 else tuple(_host(part) for part in value)


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
        if prepared[name].trials is not None:
            local["formula_trials"] = prepared[name].trials
        values.update(
            {response.names.get(key, key): value for key, value in local.items()}
        )
    return values


def _build(specs, frame, *, joint_options=None, **options):
    prepared = {
        name: _prepare_formula(formula, data=frame, family=family, **options)
        for name, (formula, family) in specs.items()
    }
    emission = emit_joint(
        {name: (part.lens, part.formula) for name, part in prepared.items()},
        **(joint_options or {}),
    )
    program = Compiler(emission.module).compile()
    return emission, prepared, program, _observations(emission, prepared)


@pytest.mark.parametrize("override", [False, True])
def test_shared_group_scale_uses_its_response_brms_default(override: bool) -> None:
    """A correlation across responses keeps each response's prior scale."""
    frame = pd.DataFrame(
        {
            "narrow": [-1.0, -0.2, 0.1, 0.8, 1.2],
            "wide": [-100.0, -10.0, 0.0, 10.0, 100.0],
            "g": ["a", "b", "a", "b", "a"],
        }
    )
    formula = (
        bf("narrow ~ 1 + (1 | shared | g)")
        + bf("wide ~ 1 + (1 | shared | g)")
        + set_rescor(False)
    )
    rule = (
        PriorSpec(class_="sd", response="wide", distribution="Exponential(1.5)")
        if override
        else None
    )
    source = formula_to_qvr(formula, data=frame, default_priors="brms", priors=rule)
    assert (
        "sample response_narrow__sigma_g_Intercept <- HalfStudentT(df=3.0, scale=2.5)"
        in source
    )
    expected = (
        "Exponential(rate=1.5)" if override else "HalfStudentT(df=3.0, scale=14.8)"
    )
    assert f"sample response_wide__sigma_g_Intercept <- {expected}" in source


def test_heterogeneous_response_density_gradients_and_checked_reference():
    frame = pd.DataFrame({"y": [0.3, -0.4, 1.2], "choice": [1, 0, 1]})
    emission, prepared, program, observations = _build(
        {"time": ("y~1", "gaussian"), "accuracy": ("choice~1", "bernoulli")}, frame
    )
    latents = {
        "response_time__intercept": torch.tensor(0.2, requires_grad=True),
        "response_time__sigma": torch.tensor(0.8, requires_grad=True),
        "response_accuracy__intercept": torch.tensor(-0.4, requires_grad=True),
    }
    result = trace(program.morphism, torch.zeros(1, 1), observations | latents)
    normal = torch.distributions.Normal
    oracle = (
        normal(0.0, 5.0).log_prob(latents["response_time__intercept"])
        + torch.distributions.HalfCauchy(2.0).log_prob(latents["response_time__sigma"])
        + normal(0.0, 5.0).log_prob(latents["response_accuracy__intercept"])
        + normal(latents["response_time__intercept"], latents["response_time__sigma"])
        .log_prob(observations["response_time__y"].to(torch.double))
        .sum()
        + torch.distributions.Bernoulli(logits=latents["response_accuracy__intercept"])
        .log_prob(torch.tensor(frame.choice.to_numpy(), dtype=torch.double))
        .sum()
    )
    torch.testing.assert_close(result.log_joint.sum(), oracle)
    for got, wanted in zip(
        torch.autograd.grad(
            result.log_joint.sum(), tuple(latents.values()), retain_graph=True
        ),
        torch.autograd.grad(oracle, tuple(latents.values())),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted)
    reference = program.run(
        "model",
        data={name: _host(value) for name, value in observations.items()},
        sites={name: _host(value.detach()) for name, value in latents.items()},
    )
    assert reference.log_joint == pytest.approx(float(oracle.detach()), abs=3e-11)
    assert {target.response for target in emission.prior_targets} == {
        "time",
        "accuracy",
    }
    assert (
        observations[emission.responses["accuracy"].names["choice"]].dtype == torch.bool
    )


@pytest.mark.parametrize("family", ["gaussian", "student_t"])
def test_correlated_responses_share_one_normalized_vector_event(family):
    from quivers.continuous._multivariate_student_t import MultivariateStudentT

    frame = pd.DataFrame({"y": [0.3, -0.4, 1.2], "z": [-0.2, 0.5, 0.1]})
    emission, _, program, observations = _build(
        {"first": ("y~1", family), "second": ("z~1", family)},
        frame,
        joint_options={"rescor": True},
    )
    means = torch.tensor([0.1, -0.2], requires_grad=True)
    scales = torch.tensor([0.8, 1.1], requires_grad=True)
    factor = torch.tensor([[1.0, 0.0], [0.3, math.sqrt(0.91)]], requires_grad=True)
    values = {"rescor": factor}
    for i, name in enumerate(("first", "second")):
        values["response_" + name + "__intercept"] = means[i]
        values["response_" + name + "__sigma"] = scales[i]
    oracle = (
        torch.distributions.LKJCholesky(2, torch.tensor(1.0)).log_prob(factor)
        + torch.distributions.Normal(0.0, 5.0).log_prob(means).sum()
        + torch.distributions.HalfCauchy(2.0).log_prob(scales).sum()
    )
    leaves = (means, scales, factor)
    scale = scales[:, None] * factor
    if family == "gaussian":
        distribution = torch.distributions.MultivariateNormal(means, scale_tril=scale)
    else:
        nu = torch.tensor(4.3, requires_grad=True)
        values["nu"] = nu
        leaves += (nu,)
        oracle = oracle + torch.distributions.Gamma(2.0, 0.1).log_prob(nu)
        distribution = MultivariateStudentT(nu, means, scale_tril=scale)
    oracle = (
        oracle
        + distribution.log_prob(
            observations["formula_joint_response"].to(torch.double)
        ).sum()
    )
    result = trace(program.morphism, torch.zeros(1, 1), observations | values)
    torch.testing.assert_close(result.log_joint.sum(), oracle)
    for got, wanted in zip(
        torch.autograd.grad(result.log_joint.sum(), leaves, retain_graph=True),
        torch.autograd.grad(oracle, leaves),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted)
    reference = program.run(
        "model",
        data={name: _host(value) for name, value in observations.items()},
        sites={name: _host(value.detach()) for name, value in values.items()},
    )
    assert reference.log_joint == pytest.approx(float(oracle.detach()), abs=3e-11)
    assert [target.site for target in emission.prior_targets].count("nu") == (
        family == "student_t"
    )


@pytest.mark.parametrize("coordinate", ["centered", "noncentered"])
def test_shared_group_id_is_one_cross_response_gaussian_block(coordinate):
    frame = pd.DataFrame(
        {"y": [0.3, -0.4, 1.2], "z": [-0.2, 0.5, 0.1], "g": ["b", "a", "b"]}
    )
    emission, prepared, program, observations = _build(
        {
            "first": ("y~1+(1|shared|g)", "gaussian"),
            "second": ("z~1+(1|shared|g)", "gaussian"),
        },
        frame,
        reparameterize=coordinate,
    )
    assert len(emission.group_layouts) == 1
    layout = emission.group_layouts[0]
    assert len(layout.members) == 2
    assert {member.prefix for member in layout.members} == {
        "response_first__",
        "response_second__",
    }
    factor = torch.tensor([[1.0, 0.0], [0.3, math.sqrt(0.91)]], requires_grad=True)
    scales = torch.tensor([0.7, 0.4], requires_grad=True)
    values = {layout.correlation_site: factor}
    for index, member in enumerate(layout.members):
        values[member.scale_site] = scales[index]
    intercepts = torch.tensor([0.1, -0.2], requires_grad=True)
    sigmas = torch.tensor([0.8, 1.1], requires_grad=True)
    for index, name in enumerate(("first", "second")):
        values["response_" + name + "__intercept"] = intercepts[index]
        values["response_" + name + "__sigma"] = sigmas[index]
    coordinates = torch.tensor([[0.2, -0.4], [0.5, 0.1]], requires_grad=True)
    if coordinate == "centered":
        values[layout.members[0].raw_site] = coordinates
        coefficients = coordinates
        group_density = (
            torch.distributions.MultivariateNormal(
                torch.zeros(2), scale_tril=scales[:, None] * factor
            )
            .log_prob(coordinates)
            .sum()
        )
    else:
        for index, member in enumerate(layout.members):
            values[member.raw_site] = coordinates[:, index]
        coefficients = (coordinates @ factor.T) * scales
        group_density = torch.distributions.Normal(0.0, 1.0).log_prob(coordinates).sum()
    result = trace(program.morphism, torch.zeros(1, 1), observations | values)
    group_indices = torch.tensor(emission.responses["first"].formula.group_indices["g"])
    oracle = (
        torch.distributions.LKJCholesky(2, torch.tensor(1.0)).log_prob(factor)
        + torch.distributions.HalfNormal(1.0).log_prob(scales).sum()
        + torch.distributions.Normal(0.0, 5.0).log_prob(intercepts).sum()
        + torch.distributions.HalfCauchy(2.0).log_prob(sigmas).sum()
        + group_density
    )
    for index, column in enumerate(("y", "z")):
        oracle = (
            oracle
            + torch.distributions.Normal(
                intercepts[index] + coefficients[group_indices, index], sigmas[index]
            )
            .log_prob(
                observations[
                    "response_" + ("first" if index == 0 else "second") + "__" + column
                ].to(torch.double)
            )
            .sum()
        )
    torch.testing.assert_close(result.log_joint.sum(), oracle)
    leaves = (scales, factor, coordinates, intercepts, sigmas)
    for got, wanted in zip(
        torch.autograd.grad(result.log_joint.sum(), leaves, retain_graph=True),
        torch.autograd.grad(oracle, leaves),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted)
    reference = program.run(
        "model",
        data={name: _host(value) for name, value in observations.items()},
        sites={name: _host(value.detach()) for name, value in values.items()},
    )
    assert reference.log_joint == pytest.approx(float(oracle.detach()), abs=3e-11)


def test_response_selector_reuses_conditionally_normalized_ordinary_prior():
    frame = pd.DataFrame({"y": [0.3, -0.4, 1.2], "z": [-0.2, 0.5, 0.1]})
    rule = PriorSpec(
        distribution="Normal(0.2, 0.7)",
        class_="Intercept",
        response="first",
        lower=-0.5,
        upper=0.8,
    )
    emission, _, program, observations = _build(
        {"first": ("y~1", "gaussian"), "second": ("z~1", "gaussian")},
        frame,
        joint_options={"priors": rule, "rescor": True},
    )
    values = {
        "response_first__intercept": torch.tensor(0.1, requires_grad=True),
        "response_second__intercept": torch.tensor(-0.2, requires_grad=True),
        "response_first__sigma": torch.tensor(0.8, requires_grad=True),
        "response_second__sigma": torch.tensor(1.1, requires_grad=True),
        "rescor": torch.eye(2, requires_grad=True),
    }
    result = trace(program.morphism, torch.zeros(1, 1), observations | values)
    means = torch.stack(
        (values["response_first__intercept"], values["response_second__intercept"])
    )
    scales = torch.stack(
        (values["response_first__sigma"], values["response_second__sigma"])
    )
    prior = torch.distributions.Normal(torch.tensor(0.2), torch.tensor(0.7))
    oracle = (
        prior.log_prob(means[0])
        - (prior.cdf(torch.tensor(0.8)) - prior.cdf(torch.tensor(-0.5))).log()
        + torch.distributions.Normal(0.0, 5.0).log_prob(means[1])
        + torch.distributions.HalfCauchy(2.0).log_prob(scales).sum()
        + torch.distributions.LKJCholesky(2, torch.tensor(1.0)).log_prob(
            values["rescor"]
        )
        + torch.distributions.MultivariateNormal(
            means, scale_tril=scales[:, None] * values["rescor"]
        )
        .log_prob(observations["formula_joint_response"].to(torch.double))
        .sum()
    )
    torch.testing.assert_close(result.log_joint.sum(), oracle)
    for got, wanted in zip(
        torch.autograd.grad(
            result.log_joint.sum(), tuple(values.values()), retain_graph=True
        ),
        torch.autograd.grad(oracle, tuple(values.values())),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted)


def test_shared_shrinkage_uses_each_response_scale_and_one_joint_hierarchy():
    frame = pd.DataFrame(
        {"y": [0.3, -0.4, 1.2], "z": [-0.2, 0.5, 0.1], "x": [0.2, -0.1, 0.7]}
    )
    hierarchy = RegularizedHorseshoe(
        df=3.0, df_global=2.0, df_slab=5.0, scale_global=0.3, scale_slab=1.2
    )
    rule = PriorSpec(distribution=hierarchy, block="shared")
    emission, _, program, observations = _build(
        {"first": ("y~0+x", "gaussian"), "second": ("z~0+x", "gaussian")},
        frame,
        joint_options={"priors": rule},
    )
    beta = torch.tensor([0.2, -0.15], requires_grad=True)
    sigma = torch.tensor([0.6, 0.9], requires_grad=True)
    global_ = torch.tensor(0.4, requires_grad=True)
    slab = torch.tensor(0.8, requires_grad=True)
    local = torch.tensor([0.7, 1.4], requires_grad=True)
    sites = {
        "prior_shared_global": global_,
        "prior_shared_slab": slab,
        "prior_shared_local": local,
    }
    for i, name in enumerate(("first", "second")):
        sites["response_" + name + "__beta_x"] = beta[i]
        sites["response_" + name + "__sigma"] = sigma[i]
    product = sigma * global_ * local
    c2 = hierarchy.scale_slab**2 * slab
    scales = product * torch.sqrt(c2 / (c2 + product.square()))
    oracle = (
        torch.distributions.StudentT(2.0, 0.0, 0.3).log_prob(global_)
        + math.log(2.0)
        + torch.distributions.InverseGamma(2.5, 2.5).log_prob(slab)
        + (
            torch.distributions.StudentT(3.0, 0.0, 1.0).log_prob(local) + math.log(2.0)
        ).sum()
        + torch.distributions.Normal(torch.zeros(2), scales).log_prob(beta).sum()
        + torch.distributions.HalfCauchy(2.0).log_prob(sigma).sum()
    )
    x = observations["response_first__x"].to(torch.double)
    for i, name in enumerate(("y", "z")):
        oracle = (
            oracle
            + torch.distributions.Normal(beta[i] * x, sigma[i])
            .log_prob(
                observations[
                    "response_" + ("first" if i == 0 else "second") + "__" + name
                ].to(torch.double)
            )
            .sum()
        )
    result = trace(program.morphism, torch.zeros(1, 1), observations | sites)
    torch.testing.assert_close(result.log_joint.sum(), oracle)
    leaves = (beta, sigma, global_, slab, local)
    for got, wanted in zip(
        torch.autograd.grad(result.log_joint.sum(), leaves, retain_graph=True),
        torch.autograd.grad(oracle, leaves),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted)
    reference = program.run(
        "model",
        data={name: _host(value) for name, value in observations.items()},
        sites={name: _host(value.detach()) for name, value in sites.items()},
    )
    assert reference.log_joint == pytest.approx(float(oracle.detach()), abs=3e-11)


def test_shared_group_keeps_distinct_scalar_and_category_coordinates():
    frame = pd.DataFrame(
        {"y": [0.3, -0.4, 1.2], "choice": [0, 2, 1], "g": ["b", "a", "b"]}
    )
    emission, _, program, observations = _build(
        {
            "time": ("y~1+(1|shared|g)", "gaussian"),
            "choice": ("choice~1+(1|shared|g)", "categorical"),
        },
        frame,
    )
    layout = emission.group_layouts[0]
    assert [member.width for member in layout.members] == [1, 2]
    factor = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, math.sqrt(0.96), 0.0], [-0.1, 0.3, math.sqrt(0.9)]],
        requires_grad=True,
    )
    scale = torch.tensor([0.6, 0.4, 0.8], requires_grad=True)
    raw = torch.tensor([[0.1, -0.4, 0.3], [-0.2, 0.3, 0.5]], requires_grad=True)
    intercept = torch.tensor([0.2, -0.3, 0.4], requires_grad=True)
    sigma = torch.tensor(0.9, requires_grad=True)
    first, second = layout.members
    sites = {
        layout.correlation_site: factor,
        first.scale_site: scale[0],
        second.scale_site: scale[1:],
        first.raw_site: raw[:, 0],
        second.raw_site: raw[:, 1:],
        "response_time__intercept": intercept[0],
        "response_choice__intercept": intercept[1:],
        "response_time__sigma": sigma,
    }
    coefficients = (raw @ factor.T) * scale
    groups = torch.tensor(emission.responses["time"].formula.group_indices["g"])
    logits = torch.cat(
        (torch.zeros(3, 1), intercept[1:] + coefficients[groups, 1:]), dim=-1
    )
    oracle = (
        torch.distributions.LKJCholesky(3, torch.tensor(1.0)).log_prob(factor)
        + torch.distributions.HalfNormal(1.0).log_prob(scale).sum()
        + torch.distributions.Normal(0.0, 1.0).log_prob(raw).sum()
        + torch.distributions.Normal(0.0, 5.0).log_prob(intercept).sum()
        + torch.distributions.HalfCauchy(2.0).log_prob(sigma)
        + torch.distributions.Normal(intercept[0] + coefficients[groups, 0], sigma)
        .log_prob(observations["response_time__y"].to(torch.double))
        .sum()
        + torch.distributions.Categorical(logits=logits)
        .log_prob(observations["response_choice__choice"])
        .sum()
    )
    result = trace(program.morphism, torch.zeros(1, 1), observations | sites)
    torch.testing.assert_close(result.log_joint.sum(), oracle, atol=3e-11, rtol=3e-11)
    leaves = (factor, scale, raw, intercept, sigma)
    for got, wanted in zip(
        torch.autograd.grad(result.log_joint.sum(), leaves, retain_graph=True),
        torch.autograd.grad(oracle, leaves),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted, atol=3e-10, rtol=3e-10)
    # The checked product group×component carrier is one flat finite index;
    # native posterior layout retains its two declared coordinate axes.
    host_sites = {name: _host(value.detach()) for name, value in sites.items()}
    host_sites[second.raw_site] = _host(raw[:, 1:].detach().reshape(-1))
    reference = program.run(
        "model",
        data={name: _host(value) for name, value in observations.items()},
        sites=host_sites,
    )
    assert reference.log_joint == pytest.approx(float(oracle.detach()), abs=3e-11)
