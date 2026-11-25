"""Independent Gaussian/support and conditional oracles for supplied group covariance."""

from copy import deepcopy

import pytest
import torch
import didactic.api as dx

from quivers.formulas import GroupCovariance


@pytest.mark.parametrize("rank", [None, 2, 1, 0])
def test_labeled_basis_reproduces_covariance_and_retains_immutable_values(rank):
    matrix = torch.tensor([[1.7, 0.3], [0.3, 0.8]], dtype=torch.float64)
    if rank == 1:
        matrix = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
    elif rank == 0:
        matrix = torch.zeros(2, 2, dtype=torch.float64)
    spec = GroupCovariance(levels=("first", "second"), matrix=matrix, rank=rank)
    matrix.zero_()
    basis = spec.basis(("second", "first"))
    projection, coordinate, factor = basis.tensors()
    expected = spec.tensor().flip((0, 1))
    torch.testing.assert_close(factor @ factor.T, expected)
    torch.testing.assert_close(
        projection.T @ projection, torch.eye(basis.rank, dtype=torch.float64)
    )
    torch.testing.assert_close(projection @ coordinate @ projection.T, expected)
    assert deepcopy(spec) == spec
    assert hash(deepcopy(spec)) == hash(spec)
    assert (
        GroupCovariance(levels=("a",), matrix=((0.1234567890123456,),)).matrix[0][0]
        == 0.1234567890123456
    )


@pytest.mark.parametrize("rank", [None, 2])
def test_conditional_mean_and_schur_match_partitioned_multivariate_gaussian(rank):
    covariance = torch.tensor(
        [[2.0, 0.4, 0.7], [0.4, 1.4, -0.2], [0.7, -0.2, 1.8]], dtype=torch.float64
    )
    spec = GroupCovariance(
        levels=("f0", "f1", "new"), matrix=covariance, rank=None if rank is None else 3
    )
    mean_map, basis = spec.conditional(("f1", "f0"), ("new",))
    ff = covariance[:2, :2].flip((0, 1))
    nf = covariance[2:, :2].flip((1,))
    expected_mean = torch.linalg.solve(ff, nf.T).T
    expected_schur = covariance[2:, 2:] - expected_mean @ nf.T
    _, _, factor = basis.tensors()
    torch.testing.assert_close(mean_map, expected_mean)
    torch.testing.assert_close(factor @ factor.T, expected_schur)
    generator = torch.Generator().manual_seed(48)
    fitted = torch.tensor([-0.3, 0.8], dtype=torch.float64)
    draws = (
        mean_map @ fitted
        + (
            factor
            @ torch.randn(basis.rank, 40000, dtype=torch.float64, generator=generator)
        ).T
    )
    torch.testing.assert_close(
        draws.mean(0), expected_mean @ fitted, atol=0.016, rtol=0
    )
    torch.testing.assert_close(
        torch.atleast_2d(draws.var(0)), expected_schur, atol=0.035, rtol=0
    )


def test_rank_deficient_conditioning_has_no_fabricated_randomness():
    spec = GroupCovariance(
        levels=("a", "b", "c"),
        matrix=((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (3.0, 6.0, 9.0)),
        rank=1,
    )
    mean_map, basis = spec.conditional(("b",), ("a", "c"))
    torch.testing.assert_close(
        mean_map, torch.tensor([[0.5], [1.5]], dtype=torch.float64)
    )
    assert basis.rank == 0
    assert basis.tensors()[2].shape == (2, 0)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"levels": ("a", "a")}, "unique"),
        ({"matrix": ((1.0, 0.2), (0.3, 1.0))}, "symmetric"),
        ({"matrix": ((1.0, 2.0), (2.0, 1.0))}, "positive definite"),
        ({"matrix": ((1.0, 1.0), (1.0, 1.0))}, "positive definite"),
        ({"matrix": ((1.0, 1.0), (1.0, 1.0)), "rank": 2}, "rank exceeds"),
        ({"rank": 0}, "omits nonzero"),
        ({"rank": True}, "int, got bool"),
        ({"matrix": ((float("nan"), 0.0), (0.0, 1.0))}, "finite"),
    ],
)
def test_invalid_declared_covariances_fail_before_emission(changes, match):
    args = dict(levels=("a", "b"), matrix=((1.0, 0.2), (0.2, 1.0))) | changes
    with pytest.raises((TypeError, ValueError, dx.ValidationError), match=match):
        GroupCovariance(**args)


def _compiled(
    parameterization,
    *,
    rank=None,
    formula="y~x+(1+x|g)",
    family="gaussian",
    thresholds_by=None,
    distributional=None,
):
    import pandas as pd
    from quivers.formulas._fit import _prepare_formula
    from quivers.dsl import Compiler

    matrix = torch.tensor([[1.7, 0.3], [0.3, 0.8]], dtype=torch.float64)
    if rank == 1:
        matrix = torch.ones(2, 2, dtype=torch.float64)
    elif rank == 0:
        matrix.zero_()
    spec = GroupCovariance(levels=("a", "b"), matrix=matrix, rank=rank)
    frame = pd.DataFrame(
        {"y": [0.2, -0.4, 0.7], "x": [-0.5, 0.2, 1.0], "g": ["b", "a", "b"]}
    )
    if family == "cumulative":
        frame.y = [0, 1, 2]
    prepared = _prepare_formula(
        formula,
        data=frame,
        family=family,
        group_covariances={"g": spec},
        reparameterize=parameterization,
        thresholds_by=thresholds_by,
        distributional=distributional,
    )
    lens, parsed = prepared.lens, prepared.formula
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
    native = Compiler(module).compile().morphism
    observations = lens.fixed_column_observations(parsed) | {
        "g_idx": torch.tensor([1, 0, 1]),
        "y": torch.as_tensor(frame.y.to_numpy().copy()),
    }
    return spec, lens, parsed, module, native, observations, complement


def _nested(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().tolist()
    if isinstance(value, list):
        return tuple(_nested(item) for item in value)
    return value


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
@pytest.mark.parametrize("rank", [None, 1, 0])
def test_full_joint_all_heads_and_reference_match_dense_kronecker_oracle(
    parameterization, rank
):
    from quivers.dsl.emit import module_to_source
    from quivers.dsl import loads
    from quivers.inference import trace
    from quivers.formulas._group_law import group_coefficients

    spec, lens, parsed, module, native, obs, complement = _compiled(
        parameterization, rank=rank
    )
    layout = complement.group_layouts[0]
    assert layout.level_covariance is not None
    basis = layout.level_covariance
    site = f"group_covariance_{layout.identity}"
    raw = (
        torch.arange(max(1, basis.rank * 2), dtype=torch.float64).reshape(-1) * 0.1
        - 0.2
    )
    values = {
        "sigma_g_Intercept": torch.tensor(0.7, dtype=torch.float64, requires_grad=True),
        "sigma_g_x": torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
        "cor_g": torch.tensor(
            [[1.0, 0.0], [0.3, (1 - 0.3**2) ** 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        ),
        "intercept": torch.tensor(0.1, dtype=torch.float64, requires_grad=True),
        "beta_x": torch.tensor(-0.2, dtype=torch.float64, requires_grad=True),
        "sigma": torch.tensor(0.8, dtype=torch.float64, requires_grad=True),
    }
    if basis.rank:
        values[site] = raw[: basis.rank * 2].reshape(basis.rank, 2).requires_grad_()
    result = trace(native, torch.zeros(1, 1), obs | values)
    sigma = torch.stack([values["sigma_g_Intercept"], values["sigma_g_x"]])
    # A factor-parameterized centered MatrixNormal differentiates on the
    # declared lower-triangular carrier, rather than its invalid upper entries.
    coefficient_factor = sigma[:, None] * (
        values["cor_g"].tril() if parameterization == "centered" else values["cor_g"]
    )
    projection, coordinate, factor = basis.tensors()
    if basis.rank:
        if parameterization == "centered":
            physical = projection @ values[site]
            covariance = torch.kron(
                coordinate.contiguous(),
                (coefficient_factor @ coefficient_factor.T).contiguous(),
            )
            prior = torch.distributions.MultivariateNormal(
                torch.zeros(basis.rank * 2, dtype=torch.float64),
                covariance_matrix=covariance,
            ).log_prob(values[site].reshape(-1))
        else:
            physical = factor @ values[site] @ coefficient_factor.T
            prior = torch.distributions.Normal(0.0, 1.0).log_prob(values[site]).sum()
    else:
        physical = torch.zeros(2, 2, dtype=torch.float64)
        prior = torch.tensor(0.0, dtype=torch.float64)
    x = torch.tensor([-0.5, 0.2, 1.0], dtype=torch.float64)
    eta = (
        values["intercept"]
        + values["beta_x"] * x
        + physical[obs["g_idx"], 0]
        + physical[obs["g_idx"], 1] * x
    )
    expected = (
        prior
        + torch.distributions.Normal(eta, values["sigma"]).log_prob(obs["y"]).sum()
    )
    expected += torch.distributions.HalfNormal(1.0).log_prob(sigma).sum()
    expected += torch.distributions.LKJCholesky(
        2, torch.tensor(1.0, dtype=torch.float64)
    ).log_prob(values["cor_g"])
    expected += torch.distributions.Normal(0.0, 5.0).log_prob(values["intercept"])
    expected += torch.distributions.Normal(
        0.0, 5.0 / x.square().mean().sqrt()
    ).log_prob(values["beta_x"])
    expected += torch.distributions.HalfCauchy(2.0).log_prob(values["sigma"])
    torch.testing.assert_close(result.log_joint.sum(), expected, rtol=2e-7, atol=2e-7)
    heads = tuple(values.values())
    actual_grad = torch.autograd.grad(result.log_joint.sum(), heads, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, heads)
    for actual, oracle in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, oracle, rtol=2e-6, atol=2e-6)
    reference = loads(module_to_source(module)).run(
        "model",
        data={k: _nested(v) for k, v in obs.items()},
        sites={k: _nested(v) for k, v in values.items()},
    )
    assert reference.log_joint == pytest.approx(
        float(expected.detach()), rel=2e-7, abs=2e-7
    )
    retained = {
        name: value.detach().reshape(1, 1, *value.shape)
        for name, value in values.items()
    }
    coefficients = group_coefficients(
        retained, complement.group_laws, parsed.group_levels
    )
    torch.testing.assert_close(
        coefficients[("", "g", "Intercept")].reshape(2), physical[:, 0].detach()
    )
    torch.testing.assert_close(
        coefficients[("", "g", "x")].reshape(2), physical[:, 1].detach()
    )


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
@pytest.mark.parametrize("policy", ["gaussian", "uncertainty", "old_levels"])
def test_group_remap_preserves_physical_known_levels_and_conditional_mean(
    parameterization, policy
):
    from quivers.formulas._group_law import KnownCovariance, block_coefficients
    from quivers.formulas._group_prediction import GroupRemap, _remap_covariance_group

    spec, lens, parsed, module, native, obs, complement = _compiled(parameterization)
    law = complement.group_laws[0]
    covariance = GroupCovariance(
        levels=("a", "b", "c"),
        matrix=((1.7, 0.3, 0.6), (0.3, 0.8, -0.1), (0.6, -0.1, 1.4)),
    )
    mean_map, conditional = covariance.conditional(("a", "b"), ("c",))
    target = covariance.basis(("b", "c", "a"))
    group = GroupRemap(
        group="g",
        fitted_size=2,
        fitted_indices=(1, -1, 0),
        row_indices=torch.tensor([0, 1, 2]),
        levels=("b", "c", "a"),
        covariance=covariance,
        covariance_laws=(law,),
        covariance_target=target,
        covariance_conditional=conditional,
        covariance_mean_map=tuple(tuple(row) for row in mean_map.tolist()),
    )
    values = {
        "sigma_g_Intercept": torch.tensor(0.7, dtype=torch.float64),
        "sigma_g_x": torch.tensor(0.5, dtype=torch.float64),
        "cor_g": torch.tensor(
            [[1.0, 0.0], [0.3, (1 - 0.3**2) ** 0.5]], dtype=torch.float64
        ),
        "group_covariance_g": torch.tensor(
            [[-0.7, 0.2], [0.9, -0.4]], dtype=torch.float64
        ),
    }
    fitted = block_coefficients(values, law, sample_ndims=0)
    compact = law.with_(level=law.level.with_(covariance=KnownCovariance(basis=target)))
    shapes = {"group_covariance_g": (3, 2)}
    draws = []
    fixed = {}
    with torch.random.fork_rng():
        torch.manual_seed(481)
        for _ in range(1500 if policy == "gaussian" else 40):
            mapped = _remap_covariance_group(group, values, shapes, policy, fixed)
            physical = block_coefficients(values | mapped, compact, sample_ndims=0)
            torch.testing.assert_close(physical[[0, 2]], fitted[[1, 0]])
            draws.append(physical[1])
    result = torch.stack(draws)
    expectation = (mean_map @ fitted)[0]
    if policy == "gaussian":
        factor = (
            torch.diag(torch.tensor([0.7, 0.5], dtype=torch.float64)) @ values["cor_g"]
        )
        expected_cov = conditional.tensors()[2].square().sum() * (factor @ factor.T)
        torch.testing.assert_close(result.mean(0), expectation, atol=0.045, rtol=0)
        torch.testing.assert_close(
            torch.cov(result.T), expected_cov, atol=0.045, rtol=0
        )
    elif policy == "old_levels":
        torch.testing.assert_close(result, result[0].expand_as(result))
    else:
        assert result.unique(dim=0).shape[0] == 2


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("rank", [None, 1, 0])
def test_fitting_and_new_labeled_prediction_share_one_covariance_path(
    parameterization, method, rank
):
    import pandas as pd
    from quivers.formulas import fit

    matrix = torch.tensor(
        [[1.7, 0.3, 0.6], [0.3, 0.8, -0.1], [0.6, -0.1, 1.4]], dtype=torch.float64
    )
    if rank == 1:
        matrix = torch.ones(3, 3, dtype=torch.float64)
    elif rank == 0:
        matrix.zero_()
    covariance = GroupCovariance(levels=("a", "b", "new"), matrix=matrix, rank=rank)
    frame = pd.DataFrame({"y": [0.2, -0.4, 0.7], "g": ["b", "a", "b"]})
    settings = (
        {"num_steps": 1}
        if method == "svi"
        else {"num_warmup": 1, "num_samples": 2, "num_chains": 1}
    )
    fitted = fit(
        "y~1+(1|g)",
        data=frame,
        group_covariances={"g": covariance},
        reparameterize=parameterization,
        method=method,
        fixed_parameters={"sigma": 1.0},
        seed=14,
        **settings,
    )
    options = {"num_samples": 2, "seed": 21}
    original = fitted.posterior_linpred(**options)
    known = fitted.posterior_linpred(new_data=frame.iloc[[1, 0]], **options)
    torch.testing.assert_close(known, original[:, :, [1, 0]])
    fresh = pd.DataFrame({"g": ["new", "a", "new"]})
    state = torch.random.get_rng_state().clone()
    predicted = fitted.posterior_linpred(
        new_data=fresh, allow_new_levels=True, sample_new_levels="gaussian", **options
    )
    torch.testing.assert_close(predicted[..., 0], predicted[..., 2])
    torch.testing.assert_close(predicted[..., 1], original[..., 1])
    torch.testing.assert_close(
        predicted,
        fitted.posterior_linpred(
            new_data=fresh,
            allow_new_levels=True,
            sample_new_levels="gaussian",
            **options,
        ),
    )
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.isfinite(fitted.log_likelihood(**options)["y"]).all()


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_prediction_can_gain_and_lose_only_declared_support_dimensions(
    parameterization,
):
    import pandas as pd
    from quivers.formulas import fit

    spec = GroupCovariance(
        levels=("a", "b", "new"),
        matrix=((0.0, 0.0, 0.0), (0.0, 1.0, 0.4), (0.0, 0.4, 1.0)),
        rank=2,
    )
    frame = pd.DataFrame({"y": [0.2, -0.1], "g": ["a", "a"]})
    fitted = fit(
        "y~1+(1|g)",
        data=frame,
        group_covariances={"g": spec},
        reparameterize=parameterization,
        method="svi",
        num_steps=1,
        fixed_parameters={"sigma": 1.0},
    )
    assert not any(
        name.startswith("group_covariance_") for name in fitted.draws(num_samples=2)
    )
    predicted = fitted.posterior_linpred(
        new_data=pd.DataFrame({"g": ["a", "new"]}),
        allow_new_levels=True,
        sample_new_levels="gaussian",
        num_samples=2,
        seed=18,
    )
    assert torch.isfinite(predicted).all()
    torch.testing.assert_close(
        predicted[..., 0],
        fitted.posterior_linpred(num_samples=2, seed=18)[..., 0].to(predicted),
    )
    with pytest.raises(ValueError, match="no fitted support innovations"):
        fitted.posterior_linpred(
            new_data=pd.DataFrame({"g": ["new"]}), allow_new_levels=True, num_samples=2
        )
    trained = fit(
        "y~1+(1|g)",
        data=pd.DataFrame({"y": [0.2, -0.1], "g": ["a", "b"]}),
        group_covariances={"g": spec},
        reparameterize=parameterization,
        method="svi",
        num_steps=1,
        fixed_parameters={"sigma": 1.0},
    )
    only_zero = trained.posterior_linpred(
        new_data=pd.DataFrame({"g": ["a"]}), num_samples=2, seed=13
    )
    torch.testing.assert_close(
        only_zero,
        trained.posterior_linpred(num_samples=2, seed=13)[..., :1].to(only_zero),
    )


def test_covariance_extension_retains_training_block_and_validates_labels_before_sampling():
    import pandas as pd
    from quivers.formulas import fit, formula_to_qvr, get_prior

    frame = pd.DataFrame({"y": [0.1, -0.2, 0.3], "g": ["a", "b", "a"]})
    base = GroupCovariance(levels=("a", "b"), matrix=((1.7, 0.3), (0.3, 0.8)))
    fitted = fit(
        "y~1+(1|g)",
        data=frame,
        group_covariances={"g": base},
        method="svi",
        num_steps=1,
    )
    extension = GroupCovariance(
        levels=("new", "b", "a"),
        matrix=((1.4, -0.1, 0.6), (-0.1, 0.8, 0.3), (0.6, 0.3, 1.7)),
    )
    extended = fitted.with_(group_covariances={"g": extension})
    fresh = pd.DataFrame({"g": ["new", "a"]})
    options = dict(
        new_data=fresh,
        allow_new_levels=True,
        sample_new_levels="gaussian",
        num_samples=2,
        seed=8,
    )
    predicted = extended.posterior_linpred(**options)
    torch.testing.assert_close(
        predicted[..., 1], fitted.posterior_linpred(num_samples=2, seed=8)[..., 0]
    )
    bad = extension.with_(matrix=((1.4, -0.1, 0.6), (-0.1, 1.8, 0.3), (0.6, 0.3, 1.7)))
    with pytest.raises(ValueError, match="fitted principal submatrix"):
        fitted.with_(group_covariances={"g": bad}).posterior_linpred(**options)
    with pytest.raises(ValueError, match="no entries for levels"):
        fitted.posterior_linpred(**options)
    with pytest.raises(ValueError, match="explicit labeled identities"):
        extended.posterior_linpred(
            new_data=pd.DataFrame({"g": [None]}), allow_new_levels=True, num_samples=2
        )
    with pytest.raises(ValueError, match="unknown grouping factors"):
        formula_to_qvr("y~1+(1|g)", data=frame, group_covariances={"wrong": base})
    assert "sigma_g_Intercept" in get_prior(
        "y~1+(1|g)", data=frame, group_covariances={"g": base}
    )


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
@pytest.mark.parametrize("name", ["categorical", "dirichlet", "logistic_normal"])
def test_component_known_covariance_fit_replay_and_fresh_labels(parameterization, name):
    import pandas as pd
    from quivers.formulas import fit, families

    covariance = GroupCovariance(
        levels=("a", "b", "new"),
        matrix=((1.4, 0.2, 0.3), (0.2, 0.8, 0.1), (0.3, 0.1, 1.0)),
    )
    frame = pd.DataFrame(
        {
            "g": ["b", "a", "b", "a"],
            "x": [-0.3, 0.2, 0.4, 0.7],
            "y": [0, 1, 2, 0],
            "a": [0.2, 0.1, 0.4, 0.5],
            "b": [0.3, 0.6, 0.2, 0.3],
            "c": [0.5, 0.3, 0.4, 0.2],
        }
    )
    formula = "y~x+(1+x|g)" if name == "categorical" else "cbind(a,b,c)~x+(1+x|g)"
    family = (
        families[name] if name == "categorical" else families[name].with_reference("b")
    )
    result = fit(
        formula,
        data=frame,
        family=family,
        group_covariances={"g": covariance},
        reparameterize=parameterization,
        method="svi",
        num_steps=1,
        seed=7,
    )
    options = {"num_samples": 2, "seed": 8}
    original = result.posterior_linpred(**options)
    known = result.posterior_linpred(new_data=frame.iloc[[1, 0]], **options)
    torch.testing.assert_close(known, original[:, :, [1, 0]])
    fresh = result.posterior_linpred(
        new_data=pd.DataFrame({"g": ["new", "a", "new"], "x": [0.2, 0.2, 0.2]}),
        allow_new_levels=True,
        sample_new_levels="gaussian",
        **options,
    )
    torch.testing.assert_close(fresh[:, :, 0], fresh[:, :, 2])
    torch.testing.assert_close(fresh[:, :, 1], original[:, :, 1])
    assert all(
        torch.isfinite(value).all()
        for value in result.log_likelihood(**options).values()
    )
    assert result.category_labels == (
        (0, 1, 2) if name == "categorical" else ("a", "b", "c")
    )


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_prepared_group_trace_does_not_refactor_known_or_learned_covariances(
    parameterization,
):
    from unittest.mock import patch
    from quivers.inference.trace import trace

    _, _, _, _, native, observations, _ = _compiled(parameterization)
    with patch(
        "torch.linalg.cholesky",
        side_effect=AssertionError("repeated covariance factorization"),
    ):
        recorded = trace(native, torch.zeros(1, 1), observations=observations)
        assert torch.isfinite(recorded.log_joint).all()
