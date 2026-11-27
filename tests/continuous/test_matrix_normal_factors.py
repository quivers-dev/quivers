"""Independent matrix-normal factor, geometry, and resource contracts."""

import itertools
import random
from unittest.mock import patch

import pytest
import torch

from quivers.continuous._matrix import MatrixNormal
from quivers.qiec.distributions import ReferenceBackend


_VARIANTS = tuple(itertools.product((False, True), repeat=2))


def _parameters(row_factor, col_factor, *, batch=False):
    loc = torch.tensor([[0.1, -0.3, 0.2], [0.4, 0.2, -0.1]], dtype=torch.double)
    row = torch.tensor([[1.2, 0.0], [0.3, 0.8]], dtype=torch.double)
    col = torch.tensor(
        [[0.7, 0.0, 0.0], [-0.2, 1.1, 0.0], [0.1, 0.3, 0.9]], dtype=torch.double
    )
    if batch:
        loc = loc.expand(4, 2, 3).clone()
        row = row.expand(4, 2, 2).clone()
    args = {
        "loc": loc,
        "row_scale_tril" if row_factor else "row_covariance": row
        if row_factor
        else row @ row.T
        if not batch
        else row @ row.transpose(-1, -2),
        "col_scale_tril" if col_factor else "col_covariance": col
        if col_factor
        else col @ col.T,
    }
    return args, row, col


@pytest.mark.parametrize(("row_factor", "col_factor"), _VARIANTS)
@pytest.mark.parametrize("batch", [False, True])
def test_density_moments_and_all_heads_match_dense_kronecker(
    row_factor, col_factor, batch
):
    args, row, col = _parameters(row_factor, col_factor, batch=batch)
    args = {key: value.clone().requires_grad_() for key, value in args.items()}
    distribution = MatrixNormal(**args)
    value = torch.tensor([[0.4, -0.1, 0.7], [-0.2, 0.8, 0.3]], dtype=torch.double)
    row_cov = args.get("row_covariance")
    if row_cov is None:
        row_cov = args["row_scale_tril"] @ args["row_scale_tril"].transpose(-1, -2)
    col_cov = args.get("col_covariance")
    if col_cov is None:
        col_cov = args["col_scale_tril"] @ args["col_scale_tril"].transpose(-1, -2)
    covariance = torch.einsum("...ij,kl->...ikjl", row_cov, col_cov).reshape(
        *distribution.batch_shape, 6, 6
    )
    dense = torch.distributions.MultivariateNormal(
        args["loc"].flatten(-2), covariance_matrix=covariance
    )
    actual = distribution.log_prob(value)
    expected = dense.log_prob(value.flatten())
    torch.testing.assert_close(actual, expected)
    for name, observed, oracle in zip(
        args,
        torch.autograd.grad(actual.sum(), tuple(args.values()), retain_graph=True),
        torch.autograd.grad(expected.sum(), tuple(args.values())),
        strict=True,
    ):
        # Factors live on the lower-triangular carrier; derivatives normal to
        # that carrier need not match a dense unconstrained matrix product.
        if name.endswith("scale_tril"):
            observed, oracle = observed.tril(), oracle.tril()
        torch.testing.assert_close(observed, oracle)
    torch.testing.assert_close(distribution.mean, args["loc"])
    torch.testing.assert_close(
        distribution.variance,
        covariance.diagonal(dim1=-2, dim2=-1).reshape(*distribution.batch_shape, 2, 3),
    )
    expanded = distribution.expand((5, *distribution.batch_shape))
    assert expanded.rsample((7,)).shape == (7, 5, *distribution.batch_shape, 2, 3)
    torch.testing.assert_close(
        expanded.log_prob(value), expected.expand(5, *distribution.batch_shape)
    )


@pytest.mark.parametrize(("row_factor", "col_factor"), _VARIANTS)
def test_reference_scores_and_seeded_draws_share_all_four_parameterizations(
    row_factor, col_factor
):
    args, _, _ = _parameters(row_factor, col_factor)
    value = ((0.4, -0.1, 0.7), (-0.2, 0.8, 0.3))
    host = {name: tensor.tolist() for name, tensor in args.items()}
    backend = ReferenceBackend()
    assert backend.family_log_prob("MatrixNormal", host, value) == pytest.approx(
        float(MatrixNormal(**args).log_prob(torch.tensor(value, dtype=torch.double))),
        abs=1e-12,
    )
    canonical, _, _ = _parameters(False, False)
    expected = backend.family_sample(
        "MatrixNormal",
        {name: value.tolist() for name, value in canonical.items()},
        random.Random(74),
    )
    torch.testing.assert_close(
        torch.tensor(
            backend.family_sample("MatrixNormal", host, random.Random(74)),
            dtype=torch.double,
        ),
        torch.tensor(expected, dtype=torch.double),
        atol=1e-15,
        rtol=1e-15,
    )


def test_supplied_factors_never_cholesky_or_materialize_covariances():
    args, _, _ = _parameters(True, True)
    with patch(
        "torch.linalg.cholesky", side_effect=AssertionError("unexpected factorization")
    ):
        distribution = MatrixNormal(**args)
        distribution.log_prob(args["loc"])
        distribution.rsample((3,))
        distribution.expand((4,)).log_prob(args["loc"])
        distribution.variance
    assert "row_covariance" not in distribution.__dict__
    assert "col_covariance" not in distribution.__dict__


@pytest.mark.parametrize(
    "bad", ["negative", "upper", "shape", "both", "missing", "asymmetric"]
)
def test_invalid_factor_contracts_match_reference(bad):
    args, _, _ = _parameters(True, True)
    if bad == "negative":
        args["row_scale_tril"][0, 0] = -1
    elif bad == "upper":
        args["row_scale_tril"][0, 1] = 0.1
    elif bad == "shape":
        args["row_scale_tril"] = torch.eye(3, dtype=torch.double)
    elif bad == "both":
        args["row_covariance"] = torch.eye(2, dtype=torch.double)
    elif bad == "asymmetric":
        del args["row_scale_tril"]
        args["row_covariance"] = torch.tensor(
            [[1.0, 0.5], [0.1, 1.0]], dtype=torch.double
        )
    else:
        del args["row_scale_tril"]
    with pytest.raises(ValueError):
        MatrixNormal(**args)
    with pytest.raises(ValueError):
        ReferenceBackend().family_log_prob(
            "MatrixNormal",
            {key: value.tolist() for key, value in args.items()},
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )


def test_factor_reverse_forward_and_second_order_derivatives():
    loc = torch.zeros(2, 3, dtype=torch.double)
    value = torch.arange(6, dtype=torch.double).reshape(2, 3) / 10

    def score(raw):
        row = torch.diag_embed(raw[:2].exp())
        col = torch.diag_embed(raw[2:].exp())
        return MatrixNormal(loc, row_scale_tril=row, col_scale_tril=col).log_prob(value)

    raw = torch.tensor(
        [0.1, -0.2, 0.3, -0.1, 0.2], dtype=torch.double, requires_grad=True
    )
    assert torch.autograd.gradcheck(score, (raw,))
    assert torch.autograd.gradgradcheck(score, (raw,))
    torch.testing.assert_close(
        torch.func.jacfwd(score)(raw), torch.func.jacrev(score)(raw)
    )


def test_batch_selection_retains_factor_coordinates_without_refactorization():
    from quivers.continuous._batch_selection import (
        select_batch,
        expand_registered_batch,
    )

    args, _, _ = _parameters(True, True, batch=True)
    distribution = MatrixNormal(**args)
    with patch(
        "torch.linalg.cholesky", side_effect=AssertionError("unexpected factorization")
    ):
        selected = select_batch(distribution, torch.tensor([True, False, True, False]))
        expanded = expand_registered_batch(selected, torch.Size((3, 2)))
        torch.testing.assert_close(selected.loc, args["loc"][[0, 2]])
        torch.testing.assert_close(
            expanded.log_prob(args["loc"][[0, 2]]),
            selected.log_prob(args["loc"][[0, 2]]).expand(3, 2),
        )
    assert "row_covariance" not in selected.__dict__
    assert "col_covariance" not in selected.__dict__


def test_conditional_matrix_normal_reuses_learned_factors_directly():
    from quivers.continuous.families import ConditionalMatrixNormal
    from quivers.continuous.spaces import Euclidean

    kernel = ConditionalMatrixNormal(
        Euclidean(name="Input", dim=1), Euclidean(name="Output", dim=6), rows=2, cols=3
    ).double()
    x = torch.tensor([[0.1], [0.4]], dtype=torch.double)
    with patch(
        "torch.linalg.cholesky",
        side_effect=AssertionError("unexpected learned refactorization"),
    ):
        law = kernel._get_dist(x)
        score = law.log_prob(torch.zeros(2, 2, 3, dtype=torch.double)).sum()
        gradients = torch.autograd.grad(score, tuple(kernel.parameters()))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert law.rsample().shape == (2, 2, 3)
    assert "row_covariance" not in law.__dict__
    assert "col_covariance" not in law.__dict__


def test_factor_sampler_has_independent_kronecker_moments_and_mixed_precision_ad():
    args, row, col = _parameters(True, True)
    loc = args["loc"].float().requires_grad_()
    row = row.clone().requires_grad_()
    col = col.float().requires_grad_()
    distribution = MatrixNormal(loc, row_scale_tril=row, col_scale_tril=col)
    assert distribution.loc.dtype == torch.double
    with torch.random.fork_rng():
        torch.manual_seed(142)
        sample = distribution.rsample((30000,))
    expected = torch.kron(row @ row.T, (col @ col.T).double())
    torch.testing.assert_close(sample.mean(0), loc.double(), atol=0.025, rtol=0)
    centered = sample.flatten(-2) - loc.double().flatten()
    torch.testing.assert_close(
        centered.T @ centered / len(sample), expected, atol=0.025, rtol=0.025
    )
    gradients = torch.autograd.grad(
        distribution.log_prob(torch.zeros(2, 3, dtype=torch.double)).sum(),
        (loc, row, col),
    )
    assert tuple(value.dtype for value in gradients) == (
        torch.float32,
        torch.float64,
        torch.float32,
    )
    assert all(torch.isfinite(value).all() for value in gradients)


def test_shared_covariances_are_factored_before_location_batch_broadcast():
    args, _, _ = _parameters(False, False)
    args["loc"] = args["loc"].expand(7, 2, 3)
    original = torch.linalg.cholesky
    with patch("torch.linalg.cholesky", wraps=original) as cholesky:
        distribution = MatrixNormal(**args)
    assert [call.args[0].shape for call in cholesky.call_args_list] == [(2, 2), (3, 3)]
    assert distribution.variance.shape == (7, 2, 3)
    assert distribution.log_prob(torch.zeros(2, 3, dtype=torch.double)).shape == (7,)
