"""Residual transforms agree with independent innovation and spatial oracles."""

import math

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal, multivariate_t
import torch

from quivers.formulas._residual_dependence import (
    ResidualKernel,
    arma_factor,
    compound_symmetry_factor,
)


def _tensor(value, *, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


@pytest.mark.parametrize("stationary", [False, True])
@pytest.mark.parametrize("phi,theta", [(0.0, 0.0), (0.3, -0.2), (-0.4, 0.5)])
def test_arma_density_and_every_parameter_derivative_match_innovation_oracle(
    stationary, phi, theta
):
    loc = _tensor([0.1, -0.2, 0.3], grad=True)
    scale = _tensor([0.7, 1.1, 0.9], grad=True)
    ar, ma = _tensor([phi], grad=True), _tensor([theta], grad=True)
    y = _tensor([0.5, -0.3, 1.2])
    actual = ResidualKernel(kind="arma", stationary=stationary)(
        {"loc": loc, "scale": scale, "ar": ar, "ma": ma}
    ).log_prob(y)
    if stationary:
        p, q = ar[0], ma[0]
        g0 = (1 + q * q + 2 * p * q) / (1 - p * p)
        g1 = (p + q) * (1 + p * q) / (1 - p * p)
        covariance = torch.stack(
            (
                torch.stack((g0, g1, p * g1)),
                torch.stack((g1, g0, g1)),
                torch.stack((p * g1, g1, g0)),
            )
        )
        covariance = scale[:, None] * covariance * scale[None, :]
        inverse = torch.linalg.inv(covariance)
        residual = y - loc
        expected = -0.5 * (
            3 * math.log(2 * math.pi)
            + torch.linalg.slogdet(covariance)[1]
            + residual @ inverse @ residual
        )
    else:
        residual = y - loc
        innovations = []
        for i in range(3):
            previous_residual = residual[i - 1] if i else residual.new_zeros(())
            previous_innovation = innovations[-1] if i else residual.new_zeros(())
            innovations.append(
                residual[i] - ar[0] * previous_residual - ma[0] * previous_innovation
            )
        standardized = torch.stack(innovations) / scale
        expected = (
            -0.5 * standardized.square().sum()
            - scale.log().sum()
            - 1.5 * math.log(2 * math.pi)
        )
    torch.testing.assert_close(actual, expected)
    parameters = (loc, scale, ar, ma)
    actual_grad = torch.autograd.grad(actual, parameters, create_graph=True)
    expected_grad = torch.autograd.grad(expected, parameters, create_graph=True)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        ag = torch.autograd.grad(
            a.sum(), parameters, retain_graph=True, allow_unused=True
        )
        bg = torch.autograd.grad(
            b.sum(), parameters, retain_graph=True, allow_unused=True
        )
        for x, z, p in zip(ag, bg, parameters, strict=True):
            torch.testing.assert_close(
                torch.zeros_like(p) if x is None else x,
                torch.zeros_like(p) if z is None else z,
            )


def test_higher_order_arma_matches_independent_numpy_triangular_filter_and_broadcasts():
    ar = _tensor([[0.3, -0.1], [0.1, 0.2]])
    ma = _tensor([0.2, -0.1, 0.05])
    scale = _tensor([0.6, 0.8, 1.0, 1.2])
    factor = arma_factor(scale, ar, ma)
    assert factor.shape == (2, 4, 4)
    for i in range(2):
        a = toeplitz(np.r_[1.0, -ar[i].numpy(), 0.0], np.array([1.0, 0.0, 0.0, 0.0]))
        b = toeplitz(np.r_[1.0, ma.numpy()], np.array([1.0, 0.0, 0.0, 0.0]))
        expected = np.linalg.solve(a, b @ np.diag(scale.numpy()))
        np.testing.assert_allclose(factor[i].numpy(), expected, atol=1e-14)
    singleton = arma_factor(_tensor([0.5]), _tensor([]), _tensor([]))
    torch.testing.assert_close(singleton, _tensor([[0.5]]))


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize(
    "kind", ["fixed", "cosy", "unstructured", "sar_error", "sar_lag"]
)
def test_native_kernels_match_scipy_and_preserve_all_parameter_gradients(kind, student):
    matrix = np.array([[1.0, 0.25, 0.1], [0.25, 1.2, 0.15], [0.1, 0.15, 0.8]])
    weights = np.array([[0.0, 0.7, 0.0], [0.2, 0.0, 0.4], [0.0, 0.5, 0.0]])
    loc = _tensor([0.1, 0.3, -0.2], grad=True)
    scale = _tensor([0.8, 1.1, 0.7], grad=True)
    rho = _tensor(0.2, grad=True)
    factor = torch.linalg.cholesky(_tensor(matrix)).requires_grad_()
    nu = _tensor(5.0, grad=True)
    values = {"loc": loc, "scale": scale, "rho": rho, "factor": factor, "nu": nu}
    known = matrix if kind == "fixed" else weights if kind.startswith("sar") else ()
    kernel = ResidualKernel(kind=kind, matrix=known, student=student)
    distribution = kernel(values)
    s = scale.detach().numpy()
    mean = loc.detach().numpy()
    if kind == "fixed":
        cov = s[:, None] * matrix * s[None, :]
    elif kind == "unstructured":
        cov = s[:, None] * matrix * s[None, :]
    elif kind == "cosy":
        cov = s[:, None] * (np.eye(3) + 0.2 * (1 - np.eye(3))) * s[None, :]
    else:
        operator = np.eye(3) - 0.2 * weights
        f = np.linalg.solve(operator, np.diag(s))
        cov = f @ f.T
        if kind == "sar_lag":
            mean = np.linalg.solve(operator, mean)
    y = _tensor([0.4, -0.6, 0.2])
    expected = (
        multivariate_t.logpdf(y.numpy(), loc=mean, shape=cov, df=5.0)
        if student
        else multivariate_normal.logpdf(y.numpy(), mean=mean, cov=cov)
    )
    np.testing.assert_allclose(
        distribution.log_prob(y).detach().numpy(), expected, rtol=1e-12, atol=1e-12
    )
    applicable = [loc, scale]
    if kind in {"cosy", "sar_error", "sar_lag"}:
        applicable.append(rho)
    if kind == "unstructured":
        applicable.append(factor)
    if student:
        applicable.append(nu)
    gradients = torch.autograd.grad(distribution.log_prob(y), applicable)
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    assert distribution.sample((7,)).shape == (7, 3)


def test_fixed_covariance_is_factored_once_and_copies_known_data(monkeypatch):
    matrix = _tensor([[1.0, 0.2], [0.2, 1.0]])
    kernel = ResidualKernel(kind="fixed", matrix=matrix)
    matrix.fill_(99)
    y = _tensor([0.3, -0.4])
    expected = multivariate_normal.logpdf(y.numpy(), cov=[[1.0, 0.2], [0.2, 1.0]])
    first = kernel({"loc": _tensor([0.0, 0.0]), "scale": _tensor([1.0, 1.0])})
    np.testing.assert_allclose(float(first.log_prob(y)), expected, rtol=1e-12)

    def fail(*args, **kwargs):
        raise AssertionError("known covariance was factored during replay")

    monkeypatch.setattr(np.linalg, "cholesky", fail)
    monkeypatch.setattr(torch.linalg, "cholesky", fail)
    for _ in range(2):
        law = kernel({"loc": _tensor([0.0, 0.0]), "scale": _tensor([1.0, 1.0])})
        np.testing.assert_allclose(float(law.log_prob(y)), expected, rtol=1e-12)


def test_exact_compound_symmetry_support_and_singular_spatial_operator_are_checked():
    for rho in [-0.5, 1.0]:
        with pytest.raises(ValueError, match="domain"):
            compound_symmetry_factor(_tensor([1.0, 1.0, 1.0]), _tensor(rho))
    singular = ResidualKernel(kind="sar_lag", matrix=torch.eye(2, dtype=torch.float64))
    score = singular(
        {"loc": _tensor([0.0, 0.0]), "scale": _tensor([1.0, 1.0]), "rho": _tensor(1.0)}
    ).log_prob(_tensor([0.1, 0.2]))
    assert score == -torch.inf
    with pytest.raises(ValueError, match="orders"):
        arma_factor(
            _tensor([1.0, 1.0]), _tensor([0.1, 0.2]), _tensor([]), stationary=True
        )
    with pytest.raises(ValueError, match="abs"):
        arma_factor(_tensor([1.0, 1.0]), _tensor([1.0]), _tensor([]), stationary=True)


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_residual_catalog_runs_checked_source_and_existing_inference_methods(
    student, method
):
    from quivers.formulas._residual_dependence import residual_catalog
    from quivers.modeling import Model, VariationalMethod, MCMCMethod
    from quivers.inference import MCMC, HMCKernel, NUTSKernel

    kernel = ResidualKernel(kind="arma", student=student)
    parameters = ",nu=nu" if student else ""
    degrees = "    sample nu <- Gamma(concentration=3.0,rate=1.0)\n" if student else ""
    source = (
        """object Row:FinSet _
object Lag:FinSet 1
object Moving:FinSet 1
object One:FinSet 1
program regression(x:Tensor[Real]([Row]),scale:Tensor[Real]([Row]),ma:Tensor[Real]([Moving]),y:Tensor[Real]([Row])):One->One
    sample beta <- Normal(loc=0.0,scale=1.0)
    sample ar:Lag <- Uniform(low=-0.6,high=0.6)
"""
        + degrees
        + """    let loc = beta * x
    observe y <- ResidualNormal(loc=loc,scale=scale,ar=ar,ma=ma"""
        + parameters
        + """)
    return beta
export regression
"""
    )
    data = {
        "x": _tensor([1.0, 2.0, 3.0]),
        "scale": _tensor([0.8, 0.9, 1.0]),
        "ma": _tensor([0.0]),
        "y": _tensor([0.1, 0.5, 0.3]),
    }
    model = Model(source, families=residual_catalog(kernel))
    if method == "svi":
        settings = VariationalMethod(num_steps=2, seed=7)
    elif method == "hmc":
        settings = MCMCMethod(
            sampler=MCMC(HMCKernel(), num_warmup=2, num_samples=3, num_chains=1, seed=7)
        )
    else:
        settings = MCMCMethod(
            sampler=MCMC(
                NUTSKernel(), num_warmup=2, num_samples=3, num_chains=1, seed=7
            )
        )
    fitted = model.fit(data, method=settings)
    scores = fitted.log_likelihood(num_samples=3, seed=11)["y"]
    assert scores.shape == (1, 3, 1)
    assert bool(torch.isfinite(scores).all())
    fresh = {
        **data,
        "x": _tensor([0.5, 1.0]),
        "scale": _tensor([0.7, 0.9]),
        "y": _tensor([0.2, 0.4]),
    }
    prediction = fitted.predict(fresh, num_samples=3, seed=13)
    assert prediction["y"].shape[-1] == 2


def test_student_innovation_filter_has_independent_radial_noise_and_linear_storage(
    monkeypatch,
):
    loc = _tensor([0.1, -0.2, 0.3, 0.5], grad=True)
    scale = _tensor([0.7, 0.9, 1.2, 0.8], grad=True)
    ar = _tensor([0.2, -0.1], grad=True)
    ma = _tensor([0.3], grad=True)
    nu = _tensor(4.5, grad=True)
    y = _tensor([0.4, 0.8, -0.1, 0.7])
    law = ResidualKernel(kind="arma", student=True)(
        {"loc": loc, "scale": scale, "ar": ar, "ma": ma, "nu": nu}
    )
    residual = y - loc
    errors = []
    for i in range(4):
        error = residual[i]
        if i > 0:
            error = error - ar[0] * residual[i - 1] - ma[0] * errors[i - 1]
        if i > 1:
            error = error - ar[1] * residual[i - 2]
        errors.append(error)
    standardized = torch.stack(errors) / scale
    expected = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * math.pi)
        - scale.log()
        - (nu + 1) / 2 * torch.log1p(standardized.square() / nu)
    ).sum()

    def no_factor(*args, **kwargs):
        raise AssertionError("innovation scoring constructed a dense residual factor")

    import quivers.formulas._residual_dependence as module

    monkeypatch.setattr(module, "arma_factor", no_factor)
    torch.testing.assert_close(law.log_prob(y), expected)
    for a, b in zip(
        torch.autograd.grad(
            law.log_prob(y), (loc, scale, ar, ma, nu), retain_graph=True
        ),
        torch.autograd.grad(expected, (loc, scale, ar, ma, nu)),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    draws = law.rsample((7,))
    assert draws.shape == (7, 4)
    transform = law.transforms[0]
    with torch.no_grad():
        torch.testing.assert_close(transform(transform.inv(draws)), draws)
        independent = torch.randn(
            7, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3)
        )
        torch.testing.assert_close(transform.inv(transform(independent)), independent)
        torch.testing.assert_close(
            transform.log_abs_det_jacobian(independent, transform(independent)),
            torch.zeros(7, dtype=torch.float64),
        )
