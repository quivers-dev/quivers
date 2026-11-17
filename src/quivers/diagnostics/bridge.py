"""Marginal likelihoods by bridge sampling, as the R ``bridgesampling`` package.

Bridge sampling [@meng1996simulating] estimates the normalizing constant

.. math::

    p(y) = \\int p(y \\mid \\theta)\\, p(\\theta)\\, d\\theta

of an unnormalized posterior :math:`q(\\theta) = p(y \\mid \\theta) p(\\theta)`
from posterior draws and draws of a normalized proposal :math:`g`. For any
bridge function :math:`h`,

.. math::

    p(y) = \\frac{E_g[q(\\theta) h(\\theta)]}{E_{\\text{post}}[g(\\theta) h(\\theta)]},

and the asymptotically optimal :math:`h` gives the fixed-point iteration

.. math::

    \\hat r^{(t+1)} = \\frac{N_1}{N_2}
    \\frac{\\sum_{i=1}^{N_2} \\ell_{2,i} / (s_1 \\ell_{2,i} + s_2 \\hat r^{(t)})}
         {\\sum_{j=1}^{N_1} 1 / (s_1 \\ell_{1,j} + s_2 \\hat r^{(t)})},
    \\qquad \\ell = q / g,

with :math:`s_1 = N_\\text{eff} / (N_\\text{eff} + N_2)` and
:math:`s_2 = N_2 / (N_\\text{eff} + N_2)`. Following the tutorial of
@gronau2017tutorial and the ``bridgesampling`` package [@gronau2020bridgesampling]
that brms calls, the draws are on the unconstrained scale, the first half of
every chain fits the proposal, the second half enters the iteration, and the
proposal is either a multivariate normal with the fitted mean and covariance
(``"normal"``) or the standard normal after the third-order warp of
@meng2002warp (``"warp3"``), which matches the posterior's mean, covariance
and skewness. The relative mean-squared error of the ``"normal"`` estimator
follows @fruhwirth2004estimating.

Every step reproduces the R implementation: Matrix ``nearPD`` for the
proposal covariance, coda ``effectiveSize`` (an autoregressive spectral
estimate at frequency zero) for :math:`N_\\text{eff}`, the median of
:math:`\\log \\ell_1` as the numerical offset, the convergence criteria and
tolerances, and the restart from the geometric mean of the last two
iterates when the first scheme does not converge. Log-domain sums replace
the package's arbitrary-precision arithmetic. Proposal draws come from a
seeded Torch stream, so estimates agree with R only within their Monte
Carlo error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Literal

import didactic.api as dx
import numpy as np
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel

type BridgeMethod = Literal["normal", "warp3"]
"""Proposal of the bridge sampler: a fitted normal or the warp-3 transformation."""

#: R's ``.Machine$double.eps ^ 0.5``, the ``all.equal`` tolerance.
_ALL_EQUAL = 1.4901161193847656e-08


class BridgeRelativeError(ValueModel):
    """Approximate error of one ``"normal"`` bridge estimate.

    Parameters
    ----------
    re2 : float
        Approximate relative mean-squared error of the marginal likelihood
        estimate, :math:`E[(\\hat p(y) - p(y))^2] / p(y)^2`.
    cv : float
        Coefficient of variation, the square root of ``re2``.
    """

    re2: float
    cv: float

    @property
    def percentage(self) -> str:
        """Return the coefficient of variation as a percentage label.

        Returns
        -------
        str
            ``cv`` times 100 with a percent sign, as ``scales::percent``.
        """
        return f"{100.0 * self.cv:.0f}%"


class BridgeRepetitionSpread(ValueModel):
    """Spread of the log marginal likelihood over independent repetitions.

    Parameters
    ----------
    min, max : float
        Smallest and largest log marginal likelihood.
    iqr : float
        Interquartile range (R ``IQR``, quantile type 7).
    """

    min: float
    max: float
    iqr: float


class BridgeResult(ValueModel):
    """A bridge-sampling estimate, as a ``bridge`` or ``bridge_list`` object.

    Parameters
    ----------
    logml_repetitions : tuple[float, ...]
        Log marginal likelihood of each independent proposal draw.
    niter : tuple[int, ...]
        Fixed-point iterations of each repetition, counting a restart's
        iterations after the ``maxiter`` of the first scheme.
    method : {"normal", "warp3"}
        Proposal used.
    neff : float
        Effective sample size of the iteration draws used in ``s1``.
    q11 : torch.Tensor
        Log unnormalized posterior at the iteration draws ``(N1,)``, warped
        for ``"warp3"``.
    q12 : torch.Tensor
        Log proposal density at the iteration draws ``(N1,)``.
    q21 : torch.Tensor
        Log unnormalized posterior at each repetition's proposal draws
        ``(R, N2)``.
    q22 : torch.Tensor
        Log proposal density at each repetition's proposal draws ``(R, N2)``.
    log_det_warp : float
        Log determinant of the warp's Cholesky factor; zero for ``"normal"``.
    nonfinite_posterior : int
        Iteration draws whose log posterior was ``-inf`` or undefined.
    nonfinite_proposal : int
        Proposal draws whose log posterior was ``-inf`` or undefined.
    """

    logml_repetitions: tuple[float, ...]
    niter: tuple[int, ...]
    method: BridgeMethod
    neff: float
    q11: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    q12: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    q21: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    q22: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_det_warp: float = 0.0
    nonfinite_posterior: int = 0
    nonfinite_proposal: int = 0

    @property
    def logml(self) -> float:
        """Return the log marginal likelihood (median over repetitions).

        Returns
        -------
        float
            The estimate, as ``bridgesampling::logml``.
        """
        return float(np.median(self.logml_repetitions))

    @property
    def repetitions(self) -> int:
        """Return the number of independent proposal draws.

        Returns
        -------
        int
            Repetition count.
        """
        return len(self.logml_repetitions)

    def error_measures(self) -> BridgeRelativeError | BridgeRepetitionSpread:
        """Return the estimator's approximate error, as ``error_measures``.

        Returns
        -------
        BridgeRelativeError or BridgeRepetitionSpread
            The relative mean-squared error of one ``"normal"`` estimate
            [@fruhwirth2004estimating], or the spread of several repetitions.

        Raises
        ------
        ValueError
            For one ``"warp3"`` repetition, for which the package defines no
            approximate error; run several repetitions instead.

        Notes
        -----
        With :math:`p = q / \\hat p(y)` the normalized posterior,
        :math:`f_1 = p(\\tilde\\theta) / (s_1 p(\\tilde\\theta) + s_2 g(\\tilde\\theta))`
        at proposal draws and
        :math:`f_2 = g(\\theta) / (s_1 p(\\theta) + s_2 g(\\theta))` at
        posterior draws, both with :math:`s_i = N_i / (N_1 + N_2)`,

        .. math::

            \\widehat{RE}^2 = \\frac{V(f_1)}{N_2 \\bar f_1^2}
            + \\frac{\\rho_{f_2}(0)}{N_1} \\frac{V(f_2)}{\\bar f_2^2},

        where :math:`\\rho_{f_2}(0)` is the autoregressive spectral density
        of :math:`f_2` at frequency zero (coda ``spectrum0.ar``).
        """
        if self.repetitions > 1:
            values = np.asarray(self.logml_repetitions, dtype=np.float64)
            q75, q25 = np.quantile(values, [0.75, 0.25])
            return BridgeRepetitionSpread(
                min=float(values.min()), max=float(values.max()), iqr=float(q75 - q25)
            )
        if self.method == "warp3":
            raise ValueError(
                "error_measures is not defined for the warp3 method with one "
                "repetition; run bridge_sampler with repetitions > 1 to assess "
                "the uncertainty of the estimate"
            )
        logml = self.logml_repetitions[0]
        q11 = self.q11.double().numpy()
        q12 = self.q12.double().numpy()
        q21 = self.q21[0].double().numpy()
        q22 = self.q22[0].double().numpy()
        n1, n2 = q11.shape[0], q22.shape[0]
        log_s1 = math.log(n1 / (n1 + n2))
        log_s2 = math.log(n2 / (n1 + n2))
        log_pg = q21 - logml
        log_pp = q11 - logml
        f1 = np.exp(log_pg - np.logaddexp(log_s1 + log_pg, log_s2 + q22))
        f2 = np.exp(q12 - np.logaddexp(log_s1 + log_pp, log_s2 + q12))
        rho = ar_spectrum0(f2)
        term1 = np.var(f1, ddof=1) / np.mean(f1) ** 2 / n2
        term2 = rho / n1 * np.var(f2, ddof=1) / np.mean(f2) ** 2
        re2 = float(term1 + term2)
        return BridgeRelativeError(re2=re2, cv=math.sqrt(re2))


def ar_spectrum0(series: np.ndarray) -> float:
    """Return the AR spectral density at frequency zero, as coda ``spectrum0.ar``.

    Parameters
    ----------
    series : numpy.ndarray
        One real series of length at least two.

    Returns
    -------
    float
        ``var.pred / (1 - sum(ar))^2`` of the Yule-Walker autoregression whose
        order minimizes the AIC (R ``ar(x, aic = TRUE)``), or zero when the
        residuals of a linear trend have zero standard deviation.

    Raises
    ------
    ValueError
        If the series is not finite or has fewer than two values.

    Notes
    -----
    The autocovariances are the biased (divisor ``n``) estimates of R
    ``acf``; the Levinson-Durbin recursion gives the innovation variances
    of orders zero to ``min(n - 1, floor(10 log10 n))``. Cost is
    O(n p + p^2) for order bound p.
    """
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n < 2 or not np.isfinite(x).all():
        raise ValueError("spectrum0.ar requires at least two finite values")
    time = np.arange(1, n + 1, dtype=np.float64)
    design = np.column_stack([np.ones(n), time])
    coefficients, *_ = np.linalg.lstsq(design, x, rcond=None)
    residuals = x - design @ coefficients
    if np.std(residuals, ddof=1) <= _ALL_EQUAL:
        return 0.0
    order_max = min(n - 1, math.floor(10.0 * math.log10(n)))
    centered = x - x.mean()
    acov = np.array(
        [np.dot(centered[: n - k], centered[k:]) / n for k in range(order_max + 1)]
    )
    if acov[0] == 0.0:
        raise ValueError("spectrum0.ar requires a series with nonzero variance")
    variances = [acov[0]]
    phi = np.zeros(0)
    fits = [phi]
    for k in range(1, order_max + 1):
        reflection = (acov[k] - np.dot(phi, acov[k - 1 : 0 : -1])) / variances[-1]
        phi = np.concatenate([phi - reflection * phi[::-1], [reflection]])
        fits.append(phi)
        variances.append(variances[-1] * (1.0 - reflection**2))
    aic = n * np.log(np.asarray(variances)) + 2.0 * np.arange(order_max + 1) + 2.0
    order = int(np.argmin(aic))
    var_pred = variances[order] * n / (n - (order + 1))
    return float(var_pred / (1.0 - fits[order].sum()) ** 2)


def effective_size(chains: Sequence[np.ndarray]) -> np.ndarray:
    """Return coda ``effectiveSize`` of an ``mcmc.list``.

    Parameters
    ----------
    chains : Sequence[numpy.ndarray]
        One ``(draws, parameters)`` array per chain.

    Returns
    -------
    numpy.ndarray
        Per-parameter effective sizes summed over chains, each chain
        contributing ``n var(x) / spectrum0.ar(x)`` (zero for a constant).
    """
    total = np.zeros(chains[0].shape[1], dtype=np.float64)
    for chain in chains:
        n = chain.shape[0]
        for column in range(chain.shape[1]):
            spec = ar_spectrum0(chain[:, column])
            if spec != 0.0:
                total[column] += n * np.var(chain[:, column], ddof=1) / spec
    return total


def near_pd(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest positive-definite matrix, as Matrix ``nearPD``.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square real matrix.

    Returns
    -------
    numpy.ndarray
        Higham's alternating projection with Dykstra's correction
        (``eig.tol = 1e-6``, ``conv.tol = 1e-7``, ``maxit = 100``, infinity
        norm), followed by the ``posd.tol = 1e-8`` eigenvalue floor, as
        ``nearPD`` with its defaults.

    Raises
    ------
    ValueError
        If the matrix is negative semidefinite or the projection does not
        converge within 100 iterations.
    """
    x = np.asarray(matrix, dtype=np.float64)
    if not np.array_equal(x, x.T):
        x = (x + x.T) / 2.0
    n = x.shape[0]
    correction = np.zeros_like(x)
    current = x
    converged = False
    for _ in range(100):
        previous = current
        residual = previous - correction
        values, vectors = np.linalg.eigh(residual)
        values, vectors = values[::-1], vectors[:, ::-1]
        keep = values > 1e-6 * values[0]
        if not keep.any():
            raise ValueError("nearPD: the matrix seems negative semidefinite")
        kept = vectors[:, keep]
        current = (kept * values[keep]) @ kept.T
        correction = current - residual
        change = np.abs(previous - current).sum(axis=1).max()
        scale = np.abs(previous).sum(axis=1).max()
        if change / scale <= 1e-7:
            converged = True
            break
    if not converged:
        raise ValueError("nearPD did not converge in 100 iterations")
    values, vectors = np.linalg.eigh(current)
    values, vectors = values[::-1].copy(), vectors[:, ::-1]
    floor = 1e-8 * abs(values[0])
    if values[n - 1] < floor:
        values[values < floor] = floor
        diagonal = np.diag(current).copy()
        current = vectors @ (values[:, None] * vectors.T)
        scale = np.sqrt(np.maximum(floor, diagonal) / np.diag(current))
        current = scale[:, None] * current * scale[None, :]
    return current


def _mvn_logpdf(values: np.ndarray, mean: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """Return multivariate normal log densities from a Cholesky factor.

    Parameters
    ----------
    values : numpy.ndarray
        Points ``(N, D)``.
    mean : numpy.ndarray
        Mean ``(D,)``.
    chol : numpy.ndarray
        Lower Cholesky factor of the covariance ``(D, D)``.

    Returns
    -------
    numpy.ndarray
        Log densities ``(N,)``.
    """
    standardized = np.linalg.solve(chol, (values - mean).T)
    dim = values.shape[1]
    return (
        -0.5 * (standardized**2).sum(axis=0)
        - np.log(np.diag(chol)).sum()
        - 0.5 * dim * math.log(2.0 * math.pi)
    )


def _iterate(
    l1: np.ndarray,
    l2: np.ndarray,
    *,
    neff: float,
    r0: float,
    tol: float,
    maxiter: int,
    criterion: Literal["r", "logml"],
) -> tuple[float | None, int, list[float]]:
    """Run the fixed-point scheme of ``.run.iterative.scheme``.

    Parameters
    ----------
    l1, l2 : numpy.ndarray
        Log ratios ``q / g`` at iteration and proposal draws.
    neff : float
        Effective size of the iteration draws.
    r0 : float
        Starting value of ``r``.
    tol : float
        Convergence tolerance.
    maxiter : int
        Iteration bound.
    criterion : {"r", "logml"}
        Relative change of ``r`` or of the log marginal likelihood.

    Returns
    -------
    tuple[float or None, int, list[float]]
        The log marginal likelihood (None without convergence), iteration
        count, and the iterates of ``r`` from ``r0``.

    Raises
    ------
    RuntimeError
        If an iterate is not finite and positive.
    """
    lstar = float(np.median(l1))
    n1, n2 = l1.shape[0], l2.shape[0]
    log_s1 = math.log(neff / (neff + n2))
    log_s2 = math.log(n2 / (neff + n2))
    log_r = math.log(r0)
    r_values = [r0]
    logml = log_r + lstar
    change = 1.0 + tol
    i = 1
    a = l2 - lstar
    b = l1 - lstar
    while i <= maxiter and change > tol:
        log_r_old, logml_old = log_r, logml
        log_num = a - np.logaddexp(log_s1 + a, log_s2 + log_r)
        log_den = -np.logaddexp(log_s1 + b, log_s2 + log_r)
        log_r = (
            math.log(n1 / n2)
            + float(np.logaddexp.reduce(log_num))
            - float(np.logaddexp.reduce(log_den))
        )
        if not math.isfinite(log_r):
            raise RuntimeError(
                "bridge sampling produced an infinite value in the iterative "
                "scheme; rerun with more posterior draws"
            )
        r_values.append(math.exp(log_r))
        logml = log_r + lstar
        if criterion == "r":
            change = abs(1.0 - math.exp(log_r_old - log_r))
        else:
            change = abs((logml - logml_old) / logml)
        i += 1
    if i >= maxiter:
        return None, i - 1, r_values
    return logml, i - 1, r_values


def _solve(
    l1: np.ndarray, l2: np.ndarray, *, neff: float, maxiter: int
) -> tuple[float, int]:
    """Solve the scheme, restarting as ``bridgesampling`` does.

    Parameters
    ----------
    l1, l2 : numpy.ndarray
        Log ratios ``q / g`` at iteration and proposal draws.
    neff : float
        Effective size of the iteration draws.
    maxiter : int
        Iteration bound of each scheme.

    Returns
    -------
    tuple[float, int]
        Log marginal likelihood and total iterations.

    Raises
    ------
    RuntimeError
        If neither the scheme from ``r0 = 0.5`` with tolerance ``1e-10`` on
        ``r`` nor the restart with tolerance ``1e-4`` on the log marginal
        likelihood converges within ``maxiter``.
    """
    logml, niter, r_values = _iterate(
        l1, l2, neff=neff, r0=0.5, tol=1e-10, maxiter=maxiter, criterion="r"
    )
    if logml is not None:
        return logml, niter
    restart = math.sqrt(r_values[-2] * r_values[-1])
    logml, niter, _ = _iterate(
        l1, l2, neff=neff, r0=restart, tol=1e-4, maxiter=maxiter, criterion="logml"
    )
    if logml is None:
        raise RuntimeError(
            f"bridge sampling did not converge within maxiter={maxiter} "
            "iterations, also after restarting; increase maxiter or the "
            "number of posterior draws"
        )
    return logml, maxiter + niter


def bridge_sampler(
    log_posterior: Callable[[np.ndarray], np.ndarray],
    chains: np.ndarray,
    *,
    method: BridgeMethod = "normal",
    repetitions: int = 1,
    use_neff: bool = True,
    maxiter: int = 1000,
    seed: int = 0,
) -> BridgeResult:
    """Estimate a log marginal likelihood from unconstrained posterior chains.

    Parameters
    ----------
    log_posterior : Callable[[numpy.ndarray], numpy.ndarray]
        Log unnormalized posterior density on the unconstrained scale,
        including every normalizing constant of the prior and likelihood and
        the log Jacobian of the constraining transforms, evaluated row-wise
        at ``(N, D)`` points.
    chains : numpy.ndarray
        Unconstrained posterior draws ``(chain, draw, D)``.
    method : {"normal", "warp3"}
        Proposal: a normal with the moments of the fitting half, or the
        standard normal after the warp-3 transformation.
    repetitions : int
        Independent proposal draws, each giving one estimate.
    use_neff : bool
        Use the median coda effective size of the iteration draws in
        ``s1``; otherwise their count.
    maxiter : int
        Iteration bound of the fixed-point scheme.
    seed : int
        Seed of the proposal draws.

    Returns
    -------
    BridgeResult
        Estimates, iteration counts, and the retained log densities.

    Raises
    ------
    ValueError
        If the draws, method, or settings are invalid, a parameter is constant
        across draws, or the log posterior is undefined at every proposal draw.
    RuntimeError
        If the scheme does not converge.

    Notes
    -----
    Cost is ``N1 + R N2`` log-posterior evaluations for ``"normal"`` and
    twice that for ``"warp3"``, plus O(N D^2 + D^3) for the proposal and
    O(N D p) for the effective sizes with autoregression order p.
    """
    if method not in ("normal", "warp3"):
        raise ValueError("bridge_sampler method must be 'normal' or 'warp3'")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("repetitions must be a positive integer")
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    draws = np.asarray(chains, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[2] < 1 or not np.isfinite(draws).all():
        raise ValueError(
            "bridge sampling requires finite unconstrained draws (chain, draw, D)"
        )
    count = draws.shape[1]
    split = round(count / 2)
    if split < 2 or count - split < 2:
        raise ValueError("bridge sampling requires at least four draws per chain")
    fit_draws = draws[:, :split].reshape(-1, draws.shape[2])
    iteration_chains = draws[:, split:]
    iteration = iteration_chains.reshape(-1, draws.shape[2])
    if np.any(np.ptp(fit_draws, axis=0) == 0.0):
        raise ValueError(
            "bridge sampling requires every unconstrained parameter to vary "
            "across the posterior draws"
        )
    n_post = iteration.shape[0]
    neff = (
        float(np.median(effective_size(list(iteration_chains))))
        if use_neff
        else float(n_post)
    )
    mean = fit_draws.mean(axis=0)
    covariance = near_pd(np.atleast_2d(np.cov(fit_draws, rowvar=False)))
    chol = np.linalg.cholesky(covariance)
    dim = mean.shape[0]
    generator = torch.Generator().manual_seed(seed)
    standard = [
        torch.randn(n_post, dim, generator=generator, dtype=torch.float64).numpy()
        for _ in range(repetitions)
    ]

    def evaluate(points: np.ndarray) -> np.ndarray:
        """Evaluate the log posterior, mapping undefined values to ``-inf``.

        Parameters
        ----------
        points : numpy.ndarray
            Unconstrained points ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            Log densities ``(N,)``.
        """
        values = np.asarray(log_posterior(points), dtype=np.float64).reshape(-1)
        if values.shape[0] != points.shape[0]:
            raise ValueError("log_posterior must return one value per row")
        return np.where(np.isnan(values), -np.inf, values)

    identity = np.eye(dim)
    zero = np.zeros(dim)
    if method == "normal":
        q12 = _mvn_logpdf(iteration, mean, chol)
        q11 = evaluate(iteration)
        proposals = [mean + z @ chol.T for z in standard]
        q22 = [_mvn_logpdf(p, mean, chol) for p in proposals]
        q21 = [evaluate(p) for p in proposals]
        log_det = 0.0
    else:
        q12 = _mvn_logpdf(np.linalg.solve(chol, (iteration - mean).T).T, zero, identity)
        q11 = np.logaddexp(evaluate(iteration), evaluate(2.0 * mean - iteration))
        q22 = [_mvn_logpdf(z, zero, identity) for z in standard]
        q21 = [
            np.logaddexp(evaluate(mean - z @ chol.T), evaluate(mean + z @ chol.T))
            for z in standard
        ]
        log_det = float(np.log(np.diag(chol)).sum())
    if any(np.all(~np.isfinite(q)) for q in q21):
        raise ValueError(
            "the log posterior is undefined or -inf at every proposal draw"
        )
    offset = -math.log(2.0) + log_det if method == "warp3" else 0.0
    l1 = offset + (q11 - q12)
    estimates: list[float] = []
    iterations: list[int] = []
    for q21_i, q22_i in zip(q21, q22, strict=True):
        logml, niter = _solve(l1, offset + (q21_i - q22_i), neff=neff, maxiter=maxiter)
        estimates.append(logml)
        iterations.append(niter)
    return BridgeResult(
        logml_repetitions=tuple(estimates),
        niter=tuple(iterations),
        method=method,
        neff=neff,
        q11=torch.from_numpy(q11),
        q12=torch.from_numpy(q12),
        q21=torch.from_numpy(np.stack(q21)),
        q22=torch.from_numpy(np.stack(q22)),
        log_det_warp=log_det,
        nonfinite_posterior=int((~np.isfinite(q11)).sum()),
        nonfinite_proposal=int(sum((~np.isfinite(q)).sum() for q in q21)),
    )


def bayes_factor_from_logml(
    logml1: float, logml2: float, *, log: bool = False
) -> float:
    """Return the Bayes factor of model 1 over model 2.

    Parameters
    ----------
    logml1, logml2 : float
        Log marginal likelihoods.
    log : bool
        Return the log Bayes factor.

    Returns
    -------
    float
        :math:`p(y \\mid M_1) / p(y \\mid M_2)` [@kass1995bayes], or its log.
    """
    difference = logml1 - logml2
    return difference if log else math.exp(difference)


def posterior_model_probabilities(
    logml: Mapping[str, float], *, prior_prob: Sequence[float] | None = None
) -> dict[str, float]:
    """Return posterior model probabilities, as ``bridgesampling::post_prob``.

    Parameters
    ----------
    logml : Mapping[str, float]
        Log marginal likelihood of each named model, at least two.
    prior_prob : Sequence[float] or None
        Prior model probabilities in the order of ``logml``; uniform when None.

    Returns
    -------
    dict[str, float]
        :math:`p(M_k \\mid y) \\propto p(y \\mid M_k) p(M_k)`, summing to one.

    Raises
    ------
    ValueError
        If fewer than two models are given, prior probabilities do not match
        the models or do not sum to one (R ``all.equal``), or an estimate is
        not finite.
    """
    if len(logml) < 2:
        raise ValueError("post_prob requires at least two models")
    values = np.asarray(list(logml.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("post_prob requires finite log marginal likelihoods")
    prior = (
        np.full(values.shape[0], 1.0 / values.shape[0])
        if prior_prob is None
        else np.asarray(prior_prob, dtype=np.float64)
    )
    if prior.shape != values.shape:
        raise ValueError(
            "the number of prior model probabilities must match the number of models"
        )
    if np.any(prior < 0) or abs(prior.sum() - 1.0) > _ALL_EQUAL:
        raise ValueError("prior model probabilities must be nonnegative and sum to one")
    log_weight = values + np.log(prior)
    posterior = np.exp(log_weight - np.logaddexp.reduce(log_weight))
    return {name: float(p) for name, p in zip(logml, posterior, strict=True)}


__all__ = [
    "BridgeMethod",
    "BridgeRelativeError",
    "BridgeRepetitionSpread",
    "BridgeResult",
    "ar_spectrum0",
    "bayes_factor_from_logml",
    "bridge_sampler",
    "effective_size",
    "near_pd",
    "posterior_model_probabilities",
]
