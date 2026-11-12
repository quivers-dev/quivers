"""Runtime distributions included in emitted PyMC programs.

`ContinuousBernoulli` defines the normalized continuous-Bernoulli log density
through `pymc.CustomDist`. `LKJCholesky` supplies a density and sampler for
correlation Cholesky factors without the standard-deviation prior carried by
`pymc.LKJCholeskyCov`.

The renderer parses these definitions with its Python grammar and copies their
function-definition subtrees into emitted source.
"""

from __future__ import annotations

import numpy as np
import pymc
import pytensor.tensor as pt


def _continuous_bernoulli_logp(value, lam):
    """Log-density of the continuous-Bernoulli distribution with
    probability parameter ``lam`` on the unit interval.

    The density is ``C(lam) * lam**x * (1 - lam)**(1 - x)`` with
    normaliser ``C(lam) = 2 * arctanh(1 - 2 * lam) / (1 - 2 * lam)``,
    which has a removable singularity at ``lam == 0.5`` where the
    constant equals ``2``."""
    z = 1.0 - 2.0 * lam
    log_norm = pt.switch(
        pt.lt(pt.abs(z), 1e-6),
        pt.log(2.0),
        pt.log(2.0 * pt.arctanh(z) / z),
    )
    return value * pt.log(lam) + (1.0 - value) * pt.log(1.0 - lam) + log_norm


def _continuous_bernoulli_random(lam, rng=None, size=None):
    """Inverse-CDF sampler for the continuous-Bernoulli distribution.

    With ``a = log(lam / (1 - lam))`` the CDF inverts to
    ``x = log(1 + (2 * lam - 1) * u / (1 - lam)) / a`` for a uniform
    draw ``u``; at ``lam == 0.5`` the distribution is uniform on the
    unit interval, so ``x == u``."""
    lam = np.asarray(lam, dtype="float64")
    draw_size = size if size is not None else lam.shape
    u = rng.uniform(size=draw_size)
    near_half = np.abs(2.0 * lam - 1.0) < 1e-6
    safe_lam = np.where(near_half, 0.25, lam)
    a = np.log(safe_lam / (1.0 - safe_lam))
    shifted = np.log(1.0 + (2.0 * safe_lam - 1.0) * u / (1.0 - safe_lam)) / a
    return np.where(near_half, u, shifted)


def _bernoulli_logit_logp(value, logit_p):
    """Log mass of a Bernoulli outcome ``value`` given by its log-odds.

    ``value * l - softplus(l)``: PyTensor's ``softplus`` is the stable
    ``log(1 + exp(l))``, so the mass keeps its digits where the
    probability is within an ulp of zero or one, as reading the log of
    the probability's complement would not."""
    return value * logit_p - pt.softplus(logit_p)


def _bernoulli_logit_random(logit_p, rng=None, size=None):
    """Draw a Bernoulli outcome at log-odds ``logit_p``."""
    logit_p = np.asarray(logit_p, dtype="float64")
    draw_size = size if size is not None else logit_p.shape
    u = rng.uniform(size=draw_size)
    return (u < 1.0 / (1.0 + np.exp(-logit_p))).astype("int64")


def BernoulliLogit(name, logit_p, **kwargs):
    """Bernoulli random variable given by its log-odds, for PyMC.

    Wraps [`pymc.CustomDist`][pymc.CustomDist] with the mass written
    from the log-odds, under the call shape of
    ``pymc.Bernoulli(name, logit_p=...)``."""
    return pymc.CustomDist(
        name,
        logit_p,
        logp=_bernoulli_logit_logp,
        random=_bernoulli_logit_random,
        dtype="int64",
        **kwargs,
    )


def ContinuousBernoulli(name, probs, **kwargs):
    """Continuous-Bernoulli random variable for PyMC.

    Wraps [`pymc.CustomDist`][pymc.CustomDist] with the exact
    continuous-Bernoulli log-density, so the family is available under
    the same call shape as a built-in ``pymc.<Family>(name, ...)``
    constructor."""
    return pymc.CustomDist(
        name,
        probs,
        logp=_continuous_bernoulli_logp,
        random=_continuous_bernoulli_random,
        dtype="float64",
        **kwargs,
    )


def _lkj_cholesky_log_normalizer(n, eta):
    """Log normalising constant of the LKJ density over correlation
    Cholesky factors of an ``n`` by ``n`` correlation matrix.

    With ``d = n - 1`` and ``alpha = eta + d / 2`` the constant is
    ``d * log(pi) / 2 + mvlgamma(alpha - 1/2, d) - d * lgamma(alpha)``,
    where the multivariate log-gamma is expanded as
    ``d (d - 1) / 4 * log(pi) + sum_j lgamma(alpha - 1/2 - j/2)`` over
    ``j`` in ``0 .. d - 1``."""
    dm1 = n - 1
    alpha = eta + 0.5 * dm1
    shifted = pt.shape_padright(alpha) - 0.5 - 0.5 * pt.arange(dm1, dtype="float64")
    numerator = 0.25 * dm1 * (dm1 - 1) * np.log(np.pi) + pt.sum(
        pt.gammaln(shifted), axis=-1
    )
    return 0.5 * dm1 * np.log(np.pi) + numerator - dm1 * pt.gammaln(alpha)


def _lkj_cholesky_logp(value, n, eta):
    """Log-density of the LKJ distribution over the lower-triangular
    Cholesky factor ``value`` of an ``n`` by ``n`` correlation matrix
    with concentration ``eta``.

    The correlation matrix ``M = value @ value.T`` has LKJ density
    proportional to ``det(M) ** (eta - 1)``, and the Jacobian of the
    factorisation contributes ``prod_i value[i, i] ** (n - i)``, so the
    density of the factor is proportional to
    ``prod_i value[i, i] ** (2 * eta - 2 + n - i)`` for one-based
    ``i``."""
    diag = pt.as_tensor(pt.diagonal(value, axis1=-2, axis2=-1))[..., 1:]
    order = 2.0 * (eta - 1.0) + n - pt.arange(2, n + 1, dtype="float64")
    unnormalized = pt.sum(order * pt.log(diag), axis=-1)
    return unnormalized - _lkj_cholesky_log_normalizer(n, eta)


def _lkj_cholesky_draw(n, eta, rng, size=None):
    """Draw the lower-triangular Cholesky factor of a correlation
    matrix distributed as LKJ with concentration ``eta`` on an ``n`` by
    ``n`` correlation matrix, by the C-vine method of Lewandowski,
    Kurowicka, and Joe.

    The canonical partial correlations are mutually independent: the
    partial correlation between variables ``i`` and ``j`` given
    ``0 .. j - 1`` (zero-based, ``j < i``) is ``2 * B - 1`` for ``B``
    drawn from ``Beta(b_j, b_j)`` with ``b_j = eta + (n - 2 - j) / 2``.
    The partial-correlation recursion
    ``L[i, j] = z[i, j] * prod_{k < j} sqrt(1 - z[i, k] ** 2)`` with
    ``L[i, i] = prod_{k < i} sqrt(1 - z[i, k] ** 2)`` maps those
    partial correlations straight to the Cholesky factor ``L`` of the
    correlation matrix ``L @ L.T``, so ``L`` follows the same LKJ law
    over factors that [`_lkj_cholesky_logp`][quivers.transpile.runtime_pymc._lkj_cholesky_logp]
    scores. Its off-diagonal correlation marginals match the LKJ law
    and stay exchangeable across off-diagonals.

    ``rng`` is the generator PyMC threads through the random callback
    and ``size`` is the requested batch shape. The result has shape
    ``batch + (n, n)`` where ``batch`` is ``size`` when given, else the
    shape of ``eta``."""
    eta = np.asarray(eta, dtype="float64")
    if size is None:
        batch = eta.shape
    else:
        batch = tuple(int(s) for s in np.atleast_1d(size))
    eta_b = np.broadcast_to(eta, batch)
    factor = np.zeros((*batch, n, n), dtype="float64")
    factor[..., 0, 0] = 1.0
    for i in range(1, n):
        acc = np.ones(batch, dtype="float64")
        for j in range(i):
            beta = eta_b + 0.5 * (n - 2 - j)
            partial = 2.0 * rng.beta(beta, beta) - 1.0
            factor[..., i, j] = partial * acc
            acc = acc * np.sqrt(1.0 - partial * partial)
        factor[..., i, i] = acc
    return factor


def LKJCholesky(name, n, eta, **kwargs):
    """LKJ random variable over correlation Cholesky factors for PyMC.

    The value is the ``n`` by ``n`` lower-triangular Cholesky factor of
    a correlation matrix, carried in unconstrained coordinates by
    [`CholeskyCorrTransform`][pymc.distributions.transforms.CholeskyCorrTransform].
    Draws come from the C-vine sampler
    [`_lkj_cholesky_draw`][quivers.transpile.runtime_pymc._lkj_cholesky_draw],
    which returns the Cholesky factor directly, and the log-density is
    the normalised density of the factor, so both the forward samples
    and the total log-density match the QVR model's
    `torch.distributions.LKJCholesky` term.

    A length-``n`` zero vector rides along as a second distribution
    parameter: it carries the correlation dimension ``n`` into PyMC's
    core-shape inference for the multivariate random callback, which
    cannot read ``n`` off the scalar ``eta`` alone. The sampler and the
    log-density both ignore its value."""
    n = int(n)
    dim_marker = np.zeros(n, dtype="float64")

    def random(eta, dim_marker, rng=None, size=None):
        return _lkj_cholesky_draw(n, eta, rng, size)

    def logp(value, eta, dim_marker):
        return _lkj_cholesky_logp(value, n, eta)

    return pymc.CustomDist(
        name,
        eta,
        dim_marker,
        random=random,
        logp=logp,
        signature="(),(n)->(n,n)",
        default_transform=pymc.distributions.transforms.CholeskyCorrTransform(
            n=n, upper=False
        ),
        **kwargs,
    )
