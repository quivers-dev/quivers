"""Runtime helper distributions for transpiled NumPyro source.

NumPyro does not ship several distribution families that QVR
supports: [`LogitNormal`][torch.distributions.LogitNormal],
[`HalfStudentT`][quivers.transpile.family_meta.HalfStudentT],
[`ContinuousBernoulli`][torch.distributions.ContinuousBernoulli],
[`FisherSnedecor`][torch.distributions.FisherSnedecor],
[`LogisticNormal`][torch.distributions.LogisticNormal],
[`OneHotCategorical`][torch.distributions.OneHotCategorical], and
[`OrderedProbit`][quivers.transpile.family_meta.OrderedProbit],
[`LogGamma`][quivers.continuous.log_space.LogGamma], and
[`LogitBeta`][quivers.continuous.log_space.LogitBeta]. The
[`NumPyroRenderer`][quivers.transpile.renderers.numpyro.NumPyroRenderer]
grafts the class definitions it needs from this module into the
emitted source, so the transpiled program is a self-contained,
runnable module. Each class is a real NumPyro
[`Distribution`][numpyro.distributions.distribution.Distribution]
whose ``log_prob`` matches the corresponding torch/QVR family up to
the shared additive constant, so the transpiled joint log-density
agrees with the QVR model.

The class bodies reference only names the emitted module already
binds: ``numpyro`` (with its ``numpyro.distributions`` and
``numpyro.distributions.transforms`` submodules), ``jnp``
(``jax.numpy``), ``jss`` (``jax.scipy.special``), and ``jrandom``
(``jax.random``). The renderer
emits the matching imports before the grafted classes; the imports
here keep this module directly importable and unit-testable.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jss
import numpyro
import numpyro.distributions
import numpyro.distributions.transforms


class LogitNormal(numpyro.distributions.TransformedDistribution):
    """``sigmoid(Normal(loc, scale))`` on the open unit interval.

    The sigmoid pushforward of a Normal; the transform's
    log-Jacobian supplies the change-of-variables term so the
    density is exact, not merely proportional.
    """

    arg_constraints = {
        "loc": numpyro.distributions.constraints.real,
        "scale": numpyro.distributions.constraints.positive,
    }
    support = numpyro.distributions.constraints.unit_interval
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
        base = numpyro.distributions.Normal(loc, scale)
        super().__init__(
            base,
            numpyro.distributions.transforms.SigmoidTransform(),
            validate_args=validate_args,
        )


class LogGamma(numpyro.distributions.Distribution):
    """``log(G)`` for ``G ~ Gamma(concentration, rate)`` on the real line.

    The density ``a (log r + y) - r exp(y) - gammaln(a)`` is exact for
    every finite ``y``. Draws stay in log space through the shape
    augmentation ``log G' + log(U) / a`` with ``G' ~ Gamma(a + 1)``
    and ``U ~ Uniform(0, 1]``, so a small shape never forms a Gamma
    draw that underflows; ``G'`` carries NumPyro's implicit
    reparameterization gradient.
    """

    arg_constraints = {
        "concentration": numpyro.distributions.constraints.positive,
        "rate": numpyro.distributions.constraints.positive,
    }
    support = numpyro.distributions.constraints.real
    reparametrized_params = ["concentration", "rate"]

    def __init__(self, concentration, rate, *, validate_args=None):
        self.concentration = jnp.asarray(concentration, dtype=jnp.result_type(float))
        self.rate = jnp.asarray(rate, dtype=jnp.result_type(float))
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.concentration), jnp.shape(self.rate)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        log_rate = jnp.log(self.rate)
        return (
            self.concentration * (log_rate + value)
            - jnp.exp(value + log_rate)
            - jss.gammaln(self.concentration)
        )

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        concentration = jnp.broadcast_to(self.concentration, shape)
        boost_key, uniform_key = jrandom.split(key)
        boosted = numpyro.distributions.Gamma(concentration + 1.0).sample(boost_key)
        uniform = jrandom.uniform(uniform_key, shape, dtype=concentration.dtype)
        return (
            jnp.log(boosted)
            + jnp.log1p(-uniform) / concentration
            - jnp.log(jnp.broadcast_to(self.rate, shape))
        )


class LogitBeta(numpyro.distributions.Distribution):
    """``logit(P)`` for ``P ~ Beta(concentration1, concentration0)``.

    The density ``a log_sigmoid(x) + b log_sigmoid(-x) - log B(a, b)``
    is exact for every finite ``x``. A draw is the difference of two
    unit-rate log-Gamma draws, each made in log space by the shape
    augmentation ``log G' + log(U) / a``.
    """

    arg_constraints = {
        "concentration1": numpyro.distributions.constraints.positive,
        "concentration0": numpyro.distributions.constraints.positive,
    }
    support = numpyro.distributions.constraints.real
    reparametrized_params = ["concentration1", "concentration0"]

    def __init__(self, concentration1, concentration0, *, validate_args=None):
        self.concentration1 = jnp.asarray(concentration1, dtype=jnp.result_type(float))
        self.concentration0 = jnp.asarray(concentration0, dtype=jnp.result_type(float))
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.concentration1), jnp.shape(self.concentration0)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        first = self.concentration1
        second = self.concentration0
        return (
            first * -jnp.logaddexp(0.0, -value)
            + second * -jnp.logaddexp(0.0, value)
            - jss.gammaln(first)
            - jss.gammaln(second)
            + jss.gammaln(first + second)
        )

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        first = jnp.broadcast_to(self.concentration1, shape)
        second = jnp.broadcast_to(self.concentration0, shape)
        keys = jrandom.split(key, 4)
        draws = []
        for shape_value, boost_key, uniform_key in (
            (first, keys[0], keys[1]),
            (second, keys[2], keys[3]),
        ):
            boosted = numpyro.distributions.Gamma(shape_value + 1.0).sample(boost_key)
            uniform = jrandom.uniform(uniform_key, shape, dtype=shape_value.dtype)
            draws.append(jnp.log(boosted) + jnp.log1p(-uniform) / shape_value)
        return draws[0] - draws[1]


class HalfStudentT(numpyro.distributions.FoldedDistribution):
    """StudentT folded about zero: ``|StudentT(df, 0, scale)|``.

    The folded density sums the two half-line contributions, which
    for the symmetric StudentT base is exactly twice the base
    density on the nonnegative reals.
    """

    arg_constraints = {
        "df": numpyro.distributions.constraints.positive,
        "scale": numpyro.distributions.constraints.positive,
    }
    support = numpyro.distributions.constraints.positive
    reparametrized_params = ["df", "scale"]

    def __init__(self, df, scale=1.0, *, validate_args=None):
        base = numpyro.distributions.StudentT(df, 0.0, scale)
        super().__init__(base, validate_args=validate_args)


class LogisticNormal(numpyro.distributions.TransformedDistribution):
    """Stick-breaking pushforward of an independent Normal onto the
    simplex.

    ``loc`` / ``scale`` are ``(..., K - 1)``-vectors; the
    stick-breaking transform maps them to a ``K``-simplex, matching
    torch's ``LogisticNormal``.
    """

    arg_constraints = {
        "loc": numpyro.distributions.constraints.real_vector,
        "scale": numpyro.distributions.constraints.positive,
    }
    support = numpyro.distributions.constraints.simplex
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc, scale, *, validate_args=None):
        base = numpyro.distributions.Normal(loc, scale).to_event(1)
        super().__init__(
            base,
            numpyro.distributions.transforms.StickBreakingTransform(),
            validate_args=validate_args,
        )


class ContinuousBernoulli(numpyro.distributions.Distribution):
    """Continuous Bernoulli on ``[0, 1]`` (Loaiza-Ganem & Cunningham,
    2019).

    The density is ``C(p) p^x (1 - p)^(1 - x)`` where ``C(p)`` is the
    parameter-dependent log-normaliser; dropping ``C(p)`` would break
    the joint density whenever ``p`` is itself random, so it is kept.
    A Taylor expansion covers the numerically unstable band around
    ``p = 1/2``.
    """

    arg_constraints = {
        "probs": numpyro.distributions.constraints.unit_interval,
    }
    support = numpyro.distributions.constraints.unit_interval
    has_rsample = True
    reparametrized_params = ["probs"]
    _lims = (0.499, 0.501)

    def __init__(self, probs, *, validate_args=None):
        self.probs = jnp.clip(
            jnp.asarray(probs, dtype=jnp.result_type(float)), 1e-6, 1.0 - 1e-6
        )
        super().__init__(batch_shape=jnp.shape(self.probs), validate_args=validate_args)

    def _outside_unstable_region(self):
        return (self.probs <= self._lims[0]) | (self.probs > self._lims[1])

    def _cut_probs(self):
        return jnp.where(
            self._outside_unstable_region(),
            self.probs,
            self._lims[0] * jnp.ones_like(self.probs),
        )

    def _cont_bern_log_norm(self):
        cut_probs = self._cut_probs()
        below = jnp.where(cut_probs <= 0.5, cut_probs, jnp.zeros_like(cut_probs))
        above = jnp.where(cut_probs >= 0.5, cut_probs, jnp.ones_like(cut_probs))
        log_norm = jnp.log(
            jnp.abs(jnp.log1p(-cut_probs) - jnp.log(cut_probs))
        ) - jnp.where(
            cut_probs <= 0.5,
            jnp.log1p(-2.0 * below),
            jnp.log(2.0 * above - 1.0),
        )
        x = jnp.square(self.probs - 0.5)
        taylor = jnp.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return jnp.where(self._outside_unstable_region(), log_norm, taylor)

    def log_prob(self, value):
        unnormalised = jss.xlogy(value, self.probs) + jss.xlog1py(
            1.0 - value, -self.probs
        )
        return unnormalised + self._cont_bern_log_norm()

    def _icdf(self, u):
        cut_probs = self._cut_probs()
        return jnp.where(
            self._outside_unstable_region(),
            (
                jnp.log1p(-cut_probs + u * (2.0 * cut_probs - 1.0))
                - jnp.log1p(-cut_probs)
            )
            / (jnp.log(cut_probs) - jnp.log1p(-cut_probs)),
            u,
        )

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        u = numpyro.distributions.Uniform(
            jnp.zeros(self.batch_shape), jnp.ones(self.batch_shape)
        ).sample(key, sample_shape)
        return jnp.broadcast_to(self._icdf(u), shape)


class FisherSnedecor(numpyro.distributions.Distribution):
    """F-distribution ``F(df1, df2)`` on the positive reals.

    Parameterised by the two degrees of freedom; the density uses
    the log-Beta normaliser. Sampling draws a single
    ``Beta(df1/2, df2/2)`` variate and maps it through the
    odds-ratio identity ``F = (df2 / df1) * B / (1 - B)``.
    """

    arg_constraints = {
        "df1": numpyro.distributions.constraints.positive,
        "df2": numpyro.distributions.constraints.positive,
    }
    support = numpyro.distributions.constraints.positive
    has_rsample = True
    reparametrized_params = ["df1", "df2"]

    def __init__(self, df1, df2, *, validate_args=None):
        self.df1 = jnp.asarray(df1, dtype=jnp.result_type(float))
        self.df2 = jnp.asarray(df2, dtype=jnp.result_type(float))
        batch_shape = jnp.broadcast_shapes(jnp.shape(self.df1), jnp.shape(self.df2))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        df1 = self.df1
        df2 = self.df2
        log_beta = (
            jss.gammaln(0.5 * df1)
            + jss.gammaln(0.5 * df2)
            - jss.gammaln(0.5 * (df1 + df2))
        )
        return (
            0.5 * df1 * jnp.log(df1)
            + 0.5 * df2 * jnp.log(df2)
            + (0.5 * df1 - 1.0) * jnp.log(value)
            - 0.5 * (df1 + df2) * jnp.log(df2 + df1 * value)
            - log_beta
        )

    def sample(self, key, sample_shape=()):
        beta = numpyro.distributions.Beta(0.5 * self.df1, 0.5 * self.df2).sample(
            key, sample_shape
        )
        return (self.df2 / self.df1) * beta / (1.0 - beta)


class OneHotCategorical(numpyro.distributions.Distribution):
    """Categorical whose draws are one-hot ``K``-vectors.

    ``log_prob`` of a one-hot ``value`` is ``sum(value * log probs)``,
    i.e. the log-probability of the selected category.
    """

    arg_constraints = {
        "probs": numpyro.distributions.constraints.simplex,
    }
    support = numpyro.distributions.constraints.simplex
    has_enumerate_support = True

    def __init__(self, probs, *, validate_args=None):
        self.probs = jnp.asarray(probs, dtype=jnp.result_type(float))
        batch_shape = jnp.shape(self.probs)[:-1]
        event_shape = jnp.shape(self.probs)[-1:]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        idx = numpyro.distributions.CategoricalProbs(self.probs).sample(
            key, sample_shape
        )
        return jnp.eye(self.event_shape[0])[idx]

    def log_prob(self, value):
        return jnp.sum(jss.xlogy(value, self.probs), axis=-1)


class OrderedProbit(numpyro.distributions.CategoricalProbs):
    """Ordered-probit categorical: a probit link over shared
    cutpoints.

    Category probabilities are successive differences of the standard
    normal CDF evaluated at ``cutpoints - eta``, giving
    ``len(cutpoints) + 1`` ordered outcomes.
    """

    arg_constraints = {
        "eta": numpyro.distributions.constraints.real,
        "cutpoints": numpyro.distributions.constraints.ordered_vector,
    }

    def __init__(self, eta, cutpoints, *, validate_args=None):
        self.eta = jnp.asarray(eta, dtype=jnp.result_type(float))
        self.cutpoints = jnp.asarray(cutpoints, dtype=jnp.result_type(float))
        cdf = jss.ndtr(self.cutpoints - self.eta[..., None])
        lead = jnp.shape(cdf)[:-1]
        cumulative = jnp.concatenate(
            [jnp.zeros(lead + (1,)), cdf, jnp.ones(lead + (1,))], axis=-1
        )
        probs = jnp.diff(cumulative, axis=-1)
        super().__init__(probs=probs, validate_args=validate_args)


__all__ = [
    "ContinuousBernoulli",
    "FisherSnedecor",
    "HalfStudentT",
    "LogGamma",
    "LogisticNormal",
    "LogitBeta",
    "LogitNormal",
    "OneHotCategorical",
    "OrderedProbit",
]
