"""Parameterized distribution families as continuous morphisms.

Each family is a ContinuousMorphism whose codomain is a continuous
space and whose conditional distribution p(y | x) belongs to a
specific parametric family. The parameters are learnable functions
of x:

- For discrete domains (FinSet): parameters are looked up from a table.
- For continuous domains (ContinuousSpace): parameters are produced
  by a small neural network.

This module wraps every reparameterizable distribution in
torch.distributions as a conditional morphism, plus custom
families (TruncatedNormal, MultivariateNormal, etc.).

Architecture
------------
Most per-dimension-independent distributions are built on a shared
generic base ``_IndependentConditional`` that handles the parameter
source, transform, and torch.distributions plumbing. The
``_make_family`` class factory generates named classes from a
specification. Distributions that need special handling
(MultivariateNormal, Dirichlet, TruncatedNormal, etc.) are
implemented as standalone classes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import constraints as _constraints

from quivers.continuous._ordered import OrderedLogistic
from quivers.core.objects import SetObject
from quivers.continuous._zip_hurdle import (
    HurdlePoisson,
    MixtureNormal,
    ZeroInflatedPoisson,
    ZeroOneInflatedBeta,
)
from quivers.continuous.bijectors import Bijector
from quivers.continuous.family_spec import (
    FamilySpec,
    ParamSpec,
    register_family as _register_family,
)
from quivers.continuous.param_transforms import resolve_transform
from quivers.continuous.spaces import (
    CholeskyFactor,
    ContinuousSpace,
    Euclidean,
)
from quivers.continuous.morphisms import (
    AnySpace,
    ContinuousMorphism,
    _event_size,
    dimension_probe,
)
from quivers.continuous.param_source import ParamSource, _make_source
from quivers.core._util import EPS


def _lower_bounded(bound: float) -> Callable:
    """Create a transform that ensures output > bound."""

    def transform(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + bound

    return transform


def _resolve_support(
    dist_class: type, override: _constraints.Constraint | None
) -> _constraints.Constraint:
    """Look up the family's output support from the torch class, or
    use the user-supplied override. Falls back to ``real`` when
    neither is available."""
    if override is not None:
        return override
    support = getattr(dist_class, "support", None)
    if isinstance(support, _constraints.Constraint):
        return support
    return _constraints.real


def _resolve_discrete(dist_class: type, override: bool | None) -> bool:
    """Discrete iff the torch distribution lacks reparameterized
    sampling (``has_rsample == False``)."""
    if override is not None:
        return override
    return not bool(getattr(dist_class, "has_rsample", True))


# ============================================================================
# generic per-dimension-independent conditional
# ============================================================================


class _IndependentConditional(ContinuousMorphism):
    """Generic conditional morphism for per-dimension-independent distributions.

    Wraps any torch.distributions.Distribution class that operates
    independently on each dimension of the codomain. Parameters are
    produced by a learnable source (lookup table or neural net) and
    transformed to satisfy distribution constraints.

    Parameters
    ----------
    domain : AnySpace
        Source space.
    codomain : ContinuousSpace
        Target space.
    dist_class : type
        A torch.distributions.Distribution subclass.
    param_specs : list of (str, Callable)
        Each entry is (parameter_name, transform_function). One raw
        scalar per codomain dimension is allocated for each spec.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        dist_class: type,
        param_specs: list[tuple[str, Callable]],
        hidden_dim: int | Sequence[int] | None = None,
        discrete: bool = False,
        param_source=None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self._dist_class = dist_class
        self._param_specs = param_specs
        self._discrete = discrete

        # total raw parameters: one scalar per spec per codomain dim
        total_raw = len(param_specs) * d
        self.param_source = _make_source(
            domain,
            total_raw,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    def _get_dist(self, x: torch.Tensor) -> D.Distribution:
        """Build the torch distribution for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.distributions.Distribution
            Parameterized distribution with batch_shape (batch, d).
        """
        raw = self.param_source(x)  # (batch, n_specs * d)
        params = {}
        offset = 0

        for name, transform in self._param_specs:
            chunk = raw[..., offset : offset + self._d]
            params[name] = transform(chunk)
            offset += self._d

        return self._dist_class(**params)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        y_typed = y.float() if self._discrete else y
        return dist.log_prob(y_typed).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        if self._discrete:
            return dist.sample(sample_shape).long()
        if getattr(dist, "has_rsample", True):
            return dist.rsample(sample_shape)
        # Continuous but non-reparameterizable (e.g. VonMises). Fall
        # back to ``.sample()`` — gradients won't flow through the
        # sample but the distribution still scores log-probabilities.
        return dist.sample(sample_shape)


# ============================================================================
# class factory
# ============================================================================


def _make_family(
    name: str,
    dist_class: type,
    param_specs: list[tuple[str, str | Bijector]],
    doc: str,
    *,
    dsl_name: str | None = None,
    support: _constraints.Constraint | None = None,
    discrete: bool | None = None,
) -> type:
    """Generate a named conditional distribution class.

    Each call also installs a `FamilySpec` into the unified
    catalog so the inline (fixed / mixed) call paths and the DSL
    compiler discover the family automatically.

    Parameters
    ----------
    name : str
        Class name (e.g. ``"ConditionalCauchy"``).
    dist_class : type
        The `torch.distributions` class.
    param_specs : list of (str, str | Bijector)
        Ordered ``(parameter_name, transform)`` pairs. Each
        transform is either a string key registered in
        [`TRANSFORM_TO_BIJECTOR`][quivers.continuous.param_transforms.TRANSFORM_TO_BIJECTOR]
        or a `Bijector` instance. String and bijector entries may
        be mixed freely within one family.
    doc : str
        Class docstring.
    dsl_name : str, optional
        DSL key. Defaults to ``name.removeprefix("Conditional")``.
    support : Constraint, optional
        Output support. Auto-detected from ``dist_class.support`` if
        absent.
    discrete : bool, optional
        Whether the family is discrete. Auto-detected from
        ``not dist_class.has_rsample`` if absent.

    Returns
    -------
    type
        A new `ContinuousMorphism` subclass.
    """
    resolved_specs = [
        (pname, resolve_transform(tname).forward) for pname, tname in param_specs
    ]
    is_discrete = _resolve_discrete(dist_class, discrete)
    out_support = _resolve_support(dist_class, support)

    class _Cls(_IndependentConditional):
        __doc__ = doc

        def __init__(
            self,
            domain: AnySpace,
            codomain: ContinuousSpace,
            hidden_dim: int | Sequence[int] | None = None,
            param_source: ParamSource | None = None,
            param_source_option: str | None = None,
        ) -> None:
            super().__init__(
                domain,
                codomain,
                dist_class,
                resolved_specs,
                hidden_dim,
                discrete=is_discrete,
                param_source=param_source,
                param_source_option=param_source_option,
            )

    _Cls.__name__ = name
    _Cls.__qualname__ = name

    # Transpile-time introspection. The conditional class exposes the
    # same class-level `arg_constraints` and `support` that the
    # underlying torch distribution publishes; the transpile lower
    # pipeline reads these without instantiating a sentinel and the
    # variational guide reads `instance.support` from the same
    # attribute (a class attribute is visible on both the class and
    # its instances).
    if isinstance(dist_class.arg_constraints, dict):
        _Cls.arg_constraints = dict(dist_class.arg_constraints)  # type: ignore[attr-defined]
    _Cls.support = out_support  # type: ignore[assignment]

    if dsl_name is None:
        dsl_name = name.removeprefix("Conditional")
    _register_family(
        FamilySpec(
            name=dsl_name,
            dist_class=dist_class,
            params=tuple(
                ParamSpec(name=pname, transform=tname) for pname, tname in param_specs
            ),
            support=out_support,
            discrete=is_discrete,
            output_kind="independent",
            docstring=doc,
            conditional_class_override=_Cls,
        )
    )
    return _Cls


# ============================================================================
# hand-written families (backward compatible, tested)
# ============================================================================


class ConditionalNormal(ContinuousMorphism):
    """Conditional normal (Gaussian) distribution.

    For each input x, produces an independent normal distribution
    on each dimension of the codomain:

        y_i ~ Normal(mu_i(x), sigma_i(x))

    Parameters are learnable: mu and log(sigma) are functions of x,
    implemented as lookup tables (discrete domain) or neural networks
    (continuous domain).

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Euclidean
        Target space.
    hidden_dim : int
        Hidden layer width for neural parameter source.

    Examples
    --------
    >>> from quivers import FinSet
    >>> from quivers.continuous.spaces import Euclidean
    >>> A = FinSet(name="context", cardinality=5)
    >>> Y = Euclidean(name="response", dim=3)
    >>> f = ConditionalNormal(A, Y)
    >>> x = torch.tensor([0, 1, 2])
    >>> samples = f.rsample(x)  # shape (3, 3)
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source=None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        # param_dim = d (mu) + d (log_sigma)
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d

    def _get_params(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mu and sigma from the parameter source.

        Returns
        -------
        mu : torch.Tensor
            Mean. Shape (batch, d).
        sigma : torch.Tensor
            Standard deviation (positive). Shape (batch, d).
        """
        raw = self.param_source(x)  # (batch, 2*d)
        mu = raw[..., : self._d]
        log_sigma = raw[..., self._d :]
        sigma = log_sigma.exp().clamp(min=EPS)
        return mu, sigma

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        # independent normal log-density, summed over dimensions
        log_p = (
            -0.5 * ((y - mu) / sigma) ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)
        )

        return log_p.sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        # reparameterization trick
        eps = torch.randn(
            *sample_shape,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )

        return mu + sigma * eps

    def base_dimension(self, x: torch.Tensor) -> int:
        """One standard-normal coordinate per independent output axis."""
        mu, _ = self._get_params(dimension_probe(x))
        return _event_size(mu.shape)

    def push_base(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """The location-scale map ``mu(x) + sigma(x) * base``.

        The same map `rsample` applies, with the caller's coordinates
        in place of a fresh draw.
        """
        mu, sigma = self._get_params(x)
        return mu + sigma * base.reshape(mu.shape)


class ConditionalLogitNormal(ContinuousMorphism):
    """Conditional logit-normal distribution on (0, 1)^d.

    If z ~ Normal(mu(x), sigma(x)), then y = sigmoid(z) ~ LogitNormal.
    Useful for modeling probabilities and bounded quantities.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Euclidean
        Target space (should have bounds [0, 1]).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.unit_interval

    def _get_params(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_source(x)
        mu = raw[..., : self._d]
        log_sigma = raw[..., self._d :]
        sigma = log_sigma.exp().clamp(min=EPS)
        return mu, sigma

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        # logit transform
        y_clamped = y.clamp(min=EPS, max=1.0 - EPS)
        logit_y = torch.log(y_clamped / (1.0 - y_clamped))

        # normal log-density at logit(y)
        log_normal = (
            -0.5 * ((logit_y - mu) / sigma) ** 2
            - sigma.log()
            - 0.5 * math.log(2 * math.pi)
        )

        # jacobian correction: -log(y * (1-y))
        log_jacobian = -torch.log(y_clamped * (1.0 - y_clamped))

        return (log_normal + log_jacobian).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        eps = torch.randn(
            *sample_shape,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )
        z = mu + sigma * eps

        return torch.sigmoid(z)


class ConditionalBeta(ContinuousMorphism):
    """Conditional beta distribution on (0, 1)^d.

    For each input x, produces an independent Beta(alpha_i(x), beta_i(x))
    on each dimension of the codomain.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Euclidean
        Target space (should have bounds [0, 1]).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.unit_interval

    def _get_params(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_source(x)
        log_alpha = raw[..., : self._d]
        log_beta = raw[..., self._d :]
        alpha = F.softplus(log_alpha) + 0.1
        beta = F.softplus(log_beta) + 0.1
        return alpha, beta

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha, beta = self._get_params(x)
        y_clamped = y.clamp(min=EPS, max=1.0 - EPS)

        dist = D.Beta(alpha, beta)
        return dist.log_prob(y_clamped).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        alpha, beta = self._get_params(x)
        dist = D.Beta(alpha, beta)

        return dist.rsample(sample_shape)


class ConditionalTruncatedNormal(ContinuousMorphism):
    """Conditional truncated normal on [low, high]^d.

    A normal distribution restricted to a bounded interval. Uses
    rejection-free sampling via the inverse CDF (Phi-based) method.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Euclidean
        Target space (must be bounded).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: Euclidean,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        if codomain.low is None or codomain.high is None:
            raise ValueError("ConditionalTruncatedNormal requires a bounded codomain")

        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d
        self._low = codomain.low
        self._high = codomain.high

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.interval(float(self._low), float(self._high))

    def _get_params(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_source(x)
        # initialize mu near center of interval
        mu = raw[..., : self._d] + (self._low + self._high) / 2.0
        log_sigma = raw[..., self._d :]
        sigma = log_sigma.exp().clamp(min=EPS)
        return mu, sigma

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        log_phi = (
            -0.5 * ((y - mu) / sigma) ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)
        )

        normal = D.Normal(0, 1)
        log_Z = torch.log(
            (
                normal.cdf((self._high - mu) / sigma)
                - normal.cdf((self._low - mu) / sigma)
            ).clamp(min=EPS)
        )

        return (log_phi - log_Z).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        mu, sigma = self._get_params(x)

        normal = D.Normal(0, 1)
        alpha = normal.cdf((self._low - mu) / sigma)
        beta = normal.cdf((self._high - mu) / sigma)

        u = torch.rand(
            *sample_shape,
            *mu.shape,
            device=mu.device,
            dtype=mu.dtype,
        )
        u_scaled = alpha + u * (beta - alpha)
        u_scaled = u_scaled.clamp(min=EPS, max=1.0 - EPS)

        return normal.icdf(u_scaled) * sigma + mu


class ConditionalDirichlet(ContinuousMorphism):
    """Conditional Dirichlet distribution on a probability simplex.

    For each input x, produces a Dirichlet(alpha(x)) distribution
    on the simplex.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : Simplex
        Target simplex.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(
            domain,
            d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.simplex

    def _get_concentration(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.param_source(x)
        return F.softplus(raw) + 0.1

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        concentration = self._get_concentration(x)
        dist = D.Dirichlet(concentration)
        return dist.log_prob(y.clamp(min=EPS))

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        concentration = self._get_concentration(x)
        dist = D.Dirichlet(concentration)
        return dist.rsample(sample_shape)


# ============================================================================
# factory-generated families (all remaining torch.distributions with rsample)
# ============================================================================

# -- loc-scale family --------------------------------------------------------

ConditionalCauchy = _make_family(
    "ConditionalCauchy",
    D.Cauchy,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional Cauchy(loc(x), scale(x)). Heavy-tailed, no finite moments.",
)
"""Conditional Cauchy(loc(x), scale(x)). Heavy-tailed, no finite moments."""

ConditionalLaplace = _make_family(
    "ConditionalLaplace",
    D.Laplace,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional Laplace(loc(x), scale(x)). Sharp peak, heavier tails than normal.",
)
"""Conditional Laplace(loc(x), scale(x)). Sharp peak, heavier tails than normal."""

ConditionalGumbel = _make_family(
    "ConditionalGumbel",
    D.Gumbel,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional Gumbel(loc(x), scale(x)). Extreme value distribution (type I).",
)
"""Conditional Gumbel(loc(x), scale(x)). Extreme value distribution (type I)."""

ConditionalLogNormal = _make_family(
    "ConditionalLogNormal",
    D.LogNormal,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional LogNormal(loc(x), scale(x)). Positive-valued, right-skewed.",
)
"""Conditional LogNormal(loc(x), scale(x)). Positive-valued, right-skewed."""

ConditionalStudentT = _make_family(
    "ConditionalStudentT",
    D.StudentT,
    [("df", "softplus_shifted"), ("loc", "id"), ("scale", "softplus")],
    "Conditional StudentT(df(x), loc(x), scale(x)). Heavy-tailed with learnable df.",
)
"""Conditional StudentT(df(x), loc(x), scale(x)). Heavy-tailed with learnable df."""

# -- positive-valued distributions -------------------------------------------

ConditionalExponential = _make_family(
    "ConditionalExponential",
    D.Exponential,
    [("rate", "softplus")],
    "Conditional Exponential(rate(x)). Memoryless, positive-valued.",
)
"""Conditional Exponential(rate(x)). Memoryless, positive-valued."""

ConditionalGamma = _make_family(
    "ConditionalGamma",
    D.Gamma,
    [("concentration", "softplus_shifted"), ("rate", "softplus")],
    "Conditional Gamma(concentration(x), rate(x)). Positive-valued, flexible shape.",
)
"""Conditional Gamma(concentration(x), rate(x)). Positive-valued, flexible shape."""

ConditionalChi2 = _make_family(
    "ConditionalChi2",
    D.Chi2,
    [("df", "softplus_shifted")],
    "Conditional Chi2(df(x)). Positive-valued, sum of squared normals.",
)
"""Conditional Chi2(df(x)). Positive-valued, sum of squared normals."""

ConditionalHalfCauchy = _make_family(
    "ConditionalHalfCauchy",
    D.HalfCauchy,
    [("scale", "softplus")],
    "Conditional HalfCauchy(scale(x)). Heavy-tailed positive prior.",
)
"""Conditional HalfCauchy(scale(x)). Heavy-tailed positive prior."""

ConditionalHalfNormal = _make_family(
    "ConditionalHalfNormal",
    D.HalfNormal,
    [("scale", "softplus")],
    "Conditional HalfNormal(scale(x)). Folded normal, positive-valued.",
)
"""Conditional HalfNormal(scale(x)). Folded normal, positive-valued."""

ConditionalInverseGamma = _make_family(
    "ConditionalInverseGamma",
    D.InverseGamma,
    [("concentration", "softplus_shifted"), ("rate", "softplus")],
    "Conditional InverseGamma(concentration(x), rate(x)). Conjugate prior for normal variance.",
)
"""Conditional InverseGamma(concentration(x), rate(x)). Conjugate prior for normal variance."""

ConditionalWeibull = _make_family(
    "ConditionalWeibull",
    D.Weibull,
    [("scale", "softplus"), ("concentration", "softplus")],
    "Conditional Weibull(scale(x), concentration(x)). Survival analysis, reliability.",
)
"""Conditional Weibull(scale(x), concentration(x)). Survival analysis, reliability."""

ConditionalPareto = _make_family(
    "ConditionalPareto",
    D.Pareto,
    [("scale", "softplus"), ("alpha", "softplus")],
    "Conditional Pareto(scale(x), alpha(x)). Power-law tail.",
)
"""Conditional Pareto(scale(x), alpha(x)). Power-law tail."""

# -- (0, 1)-valued distributions ---------------------------------------------

ConditionalKumaraswamy = _make_family(
    "ConditionalKumaraswamy",
    D.Kumaraswamy,
    [("concentration1", "softplus_shifted"), ("concentration0", "softplus_shifted")],
    "Conditional Kumaraswamy(a(x), b(x)). Beta-like on (0,1), closed-form CDF.",
)
"""Conditional Kumaraswamy(a(x), b(x)). Beta-like on (0,1), closed-form CDF."""

ConditionalContinuousBernoulli = _make_family(
    "ConditionalContinuousBernoulli",
    D.ContinuousBernoulli,
    [("logits", "id")],
    "Conditional ContinuousBernoulli(logits(x)). Continuous relaxation of Bernoulli.",
)
"""Conditional ContinuousBernoulli(logits(x)). Continuous relaxation of Bernoulli."""

# -- two-df distributions ----------------------------------------------------

ConditionalFisherSnedecor = _make_family(
    "ConditionalFisherSnedecor",
    D.FisherSnedecor,
    [("df1", "softplus_shifted"), ("df2", "softplus_shifted")],
    "Conditional FisherSnedecor(df1(x), df2(x)). F-distribution, ratio of chi-squared.",
)
"""Conditional FisherSnedecor(df1(x), df2(x)). F-distribution, ratio of chi-squared."""

# -- discrete count distributions --------------------------------------------

ConditionalPoisson = _make_family(
    "ConditionalPoisson",
    D.Poisson,
    [("rate", "softplus")],
    "Conditional Poisson(rate(x)). Discrete, non-negative integer counts.",
)
"""Conditional Poisson(rate(x)). Discrete, non-negative integer counts."""

ConditionalGeometric = _make_family(
    "ConditionalGeometric",
    D.Geometric,
    [("probs", "sigmoid")],
    "Conditional Geometric(probs(x)). Number of failures before first success.",
)
"""Conditional Geometric(probs(x)). Number of failures before first success."""

ConditionalNegativeBinomial = _make_family(
    "ConditionalNegativeBinomial",
    D.NegativeBinomial,
    [("total_count", "softplus_shifted"), ("probs", "sigmoid")],
    "Conditional NegativeBinomial(total_count(x), probs(x)). "
    "Discrete, over-dispersed count distribution.",
)
"""Conditional NegativeBinomial(total_count(x), probs(x)). Discrete, over-dispersed count distribution."""

# -- circular distributions --------------------------------------------------

# VonMises is continuous (angles on the real line) but torch flags it
# ``has_rsample=False`` because reparameterized gradients are not
# implemented. Override the discrete-detection so the conditional
# class uses ``.sample()`` without casting to long.
ConditionalVonMises = _make_family(
    "ConditionalVonMises",
    D.VonMises,
    [("loc", "id"), ("concentration", "softplus")],
    "Conditional VonMises(loc(x), concentration(x)). "
    "Circular analogue of the Normal distribution.",
    discrete=False,
)
"""Conditional VonMises(loc(x), concentration(x)). Circular analogue of the Normal distribution."""

# -- uniform distribution (special parameterization) -------------------------


class ConditionalUniform(ContinuousMorphism):
    """Conditional uniform distribution on a learnable interval.

    Parameterized as Uniform(loc - width/2, loc + width/2) where
    loc is unconstrained and width is positive. This ensures
    low < high is always satisfied.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        # param_dim = d (loc) + d (raw_width)
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d
        # The bounds are data-dependent, so we cannot pin a single
        # interval at construction time; advertise the codomain's
        # declared bounds when available, otherwise fall back to the
        # real line.
        low, high = getattr(codomain, "low", None), getattr(codomain, "high", None)
        if low is not None and high is not None:
            self._support_cache: _constraints.Constraint = _constraints.interval(
                float(low), float(high)
            )
        else:
            self._support_cache = _constraints.real

    @property
    def support(self) -> _constraints.Constraint:
        return self._support_cache

    def _get_dist(self, x: torch.Tensor) -> D.Uniform:
        raw = self.param_source(x)
        loc = raw[..., : self._d]
        width = F.softplus(raw[..., self._d :]) + EPS
        low = loc - width / 2.0
        high = loc + width / 2.0
        return D.Uniform(low, high)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


# ============================================================================
# multivariate distributions (special parameter structures)
# ============================================================================


class ConditionalMultivariateNormal(ContinuousMorphism):
    """Conditional multivariate normal with full covariance.

    Parameterized via Cholesky factor: the parameter source outputs
    loc (d values) and the lower-triangular entries of L (d*(d+1)/2
    values), where Sigma = L @ L^T.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space (d-dimensional).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        n_tril = d * (d + 1) // 2
        self.param_source = _make_source(
            domain,
            d + n_tril,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d
        self._n_tril = n_tril

    def _get_dist(self, x: torch.Tensor) -> D.MultivariateNormal:
        raw = self.param_source(x)
        loc = raw[..., : self._d]
        tril_raw = raw[..., self._d :]

        # build lower-triangular matrix
        batch_shape = tril_raw.shape[:-1]
        L = torch.zeros(
            *batch_shape,
            self._d,
            self._d,
            device=tril_raw.device,
            dtype=tril_raw.dtype,
        )

        # fill lower triangle
        idx = torch.tril_indices(self._d, self._d)
        L[..., idx[0], idx[1]] = tril_raw

        # ensure positive diagonal (for positive definiteness)
        diag_idx = torch.arange(self._d)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + EPS

        return D.MultivariateNormal(loc, scale_tril=L)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


class ConditionalLowRankMVN(ContinuousMorphism):
    """Conditional low-rank multivariate normal.

    Parameterized as loc + low-rank factor + diagonal:
        Sigma = W @ W^T + diag(d)

    This is more parameter-efficient than full MVN for high dimensions.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space (d-dimensional).
    rank : int
        Rank of the low-rank factor W.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        rank: int = 2,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self._rank = rank

        # loc (d) + factor (d * rank) + diag (d)
        total = d + d * rank + d
        self.param_source = _make_source(
            domain,
            total,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    def _get_dist(self, x: torch.Tensor) -> D.LowRankMultivariateNormal:
        raw = self.param_source(x)
        d = self._d
        r = self._rank

        loc = raw[..., :d]
        factor_raw = raw[..., d : d + d * r]
        diag_raw = raw[..., d + d * r :]

        cov_factor = factor_raw.reshape(*raw.shape[:-1], d, r)
        cov_diag = F.softplus(diag_raw) + EPS

        return D.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


# ============================================================================
# relaxed discrete distributions (continuous outputs)
# ============================================================================


class ConditionalRelaxedBernoulli(ContinuousMorphism):
    """Conditional relaxed Bernoulli (concrete) distribution.

    Outputs continuous values in (0, 1) that approximate Bernoulli
    samples. The temperature controls the relaxation: lower temperature
    = closer to discrete.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space (should be 1-d per Bernoulli component).
    temperature : float
        Relaxation temperature.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        temperature: float = 0.5,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(
            domain,
            d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d
        self._temperature = temperature

    def _get_dist(self, x: torch.Tensor) -> D.RelaxedBernoulli:
        logits = self.param_source(x)
        temp = torch.tensor(
            self._temperature,
            device=logits.device,
            dtype=logits.dtype,
        )
        return D.RelaxedBernoulli(temperature=temp, logits=logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y.clamp(min=EPS, max=1.0 - EPS)).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


class ConditionalRelaxedOneHotCategorical(ContinuousMorphism):
    """Conditional relaxed one-hot categorical (Gumbel-Softmax).

    Outputs continuous vectors on the simplex that approximate
    one-hot categorical samples.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space (simplex or d-dimensional).
    temperature : float
        Relaxation temperature.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        temperature: float = 0.5,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self.param_source = _make_source(
            domain,
            d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d
        self._temperature = temperature

    def _get_dist(self, x: torch.Tensor) -> D.RelaxedOneHotCategorical:
        logits = self.param_source(x)
        temp = torch.tensor(
            self._temperature,
            device=logits.device,
            dtype=logits.dtype,
        )
        return D.RelaxedOneHotCategorical(temperature=temp, logits=logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y.clamp(min=EPS))

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


# ============================================================================
# wishart (matrix-valued)
# ============================================================================


class ConditionalWishart(ContinuousMorphism):
    """Conditional Wishart distribution over positive-definite matrices.

    Produces random d x d positive-definite matrices. Parameterized
    by degrees of freedom df(x) and a scale matrix V(x).

    The codomain dimension is interpreted as d, and outputs are
    d x d matrices flattened to d^2.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space. dim is the matrix size d (output is d x d).
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        n_tril = d * (d + 1) // 2
        # df (1) + lower-triangular scale (n_tril)
        self.param_source = _make_source(
            domain,
            1 + n_tril,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d
        self._n_tril = n_tril

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.positive_definite

    def _get_dist(self, x: torch.Tensor) -> D.Wishart:
        raw = self.param_source(x)
        d = self._d

        # df must be > d - 1
        df = F.softplus(raw[..., 0]) + d

        tril_raw = raw[..., 1:]
        batch_shape = tril_raw.shape[:-1]
        L = torch.zeros(
            *batch_shape,
            d,
            d,
            device=tril_raw.device,
            dtype=tril_raw.dtype,
        )
        idx = torch.tril_indices(d, d)
        L[..., idx[0], idx[1]] = tril_raw
        diag_idx = torch.arange(d)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + EPS

        return D.Wishart(df=df, scale_tril=L)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        d = self._d
        y_mat = y.reshape(*y.shape[:-1], d, d)
        return dist.log_prob(y_mat)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        mat = dist.rsample(sample_shape)
        # flatten d x d to d^2
        return mat.reshape(*mat.shape[:-2], self._d * self._d)


# ============================================================================
# matrix-valued: MatrixNormal (Kronecker covariance)
# ============================================================================


class ConditionalMatrixNormal(ContinuousMorphism):
    """Conditional Matrix-Normal :math:`\\mathcal{MN}(M, U, V)`.

    The matrix-Normal distribution on :math:`\\mathbb{R}^{n \\times p}`
    factorises with a Kronecker-product covariance: if
    :math:`X \\sim \\mathcal{MN}(M, U, V)` then
    :math:`\\mathrm{vec}(X) \\sim \\mathcal{N}(\\mathrm{vec}(M), V \\otimes U)`
    with :math:`U \\in \\mathbb{R}^{n \\times n}` the row covariance
    and :math:`V \\in \\mathbb{R}^{p \\times p}` the column covariance.

    Categorically, the Kronecker structure is the tensor product
    of two Gaussians; it is strictly more constrained than the
    flat :math:`np`-dim MVN that the same parameter tensor could
    carry, so the surface keeps the two families distinct (no
    auto-substitution).  Use this when the prior should express
    independent row and column correlation structure.

    The codomain's product factorisation supplies the row and
    column dimensions.  The grammar surface
    ``~ MatrixNormal(loc, row_scale, col_scale) over (dom, cod)``
    binds the first axis listed in ``over`` to the row covariance
    and the second to the column covariance.

    Parameters
    ----------
    domain : AnySpace
        Source space.
    codomain : ContinuousSpace
        Target space whose factorisation supplies ``(n, p)``.  Must
        carry a product structure of two factors; the first is the
        row axis, the second the column.
    rows : int
        Row dimension :math:`n`.
    cols : int
        Column dimension :math:`p`.
    hidden_dim : int
        Hidden layer width for the parameter network.
    """

    event_rank: int = 2

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        rows: int,
        cols: int,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        self._rows = int(rows)
        self._cols = int(cols)
        n_loc = self._rows * self._cols
        n_row_tril = self._rows * (self._rows + 1) // 2
        n_col_tril = self._cols * (self._cols + 1) // 2
        self._n_loc = n_loc
        self._n_row_tril = n_row_tril
        self._n_col_tril = n_col_tril
        self.param_source = _make_source(
            domain,
            n_loc + n_row_tril + n_col_tril,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    def _build_tril(self, raw: torch.Tensor, d: int, n_tril: int) -> torch.Tensor:
        batch_shape = raw.shape[:-1]
        L = torch.zeros(*batch_shape, d, d, device=raw.device, dtype=raw.dtype)
        idx = torch.tril_indices(d, d)
        L[..., idx[0], idx[1]] = raw[..., :n_tril]
        diag_idx = torch.arange(d)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + EPS
        return L

    def _get_dist(self, x: torch.Tensor) -> D.MultivariateNormal:
        raw = self.param_source(x)
        n, p = self._rows, self._cols
        offset = 0
        loc_flat = raw[..., offset : offset + self._n_loc]
        offset += self._n_loc
        row_raw = raw[..., offset : offset + self._n_row_tril]
        offset += self._n_row_tril
        col_raw = raw[..., offset : offset + self._n_col_tril]

        U_chol = self._build_tril(row_raw, n, self._n_row_tril)
        V_chol = self._build_tril(col_raw, p, self._n_col_tril)
        # Cholesky of the Kronecker covariance Sigma = V (X) U is
        # L_Sigma = V_chol (X) U_chol so vec(X) ~ N(vec(M), L L^T).
        L_kron = torch.einsum("...ij,...kl->...ikjl", V_chol, U_chol).reshape(
            *loc_flat.shape[:-1], n * p, n * p
        )
        return D.MultivariateNormal(loc_flat, scale_tril=L_kron)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y has shape (..., n*p) or (..., n, p); flatten if matrix.
        if y.dim() >= 2 and y.shape[-2:] == (self._rows, self._cols):
            y = y.reshape(*y.shape[:-2], self._rows * self._cols)
        return self._get_dist(x).log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        sample_flat = self._get_dist(x).rsample(sample_shape)
        return sample_flat.reshape(*sample_flat.shape[:-1], self._rows, self._cols)


# ============================================================================
# matrix-valued: InverseWishart
# ============================================================================


class ConditionalInverseWishart(ContinuousMorphism):
    """Conditional Inverse-Wishart over positive-definite matrices.

    Conjugate prior for the covariance of a multivariate normal.
    Realised as a deterministic inversion of a Wishart sample so
    autograd flows; equivalent in distribution to drawing
    :math:`\\Sigma^{-1} \\sim \\mathcal{W}(\\nu, V^{-1})` and
    inverting.  See Gelman et al. (2013) §3.6 for the conjugacy
    statement.

    Parameters
    ----------
    domain : AnySpace
        Source space.
    codomain : ContinuousSpace
        Target space whose ``dim`` is the matrix size :math:`d`.
    hidden_dim : int
        Hidden layer width for the parameter network.
    """

    event_rank: int = 2

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        n_tril = d * (d + 1) // 2
        self._d = d
        self._n_tril = n_tril
        self.param_source = _make_source(
            domain,
            1 + n_tril,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.positive_definite

    def _get_wishart(self, x: torch.Tensor) -> D.Wishart:
        raw = self.param_source(x)
        d = self._d
        # df must exceed d - 1.
        df = F.softplus(raw[..., 0]) + d
        tril_raw = raw[..., 1:]
        batch_shape = tril_raw.shape[:-1]
        L = torch.zeros(
            *batch_shape, d, d, device=tril_raw.device, dtype=tril_raw.dtype
        )
        idx = torch.tril_indices(d, d)
        L[..., idx[0], idx[1]] = tril_raw
        diag_idx = torch.arange(d)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + EPS
        # IW(nu, V) is the inverse of W(nu, V^-1); we parameterise
        # the wishart with scale_tril L and invert the sample.
        return D.Wishart(df=df, scale_tril=L)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y is positive-definite matrix Sigma; the wishart variate
        # is Sigma^{-1}.  log p_IW(Sigma) = log p_W(Sigma^{-1}) +
        # log |d Sigma^{-1} / d Sigma| where the Jacobian of matrix
        # inversion at a d x d positive-definite matrix is
        # (-1)^d det(Sigma)^{-(d+1)} so the log absolute Jacobian
        # is -(d + 1) log det(Sigma).
        d = self._d
        y_mat = (
            y.reshape(*y.shape[:-1], d, d)
            if y.dim() < 2 or y.shape[-2:] != (d, d)
            else y
        )
        if y_mat.shape[-2:] != (d, d):
            y_mat = y.reshape(*y.shape[:-1], d, d)
        y_inv = torch.linalg.inv(y_mat)
        log_det = torch.logdet(y_mat)
        wishart = self._get_wishart(x)
        return wishart.log_prob(y_inv) - (d + 1) * log_det

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        w = self._get_wishart(x).rsample(sample_shape)
        sigma = torch.linalg.inv(w)
        return sigma.reshape(*sigma.shape[:-2], self._d * self._d)


# ============================================================================
# optional: GeneralizedPareto (may not be in all torch versions)
# ============================================================================

# ============================================================================
# discrete-valued conditional distributions
# ============================================================================


class ConditionalBernoulli(ContinuousMorphism):
    """Conditional Bernoulli: continuous probability -> discrete truth value.

    Takes a continuous input x and produces learnable logits that
    parameterize a Bernoulli distribution. The output is a discrete
    sample in {0, 1}, returned as a LongTensor.

    This is the key bridge used in PDS (Grove & White) for the
    ``Bern x`` pattern, where a LogitNormal draw x in (0,1)
    parameterizes a Bernoulli over truth values.

    The codomain must be a FinSet of size 2 (representing {False, True}
    or {0, 1}).

    Note
    ----
    Sampling from Bernoulli is NOT reparameterizable. Gradients
    do not flow through the discrete samples. Use score function
    estimators (REINFORCE) or the Gumbel-Softmax trick if
    differentiable samples are needed.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space (typically UnitInterval or a FinSet).
    codomain : SetObject
        Target FinSet of size 2.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        from quivers.core.objects import SetObject

        if not isinstance(codomain, SetObject) or codomain.size != 2:
            raise ValueError(
                f"ConditionalBernoulli requires a FinSet(2) codomain, got {codomain!r}"
            )

        super().__init__(domain, codomain)

        # one logit per input
        self.param_source = _make_source(
            domain,
            1,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    def _get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Bernoulli probabilities from input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Probabilities. Shape (batch,).
        """
        logits = self.param_source(x).squeeze(-1)  # (batch,)
        return torch.sigmoid(logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability of discrete output y given input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Discrete output in {0, 1}. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        probs = self._get_probs(x)
        dist = D.Bernoulli(probs=probs)
        return dist.log_prob(y.float())

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Sample from Bernoulli (not reparameterizable).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Discrete samples in {0, 1}. Shape (*sample_shape, batch).
        """
        probs = self._get_probs(x)
        dist = D.Bernoulli(probs=probs)
        return dist.sample(sample_shape).long()


class ConditionalCategorical(ContinuousMorphism):
    """Conditional Categorical: continuous input -> discrete category.

    Generalizes ConditionalBernoulli to k > 2 categories.
    Takes a continuous input and produces learnable logits over k
    categories. The output is a discrete sample in {0, ..., k-1}.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : SetObject
        Target FinSet of size k.
    hidden_dim : int
        Hidden layer width for neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        from quivers.core.objects import SetObject

        if not isinstance(codomain, SetObject):
            raise ValueError(
                f"ConditionalCategorical requires a FinSet codomain, got {codomain!r}"
            )

        super().__init__(domain, codomain)
        self._k = codomain.size
        self.param_source = _make_source(
            domain,
            self._k,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self):
        """Discrete-integer support over ``{0, …, k-1}``."""
        return _constraints.integer_interval(0, self._k - 1)

    def _get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute category logits from input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Logits. Shape (batch, k).
        """
        return self.param_source(x)  # (batch, k)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability of discrete output y given input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Discrete output in {0, ..., k-1}. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        logits = self._get_logits(x)
        dist = D.Categorical(logits=logits)
        return dist.log_prob(y.long())

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Sample from Categorical (not reparameterizable).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Discrete samples in {0, ..., k-1}. Shape (*sample_shape, batch).
        """
        logits = self._get_logits(x)
        dist = D.Categorical(logits=logits)
        return dist.sample(sample_shape).long()


class ConditionalBinomial(ContinuousMorphism):
    """Conditional Binomial(total_count, probs(x)).

    The ``total_count`` (number of trials) is a fixed
    hyperparameter set at construction time, typical for binomial
    likelihoods where ``n`` is known per observation. Only the
    ``probs`` parameter is learnable.

    Outputs integer counts in ``{0, 1, ..., total_count}``.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space.
    total_count : int
        Number of Bernoulli trials per observation.
    hidden_dim : int
        Hidden layer width for the parameter source.
    """

    # Class-level transpile-time metadata. `support` is the broadest
    # interval `nonnegative_integer`; the per-instance upper bound
    # `total_count` lives on the instance and the variational guide
    # picks it up via `_total_count` for bijector construction.
    arg_constraints: dict[str, _constraints.Constraint] = dict(
        D.Binomial.arg_constraints
    )
    support: _constraints.Constraint = _constraints.nonnegative_integer  # type: ignore[assignment]

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        total_count: int = 1,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        if total_count < 1:
            raise ValueError(
                f"ConditionalBinomial: total_count must be >= 1, got {total_count}"
            )
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self._total_count = int(total_count)
        self.param_source = _make_source(
            domain,
            d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    def _get_dist(self, x: torch.Tensor) -> D.Binomial:
        logits = self.param_source(x)
        return D.Binomial(total_count=self._total_count, logits=logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y.float()).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.sample(sample_shape).long()


class ConditionalLogisticNormal(ContinuousMorphism):
    """Conditional LogisticNormal on the simplex.

    Pushes a Normal(loc(x), scale(x)) draw through the softmax
    transform to produce a simplex-valued sample. Multivariate
    analogue of `ConditionalLogitNormal`. Useful as an
    alternative to `ConditionalDirichlet` when the
    underlying simplex distribution should be Gaussian in
    logit space rather than Beta-shaped.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space; ``codomain.dim`` is the simplex dimension.
    hidden_dim : int
        Hidden layer width for the parameter source.
    """

    arg_constraints: dict[str, _constraints.Constraint] = dict(
        D.LogisticNormal.arg_constraints
    )
    support: _constraints.Constraint = _constraints.simplex  # type: ignore[assignment]

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        # We use a Normal in (d-1)-dim space and the
        # StickBreakingTransform to land on the d-simplex.
        # torch.distributions.LogisticNormal handles this.
        self.param_source = _make_source(
            domain,
            2 * (d - 1),
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )
        self._d = d

    def _get_dist(self, x: torch.Tensor) -> D.LogisticNormal:
        raw = self.param_source(x)
        d_minus_1 = self._d - 1
        loc = raw[..., :d_minus_1]
        scale = F.softplus(raw[..., d_minus_1:]) + EPS
        return D.LogisticNormal(loc, scale)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.rsample(sample_shape)


class ConditionalZeroInflatedPoisson(ContinuousMorphism):
    """Conditional ZeroInflatedPoisson(zero_prob(x), rate(x)).

    The parameter source produces `2 * codomain.dim` numbers per
    input row: the first half feeds a sigmoid to produce the
    zero-inflation probability, the second half a softplus to
    produce the Poisson rate. Outputs non-negative integer counts.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.nonnegative_integer

    def _get_dist(self, x: torch.Tensor) -> ZeroInflatedPoisson:
        raw = self.param_source(x)
        zero_prob = torch.sigmoid(raw[..., : self._d]).clamp(EPS, 1.0 - EPS)
        rate = F.softplus(raw[..., self._d :]) + EPS
        return ZeroInflatedPoisson(zero_prob, rate)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._get_dist(x).log_prob(y.float()).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        return self._get_dist(x).sample(sample_shape).long()


class ConditionalHurdlePoisson(ContinuousMorphism):
    """Conditional HurdlePoisson(zero_prob(x), rate(x)).

    Same parameter shape as `ConditionalZeroInflatedPoisson` but the
    two-stage hurdle log-density: a Bernoulli for zero vs positive,
    then a zero-truncated Poisson for the strictly-positive branch.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.nonnegative_integer

    def _get_dist(self, x: torch.Tensor) -> HurdlePoisson:
        raw = self.param_source(x)
        zero_prob = torch.sigmoid(raw[..., : self._d]).clamp(EPS, 1.0 - EPS)
        rate = F.softplus(raw[..., self._d :]) + EPS
        return HurdlePoisson(zero_prob, rate)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._get_dist(x).log_prob(y.float()).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        return self._get_dist(x).sample(sample_shape).long()


class ConditionalZeroOneInflatedBeta(ContinuousMorphism):
    """Conditional ZeroOneInflatedBeta(mu(x), phi(x), zoi(x), coi(x)).

    Parameter source emits ``4 * dim`` numbers per row: logit-mu,
    log-phi, logit-zoi, logit-coi. Used by the runtime family registry
    for ``~ ZeroOneInflatedBeta`` declarations; formula frontend
    prefers the inline ``observe ... <- ZeroOneInflatedBeta(mu, phi, zoi, coi)``.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self.param_source = _make_source(
            domain,
            4 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.unit_interval

    def _get_dist(self, x: torch.Tensor) -> ZeroOneInflatedBeta:
        raw = self.param_source(x)
        d = self._d
        mu = torch.sigmoid(raw[..., :d]).clamp(EPS, 1.0 - EPS)
        phi = F.softplus(raw[..., d : 2 * d]) + EPS
        zoi = torch.sigmoid(raw[..., 2 * d : 3 * d]).clamp(EPS, 1.0 - EPS)
        coi = torch.sigmoid(raw[..., 3 * d : 4 * d]).clamp(EPS, 1.0 - EPS)
        return ZeroOneInflatedBeta(mu, phi, zoi, coi)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._get_dist(x).log_prob(y.float()).sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        return self._get_dist(x).sample(sample_shape)


class ConditionalMixtureNormal(ContinuousMorphism):
    """Conditional finite Gaussian mixture with input-driven weights,
    locations, and scales.

    The parameter source produces `3 * K` numbers per row, where
    `K` is the number of mixture components (fixed at construction).
    Weights pass through softmax, locations are emitted directly,
    scales pass through softplus + epsilon.

    The codomain is assumed scalar (1-d real); a higher-dimensional
    extension would replace `Normal` with `Independent(Normal(...), 1)`
    and triple the per-component parameter count.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        num_components: int = 2,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        if num_components < 2:
            raise ValueError(
                "ConditionalMixtureNormal: num_components must be >= 2, "
                f"got {num_components}"
            )
        super().__init__(domain, codomain)
        self._k = int(num_components)
        self.param_source = _make_source(
            domain,
            3 * self._k,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.real

    @property
    def num_components(self) -> int:
        return self._k

    def _get_dist(self, x: torch.Tensor) -> MixtureNormal:
        raw = self.param_source(x)
        weights = F.softmax(raw[..., : self._k], dim=-1)
        loc = raw[..., self._k : 2 * self._k]
        scale = F.softplus(raw[..., 2 * self._k :]) + EPS
        return MixtureNormal(weights, loc, scale)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._get_dist(x).log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        return self._get_dist(x).sample(sample_shape)


class ConditionalOneHotCategorical(ContinuousMorphism):
    """Conditional OneHotCategorical(probs(x)).

    Generalises `ConditionalCategorical` to one-hot
    encoded outputs (vector of zeros with a single one). Useful
    as a discrete-output observation kernel where downstream
    code wants a vector rather than an integer index.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space; ``codomain.dim`` is the number of categories.
    hidden_dim : int
        Hidden layer width for the parameter source.
    """

    # torch's `OneHotCategorical.support` is `OneHot()`; the
    # variational guide treats it as a simplex bijector target.
    arg_constraints: dict[str, _constraints.Constraint] = dict(
        D.OneHotCategorical.arg_constraints
    )
    support: _constraints.Constraint = D.OneHotCategorical.support  # type: ignore[assignment, misc]

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self.param_source = _make_source(
            domain,
            d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    def _get_dist(self, x: torch.Tensor) -> D.OneHotCategorical:
        logits = self.param_source(x)
        return D.OneHotCategorical(logits=logits)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y.float())

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.sample(sample_shape).float()


class ConditionalMixture(ContinuousMorphism):
    """K-component mixture of a conditional family.

    Wraps a single conditional family class (one of the registered
    ``ConditionalX`` types) and gives it K independent
    parameterizations plus learnable mixture logits, producing

        p(y | x) = sum_k pi_k(x) * f_k(y | x)

    where each ``f_k`` is an instance of the component class and
    ``pi`` is the softmax of ``K`` learnable logits.

    Sampling is via ancestral simulation (Categorical pick + the
    chosen component's ``rsample``). ``log_prob`` evaluates the
    log-sum-exp of the per-component log-densities. The Categorical
    pick is non-reparameterizable; gradient flow through the
    weights uses the score-function path (which higher-level
    objectives like IWAE can route through).

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space; matches the component family's codomain.
    component_class : type
        A ``ConditionalX`` class accepting ``(domain, codomain,
        hidden_dim)`` constructor args.
    num_components : int
        Number of mixture components.
    hidden_dim : int
        Hidden width for both the mixture-logit MLP and each
        component's parameter source.
    """

    # Class-level transpile-time metadata. The user-facing argument
    # is the per-component mixing probability vector; per-component
    # parameters travel through the component morphisms themselves.
    # The output `support` mirrors the first component's support and
    # is exposed at instance access via the component delegation
    # below.
    arg_constraints: dict[str, _constraints.Constraint] = {
        "weights": _constraints.simplex,
    }
    support: _constraints.Constraint = _constraints.real  # type: ignore[assignment]

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        component_class: type,
        num_components: int = 4,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        if num_components < 2:
            raise ValueError(
                f"ConditionalMixture: num_components must be >= 2, got {num_components}"
            )
        super().__init__(domain, codomain)
        self._K = int(num_components)
        self._components = torch.nn.ModuleList(
            [component_class(domain, codomain, hidden_dim) for _ in range(self._K)]
        )
        self.mixture_logits = _make_source(
            domain,
            self._K,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    def _log_weights(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mixture_logits(x)
        return torch.log_softmax(logits, dim=-1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_w = self._log_weights(x)
        per_comp = torch.stack(
            [comp.log_prob(x, y) for comp in self._components], dim=-1
        )
        return torch.logsumexp(log_w + per_comp, dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        log_w = self._log_weights(x)
        cat = D.Categorical(logits=log_w)
        comp_idx = cat.sample(sample_shape)  # (*sample_shape, batch)
        # Draw from every component, then gather along a new K axis.
        comp_samples = torch.stack(
            [comp.rsample(x, sample_shape) for comp in self._components],
            dim=-1,
        )
        # ``comp_samples`` has shape ``(*sample_shape, batch, *event,
        # K)``. Broadcast ``comp_idx`` (shape ``(*sample_shape,
        # batch)``) to match every event axis with size 1 then 1 on
        # the K axis, so the final gather picks one element per
        # batch row.
        idx = comp_idx
        for _ in range(comp_samples.dim() - 1 - idx.dim()):
            idx = idx.unsqueeze(-1)
        idx = idx.unsqueeze(-1)  # K-axis broadcast dim
        idx = idx.expand(*comp_samples.shape[:-1], 1)
        return comp_samples.gather(-1, idx).squeeze(-1)


class ConditionalIndependent(ContinuousMorphism):
    """Reinterpret the trailing batch dimension of a base conditional
    family as an event dimension.

    Equivalent to wrapping the base distribution in
    `torch.distributions.Independent` with
    ``reinterpreted_batch_ndims = 1``. Used to make per-element
    independence explicit when a downstream guide wants to score
    a vector-valued observation as a single event.

    Parameters
    ----------
    base : ContinuousMorphism
        The base conditional family. The wrapped distribution sums
        the base's per-element log-probabilities along the last
        axis to score a vector-valued observation.
    """

    # Class-level transpile-time metadata. Argument-shape constraints
    # are empty: the wrapped morphism's parameters surface through
    # the base reference. `support` defaults to `real` at the class
    # level; the wrapped base's support is recoverable via the
    # `_base` attribute at instance access time.
    arg_constraints: dict[str, _constraints.Constraint] = {}
    support: _constraints.Constraint = _constraints.real  # type: ignore[assignment]

    def __init__(self, base: ContinuousMorphism) -> None:
        super().__init__(base.domain, base.codomain)
        self._base = base

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The base's log_prob already sums along the last axis for
        # ``_IndependentConditional``; ``Independent`` adds the
        # explicit reinterpretation for any base.
        lp = self._base.log_prob(x, y)
        if lp.dim() > 1:
            return lp.sum(dim=-1)
        return lp

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        return self._base.rsample(x, sample_shape)


class ConditionalTransformed(ContinuousMorphism):
    """A base conditional family composed with a chain of bijectors.

    Equivalent to `torch.distributions.TransformedDistribution`
    lifted to ContinuousMorphism. The ``transforms`` are applied in
    forward order to ``rsample`` outputs; ``log_prob`` includes the
    log-determinant Jacobian correction.

    Parameters
    ----------
    base : ContinuousMorphism
        Base conditional family.
    transforms : list of torch.distributions.Transform
        Bijectors applied in forward order. Each must implement
        ``__call__``, ``inv``, and ``log_abs_det_jacobian``.
    """

    # Class-level transpile-time metadata. Argument-shape constraints
    # are empty: the wrapped morphism's parameters surface through
    # the base reference and each transform exposes its own
    # `domain` / `codomain` constraint. `support` defaults to `real`
    # at the class level; the composed transforms' codomain
    # constraint is recoverable via the `_transforms` attribute at
    # instance access time.
    arg_constraints: dict[str, _constraints.Constraint] = {}
    support: _constraints.Constraint = _constraints.real  # type: ignore[assignment]

    def __init__(
        self,
        base: ContinuousMorphism,
        transforms: list,
    ) -> None:
        super().__init__(base.domain, base.codomain)
        self._base = base
        self._transforms = list(transforms)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        z = self._base.rsample(x, sample_shape)
        for tf in self._transforms:
            z = tf(z)
        return z

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Invert the transforms to recover the base sample, then
        # accumulate the log-det-Jacobian corrections.
        z = y
        log_det = torch.zeros(
            y.shape[:-1] if y.dim() > 1 else (1,),
            device=y.device,
            dtype=y.dtype,
        )
        for tf in reversed(self._transforms):
            z_prev = tf.inv(z)
            ldj = tf.log_abs_det_jacobian(z_prev, z)
            if ldj.dim() > log_det.dim():
                ldj = ldj.sum(dim=tuple(range(log_det.dim(), ldj.dim())))
            log_det = log_det + ldj
            z = z_prev
        return self._base.log_prob(x, z) - log_det


class ConditionalLKJCholesky(ContinuousMorphism):
    """Conditional LKJCholesky(dim, concentration(x)).

    Produces lower-triangular Cholesky factors of correlation
    matrices on the LKJ distribution (Lewandowski-Kurowicka-Joe
    2009, doi:10.1016/j.jmva.2009.04.008). The matrix dimension
    is taken from ``codomain.dim``; only the concentration parameter
    is learnable.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space; ``codomain.dim`` is the correlation-matrix size.
    hidden_dim : int
        Hidden layer width for the parameter source.
    """

    arg_constraints: dict[str, _constraints.Constraint] = dict(
        D.LKJCholesky.arg_constraints
    )
    support: _constraints.Constraint = _constraints.corr_cholesky  # type: ignore[assignment]

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        self._matrix_dim = codomain.dim
        self.param_source = _make_source(
            domain,
            1,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    def _get_dist(self, x: torch.Tensor) -> D.LKJCholesky:
        raw = self.param_source(x)
        concentration = F.softplus(raw.squeeze(-1)) + 0.1
        return D.LKJCholesky(dim=self._matrix_dim, concentration=concentration)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        dist = self._get_dist(x)
        return dist.sample(sample_shape)


# ============================================================================
# Gaussian process: covariance kernel evaluated at input locations
# ============================================================================


class _GPCholeskyError(RuntimeError):
    """Raised when the jittered GP kernel matrix fails Cholesky.

    The kernel matrix `K + jitter * I` should be positive definite by
    construction; failure indicates that the chosen length scale,
    amplitude, or input locations have produced a numerically
    degenerate covariance (e.g. duplicate locations with near-zero
    length scale). The error surfaces the diagnostic moments rather
    than silently inflating the jitter, which would corrupt the
    log-density without notice.
    """


_GP_KERNEL_CHOICES = ("rbf", "matern52", "linear")


def _gp_squared_distance(x: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance over the last-but-one axis.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(..., N, D)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(..., N, N)`` whose ``[i, j]`` entry is
        ``||x[..., i, :] - x[..., j, :]||^2``.
    """
    norms = (x * x).sum(dim=-1, keepdim=True)
    cross = x @ x.transpose(-1, -2)
    sq = norms + norms.transpose(-1, -2) - 2.0 * cross
    return sq.clamp(min=0.0)


class ConditionalGaussianProcess(ContinuousMorphism):
    """Gaussian process prior with mean zero and a chosen covariance kernel.

    A [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process)
    is a Markov kernel ``X^N -> G(R^N)`` whose value at a finite set
    of input locations ``x_1, ..., x_N`` follows a multivariate
    Normal with covariance matrix ``K(x_i, x_j)``. Unlike the
    parametric families that derive their distribution parameters
    from a neural network on the input, the GP's "parameters" are
    the input locations themselves: the kernel function evaluated
    on the inputs produces the covariance directly.

    Reference: Rasmussen & Williams (2006),
    [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/).

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space. Its ``dim`` is the per-location feature
        dimensionality ``D`` of the inputs.
    codomain : ContinuousSpace
        Target space. Its ``dim`` is the number of input locations
        ``N`` at which the GP is evaluated.
    kernel : {"rbf", "matern52", "linear"}
        Covariance kernel. ``"rbf"`` is the squared-exponential
        kernel; ``"matern52"`` is the Matern kernel with smoothness
        ``nu = 5/2``; ``"linear"`` is the inner-product kernel.
    length_scale : float
        Initial length scale of the kernel (positive; learnable).
        Ignored by the linear kernel.
    amplitude : float
        Initial amplitude (positive; learnable). Multiplies the
        kernel by ``amplitude^2``.
    jitter : float
        Diagonal regulariser added to ``K`` for numerical
        positive-definiteness of the Cholesky factorisation.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        amplitude: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(domain, codomain)
        if kernel not in _GP_KERNEL_CHOICES:
            raise ValueError(
                f"ConditionalGaussianProcess: unknown kernel {kernel!r}; "
                f"valid choices: {_GP_KERNEL_CHOICES}"
            )
        if length_scale <= 0.0:
            raise ValueError(
                f"ConditionalGaussianProcess: length_scale must be > 0, got {length_scale!r}"
            )
        if amplitude <= 0.0:
            raise ValueError(
                f"ConditionalGaussianProcess: amplitude must be > 0, got {amplitude!r}"
            )
        self._kernel = kernel
        self._jitter = jitter
        self._n = codomain.dim
        self._d = getattr(domain, "dim", None)
        # Store raw (pre-softplus) parameters so the transformed value
        # is strictly positive and the optimiser sees unconstrained
        # variables.
        inv_softplus_ls = math.log(math.expm1(length_scale))
        inv_softplus_amp = math.log(math.expm1(amplitude))
        self._raw_length_scale = torch.nn.Parameter(
            torch.tensor(inv_softplus_ls, dtype=torch.get_default_dtype())
        )
        self._raw_amplitude = torch.nn.Parameter(
            torch.tensor(inv_softplus_amp, dtype=torch.get_default_dtype())
        )

    @property
    def length_scale(self) -> torch.Tensor:
        """Current (positive) length scale of the kernel."""
        return F.softplus(self._raw_length_scale) + EPS

    @property
    def amplitude(self) -> torch.Tensor:
        """Current (positive) amplitude of the kernel."""
        return F.softplus(self._raw_amplitude) + EPS

    def _kernel_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix ``K(x, x)`` of shape ``(..., N, N)``."""
        amp2 = self.amplitude * self.amplitude
        if self._kernel == "linear":
            inner = x @ x.transpose(-1, -2)
            return amp2 * inner
        ls = self.length_scale
        sq = _gp_squared_distance(x)
        if self._kernel == "rbf":
            return amp2 * torch.exp(-0.5 * sq / (ls * ls))
        # matern52
        r = torch.sqrt(sq + EPS)
        sqrt5_r_over_l = math.sqrt(5.0) * r / ls
        poly = 1.0 + sqrt5_r_over_l + (5.0 / 3.0) * sq / (ls * ls)
        return amp2 * poly * torch.exp(-sqrt5_r_over_l)

    def _jittered_cholesky(self, x: torch.Tensor) -> torch.Tensor:
        K = self._kernel_matrix(x)
        n = K.shape[-1]
        eye = torch.eye(n, dtype=K.dtype, device=K.device)
        K_j = K + self._jitter * eye
        try:
            return torch.linalg.cholesky(K_j)
        except RuntimeError as exc:
            diag_min = K_j.diagonal(dim1=-2, dim2=-1).min().item()
            diag_max = K_j.diagonal(dim1=-2, dim2=-1).max().item()
            raise _GPCholeskyError(
                "ConditionalGaussianProcess: Cholesky failed on the "
                f"jittered kernel matrix (kernel={self._kernel!r}, "
                f"length_scale={self.length_scale.item():.6g}, "
                f"amplitude={self.amplitude.item():.6g}, "
                f"jitter={self._jitter:.3g}, N={n}, "
                f"diag(K_j) in [{diag_min:.3g}, {diag_max:.3g}]). "
                "Check for duplicate input locations or an excessively "
                "small length scale."
            ) from exc

    def _validate_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``x`` to ``(..., N, D)`` and validate dims."""
        if x.dim() < 2:
            raise ValueError(
                "ConditionalGaussianProcess: input must have shape "
                f"(..., N, D) with N={self._n}; got tensor of shape {tuple(x.shape)}"
            )
        if x.shape[-2] != self._n:
            raise ValueError(
                "ConditionalGaussianProcess: input location count "
                f"x.shape[-2] = {x.shape[-2]} does not match codomain dim "
                f"N = {self._n}"
            )
        return x

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._validate_input_shape(x)
        L = self._jittered_cholesky(x)
        loc = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        mvn = D.MultivariateNormal(loc, scale_tril=L)
        return mvn.log_prob(y)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        x = self._validate_input_shape(x)
        L = self._jittered_cholesky(x)
        loc = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        mvn = D.MultivariateNormal(loc, scale_tril=L)
        return mvn.rsample(sample_shape)


# ============================================================================
# Horseshoe: global-local shrinkage prior with numerical marginal density
# ============================================================================


def _gauss_legendre_16(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """16-point Gauss-Legendre nodes and weights on ``[0, 1]``.

    Standard nodes / weights on ``[-1, 1]`` (from Abramowitz & Stegun,
    table 25.4) are mapped to ``[0, 1]`` via ``t = (1 + xi) / 2`` with
    weights scaled by ``1/2``.
    """
    nodes_pm1 = (
        -0.9894009349916499,
        -0.9445750230732326,
        -0.8656312023878318,
        -0.7554044083550030,
        -0.6178762444026438,
        -0.4580167776572274,
        -0.2816035507792589,
        -0.0950125098376374,
        0.0950125098376374,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026438,
        0.7554044083550030,
        0.8656312023878318,
        0.9445750230732326,
        0.9894009349916499,
    )
    weights_pm1 = (
        0.0271524594117541,
        0.0622535239386479,
        0.0951585116824928,
        0.1246289712555339,
        0.1495959888165767,
        0.1691565193950025,
        0.1826034150449236,
        0.1894506104550685,
        0.1894506104550685,
        0.1826034150449236,
        0.1691565193950025,
        0.1495959888165767,
        0.1246289712555339,
        0.0951585116824928,
        0.0622535239386479,
        0.0271524594117541,
    )
    nodes = torch.tensor(
        [(1.0 + n) * 0.5 for n in nodes_pm1], device=device, dtype=dtype
    )
    weights = torch.tensor([w * 0.5 for w in weights_pm1], device=device, dtype=dtype)
    return nodes, weights


class ConditionalHorseshoe(ContinuousMorphism):
    """Carvalho-Polson-Scott horseshoe prior.

    The [horseshoe prior](https://doi.org/10.1093/biomet/asq017) places
    a global-local shrinkage structure on each coordinate:

    .. code-block:: text

       tau ~ HalfCauchy(scale)
       lambda_d ~ HalfCauchy(1)            for d = 1, ..., D
       beta_d | tau, lambda_d ~ Normal(0, (tau * lambda_d)^2)

    The marginal density of ``beta_d`` after integrating the local
    scale ``lambda_d`` has no closed form; this implementation uses a
    16-point Gauss-Legendre quadrature after mapping the half-line
    ``lambda in (0, inf)`` to ``t in (0, 1)`` via the change of
    variables ``lambda = tan(pi * t / 2)``, whose Jacobian is
    ``(pi / 2) * sec^2(pi * t / 2)``.

    Reference: Carvalho, Polson & Scott (2010),
    [The horseshoe estimator for sparse signals](https://doi.org/10.1093/biomet/asq017).

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space. The prior is conditionally independent of ``x``;
        ``x`` only carries the batch shape.
    codomain : ContinuousSpace
        Target space. Its ``dim`` is the coordinate count ``D``.
    scale : float
        Initial global shrinkage ``tau`` (positive; learnable).
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        scale: float = 1.0,
    ) -> None:
        super().__init__(domain, codomain)
        if scale <= 0.0:
            raise ValueError(f"ConditionalHorseshoe: scale must be > 0, got {scale!r}")
        self._d = codomain.dim
        inv_softplus_scale = math.log(math.expm1(scale))
        self._raw_scale = torch.nn.Parameter(
            torch.tensor(inv_softplus_scale, dtype=torch.get_default_dtype())
        )

    @property
    def scale(self) -> torch.Tensor:
        """Current (positive) global shrinkage ``tau``."""
        return F.softplus(self._raw_scale) + EPS

    def _batch_shape(self, x: torch.Tensor) -> torch.Size:
        """Infer the batch shape from ``x``: discrete domains have
        ``x.shape`` directly, continuous domains drop the trailing
        feature dim."""
        if x.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return x.shape
        if x.dim() == 0:
            return torch.Size(())
        return x.shape[:-1]

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        batch = self._batch_shape(x)
        shape = torch.Size(sample_shape) + batch + (self._d,)
        tau = self.scale
        half_cauchy_local = D.HalfCauchy(
            torch.ones((), dtype=tau.dtype, device=tau.device)
        )
        lam = half_cauchy_local.rsample(shape)
        z = torch.randn(shape, dtype=tau.dtype, device=tau.device)
        return tau * lam * z

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tau = self.scale
        device = y.device
        dtype = y.dtype
        t_nodes, t_weights = _gauss_legendre_16(device, dtype)
        # Change of variables: lambda = tan(pi * t / 2), dlambda/dt
        # = (pi / 2) * sec^2(pi * t / 2). Integrand:
        #   p(beta) = int_0^inf Normal(beta; 0, (tau * lambda)^2)
        #             * HalfCauchy(lambda; 0, 1) dlambda
        pi = math.pi
        half_pi_t = 0.5 * pi * t_nodes
        lam = torch.tan(half_pi_t)
        sec2 = 1.0 / (torch.cos(half_pi_t) ** 2)
        jacobian = 0.5 * pi * sec2
        # Half-Cauchy(0, 1) density on lambda: 2 / (pi * (1 + lambda^2)).
        log_hc = math.log(2.0 / pi) - torch.log1p(lam * lam)
        # Broadcast: y has shape (..., D); we need per-coordinate
        # log-density. Add a quadrature axis at position 0.
        y_expanded = y.unsqueeze(-1)  # (..., D, 1)
        sigma = tau * lam  # (Q,)
        # Normal log-density: -0.5 * log(2*pi) - log(sigma) - 0.5 * (y/sigma)^2
        log_norm = (
            -0.5 * math.log(2.0 * pi)
            - torch.log(sigma)
            - 0.5 * (y_expanded / sigma) ** 2
        )  # (..., D, Q)
        log_integrand = log_norm + log_hc + torch.log(jacobian)
        # Sum (in log-space) weighted by quadrature weights.
        log_weights = torch.log(t_weights)
        log_p_per_coord = torch.logsumexp(
            log_integrand + log_weights, dim=-1
        )  # (..., D)
        return log_p_per_coord.sum(dim=-1)


class ConditionalLearnedOrderedLogistic(ContinuousMorphism):
    """Conditional OrderedLogistic(predictor(x), cutpoints(x)), cutpoints learned.

    This is the morphism a ``~ OrderedLogistic`` declaration compiles to.
    [`ConditionalOrderedLogistic`][quivers.continuous.ordered.ConditionalOrderedLogistic]
    is the variant that takes its cutpoints explicitly.

    The continuous input drives a parameter source that produces
    `1 + (K - 1)` numbers: one real predictor and a `K - 1` cutpoint
    vector. The cutpoints are passed through a strictly-monotonic
    transform (first entry free, subsequent entries via cumulative
    softplus) so the cumulative-link contract `c_0 < c_1 < ... <
    c_{K-2}` is satisfied unconditionally.

    Outputs integer categories in `{0, ..., K - 1}` where `K =
    codomain.size`.

    Parameters
    ----------
    domain : AnySpace
        Source space.
    codomain : AnySpace
        A finite set of at least two ordered categories.
    hidden_dim : int or sequence of int
        Hidden widths, read only by a source that has hidden layers.
    param_source, param_source_option
        Select the parameter source directly, or by the DSL's
        ``[param_source=...]`` text.

    Raises
    ------
    ValueError
        If the codomain is not a finite set of at least two categories.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        if not isinstance(codomain, SetObject):
            raise ValueError(
                "ConditionalLearnedOrderedLogistic requires a FinSet codomain, "
                f"got {codomain!r}"
            )
        if codomain.size < 2:
            raise ValueError(
                "ConditionalLearnedOrderedLogistic requires K >= 2 categories, "
                f"got codomain.size = {codomain.size}"
            )
        super().__init__(domain, codomain)
        self._k = codomain.size
        self.param_source = _make_source(
            domain,
            1 + (self._k - 1),
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.integer_interval(0, self._k - 1)

    def _get_dist(self, x: torch.Tensor) -> OrderedLogistic:
        raw = self.param_source(x)
        predictor = raw[..., 0]
        cut_raw = raw[..., 1:]
        first = cut_raw[..., :1]
        rest = F.softplus(cut_raw[..., 1:]) + EPS
        cutpoints = torch.cat([first, first + torch.cumsum(rest, dim=-1)], dim=-1)
        return OrderedLogistic(predictor, cutpoints)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._get_dist(x).log_prob(y.long())

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        return self._get_dist(x).sample(sample_shape).long()


# ============================================================================
# optional: generalized Pareto (requires recent torch)
# ============================================================================

try:
    _GPD = D.GeneralizedPareto

    class ConditionalGeneralizedPareto(ContinuousMorphism):
        """Conditional generalized Pareto distribution.

        Parameters
        ----------
        domain : SetObject or ContinuousSpace
            Source space.
        codomain : ContinuousSpace
            Target space.
        hidden_dim : int or sequence of int
            Hidden widths, read only by a source that has hidden layers.
        param_source, param_source_option
            Select the parameter source directly, or by the DSL's
            ``[param_source=...]`` text.
        """

        def __init__(
            self,
            domain: AnySpace,
            codomain: ContinuousSpace,
            hidden_dim: int | Sequence[int] | None = None,
            param_source: ParamSource | None = None,
            param_source_option: str | None = None,
        ) -> None:
            super().__init__(domain, codomain)
            d = codomain.dim
            # loc + scale + concentration
            self.param_source = _make_source(
                domain,
                3 * d,
                hidden_dim,
                param_source=param_source,
                param_source_option=param_source_option,
            )
            self._d = d

        def _get_dist(self, x: torch.Tensor) -> D.GeneralizedPareto:
            raw = self.param_source(x)
            d = self._d
            loc = raw[..., :d]
            scale = F.softplus(raw[..., d : 2 * d]) + EPS
            concentration = raw[..., 2 * d :]
            return _GPD(loc, scale, concentration)

        def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            dist = self._get_dist(x)
            return dist.log_prob(y).sum(dim=-1)

        def rsample(
            self,
            x: torch.Tensor,
            sample_shape: torch.Size = torch.Size(),
        ) -> torch.Tensor:
            dist = self._get_dist(x)
            return dist.rsample(sample_shape)

    _HAS_GPD = True

except AttributeError:
    _HAS_GPD = False


# ============================================================================
# FamilySpec registrations for hand-written classes
# ============================================================================
#
# The factory-generated families (Cauchy, Laplace, ..., VonMises) are
# registered inside `_make_family`. The hand-written specials
# below (Normal, LogitNormal, Beta, TruncatedNormal, Dirichlet, Uniform,
# MultivariateNormal, LowRankMVN, RelaxedBernoulli,
# RelaxedOneHotCategorical, Wishart, Bernoulli, Categorical) require
# explicit registration. Each `FamilySpec` records the same
# canonical info the factory-generated ones do; the
# ``conditional_class_override`` field carries the hand-written class
# so the unified catalog can build a ConditionalX from any DSL name.


_register_family(
    FamilySpec(
        name="Normal",
        dist_class=D.Normal,
        params=(
            ParamSpec(name="loc", transform="id"),
            ParamSpec(name="scale", transform="softplus"),
        ),
        support=_constraints.real,
        output_kind="independent",
        docstring="Conditional Normal(loc(x), scale(x)).",
        conditional_class_override=ConditionalNormal,
    )
)

_register_family(
    FamilySpec(
        name="LogitNormal",
        dist_class=D.TransformedDistribution,
        params=(
            ParamSpec(name="mu", transform="id"),
            ParamSpec(name="sigma", transform="softplus"),
        ),
        support=_constraints.unit_interval,
        output_kind="independent",
        docstring="LogitNormal(mu(x), sigma(x)) — Normal pushed through sigmoid.",
        conditional_class_override=ConditionalLogitNormal,
    )
)

_register_family(
    FamilySpec(
        name="Beta",
        dist_class=D.Beta,
        params=(
            ParamSpec(name="concentration1", transform="softplus_shifted"),
            ParamSpec(name="concentration0", transform="softplus_shifted"),
        ),
        support=_constraints.unit_interval,
        output_kind="independent",
        docstring="Conditional Beta(alpha(x), beta(x)) on (0, 1).",
        conditional_class_override=ConditionalBeta,
    )
)

_register_family(
    FamilySpec(
        name="TruncatedNormal",
        dist_class=D.Normal,  # surrogate; inline path handles truncation
        params=(
            ParamSpec(name="mu", transform="id"),
            ParamSpec(name="sigma", transform="softplus"),
        ),
        support=_constraints.real,  # interval set per-call when bounds are literal
        output_kind="independent",
        docstring="TruncatedNormal(mu(x), sigma(x), low, high) — Normal truncated to [low, high].",
        conditional_class_override=ConditionalTruncatedNormal,
    )
)

_register_family(
    FamilySpec(
        name="Dirichlet",
        dist_class=D.Dirichlet,
        params=(
            ParamSpec(
                name="concentration", transform="softplus_shifted", kind="vector"
            ),
        ),
        support=_constraints.simplex,
        output_kind="vector",
        docstring="Conditional Dirichlet(alpha(x)) on the simplex.",
        conditional_class_override=ConditionalDirichlet,
    )
)

_register_family(
    FamilySpec(
        name="Uniform",
        dist_class=D.Uniform,
        params=(
            ParamSpec(name="low", transform="id"),
            ParamSpec(name="high", transform="id"),
        ),
        support=_constraints.real,  # interval set per-call when bounds are literal
        output_kind="independent",
        docstring="Conditional Uniform(low(x), high(x)).",
        conditional_class_override=ConditionalUniform,
    )
)

_register_family(
    FamilySpec(
        name="MultivariateNormal",
        dist_class=D.MultivariateNormal,
        params=(
            ParamSpec(name="loc", transform="id"),
            ParamSpec(name="scale_tril", transform="id"),
        ),
        support=_constraints.real_vector,
        output_kind="mvn",
        docstring="Conditional MultivariateNormal(loc(x), scale_tril(x)).",
        conditional_class_override=ConditionalMultivariateNormal,
    )
)

_register_family(
    FamilySpec(
        name="LowRankMVN",
        dist_class=D.LowRankMultivariateNormal,
        params=(
            ParamSpec(name="loc", transform="id"),
            ParamSpec(name="cov_factor", transform="id"),
            ParamSpec(name="cov_diag", transform="softplus"),
        ),
        support=_constraints.real_vector,
        output_kind="mvn",
        docstring="Conditional LowRankMVN(loc(x), W(x), D(x)) — Σ = W W^T + diag(D).",
        conditional_class_override=ConditionalLowRankMVN,
    )
)

_register_family(
    FamilySpec(
        name="RelaxedBernoulli",
        dist_class=D.RelaxedBernoulli,
        params=(
            ParamSpec(name="temperature", transform="softplus"),
            ParamSpec(name="logits", transform="id"),
        ),
        support=_constraints.unit_interval,
        output_kind="independent",
        docstring="RelaxedBernoulli(temperature, logits(x)) — Gumbel-softmax / continuous Bernoulli.",
        conditional_class_override=ConditionalRelaxedBernoulli,
    )
)

_register_family(
    FamilySpec(
        name="RelaxedOneHotCategorical",
        dist_class=D.RelaxedOneHotCategorical,
        params=(
            ParamSpec(name="temperature", transform="softplus"),
            ParamSpec(name="logits", transform="id", kind="vector"),
        ),
        support=_constraints.simplex,
        output_kind="vector",
        docstring="RelaxedOneHotCategorical(temperature, logits(x)) — Gumbel-softmax on the simplex.",
        conditional_class_override=ConditionalRelaxedOneHotCategorical,
    )
)

_register_family(
    FamilySpec(
        name="Wishart",
        dist_class=D.Wishart,
        params=(
            ParamSpec(name="df", transform="softplus_shifted"),
            ParamSpec(name="scale_tril", transform="id"),
        ),
        support=_constraints.positive_definite,
        output_kind="matrix",
        docstring="Conditional Wishart(df(x), scale_tril(x)) — positive-definite matrices.",
        conditional_class_override=ConditionalWishart,
    )
)

_register_family(
    FamilySpec(
        name="Bernoulli",
        dist_class=D.Bernoulli,
        params=(ParamSpec(name="probs", transform="sigmoid"),),
        support=_constraints.boolean,
        discrete=True,
        output_kind="categorical",
        docstring="Conditional Bernoulli(probs(x)).",
        conditional_class_override=ConditionalBernoulli,
    )
)

_register_family(
    FamilySpec(
        name="Categorical",
        dist_class=D.Categorical,
        params=(ParamSpec(name="probs", transform="id", kind="vector"),),
        support=_constraints.nonnegative_integer,
        discrete=True,
        output_kind="categorical",
        docstring="Conditional Categorical(probs(x)) over {0, ..., k-1}.",
        conditional_class_override=ConditionalCategorical,
    )
)

_register_family(
    FamilySpec(
        name="Binomial",
        dist_class=D.Binomial,
        params=(
            ParamSpec(name="total_count", transform="softplus_shifted", kind="integer"),
            ParamSpec(name="probs", transform="sigmoid"),
        ),
        support=_constraints.nonnegative_integer,
        discrete=True,
        output_kind="categorical",
        docstring="Conditional Binomial(n=total_count, probs(x)).",
        conditional_class_override=ConditionalBinomial,
    )
)


_register_family(
    FamilySpec(
        name="LogisticNormal",
        dist_class=D.LogisticNormal,
        params=(
            ParamSpec(name="loc", transform="id"),
            ParamSpec(name="scale", transform="softplus"),
        ),
        support=_constraints.simplex,
        output_kind="vector",
        docstring="Conditional LogisticNormal(loc(x), scale(x)) on the simplex.",
        conditional_class_override=ConditionalLogisticNormal,
    )
)


_register_family(
    FamilySpec(
        name="OneHotCategorical",
        dist_class=D.OneHotCategorical,
        params=(ParamSpec(name="logits", transform="id", kind="vector"),),
        support=D.OneHotCategorical.support,
        discrete=False,  # vector output, not integer-scalar
        output_kind="vector",
        docstring="Conditional OneHotCategorical(logits(x)) — one-hot vector output.",
        conditional_class_override=ConditionalOneHotCategorical,
    )
)


_register_family(
    FamilySpec(
        name="LKJCholesky",
        dist_class=D.LKJCholesky,
        params=(ParamSpec(name="concentration", transform="softplus_shifted"),),
        support=_constraints.corr_cholesky,
        output_kind="matrix",
        docstring="Conditional LKJCholesky(matrix_dim, concentration(x)) for correlation Cholesky factors.",
        conditional_class_override=ConditionalLKJCholesky,
    )
)


_register_family(
    FamilySpec(
        name="GP",
        dist_class=D.MultivariateNormal,
        params=(
            ParamSpec(name="length_scale", transform="softplus"),
            ParamSpec(name="amplitude", transform="softplus"),
        ),
        support=_constraints.real_vector,
        output_kind="mvn",
        docstring="Gaussian process prior with covariance kernel evaluated at the inputs.",
        conditional_class_override=ConditionalGaussianProcess,
    )
)


_register_family(
    FamilySpec(
        name="Horseshoe",
        dist_class=D.Normal,  # surrogate; the marginal is numerical
        params=(ParamSpec(name="scale", transform="softplus"),),
        support=_constraints.real,
        output_kind="vector",
        docstring="Carvalho-Polson-Scott horseshoe prior with numerical marginal density.",
        conditional_class_override=ConditionalHorseshoe,
    )
)


if _HAS_GPD:
    _register_family(
        FamilySpec(
            name="GeneralizedPareto",
            dist_class=_GPD,  # type: ignore[name-defined]
            params=(
                ParamSpec(name="loc", transform="id"),
                ParamSpec(name="scale", transform="softplus"),
                ParamSpec(name="concentration", transform="id"),
            ),
            support=_constraints.real,
            output_kind="independent",
            docstring="Conditional GeneralizedPareto(loc(x), scale(x), concentration(x)).",
            conditional_class_override=ConditionalGeneralizedPareto,  # type: ignore[name-defined]
        )
    )
# ---------------------------------------------------------------------------
# Compound / shim families: Beta-Binomial, Logistic, Half-StudentT
# ---------------------------------------------------------------------------


class ConditionalBetaBinomial(ContinuousMorphism):
    """Conjugate Beta-Binomial likelihood with learnable Beta parameters.

    The number of successes ``y`` arises by drawing
    ``p ~ Beta(concentration1(x), concentration0(x))`` and then
    ``y ~ Binomial(total_count, p)``. The Beta is integrated out
    analytically, yielding the log-probability

    .. math::

        \\log p(y \\mid x) = \\log \\Gamma(\\alpha + \\beta)
            + \\log \\Gamma(\\alpha + y) + \\log \\Gamma(\\beta + n - y)
            - \\log \\Gamma(\\alpha) - \\log \\Gamma(\\beta)
            - \\log \\Gamma(\\alpha + \\beta + n)
            + \\log \\binom{n}{y}

    where :math:`\\alpha = \\mathrm{concentration1}(x)`,
    :math:`\\beta = \\mathrm{concentration0}(x)`, and :math:`n` is
    ``total_count``.

    Sampling is not reparameterisable; ``rsample`` raises and
    ``sample`` draws ``p`` from the Beta and ``y`` from the
    resulting Binomial.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space. ``codomain.dim`` independent Beta-Binomial
        observations are produced per input.
    total_count : int
        Number of Bernoulli trials per observation; must be a
        positive integer.
    hidden_dim : int
        Hidden layer width for the neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        total_count: int,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        if total_count < 1:
            raise ValueError(
                f"ConditionalBetaBinomial: total_count must be >= 1, got {total_count}"
            )
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self._total_count = int(total_count)
        # Two raw scalars per codomain dim: alpha and beta concentrations.
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.integer_interval(0, self._total_count)

    def _get_concentrations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_source(x)
        log_alpha = raw[..., : self._d]
        log_beta = raw[..., self._d :]
        alpha = F.softplus(log_alpha) + EPS
        beta = F.softplus(log_beta) + EPS
        return alpha, beta

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha, beta = self._get_concentrations(x)
        n = float(self._total_count)
        y_f = y.float()
        lgamma = torch.lgamma
        n_tensor = torch.tensor(n, device=y_f.device, dtype=y_f.dtype)
        log_binom = (
            lgamma(n_tensor + 1.0) - lgamma(y_f + 1.0) - lgamma(n_tensor - y_f + 1.0)
        )
        log_p = (
            lgamma(alpha + beta)
            + lgamma(alpha + y_f)
            + lgamma(beta + n_tensor - y_f)
            - lgamma(alpha)
            - lgamma(beta)
            - lgamma(alpha + beta + n_tensor)
            + log_binom
        )
        return log_p.sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        raise NotImplementedError(
            "ConditionalBetaBinomial.rsample is not supported: "
            "Beta-Binomial sampling is not reparameterisable; "
            "use .sample() instead."
        )

    def sample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        with torch.no_grad():
            alpha, beta = self._get_concentrations(x)
            p = D.Beta(alpha, beta).sample(sample_shape)
            return D.Binomial(total_count=self._total_count, probs=p).sample().long()


class ConditionalLogistic(ContinuousMorphism):
    """Conditional logistic distribution on the real line.

    Independent per codomain dim with learnable location and scale:

    .. math::

        \\log p(y \\mid x) = -z - 2 \\log(1 + e^{-z}) - \\log \\sigma

    where :math:`z = (y - \\mu) / \\sigma`. Samples are
    reparameterised via the inverse-CDF transform of a uniform draw,
    :math:`y = \\mu + \\sigma \\log(u / (1 - u))`.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space. ``codomain.dim`` independent logistic
        components are produced per input.
    hidden_dim : int
        Hidden layer width for the neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        # param_dim = d (loc) + d (raw_scale).
        self.param_source = _make_source(
            domain,
            2 * d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.real

    def _get_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.param_source(x)
        loc = raw[..., : self._d]
        scale = F.softplus(raw[..., self._d :]) + EPS
        return loc, scale

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loc, scale = self._get_params(x)
        z = (y - loc) / scale
        log_p = -z - 2.0 * F.softplus(-z) - scale.log()
        return log_p.sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        loc, scale = self._get_params(x)
        u = torch.rand(
            *sample_shape,
            *loc.shape,
            device=loc.device,
            dtype=loc.dtype,
        ).clamp(min=EPS, max=1.0 - EPS)
        return loc + scale * (u.log() - (-u).log1p())


class ConditionalHalfStudentT(ContinuousMorphism):
    """Conditional half-StudentT on the positive reals.

    A StudentT with ``df`` fixed at construction time, learnable
    scale, and location fixed at zero, restricted to nonnegative
    values via reflection. The folded log-density at ``y >= 0`` is
    ``log 2 + log_prob_studentt(0, scale(x), df; y)`` and ``-inf``
    elsewhere.

    Sampling reflects a base StudentT draw through the origin:
    ``y = |z|`` with ``z ~ StudentT(df, 0, scale)``.
    Reparameterisation flows through the absolute-value operation.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : ContinuousSpace
        Target space. ``codomain.dim`` independent half-StudentT
        components are produced per input.
    df : float
        Degrees-of-freedom hyperparameter (positive). Held fixed
        across observations; only ``scale`` is learnable.
    hidden_dim : int
        Hidden layer width for the neural parameter source.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: ContinuousSpace,
        df: float,
        hidden_dim: int | Sequence[int] | None = None,
        param_source: ParamSource | None = None,
        param_source_option: str | None = None,
    ) -> None:
        if df <= 0.0:
            raise ValueError(f"ConditionalHalfStudentT: df must be > 0, got {df!r}")
        super().__init__(domain, codomain)
        d = codomain.dim
        self._d = d
        self._df = float(df)
        self.param_source = _make_source(
            domain,
            d,
            hidden_dim,
            param_source=param_source,
            param_source_option=param_source_option,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return _constraints.greater_than(0.0)

    def _get_scale(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.param_source(x)
        return F.softplus(raw) + EPS

    def _base_dist(self, scale: torch.Tensor) -> D.StudentT:
        df_tensor = torch.full_like(scale, self._df)
        loc = torch.zeros_like(scale)
        return D.StudentT(df_tensor, loc, scale)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale = self._get_scale(x)
        base = self._base_dist(scale)
        in_support = y >= 0.0
        # Fold: log p_half(y) = log 2 + log p_base(y) for y >= 0,
        # else -inf. Clamp y to a finite nonnegative value before
        # evaluating the base density to avoid NaN from StudentT at
        # negative inputs (which is fine; the mask zeros it out).
        y_safe = y.clamp(min=0.0)
        base_lp = base.log_prob(y_safe)
        folded = math.log(2.0) + base_lp
        masked = torch.where(in_support, folded, torch.full_like(folded, float("-inf")))
        return masked.sum(dim=-1)

    def rsample(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        scale = self._get_scale(x)
        base = self._base_dist(scale)
        return base.rsample(sample_shape).abs()


# ---------------------------------------------------------------------------
# LKJ correlation prior on Cholesky factors
# ---------------------------------------------------------------------------


class LKJCorrelationFactor(ContinuousMorphism):
    """LKJ prior on Cholesky factors ``LKJ(K, η)`` over CholeskyFactor(K).

    Density on the Cholesky factor:

    .. math::

        p(L) \\propto \\prod_{k=2}^{K} L_{kk}^{K - k + 2(\\eta - 1)}.

    A higher concentration :math:`\\eta > 1` pulls toward the
    identity correlation; :math:`\\eta = 1` is uniform on
    correlations. Sampling uses the *onion method* of
    Lewandowski-Kurowicka-Joe 2009: draw row-norm partial
    correlations from Beta distributions and form :math:`L`
    row-by-row.

    Parameters
    ----------
    dim : int
        Correlation-matrix size :math:`K \\ge 2`.
    eta : float
        Concentration :math:`\\eta > 0`.
    domain : AnySpace
        The morphism's source (parameter conditioning); typically
        the program's input space. The LKJ prior itself does not
        consume per-observation conditioning, so the rsample path
        broadcasts the prior across the batch dimension.
    """

    # Class-level transpile-time metadata. The constructor takes the
    # correlation-matrix dimension `dim` (a positive integer) and the
    # concentration `eta` (a positive real). Output is a Cholesky
    # factor of a correlation matrix.
    arg_constraints: dict[str, _constraints.Constraint] = {
        "dim": _constraints.positive_integer,
        "eta": _constraints.positive,
    }
    support: _constraints.Constraint = _constraints.corr_cholesky  # type: ignore[assignment]

    def __init__(self, dim: int, eta: float, domain: AnySpace) -> None:
        if dim < 2:
            raise ValueError(f"LKJ requires dim >= 2; got {dim}")
        if eta <= 0:
            raise ValueError(f"LKJ requires eta > 0; got {eta}")
        codomain = CholeskyFactor(name=f"L({dim})", dim=dim)
        super().__init__(domain, codomain)
        self._dim = dim
        self._eta = float(eta)

    def _distribution(self, x: torch.Tensor) -> D.LKJCholesky:
        """The LKJ distribution, on the input's device and float dtype.

        Parameters
        ----------
        x : torch.Tensor
            The input; it carries no parameters.

        Returns
        -------
        torch.distributions.LKJCholesky
            The prior over ``dim``-by-``dim`` Cholesky factors.
        """
        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        concentration = torch.tensor(self._eta, device=x.device, dtype=dtype)
        return D.LKJCholesky(self._dim, concentration)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Draw one Cholesky factor per input row, by the onion method.

        Parameters
        ----------
        x : torch.Tensor
            The input; only its row count, device, and dtype are read.
        sample_shape : torch.Size
            Leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Flat factors, shape ``(*sample_shape, batch, dim * dim)``.
        """
        batch = x.shape[0]
        factors = self._distribution(x).sample(torch.Size(sample_shape) + (batch,))
        return factors.reshape(*factors.shape[:-2], self._dim * self._dim)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Normalized log density of the LKJ prior at Cholesky factors.

        Parameters
        ----------
        x : torch.Tensor
            The input; it carries no parameters.
        y : torch.Tensor
            Factors, flat ``(..., dim * dim)`` or square
            ``(..., dim, dim)``.

        Returns
        -------
        torch.Tensor
            One log density per factor.
        """
        K = self._dim
        if y.shape[-2:] != (K, K):
            y = y.reshape(*y.shape[:-1], K, K)
        return self._distribution(x).log_prob(y)

    def __repr__(self) -> str:
        return f"LKJCorrelationFactor(dim={self._dim}, eta={self._eta})"


# ---------------------------------------------------------------------------
# Generic truncation combinator
# ---------------------------------------------------------------------------


class Truncated(ContinuousMorphism):
    """Truncate a base family to an interval :math:`[a, b]`.

    Categorical denotation: given a base family
    :math:`F : \\Theta \\to \\mathcal{G}(\\mathbb{R})` and constants
    :math:`a, b \\in \\bar{\\mathbb{R}}` with :math:`a < b`, the
    truncated family has density

    .. math::

        p_{F_{|[a,b]}}(x) = \\frac{p_F(x)}{F_{\\text{cdf}}(b)
        - F_{\\text{cdf}}(a)} \\cdot \\mathbb{1}_{[a,b]}(x)

    and the morphism :math:`F_{|[a,b]} : \\Theta \\to
    \\mathcal{G}([a,b])`. Sampling uses inverse-CDF when
    `base` supports it; otherwise rejection sampling.

    Parameters
    ----------
    base : ContinuousMorphism
        The base distribution-family morphism. Must expose
        ``log_prob`` and ``rsample`` plus an ``icdf`` method or a
        ``base_distribution`` torch ``Distribution`` for inverse-CDF
        sampling. Falls back to rejection sampling otherwise.
    lower : float or None
        Lower bound :math:`a`. ``None`` means :math:`-\\infty`.
    upper : float or None
        Upper bound :math:`b`. ``None`` means :math:`+\\infty`.
    max_rejection_iterations : int
        Cap on rejection-sampling attempts before raising.
    """

    # Class-level transpile-time metadata. The base wrapped morphism
    # carries its own argument constraints; the bounds are simple
    # real scalars. Class-level `support` defaults to `real`; per-
    # instance the lookup tightens via `support_at_bounds()`.
    arg_constraints: dict[str, _constraints.Constraint] = {
        "lower": _constraints.real,
        "upper": _constraints.real,
    }
    support: _constraints.Constraint = _constraints.real  # type: ignore[assignment]

    def __init__(
        self,
        base: ContinuousMorphism,
        lower: float | None = None,
        upper: float | None = None,
        max_rejection_iterations: int = 64,
    ) -> None:
        super().__init__(base.domain, base.codomain)
        if lower is None and upper is None:
            raise ValueError(
                "Truncated requires at least one of lower / upper to be finite; "
                "without truncation, use the base family directly"
            )
        if lower is not None and upper is not None and not (lower < upper):
            raise ValueError(
                f"Truncated requires lower < upper; got lower={lower}, upper={upper}"
            )
        self._base = base
        self._lower = lower
        self._upper = upper
        self._max_iters = max_rejection_iterations
        # Attach so the parent nn.Module tracks parameters.
        self._base_mod = base

    def _interval_logmass(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate :math:`\\log(F(b) - F(a))` via Monte-Carlo on the base.

        For families with analytic CDFs (Normal, etc.) we could use
        a closed form, but a general primitive operating across
        every family in the registry uses 256 base samples to
        estimate the truncation mass. This keeps the combinator
        usable on every family without per-family special-casing.
        """
        with torch.no_grad():
            samples = self._base.rsample(x.repeat(256, *([1] * (x.ndim - 1))))
            mask = torch.ones_like(samples)
            if self._lower is not None:
                mask = mask * (samples >= self._lower).float()
            if self._upper is not None:
                mask = mask * (samples <= self._upper).float()
            in_interval = mask.reshape(256, x.shape[0], -1).mean(dim=0)
        return torch.log(in_interval.clamp(min=1e-30)).squeeze(-1)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        del sample_shape
        """Rejection-sample from the base until every entry is in :math:`[a,b]`."""
        batch = x.shape[0]
        out = self._base.rsample(x)
        mask = self._in_bounds(out)
        for _ in range(self._max_iters):
            if mask.all():
                return out
            replacement = self._base.rsample(x)
            out = torch.where(
                mask.reshape(batch, *([1] * (out.ndim - 1))).expand_as(out),
                out,
                replacement,
            )
            mask = self._in_bounds(out)
        if not mask.all():
            raise RuntimeError(
                f"Truncated rejection sampling failed to fill the batch in "
                f"{self._max_iters} iterations; the truncation mass is "
                "vanishingly small for the supplied parameters"
            )
        return out

    def _in_bounds(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.ones(x.shape[:1], device=x.device, dtype=torch.bool)
        if self._lower is not None:
            m = m & (x.reshape(x.shape[0], -1) >= self._lower).all(dim=-1)
        if self._upper is not None:
            m = m & (x.reshape(x.shape[0], -1) <= self._upper).all(dim=-1)
        return m

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        base_lp = self._base.log_prob(x, y)
        # Hard-zero outside the interval.
        in_bounds = self._in_bounds(y)
        truncation_lp = self._interval_logmass(x)
        adjusted = base_lp - truncation_lp
        return torch.where(
            in_bounds,
            adjusted,
            torch.full_like(adjusted, float("-inf")),
        )

    def __repr__(self) -> str:
        return f"Truncated({self._base!r}, lower={self._lower}, upper={self._upper})"


__all__ = [
    "ConditionalNormal",
    "ConditionalLogitNormal",
    "ConditionalBeta",
    "ConditionalTruncatedNormal",
    "ConditionalDirichlet",
    "ConditionalCauchy",
    "ConditionalLaplace",
    "ConditionalGumbel",
    "ConditionalLogNormal",
    "ConditionalStudentT",
    "ConditionalExponential",
    "ConditionalGamma",
    "ConditionalChi2",
    "ConditionalHalfCauchy",
    "ConditionalHalfNormal",
    "ConditionalInverseGamma",
    "ConditionalWeibull",
    "ConditionalPareto",
    "ConditionalKumaraswamy",
    "ConditionalContinuousBernoulli",
    "ConditionalFisherSnedecor",
    "ConditionalPoisson",
    "ConditionalGeometric",
    "ConditionalNegativeBinomial",
    "ConditionalVonMises",
    "ConditionalUniform",
    "ConditionalMultivariateNormal",
    "ConditionalLowRankMVN",
    "ConditionalRelaxedBernoulli",
    "ConditionalRelaxedOneHotCategorical",
    "ConditionalWishart",
    "ConditionalMatrixNormal",
    "ConditionalInverseWishart",
    "ConditionalBernoulli",
    "ConditionalCategorical",
    "ConditionalBinomial",
    "ConditionalLogisticNormal",
    "ConditionalLearnedOrderedLogistic",
    "ConditionalZeroInflatedPoisson",
    "ConditionalHurdlePoisson",
    "ConditionalZeroOneInflatedBeta",
    "ConditionalMixtureNormal",
    "ConditionalOneHotCategorical",
    "ConditionalMixture",
    "ConditionalIndependent",
    "ConditionalTransformed",
    "ConditionalLKJCholesky",
    "ConditionalGaussianProcess",
    "ConditionalHorseshoe",
    "ConditionalBetaBinomial",
    "ConditionalLogistic",
    "ConditionalHalfStudentT",
    "LKJCorrelationFactor",
    "Truncated",
]
if _HAS_GPD:
    __all__.append("ConditionalGeneralizedPareto")
