"""Shared log-space ordinal probability kernels and discrete distribution adapter."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints

OrdinalKind = Literal["cumulative", "acat", "sratio", "cratio"]
OrdinalLink = Literal[
    "logit", "probit", "probit_approx", "cloglog", "cauchit", "softit"
]


def _log1mexp(x: Tensor) -> Tensor:
    """Evaluate log(1-exp(x)) without cancellation or inactive-branch NaNs.

    Parameters
    ----------
    x : Tensor
        Nonpositive logarithms.

    Returns
    -------
    Tensor
        Logarithms of complementary probabilities.
    """
    near = x > -0.6931471805599453
    safe_near = torch.where(near, x, -torch.ones_like(x))
    safe_far = torch.where(near, -torch.ones_like(x), x)
    return torch.where(
        near, torch.log(-torch.expm1(safe_near)), torch.log1p(-safe_far.exp())
    )


def _log_cdf_pair(z: Tensor, link: OrdinalLink) -> tuple[Tensor, Tensor]:
    """Return log CDF and log survival at finite ordinal contrasts.

    Parameters
    ----------
    z : Tensor
        Finite contrasts.
    link : {"logit", "probit", "probit_approx", "cloglog", "cauchit", "softit"}
        Static inverse link.

    Returns
    -------
    tuple of Tensor
        Log CDF and log survival.

    Raises
    ------
    ValueError
        If the link is unsupported.
    """
    if link == "logit":
        return torch.nn.functional.logsigmoid(z), torch.nn.functional.logsigmoid(-z)
    if link == "probit":
        return torch.special.log_ndtr(z), torch.special.log_ndtr(-z)
    if link == "probit_approx":
        argument = 0.07056 * z.pow(3) + 1.5976 * z
        return torch.nn.functional.logsigmoid(argument), torch.nn.functional.logsigmoid(
            -argument
        )
    if link == "cloglog":
        finite_exp = z <= math.log(torch.finfo(z.dtype).max)
        safe = torch.where(finite_exp, z, torch.zeros_like(z))
        log_sf = torch.where(finite_exp, -safe.exp(), -torch.inf)
        small = z < -35
        regular = torch.where(small, -torch.ones_like(z), log_sf)
        return torch.where(small, z, _log1mexp(regular)), log_sf
    if link == "cauchit":
        tail = z.abs() > 1
        magnitude = torch.where(tail, z.abs(), torch.ones_like(z))
        log_small = (1 / magnitude).atan().log() - math.log(math.pi)
        log_large = _log1mexp(log_small)
        safe = torch.where(tail, torch.zeros_like(z), z)
        cdf = (0.5 + safe.atan() / math.pi).log()
        sf = (0.5 - safe.atan() / math.pi).log()
        return (
            torch.where(tail, torch.where(z < 0, log_small, log_large), cdf),
            torch.where(tail, torch.where(z < 0, log_large, log_small), sf),
        )
    if link == "softit":
        positive = torch.nn.functional.softplus(z)
        small = z < -35
        safe = torch.where(small, torch.zeros_like(z), z)
        log_positive = torch.where(small, z, torch.nn.functional.softplus(safe).log())
        denominator = positive.log1p()
        return log_positive - denominator, -denominator
    raise ValueError(f"Unsupported ordinal link {link!r}")


def category_log_probs(
    predictor: Tensor,
    cutpoints: Tensor,
    *,
    kind: OrdinalKind = "cumulative",
    link: OrdinalLink = "logit",
    discrimination: Tensor | float = 1.0,
    category_effects: Tensor | None = None,
) -> Tensor:
    """Compute ordinal category log probabilities in linear time and storage.

    Parameters
    ----------
    predictor : Tensor
        Scalar predictor, with arbitrary batch dimensions.
    cutpoints : Tensor
        Thresholds with last dimension K-1. Cumulative contrasts must increase.
    kind : {"cumulative", "acat", "sratio", "cratio"}
        Static ordinal likelihood construction.
    link : {"logit", "probit", "probit_approx", "cloglog", "cauchit", "softit"}
        Static inverse link for transition probabilities.
    discrimination : Tensor or float
        Positive multiplier, broadcast over the predictor batch.
    category_effects : Tensor or None
        Optional predictor increments with explicit final transition axis K-1.

    Returns
    -------
    Tensor
        Normalized log probabilities with shape (*batch, K). Parameters must
        already have been validated by the distribution adapter. This pure
        kernel also supports automatic differentiation transforms.

    Raises
    ------
    ValueError
        If the static kind or link is unsupported.
    """
    disc = torch.as_tensor(
        discrimination, dtype=predictor.dtype, device=predictor.device
    )
    eta = predictor.unsqueeze(-1)
    if category_effects is not None:
        eta = eta + category_effects
    z = disc.unsqueeze(-1) * (cutpoints - eta)
    if kind == "acat":
        if link == "logit":
            log_odds = -z
        else:
            log_q, log_not_q = _log_cdf_pair(-z, link)
            log_odds = log_q - log_not_q
        logits = torch.cat((torch.zeros_like(z[..., :1]), log_odds.cumsum(-1)), -1)
        return logits.log_softmax(-1)
    log_cdf, log_sf = _log_cdf_pair(-z if kind == "cratio" else z, link)
    if kind == "cumulative":
        if link == "logit":
            # sigmoid(b)-sigmoid(a) = sigmoid(b)*sigmoid(-a)*(1-exp(a-b)).
            middle = (
                log_cdf[..., 1:]
                + log_sf[..., :-1]
                + _log1mexp(z[..., :-1] - z[..., 1:])
            )
        else:
            # Mirror positive-tail intervals before differencing log CDFs.
            mirror = log_cdf[..., :-1] > -math.log(2)
            high = torch.where(mirror, log_sf[..., :-1], log_cdf[..., 1:])
            low = torch.where(mirror, log_sf[..., 1:], log_cdf[..., :-1])
            finite = torch.isfinite(high)
            gap = torch.where(finite, low, -torch.ones_like(low)) - torch.where(
                finite, high, torch.zeros_like(high)
            )
            middle = torch.where(finite, high + _log1mexp(gap), -torch.inf)
        return torch.cat((log_cdf[..., :1], middle, log_sf[..., -1:]), -1)
    if kind == "sratio":
        log_stop, log_continue = log_cdf, log_sf
    elif kind == "cratio":
        log_continue, log_stop = log_cdf, log_sf
    else:
        raise ValueError(f"Unsupported ordinal kind {kind!r}")
    prefix = torch.cat((torch.zeros_like(z[..., :1]), log_continue.cumsum(-1)), -1)
    return torch.cat((log_stop + prefix[..., :-1], prefix[..., -1:]), -1)


class OrdinalDistribution(Distribution):
    """Validated scalar ordinal distribution with shared category probabilities.

    Parameters
    ----------
    predictor : Tensor
        Finite floating point predictor with arbitrary batch dimensions.
    cutpoints : Tensor
        Finite thresholds with final transition axis K-1, K at least two.
    kind : {"cumulative", "acat", "sratio", "cratio"}
        Static ordinal likelihood construction.
    link : {"logit", "probit", "probit_approx", "cloglog", "cauchit", "softit"}
        Static inverse link for transition probabilities.
    discrimination : Tensor or float
        Finite, strictly positive discrimination; fixed to one by default.
    category_effects : Tensor or None
        Finite predictor increments with final transition axis K-1.
    hurdle_prob : Tensor or float or None
        Optional probability for an additional lowest category zero. The base
        ordinal categories are shifted up by one.
    validate_args : bool or None
        Validate finite parameters, positive discrimination, ordered cumulative
        contrasts, and integral category values. Follows PyTorch's default.
    """

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Parameter constraints are checked jointly in the constructor.

        Returns
        -------
        dict
            Empty independent-constraint mapping.
        """
        return {}

    has_rsample = False

    def __init__(
        self,
        predictor: Tensor,
        cutpoints: Tensor,
        *,
        kind: OrdinalKind = "cumulative",
        link: OrdinalLink = "logit",
        discrimination: Tensor | float = 1.0,
        category_effects: Tensor | None = None,
        hurdle_prob: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if isinstance(discrimination, bool) or (
            isinstance(discrimination, Tensor) and discrimination.dtype == torch.bool
        ):
            raise ValueError("ordinal discrimination cannot be boolean")
        if isinstance(hurdle_prob, bool) or (
            isinstance(hurdle_prob, Tensor) and hurdle_prob.dtype == torch.bool
        ):
            raise ValueError("ordinal hurdle probability cannot be boolean")
        if category_effects is not None and not category_effects.is_floating_point():
            raise ValueError("ordinal category_effects must be real floating tensors")
        if kind not in ("cumulative", "acat", "sratio", "cratio"):
            raise ValueError(f"Unsupported ordinal kind {kind!r}")
        if link not in (
            "logit",
            "probit",
            "probit_approx",
            "cloglog",
            "cauchit",
            "softit",
        ):
            raise ValueError(f"Unsupported ordinal link {link!r}")
        if kind in {"sratio", "cratio"} and link == "softit":
            raise ValueError("softit is not a supported sequential ordinal link")
        if not predictor.is_floating_point() or not cutpoints.is_floating_point():
            raise ValueError(
                "ordinal predictor and cutpoints must be real floating tensors"
            )
        if cutpoints.ndim == 0 or cutpoints.shape[-1] < 1:
            raise ValueError("ordinal cutpoints require a final dimension of size >= 1")
        if category_effects is not None and (
            category_effects.ndim == 0
            or category_effects.shape[-1] != cutpoints.shape[-1]
        ):
            raise ValueError("ordinal category_effects must have K-1 transitions")
        self.predictor = predictor
        self.cutpoints = cutpoints
        self.kind: OrdinalKind = kind
        self.link: OrdinalLink = link
        self.discrimination = torch.as_tensor(
            discrimination, dtype=predictor.dtype, device=predictor.device
        )
        self.category_effects = category_effects
        self.hurdle_prob = (
            None
            if hurdle_prob is None
            else torch.as_tensor(
                hurdle_prob, dtype=predictor.dtype, device=predictor.device
            )
        )
        shapes = [predictor.shape, cutpoints.shape[:-1], self.discrimination.shape]
        if category_effects is not None:
            shapes.append(category_effects.shape[:-1])
        if self.hurdle_prob is not None:
            shapes.append(self.hurdle_prob.shape)
        batch_shape = torch.broadcast_shapes(*shapes)
        self._num_categories = cutpoints.shape[-1] + 1 + (hurdle_prob is not None)
        super().__init__(batch_shape, torch.Size(), validate_args=validate_args)
        if self._validate_args:
            values = [predictor, cutpoints, self.discrimination]
            if category_effects is not None:
                values.append(category_effects)
            if not all(bool(torch.isfinite(value).all()) for value in values):
                raise ValueError("ordinal parameters must be finite")
            if not bool((self.discrimination > 0).all()):
                raise ValueError("ordinal discrimination must be strictly positive")
            if kind == "cumulative":
                contrasts = cutpoints
                if category_effects is not None:
                    contrasts = contrasts - category_effects
                if not bool((contrasts[..., 1:] > contrasts[..., :-1]).all()):
                    raise ValueError(
                        "cumulative cutpoints/contrasts must be strictly increasing"
                    )
            if self.hurdle_prob is not None and not bool(
                (
                    torch.isfinite(self.hurdle_prob)
                    & (self.hurdle_prob >= 0)
                    & (self.hurdle_prob <= 1)
                ).all()
            ):
                raise ValueError(
                    "ordinal hurdle probability must be finite and in [0, 1]"
                )

    @property
    def num_categories(self) -> int:
        """Number of response categories, including an optional hurdle category.

        Returns
        -------
        int
            Number of categories.
        """
        return self._num_categories

    @constraints.dependent_property
    def support(self):
        """Integer response support.

        Returns
        -------
        Constraint
            Inclusive interval from zero to K-1.
        """
        return constraints.integer_interval(0, self._num_categories - 1)

    def category_log_probs(self) -> Tensor:
        """Return all category log probabilities.

        Returns
        -------
        Tensor
            Log probabilities with shape (*batch, K).
        """
        result = category_log_probs(
            self.predictor,
            self.cutpoints,
            kind=self.kind,
            link=self.link,
            discrimination=self.discrimination,
            category_effects=self.category_effects,
        )
        result = result.expand(*self.batch_shape, result.shape[-1])
        if self.hurdle_prob is None:
            return result
        hu = self.hurdle_prob.expand(self.batch_shape)
        # Mask endpoints before logs: inactive singular derivatives must not
        # contaminate the derivative of the finite branch.
        interior = (hu > 0) & (hu < 1)
        safe = torch.where(interior, hu, torch.full_like(hu, 0.5))
        log_hu = torch.where(hu == 0, -torch.inf, safe.log())
        log_hu = torch.where(hu == 1, torch.zeros_like(hu), log_hu)
        log_rest = torch.where(hu == 1, -torch.inf, torch.log1p(-safe))
        log_rest = torch.where(hu == 0, torch.zeros_like(hu), log_rest)
        return torch.cat((log_hu.unsqueeze(-1), log_rest.unsqueeze(-1) + result), -1)

    def category_probs(self) -> Tensor:
        """Return all category probabilities.

        Returns
        -------
        Tensor
            Probabilities with shape (*batch, K).
        """
        if self.hurdle_prob is None:
            return self.category_log_probs().exp()
        base = (
            category_log_probs(
                self.predictor,
                self.cutpoints,
                kind=self.kind,
                link=self.link,
                discrimination=self.discrimination,
                category_effects=self.category_effects,
            )
            .exp()
            .expand(*self.batch_shape, self.num_categories - 1)
        )
        hu = self.hurdle_prob.expand(self.batch_shape).unsqueeze(-1)
        return torch.cat((hu, (1 - hu) * base), -1)

    def _category_log_probs(self) -> Tensor:
        """Compatibility alias for category_log_probs.

        Returns
        -------
        Tensor
            Log probabilities with shape (*batch, K).
        """
        return self.category_log_probs()

    def _category_probs(self) -> Tensor:
        """Compatibility alias for category_probs.

        Returns
        -------
        Tensor
            Probabilities with shape (*batch, K).
        """
        return self.category_probs()

    @property
    def mean(self) -> Tensor:
        """Response expectation on zero-based category codes.

        Returns
        -------
        Tensor
            Batch of means.
        """
        weights = torch.arange(
            self.num_categories,
            dtype=self.predictor.dtype,
            device=self.predictor.device,
        )
        return (self.category_probs() * weights).sum(-1)

    @property
    def mode(self) -> Tensor:
        """Most probable response category.

        Returns
        -------
        Tensor
            Batch of integer category codes.
        """
        return self.category_log_probs().argmax(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        """Score integral category values without coercing invalid observations.

        Parameters
        ----------
        value : Tensor
            Category codes broadcast against the distribution batch.

        Returns
        -------
        Tensor
            Log masses, or negative infinity outside support when unchecked.

        Raises
        ------
        ValueError
            For boolean values, or invalid categories when validation is enabled.
        """
        if value.dtype == torch.bool:
            raise ValueError("ordinal response categories cannot be boolean")
        if self._validate_args:
            self._validate_sample(value)
        log_probs = self.category_log_probs()
        shape = torch.broadcast_shapes(value.shape, self.batch_shape)
        valid = (value >= 0) & (value < self.num_categories) & (value == value.floor())
        safe = torch.where(valid, value, torch.zeros_like(value))
        index = safe.long().expand(shape)
        result = (
            log_probs.expand(*shape, self.num_categories)
            .gather(-1, index.unsqueeze(-1))
            .squeeze(-1)
        )
        return torch.where(valid.expand(shape), result, -torch.inf)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        """Draw non-reparameterized integer category samples.

        Parameters
        ----------
        sample_shape : Sequence of int
            Additional leading sample dimensions.

        Returns
        -------
        Tensor
            Integer samples with shape (*sample_shape, *batch_shape).
        """
        sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.batch_shape
        if shape.numel() == 0:
            return torch.empty(shape, dtype=torch.long, device=self.predictor.device)
        return torch.distributions.Categorical(logits=self.category_log_probs()).sample(
            sample_shape
        )
