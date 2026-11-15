"""Response families and link functions for the formula frontend.

Each `Family` knows how to emit a QVR ``observe`` step plus
the inverse-link expression that wraps a linear predictor.  The
registry maps brms-style family names to a typed family record;
the compiler dispatches on the family name to slot the right
inverse-link / observe lines into the emitted ``.qvr`` source.

Each family declares location and auxiliary links separately. The compiler
converts regression parameters to QVR constructor parameters, for instance
Weibull mean/shape to scale/shape. Categorical and finite-mixture families
add compiler-owned component axes because their likelihood parameters are
vectors rather than scalar auxiliaries.
"""

from __future__ import annotations

from typing import Mapping

import didactic.api as dx


class Link(dx.Model):
    """Inverse-link function bridging a linear predictor on
    :math:`\\mathbb{R}` to the family's natural parameter space.

    Attributes
    ----------
    name : str
        Canonical name (e.g. ``"identity"``, ``"logit"``, ``"log"``,
        ``"probit"``, ``"cloglog"``).
    inverse_expr : str
        QVR let-expression template applied to a linear predictor
        ``eta`` to produce the family parameter.  Substitution
        token ``{eta}`` is replaced by the linear predictor's
        variable name. ``"{eta}"`` itself is the identity link. The
        compiler lowers the canonical names in `links` directly to AST;
        this expression records the corresponding inverse for inspection.
    """

    name: str
    inverse_expr: str


class AuxParam(dx.Model):
    """An auxiliary family parameter (scale, dispersion, etc.).

    Attributes
    ----------
    name : str
        Variable name to emit in the ``.qvr`` source.
    prior : str
        Prior distribution expression (e.g. ``"HalfCauchy(2.0)"``).
    link : Link
        Link applied to a per-row linear predictor if the parameter
        is itself distributionally regressed (``y ~ x``, ``sigma ~
        x``, etc.); otherwise unused.
    """

    name: str
    prior: str
    link: Link


class Family(dx.Model):
    """Response family: observation kernel plus its link and any
    auxiliary parameters.

    The compiler emits, for each family-keyed regression:

    1. The auxiliary-parameter latent draws (intercept-only by
       default; one per parameter named in `aux_params`).
    2. A ``let eta = <linear predictor>`` binding the location's
       linear predictor.
    3. A ``let mu = <link.inverse_expr>`` binding the natural
       parameter.
    4. A single ``observe y : Resp <- <observe_family>(mu, ...)``
       step parameterised by ``mu`` and the auxiliary parameters.
    """

    name: str
    location_link: Link
    observe_family: str
    extra_observe_args: tuple[str, ...] = ()
    aux_params: tuple[AuxParam, ...] = ()

    def with_links(self, **parameters: str) -> Family:
        """Return a family with selected parameter links.

        Parameters
        ----------
        **parameters : str
            Link names keyed by ``mu`` for the location or an auxiliary
            parameter name, for instance ``sigma="softplus"``.

        Returns
        -------
        Family
            An independent family value with the requested links.

        Raises
        ------
        ValueError
            If a parameter or link is unknown. Categorical models require
            their multinomial softmax location link.
        """
        known = {"mu", *(aux.name for aux in self.aux_params)}
        unknown = parameters.keys() - known
        if unknown:
            raise ValueError(
                f"unknown family parameters {sorted(unknown)}; choices are {sorted(known)}"
            )
        invalid = set(parameters.values()) - links.keys()
        if invalid:
            raise ValueError(
                f"unknown links {sorted(invalid)}; choices are {sorted(links)}"
            )
        if self.name == "categorical" and parameters.get("mu", "softmax") != "softmax":
            raise ValueError("categorical models require the softmax location link")
        if self.name == "cumulative" and parameters.get("mu", "identity") not in {
            "identity",
            "logit",
            "probit",
        }:
            raise ValueError("cumulative models support logit and probit links")
        location_link = (
            links[parameters["mu"]] if "mu" in parameters else self.location_link
        )
        return Family(
            name=self.name,
            location_link=location_link,
            observe_family=(
                "OrderedProbit" if location_link.name == "probit" else "OrderedLogistic"
            )
            if self.name == "cumulative"
            else self.observe_family,
            extra_observe_args=self.extra_observe_args,
            aux_params=tuple(
                AuxParam(
                    name=aux.name,
                    prior=aux.prior,
                    link=links[parameters[aux.name]]
                    if aux.name in parameters
                    else aux.link,
                )
                for aux in self.aux_params
            ),
        )


_IDENTITY = Link(name="identity", inverse_expr="{eta}")
_LOGIT = Link(name="logit", inverse_expr="sigmoid({eta})")
_LOG = Link(name="log", inverse_expr="exp({eta})")
_SOFTMAX = Link(name="softmax", inverse_expr="softmax({eta})")
_INVERSE = Link(name="inverse", inverse_expr="1.0 / {eta}")


links: Mapping[str, Link] = {
    "identity": _IDENTITY,
    "logit": _LOGIT,
    "log": _LOG,
    "softmax": _SOFTMAX,
    "inverse": _INVERSE,
    "probit": Link(
        name="probit", inverse_expr="0.5 * (1.0 + erf({eta} / 1.4142135623730951))"
    ),
    "probit_approx": Link(
        name="probit_approx",
        inverse_expr="sigmoid(0.07056 * ({eta}) ** 3.0 + 1.5976 * {eta})",
    ),
    "log1p": Link(name="log1p", inverse_expr="expm1({eta})"),
    "cloglog": Link(name="cloglog", inverse_expr="-expm1(-exp({eta}))"),
    "cauchit": Link(
        name="cauchit", inverse_expr="0.5 + atan({eta}) / 3.141592653589793"
    ),
    "sqrt": Link(name="sqrt", inverse_expr="({eta}) ** 2.0"),
    "inverse_square": Link(name="inverse_square", inverse_expr="1.0 / ({eta}) ** 0.5"),
    "logm1": Link(name="logm1", inverse_expr="1.0 + exp({eta})"),
    "softplus": Link(
        name="softplus",
        inverse_expr="0.5 * {eta} + 0.5 * abs({eta}) + log1p(exp(-abs({eta})))",
    ),
    "tan_half": Link(name="tan_half", inverse_expr="2.0 * atan({eta})"),
}
"""The built-in inverse links, keyed by link name."""


families: Mapping[str, Family] = {
    "gaussian": Family(
        name="gaussian",
        location_link=_IDENTITY,
        observe_family="Normal",
        extra_observe_args=("sigma",),
        aux_params=(
            AuxParam(
                name="sigma",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
        ),
    ),
    "bernoulli": Family(
        name="bernoulli",
        location_link=_LOGIT,
        observe_family="Bernoulli",
    ),
    "binomial": Family(
        name="binomial",
        location_link=_LOGIT,
        observe_family="Binomial",
    ),
    "beta_binomial": Family(
        name="beta_binomial",
        location_link=_LOGIT,
        observe_family="BetaBinomial",
        extra_observe_args=("phi",),
        aux_params=(AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),),
    ),
    "categorical": Family(
        name="categorical",
        location_link=_SOFTMAX,
        observe_family="Categorical",
    ),
    "poisson": Family(
        name="poisson",
        location_link=_LOG,
        observe_family="Poisson",
    ),
    "negative_binomial": Family(
        name="negative_binomial",
        location_link=_LOG,
        observe_family="NegativeBinomial",
        extra_observe_args=("disp",),
        aux_params=(
            AuxParam(
                name="disp",
                prior="Gamma(2.0, 2.0)",
                link=_LOG,
            ),
        ),
    ),
    "gamma": Family(
        name="gamma",
        location_link=_LOG,
        observe_family="Gamma",
        extra_observe_args=("shape",),
        aux_params=(
            AuxParam(
                name="shape",
                prior="Gamma(2.0, 2.0)",
                link=_LOG,
            ),
        ),
    ),
    "beta": Family(
        name="beta",
        location_link=_LOGIT,
        observe_family="Beta",
        extra_observe_args=("phi",),
        aux_params=(
            AuxParam(
                name="phi",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
        ),
    ),
    "student_t": Family(
        name="student_t",
        location_link=_IDENTITY,
        observe_family="StudentT",
        extra_observe_args=("nu", "sigma"),
        aux_params=(
            AuxParam(
                name="nu",
                prior="Gamma(2.0, 0.1)",
                link=links["logm1"],
            ),
            AuxParam(
                name="sigma",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
        ),
    ),
    "cumulative": Family(
        name="cumulative",
        location_link=_IDENTITY,
        observe_family="OrderedLogistic",
    ),
    "zero_inflated_poisson": Family(
        name="zero_inflated_poisson",
        location_link=_LOG,
        observe_family="ZeroInflatedPoisson",
        extra_observe_args=("zi",),
        aux_params=(
            AuxParam(
                name="zi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
        ),
    ),
    "hurdle_poisson": Family(
        name="hurdle_poisson",
        location_link=_LOG,
        observe_family="HurdlePoisson",
        extra_observe_args=("zi",),
        aux_params=(
            AuxParam(
                name="zi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
        ),
    ),
    "zero_inflated_beta": Family(
        name="zero_inflated_beta",
        location_link=_LOGIT,
        observe_family="ZeroOneInflatedBeta",
        extra_observe_args=("phi", "zi"),
        aux_params=(
            AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "zero_one_inflated_beta": Family(
        name="zero_one_inflated_beta",
        location_link=_LOGIT,
        observe_family="ZeroOneInflatedBeta",
        extra_observe_args=("phi", "zoi", "coi"),
        aux_params=(
            AuxParam(
                name="phi",
                prior="HalfCauchy(2.0)",
                link=_LOG,
            ),
            AuxParam(
                name="zoi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
            AuxParam(
                name="coi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
        ),
    ),
    "lognormal": Family(
        name="lognormal",
        location_link=_IDENTITY,
        observe_family="LogNormal",
        extra_observe_args=("sigma",),
        aux_params=(AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),),
    ),
    "asym_laplace": Family(
        name="asym_laplace",
        location_link=_IDENTITY,
        observe_family="AsymmetricLaplace",
        extra_observe_args=("sigma", "quantile"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="quantile", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "exgaussian": Family(
        name="exgaussian",
        location_link=_IDENTITY,
        observe_family="ExGaussian",
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="beta", prior="HalfNormal(1.0)", link=_LOG),
        ),
    ),
    "inverse_gaussian": Family(
        name="inverse_gaussian",
        location_link=_LOG,
        observe_family="InverseGaussian",
        extra_observe_args=("shape",),
        aux_params=(AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "exponential": Family(
        name="exponential",
        location_link=_LOG,
        observe_family="Exponential",
    ),
    "geometric": Family(
        name="geometric",
        location_link=_LOG,
        observe_family="Geometric",
    ),
    "weibull": Family(
        name="weibull",
        location_link=_LOG,
        observe_family="Weibull",
        extra_observe_args=("shape",),
        aux_params=(AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "von_mises": Family(
        name="von_mises",
        location_link=links["tan_half"],
        observe_family="VonMises",
        extra_observe_args=("kappa",),
        aux_params=(AuxParam(name="kappa", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "mixture": Family(
        name="mixture",
        location_link=_IDENTITY,
        observe_family="MixtureNormal",
    ),
}
"""The built-in response families, keyed by brms-style family name.

The compiler dispatches on the family name and slots each family's
observe step and link into the emitted ``.qvr`` source.
"""


__all__ = [
    "Link",
    "AuxParam",
    "Family",
    "links",
    "families",
]
