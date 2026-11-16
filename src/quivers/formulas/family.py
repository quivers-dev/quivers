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

from typing import Literal, Mapping

import didactic.api as dx

from quivers._splines import MSplineBasis
from quivers.formulas._survival import BaselineSpec


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


class OrdinalSpec(dx.Model):
    """Static ordinal likelihood and threshold configuration.

    Attributes
    ----------
    kind : str
        Cumulative, adjacent-category, stopping-ratio, or continuation-ratio
        construction. The hurdle variant adds a separate lowest category zero.
    link : str
        Inverse link for cumulative or transition probabilities.
    thresholds : str
        Flexible, equally spaced, or explicitly sum-centered thresholds.
    """

    kind: Literal["cumulative", "acat", "sratio", "cratio", "hurdle_cumulative"]
    link: Literal[
        "logit", "probit", "probit_approx", "cloglog", "cauchit", "softit"
    ] = "logit"
    thresholds: Literal["flexible", "equidistant", "sum_to_zero"] = "flexible"


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
    fixed_default : float or None
        Optional constant used unless the parameter is explicitly regressed,
        fixed, or given a prior. Ordinal discrimination defaults to one.
    """

    name: str
    prior: str
    link: Link
    fixed_default: float | None = None


class ResponseSpec(dx.Model):
    """Retain the component coordinate system of a categorical or vector law.

    Parameters
    ----------
    kind : str
        Native response construction using the shared component predictor.
    reference : int or str or None
        Zero-predictor component by position or retained label. None gives
        every component its own predictor under proper coefficient priors.
    """

    kind: Literal[
        "categorical",
        "multinomial",
        "dirichlet_multinomial",
        "dirichlet",
        "dirichlet2",
        "logistic_normal",
    ]
    reference: int | str | None = 0

    def reference_index(self, labels: tuple[str, ...]) -> int | None:
        """Resolve a reference against retained component or factor labels.

        Parameters
        ----------
        labels : tuple[str, ...]
            Complete component order.

        Returns
        -------
        int or None
            Reference position, or None for all-component predictors.

        Raises
        ------
        ValueError
            If the reference is missing, invalid, or incompatible with ALR.
        """
        reference = self.reference
        if reference is None:
            if self.kind == "logistic_normal":
                raise ValueError(
                    "logistic_normal requires an explicit reference component"
                )
            return None
        if isinstance(reference, str):
            if reference not in labels:
                raise ValueError(
                    f"unknown reference component {reference!r}; choices are {labels}"
                )
            return labels.index(reference)
        if (
            isinstance(reference, bool)
            or not isinstance(reference, int)
            or not 0 <= reference < len(labels)
        ):
            raise ValueError("reference component must identify a retained component")
        return reference

    def coordinate_indices(self, labels: tuple[str, ...]) -> tuple[int, ...]:
        """Return coefficient coordinates in retained component order.

        Parameters
        ----------
        labels : tuple[str, ...]
            Complete component order.

        Returns
        -------
        tuple[int, ...]
            Component positions carrying free predictors.
        """
        reference = self.reference_index(labels)
        return tuple(index for index in range(len(labels)) if index != reference)


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

    ``trials_parameter`` identifies families that require a constant or
    per-row number of trials. It names the native constructor parameter;
    fitting and prediction use the same metadata to validate trial counts.
    ``ordinal`` retains static likelihood, inverse-link, and threshold choices;
    ordinary QVR expressions lower these choices to checked category logits.
    ``response`` retains joint component predictors, reference coordinates,
    and the distinction between scalar categorical and vector response events.

    Parameters
    ----------
    name : str
        Formula registry key.
    location_link : Link
        Location or mean inverse link.
    observe_family : str
        Checked native observation family.
    extra_observe_args : tuple[str, ...]
        Additional canonical constructor arguments.
    aux_params : tuple[AuxParam, ...]
        Auxiliary priors and distributional links.
    trials_parameter : str or None
        Native parameter receiving exact per-row trial counts.
    ordinal : OrdinalSpec or None
        Retained ordinal thresholds and transition construction.
    response : ResponseSpec or None
        Joint component coordinate system and reference selection.
    baseline : BaselineSpec or None
        Retained normalized spline structure for proportional hazards.
    """

    name: str
    location_link: Link
    observe_family: str
    extra_observe_args: tuple[str, ...] = ()
    aux_params: tuple[AuxParam, ...] = ()
    trials_parameter: str | None = None
    ordinal: OrdinalSpec | None = None
    response: ResponseSpec | None = None
    baseline: BaselineSpec | None = None

    def with_baseline(
        self,
        basis: MSplineBasis | None = None,
        *,
        df: int = 5,
        degree: int = 3,
        by: str | None = None,
    ) -> Family:
        """Configure a retained normalized baseline for proportional hazards.

        Parameters
        ----------
        basis : MSplineBasis or None
            Explicit immutable knots and tail policy, or automatic training knots.
        df : int
            Number of basis components for automatic selection.
        degree : int
            Polynomial degree for automatic selection.
        by : str or None
            Grouping factor with a separate normalized baseline simplex per level.
            Every group retains the same knot structure and endpoint tail policy.

        Returns
        -------
        Family
            Independent family retaining the selected baseline structure.

        Raises
        ------
        ValueError
            If this is not a baseline-hazard family or automatic dimensions are invalid.
        """
        if self.baseline is None:
            raise ValueError(
                "baseline configuration requires a proportional-hazards family"
            )
        if basis is not None and not isinstance(basis, MSplineBasis):
            raise ValueError("basis must be an immutable MSplineBasis")
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValueError("baseline degree must be a nonnegative integer")
        if isinstance(df, bool) or not isinstance(df, int) or df < degree + 1:
            raise ValueError("baseline df must be an integer at least degree + 1")
        if by is not None and (not isinstance(by, str) or not by):
            raise ValueError("baseline by must be a nonempty grouping column name")
        return self.with_(
            baseline=BaselineSpec(
                df=basis.size if basis is not None else df,
                degree=basis.degree if basis is not None else degree,
                boundary_knots=basis.boundary_knots if basis is not None else None,
                interior_knots=basis.interior_knots if basis is not None else (),
                by=by,
            )
        )

    def with_reference(self, reference: int | str | None) -> Family:
        """Select the reference component of a joint categorical predictor.

        Parameters
        ----------
        reference : int or str or None
            Component position or label; None estimates every component.

        Returns
        -------
        Family
            Independent family retaining the selected coordinate system.

        Raises
        ------
        ValueError
            If this family has no component predictor or forbids references.
        """
        if self.response is None:
            raise ValueError(
                "reference components require a categorical or vector family"
            )
        if self.response.kind == "dirichlet2" and reference is not None:
            raise ValueError(
                "dirichlet2 estimates every concentration and has no reference"
            )
        if self.response.kind == "logistic_normal" and reference is None:
            raise ValueError("logistic_normal requires an explicit reference component")
        return self.with_(response=self.response.with_(reference=reference))

    def with_thresholds(
        self, thresholds: Literal["flexible", "equidistant", "sum_to_zero"]
    ) -> Family:
        """Return an ordinal family with the requested threshold structure.

        Parameters
        ----------
        thresholds : {"flexible", "equidistant", "sum_to_zero"}
            Threshold configuration; ordering applies only to cumulative models.

        Returns
        -------
        Family
            Independent family value preserving links and auxiliary parameters.

        Raises
        ------
        ValueError
            If the family is not ordinal or the threshold option is unknown.
        """
        if self.ordinal is None:
            raise ValueError("threshold configuration requires an ordinal family")
        if thresholds not in {"flexible", "equidistant", "sum_to_zero"}:
            raise ValueError(f"unknown ordinal threshold configuration {thresholds!r}")
        return self.with_(ordinal=self.ordinal.with_(thresholds=thresholds))

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
        if self.response is not None and self.response.kind in {
            "categorical",
            "multinomial",
            "dirichlet_multinomial",
            "dirichlet",
        }:
            if parameters.get("mu") == "logit":
                parameters["mu"] = "softmax"
            if parameters.get("mu", "softmax") != "softmax":
                raise ValueError(
                    "compositional means require the softmax location link"
                )
        if (
            self.response is not None
            and self.response.kind == "logistic_normal"
            and parameters.get("mu", "identity") != "identity"
        ):
            raise ValueError(
                "logistic_normal log ratios require the identity location link"
            )
        invalid = set(parameters.values()) - links.keys()
        if invalid:
            raise ValueError(
                f"unknown links {sorted(invalid)}; choices are {sorted(links)}"
            )
        if self.name == "categorical" and parameters.get("mu", "softmax") != "softmax":
            raise ValueError("categorical models require the softmax location link")
        ordinal = self.ordinal
        if ordinal is not None and "mu" in parameters:
            requested = parameters["mu"]
            if requested == "identity" and self.name == "cumulative":
                requested = "logit"
            allowed = {
                "logit",
                "probit",
                "probit_approx",
                "cloglog",
                "cauchit",
                "softit",
            }
            if ordinal.kind in {"sratio", "cratio"}:
                allowed.remove("softit")
            if requested not in allowed:
                raise ValueError(
                    f"{ordinal.kind} ordinal models support links {sorted(allowed)}"
                )
            ordinal = ordinal.with_(link=requested)
        location_link = (
            links[parameters["mu"]] if "mu" in parameters else self.location_link
        )
        return Family(
            name=self.name,
            location_link=location_link,
            observe_family=(
                "OrderedProbit" if ordinal.link == "probit" else "OrderedLogistic"
            )
            if ordinal is not None
            and ordinal.kind == "cumulative"
            and ordinal.link in {"logit", "probit"}
            else "Categorical"
            if ordinal is not None
            else self.observe_family,
            extra_observe_args=self.extra_observe_args,
            trials_parameter=self.trials_parameter,
            ordinal=ordinal,
            response=self.response,
            baseline=self.baseline,
            aux_params=tuple(
                AuxParam(
                    name=aux.name,
                    prior=aux.prior,
                    link=links[parameters[aux.name]]
                    if aux.name in parameters
                    else aux.link,
                    fixed_default=aux.fixed_default,
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
    "squareplus": Link(
        name="squareplus", inverse_expr="({eta} + sqrt(({eta}) ** 2.0 + 4.0)) / 2.0"
    ),
    "softit": Link(
        name="softit", inverse_expr="log1p(exp({eta})) / (1.0 + log1p(exp({eta})))"
    ),
    "tan_half": Link(name="tan_half", inverse_expr="2.0 * atan({eta})"),
}
"""The built-in inverse links, keyed by link name."""


families: Mapping[str, Family] = {
    "wiener": Family(
        name="wiener",
        location_link=_IDENTITY,
        observe_family="Wiener",
        aux_params=(
            AuxParam(name="bs", prior="Gamma(1.0, 1.0)", link=_LOG),
            AuxParam(name="ndt", prior="LogNormal(-2.0, 0.5)", link=_LOG),
            AuxParam(name="bias", prior="Beta(1.0, 1.0)", link=_LOGIT),
        ),
    ),
    "com_poisson": Family(
        name="com_poisson",
        location_link=_LOG,
        observe_family="COMPoisson",
        extra_observe_args=("shape",),
        aux_params=(AuxParam(name="shape", prior="Gamma(0.01, 0.01)", link=_LOG),),
    ),
    "discrete_weibull": Family(
        name="discrete_weibull",
        location_link=_LOGIT,
        observe_family="DiscreteWeibull",
        extra_observe_args=("shape",),
        aux_params=(AuxParam(name="shape", prior="HalfCauchy(2.0)", link=_LOG),),
    ),
    "zero_inflated_asym_laplace": Family(
        name="zero_inflated_asym_laplace",
        location_link=_IDENTITY,
        observe_family="ZeroInflatedAsymmetricLaplace",
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="quantile", prior="Beta(2.0, 2.0)", link=_LOGIT),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "negbinomial2": Family(
        name="negbinomial2",
        location_link=_LOG,
        observe_family="NegativeBinomial",
        aux_params=(AuxParam(name="sigma", prior="Gamma(0.4, 0.3)", link=_LOG),),
    ),
    "skew_normal": Family(
        name="skew_normal",
        location_link=_IDENTITY,
        observe_family="SkewNormal",
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="alpha", prior="Normal(0.0, 4.0)", link=_IDENTITY),
        ),
    ),
    "shifted_lognormal": Family(
        name="shifted_lognormal",
        location_link=_IDENTITY,
        observe_family="ShiftedLogNormal",
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="ndt", prior="Uniform(0.0, response_min)", link=_LOG),
        ),
    ),
    "frechet": Family(
        name="frechet",
        location_link=_LOG,
        observe_family="Frechet",
        aux_params=(
            AuxParam(
                name="nu",
                prior="Restrict(base=Gamma(concentration=2.0, rate=0.1), low=1.0)",
                link=links["logm1"],
            ),
        ),
    ),
    "gen_extreme_value": Family(
        name="gen_extreme_value",
        location_link=_IDENTITY,
        observe_family="GeneralizedExtremeValue",
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="xi", prior="Normal(0.0, 0.3)", link=_IDENTITY),
        ),
    ),
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
        trials_parameter="total_count",
    ),
    "beta_binomial": Family(
        name="beta_binomial",
        location_link=_LOGIT,
        observe_family="BetaBinomial",
        trials_parameter="total_count",
        extra_observe_args=("phi",),
        aux_params=(AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),),
    ),
    "categorical": Family(
        name="categorical",
        location_link=_SOFTMAX,
        observe_family="Categorical",
        response=ResponseSpec(kind="categorical"),
    ),
    "multinomial": Family(
        name="multinomial",
        location_link=_SOFTMAX,
        observe_family="Multinomial",
        trials_parameter="total_count",
        response=ResponseSpec(kind="multinomial"),
    ),
    "dirichlet_multinomial": Family(
        name="dirichlet_multinomial",
        location_link=_SOFTMAX,
        observe_family="DirichletMultinomial",
        trials_parameter="total_count",
        aux_params=(AuxParam(name="phi", prior="Gamma(2.0, 0.1)", link=_LOG),),
        response=ResponseSpec(kind="dirichlet_multinomial"),
    ),
    "dirichlet": Family(
        name="dirichlet",
        location_link=_SOFTMAX,
        observe_family="Dirichlet",
        aux_params=(AuxParam(name="phi", prior="Gamma(2.0, 0.1)", link=_LOG),),
        response=ResponseSpec(kind="dirichlet"),
    ),
    "dirichlet2": Family(
        name="dirichlet2",
        location_link=_LOG,
        observe_family="Dirichlet",
        response=ResponseSpec(kind="dirichlet2", reference=None),
    ),
    "logistic_normal": Family(
        name="logistic_normal",
        location_link=_IDENTITY,
        observe_family="AdditiveLogRatioNormal",
        aux_params=(AuxParam(name="sigma", prior="HalfNormal(1.0)", link=_LOG),),
        response=ResponseSpec(kind="logistic_normal"),
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
        ordinal=OrdinalSpec(kind="cumulative"),
        aux_params=(
            AuxParam(
                name="disc", prior="LogNormal(0.0, 1.0)", link=_LOG, fixed_default=1.0
            ),
        ),
    ),
    "hurdle_cumulative": Family(
        name="hurdle_cumulative",
        location_link=_LOGIT,
        observe_family="Categorical",
        ordinal=OrdinalSpec(kind="hurdle_cumulative"),
        aux_params=(
            AuxParam(
                name="disc", prior="LogNormal(0.0, 1.0)", link=_LOG, fixed_default=1.0
            ),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "acat": Family(
        name="acat",
        location_link=_LOGIT,
        observe_family="Categorical",
        ordinal=OrdinalSpec(kind="acat"),
        aux_params=(
            AuxParam(
                name="disc", prior="LogNormal(0.0, 1.0)", link=_LOG, fixed_default=1.0
            ),
        ),
    ),
    "sratio": Family(
        name="sratio",
        location_link=_LOGIT,
        observe_family="Categorical",
        ordinal=OrdinalSpec(kind="sratio"),
        aux_params=(
            AuxParam(
                name="disc", prior="LogNormal(0.0, 1.0)", link=_LOG, fixed_default=1.0
            ),
        ),
    ),
    "cratio": Family(
        name="cratio",
        location_link=_LOGIT,
        observe_family="Categorical",
        ordinal=OrdinalSpec(kind="cratio"),
        aux_params=(
            AuxParam(
                name="disc", prior="LogNormal(0.0, 1.0)", link=_LOG, fixed_default=1.0
            ),
        ),
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
        extra_observe_args=("hu",),
        aux_params=(
            AuxParam(
                name="hu",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
            ),
        ),
    ),
    "zero_inflated_negbinomial": Family(
        name="zero_inflated_negbinomial",
        location_link=_LOG,
        observe_family="ZeroInflatedNegativeBinomial",
        aux_params=(
            AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "zero_inflated_binomial": Family(
        name="zero_inflated_binomial",
        location_link=_LOGIT,
        observe_family="ZeroInflatedBinomial",
        trials_parameter="total_count",
        aux_params=(AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT),),
    ),
    "zero_inflated_beta_binomial": Family(
        name="zero_inflated_beta_binomial",
        location_link=_LOGIT,
        observe_family="ZeroInflatedBetaBinomial",
        trials_parameter="total_count",
        aux_params=(
            AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "hurdle_negbinomial": Family(
        name="hurdle_negbinomial",
        location_link=_LOG,
        observe_family="HurdleNegativeBinomial",
        aux_params=(
            AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "hurdle_gamma": Family(
        name="hurdle_gamma",
        location_link=_LOG,
        observe_family="HurdleGamma",
        aux_params=(
            AuxParam(name="shape", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "hurdle_lognormal": Family(
        name="hurdle_lognormal",
        location_link=_IDENTITY,
        observe_family="HurdleLogNormal",
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT),
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
    "xbeta": Family(
        name="xbeta",
        location_link=_LOGIT,
        observe_family="ExtendedBeta",
        extra_observe_args=("phi", "kappa"),
        aux_params=(
            AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="kappa", prior="Gamma(0.01, 0.01)", link=_LOG),
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
    "cox": Family(
        name="cox",
        location_link=_LOG,
        observe_family="BernsteinHazard",
        baseline=BaselineSpec(),
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
    "OrdinalSpec",
    "ResponseSpec",
    "BaselineSpec",
    "Family",
    "links",
    "families",
]
