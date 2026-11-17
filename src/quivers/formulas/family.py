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
from quivers.formulas._custom_types import CustomFamilySpec
from quivers.formulas._link_whitelist import check_family_links
from quivers.formulas._family_registry import FamilyRegistry
from quivers._values import OPAQUE_BY_VALUE, ValueModel


class Binding(dx.Model):
    """One named quantity defined by a QVR expression over earlier names.

    Attributes
    ----------
    name : str
        Name bound in the emitted program.
    expr : str
        Deterministic QVR expression defining it.
    """

    name: str
    expr: str


class Link(dx.Model):
    """Inverse-link function bridging a linear predictor on
    :math:`\\mathbb{R}` to the family's natural parameter space.

    Attributes
    ----------
    name : str
        Canonical name (e.g. ``"identity"``, ``"logit"``, ``"log"``,
        ``"probit"``, ``"cloglog"``).
    inverse_expr : str
        QVR expression giving the family parameter from the linear
        predictor, written ``{eta}``. The compiler emits exactly this
        expression. ``{name}`` is the parameter being defined, for
        referring to `intermediates`.
    intermediates : tuple[Binding, ...]
        Shared quantities bound, in order, before the inverse; each name
        and expression may use ``{eta}`` and ``{name}``.
    """

    name: str
    inverse_expr: str
    intermediates: tuple[Binding, ...] = ()


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


# mixtures: brms mixture() components and ordering (quivers.formulas._mixture).
class MixtureSpec(ValueModel):
    """Retain the components and identification of a finite mixture family.

    Parameters
    ----------
    components : tuple[Family, ...]
        Component families in brms order; ``nmix`` repetitions are expanded.
    order : {"mu", "none"}
        ``"mu"`` orders the component location intercepts; ``"none"``
        applies no ordering constraint.
    equal : tuple[tuple[str, str], ...]
        Distributional parameters set equal to another parameter of the same
        class, brms ``sigma2 = "sigma1"``, as (target, source) pairs.
    """

    components: tuple[Family, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    order: Literal["mu", "none"] = "none"
    equal: tuple[tuple[str, str], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


class Family(ValueModel):
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

    The native constructor rarely shares the formula parameterization (a
    mean and a shape against a concentration and a rate). ``observe_lets``
    and ``observe_args`` state that map as data, so the compiler emits any
    family the same way.

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
    observe_lets : tuple[Binding, ...]
        Quantities derived from ``mu`` and the auxiliary parameters, bound in
        order before the observation: the map from the formula
        parameterization to the native constructor's.
    observe_args : tuple[str or float, ...]
        Native constructor arguments in schema order: names of formula
        parameters or of `observe_lets`, literal constants, or ``"trials"``
        for the per-row trial counts of a family with `trials_parameter`.
    decision_event : str or None
        Expression forming the scored event from ``formula_response`` and
        the binary ``formula_decision`` of a ``dec()`` term. None for a
        family whose event is its response.
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
    custom : CustomFamilySpec or None
        Checked custom constructor bindings and compilation-local native law.
    mixture : MixtureSpec or None
        Components and ordering of a finite mixture built by
        `quivers.formulas.mixture`.
    """

    name: str
    location_link: Link
    observe_family: str
    observe_lets: tuple[Binding, ...] = ()
    observe_args: tuple[str | float, ...] = ("mu",)
    decision_event: str | None = None
    aux_params: tuple[AuxParam, ...] = ()
    trials_parameter: str | None = None
    ordinal: OrdinalSpec | None = None
    response: ResponseSpec | None = None
    baseline: BaselineSpec | None = None
    custom: CustomFamilySpec | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    mixture: MixtureSpec | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

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

    def response_ordinal(self) -> OrdinalSpec | None:
        """Read the ordinal construction that codes the response categories.

        Returns
        -------
        OrdinalSpec or None
            The family's own ordinal specification, that of the first
            component for a mixture of ordinal families, or None for a
            response without ordered categories.
        """
        if self.mixture is not None:
            return self.mixture.components[0].ordinal
        return self.ordinal

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
            If a parameter or link is unknown, or if brms does not admit the
            link for that family parameter; the error lists the admissible
            links. Categorical models require their multinomial softmax
            location link.
        """
        if self.mixture is not None:
            raise ValueError(
                "mixture links belong to the components; set them before mixture()"
            )
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
                    f"link {requested!r} is not supported for parameter 'mu' of "
                    f"family {self.name!r}; admissible links are {sorted(allowed)}"
                )
            ordinal = ordinal.with_(link=requested)
        if self.custom is None:
            # Ordinal families check the link their thresholds will use.
            effective = dict(parameters)
            if ordinal is not None and "mu" in parameters:
                effective["mu"] = ordinal.link
            check_family_links(self.name, effective)
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
            observe_lets=self.observe_lets,
            observe_args=self.observe_args,
            decision_event=self.decision_event,
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


_SOFTPLUS_EXPR = (
    "select({eta} >= 0.0, "
    "{eta} + log1p(exp(-1.0 * select({eta} >= 0.0, {eta}, 0.0))), "
    "log1p(exp(select({eta} >= 0.0, 0.0, {eta}))))"
)
"""Softplus with each branch evaluated only on its own half-line."""


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
        inverse_expr="sigmoid(0.07056 * ({eta} * ({eta} * {eta})) + 1.5976 * {eta})",
    ),
    "log1p": Link(name="log1p", inverse_expr="expm1({eta})"),
    "cloglog": Link(name="cloglog", inverse_expr="-1.0 * expm1(-1.0 * exp({eta}))"),
    "cauchit": Link(
        name="cauchit", inverse_expr="0.5 + atan({eta}) / 3.141592653589793"
    ),
    "sqrt": Link(name="sqrt", inverse_expr="{eta} * {eta}"),
    "inverse_square": Link(name="inverse_square", inverse_expr="1.0 / sqrt({eta})"),
    "logm1": Link(name="logm1", inverse_expr="1.0 + exp({eta})"),
    "softplus": Link(name="softplus", inverse_expr=_SOFTPLUS_EXPR),
    # The scaled root keeps the square finite for large predictors; select
    # (not abs) retains the right branch's derivative at zero.
    "squareplus": Link(
        name="squareplus",
        intermediates=(
            Binding(name="{name}_link_half", expr="0.5 * {eta}"),
            Binding(name="{name}_link_scale", expr="max(1.0, abs({name}_link_half))"),
            Binding(
                name="{name}_link_root",
                expr=(
                    "{name}_link_scale * sqrt("
                    "{name}_link_half / {name}_link_scale"
                    " * ({name}_link_half / {name}_link_scale)"
                    " + 1.0 / {name}_link_scale * (1.0 / {name}_link_scale))"
                ),
            ),
            Binding(
                name="{name}_link_magnitude",
                expr=(
                    "{name}_link_root + select({name}_link_half >= 0.0, "
                    "{name}_link_half, -1.0 * {name}_link_half)"
                ),
            ),
        ),
        inverse_expr=(
            "select({name}_link_half >= 0.0, {name}_link_magnitude, "
            "1.0 / {name}_link_magnitude)"
        ),
    ),
    "softit": Link(
        name="softit",
        intermediates=(Binding(name="{name}_link_positive", expr=_SOFTPLUS_EXPR),),
        inverse_expr="{name}_link_positive / (1.0 + {name}_link_positive)",
    ),
    "tan_half": Link(name="tan_half", inverse_expr="2.0 * atan({eta})"),
}
"""The built-in inverse links, keyed by link name."""


families: Mapping[str, Family] = FamilyRegistry({
    "wiener": Family(
        name="wiener",
        location_link=_IDENTITY,
        observe_family="Wiener",
        decision_event="formula_response * (2.0 * real(formula_decision) - 1.0)",
        observe_args=("bs", "ndt", "bias", "mu"),
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
        observe_args=("mu", "shape"),
        aux_params=(AuxParam(name="shape", prior="Gamma(0.01, 0.01)", link=_LOG),),
    ),
    "discrete_weibull": Family(
        name="discrete_weibull",
        location_link=_LOGIT,
        observe_family="DiscreteWeibull",
        observe_args=("mu", "shape"),
        aux_params=(AuxParam(name="shape", prior="HalfCauchy(2.0)", link=_LOG),),
    ),
    "zero_inflated_asym_laplace": Family(
        name="zero_inflated_asym_laplace",
        location_link=_IDENTITY,
        observe_family="ZeroInflatedAsymmetricLaplace",
        observe_args=("zi", "mu", "sigma", "quantile"),
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
        observe_lets=(
            Binding(name="negative_binomial_shape", expr="1.0 / sigma"),
            Binding(name="negative_binomial_probs", expr="mu / (mu + negative_binomial_shape)"),
        ),
        observe_args=("negative_binomial_shape", "negative_binomial_probs"),
        aux_params=(AuxParam(name="sigma", prior="Gamma(0.4, 0.3)", link=_LOG),),
    ),
    "skew_normal": Family(
        name="skew_normal",
        location_link=_IDENTITY,
        observe_family="SkewNormal",
        observe_lets=(
            Binding(name="skew_delta", expr="sin(atan(alpha))"),
            Binding(name="skew_center", expr="skew_delta * 0.7978845608028654"),
            Binding(name="skew_scale", expr="sigma / sqrt(1.0 - skew_center * skew_center)"),
            Binding(name="skew_location", expr="mu - skew_scale * skew_center"),
        ),
        observe_args=("skew_location", "skew_scale", "alpha"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="alpha", prior="Normal(0.0, 4.0)", link=_IDENTITY),
        ),
    ),
    "shifted_lognormal": Family(
        name="shifted_lognormal",
        location_link=_IDENTITY,
        observe_family="ShiftedLogNormal",
        observe_args=("mu", "sigma", "ndt"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="ndt", prior="Uniform(0.0, response_min)", link=_LOG),
        ),
    ),
    "frechet": Family(
        name="frechet",
        location_link=_LOG,
        observe_family="Frechet",
        observe_lets=(
            Binding(name="frechet_scale", expr="mu * exp(-1.0 * lgamma(1.0 - 1.0 / nu))"),
        ),
        observe_args=("frechet_scale", "nu"),
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
        observe_args=("mu", "sigma", "xi"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="xi", prior="Normal(0.0, 0.3)", link=_IDENTITY),
        ),
    ),
    "gaussian": Family(
        name="gaussian",
        location_link=_IDENTITY,
        observe_family="Normal",
        observe_args=("mu", "sigma"),
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
        observe_args=("trials", "mu"),
        trials_parameter="total_count",
    ),
    "beta_binomial": Family(
        name="beta_binomial",
        location_link=_LOGIT,
        observe_family="BetaBinomial",
        observe_lets=(
            Binding(name="beta_binomial_concentration1", expr="mu * phi"),
            Binding(name="beta_binomial_concentration0", expr="(1.0 - mu) * phi"),
        ),
        observe_args=("trials", "beta_binomial_concentration1", "beta_binomial_concentration0"),
        trials_parameter="total_count",
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
        observe_args=("trials", "mu"),
        trials_parameter="total_count",
        response=ResponseSpec(kind="multinomial"),
    ),
    "dirichlet_multinomial": Family(
        name="dirichlet_multinomial",
        location_link=_SOFTMAX,
        observe_family="DirichletMultinomial",
        observe_args=("trials", "mu"),
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
        observe_lets=(
            Binding(name="negative_binomial_probs", expr="mu / (mu + disp)"),
        ),
        observe_args=("disp", "negative_binomial_probs"),
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
        observe_lets=(
            Binding(name="gamma_rate", expr="shape / mu"),
        ),
        observe_args=("shape", "gamma_rate"),
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
        observe_lets=(
            Binding(name="beta_concentration1", expr="mu * phi"),
            Binding(name="beta_concentration0", expr="(1.0 - mu) * phi"),
        ),
        observe_args=("beta_concentration1", "beta_concentration0"),
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
        observe_args=("nu", "mu", "sigma"),
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
        observe_args=("zi", "mu"),
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
        observe_args=("hu", "mu"),
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
        observe_lets=(
            Binding(name="negative_binomial_probs", expr="mu / (mu + shape)"),
        ),
        observe_args=("zi", "shape", "negative_binomial_probs"),
        aux_params=(
            AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "zero_inflated_binomial": Family(
        name="zero_inflated_binomial",
        location_link=_LOGIT,
        observe_family="ZeroInflatedBinomial",
        observe_args=("zi", "trials", "mu"),
        trials_parameter="total_count",
        aux_params=(AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT),),
    ),
    "zero_inflated_beta_binomial": Family(
        name="zero_inflated_beta_binomial",
        location_link=_LOGIT,
        observe_family="ZeroInflatedBetaBinomial",
        observe_lets=(
            Binding(name="beta_binomial_concentration1", expr="mu * phi"),
            Binding(name="beta_binomial_concentration0", expr="(1.0 - mu) * phi"),
        ),
        observe_args=("zi", "trials", "beta_binomial_concentration1", "beta_binomial_concentration0"),
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
        observe_lets=(
            Binding(name="negative_binomial_probs", expr="mu / (mu + shape)"),
        ),
        observe_args=("hu", "shape", "negative_binomial_probs"),
        aux_params=(
            AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "hurdle_gamma": Family(
        name="hurdle_gamma",
        location_link=_LOG,
        observe_family="HurdleGamma",
        observe_lets=(
            Binding(name="hurdle_gamma_rate", expr="shape / mu"),
        ),
        observe_args=("hu", "shape", "hurdle_gamma_rate"),
        aux_params=(
            AuxParam(name="shape", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "hurdle_lognormal": Family(
        name="hurdle_lognormal",
        location_link=_IDENTITY,
        observe_family="HurdleLogNormal",
        observe_args=("hu", "mu", "sigma"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "zero_inflated_beta": Family(
        name="zero_inflated_beta",
        location_link=_LOGIT,
        observe_family="ZeroOneInflatedBeta",
        observe_args=("mu", "phi", "zi", 0.0),
        aux_params=(
            AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "xbeta": Family(
        name="xbeta",
        location_link=_LOGIT,
        observe_family="ExtendedBeta",
        observe_args=("mu", "phi", "kappa"),
        aux_params=(
            AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="kappa", prior="Gamma(0.01, 0.01)", link=_LOG),
        ),
    ),
    "zero_one_inflated_beta": Family(
        name="zero_one_inflated_beta",
        location_link=_LOGIT,
        observe_family="ZeroOneInflatedBeta",
        observe_args=("mu", "phi", "zoi", "coi"),
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
        observe_args=("mu", "sigma"),
        aux_params=(AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),),
    ),
    "asym_laplace": Family(
        name="asym_laplace",
        location_link=_IDENTITY,
        observe_family="AsymmetricLaplace",
        observe_args=("mu", "sigma", "quantile"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="quantile", prior="Beta(2.0, 2.0)", link=_LOGIT),
        ),
    ),
    "exgaussian": Family(
        name="exgaussian",
        location_link=_IDENTITY,
        observe_family="ExGaussian",
        observe_lets=(
            Binding(name="exgaussian_location", expr="mu - beta"),
            Binding(name="exgaussian_rate", expr="1.0 / beta"),
        ),
        observe_args=("exgaussian_location", "sigma", "exgaussian_rate"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="beta", prior="HalfNormal(1.0)", link=_LOG),
        ),
    ),
    "inverse_gaussian": Family(
        name="inverse_gaussian",
        location_link=_LOG,
        observe_family="InverseGaussian",
        observe_args=("mu", "shape"),
        aux_params=(AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "exponential": Family(
        name="exponential",
        location_link=_LOG,
        observe_family="Exponential",
        observe_lets=(
            Binding(name="exponential_rate", expr="1.0 / mu"),
        ),
        observe_args=("exponential_rate",),
    ),
    "geometric": Family(
        name="geometric",
        location_link=_LOG,
        observe_family="Geometric",
        observe_lets=(
            Binding(name="geometric_probs", expr="1.0 / (1.0 + mu)"),
        ),
        observe_args=("geometric_probs",),
    ),
    "weibull": Family(
        name="weibull",
        location_link=_LOG,
        observe_family="Weibull",
        observe_lets=(
            Binding(name="weibull_scale", expr="mu * exp(-1.0 * lgamma(1.0 + 1.0 / shape))"),
        ),
        observe_args=("weibull_scale", "shape"),
        aux_params=(AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "von_mises": Family(
        name="von_mises",
        location_link=links["tan_half"],
        observe_family="VonMises",
        observe_args=("mu", "kappa"),
        aux_params=(AuxParam(name="kappa", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "cox": Family(
        name="cox",
        location_link=_LOG,
        observe_family="BernsteinHazard",
        baseline=BaselineSpec(),
    ),
    # Expanded by quivers.formulas._mixture to mixture(gaussian, ...).
    "mixture": Family(
        name="mixture",
        location_link=_IDENTITY,
        observe_family="Mixture",
    ),
})  # fmt: skip
"""The built-in response families, keyed by brms-style family name.

Each family states its link and its map to the native constructor; the
compiler emits them without consulting the family name. Lookup also
accepts the brms spellings of a name (``negbinomial``, ``student``,
``Gamma``, ``inverse.gaussian``, ``zi_poisson``), each resolving to the
one registered family.
"""


__all__ = [
    "Binding",
    "Link",
    "AuxParam",
    "OrdinalSpec",
    "ResponseSpec",
    "BaselineSpec",
    "MixtureSpec",
    "Family",
    "links",
    "families",
]
