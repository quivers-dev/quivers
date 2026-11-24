"""Response families and link functions for the formula frontend.

Each `Family` knows how to emit a QVR ``observe`` step plus
the inverse-link expression that wraps a linear predictor.  The
registry maps brms-style family names to a typed family record. The
record declares every property the frontend reads: the compiler, the
posterior methods and the response terms consult those declarations and
never the family name, so a new family opts in to whatever it declares.

Each family declares location and auxiliary links separately. The compiler
converts regression parameters to QVR constructor parameters, for instance
Weibull mean/shape to scale/shape. Categorical and finite-mixture families
add compiler-owned component axes because their likelihood parameters are
vectors rather than scalar auxiliaries.
"""

from __future__ import annotations

import math
from typing import Literal, Mapping

import didactic.api as dx
import numpy as np

from quivers._splines import MSplineBasis
from quivers.formulas._survival import BaselineSpec
from quivers.formulas._custom_types import CustomFamilySpec
from quivers.formulas._family_registry import FamilyRegistry
from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.qiec.families import DistributionFamily
from quivers.qiec.families import family as semantic_family


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


class Exposure(dx.Model):
    """The law of a count accumulated over ``exposure`` unit denominators.

    A family carrying one is closed under summing independent unit-exposure
    counts, which is what a ``rate()`` response term requires.

    Attributes
    ----------
    observe_lets : tuple[Binding, ...]
        Quantities bound after the family's own, written over the formula
        parameters, the family's bindings and ``exposure``.
    observe_args : tuple[str or float, ...]
        Native constructor arguments of the accumulated count; ``"exposure"``
        names the denominator itself.
    observe_family : str or None
        Native law of the accumulated count when it leaves the family's own.
    """

    observe_lets: tuple[Binding, ...] = ()
    observe_args: tuple[str | float, ...]
    observe_family: str | None = None


class Domain(dx.Model):
    """The interval of values a family parameter or response may take.

    Attributes
    ----------
    lower : float or None
        Lower endpoint, or None when unbounded below.
    upper : float or None
        Upper endpoint, or None when unbounded above.
    lower_closed : bool
        Whether the lower endpoint belongs to the interval.
    upper_closed : bool
        Whether the upper endpoint belongs to the interval.
    """

    lower: float | None = 0.0
    upper: float | None = None
    lower_closed: bool = False
    upper_closed: bool = False

    def contains(self, value: float) -> bool:
        """Test membership of one finite value.

        Parameters
        ----------
        value : float
            Candidate value.

        Returns
        -------
        bool
            True when the value lies in the interval.
        """
        above = self.lower is None or (
            value >= self.lower if self.lower_closed else value > self.lower
        )
        below = self.upper is None or (
            value <= self.upper if self.upper_closed else value < self.upper
        )
        return above and below

    def within(self, other: Domain) -> bool:
        """Test whether this interval is contained in another.

        Parameters
        ----------
        other : Domain
            Candidate enclosing interval.

        Returns
        -------
        bool
            True when every value of this interval lies in ``other``.
        """
        above = other.lower is None or (
            self.lower is not None
            and (
                self.lower > other.lower
                or self.lower == other.lower
                and (other.lower_closed or not self.lower_closed)
            )
        )
        below = other.upper is None or (
            self.upper is not None
            and (
                self.upper < other.upper
                or self.upper == other.upper
                and (other.upper_closed or not self.upper_closed)
            )
        )
        return above and below

    def requirement(self) -> str:
        """Describe the interval as the completion of "must ...".

        Returns
        -------
        str
            A verb phrase stating the constraint.
        """
        if self.lower is None and self.upper is None:
            return "be a real number"
        if self.upper is None:
            if self.lower == 0:
                return "be nonnegative" if self.lower_closed else "be strictly positive"
            return (
                f"be at least {self.lower:g}"
                if self.lower_closed
                else f"exceed {self.lower:g}"
            )
        if self.lower is None:
            return (
                f"be at most {self.upper:g}"
                if self.upper_closed
                else f"be below {self.upper:g}"
            )
        if not self.lower_closed and not self.upper_closed:
            return f"be strictly between {self.lower:g} and {self.upper:g}"
        left = "[" if self.lower_closed else "("
        right = "]" if self.upper_closed else ")"
        return f"be in {left}{self.lower:g}, {self.upper:g}{right}"


RESPONSE_MIN = "response_min"
"""Name of the smallest response in the prior of a shift below it."""

_POSITIVE_PARAMETER_LINKS = ("log", "identity", "softplus", "squareplus")
"""brms's links of an auxiliary parameter on the positive half-line."""

_PROBABILITY_PARAMETER_LINKS = ("logit", "identity")
"""brms's links of an auxiliary parameter on the unit interval."""

_REAL = Domain(lower=None)
_NONNEGATIVE = Domain(lower_closed=True)
_UNIT_OPEN = Domain(upper=1.0)
_UNIT_CLOSED = Domain(upper=1.0, lower_closed=True, upper_closed=True)

_SUPPORT_DOMAINS: Mapping[str, Domain] = {
    "real": _REAL,
    "positive": Domain(),
    "nonnegative": _NONNEGATIVE,
    "unit_interval": _UNIT_OPEN,
}
"""The interval of each parameter-free continuous support kind."""


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
    image : Domain or None
        Interval the inverse maps the real line onto, coordinate by
        coordinate for a vector link. None when it is not declared, which
        a parameter whose domain has a lower endpoint above zero refuses.
    brms_intercept_prior : str or None
        brms's default prior of a centered location intercept under this
        link, where it is not the response-scaled Student-t.
    brms_response_expr : str or None
        The link applied to the response ``{y}``, where brms reads the
        scales of its default priors from the linked response; None where
        brms keeps its fixed scale.
    """

    name: str
    inverse_expr: str
    intermediates: tuple[Binding, ...] = ()
    image: Domain | None = None
    brms_intercept_prior: str | None = None
    brms_response_expr: str | None = None

    @property
    def is_identity(self) -> bool:
        """Whether the parameter is its linear predictor.

        Returns
        -------
        bool
            True when the inverse is the identity.
        """
        return self.inverse_expr == "{eta}"

    @property
    def is_log(self) -> bool:
        """Whether the linear predictor is the logarithm of the parameter.

        Returns
        -------
        bool
            True when the inverse is the exponential.
        """
        return self.inverse_expr == "exp({eta})"


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
    brms_class : str or None
        brms parameter class when it differs from `name`.
    brms_prior : str or None
        brms's default prior where it is specific to this family's parameter;
        otherwise the default of the parameter's brms class applies.
    domain : Domain
        Values the parameter may take; the positive half-line by default. A
        fixed value is checked against it. A lower endpoint above zero is
        narrower than the native constraint, so a prior is restricted to it
        and a link must map into it.
    links : tuple[str, ...] or None
        Links `Family.with_links` admits for the parameter, where they are
        not those brms gives every parameter of the domain
        (`admissible_links`).
    below_response : bool
        Whether the parameter is a shift every response exceeds. A fixed
        value is checked against the responses, `prior` and `brms_prior` may
        name the smallest response as ``response_min``, and inference starts
        at a quarter of it unless the prior is already bounded by it.
    per_component : bool
        Whether the parameter has one value per free component of the
        family's `ResponseSpec`.
    """

    name: str
    prior: str
    link: Link
    fixed_default: float | None = None
    domain: Domain = Domain()
    brms_class: str | None = None
    brms_prior: str | None = None
    links: tuple[str, ...] | None = None
    below_response: bool = False
    per_component: bool = False

    @property
    def parameter_class(self) -> str:
        """The brms class of the parameter.

        Returns
        -------
        str
            The declared class, or the parameter's own name.
        """
        return self.name if self.brms_class is None else self.brms_class

    @property
    def admissible_links(self) -> tuple[str, ...] | None:
        """The links brms admits for the parameter.

        Returns
        -------
        tuple[str, ...] or None
            The declared links, or those brms gives every parameter on the
            positive half-line or the unit interval. None for a parameter on
            another interval that declares no links.
        """
        if self.links is not None:
            return self.links
        if self.domain.lower != 0.0:
            return None
        if self.domain.upper is None:
            return _POSITIVE_PARAMETER_LINKS
        return _PROBABILITY_PARAMETER_LINKS if self.domain.upper == 1.0 else None

    @property
    def bounded_by_response(self) -> bool:
        """Whether the default prior is bounded by the smallest response.

        Returns
        -------
        bool
            True when `prior` names ``response_min``.
        """
        return RESPONSE_MIN in self.prior

    def with_response_minimum(self, prior: str, response: np.ndarray) -> str:
        """Bind the smallest response in a prior of this parameter.

        Parameters
        ----------
        prior : str
            Prior expression, possibly naming ``response_min``.
        response : np.ndarray
            Training responses.

        Returns
        -------
        str
            The prior with ``response_min`` replaced by its value.

        Raises
        ------
        ValueError
            If the prior names ``response_min`` for a parameter that does
            not declare `below_response`.
        """
        if RESPONSE_MIN not in prior:
            return prior
        if not self.below_response:
            raise ValueError(
                f"the prior of {self.name!r} names {RESPONSE_MIN}, which only a "
                "parameter declaring AuxParam.below_response may"
            )
        return prior.replace(RESPONSE_MIN, repr(float(np.min(response))))


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


class Elliptical(dx.Model):
    """The elliptical residual law of a location-scale family.

    A family declaring one has the density of a multivariate Normal or
    Student-t in one coordinate, so residuals may be correlated across rows
    (autocorrelation terms) or across responses (``set_rescor``) under the
    multivariate law with the same scale and degrees of freedom.

    Attributes
    ----------
    scale : str
        Auxiliary parameter scaling the residual.
    df : str or None
        Auxiliary parameter giving the Student degrees of freedom; None for
        the Normal law.
    """

    scale: str
    df: str | None = None

    @property
    def kernel(self) -> str:
        """The native law of a correlated residual vector.

        Returns
        -------
        str
            The multivariate Normal, or Student-t under degrees of freedom.
        """
        return "MultivariateNormal" if self.df is None else "MultivariateStudentT"

    @property
    def parameterization(self) -> tuple[str, ...]:
        """The constructor parameters of `kernel` in schema order.

        Returns
        -------
        tuple[str, ...]
            Location and Cholesky scale factor, after the degrees of freedom
            of a Student-t.
        """
        location = ("loc", "scale_tril")
        return location if self.df is None else ("df", *location)


# mixtures: brms mixture() components and ordering (quivers.formulas._mixture).
class MixtureSpec(ValueModel):
    """Retain the components and identification of a finite mixture family.

    Parameters
    ----------
    components : tuple[Family, ...]
        Component families in brms order; ``nmix`` repetitions are expanded.
        One component is a template, repeated ``mixture_components`` times.
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
    exposure : Exposure or None
        Law of the count over a ``rate()`` denominator; None when the family
        is not closed under summing unit-exposure counts.
    log_location_argument : str or None
        Native constructor parameter that accepts the location on the log
        scale. Under a log location link the linear predictor is passed to it
        directly, so a rate too large to exponentiate still scores.
    circular : bool
        Whether the response is an angle, for which an expectation needs an
        angular convention and no arithmetic mean is defined.
    log_response : bool
        Whether the location and scale parameterize the logarithm of the
        response, so default prior scales are taken on the log response.
    response_domain : Domain or None
        Interval every observed response must lie in, stated only where the
        native support depends on parameters or the family narrows it;
        otherwise the native support and its atoms give it.
    decision_event : str or None
        Expression forming the scored event from ``formula_response`` and
        the binary ``formula_decision`` of a ``dec()`` term. None for a
        family whose event is its response.
    location_links : tuple[str, ...] or None
        Links `with_links` admits for the location; None declares no
        decision, and `with_links` refuses a location link.
    recorded_events : bool
        Whether the family admits the ``cens()`` and ``trunc()`` terms.
    error_scale : str or None
        Auxiliary parameter giving the residual scale that the known errors
        of a ``se()`` term replace, or join in quadrature; None when the
        family admits no ``se()``.
    elliptical : Elliptical or None
        Elliptical residual law, under which residuals may be correlated
        across rows or responses.
    r_squared : {"model", "residual"}
        Variance ``bayes_r2`` reads by default: the law's own conditional
        variance or that of the residuals.
    factor_levels : int or None
        Number of levels of a factor response the family codes
        ``0, ..., n - 1``, stated only where the native support depends on
        parameters (`coded_levels`).
    mixture_law : str or None
        Native law of a finite mixture whose components all have this
        family, over the stacked native arguments of the components. The law
        must truncate each component itself. None mixes the component laws.
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
    exposure: Exposure | None = None
    log_location_argument: str | None = None
    circular: bool = False
    log_response: bool = False
    response_domain: Domain | None = None
    decision_event: str | None = None
    location_links: tuple[str, ...] | None = None
    recorded_events: bool = False
    error_scale: str | None = None
    elliptical: Elliptical | None = None
    r_squared: Literal["model", "residual"] = "residual"
    factor_levels: int | None = None
    mixture_law: str | None = None

    @property
    def response_interval(self) -> Domain | None:
        """The interval a scalar continuous response must lie in.

        An atom of the native law at an endpoint of its support closes that
        endpoint.

        Returns
        -------
        Domain or None
            The declared or native interval, or None for a discrete, vector
            or composite response, whose values are checked by their coding.
        """
        if self.response_domain is not None:
            return self.response_domain
        native = self.native
        if native.discrete or native.compositional or native.event_rank:
            return None
        domain = _SUPPORT_DOMAINS.get(str(native.support))
        if domain is None:
            return None
        atoms = native.atom_locations
        return domain.with_(
            lower_closed=domain.lower_closed or atoms is None or domain.lower in atoms,
            upper_closed=domain.upper_closed or atoms is None or domain.upper in atoms,
        )

    @property
    def native(self) -> DistributionFamily:
        """The semantic record of the observed native law.

        Support, element type, discreteness, atoms and event rank are read
        from it; the family does not restate them.

        Returns
        -------
        DistributionFamily
            The record registered under `observe_family`, from the custom
            catalog of a custom family.
        """
        if self.custom is not None:
            return self.custom.catalog[self.observe_family]
        return semantic_family(self.observe_family)

    @property
    def retains_log_location(self) -> bool:
        """Whether the observation takes the linear predictor on the log scale.

        Returns
        -------
        bool
            True when the native law accepts a log-scale location and the
            location link is the logarithm.
        """
        return (
            self.log_location_argument is not None and self.location_link.is_log
        )

    @property
    def vector_response(self) -> bool:
        """Whether one response event is a vector of components.

        Returns
        -------
        bool
            True for compositional and count-vector responses.
        """
        return self.response is not None and not self.unordered_categories

    @property
    def coded_levels(self) -> int | None:
        """Number of levels of a factor the family reads as integer codes.

        Returns
        -------
        int or None
            The declared number, or the size of a native support
            ``0, ..., n - 1``. None for a family without such codes; a
            `category_response` takes any number of levels.
        """
        if self.factor_levels is not None:
            return self.factor_levels
        support = self.native.finite_support
        if support is None or list(support) != list(range(len(support))):
            return None
        return len(support)

    @property
    def scores_cutpoints(self) -> bool:
        """Whether the native law of an ordinal family takes its cutpoints.

        Returns
        -------
        bool
            True for a law over a predictor and cutpoints, False for a law
            over category logits.
        """
        return "cutpoints" in self.native.parameter_names

    @property
    def component_scale(self) -> AuxParam | None:
        """The auxiliary parameter with one value per free component.

        Returns
        -------
        AuxParam or None
            The parameter declaring `AuxParam.per_component`, if any.
        """
        return next((aux for aux in self.aux_params if aux.per_component), None)

    @property
    def residual_scale(self) -> AuxParam | None:
        """The residual scale, the parameter of brms class ``sigma``.

        Returns
        -------
        AuxParam or None
            The parameter of that class, or None.
        """
        return next(
            (aux for aux in self.aux_params if aux.parameter_class == "sigma"), None
        )

    @property
    def latent_response_obstacle(self) -> str | None:
        """Why a missing response cannot be a latent coordinate.

        A missing ``mi()`` response is a latent coordinate drawn from the
        response law, so the law needs a density on an interval and nothing
        fixed by the recorded responses.

        Returns
        -------
        str or None
            The property that rules it out, or None when the family admits
            ``mi()`` responses.
        """
        native = self.native
        if self.response is not None or native.event_rank:
            return "its response is a vector or a category"
        if native.discrete:
            return "its response is discrete"
        if native.compositional:
            return "its response mixes component laws"
        if native.atom_locations is None or native.atom_locations:
            return "its response law has point masses"
        if self.decision_event is not None:
            return "its event combines the response with a decision"
        if self.baseline is not None:
            return "its baseline hazard is placed on the recorded times"
        if any(aux.below_response for aux in self.aux_params):
            return "its shift is bounded by the recorded responses"
        return None

    def native_parameter(self, name: str) -> str:
        """Name the native constructor parameter that receives a quantity.

        Parameters
        ----------
        name : str
            Formula parameter or derived binding passed to the constructor.

        Returns
        -------
        str
            The constructor parameter in the native schema.

        Raises
        ------
        ValueError
            If the quantity is not itself a native constructor argument.
        """
        if self.custom is not None:
            arguments = {bound: native for native, bound in self.custom.bindings}
        else:
            arguments = {
                argument: native
                for argument, native in zip(
                    self.observe_args, self.native.parameter_names, strict=False
                )
                if isinstance(argument, str)
            }
        if name not in arguments:
            raise ValueError(
                f"family {self.name!r} does not pass {name!r} to its native "
                "constructor (Family.observe_args)"
            )
        return arguments[name]

    def admissible_links(self, parameter: str) -> tuple[str, ...]:
        """List the links the family admits for one parameter.

        Parameters
        ----------
        parameter : str
            ``mu`` for the location or an auxiliary parameter name.

        Returns
        -------
        tuple[str, ...]
            Admissible link names.

        Raises
        ------
        ValueError
            If the parameter is unknown or declares no links.
        """
        if parameter == "mu":
            declared, field = self.location_links, "Family.location_links"
        else:
            aux = next((aux for aux in self.aux_params if aux.name == parameter), None)
            if aux is None:
                raise ValueError(
                    f"unknown family parameter {parameter!r} of {self.name!r}"
                )
            declared, field = aux.admissible_links, "AuxParam.links"
        if declared is None:
            raise ValueError(
                f"family {self.name!r} declares no admissible links for parameter "
                f"{parameter!r} ({field})"
            )
        return declared

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

    @dx.model_validator()
    def _validate_declarations(self) -> None:
        """Require declared parameter names to be auxiliary parameters.

        Raises
        ------
        ValueError
            If a declaration names a parameter the family does not have, or a
            per-component parameter has no components.
        """
        names = {aux.name for aux in self.aux_params}
        declared = {"error_scale": self.error_scale}
        if self.elliptical is not None:
            declared |= {
                "elliptical.scale": self.elliptical.scale,
                "elliptical.df": self.elliptical.df,
            }
        for field, name in declared.items():
            if name is not None and name not in names:
                raise ValueError(
                    f"family {self.name!r} declares {field} {name!r}, which is not "
                    f"one of its auxiliary parameters {sorted(names)}"
                )
        if self.response is None and self.component_scale is not None:
            raise ValueError(
                f"family {self.name!r} declares a per-component parameter without "
                "a ResponseSpec"
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

    @property
    def unordered_categories(self) -> bool:
        """Whether the response is an unordered category code.

        Returns
        -------
        bool
            True for a scalar categorical response.
        """
        return self.response is not None and self.response.kind == "categorical"

    @property
    def category_response(self) -> bool:
        """Whether the response is a category code with class probabilities.

        Returns
        -------
        bool
            True for ordinal and unordered categorical responses, and for a
            mixture of ordinal components.
        """
        return self.response_ordinal() is not None or self.unordered_categories

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
        if self.mixture is not None and len(self.mixture.components) > 1:
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
        ordinal = self.ordinal
        observe_family = self.observe_family
        requested = dict(parameters)
        if ordinal is not None and "mu" in parameters:
            # The location of a cumulative family is its own linear predictor;
            # the identity keeps it, with thresholds on the logit scale.
            if parameters["mu"] == "identity" and ordinal.kind == "cumulative":
                requested["mu"] = "logit"
        for parameter, link in requested.items():
            allowed = self.admissible_links(parameter)
            if link not in allowed:
                raise ValueError(
                    f"link {link!r} is not supported for parameter {parameter!r} of "
                    f"family {self.name!r}; admissible links are {list(allowed)}"
                )
        if ordinal is not None and "mu" in requested:
            ordinal = ordinal.with_(link=requested["mu"])
            observe_family = (
                _CUTPOINT_LAWS.get(ordinal.link, "Categorical")
                if ordinal.kind == "cumulative"
                else "Categorical"
            )
        return self.with_(
            location_link=links[parameters["mu"]]
            if "mu" in parameters
            else self.location_link,
            observe_family=observe_family,
            ordinal=ordinal,
            aux_params=tuple(
                aux.with_(link=links[parameters[aux.name]])
                if aux.name in parameters
                else aux
                for aux in self.aux_params
            ),
        )


_CUTPOINT_LAWS: Mapping[str, str] = {
    "logit": "OrderedLogistic",
    "probit": "OrderedProbit",
}
"""Native cumulative laws over cutpoints, by ordinal link."""

_IDENTITY = Link(
    name="identity",
    inverse_expr="{eta}",
    image=_REAL,
    brms_response_expr="{y}",
)
_LOGIT = Link(name="logit", inverse_expr="sigmoid({eta})", image=_UNIT_OPEN)
_LOG = Link(
    name="log",
    inverse_expr="exp({eta})",
    image=Domain(),
    brms_response_expr="log({y})",
)
_SOFTMAX = Link(name="softmax", inverse_expr="softmax({eta})", image=_UNIT_OPEN)
_INVERSE = Link(
    name="inverse",
    inverse_expr="1.0 / {eta}",
    image=_REAL,
    brms_response_expr="1.0 / {y}",
)


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
        name="probit",
        inverse_expr="0.5 * (1.0 + erf({eta} / 1.4142135623730951))",
        image=_UNIT_OPEN,
    ),
    "probit_approx": Link(
        name="probit_approx",
        inverse_expr="sigmoid(0.07056 * ({eta} * ({eta} * {eta})) + 1.5976 * {eta})",
        image=_UNIT_OPEN,
    ),
    "log1p": Link(
        name="log1p", inverse_expr="expm1({eta})", image=Domain(lower=-1.0)
    ),
    "cloglog": Link(
        name="cloglog",
        inverse_expr="-1.0 * expm1(-1.0 * exp({eta}))",
        image=_UNIT_OPEN,
    ),
    "cauchit": Link(
        name="cauchit",
        inverse_expr="0.5 + atan({eta}) / 3.141592653589793",
        image=_UNIT_OPEN,
    ),
    "sqrt": Link(
        name="sqrt",
        inverse_expr="{eta} * {eta}",
        image=_NONNEGATIVE,
        brms_response_expr="sqrt({y})",
    ),
    "inverse_square": Link(
        name="inverse_square",
        inverse_expr="1.0 / sqrt({eta})",
        image=Domain(),
        brms_response_expr="1.0 / ({y} * {y})",
    ),
    "logm1": Link(
        name="logm1", inverse_expr="1.0 + exp({eta})", image=Domain(lower=1.0)
    ),
    "softplus": Link(name="softplus", inverse_expr=_SOFTPLUS_EXPR, image=Domain()),
    # The scaled root keeps the square finite for large predictors; select
    # (not abs) retains the right branch's derivative at zero.
    "squareplus": Link(
        name="squareplus",
        image=Domain(),
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
        image=_UNIT_OPEN,
        intermediates=(Binding(name="{name}_link_positive", expr=_SOFTPLUS_EXPR),),
        inverse_expr="{name}_link_positive / (1.0 + {name}_link_positive)",
    ),
    "tan_half": Link(
        name="tan_half",
        inverse_expr="2.0 * atan({eta})",
        image=Domain(lower=-math.pi, upper=math.pi),
        brms_intercept_prior="StudentT(1.0, 0.0, 1.0)",
    ),
}
"""The built-in inverse links, keyed by link name."""


_REAL_LINKS = (
    "identity",
    "log",
    "inverse",
    "softplus",
    "squareplus",
    "logit",
    "probit",
    "probit_approx",
    "cloglog",
    "cauchit",
    "softit",
)
"""brms's location links of a family whose location is any real number."""

_REAL_EXTENSIONS = ("log1p", "sqrt", "inverse_square", "logm1", "tan_half")
"""The other real-valued links of the registry, admitted beyond brms where
the location is any real number, since each gives a valid location."""

_POSITIVE_LINKS = ("log", "identity", "inverse", "softplus", "squareplus")
_COUNT_LINKS = ("log", "identity", "sqrt", "softplus", "squareplus")
_UNIT_LINKS = (
    "logit",
    "probit",
    "probit_approx",
    "cloglog",
    "cauchit",
    "softit",
    "identity",
    "log",
)
_ORDINAL_LINKS = ("logit", "probit", "probit_approx", "cloglog", "cauchit", "softit")
_RATIO_LINKS = ("logit", "probit", "probit_approx", "cloglog", "cauchit")
_SHIFTED_LINKS = ("identity", "log", "inverse", "softplus", "squareplus")
_LOG_LOCATION_LINKS = ("identity", "inverse")
_COMPOSITIONAL_LINKS = ("logit", "softmax")

_GAUSSIAN = Family(
    name="gaussian",
    location_link=_IDENTITY,
    location_links=_REAL_LINKS + _REAL_EXTENSIONS,
    observe_family="Normal",
    observe_args=("mu", "sigma"),
    recorded_events=True,
    error_scale="sigma",
    elliptical=Elliptical(scale="sigma"),
    r_squared="model",
    mixture_law="MixtureNormal",
    aux_params=(AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),),
)


families: Mapping[str, Family] = FamilyRegistry({
    "wiener": Family(
        name="wiener",
        location_link=_IDENTITY,
        location_links=("identity", "log", "softplus", "squareplus"),
        observe_family="Wiener",
        response_domain=Domain(),
        decision_event="formula_response * (2.0 * real(formula_decision) - 1.0)",
        observe_args=("bs", "ndt", "bias", "mu"),
        aux_params=(
            AuxParam(name="bs", prior="Gamma(1.0, 1.0)", link=_LOG),
            AuxParam(
                name="ndt",
                prior="LogNormal(-2.0, 0.5)",
                link=_LOG,
                domain=_NONNEGATIVE,
                below_response=True,
                brms_prior="Uniform(0.0, response_min)",
            ),
            AuxParam(name="bias", prior="Beta(1.0, 1.0)", link=_LOGIT, domain=_UNIT_OPEN),
        ),
    ),
    "com_poisson": Family(
        name="com_poisson",
        location_link=_LOG,
        location_links=_COUNT_LINKS,
        observe_family="COMPoisson",
        recorded_events=True,
        log_location_argument="log_rate",
        observe_args=("mu", "shape"),
        aux_params=(AuxParam(name="shape", prior="Gamma(0.01, 0.01)", link=_LOG),),
    ),
    "discrete_weibull": Family(
        name="discrete_weibull",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS[:-1],
        observe_family="DiscreteWeibull",
        recorded_events=True,
        observe_args=("mu", "shape"),
        aux_params=(AuxParam(name="shape", prior="HalfCauchy(2.0)", link=_LOG),),
    ),
    "zero_inflated_asym_laplace": Family(
        name="zero_inflated_asym_laplace",
        location_link=_IDENTITY,
        location_links=_SHIFTED_LINKS,
        observe_family="ZeroInflatedAsymmetricLaplace",
        recorded_events=True,
        observe_args=("zi", "mu", "sigma", "quantile"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="quantile", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_OPEN),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),
        ),
    ),
    "negbinomial2": Family(
        name="negbinomial2",
        location_link=_LOG,
        location_links=_COUNT_LINKS,
        observe_family="NegativeBinomial",
        recorded_events=True,
        exposure=Exposure(
            observe_lets=(
                Binding(name="rate_shape", expr="negative_binomial_shape * exposure"),
            ),
            observe_args=("rate_shape", "negative_binomial_probs"),
        ),
        observe_lets=(
            Binding(name="negative_binomial_shape", expr="1.0 / sigma"),
            Binding(name="negative_binomial_probs", expr="mu / (mu + negative_binomial_shape)"),
        ),
        observe_args=("negative_binomial_shape", "negative_binomial_probs"),
        aux_params=(AuxParam(
                name="sigma",
                prior="Gamma(0.4, 0.3)",
                link=_LOG,
                brms_prior="Gamma(0.4, 0.3)",
            ),),
    ),
    "skew_normal": Family(
        name="skew_normal",
        location_link=_IDENTITY,
        location_links=_REAL_LINKS + _REAL_EXTENSIONS,
        observe_family="SkewNormal",
        recorded_events=True,
        error_scale="sigma",
        observe_lets=(
            Binding(name="skew_delta", expr="sin(atan(alpha))"),
            Binding(name="skew_center", expr="skew_delta * 0.7978845608028654"),
            Binding(name="skew_scale", expr="sigma / sqrt(1.0 - skew_center * skew_center)"),
            Binding(name="skew_location", expr="mu - skew_scale * skew_center"),
        ),
        observe_args=("skew_location", "skew_scale", "alpha"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(
                name="alpha",
                prior="Normal(0.0, 4.0)",
                link=_IDENTITY,
                domain=_REAL,
                links=("identity", "log", "softplus", "squareplus"),
            ),
        ),
    ),
    "shifted_lognormal": Family(
        name="shifted_lognormal",
        location_link=_IDENTITY,
        location_links=_LOG_LOCATION_LINKS,
        observe_family="ShiftedLogNormal",
        recorded_events=True,
        log_response=True,
        response_domain=Domain(),
        observe_args=("mu", "sigma", "ndt"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(
                name="ndt",
                prior="Uniform(0.0, response_min)",
                link=_LOG,
                domain=_NONNEGATIVE,
                below_response=True,
                brms_prior="Uniform(0.0, response_min)",
            ),
        ),
    ),
    "frechet": Family(
        name="frechet",
        location_link=_LOG,
        location_links=_POSITIVE_LINKS,
        observe_family="Frechet",
        recorded_events=True,
        observe_lets=(
            Binding(name="frechet_scale", expr="mu * exp(-1.0 * lgamma(1.0 - 1.0 / nu))"),
        ),
        observe_args=("frechet_scale", "nu"),
        aux_params=(
            AuxParam(
                name="nu",
                prior="Restrict(base=Gamma(concentration=2.0, rate=0.1), low=1.0)",
                link=links["logm1"],
                domain=Domain(lower=1.0),
                links=("logm1",),
            ),
        ),
    ),
    "gen_extreme_value": Family(
        name="gen_extreme_value",
        location_link=_IDENTITY,
        location_links=_SHIFTED_LINKS,
        observe_family="GeneralizedExtremeValue",
        recorded_events=True,
        observe_args=("mu", "sigma", "xi"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(
                name="xi",
                prior="Normal(0.0, 0.3)",
                link=_IDENTITY,
                domain=_REAL,
                links=("log1p", "identity"),
            ),
        ),
    ),
    "gaussian": _GAUSSIAN,
    "bernoulli": Family(
        name="bernoulli",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS,
        observe_family="Bernoulli",
        r_squared="model",
    ),
    "binomial": Family(
        name="binomial",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS,
        observe_family="Binomial",
        recorded_events=True,
        factor_levels=2,
        observe_args=("trials", "mu"),
        trials_parameter="total_count",
    ),
    "beta_binomial": Family(
        name="beta_binomial",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS[:-1],
        observe_family="BetaBinomial",
        recorded_events=True,
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
        location_links=_COMPOSITIONAL_LINKS,
        observe_family="Categorical",
        response=ResponseSpec(kind="categorical"),
    ),
    "multinomial": Family(
        name="multinomial",
        location_link=_SOFTMAX,
        location_links=_COMPOSITIONAL_LINKS,
        observe_family="Multinomial",
        observe_args=("trials", "mu"),
        trials_parameter="total_count",
        response=ResponseSpec(kind="multinomial"),
    ),
    "dirichlet_multinomial": Family(
        name="dirichlet_multinomial",
        location_link=_SOFTMAX,
        location_links=_COMPOSITIONAL_LINKS,
        observe_family="DirichletMultinomial",
        observe_args=("trials", "mu"),
        trials_parameter="total_count",
        aux_params=(AuxParam(name="phi", prior="Gamma(2.0, 0.1)", link=_LOG),),
        response=ResponseSpec(kind="dirichlet_multinomial"),
    ),
    "dirichlet": Family(
        name="dirichlet",
        location_link=_SOFTMAX,
        location_links=_COMPOSITIONAL_LINKS,
        observe_family="Dirichlet",
        aux_params=(AuxParam(name="phi", prior="Gamma(2.0, 0.1)", link=_LOG),),
        response=ResponseSpec(kind="dirichlet"),
    ),
    "dirichlet2": Family(
        name="dirichlet2",
        location_link=_LOG,
        location_links=("log", "softplus", "squareplus", "identity", "logm1"),
        observe_family="Dirichlet",
        response=ResponseSpec(kind="dirichlet2", reference=None),
    ),
    "logistic_normal": Family(
        name="logistic_normal",
        location_link=_IDENTITY,
        location_links=("identity",),
        observe_family="AdditiveLogRatioNormal",
        aux_params=(AuxParam(
                name="sigma", prior="HalfNormal(1.0)", link=_LOG, per_component=True
            ),),
        response=ResponseSpec(kind="logistic_normal"),
    ),
    "poisson": Family(
        name="poisson",
        location_link=_LOG,
        location_links=_COUNT_LINKS,
        observe_family="Poisson",
        recorded_events=True,
        exposure=Exposure(
            observe_lets=(Binding(name="rate_mu", expr="mu * exposure"),),
            observe_args=("rate_mu",),
        ),
    ),
    "negative_binomial": Family(
        name="negative_binomial",
        location_link=_LOG,
        location_links=_COUNT_LINKS,
        observe_family="NegativeBinomial",
        recorded_events=True,
        exposure=Exposure(
            observe_lets=(Binding(name="rate_shape", expr="disp * exposure"),),
            observe_args=("rate_shape", "negative_binomial_probs"),
        ),
        observe_lets=(
            Binding(name="negative_binomial_probs", expr="mu / (mu + disp)"),
        ),
        observe_args=("disp", "negative_binomial_probs"),
        aux_params=(
            AuxParam(
                name="disp",
                prior="Gamma(2.0, 2.0)",
                link=_LOG,
                brms_class="shape",
                brms_prior="InverseGamma(0.4, 0.3)",
            ),
        ),
    ),
    "gamma": Family(
        name="gamma",
        location_link=_LOG,
        location_links=_POSITIVE_LINKS,
        observe_family="Gamma",
        recorded_events=True,
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
        location_links=_UNIT_LINKS,
        observe_family="Beta",
        recorded_events=True,
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
        location_links=_REAL_LINKS + _REAL_EXTENSIONS,
        observe_family="StudentT",
        recorded_events=True,
        error_scale="sigma",
        elliptical=Elliptical(scale="sigma", df="nu"),
        observe_args=("nu", "mu", "sigma"),
        aux_params=(
            AuxParam(
                name="nu",
                prior="Gamma(2.0, 0.1)",
                link=links["logm1"],
                links=("logm1", "identity"),
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
        location_links=_ORDINAL_LINKS,
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
        location_links=_ORDINAL_LINKS,
        observe_family="Categorical",
        ordinal=OrdinalSpec(kind="hurdle_cumulative"),
        aux_params=(
            AuxParam(
                name="disc", prior="LogNormal(0.0, 1.0)", link=_LOG, fixed_default=1.0
            ),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),
        ),
    ),
    "acat": Family(
        name="acat",
        location_link=_LOGIT,
        location_links=_ORDINAL_LINKS,
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
        location_links=_RATIO_LINKS,
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
        location_links=_RATIO_LINKS,
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
        location_links=_COUNT_LINKS,
        observe_family="ZeroInflatedPoisson",
        recorded_events=True,
        observe_args=("zi", "mu"),
        aux_params=(
            AuxParam(
                name="zi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
                domain=_UNIT_CLOSED,
            ),
        ),
    ),
    "hurdle_poisson": Family(
        name="hurdle_poisson",
        location_link=_LOG,
        location_links=_COUNT_LINKS,
        observe_family="HurdlePoisson",
        recorded_events=True,
        observe_args=("hu", "mu"),
        aux_params=(
            AuxParam(
                name="hu",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
                domain=_UNIT_CLOSED,
            ),
        ),
    ),
    "zero_inflated_negbinomial": Family(
        name="zero_inflated_negbinomial",
        location_link=_LOG,
        location_links=_COUNT_LINKS,
        observe_family="ZeroInflatedNegativeBinomial",
        recorded_events=True,
        observe_lets=(
            Binding(name="negative_binomial_probs", expr="mu / (mu + shape)"),
        ),
        observe_args=("zi", "shape", "negative_binomial_probs"),
        aux_params=(
            AuxParam(
                name="shape",
                prior="Gamma(2.0, 2.0)",
                link=_LOG,
                brms_prior="InverseGamma(0.4, 0.3)",
            ),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),
        ),
    ),
    "zero_inflated_binomial": Family(
        name="zero_inflated_binomial",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS,
        observe_family="ZeroInflatedBinomial",
        recorded_events=True,
        observe_args=("zi", "trials", "mu"),
        trials_parameter="total_count",
        aux_params=(AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),),
    ),
    "zero_inflated_beta_binomial": Family(
        name="zero_inflated_beta_binomial",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS,
        observe_family="ZeroInflatedBetaBinomial",
        recorded_events=True,
        observe_lets=(
            Binding(name="beta_binomial_concentration1", expr="mu * phi"),
            Binding(name="beta_binomial_concentration0", expr="(1.0 - mu) * phi"),
        ),
        observe_args=("zi", "trials", "beta_binomial_concentration1", "beta_binomial_concentration0"),
        trials_parameter="total_count",
        aux_params=(
            AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),
        ),
    ),
    "hurdle_negbinomial": Family(
        name="hurdle_negbinomial",
        location_link=_LOG,
        location_links=_COUNT_LINKS,
        observe_family="HurdleNegativeBinomial",
        recorded_events=True,
        observe_lets=(
            Binding(name="negative_binomial_probs", expr="mu / (mu + shape)"),
        ),
        observe_args=("hu", "shape", "negative_binomial_probs"),
        aux_params=(
            AuxParam(
                name="shape",
                prior="Gamma(2.0, 2.0)",
                link=_LOG,
                brms_prior="InverseGamma(0.4, 0.3)",
            ),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),
        ),
    ),
    "hurdle_gamma": Family(
        name="hurdle_gamma",
        location_link=_LOG,
        location_links=_POSITIVE_LINKS,
        observe_family="HurdleGamma",
        recorded_events=True,
        observe_lets=(
            Binding(name="hurdle_gamma_rate", expr="shape / mu"),
        ),
        observe_args=("hu", "shape", "hurdle_gamma_rate"),
        aux_params=(
            AuxParam(name="shape", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),
        ),
    ),
    "hurdle_lognormal": Family(
        name="hurdle_lognormal",
        location_link=_IDENTITY,
        location_links=_LOG_LOCATION_LINKS,
        observe_family="HurdleLogNormal",
        recorded_events=True,
        log_response=True,
        observe_args=("hu", "mu", "sigma"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="hu", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),
        ),
    ),
    "zero_inflated_beta": Family(
        name="zero_inflated_beta",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS,
        observe_family="ZeroOneInflatedBeta",
        recorded_events=True,
        response_domain=Domain(upper=1.0, lower_closed=True),
        observe_args=("mu", "phi", "zi", 0.0),
        aux_params=(
            AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="zi", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_CLOSED),
        ),
    ),
    "xbeta": Family(
        name="xbeta",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS,
        observe_family="ExtendedBeta",
        observe_args=("mu", "phi", "kappa"),
        aux_params=(
            AuxParam(name="phi", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(
                name="kappa",
                prior="Gamma(0.01, 0.01)",
                link=_LOG,
                brms_prior="Gamma(0.01, 0.01)",
            ),
        ),
    ),
    "zero_one_inflated_beta": Family(
        name="zero_one_inflated_beta",
        location_link=_LOGIT,
        location_links=_UNIT_LINKS,
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
                domain=_UNIT_CLOSED,
            ),
            AuxParam(
                name="coi",
                prior="Beta(2.0, 2.0)",
                link=_LOGIT,
                domain=_UNIT_CLOSED,
            ),
        ),
    ),
    "lognormal": Family(
        name="lognormal",
        location_link=_IDENTITY,
        location_links=_LOG_LOCATION_LINKS,
        observe_family="LogNormal",
        recorded_events=True,
        log_response=True,
        observe_args=("mu", "sigma"),
        aux_params=(AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),),
    ),
    "asym_laplace": Family(
        name="asym_laplace",
        location_link=_IDENTITY,
        location_links=_SHIFTED_LINKS,
        observe_family="AsymmetricLaplace",
        recorded_events=True,
        observe_args=("mu", "sigma", "quantile"),
        aux_params=(
            AuxParam(name="sigma", prior="HalfCauchy(2.0)", link=_LOG),
            AuxParam(name="quantile", prior="Beta(2.0, 2.0)", link=_LOGIT, domain=_UNIT_OPEN),
        ),
    ),
    "exgaussian": Family(
        name="exgaussian",
        location_link=_IDENTITY,
        location_links=_SHIFTED_LINKS,
        observe_family="ExGaussian",
        recorded_events=True,
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
        location_links=("inverse_square", "inverse", "identity", "log", "softplus", "squareplus"),
        observe_family="InverseGaussian",
        recorded_events=True,
        observe_args=("mu", "shape"),
        aux_params=(AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "exponential": Family(
        name="exponential",
        location_link=_LOG,
        location_links=_POSITIVE_LINKS,
        observe_family="Exponential",
        recorded_events=True,
        observe_lets=(
            Binding(name="exponential_rate", expr="1.0 / mu"),
        ),
        observe_args=("exponential_rate",),
    ),
    "geometric": Family(
        name="geometric",
        location_link=_LOG,
        location_links=_COUNT_LINKS,
        observe_family="Geometric",
        recorded_events=True,
        exposure=Exposure(
            observe_family="NegativeBinomial",
            observe_lets=(Binding(name="rate_probs", expr="mu / (1.0 + mu)"),),
            observe_args=("exposure", "rate_probs"),
        ),
        observe_lets=(
            Binding(name="geometric_probs", expr="1.0 / (1.0 + mu)"),
        ),
        observe_args=("geometric_probs",),
    ),
    "weibull": Family(
        name="weibull",
        location_link=_LOG,
        location_links=_POSITIVE_LINKS,
        observe_family="Weibull",
        recorded_events=True,
        observe_lets=(
            Binding(name="weibull_scale", expr="mu * exp(-1.0 * lgamma(1.0 + 1.0 / shape))"),
        ),
        observe_args=("weibull_scale", "shape"),
        aux_params=(AuxParam(name="shape", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "von_mises": Family(
        name="von_mises",
        location_link=links["tan_half"],
        location_links=("tan_half", "identity"),
        observe_family="VonMises",
        recorded_events=True,
        circular=True,
        observe_args=("mu", "kappa"),
        aux_params=(AuxParam(name="kappa", prior="Gamma(2.0, 2.0)", link=_LOG),),
    ),
    "cox": Family(
        name="cox",
        location_link=_LOG,
        location_links=("log", "identity", "softplus", "squareplus"),
        observe_family="BernsteinHazard",
        recorded_events=True,
        observe_args=("baseline_breaks", "baseline_controls", "mu"),
        log_location_argument="log_rate",
        baseline=BaselineSpec(),
    ),
    # A template: quivers.formulas._mixture repeats its one component.
    "mixture": Family(
        name="mixture",
        location_link=_IDENTITY,
        location_links=_REAL_LINKS,
        observe_family="Mixture",
        recorded_events=True,
        mixture=MixtureSpec(components=(_GAUSSIAN,)),
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
    "Domain",
    "Elliptical",
    "Exposure",
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
