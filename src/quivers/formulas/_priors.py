"""Typed prior selection over the formula compiler's existing latent sites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel


class RegularizedHorseshoe(ValueModel):
    """A sampled global/local shrinkage hierarchy with a Student-t slab.

    Parameters
    ----------
    df, df_global, df_slab : float
        Positive degrees of freedom for local, global, and slab priors.
    scale_global, scale_slab : float
        Positive global and slab scales.
    par_ratio : float or None
        If supplied, replace scale_global by par_ratio/sqrt(training_rows).
    autoscale : bool
        Multiply the global scale by a scalar residual sigma when available.
        Distributional-parameter coefficients are not autoscaled.
    main : bool
        brms ``main = TRUE``: when one predictor's shrinkage spans several
        parameter classes, the classes join one hierarchy whose arguments
        are taken from the single rule marked main.
    """

    df: float = 1.0
    scale_global: float = 1.0
    df_global: float = 1.0
    scale_slab: float = 2.0
    df_slab: float = 4.0
    par_ratio: float | None = None
    autoscale: bool = True
    main: bool = False

    @dx.model_validator()
    def _validate(self) -> None:
        for name in ("df", "scale_global", "df_global", "scale_slab", "df_slab"):
            _positive(getattr(self, name), name)
        if self.par_ratio is not None:
            _positive(self.par_ratio, "par_ratio")
        if type(self.autoscale) is not bool or type(self.main) is not bool:
            raise TypeError("autoscale and main must be Boolean")


class R2D2(ValueModel):
    """A Gaussian-kernel R2D2 variance-allocation hierarchy.

    Parameters
    ----------
    mean_R2 : float
        Mean of the Beta prior on explained-variance fraction, in (0,1).
    prec_R2 : float
        Positive Beta precision.
    cons_D2 : float or tuple[float, ...]
        Positive Dirichlet concentration, shared or per selected coordinate.
    autoscale : bool
        Multiply coefficient scales by a scalar residual sigma when available.
    main : bool
        brms ``main = TRUE``: the rule whose arguments a hierarchy spanning
        several parameter classes of one predictor uses.
    """

    mean_R2: float = 0.5
    prec_R2: float = 2.0
    cons_D2: float | tuple[float, ...] = dx.field(
        default=0.5, opaque=True, extras=OPAQUE_BY_VALUE
    )
    autoscale: bool = True
    main: bool = False

    @dx.model_validator()
    def _validate(self) -> None:
        _positive(self.mean_R2, "mean_R2")
        if self.mean_R2 >= 1:
            raise ValueError("mean_R2 must lie strictly between zero and one")
        _positive(self.prec_R2, "prec_R2")
        values = self.cons_D2 if isinstance(self.cons_D2, tuple) else (self.cons_D2,)
        if not values:
            raise ValueError("cons_D2 must not be empty")
        for value in values:
            _positive(value, "cons_D2")
        if type(self.autoscale) is not bool or type(self.main) is not bool:
            raise TypeError("autoscale and main must be Boolean")


class ImproperFlat(ValueModel):
    """The improper flat prior brms writes as an empty prior string.

    Its density is constant on the target's domain, so it is the Lebesgue
    measure there and has no normalizing constant. Fitting accepts it, since
    the posterior can still be proper. Prior draws, ``sample_prior``, and
    marginal-likelihood estimates require a normalized prior and refuse it
    with an error naming the flat sites.
    """


class ConstantPrior(ValueModel):
    """brms ``constant(const, broadcast)``: fix a parameter to known values.

    Parameters
    ----------
    value : float or tuple[float, ...]
        One value, or one value per selected coordinate in the target's
        coefficient order.
    broadcast : bool
        Repeat a single value over every coordinate of the target, as brms
        does by default. With False, a tuple must match the target's width.
    """

    value: float | tuple[float, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    broadcast: bool = True

    @dx.model_validator()
    def _validate(self) -> None:
        values = self.value if isinstance(self.value, tuple) else (self.value,)
        if not values or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in values
        ):
            raise ValueError("a constant prior requires finite real values")
        if type(self.broadcast) is not bool:
            raise TypeError("broadcast must be Boolean")


def _positive(value, name):
    """Validate one finite positive hyperparameter."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


type PriorDistribution = (
    str | RegularizedHorseshoe | R2D2 | ImproperFlat | ConstantPrior
)
"""A compositional QVR prior expression, a sampled shrinkage hierarchy, the
improper flat prior, or a constant."""


class PriorSpec(ValueModel):
    """Select a prior using authored formula labels rather than generated names.

    Parameters
    ----------
    distribution : str or RegularizedHorseshoe or R2D2
        A compositional QVR distribution or a sampled shrinkage hierarchy.
    class_ : str or None
        Parameter class: b excludes intercepts; Intercept, sd, cor, and
        auxiliary parameter names select their corresponding roles.
    coef, group, dpar, nlpar, site, response : str or None
        Expanded coefficient label, original group name, distributional or
        nonlinear parameter name, exact generated site, or authored joint-response
        label. None is a wildcard.
    lower, upper : float or None
        Optional finite bounds, intersected with the target's domain.
    block : str or None
        Shared shrinkage identity. Rules with the same block share global
        hyperparameters and allocations and must specify identical hierarchies.
    tag : str or None
        brms ``tag``: a label carried into the prior table, so a subset of
        priors can be selected for sensitivity analysis.
    """

    distribution: PriorDistribution = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    class_: str | None = "b"
    coef: str | None = None
    group: str | None = None
    dpar: str | None = None
    nlpar: str | None = None
    site: str | None = None
    response: str | None = None
    lower: float | None = None
    upper: float | None = None
    block: str | None = None
    tag: str | None = None

    @dx.model_validator()
    def _validate(self) -> None:
        if isinstance(self.distribution, str) and not self.distribution.strip():
            raise ValueError("a prior distribution must not be empty")
        if self.tag is not None and (not isinstance(self.tag, str) or not self.tag):
            raise ValueError("a prior tag must be nonempty text or None")
        if isinstance(self.distribution, ConstantPrior) and (
            self.lower is not None or self.upper is not None
        ):
            raise ValueError("a constant prior takes no bounds")
        for name in (*_SELECTORS, "block"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"prior selector {name} must be nonempty text or None")
        if self.dpar is not None and self.nlpar is not None:
            raise ValueError("a prior cannot select both dpar and nlpar")
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("prior bounds must be finite real numbers or None")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("prior lower bound must be below upper bound")
        if self.block is not None and isinstance(
            self.distribution, (str, ImproperFlat, ConstantPrior)
        ):
            raise ValueError(
                "a shared block is only meaningful for a shrinkage hierarchy"
            )

    @property
    def selector(self) -> dict[str, str]:
        """Return exactly the explicitly selected role coordinates."""
        return {
            name: getattr(self, name)
            for name in _SELECTORS
            if getattr(self, name) is not None
        }

    def __add__(self, other: PriorSpec | PriorSet) -> PriorSet:
        return PriorSet(rules=(self,) + normalize_priors(other).rules)


class PriorSet(ValueModel):
    """An immutable collection of prior rules with explicit override semantics.

    Parameters
    ----------
    rules : tuple[PriorSpec, ...]
        Ordered rules. A strict selector refinement overrides a broader
        rule; ambiguous competing rules and unused rules are errors.
    """

    rules: tuple[PriorSpec, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def __add__(self, other: PriorSpec | PriorSet) -> PriorSet:
        return PriorSet(rules=self.rules + normalize_priors(other).rules)


type PriorInput = Mapping[str, str] | PriorSpec | PriorSet | Sequence[PriorSpec]
"""Authored exact-name overrides or typed rules normalized to one immutable PriorSet."""
_SELECTORS = ("class_", "coef", "group", "dpar", "nlpar", "site", "response")


def normalize_priors(value: PriorInput | None) -> PriorSet:
    """Adapt legacy named priors and typed rules into one immutable representation.

    Parameters
    ----------
    value : PriorInput or None
        Exact-name mapping, one rule, or a collection of typed rules.

    Returns
    -------
    PriorSet
        The canonical representation retained for compilation and replay.

    Raises
    ------
    TypeError
        If values are neither named distribution strings nor typed rules.
    """
    if value is None:
        return PriorSet()
    if isinstance(value, PriorSet):
        return value
    if isinstance(value, PriorSpec):
        return PriorSet(rules=(value,))
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            raise TypeError("named priors must map site names to distribution strings")
        return PriorSet(
            rules=tuple(
                PriorSpec(distribution=v, class_=None, site=k) for k, v in value.items()
            )
        )
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not all(isinstance(v, PriorSpec) for v in value)
    ):
        raise TypeError("priors require a named mapping or typed PriorSpec collection")
    return PriorSet(rules=tuple(value))


class PriorTarget(ValueModel):
    """One overrideable site with semantic labels and nominal component axes.

    Parameters
    ----------
    site, class_ : str
        Generated binder and semantic parameter class.
    coef, group, dpar, nlpar, response : str or None
        Authored coefficient, grouping, predictor, and joint-response labels.
    axes : tuple[str, ...]
        Nominal independent coefficient axes in source order.
    shape : tuple[int, ...]
        Their declared cardinalities, distinct from native event dimensions.
    lower, upper : float or None
        Admissible coefficient-domain bounds.
    event_rank : int
        Native distribution event rank, separate from independent axes.
    component_labels : tuple[str, ...]
        Free response-component labels in the retained coefficient order.
    """

    site: str
    class_: str
    coef: str | None = None
    group: str | None = None
    dpar: str | None = None
    nlpar: str | None = None
    response: str | None = None
    axes: tuple[str, ...] = ()
    shape: tuple[int, ...] = ()
    lower: float | None = None
    upper: float | None = None
    event_rank: int = 0
    component_labels: tuple[str, ...] = ()

    @dx.model_validator()
    def _validate(self) -> None:
        if len(self.axes) != len(self.shape) or any(
            type(n) is not int or n < 1 for n in self.shape
        ):
            raise ValueError("prior target axes require aligned positive cardinalities")

    @property
    def width(self) -> int:
        """Return the number of independent selected coordinates."""
        return math.prod(self.shape)


def brms_class_names(target: PriorTarget) -> frozenset[str]:
    """Return the brms prior class names that select a target.

    brms names the residual correlation of a multivariate model ``rescor``,
    the correlation of a logistic-normal response ``lncor``, the spacing of
    equidistant ordinal thresholds ``delta``, and accepts the Cholesky
    spellings ``L``, ``Lrescor`` and ``Llncor`` of the correlation classes.

    Parameters
    ----------
    target : PriorTarget
        One registered site.

    Returns
    -------
    frozenset[str]
        The target's own class and its brms names.
    """
    names = {target.class_}
    if target.class_ == "cor":
        names |= {"L"} if target.group is not None else {"rescor", "Lrescor"}
    elif target.class_ == "rescor":
        names |= {"lncor", "Llncor"}
    elif target.class_ == "cutpoint_spacing":
        names |= {"delta"}
    return frozenset(names)


def _selects(target: PriorTarget, key: str, value: str | None) -> bool:
    """Report whether one selector field admits a target.

    Parameters
    ----------
    target : PriorTarget
        One registered site.
    key : str
        Selector field.
    value : str or None
        Its authored value.

    Returns
    -------
    bool
        Equality on the field; a class also matches by its brms names.
    """
    if key == "class_" and value is not None:
        return value in brms_class_names(target)
    return getattr(target, key) == value


def resolve_priors(
    priors: PriorSet, targets: Sequence[PriorTarget]
) -> dict[str, PriorSpec]:
    """Resolve strict selector refinements and reject ambiguity or unused rules.

    Parameters
    ----------
    priors : PriorSet
        Authored selector rules.
    targets : Sequence[PriorTarget]
        Canonical emitted-site catalogue.

    Returns
    -------
    dict[str, PriorSpec]
        One winning rule per selected site.

    Raises
    ------
    ValueError
        If a rule matches no target, is completely shadowed, or conflicts
        with an incomparable rule on at least one target.
    """
    matched = [False] * len(priors.rules)
    won = [False] * len(priors.rules)
    selected = {}
    for target in targets:
        candidates = []
        for index, rule in enumerate(priors.rules):
            if all(
                _selects(target, key, value) for key, value in rule.selector.items()
            ):
                matched[index] = True
                candidates.append(index)
        winners = [
            i
            for i in candidates
            if not any(
                (priors.rules[i].site is None and priors.rules[j].site is not None)
                or priors.rules[i].selector.items() < priors.rules[j].selector.items()
                for j in candidates
            )
        ]
        if len(winners) > 1:
            raise ValueError(
                f"conflicting prior selectors for {target.site!r}: "
                + ", ".join(str(priors.rules[i].selector) for i in winners)
            )
        if winners:
            index = winners[0]
            won[index] = True
            selected[target.site] = priors.rules[index]
    for index, rule in enumerate(priors.rules):
        if not matched[index]:
            raise ValueError(
                f"unused prior selector {rule.selector}; available targets are {[t.site for t in targets]}"
            )
        if not won[index]:
            raise ValueError(f"prior selector {rule.selector} is completely shadowed")
    return selected


__all__ = [
    "PriorSpec",
    "PriorSet",
    "PriorTarget",
    "RegularizedHorseshoe",
    "R2D2",
    "ImproperFlat",
    "ConstantPrior",
    "PriorInput",
    "PriorDistribution",
    "normalize_priors",
]
