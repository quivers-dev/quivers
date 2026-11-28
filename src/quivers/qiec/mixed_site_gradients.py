"""Compose finite Bernoulli estimators with checked Normal pathwise actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction

from quivers._values import HeldValue
from quivers.qiec._random_site_graph import (
    _Address,
    _check_random_graph,
    _random_computation,
)
from quivers.qiec.bernoulli_estimators import (
    BernoulliGradientPlan,
    BernoulliGradientSample,
    BernoulliVisit,
    _assemble_finite_tree,
    _check_request as _check_bernoulli,
    _Estimator,
    _NeedChoice,
    _probability_argument,
)
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import (
    ExactJvp,
    JvpLimit,
    JvpUnsupported,
    _fraction,
)
from quivers.qiec.effects import EffectRequest
from quivers.qiec.evaluator import (
    FuelExhaustedError,
    RuntimeRequest,
    _Continue,
    _Frame,
    _Returned,
)
from quivers.qiec.families import FAMILIES
from quivers.qiec.identifiers import ComputationId, EffectInstanceId, SiteId
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import QiecModule
from quivers.qiec.normal_pathwise import (
    NormalPathwisePlan,
    NormalVisit,
    _check_request as _check_normal,
    _NoiseDual,
    _normal_path,
    _NormalEvaluator,
)
from quivers.qiec.polynomial_laws import PolynomialLawLimit, _PolynomialRing
from quivers.qiec.scope import Scope
from quivers.qiec.terms import DistributionValue, Local, Value
from quivers.qiec.types import BOOL


@dataclass(frozen=True, slots=True)
class MixedNormalVisit:
    """One Normal coordinate for a source occurrence at a finite history.

    Parameters
    ----------
    prefix : tuple[bool, ...]
        The Bernoulli choices already made when this request is reached.
    visit : NormalVisit
        The canonical source occurrence and its checked scale.
    """

    prefix: tuple[bool, ...]
    visit: NormalVisit


@dataclass(frozen=True, slots=True)
class MixedNormalPath:
    """The conditional Normal action for one complete finite branch.

    Parameters
    ----------
    prefix : tuple[bool, ...]
        The complete Bernoulli choices for that branch.
    coordinates : tuple[int, ...]
        Indices into the mixed plan's full Normal-coordinate table, in
        order of the branch's Normal draws.
    normal : NormalPathwisePlan
        The checked exact and pointwise action of the actual branch.
    """

    prefix: tuple[bool, ...]
    coordinates: tuple[int, ...]
    normal: NormalPathwisePlan


@dataclass(frozen=True, slots=True)
class MixedSiteGradientPlan:
    """A finite estimator tree whose leaves use checked Normal rules.

    Parameters
    ----------
    finite : BernoulliGradientPlan
        The finite estimator actions with the Normal noise integrated out.
    normal_visits : tuple[MixedNormalVisit, ...]
        A fresh coordinate per source occurrence and preceding finite
        history. Histories share a coordinate when they reach exactly
        the same Normal request before they diverge.
    normal_paths : tuple[MixedNormalPath, ...]
        The complete finite histories and their conditional Normal actions.
    polynomial_work : int
        The total polynomial and moment work used by all reference replays.
    """

    finite: BernoulliGradientPlan
    normal_visits: tuple[MixedNormalVisit, ...]
    normal_paths: tuple[MixedNormalPath, ...]
    polynomial_work: int

    @property
    def exact(self) -> ExactJvp:
        """Give the expected source return and its directional derivative.

        Returns
        -------
        ExactJvp
            The expectation over both Bernoulli and Normal variables.
        """
        return self.finite.exact

    def sample(self, *, seed: int | None = None) -> BernoulliGradientSample:
        """Run the admitted finite estimators with Normal noise integrated.

        Parameters
        ----------
        seed : int or None
            An optional seed for independent finite estimator draws.

        Returns
        -------
        BernoulliGradientSample
            An unbiased estimate of the signed expected return and its
            directional derivative in the checked local fragment.
        """
        return self.finite.sample(seed=seed)

    def at_noise(self, noise: tuple[int | Fraction, ...]) -> BernoulliGradientPlan:
        """Fix every Normal coordinate before applying the finite estimators.

        Parameters
        ----------
        noise : tuple[int or Fraction, ...]
            The fixed coordinates in normal_visits order.

        Returns
        -------
        BernoulliGradientPlan
            The finite estimator and its exact conditional mean at these
            coordinates. These inputs are supplied by the caller.

        Raises
        ------
        ValueError
            If the noise vector has the wrong size or invalid entries.
        JvpLimit
            If conditional values exceed the exact arithmetic bounds.
        """
        if len(noise) != len(self.normal_visits):
            raise ValueError("noise coordinates must match the mixed Normal visits")
        records: dict[tuple[bool, ...], ExactJvp | BernoulliVisit] = {
            visit.prefix: visit for visit in self.finite.visits
        }
        for path in self.normal_paths:
            records[path.prefix] = path.normal.at_noise(
                tuple(noise[index] for index in path.coordinates)
            )
        tree, exact = _assemble_finite_tree(records)
        return replace(self.finite, exact=exact, tree=tree)


def _check_request(request: EffectRequest, instance: EffectInstanceId) -> None:
    """Check the actual nominal family and direct source request shape.

    Parameters
    ----------
    request : EffectRequest
        The operation request from the checked call graph.
    instance : EffectInstanceId
        The one nominal Random instance of the entry.

    Raises
    ------
    JvpUnsupported
        If the request leaves the Bernoulli or Normal fragments.
    """
    if request.result_type == BOOL:
        _check_bernoulli(request, instance)
    else:
        _check_normal(request, instance)


class _MixedEvaluator(_NormalEvaluator):
    """Replay one finite prefix with the exact Normal interpretation.

    Parameters
    ----------
    module : QiecModule
        The validated source and actual named calls.
    fuel : int
        The per-replay source step budget.
    ring : _PolynomialRing
        The bound and arithmetic for the Normal coordinates.
    normal : frozenset[SiteId]
        The sites with the explicit Normal rule.
    bernoulli : Mapping[SiteId, _Estimator]
        The finite sites with their declared rules.
    prefix : tuple[bool, ...]
        The finite decisions supplied to this replay.
    """

    def __init__(
        self,
        module: QiecModule,
        fuel: int,
        ring: _PolynomialRing,
        normal: frozenset[SiteId],
        bernoulli: Mapping[SiteId, _Estimator],
        prefix: tuple[bool, ...],
    ) -> None:
        super().__init__(module, fuel, ring, normal)
        self.bernoulli = bernoulli
        self.prefix = prefix
        self.choices: set[_Address] = set()
        self.normal_prefixes: list[tuple[bool, ...]] = []

    def _value(self, value: Value, environment: Mapping[Local, object]) -> object:
        """Add the finite Bernoulli operand to the Normal value algebra.

        Parameters
        ----------
        value : Value
            The actual source term.
        environment : Mapping[Local, object]
            The checked lexical bindings of this replay.

        Returns
        -------
        object
            A polynomial scalar, finite probability, or other machine value.
        """
        if (
            isinstance(value, DistributionValue)
            and value.family == FAMILIES["Bernoulli"].id
        ):
            return self._value(_probability_argument(value), environment)
        return super()._value(value, environment)

    def _dispatch(
        self,
        request: RuntimeRequest,
        environment: Scope[Local, HeldValue],
        stack: list[_Frame],
        search_index: int,
    ) -> _Continue:
        """Supply the checked Normal or recorded Bernoulli result.

        Parameters
        ----------
        request : RuntimeRequest
            The addressed operation on the selected Random row.
        environment : Scope[Local, HeldValue]
            The unmodified lexical state for the resumed continuation.
        stack : list[_Frame]
            The unmodified stack.
        search_index : int
            The current handler search position.

        Returns
        -------
        _Continue
            The same continuation supplied with its licensed result.

        Raises
        ------
        JvpUnsupported
            If the finite probability depends on noise or is outside
            its open support, an address repeats, or a policy is missing.
        _NeedChoice
            If the replay reaches the next unresolved finite draw.
        """
        position = len(self.choices)
        if request.core.result_type != BOOL:
            resumed = super()._dispatch(request, environment, stack, search_index)
            self.normal_prefixes.append(self.prefix[:position])
            return resumed
        site = request.core.origin.static_site
        method = self.bernoulli.get(site)
        if method is None:
            raise JvpUnsupported(f"missing finite estimator for reached site {site}")
        _, probability = request.arguments
        assert isinstance(probability, _NoiseDual)
        if probability.noisy:
            raise JvpUnsupported(
                "mixed Bernoulli probabilities must be noise-independent"
            )
        p = self.constant(probability.primal)
        if not 0 < p < 1:
            raise JvpUnsupported(
                "conditional Bernoulli probabilities must be strictly between zero and one"
            )
        if request.address in self.choices:
            raise JvpUnsupported(
                f"repeated Bernoulli address along a trace {request.address!r}"
            )
        self.choices.add(request.address)
        if position == len(self.prefix):
            raise _NeedChoice(
                BernoulliVisit(
                    site,
                    request.address,
                    self.prefix,
                    method,
                    p,
                    self.constant(probability.tangent),
                )
            )
        return _Continue(
            current=_Returned(
                value=self.numbers.number("truth", self.prefix[position])
            ),
            environment=environment,
            resumption_path=request.resumption_path,
        )


def plan_mixed_site_gradient(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    policies: tuple[SiteEstimator, ...],
    *,
    max_paths: int = 4096,
    max_sites: int = 64,
    max_work: int = 250_000,
    fuel: int = 50_000,
) -> MixedSiteGradientPlan:
    """Check a bounded nominal Bernoulli/Normal estimator program.

    Bernoulli probabilities may use shared inputs and earlier Boolean
    outcomes. Each must be noise-independent and in the open interval
    from zero to one. The Normal operations must meet the scalar pathwise
    checks on all finite traces. The finite estimator consumes the signed
    conditional values and their derivatives linearly.

    Parameters
    ----------
    module : QiecModule
        The checked source module.
    computation : ComputationId
        The named scalar-returning entry on one nominal Random row.
    arguments, direction : tuple[int or Fraction, ...]
        The exact parameter point and direction in source parameter order.
    policies : tuple[SiteEstimator, ...]
        One explicit policy for each reached static site.
    max_paths : int
        Maximum number of complete finite traces.
    max_sites : int
        Maximum number of distinct Normal coordinates per finite trace.
    max_work : int
        Total polynomial and moment work over the checked reference runs.
    fuel : int
        Maximum source steps in each reference run.

    Returns
    -------
    MixedSiteGradientPlan
        Both the joint integrated result and conditional pathwise actions.

    Raises
    ------
    ValueError
        If the arguments, source name, declaration or a bound is malformed.
    JvpUnsupported
        If the actual source leaves the checked fragment.
    JvpLimit
        If a trace, occurrence, arithmetic or source step limit is exceeded.
    """
    for name, limit in (
        ("max_paths", max_paths),
        ("max_sites", max_sites),
        ("max_work", max_work),
        ("fuel", fuel),
    ):
        _validate_limit(name, limit)
    registry, owner = _random_computation(module, computation, "mixed site")
    if not len(arguments) == len(direction) == len(owner.parameters):
        raise ValueError("arguments and direction must match the computation's arity")
    declared = {policy.site: policy.estimator for policy in policies}
    if len(declared) != len(policies):
        raise ValueError("duplicate estimator for one mixed site")
    bernoulli: dict[SiteId, _Estimator] = {}
    normal: set[SiteId] = set()
    for site, method in declared.items():
        if method in ("enumerate", "score", "measure"):
            bernoulli[site] = method
        elif method == "reparam":
            normal.add(site)
        else:
            raise JvpUnsupported(f"unsupported mixed site estimator {method!r}")
    _check_random_graph(module, owner, "mixed site", _check_request)
    point = tuple(
        (parameter, _fraction(value), _fraction(tangent))
        for parameter, value, tangent in zip(
            owner.parameters, arguments, direction, strict=True
        )
    )
    normal_sites = frozenset(normal)
    width = min(max_sites, max_work, fuel)
    remaining = max_work
    pending: list[tuple[bool, ...]] = [()]
    records: dict[tuple[bool, ...], ExactJvp | BernoulliVisit] = {}
    visits: list[BernoulliVisit] = []
    normal_visits: list[MixedNormalVisit] = []
    coordinates: dict[tuple[tuple[bool, ...], _Address], int] = {}
    paths: list[MixedNormalPath] = []
    path_count, steps = 1, 0
    while pending:
        prefix = pending.pop()
        ring = _PolynomialRing(width, remaining)
        evaluator = _MixedEvaluator(module, fuel, ring, normal_sites, bernoulli, prefix)
        environment = {
            parameter: evaluator.input(value, tangent, parameter=True)
            for parameter, value, tangent in point
        }
        try:
            result = evaluator.evaluate_validated(owner.body, registry, environment)
            assert isinstance(result, _NoiseDual) and len(evaluator.choices) == len(
                prefix
            )
            leaf = _normal_path(computation, evaluator, result, remaining)
        except _NeedChoice as choice:
            records[prefix] = choice.visit
            visits.append(choice.visit)
            path_count += 1
            if path_count > max_paths:
                raise JvpLimit(
                    f"mixed enumeration exceeds {max_paths} complete paths"
                ) from None
            pending.extend(((*prefix, True), (*prefix, False)))
        except FuelExhaustedError as error:
            raise JvpLimit(
                f"mixed replay exceeded its step budget of {fuel} at {prefix!r}"
            ) from error
        except PolynomialLawLimit as error:
            raise JvpLimit(f"mixed Normal expansion: {error}") from error
        else:
            records[prefix] = leaf.exact
            indices: list[int] = []
            for history, visit in zip(
                evaluator.normal_prefixes, leaf.visits, strict=True
            ):
                coord = coordinates.setdefault(
                    (history, visit.address), len(normal_visits)
                )
                if coord == len(normal_visits):
                    normal_visits.append(MixedNormalVisit(history, visit))
                indices.append(coord)
            paths.append(MixedNormalPath(prefix, tuple(indices), leaf))
        finally:
            steps += evaluator._steps
            remaining = ring.remaining
    reached = {visit.site for visit in visits} | {
        item.visit.site for item in normal_visits
    }
    unused = sorted(str(site) for site in declared.keys() - reached)
    if unused:
        raise JvpUnsupported(
            f"estimator policies name unreached sites: {', '.join(unused)}"
        )
    tree, exact = _assemble_finite_tree(records)
    ordered = tuple(
        sorted(
            (policy for policy in policies if policy.site in bernoulli),
            key=lambda p: str(p.site),
        )
    )
    finite = BernoulliGradientPlan(
        computation, exact, ordered, tuple(visits), path_count, steps, tree
    )
    return MixedSiteGradientPlan(
        finite, tuple(normal_visits), tuple(paths), max_work - remaining
    )


__all__ = [
    "MixedNormalPath",
    "MixedNormalVisit",
    "MixedSiteGradientPlan",
    "plan_mixed_site_gradient",
]
