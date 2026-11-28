"""Exact and stochastic directional derivatives for checked finite Bernoulli sites."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from random import Random
from typing import Literal

from quivers._values import HeldValue
from quivers.qiec._random_site_graph import (
    _Address,
    _check_random_graph,
    _random_computation,
)
from quivers.qiec.builtins import RANDOM, RANDOM_SAMPLE
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import (
    ExactJvp,
    JvpLimit,
    JvpUnsupported,
    _bound,
    _Dual,
    _dual,
    _fraction,
    _ScalarDerivativeEvaluator,
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
from quivers.qiec.module import QiecModule
from quivers.qiec.scope import Scope
from quivers.qiec.terms import (
    DistributionValue,
    Local,
    SiteValue,
    Value,
)
from quivers.qiec.types import BOOL

type _Estimator = Literal["enumerate", "score", "measure"]


@dataclass(frozen=True, slots=True)
class BernoulliVisit:
    """A reached draw and its conditional probability at a source history.

    Parameters
    ----------
    site : SiteId
        The static identity named by the policy.
    address : tuple
        The checked evaluator's request address.
    prefix : tuple[bool, ...]
        The source choices preceding this request in the trace.
    estimator : {"enumerate", "score", "measure"}
        The policy declared for the static site of this occurrence.
    probability : Fraction
        The conditional probability of true.
    tangent : Fraction
        The exact directional derivative of that probability.
    """

    site: SiteId
    address: _Address
    prefix: tuple[bool, ...]
    estimator: _Estimator
    probability: Fraction
    tangent: Fraction


@dataclass(frozen=True, slots=True)
class BernoulliScoreDraw:
    """One stochastic choice made by the score estimator.

    Parameters
    ----------
    visit : BernoulliVisit
        The source occurrence and conditional law.
    choice : bool
        The outcome selected. Enumerated branches do not make these draws.
    """

    visit: BernoulliVisit
    choice: bool


@dataclass(frozen=True, slots=True)
class BernoulliWeakSelector:
    """One primal branch selector for a weak derivative.

    Parameters
    ----------
    visit : BernoulliVisit
        The source occurrence and conditional law.
    choice : bool
        The selected primal branch. Both branches contribute to the
        weak derivative, including the one this selector did not choose.
    """

    visit: BernoulliVisit
    choice: bool


@dataclass(frozen=True, slots=True)
class BernoulliGradientSample:
    """One realization of the chosen per-site derivative estimator.

    Parameters
    ----------
    value : Fraction
        The resulting estimate of the expected return.
    tangent : Fraction
        The resulting unbiased estimate of its directional derivative
        within the checked local fragment.
    visits : tuple[BernoulliVisit, ...]
        All visited requests in order, including deterministic enumeration.
    draws : tuple[BernoulliScoreDraw, ...]
        The stochastic choices in order. Enumeration may visit both
        conditional branches and a score choice visits only its own.
    selectors : tuple[BernoulliWeakSelector, ...]
        The primal selectors of weak derivative sites, in execution order.
    virtual_branches : tuple[tuple[bool, ...], ...]
        The actually launched weak derivative arms, as source prefixes.
        Both arms of each weak derivative appear in execution order.
    """

    value: Fraction
    tangent: Fraction
    visits: tuple[BernoulliVisit, ...]
    draws: tuple[BernoulliScoreDraw, ...]
    selectors: tuple[BernoulliWeakSelector, ...] = ()
    virtual_branches: tuple[tuple[bool, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class BernoulliBranch:
    """A conditional visit with one continuation for each Boolean result.

    Parameters
    ----------
    visit : BernoulliVisit
        The checked conditional draw.
    no, yes : ExactJvp or BernoulliBranch
        What the source does after false and true, respectively.
    """

    visit: BernoulliVisit
    no: ExactJvp | BernoulliBranch
    yes: ExactJvp | BernoulliBranch


type _Tree = ExactJvp | BernoulliBranch


@dataclass(frozen=True, slots=True)
class _WeakArm:
    """An individual weak branch to visit at its actual launch time.

    Parameters
    ----------
    prefix : tuple[bool, ...]
        The original choice history of the arm.
    tree : _Tree
        The conditional continuation to execute.
    """

    prefix: tuple[bool, ...]
    tree: _Tree


@dataclass(frozen=True, slots=True)
class BernoulliGradientPlan:
    """A finite addressed tree and its expected return and derivative.

    Parameters
    ----------
    computation : ComputationId
        The checked source computation being differentiated.
    exact : ExactJvp
        The exact expected return and its directional derivative,
        computed by enumerating every source trace.
    policies : tuple[SiteEstimator, ...]
        The explicit policies, ordered by static site identity.
    visits : tuple[BernoulliVisit, ...]
        The adaptive occurrences in the exhaustive trace tree.
    paths : int
        The number of complete source traces in that tree.
    source_steps : int
        The total measured reference-machine steps used to build the tree.
    tree : ExactJvp or BernoulliBranch
        The checked tree supplying both the exact and sampled results.
    """

    computation: ComputationId
    exact: ExactJvp
    policies: tuple[SiteEstimator, ...]
    visits: tuple[BernoulliVisit, ...]
    paths: int
    source_steps: int
    tree: ExactJvp | BernoulliBranch = field(repr=False)

    @property
    def replays(self) -> int:
        """Count the checked reference runs used during tree construction.

        Returns
        -------
        int
            The number of finished or choice-suspended replays.
        """
        return self.paths + len(self.visits)

    def sample(self, *, seed: int | None = None) -> BernoulliGradientSample:
        """Run the chosen mixture of enumeration, score and weak derivatives.

        The result estimates the same objective as `exact`. The draw list
        records only the score choices. The weak selector and virtual
        branch lists report the effects of the weak derivative separately.
        The execution does not rerun the original probabilistic program.
        Exact rational score weights may be large near the boundary.

        Parameters
        ----------
        seed : int or None
            An optional seed for the estimator's independent random draws.

        Returns
        -------
        BernoulliGradientSample
            A value, derivative estimate, and ordered stochastic choices.
        """
        return _draw(self.tree, Random(seed))


def _combine(visit: BernoulliVisit, no: ExactJvp, yes: ExactJvp) -> ExactJvp:
    """Apply the product rule to one conditional expectation.

    Parameters
    ----------
    visit : BernoulliVisit
        The conditional true probability and its tangent.
    no, yes : ExactJvp
        The values and derivatives from the two conditional branches.

    Returns
    -------
    ExactJvp
        The weighted value and its derivative.
    """
    p = visit.probability
    value = (1 - p) * no.value + p * yes.value
    derivative = (
        (1 - p) * no.tangent + p * yes.tangent + visit.tangent * (yes.value - no.value)
    )
    return ExactJvp(value, derivative)


def _assemble_finite_tree(
    records: dict[tuple[bool, ...], ExactJvp | BernoulliVisit],
) -> tuple[_Tree, ExactJvp]:
    """Assemble the checked finite tree and integrate its signed return.

    Parameters
    ----------
    records : dict
        Conditional requests and completed leaves, parents before children.

    Returns
    -------
    tuple
        The finite estimator tree and its exact expectation and derivative.
    """
    trees: dict[tuple[bool, ...], _Tree] = {}
    exact: dict[tuple[bool, ...], ExactJvp] = {}
    for prefix in reversed(records):
        result = records[prefix]
        if isinstance(result, ExactJvp):
            trees[prefix] = exact[prefix] = result
        else:
            no, yes = (*prefix, False), (*prefix, True)
            trees[prefix] = BernoulliBranch(result, trees.pop(no), trees.pop(yes))
            expected = _combine(result, exact.pop(no), exact.pop(yes))
            # Reject a plan during compilation, never a rare score outcome.
            _bound(expected.value, expected.tangent)
            exact[prefix] = expected
    return trees[()], exact[()]


def _draw(tree: _Tree, random: Random) -> BernoulliGradientSample:
    """Interpret the three estimator policies on one checked tree.

    Parameters
    ----------
    tree : _Tree
        The exhaustive conditional tree.
    random : Random
        Independent uniform integer randomness used for exact rational draws.

    Returns
    -------
    BernoulliGradientSample
        A realization of the mixed estimator.
    """

    def choose(visit: BernoulliVisit) -> bool:
        """Draw the Boolean outcome at one stochastic policy visit.

        Parameters
        ----------
        visit : BernoulliVisit
            The reached score or measure occurrence.

        Returns
        -------
        bool
            An exact Bernoulli choice under its conditional probability.
        """
        p = visit.probability
        return random.randrange(p.denominator) < p.numerator

    return _interpret(tree, choose)


def _interpret(
    tree: _Tree, choose: Callable[[BernoulliVisit], bool]
) -> BernoulliGradientSample:
    """Interpret the finite policies using the supplied selector outcomes.

    Parameters
    ----------
    tree : _Tree
        The checked conditional tree.
    choose : Callable
        The action for the next actual score draw or weak selector.

    Returns
    -------
    BernoulliGradientSample
        The mixed estimator driven by exactly those selector outcomes.
    """
    pending: list[_Tree | _WeakArm | tuple[BernoulliVisit, bool | None]] = [tree]
    values: list[ExactJvp] = []
    visits: list[BernoulliVisit] = []
    draws: list[BernoulliScoreDraw] = []
    selectors: list[BernoulliWeakSelector] = []
    virtual_branches: list[tuple[bool, ...]] = []
    while pending:
        next_ = pending.pop()
        if isinstance(next_, ExactJvp):
            values.append(next_)
        elif isinstance(next_, _WeakArm):
            virtual_branches.append(next_.prefix)
            pending.append(next_.tree)
        elif isinstance(next_, BernoulliBranch):
            visit = next_.visit
            visits.append(visit)
            if visit.estimator == "enumerate":
                pending.extend(((visit, None), next_.yes, next_.no))
            else:
                answer = choose(visit)
                if visit.estimator == "measure":
                    selectors.append(BernoulliWeakSelector(visit, answer))
                    no = _WeakArm((*visit.prefix, False), next_.no)
                    yes = _WeakArm((*visit.prefix, True), next_.yes)
                    pending.extend(((visit, answer), yes, no))
                else:
                    draws.append(BernoulliScoreDraw(visit, answer))
                    pending.extend(((visit, answer), next_.yes if answer else next_.no))
        else:
            visit, answer = next_
            result = values.pop()
            if answer is None:
                values.append(_combine(visit, values.pop(), result))
            elif visit.estimator == "measure":
                no, yes = values.pop(), result
                selected = yes if answer else no
                weak = visit.tangent * (yes.value - no.value)
                values.append(ExactJvp(selected.value, selected.tangent + weak))
            else:
                probability = visit.probability if answer else 1 - visit.probability
                tangent = visit.tangent if answer else -visit.tangent
                gradient = result.tangent + result.value * tangent / probability
                values.append(ExactJvp(result.value, gradient))
    (result,) = values
    return BernoulliGradientSample(
        result.value,
        result.tangent,
        tuple(visits),
        tuple(draws),
        tuple(selectors),
        tuple(virtual_branches),
    )


def _probability_argument(value: DistributionValue) -> Value:
    """Require the admitted nominal scalar Bernoulli parameterization.

    Parameters
    ----------
    value : DistributionValue
        The source constructor of the sampleable.

    Returns
    -------
    Value
        The scalar probability operand.

    Raises
    ------
    JvpUnsupported
        If the constructor is unspecialized, plated, or has other parameters.
    """
    if (
        value.family != FAMILIES["Bernoulli"].id
        or value.name != "Bernoulli"
        or not value.plate.empty
        or tuple(name for name, _ in value.arguments) != ("probs",)
    ):
        raise JvpUnsupported("site estimation requires unplated Bernoulli(probs=...)")
    return value.arguments[0][1]


def _check_request(request: EffectRequest, instance: EffectInstanceId) -> None:
    """Require a direct checked sample on the selected nominal instance.

    Parameters
    ----------
    request : EffectRequest
        The source operation being checked.
    instance : EffectInstanceId
        The unique Random instance opened by the selected computation.

    Raises
    ------
    JvpUnsupported
        If it is not a directly constructed Boolean Bernoulli sample.
    """
    if (
        request.instance != instance
        or request.effect != RANDOM
        or request.operation != RANDOM_SAMPLE
        or request.result_type != BOOL
        or request.static_arguments != (BOOL,)
    ):
        raise JvpUnsupported(
            "site estimation requires Random.sample[Bool] on one instance"
        )
    site, sampleable = request.arguments
    if not isinstance(site, SiteValue) or not isinstance(sampleable, DistributionValue):
        raise JvpUnsupported(
            "site estimation requires a direct site and Bernoulli constructor"
        )
    _probability_argument(sampleable)


class _NeedChoice(Exception):
    """A replay reached the next unresolved conditional draw.

    Parameters
    ----------
    visit : BernoulliVisit
        The pending request and its exact conditional probability.
    """

    def __init__(self, visit: BernoulliVisit) -> None:
        super().__init__(visit.address)
        self.visit = visit


class _BernoulliEvaluator(_ScalarDerivativeEvaluator):
    """Evaluate one exact source path through the reference machine.

    Parameters
    ----------
    module : QiecModule
        The validated module supplying its actual callees.
    fuel : int
        Maximum evaluator steps during each replay.
    prefix : tuple[bool, ...]
        The known choices in this replay.
    methods : Mapping[SiteId, _Estimator]
        The per-static-site policies.
    """

    def __init__(
        self,
        module: QiecModule,
        fuel: int,
        prefix: tuple[bool, ...],
        methods: Mapping[SiteId, _Estimator],
    ) -> None:
        super().__init__(module=module, fuel=fuel)
        self.prefix = prefix
        self.methods = methods
        self.seen: set[_Address] = set()

    def _value(self, value: Value, environment: Mapping[Local, object]) -> object:
        """Evaluate exact operands including a Bernoulli's probability.

        Parameters
        ----------
        value : Value
            The admitted expression being evaluated.
        environment : Mapping[Local, object]
            Its checked runtime scope.

        Returns
        -------
        object
            The normal exact operand, site label, or dual probability.

        Raises
        ------
        JvpUnsupported
            If a reached distribution has no admitted probability.
        """
        if isinstance(value, SiteValue):
            return value.label
        if isinstance(value, DistributionValue):
            return self._value(_probability_argument(value), environment)
        return super()._value(value, environment)

    def _dispatch(
        self,
        request: RuntimeRequest,
        environment: Scope[Local, HeldValue],
        stack: list[_Frame],
        search_index: int,
    ) -> _Continue:
        """Return a replayed choice or request the next finite branch.

        Parameters
        ----------
        request : RuntimeRequest
            The already evaluated, canonically addressed request.
        environment : Scope[Local, HeldValue]
            The scope to continue into after its Boolean answer.
        stack : list[_Frame]
            The reference machine's continuation, preserved intact.
        search_index : int
            The reference handler search position. This fragment has no
            installed handler on the selected instance.

        Returns
        -------
        _Continue
            The reference machine state after a recorded choice.

        Raises
        ------
        JvpUnsupported
            If a probability is inadmissible, a dynamic address repeats,
            or a reached site has no declared policy.
        _NeedChoice
            If this is the first request beyond the replay prefix.
        """
        address = request.address
        position = len(self.seen)
        if address in self.seen:
            raise JvpUnsupported(
                f"repeated Bernoulli address along a trace {address!r}"
            )
        self.seen.add(address)
        _, p = request.arguments
        if not isinstance(p, _Dual) or not 0 < p.value < 1:
            raise JvpUnsupported(
                "conditional Bernoulli probabilities must be strictly between zero and one"
            )
        static_site = request.core.origin.static_site
        method = self.methods.get(static_site)
        if method is None:
            raise JvpUnsupported(f"missing estimator for reached site {static_site}")
        if position == len(self.prefix):
            raise _NeedChoice(
                BernoulliVisit(
                    static_site, address, self.prefix, method, p.value, p.tangent
                )
            )
        value = self.numbers.number("truth", self.prefix[position])
        return _Continue(
            current=_Returned(value=value),
            environment=environment,
            resumption_path=request.resumption_path,
        )


def plan_bernoulli_gradient(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    policies: tuple[SiteEstimator, ...],
    *,
    max_paths: int = 1_024,
    fuel: int = 65_536,
) -> BernoulliGradientPlan:
    """Build a checked, bounded per-site Bernoulli estimator.

    The entry must take scalar reals and return a scalar real with one
    nominal Random instance in its row. The fragment admits ordinary
    sequencing, conditionals and calls and direct scalar Boolean draws
    from Bernoulli probabilities. Every reached site needs one explicit
    `enumerate`, `score` or `measure` policy. The conditional probabilities must
    be strictly interior, and comparisons must be locally stable under
    the exact scalar derivative rules.

    The plan exhausts every source choice at the supplied point to get
    the exact expected value and directional derivative. Sampling its
    chosen estimator is a separate operation with its own draw record.

    Parameters
    ----------
    module : QiecModule
        The checked source compilation unit.
    computation : ComputationId
        The named computation whose expected return is differentiated.
    arguments : tuple[int or Fraction, ...]
        Exact parameter values in source order.
    direction : tuple[int or Fraction, ...]
        Their exact directional tangents.
    policies : tuple[SiteEstimator, ...]
        One method declaration for every reached static site.
    max_paths : int
        The maximum number of complete source traces.
    fuel : int
        The maximum reference evaluator steps for each replay.

    Returns
    -------
    BernoulliGradientPlan
        The exhaustive checked expectation and sampled estimator.

    Raises
    ------
    JvpUnsupported
        If the source, a probability, or its estimator policy is outside
        the finite first-order fragment.
    JvpLimit
        If exhaustion requires too many traces or steps.
    ValueError
        If an input, declaration name, or resource limit is malformed.
    """
    for name, limit in (("max_paths", max_paths), ("fuel", fuel)):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    registry, owner = _random_computation(module, computation, "Bernoulli")
    if not len(arguments) == len(direction) == len(owner.parameters):
        raise ValueError("arguments and direction must match the computation's arity")
    methods: dict[SiteId, _Estimator] = {}
    for policy in policies:
        if policy.site in methods:
            raise ValueError(f"duplicate estimator for site {policy.site}")
        if policy.estimator not in ("enumerate", "score", "measure"):
            raise JvpUnsupported(
                f"unsupported Bernoulli estimator {policy.estimator!r}"
            )
        methods[policy.site] = policy.estimator
    _check_random_graph(module, owner, "Bernoulli", _check_request)
    environment = {
        parameter: _dual(_fraction(value), _fraction(tangent), True)
        for parameter, value, tangent in zip(
            owner.parameters, arguments, direction, strict=True
        )
    }

    records: dict[tuple[bool, ...], ExactJvp | BernoulliVisit] = {}
    visits: list[BernoulliVisit] = []
    pending: list[tuple[bool, ...]] = [()]
    paths = 1
    source_steps = 0
    while pending:
        prefix = pending.pop()
        evaluator = _BernoulliEvaluator(module, fuel, prefix, methods)
        try:
            result = evaluator.evaluate_validated(owner.body, registry, environment)
        except _NeedChoice as choice:
            records[prefix] = choice.visit
            visits.append(choice.visit)
            paths += 1
            if paths > max_paths:
                raise JvpLimit(
                    f"Bernoulli enumeration exceeds {max_paths} complete paths"
                ) from None
            pending.extend(((*prefix, True), (*prefix, False)))
        except FuelExhaustedError as error:
            raise JvpLimit(
                f"Bernoulli replay exceeded its step budget of {fuel} at {prefix!r}"
            ) from error
        else:
            assert isinstance(result, _Dual) and len(evaluator.seen) == len(prefix)
            records[prefix] = ExactJvp(result.value, result.tangent)
        finally:
            source_steps += evaluator._steps
    reached = {visit.site for visit in visits}
    unused = sorted(str(site) for site in methods if site not in reached)
    if unused:
        raise JvpUnsupported(
            f"estimator policies name unreached sites: {', '.join(unused)}"
        )

    tree, exact = _assemble_finite_tree(records)
    ordered = tuple(sorted(policies, key=lambda policy: str(policy.site)))
    return BernoulliGradientPlan(
        computation, exact, ordered, tuple(visits), paths, source_steps, tree
    )


__all__ = [
    "BernoulliBranch",
    "BernoulliGradientPlan",
    "BernoulliGradientSample",
    "BernoulliScoreDraw",
    "BernoulliVisit",
    "BernoulliWeakSelector",
    "plan_bernoulli_gradient",
]
