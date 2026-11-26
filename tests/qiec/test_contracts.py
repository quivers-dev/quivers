"""Contract records: round trips, the composition law, orders and refusals."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from fractions import Fraction

import didactic.api as dx
import pytest

from quivers.qiec import contracts as c
from quivers.qiec.identifiers import ComputationId, FamilyId, SiteId
from quivers.qiec.serialization import (
    _NODES,
    SerializationError,
    dumps,
    from_data,
    loads,
    to_data,
)

ALGEBRA = c.ClaimSubject(kind="algebra", name="quivers.core.algebras.LogProbAlgebra")
PROOF = c.Proved(
    argument="docs/api/qiec/contracts.md#the-composition-law",
    sources=(
        c.Citation(
            reference="Li and Eisner 2009, First- and second-order expectation semirings",
            locator="section 3",
            doi="10.3115/1699510.1699517",
        ),
    ),
)
TEST = c.Tested(check="tests/test_algebra_laws.py", cases=10, domain="float64 grid")
DECIDED = c.Decided(
    procedure="exhaustive", domain="the Boolean carrier", check="tests/test_x.py"
)
ASSUMED = c.Assumed(declared_by="a user", reason="a learned operation")
HALF = c.Rational.of(Fraction(1, 2))
QUARTER = c.Rational.of(Fraction(1, 4))
ONE = c.Rational.of(1)
LOG = c.AbsoluteLogError()
P1 = c.ParameterId.derive("rule", "a")
P2 = c.ParameterId.derive("rule", "b")
OCC = c.OccurrenceId.derive("axiom", 0)
SITE_A = SiteId.derive("site", "a")
SITE_B = SiteId.derive("site", "b")


def claim(law: c.Law, evidence: c.LawEvidence = PROOF) -> c.LawClaim:
    """A claim about the algebra.

    Parameters
    ----------
    law : Law
        The law.
    evidence : LawEvidence
        Its evidence.

    Returns
    -------
    LawClaim
        The claim.
    """
    return c.LawClaim(law=law, subject=ALGEBRA, evidence=evidence)


CLAIMS = tuple(
    claim(law)
    for law in (
        *c.SEMIRING_LAWS,
        c.TIMES_COMMUTATIVE,
        c.PLUS_SELECTIVE,
        c.SUPERIOR,
        c.ZERO_SUM_FREE,
        c.NO_ZERO_DIVISORS,
        c.STAR_LEAST_SOLUTION,
        c.OMEGA_CONTINUOUS,
        c.DIVISION,
        c.SEMIMODULE_ACTION,
    )
)


def query(
    *,
    observable: c.Observable = c.Weight(),
    differentiate_with_respect_to: tuple[c.ParameterId | c.OccurrenceId, ...] = (),
    batch_axes: tuple[str, ...] = (),
    retention: c.Retention = c.NoRetention(),
) -> c.QuerySpec:
    """A query of the test presentation.

    Parameters
    ----------
    observable : Observable
        What it observes.
    differentiate_with_respect_to : tuple of ParameterId or OccurrenceId
        The inputs of its gradient.
    batch_axes : tuple of str
        Its batch axes.
    retention : Retention
        The provenance it keeps.

    Returns
    -------
    QuerySpec
        The query.
    """
    return c.QuerySpec(
        presentation=c.PresentationId.derive("g"),
        interpretation=c.InterpretationId.derive("i"),
        observable=observable,
        differentiate_with_respect_to=differentiate_with_respect_to,
        batch_axes=batch_axes,
        retention=retention,
    )


def contract(
    *,
    guarantee: c.Guarantee = c.Exact(),
    gradient: c.DerivativeSemantics = c.NOT_DIFFERENTIATED,
    numerics: c.Numerics = c.Numerics(dtype="float64", domain="log"),
    controls: tuple[c.Control, ...] = (),
    budget: c.ResourceBudget = c.UNLIMITED,
    on_failure: c.FailurePolicy = c.FailurePolicy.REFUSE,
) -> c.AccuracyContract:
    """A contract on the goal weight in the absolute log metric.

    Parameters
    ----------
    guarantee : Guarantee
        The requested guarantee.
    gradient : DerivativeSemantics
        The requested derivative.
    numerics : Numerics
        The precision.
    controls : tuple of Control
        Method parameters.
    budget : ResourceBudget
        Limits on work.
    on_failure : FailurePolicy
        The failure policy.

    Returns
    -------
    AccuracyContract
        The contract.
    """
    return c.AccuracyContract(
        numerics=numerics,
        target=c.Target(quantity="goal weight", metric=LOG),
        guarantee=guarantee,
        controls=controls,
        budget=budget,
        on_failure=on_failure,
        gradient=gradient,
    )


def codes(found: tuple[c.Refusal, ...]) -> set[c.RefusalCode]:
    """The kinds of some refusals.

    Parameters
    ----------
    found : tuple of Refusal
        The refusals.

    Returns
    -------
    set of RefusalCode
        Their codes.
    """
    return {item.code for item in found}


def refusals(
    spec: c.QuerySpec,
    recursion: c.Recursion,
    accuracy: c.AccuracyContract,
    claims: tuple[c.LawClaim, ...] = CLAIMS,
) -> set[c.RefusalCode]:
    """The refusals of a combination.

    Parameters
    ----------
    spec : QuerySpec
        The query.
    recursion : Recursion
        Its recursion.
    accuracy : AccuracyContract
        Its contract.
    claims : tuple of LawClaim
        The claims.

    Returns
    -------
    set of RefusalCode
        Their codes.
    """
    return codes(
        c.admissibility(spec, recursion, accuracy, algebra=ALGEBRA, claims=claims)
    )


# ---------------------------------------------------------------------------
# round trips


def _examples() -> list[dx.Model]:
    """One instance of every serializable contract record.

    Returns
    -------
    list of didactic.api.Model
        The instances.
    """
    estimated = c.DerivativeSemantics.estimated(
        c.SiteEstimator(site=SITE_B, estimator="score"),
        c.SiteEstimator(site=SITE_A, estimator="reparam"),
        bias=c.BiasBound(bound=QUARTER),
    )
    entries = (
        c.DerivativeEntry(output="z", wrt=P2, semantics=estimated),
        c.DerivativeEntry(output="z", wrt=OCC, semantics=estimated),
    )
    enclosure = c.Enclosure(lower=QUARTER, upper=HALF, conditions=("outward",))
    composition = c.Composition(
        operation="logsumexp",
        terms=(c.ComposedTerm(lipschitz=ONE, bound=QUARTER),),
        conditions=("x",),
    )
    work = c.WorkReport(
        phase=c.Phase.EXECUTION, scalar_sums=4, seconds=c.Rational.of(0.5)
    )
    report = c.AccuracyReport(
        attainment=c.Attainment(
            guarantee=c.Certified(bound=QUARTER, certificate=enclosure),
            metric=LOG,
            rounding=c.ZERO,
            derivative=estimated,
        ),
        stop=c.ControlReached(control="tolerance", value=c.Rational.of(0.001)),
        unmet=(c.ContractField.GRADIENT,),
        derivatives=c.DerivativeReport(entries=entries),
        work=(work,),
        claims=(claim(c.PLUS_COMMUTATIVE, DECIDED),),
    )
    best = c.BestKnown(
        sources=(c.Citation(reference="Eisner 2023", locator="section 6"),)
    )
    return [
        c.Rational.of(Fraction(-3, 7)),
        c.Citation(reference="r", locator="l"),
        c.Law(name="n", statement="s"),
        ALGEBRA,
        PROOF,
        DECIDED,
        TEST,
        ASSUMED,
        claim(c.PLUS_SELECTIVE, ASSUMED),
        c.Requirement(law=c.DIVISION, subject=ALGEBRA),
        c.SPECTRAL_RADIUS_BELOW_ONE,
        c.FIRST_MAXIMIZER,
        c.Unbiased(),
        c.BiasBound(bound=HALF),
        c.BiasUnknown(),
        c.SiteEstimator(site=SITE_A, estimator="enumerate"),
        c.DerivativeSemantics(
            almost_everywhere=True,
            tie_rule=c.FIRST_MAXIMIZER,
            implicit_conditions=(c.SPECTRAL_RADIUS_BELOW_ONE,),
            unrolled=(5, 2),
        ),
        c.DerivativeSemantics.none("not requested"),
        entries[0],
        c.DerivativeReport(entries=entries),
        c.AbsoluteError(),
        LOG,
        c.RelativeError(),
        c.TotalVariation(),
        c.KLDivergence(direction="target-to-computed"),
        c.ObjectiveGap(),
        enclosure,
        c.DualityGap(primal=QUARTER, dual=HALF),
        c.Argument(value=QUARTER, claim=claim(c.Law(name="rate", statement="q"))),
        c.ComposedTerm(lipschitz=ONE, bound=HALF),
        composition,
        c.Exact(),
        c.Certified(bound=HALF, conditions=("a",), certificate=composition),
        c.Probabilistic(epsilon=HALF, delta=QUARTER, assumptions=("iid",)),
        c.Heuristic(reason="beam"),
        c.Numerics(dtype="float32", rounding=c.Rational.of(2.0**-20)),
        c.Target(quantity="q", metric=c.KLDivergence(direction="computed-to-target")),
        c.Control(name="beam", value=c.Rational.of(8)),
        c.Control(name="method", value="newton"),
        c.ResourceBudget(iterations=10, bytes=1024, seconds=HALF),
        contract(
            guarantee=c.Probabilistic(epsilon=HALF, delta=QUARTER),
            controls=(c.Control(name="particles", value=c.Rational.of(100)),),
            on_failure=c.FailurePolicy.RETURN_UNESTABLISHED,
            gradient=estimated,
        ),
        c.Acyclic(),
        c.LinearCyclic(),
        c.MonotonePolynomial(),
        c.Bounded(measure="length", limit=12),
        c.Unrolled(iterations=7),
        c.NoRetention(),
        c.ParentCounts(),
        c.JudgmentPattern(family=FamilyId.derive("f")),
        c.RequestedEvidence(patterns=(c.JudgmentPattern(family=FamilyId.derive("f")),)),
        c.PackedForest(),
        c.Support(),
        c.Weight(),
        c.Best(),
        c.KBest(k=3),
        c.Outcome(key=ComputationId.derive("key")),
        c.Statistic(statistic=ComputationId.derive("stat")),
        c.Marginal(normalized=True),
        c.Sample(n=4),
        c.Fold(fold=ComputationId.derive("fold")),
        c.NonlinearFold(fold=ComputationId.derive("square")),
        c.Forest(),
        query(
            differentiate_with_respect_to=(P2, OCC, P1),
            batch_axes=("batch",),
            retention=c.PackedForest(),
        ),
        c.ALGEBRA_OPERATIONS,
        c.ComplexityParameter(name="V", meaning="items"),
        c.ProvedOptimal(
            lower_bound="V", model=c.ALGEBRA_OPERATIONS, sources=best.sources
        ),
        c.Conditional(hypothesis="APSP", sources=best.sources),
        best,
        c.Unresolved(obligation="close the gap"),
        c.Deficient(
            better_bound="V", justification="j", reviewed_by="r", sources=best.sources
        ),
        c.ComplexityBound(
            phase=c.Phase.REFIT, resource="space", expression="V", status=best
        ),
        c.PhaseInapplicable(phase=c.Phase.REFIT, reason="none"),
        c.ComplexityClaim(
            query="Q1",
            parameters=(c.ComplexityParameter(name="V", meaning="items"),),
            model=c.WORD_RAM_EXPECTED_HASHING,
            bounds=(),
        ),
        work,
        report.attainment,
        c.ArgumentBound(metric=LOG, lipschitz=ONE, affine=True),
        c.OperationBound(name="op", arguments=(), metric=LOG, rounding=c.ZERO),
        c.Completed(),
        c.BudgetExhausted(resource="iterations", limit=c.Rational.of(10)),
        report.stop,
        report,
        c.Refusal(code=c.RefusalCode.MISSING_CLAIM, reason="r"),
    ]


EXAMPLES = _examples()


def test_every_contract_record_is_on_the_wire_and_has_an_example() -> None:
    """Require the codec to carry exactly the contract records exemplified."""
    on_wire = {cls for cls in _NODES.values() if cls.__module__ == c.__name__}
    assert on_wire == {type(example) for example in EXAMPLES}


@pytest.mark.parametrize("record", EXAMPLES, ids=lambda r: type(r).__name__)
def test_record_round_trips(record: dx.Model) -> None:
    """Require equality and stable bytes after a round trip.

    Parameters
    ----------
    record : didactic.api.Model
        The record.
    """
    text = dumps(record)
    decoded = loads(text)
    assert decoded == record
    assert type(decoded) is type(record)
    assert dumps(decoded) == text


def test_identities_keep_their_order_and_class() -> None:
    """Require the inputs of a gradient to keep order and identity."""
    spec = query(differentiate_with_respect_to=(P2, OCC, P1))
    decoded = loads(dumps(spec))
    assert isinstance(decoded, c.QuerySpec)
    assert decoded.differentiate_with_respect_to == (P2, OCC, P1)
    assert [type(item) for item in decoded.differentiate_with_respect_to] == [
        c.ParameterId,
        c.OccurrenceId,
        c.ParameterId,
    ]


def test_identifier_namespaces_are_checked() -> None:
    """Require an occurrence identity not to decode as a parameter."""
    data = to_data(P1)
    root = data["root"]
    assert isinstance(root, dict)
    root["value"] = str(OCC)
    with pytest.raises(SerializationError):
        from_data(data)


def test_runtime_wrapper_does_not_cross_the_wire() -> None:
    """Require an unestablished value to stay off the wire."""
    report = EXAMPLES[-2]
    assert isinstance(report, c.AccuracyReport)
    with pytest.raises(SerializationError):
        dumps(c.Unestablished(value=1.5, report=report))


def test_a_rational_not_in_lowest_terms_is_refused_on_decode() -> None:
    """Require the decoder to apply the record's own validation."""
    text = dumps(HALF)
    tampered = text.replace('"value":1}', '"value":2}').replace(
        '"value":2},{"name":"tag"', '"value":4},{"name":"tag"'
    )
    tampered = tampered.replace(
        '{"name":"denominator","value":2}', '{"name":"denominator","value":4}'
    )
    assert tampered != text
    with pytest.raises(SerializationError):
        loads(tampered)


# ---------------------------------------------------------------------------
# single records


def test_rationals_are_exact() -> None:
    """Require exact conversion and refusal of what is not a finite number."""
    assert c.Rational.of(0.1).fraction == Fraction(0.1)
    assert c.Rational.of(0.1).fraction != Fraction(1, 10)
    with pytest.raises(ValueError):
        c.Rational.of(math.inf)
    with pytest.raises(ValueError):
        c.Rational.of(True)
    with pytest.raises(dx.ValidationError):
        c.Rational(numerator=2, denominator=4)
    with pytest.raises(dx.ValidationError):
        c.Rational(numerator=1, denominator=0)


@pytest.mark.parametrize(
    "build",
    [
        lambda: c.Probabilistic(epsilon=HALF, delta=c.ZERO),
        lambda: c.Probabilistic(epsilon=HALF, delta=ONE),
        lambda: c.Certified(bound=c.Rational.of(-1)),
        lambda: c.BiasBound(bound=c.ZERO),
        lambda: c.DerivativeSemantics(missing=("x",), almost_everywhere=True),
        lambda: c.DerivativeSemantics(bias=c.BiasUnknown()),
        lambda: c.DerivativeSemantics(unrolled=(0,)),
        lambda: c.DerivativeSemantics(
            estimators=(
                c.SiteEstimator(site=SITE_A, estimator="score"),
                c.SiteEstimator(site=SITE_A, estimator="reparam"),
            )
        ),
        lambda: c.Enclosure(lower=HALF, upper=QUARTER),
        lambda: c.KBest(k=0),
        lambda: c.Sample(n=0),
        lambda: c.Unrolled(iterations=0),
        lambda: c.Bounded(measure="length", limit=-1),
        lambda: c.Tested(check="t", cases=0, domain="d"),
        lambda: c.Heuristic(reason=" "),
        lambda: c.ResourceBudget(iterations=-1),
        lambda: c.WorkReport(phase=c.Phase.EXECUTION, scalar_sums=-1),
        lambda: c.RequestedEvidence(patterns=()),
        lambda: query(differentiate_with_respect_to=(P1, P1)),
        lambda: c.ComplexityParameter(name="log", meaning="m"),
        lambda: c.ComplexityParameter(name="2n", meaning="m"),
        lambda: c.Deficient(
            better_bound="V", justification="j", reviewed_by="", sources=()
        ),
        lambda: c.ProvedOptimal(
            lower_bound="V", model=c.ALGEBRA_OPERATIONS, sources=()
        ),
    ],
)
def test_malformed_records_are_refused(build: Callable[[], dx.Model]) -> None:
    """Require every malformed single record to be refused at construction.

    Parameters
    ----------
    build : Callable[[], didactic.api.Model]
        A function that builds the record.
    """
    with pytest.raises((dx.ValidationError, ValueError)):
        build()


def test_status_is_read_from_the_evidence() -> None:
    """Require the status and its strength to follow the evidence."""
    statuses = {
        evidence.tag: claim(c.PLUS_UNIT, evidence)
        for evidence in (PROOF, DECIDED, TEST, ASSUMED)
    }
    assert {tag: item.status.value for tag, item in statuses.items()} == {
        "proved": "proved",
        "decided": "decided",
        "tested": "tested",
        "assumed": "assumed",
    }
    assert [item.established for item in statuses.values()] == [
        True,
        True,
        False,
        False,
    ]


def test_metrics_that_fail_the_triangle_inequality() -> None:
    """Require the counterexamples stated for the relative error and KL."""

    def relative(a: float, b: float) -> float:
        """The relative error of a computed value against a target.

        Parameters
        ----------
        a : float
            The computed value.
        b : float
            The target.

        Returns
        -------
        float
            ``|a - b| / |b|``.
        """
        return abs(a - b) / abs(b)

    assert relative(101, 1) > relative(101, 100) + relative(100, 1)

    def kl(p: float, q: float) -> float:
        """The divergence between two Bernoulli laws.

        Parameters
        ----------
        p : float
            The mean of the first law.
        q : float
            The mean of the second law.

        Returns
        -------
        float
            ``D(Bernoulli(p) || Bernoulli(q))``.
        """
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    assert round(kl(0.1, 0.9), 3) == 1.758
    assert round(kl(0.1, 0.5), 3) == 0.368
    assert round(kl(0.5, 0.9), 3) == 0.511
    assert kl(0.1, 0.9) > kl(0.1, 0.5) + kl(0.5, 0.9)
    assert not c.RelativeError().triangle_inequality
    assert not c.KLDivergence(direction="target-to-computed").triangle_inequality
    assert c.AbsoluteLogError().triangle_inequality


# ---------------------------------------------------------------------------
# the order of derivative semantics


def _semantics() -> list[c.DerivativeSemantics]:
    """A grid of derivative semantics covering every component.

    Returns
    -------
    list of DerivativeSemantics
        The grid.
    """
    other_rule = c.TieRule(name="equal-shares", statement="split")
    grid = [c.DerivativeSemantics.none("x")]
    for ae, rule, implicit, unrolled, sites, bias in itertools.product(
        (False, True),
        (None, c.FIRST_MAXIMIZER, other_rule),
        ((), (c.SPECTRAL_RADIUS_BELOW_ONE,)),
        ((), (3,), (3, 3)),
        ((), (c.SiteEstimator(site=SITE_A, estimator="score"),)),
        (c.Unbiased(), c.BiasBound(bound=HALF), c.BiasUnknown()),
    ):
        if sites == () and not isinstance(bias, c.Unbiased):
            continue
        grid.append(
            c.DerivativeSemantics(
                almost_everywhere=ae,
                tie_rule=rule,
                implicit_conditions=implicit,
                unrolled=unrolled,
                estimators=sites,
                bias=bias,
            )
        )
    return grid


GRID = _semantics()


def test_at_least_is_a_preorder_with_none_at_the_bottom() -> None:
    """Require reflexivity, transitivity, and no derivative as the weakest."""
    for a in GRID:
        assert a.at_least(a)
        assert a.at_least(GRID[0])
    for a, b, d in itertools.product(GRID[::3], GRID[::2], GRID[::3]):
        if a.at_least(b) and b.at_least(d):
            assert a.at_least(d), (a, b, d)


def test_meet_is_the_greatest_lower_bound() -> None:
    """Require the meet below both, and above everything below both."""
    for a, b in itertools.product(GRID[::4], GRID[::3]):
        try:
            m = a.meet(b)
        except ValueError:
            assert a.tie_rule is not None and b.tie_rule is not None
            assert a.tie_rule != b.tie_rule
            continue
        assert a.at_least(m) and b.at_least(m)
        for z in GRID[::5]:
            if z.at_least(a) and z.at_least(b):
                assert z.at_least(m), (a, b, z)


def test_kind_names_the_weakest_form() -> None:
    """Require the named form to be the weakest component present."""
    assert c.DerivativeSemantics.exact().kind == "exact-at-regular-points"
    assert c.DerivativeSemantics.selected_branch(c.FIRST_MAXIMIZER).kind == (
        "selected-branch"
    )
    assert (
        c.DerivativeSemantics.implicit(c.SPECTRAL_RADIUS_BELOW_ONE).kind == "implicit"
    )
    mixed = c.DerivativeSemantics(
        unrolled=(2,), estimators=(c.SiteEstimator(site=SITE_A, estimator="score"),)
    )
    assert mixed.kind == "unrolled"
    assert c.NOT_DIFFERENTIATED.kind == "none"


def test_a_derivative_report_summarizes_by_the_meet() -> None:
    """Require the summary of a report to be the meet of its entries."""
    exact = c.DerivativeSemantics.exact(almost_everywhere=True)
    unrolled = c.DerivativeSemantics.unrolled_iterations(4)
    report = c.DerivativeReport(
        entries=(
            c.DerivativeEntry(output="y", wrt=P1, semantics=exact),
            c.DerivativeEntry(output="y", wrt=P2, semantics=unrolled),
        )
    )
    assert report.summary == exact.meet(unrolled)
    assert report.summary.kind == "unrolled"
    assert c.DerivativeReport().summary == c.NOT_DIFFERENTIATED
    with pytest.raises(dx.ValidationError):
        c.DerivativeReport(entries=(report.entries[0], report.entries[0]))


# ---------------------------------------------------------------------------
# the composition law


def part(
    guarantee: c.Guarantee,
    rounding: c.Rational | None = c.ZERO,
    derivative: c.DerivativeSemantics = c.NOT_DIFFERENTIATED,
    metric: c.Metric = LOG,
) -> c.Attainment:
    """One part of a composition.

    Parameters
    ----------
    guarantee : Guarantee
        Its guarantee.
    rounding : Rational or None
        Its rounding bound.
    derivative : DerivativeSemantics
        Its derivative.
    metric : Metric
        Its metric.

    Returns
    -------
    Attainment
        The part.
    """
    return c.Attainment(
        guarantee=guarantee, metric=metric, rounding=rounding, derivative=derivative
    )


def operation(
    *constants: c.Rational | None,
    metric: c.Metric = LOG,
    rounding: c.Rational | None = c.ZERO,
    affine: bool = False,
    derivative: c.DerivativeSemantics = c.NOT_DIFFERENTIATED,
) -> c.OperationBound:
    """An operation with given Lipschitz constants.

    Parameters
    ----------
    *constants : Rational or None
        One constant per argument.
    metric : Metric
        The metric of its result and arguments.
    rounding : Rational or None
        Its own rounding bound.
    affine : bool
        Whether it is affine in every argument.
    derivative : DerivativeSemantics
        Its own derivative.

    Returns
    -------
    OperationBound
        The operation.
    """
    return c.OperationBound(
        name="op",
        arguments=tuple(
            c.ArgumentBound(metric=metric, lipschitz=constant, affine=affine)
            for constant in constants
        ),
        metric=metric,
        rounding=rounding,
        derivative=derivative,
    )


def certified(bound: Fraction) -> c.Certified:
    """A certified bound shown by an enclosure.

    Parameters
    ----------
    bound : Fraction
        The width.

    Returns
    -------
    Certified
        The guarantee.
    """
    return c.Certified(
        bound=c.Rational.of(bound),
        certificate=c.Enclosure(lower=c.ZERO, upper=c.Rational.of(bound)),
    )


def test_exact_parts_give_an_exact_composite_with_added_rounding() -> None:
    """Require exactness and the rounding ``r + sum L_i r_i``."""
    eps = c.Rational.of(2.0**-52)
    composite = c.compose(
        (part(c.Exact(), eps), part(c.Exact(), eps)),
        operation(ONE, c.Rational.of(2), rounding=eps),
    )
    assert isinstance(composite.guarantee, c.Exact)
    assert composite.rounding is not None
    assert composite.rounding.fraction == 4 * eps.fraction
    unknown = c.compose(
        (part(c.Exact(), None), part(c.Exact(), eps)), operation(ONE, ONE)
    )
    assert isinstance(unknown.guarantee, c.Exact)
    assert unknown.rounding is None
    assert unknown.total_bound is None


def test_certified_bounds_add_with_their_constants() -> None:
    """Require ``sum L_i e_i`` and a certificate that shows it."""
    composite = c.compose(
        (
            part(certified(Fraction(1, 4))),
            part(c.Exact()),
            part(certified(Fraction(1, 8))),
        ),
        operation(ONE, c.Rational.of(5), c.Rational.of(2)),
    )
    guarantee = composite.guarantee
    assert isinstance(guarantee, c.Certified)
    assert guarantee.bound.fraction == Fraction(1, 4) + 2 * Fraction(1, 8)
    assert guarantee.certificate is not None
    assert guarantee.certificate.bound == guarantee.bound.fraction
    assert composite.total_bound == guarantee.bound.fraction


def test_probabilities_of_failure_add_and_can_become_vacuous() -> None:
    """Require the union bound, and a heuristic when it says nothing."""
    likely = c.Probabilistic(epsilon=HALF, delta=c.Rational.of(Fraction(3, 10)))
    composite = c.compose(
        (part(likely), part(certified(Fraction(1, 2)))), operation(ONE, ONE)
    )
    guarantee = composite.guarantee
    assert isinstance(guarantee, c.Probabilistic)
    assert guarantee.epsilon.fraction == 1
    assert guarantee.delta.fraction == Fraction(3, 10)
    two = c.compose((part(likely), part(likely)), operation(ONE, ONE))
    assert isinstance(two.guarantee, c.Probabilistic)
    assert two.guarantee.delta.fraction == Fraction(3, 5)
    four = c.compose((part(likely),) * 4, operation(ONE, ONE, ONE, ONE))
    assert isinstance(four.guarantee, c.Heuristic)


def test_local_tolerances_give_a_heuristic() -> None:
    """Require a heuristic part to make the composite heuristic."""
    tolerance = c.Heuristic(reason="residual below 1e-3")
    composite = c.compose(
        (part(tolerance), part(certified(Fraction(1, 4)))), operation(ONE, ONE)
    )
    assert isinstance(composite.guarantee, c.Heuristic)
    ignored = c.compose(
        (part(tolerance), part(certified(Fraction(1, 4)))), operation(c.ZERO, ONE)
    )
    assert isinstance(ignored.guarantee, c.Certified)


def test_missing_constants_and_other_metrics_give_a_heuristic() -> None:
    """Require a constant in the part's own metric for every bounded part."""
    bounded = part(certified(Fraction(1, 4)))
    assert isinstance(c.compose((bounded,), operation(None)).guarantee, c.Heuristic)
    absolute = part(certified(Fraction(1, 4)), metric=c.AbsoluteError())
    assert isinstance(c.compose((absolute,), operation(ONE)).guarantee, c.Heuristic)
    for metric in (c.RelativeError(), c.KLDivergence(direction="target-to-computed")):
        composite = c.compose(
            (part(certified(Fraction(1, 4)), metric=metric),),
            operation(ONE, metric=metric),
        )
        assert isinstance(composite.guarantee, c.Heuristic)
        assert composite.rounding is None
    with pytest.raises(ValueError):
        c.compose((bounded,), operation(ONE, ONE))


def test_a_small_residual_is_not_a_small_error() -> None:
    """Require the iteration stated in the documentation to stop far from its value."""
    weight, total, steps = 0.999, 1.0, 1
    while math.log(total + weight**steps) - math.log(total) >= 1e-3:
        total += weight**steps
        steps += 1
    stopped = math.log(total + weight**steps)
    assert steps == 693
    assert round(stopped, 4) == 6.2158
    assert round(math.log(1000), 4) == 6.9078


def test_the_bound_holds_for_log_sum_exp() -> None:
    """Require the composed bound to cover the error of log-sum-exp.

    Log-sum-exp is 1-Lipschitz in each argument in the absolute metric of
    log weights, so perturbations within certified bounds move it by at
    most their sum.
    """
    bound = c.compose(
        (part(certified(Fraction(1, 4))), part(certified(Fraction(1, 8)))),
        operation(ONE, ONE),
    ).guarantee
    assert isinstance(bound, c.Certified)
    limit = float(bound.bound.fraction)

    def lse(a: float, b: float) -> float:
        """The log of the sum of two exponentials.

        Parameters
        ----------
        a, b : float
            The log weights.

        Returns
        -------
        float
            ``log(exp(a) + exp(b))``, computed stably.
        """
        top = max(a, b)
        return top + math.log(math.exp(a - top) + math.exp(b - top))

    for a, b in itertools.product((-30.0, -1.0, 0.0, 2.5), (-4.0, 0.0, 7.0)):
        for da, db in itertools.product((-0.25, 0.0, 0.25), (-0.125, 0.125)):
            assert abs(lse(a + da, b + db) - lse(a, b)) <= limit + 1e-12


def test_composite_derivatives() -> None:
    """Require the weakest derivative, with unbiasedness only through affine maps."""
    unbiased = c.DerivativeSemantics.estimated(
        c.SiteEstimator(site=SITE_A, estimator="score")
    )
    exact = c.DerivativeSemantics.exact(almost_everywhere=True)
    through_sum = c.compose(
        (part(c.Exact(), derivative=unbiased), part(c.Exact(), derivative=exact)),
        operation(ONE, ONE, affine=True, derivative=exact),
    )
    assert isinstance(through_sum.derivative.bias, c.Unbiased)
    assert through_sum.derivative.estimators == unbiased.estimators
    assert not through_sum.derivative.almost_everywhere
    through_square = c.compose(
        (part(c.Exact(), derivative=unbiased),), operation(ONE, derivative=exact)
    )
    assert isinstance(through_square.derivative.bias, c.BiasUnknown)
    loops = c.compose(
        (
            part(c.Exact(), derivative=c.DerivativeSemantics.unrolled_iterations(5)),
            part(c.Exact(), derivative=c.DerivativeSemantics.unrolled_iterations(5)),
        ),
        operation(ONE, ONE, derivative=exact),
    )
    assert loops.derivative.unrolled == (5, 5)
    for piece, inputs in (
        (through_sum, (unbiased, exact)),
        (through_square, (unbiased,)),
        (loops, (c.DerivativeSemantics.unrolled_iterations(5),)),
    ):
        assert all(item.at_least(piece.derivative) for item in inputs)
    rules = (
        part(
            c.Exact(),
            derivative=c.DerivativeSemantics.selected_branch(c.FIRST_MAXIMIZER),
        ),
        part(
            c.Exact(),
            derivative=c.DerivativeSemantics.selected_branch(
                c.TieRule(name="equal-shares", statement="split")
            ),
        ),
    )
    with pytest.raises(ValueError):
        c.compose(rules, operation(ONE, ONE, derivative=exact))
    none = c.compose(
        (part(c.Exact(), derivative=exact),),
        operation(ONE, derivative=c.NOT_DIFFERENTIATED),
    )
    assert none.derivative.kind == "none"


# ---------------------------------------------------------------------------
# admissibility


@pytest.mark.parametrize(
    ("spec", "recursion", "accuracy"),
    [
        (query(), c.Acyclic(), contract()),
        (
            query(observable=c.Best(), differentiate_with_respect_to=(P1,)),
            c.Acyclic(),
            contract(gradient=c.DerivativeSemantics.selected_branch(c.FIRST_MAXIMIZER)),
        ),
        (query(observable=c.KBest(k=2)), c.LinearCyclic(), contract()),
        (query(observable=c.Support()), c.Acyclic(), contract()),
        (
            query(observable=c.Forest(), retention=c.PackedForest()),
            c.LinearCyclic(),
            contract(),
        ),
        (
            query(
                observable=c.NonlinearFold(fold=ComputationId.derive("sq")),
                retention=c.PackedForest(),
            ),
            c.Bounded(measure="height", limit=4),
            contract(),
        ),
        (
            query(
                observable=c.NonlinearFold(fold=ComputationId.derive("sq")),
                retention=c.PackedForest(),
            ),
            c.LinearCyclic(),
            contract(guarantee=c.Heuristic(reason="truncated enumeration")),
        ),
        (
            query(differentiate_with_respect_to=(P1, OCC)),
            c.MonotonePolynomial(),
            contract(
                guarantee=c.Certified(bound=QUARTER),
                gradient=c.DerivativeSemantics.implicit(c.SPECTRAL_RADIUS_BELOW_ONE),
            ),
        ),
        (
            query(differentiate_with_respect_to=(P1,)),
            c.Unrolled(iterations=10),
            contract(
                guarantee=c.Heuristic(reason="unrolled"),
                gradient=c.DerivativeSemantics.unrolled_iterations(10),
            ),
        ),
        (query(observable=c.Marginal(normalized=True)), c.Acyclic(), contract()),
        (query(observable=c.Sample(n=3)), c.Acyclic(), contract()),
        (
            query(observable=c.Statistic(statistic=ComputationId.derive("s"))),
            c.Acyclic(),
            contract(),
        ),
    ],
)
def test_admissible_combinations(
    spec: c.QuerySpec, recursion: c.Recursion, accuracy: c.AccuracyContract
) -> None:
    """Require admissible combinations to be admitted.

    Parameters
    ----------
    spec : QuerySpec
        The query.
    recursion : Recursion
        Its recursion.
    accuracy : AccuracyContract
        Its contract.
    """
    assert refusals(spec, recursion, accuracy) == set()
    c.check_admissible(spec, recursion, accuracy, algebra=ALGEBRA, claims=CLAIMS)


def test_a_fold_needs_its_own_linearity_claim() -> None:
    """Require the linearity claim of a fold, about the fold."""
    fold = ComputationId.derive("count")
    spec = query(observable=c.Fold(fold=fold), retention=c.PackedForest())
    assert refusals(spec, c.Acyclic(), contract()) == {c.RefusalCode.MISSING_CLAIM}
    linear = c.LawClaim(
        law=c.FOLD_LINEAR,
        subject=c.ClaimSubject(kind="computation", name=str(fold)),
        evidence=PROOF,
    )
    assert refusals(spec, c.Acyclic(), contract(), (*CLAIMS, linear)) == set()


def test_used_laws_need_claims_and_heuristics_need_no_establishment() -> None:
    """Require claims for further laws, and tested claims for heuristics only."""
    tested = tuple(claim(item.law, TEST) for item in CLAIMS)
    assert refusals(query(), c.Acyclic(), contract(), tested) == {
        c.RefusalCode.CLAIM_NOT_ESTABLISHED
    }
    heuristic = contract(guarantee=c.Heuristic(reason="beam"))
    assert refusals(query(), c.Acyclic(), heuristic, tested) == set()
    uses = (c.Requirement(law=c.PLUS_IDEMPOTENT, subject=ALGEBRA),)
    found = c.admissibility(
        query(), c.Acyclic(), contract(), algebra=ALGEBRA, claims=CLAIMS, uses=uses
    )
    assert codes(found) == {c.RefusalCode.MISSING_CLAIM}


def test_every_reason_is_reported_together() -> None:
    """Require one refusal per reason, raised together."""
    spec = query(observable=c.Support(), differentiate_with_respect_to=(P1,))
    with pytest.raises(c.InadmissibleContract) as caught:
        c.check_admissible(
            spec,
            c.MonotonePolynomial(),
            contract(gradient=c.DerivativeSemantics.exact()),
            algebra=ALGEBRA,
            claims=(),
        )
    assert codes(caught.value.refusals) == {
        c.RefusalCode.MISSING_CLAIM,
        c.RefusalCode.EXACT_WITHOUT_FINITE_METHOD,
        c.RefusalCode.DERIVATIVE_UNAVAILABLE,
        c.RefusalCode.OBSERVABLE_NOT_DIFFERENTIABLE,
    }


# ---------------------------------------------------------------------------
# reports and failure


def report(
    guarantee: c.Guarantee,
    accuracy: c.AccuracyContract,
    *,
    stop: c.StopReason = c.Completed(),
    work: tuple[c.WorkReport, ...] = (),
    rounding: c.Rational | None = c.ZERO,
) -> c.AccuracyReport:
    """A report whose unmet requirements are computed honestly.

    Parameters
    ----------
    guarantee : Guarantee
        The attained guarantee.
    accuracy : AccuracyContract
        The contract.
    stop : StopReason
        Why the run stopped.
    work : tuple of WorkReport
        Its work.
    rounding : Rational or None
        Its rounding bound.

    Returns
    -------
    AccuracyReport
        The report.
    """
    attainment = c.Attainment(guarantee=guarantee, metric=LOG, rounding=rounding)
    return c.AccuracyReport(
        attainment=attainment,
        stop=stop,
        unmet=c.unmet_fields(accuracy, attainment, work),
        work=work,
    )


def test_a_met_contract_returns_the_value() -> None:
    """Require the plain value when every requirement is met."""
    accuracy = contract()
    assert c.settle(accuracy, 2.5, report(c.Exact(), accuracy)) == 2.5


def test_an_unmet_contract_refuses_or_returns_unestablished() -> None:
    """Require the failure policy, and never a plain value."""
    refuse = contract()
    unmet = report(c.Heuristic(reason="beam"), refuse)
    assert unmet.unmet == (c.ContractField.GUARANTEE,)
    with pytest.raises(c.ContractNotMet):
        c.settle(refuse, 2.5, unmet)
    keep = contract(on_failure=c.FailurePolicy.RETURN_UNESTABLISHED)
    settled = c.settle(keep, 2.5, report(c.Heuristic(reason="beam"), keep))
    assert isinstance(settled, c.Unestablished)
    assert settled.value == 2.5
    with pytest.raises(TypeError):
        float(settled)  # type: ignore[arg-type]


def test_an_inconsistent_report_is_refused_by_settle() -> None:
    """Require settle to refuse a report that hides an unmet requirement."""
    accuracy = contract()
    attainment = c.Attainment(guarantee=c.Heuristic(reason="beam"), metric=LOG)
    hidden = c.AccuracyReport(attainment=attainment)
    with pytest.raises(c.InadmissibleReport):
        c.settle(accuracy, 2.5, hidden)


def test_work_beyond_the_budget_is_unmet() -> None:
    """Require exceeded limits to be reported, never read as a bound."""
    accuracy = contract(budget=c.ResourceBudget(iterations=5, scalar_work=10))
    work = (
        c.WorkReport(phase=c.Phase.CONSTRUCTION, iterations=3, scalar_sums=4),
        c.WorkReport(phase=c.Phase.EXECUTION, iterations=3, scalar_products=4),
    )
    assert report(c.Exact(), accuracy, work=work).unmet == (c.ContractField.BUDGET,)
    within = contract(budget=c.ResourceBudget(iterations=6, scalar_work=8))
    assert report(c.Exact(), within, work=work).unmet == ()


def test_rounding_and_gradient_requirements() -> None:
    """Require a known rounding within the contract's, and the derivative asked."""
    accuracy = contract(
        numerics=c.Numerics(dtype="float64", domain="log", rounding=QUARTER),
    )
    assert report(c.Exact(), accuracy, rounding=None).unmet == (
        c.ContractField.NUMERICS,
    )
    assert report(c.Exact(), accuracy, rounding=HALF).unmet == (
        c.ContractField.NUMERICS,
    )
    gradient = contract(gradient=c.DerivativeSemantics.exact())
    assert report(c.Exact(), gradient).unmet == (c.ContractField.GRADIENT,)


def test_an_early_stop_claims_only_its_certificate() -> None:
    """Require the certificate's bound after an early stop, and nothing else."""
    stop = c.ControlReached(control="tolerance", value=c.Rational.of(0.001))
    enclosure = c.Enclosure(lower=c.Rational.of(6.9), upper=c.Rational.of(6.91))
    guarantee = c.attained_guarantee(stop, c.Exact(), enclosure)
    assert isinstance(guarantee, c.Certified)
    assert guarantee.bound.fraction == enclosure.bound
    assert isinstance(c.attained_guarantee(stop, c.Exact(), None), c.Heuristic)
    assert isinstance(c.attained_guarantee(c.Completed(), c.Exact(), None), c.Exact)
    accuracy = contract(guarantee=c.Certified(bound=QUARTER))
    assert c.report_refusals(accuracy, report(guarantee, accuracy, stop=stop)) == ()


# ---------------------------------------------------------------------------
# complexity claims


def test_complexity_claims() -> None:
    """Require a complete claim to pass and an incomplete one to be refused."""
    best = c.BestKnown(sources=(c.Citation(reference="Eisner 2023", locator="s6"),))
    bounds = tuple(
        c.ComplexityBound(
            phase=phase, resource=resource, expression="V + E", status=best
        )
        for phase in (c.Phase.CONSTRUCTION, c.Phase.EXECUTION, c.Phase.DIFFERENTIATION)
        for resource in ("time", "space")
    )
    claim_record = c.ComplexityClaim(
        query="Q1",
        parameters=(
            c.ComplexityParameter(name="V", meaning="items"),
            c.ComplexityParameter(name="E", meaning="instances"),
        ),
        model=c.ALGEBRA_OPERATIONS,
        bounds=bounds,
        inapplicable=(c.PhaseInapplicable(phase=c.Phase.REFIT, reason="none"),),
    )
    c.check_complexity(claim_record)
    both = claim_record.with_(
        bounds=(
            *bounds,
            c.ComplexityBound(
                phase=c.Phase.REFIT, resource="time", expression="V", status=best
            ),
            c.ComplexityBound(
                phase=c.Phase.REFIT, resource="space", expression="V", status=best
            ),
        )
    )
    with pytest.raises(c.InadmissibleContract) as caught:
        c.check_complexity(both)
    assert codes(caught.value.refusals) == {c.RefusalCode.INCOMPLETE_CLAIM}
    log_bound = claim_record.with_(
        bounds=(
            bounds[0].with_(expression="(V + E) log(V) + BM(V)^omega"),
            *bounds[1:],
        )
    )
    assert c.complexity_refusals(log_bound) == ()
