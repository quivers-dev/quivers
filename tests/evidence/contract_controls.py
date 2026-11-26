"""Checks and deliberately wrong versions of the contract records.

Contract records are not executors, so their checks are not cells of the
grid of executors by problem classes. They follow the same discipline:
every check passes on the library, every mutation installs one wrong
version of a library function, and the checks it names reject it with a
recorded minimum effect.

Three kinds of check:

- a refusal check builds one inadmissible combination of records and
  requires exactly the refusal of its kind, after requiring the
  admissible combination it varies to be admitted;
- a status check settles a run that stopped early without a certificate
  and requires a heuristic result returned as unestablished, so a run
  that reports its budget or its residual as an error bound, or labels
  itself exact, is rejected;
- an identity check sends records with parameter, occurrence and site
  identities through the wire codec and counts identities that come
  back different or in another place.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

from quivers.qiec import contracts as c
from quivers.qiec.identifiers import ComputationId, FamilyId, SiteId
from quivers.qiec.serialization import dumps, loads
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Mismatch
from tests.evidence.mutations import Mutation, Rewrite

CONTRACTS = "quivers.qiec.contracts"
SERIALIZATION = "quivers.qiec.serialization"

ALGEBRA = c.ClaimSubject(kind="algebra", name="quivers.core.algebras.LogProbAlgebra")
"""The weight algebra of every combination."""

PROOF = c.Proved(
    argument="docs/api/qiec/contracts.md#the-composition-law",
    sources=(
        c.Citation(
            reference="Goodman 1999, Semiring Parsing",
            locator="section 2",
            doi="10.5555/973215.973218",
        ),
    ),
)
"""Evidence of an established claim."""

TEST = c.Tested(check="tests/test_algebra_laws.py", cases=64, domain="float64 grid")
"""Evidence of a claim that is only tested."""

PARAMETERS = (
    c.ParameterId.derive("rule", "s -> np vp"),
    c.ParameterId.derive("rule", "np -> det n"),
)
"""Two parameters, out of the order of their digests."""

OCCURRENCE = c.OccurrenceId.derive("axiom", "word", 0)
"""An axiom occurrence."""

SITES = (SiteId.derive("draw", "z"), SiteId.derive("draw", "a"))
"""Two random sites."""


def claim(law: c.Law, evidence: c.LawEvidence = PROOF) -> c.LawClaim:
    """A claim about the weight algebra.

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


ALL_LAWS = (
    *c.SEMIRING_LAWS,
    c.PLUS_SELECTIVE,
    c.ZERO_SUM_FREE,
    c.NO_ZERO_DIVISORS,
    c.STAR_LEAST_SOLUTION,
    c.OMEGA_CONTINUOUS,
)
"""Every law some combination needs."""

CLAIMS = tuple(claim(law) for law in ALL_LAWS)
"""Established claims of every law."""


def query(
    **changes: c.Observable | c.Retention | tuple[c.ParameterId | c.OccurrenceId, ...],
) -> c.QuerySpec:
    """The weight query, with changes.

    Parameters
    ----------
    **changes : Observable or Retention or tuple of ParameterId or OccurrenceId
        Fields to replace.

    Returns
    -------
    QuerySpec
        The query.
    """
    base = c.QuerySpec(
        presentation=c.PresentationId.derive("grammar"),
        interpretation=c.InterpretationId.derive("inside"),
        observable=c.Weight(),
    )
    return base.with_(**changes)


def contract(
    guarantee: c.Guarantee = c.Exact(),
    gradient: c.DerivativeSemantics = c.NOT_DIFFERENTIATED,
    controls: tuple[c.Control, ...] = (),
    on_failure: c.FailurePolicy = c.FailurePolicy.REFUSE,
) -> c.AccuracyContract:
    """An accuracy contract in the absolute log metric.

    Parameters
    ----------
    guarantee : Guarantee
        The requested guarantee.
    gradient : DerivativeSemantics
        The requested derivative.
    controls : tuple of Control
        Method parameters.
    on_failure : FailurePolicy
        The failure policy.

    Returns
    -------
    AccuracyContract
        The contract.
    """
    return c.AccuracyContract(
        numerics=c.Numerics(dtype="float64", domain="log"),
        target=c.Target(quantity="the weight of the goal", metric=c.AbsoluteLogError()),
        guarantee=guarantee,
        controls=controls,
        budget=c.ResourceBudget(iterations=100),
        on_failure=on_failure,
        gradient=gradient,
    )


HALF = c.Rational.of(Fraction(1, 2))
QUARTER = c.Rational.of(Fraction(1, 4))
EIGHTH = c.Rational.of(Fraction(1, 8))

type Combination = tuple[
    c.QuerySpec, c.Recursion, c.AccuracyContract, tuple[c.LawClaim, ...]
]
"""A query, its recursion, its contract and the claims available."""

ADMISSIBLE: Combination = (query(), c.Acyclic(), contract(), CLAIMS)
"""The admissible combination every contract refusal varies."""

EXACT_GRADIENT = c.DerivativeSemantics.exact()


def _replace_claim(
    law: c.Law, evidence: c.LawEvidence | None
) -> tuple[c.LawClaim, ...]:
    """The claims with one law's claim replaced or removed.

    Parameters
    ----------
    law : Law
        The law.
    evidence : LawEvidence or None
        Its new evidence, or ``None`` to remove its claim.

    Returns
    -------
    tuple of LawClaim
        The claims.
    """
    kept = tuple(item for item in CLAIMS if item.law != law)
    return kept if evidence is None else (*kept, claim(law, evidence))


CONTRACT_CASES: dict[c.RefusalCode, Combination] = {
    c.RefusalCode.MISSING_CLAIM: (
        query(),
        c.Acyclic(),
        contract(),
        _replace_claim(c.PLUS_COMMUTATIVE, None),
    ),
    c.RefusalCode.CLAIM_NOT_ESTABLISHED: (
        query(),
        c.Acyclic(),
        contract(),
        _replace_claim(c.PLUS_COMMUTATIVE, TEST),
    ),
    c.RefusalCode.EXACT_WITHOUT_FINITE_METHOD: (
        query(),
        c.MonotonePolynomial(),
        contract(),
        CLAIMS,
    ),
    c.RefusalCode.DERIVATIVE_UNAVAILABLE: (
        query(differentiate_with_respect_to=PARAMETERS),
        c.Unrolled(iterations=5),
        contract(guarantee=c.Certified(bound=HALF), gradient=EXACT_GRADIENT),
        CLAIMS,
    ),
    c.RefusalCode.GRADIENT_WITHOUT_INPUTS: (
        query(),
        c.Acyclic(),
        contract(gradient=EXACT_GRADIENT),
        CLAIMS,
    ),
    c.RefusalCode.INPUTS_WITHOUT_GRADIENT: (
        query(differentiate_with_respect_to=PARAMETERS),
        c.Acyclic(),
        contract(),
        CLAIMS,
    ),
    c.RefusalCode.OBSERVABLE_NOT_DIFFERENTIABLE: (
        query(observable=c.Support(), differentiate_with_respect_to=PARAMETERS),
        c.Acyclic(),
        contract(gradient=EXACT_GRADIENT),
        CLAIMS,
    ),
    c.RefusalCode.SELECTION_WITHOUT_TIE_RULE: (
        query(observable=c.Best(), differentiate_with_respect_to=PARAMETERS),
        c.Acyclic(),
        contract(gradient=EXACT_GRADIENT),
        CLAIMS,
    ),
    c.RefusalCode.FOREST_NOT_RETAINED: (
        query(observable=c.Forest()),
        c.Acyclic(),
        contract(),
        CLAIMS,
    ),
    c.RefusalCode.INFINITE_DERIVATIONS: (
        query(
            observable=c.NonlinearFold(fold=ComputationId.derive("square")),
            retention=c.PackedForest(),
        ),
        c.LinearCyclic(),
        contract(),
        CLAIMS,
    ),
    c.RefusalCode.REQUEST_WITH_CERTIFICATE: (
        query(),
        c.Acyclic(),
        contract(
            guarantee=c.Certified(
                bound=QUARTER,
                certificate=c.Enclosure(lower=c.ZERO, upper=QUARTER),
            )
        ),
        CLAIMS,
    ),
    c.RefusalCode.DUPLICATE_CONTROL: (
        query(),
        c.Acyclic(),
        contract(
            controls=(
                c.Control(name="beam", value=c.Rational.of(8)),
                c.Control(name="beam", value=c.Rational.of(16)),
            )
        ),
        CLAIMS,
    ),
}
"""One inadmissible combination per refusal of `admissibility`."""


def _admissibility(combination: Combination) -> set[c.RefusalCode]:
    """The refusals of a combination.

    Parameters
    ----------
    combination : Combination
        The query, recursion, contract and claims.

    Returns
    -------
    set of RefusalCode
        The kinds of refusal.
    """
    spec, recursion, accuracy, claims = combination
    found = c.admissibility(spec, recursion, accuracy, algebra=ALGEBRA, claims=claims)
    return {item.code for item in found}


def _require(
    found: set[c.RefusalCode], expected: set[c.RefusalCode], what: str
) -> None:
    """Require exactly the expected refusals.

    Parameters
    ----------
    found : set of RefusalCode
        The refusals produced.
    expected : set of RefusalCode
        The refusals required.
    what : str
        The combination, named in the diagnostic.

    Raises
    ------
    Mismatch
        If they differ.
    """
    if found != expected:
        raise Mismatch(
            "refusal",
            math.inf,
            f"{what}: refused {sorted(code.value for code in found)}, required "
            f"{sorted(code.value for code in expected)}",
        )


def check_contract_refusal(code: c.RefusalCode) -> None:
    """Require one inadmissible combination to be refused for its reason.

    Parameters
    ----------
    code : RefusalCode
        The refusal.

    Raises
    ------
    Mismatch
        If the admissible combination is refused, or the inadmissible one
        is not refused for exactly this reason.
    """
    _require(_admissibility(ADMISSIBLE), set(), "the admissible combination")
    _require(_admissibility(CONTRACT_CASES[code]), {code}, code.value)


def _report(
    guarantee: c.Guarantee,
    *,
    stop: c.StopReason = c.Completed(),
    unmet: tuple[c.ContractField, ...] = (),
    claims: tuple[c.LawClaim, ...] = (),
    metric: c.Metric = c.AbsoluteLogError(),
) -> c.AccuracyReport:
    """A report of a run without a derivative.

    Parameters
    ----------
    guarantee : Guarantee
        The attained guarantee.
    stop : StopReason
        Why the run stopped.
    unmet : tuple of ContractField
        The requirements reported unmet.
    claims : tuple of LawClaim
        The claims the run used.
    metric : Metric
        The metric of the attainment.

    Returns
    -------
    AccuracyReport
        The report.
    """
    return c.AccuracyReport(
        attainment=c.Attainment(guarantee=guarantee, metric=metric, rounding=c.ZERO),
        stop=stop,
        unmet=unmet,
        work=(c.WorkReport(phase=c.Phase.EXECUTION, iterations=3, scalar_sums=40),),
        claims=claims,
    )


type ReportCase = tuple[c.AccuracyContract, c.AccuracyReport, c.Recursion]
"""A contract, a report of a run under it, and the run's recursion."""

CERTIFIED_REQUEST = contract(guarantee=c.Certified(bound=HALF))
"""A contract that asks for a bound of one half."""

ADMISSIBLE_REPORT: ReportCase = (contract(), _report(c.Exact()), c.Acyclic())
"""The admissible report every report refusal varies."""

REPORT_CASES: dict[c.RefusalCode, ReportCase] = {
    c.RefusalCode.EARLY_STOP_LABELLED_EXACT: (
        contract(),
        _report(
            c.Exact(),
            stop=c.BudgetExhausted(resource="iterations", limit=c.Rational.of(100)),
        ),
        c.Acyclic(),
    ),
    c.RefusalCode.CERTIFIED_WITHOUT_CERTIFICATE: (
        CERTIFIED_REQUEST,
        _report(c.Certified(bound=QUARTER)),
        c.Acyclic(),
    ),
    c.RefusalCode.BOUND_BELOW_CERTIFICATE: (
        CERTIFIED_REQUEST,
        _report(
            c.Certified(
                bound=EIGHTH, certificate=c.Enclosure(lower=QUARTER, upper=HALF)
            )
        ),
        c.Acyclic(),
    ),
    c.RefusalCode.ARGUMENT_NOT_ESTABLISHED: (
        CERTIFIED_REQUEST,
        _report(
            c.Certified(
                bound=QUARTER,
                certificate=c.Argument(
                    value=QUARTER,
                    claim=claim(c.Law(name="contraction", statement="rate 1/2"), TEST),
                ),
            )
        ),
        c.Acyclic(),
    ),
    c.RefusalCode.CERTIFICATE_METRIC: (
        CERTIFIED_REQUEST,
        _report(
            c.Certified(
                bound=QUARTER, certificate=c.DualityGap(primal=QUARTER, dual=HALF)
            )
        ),
        c.Acyclic(),
    ),
    c.RefusalCode.USED_CLAIM_NOT_ESTABLISHED: (
        contract(),
        _report(c.Exact(), claims=(claim(c.PLUS_COMMUTATIVE, TEST),)),
        c.Acyclic(),
    ),
    c.RefusalCode.EXACT_FOR_INEXACT_RECURSION: (
        contract(),
        _report(c.Exact()),
        c.MonotonePolynomial(),
    ),
    c.RefusalCode.UNMET_MISREPORTED: (
        contract(),
        _report(c.Heuristic(reason="beam search")),
        c.Acyclic(),
    ),
}
"""One inadmissible report per refusal of `report_refusals`."""


def _report_refusals(case: ReportCase) -> set[c.RefusalCode]:
    """The refusals of a report.

    Parameters
    ----------
    case : ReportCase
        The contract, report and recursion.

    Returns
    -------
    set of RefusalCode
        The kinds of refusal.
    """
    accuracy, report, recursion = case
    return {item.code for item in c.report_refusals(accuracy, report, recursion)}


def check_report_refusal(code: c.RefusalCode) -> None:
    """Require one inadmissible report to be refused for its reason.

    Parameters
    ----------
    code : RefusalCode
        The refusal.

    Raises
    ------
    Mismatch
        If the admissible report is refused, or the inadmissible one is
        not refused for exactly this reason.
    """
    _require(_report_refusals(ADMISSIBLE_REPORT), set(), "the admissible report")
    _require(_report_refusals(REPORT_CASES[code]), {code}, code.value)


def _bound(
    phase: c.Phase, resource: str, expression: str, status: c.OptimalityStatus
) -> c.ComplexityBound:
    """One bound of a complexity claim.

    Parameters
    ----------
    phase : Phase
        The phase.
    resource : str
        ``"time"`` or ``"space"``.
    expression : str
        The bound.
    status : OptimalityStatus
        Its status.

    Returns
    -------
    ComplexityBound
        The bound.
    """
    return c.ComplexityBound(
        phase=phase,
        resource="time" if resource == "time" else "space",
        expression=expression,
        status=status,
    )


EISNER = c.Citation(
    reference="Eisner 2023, Time-and-Space-Efficient Weighted Deduction",
    locator="sections 5 and 6",
    doi="10.1162/tacl_a_00588",
)
BEST = c.BestKnown(sources=(EISNER,))
READING = c.ProvedOptimal(
    lower_bound="V + E",
    model=c.ALGEBRA_OPERATIONS,
    sources=(
        c.Citation(
            reference="every hyperedge is read by an adversary argument",
            locator="complexity claims, Q10",
        ),
    ),
)


def _claim(
    bounds: tuple[c.ComplexityBound, ...],
    inapplicable: tuple[c.PhaseInapplicable, ...] = (),
) -> c.ComplexityClaim:
    """A complexity claim of the acyclic weight query.

    Parameters
    ----------
    bounds : tuple of ComplexityBound
        Its bounds.
    inapplicable : tuple of PhaseInapplicable
        The phases that do not apply.

    Returns
    -------
    ComplexityClaim
        The claim.
    """
    return c.ComplexityClaim(
        query="weight of all items, acyclic derived hypergraph",
        parameters=(
            c.ComplexityParameter(name="V", meaning="items"),
            c.ComplexityParameter(name="E", meaning="grounded rule instances"),
        ),
        model=c.ALGEBRA_OPERATIONS,
        bounds=bounds,
        inapplicable=inapplicable,
    )


COMPLETE_BOUNDS = (
    _bound(c.Phase.CONSTRUCTION, "time", "V + E", BEST),
    _bound(c.Phase.CONSTRUCTION, "space", "V", BEST),
    _bound(c.Phase.EXECUTION, "time", "V + E", READING),
    _bound(c.Phase.EXECUTION, "space", "V", BEST),
    _bound(c.Phase.DIFFERENTIATION, "time", "V + E", BEST),
    _bound(c.Phase.DIFFERENTIATION, "space", "E", BEST),
)
REFIT = (c.PhaseInapplicable(phase=c.Phase.REFIT, reason="the query has no refit"),)

ADMISSIBLE_CLAIM = _claim(COMPLETE_BOUNDS, REFIT)
"""The admissible complexity claim every claim refusal varies."""

CLAIM_CASES: dict[c.RefusalCode, c.ComplexityClaim] = {
    c.RefusalCode.INCOMPLETE_CLAIM: _claim(COMPLETE_BOUNDS),
    c.RefusalCode.UNDECLARED_PARAMETER: _claim(
        (*COMPLETE_BOUNDS[:-1], _bound(c.Phase.DIFFERENTIATION, "space", "E n", BEST)),
        REFIT,
    ),
    c.RefusalCode.LOWER_BOUND_MISMATCH: _claim(
        (
            *COMPLETE_BOUNDS[:2],
            _bound(c.Phase.EXECUTION, "time", "V + E", READING.with_(lower_bound="V")),
            *COMPLETE_BOUNDS[3:],
        ),
        REFIT,
    ),
}
"""One inadmissible complexity claim per refusal of `complexity_refusals`."""


def check_claim_refusal(code: c.RefusalCode) -> None:
    """Require one inadmissible complexity claim to be refused for its reason.

    Parameters
    ----------
    code : RefusalCode
        The refusal.

    Raises
    ------
    Mismatch
        If the admissible claim is refused, or the inadmissible one is
        not refused for exactly this reason.
    """
    _require(
        {item.code for item in c.complexity_refusals(ADMISSIBLE_CLAIM)},
        set(),
        "the admissible claim",
    )
    _require(
        {item.code for item in c.complexity_refusals(CLAIM_CASES[code])},
        {code},
        code.value,
    )


def check_status(stop: c.StopReason) -> None:
    """Require a run stopped early without a certificate to be heuristic.

    The run's method is exact when it completes. Stopped early, it has no
    certificate, so it claims nothing, its contract is not met, and the
    value comes back unestablished.

    Parameters
    ----------
    stop : StopReason
        Why the run stopped.

    Raises
    ------
    Mismatch
        If the report is refused, the guarantee is not heuristic, or the
        value is not returned unestablished.
    """
    accuracy = contract(on_failure=c.FailurePolicy.RETURN_UNESTABLISHED)
    guarantee = c.attained_guarantee(stop, c.Exact(), None)
    attainment = c.Attainment(
        guarantee=guarantee, metric=c.AbsoluteLogError(), rounding=c.ZERO
    )
    work = (c.WorkReport(phase=c.Phase.EXECUTION, iterations=100),)
    report = c.AccuracyReport(
        attainment=attainment,
        stop=stop,
        unmet=c.unmet_fields(accuracy, attainment, work),
        work=work,
    )
    refused = c.report_refusals(accuracy, report, c.MonotonePolynomial())
    if refused:
        raise Mismatch(
            "refusal",
            math.inf,
            "; ".join(f"{item.code.value}: {item.reason}" for item in refused),
        )
    if not isinstance(guarantee, c.Heuristic):
        raise Mismatch("value", math.inf, f"a stopped run claimed {guarantee.tag}")
    settled = c.settle(accuracy, 6.2158, report)
    if not isinstance(settled, c.Unestablished):
        raise Mismatch("value", math.inf, "a stopped run returned a plain value")


def _identities(value: c.QuerySpec | c.AccuracyReport) -> list[str]:
    """The identities a record holds, in order and with their classes.

    Parameters
    ----------
    value : QuerySpec or AccuracyReport
        The record.

    Returns
    -------
    list of str
        ``class:identity`` of every parameter, occurrence and site.
    """
    if isinstance(value, c.QuerySpec):
        held = list(value.differentiate_with_respect_to)
    else:
        held = [
            identity
            for entry in value.derivatives.entries
            for identity in (
                entry.wrt,
                *(item.site for item in entry.semantics.estimators),
            )
        ]
    return [f"{type(item).__name__}:{item}" for item in held]


def identity_records() -> tuple[c.QuerySpec, c.AccuracyReport]:
    """A query and a report that hold parameter, occurrence and site identities.

    Returns
    -------
    tuple of QuerySpec and AccuracyReport
        The records.
    """
    spec = query(
        observable=c.Outcome(key=ComputationId.derive("meaning")),
        differentiate_with_respect_to=(*PARAMETERS, OCCURRENCE),
        retention=c.RequestedEvidence(
            patterns=(c.JudgmentPattern(family=FamilyId.derive("parse")),)
        ),
    )
    estimated = c.DerivativeSemantics.estimated(
        *(c.SiteEstimator(site=site, estimator="score") for site in SITES)
    )
    entries = tuple(
        c.DerivativeEntry(output="weight", wrt=wrt, semantics=estimated)
        for wrt in (*PARAMETERS, OCCURRENCE)
    )
    report = c.AccuracyReport(
        attainment=c.Attainment(
            guarantee=c.Exact(), metric=c.AbsoluteLogError(), derivative=estimated
        ),
        derivatives=c.DerivativeReport(entries=entries),
    )
    return spec, report


def check_identities() -> None:
    """Require parameter, occurrence and site identities to survive the codec.

    Raises
    ------
    Mismatch
        If a decoded record differs, with the number of identities that
        came back different or in another place as the deviation.
    """
    for record in identity_records():
        decoded = loads(dumps(record))
        before = _identities(record)
        after = (
            _identities(decoded)
            if isinstance(decoded, (c.QuerySpec, c.AccuracyReport))
            else []
        )
        lost = sum(
            1
            for index, identity in enumerate(before)
            if index >= len(after) or after[index] != identity
        )
        if lost or decoded != record:
            raise Mismatch(
                "value",
                float(max(lost, 1)),
                f"{lost} of {len(before)} identities changed in a round trip",
            )


CHECKS: dict[str, Check] = {}
"""Every check of the contract records by name."""


def _add(name: str, run: Callable[[], None]) -> None:
    """Register one check.

    Parameters
    ----------
    name : str
        Its name.
    run : Callable[[], None]
        Its body.

    Raises
    ------
    ValueError
        If the name is taken.
    """
    if name in CHECKS:
        raise ValueError(f"the check {name!r} is registered twice")
    CHECKS[name] = Check(name=name, executor="contracts", question="records", run=run)


def _register() -> None:
    """Register every check."""
    for code in CONTRACT_CASES:
        _add(
            f"contracts/admissibility/{code.value}", _bind(check_contract_refusal, code)
        )
    for code in REPORT_CASES:
        _add(f"contracts/report/{code.value}", _bind(check_report_refusal, code))
    for code in CLAIM_CASES:
        _add(f"contracts/complexity/{code.value}", _bind(check_claim_refusal, code))
    _add(
        "contracts/status/budget",
        lambda: check_status(
            c.BudgetExhausted(resource="iterations", limit=c.Rational.of(100))
        ),
    )
    _add(
        "contracts/status/residual",
        lambda: check_status(
            c.ControlReached(
                control="tolerance", value=c.Rational.of(Fraction(1, 1000))
            )
        ),
    )
    _add("contracts/codec/identities", check_identities)


def _bind(
    check: Callable[[c.RefusalCode], None], code: c.RefusalCode
) -> Callable[[], None]:
    """A check with its refusal fixed.

    Parameters
    ----------
    check : Callable[[RefusalCode], None]
        The check.
    code : RefusalCode
        The refusal.

    Returns
    -------
    Callable[[], None]
        The bound check.
    """
    return lambda: check(code)


_register()


def _removed(code: c.RefusalCode, function: str, old: str, new: str) -> Mutation:
    """A mutation that removes one refusal.

    Parameters
    ----------
    code : RefusalCode
        The refusal.
    function : str
        The function that raises it.
    old : str
        The text of its condition.
    new : str
        A condition that never holds, or that holds elsewhere.

    Returns
    -------
    Mutation
        The mutation, rejected by the check of the refusal.
    """
    prefix = {
        "admissibility": "admissibility",
        "report_refusals": "report",
        "complexity_refusals": "complexity",
    }[function]
    return Mutation(
        name=f"refusal-{code.value}-removed",
        defect_class="refusal",
        provenance=f"the checker admits a combination it has to refuse: {code.value}",
        target=f"{CONTRACTS}:{function}",
        rewrites=(Rewrite(old=old, new=new),),
        rejected_by=(f"contracts/{prefix}/{code.value}",),
        min_effect=1.0,
    )


R = c.RefusalCode

MUTATIONS: tuple[Mutation, ...] = (
    _removed(R.MISSING_CLAIM, "admissibility", "if not matching:", "if False:"),
    _removed(
        R.CLAIM_NOT_ESTABLISHED,
        "admissibility",
        "if matching and guaranteed and not any(c.established for c in matching):",
        "if False:",
    ),
    _removed(
        R.EXACT_WITHOUT_FINITE_METHOD,
        "admissibility",
        "if isinstance(contract.guarantee, Exact) and not recursion.finite_method:",
        "if False:",
    ),
    _removed(
        R.DERIVATIVE_UNAVAILABLE,
        "admissibility",
        "if not recursion.admits(wanted):",
        "if False:",
    ),
    _removed(
        R.GRADIENT_WITHOUT_INPUTS,
        "admissibility",
        "if not wanted.missing and not query.differentiate_with_respect_to:",
        "if False:",
    ),
    _removed(
        R.INPUTS_WITHOUT_GRADIENT,
        "admissibility",
        "if wanted.missing and query.differentiate_with_respect_to:",
        "if False:",
    ),
    _removed(
        R.OBSERVABLE_NOT_DIFFERENTIABLE,
        "admissibility",
        "if query.differentiate_with_respect_to and not observable.differentiable:",
        "if False:",
    ),
    _removed(
        R.SELECTION_WITHOUT_TIE_RULE,
        "admissibility",
        "if observable.selects and not wanted.missing and wanted.tie_rule is None:",
        "if False:",
    ),
    _removed(
        R.FOREST_NOT_RETAINED,
        "admissibility",
        "if observable.needs_forest and not query.retention.keeps_forest:",
        "if False:",
    ),
    _removed(
        R.INFINITE_DERIVATIONS,
        "admissibility",
        "and not recursion.finite_derivations",
        "and False",
    ),
    _removed(
        R.REQUEST_WITH_CERTIFICATE,
        "admissibility",
        "and contract.guarantee.certificate is not None",
        "and False",
    ),
    _removed(
        R.DUPLICATE_CONTROL,
        "admissibility",
        "if len(set(names)) != len(names):",
        "if False:",
    ),
    _removed(
        R.EARLY_STOP_LABELLED_EXACT,
        "report_refusals",
        "if report.stop.early and isinstance(attained, Exact):",
        "if False:",
    ),
    _removed(
        R.CERTIFIED_WITHOUT_CERTIFICATE,
        "report_refusals",
        "if claimed is not None and certificate is None:",
        "if False:",
    ),
    _removed(
        R.BOUND_BELOW_CERTIFICATE,
        "report_refusals",
        "and claimed < certificate.bound:",
        "and False:",
    ),
    _removed(
        R.ARGUMENT_NOT_ESTABLISHED,
        "report_refusals",
        "if isinstance(certificate, Argument) and not certificate.claim.established:",
        "if False:",
    ),
    _removed(
        R.CERTIFICATE_METRIC,
        "report_refusals",
        "and not _certificate_metric(",
        "and False and not _certificate_metric(",
    ),
    _removed(R.USED_CLAIM_NOT_ESTABLISHED, "report_refusals", "if weak:", "if False:"),
    _removed(
        R.EXACT_FOR_INEXACT_RECURSION,
        "report_refusals",
        "and not recursion.finite_method",
        "and False",
    ),
    _removed(
        R.UNMET_MISREPORTED,
        "report_refusals",
        "if report.unmet != expected:",
        "if False:",
    ),
    _removed(
        R.INCOMPLETE_CLAIM,
        "complexity_refusals",
        "if bounded == absent or (bounded and inapplicable[phase]):",
        "if False:",
    ),
    _removed(R.UNDECLARED_PARAMETER, "complexity_refusals", "if unknown:", "if False:"),
    _removed(
        R.LOWER_BOUND_MISMATCH,
        "complexity_refusals",
        "isinstance(status, ProvedOptimal) and _normal(",
        "False and _normal(",
    ),
    Mutation(
        name="early-stop-claims-the-method-guarantee",
        defect_class="status",
        provenance=(
            "a run that ran out of budget or reached its tolerance reports the "
            "guarantee its method has when it completes, so exhaustion is labelled "
            "exact"
        ),
        target=f"{CONTRACTS}:attained_guarantee",
        rewrites=(
            Rewrite(
                old='return Heuristic(reason=f"stopped early by {stop.tag} without '
                'a certificate")',
                new="return method",
            ),
        ),
        rejected_by=("contracts/status/budget", "contracts/status/residual"),
        min_effect=1.0,
    ),
    Mutation(
        name="budget-reported-as-error-bound",
        defect_class="status",
        provenance="the iteration cap of an exhausted run is reported as its error bound",
        target=f"{CONTRACTS}:attained_guarantee",
        rewrites=(
            Rewrite(
                old='return Heuristic(reason=f"stopped early by {stop.tag} without '
                'a certificate")',
                new="return Certified(bound=stop.limit)",
            ),
        ),
        rejected_by=("contracts/status/budget",),
        min_effect=1.0,
    ),
    Mutation(
        name="residual-reported-as-error-bound",
        defect_class="status",
        provenance="the last residual of a run stopped by its tolerance is its bound",
        target=f"{CONTRACTS}:attained_guarantee",
        rewrites=(
            Rewrite(
                old='return Heuristic(reason=f"stopped early by {stop.tag} without '
                'a certificate")',
                new="return Certified(bound=stop.value)",
            ),
        ),
        rejected_by=("contracts/status/residual",),
        min_effect=1.0,
    ),
    Mutation(
        name="identities-sorted-on-the-wire",
        defect_class="serialization",
        provenance=(
            "the encoder writes tuples in a canonical sorted order, so the order "
            "of the gradient's inputs is lost"
        ),
        target=f"{SERIALIZATION}:_encode",
        rewrites=(
            Rewrite(
                old='return {"$tuple": [_encode(item) for item in value]}',
                new='return {"$tuple": [_encode(item) for item in sorted(value, key=str)]}',
            ),
        ),
        rejected_by=("contracts/codec/identities",),
        min_effect=1.0,
    ),
    Mutation(
        name="identities-derived-again-from-their-text",
        defect_class="serialization",
        provenance=(
            "the decoder derives an identifier from its wire text instead of "
            "parsing it, so every parameter and site gets a new identity"
        ),
        target=f"{SERIALIZATION}:_decode",
        rewrites=(
            Rewrite(
                old="return _IDS[tag].parse(encoded)",
                new="return _IDS[tag].derive(encoded)",
            ),
        ),
        rejected_by=("contracts/codec/identities",),
        min_effect=2.0,
    ),
)
"""Every wrong version of the contract records and the checks that reject it."""
