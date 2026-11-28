"""Report the actual checked handler shapes and nominal law checks for a plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from quivers.qiec.differentiation import ExactJvp, JvpLimit, JvpUnsupported, exact_jvp
from quivers.qiec.discharge import (
    DischargeObstacle,
    DischargedModule,
    DischargeRewrite,
    discharge_module,
)
from quivers.qiec.effect_law_schema import (
    EffectLawBundle,
    EffectLawEvidence,
    EffectLawObligation,
    _module_digest,
)
from quivers.qiec.effect_laws import (
    FiniteEffectLawDecision,
    ScalarEffectLawDecision,
    check_finite_effect_law,
    check_scalar_effect_law,
)
from quivers.qiec.handler_carriers import HandlerCarrierShape, classify_handler_carrier
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.handler_interchange import (
    FiniteHandlerInterchange,
    check_finite_handler_interchange,
)
from quivers.qiec.laws import FiniteLawDecision, FiniteLawLimit, FiniteLawUnsupported
from quivers.qiec.module import (
    QiecModule,
    _execution_reachability,
    reached_handlers,
    validate_module,
)
from quivers.qiec.polynomial_laws import PolynomialLawLimit, PolynomialLawUnsupported
from quivers.qiec.rational_laws import RationalLawLimit, RationalLawUnsupported

type _Decision = FiniteEffectLawDecision | ScalarEffectLawDecision
type _Status = Literal["checked", "refuted", "unresolved"]


@dataclass(frozen=True, slots=True)
class EffectLawReport:
    """The result of checking one declared obligation against its source.

    Parameters
    ----------
    obligation : EffectLawObligation
        The exact requested equation, handler, method, and guard conditions.
    decision : FiniteEffectLawDecision or ScalarEffectLawDecision or None
        The freshly checked result, including a witness when the law fails.
    reason : str or None
        Why the checker could not decide the obligation, if it could not.
    declared_evidence : EffectLawEvidence or None
        The test or assumption attached to this exact source.
    """

    obligation: EffectLawObligation
    decision: _Decision | None
    reason: str | None
    declared_evidence: EffectLawEvidence | None = None

    @property
    def status(self) -> _Status:
        """The result of the check.

        Returns
        -------
        _Status
            Whether the law was checked, refuted, or left unresolved.
        """
        if self.decision is None:
            return "unresolved"
        return "checked" if self.decision.certified else "refuted"


@dataclass(frozen=True, slots=True)
class EffectInterchangeReport:
    """The returned-value observation of one requested finite interchange.

    Parameters
    ----------
    statement : FiniteHandlerInterchange
        The two named computations whose installation orders are compared.
    decision : FiniteLawDecision or None
        The result of the finite comparison when it can be decided.
    reason : str or None
        The obstacle to deciding the interchange, if there is one.
    """

    statement: FiniteHandlerInterchange
    decision: FiniteLawDecision | None
    reason: str | None

    @property
    def status(self) -> _Status:
        """The status of the returned-value comparison.

        Returns
        -------
        _Status
            Whether the interchange was checked, refuted, or left unresolved.
        """
        if self.decision is None:
            return "unresolved"
        return "checked" if self.decision.counterexample is None else "refuted"


@dataclass(frozen=True, slots=True)
class EffectExecutionStage:
    """One source computation reached in the resulting execution graph.

    Parameters
    ----------
    computation : ComputationId
        The original checked identity, preserved by lowering.
    stage : {"direct", "handler-discharge", "recursive-state", "reference"}
        The stage selected by ordinary discharge. `direct` has no
        local rewrite and `reference` retains a handler. A direct
        caller can reach another computation at a different stage.
    rewrite : DischargeRewrite or None
        The checked transformation and its companions, if one ran.
    obstacle : DischargeObstacle or None
        The reason a handler remains for reference execution.
    """

    computation: ComputationId
    stage: Literal["direct", "handler-discharge", "recursive-state", "reference"]
    rewrite: DischargeRewrite | None = None
    obstacle: DischargeObstacle | None = None


@dataclass(frozen=True, slots=True)
class EffectJvpCheck:
    """A pointwise comparison of exact source and staged derivatives.

    Parameters
    ----------
    computation : ComputationId
        The checked source identity selected for both evaluations.
    arguments : tuple[int or Fraction, ...]
        The source coordinates in parameter order.
    direction : tuple[int or Fraction, ...]
        The supplied tangent in that order.
    source : ExactJvp or None
        The source machine's result when it admits the calculation.
    staged : ExactJvp or None
        The selected staged graph's result under the same calculation.
    source_obstacle : str or None
        The actual derivative refusal in the source graph, if any.
    staged_obstacle : str or None
        The actual derivative refusal in the staged graph, if any.
    """

    computation: ComputationId
    arguments: tuple[int | Fraction, ...]
    direction: tuple[int | Fraction, ...]
    source: ExactJvp | None
    staged: ExactJvp | None
    source_obstacle: str | None
    staged_obstacle: str | None

    @property
    def status(self) -> _Status:
        """Report whether the two actual observations agree.

        Returns
        -------
        _Status
            Whether both results agree, differ, or a result is unavailable.
        """
        if self.source is None or self.staged is None:
            return "unresolved"
        return "checked" if self.source == self.staged else "refuted"


@dataclass(frozen=True, slots=True)
class EffectPlan:
    """A view of the source, carrier analysis, law results, and discharge.

    Parameters
    ----------
    bundle : EffectLawBundle
        The checked source and all the law obligations requested for it.
    computation : ComputationId
        The selected source computation.
    carriers : tuple[HandlerCarrierShape, ...]
        Carrier shapes for handlers in the computation's checked call graph.
    laws : tuple[EffectLawReport, ...]
        One result per requested obligation, regardless of reachability
        from the selected computation.
    discharge : DischargedModule
        The result of ordinary checked discharge, including its obstacles.
        Ordinary discharge applies its own checked rules.
    interchanges : tuple[EffectInterchangeReport, ...]
        Returned-value observations from independently checked finite interchanges.
    stages : tuple[EffectExecutionStage, ...]
        The stages of reachable source computations in the discharged
        graph. Generated computations appear on their originating rewrite.
    """

    bundle: EffectLawBundle
    computation: ComputationId
    carriers: tuple[HandlerCarrierShape, ...]
    laws: tuple[EffectLawReport, ...]
    discharge: DischargedModule
    interchanges: tuple[EffectInterchangeReport, ...] = ()
    stages: tuple[EffectExecutionStage, ...] = ()

    def check_jvp(
        self,
        arguments: tuple[int | Fraction, ...],
        direction: tuple[int | Fraction, ...],
        *,
        fuel: int = 65_536,
    ) -> EffectJvpCheck:
        """Compare exact scalar JVPs in the source and staged QIEC graphs.

        Both evaluations use the selected source identity. The reported
        equality concerns the exact supplied point and direction in the
        regular scalar fragment.

        Parameters
        ----------
        arguments : tuple[int or Fraction, ...]
            Exact input coordinates in source order.
        direction : tuple[int or Fraction, ...]
            Exact tangent coordinates in the same order.
        fuel : int
            Reference steps allowed independently in each graph.

        Returns
        -------
        EffectJvpCheck
            Both results or the specified refusals at the point.

        Raises
        ------
        ValueError
            If the computation identity, inputs, or fuel are invalid.
        """
        source = self.bundle.module
        staged = self.discharge.module
        owner = next(
            (entry for entry in source.computations if entry.id == self.computation),
            None,
        )
        if owner is None:
            raise ValueError("the effect plan does not name a source computation")
        if not any(
            entry.id == owner.id and entry.name == owner.name
            for entry in staged.computations
        ):
            raise ValueError("the staged graph does not preserve the source identity")
        observations: list[tuple[ExactJvp | None, str | None]] = []
        for graph in (source, staged):
            try:
                answer = exact_jvp(graph, owner.name, arguments, direction, fuel=fuel)
            except (JvpUnsupported, JvpLimit) as error:
                observations.append((None, str(error)))
            else:
                observations.append((answer, None))
        (left, left_obstacle), (right, right_obstacle) = observations
        return EffectJvpCheck(
            self.computation,
            arguments,
            direction,
            left,
            right,
            left_obstacle,
            right_obstacle,
        )


def _law_report(
    module: QiecModule,
    obligation: EffectLawObligation,
    *,
    max_assignments: int,
    max_work: int,
    fuel: int,
) -> EffectLawReport:
    """Independently decide a supported obligation and keep its refusal.

    Parameters
    ----------
    module : QiecModule
        The source supplying the actual handler and equation.
    obligation : EffectLawObligation
        The requested method and scope.
    max_assignments : int
        The finite check's assignment bound.
    max_work : int
        The scalar check's normalization budget.
    fuel : int
        The exact interpreter's step bound per probe.

    Returns
    -------
    EffectLawReport
        The freshly checked evidence or a specific unresolved reason.
    """
    if obligation.guards:
        names = ", ".join(guard.name for guard in obligation.guards)
        return EffectLawReport(
            obligation, None, f"declared guard conditions remain unproved: {names}"
        )
    try:
        decision: _Decision
        if obligation.method == "finite":
            decision = check_finite_effect_law(
                module,
                obligation.equation,
                obligation.handler,
                max_assignments=max_assignments,
                fuel=fuel,
            )
        else:
            decision = check_scalar_effect_law(
                module,
                obligation.equation,
                obligation.handler,
                method=obligation.method,
                max_work=max_work,
                fuel=fuel,
            )
    except (
        FiniteLawLimit,
        FiniteLawUnsupported,
        PolynomialLawLimit,
        PolynomialLawUnsupported,
        RationalLawLimit,
        RationalLawUnsupported,
    ) as error:
        return EffectLawReport(obligation, None, str(error))
    return EffectLawReport(obligation, decision, None)


def checked_effect_plan(
    bundle: EffectLawBundle,
    computation: ComputationId,
    *,
    interchanges: tuple[FiniteHandlerInterchange, ...] = (),
    law_evidence: tuple[EffectLawEvidence, ...] = (),
    max_assignments: int = 65_536,
    max_work: int = 65_536,
    fuel: int = 65_536,
) -> EffectPlan:
    """Report actual carrier shapes, exact law evidence, and ordinary discharge.

    Each law is checked against the source in the bundle. An unsupported
    method or exhausted limit stays visible beside decisions for other
    obligations. A named guard condition is an open hypothesis. Existing
    discharge uses its own checks and does not consume law reports.

    Parameters
    ----------
    bundle : EffectLawBundle
        The source module and the nominal obligations to inspect.
    computation : ComputationId
        The source declaration whose reachable handlers to classify.
    interchanges : tuple[FiniteHandlerInterchange, ...]
        Named finite observations checked independently of discharge.
    law_evidence : tuple[EffectLawEvidence, ...]
        Bound empirical or assumed evidence for requested obligations.
    max_assignments : int
        The maximum number of assignments per finite obligation.
    max_work : int
        The maximum symbolic work per scalar obligation.
    fuel : int
        The interpreter step limit per side and assignment when used by a
        finite or rational check.

    Returns
    -------
    EffectPlan
        A report over the actual checked source, including every
        independently unresolved obligation.

    Raises
    ------
    ValueError
        If the computation, a bound, or an evidence attachment is invalid.
    """
    for name, limit in (
        ("max_assignments", max_assignments),
        ("max_work", max_work),
        ("fuel", fuel),
    ):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    module = bundle.module
    validate_module(module)
    if not any(item.id == computation for item in module.computations):
        raise ValueError(f"unknown computation for effect plan `{computation}`")
    reach = reached_handlers(module, computation)
    carriers = tuple(
        classify_handler_carrier(module, handler.id)
        for handler in module.handlers
        if handler.id in reach
    )
    by_obligation: dict[EffectLawObligation, EffectLawEvidence] = {}
    if law_evidence:
        digest = _module_digest(module)
        obligations = frozenset(bundle.obligations)
        for assertion in law_evidence:
            if (
                assertion.module_digest != digest
                or assertion.obligation not in obligations
            ):
                raise ValueError(
                    "effect law evidence names a different module or obligation"
                )
            if assertion.obligation in by_obligation:
                raise ValueError(
                    "duplicate declared evidence for an effect law obligation"
                )
            by_obligation[assertion.obligation] = assertion
    laws = tuple(
        replace(
            _law_report(
                module,
                obligation,
                max_assignments=max_assignments,
                max_work=max_work,
                fuel=fuel,
            ),
            declared_evidence=by_obligation.get(obligation),
        )
        for obligation in bundle.obligations
    )
    comparisons: list[EffectInterchangeReport] = []
    for statement in interchanges:
        try:
            decision = check_finite_handler_interchange(
                module, statement, max_assignments=max_assignments, fuel=fuel
            )
        except (FiniteLawLimit, FiniteLawUnsupported) as error:
            comparisons.append(EffectInterchangeReport(statement, None, str(error)))
        else:
            comparisons.append(EffectInterchangeReport(statement, decision, None))
    discharge = discharge_module(module)
    reached, _ = _execution_reachability(discharge.module, computation)
    rewrites = {rewrite.computation: rewrite for rewrite in discharge.rewrites}
    obstacles = {obstacle.computation: obstacle for obstacle in discharge.obstacles}
    stages: list[EffectExecutionStage] = []
    for source in module.computations:
        if source.id not in reached:
            continue
        if (obstacle := obstacles.get(source.name)) is not None:
            stage = EffectExecutionStage(source.id, "reference", obstacle=obstacle)
        elif (rewrite := rewrites.get(source.id)) is not None:
            stage = EffectExecutionStage(source.id, rewrite.method, rewrite=rewrite)
        else:
            stage = EffectExecutionStage(source.id, "direct")
        stages.append(stage)
    return EffectPlan(
        bundle,
        computation,
        carriers,
        laws,
        discharge,
        tuple(comparisons),
        tuple(stages),
    )


__all__ = [
    "EffectExecutionStage",
    "EffectInterchangeReport",
    "EffectJvpCheck",
    "EffectLawReport",
    "EffectPlan",
    "checked_effect_plan",
]
