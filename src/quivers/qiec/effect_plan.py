"""Report the actual checked handler shapes and nominal law checks for a plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from quivers.qiec.contracts import DerivativeSemantics
from quivers.qiec.derivative_rule_schema import RationalDerivativeDeclaration
from quivers.qiec.derivative_rules import (
    RationalDerivativeDecision,
    check_rational_derivative_rule,
)
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
    installed_handlers,
    program_computations,
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
    """One source computation reached in the selected execution graph.

    Parameters
    ----------
    computation : ComputationId
        The original checked identity, preserved by lowering.
    stage : {"direct", "handler-discharge", "recursive-state", "reference", "source-handler"}
        `direct` has no local rewrite. Ordinary discharge uses `reference`
        for a retained handler and names its obstacle. A target selecting
        the unchanged source uses `source-handler` at original installations.
        A direct caller can reach a computation at another stage.
    rewrite : DischargeRewrite or None
        The checked transformation and its companions, if one ran.
    obstacle : DischargeObstacle or None
        The reason a handler remains for reference execution.
    """

    computation: ComputationId
    stage: Literal[
        "direct", "handler-discharge", "recursive-state", "reference", "source-handler"
    ]
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
class EffectTargetReport:
    """An actual standalone export outcome with scoped derivative evidence.

    Parameters
    ----------
    target : str
        The requested public target key.
    computation : ComputationId
        The source entry whose execution stages are reported.
    projection : {"source", "discharged"} or None
        The QIEC graph selected by the renderer after successful emission.
    stages : tuple[EffectExecutionStage, ...]
        Source entries reached by this export's selected QIEC graph.
    refusals : tuple[str, ...]
        The target's named refusal kinds, if the export failed.
    detail : str or None
        The backend diagnostic including any failed hypothesis.
    laws : tuple[EffectLawReport, ...]
        The plan's source law reports and their existing evidence.
    native_jvp : EffectJvpCheck or None
        Requested exact comparison of source and selected QIEC graph.
    derivative_rule : RationalDerivativeDeclaration or None
        The nominal source derivative formula, if requested.
    rule_decision : RationalDerivativeDecision or None
        The exact recheck of that formula against the actual source.
    rule_obstacle : str or None
        The reason the source formula could not be checked, if any.
    external_derivative : DerivativeSemantics
        The report for the emitted backend's derivative. Emission and
        QIEC checks provide no evidence about that derivative.
    """

    target: str
    computation: ComputationId
    projection: Literal["source", "discharged"] | None
    stages: tuple[EffectExecutionStage, ...]
    refusals: tuple[str, ...]
    detail: str | None
    laws: tuple[EffectLawReport, ...]
    native_jvp: EffectJvpCheck | None
    derivative_rule: RationalDerivativeDeclaration | None
    rule_decision: RationalDerivativeDecision | None
    rule_obstacle: str | None
    external_derivative: DerivativeSemantics


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
        return _compare_jvp(
            self.bundle.module,
            self.discharge.module,
            self.computation,
            arguments,
            direction,
            fuel,
        )

    def check_target(
        self,
        target: str,
        *,
        arguments: tuple[int | Fraction, ...] | None = None,
        direction: tuple[int | Fraction, ...] | None = None,
        derivative_rule: RationalDerivativeDeclaration | None = None,
        fuel: int = 65_536,
        max_rule_work: int = 65_536,
    ) -> EffectTargetReport:
        """Report the actual export stage of a standalone QIEC computation.

        The registered projector selects the QIEC graph, and the regular
        renderer and emitter must accept the full standalone module.
        The optional pointwise comparison runs on QIEC. A supplied rule
        is checked separately on the original source. The backend code
        is emitted without executing its derivative.

        Parameters
        ----------
        target : str
            The public backend key accepted by the transpiler.
        arguments, direction : tuple[int or Fraction, ...] or None
            Optional input coordinates and tangent in source order.
            Supply both for a native pointwise comparison.
        derivative_rule : RationalDerivativeDeclaration or None
            A nominal source formula to certify or refute.
        fuel : int
            The step limit of each native derivative check.
        max_rule_work : int
            The symbolic work limit of the rule check.

        Returns
        -------
        EffectTargetReport
            The selected stage or refusal, with each observation scoped
            to the graph on which it was checked. Probabilistic programs
            require the separate source-level program target plan.

        Raises
        ------
        ValueError
            If the source identity, inputs or a requested bound is invalid.
        """
        from panproto import PanprotoError

        from quivers.transpile._api import UnsupportedConstruct
        from quivers.transpile._entry import _emit_target, available_targets
        from quivers.transpile.ir import IRProgram
        from quivers.transpile.qiec_ir import (
            capabilities_for_target,
            project_for_target,
        )

        source = self.bundle.module
        owner = next(
            (item for item in source.computations if item.id == self.computation), None
        )
        if owner is None:
            raise ValueError("the effect plan does not name a source computation")
        if (arguments is None) != (direction is None):
            raise ValueError(
                "supply both arguments and direction for a native comparison"
            )
        rule, rule_obstacle = _declared_rule(
            source, self.computation, derivative_rule, fuel, max_rule_work
        )
        stages: tuple[EffectExecutionStage, ...] = ()
        native_jvp = None
        projection: Literal["source", "discharged"] | None = None
        detail = None
        refusals: tuple[str, ...] = ()
        try:
            targets = available_targets()
            if target not in targets:
                raise UnsupportedConstruct(
                    target, [f"target:unknown:{target}:{','.join(targets)}"]
                )
            if source.gap:
                raise UnsupportedConstruct(
                    f"qvr-{target}", [source.gap.partition("; ")[0]]
                )
            if source.entries or self.computation in program_computations(source):
                public_name = next(
                    (
                        entry.name
                        for entry in source.entries
                        if entry.computation == self.computation
                    ),
                    owner.name,
                )
                raise UnsupportedConstruct(
                    f"qvr-{target}",
                    [f"qiec:target-stage:program-plan-required:{public_name}"],
                )
            capabilities = capabilities_for_target(target)
            ir = project_for_target(
                source, self.discharge, target, capabilities=capabilities
            )
            if not any(
                item.id.text == str(self.computation) and item.name == owner.name
                for item in ir.computations
            ):
                raise UnsupportedConstruct(
                    f"qvr-{target}",
                    [f"qiec:target-stage:entry-not-projected:{owner.name}"],
                )
            _emit_target(
                IRProgram(name=ir.module, inputs=(), body=(), module=ir), target
            )
        except UnsupportedConstruct as error:
            refusals = tuple(error.kinds)
            detail = str(error)
        except PanprotoError as error:
            refusals = (f"qiec:target-emission:{type(error).__name__}",)
            detail = str(error)
        else:
            if capabilities.supports("handle"):
                projection, selected = "source", source
                reached, _ = _execution_reachability(source, self.computation)
                stages = tuple(
                    EffectExecutionStage(
                        item.id,
                        "source-handler" if installed_handlers(item.body) else "direct",
                    )
                    for item in source.computations
                    if item.id in reached
                )
            else:
                projection, selected = "discharged", self.discharge.module
                stages = _ordinary_stages(source, self.discharge, self.computation)
            if arguments is not None and direction is not None:
                native_jvp = _compare_jvp(
                    source, selected, self.computation, arguments, direction, fuel
                )

        external = DerivativeSemantics.none(
            "target export refused" if refusals else "external derivative unverified"
        )
        return EffectTargetReport(
            target,
            self.computation,
            projection,
            stages,
            refusals,
            detail,
            self.laws,
            native_jvp,
            derivative_rule,
            rule,
            rule_obstacle,
            external,
        )


def _declared_rule(
    source: QiecModule,
    computation: ComputationId,
    declaration: RationalDerivativeDeclaration | None,
    fuel: int,
    max_work: int,
) -> tuple[RationalDerivativeDecision | None, str | None]:
    """Check the optional authored rule against the actual named source.

    Parameters
    ----------
    source : QiecModule
        The checked source defining the primal and action.
    computation : ComputationId
        The entry's nominal identity.
    declaration : RationalDerivativeDeclaration or None
        The optional authored rule to check.
    fuel, max_work : int
        The interpretation and symbolic proof limits.

    Returns
    -------
    tuple[RationalDerivativeDecision or None, str or None]
        The exact decision, including any counterexample, or an obstacle.

    Raises
    ------
    ValueError
        If a requested proof bound is invalid.
    """
    if declaration is None:
        return None, None
    if declaration.primal != computation:
        return None, "the declared derivative belongs to another named computation"
    try:
        decision = check_rational_derivative_rule(
            source, declaration, fuel=fuel, max_work=max_work
        )
    except (RationalLawLimit, RationalLawUnsupported) as error:
        return None, str(error)
    return decision, None


def _compare_jvp(
    source: QiecModule,
    staged: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    fuel: int,
) -> EffectJvpCheck:
    """Compare exact observations from two graphs of the same checked entry.

    Parameters
    ----------
    source, staged : QiecModule
        The original source and the selected QIEC graph.
    computation : ComputationId
        The nominal entry identity that both graphs preserve.
    arguments, direction : tuple[int or Fraction, ...]
        The exact input coordinates and tangent in positional order.
    fuel : int
        Steps permitted in each evaluation.

    Returns
    -------
    EffectJvpCheck
        Both observations or the reference refusals at that point.

    Raises
    ------
    ValueError
        If either graph lacks the identity or a point or bound is invalid.
    """
    owner = next(
        (entry for entry in source.computations if entry.id == computation),
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
        computation,
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


def _ordinary_stages(
    module: QiecModule, discharge: DischargedModule, computation: ComputationId
) -> tuple[EffectExecutionStage, ...]:
    """Read the stages of the source computations reached after discharge.

    Parameters
    ----------
    module : QiecModule
        The source graph with the named roots and their stable identities.
    discharge : DischargedModule
        Its actual discharged graph and the reasons for retained handlers.
    computation : ComputationId
        The checked root of this request.

    Returns
    -------
    tuple[EffectExecutionStage, ...]
        The selected native stages in source order.
    """
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
    return tuple(stages)


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
    return EffectPlan(
        bundle,
        computation,
        carriers,
        laws,
        discharge,
        tuple(comparisons),
        _ordinary_stages(module, discharge, computation),
    )


__all__ = [
    "EffectExecutionStage",
    "EffectInterchangeReport",
    "EffectJvpCheck",
    "EffectLawReport",
    "EffectPlan",
    "EffectTargetReport",
    "checked_effect_plan",
]
