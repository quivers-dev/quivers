"""Portable observations of checked effect target stages."""

from __future__ import annotations

import re
from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import DerivativeSemantics
from quivers.qiec.identifiers import ComputationId

type _Stage = Literal[
    "direct",
    "handler-discharge",
    "recursive-state",
    "reference",
    "source-handler",
    "program-plan",
]
type _HelperProjection = Literal["source", "discharged"]
type _TargetProjection = Literal["source", "discharged", "program-plan"]
type _ObstacleKind = Literal[
    "foreign", "forwarding", "recursion", "case", "growth", "address"
]

_PROJECTION_STAGES: dict[_TargetProjection, frozenset[_Stage]] = {
    "source": frozenset(("direct", "source-handler")),
    "discharged": frozenset(
        ("direct", "handler-discharge", "recursive-state", "reference")
    ),
    "program-plan": frozenset(("program-plan",)),
}


class EffectStageRecord(QiecModel):
    """One named source computation's observed target stage.

    Parameters
    ----------
    computation : ComputationId
        The original computation identity.
    stage : str
        A recognized effect execution stage in the emitted projection.
    companions : tuple of ComputationId
        The generated computations retained by a recursive state rewrite.
    obstacle : str or None
        The named discharge obstacle for a retained reference computation.
    """

    computation: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    stage: _Stage
    companions: tuple[ComputationId, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    obstacle: _ObstacleKind | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Check the stage's allowed rewrite and refusal information.

        Raises
        ------
        ValueError
            If the named stage does not match the companions or obstacle.
        """
        self._validate_recursive_fields()
        if (self.stage == "reference") != (self.obstacle is not None):
            raise ValueError("only a reference stage needs a discharge obstacle")
        if (self.stage == "recursive-state") != bool(self.companions):
            raise ValueError("only a recursive state stage needs companions")
        if self.computation in self.companions or len(set(self.companions)) != len(
            self.companions
        ):
            raise ValueError(
                "effect stage companions need distinct generated identities"
            )


class EffectTargetEvidence(QiecModel):
    """An inert receipt of target emission or a named refusal.

    Loading checks the shape. The effect plan that owns the checked source
    can independently verify the receipt by repeating target preparation.

    Parameters
    ----------
    source_digest : str
        The SHA-256 digest of the canonical checked QIEC module.
    computation : ComputationId
        The selected source entry.
    target : str
        The requested public backend key.
    projection : str or None
        The observed source, discharged, or program-plan projection, if emitted.
    helper_projection : str or None
        The separate helper projection for an emitted program plan.
    stages : tuple of EffectStageRecord
        The emitted graph's source stage records in report order.
    artifact_digest : str or None
        SHA-256 of the exact emitted bytes, if emission succeeded.
    refusals : tuple of str
        The named refusals returned by the target preparation or emitter.
    """

    source_digest: str
    computation: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    target: str
    projection: _TargetProjection | None
    helper_projection: _HelperProjection | None
    stages: tuple[EffectStageRecord, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    artifact_digest: str | None
    refusals: tuple[str, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require canonical identity and consistent emitted or refused facts.

        Raises
        ------
        ValueError
            If an identity is malformed or the reported stage shape is inconsistent.
        """
        self._validate_recursive_fields()
        if re.fullmatch(r"[0-9a-f]{64}", self.source_digest) is None or (
            self.artifact_digest is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.artifact_digest) is None
        ):
            raise ValueError("effect target evidence needs lowercase SHA-256 digests")
        if not self.target.strip() or any(not kind.strip() for kind in self.refusals):
            raise ValueError(
                "effect target evidence needs nonempty target and refusal names"
            )
        if len(set(self.refusals)) != len(self.refusals):
            raise ValueError("effect target evidence repeats a refusal")
        if self.projection is None:
            if (
                not self.refusals
                or self.artifact_digest is not None
                or self.helper_projection is not None
                or self.stages
            ):
                raise ValueError(
                    "a refused target cannot claim emitted stages or bytes"
                )
            return
        if self.artifact_digest is None or self.refusals:
            raise ValueError(
                "an emitted target needs an artifact digest and no refusals"
            )
        program = self.projection == "program-plan"
        if program != (self.helper_projection is not None) or (
            program and len(self.stages) != 1
        ):
            raise ValueError(
                "a program projection needs its single stage and helper projection"
            )
        computations = {stage.computation for stage in self.stages}
        if (
            len(computations) != len(self.stages)
            or self.computation not in computations
        ):
            raise ValueError(
                "effect target stages need distinct identities and the selected entry"
            )
        if any(
            stage.stage not in _PROJECTION_STAGES[self.projection]
            for stage in self.stages
        ):
            raise ValueError(
                "an effect stage is incompatible with the target projection"
            )

    @property
    def external_derivative(self) -> DerivativeSemantics:
        """Report the derivative scope that target emission establishes.

        Returns
        -------
        DerivativeSemantics
            The existing external-unverified or export-refused observation.
        """
        reason = (
            "target export refused"
            if self.refusals
            else "external derivative unverified"
        )
        return DerivativeSemantics.none(reason)


__all__ = ["EffectStageRecord", "EffectTargetEvidence"]
