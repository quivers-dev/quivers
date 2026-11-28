"""Stable declarations of QIEC effect equations and obligations."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import re
from typing import Literal, cast

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.laws import Assumed, Tested
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import Condition
from quivers.qiec.identifiers import (
    ComputationId,
    EffectId,
    EffectInstanceId,
    HandlerId,
    OperationId,
)
from quivers.qiec.module import QiecModule

type _CheckMethod = Literal["finite", "polynomial", "rational"]


class EffectEquation(QiecModel):
    """Declare a pair of checked computations over one nominal effect.

    The law's name is scoped to its effect. Its statement is the two
    computation bodies, checked under the listed nominal operation set.

    Parameters
    ----------
    effect : EffectId
        The declared interface owning the law.
    name : str
        The law's name within that interface.
    instance : EffectInstanceId
        The concrete instance whose requests form the two sides.
    operations : tuple[OperationId, ...]
        The operations occurring across both bodies.
    left : ComputationId
        The left computation.
    right : ComputationId
        The right computation, with the same positional domain and result.
    """

    effect: EffectId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    instance: EffectInstanceId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    operations: tuple[OperationId, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    left: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    right: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    def __init__(
        self,
        effect: EffectId,
        name: str,
        instance: EffectInstanceId,
        operations: tuple[OperationId, ...],
        left: ComputationId,
        right: ComputationId,
    ) -> None:
        # Didactic reads constructor fields from the concrete model.
        initialize = cast(Callable[..., None], super().__init__)
        initialize(
            effect=effect,
            name=name,
            instance=instance,
            operations=operations,
            left=left,
            right=right,
        )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a law name and a nonempty, distinct operation set.

        Raises
        ------
        ValueError
            If the name is empty or the operation set is empty or repeated.
        """
        if not self.name.strip():
            raise ValueError("an effect equation needs a name")
        if not self.operations or len(set(self.operations)) != len(self.operations):
            raise ValueError("an effect equation needs distinct performed operations")
        self._normalize_field("operations", tuple(sorted(self.operations, key=str)))


class EffectLawObligation(QiecModel):
    """A request to check one handler against a nominal effect equation.

    Parameters
    ----------
    equation : EffectEquation
        The exact nominal statement to check.
    handler : HandlerId
        The implementation the obligation addresses.
    method : {"finite", "polynomial", "rational"}
        The exact checking method requested.
    guards : tuple[Condition, ...]
        The named hypotheses declared with this obligation. The ordinary
        exact checks compare each side throughout the method's natural
        domain.
    """

    equation: EffectEquation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    handler: HandlerId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    method: _CheckMethod
    guards: tuple[Condition, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Canonicalize a list of distinct guard conditions.

        Raises
        ------
        ValueError
            If a guard name carries surrounding whitespace or is repeated.
        TypeError
            If a guard or another recursive field has an invalid type.
        """
        self._validate_recursive_fields()
        names = [guard.name for guard in self.guards]
        if any(name != name.strip() for name in names):
            raise ValueError(
                "effect law guard names must have no surrounding whitespace"
            )
        if len(set(names)) != len(names):
            raise ValueError("duplicate effect law guard name")
        self._normalize_field(
            "guards", tuple(sorted(self.guards, key=lambda guard: guard.name))
        )


class EffectLawBundle(QiecModel):
    """A QIEC module with declarative law checks for that module.

    Parameters
    ----------
    module : QiecModule
        The unchanged QIEC compilation unit.
    obligations : tuple[EffectLawObligation, ...]
        The requested checks, with one statement per scoped effect-law
        name and one obligation per equation, handler and method.
    """

    module: QiecModule = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    obligations: tuple[EffectLawObligation, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Check the nominal links and uniqueness of the declared obligations.

        Raises
        ------
        ValueError
            If an obligation names a mismatched module declaration or
            repeats a subject, or two equations reuse a scoped law name.
        TypeError
            If the module or an obligation has an invalid type.
        """
        self._validate_recursive_fields()
        effects = {
            item.ref.id: {op.id for op in item.operations}
            for item in self.module.effects
        }
        instances = {
            item.entry.instance: item.entry.effect for item in self.module.instances
        }
        handlers = {item.id: item.effect for item in self.module.handlers}
        computations = {item.id for item in self.module.computations}
        equations: dict[tuple[EffectId, str], EffectEquation] = {}
        subjects: set[tuple[EffectId, str, HandlerId, _CheckMethod]] = set()
        for obligation in self.obligations:
            eq = obligation.equation
            instance = instances.get(eq.instance)
            if (
                instance is None
                or instance.id != eq.effect
                or handlers.get(obligation.handler) != instance
                or not set(eq.operations) <= effects.get(eq.effect, set())
                or eq.left not in computations
                or eq.right not in computations
            ):
                raise ValueError(
                    "effect law obligation names a mismatched module declaration"
                )
            key = (eq.effect, eq.name)
            if eq != equations.setdefault(key, eq):
                raise ValueError("different equations reuse a nominal effect law name")
            subject = (*key, obligation.handler, obligation.method)
            if subject in subjects:
                raise ValueError("duplicate effect law obligation")
            subjects.add(subject)
        self._normalize_field(
            "obligations",
            tuple(sorted(self.obligations, key=_obligation_key)),
        )


class EffectLawEvidence(QiecModel):
    """Record tested or assumed evidence for one exact authored law subject.

    Parameters
    ----------
    obligation : EffectLawObligation
        The nominal equation, handler, method, and hypotheses at issue.
    module_digest : str
        The SHA-256 digest of the exact canonical module supplying it.
    evidence : Tested or Assumed
        The empirical test or assumption declared with this obligation.
    """

    obligation: EffectLawObligation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    module_digest: str
    evidence: Tested | Assumed = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require a canonical lowercase SHA-256 digest.

        Raises
        ------
        ValueError
            If the digest is malformed.
        """
        self._validate_recursive_fields()
        if re.fullmatch(r"[0-9a-f]{64}", self.module_digest) is None:
            raise ValueError(
                "an effect law evidence digest must be a lowercase SHA-256"
            )

    @classmethod
    def for_module(
        cls,
        module: QiecModule,
        obligation: EffectLawObligation,
        evidence: Tested | Assumed,
    ) -> EffectLawEvidence:
        """Bind declared evidence to the exact module and law obligation.

        Parameters
        ----------
        module : QiecModule
            The executable QIEC module to which the evidence refers.
        obligation : EffectLawObligation
            The law and handler to which the evidence refers.
        evidence : Tested or Assumed
            The empirical result or assumption to record.

        Returns
        -------
        EffectLawEvidence
            Evidence with a checked nominal subject and source digest.
        """
        EffectLawBundle(module=module, obligations=(obligation,))
        return cls(
            obligation=obligation,
            module_digest=_module_digest(module),
            evidence=evidence,
        )


def _module_digest(module: QiecModule) -> str:
    """Identify the exact canonical executable source of an obligation.

    Parameters
    ----------
    module : QiecModule
        The checked executable module.

    Returns
    -------
    str
        The SHA-256 hexadecimal digest of the canonical QIEC encoding.
    """
    from quivers.qiec.serialization import dumps

    return sha256(dumps(module).encode()).hexdigest()


def _obligation_key(obligation: EffectLawObligation) -> tuple[str, str, str, str]:
    """Order the obligation set by nominal owner and requested check.

    Parameters
    ----------
    obligation : EffectLawObligation
        One declared check on the module.

    Returns
    -------
    tuple[str, str, str, str]
        The effect ID, local law name, handler ID and method.
    """
    return (
        str(obligation.equation.effect),
        obligation.equation.name,
        str(obligation.handler),
        obligation.method,
    )


__all__ = [
    "EffectEquation",
    "EffectLawBundle",
    "EffectLawEvidence",
    "EffectLawObligation",
]
