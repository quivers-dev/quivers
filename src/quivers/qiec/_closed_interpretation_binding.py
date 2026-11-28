"""Standalone contracts for closed lexical proof interpretations."""

from __future__ import annotations

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._lexical_proof_types import export_lexical_proofs
from quivers.qiec._models import QiecModel
from quivers.qiec._proof_interpretation import prepare_closed_interpretation
from quivers.qiec.checking import KernelError
from quivers.qiec.contracts import InterpretationId, PresentationId
from quivers.qiec.effects import EffectRow
from quivers.qiec.foreign import DeductionOutcomes, DeductionProofs
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.module import NamedEffectInstance, QiecModule, validate_module
from quivers.qiec.rules import ORDERED, InterpretationDecl, RuleError, RulePresentation
from quivers.qiec.types import TypeExpr


class ClosedInterpretationBinding(QiecModel):
    """A checked source association for one lexical proof interpretation.

    Each association serializes as a companion to the executable module.
    Its presentation describes the proof constructors. When a packed
    query is available, validation also compares the query's presentation.

    Parameters
    ----------
    presentation, interpretation
        The indexed proof presentation and the assignments of its
        constructors to nominal rule operations.
    computation
        The exported computation over the presentation's proof family.
    carrier
        The result type over the judgment's original index binders.
    effects, instances
        The closed row and its named lexical effect requirements.
    """

    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    interpretation: InterpretationDecl = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    computation: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    carrier: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    effects: EffectRow = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    instances: tuple[NamedEffectInstance, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def validate(self, module: QiecModule) -> None:
        """Reconstruct the fold and compare its exact checked declaration.

        Parameters
        ----------
        module
            The checked executable this companion describes.

        Raises
        ------
        RuleError
            If a presentation, root, rule assignment or requirement is
            stale or mismatches its executable counterpart.
        KernelError
            If the executable module itself is ill typed.
        """
        registry = validate_module(module)
        presentation = self.presentation
        interpretation = self.interpretation
        if (
            presentation.id != PresentationId.derive(presentation.name)
            or presentation.id != interpretation.presentation
            or presentation.structure != ORDERED
            or len(presentation.judgments) != 1
        ):
            raise RuleError("nominal closed interpretation presentation is stale")
        (judgment,) = presentation.judgments
        root = next(
            (item for item in module.computations if item.id == self.computation), None
        )
        if root is None:
            raise RuleError("unknown closed interpretation root")
        origin = root.origin.with_(structural_path=("proof_interpretations",))
        if interpretation.id != InterpretationId.derive(
            module.module, *origin.structural_path, root.name
        ):
            raise RuleError("nominal closed interpretation identity is stale")
        if root.type.effects != self.effects:
            raise RuleError("closed interpretation effect row does not match its root")
        known = {item.entry.instance: item for item in module.instances}
        required = {entry.instance: entry for entry in self.effects.entries}
        supplied = {item.entry.instance: item for item in self.instances}
        if (
            len(supplied) != len(self.instances)
            or supplied.keys() != required.keys()
            or any(
                known.get(identity) != item or required[identity] != item.entry
                for identity, item in supplied.items()
            )
        ):
            raise RuleError(
                "closed interpretation lexical effect requirements are stale"
            )

        for handler in module.handlers:
            query = handler.foreign
            if isinstance(query, DeductionProofs):
                query = query.chart
            if (
                isinstance(query, DeductionOutcomes)
                and isinstance(query.presentation, RulePresentation)
                and query.presentation.id == presentation.id
                and export_lexical_proofs(query.presentation, presentation.name)
                != presentation
            ):
                raise RuleError("closed interpretation differs from its packed query")
        try:
            (expected,) = prepare_closed_interpretation(
                registry,
                presentation,
                interpretation,
                names={judgment.id: root.name},
                carriers={judgment.id: self.carrier},
                origin=origin,
            )
        except KernelError as error:
            raise RuleError(f"invalid closed interpretation: {error}") from error
        if expected != root:
            raise RuleError("closed interpretation differs from its checked fold")
