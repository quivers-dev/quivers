"""A serializable scoped interpretation contract separate from executable QIEC."""

from __future__ import annotations

from dataclasses import replace

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec._scoped_proof_fold import (
    _operation_signature,
    _scoped_definitions,
    _scoped_entry,
    scoped_proof_fold,
)
from quivers.qiec._scoped_proof_types import scoped_proof_types
from quivers.qiec.contracts import InterpretationId
from quivers.qiec.effects import EffectRow
from quivers.qiec.identifiers import ComputationId, FamilyId
from quivers.qiec.module import NamedEffectInstance, QiecModule, validate_module
from quivers.qiec.rules import InterpretationDecl, RuleError, RulePresentation
from quivers.qiec.types import TypeExpr


class ScopedInterpretationBinding(QiecModel):
    """A nominal association from a structural interpretation to checked code.

    This is a companion record. Its codec never adds fields to the
    executable module. A decoded record is checked against its module
    before it is used as schema evidence.

    Parameters
    ----------
    presentation, interpretation
        The original structural rules and their nominal operation assignments.
    computation, judgment
        The exported computation and the judgment it interprets.
    carriers, rows
        The exact semantic carrier and closed effect row per judgment.
    instances
        The lexical names and identities of the effects required by those rows.
    """

    presentation: RulePresentation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    interpretation: InterpretationDecl = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    computation: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    judgment: FamilyId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    carriers: tuple[tuple[FamilyId, TypeExpr], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    rows: tuple[tuple[FamilyId, EffectRow], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    instances: tuple[NamedEffectInstance, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def validate(self, module: QiecModule) -> None:
        """Check the companion against its actual executable declarations.

        Check the original structural rules against the generated proof
        declarations and check the entry, applications and fold bodies
        against standard QIEC generation at the claimed carriers and
        rows. The stable rule-to-operation mapping must agree on every
        branch. The scoped source contract carries no weight parameters
        or claims about preserving equations.

        Parameters
        ----------
        module
            The executable module the companion purports to describe.

        Raises
        ------
        RuleError
            If a scoped rule, entry, effect, carrier, or operation
            assignment does not match.
        KernelError
            If the executable QIEC module itself is ill typed.
        """
        validate_module(module)
        presentation = self.presentation
        interpretation = self.interpretation
        if presentation.id != interpretation.presentation or interpretation.preserves:
            raise RuleError(
                "scoped interpretation does not match its structural presentation"
            )
        if (
            presentation.axioms
            or presentation.index_sorts
            or any(rule.weight is not None for rule in presentation.rules)
        ):
            raise RuleError(
                "scoped interpretation bindings require structural source rules"
            )

        computations = {item.id: item for item in module.computations}
        root = computations.get(self.computation)
        if root is None or self.judgment not in presentation.judgments_by_id:
            raise RuleError("unknown scoped interpretation root or judgment")
        expected_id = InterpretationId.derive(
            module.module, root.name, "scoped-interpretation"
        )
        if interpretation.id != expected_id:
            raise RuleError(
                "nominal scoped interpretation identity does not match its root"
            )
        carriers = dict(self.carriers)
        rows = dict(self.rows)
        if len(carriers) != len(self.carriers) or root.type.result != carriers.get(
            self.judgment
        ):
            raise RuleError(
                "scoped interpretation carrier does not match the checked root"
            )
        if len(rows) != len(self.rows) or root.type.effects != rows.get(self.judgment):
            raise RuleError(
                "scoped interpretation effect row does not match the checked root"
            )

        required = {
            entry.instance: entry for row in rows.values() for entry in row.entries
        }
        known = {instance.entry.instance: instance for instance in module.instances}
        named = {instance.entry.instance: instance for instance in self.instances}
        if (
            len(named) != len(self.instances)
            or named.keys() != required.keys()
            or any(
                known.get(identity) != instance or instance.entry != required[identity]
                for identity, instance in named.items()
            )
        ):
            raise RuleError(
                "scoped interpretation lexical effect requirements are stale"
            )

        syntax = scoped_proof_types(presentation)
        declared = syntax.module
        families = {family.id: family for family in module.families}
        constructors = {
            constructor.id: constructor for constructor in module.constructors
        }
        if any(
            families.get(family.id) != family for family in declared.families
        ) or any(
            constructors.get(constructor.id) != constructor
            for constructor in declared.constructors
        ):
            raise RuleError(
                "scoped interpretation presentation differs from checked proof types"
            )
        positioned = replace(
            syntax,
            module=declared.with_(
                module=module.module, source_protocol=module.source_protocol
            ),
        )
        plan = scoped_proof_fold(positioned, carriers, rows, name=root.name)
        assignments = dict(interpretation.operations)
        if len(assignments) != len(interpretation.operations) or any(
            operation not in computations for operation in assignments.values()
        ):
            raise RuleError("scoped interpretation operation assignments are invalid")
        operations = {
            source: _operation_signature(computations[operation])
            for source, operation in assignments.items()
        }
        generated, _ = _scoped_definitions(plan, operations)
        if _scoped_entry(plan, self.judgment, root.origin) != root:
            raise RuleError(
                "scoped interpretation entry does not match the checked root"
            )
        if any(computations.get(item.id) != item for item in generated):
            raise RuleError(
                "scoped interpretation assignments differ from checked fold bodies"
            )
