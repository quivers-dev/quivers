"""Elaborate an explicitly Cartesian interpretation of a structural signature."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import didactic.api as dx

from quivers.dsl import ast_nodes as surface
from quivers.dsl.compiler.structural import _signature_from_declaration
from quivers.qiec import scoped_proof_fold, scoped_proof_types
from quivers.qiec._scoped_proof_fold import _scoped_definitions
from quivers.qiec._scoped_proof_types import ScopedProofTypes, _at
from quivers.qiec.checking import ComputationSignature, KernelError
from quivers.qiec.effects import ComputationType, EffectRow, RowEntry
from quivers.qiec.identifiers import ComputationId, ConstructorId, FamilyId
from quivers.qiec.kinds import TYPE, TypeBinder
from quivers.qiec.module import NamedComputation
from quivers.qiec.rules import CARTESIAN, presentation_of
from quivers.qiec.substitution import instantiate_telescope, substitute_type
from quivers.qiec.terms import Call, ConstructorValue, Local, Var
from quivers.qiec.types import TypeVariable

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator


@dataclass(frozen=True)
class ScopedSource:
    """The generated declarations and entry judgment of a structural signature."""

    syntax: ScopedProofTypes
    root: FamilyId


def declare_scoped_types(elaborator: _Elaborator) -> dict[str, ScopedSource]:
    """Register scoped proof types before resolving authored operation signatures."""
    made: dict[str, ScopedSource] = {}
    for declaration in elaborator.statements:
        if not isinstance(declaration, surface.QiecInterpretationDecl):
            continue
        name = declaration.deduction
        structural = elaborator._structural_signatures.get(name)
        if structural is None:
            if declaration.options:
                elaborator._fail(
                    declaration,
                    "Cartesian interpretation options require a structural signature",
                    code="qiec-interpretation",
                )
            continue
        permissions = [
            option.value for option in declaration.options if option.key == "structure"
        ]
        if (
            len(permissions) != 1
            or not isinstance(permissions[0], surface.OptionName)
            or permissions[0].value != "cartesian"
        ):
            elaborator._fail(
                declaration,
                "a structural interpretation requires explicit [structure=cartesian]",
                code="qiec-interpretation",
            )
        for option in declaration.options:
            if option.key not in {"structure", "effects"}:
                elaborator._fail(
                    option,
                    f"unknown scoped interpretation option {option.key!r}",
                    code="qiec-interpretation",
                )
        if name in made:
            continue
        try:
            signature = _signature_from_declaration(structural.declaration)
            presentation = presentation_of(signature).with_(structure=CARTESIAN)
            syntax = scoped_proof_types(presentation)
        except (KernelError, TypeError, ValueError, dx.ValidationError) as error:
            elaborator._fail_kernel(declaration, error, fallback="qiec-interpretation")
        roots = tuple(
            judgment.id
            for judgment in presentation.judgments
            if judgment.name == structural.root
        )
        if not roots:
            elaborator._fail(
                declaration,
                f"structural signature {name!r} has no object root for a scoped fold",
                code="qiec-interpretation",
            )
        origin = elaborator._origin(
            declaration, ("scoped_interpretations",), "computation"
        )
        module = syntax.module.with_(
            module=origin.module, source_protocol=origin.source_protocol
        )
        syntax = replace(syntax, module=module)
        for family in module.families:
            if family.name in elaborator.families:
                elaborator._fail(
                    declaration,
                    f"duplicate scoped proof family {family.name!r}",
                    code="qiec-interpretation",
                )
            elaborator.families[family.name] = family
            elaborator.registry.register_family(family)
        for constructor in module.constructors:
            if constructor.name in elaborator.constructors:
                elaborator._fail(
                    declaration,
                    f"duplicate scoped proof constructor {constructor.name!r}",
                    code="qiec-interpretation",
                )
            elaborator.constructors[constructor.name] = constructor
            elaborator.registry.register_constructor(constructor)
        made[name] = ScopedSource(syntax=syntax, root=roots[0])
    return made


def _scoped_effect_row(
    elaborator: _Elaborator,
    declaration: surface.QiecInterpretationDecl,
    source: ScopedSource,
) -> EffectRow:
    """Resolve an explicit closed source row to its lexical effect instances."""
    effect_options = [
        option for option in declaration.options if option.key == "effects"
    ]
    if not effect_options:
        return EffectRow()
    option = effect_options[0]
    if len(effect_options) != 1 or not isinstance(option.value, surface.OptionList):
        elaborator._fail(
            option,
            "scoped effects require one effects=[instance, ...] option",
            code="qiec-interpretation",
        )
    entries: dict[str, RowEntry] = {}
    for item in option.value.items:
        if not isinstance(item, surface.OptionName):
            elaborator._fail(
                option,
                "scoped effects must name lexical effect instances",
                code="qiec-interpretation",
            )
        instance = elaborator.instances.get(item.value)
        if instance is None:
            elaborator._fail(
                option,
                f"unknown scoped effect instance {item.value!r}",
                code="qiec-interpretation",
            )
        if item.value in entries:
            elaborator._fail(
                option,
                f"duplicate scoped effect instance {item.value!r}",
                code="qiec-interpretation",
            )
        entries[item.value] = instance.entry
    if entries and len(source.syntax.proofs) != 1:
        elaborator._fail(
            option,
            "an authored scoped effect row requires a single object sort",
            code="qiec-interpretation",
        )
    return EffectRow(entries=tuple(entries.values()))


def lower_scoped_interpretation(
    elaborator: _Elaborator,
    declaration: surface.QiecInterpretationDecl,
    source: ScopedSource,
) -> tuple[NamedComputation, ...]:
    """Make ordinary named computations at the declared carrier and row."""
    syntax = source.syntax
    row = _scoped_effect_row(elaborator, declaration, source)
    semantic = elaborator._lower_type(declaration.carrier, ())
    carriers = {family: semantic for family in syntax.proofs}
    plan = scoped_proof_fold(
        syntax, carriers, dict.fromkeys(carriers, row), name=declaration.name
    )
    rules = {rule.name: rule.id for rule in syntax.presentation.rules}
    selected: dict[ConstructorId, ComputationSignature] = {}
    ambient = TypeVariable(name="scoped_rule_context")
    for clause in declaration.clauses:
        identity = (
            rules.get(clause.constructor) if clause.constructor_kind == "rule" else None
        )
        if identity is None:
            elaborator._fail(
                clause,
                f"unknown scoped structural rule {clause.constructor!r}",
                code="qiec-interpretation",
            )
        if identity in selected:
            elaborator._fail(
                clause,
                f"duplicate interpretation for rule {clause.constructor!r}",
                code="qiec-interpretation",
            )
        operation = elaborator.computation_signatures.get(clause.operation)
        if operation is None:
            elaborator._fail(
                clause,
                f"unknown interpretation operation {clause.operation!r}",
                code="qiec-interpretation",
            )
        if (
            len(operation.telescope) != 1
            or not isinstance(operation.telescope[0], TypeBinder)
            or operation.telescope[0].kind != TYPE
        ):
            elaborator._fail(
                clause,
                f"scoped operation {operation.name!r} needs one ambient Type binder",
                code="qiec-interpretation",
            )
        substitution = instantiate_telescope(operation.telescope, (ambient,))
        parameters = tuple(
            substitute_type(type_, substitution) for type_ in operation.parameters
        )
        result = substitute_type(operation.result, substitution)
        if (
            parameters != plan.operation_parameters(identity, ambient)
            or result != semantic
        ):
            elaborator._fail(
                clause,
                f"scoped operation {operation.name!r} needs the rule's typed arguments "
                "and the declared carrier",
                code="qiec-interpretation",
            )
        if operation.effects.tail is not None or any(
            row.lookup(entry.instance) != entry.effect
            for entry in operation.effects.entries
        ):
            elaborator._fail(
                clause,
                f"scoped operation {operation.name!r} declares effects outside "
                "the interpretation's closed effects row",
                code="qiec-interpretation",
            )
        selected[identity] = operation
    missing = [name for name, identity in rules.items() if identity not in selected]
    if missing:
        elaborator._fail(
            declaration,
            f"missing scoped interpretation for {', '.join(missing)}",
            code="qiec-interpretation",
        )
    generated, folds = _scoped_definitions(plan, selected)
    chosen = folds[source.root]
    empty_type = _at(syntax.empty)
    proof_type = _at(
        syntax.proofs[source.root], *syntax.carrier_types(carriers), empty_type
    )
    proof = Local(name="proof", type=proof_type)
    origin = elaborator._origin(
        declaration, ("scoped_interpretations", "entry"), "computation"
    )
    return (
        *generated,
        NamedComputation(
            id=ComputationId.derive(
                origin.module, declaration.name, "scoped-interpretation"
            ),
            name=declaration.name,
            telescope=(),
            parameters=(proof,),
            type=ComputationType(effects=row, result=semantic),
            origin=origin,
            body=Call(
                callee=chosen.id,
                name=chosen.name,
                static_arguments=(empty_type,),
                arguments=(
                    Var(local=proof),
                    ConstructorValue(
                        constructor=syntax.empty.constructors[0],
                        static_arguments=(),
                        fields=(),
                        result_type=empty_type,
                    ),
                ),
                result_type=semantic,
                effects=row,
                origin=origin,
            ),
        ),
    )
