"""Elaborate an explicitly Cartesian interpretation of a structural signature."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import didactic.api as dx

from quivers.dsl import ast_nodes as surface
from quivers.dsl.compiler.structural import _signature_from_declaration
from quivers.dsl.structural_elaboration import _root_sort
from quivers.qiec import scoped_proof_fold, scoped_proof_types
from quivers.qiec._scoped_interpretation_binding import ScopedInterpretationBinding
from quivers.qiec._scoped_proof_fold import _fold_id, _scoped_definitions, _scoped_entry
from quivers.qiec._scoped_proof_types import ScopedProofTypes
from quivers.qiec.checking import ComputationSignature, KernelError
from quivers.qiec.contracts import InterpretationId
from quivers.qiec.effects import EffectRow, RowEntry
from quivers.qiec.identifiers import ComputationId, ConstructorId, FamilyId
from quivers.qiec.kinds import TYPE, TypeBinder
from quivers.qiec.module import (
    NamedComputation,
    QiecModule,
    _execution_reachability,
    _validate_named_static_scope,
)
from quivers.qiec.rules import (
    CARTESIAN,
    InterpretationDecl,
    RuleError,
    RulePresentation,
    presentation_of,
)
from quivers.qiec.substitution import instantiate_telescope, substitute_type
from quivers.qiec.types import TypeExpr, TypeVariable

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator


@dataclass(frozen=True)
class ScopedSource:
    """The generated declarations and entry judgment of a structural signature."""

    syntax: ScopedProofTypes
    root: FamilyId


@dataclass(frozen=True)
class _ScopedAction:
    """A named operation and its type instantiated at the ambient context."""

    clause: surface.QiecInterpretationClause
    operation: ComputationSignature
    parameters: tuple[TypeExpr, ...]
    result: TypeExpr


def _scoped_presentation(source: surface.SignatureDecl) -> RulePresentation:
    """Apply the explicit Cartesian contract to a structural source signature."""
    return presentation_of(_signature_from_declaration(source)).with_(
        structure=CARTESIAN
    )


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
            if option.key not in {"structure", "effects", "carriers"}:
                elaborator._fail(
                    option,
                    f"unknown scoped interpretation option {option.key!r}",
                    code="qiec-interpretation",
                )
        if name in made:
            continue
        try:
            presentation = _scoped_presentation(structural.declaration)
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
) -> EffectRow:
    """Resolve the closed lexical row shared by the source judgments."""
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
    return EffectRow(entries=tuple(entries.values()))


def _scoped_carriers(
    elaborator: _Elaborator,
    declaration: surface.QiecInterpretationDecl,
    source: ScopedSource,
    actions: dict[ConstructorId, _ScopedAction],
) -> dict[FamilyId, TypeExpr]:
    """Choose common carriers or infer a closed codomain for each judgment."""
    semantic = elaborator._lower_type(declaration.carrier, ())
    options = [option for option in declaration.options if option.key == "carriers"]
    if not options:
        return dict.fromkeys(source.syntax.proofs, semantic)
    if (
        len(options) != 1
        or not isinstance(options[0].value, surface.OptionName)
        or options[0].value.value != "by_judgment"
    ):
        elaborator._fail(
            options[0],
            "scoped carriers require one carriers=by_judgment option",
            code="qiec-interpretation",
        )
    presentation = source.syntax.presentation
    carriers: dict[FamilyId, TypeExpr] = {source.root: semantic}
    for rule in presentation.rules:
        action = actions[rule.id]
        family = rule.conclusion.family
        name = presentation.judgments_by_id[family].name
        try:
            _validate_named_static_scope(action.result, ())
        except KernelError:
            elaborator._fail(
                action.clause,
                f"scoped carrier for judgment {name!r} must be closed independently "
                "of the ambient Type binder",
                code="qiec-interpretation",
            )
        expected = carriers.setdefault(family, action.result)
        if action.result != expected:
            if family == source.root:
                message = f"scoped operation {action.operation.name!r} disagrees with the declared root carrier"
            else:
                message = (
                    f"scoped producers for judgment {name!r} disagree on their carriers"
                )
            elaborator._fail(
                action.clause,
                f"{message}: expected {elaborator._render(expected)}, "
                f"got {elaborator._render(action.result)}",
                code="qiec-interpretation",
            )
    missing = [
        judgment.name
        for judgment in presentation.judgments
        if judgment.id not in carriers
    ]
    if missing:
        elaborator._fail(
            declaration,
            f"no declared rule operation produces a closed carrier for {', '.join(missing)}",
            code="qiec-interpretation",
        )
    return carriers


def lower_scoped_interpretation(
    elaborator: _Elaborator,
    declaration: surface.QiecInterpretationDecl,
    source: ScopedSource,
) -> tuple[NamedComputation, ...]:
    """Make ordinary named computations at each judgment's checked carrier."""
    syntax = source.syntax
    row = _scoped_effect_row(elaborator, declaration)
    rules = {rule.name: rule.id for rule in syntax.presentation.rules}
    actions: dict[ConstructorId, _ScopedAction] = {}
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
        if identity in actions:
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
        actions[identity] = _ScopedAction(clause, operation, parameters, result)
    missing = [name for name, identity in rules.items() if identity not in actions]
    if missing:
        elaborator._fail(
            declaration,
            f"missing scoped interpretation for {', '.join(missing)}",
            code="qiec-interpretation",
        )
    carriers = _scoped_carriers(elaborator, declaration, source, actions)
    plan = scoped_proof_fold(
        syntax, carriers, dict.fromkeys(carriers, row), name=declaration.name
    )
    for rule in syntax.presentation.rules:
        action = actions[rule.id]
        if (
            action.parameters != plan.operation_parameters(rule.id, ambient)
            or action.result != carriers[rule.conclusion.family]
        ):
            elaborator._fail(
                action.clause,
                f"scoped operation {action.operation.name!r} needs the rule's typed arguments "
                "and the declared carrier",
                code="qiec-interpretation",
            )
    generated, _ = _scoped_definitions(
        plan, {identity: action.operation for identity, action in actions.items()}
    )
    origin = elaborator._origin(
        declaration, ("scoped_interpretations", "entry"), "computation"
    )
    return (*generated, _scoped_entry(plan, source.root, origin))


def selected_scoped_interpretations(
    source: surface.Module,
    module: QiecModule,
    selected: ComputationId,
    *,
    reached: set[ComputationId] | None = None,
) -> tuple[ScopedInterpretationBinding, ...]:
    """Extract and check the scoped contracts reached from one program."""
    declarations = [
        declaration
        for declaration in source.statements
        if isinstance(declaration, surface.QiecInterpretationDecl)
        and declaration.options
    ]
    if not declarations:
        return ()
    scopes = {
        declaration.name: declaration
        for declaration in source.statements
        if isinstance(declaration, surface.SignatureDecl)
    }
    if reached is None:
        reached, _ = _execution_reachability(module, selected)
    compiled = {computation.name: computation for computation in module.computations}
    by_id = {computation.id: computation for computation in compiled.values()}
    prepared: dict[str, RulePresentation] = {}
    bindings = []
    for declaration in declarations:
        root = compiled[declaration.name]
        if root.id not in reached:
            continue
        scope = scopes[declaration.deduction]
        presentation = prepared.get(scope.name)
        if presentation is None:
            presentation = _scoped_presentation(scope)
            prepared[scope.name] = presentation
        root_sort = _root_sort(scope)
        judgment = next(
            item.id for item in presentation.judgments if item.name == root_sort
        )
        rules = {rule.name: rule.id for rule in presentation.rules}
        operations = tuple(
            (rules[clause.constructor], compiled[clause.operation].id)
            for clause in declaration.clauses
        )
        identities = {entry.instance for entry in root.type.effects.entries}
        carriers = []
        for item in presentation.judgments:
            fold = by_id.get(_fold_id(module.module, root.name, item.id))
            if fold is None:
                raise RuleError(f"no checked scoped fold for judgment {item.name!r}")
            carriers.append((item.id, fold.type.result))
        binding = ScopedInterpretationBinding(
            presentation=presentation,
            interpretation=InterpretationDecl(
                id=InterpretationId.derive(
                    module.module, declaration.name, "scoped-interpretation"
                ),
                presentation=presentation.id,
                operations=operations,
            ),
            computation=root.id,
            judgment=judgment,
            carriers=tuple(carriers),
            rows=tuple((item.id, root.type.effects) for item in presentation.judgments),
            instances=tuple(
                item for item in module.instances if item.entry.instance in identities
            ),
        )
        binding.validate(module)
        bindings.append(binding)
    return tuple(bindings)
