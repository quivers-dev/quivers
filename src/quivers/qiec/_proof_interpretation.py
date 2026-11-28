"""Lower typed closed rule interpretations to ordinary QIEC proof folds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from quivers.qiec.checking import ComputationSignature, KernelError, KernelRegistry
from quivers.qiec.coverage import constructor_skolems
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl
from quivers.qiec.effects import ComputationType, EffectRow
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    FamilyId,
    SourceOrigin,
    StaticScopeId,
)
from quivers.qiec.kinds import IndexBinder
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.rules import AxiomSlot, InterpretationDecl, Rule, RulePresentation
from quivers.qiec.substitution import instantiate_telescope, substitute_type
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    Local,
    Var,
)
from quivers.qiec.types import IndexVariable, TypeApplication, TypeExpr


@dataclass(frozen=True)
class _RuleOperation:
    """A checked rule operation with its ordered premise and conclusion families.

    Parameters
    ----------
    premises
        The judgment family of each proof field, in constructor order.
    conclusion
        The judgment family the constructor proves.
    computation
        The checked computation acting on folded premises and value fields.
    """

    premises: tuple[FamilyId, ...]
    conclusion: FamilyId
    computation: ComputationSignature


def lower_closed_interpretation(
    module: QiecModule,
    presentation: RulePresentation,
    interpretation: InterpretationDecl,
    *,
    name: str,
    carriers: Mapping[FamilyId, TypeExpr],
) -> QiecModule:
    """Add a typed fold for each judgment family of a rule presentation.

    Each fold receives a checked proof and calls the corresponding rule
    operation after folding its premises in their declared order. No proof
    occurrence is reused across calls. Within a branch, the rule's rigid
    constructor indices specialize the recursive calls. A result carrier is
    closed, so those indices cannot escape in a result type.

    Parameters
    ----------
    module
        The checked module containing the interpretation's operations and
        the presentation's proof-family declarations.
    presentation
        The unconditional rule presentation of the input proofs.
    interpretation
        One checked computation identity per rule or axiom slot.
    name
        The generated fold name prefix, followed by `__` and the family name.
    carriers
        One closed result type per judgment family.

    Returns
    -------
    QiecModule
        The original module extended by checked, serializable computations.

    Raises
    ------
    KernelError
        If the signature, equation, context, or effect row falls outside this
        fold's contract, or the proof declarations and checked module disagree.
    """
    registry = validate_module(module)
    generated = prepare_closed_interpretation(
        registry,
        presentation,
        interpretation,
        names={
            judgment.id: f"{name}__{judgment.name}"
            for judgment in presentation.judgments
        },
        carriers=carriers,
        origin=SourceOrigin(
            module=module.module,
            structural_path=("proof_interpretations",),
            role="computation",
            source_protocol=module.source_protocol,
        ),
    )
    lowered = module.with_(computations=(*module.computations, *generated))
    validate_module(lowered)
    return lowered


def prepare_closed_interpretation(
    registry: KernelRegistry,
    presentation: RulePresentation,
    interpretation: InterpretationDecl,
    *,
    names: Mapping[FamilyId, str],
    carriers: Mapping[FamilyId, TypeExpr],
    origin: SourceOrigin,
) -> tuple[NamedComputation, ...]:
    """Lower a checked fold against signatures before surrounding bodies exist.

    Parameters
    ----------
    registry
        The kernel registry containing the proof constructors and operations.
    presentation
        The unconditional rule presentation of the input proofs.
    interpretation
        One checked computation identity per rule or axiom slot.
    names
        The computation name for each judgment family.
    carriers
        The closed result type for each judgment family.
    origin
        The source declaration to which the generated computations belong.

    Returns
    -------
    tuple of NamedComputation
        The generated cases and ordered calls, awaiting module validation.

    Raises
    ------
    KernelError
        If the interpretation, declaration signatures, or effects do not
        support the closed fold.
    """
    families, constructors = presentation.declarations()
    sources = presentation.constructors_by_id
    if interpretation.presentation != presentation.id:
        raise KernelError("a proof fold must interpret its own presentation")
    if presentation.conditions or presentation.equations or interpretation.preserves:
        raise KernelError(
            "closed proof folds require an unconditional presentation without equations"
        )
    if any(
        premise.hypotheses for rule in presentation.rules for premise in rule.premises
    ):
        raise KernelError(
            "closed proof folds cannot interpret a premise under hypotheses"
        )
    if any(registry.families.get(family.id) != family for family in families) or any(
        registry.constructors.get(constructor.id) != constructor
        for constructor in constructors
    ):
        raise KernelError(
            "a proof fold needs the presentation's checked proof declarations"
        )
    declared = {family.id for family in families}
    if set(names) != declared or len(set(names.values())) != len(declared):
        raise KernelError(
            "a proof fold needs a distinct name for every judgment family"
        )
    if set(carriers) != declared:
        raise KernelError(
            "a proof fold needs one closed result type per judgment family"
        )
    for type_ in carriers.values():
        registry.validate_type(type_)
    selected = dict(interpretation.operations)
    if set(selected) != set(sources) or len(selected) != len(interpretation.operations):
        raise KernelError(
            "a proof fold needs exactly one declared operation per constructor"
        )

    operations = {
        identity: _check_operation(
            source, registry.computation(selected[identity]), carriers
        )
        for identity, source in sources.items()
    }
    rows = _fold_rows(carriers, operations)
    signatures = {
        family.id: _fold_signature(
            origin.module,
            names[family.id],
            family,
            carriers[family.id],
            rows[family.id],
        )
        for family in families
    }
    return tuple(
        _fold(origin, family, signatures, operations, registry.constructors)
        for family in families
    )


def _check_operation(
    source: Rule | AxiomSlot,
    signature: ComputationSignature,
    carriers: Mapping[FamilyId, TypeExpr],
) -> _RuleOperation:
    """Check ordinary parameters, result and effect row against a rule.

    Parameters
    ----------
    source
        The rule or axiom slot being interpreted.
    signature
        The checked operation selected for this constructor.
    carriers
        The chosen result types.

    Returns
    -------
    _RuleOperation
        The typed operation and its premise and conclusion families.

    Raises
    ------
    KernelError
        If the operation has the wrong parameters, result, or effect row.
    """
    if isinstance(source, Rule):
        premises = tuple(premise.judgment.family for premise in source.premises)
        conclusion = source.conclusion.family
    else:
        premises = ()
        conclusion = source.judgment.family
    expected = (
        *(carriers[family] for family in premises),
        *(value.type for value in source.values),
    )
    if (
        signature.telescope
        or signature.effects.tail is not None
        or signature.parameters != expected
        or signature.result != carriers[conclusion]
    ):
        raise KernelError(
            f"operation {signature.name!r} does not match the closed interpretation of {source.name!r}",
            "qiec-call",
        )
    return _RuleOperation(premises, conclusion, signature)


def _fold_rows(
    carriers: Mapping[FamilyId, TypeExpr],
    operations: Mapping[ConstructorId, _RuleOperation],
) -> dict[FamilyId, EffectRow]:
    """Compute the least closed effect rows for mutually recursive folds.

    Parameters
    ----------
    carriers
        The judgment families and their result types.
    operations
        The typed operations by proof constructor.

    Returns
    -------
    dict of FamilyId to EffectRow
        The union of the rule rows each fold can reach.
    """
    direct = {family: EffectRow() for family in carriers}
    children: dict[FamilyId, set[FamilyId]] = {family: set() for family in carriers}
    for operation in operations.values():
        family = operation.conclusion
        direct[family] = direct[family].union(operation.computation.effects)
        children[family].update(operation.premises)
    rows = {}
    for family in carriers:
        seen: set[FamilyId] = set()
        pending = [family]
        row = EffectRow()
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                row = row.union(direct[current])
                pending.extend(children[current])
        rows[family] = row
    return rows


def _fold_signature(
    module: str, name: str, family: FamilyDecl, output: TypeExpr, effects: EffectRow
) -> ComputationSignature:
    """Give the fold an ordinary function signature over all proof indices.

    Parameters
    ----------
    module, name
        The module and the generated fold's name.
    family
        The indexed family of proofs.
    output, effects
        The closed result type and the reachable effect row.

    Returns
    -------
    ComputationSignature
        The signature of a fold over a single proof.
    """
    assert all(isinstance(index, IndexBinder) for index in family.indices)
    telescope = tuple(
        index.with_(name=f"i{position}", refinable=False)
        for position, index in enumerate(family.indices)
    )
    proof = TypeApplication(
        constructor=family.type_constructor,
        arguments=tuple(
            IndexVariable(name=index.name, sort=index.sort) for index in telescope
        ),
    )
    return ComputationSignature(
        id=ComputationId.derive(module, "computation", name),
        name=name,
        telescope=telescope,
        parameters=(proof,),
        result=output,
        effects=effects,
    )


def _fold(
    declaration: SourceOrigin,
    family: FamilyDecl,
    signatures: Mapping[FamilyId, ComputationSignature],
    operations: Mapping[ConstructorId, _RuleOperation],
    constructors: Mapping[ConstructorId, ConstructorDecl],
) -> NamedComputation:
    """Build the indexed case analysis of a checked proof.

    Parameters
    ----------
    declaration, family
        The declaration of the interpretation and the proof family consumed.
    signatures, operations, constructors
        The folds, checked interpretations, and proof constructors in scope.

    Returns
    -------
    NamedComputation
        The ordinary kernel case analysis for one family.
    """
    signature = signatures[family.id]
    proof = Local(name="proof", type=signature.parameters[0])
    origin = declaration.with_(
        structural_path=(*declaration.structural_path, signature.name)
    )
    return NamedComputation(
        id=signature.id,
        name=signature.name,
        telescope=signature.telescope,
        parameters=(proof,),
        type=ComputationType(effects=signature.effects, result=signature.result),
        body=Case(
            scrutinee=Var(local=proof),
            motive=CaseMotive(indices=family.indices, result_type=signature.result),
            branches=tuple(
                _fold_branch(constructors[id], operations[id], signatures, origin)
                for id in family.constructors
            ),
        ),
        origin=origin,
    )


def _fold_branch(
    constructor: ConstructorDecl,
    operation: _RuleOperation,
    signatures: Mapping[FamilyId, ComputationSignature],
    origin: SourceOrigin,
) -> CaseBranch:
    """Bind each folded premise before calling the constructor's operation.

    Parameters
    ----------
    constructor, operation
        The proof constructor and its checked interpretation.
    signatures
        One fold signature per premise family.
    origin
        The containing computation's source identity.

    Returns
    -------
    CaseBranch
        The branch with canonical rigid constructor indices and ordered calls.
    """
    path = (*origin.structural_path, "branch", constructor.name)
    scope = StaticScopeId.derive(origin.module, path)
    skolems = constructor_skolems(constructor, scope)
    substitution = instantiate_telescope(constructor.telescope, skolems)
    fields = tuple(
        Local(name=field.name, type=substitute_type(field.type, substitution))
        for field in constructor.fields
    )
    arity = len(operation.premises)
    children = []
    for index, (family, child) in enumerate(
        zip(operation.premises, fields[:arity], strict=True)
    ):
        signature = signatures[family]
        assert isinstance(child.type, TypeApplication)
        folded = Local(name=f"folded_premise{index}", type=signature.result)
        children.append(
            (
                folded,
                Call(
                    callee=signature.id,
                    name=signature.name,
                    static_arguments=child.type.arguments,
                    arguments=(Var(local=child),),
                    result_type=signature.result,
                    effects=signature.effects,
                    origin=origin.with_(
                        structural_path=(*path, "premise", index), role="call"
                    ),
                ),
            )
        )
    signature = operation.computation
    body: Computation = Call(
        callee=signature.id,
        name=signature.name,
        static_arguments=(),
        arguments=tuple(
            Var(local=value)
            for value in (*(local for local, _ in children), *fields[arity:])
        ),
        result_type=signature.result,
        effects=signature.effects,
        origin=origin.with_(structural_path=(*path, "operation"), role="call"),
    )
    for local, call in reversed(children):
        body = Bind(binder=local, first=call, then=body)
    return CaseBranch(
        constructor=constructor.id,
        static_arguments=skolems,
        fields=fields,
        body=body,
        scope=scope,
    )
