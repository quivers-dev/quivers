"""Lower typed closed rule interpretations to ordinary QIEC proof folds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quivers.qiec.checking import (
    ComputationSignature,
    KernelError,
    KernelRegistry,
    StaticScope,
)
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
from quivers.qiec.kinds import IndexBinder, Telescope
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.rules import (
    AxiomSlot,
    InterpretationDecl,
    JudgmentDecl,
    Rule,
    RulePresentation,
)
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
from quivers.qiec.types import IndexVariable, StaticArgument, TypeApplication, TypeExpr


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


def bind_operands(
    operands: Sequence[tuple[Local, Computation]], body: Computation
) -> Computation:
    """Evaluate ordered operands once each before a typed operation.

    Parameters
    ----------
    operands
        Fresh result locals paired with their computations, in source order.
    body
        The operation using those results. Its locals are bound by the operands.

    Returns
    -------
    Computation
        The existing QIEC binds that sequence the operand effects.
    """
    for local, computation in reversed(operands):
        body = Bind(binder=local, first=computation, then=body)
    return body


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
    constructor indices specialize the recursive calls and their result
    carriers. Only the declared judgment indices can appear in the
    carrier of the resulting proof fold.

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
        One result type per judgment family, over that family's indices.

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
        The result type for each judgment family, over its declared indices.
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
        raise KernelError("a proof fold needs one result type per judgment family")
    judgments = presentation.judgments_by_id
    for family in families:
        scope = StaticScope().extend(
            judgments[family.id].indices, subject="proof interpretation carrier"
        )
        registry.validate_type(carriers[family.id], scope)
    selected = dict(interpretation.operations)
    if set(selected) != set(sources) or len(selected) != len(interpretation.operations):
        raise KernelError(
            "a proof fold needs exactly one declared operation per constructor"
        )

    operations = {
        identity: _check_operation(
            source,
            registry.computation(selected[identity]),
            judgments,
            carriers,
            registry,
        )
        for identity, source in sources.items()
    }
    rows = _fold_rows(carriers, operations)
    signatures = {
        family.id: _fold_signature(
            origin.module,
            names[family.id],
            family,
            judgments[family.id].indices,
            carriers[family.id],
            rows[family.id],
        )
        for family in families
    }
    return tuple(
        _fold(
            origin,
            family,
            signatures,
            operations,
            registry.constructors,
            CaseMotive(
                indices=judgments[family.id].indices, result_type=carriers[family.id]
            ),
        )
        for family in families
    )


def _carrier_at(
    carrier: TypeExpr, indices: Telescope, arguments: tuple[StaticArgument, ...]
) -> TypeExpr:
    """Instantiate a result carrier at one judgment's indices.

    Parameters
    ----------
    carrier, indices
        The carrier type and the binders it abstracts.
    arguments
        The checked static arguments supplied to those binders.

    Returns
    -------
    TypeExpr
        The result type at this instance of the judgment.
    """
    return substitute_type(carrier, instantiate_telescope(indices, arguments))


def _check_operation(
    source: Rule | AxiomSlot,
    signature: ComputationSignature,
    judgments: Mapping[FamilyId, JudgmentDecl],
    carriers: Mapping[FamilyId, TypeExpr],
    registry: KernelRegistry,
) -> _RuleOperation:
    """Check ordinary parameters, result and effect row against a rule.

    Parameters
    ----------
    source
        The rule or axiom slot being interpreted.
    signature
        The checked operation selected for this constructor.
    judgments
        The declaration of each judgment family, including its index binders.
    carriers
        The carrier types abstracted over those binders.
    registry
        The kernel declarations needed to validate instance references.

    Returns
    -------
    _RuleOperation
        The typed operation and its premise and conclusion families.

    Raises
    ------
    KernelError
        If the operation has the wrong parameters, result, or effect row.
    """
    premises = (
        tuple(premise.judgment for premise in source.premises)
        if isinstance(source, Rule)
        else ()
    )
    conclusion = source.conclusion if isinstance(source, Rule) else source.judgment
    error = f"operation {signature.name!r} does not match the closed interpretation of {source.name!r}"
    if signature.telescope and not all(
        isinstance(binder, IndexBinder) for binder in source.telescope
    ):
        raise KernelError(error, "qiec-call")
    arguments = (
        tuple(
            IndexVariable(name=binder.name, sort=binder.sort)
            for binder in source.telescope
            if isinstance(binder, IndexBinder)
        )
        if signature.telescope
        else ()
    )
    try:
        substitution = instantiate_telescope(signature.telescope, arguments)
    except (KernelError, TypeError, ValueError) as exc:
        raise KernelError(error, "qiec-call") from exc
    expected = (
        *(
            _carrier_at(carriers[p.family], judgments[p.family].indices, p.arguments)
            for p in premises
        ),
        *(value.type for value in source.values),
    )
    if (
        signature.effects.tail is not None
        or tuple(substitute_type(type_, substitution) for type_ in signature.parameters)
        != expected
        or substitute_type(signature.result, substitution)
        != _carrier_at(
            carriers[conclusion.family],
            judgments[conclusion.family].indices,
            conclusion.arguments,
        )
    ):
        raise KernelError(error, "qiec-call")
    for entry in signature.effects.entries:
        registry.validate_static(entry.effect, StaticScope())
    return _RuleOperation(
        tuple(p.family for p in premises), conclusion.family, signature
    )


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
    module: str,
    name: str,
    family: FamilyDecl,
    indices: Telescope,
    output: TypeExpr,
    effects: EffectRow,
) -> ComputationSignature:
    """Give the fold an ordinary function signature over all proof indices.

    Parameters
    ----------
    module, name
        The module and the generated fold's name.
    family
        The indexed family of proofs.
    indices
        The original judgment index names as used in the result carrier.
    output, effects
        The result type over the proof indices and the reachable effect row.

    Returns
    -------
    ComputationSignature
        The signature of a fold over a single proof.
    """
    assert all(isinstance(index, IndexBinder) for index in family.indices)
    occupied = {index.name for index in indices}
    telescope = []
    for position, index in enumerate(family.indices):
        binder_name = f"i{position}"
        while binder_name in occupied:
            binder_name = f"_{binder_name}"
        occupied.add(binder_name)
        telescope.append(index.with_(name=binder_name, refinable=False))
    proof = TypeApplication(
        constructor=family.type_constructor,
        arguments=tuple(
            IndexVariable(name=index.name, sort=index.sort) for index in telescope
        ),
    )
    return ComputationSignature(
        id=ComputationId.derive(module, "computation", name),
        name=name,
        telescope=tuple(telescope),
        parameters=(proof,),
        result=_carrier_at(output, indices, proof.arguments),
        effects=effects,
    )


def _fold(
    declaration: SourceOrigin,
    family: FamilyDecl,
    signatures: Mapping[FamilyId, ComputationSignature],
    operations: Mapping[ConstructorId, _RuleOperation],
    constructors: Mapping[ConstructorId, ConstructorDecl],
    motive: CaseMotive,
) -> NamedComputation:
    """Build the indexed case analysis of a checked proof.

    Parameters
    ----------
    declaration, family
        The declaration of the interpretation and the proof family consumed.
    signatures, operations, constructors
        The folds, checked interpretations, and proof constructors in scope.
    motive
        The result carrier over the original judgment's abstract indices.

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
            motive=motive,
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
        result = _carrier_at(
            signature.result, signature.telescope, child.type.arguments
        )
        folded = Local(name=f"folded_premise{index}", type=result)
        children.append(
            (
                folded,
                Call(
                    callee=signature.id,
                    name=signature.name,
                    static_arguments=child.type.arguments,
                    arguments=(Var(local=child),),
                    result_type=result,
                    effects=signature.effects,
                    origin=origin.with_(
                        structural_path=(*path, "premise", index), role="call"
                    ),
                ),
            )
        )
    signature = operation.computation
    static = skolems if signature.telescope else ()
    body: Computation = Call(
        callee=signature.id,
        name=signature.name,
        static_arguments=static,
        arguments=tuple(
            Var(local=value)
            for value in (*(local for local, _ in children), *fields[arity:])
        ),
        result_type=_carrier_at(signature.result, signature.telescope, static),
        effects=signature.effects,
        origin=origin.with_(structural_path=(*path, "operation"), role="call"),
    )
    return CaseBranch(
        constructor=constructor.id,
        static_arguments=skolems,
        fields=fields,
        body=bind_operands(children, body),
        scope=scope,
    )
