"""Compile a right-scoped proof fold using ordinary first-order QIEC.

The semantic argument of a scoped premise is a product of the premise proof
and its captured ambient environment. Constructing it executes neither
the proof nor its effects. Applying it extends that environment with the
given semantic hypotheses in source order and executes the child fold once.
An action can ignore it, invoke it once, or invoke it repeatedly where the
presentation grants Cartesian permissions. Each call runs its own child
effects through the ordinary evaluator. Eager premises run left to right.

For finite proof trees and terminating actions that use their supplied
arguments finitely, termination follows by induction on proof height.
Each invocation of a supplied argument descends into a strict child.
This does not restrict other general recursion that an action may declare.
These definitions establish the scoped-application equations above.
An action that inspects the suspended proof itself is a proof observer.
To interpret suspended arguments extensionally, an action must consume
them through application. No equation about an arbitrary action follows
from type checking. In particular, replacing an effectful proof by a
previously computed value can change the number and order of requests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from quivers.qiec._scoped_proof_interpretation import scoped_lookup_computations
from quivers.qiec._scoped_proof_types import ScopedProofTypes, _at
from quivers.qiec.checking import ComputationSignature
from quivers.qiec.coverage import constructor_skolems, refine_branch
from quivers.qiec.effects import ComputationType, EffectRow
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    FamilyId,
    SourceOrigin,
    StaticScopeId,
)
from quivers.qiec.kinds import TypeBinder
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.rules import AxiomSlot, Premise, Rule, RuleError
from quivers.qiec.substitution import instantiate_telescope, substitute_type
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    ConstructorValue,
    Local,
    Projection,
    TransportValue,
    TupleValue,
    Value,
    Var,
)
from quivers.qiec.types import STRING, TypeExpr, TypeVariable, product_type


@dataclass(frozen=True)
class ScopedProofFold:
    """The closed carrier and effect contract of a scoped interpretation.

    Parameters
    ----------
    syntax
        The shared typed proof and context declarations.
    carriers
        The closed type of meanings for each judgment.
    rows
        The declared closed effect row of the fold for each judgment.
        A suspended premise uses the row for the judgment it proves.
    name
        The interpretation's stable name within this presentation.
    """

    syntax: ScopedProofTypes
    carriers: Mapping[FamilyId, TypeExpr]
    rows: Mapping[FamilyId, EffectRow]
    name: str

    def operation_parameters(
        self, identity: ConstructorId, ambient: TypeExpr
    ) -> tuple[TypeExpr, ...]:
        """Describe a source rule's operation parameters at this context.

        Parameters
        ----------
        identity, ambient
            The rule or axiom slot identity and its abstract environment.

        Returns
        -------
        tuple
            In order, eager meanings and suspended argument products, then
            the value payloads. An input axiom first receives its occurrence ID.
        """
        source = self.syntax.presentation.constructors_by_id[identity]
        payload = tuple(field.type for field in source.values)
        if isinstance(source, AxiomSlot):
            return (STRING, *payload)
        return (
            *(
                _closure(self, premise, ambient)
                if premise.after
                else self.carriers[premise.judgment.family]
                for premise in source.premises
            ),
            *payload,
        )

    def applications(self) -> dict[tuple[ConstructorId, int], NamedComputation]:
        """Declare the ordinary computation that applies each scoped premise.

        Returns
        -------
        dict
            The application declarations keyed by source rule and position.
            Each takes its suspended argument and the bound meanings in order.
        """
        return {
            (rule.id, position): _application(self, rule, position, premise)
            for rule in self.syntax.presentation.rules
            for position, premise in enumerate(rule.premises)
            if premise.after
        }

    def assemble(
        self,
        operations: Mapping[ConstructorId, NamedComputation],
        *,
        module: QiecModule | None = None,
    ) -> tuple[QiecModule, dict[FamilyId, NamedComputation]]:
        """Assemble and kernel-check a complete scoped interpretation.

        Parameters
        ----------
        operations
            One ordinary named computation per source rule or axiom, generic
            in its ambient context and receiving its typed operation parameters.
            Operations may call the generated scoped application computations.
        module
            The syntax module with any added effects and carrier declarations.
            Omitting it uses the original syntax module.

        Returns
        -------
        tuple
            The checked module and the fold computation of each judgment.

        Raises
        ------
        RuleError
            If the operations do not cover exactly the source constructors.
        KernelError
            If an operation, fold result, declared row, or call fails checking.
        """
        signatures = {
            source: _operation_signature(operation)
            for source, operation in operations.items()
        }
        generated, folds = _scoped_definitions(self, signatures)
        base = self.syntax.module if module is None else module
        checked = base.with_(
            computations=(
                *base.computations,
                *operations.values(),
                *generated,
            )
        )
        validate_module(checked)
        return checked, folds


def _operation_signature(operation: NamedComputation) -> ComputationSignature:
    """Retain the nominal signature needed to generate a fold call.

    Parameters
    ----------
    operation
        The complete checked operation declaration.

    Returns
    -------
    ComputationSignature
        Its ordinary static and value parameters, result, and row.
    """
    return ComputationSignature(
        id=operation.id,
        name=operation.name,
        telescope=operation.telescope,
        parameters=tuple(local.type for local in operation.parameters),
        result=operation.type.result,
        effects=operation.type.effects,
    )


def _scoped_entry(
    fold: ScopedProofFold, root: FamilyId, origin: SourceOrigin
) -> NamedComputation:
    """Export one judgment's fold with an empty initial environment.

    Parameters
    ----------
    fold, root, origin
        The scoped interpretation, root judgment, and declaration site.

    Returns
    -------
    NamedComputation
        The closed named call taking a proof at its exact empty context.
    """
    syntax = fold.syntax
    empty_type = _at(syntax.empty)
    proof_type = _at(
        syntax.proofs[root], *syntax.carrier_types(fold.carriers), empty_type
    )
    proof = Local(name="proof", type=proof_type)
    empty = ConstructorValue(
        constructor=syntax.empty.constructors[0],
        static_arguments=(),
        fields=(),
        result_type=empty_type,
    )
    call = _fold_call(fold, root, empty_type, Var(local=proof), empty, origin)
    return NamedComputation(
        id=ComputationId.derive(origin.module, fold.name, "scoped-interpretation"),
        name=fold.name,
        telescope=(),
        parameters=(proof,),
        type=ComputationType(effects=call.effects, result=call.result_type),
        origin=origin,
        body=call,
    )


def _scoped_definitions(
    fold: ScopedProofFold, operations: Mapping[ConstructorId, ComputationSignature]
) -> tuple[tuple[NamedComputation, ...], dict[FamilyId, NamedComputation]]:
    """Generate scoped definitions once action signatures are available.

    Parameters
    ----------
    fold, operations
        The closed interpretation contract and a signature per source constructor.

    Returns
    -------
    tuple
        The generated computations and the selected fold by judgment. Ordinary
        module checking validates these definitions together with the actions.

    Raises
    ------
    RuleError
        If the operations do not cover exactly the source constructors.
    """
    syntax = fold.syntax
    if operations.keys() != syntax.rules.keys():
        raise RuleError("scoped operations must cover exactly the source constructors")
    lookups = scoped_lookup_computations(syntax, fold.carriers, name=fold.name)
    source_ids = {declared.id: source for source, declared in syntax.rules.items()}
    folds = {
        family: _fold(fold, family, lookups[family], operations, source_ids)
        for family in syntax.proofs
    }
    return (*lookups.values(), *fold.applications().values(), *folds.values()), folds


def scoped_proof_fold(
    syntax: ScopedProofTypes,
    carriers: Mapping[FamilyId, TypeExpr],
    rows: Mapping[FamilyId, EffectRow],
    *,
    name: str,
) -> ScopedProofFold:
    """Fix the closed result and effect contract for a scoped interpretation.

    Parameters
    ----------
    syntax
        The standard right-scoped proof declarations.
    carriers, rows
        A carrier and a closed effect row for each judgment. The kernel
        checks these rows against all fold branches and their calls.
    name
        The stable source name of the interpretation.

    Returns
    -------
    ScopedProofFold
        The contract needed to author typed rule and premise applications.

    Raises
    ------
    RuleError
        If a carrier or closed row is missing, or the name is empty.
    """
    syntax.carrier_types(carriers)
    if set(rows) != syntax.proofs.keys() or any(
        row.tail is not None for row in rows.values()
    ):
        raise RuleError(
            "scoped fold rows must be closed and cover exactly the judgments"
        )
    if not name:
        raise RuleError("a scoped interpretation needs a stable name")
    return ScopedProofFold(
        syntax, MappingProxyType(dict(carriers)), MappingProxyType(dict(rows)), name
    )


def _closure(fold: ScopedProofFold, premise: Premise, ambient: TypeExpr) -> TypeExpr:
    """Give one suspended premise its first-order product type.

    Parameters
    ----------
    fold, premise, ambient
        The interpretation, the scoped premise and its captured context.

    Returns
    -------
    TypeExpr
        A product carrying the proof under its extended context and the
        original environment.
    """
    syntax = fold.syntax
    extended = ambient
    for assumption in premise.after:
        passed = assumption.family
        extended = _at(syntax.frames[passed], extended, fold.carriers[passed])
    proof = _at(
        syntax.proofs[premise.judgment.family],
        *syntax.carrier_types(fold.carriers),
        extended,
    )
    return product_type(proof, ambient)


def _origin(fold: ScopedProofFold, *path: str) -> SourceOrigin:
    """Locate generated computations and call sites by structural path.

    Parameters
    ----------
    fold, path
        The interpretation and its local structural path.

    Returns
    -------
    SourceOrigin
        Stable source identity within the presentation.
    """
    base = fold.syntax.module
    return SourceOrigin(
        module=base.module,
        source_protocol=base.source_protocol,
        structural_path=("scoped_interpretation", fold.name, *path),
        role="computation",
    )


def _fold_call(
    fold: ScopedProofFold,
    family: FamilyId,
    ambient: TypeExpr,
    proof: Value,
    environment: Value,
    origin: SourceOrigin,
) -> Call:
    """Call the family fold at an exactly matched proof/environment type.

    Parameters
    ----------
    fold, family, ambient
        The interpretation, desired judgment, and scope of the proof.
    proof, environment, origin
        The arguments recorded at this call site.

    Returns
    -------
    Call
        A call of the named fold with the declared result and effect row.
    """
    judgment = fold.syntax.presentation.judgments_by_id[family]
    return Call(
        callee=ComputationId.derive(
            fold.syntax.module.module, fold.name, family, "scoped-fold"
        ),
        name=f"{fold.name}__fold_{judgment.name}",
        static_arguments=(ambient,),
        arguments=(proof, environment),
        result_type=fold.carriers[family],
        effects=fold.rows[family],
        origin=origin,
    )


def _application(
    fold: ScopedProofFold, rule: Rule, position: int, premise: Premise
) -> NamedComputation:
    """Generate one suspended premise's explicit application computation.

    Parameters
    ----------
    fold, rule, position, premise
        The interpretation and the source of the argument being applied.

    Returns
    -------
    NamedComputation
        Its environment extension followed by one checked fold of the proof.
    """
    syntax = fold.syntax
    family = premise.judgment.family
    ambient = TypeVariable(name="application_context")
    argument = Local(name="argument", type=_closure(fold, premise, ambient))
    environment: Value = Projection(
        value=Var(local=argument), position=1, result_type=ambient
    )
    extended: TypeExpr = ambient
    values: list[Local] = []
    for number, assumption in enumerate(premise.after):
        passed = assumption.family
        semantic = fold.carriers[passed]
        frame = syntax.frames[passed]
        local = Local(name=f"given{number}", type=semantic)
        arguments = (extended, semantic)
        extended = _at(frame, *arguments)
        environment = ConstructorValue(
            constructor=frame.constructors[0],
            static_arguments=arguments,
            fields=(environment, Var(local=local)),
            result_type=extended,
        )
        values.append(local)
    proof_type = _at(
        syntax.proofs[family], *syntax.carrier_types(fold.carriers), extended
    )
    proof = Projection(value=Var(local=argument), position=0, result_type=proof_type)
    origin = _origin(fold, "rule", rule.name, "premise", str(position))
    call = _fold_call(fold, family, extended, proof, environment, origin)
    return NamedComputation(
        id=ComputationId.derive(
            syntax.module.module, fold.name, rule.id, position, "scoped-apply"
        ),
        name=f"{fold.name}__apply_{rule.name}_{position}",
        telescope=(TypeBinder(name=ambient.name),),
        parameters=(argument, *values),
        type=ComputationType(effects=call.effects, result=call.result_type),
        body=call,
        origin=origin,
    )


def _fold(
    fold: ScopedProofFold,
    family: FamilyId,
    lookup: NamedComputation,
    operations: Mapping[ConstructorId, ComputationSignature],
    source_ids: Mapping[ConstructorId, ConstructorId],
) -> NamedComputation:
    """Compile case analysis over every constructor of a proof family.

    Parameters
    ----------
    fold, family, lookup, operations
        The interpretation, selected judgment, variable lookup and actions.
    source_ids
        The index from scoped constructors to source rule identities.

    Returns
    -------
    NamedComputation
        The recursive case with pure hypotheses and typed rule applications.
    """
    syntax = fold.syntax
    semantic = syntax.carrier_types(fold.carriers)
    ambient = TypeVariable(name="fold_context")
    proof_family = syntax.proofs[family]
    proof_type = _at(proof_family, *semantic, ambient)
    proof = Local(name="proof", type=proof_type)
    environment = Local(name="environment", type=ambient)
    origin = _origin(fold, "judgment", proof_family.name)
    nominal = _fold_call(
        fold, family, ambient, Var(local=proof), Var(local=environment), origin
    )
    branches = []
    for identity in proof_family.constructors:
        constructor = syntax.registry.constructor(identity)
        scope = StaticScopeId.derive(nominal.callee, identity, "scoped-proof")
        skolems = constructor_skolems(constructor, scope)
        (branch_context,) = skolems
        assert isinstance(branch_context, TypeVariable)
        substitution = instantiate_telescope(
            (*proof_family.parameters, *constructor.telescope),
            (*semantic, branch_context),
        )
        fields = tuple(
            Local(name=field.name, type=substitute_type(field.type, substitution))
            for field in constructor.fields
        )
        (given,) = refine_branch(proof_family, constructor, proof_type, skolems).givens
        captured = TransportValue(
            evidence=given, value=Var(local=environment), target_type=branch_context
        )
        location = _origin(
            fold, "judgment", proof_family.name, "constructor", constructor.name
        )
        if identity == syntax.holes[family]:
            (reference,) = fields
            body: Computation = Call(
                callee=lookup.id,
                name=lookup.name,
                static_arguments=(branch_context,),
                arguments=(Var(local=reference), captured),
                result_type=fold.carriers[family],
                effects=EffectRow(),
                origin=location,
            )
        else:
            original = source_ids[identity]
            body = _action(
                fold,
                original,
                operations[original],
                branch_context,
                captured,
                fields,
                location,
            )
        branches.append(
            CaseBranch(
                constructor=identity,
                static_arguments=skolems,
                fields=fields,
                body=body,
                scope=scope,
            )
        )
    return NamedComputation(
        id=nominal.callee,
        name=nominal.name,
        telescope=(TypeBinder(name=ambient.name),),
        parameters=(proof, environment),
        type=ComputationType(effects=nominal.effects, result=nominal.result_type),
        body=Case(
            scrutinee=Var(local=proof),
            motive=CaseMotive(
                indices=proof_family.indices, result_type=nominal.result_type
            ),
            branches=tuple(branches),
        ),
        origin=origin,
    )


def _action(
    fold: ScopedProofFold,
    identity: ConstructorId,
    operation: ComputationSignature,
    ambient: TypeExpr,
    captured: Value,
    fields: tuple[Local, ...],
    origin: SourceOrigin,
) -> Computation:
    """Sequence eager children and pass scoped children without running them.

    Parameters
    ----------
    fold, identity, operation
        The interpretation, source rule identity and its typed action.
    ambient, captured
        The selected rule's abstract context and transported environment.
    fields, origin
        The rule's ordered fields and stable source site.

    Returns
    -------
    Computation
        A left-to-right sequence of eager calls followed by the action.
    """
    source = fold.syntax.presentation.constructors_by_id[identity]
    premises = source.premises if isinstance(source, Rule) else ()
    values: list[Value] = []
    bindings: list[tuple[Local, Call]] = []
    for position, premise in enumerate(premises):
        child = Var(local=fields[position])
        semantic = premise.judgment.family
        if premise.after:
            values.append(
                TupleValue(
                    items=(child, captured),
                    result_type=_closure(fold, premise, ambient),
                )
            )
        else:
            location = origin.with_(
                structural_path=(*origin.structural_path, "premise", str(position))
            )
            call = _fold_call(fold, semantic, ambient, child, captured, location)
            local = Local(name=f"meaning{position}", type=call.result_type)
            bindings.append((local, call))
            values.append(Var(local=local))
    values.extend(Var(local=field) for field in fields[len(premises) :])
    body: Computation = Call(
        callee=operation.id,
        name=operation.name,
        static_arguments=(ambient,),
        arguments=tuple(values),
        result_type=operation.result,
        effects=operation.effects,
        origin=origin,
    )
    for local, call in reversed(bindings):
        body = Bind(binder=local, first=call, then=body)
    return body
