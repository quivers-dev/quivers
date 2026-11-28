"""Checked environment operations for interpreting right-scoped proofs."""

from __future__ import annotations

from collections.abc import Mapping

from quivers.qiec._scoped_proof_types import ScopedProofTypes, _at
from quivers.qiec.coverage import constructor_skolems, refine_branch
from quivers.qiec.effects import ComputationType, EffectRow
from quivers.qiec.identifiers import (
    ComputationId,
    FamilyId,
    SourceOrigin,
    StaticScopeId,
)
from quivers.qiec.kinds import TypeBinder
from quivers.qiec.module import NamedComputation
from quivers.qiec.terms import (
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    Local,
    Return,
    TransportValue,
    Var,
)
from quivers.qiec.types import TypeExpr, TypeVariable


def scoped_lookup_computations(
    syntax: ScopedProofTypes, carriers: Mapping[FamilyId, TypeExpr]
) -> dict[FamilyId, NamedComputation]:
    """Generate checked-code definitions for typed environment lookup.

    Case on the membership proof gives the whole-type equality between
    the input environment and a frame. Transport permits matching that
    frame, whose outer environment has exactly the context type of the
    membership tail. Here returns the top field. There decreases the
    witness length and consults only the outer environment. Induction
    gives the input value at the witness's exact de Bruijn position.
    Each lookup is pure and performs one frame elimination per witness
    constructor.

    Parameters
    ----------
    syntax
        The checked frame and membership declarations.
    carriers
        The closed semantic type of each judgment in this interpretation.

    Returns
    -------
    dict
        Recursively defined QIEC computations, one per judgment, to
        register with the syntax's module and any extra carrier declarations.
    """
    semantic = syntax.carrier_types(carriers)
    ambient = TypeVariable(name="lookup_context")
    domains = (TypeBinder(name=ambient.name),)
    row = EffectRow()
    names = {
        source: f"{family.name}__lookup" for source, family in syntax.members.items()
    }
    lookups: dict[FamilyId, NamedComputation] = {}
    for selected, member in syntax.members.items():
        answer = carriers[selected]
        identity = ComputationId.derive(syntax.module.module, selected, "scoped-lookup")
        name = names[selected]
        origin = SourceOrigin(
            module=syntax.module.module,
            source_protocol=syntax.module.source_protocol,
            structural_path=("scoped_lookup", str(selected)),
            role="computation",
        )
        reference = Local(name="reference", type=_at(member, *semantic, ambient))
        environment = Local(name="environment", type=ambient)
        branches: list[CaseBranch] = []
        cases = (
            (selected, syntax.here[selected], True),
            *(
                (passed, syntax.there[selected, passed], False)
                for passed in syntax.frames
            ),
        )
        for passed, constructor, at_top in cases:
            declaration = syntax.registry.constructor(constructor)
            scope = StaticScopeId.derive(identity, constructor, "membership")
            static_arguments = constructor_skolems(declaration, scope)
            (previous_type,) = static_arguments
            assert isinstance(previous_type, TypeVariable)
            frame = syntax.frames[passed]
            frame_type = _at(frame, previous_type, carriers[passed])
            (cast,) = refine_branch(
                member, declaration, reference.type, static_arguments
            ).givens
            previous = Local(name="previous", type=previous_type)
            hypothesis = Local(name="hypothesis", type=carriers[passed])
            if at_top:
                fields = ()
                body = Return(value=Var(local=hypothesis))
            else:
                tail = Local(name="tail", type=_at(member, *semantic, previous_type))
                fields = (tail,)
                body = Call(
                    callee=identity,
                    name=name,
                    static_arguments=(previous_type,),
                    arguments=(Var(local=tail), Var(local=previous)),
                    result_type=answer,
                    effects=row,
                    origin=origin,
                )
            frame_constructor = frame.constructors[0]
            eliminate_frame = Case(
                scrutinee=TransportValue(
                    evidence=cast, value=Var(local=environment), target_type=frame_type
                ),
                motive=CaseMotive(indices=(), result_type=answer),
                branches=(
                    CaseBranch(
                        constructor=frame_constructor,
                        static_arguments=(),
                        fields=(previous, hypothesis),
                        body=body,
                        scope=StaticScopeId.derive(scope, frame_constructor, "frame"),
                    ),
                ),
            )
            branches.append(
                CaseBranch(
                    constructor=constructor,
                    static_arguments=static_arguments,
                    fields=fields,
                    body=eliminate_frame,
                    scope=scope,
                )
            )
        lookups[selected] = NamedComputation(
            id=identity,
            name=name,
            telescope=domains,
            parameters=(reference, environment),
            type=ComputationType(effects=row, result=answer),
            body=Case(
                scrutinee=Var(local=reference),
                motive=CaseMotive(indices=member.indices, result_type=answer),
                branches=tuple(branches),
            ),
            origin=origin,
        )
    return lookups
