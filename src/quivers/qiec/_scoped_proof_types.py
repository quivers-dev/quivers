"""Checked proof syntax for right-scoped rule presentations.

If the carrier of judgment J is B_J, the type of an environment with
J introduced on the right is the nominal product Frame_J[Gamma, B_J].
Different judgments have different frame identities even when their
carriers agree. The proof and membership families are uniform in all B_J,
so all interpretations of a presentation use the same constructor identities.

Here_J selects the newest position of Gamma.J. There_JK embeds a position
of Gamma into Gamma.K, raising its de Bruijn index by one. Induction on the
position proves that these introductions encode exactly the in-scope
occurrences of J, including distinct occurrences of the same judgment.
For a rule with a right-scoped premise at J_1,...,J_n, its proof field
has the index obtained by pushing those frames in that order. Induction on
rule applications then gives the exact ambient context of every proof field.

For J judgments there are J(J+1) membership constructors. Each written
membership type has J uniform carrier arguments and one context argument.
The other generated constructors follow the source rules, premises, and
right extensions. Open proof conversion visits every node once and emits
one membership constructor per skipped position plus one for selection
at each hole. The context spine is shared in memory. Validation and wire
encoding additionally process the type annotations, including nested
frame applications. Encoding concrete proofs does not replace a packed
parser when the requested observable admits aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from quivers.qiec.checking import KernelError, KernelRegistry, infer_value
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import TypeBinder
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.rules import (
    AxiomSlot,
    CARTESIAN,
    Derivation,
    Hole,
    OpenDerivation,
    ProofNode,
    RuleError,
    RulePresentation,
)
from quivers.qiec.terms import ConstructorValue, LiteralValue
from quivers.qiec.types import STRING, UNIT, TypeApplication, TypeExpr, TypeVariable


@dataclass(frozen=True)
class _Context:
    """A context and its innermost entry.

    Parameters
    ----------
    type
        The context's environment carrier.
    family
        The newest judgment in the context, absent at the root.
    previous
        The context before introducing this entry.
    """

    type: TypeExpr
    family: FamilyId | None = None
    previous: _Context | None = None


@dataclass(frozen=True)
class ScopedProofTypes:
    """The checked proof families of an unindexed right-binding presentation.

    Parameters
    ----------
    presentation
        The original rule declarations with their ordered premises.
    module, registry
        The generated standard QIEC declarations and their checked registry.
    empty, frames
        The types of an empty environment and a one-judgment extension.
    members, proofs
        The nominal families of context membership and proofs per judgment.
    holes, rules
        The constructors recording references and applications of source rules.
    here, there
        The constructors selecting and passing the innermost position.
        Passing is indexed by the selected judgment and then the passed one.
    """

    presentation: RulePresentation
    module: QiecModule
    registry: KernelRegistry
    empty: FamilyDecl
    frames: Mapping[FamilyId, FamilyDecl]
    members: Mapping[FamilyId, FamilyDecl]
    proofs: Mapping[FamilyId, FamilyDecl]
    holes: Mapping[FamilyId, ConstructorId]
    rules: Mapping[ConstructorId, ConstructorDecl]
    here: Mapping[FamilyId, ConstructorId]
    there: Mapping[tuple[FamilyId, FamilyId], ConstructorId]

    def carrier_types(
        self, carriers: Mapping[FamilyId, TypeExpr] | None
    ) -> tuple[TypeExpr, ...]:
        """Put semantic types in the uniform family parameter order.

        Parameters
        ----------
        carriers
            One carrier per judgment. Omitting it uses Unit in every frame.

        Returns
        -------
        tuple[TypeExpr, ...]
            The ordered family parameters for a chosen interpretation.

        Raises
        ------
        RuleError
            If a supplied mapping does not cover exactly the judgments.
        """
        if carriers is None:
            return (UNIT,) * len(self.frames)
        if set(carriers) != self.frames.keys():
            raise RuleError(
                "scoped carriers must cover exactly the presentation's judgments"
            )
        return tuple(carriers[family] for family in self.frames)

    def value(
        self,
        derivation: OpenDerivation,
        carriers: Mapping[FamilyId, TypeExpr] | None = None,
        *,
        registry: KernelRegistry | None = None,
    ) -> ConstructorValue:
        """Lower and check an open derivation at its exact ordered context.

        Parameters
        ----------
        derivation
            The proof, including its free hypotheses in outermost-first order.
        carriers
            One semantic carrier per judgment, defaulting to Unit for quotation.
        registry
            A checked extension of the generated registry when the carriers
            refer to additional declarations.

        Returns
        -------
        ConstructorValue
            A checked, serializable proof value with explicit membership witnesses.

        Raises
        ------
        RuleError
            If a rule, hypothesis, payload, or bound scope is inconsistent
            with the original presentation.
        """
        parameters = self.carrier_types(carriers)
        by_family = dict(zip(self.frames, parameters, strict=True))
        root = _Context(_at(self.empty))
        for item in derivation.context:
            if item.arguments:
                raise RuleError("a scoped proof context requires unindexed hypotheses")
            root = self._extend(root, item.family, by_family)
        pending: list[tuple[ProofNode, _Context, bool]] = [
            (derivation.body, root, False)
        ]
        values: list[ConstructorValue] = []
        while pending:
            node, context, entered = pending.pop()
            if isinstance(node, Hole):
                values.append(self._hole(node, context, parameters))
                continue
            source = self.presentation.constructors_by_id.get(node.constructor)
            if source is None:
                raise RuleError(
                    "a scoped proof names a constructor from another presentation"
                )
            if isinstance(source, AxiomSlot) or entered:
                values.append(self._applied(node, context, values, parameters))
                continue
            if len(node.premises) != len(source.premises):
                raise RuleError(
                    f"rule {source.name!r} has the wrong number of premises"
                )
            pending.append((node, context, True))
            for premise, proof in reversed(
                tuple(zip(source.premises, node.premises, strict=True))
            ):
                child = context
                for assumption in premise.after:
                    child = self._extend(child, assumption.family, by_family)
                pending.append((proof, child, False))
        (value,) = values
        try:
            infer_value(value, self.registry if registry is None else registry)
        except (KernelError, TypeError, ValueError) as error:
            raise RuleError(f"QIEC refuses the scoped proof: {error}") from error
        return value

    def _extend(
        self, context: _Context, family: FamilyId, carriers: Mapping[FamilyId, TypeExpr]
    ) -> _Context:
        """Add one assumption at the innermost end.

        Parameters
        ----------
        context, family, carriers
            The existing context, introduced judgment, and semantic carriers.

        Returns
        -------
        _Context
            The nominal context extension.

        Raises
        ------
        RuleError
            If the judgment belongs to a different presentation.
        """
        frame = self.frames.get(family)
        if frame is None:
            raise RuleError(
                "a scoped proof assumes a judgment from another presentation"
            )
        type_ = _at(frame, context.type, carriers[family])
        return _Context(type=type_, family=family, previous=context)

    def _hole(
        self, hole: Hole, context: _Context, parameters: tuple[TypeExpr, ...]
    ) -> ConstructorValue:
        """Encode a de Bruijn position using Here and There witnesses.

        Parameters
        ----------
        hole, context, parameters
            The claimed judgment at a position, enclosing context and carriers.

        Returns
        -------
        ConstructorValue
            The proof constructor carrying its typed context membership.

        Raises
        ------
        RuleError
            If the position is outside the context or has the wrong judgment.
        """
        current = context
        skipped: list[_Context] = []
        for _ in range(hole.index):
            if current.previous is None:
                raise RuleError(f"hypothesis {hole.index} lies outside its context")
            skipped.append(current)
            current = current.previous
        if current.previous is None:
            raise RuleError(f"hypothesis {hole.index} lies outside its context")
        family = hole.item.family
        if current.family != family or hole.item.arguments:
            raise RuleError("a scoped hypothesis does not match its context position")
        member = self.members[family]
        witness = ConstructorValue(
            constructor=self.here[family],
            static_arguments=(*parameters, current.previous.type),
            fields=(),
            result_type=_at(member, *parameters, current.type),
        )
        for outside in reversed(skipped):
            assert outside.family is not None and outside.previous is not None
            witness = ConstructorValue(
                constructor=self.there[family, outside.family],
                static_arguments=(*parameters, outside.previous.type),
                fields=(witness,),
                result_type=_at(member, *parameters, outside.type),
            )
        return ConstructorValue(
            constructor=self.holes[family],
            static_arguments=(*parameters, context.type),
            fields=(witness,),
            result_type=_at(self.proofs[family], *parameters, context.type),
        )

    def _applied(
        self,
        node: Derivation,
        context: _Context,
        stack: list[ConstructorValue],
        parameters: tuple[TypeExpr, ...],
    ) -> ConstructorValue:
        """Finish a constructor after converting its ordered premises.

        Parameters
        ----------
        node, context
            The original application and its ambient context.
        stack, parameters
            The encoded values available to consume and the semantic carriers.

        Returns
        -------
        ConstructorValue
            A nominal proof value preserving payloads and axiom identities.

        Raises
        ------
        RuleError
            If the node has an invalid judgment, payload or occurrence record.
        """
        source = self.presentation.constructors_by_id[node.constructor]
        constructor = self.rules[node.constructor]
        family = self.proofs.get(node.item.family)
        if family is None or node.item.arguments:
            raise RuleError("a scoped proof node claims an unknown or indexed judgment")
        if len(node.values) != len(source.values):
            raise RuleError(
                f"scoped constructor {source.name!r} has the wrong value arity"
            )
        if node.binding:
            raise RuleError("an unindexed scoped proof cannot carry static bindings")
        arguments = tuple(
            LiteralValue(value=value, type=field.type)
            for value, field in zip(node.values, source.values, strict=True)
        )
        if isinstance(source, AxiomSlot):
            if node.occurrence is None or node.premises:
                raise RuleError(
                    "a scoped input axiom needs an occurrence identity and no premises"
                )
            arguments = (
                LiteralValue(value=str(node.occurrence), type=STRING),
                *arguments,
            )
        else:
            if node.occurrence is not None:
                raise RuleError(
                    "a rule application does not name an input axiom occurrence"
                )
            size = len(source.premises)
            if size:
                arguments = (*stack[-size:], *arguments)
                del stack[-size:]
        return ConstructorValue(
            constructor=constructor.id,
            static_arguments=(*parameters, context.type),
            fields=arguments,
            result_type=_at(family, *parameters, context.type),
        )


def _at(family: FamilyDecl, *types: TypeExpr) -> TypeApplication:
    """Apply one generated family at its type parameters and indices.

    Parameters
    ----------
    family, types
        The nominal family and its type arguments in declaration order.

    Returns
    -------
    TypeApplication
        The family applied to these arguments.
    """
    return TypeApplication(constructor=family.type_constructor, arguments=types)


def _environments(
    presentation: RulePresentation, prefix: str
) -> tuple[FamilyDecl, dict[FamilyId, FamilyDecl], tuple[ConstructorDecl, ...]]:
    """Declare the empty context and its nominal one-step extensions.

    Parameters
    ----------
    presentation, prefix
        The source identities and generated name prefix.

    Returns
    -------
    tuple
        The empty family, the frame families, and their constructors.
    """
    empty = FamilyDecl(
        id=FamilyId.derive(presentation.id, "scoped-empty"),
        name=f"{prefix}Empty",
        parameters=(),
        indices=(),
        constructors=(ConstructorId.derive(presentation.id, "scoped-empty"),),
    )
    declared = [
        ConstructorDecl(
            id=empty.constructors[0],
            family=empty.id,
            name=empty.name,
            telescope=(),
            fields=(),
            result_indices=(),
        )
    ]
    frames: dict[FamilyId, FamilyDecl] = {}
    for item in presentation.judgments:
        previous = TypeBinder(name="previous_context")
        carrier = TypeBinder(name="hypothesis_type")
        frame = FamilyDecl(
            id=FamilyId.derive(item.id, "scoped-frame"),
            name=f"{prefix}Frame_{item.name}",
            parameters=(previous, carrier),
            indices=(),
            constructors=(ConstructorId.derive(item.id, "scoped-frame"),),
        )
        frames[item.id] = frame
        declared.append(
            ConstructorDecl(
                id=frame.constructors[0],
                family=frame.id,
                name=frame.name,
                telescope=(),
                fields=(
                    FieldDef(name="previous", type=TypeVariable(name=previous.name)),
                    FieldDef(name="hypothesis", type=TypeVariable(name=carrier.name)),
                ),
                result_indices=(),
            )
        )
    return empty, frames, tuple(declared)


def scoped_proof_types(presentation: RulePresentation) -> ScopedProofTypes:
    """Generate checked proof syntax with explicit ordered context types.

    Parameters
    ----------
    presentation
        A Cartesian presentation with unindexed judgments, right-scoped
        premises, and no guards or equations.

    Returns
    -------
    ScopedProofTypes
        Ordinary indexed QIEC declarations and a typed conversion from proofs.

    Raises
    ------
    RuleError
        If the source uses a feature outside this fragment.
    """
    if presentation.structure != CARTESIAN:
        raise RuleError("scoped proof types require Cartesian structural permissions")
    if presentation.conditions or presentation.equations:
        raise RuleError(
            "scoped proof types require unconditional syntax without equations"
        )
    sources = presentation.constructors_by_id
    if any(item.indices for item in presentation.judgments) or any(
        rule.telescope for rule in sources.values()
    ):
        raise RuleError(
            "scoped proof types require unindexed judgments and rule declarations"
        )
    if any(premise.before for rule in presentation.rules for premise in rule.premises):
        raise RuleError("scoped proof types admit only hypotheses added on the right")

    judgments = presentation.judgments_by_id
    prefix = f"{presentation.name}__Scoped"
    empty, frames, environments = _environments(presentation, prefix)
    parameters = tuple(
        TypeBinder(name=f"semantic{index}") for index in range(len(judgments))
    )
    carriers = tuple(TypeVariable(name=binder.name) for binder in parameters)
    context = TypeVariable(name="ambient")
    pushed = {
        family: _at(frames[family], context, carrier)
        for family, carrier in zip(judgments, carriers, strict=True)
    }
    carrier_by_family = dict(zip(judgments, carriers, strict=True))
    ctx_binder = TypeBinder(name=context.name)
    ctx_index = TypeBinder(name="indexed_context", refinable=True)
    here = {family: ConstructorId.derive(family, "scoped-here") for family in judgments}
    there = {
        (target, passed): ConstructorId.derive(target, passed, "scoped-there")
        for target in judgments
        for passed in judgments
    }
    holes = {
        family: ConstructorId.derive(family, "scoped-hypothesis")
        for family in judgments
    }
    source_ids = {
        identity: ConstructorId.derive(identity, "scoped") for identity in sources
    }
    members = {
        family: FamilyDecl(
            id=FamilyId.derive(family, "scoped-membership"),
            name=f"{prefix}Member_{item.name}",
            parameters=parameters,
            indices=(ctx_index,),
            constructors=(here[family], *(there[family, other] for other in judgments)),
        )
        for family, item in judgments.items()
    }
    proof_ids: dict[FamilyId, list[ConstructorId]] = {
        family: [hole] for family, hole in holes.items()
    }
    for identity, source in sources.items():
        conclusion = (
            source.judgment if isinstance(source, AxiomSlot) else source.conclusion
        )
        proof_ids[conclusion.family].append(source_ids[identity])
    proofs = {
        family: FamilyDecl(
            id=FamilyId.derive(family, "scoped-proof"),
            name=f"{prefix}Proof_{item.name}",
            parameters=parameters,
            indices=(ctx_index,),
            constructors=tuple(proof_ids[family]),
        )
        for family, item in judgments.items()
    }
    declared = list(environments)
    for family, item in judgments.items():
        member = members[family]
        declared.extend(
            (
                ConstructorDecl(
                    id=here[family],
                    family=member.id,
                    name=f"{prefix}Here_{item.name}",
                    telescope=(ctx_binder,),
                    fields=(),
                    result_indices=(pushed[family],),
                ),
                ConstructorDecl(
                    id=holes[family],
                    family=proofs[family].id,
                    name=f"{prefix}Hypothesis_{item.name}",
                    telescope=(ctx_binder,),
                    fields=(
                        FieldDef(
                            name="reference", type=_at(member, *carriers, context)
                        ),
                    ),
                    result_indices=(context,),
                ),
            )
        )
        for other, passed in judgments.items():
            declared.append(
                ConstructorDecl(
                    id=there[family, other],
                    family=member.id,
                    name=f"{prefix}There_{item.name}_{passed.name}",
                    telescope=(ctx_binder,),
                    fields=(
                        FieldDef(name="previous", type=_at(member, *carriers, context)),
                    ),
                    result_indices=(pushed[other],),
                )
            )
    rules: dict[ConstructorId, ConstructorDecl] = {}
    for identity, source in sources.items():
        fields: list[FieldDef] = []
        if isinstance(source, AxiomSlot):
            family = source.judgment.family
            fields.append(FieldDef(name="occurrence", type=STRING))
        else:
            family = source.conclusion.family
            for position, premise in enumerate(source.premises):
                child: TypeExpr = context
                for hypothesis in premise.after:
                    child = _at(
                        frames[hypothesis.family],
                        child,
                        carrier_by_family[hypothesis.family],
                    )
                fields.append(
                    FieldDef(
                        name=f"premise{position}",
                        type=_at(proofs[premise.judgment.family], *carriers, child),
                    )
                )
        fields.extend(
            field.with_(name=f"payload{position}")
            for position, field in enumerate(source.values)
        )
        rules[identity] = ConstructorDecl(
            id=source_ids[identity],
            family=proofs[family].id,
            name=f"{prefix}Rule_{source.name}",
            telescope=(ctx_binder,),
            fields=tuple(fields),
            result_indices=(context,),
        )
    module = presentation.to_module().with_(
        families=(empty, *frames.values(), *members.values(), *proofs.values()),
        constructors=(*declared, *rules.values()),
    )
    return ScopedProofTypes(
        presentation=presentation,
        module=module,
        registry=validate_module(module),
        empty=empty,
        frames=MappingProxyType(frames),
        members=MappingProxyType(members),
        proofs=MappingProxyType(proofs),
        holes=MappingProxyType(holes),
        rules=MappingProxyType(rules),
        here=MappingProxyType(here),
        there=MappingProxyType(there),
    )
