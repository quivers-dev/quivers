"""Checked elaboration of typed finite lexical outcome queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import didactic.api as dx

from quivers.core.algebras import LOG_PROB
from quivers.dsl.ast_nodes import (
    LetExprCall,
    LetExprVar,
    ProgramDecl,
    QiecComputationDecl,
)
from quivers.qiec._lexical_proof_types import export_lexical_proofs
from quivers.qiec.canonical import tensor_type
from quivers.qiec.checking import ComputationSignature, KernelError
from quivers.qiec.contracts import ParameterId
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.deduction_outcomes import proof_entry_type, validate_outcome_key
from quivers.qiec.effects import (
    ArgumentDef,
    EffectDef,
    EffectRef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    instantiate_effect,
)
from quivers.qiec.foreign import (
    DeductionOutcomeGrouping,
    DeductionOutcomes,
    DeductionProofs,
)
from quivers.qiec.identifiers import (
    ComputationId,
    ConstructorId,
    EffectId,
    FamilyId,
    HandlerId,
    OperationId,
    SiteProvenance,
    StaticScopeId,
)
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.module import (
    NamedComputation,
    NamedEffectInstance,
    called_computations,
    installed_handlers,
)
from quivers.qiec.rules import RulePresentation
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    ConstructorValue,
    Handle,
    Local,
    Perform,
    Return,
    TensorValue,
    Value,
    Var,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    IndexLiteral,
    IndexVariable,
    TypeApplication,
    TypeExpr,
)

if TYPE_CHECKING:
    from quivers.dsl.deduction_elaboration import _Context, _System
    from quivers.dsl.qiec_lowering import _Elaborator


PROOF_CALLS = frozenset({"parse_proofs", "parse_normalized_proofs"})
OUTCOME_CALLS = frozenset({"parse_outcomes", "parse_normalized_outcomes"}) | PROOF_CALLS
OUTCOME_CODE = "qiec-deduction-outcomes"
_SPAN = "span"
_QUERY = "query"
_GROUP = "group"


@dataclass(frozen=True)
class OutcomeQuery:
    """The registered query, its ordered rule weights, and its result family."""

    instance: NamedEffectInstance
    handler: HandlerDef
    terms: TypeApplication
    weights: TypeApplication
    production_indices: tuple[int, ...]


@dataclass(frozen=True)
class OutcomeKey:
    """A named projection's checked result family and native grouping operation."""

    node: LetExprVar
    result: TypeApplication
    empty: ConstructorDecl
    entry: ConstructorDecl
    instance: NamedEffectInstance
    handler: HandlerDef
    operation: OperationDef


def outcome_call_sites(elaborator: _Elaborator) -> tuple[LetExprCall, ...]:
    """Find the direct program bindings that can introduce an outcome family."""
    return tuple(
        value
        for statement in elaborator.source.syntax.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(value := getattr(step, "value", None), LetExprCall)
        and value.func in OUTCOME_CALLS
        and value.args
        and isinstance(value.args[0], LetExprVar)
    )


def declare_outcome_key_types(
    elaborator: _Elaborator, calls: tuple[LetExprCall, ...]
) -> dict[tuple[str, str], OutcomeKey]:
    """Register each named key's result type before ordinary signatures."""
    declarations = {
        item.name: item
        for item in elaborator.source.syntax.statements
        if isinstance(item, QiecComputationDecl)
    }
    keys = {}
    for call in calls:
        if call.func in PROOF_CALLS or len(call.args) < 3:
            continue
        deduction, _, selector, *_ = call.args
        if not isinstance(deduction, LetExprVar) or not isinstance(
            selector, LetExprVar
        ):
            continue
        identity = deduction.name, selector.name
        if identity in keys or deduction.name not in elaborator._deductions:
            continue
        system = elaborator._deductions[deduction.name]
        declaration = declarations.get(selector.name)
        if declaration is None or declaration.binders:
            elaborator._fail(
                selector,
                "a deduction outcome key must name a monomorphic checked computation",
                code=OUTCOME_CODE,
            )
        key_type = elaborator._lower_type(declaration.result_type, ())
        try:
            validate_outcome_key(key_type, elaborator.registry)
        except KernelError as error:
            elaborator._fail(selector, str(error), code=OUTCOME_CODE)
        keys[identity] = _declare_key(elaborator, system, selector, key_type)
    return keys


def _declare_key(
    elaborator: _Elaborator, system: _System, node: LetExprVar, key_type: TypeExpr
) -> OutcomeKey:
    """Declare one weighted row type and reducer for a named projection."""
    module = elaborator.source.module_name
    stem = system.name(node.name)
    name = f"{stem}__Outcomes"
    family_id = FamilyId.derive(module, "family", name)
    empty_id = ConstructorId.derive(family_id, "empty")
    entry_id = ConstructorId.derive(family_id, "entry")
    family = FamilyDecl(
        id=family_id,
        name=name,
        parameters=(),
        indices=(),
        constructors=(empty_id, entry_id),
    )
    result = TypeApplication(constructor=family.type_constructor, arguments=())
    empty, entry = _register_rows(
        elaborator, family, result, key_type, f"{stem}__outcome"
    )
    ref = EffectRef(
        id=EffectId.derive(module, "effect", f"{stem}__OutcomeGrouping"),
        name=f"{stem}__OutcomeGrouping",
    )
    operation = OperationDef(
        id=OperationId.derive(ref.id, "operation", _GROUP),
        name=_GROUP,
        telescope=(),
        arguments=(
            ArgumentDef(name="rows", type=result),
            ArgumentDef(name="normalize", type=BOOL),
        ),
        result_type=result,
    )
    instance, handler = _register_query_handler(
        elaborator,
        node,
        ref,
        operation,
        f"{stem}__outcome_grouping",
        f"{stem}__group_outcomes",
        DeductionOutcomeGrouping(key_type=key_type, empty=empty_id, entry=entry_id),
    )
    return OutcomeKey(node, result, empty, entry, instance, handler, operation)


def declare_outcome_key_computations(
    elaborator: _Elaborator,
) -> tuple[NamedComputation, ...]:
    """Build ordinary QIEC projection calls followed by typed mass grouping."""
    return tuple(
        computation
        for (deduction, selector), key in elaborator._outcome_keys.items()
        for computation in _KeyProjection(
            elaborator, elaborator._deductions[deduction], selector, key
        ).computations()
    )


def check_outcome_key_bodies(
    elaborator: _Elaborator, computations: tuple[NamedComputation, ...]
) -> None:
    """Require a closed key to be deterministic through its known call graph."""
    if not elaborator._outcome_keys:
        return
    by_id = {computation.id: computation for computation in computations}
    visited: set[ComputationId] = set()
    for key in elaborator._outcome_keys.values():
        pending = [elaborator.computation_signatures[key.node.name].id]
        while pending:
            identity = pending.pop()
            if identity in visited:
                continue
            visited.add(identity)
            computation = by_id[identity]
            # An empty row can hide effects from a locally installed
            # handler. Those have no general determinism/sharing law.
            if installed_handlers(computation.body):
                elaborator._fail(
                    key.node,
                    f"deduction outcome key {key.node.name!r} must be deterministic; "
                    f"{computation.name!r} installs an effect handler",
                    code=OUTCOME_CODE,
                )
            pending.extend(called_computations(computation.body))


class _KeyProjection:
    """Synthesize the checked traversal and the closed grouping call."""

    def __init__(
        self, elaborator: _Elaborator, system: _System, selector: str, key: OutcomeKey
    ) -> None:
        self.elaborator = elaborator
        self.key = key
        self.module = elaborator.source.module_name
        self.stem = system.name(selector)
        self.path = ("deductions", system.decl.name, "outcomes", selector)
        self.selector = elaborator.computation_signatures[selector]
        if (
            self.selector.telescope
            or self.selector.parameters != (system.item,)
            or self.selector.effects != EffectRow()
        ):
            elaborator._fail(
                key.node,
                f"deduction outcome key {selector!r} must be a pure computation "
                f"taking one {system.name('Item')} argument",
                code=OUTCOME_CODE,
            )
        self.raw = elaborator._outcome_queries[system.decl.name].handler.output_type
        self.raw_empty = elaborator.constructors[system.name("outcome_end")]
        self.raw_entry = elaborator.constructors[system.name("outcome_entry")]
        self.project = self.signature("map_outcomes", (self.raw,))
        self.group = self.signature("key_outcomes", (self.raw, BOOL))

    def signature(
        self, suffix: str, parameters: tuple[TypeExpr, ...]
    ) -> ComputationSignature:
        """Register one generated pure computation's visible signature."""
        name = f"{self.stem}__{suffix}"
        signature = ComputationSignature(
            id=ComputationId.derive(self.module, "computation", name),
            name=name,
            telescope=(),
            parameters=parameters,
            result=self.key.result,
            effects=EffectRow(),
        )
        self.elaborator.registry.register_computation(signature)
        self.elaborator.computation_signatures[name] = signature
        return signature

    def call(self, signature: ComputationSignature, *arguments: Value) -> Call:
        """Emit the ordinary checked call, with the source selector's origin."""
        return Call(
            callee=signature.id,
            name=signature.name,
            static_arguments=(),
            arguments=arguments,
            result_type=signature.result,
            effects=signature.effects,
            origin=self.elaborator._origin(
                self.key.node, (*self.path, signature.name), "call"
            ),
        )

    def row(self, constructor: ConstructorDecl, *fields: Value) -> ConstructorValue:
        """Construct one expression of the declared row type."""
        return ConstructorValue(
            constructor=constructor.id,
            static_arguments=(),
            fields=fields,
            result_type=self.key.result,
        )

    def computations(self) -> tuple[NamedComputation, ...]:
        """Check the map and reducer that follow one raw chart computation."""
        key = self.key
        rows = Local(name="rows", type=self.raw)
        normalize = Local(name="normalize", type=BOOL)
        mapped = Local(name="mapped", type=key.result)
        item, weight, tail = tuple(
            Local(name=field.name, type=field.type) for field in self.raw_entry.fields
        )
        projected = Local(name="projected", type=self.selector.result)
        mapped_tail = Local(name="mapped_tail", type=key.result)
        map_body = Case(
            scrutinee=Var(local=rows),
            motive=CaseMotive(indices=(), result_type=key.result),
            branches=(
                CaseBranch(
                    constructor=self.raw_empty.id,
                    static_arguments=(),
                    fields=(),
                    body=Return(value=self.row(key.empty)),
                    scope=StaticScopeId.derive(self.module, self.path, "empty"),
                ),
                CaseBranch(
                    constructor=self.raw_entry.id,
                    static_arguments=(),
                    fields=(item, weight, tail),
                    body=Bind(
                        binder=projected,
                        first=self.call(self.selector, Var(local=item)),
                        then=Bind(
                            binder=mapped_tail,
                            first=self.call(self.project, Var(local=tail)),
                            then=Return(
                                value=self.row(
                                    key.entry,
                                    Var(local=projected),
                                    Var(local=weight),
                                    Var(local=mapped_tail),
                                )
                            ),
                        ),
                    ),
                    scope=StaticScopeId.derive(self.module, self.path, "entry"),
                ),
            ),
        )
        request = EffectRequest(
            instance=key.instance.entry.instance,
            effect=key.instance.entry.effect,
            operation=key.operation.id,
            static_arguments=(),
            arguments=(Var(local=mapped), Var(local=normalize)),
            result_type=key.result,
            origin=SiteProvenance(
                origin=self.elaborator._origin(
                    key.node, (*self.path, "group"), "effect-request"
                )
            ),
        )
        group_body = Bind(
            binder=mapped,
            first=self.call(self.project, Var(local=rows)),
            then=Handle(
                instance=request.instance,
                handler=key.handler.id,
                computation=Perform(request=request),
                static_arguments=(),
            ),
        )
        return (
            self.elaborator._checked_computation(
                self.project, (rows,), map_body, key.node, (*self.path, "map")
            ),
            self.elaborator._checked_computation(
                self.group, (rows, normalize), group_body, key.node, self.path
            ),
        )


def declare_lexical_proofs(
    elaborator: _Elaborator, system: _System, node: object
) -> RulePresentation:
    """Expose indexed constructors without making a claim about chart finiteness."""
    presented = system.presentation_cell[0]
    assert presented is not None
    if not system.entries or tuple(
        judgment.name for judgment in presented.presentation.judgments
    ) != (_SPAN,):
        elaborator._fail(
            node, "proof types require a lexical span deduction", code=OUTCOME_CODE
        )
    try:
        named = export_lexical_proofs(presented.presentation, system.decl.name)
        families, constructors = named.declarations()
        for sort in named.index_sorts:
            if sort.name in elaborator.index_sorts:
                elaborator._fail(
                    node,
                    f"generated proof sort {sort.name!r} already exists",
                    code=OUTCOME_CODE,
                )
            elaborator.index_sorts[sort.name] = sort
        for family in families:
            if family.name in elaborator.families:
                elaborator._fail(
                    node,
                    f"generated proof family {family.name!r} already exists",
                    code=OUTCOME_CODE,
                )
            elaborator.registry.register_family(family)
            elaborator.families[family.name] = family
        for constructor in constructors:
            if constructor.name in elaborator.constructors:
                elaborator._fail(
                    node,
                    f"generated proof constructor {constructor.name!r} already exists",
                    code=OUTCOME_CODE,
                )
            elaborator.registry.register_constructor(constructor)
            elaborator.constructors[constructor.name] = constructor
    except (KernelError, ValueError, dx.ValidationError) as error:
        elaborator._fail_kernel(node, error, fallback=OUTCOME_CODE)
    return named


def declare_outcomes(
    elaborator: _Elaborator, system: _System, node: object, *, proofs: bool = False
) -> OutcomeQuery:
    """Declare the result family and one operation fixed to this presentation."""
    if system.semiring.algebra != LOG_PROB.subject or not system.entries:
        elaborator._fail(
            node,
            "outcome queries require a lexical LogProb deduction",
            code=OUTCOME_CODE,
        )
    if system.depth_cell[0] is not None:
        elaborator._fail(
            node,
            "outcome queries do not yet support a category depth bound",
            code=OUTCOME_CODE,
        )
    if not system.normalized_cell[0] and any(
        rule.presented.weighted for rule in system.rules
    ):
        elaborator._fail(
            node,
            "outcome queries require fixed or normalize=lhs rule weights. Free weights keyed by a rule's bindings are outside this fragment",
            code=OUTCOME_CODE,
        )
    presented = system.presentation_cell[0]
    assert presented is not None
    presentation = presented.presentation
    module = elaborator.source.module_name
    production_indices: tuple[int, ...] = ()
    parameters: dict[int, ParameterId] = {}
    if system.normalized_cell[0]:
        production_indices = tuple(
            dict.fromkeys(rule.index for rule in presented.rules)
        )
        parameters = {
            index: ParameterId.derive(presentation.id, "production", index)
            for index in production_indices
        }
        presentation = presentation.with_(
            rules=tuple(
                rule.rule.with_(weight=parameters[rule.index])
                for rule in presented.rules
            )
        )
    family_name = system.name("Proofs" if proofs else "Outcomes")
    family_id = FamilyId.derive(module, "family", family_name)
    empty_id = ConstructorId.derive(family_id, "empty")
    entry_id = ConstructorId.derive(family_id, "entry")
    family = FamilyDecl(
        id=family_id,
        name=family_name,
        parameters=(),
        indices=(),
        constructors=(empty_id, entry_id),
    )
    result = TypeApplication(constructor=family.type_constructor, arguments=())
    try:
        chart = DeductionOutcomes(
            presentation=presentation,
            words=tuple(entry.word for entry in system.entries),
            parameters=tuple(parameters.values()),
            span=_SPAN,
            start=system.start,
            item_type=system.item,
            item_constructors=tuple(
                (name, constructor.id)
                for name, constructor in system.constructors.items()
            ),
            empty=empty_id,
            entry=entry_id,
        )
        semantics: DeductionOutcomes | DeductionProofs = chart
        if proofs:
            semantics = DeductionProofs(chart=chart)
    except (ValueError, dx.ValidationError) as error:
        elaborator._fail_kernel(node, error, fallback=OUTCOME_CODE)
    key_binders: tuple[IndexBinder, ...] = ()
    key_type = system.item
    if proofs:
        (proof_family,), _ = elaborator._lexical_proofs[system.decl.name].declarations()
        key_binders, key_type = proof_entry_type(proof_family)
    _register_rows(
        elaborator,
        family,
        result,
        key_type,
        system.name("proof" if proofs else "outcome"),
        key_binders=key_binders,
    )

    arity = 2 if presented.uses_forms else 1
    terms = tensor_type(
        system.item, (IndexLiteral(value=arity * len(system.entries), sort=NAT),)
    )
    weights = tensor_type(
        REAL, (IndexLiteral(value=len(system.entries) + len(parameters), sort=NAT),)
    )
    effect_name = system.name("ProofQuery" if proofs else "OutcomeQuery")
    effect_id = EffectId.derive(module, "effect", effect_name)
    extent = IndexVariable(name="n", sort=NAT)
    operation = OperationDef(
        id=OperationId.derive(effect_id, "operation", _QUERY),
        name=_QUERY,
        telescope=(IndexBinder(name=extent.name, sort=NAT),),
        arguments=(
            ArgumentDef(name="tokens", type=tensor_type(STRING, (extent,))),
            ArgumentDef(name="terms", type=terms),
            ArgumentDef(name="weights", type=weights),
            ArgumentDef(name="limit", type=INT)
            if proofs
            else ArgumentDef(name="key", type=STRING),
            ArgumentDef(name="normalize", type=BOOL),
        ),
        result_type=result,
    )
    instance, handler = _register_query_handler(
        elaborator,
        system.decl,
        EffectRef(id=effect_id, name=effect_name),
        operation,
        system.name("proof_query" if proofs else "outcome_query"),
        system.name("query_proofs" if proofs else "query_outcomes"),
        semantics,
    )
    return OutcomeQuery(instance, handler, terms, weights, production_indices)


def _register_query_handler(
    elaborator: _Elaborator,
    node: object,
    ref: EffectRef,
    operation: OperationDef,
    instance_name: str,
    handler_name: str,
    semantics: DeductionOutcomes | DeductionProofs | DeductionOutcomeGrouping,
) -> tuple[NamedEffectInstance, HandlerDef]:
    """Register a closed linear query with one private effect instance."""
    module = elaborator.source.module_name
    effect = EffectDef(ref=ref, telescope=(), operations=(operation,))
    elaborator.effects[ref.name] = effect
    elaborator.registry.register_effect(effect)
    row = instantiate_effect(
        effect.ref, module=module, lexical_path=("instances", instance_name)
    )
    instance = NamedEffectInstance(
        name=instance_name,
        entry=row,
        origin=elaborator._origin(
            node, ("instances", instance_name), "effect-instance"
        ),
    )
    elaborator.instances[instance_name] = instance
    handler = HandlerDef(
        id=HandlerId.derive(module, "handler", handler_name),
        name=handler_name,
        effect=effect.ref,
        clauses=(
            HandlerClauseDef(operation=operation.id, grade=ResumptionGrade.LINEAR),
        ),
        input_type=operation.result_type,
        output_type=operation.result_type,
        introduced=EffectRow(),
        foreign=semantics,
    )
    elaborator.handlers[handler_name] = handler
    return instance, handler


def _register_rows(
    elaborator: _Elaborator,
    family: FamilyDecl,
    result: TypeApplication,
    key_type: TypeExpr,
    constructor_stem: str,
    *,
    key_binders: tuple[IndexBinder, ...] = (),
) -> tuple[ConstructorDecl, ConstructorDecl]:
    """Register a nominal typed list of weighted outcome keys."""
    elaborator.families[family.name] = family
    elaborator.registry.register_family(family)
    empty_id, entry_id = family.constructors
    empty = ConstructorDecl(
        id=empty_id,
        name=f"{constructor_stem}_end",
        family=family.id,
        telescope=(),
        fields=(),
        result_indices=(),
    )
    entry = ConstructorDecl(
        id=entry_id,
        name=f"{constructor_stem}_entry",
        family=family.id,
        telescope=key_binders,
        fields=(
            FieldDef(name="key", type=key_type),
            FieldDef(name="weight", type=REAL),
            FieldDef(name="tail", type=result),
        ),
        result_indices=(),
    )
    for constructor in (empty, entry):
        elaborator.constructors[constructor.name] = constructor
        elaborator.registry.register_constructor(constructor)
    return empty, entry


def elaborate_outcomes(
    context: _Context, query: OutcomeQuery, signature: ComputationSignature
) -> NamedComputation:
    """Read weights through the grammar's checked readers and run the query."""
    system = context.system
    proofs = isinstance(query.handler.foreign, DeductionProofs)
    tokens, selector, normalize = tuple(
        Local(name=name, type=type_)
        for name, type_ in zip(
            ("tokens", "limit" if proofs else "key", "normalize"),
            context.inputs(signature),
            strict=True,
        )
    )
    terms: list[Value] = []
    for entry in system.entries:
        terms.append(context.ground(entry.category))
        if entry.form is not None:
            terms.append(context.ground(entry.form))
    readings = [
        context.entry_weight(index, entry, lambda weight: Return(value=weight))
        for index, entry in enumerate(system.entries)
    ] + [
        context.production_weight(("rule", index), lambda weight: Return(value=weight))
        for index in query.production_indices
    ]
    values = tuple(system.local("weight", system.carrier) for _ in readings)
    body: Computation = Handle(
        instance=query.instance.entry.instance,
        handler=query.handler.id,
        computation=context.request(
            query.instance,
            _QUERY,
            (context.extent,),
            (
                Var(local=tokens),
                TensorValue(items=tuple(terms), result_type=query.terms),
                TensorValue(
                    items=tuple(context.real_of(Var(local=local)) for local in values),
                    result_type=query.weights,
                ),
                Var(local=selector),
                Var(local=normalize),
            ),
            signature.result,
            "proofs" if proofs else "outcomes",
        ),
        static_arguments=(),
    )
    for local, reading in reversed(tuple(zip(values, readings, strict=True))):
        body = Bind(binder=local, first=reading, then=body)
    return context.computation(signature, (tokens, selector, normalize), body)
