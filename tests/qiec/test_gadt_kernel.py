import pytest

from quivers.qiec import (
    BOOL,
    INT,
    NAT,
    SHAPE,
    STRING,
    TYPE,
    UNIT,
    BranchPattern,
    Case,
    CaseBranch,
    CaseMotive,
    CheckContext,
    ConstructorDecl,
    ConstructorId,
    ConstructorValue,
    CoverageStatus,
    EffectDef,
    EffectBinder,
    EffectId,
    EffectRef,
    EffectRequest,
    EffectVariable,
    EvidenceValue,
    FamilyDecl,
    FamilyId,
    FieldDef,
    FunctionType,
    EqualityType,
    IndexBinder,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    KernelError,
    KernelRegistry,
    LiteralValue,
    Local,
    OperationDef,
    OperationId,
    Perform,
    Reachability,
    Return,
    SiteProvenance,
    SourceOrigin,
    StaticScopeId,
    TypeApplication,
    TypeBinder,
    TypeConstructorRef,
    TypeId,
    TypeVariable,
    UserIndexSort,
    Var,
    check_indexed_coverage,
    constructor_skolems,
    infer_computation,
    infer_value,
    instantiate_effect,
    refine_branch,
)


def _expr_family() -> tuple[
    FamilyDecl,
    ConstructorDecl,
    ConstructorDecl,
    ConstructorDecl,
]:
    family_id = FamilyId.derive("tests", "Expr")
    bool_id = ConstructorId.derive(str(family_id), "Bool")
    int_id = ConstructorId.derive(str(family_id), "Int")
    any_id = ConstructorId.derive(str(family_id), "Any")
    family = FamilyDecl(
        id=family_id,
        name="Expr",
        parameters=(),
        indices=(TypeBinder(name="a", kind=TYPE, refinable=True),),
        constructors=(bool_id, int_id, any_id),
    )
    bool_constructor = ConstructorDecl(
        id=bool_id,
        family=family_id,
        name="Bool",
        telescope=(),
        fields=(),
        result_indices=(BOOL,),
    )
    int_constructor = ConstructorDecl(
        id=int_id,
        family=family_id,
        name="Int",
        telescope=(),
        fields=(),
        result_indices=(INT,),
    )
    any_constructor = ConstructorDecl(
        id=any_id,
        family=family_id,
        name="Any",
        telescope=(TypeBinder(name="x"),),
        fields=(),
        result_indices=(TypeVariable(name="x"),),
    )
    return family, bool_constructor, int_constructor, any_constructor


def _registry_with_expr() -> tuple[
    KernelRegistry,
    FamilyDecl,
    ConstructorDecl,
    ConstructorDecl,
    ConstructorDecl,
]:
    family, bool_constructor, int_constructor, any_constructor = _expr_family()
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(bool_constructor)
    registry.register_constructor(int_constructor)
    registry.register_constructor(any_constructor)
    return registry, family, bool_constructor, int_constructor, any_constructor


def _expr(family: FamilyDecl, index: object) -> TypeApplication:
    return TypeApplication(constructor=family.type_constructor, arguments=(index,))  # type: ignore[arg-type]


def _origin() -> SiteProvenance:
    return SiteProvenance(
        origin=SourceOrigin(
            module="tests",
            structural_path=("case", 1),
            role="ask",
            source_protocol="qvr-source/v0.19",
        )
    )


def _scope(label: str) -> StaticScopeId:
    return StaticScopeId.derive("tests", "case-branch", label)


def test_constructor_result_indices_are_checked() -> None:
    registry, family, bool_constructor, _, _ = _registry_with_expr()
    value = ConstructorValue(
        constructor=bool_constructor.id,
        static_arguments=(),
        fields=(),
        result_type=_expr(family, BOOL),
    )
    assert infer_value(value, registry) == _expr(family, BOOL)

    malformed = ConstructorValue(
        constructor=bool_constructor.id,
        static_arguments=(),
        fields=(),
        result_type=_expr(family, INT),
    )
    with pytest.raises(KernelError, match="constructor result"):
        infer_value(malformed, registry)


def test_indexed_coverage_omits_only_certified_impossible_constructors() -> None:
    registry, family, bool_constructor, _, any_constructor = _registry_with_expr()
    constructors = tuple(
        registry.constructor(constructor) for constructor in family.constructors
    )
    scrutinee = _expr(family, BOOL)

    missing_polymorphic = check_indexed_coverage(
        family,
        constructors,
        scrutinee,
        (BranchPattern(constructor=bool_constructor.id),),
    )
    assert missing_polymorphic.status is CoverageStatus.INCOMPLETE
    assert missing_polymorphic.missing == (any_constructor.id,)

    complete = check_indexed_coverage(
        family,
        constructors,
        scrutinee,
        (
            BranchPattern(constructor=bool_constructor.id),
            BranchPattern(constructor=any_constructor.id),
        ),
    )
    assert complete.status is CoverageStatus.COMPLETE


def test_indexed_coverage_distinguishes_nullary_and_applied_index_constructors() -> (
    None
):
    natural = UserIndexSort(name="Nat", constructors=("Z", "S"), arities=(0, 1))
    family_id = FamilyId.derive("tests", "Vec")
    nil_id = ConstructorId.derive(family_id, "Nil")
    cons_id = ConstructorId.derive(family_id, "Cons")
    family = FamilyDecl(
        id=family_id,
        name="Vec",
        parameters=(),
        indices=(IndexBinder(name="n", sort=natural, refinable=True),),
        constructors=(nil_id, cons_id),
    )
    nil = ConstructorDecl(
        id=nil_id,
        family=family_id,
        name="Nil",
        telescope=(),
        fields=(),
        result_indices=(IndexLiteral(value="Z", sort=natural),),
    )
    cons = ConstructorDecl(
        id=cons_id,
        family=family_id,
        name="Cons",
        telescope=(IndexBinder(name="m", sort=natural),),
        fields=(),
        result_indices=(
            IndexConstructor(
                name="S",
                arguments=(IndexVariable(name="m", sort=natural),),
                sort=natural,
            ),
        ),
    )
    scrutinee = TypeApplication(
        constructor=family.type_constructor,
        arguments=(
            IndexConstructor(
                name="S",
                arguments=(IndexVariable(name="n", sort=natural),),
                sort=natural,
            ),
        ),
    )

    coverage = check_indexed_coverage(
        family,
        (nil, cons),
        scrutinee,
        (BranchPattern(constructor=cons_id),),
    )

    assert coverage.status is CoverageStatus.COMPLETE


def test_unknown_branch_refinement_retains_its_effect_row() -> None:
    registry, family, bool_constructor, _, any_constructor = _registry_with_expr()
    effect_id = EffectId.derive("tests", "Ask")
    operation = OperationDef(
        id=OperationId.derive(str(effect_id), "ask"),
        name="ask",
        telescope=(),
        arguments=(),
        result_type=STRING,
    )
    effect = EffectDef(
        ref=EffectRef(id=effect_id, name="Ask"), telescope=(), operations=(operation,)
    )
    registry.register_effect(effect)
    instance = instantiate_effect(effect.ref, module="tests", lexical_path=("case",))
    request = EffectRequest(
        instance=instance.instance,
        effect=effect.ref,
        operation=operation.id,
        static_arguments=(),
        arguments=(),
        result_type=STRING,
        origin=_origin(),
    )
    scrutinee_type = _expr(family, BOOL)
    scrutinee = ConstructorValue(
        constructor=bool_constructor.id,
        static_arguments=(),
        fields=(),
        result_type=scrutinee_type,
    )
    bool_scope = _scope("effect-bool")
    any_scope = _scope("effect-any")
    unknown_type = constructor_skolems(any_constructor, any_scope)[0]
    case = Case(
        scrutinee=scrutinee,
        motive=CaseMotive(
            indices=(TypeBinder(name="result_index", kind=TYPE, refinable=True),),
            result_type=STRING,
        ),
        branches=(
            CaseBranch(
                constructor=bool_constructor.id,
                static_arguments=(),
                fields=(),
                body=Return(value=LiteralValue(value="bool", type=STRING)),
                scope=bool_scope,
            ),
            CaseBranch(
                constructor=any_constructor.id,
                static_arguments=(unknown_type,),
                fields=(),
                body=Perform(request=request),
                scope=any_scope,
            ),
        ),
    )

    inferred = infer_computation(case, registry)
    assert inferred.result == STRING
    assert inferred.effects.lookup(instance.instance) == effect.ref
    refinement = refine_branch(
        family,
        any_constructor,
        scrutinee_type,
        (unknown_type,),
    )
    assert refinement.reachability is Reachability.UNKNOWN


def test_branch_equality_evidence_cannot_escape_its_context() -> None:
    registry, family, _, _, any_constructor = _registry_with_expr()
    refinement = refine_branch(
        family,
        any_constructor,
        _expr(family, BOOL),
        (TypeVariable(name="x"),),
    )
    evidence = EvidenceValue(evidence=refinement.givens[0])

    with pytest.raises(KernelError, match="escaped or was forged"):
        infer_value(evidence, registry)
    assert (
        infer_value(
            evidence,
            registry,
            CheckContext().with_givens(refinement.givens),
        )
        == refinement.givens[0].equality
    )


@pytest.mark.parametrize(
    "forbidden_type",
    (
        FunctionType(parameter=INT, result=INT),
        EqualityType(kind=TYPE, left=INT, right=INT),
        TypeApplication(
            constructor=TypeConstructorRef(id=TypeId.derive("tests", "Box"), name="Box")
        ),
    ),
)
def test_literals_are_restricted_to_primitive_types(forbidden_type: object) -> None:
    with pytest.raises(KernelError, match="restricted to"):
        infer_value(
            LiteralValue(value=None, type=forbidden_type),  # type: ignore[arg-type]
            KernelRegistry(),
        )


def test_case_branch_cannot_choose_a_constructor_existential() -> None:
    family_id = FamilyId.derive("tests", "Packed")
    constructor_id = ConstructorId.derive(str(family_id), "Pack")
    family = FamilyDecl(
        id=family_id,
        name="Packed",
        parameters=(),
        indices=(),
        constructors=(constructor_id,),
    )
    constructor = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Pack",
        telescope=(TypeBinder(name="packed"),),
        fields=(FieldDef(name="value", type=TypeVariable(name="packed")),),
        result_indices=(),
    )
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(constructor)
    packed_type = TypeApplication(constructor=family.type_constructor)
    scrutinee = ConstructorValue(
        constructor=constructor_id,
        static_arguments=(STRING,),
        fields=(LiteralValue(value="not an integer", type=STRING),),
        result_type=packed_type,
    )
    field = Local(name="value", type=INT)
    computation = Case(
        scrutinee=scrutinee,
        motive=CaseMotive(indices=(), result_type=INT),
        branches=(
            CaseBranch(
                constructor=constructor_id,
                static_arguments=(INT,),
                fields=(field,),
                body=Return(value=Var(local=field)),
                scope=_scope("forged-pack"),
            ),
        ),
    )

    with pytest.raises(KernelError, match="canonical rigid skolems"):
        infer_computation(computation, registry)


def test_constructor_skolem_cannot_escape_through_a_case_motive() -> None:
    family_id = FamilyId.derive("tests", "EscapingPack")
    constructor_id = ConstructorId.derive(family_id, "Pack")
    family = FamilyDecl(
        id=family_id,
        name="EscapingPack",
        parameters=(),
        indices=(),
        constructors=(constructor_id,),
    )
    constructor = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Pack",
        telescope=(TypeBinder(name="packed"),),
        fields=(FieldDef(name="value", type=TypeVariable(name="packed")),),
        result_indices=(),
    )
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(constructor)
    packed_type = TypeApplication(constructor=family.type_constructor)
    scrutinee = ConstructorValue(
        constructor=constructor_id,
        static_arguments=(STRING,),
        fields=(LiteralValue(value="hidden", type=STRING),),
        result_type=packed_type,
    )
    scope = _scope("escape")
    skolem = constructor_skolems(constructor, scope)[0]
    assert isinstance(skolem, TypeVariable)
    field = Local(name="hidden", type=skolem)
    computation = Case(
        scrutinee=scrutinee,
        motive=CaseMotive(indices=(), result_type=skolem),
        branches=(
            CaseBranch(
                constructor=constructor_id,
                static_arguments=(skolem,),
                fields=(field,),
                body=Return(value=Var(local=field)),
                scope=scope,
            ),
        ),
    )

    with pytest.raises(KernelError, match="escapes its case branch"):
        infer_computation(computation, registry)

    effect_id = EffectId.derive("tests", "SkolemEffect")
    operation = OperationDef(
        id=OperationId.derive(effect_id, "emit"),
        name="emit",
        telescope=(),
        arguments=(),
        result_type=UNIT,
    )
    effect = EffectDef(
        ref=EffectRef(id=effect_id, name="SkolemEffect"),
        telescope=(TypeBinder(name="payload"),),
        operations=(operation,),
    )
    registry.register_effect(effect)
    effect_scope = _scope("effect-row-escape")
    effect_skolem = constructor_skolems(constructor, effect_scope)[0]
    assert isinstance(effect_skolem, TypeVariable)
    applied = effect.apply((effect_skolem,))
    instance = instantiate_effect(
        applied,
        module="tests",
        lexical_path=("effect-row-escape",),
    )
    request = EffectRequest(
        instance=instance.instance,
        effect=applied,
        operation=operation.id,
        static_arguments=(),
        arguments=(),
        result_type=UNIT,
        origin=_origin(),
    )
    effect_case = Case(
        scrutinee=scrutinee,
        motive=CaseMotive(indices=(), result_type=UNIT),
        branches=(
            CaseBranch(
                constructor=constructor_id,
                static_arguments=(effect_skolem,),
                fields=(Local(name="ignored", type=effect_skolem),),
                body=Perform(request=request),
                scope=effect_scope,
            ),
        ),
    )

    with pytest.raises(KernelError, match="case branch effect row"):
        infer_computation(effect_case, registry)


def test_constructor_registration_checks_scope_shadowing_and_index_sort() -> None:
    family_id = FamilyId.derive("tests", "Scoped")
    constructor_id = ConstructorId.derive(str(family_id), "Scoped")
    family = FamilyDecl(
        id=family_id,
        name="Scoped",
        parameters=(TypeBinder(name="parameter"),),
        indices=(IndexBinder(name="index", sort=NAT, refinable=True),),
        constructors=(constructor_id,),
    )
    registry = KernelRegistry()
    registry.register_family(family)

    shadowing = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Scoped",
        telescope=(TypeBinder(name="parameter"),),
        fields=(),
        result_indices=(IndexVariable(name="missing", sort=NAT),),
    )
    with pytest.raises(KernelError, match="shadows family binders"):
        registry.register_constructor(shadowing)

    unbound = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Scoped",
        telescope=(),
        fields=(FieldDef(name="bad", type=TypeVariable(name="missing")),),
        result_indices=(IndexVariable(name="missing", sort=NAT),),
    )
    with pytest.raises(KernelError, match="unbound type variable"):
        registry.register_constructor(unbound)

    unbound_index = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Scoped",
        telescope=(),
        fields=(),
        result_indices=(IndexVariable(name="missing", sort=NAT),),
    )
    with pytest.raises(KernelError, match="unbound index variable"):
        registry.register_constructor(unbound_index)

    effect_carrier = TypeConstructorRef(
        id=TypeId.derive("tests", "EffectCarrier"),
        name="EffectCarrier",
        telescope=(EffectBinder(name="effect"),),
    )
    unbound_effect = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Scoped",
        telescope=(),
        fields=(
            FieldDef(
                name="bad",
                type=TypeApplication(
                    constructor=effect_carrier,
                    arguments=(EffectVariable(name="missing"),),
                ),
            ),
        ),
        result_indices=(IndexVariable(name="also_missing", sort=NAT),),
    )
    with pytest.raises(KernelError, match="unbound effect variable"):
        registry.register_constructor(unbound_effect)

    wrong_sort = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Scoped",
        telescope=(IndexBinder(name="shape", sort=SHAPE),),
        fields=(),
        result_indices=(IndexVariable(name="shape", sort=SHAPE),),
    )
    with pytest.raises(KernelError, match="wrong index sort"):
        registry.register_constructor(wrong_sort)


def test_constructor_skolems_are_alpha_stable_and_scope_fresh() -> None:
    _, _, _, _, constructor = _registry_with_expr()
    left_scope = _scope("left")
    right_scope = _scope("right")
    left = constructor_skolems(constructor, left_scope)[0]
    right = constructor_skolems(constructor, right_scope)[0]

    assert isinstance(left, TypeVariable)
    assert isinstance(right, TypeVariable)
    assert left != right
    assert left.identity != right.identity
    assert TypeVariable(name="renamed", kind=left.kind, identity=left.identity) == left

    with pytest.raises(KernelError, match="reused an enclosing static scope"):
        CheckContext().with_static_scope(left_scope).with_static_scope(left_scope)


def test_case_branch_scopes_must_be_globally_fresh() -> None:
    registry, family, bool_constructor, _, any_constructor = _registry_with_expr()
    repeated = _scope("repeated")
    scrutinee = ConstructorValue(
        constructor=bool_constructor.id,
        static_arguments=(),
        fields=(),
        result_type=_expr(family, BOOL),
    )
    computation = Case(
        scrutinee=scrutinee,
        motive=CaseMotive(
            indices=(TypeBinder(name="index", kind=TYPE, refinable=True),),
            result_type=STRING,
        ),
        branches=(
            CaseBranch(
                constructor=bool_constructor.id,
                static_arguments=(),
                fields=(),
                body=Return(value=LiteralValue(value="bool", type=STRING)),
                scope=repeated,
            ),
            CaseBranch(
                constructor=any_constructor.id,
                static_arguments=constructor_skolems(any_constructor, repeated),
                fields=(),
                body=Return(value=LiteralValue(value="any", type=STRING)),
                scope=repeated,
            ),
        ),
    )

    with pytest.raises(KernelError, match="globally fresh"):
        infer_computation(computation, registry)


def test_constructor_instantiation_deeply_validates_static_arguments() -> None:
    family_id = FamilyId.derive("tests", "DeepArguments")
    constructor_id = ConstructorId.derive(family_id, "Deep")
    family = FamilyDecl(
        id=family_id,
        name="DeepArguments",
        parameters=(TypeBinder(name="value"),),
        indices=(),
        constructors=(constructor_id,),
    )
    constructor = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Deep",
        telescope=(),
        fields=(),
        result_indices=(),
    )
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(constructor)
    malformed = TypeApplication(
        constructor=TypeConstructorRef(
            id=TypeId.derive("tests", "NeedsIndex"),
            name="NeedsIndex",
            telescope=(IndexBinder(name="index", sort=NAT),),
        ),
        arguments=(INT,),
    )
    result = TypeApplication(
        constructor=family.type_constructor, arguments=(malformed,)
    )

    with pytest.raises(TypeError, match="index argument"):
        infer_value(
            ConstructorValue(
                constructor=constructor_id,
                static_arguments=(malformed,),
                fields=(),
                result_type=result,
            ),
            registry,
        )


def test_family_lookup_enforces_canonical_type_constructor_metadata() -> None:
    registry, family, _, _, _ = _registry_with_expr()
    renamed = TypeConstructorRef(
        id=family.type_constructor.id,
        name="RenamedExpr",
        telescope=family.type_constructor.telescope,
    )
    assert (
        registry.family_for_type(
            TypeApplication(constructor=renamed, arguments=(BOOL,))
        )
        == family
    )

    conflicting = TypeConstructorRef(
        id=family.type_constructor.id,
        name="RenamedExpr",
        telescope=(),
    )
    with pytest.raises(KernelError, match="conflicting telescope metadata"):
        registry.family_for_type(TypeApplication(constructor=conflicting))
