"""Natural-number shape arithmetic preserves sorts, identities, and exact evidence."""

import didactic.api as dx
import pytest

from quivers.qiec import (
    NAT,
    IndexBinder,
    IndexLiteral,
    IndexSum,
    IndexVariable,
    ShapeIndex,
    StaticVariableId,
    UserIndexSort,
    add_indices,
    dumps,
    loads,
    render_static,
    shift_index,
)
from quivers.qiec.checking import KernelError, StaticScope, check_static
from quivers.qiec.index_arithmetic import infer_index_extents, solve_index_equation
from quivers.qiec.substitution import StaticSubstitution, substitute_index
from quivers.qiec.types import check_static_arguments, index_sort


def nat(value):
    return IndexLiteral(value=value, sort=NAT)


def variable(name):
    return IndexVariable(name=name, sort=NAT)


def test_addition_has_one_normal_form_under_association_commutation_and_substitution():
    n, m = variable("n"), variable("m")
    left = add_indices(shift_index(n, 2), m, nat(3))
    right = add_indices(m, n, nat(5))
    assert left == right and hash(left) == hash(right)
    assert render_static(left) == "nat_add(nat_add(m, n), 5)"
    assert add_indices() == nat(0)
    assert add_indices(n, nat(0)) is n
    assert add_indices(nat(2), nat(3)) == nat(5)
    assert substitute_index(
        left, StaticSubstitution(indices=(("n", nat(4)), ("m", nat(7))))
    ) == nat(16)
    assert substitute_index(
        left, StaticSubstitution(indices=(("n", shift_index(m, 1)),))
    ) == add_indices(m, m, nat(6))


def test_literal_and_symbolic_predecessors_require_a_present_nonnegative_constant():
    n = variable("n")
    assert shift_index(nat(3), -3) == nat(0)
    assert shift_index(shift_index(n, 2), -2) == n
    assert shift_index(shift_index(n, 2), -1) == shift_index(n, 1)
    for term, offset in ((n, -1), (nat(0), -1), (shift_index(n, 2), -3)):
        with pytest.raises(ValueError, match="unproved"):
            shift_index(term, offset)
    for offset in (True, 1.5):
        with pytest.raises(TypeError, match="integer"):
            shift_index(n, offset)


def test_rigid_names_are_never_captured_and_serialization_retains_their_identity():
    n = variable("n")
    skolem = IndexVariable(
        name="n", sort=NAT, identity=StaticVariableId.derive("nat", "branch")
    )
    term = add_indices(n, skolem, nat(1))
    rewritten = substitute_index(term, StaticSubstitution(indices=(("n", nat(3)),)))
    assert rewritten == shift_index(skolem, 4)
    assert loads(dumps(term)) == term
    assert loads(dumps(rewritten)) == rewritten
    with pytest.raises(TypeError, match="rigid"):
        solve_index_equation(term, nat(5), skolem)
    with pytest.raises(TypeError, match="rigid"):
        infer_index_extents([(term, 5)])


def test_sum_scope_and_sort_checking_recurse_into_every_summand():
    n, m = variable("n"), variable("m")
    term = add_indices(n, m, nat(1))
    scope = StaticScope(indices=(("n", NAT), ("m", NAT)))
    check_static(term, scope)
    assert index_sort(term) == NAT
    assert check_static_arguments((IndexBinder(name="k", sort=NAT),), (term,))[1] == (
        ("k", term),
    )
    with pytest.raises(KernelError, match="unbound.*m"):
        check_static(term, StaticScope(indices=(("n", NAT),)))
    state = UserIndexSort(name="State", constructors=("cold", "warm"))
    with pytest.raises(TypeError, match="Nat"):
        add_indices(n, IndexLiteral(value="cold", sort=state))
    with pytest.raises(TypeError, match="Nat"):
        add_indices(ShapeIndex(dimensions=(n,)))
    with pytest.raises((ValueError, TypeError, dx.ValidationError)):
        IndexSum(terms=(n,), constant=-1)
    with pytest.raises((ValueError, TypeError, dx.ValidationError)):
        IndexSum(terms=(n,), constant=0)


def test_symbolic_equation_cancellation_returns_only_provably_natural_terms():
    n, m = variable("n"), variable("m")
    assert solve_index_equation(shift_index(n, 1), shift_index(m, 1), n) == m
    assert solve_index_equation(shift_index(n, 1), nat(4), n) == nat(3)
    assert solve_index_equation(add_indices(n, n), add_indices(m, m), n) == m
    assert solve_index_equation(add_indices(n, n), nat(6), n) == nat(3)
    assert solve_index_equation(n, m, n) == m
    assert solve_index_equation(shift_index(n, 1), m, n) is None
    assert solve_index_equation(shift_index(n, 1), nat(0), n) is None
    assert solve_index_equation(add_indices(n, n), nat(5), n) is None
    assert solve_index_equation(n, n, n) is None


def test_exact_affine_shape_evidence_solves_multiple_axes_without_equal_size_guesses():
    n, m = variable("n"), variable("m")
    assert infer_index_extents([(shift_index(n, 1), 4)]) == {"n": 3}
    assert infer_index_extents([(add_indices(n, m), 5), (add_indices(n, m, m), 8)]) == {
        "m": 3,
        "n": 2,
    }
    assert infer_index_extents([(add_indices(n, m), 5)], {"m": 3}) == {"m": 3, "n": 2}
    assert infer_index_extents([(add_indices(n, m), 5)]) == {}
    assert infer_index_extents([(add_indices(n, m), 0)]) == {"m": 0, "n": 0}
    assert infer_index_extents([(n, 3), (add_indices(n, m), 6)]) == {"m": 3, "n": 3}
    for equations in (
        [(shift_index(n, 1), 0)],
        [(n, 2), (shift_index(n, 1), 4)],
        [(add_indices(n, n), 5)],
        [(nat(2), 3)],
    ):
        with pytest.raises(ValueError, match="solution|inconsistent"):
            infer_index_extents(equations)
    with pytest.raises(TypeError, match="integer"):
        infer_index_extents([(n, True)])


def test_rigid_sum_cannot_escape_through_a_case_result_type():
    from quivers.qiec import (
        REAL,
        Case,
        CaseBranch,
        CaseMotive,
        ConstructorDecl,
        ConstructorId,
        ConstructorValue,
        FamilyDecl,
        FamilyId,
        FieldDef,
        KernelRegistry,
        LiteralValue,
        Local,
        Return,
        StaticScopeId,
        TypeApplication,
        TensorValue,
        Var,
        constructor_skolems,
        infer_computation,
    )
    from quivers.qiec.canonical import tensor_type

    family_id = FamilyId.derive("addition", "HiddenWidth")
    constructor_id = ConstructorId.derive(family_id, "Pack")
    family = FamilyDecl(
        id=family_id,
        name="HiddenWidth",
        parameters=(),
        indices=(),
        constructors=(constructor_id,),
    )
    constructor = ConstructorDecl(
        id=constructor_id,
        family=family_id,
        name="Pack",
        telescope=(IndexBinder(name="n", sort=NAT),),
        fields=(
            FieldDef(
                name="values", type=tensor_type(REAL, (shift_index(variable("n"), 1),))
            ),
        ),
        result_indices=(),
    )
    registry = KernelRegistry()
    registry.register_family(family)
    registry.register_constructor(constructor)
    packed_type = TypeApplication(constructor=family.type_constructor)
    scrutinee = ConstructorValue(
        constructor=constructor_id,
        static_arguments=(nat(2),),
        fields=(
            TensorValue(
                items=tuple(
                    LiteralValue(value=value, type=REAL) for value in (1.0, 2.0, 3.0)
                ),
                result_type=tensor_type(REAL, (nat(3),)),
            ),
        ),
        result_type=packed_type,
    )
    scope = StaticScopeId.derive("addition", "hidden-width-branch")
    skolem = constructor_skolems(constructor, scope)[0]
    result_type = tensor_type(REAL, (shift_index(skolem, 1),))
    field = Local(name="values", type=result_type)
    case = Case(
        scrutinee=scrutinee,
        motive=CaseMotive(indices=(), result_type=result_type),
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
        infer_computation(case, registry)
