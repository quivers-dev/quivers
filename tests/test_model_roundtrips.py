"""JSON round-trip tests for every migrated didactic Model.

Each test case is a representative instance of a Model variant; the
roundtrip fixture dumps it to JSON, validates back from JSON, and asserts
equality. This pins didactic's serialization contract for the migrated
classes and surfaces any encoding/decoding asymmetries.

For TaggedUnion variants the round-trip goes through the union root's
:py:meth:`model_validate_json`, exercising the ``kind`` discriminator
dispatch; flat :class:`dx.Model` subclasses validate against themselves.
"""

from __future__ import annotations

from typing import TypeVar

import pytest

# AST nodes
from quivers.dsl.ast_nodes import (
    BindStep,
    CategoryDecl,
    CompositionDecl,
    ContinuousConstructor,
    DefineDecl,
    DiscreteConstructor,
    DrawArgName as _DrawArgName,
    DrawStep,
    ExportDecl,
    Expr,
    ExprChartFold,
    ExprCompose,
    ExprCurry,
    ExprFan,
    ExprIdent,
    ExprIdentity,
    ExprMarginalize,
    ExprParser,
    ExprRepeat,
    ExprScan,
    ExprStack,
    ExprTensorProduct,
    LetExprBinOp,
    LetExprCall,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
    LetStep,
    Module,
    MorphismDecl,
    ObjectCoproduct,
    ObjectDecl,
    ObjectEffectApply,
    ObjectExpr,
    ObjectProduct,
    ObjectSlash,
    ProgramDecl,
    ProgramStep,
    RuleDecl,
    Statement,
    TypeFromExpr,
    TypeName,
)

# stochastic Category union
from quivers.stochastic.categories import (
    AtomicCategory,
    Category,
    ModalCategory,
    ProductCategory,
    SlashCategory,
    UnitCategory,
)

# RuleSystem
from quivers.stochastic._rule_system import RuleSystem

# core/objects
from quivers.core.objects import (
    CoproductSet,
    EnumSet,
    FinSet,
    FreeMonoid,
    FreeResiduated,
    ProductSet,
    SetObject,
)

# categorical/monoidal
from quivers.categorical.monoidal import EmptySet

# continuous/spaces
from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    ProductSpace,
    Simplex,
)

# monadic/algebraic + bridges
from quivers.monadic.algebraic import (
    EffectSignature,
    FreeMonad,
    Handler,
    Operation,
)
from quivers.monadic.bridges import ArrowMonad, Kleisli


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------


# AST: object / type expressions
_TYPE_NAME = TypeName(name="State")
_TYPE_PRODUCT = ObjectProduct(components=(TypeName(name="A"), TypeName(name="B")))
_TYPE_COPRODUCT = ObjectCoproduct(components=(_TYPE_NAME, TypeName(name="X")))
_TYPE_SLASH = ObjectSlash(
    result=TypeName(name="X"), argument=TypeName(name="Y"), direction="/"
)
_TYPE_EFFECT_APPLY = ObjectEffectApply(effect="Cont_S", args=(TypeName(name="NP"),))
_DISCRETE_CTOR = DiscreteConstructor(constructor="FinSet", args=("3",))
_CONTINUOUS_CTOR = ContinuousConstructor(
    constructor="Real", args=("3",), kwargs={"low": 0.0, "high": 1.0}
)

# AST: value expressions
_E_IDENT = ExprIdent(name="f")
_E_IDENTITY = ExprIdentity(object_name="A")
_E_COMPOSE = ExprCompose(left=_E_IDENT, right=ExprIdent(name="g"))
_E_TENSOR = ExprTensorProduct(left=_E_IDENT, right=ExprIdent(name="g"))
_E_FAN = ExprFan(exprs=(_E_IDENT, ExprIdent(name="g")))
_E_REPEAT = ExprRepeat(expr=_E_IDENT, count=3)
_E_STACK = ExprStack(expr=_E_IDENT, count=4)
_E_SCAN = ExprScan(expr=_E_IDENT, init="zeros")
_E_MARG = ExprMarginalize(inner=_E_IDENT, names=("A",))
_E_PARSER = ExprParser(
    rules=("evaluation",),
    categories=("S", "NP"),
    terminal="Token",
    start="S",
    depth=1,
)
_E_CURRY = ExprCurry(inner=_E_IDENT, direction="right")
_E_CHART_FOLD = ExprChartFold(
    lex=ExprIdent(name="lex"),
    binary=ExprIdent(name="combine"),
    unary=None,
    start="S",
    depth=2,
    effect_depth=0,
)

# AST: let-expr nodes
_LE_LIT = LetExprLiteral(value=0.5)
_LE_VAR = LetExprVar(name="theta")
_LE_BINOP = LetExprBinOp(op="+", left=_LE_VAR, right=_LE_LIT)
_LE_UNARY = LetExprUnaryOp(operand=_LE_VAR)
_LE_CALL = LetExprCall(func="log", args=(_LE_VAR,))

# AST: program steps
_DRAW = DrawStep(vars=("x",), morphism="f", args=None)
_BIND = BindStep(vars=("y",), morphism="g", args=(_DrawArgName(text="x"),))
_LET_STEP = LetStep(name="y", value=_LE_BINOP)

# AST: top-level statements
_COMPDECL = CompositionDecl(name="product_fuzzy", level="algebra")
_CDECL = CategoryDecl(names=("S",))
_RDECL = RuleDecl(
    name="app",
    variables=("X", "Y"),
    premises=(_TYPE_SLASH, TypeName(name="Y")),
    conclusion=TypeName(name="X"),
)
_ODECL = ObjectDecl(names=("State",), init=TypeFromExpr(expr=_TYPE_NAME))
_MDECL = MorphismDecl(
    names=("f",),
    domain=_TYPE_NAME,
    codomain=_TYPE_NAME,
)
_PDECL = ProgramDecl(
    name="p",
    params=None,
    domain=_TYPE_NAME,
    codomain=_TYPE_NAME,
    draws=(_DRAW,),
    return_vars=("x",),
)
_LDECL = DefineDecl(name="h", expr=_E_IDENT)
_OUTDECL = ExportDecl(expr=_E_IDENT)

_MODULE = Module(statements=(_COMPDECL, _ODECL, _MDECL, _OUTDECL))

# stochastic categories
_CAT_ATOM = AtomicCategory(name="S")
_CAT_SLASH_RT = SlashCategory(
    result=_CAT_ATOM, argument=AtomicCategory(name="NP"), direction="/"
)
_CAT_PROD_RT = ProductCategory(left=_CAT_ATOM, right=AtomicCategory(name="N"))
_CAT_UNIT = UnitCategory()
_CAT_MODAL = ModalCategory(modality="◇", inner=_CAT_ATOM)

# RuleSystem
_RULE_SYS = RuleSystem(
    binary_rules=((0, 1, 2),),
    unary_rules=((0, 1),),
    n_categories=3,
    description="demo",
    binary_weights=(1.0,),
    unary_weights=(0.5,),
)

# core objects
_FINSET = FinSet(name="X", cardinality=4)
_PRODUCT_SET = ProductSet(components=(_FINSET, FinSet(name="Y", cardinality=3)))
_COPRODUCT_SET = CoproductSet(components=(_FINSET, _FINSET))
_FREE_MONOID = FreeMonoid(generators=_FINSET, max_length=2)
_ENUM_SET = EnumSet(name="Atoms", elements=("NP", "S", "VP"))
_FREE_RES = FreeResiduated(generators=_ENUM_SET, depth=1, ops=("slash",))

# EmptySet (categorical/monoidal)
_EMPTY = EmptySet()

# continuous spaces
_EUCLID = Euclidean(name="x", dim=3)
_EUCLID_BOUNDED = Euclidean(name="u", dim=2, low=0.0, high=1.0)
_SIMPLEX = Simplex(name="probs", dim=4)
_POSREAL = PositiveReals(name="sigma", dim=1)
_PRODUCT_SPACE = ProductSpace(components=(_EUCLID, _SIMPLEX))

# ---------------------------------------------------------------------------
# (root, instance) pairs for parametrization
# ---------------------------------------------------------------------------


CASES: list[tuple[type, object]] = [
    # AST: object / type expressions
    (ObjectExpr, _TYPE_NAME),
    (ObjectExpr, _TYPE_PRODUCT),
    (ObjectExpr, _TYPE_COPRODUCT),
    (ObjectExpr, _TYPE_SLASH),
    (ObjectExpr, _TYPE_EFFECT_APPLY),
    (ObjectExpr, _DISCRETE_CTOR),
    (ObjectExpr, _CONTINUOUS_CTOR),
    # AST: value expressions
    (Expr, _E_IDENT),
    (Expr, _E_IDENTITY),
    (Expr, _E_COMPOSE),
    (Expr, _E_TENSOR),
    (Expr, _E_FAN),
    (Expr, _E_REPEAT),
    (Expr, _E_STACK),
    (Expr, _E_SCAN),
    (Expr, _E_MARG),
    (Expr, _E_PARSER),
    (Expr, _E_CURRY),
    (Expr, _E_CHART_FOLD),
    # AST: let-expr nodes
    (LetExprNode, _LE_LIT),
    (LetExprNode, _LE_VAR),
    (LetExprNode, _LE_BINOP),
    (LetExprNode, _LE_UNARY),
    (LetExprNode, _LE_CALL),
    # AST: program steps
    (ProgramStep, _DRAW),
    (ProgramStep, _BIND),
    (ProgramStep, _LET_STEP),
    # AST: top-level statements
    (Statement, _COMPDECL),
    (Statement, _CDECL),
    (Statement, _RDECL),
    (Statement, _ODECL),
    (Statement, _MDECL),
    (Statement, _PDECL),
    (Statement, _LDECL),
    (Statement, _OUTDECL),
    # AST: module
    (Module, _MODULE),
    # stochastic categories
    (Category, _CAT_ATOM),
    (Category, _CAT_SLASH_RT),
    (Category, _CAT_PROD_RT),
    (Category, _CAT_UNIT),
    (Category, _CAT_MODAL),
    # RuleSystem
    (RuleSystem, _RULE_SYS),
    # core objects
    (SetObject, _FINSET),
    (SetObject, _PRODUCT_SET),
    (SetObject, _COPRODUCT_SET),
    (SetObject, _FREE_MONOID),
    (SetObject, _ENUM_SET),
    (SetObject, _FREE_RES),
    (SetObject, _EMPTY),
    # continuous spaces
    (ContinuousSpace, _EUCLID),
    (ContinuousSpace, _EUCLID_BOUNDED),
    (ContinuousSpace, _SIMPLEX),
    (ContinuousSpace, _POSREAL),
    (ContinuousSpace, _PRODUCT_SPACE),
]


# ---------------------------------------------------------------------------
# monadic/algebraic + bridges
# ---------------------------------------------------------------------------

_OP_GET = Operation(
    name="get",
    parameter=FinSet(name="P", cardinality=2),
    result=FinSet(name="R", cardinality=3),
)
_OP_PUT = Operation(
    name="put",
    parameter=FinSet(name="R", cardinality=3),
    result=FinSet(name="P", cardinality=2),
)
_EFFECT_SIG = EffectSignature(name="IO", operations=(_OP_GET, _OP_PUT))
_FREE_MONAD = FreeMonad(signature=_EFFECT_SIG)


T = TypeVar("T")


@pytest.mark.parametrize(
    "root,instance",
    [
        (Operation, _OP_GET),
        (EffectSignature, _EFFECT_SIG),
        (FreeMonad, _FREE_MONAD),
    ],
    ids=["Operation", "EffectSignature-tuple-of-Operation", "FreeMonad"],
)
def test_algebraic_full_roundtrip(root: type, instance: object) -> None:
    """Operation, EffectSignature, and FreeMonad survive full JSON round-trips."""
    raw = instance.model_dump_json()  # type: ignore[attr-defined]
    parsed = root.model_validate_json(raw)  # type: ignore[attr-defined]
    assert parsed == instance


class _RuntimeMonad:
    """Stand-in for a typeclass-instance object held in an opaque field."""


def test_handler_opaque_fields_do_not_serialise() -> None:
    """``Handler`` round-trips its ``signature`` but drops opaque fields."""
    monad = _RuntimeMonad()
    h = Handler(
        signature=_EFFECT_SIG,
        target=monad,
        return_clause="ret-morphism",
        operation_clauses={"get": "gc"},
    )
    assert h.target is monad
    assert h.return_clause == "ret-morphism"
    assert h.operation_clauses == {"get": "gc"}

    restored = Handler.model_validate_json(h.model_dump_json())
    assert restored.signature == _EFFECT_SIG
    assert restored.target is None
    assert restored.return_clause is None
    assert restored.operation_clauses == {}


def test_kleisli_opaque_monad_field() -> None:
    """``Kleisli`` holds its monad opaquely; identity preserved in-process."""
    monad = _RuntimeMonad()
    kl = Kleisli(monad=monad)
    assert kl.monad is monad
    restored = Kleisli.model_validate_json(kl.model_dump_json())
    assert restored.monad is None


def test_arrow_monad_opaque_arrow_field() -> None:
    """``ArrowMonad`` holds its arrow opaquely; identity preserved in-process."""
    arrow = _RuntimeMonad()
    am = ArrowMonad(arrow=arrow)
    assert am.arrow is arrow
    restored = ArrowMonad.model_validate_json(am.model_dump_json())
    assert restored.arrow is None


def _id(case: tuple[type, object]) -> str:
    root, instance = case
    return f"{root.__name__}-{type(instance).__name__}"


@pytest.mark.parametrize("root,instance", CASES, ids=[_id(c) for c in CASES])
def test_json_roundtrip(root: type, instance: object) -> None:
    """Each migrated dx.Model survives a JSON round-trip via its root."""
    raw = instance.model_dump_json()  # type: ignore[attr-defined]
    parsed = root.model_validate_json(raw)  # type: ignore[attr-defined]
    assert parsed == instance, (
        f"round-trip mismatch for {type(instance).__name__} via {root.__name__}\n"
        f"  json   = {raw}\n"
        f"  parsed = {parsed}\n"
        f"  expect = {instance}"
    )
