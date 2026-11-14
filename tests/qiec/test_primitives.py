"""Pure primitives, tuples, and projections in the kernel."""

from __future__ import annotations

import pytest
import didactic.api as dx

from quivers.qiec import (
    BOOL,
    INT,
    REAL,
    STRING,
    CheckContext,
    Evaluator,
    KernelError,
    KernelRegistry,
    LiteralValue,
    PrimitiveApplication,
    PrimitiveId,
    Projection,
    Return,
    TupleValue,
    Var,
    infer_value,
    loads,
    dumps,
    primitive,
    PRIMITIVES,
    IMPLEMENTATIONS,
)
from quivers.qiec.evaluator import EvaluationError
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.substitution import StaticSubstitution, substitute_value
from quivers.qiec.terms import Local, LiteralData
from quivers.qiec.types import TypeVariable, product_type

ORIGIN = SourceOrigin(
    module="primitives",
    structural_path=("test",),
    role="primitive",
    source_protocol="test",
)


def test_primitive_signature_updates_validate_recursive_types() -> None:
    """Primitive signatures compare typed fields and reject malformed updates."""
    signature = primitive("add_real")
    assert signature == signature.with_(parameters=(REAL, REAL))
    assert hash(signature) == hash(signature.with_(parameters=(REAL, REAL)))
    assert signature != signature.with_(result=INT)
    with pytest.raises(dx.ValidationError, match="parameters"):
        signature.with_(parameters=(REAL, "Real"))
    with pytest.raises(dx.ValidationError):
        signature.with_(capability="unknown")


def _apply(name: str, *arguments: LiteralData) -> PrimitiveApplication:
    """Apply a primitive to literal arguments at its own signature.

    Parameters
    ----------
    name : str
        The primitive's name.
    *arguments : LiteralData
        Literal values, typed by the signature's parameters.

    Returns
    -------
    PrimitiveApplication
        The well-typed application.
    """
    signature = primitive(name)
    return PrimitiveApplication(
        primitive=signature.id,
        name=name,
        arguments=tuple(
            LiteralValue(value=argument, type=parameter)
            for argument, parameter in zip(arguments, signature.parameters, strict=True)
        ),
        result_type=signature.result,
        origin=ORIGIN,
    )


def test_every_primitive_has_an_implementation_and_a_stable_identity() -> None:
    """The registry is closed and every entry is runnable."""
    assert set(PRIMITIVES) == set(IMPLEMENTATIONS)
    identities = {signature.id for signature in PRIMITIVES.values()}
    assert len(identities) == len(PRIMITIVES)
    assert primitive("add_int").id == PrimitiveId.derive("builtin", "add_int")
    with pytest.raises(KeyError):
        primitive("add_complex")


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("add_int", (2, 3), 5),
        ("div_int", (7, 2), 3),
        ("div_int", (-7, 2), -3),
        ("mod_int", (-7, 2), -1),
        ("mod_int", (7, -2), 1),
        ("div_real", (1.0, 4.0), 0.25),
        ("pow_real", (2.0, 3.0), 8.0),
        ("lt_int", (1, 2), True),
        ("eq_string", ("a", "a"), True),
        ("and", (True, False), False),
        ("not", (False,), True),
        ("concat", ("ab", "cd"), "abcd"),
        ("int_to_real", (3,), 3.0),
        ("real_to_int", (-2.7,), -2),
    ],
)
def test_primitives_check_and_evaluate(
    name: str, arguments: tuple[LiteralData, ...], expected: object
) -> None:
    """A well-typed application infers the registry's result type and runs."""
    application = _apply(name, *arguments)
    assert infer_value(application, KernelRegistry()) == primitive(name).result
    assert Evaluator().evaluate(Return(value=application)) == expected


def test_integer_division_by_zero_is_an_evaluation_error() -> None:
    with pytest.raises(EvaluationError, match="div_int"):
        Evaluator().evaluate(Return(value=_apply("div_int", 1, 0)))


def test_primitive_applications_are_checked_against_the_registry() -> None:
    """Arity, argument types, the claimed result, and the identity all matter."""
    registry = KernelRegistry()
    signature = primitive("add_int")
    with pytest.raises(KernelError, match="takes 2 argument"):
        infer_value(
            PrimitiveApplication(
                primitive=signature.id,
                name="add_int",
                arguments=(LiteralValue(value=1, type=INT),),
                result_type=INT,
                origin=ORIGIN,
            ),
            registry,
        )
    with pytest.raises(KernelError, match="argument 1 of primitive"):
        infer_value(
            PrimitiveApplication(
                primitive=signature.id,
                name="add_int",
                arguments=(
                    LiteralValue(value=1, type=INT),
                    LiteralValue(value=1.0, type=REAL),
                ),
                result_type=INT,
                origin=ORIGIN,
            ),
            registry,
        )
    with pytest.raises(KernelError, match="produces"):
        infer_value(
            PrimitiveApplication(
                primitive=signature.id,
                name="add_int",
                arguments=(
                    LiteralValue(value=1, type=INT),
                    LiteralValue(value=2, type=INT),
                ),
                result_type=REAL,
                origin=ORIGIN,
            ),
            registry,
        )
    with pytest.raises(KernelError, match="identity of another"):
        infer_value(
            PrimitiveApplication(
                primitive=primitive("sub_int").id,
                name="add_int",
                arguments=(
                    LiteralValue(value=1, type=INT),
                    LiteralValue(value=2, type=INT),
                ),
                result_type=INT,
                origin=ORIGIN,
            ),
            registry,
        )
    with pytest.raises(KernelError, match="unknown primitive"):
        infer_value(
            PrimitiveApplication(
                primitive=signature.id,
                name="add_complex",
                arguments=(),
                result_type=INT,
                origin=ORIGIN,
            ),
            registry,
        )


def test_tuples_and_projections_check_evaluate_and_round_trip() -> None:
    registry = KernelRegistry()
    pair = TupleValue(
        items=(LiteralValue(value=1, type=INT), LiteralValue(value="x", type=STRING)),
        result_type=product_type(INT, STRING),
    )
    assert infer_value(pair, registry) == product_type(INT, STRING)
    second = Projection(value=pair, position=1, result_type=STRING)
    assert infer_value(second, registry) == STRING
    assert Evaluator().evaluate(Return(value=pair)) == (1, "x")
    assert Evaluator().evaluate(Return(value=second)) == "x"
    assert loads(dumps(Return(value=second))) == Return(value=second)
    assert loads(dumps(Return(value=_apply("add_int", 1, 2)))) == Return(
        value=_apply("add_int", 1, 2)
    )

    with pytest.raises(KernelError, match="not the claimed"):
        infer_value(
            TupleValue(items=pair.items, result_type=product_type(INT, INT)), registry
        )
    with pytest.raises(KernelError, match="outside a product"):
        infer_value(Projection(value=pair, position=2, result_type=INT), registry)
    with pytest.raises(KernelError, match="non-product"):
        infer_value(
            Projection(
                value=LiteralValue(value=1, type=INT), position=0, result_type=INT
            ),
            registry,
        )
    with pytest.raises(KernelError, match="not the claimed"):
        infer_value(Projection(value=pair, position=0, result_type=BOOL), registry)


def test_substitution_reaches_tuple_and_primitive_argument_types() -> None:
    """Static substitution rewrites the types inside the new value forms."""
    variable = TypeVariable(name="A")
    local = Local(name="x", type=variable)
    pair = TupleValue(
        items=(Var(local=local), LiteralValue(value=1, type=INT)),
        result_type=product_type(variable, INT),
    )
    substitution = StaticSubstitution(types=(("A", REAL),))
    rewritten = substitute_value(pair, substitution)
    assert rewritten == TupleValue(
        items=(Var(local=Local(name="x", type=REAL)), LiteralValue(value=1, type=INT)),
        result_type=product_type(REAL, INT),
    )
    projected = substitute_value(
        Projection(value=pair, position=0, result_type=variable), substitution
    )
    assert projected == Projection(value=rewritten, position=0, result_type=REAL)
    context = CheckContext().extend(Local(name="x", type=REAL))
    assert infer_value(projected, KernelRegistry(), context) == REAL
