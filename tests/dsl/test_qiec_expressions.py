"""The shared pure-expression tree lowered to typed QIEC values."""

from __future__ import annotations

import math

import pytest

from quivers.dsl import parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.pure_builtins import (
    BUILTIN_REGISTRY,
    EAGER_BUILTIN_NAMES,
    HOST_ONLY_BUILTINS,
    PURE_BUILTINS,
    _BUILTIN_PRIMITIVES,
    _COLLECTION_BUILTINS,
    _REDUCTIONS,
    _ROWWISE,
)
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec import (
    BOOL,
    dumps,
    INT,
    REAL,
    STRING,
    LiteralValue,
    PrimitiveApplication,
    Projection,
    Return,
    TupleValue,
    run_named,
)
from quivers.qiec.types import product_type


def _source_value(type_) -> str:
    return {
        INT: "2",
        REAL: "1.5",
        BOOL: "true",
        STRING: '"x"',
    }.get(type_, "weight(0.0)")


def _lower(body: str, signature: str = "() : Int !{}"):
    """Lower one computation and return it.

    Parameters
    ----------
    body : str
        The computation body, one indented block.
    signature : str
        The header after the name.

    Returns
    -------
    NamedComputation
        The lowered computation named ``probe``.
    """
    module = lower_qvr_to_qiec(
        parse(f"define probe{signature} =\n{body}"), file_path="probe.qvr"
    )
    return next(item for item in module.computations if item.name == "probe")


def test_operators_lower_to_primitives_chosen_by_operand_type() -> None:
    """``+`` on integers, reals, and strings names three primitives."""
    ints = _lower("    return (1 + 2) * 3\n")
    assert isinstance(ints.body, Return)
    outer = ints.body.value
    assert isinstance(outer, PrimitiveApplication) and outer.name == "mul_int"
    inner = outer.arguments[0]
    assert isinstance(inner, PrimitiveApplication) and inner.name == "add_int"
    assert inner.arguments == (
        LiteralValue(value=1, type=INT),
        LiteralValue(value=2, type=INT),
    )

    reals = _lower("    return 1.5 / 2.0\n", "() : Real !{}")
    assert isinstance(reals.body, Return)
    assert isinstance(reals.body.value, PrimitiveApplication)
    assert reals.body.value.name == "div_real"

    strings = _lower('    return "a" + "b"\n', "() : String !{}")
    assert isinstance(strings.body, Return)
    assert isinstance(strings.body.value, PrimitiveApplication)
    assert strings.body.value.name == "concat"

    flags = _lower("    return (1 < 2) && not false\n", "() : Bool !{}")
    assert isinstance(flags.body, Return)
    assert isinstance(flags.body.value, PrimitiveApplication)
    assert flags.body.value.name == "and"
    assert [
        argument.name  # type: ignore[union-attr]
        for argument in flags.body.value.arguments
    ] == ["lt_int", "not"]


def test_negated_literals_fold_to_literals() -> None:
    """``-1`` stays a literal, as a static target needs it to."""
    negative = _lower("    return -1\n")
    assert isinstance(negative.body, Return)
    assert negative.body.value == LiteralValue(value=-1, type=INT)
    real = _lower("    return -2.5\n", "() : Real !{}")
    assert isinstance(real.body, Return)
    assert real.body.value == LiteralValue(value=-2.5, type=REAL)


def test_tuples_and_projections_lower_and_run() -> None:
    computation = _lower(
        '    let pair = (1, "x")\n    return pair[1]\n', "() : String !{}"
    )
    module = lower_qvr_to_qiec(
        parse(
            'define probe() : String !{} =\n    let pair = (1, "x")\n    return pair[1]\n'
        ),
        file_path="probe.qvr",
    )
    binding = computation.body
    assert hasattr(binding, "first")
    tuple_value = binding.first.value  # type: ignore[union-attr]
    assert isinstance(tuple_value, TupleValue)
    assert tuple_value.result_type == product_type(INT, STRING)
    projection = binding.then.value  # type: ignore[union-attr]
    assert isinstance(projection, Projection)
    assert projection.position == 1 and projection.result_type == STRING
    assert run_named(module, "probe").value == "x"


def test_an_integer_beside_a_real_is_converted() -> None:
    module = lower_qvr_to_qiec(
        parse(
            "define probe(n : Int, xs : Tensor[Int]([3])) : Real !{} =\n"
            "    let mixed = n * 2.5 + 1\n"
            "    let scaled = xs * 0.5\n"
            "    return mixed + sum(scaled)\n"
        ),
        file_path="probe.qvr",
    )
    rendered = dumps(module)
    assert rendered.count('"int_to_real"') == 3
    assert run_named(module, "probe", (3, (1, 2, 3))).value == pytest.approx(
        3 * 2.5 + 1 + 3.0
    )


def test_builtins_resolve_by_argument_types() -> None:
    module = lower_qvr_to_qiec(
        parse(
            "define probe(n : Int) : Real !{} =\n"
            "    let scaled = real(n) / 4.0\n"
            "    let clipped = max(scaled, 0.5)\n"
            "    return sqrt(clipped) + real(abs(n % 3))\n"
        ),
        file_path="probe.qvr",
    )
    assert run_named(module, "probe", (-7,)).value == pytest.approx(
        (max(-7 / 4.0, 0.5)) ** 0.5 + 1.0
    )


@pytest.mark.parametrize(
    ("name", "signature"),
    [
        (name, signature)
        for name, overloads in _BUILTIN_PRIMITIVES.items()
        for signature in overloads
    ],
)
def test_every_checked_primitive_overload_has_a_typed_lowering(
    name: str, signature: tuple
) -> None:
    arguments = ", ".join(_source_value(type_) for type_ in signature)
    _lower(f"    let value = {name}({arguments})\n    return 0\n")


@pytest.mark.parametrize("name", sorted({*_REDUCTIONS, *_ROWWISE}))
def test_every_checked_tensor_builtin_has_a_typed_lowering(name: str) -> None:
    _lower(f"    let value = {name}([1.0, 2.0])\n    return 0\n")


@pytest.mark.parametrize("name", sorted(PURE_BUILTINS))
def test_every_checked_builtin_rejects_the_wrong_arity(name: str) -> None:
    with pytest.raises(QiecDiagnosticError):
        _lower(f"    let value = {name}()\n    return 0\n")


def test_builtin_registry_is_the_complete_capability_inventory() -> None:
    assert frozenset(BUILTIN_REGISTRY) == EAGER_BUILTIN_NAMES | _COLLECTION_BUILTINS
    assert PURE_BUILTINS | HOST_ONLY_BUILTINS == frozenset(BUILTIN_REGISTRY)
    assert not (PURE_BUILTINS & HOST_ONLY_BUILTINS)
    assert all(
        BUILTIN_REGISTRY[name].targets == {"eager"} for name in HOST_ONLY_BUILTINS
    )
    assert all(BUILTIN_REGISTRY[name].eager is not None for name in EAGER_BUILTIN_NAMES)
    assert BUILTIN_REGISTRY["filter"].qiec_form == "unsupported"


@pytest.mark.parametrize(
    ("body", "signature", "fragment"),
    [
        ("    return 1 + true\n", "() : Int !{}", "both sides must agree"),
        ("    return 1.0 % 2.0\n", "() : Real !{}", "`%` is not defined at Real"),
        ("    return not 1\n", "() : Bool !{}", "`not` takes Bool"),
        ("    return sqrt(4)\n", "() : Real !{}", "not defined at (Int)"),
        ("    return hypot(1.0, 2.0)\n", "() : Real !{}", "unknown builtin"),
        ("    return [1, 2]\n", "() : Int !{}", "list literal"),
        (
            "    let n = 1\n    return n[0]\n",
            "() : Int !{}",
            "selects a tuple component",
        ),
        ("    let p = (1, 2)\n    return p[2]\n", "() : Int !{}", "outside a product"),
        (
            "    let p = (1, 2)\n    return p[0, 1]\n",
            "() : Int !{}",
            "one integer literal",
        ),
    ],
)
def test_ill_typed_expressions_are_source_located(
    body: str, signature: str, fragment: str
) -> None:
    with pytest.raises(QiecDiagnosticError) as captured:
        _lower(body, signature)
    assert captured.value.code == "qiec-primitive"
    assert fragment in str(captured.value)
    assert captured.value.line >= 2


def test_a_computation_applied_as_a_builtin_is_explained() -> None:
    source = (
        "define helper() : Int !{} =\n    return 1\n\n"
        "define probe() : Int !{} =\n    return helper()\n"
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        lower_qvr_to_qiec(parse(source), file_path="probe.qvr")
    assert "let x <- helper(...)" in str(captured.value)


def test_expressions_emit_canonically_and_round_trip() -> None:
    source = (
        "define probe(x : Int, y : Real) : Real !{} =\n"
        "    let doubled = x * 2 + 1\n"
        "    let scaled = real(doubled) / y\n"
        "    let flag = x > 0 && not (y == 1.0)\n"
        "    let grouped = (x + 1) * 2 - (3 - 1)\n"
        "    let pair = (doubled, y)\n"
        "    return pair[1] - scaled\n"
    )
    module = parse(source)
    assert module_to_source(module) == source
    assert parse(module_to_source(module)) == module
    lowered = lower_qvr_to_qiec(module, file_path="probe.qvr")
    assert run_named(lowered, "probe", (3, 2.0)).value == pytest.approx(2.0 - 3.5)
    assert lowered.computations[0].type.result == REAL


def test_tensor_arithmetic_reductions_rowwise_and_comprehensions_run() -> None:
    source = """\
object K : FinSet 3

define scaled(v : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    return tanh(v) * 2.0 + 1.0

define total(v : Tensor[Real]([3])) : Real !{} =
    return sum(v) + max(v) - logsumexp(v) + mean(v)

define weights(v : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    return softmax(v)

define squares() : Tensor[Real]([3]) !{} =
    return factor k : K in real(k) * real(k)

define pick(v : Tensor[Real]([3]), i : Int) : Real !{} =
    return v[i] + sigmoid(0.0)

define table() : Tensor[Real]([3, 2]) !{} =
    return factor k : K, j : FinSet 2 in real(k) + real(j)
"""
    module = lower_qvr_to_qiec(parse(source), file_path="tensors.qvr")
    v = (0.0, 1.0, -1.0)
    scaled = run_named(module, "scaled", (v,)).value
    assert scaled == pytest.approx(
        (1.0, 1.0 + 2.0 * math.tanh(1.0), 1.0 - 2.0 * math.tanh(1.0))
    )
    total = run_named(module, "total", (v,)).value
    lse = math.log(sum(math.exp(x) for x in v))
    assert total == pytest.approx(0.0 + 1.0 - lse + 0.0)
    weights = run_named(module, "weights", (v,)).value
    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] == pytest.approx(math.exp(1.0) / sum(math.exp(x) for x in v))
    assert run_named(module, "squares").value == (0.0, 1.0, 4.0)
    assert run_named(module, "pick", (v, 2)).value == pytest.approx(-0.5)
    assert run_named(module, "table").value == ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))
    assert module_to_source(parse(source)) == source


def test_tensor_expressions_are_rejected_at_the_wrong_shapes() -> None:
    with pytest.raises(QiecDiagnosticError, match="differing shapes"):
        lower_qvr_to_qiec(
            parse(
                "define bad(a : Tensor[Real]([2]), b : Tensor[Real]([3])) : "
                "Tensor[Real]([2]) !{} =\n    return a + b\n"
            ),
            file_path="bad.qvr",
        )
    with pytest.raises(QiecDiagnosticError, match="one Tensor"):
        lower_qvr_to_qiec(
            parse("define bad(a : Real) : Real !{} =\n    return softmax(a)\n"),
            file_path="bad.qvr",
        )
