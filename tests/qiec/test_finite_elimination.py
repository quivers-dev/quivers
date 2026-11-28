"""Exact finite elimination against a dense calculation in written order."""

from itertools import product
from typing import cast

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import QiecModule, dumps, loads
from quivers.qiec.finite_elimination import (
    FiniteAlgebra,
    FiniteAlgebraLawError,
    FiniteFactor,
    check_finite_algebra,
    eliminate_finite,
)
from quivers.qiec.laws import FiniteLawLimit, FiniteLawUnsupported, FiniteValue

_BOOLEAN_SOURCE = """
define add(a : Bool, b : Bool) : Bool !{} =
    if a then
        return true
    else
        return b

define multiply(a : Bool, b : Bool) : Bool !{} =
    if a then
        return b
    else
        return false
"""


def _compiled(source: str, name: str) -> QiecModule:
    compiler = Compiler(parse(source), module_name=name)
    module = compiler.qiec_module
    assert module is not None
    return module


def _algebra(module, zero, one, *, addition="add", multiplication="multiply"):
    ids = {item.name: item.id for item in module.computations}
    return FiniteAlgebra(ids[addition], ids[multiplication], zero, one)


@pytest.fixture(scope="module")
def boolean():
    module = _compiled(_BOOLEAN_SOURCE, "finite-boolean")
    return module, _algebra(module, False, True)


def _dense(factors, output, extents, add, multiply, zero, one):
    """Enumerate every complete assignment and fold all written factors."""
    scopes = tuple(factor.axes for factor in factors)
    tables = tuple(
        dict(
            zip(
                product(*(range(extents[axis]) for axis in axes)),
                factor.cells,
                strict=True,
            )
        )
        for axes, factor in zip(scopes, factors, strict=True)
    )
    variables = tuple(extents)
    totals = dict.fromkeys(product(*(range(extents[axis]) for axis in output)), zero)
    for point in product(*(range(extents[axis]) for axis in variables)):
        assignment = dict(zip(variables, point, strict=True))
        key = tuple(assignment[axis] for axis in output)
        value = one
        for axes, factor in zip(scopes, tables, strict=True):
            value = multiply(value, factor[tuple(assignment[axis] for axis in axes)])
        totals[key] = add(totals[key], value)
    return tuple(totals.values())


def test_commutative_plan_can_join_a_noncontiguous_bucket(boolean):
    module, algebra = boolean
    factors = (
        FiniteFactor(("x",), (True, False)),
        FiniteFactor(("y",), (False, True)),
        FiniteFactor(("x",), (True, True)),
    )
    extents = {"x": 2, "y": 2}
    result = eliminate_finite(module, algebra, factors, ("y",), extents)
    assert result.proof.noncommutative is None
    assert result.proof.values == 2 and result.proof.checks == 60
    assert result.steps[0].positions == (0, 2)
    assert result.steps[0].bag == ("x",)
    assert result.largest == 2
    assert (
        result.factor.cells
        == _dense(
            factors,
            ("y",),
            extents,
            lambda a, b: a or b,
            lambda a, b: a and b,
            False,
            True,
        )
        == (False, True)
    )


@pytest.mark.parametrize("length", (2, 3, 5, 9))
def test_commutative_chain_matches_complete_dense_assignment_table(boolean, length):
    module, algebra = boolean
    variables = tuple(f"variable_{index}" for index in range(length + 1))
    factors = tuple(
        FiniteFactor(
            (variables[index + 1], variables[index]),
            tuple(
                (a == b) if index % 3 else (a != b)
                for a, b in product(range(2), repeat=2)
            ),
        )
        for index in range(length)
    )
    output = variables[-1], variables[0]
    extents = dict.fromkeys(variables, 2)
    result = eliminate_finite(module, algebra, factors, output, extents)
    assert result.factor.axes == output
    assert result.factor.cells == _dense(
        factors,
        output,
        extents,
        lambda a, b: a or b,
        lambda a, b: a and b,
        False,
        True,
    )
    assert result.largest <= 8
    assert len(result.steps) == length - 1


_MATRIX = "(Bool * Bool * Bool * Bool)"
_MATRIX_CASES = tuple((a, b) for a in range(2) for b in range(2))
_MATRIX_ADDITION = "\n".join(
    f"    let c{i} <- any_bit(a[{i}], b[{i}])" for i in range(4)
)
_MATRIX_PRODUCT = "\n".join(
    f"    let left{k} <- all_bits(a[{2 * i}], b[{j}])\n"
    f"    let right{k} <- all_bits(a[{2 * i + 1}], b[{2 + j}])\n"
    f"    let c{k} <- any_bit(left{k}, right{k})"
    for k, (i, j) in enumerate(_MATRIX_CASES)
)
_BIT_OPERATIONS = _BOOLEAN_SOURCE.replace("define add(", "define any_bit(").replace(
    "define multiply(", "define all_bits("
)
_MATRIX_SOURCE = (
    _BIT_OPERATIONS
    + f"""
define add(a : {_MATRIX}, b : {_MATRIX}) : {_MATRIX} !{{}} =
{_MATRIX_ADDITION}
    return (c0, c1, c2, c3)

define multiply(a : {_MATRIX}, b : {_MATRIX}) : {_MATRIX} !{{}} =
{_MATRIX_PRODUCT}
    return (c0, c1, c2, c3)
"""
)
_EMPTY = (False, False, False, False)
_IDENTITY = (True, False, False, True)
_E11 = (True, False, False, False)
_E12 = (False, True, False, False)
_E21 = (False, False, True, False)
_E22 = (False, False, False, True)


def _union(left, right):
    return tuple(a or b for a, b in zip(left, right, strict=True))


def _relation_product(left, right):
    left_edges = {
        edge for edge, active in zip(_MATRIX_CASES, left, strict=True) if active
    }
    right_edges = {
        edge for edge, active in zip(_MATRIX_CASES, right, strict=True) if active
    }
    composition = {
        (source, target)
        for source, joint in left_edges
        for other, target in right_edges
        if joint == other
    }
    return tuple(edge in composition for edge in _MATRIX_CASES)


@pytest.fixture(scope="module")
def matrix():
    module = _compiled(_MATRIX_SOURCE, "finite-boolean-matrices")
    return module, _algebra(module, _EMPTY, _IDENTITY)


def test_noncommutative_product_keeps_its_order_with_an_intervening_factor(matrix):
    module, algebra = matrix
    factors = (
        FiniteFactor((), (_E21,)),
        FiniteFactor(("x",), (_E12, _EMPTY)),
        FiniteFactor(("y",), (_E21, _E11)),
        FiniteFactor(("x",), (_E11, _EMPTY)),
        FiniteFactor((), (_E12,)),
    )
    extents = {"y": 2, "x": 2}
    result = eliminate_finite(
        module, algebra, factors, ("y",), extents, max_cells=4, max_checks=17_504
    )
    assert result.proof.checks == 17_504
    witness = result.proof.noncommutative
    assert witness is not None
    left, right = witness.arguments
    assert witness.left == _relation_product(left, right)
    assert witness.right == _relation_product(right, left)
    assert witness.left != witness.right
    assert result.steps[0].positions == (1, 2, 3)
    assert result.largest == 4
    assert (
        result.factor.cells
        == _dense(
            factors,
            ("y",),
            extents,
            _union,
            _relation_product,
            _EMPTY,
            _IDENTITY,
        )
        == (_E22, _EMPTY)
    )
    wrong = factors[:2] + factors[3:4] + factors[2:3] + factors[4:]
    assert _dense(
        wrong, ("y",), extents, _union, _relation_product, _EMPTY, _IDENTITY
    ) == (_EMPTY, _EMPTY)
    with pytest.raises(FiniteLawLimit, match="bag needs 4 cells"):
        eliminate_finite(module, algebra, factors, ("y",), extents, max_cells=3)


def _bits(index):
    return bool(index & 2), bool(index & 1)


def _table(name, rows):
    """Emit an ordered branch table for two binary-pair arguments."""

    def branches(inputs, values, indent):
        if not inputs:
            pair = ", ".join("true" if bit else "false" for bit in _bits(values[0]))
            return f"{indent}return ({pair})"
        midpoint = len(values) // 2
        yes = branches(inputs[1:], values[midpoint:], indent + "    ")
        no = branches(inputs[1:], values[:midpoint], indent + "    ")
        return f"{indent}if {inputs[0]} then\n{yes}\n{indent}else\n{no}"

    carrier = "(Bool * Bool)"
    header = f"define {name}(a : {carrier}, b : {carrier}) : {carrier} !{{}} =\n"
    inputs = ("a[0]", "a[1]", "b[0]", "b[1]")
    return header + branches(inputs, [entry for row in rows for entry in row], "    ")


_DOMAIN = range(4)
_RIGHT_DISTRIBUTIVE = (
    (0, 0, 0, 0),
    (0, 1, 2, 3),
    (0, 2, 2, 3),
    (0, 3, 0, 0),
)
_XOR = tuple(tuple(a ^ b for b in _DOMAIN) for a in _DOMAIN)
_LEFT_BIASED_SUM = tuple(tuple(a if a else b for b in _DOMAIN) for a in _DOMAIN)
_MINIMUM = tuple(tuple(min(a, b) for b in _DOMAIN) for a in _DOMAIN)


@pytest.mark.parametrize("direction", ("left", "right"))
def test_one_sided_distribution_law_cannot_authorize_elimination(direction):
    multiplication = _RIGHT_DISTRIBUTIVE
    if direction == "right":
        multiplication = tuple(zip(*multiplication, strict=True))
    source = _table("add", _XOR) + "\n\n" + _table("multiply", multiplication)
    module = _compiled(source, f"one-sided-{direction}")
    algebra = _algebra(module, _bits(0), _bits(1))
    pairs = {
        "left": [
            (
                a,
                b,
                c,
                multiplication[a][b ^ c],
                multiplication[a][b] ^ multiplication[a][c],
            )
            for a, b, c in product(_DOMAIN, repeat=3)
        ],
        "right": [
            (
                a,
                b,
                c,
                multiplication[a ^ b][c],
                multiplication[a][c] ^ multiplication[b][c],
            )
            for a, b, c in product(_DOMAIN, repeat=3)
        ],
    }
    other = "right" if direction == "left" else "left"
    assert all(lhs == rhs for *_, lhs, rhs in pairs[other])
    a, b, c, lhs, rhs = next(item for item in pairs[direction] if item[-2] != item[-1])
    factors = (FiniteFactor(("x",), (_bits(1), _bits(2))),)
    with pytest.raises(
        FiniteAlgebraLawError, match=f"{direction} distributivity"
    ) as exc:
        eliminate_finite(module, algebra, factors, (), {"x": 2})
    witness = exc.value.counterexample
    assert witness.arguments == tuple(map(_bits, (a, b, c)))
    assert (witness.left, witness.right) == (_bits(lhs), _bits(rhs))


def test_noncommutative_addition_is_refused_with_its_exact_witness():
    source = _table("add", _LEFT_BIASED_SUM) + "\n\n" + _table("multiply", _MINIMUM)
    module = _compiled(source, "finite-ordered-addition")
    algebra = _algebra(module, _bits(0), _bits(3))
    with pytest.raises(FiniteAlgebraLawError, match="additive commutativity") as exc:
        check_finite_algebra(module, algebra)
    witness = exc.value.counterexample
    assert witness.arguments == (_bits(1), _bits(2))
    assert (witness.left, witness.right) == (_bits(1), _bits(2))


def test_proof_budget_is_exact_and_scope_is_rechecked(boolean):
    module, algebra = boolean
    with pytest.raises(FiniteLawLimit, match="needs 60 checks, exceeding 59"):
        check_finite_algebra(module, algebra, max_checks=59)
    with pytest.raises(FiniteLawLimit, match="fuel exhausted"):
        check_finite_algebra(module, algebra, fuel=1)
    proof = check_finite_algebra(module, algebra, max_checks=60)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert check_finite_algebra(restored, algebra) == proof
    source_add, source_product = module.computations
    changed = module.with_(
        computations=(
            source_add.with_(body=source_product.body),
            source_product,
        )
    )
    with pytest.raises(FiniteAlgebraLawError, match="additive identity"):
        check_finite_algebra(changed, algebra)


def test_empty_sums_and_empty_outputs_do_not_evade_the_cell_bound(boolean):
    module, algebra = boolean
    factors = (
        FiniteFactor(("empty_left", "row"), ()),
        FiniteFactor(("row", "empty_right"), ()),
    )
    extents = {"empty_left": 0, "row": 3, "empty_right": 0}
    with pytest.raises(FiniteLawLimit, match="needs 3 cells"):
        eliminate_finite(module, algebra, factors, (), extents, max_cells=2)
    contracted = eliminate_finite(module, algebra, factors, (), extents, max_cells=3)
    assert contracted.factor == FiniteFactor((), (False,))
    assert contracted.largest == 3
    assert all(step.cells == 0 for step in contracted.steps)
    empty_output = eliminate_finite(
        module,
        algebra,
        (FiniteFactor(("empty", "row"), ()),),
        ("row", "empty"),
        {"empty": 0, "row": 10**30},
        max_cells=1,
    )
    assert empty_output.factor == FiniteFactor(("row", "empty"), ())


def test_factors_and_identities_are_type_checked_without_coercing_booleans(boolean):
    module, algebra = boolean
    with pytest.raises(FiniteLawUnsupported, match="outside its finite type"):
        eliminate_finite(
            module,
            algebra,
            (FiniteFactor(("x",), (cast(FiniteValue, 0),)),),
            (),
            {"x": 1},
        )
    with pytest.raises(ValueError, match="do not match"):
        eliminate_finite(
            module, algebra, (FiniteFactor(("x",), (False,)),), (), {"x": 2}
        )
    with pytest.raises(ValueError, match="repeat an axis"):
        eliminate_finite(
            module, algebra, (FiniteFactor(("x", "x"), (False,)),), (), {"x": 1}
        )
    wrong = FiniteAlgebra(algebra.addition, algebra.multiplication, True, True)
    with pytest.raises(FiniteAlgebraLawError, match="additive identity") as exc:
        check_finite_algebra(module, wrong)
    assert exc.value.counterexample.arguments == (False,)
    assert exc.value.counterexample.left is True
