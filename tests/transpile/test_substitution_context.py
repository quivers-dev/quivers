"""Context-aware substitution tests for
[`_substitute_let_expr`][quivers.transpile.renderers._stan_helpers._substitute_let_expr].

The helper distinguishes two replacement slots: the `index_value`
substituted into the `indices` tuple of every
[`LetExprIndex`][quivers.dsl.ast_nodes.LetExprIndex] (the
``[ ]``-bracketed positions), and the `scalar_value` substituted
everywhere else. Stan needs the two values to differ by one
(Stan arrays are 1-based, QVR arithmetic on factor binders is
0-based); Python / NumPyro / Pyro / PyMC / Edward2 / JavaScript /
WebPPL pass the same value for both slots because their host
arrays are already 0-based.

These tests exercise the helper directly at the AST level rather
than through a full renderer; the substituted tree is structurally
equal to the hand-built expected tree.
"""

from __future__ import annotations

import pytest

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprBool,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprLiteral,
    LetExprNode,
    LetExprTuple,
    LetExprUnaryOp,
    LetExprUnit,
    LetExprVar,
    LetFactorBinder,
    QiecConstructorValue,
    QiecTypeName,
    TypeName,
)
from quivers.dsl.let_expr_traversal import free_let_names, map_let_expr
from quivers.transpile.ir import LetAffineSource, LetExprAffineMap
from quivers.transpile.ir import DimStatic, IRArgNumber, Plate
from quivers.transpile.renderers._base import _substitute_latent_expr
from quivers.transpile.renderers._bugs_helpers import (
    _substitute as bugs_substitute,
    collect_letexpr_vars,
    index_letexpr_refs,
    inline_letexpr,
)
from quivers.transpile.renderers._julia_helpers import _substitute as julia_substitute
from quivers.transpile.renderers.gen import _rename_let_expr
from quivers.transpile.renderers._let_expr_traversal import AFFINE_EXPRESSION
from quivers.transpile.renderers._stan_helpers import (
    _substitute_let_expr,
)


def _stan_substitute(expr: LetExprNode, name: str, value: int) -> LetExprNode:
    """Stan call-site convention: 1-indexed in index slots, 0-indexed
    in scalar slots (matches `_build_nested_array` in `_stan_helpers`)."""
    return _substitute_let_expr(
        expr,
        name,
        index_value=LetExprLiteral(value=value + 1),
        scalar_value=LetExprLiteral(value=value),
    )


def _python_substitute(expr: LetExprNode, name: str, value: int) -> LetExprNode:
    """Python / NumPyro / Pyro / PyMC / Edward2 call-site convention:
    same 0-indexed value in both slots (matches `_build_nested_python`
    in `_python_helpers` and `_build_nested_array` in
    `_javascript_helpers`)."""
    literal = LetExprLiteral(value=value)
    return _substitute_let_expr(
        expr,
        name,
        index_value=literal,
        scalar_value=literal,
    )


def test_index_slot_only_stan_picks_one_based() -> None:
    """`delta[v]` with `v=0` substitutes to `delta[1]` in Stan (the
    binder appears in an index slot only, so the Stan call site's
    1-indexed `index_value` wins)."""
    expr = LetExprIndex(
        array=LetExprVar(name="delta"),
        indices=(LetExprVar(name="v"),),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    assert isinstance(substituted.array, LetExprVar)
    assert substituted.array.name == "delta"
    assert len(substituted.indices) == 1
    idx = substituted.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 1


def test_index_slot_only_python_picks_zero_based() -> None:
    """`delta[v]` with `v=0` substitutes to `delta[0]` in Python /
    NumPyro / Pyro / PyMC / Edward2 (the binder appears in an index
    slot but the call site supplies the 0-indexed value for both
    slots, since the host arrays are already 0-based)."""
    expr = LetExprIndex(
        array=LetExprVar(name="delta"),
        indices=(LetExprVar(name="v"),),
    )
    substituted = _python_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    assert len(substituted.indices) == 1
    idx = substituted.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 0


def test_scalar_slot_only_stan_picks_zero_based() -> None:
    """`2 * v + 1` with `v=0` substitutes to `2 * 0 + 1` in Stan
    (every binder occurrence is in scalar position, so the
    0-indexed `scalar_value` wins and the expression evaluates to
    `1` rather than the `3` a context-blind substitution would
    produce)."""
    expr = LetExprBinOp(
        op="+",
        left=LetExprBinOp(
            op="*",
            left=LetExprLiteral(value=2),
            right=LetExprVar(name="v"),
        ),
        right=LetExprLiteral(value=1),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprBinOp)
    assert substituted.op == "+"
    left = substituted.left
    assert isinstance(left, LetExprBinOp)
    assert left.op == "*"
    assert isinstance(left.left, LetExprLiteral)
    assert left.left.value == 2
    assert isinstance(left.right, LetExprLiteral)
    assert left.right.value == 0
    assert isinstance(substituted.right, LetExprLiteral)
    assert substituted.right.value == 1


def test_mixed_slots_in_one_expression_stan() -> None:
    """`delta[v] * v` with `v=0` substitutes to `delta[1] * 0` in
    Stan: the binder occurrence under `LetExprIndex.indices` uses
    the 1-indexed `index_value`, the bare occurrence on the right
    of the multiply uses the 0-indexed `scalar_value`."""
    expr = LetExprBinOp(
        op="*",
        left=LetExprIndex(
            array=LetExprVar(name="delta"),
            indices=(LetExprVar(name="v"),),
        ),
        right=LetExprVar(name="v"),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprBinOp)
    assert substituted.op == "*"
    left = substituted.left
    assert isinstance(left, LetExprIndex)
    assert isinstance(left.array, LetExprVar)
    assert left.array.name == "delta"
    idx = left.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 1
    right = substituted.right
    assert isinstance(right, LetExprLiteral)
    assert right.value == 0


def test_mixed_slots_in_one_expression_python() -> None:
    """`delta[v] * v` with `v=0` substitutes to `delta[0] * 0` in
    Python: both occurrences resolve to 0 because the call site
    passes the same `LetExprLiteral(value=0)` for both slots."""
    expr = LetExprBinOp(
        op="*",
        left=LetExprIndex(
            array=LetExprVar(name="delta"),
            indices=(LetExprVar(name="v"),),
        ),
        right=LetExprVar(name="v"),
    )
    substituted = _python_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprBinOp)
    left = substituted.left
    assert isinstance(left, LetExprIndex)
    idx = left.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 0
    right = substituted.right
    assert isinstance(right, LetExprLiteral)
    assert right.value == 0


def test_array_child_of_indexed_is_scalar_slot_stan() -> None:
    """The `array` child of a `LetExprIndex` is a scalar slot, not
    an index slot. A pathological body `v[v]` with `v=0` substitutes
    to `0[1]` in Stan: the outer `v` is the array being indexed
    (scalar), the inner `v` is the index (1-based)."""
    expr = LetExprIndex(
        array=LetExprVar(name="v"),
        indices=(LetExprVar(name="v"),),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    array = substituted.array
    assert isinstance(array, LetExprLiteral)
    assert array.value == 0
    idx = substituted.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 1


def test_nested_indices_inner_index_is_index_slot_stan() -> None:
    """`arr[subj[v]]` with `v=0` substitutes to `arr[subj[1]]` in
        Stan. The inner `LetExprIndex` lives inside the outer node's
        `indices` tuple, but the helper resets `in_index_slot` on
        descent into the inner's `array` child and re-sets it on
        descent into the inner's `indices` tuple. The bare `v` ends
    up in `subj[...]`'s index slot and thus picks the
        1-indexed value."""
    expr = LetExprIndex(
        array=LetExprVar(name="arr"),
        indices=(
            LetExprIndex(
                array=LetExprVar(name="subj"),
                indices=(LetExprVar(name="v"),),
            ),
        ),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    inner = substituted.indices[0]
    assert isinstance(inner, LetExprIndex)
    assert isinstance(inner.array, LetExprVar)
    assert inner.array.name == "subj"
    inner_idx = inner.indices[0]
    assert isinstance(inner_idx, LetExprLiteral)
    assert inner_idx.value == 1


def test_unrelated_var_unchanged() -> None:
    """A `LetExprVar` whose name differs from `name` passes through
    untouched regardless of slot context."""
    expr = LetExprIndex(
        array=LetExprVar(name="arr"),
        indices=(LetExprVar(name="k"),),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    assert isinstance(substituted.array, LetExprVar)
    assert substituted.array.name == "arr"
    idx = substituted.indices[0]
    assert isinstance(idx, LetExprVar)
    assert idx.name == "k"


def test_unary_operand_is_scalar_slot() -> None:
    """The operand of a `LetExprUnaryOp` is a scalar slot; `-v`
    with `v=0` becomes `-0` under the Stan call-site convention."""
    expr = LetExprUnaryOp(operand=LetExprVar(name="v"))
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprUnaryOp)
    operand = substituted.operand
    assert isinstance(operand, LetExprLiteral)
    assert operand.value == 0


def test_call_args_are_scalar_slots() -> None:
    """Arguments to a `LetExprCall` are scalar slots; `log(v + 1)`
    with `v=0` becomes `log(0 + 1)` under the Stan call-site
    convention."""
    expr = LetExprCall(
        func="log",
        args=(
            LetExprBinOp(
                op="+",
                left=LetExprVar(name="v"),
                right=LetExprLiteral(value=1),
            ),
        ),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprCall)
    assert substituted.func == "log"
    inner = substituted.args[0]
    assert isinstance(inner, LetExprBinOp)
    assert isinstance(inner.left, LetExprLiteral)
    assert inner.left.value == 0


def test_boolean_negation_and_authored_locations_survive_substitution() -> None:
    expression = LetExprUnaryOp(
        op="not", operand=LetExprVar(name="flag"), line=7, col=13
    )
    result = _substitute_let_expr(
        expression,
        "flag",
        index_value=LetExprBool(value=False),
        scalar_value=LetExprBool(value=False),
    )
    assert result == expression.with_(operand=LetExprBool(value=False))


@pytest.mark.parametrize(
    "expression,expected",
    [
        (LetExprBool(value=True), LetExprBool(value=True)),
        (LetExprUnit(), LetExprUnit()),
        (
            LetExprTuple(items=(LetExprVar(name="v"), LetExprUnit())),
            LetExprTuple(items=(LetExprLiteral(value=0), LetExprUnit())),
        ),
        (
            QiecConstructorValue(
                constructor="Box",
                fields=(LetExprVar(name="v"),),
                result_type=QiecTypeName(name="Box"),
                line=8,
                col=4,
            ),
            QiecConstructorValue(
                constructor="Box",
                fields=(LetExprLiteral(value=0),),
                result_type=QiecTypeName(name="Box"),
                line=8,
                col=4,
            ),
        ),
    ],
)
def test_shared_expression_variants_reach_renderers_without_a_second_case_table(
    expression, expected
) -> None:
    assert _python_substitute(expression, "v", 0) == expected


@pytest.mark.parametrize("binder_kind", ["lambda", "factor"])
def test_replacement_remains_free_under_a_capturing_renderer_binder(binder_kind):
    body = LetExprTuple(items=(LetExprVar(name="v"), LetExprVar(name="j")))
    expression = (
        LetExprLambda(param="j", body=body)
        if binder_kind == "lambda"
        else LetExprFactor(
            binders=(LetFactorBinder(var="j", index=TypeName(name="Row")),),
            body=body,
        )
    )
    replacement = LetExprVar(name="j")
    result = _substitute_let_expr(
        expression, "v", index_value=replacement, scalar_value=replacement
    )
    assert free_let_names(result) == ("j",)
    assert result.body == LetExprTuple(
        items=(LetExprVar(name="j"), LetExprVar(name="j_1"))
    )
    assert (result.param if binder_kind == "lambda" else result.binders[0].var) == "j_1"
    # An explicitly shadowing binder remains untouched, including index slots.
    assert (
        _substitute_let_expr(
            expression, "j", index_value=replacement, scalar_value=replacement
        )
        == expression
    )


def test_index_replacement_cannot_be_captured_by_an_enclosing_binder():
    expression = LetExprLambda(
        param="j",
        body=LetExprIndex(array=LetExprVar(name="a"), indices=(LetExprVar(name="i"),)),
    )
    result = _substitute_let_expr(
        expression,
        "i",
        index_value=LetExprVar(name="j"),
        scalar_value=LetExprLiteral(value=0),
    )
    assert result.param == "j_1"
    assert result.body.indices == (LetExprVar(name="j"),)
    assert free_let_names(result) == ("a", "j")


def test_affine_extension_preserves_layout_and_uses_canonical_capture_avoidance():
    expression = LetExprAffineMap(
        weight=LetExprIndex(
            array=LetExprVar(name="w"), indices=(LetExprVar(name="v"),)
        ),
        bias=LetExprVar(name="v"),
        sources=(
            LetAffineSource(
                value=LetExprLambda(param="j", body=LetExprVar(name="v")), width=3
            ),
        ),
        row_offset=2,
        rows=4,
        transform="exp",
    )
    result = _substitute_let_expr(
        expression,
        "v",
        index_value=LetExprLiteral(value=1),
        scalar_value=LetExprVar(name="j"),
    )
    assert isinstance(result, LetExprAffineMap)
    assert result.weight.indices == (LetExprLiteral(value=1),)
    assert result.bias == LetExprVar(name="j")
    assert result.sources[0].value.param == "j_1"
    assert result.sources[0].value.body == LetExprVar(name="j")
    assert (result.row_offset, result.rows, result.transform) == (2, 4, "exp")
    assert result.sources[0].width == 3
    assert free_let_names(result, extension=AFFINE_EXPRESSION) == ("w", "j")
    assert (
        map_let_expr(result, lambda node: node, extension=AFFINE_EXPRESSION) == result
    )


@pytest.mark.parametrize(
    "rewrite",
    [
        lambda expr: julia_substitute(expr, "x", LetExprVar(name="y")),
        lambda expr: inline_letexpr(expr, {"x": LetExprVar(name="y")}),
        lambda expr: _rename_let_expr(expr, {"x": "y"}),
    ],
)
def test_backend_adapters_share_capture_avoidance_and_keep_not(rewrite):
    expr = LetExprLambda(
        param="y",
        body=LetExprTuple(
            items=(
                LetExprUnaryOp(op="not", operand=LetExprVar(name="x"), line=17, col=9),
                LetExprVar(name="y"),
                LetExprUnit(),
            )
        ),
    )
    actual = rewrite(expr)
    assert actual.param == "y_1"
    assert actual.body.items[0].op == "not"
    assert actual.body.items[0].operand == LetExprVar(name="y")
    assert (actual.body.items[0].line, actual.body.items[0].col) == (17, 9)
    assert actual.body.items[1] == LetExprVar(name="y_1")
    assert free_let_names(actual) == ("y",)


def test_integer_factor_substitution_preserves_shadowing_and_large_integers():
    integer = 2**53 + 1
    expr = LetExprTuple(
        items=(
            LetExprVar(name="x"),
            LetExprLambda(param="x", body=LetExprVar(name="x")),
        )
    )
    actual = bugs_substitute(expr, {"x": integer})
    assert actual.items[0].value == integer
    assert isinstance(actual.items[0].value, int)
    assert actual.items[1] == expr.items[1]


def test_graph_free_name_collection_handles_all_constructors_and_bound_names():
    expr = LetExprLambda(
        param="bound",
        body=LetExprTuple(
            items=(
                LetExprVar(name="bound"),
                LetExprVar(name="outside"),
                QiecConstructorValue(
                    constructor="Some",
                    fields=(LetExprVar(name="nested"),),
                    result_type=QiecTypeName(name="Option"),
                ),
                LetExprBool(value=True),
                LetExprUnit(),
            )
        ),
    )
    assert collect_letexpr_vars(expr) == frozenset({"outside", "nested"})


def test_marginal_atom_substitution_keeps_not_and_simplifies_only_free_selection():
    expr = LetExprTuple(
        items=(
            LetExprUnaryOp(
                op="not",
                operand=LetExprCall(
                    func="equals", args=(LetExprVar(name="z"), LetExprLiteral(value=0))
                ),
                line=8,
            ),
            LetExprCall(
                func="select",
                args=(
                    LetExprVar(name="z"),
                    LetExprLiteral(value=7),
                    LetExprLiteral(value=9),
                ),
            ),
            LetExprLambda(
                param="z",
                body=LetExprCall(
                    func="select",
                    args=(
                        LetExprVar(name="z"),
                        LetExprLiteral(value=7),
                        LetExprLiteral(value=9),
                    ),
                ),
            ),
        )
    )
    actual = _substitute_latent_expr(expr, "z", IRArgNumber(value=0))
    assert actual.items[0].op == "not"
    assert actual.items[0].line == 8
    assert actual.items[0].operand.args[0].value == 0
    assert actual.items[1].value == 9
    assert actual.items[2] == expr.items[2]


def test_graph_loop_indexing_preserves_not_tuple_and_gather_origins():
    row = DimStatic(name="Row", size=3)
    category = DimStatic(name="Category", size=2)
    expr = LetExprTuple(
        items=(
            LetExprUnaryOp(op="not", operand=LetExprVar(name="flag"), line=12),
            LetExprIndex(
                array=LetExprVar(name="table"),
                indices=(LetExprVar(name="group_idx"),),
                line=13,
            ),
        )
    )
    plates = {
        "flag": Plate(batch_dims=(row, category), event_dims=()),
        "group_idx": Plate(batch_dims=(row,), event_dims=()),
        "table": Plate(
            batch_dims=(DimStatic(name="Group", size=4), category), event_dims=()
        ),
    }
    actual = index_letexpr_refs(
        expr, plates, Plate(batch_dims=(row, category), event_dims=()), ("i", "k")
    )
    assert actual.items[0].op == "not"
    assert actual.items[0].line == 12
    assert actual.items[0].operand.indices == (
        LetExprVar(name="i"),
        LetExprVar(name="k"),
    )
    gather = actual.items[1]
    assert gather.array == LetExprVar(name="table")
    assert gather.line == 13
    assert gather.indices == (
        LetExprIndex(
            array=LetExprVar(name="group_idx"), indices=(LetExprVar(name="i"),)
        ),
        LetExprVar(name="k"),
    )
