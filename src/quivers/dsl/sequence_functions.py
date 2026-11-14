"""Checked elaboration of learned functions over whole sequences.

A ``[role=function]`` morphism whose ``param_source`` is
``self_attention`` or ``positions(N)`` maps a sequence of rows to a
sequence of rows: a program calls it in a ``let`` on a table
``Tensor[Real]([T, d])``, the sequence a scan with ``outputs=all`` binds
or a host datum, and the value is the table ``Tensor[Real]([T, m])``. The
position extent ``T`` may be open, so the elaboration builds the map from
comprehensions over it rather than from ``factor`` binders over declared
objects.

Self-attention with ``h`` heads of width ``e = m / h`` elaborates to

* the query, key, and value tables, one affine map per row:
  ``q[i] = Wq x[i] + bq``, and likewise ``k`` and ``v``;
* per head ``g`` and row ``i``, the scores
  ``s[j] = (1 / sqrt(e)) * sum_u q[i, g e + u] k[j, g e + u]`` and the
  attention weights, ``softmax(s)`` or, when causal, the weights
  ``w[j] = [j <= i] exp(s[j] - max_{j' <= i} s[j'])`` normalized to sum
  to one, which equal the softmax over the positions ``j <= i`` and are
  zero after ``i``;
* the context ``c[i, col] = sum_j a[col div e, i, j] v[j, col]``;
* the output projection ``Wo c[i] + bo`` at every row.

A position table elaborates to ``x[i, k] + P[i, k]``. The weights are
program inputs with roles ``weight`` and ``bias``, named after the
function as a learned function's are, so the reference machine runs a
call with the compiled program's weights supplied as data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quivers.continuous.param_source import SEQUENCE_SOURCE_KINDS
from quivers.dsl.ast_nodes import MorphismDecl, OptionCall, OptionNumber
from quivers.dsl.ast_nodes.let_expressions import LetExprCall, LetExprVar
from quivers.qiec.canonical import tensor_shape, tensor_type
from quivers.qiec.terms import (
    AffineMap,
    Comprehension,
    Gather,
    LetBinding,
    LetValue,
    LiteralValue,
    Local,
    Reduction,
    Rowwise,
    Value,
    Var,
)
from quivers.qiec.kinds import NAT
from quivers.qiec.types import INT, REAL, IndexLiteral, IndexTerm, TypeExpr

if TYPE_CHECKING:
    from quivers.dsl.program_elaboration import _ProgramState, _Scope
    from quivers.dsl.qiec_lowering import _Elaborator


def sequence_source_kind(morphism: MorphismDecl) -> str | None:
    """The sequence network a learned function's ``param_source`` names.

    Parameters
    ----------
    morphism : MorphismDecl
        The ``[role=function]`` declaration.

    Returns
    -------
    str | None
        ``self_attention`` or ``positions``, else ``None``.
    """
    for entry in morphism.options:
        if entry.key != "param_source":
            continue
        value = entry.value
        kind = value.func if isinstance(value, OptionCall) else None
        if kind is None:
            kind = str(getattr(value, "value", ""))
        return kind if kind in SEQUENCE_SOURCE_KINDS else None
    return None


def _option(morphism: MorphismDecl, key: str) -> str | None:
    """The text of a name- or number-valued option, when given.

    Parameters
    ----------
    morphism : MorphismDecl
        The declaration.
    key : str
        The option's key.

    Returns
    -------
    str | None
        The name, the number as an integer's text when integral, or
        ``None`` when the option is absent.
    """
    for entry in morphism.options:
        if entry.key != key:
            continue
        value = entry.value
        if isinstance(value, OptionNumber):
            number = float(value.value)
            return str(int(number)) if number.is_integer() else str(number)
        return str(getattr(value, "value", ""))
    return None


def _position_count(morphism: MorphismDecl) -> int | None:
    """The number of positions a ``positions(N)`` table holds.

    Parameters
    ----------
    morphism : MorphismDecl
        The declaration.

    Returns
    -------
    int | None
        ``N``, or ``None`` when the source names no single integral
        count (which the compiler refuses at the declaration).
    """
    for entry in morphism.options:
        if entry.key == "param_source" and isinstance(entry.value, OptionCall):
            args = entry.value.args
            if len(args) == 1 and isinstance(args[0], OptionNumber):
                number = float(args[0].value)
                if number.is_integer() and number >= 1:
                    return int(number)
    return None


class _SequenceFunctionElaboration:
    """Elaborate calls of learned functions that read whole sequences."""

    def _sequence_function_call(
        self: _Elaborator,
        morphism: MorphismDecl,
        kind: str,
        call: LetExprCall,
        width: int,
        out_width: int,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> tuple[Value, TypeExpr]:
        """Elaborate one call of a sequence function to its network.

        Parameters
        ----------
        morphism : MorphismDecl
            The ``[role=function]`` declaration.
        kind : str
            ``self_attention`` or ``positions``.
        call : LetExprCall
            The call, its one argument free of learned-function calls.
        width : int
            The width ``d`` of a row of the domain.
        out_width : int
            The width ``m`` of a row of the codomain.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        tuple[Value, TypeExpr]
            The image and its type, ``Tensor[Real]([T, m])`` for an
            argument ``Tensor[Real]([T, d])``.

        Raises
        ------
        QiecDiagnosticError
            If the call passes other than one argument, the argument is
            not a table of ``d``-wide rows, or the declaration's options
            do not describe the network.
        """
        name = call.func
        if len(call.args) != 1:
            self._fail(
                node,
                f"the sequence function {name!r} reads one sequence, but is "
                f"called with {len(call.args)} arguments",
                code="qiec-program",
            )
        (argument,) = call.args
        if (
            isinstance(argument, LetExprVar)
            and scope.lookup(argument.name) is None
            and argument.name not in self._lambda_macros
        ):
            # A host datum passed straight to the function is a sequence
            # of rows of the domain's width, over an open extent.
            self._declare_parameter(
                argument.name,
                tensor_type(
                    REAL,
                    (
                        self._open_extent(f"{argument.name}_extent", state),
                        IndexLiteral(value=width, sort=NAT),
                    ),
                ),
                "data",
                scope,
                state,
                node,
            )
        value = self._let_value(argument, scope, state, node)
        type_ = self._value_type(value, scope.context(), argument)
        shape = tensor_shape(type_)
        if (
            shape is None
            or shape[0] != REAL
            or len(shape[1]) != 2
            or shape[1][1] != IndexLiteral(value=width, sort=NAT)
        ):
            self._fail(
                node,
                f"the sequence function {name!r} reads a sequence of "
                f"{width}-wide real rows, Tensor[Real]([T, {width}]), but its "
                f"argument has type {self._render(type_)}; a sequence is the "
                f"every-state binding of a scan with outputs=all, a let over "
                f"one, or a host table",
                code="qiec-program",
            )
        length = shape[1][0]
        serial = state.function_calls_cell[0]
        if kind == "positions":
            return self._position_table(
                morphism, value, length, width, serial, scope, state, node
            )
        return self._self_attention(
            morphism, value, length, width, out_width, serial, scope, state, node
        )

    def _position_table(
        self: _Elaborator,
        morphism: MorphismDecl,
        value: Value,
        length: IndexTerm,
        width: int,
        serial: int,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> tuple[Value, TypeExpr]:
        """Elaborate a position table: ``x[i, k] + P[i, k]``.

        Parameters
        ----------
        morphism : MorphismDecl
            The declaration.
        value : Value
            The sequence, ``Tensor[Real]([T, d])``.
        length : IndexTerm
            The sequence's extent ``T``.
        width : int
            The row width ``d``.
        serial : int
            The call's serial number, which keeps its binders apart from
            every other call's.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        tuple[Value, TypeExpr]
            The sum and its type, the sequence's own.

        Raises
        ------
        QiecDiagnosticError
            If the table's length is not a positive integer.
        """
        name = morphism.names[0]
        count = _position_count(morphism)
        if count is None:
            self._fail(
                node,
                f"the position table {name!r} names no length; declare it "
                f"param_source=positions(N)",
                code="qiec-program",
            )
        d = IndexLiteral(value=width, sort=NAT)
        row = tensor_type(REAL, (d,))
        table = self._declare_parameter(
            f"{name}_param_positions",
            tensor_type(REAL, (IndexLiteral(value=count, sort=NAT), d)),
            "weight",
            scope,
            state,
            node,
            axes=(f"{name}_position", f"{name}_param_col"),
        )
        prefix = f"{name}${serial}"
        result = tensor_type(REAL, (length, d))
        source = Local(name=f"{prefix}$x", type=result)
        i = Local(name=f"{prefix}$i", type=INT)
        k = Local(name=f"{prefix}$k", type=INT)
        path = ("programs", state.declaration.name, "functions", name, serial)
        entry = self._primitive(
            "add_real",
            (
                Gather(
                    value=Gather(
                        value=Var(local=source), index=Var(local=i), result_type=row
                    ),
                    index=Var(local=k),
                    result_type=REAL,
                ),
                Gather(
                    value=Gather(
                        value=Var(local=table), index=Var(local=i), result_type=row
                    ),
                    index=Var(local=k),
                    result_type=REAL,
                ),
            ),
            node,
            (*path, "add"),
        )
        image = Comprehension(
            binder=i,
            extent=length,
            body=Comprehension(binder=k, extent=d, body=entry, result_type=row),
            result_type=result,
        )
        return LetValue(
            bindings=(LetBinding(binder=source, value=value),),
            body=image,
            result_type=result,
        ), result

    def _self_attention(
        self: _Elaborator,
        morphism: MorphismDecl,
        value: Value,
        length: IndexTerm,
        width: int,
        out_width: int,
        serial: int,
        scope: _Scope,
        state: _ProgramState,
        node: object,
    ) -> tuple[Value, TypeExpr]:
        """Elaborate multi-head scaled dot-product self-attention.

        Parameters
        ----------
        morphism : MorphismDecl
            The declaration.
        value : Value
            The sequence, ``Tensor[Real]([T, d])``.
        length : IndexTerm
            The sequence's extent ``T``.
        width : int
            The input row width ``d``.
        out_width : int
            The model width ``m``.
        serial : int
            The call's serial number, which keeps its binders apart from
            every other call's.
        scope : _Scope
            The scope.
        state : _ProgramState
            The program's accumulating state.
        node : object
            The source node.

        Returns
        -------
        tuple[Value, TypeExpr]
            The attended sequence and its type ``Tensor[Real]([T, m])``.

        Raises
        ------
        QiecDiagnosticError
            If the head count does not divide the model width or the
            declaration does not say whether the attention is causal.
        """
        name = morphism.names[0]
        heads_text = _option(morphism, "heads")
        heads = 1 if heads_text is None else int(float(heads_text))
        causal_text = _option(morphism, "causal")
        if causal_text not in ("true", "false"):
            self._fail(
                node,
                f"the self-attention function {name!r} does not say whether it "
                f"is causal; declare causal=true or causal=false",
                code="qiec-program",
            )
        if heads < 1 or out_width % heads != 0:
            self._fail(
                node,
                f"the self-attention function {name!r} splits its model width "
                f"{out_width} into {heads} heads of equal width",
                code="qiec-program",
            )
        causal = causal_text == "true"
        head_width = out_width // heads
        d = IndexLiteral(value=width, sort=NAT)
        m = IndexLiteral(value=out_width, sort=NAT)
        e = IndexLiteral(value=head_width, sort=NAT)
        h = IndexLiteral(value=heads, sort=NAT)
        in_row = tensor_type(REAL, (d,))
        model_row = tensor_type(REAL, (m,))
        sequence = tensor_type(REAL, (length, m))
        position_row = tensor_type(REAL, (length,))
        weights_type = tensor_type(REAL, (h, length, length))
        path = ("programs", state.declaration.name, "functions", name, serial)

        def projection(role: str, fan_in: IndexLiteral) -> tuple[Local, Local]:
            weight = self._declare_parameter(
                f"{name}_param_{role}_weight",
                tensor_type(REAL, (m, fan_in)),
                "weight",
                scope,
                state,
                node,
                axes=(f"{name}_{role}_row", f"{name}_{role}_col"),
            )
            bias = self._declare_parameter(
                f"{name}_param_{role}_bias",
                tensor_type(REAL, (m,)),
                "bias",
                scope,
                state,
                node,
                axes=(f"{name}_{role}_row",),
            )
            return weight, bias

        prefix = f"{name}${serial}"
        source = Local(name=f"{prefix}$x", type=tensor_type(REAL, (length, d)))
        i = Local(name=f"{prefix}$i", type=INT)
        j = Local(name=f"{prefix}$j", type=INT)
        g = Local(name=f"{prefix}$g", type=INT)
        u = Local(name=f"{prefix}$u", type=INT)
        col = Local(name=f"{prefix}$col", type=INT)

        def prim(operator: str, *arguments: Value) -> Value:
            return self._primitive(operator, arguments, node, (*path, operator))

        def entry(table: Local, row: Value, column: Value, row_type: TypeExpr) -> Value:
            return Gather(
                value=Gather(value=Var(local=table), index=row, result_type=row_type),
                index=column,
                result_type=REAL,
            )

        def int_literal(number: int) -> Value:
            return LiteralValue(value=number, type=INT)

        def real_literal(number: float) -> Value:
            return LiteralValue(value=number, type=REAL)

        bindings: list[LetBinding] = [LetBinding(binder=source, value=value)]
        projected: dict[str, Local] = {}
        for role in ("query", "key", "value"):
            weight, bias = projection(role, d)
            local = Local(name=f"{prefix}${role}", type=sequence)
            bindings.append(
                LetBinding(
                    binder=local,
                    value=Comprehension(
                        binder=i,
                        extent=length,
                        body=AffineMap(
                            weight=Var(local=weight),
                            bias=Var(local=bias),
                            sources=(
                                Gather(
                                    value=Var(local=source),
                                    index=Var(local=i),
                                    result_type=in_row,
                                ),
                            ),
                            row_offset=0,
                            rows=out_width,
                            transform="identity",
                            result_type=model_row,
                        ),
                        result_type=sequence,
                    ),
                )
            )
            projected[role] = local
        # Coordinate ``u`` of head ``g`` is model coordinate ``g e + u``.
        coordinate = prim(
            "add_int",
            prim("mul_int", Var(local=g), int_literal(head_width)),
            Var(local=u),
        )
        score = prim(
            "mul_real",
            real_literal(1.0 / math.sqrt(head_width)),
            Reduction(
                operator="sum",
                value=Comprehension(
                    binder=u,
                    extent=e,
                    body=prim(
                        "mul_real",
                        entry(projected["query"], Var(local=i), coordinate, model_row),
                        entry(projected["key"], Var(local=j), coordinate, model_row),
                    ),
                    result_type=tensor_type(REAL, (e,)),
                ),
                result_type=REAL,
            ),
        )
        scores = Local(name=f"{prefix}$scores", type=position_row)
        scores_value = Comprehension(
            binder=j, extent=length, body=score, result_type=position_row
        )
        if causal:
            # The weights over the positions ``j <= i``: the row's peak is
            # taken over them (a later position reads the score at ``i``,
            # which is among them), and a later position's exponent is
            # zero before its indicator zeroes it, so no term overflows.
            peak = Local(name=f"{prefix}$peak", type=REAL)
            visible = prim("le_int", Var(local=j), Var(local=i))
            at_i = Gather(value=Var(local=scores), index=Var(local=i), result_type=REAL)
            at_j = Gather(value=Var(local=scores), index=Var(local=j), result_type=REAL)
            row_weights: Value = LetValue(
                bindings=(
                    LetBinding(binder=scores, value=scores_value),
                    LetBinding(
                        binder=peak,
                        value=Reduction(
                            operator="max",
                            value=Comprehension(
                                binder=j,
                                extent=length,
                                body=prim("select_real", visible, at_j, at_i),
                                result_type=position_row,
                            ),
                            result_type=REAL,
                        ),
                    ),
                ),
                body=Rowwise(
                    operator="normalize",
                    value=Comprehension(
                        binder=j,
                        extent=length,
                        body=prim(
                            "mul_real",
                            prim("bool_to_real", visible),
                            prim(
                                "exp",
                                prim(
                                    "select_real",
                                    visible,
                                    prim("sub_real", at_j, Var(local=peak)),
                                    real_literal(0.0),
                                ),
                            ),
                        ),
                        result_type=position_row,
                    ),
                    result_type=position_row,
                ),
                result_type=position_row,
            )
        else:
            row_weights = Rowwise(
                operator="softmax", value=scores_value, result_type=position_row
            )
        attention = Local(name=f"{prefix}$weights", type=weights_type)
        bindings.append(
            LetBinding(
                binder=attention,
                value=Comprehension(
                    binder=g,
                    extent=h,
                    body=Comprehension(
                        binder=i,
                        extent=length,
                        body=row_weights,
                        result_type=tensor_type(REAL, (length, length)),
                    ),
                    result_type=weights_type,
                ),
            )
        )
        # Model coordinate ``col`` belongs to head ``col div e``.
        head_of = prim("div_int", Var(local=col), int_literal(head_width))
        context = Local(name=f"{prefix}$context", type=sequence)
        bindings.append(
            LetBinding(
                binder=context,
                value=Comprehension(
                    binder=i,
                    extent=length,
                    body=Comprehension(
                        binder=col,
                        extent=m,
                        body=Reduction(
                            operator="sum",
                            value=Comprehension(
                                binder=j,
                                extent=length,
                                body=prim(
                                    "mul_real",
                                    Gather(
                                        value=Gather(
                                            value=Gather(
                                                value=Var(local=attention),
                                                index=head_of,
                                                result_type=tensor_type(
                                                    REAL, (length, length)
                                                ),
                                            ),
                                            index=Var(local=i),
                                            result_type=position_row,
                                        ),
                                        index=Var(local=j),
                                        result_type=REAL,
                                    ),
                                    entry(
                                        projected["value"],
                                        Var(local=j),
                                        Var(local=col),
                                        model_row,
                                    ),
                                ),
                                result_type=position_row,
                            ),
                            result_type=REAL,
                        ),
                        result_type=model_row,
                    ),
                    result_type=sequence,
                ),
            )
        )
        out_weight, out_bias = (
            self._declare_parameter(
                f"{name}_param_weight",
                tensor_type(REAL, (m, m)),
                "weight",
                scope,
                state,
                node,
                axes=(f"{name}_param_row", f"{name}_param_col"),
            ),
            self._declare_parameter(
                f"{name}_param_bias",
                tensor_type(REAL, (m,)),
                "bias",
                scope,
                state,
                node,
                axes=(f"{name}_param_row",),
            ),
        )
        image = Comprehension(
            binder=i,
            extent=length,
            body=AffineMap(
                weight=Var(local=out_weight),
                bias=Var(local=out_bias),
                sources=(
                    Gather(
                        value=Var(local=context),
                        index=Var(local=i),
                        result_type=model_row,
                    ),
                ),
                row_offset=0,
                rows=out_width,
                transform="identity",
                result_type=model_row,
            ),
            result_type=sequence,
        )
        return LetValue(
            bindings=tuple(bindings), body=image, result_type=sequence
        ), sequence


__all__ = []
