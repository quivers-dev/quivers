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
                        IndexLiteral(width, NAT),
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
            or shape[1][1] != IndexLiteral(width, NAT)
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
        serial = state.function_calls
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
        d = IndexLiteral(width, NAT)
        row = tensor_type(REAL, (d,))
        table = self._declare_parameter(
            f"{name}_param_positions",
            tensor_type(REAL, (IndexLiteral(count, NAT), d)),
            "weight",
            scope,
            state,
            node,
            axes=(f"{name}_position", f"{name}_param_col"),
        )
        prefix = f"{name}${serial}"
        result = tensor_type(REAL, (length, d))
        source = Local(f"{prefix}$x", result)
        i = Local(f"{prefix}$i", INT)
        k = Local(f"{prefix}$k", INT)
        path = ("programs", state.declaration.name, "functions", name, serial)
        entry = self._primitive(
            "add_real",
            (
                Gather(Gather(Var(source), Var(i), row), Var(k), REAL),
                Gather(Gather(Var(table), Var(i), row), Var(k), REAL),
            ),
            node,
            (*path, "add"),
        )
        image = Comprehension(i, length, Comprehension(k, d, entry, row), result)
        return LetValue((LetBinding(source, value),), image, result), result

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
        d = IndexLiteral(width, NAT)
        m = IndexLiteral(out_width, NAT)
        e = IndexLiteral(head_width, NAT)
        h = IndexLiteral(heads, NAT)
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
        source = Local(f"{prefix}$x", tensor_type(REAL, (length, d)))
        i = Local(f"{prefix}$i", INT)
        j = Local(f"{prefix}$j", INT)
        g = Local(f"{prefix}$g", INT)
        u = Local(f"{prefix}$u", INT)
        col = Local(f"{prefix}$col", INT)

        def prim(operator: str, *arguments: Value) -> Value:
            return self._primitive(operator, arguments, node, (*path, operator))

        def entry(table: Local, row: Value, column: Value, row_type: TypeExpr) -> Value:
            return Gather(Gather(Var(table), row, row_type), column, REAL)

        def int_literal(number: int) -> Value:
            return LiteralValue(number, INT)

        def real_literal(number: float) -> Value:
            return LiteralValue(number, REAL)

        bindings: list[LetBinding] = [LetBinding(source, value)]
        projected: dict[str, Local] = {}
        for role in ("query", "key", "value"):
            weight, bias = projection(role, d)
            local = Local(f"{prefix}${role}", sequence)
            bindings.append(
                LetBinding(
                    local,
                    Comprehension(
                        i,
                        length,
                        AffineMap(
                            Var(weight),
                            Var(bias),
                            (Gather(Var(source), Var(i), in_row),),
                            0,
                            out_width,
                            "identity",
                            model_row,
                        ),
                        sequence,
                    ),
                )
            )
            projected[role] = local
        # Coordinate ``u`` of head ``g`` is model coordinate ``g e + u``.
        coordinate = prim(
            "add_int", prim("mul_int", Var(g), int_literal(head_width)), Var(u)
        )
        score = prim(
            "mul_real",
            real_literal(1.0 / math.sqrt(head_width)),
            Reduction(
                "sum",
                Comprehension(
                    u,
                    e,
                    prim(
                        "mul_real",
                        entry(projected["query"], Var(i), coordinate, model_row),
                        entry(projected["key"], Var(j), coordinate, model_row),
                    ),
                    tensor_type(REAL, (e,)),
                ),
                REAL,
            ),
        )
        scores = Local(f"{prefix}$scores", position_row)
        scores_value = Comprehension(j, length, score, position_row)
        if causal:
            # The weights over the positions ``j <= i``: the row's peak is
            # taken over them (a later position reads the score at ``i``,
            # which is among them), and a later position's exponent is
            # zero before its indicator zeroes it, so no term overflows.
            peak = Local(f"{prefix}$peak", REAL)
            visible = prim("le_int", Var(j), Var(i))
            at_i = Gather(Var(scores), Var(i), REAL)
            at_j = Gather(Var(scores), Var(j), REAL)
            row_weights: Value = LetValue(
                (
                    LetBinding(scores, scores_value),
                    LetBinding(
                        peak,
                        Reduction(
                            "max",
                            Comprehension(
                                j,
                                length,
                                prim("select_real", visible, at_j, at_i),
                                position_row,
                            ),
                            REAL,
                        ),
                    ),
                ),
                Rowwise(
                    "normalize",
                    Comprehension(
                        j,
                        length,
                        prim(
                            "mul_real",
                            prim("bool_to_real", visible),
                            prim(
                                "exp",
                                prim(
                                    "select_real",
                                    visible,
                                    prim("sub_real", at_j, Var(peak)),
                                    real_literal(0.0),
                                ),
                            ),
                        ),
                        position_row,
                    ),
                    position_row,
                ),
                position_row,
            )
        else:
            row_weights = Rowwise("softmax", scores_value, position_row)
        attention = Local(f"{prefix}$weights", weights_type)
        bindings.append(
            LetBinding(
                attention,
                Comprehension(
                    g,
                    h,
                    Comprehension(
                        i,
                        length,
                        row_weights,
                        tensor_type(REAL, (length, length)),
                    ),
                    weights_type,
                ),
            )
        )
        # Model coordinate ``col`` belongs to head ``col div e``.
        head_of = prim("div_int", Var(col), int_literal(head_width))
        context = Local(f"{prefix}$context", sequence)
        bindings.append(
            LetBinding(
                context,
                Comprehension(
                    i,
                    length,
                    Comprehension(
                        col,
                        m,
                        Reduction(
                            "sum",
                            Comprehension(
                                j,
                                length,
                                prim(
                                    "mul_real",
                                    Gather(
                                        Gather(
                                            Gather(
                                                Var(attention),
                                                head_of,
                                                tensor_type(REAL, (length, length)),
                                            ),
                                            Var(i),
                                            position_row,
                                        ),
                                        Var(j),
                                        REAL,
                                    ),
                                    entry(
                                        projected["value"], Var(j), Var(col), model_row
                                    ),
                                ),
                                position_row,
                            ),
                            REAL,
                        ),
                        model_row,
                    ),
                    sequence,
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
            i,
            length,
            AffineMap(
                Var(out_weight),
                Var(out_bias),
                (Gather(Var(context), Var(i), model_row),),
                0,
                out_width,
                "identity",
                model_row,
            ),
            sequence,
        )
        return LetValue(tuple(bindings), image, sequence), sequence
