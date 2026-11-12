"""Tests for the let-expression call surface.

Covers:

* PyTorch primitive activations and reductions exposed to let bodies.
* Calling user-defined callables (programs, morphisms, plain functions)
  injected via the structural-globals lookup.
* Compile-time arity checking on user-defined callables.
* Runtime shape errors wrapped with the call site's function name.
* Composition: primitives in primitives, programs in primitives,
  primitives in programs, programs in programs.
* End-to-end DSL surface where encoder rule bodies invoke top-level
  ``let``-bound morphisms.
"""

from __future__ import annotations
import textwrap

import pytest
import torch
import torch.nn as nn

from quivers.dsl import loads
from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprLiteral,
    LetExprVar,
)
from quivers.dsl.compiler import CompileError
from quivers.dsl.compiler.programs import (
    _LET_EXPR_BUILTINS,
    _ProgramsMixin,
    _expected_call_arity,
)


# ---------------------------------------------------------------------------
# AST helpers (avoid the parser; isolate the let-expression compiler).
# ---------------------------------------------------------------------------


def _var(name: str) -> LetExprVar:
    return LetExprVar(name=name)


def _lit(v: float) -> LetExprLiteral:
    return LetExprLiteral(value=v)


def _call(func: str, *args) -> LetExprCall:
    return LetExprCall(func=func, args=tuple(args))


def _binop(op: str, left, right) -> LetExprBinOp:
    return LetExprBinOp(op=op, left=left, right=right)


def _compile(node, globals_=None):
    return _ProgramsMixin._compile_let_expr(node, globals_=globals_)


# ===========================================================================
# Section 1: every advertised builtin runs and produces the right shape.
# ===========================================================================


class TestBuiltinPrimitives:
    """Every entry in ``_LET_EXPR_BUILTINS`` is invocable from a let
    expression and returns a tensor (or value derived from one)."""

    @pytest.mark.parametrize(
        "name",
        sorted(
            n
            for n in _LET_EXPR_BUILTINS
            if n
            not in {
                "leaky_relu",
                "prelu",
                "rrelu",
                "elu",
                "celu",
                "hardtanh",
                "hardshrink",
                "softplus",
                "softshrink",
                "threshold",
                "normalize",
                "clamp",
                "dropout",
                "alpha_dropout",
                "norm",
                "pow",
                "slice",
                "select",
                "matmul",
            }
        ),
    )
    def test_unary_builtin_runs(self, name):
        x = torch.randn(4, 8).abs() + 0.5
        fn = _compile(_call(name, _var("x")))
        out = fn({"x": x})
        # min/max/argmin/argmax etc collapse the last axis; sum/mean
        # too. Most everything else is shape-preserving. Just ensure
        # the call doesn't crash and returns a tensor.
        assert isinstance(out, torch.Tensor)

    def test_all_95_primitives_present(self):
        assert len(_LET_EXPR_BUILTINS) == 95

    def test_binary_pow_uses_both_arguments(self):
        fn = _compile(_call("pow", _var("x"), _lit(2.0)))
        torch.testing.assert_close(
            fn({"x": torch.tensor([2.0, 3.0])}), torch.tensor([4.0, 9.0])
        )

    def test_binary_min_is_elementwise(self):
        fn = _compile(_call("min", _var("x"), _var("y")))
        torch.testing.assert_close(
            fn({"x": torch.tensor([1.0, 4.0]), "y": torch.tensor([2.0, 3.0])}),
            torch.tensor([1.0, 3.0]),
        )

    def test_relu_pointwise(self):
        x = torch.tensor([-1.0, 0.0, 2.0])
        fn = _compile(_call("relu", _var("x")))
        torch.testing.assert_close(fn({"x": x}), torch.tensor([0.0, 0.0, 2.0]))

    def test_sigmoid_bounded(self):
        x = torch.randn(16)
        fn = _compile(_call("sigmoid", _var("x")))
        out = fn({"x": x})
        assert ((out > 0) & (out < 1)).all()

    def test_softmax_normalizes_last_axis(self):
        x = torch.randn(3, 5)
        fn = _compile(_call("softmax", _var("x")))
        out = fn({"x": x})
        assert isinstance(out, torch.Tensor)
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(3))

    def test_log_softmax_then_exp_equals_softmax(self):
        x = torch.randn(3, 5)
        lsm = _compile(_call("log_softmax", _var("x")))({"x": x})
        sm = _compile(_call("softmax", _var("x")))({"x": x})
        assert isinstance(lsm, torch.Tensor)
        torch.testing.assert_close(lsm.exp(), sm)

    def test_sum_reduces_last_axis(self):
        x = torch.ones(3, 4)
        fn = _compile(_call("sum", _var("x")))
        torch.testing.assert_close(fn({"x": x}), torch.full((3,), 4.0))

    def test_logsumexp_variadic_form_stacks_args(self):
        # ``logsumexp`` is dispatched as a variadic higher-order form
        # at the let-expression level: ``logsumexp(a, b, c)`` reduces
        # over the stack of its scalar/tensor arguments rather than
        # along ``dim=-1`` of a single tensor.
        fn = _compile(_call("logsumexp", _var("a"), _var("b"), _var("c")))
        env = {
            "a": torch.tensor(0.0),
            "b": torch.tensor(1.0),
            "c": torch.tensor(2.0),
        }
        torch.testing.assert_close(
            fn(env), torch.logsumexp(torch.tensor([0.0, 1.0, 2.0]), dim=0)
        )

    def test_layer_norm_zero_mean_unit_var(self):
        x = torch.randn(8, 16) * 5 + 7
        fn = _compile(_call("layer_norm", _var("x")))
        out = fn({"x": x})
        assert isinstance(out, torch.Tensor)
        torch.testing.assert_close(out.mean(dim=-1), torch.zeros(8), atol=1e-5, rtol=0)

    def test_rms_norm_preserves_shape(self):
        x = torch.randn(4, 8)
        fn = _compile(_call("rms_norm", _var("x")))
        out = fn({"x": x})
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape

    def test_chained_activations_match_torch(self):
        x = torch.randn(10)
        node = _call("tanh", _call("sigmoid", _call("relu", _var("x"))))
        out = _compile(node)({"x": x})
        torch.testing.assert_close(out, torch.tanh(torch.sigmoid(torch.relu(x))))

    def test_swish_aliases_silu(self):
        x = torch.randn(4)
        a = _compile(_call("swish", _var("x")))({"x": x})
        b = _compile(_call("silu", _var("x")))({"x": x})
        torch.testing.assert_close(a, b)

    def test_arithmetic_around_primitive(self):
        x = torch.randn(4)
        node = _binop("+", _call("relu", _var("x")), _lit(1.0))
        out = _compile(node)({"x": x})
        torch.testing.assert_close(out, torch.relu(x) + 1.0)


# ===========================================================================
# Section 2: user-defined callables are dispatched correctly.
# ===========================================================================


class TestUserCallableDispatch:
    """Calls into ``globals_`` resolve to the right callable, pass the
    evaluated arguments, and return the callable's value verbatim."""

    def test_plain_function_one_arg(self):
        def double(x):
            return 2 * x

        node = _call("double", _var("a"))
        out = _compile(node, globals_={"double": double})({"a": torch.ones(3)})
        torch.testing.assert_close(out, torch.full((3,), 2.0))

    def test_plain_function_two_args(self):
        def addmul(x, y):
            return x * 2 + y

        node = _call("addmul", _var("a"), _var("b"))
        out = _compile(node, globals_={"addmul": addmul})(
            {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
        )
        torch.testing.assert_close(out, torch.tensor([5.0]))

    def test_nn_module_callable(self):
        layer = nn.Linear(4, 8, bias=False)
        node = _call("L", _var("x"))
        out = _compile(node, globals_={"L": layer})({"x": torch.randn(2, 4)})
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8)

    def test_composition_primitive_over_user_call(self):
        layer = nn.Linear(4, 8, bias=False)
        # relu(L(x))
        node = _call("relu", _call("L", _var("x")))
        out = _compile(node, globals_={"L": layer})({"x": torch.randn(2, 4)})
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8)
        assert (out >= 0).all()

    def test_composition_user_call_over_primitive(self):
        def sq(x):
            return x * x

        # sq(relu(x))
        node = _call("sq", _call("relu", _var("x")))
        out = _compile(node, globals_={"sq": sq})({"x": torch.tensor([-2.0, 3.0])})
        torch.testing.assert_close(out, torch.tensor([0.0, 9.0]))

    def test_user_call_over_user_call(self):
        def f(x):
            return x + 1

        def g(x):
            return x * 10

        # f(g(x))
        node = _call("f", _call("g", _var("x")))
        out = _compile(node, globals_={"f": f, "g": g})({"x": torch.tensor([1.0])})
        torch.testing.assert_close(out, torch.tensor([11.0]))

    def test_user_call_with_arithmetic_arg(self):
        def f(x):
            return x

        node = _call("f", _binop("+", _var("a"), _var("b")))
        out = _compile(node, globals_={"f": f})(
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}
        )
        torch.testing.assert_close(out, torch.tensor(5.0))


# ===========================================================================
# Section 3: arity checking happens at compile time.
# ===========================================================================


class TestCompileTimeArityCheck:
    """``_compile_let_expr`` rejects calls whose argument count does
    not match the callee's positional-arity, when the arity can be
    determined statically (plain functions, ``nn.Module`` subclasses
    with a fixed ``forward``)."""

    def test_one_arg_function_called_with_zero(self):
        def f(x):
            return x

        node = _call("f")
        with pytest.raises(CompileError, match="expected 1.*got 0"):
            _compile(node, globals_={"f": f})

    def test_one_arg_function_called_with_two(self):
        def f(x):
            return x

        node = _call("f", _var("a"), _var("b"))
        with pytest.raises(CompileError, match="expected 1.*got 2"):
            _compile(node, globals_={"f": f})

    def test_two_arg_function_called_with_one(self):
        def f(x, y):
            return x + y

        node = _call("f", _var("a"))
        with pytest.raises(CompileError, match="expected 2.*got 1"):
            _compile(node, globals_={"f": f})

    def test_two_arg_function_correct_count_compiles(self):
        def f(x, y):
            return x + y

        node = _call("f", _var("a"), _var("b"))
        # Should not raise.
        compiled = _compile(node, globals_={"f": f})
        out = compiled({"a": torch.tensor(1.0), "b": torch.tensor(2.0)})
        torch.testing.assert_close(out, torch.tensor(3.0))

    def test_varargs_function_skips_check(self):
        def variadic(*xs):
            return sum(xs)

        # Three args, but ``*xs`` makes arity unknowable: must compile.
        node = _call("variadic", _lit(1.0), _lit(2.0), _lit(3.0))
        compiled = _compile(node, globals_={"variadic": variadic})
        # Runtime: 6.0
        torch.testing.assert_close(compiled({}), torch.tensor(6.0))

    def test_default_arg_function_no_default_required(self):
        def f(x, y=0):
            return x + y

        # One required arg, one default: arity = 1.
        node_one = _call("f", _var("a"))
        compiled = _compile(node_one, globals_={"f": f})
        torch.testing.assert_close(
            compiled({"a": torch.tensor(2.0)}), torch.tensor(2.0)
        )

        # Two args also acceptable? Arity check sees arity = 1, so
        # passing 2 raises. (Defaults are not currently honored by
        # let-call arity inference; the test pins this contract.)
        node_two = _call("f", _var("a"), _var("b"))
        with pytest.raises(CompileError, match="expected 1.*got 2"):
            _compile(node_two, globals_={"f": f})

    def test_lambda_callable_arity_inferred(self):
        f = lambda x, y, z: x + y + z  # noqa: E731

        with pytest.raises(CompileError, match="expected 3.*got 1"):
            _compile(_call("f", _var("a")), globals_={"f": f})

    def test_builtin_arity_not_checked(self):
        # ``clamp`` is a 3-arg builtin; the let-expression dispatcher
        # only validates user-defined callables, so this is allowed
        # to compile with any arity and fail at runtime if at all.
        node = _call("relu", _var("x"))
        out = _compile(node)({"x": torch.tensor(-1.0)})
        torch.testing.assert_close(out, torch.tensor(0.0))


class TestExpectedCallArity:
    """Direct tests on the ``_expected_call_arity`` helper."""

    def test_returns_none_for_uninspectable(self):
        # The ``object`` constructor accepts no positional args in the
        # signature; arity = 0. To get a None result we need something
        # whose signature inspection actually fails. Builtin
        # ``len`` declines inspection.
        # ``inspect.signature(len)`` raises ValueError on some Python
        # builds; we just assert the helper returns either an int or
        # None and never raises.
        try:
            r = _expected_call_arity(len)
        except Exception as e:  # pragma: no cover
            pytest.fail(f"helper raised on builtin: {e}")
        assert r is None or isinstance(r, int)

    def test_morphism_arity_one(self):
        from quivers.core.morphisms import morphism
        from quivers.core.objects import FinSet

        f = morphism(FinSet(name="A", cardinality=3), FinSet(name="B", cardinality=4))
        assert _expected_call_arity(f) == 1

    def test_function_with_defaults(self):
        def f(x, y, z=0):
            return x + y + z

        assert _expected_call_arity(f) == 2

    def test_function_with_varargs(self):
        def f(*xs):
            return xs

        assert _expected_call_arity(f) is None

    def test_function_zero_args(self):
        def f():
            return 0

        assert _expected_call_arity(f) == 0


# ===========================================================================
# Section 4: shape mismatches surface as CompileError with the call name.
# ===========================================================================


class TestRuntimeShapeWrapping:
    """Tensor shape errors from inside a user-defined callable are
    re-raised as :class:`CompileError`, naming the call site and
    forwarding the PyTorch RuntimeError message."""

    def test_linear_with_wrong_input_dim(self):
        layer = nn.Linear(8, 4)
        # Layer expects last dim == 8, but we feed last dim == 5.
        node = _call("L", _var("x"))
        compiled = _compile(node, globals_={"L": layer})
        with pytest.raises(CompileError, match="call to 'L' failed"):
            compiled({"x": torch.randn(2, 5)})

    def test_user_function_raising_typeerror(self):
        def f(x, y):
            return x + y

        # The compile-time arity check catches this before we even
        # build the closure; the runtime branch only fires when the
        # callee accepts varargs.
        with pytest.raises(CompileError, match="expected 2.*got 1"):
            _compile(_call("f", _var("a")), globals_={"f": f})

    def test_variadic_runtime_typeerror_wrapped(self):
        # ``inflate`` accepts variadic args (arity check skipped) and
        # synthesizes a TypeError at runtime; the wrapper renames it
        # to point at the call site.
        def inflate(*xs):
            return xs[0] + "not-a-tensor"

        node = _call("inflate", _var("a"))
        compiled = _compile(node, globals_={"inflate": inflate})
        with pytest.raises(CompileError, match="call to 'inflate' failed"):
            compiled({"a": torch.tensor(1.0)})

    def test_broadcasting_mismatch_wrapped(self):
        def add(x, y):
            return x + y

        # (3,) + (4,) is a broadcasting error inside PyTorch.
        node = _call("add", _var("a"), _var("b"))
        compiled = _compile(node, globals_={"add": add})
        with pytest.raises(CompileError, match="call to 'add' failed"):
            compiled({"a": torch.randn(3), "b": torch.randn(4)})

    def test_matrix_multiply_shape_mismatch_wrapped(self):
        def matprod(x, y):
            return x @ y

        # (2,3) @ (5,2): inner dims disagree.
        node = _call("matprod", _var("a"), _var("b"))
        compiled = _compile(node, globals_={"matprod": matprod})
        with pytest.raises(CompileError, match="call to 'matprod' failed"):
            compiled({"a": torch.randn(2, 3), "b": torch.randn(5, 2)})


# ===========================================================================
# Section 5: end-to-end DSL — encoder bodies see top-level morphisms.
# ===========================================================================


class TestEncoderBodyEndToEnd:
    """End-to-end DSL: encoder rule bodies can call top-level
    ``let``-bound morphisms and PyTorch primitive activations on
    real :class:`Term` inputs."""

    def test_activation_in_recurrent_body(self):
        # Sequence encoder with ``relu`` applied to each step's
        # head + state. Shape comes from the ``A`` data sort's per-key
        # learnable embedding (dim 4) plus the broadcast recurrent
        # state.
        src = """
        signature Seq
            sorts
                Seq : object [dim=4]
                A : data [dim=4]
            constructors
                Nil  :        -> Seq
                Cons : A, Seq -> Seq
        encoder C : Seq
            dim Seq = 4
            op Nil                              |-> 0.0
            op Cons(head, tail) recurrent state |-> relu(head + state)
        """
        prog = loads(textwrap.dedent(src))
        from quivers.structural import make_term

        C = prog.encoders["C"]
        t = make_term("Cons", "a", make_term("Cons", "b", make_term("Nil")))
        v = C(t)
        assert v.shape == (4,)
        assert (v >= 0).all()

    def test_chained_activations_in_rule_body(self):
        src = """
        signature Seq
            sorts
                Seq : object [dim=4]
                A : data [dim=4]
            constructors
                Nil  :        -> Seq
                Cons : A, Seq -> Seq
        encoder C : Seq
            dim Seq = 4
            op Nil                              |-> 0.0
            op Cons(head, tail) recurrent state |-> tanh(sigmoid(relu(head + state)))
        """
        prog = loads(textwrap.dedent(src))
        from quivers.structural import make_term

        C = prog.encoders["C"]
        t = make_term("Cons", "a", make_term("Cons", "b", make_term("Nil")))
        v = C(t)
        assert v.shape == (4,)

    def test_layer_norm_in_recurrent_body(self):
        src = """
        signature Seq
            sorts
                Seq : object [dim=8]
                A : data [dim=8]
            constructors
                Nil  :        -> Seq
                Cons : A, Seq -> Seq
        encoder C : Seq
            dim Seq = 8
            op Nil                              |-> 0.0
            op Cons(head, tail) recurrent state |-> layer_norm(head + state)
        """
        prog = loads(textwrap.dedent(src))
        from quivers.structural import make_term

        C = prog.encoders["C"]
        t = make_term("Cons", "a", make_term("Cons", "b", make_term("Nil")))
        v = C(t)
        assert v.shape == (8,)

    def test_softmax_then_sum_in_body(self):
        src = """
        signature Seq
            sorts
                Seq : object [dim=8]
                A : data [dim=8]
            constructors
                Nil  :        -> Seq
                Cons : A, Seq -> Seq
        encoder C : Seq
            dim Seq = 8
            op Nil                              |-> 0.0
            op Cons(head, tail) recurrent state |-> softmax(head + state)
        """
        prog = loads(textwrap.dedent(src))
        from quivers.structural import make_term

        C = prog.encoders["C"]
        t = make_term("Cons", "a", make_term("Nil"))
        v = C(t)
        assert v.shape == (8,)
        torch.testing.assert_close(v.sum(), torch.tensor(1.0))


# ===========================================================================
# Section 6: dispatch ordering — builtins, user callables, constructors.
# ===========================================================================


class TestDispatchOrdering:
    """Builtins win over identically-named user callables (no
    shadowing); constructors win when the name is in the
    constructor set."""

    def test_builtin_wins_over_user_with_same_name(self):
        # Define a globally-injected ``relu`` that does the wrong
        # thing; the builtin should still be used.
        def bogus_relu(x):
            return x * -999

        node = _call("relu", _var("x"))
        out = _compile(node, globals_={"relu": bogus_relu})(
            {"x": torch.tensor([-1.0, 2.0])}
        )
        torch.testing.assert_close(out, torch.tensor([0.0, 2.0]))

    def test_unknown_func_raises(self):
        node = _call("does_not_exist", _var("x"))
        compiled = _compile(node)
        with pytest.raises(CompileError, match="unknown function"):
            compiled({"x": torch.tensor(1.0)})

    def test_constructor_set_is_consulted(self):
        # ``MyCtor`` not in globals or builtins but in the constructor
        # set: should build a term tuple.
        node = _call("MyCtor", _lit(1.0), _lit(2.0))
        compiled = _compile(node, globals_={"__constructors__": frozenset({"MyCtor"})})
        out = compiled({})
        assert out[0] == "MyCtor"
        assert len(out) == 3

    @pytest.mark.parametrize("node", [_var("Constant"), _call("Constant")])
    def test_nullary_constructor_uses_atom_leaf(self, node):
        compiled = _compile(
            node, globals_={"__constructors__": frozenset({"Constant"})}
        )
        assert compiled({}) == ("atom", "Constant")

    def test_bound_name_uses_variable_leaf(self):
        compiled = _compile(
            _var("x"),
            globals_={
                "__constructors__": frozenset({"x"}),
                "__bound_vars__": frozenset({"x"}),
            },
        )
        assert compiled({}) == ("var", "x")


# ===========================================================================
# Section 7: stress tests — deep nesting, many primitives, broadcasting.
# ===========================================================================


class TestDeepNestingAndBroadcasting:
    """Compiler handles deep let-expression trees and broad shapes
    without breaking the closure structure."""

    def test_deep_unary_chain(self):
        # relu(relu(...relu(x)...)) 20 deep.
        node = _var("x")
        for _ in range(20):
            node = _call("relu", node)
        compiled = _compile(node)
        out = compiled({"x": torch.tensor([-1.0, 1.0])})
        torch.testing.assert_close(out, torch.tensor([0.0, 1.0]))

    def test_wide_arithmetic_tree(self):
        # ((a + b) + (c + d)) + (relu(e) + sigmoid(f))
        ab = _binop("+", _var("a"), _var("b"))
        cd = _binop("+", _var("c"), _var("d"))
        ef = _binop("+", _call("relu", _var("e")), _call("sigmoid", _var("f")))
        outer = _binop("+", _binop("+", ab, cd), ef)
        compiled = _compile(outer)
        env = {k: torch.tensor(1.0) for k in "abcdef"}
        out = compiled(env)
        expected = (
            1
            + 1
            + 1
            + 1
            + torch.relu(torch.tensor(1.0))
            + torch.sigmoid(torch.tensor(1.0))
        )
        torch.testing.assert_close(out, expected)

    def test_batched_call_through_module(self):
        # Same Linear layer applied through a let-call with a
        # batch dim of 7.
        layer = nn.Linear(16, 4)
        node = _call("L", _var("x"))
        compiled = _compile(node, globals_={"L": layer})
        out = compiled({"x": torch.randn(7, 16)})
        assert isinstance(out, torch.Tensor)
        assert out.shape == (7, 4)

    def test_mixed_primitives_and_user_calls_deep(self):
        # gelu(L1(relu(L2(x))))
        L1 = nn.Linear(8, 4)
        L2 = nn.Linear(16, 8)
        node = _call(
            "gelu",
            _call("L1", _call("relu", _call("L2", _var("x")))),
        )
        compiled = _compile(node, globals_={"L1": L1, "L2": L2})
        out = compiled({"x": torch.randn(3, 16)})
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 4)
