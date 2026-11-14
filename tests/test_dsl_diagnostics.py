"""Diagnostic-quality regression tests for the QVR front end.

Malformed source must raise ``ParseError`` or ``CompileError`` with
the offending token's line and column. Well-formed numeric literals
must retain their values. Each case checks one diagnostic contract.

Coverage:

* Misplaced constructor keys on a morphism option block produce the
  braces hint (``Real 1 {low=...}``).
* Signed, leading-dot, and exponent-only numeric literals parse to
  their typed values end-to-end.
* Empty option blocks, square-bracket sample tuples, tuple observes,
  identifier ``repeat`` counts, non-identifier ``identity`` arguments,
  top-level ``let``, ``=>`` rule conclusions, the removed compose
  operators, missing ``return`` steps, unknown option keys, unknown
  role values, typo'd keywords, and negative constructor positional
  arguments all raise with the expected message fragment.
* Line/column accuracy: the message carries the offending token's
  ``line N`` for every location-pinned case.
"""

from __future__ import annotations

import pytest

from quivers.continuous.spaces import Euclidean
from quivers.dsl import CompileError, ParseError, loads, parse
from quivers.dsl.ast_nodes import (
    ContinuousConstructor,
    ObjectDecl,
    TypeFromExpr,
)


# ---------------------------------------------------------------------------
# Error-diagnostic table
# ---------------------------------------------------------------------------


_ERROR_CASES: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "misplaced-constructor-key-hints-braces",
        ("object A : FinSet 3\nmorphism f : A -> Real 1 [low=-1.0]\nexport f\n"),
        ("line 2", "unknown option 'low'", "braces", "Real 1 {low="),
    ),
    (
        "empty-option-block",
        (
            "object A : FinSet 3\n"
            "object B : FinSet 3\n"
            "morphism f : A -> B []\n"
            "export f\n"
        ),
        ("line 3", "missing identifier"),
    ),
    (
        "square-bracket-sample-tuple",
        (
            "object A : FinSet 3\n"
            "object R : Real 2\n"
            "morphism f : A -> R [role=kernel] ~ Normal\n"
            "program p : A -> R\n"
            "    sample [x, y] <- f\n"
            "    return x\n"
            "export p\n"
        ),
        ("line 5", "syntax error"),
    ),
    (
        "tuple-observe-rejected-at-compile",
        (
            "object A : FinSet 3\n"
            "object R : Real 2\n"
            "morphism f : A -> R [role=kernel] ~ Normal\n"
            "program p : A -> R\n"
            "    sample x <- f\n"
            "    observe (u, v) <- f\n"
            "    return x\n"
            "export p\n"
        ),
        ("line 6", "observe takes a single variable", "(u, v)"),
    ),
    (
        "repeat-identifier-count",
        (
            "object A : FinSet 3\n"
            "morphism g : A -> A [role=parameter]\n"
            "define h = repeat(g, n)\n"
            "export h\n"
        ),
        ("line 3", "syntax error"),
    ),
    (
        "identity-non-identifier-argument",
        (
            "object A : FinSet 3\n"
            "object B : FinSet 3\n"
            "define i = identity(A * B)\n"
            "export i\n"
        ),
        ("line 3", "syntax error"),
    ),
    (
        "top-level-let",
        (
            "object A : FinSet 3\n"
            "morphism f : A -> A [role=parameter]\n"
            "let h = f >> f\n"
            "export h\n"
        ),
        ("line 3", "let h = f >> f"),
    ),
    (
        "rule-fat-arrow-conclusion",
        (
            "object Atoms : {NP, S}\n"
            "object Cat : FreeResiduated(Atoms, depth=2, ops=[slash])\n"
            "rule app(X, Y) : X/Y, Y => X\n"
        ),
        ("line 3", "syntax error"),
    ),
    (
        "missing-return-step",
        (
            "object A : FinSet 3\n"
            "object R : Real 2\n"
            "morphism f : A -> R [role=kernel] ~ Normal\n"
            "program p : A -> R\n"
            "    sample x <- f\n"
            "export p\n"
        ),
        ("return",),
    ),
    (
        "unknown-option-key-did-you-mean",
        ("object A : FinSet 3\nmorphism f : A -> A [rol=latent]\nexport f\n"),
        ("line 2", "unknown option 'rol'", "did you mean 'role'"),
    ),
    (
        "unknown-role-value-lists-roles",
        ("object A : FinSet 3\nmorphism f : A -> A [role=latentt]\nexport f\n"),
        (
            "line 2",
            "unknown role 'latentt'",
            "'kernel'",
            "'parameter'",
            "'observed'",
        ),
    ),
    (
        # `scale` configures a latent morphism's init. A family-backed
        # kernel draws its parameters from its param_source, so the
        # option means nothing there and must not be swallowed.
        "scale-on-kernel-names-the-role-that-reads-it",
        ("object S : Real 4\nmorphism f : S -> S [scale=0.1] ~ Normal\nexport f\n"),
        ("line 2", "'scale'", "not read by role=kernel", "role=parameter"),
    ),
    (
        "bins-on-kernel-names-the-role-that-reads-it",
        ("object S : Real 4\nmorphism f : S -> S [bins=4] ~ Normal\nexport f\n"),
        ("line 2", "'bins'", "not read by role=kernel", "role=discretize"),
    ),
    (
        "param-source-on-latent-names-the-role-that-reads-it",
        (
            "object A : FinSet 3\n"
            "morphism f : A -> A [role=parameter, param_source=mlp]\n"
            "export f\n"
        ),
        ("line 2", "'param_source'", "not read by role=parameter", "role=kernel"),
    ),
    (
        "typo-keyword",
        ("object A : FinSet 3\nmorphsim f : A -> A\nexport f\n"),
        ("line 2", "morphsim"),
    ),
    (
        "negative-constructor-positional",
        ("object A : Real -1\nexport A\n"),
        ("line 1", "syntax error"),
    ),
]


@pytest.mark.parametrize(
    "source,fragments",
    [(src, frags) for _, src, frags in _ERROR_CASES],
    ids=[case_id for case_id, _, _ in _ERROR_CASES],
)
def test_malformed_source_raises_with_precise_diagnostic(
    source: str,
    fragments: tuple[str, ...],
) -> None:
    """Each malformed module raises loudly, and the message carries
    the expected fragments (including the offending line number for
    the location-pinned cases)."""
    with pytest.raises((ParseError, CompileError)) as excinfo:
        loads(source)
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, f"expected {fragment!r} in diagnostic:\n{message}"


# ---------------------------------------------------------------------------
# Removed compose operators fail loudly
# ---------------------------------------------------------------------------


_CUT_OPERATORS = (">=>", "*>", "~>", "||>", "?>", "&&>", "+>", "$>", "%>")


@pytest.mark.parametrize("op", _CUT_OPERATORS)
def test_cut_compose_operator_is_a_parse_error(op: str) -> None:
    """The compose operators outside ``>>``/``<<``/``>>>``/``@`` do
    not parse; the diagnostic points at the operator's line."""
    source = (
        "object A : FinSet 3\n"
        "morphism f : A -> A [role=parameter]\n"
        "morphism g : A -> A [role=parameter]\n"
        f"define h = f {op} g\n"
        "export h\n"
    )
    with pytest.raises(ParseError) as excinfo:
        parse(source)
    assert "line 4" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Numeric literals parse to typed values
# ---------------------------------------------------------------------------


def _constructor_kwargs(source: str) -> dict[str, float | int | str]:
    """Parse ``source`` and return the kwargs of the constructor in
    its second statement's initializer."""
    module = parse(source)
    stmt = module.statements[1]
    assert isinstance(stmt, ObjectDecl)
    assert isinstance(stmt.init, TypeFromExpr)
    ctor = stmt.init.expr
    assert isinstance(ctor, ContinuousConstructor)
    return ctor.kwargs


def test_signed_constructor_kwargs_parse_to_negative_floats() -> None:
    """``low=-1.0`` inside braces survives as the float ``-1.0``,
    both in the AST and on the compiled continuous space."""
    source = (
        "object A : FinSet 3\n"
        "object R : Real 2 {low=-1.0, high=1.0}\n"
        "morphism f : A -> R [role=kernel] ~ Normal\n"
        "export f\n"
    )
    kwargs = _constructor_kwargs(source)
    assert kwargs == {"low": -1.0, "high": 1.0}
    program = loads(source)
    morphism = program.morphism
    assert morphism is not None
    codomain = morphism.codomain
    assert isinstance(codomain, Euclidean)
    assert codomain.low == -1.0
    assert codomain.high == 1.0


def test_leading_dot_float_parses_to_half() -> None:
    source = "object A : FinSet 3\nobject R : Real 2 {low=.5}\nexport A\n"
    assert _constructor_kwargs(source) == {"low": 0.5}


def test_exponent_only_float_parses() -> None:
    source = "object A : FinSet 3\nobject R : Real 2 {low=1e-3}\nexport A\n"
    assert _constructor_kwargs(source) == {"low": 0.001}


# ---------------------------------------------------------------------------
# review-driven regressions: silent misbehavior the front end must reject
# ---------------------------------------------------------------------------


def test_kernel_deterministic_init_is_rejected() -> None:
    """A role-less (default-kernel) morphism with a deterministic
    ``~ <expression>`` initializer must error rather than silently
    compiling to a bare random kernel; the message names the fixing
    roles."""
    source = "object X : FinSet 3\nmorphism f : X -> X ~ identity(X)\nexport f\n"
    with pytest.raises(CompileError) as excinfo:
        loads(source)
    message = str(excinfo.value)
    assert "role=observed" in message
    assert "line 2" in message


def test_object_slash_direction_ignores_intervening_comment() -> None:
    """The slash direction comes from the grammar field, so a block
    comment carrying a backslash between the operator and its right
    operand does not flip ``/`` to ``\\``."""
    from quivers.dsl.ast_nodes import ObjectSlash, TypeFromExpr

    module = parse("object Z : A /#{x}# B\n")
    stmt = module.statements[0]
    assert isinstance(stmt, ObjectDecl)
    assert isinstance(stmt.init, TypeFromExpr)
    slash = stmt.init.expr
    assert isinstance(slash, ObjectSlash)
    assert slash.direction == "/"


def test_compound_bracket_index_is_rejected() -> None:
    """A bracket index that is not a bare identifier is rejected at
    parse time rather than silently dropped to an empty index tuple."""
    source = (
        "object P : FinSet 2\n"
        "program q : P -> P [n=1]\n"
        "    sample x <- f(theta[i * j])\n"
        "    return x\n"
    )
    with pytest.raises(ParseError) as excinfo:
        parse(source)
    assert "bare identifier" in str(excinfo.value)


def test_escaped_string_round_trips_as_a_fixed_point() -> None:
    """A lexicon word containing an escaped quote survives
    parse -> emit -> parse -> emit without growing."""
    from quivers.dsl.emit import module_to_source

    source = 'deduction d : T -> T\n    atoms x\n    lexicon\n        "a\\"b" : x = x\n'
    first = module_to_source(parse(source))
    second = module_to_source(parse(first))
    assert first == second
    assert '"a\\"b"' in first
