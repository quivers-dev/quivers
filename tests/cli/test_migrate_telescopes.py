"""Released parameter lists migrate by checked CST spans and binding roles."""

from __future__ import annotations

import pytest
import torch

from quivers.cli.migrations import MigrationError, compose_migration
from quivers.cli.migrations import v0_64_2_to_head as telescopes
from quivers.cli.migrations._common import parse_validated_source
from quivers.dsl import parse


SOURCE = b"""# Header text p(A, kernel) and Real remain ordinary comments.
object A : FinSet 2
object Out : Real 1
program p(scale : Real, K : FinSet, f : Mor[K, Out], count : Nat) : K -> Out
    sample x <- f
    return x
program host : A -> Out
    sample x <- p(2.0, A, kernel, 3)
    observe y <- p(2.0, A, kernel, 3)
    marginalize z <- p(2.0, A, kernel, 3) [reduction=logsumexp]
        observe w <- Normal(loc=z, scale=1.0)
    return x
"""


def test_template_parameters_and_all_draw_calls_partition_in_original_order() -> None:
    expected = SOURCE.replace(
        b"p(scale : Real, K : FinSet, f : Mor[K, Out], count : Nat)",
        b"p[K : FinSet, f : Mor[K, Out]](scale : Real, count : Nat)",
    ).replace(b"<- p(2.0, A, kernel, 3)", b"<- p[A, kernel](2.0, 3)")
    result = telescopes.migrate(SOURCE)
    assert result == expected
    assert b"# Header text p(A, kernel)" in result
    parse(result.decode())


def test_static_only_expression_calls_retain_required_runtime_parentheses() -> None:
    source = b"""object A : FinSet 1
object Out : Real 1
program p(K : FinSet, f : Mor[K, Out]) : K -> Out
    sample x <- f
    return x
define instance = p(A, kernel)
export p(A, kernel)
"""
    expected = source.replace(
        b"p(K : FinSet, f : Mor[K, Out])", b"p[K : FinSet, f : Mor[K, Out]]"
    ).replace(b"p(A, kernel)", b"p[A, kernel]()")
    assert telescopes.migrate(source) == expected


def test_static_schema_and_rule_delimiters_preserve_all_other_bytes() -> None:
    source = """# λ: schema f(X) is prose.
schema forward(X, Y : Cat) : (X / Y) * Y -> X
rule apply(X, Y) : X / Y, Y |- X
""".encode()
    expected = source.replace(b"forward(X, Y : Cat)", b"forward[X, Y : Cat]").replace(
        b"apply(X, Y)", b"apply[X, Y]"
    )
    assert compose_migration("v0.64.2", "HEAD")(source) == expected


def test_template_multiline_comments_stay_once_with_their_items() -> None:
    source = """# λ before byte offsets
object Out : Real 1
program p(
    scale : Real, # Scalar setting
    K : FinSet, # Static plate
) : K -> Out
    sample z : K <- Normal(loc=0.0, scale=scale)
    return z
program host : A -> Out
    sample z <- p(
        2.0, # Actual scalar
        A, # Actual plate
    )
    return z
""".encode()
    result = telescopes.migrate(source)
    for comment in (
        b"# \xce\xbb before byte offsets",
        b"# Scalar setting",
        b"# Static plate",
        b"# Actual scalar",
        b"# Actual plate",
    ):
        assert result.count(comment) == 1
    assert result.index(b"K : FinSet") < result.index(b"# Static plate")
    assert result.index(b"scale : Real") < result.index(b"# Scalar setting")
    assert b"    sample z : K <- Normal(loc=0.0, scale=scale)\n    return z\n" in result
    parse_validated_source("HEAD", result)
    assert all(line == line.rstrip(b" \t") for line in result.splitlines())


def test_mixed_parameter_lists_retain_crlf_and_body_comments() -> None:
    source = b"""object Out : Real 1
program p(K : FinSet, scale : Real) : K -> Out
    # Keep this body comment.
    sample z : K <- Normal(loc=0.0, scale=scale)
    return z
""".replace(b"\n", b"\r\n")
    result = telescopes.migrate(source)
    assert result == source.replace(
        b"p(K : FinSet, scale : Real)", b"p[K : FinSet](scale : Real)"
    )
    assert result.count(b"\r\n") == source.count(b"\r\n")


def test_unaffected_scalar_headers_and_builtin_calls_are_byte_identical() -> None:
    source = b"""object A : FinSet 1
object Out : Real 1
program p(scale : Real, count : Nat) : A -> Out
    sample z <- Normal(loc=0.0, scale=scale)
    return z
"""
    assert telescopes.migrate(source) == source
    assert telescopes.migrate(telescopes.migrate(source)) == source


def test_reapplying_static_migration_is_rejected_by_historical_source_gate() -> None:
    migrated = telescopes.migrate(SOURCE)
    with pytest.raises(MigrationError, match="source does not parse under v0.64.2"):
        telescopes.migrate(migrated)


@pytest.mark.parametrize("actuals", [b"2.0, A", b"2.0, A, kernel, 3, 4"])
def test_ambiguous_call_arity_is_refused_with_actionable_location(
    actuals: bytes,
) -> None:
    source = SOURCE.replace(b"p(2.0, A, kernel, 3)", b"p(" + actuals + b")")
    with pytest.raises(
        MigrationError, match="line .*column .*complete positional argument list"
    ):
        telescopes.migrate(source)


@pytest.mark.parametrize("source", [b"program p( : A -> B\n", b"\x00"])
def test_malformed_input_is_refused(source: bytes) -> None:
    with pytest.raises(MigrationError, match="source does not parse under v0.64.2"):
        telescopes.migrate(source)


def test_migrated_template_call_retains_its_numeric_density() -> None:
    from quivers.modeling import Model

    source = b"""object Row : FinSet 2
object Out : Real 1
program effects(spread : Real, K : FinSet) : K -> Out
    sample z : K <- Normal(loc=0.0, scale=spread)
    return z
program host : Row -> Out
    sample theta <- effects(2.5, Row)
    return theta
export host
"""
    migrated = telescopes.migrate(source)
    bound = Model(migrated.decode()).bind({})
    point = torch.tensor([0.25, -0.75])
    actual = bound.trace({"theta": point}).log_joint
    expected = torch.distributions.Normal(0.0, 2.5).log_prob(point).sum()
    torch.testing.assert_close(actual.sum(), expected)


def test_inline_block_comments_and_nested_type_punctuation_are_preserved() -> None:
    source = b"""object Out : Real 1
program p(#{ scalar, not a separator }# scale : Real, K : FinSet, f : Mor[K, Out]) : K -> Out
    sample x <- f
    return x
"""
    migrated = telescopes.migrate(source)
    assert migrated.count(b"#{ scalar, not a separator }#") == 1
    assert migrated.count(b"Mor[K, Out]") == 1
    parse_validated_source("HEAD", migrated)
