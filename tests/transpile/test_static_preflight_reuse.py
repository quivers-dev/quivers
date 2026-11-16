"""Static renderers share one render-local capability selection."""

from unittest.mock import patch

import pytest

from quivers.dsl import parse
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile.plan import Lower
from quivers.transpile.qiec_ir import analyze_qiec_capabilities


PURE = "define answer() : Int !{} =\n    return 42\n"
UNSUPPORTED = """\
define label() : String !{} =
    return "ok"
"""
PROGRAM = """\
object Out : Real 1
program model : Out -> Out
    sample x <- Normal(loc=0.0, scale=1.0)
    return x
export model
"""


@pytest.mark.parametrize("target", ["stan", "bugs", "jags"])
def test_static_render_analyzes_once_and_does_not_cache_across_renders(target):
    syntax = parse(PURE)
    with patch(
        "quivers.transpile.renderers._qiec.analyze_qiec_capabilities",
        wraps=analyze_qiec_capabilities,
    ) as analyze:
        first = transpile(syntax, target=target)
        assert analyze.call_count == 1
        second = transpile(syntax, target=target)
        assert analyze.call_count == 2
    assert first == second
    assert b"42" in first


@pytest.mark.parametrize("target", ["stan", "bugs", "jags"])
def test_unused_unrepresentable_computation_is_omitted_without_losing_supported_one(
    target,
):
    syntax = parse(PURE + UNSUPPORTED + PROGRAM)
    with patch(
        "quivers.transpile.renderers._qiec.analyze_qiec_capabilities",
        wraps=analyze_qiec_capabilities,
    ) as analyze:
        output = transpile(syntax, target=target)
    assert analyze.call_count == 1
    assert b"42" in output and b"label" not in output


@pytest.mark.parametrize("target", ["stan", "bugs", "jags"])
def test_needed_computation_preserves_every_capability_refusal(target):
    syntax = parse(UNSUPPORTED)
    ir = Lower().forward(syntax, target=target)
    diagnostics = analyze_qiec_capabilities(ir.module, target)
    expected = UnsupportedConstruct(f"qvr-{target}", [d.kind for d in diagnostics])
    assert diagnostics
    with (
        patch(
            "quivers.transpile.renderers._qiec.analyze_qiec_capabilities",
            wraps=analyze_qiec_capabilities,
        ) as analyze,
        pytest.raises(UnsupportedConstruct) as caught,
    ):
        transpile(syntax, target=target)
    assert analyze.call_count == 1
    assert caught.value.kinds == expected.kinds
    assert str(caught.value) == str(expected)
