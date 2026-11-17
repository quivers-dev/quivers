"""Conditional values keep branch-local partial expressions lazy on real hosts."""

import ast
import math
import subprocess

import pytest
import torch

from quivers.dsl import parse
from quivers.transpile import Lower, UnsupportedConstruct
from quivers.transpile._entry import _RENDERERS
from quivers.transpile._pipeline import parser_registry
from quivers.transpile.qiec_ir import (
    IRQiecBind,
    IRQiecConditionalValue,
    IRQiecIf,
    IRQiecLetBinding,
    IRQiecLetValue,
    IRQiecReturn,
)
from tests.transpile._tools import require_scheme, require_tool
from tests.transpile._webppl import run_webppl


SOURCE = """define guarded(x : Real) : Real !{} =
    if x < 0.0 then
        let magnitude = 0.0 - x
        let partial = log(magnitude)
        return partial + partial
    else
        let magnitude = x
        let partial = log(magnitude)
        return partial + partial
"""
DYNAMIC = ("pyro", "numpyro", "pymc", "edward2", "turing", "gen", "webppl", "church")


def _ir():
    plan = Lower().forward(parse(SOURCE))
    module = plan.module
    assert module is not None
    computation = module.computations[0]
    body = computation.body
    assert isinstance(body, IRQiecIf)

    def value(branch):
        assert isinstance(branch, IRQiecBind)
        assert isinstance(branch.then, IRQiecReturn)
        assert all(isinstance(step.first, IRQiecReturn) for step in branch.steps)
        return IRQiecLetValue(
            bindings=tuple(
                IRQiecLetBinding(binder=step.binder, value=step.first.value)
                for step in branch.steps
            ),
            body=branch.then.value,
            result_type=computation.type.result,
        )

    conditional = IRQiecConditionalValue(
        condition=body.condition,
        then=value(body.then),
        otherwise=value(body.otherwise),
        result_type=computation.type.result,
    )
    return plan.with_(
        module=module.with_(
            computations=(computation.with_(body=IRQiecReturn(value=conditional)),)
        )
    )


def _render(target):
    renderer, grammar, _ = _RENDERERS[target]
    return bytes(parser_registry().emit_pretty(grammar, renderer().render(_ir())))


@pytest.mark.parametrize("target", DYNAMIC)
def test_dynamic_hosts_render_the_common_lazy_value(target):
    source = _render(target)
    assert b"_qvr_qiec_if" in source or b"_qvr-qiec-if" in source
    if target in DYNAMIC[:4]:
        ast.parse(source)


@pytest.mark.parametrize("target", ["bugs", "jags"])
def test_graph_targets_refuse_conditional_values_by_capability(target):
    with pytest.raises(UnsupportedConstruct, match="if"):
        _render(target)


@pytest.mark.parametrize("x", [-0.2, 0.2])
def test_pyro_branch_local_bindings_preserve_value_and_second_derivatives(x):
    import pyro

    namespace = {"torch": torch, "pyro": pyro}
    exec(_render("pyro"), namespace)
    argument = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    result = namespace["qiec_guarded"](argument)
    first = torch.autograd.grad(result, argument, create_graph=True)[0]
    second = torch.autograd.grad(first, argument)[0]
    assert float(result.detach()) == pytest.approx(2 * math.log(abs(x)), rel=1e-14)
    assert float(first.detach()) == pytest.approx(2 / x, rel=1e-14)
    assert float(second) == pytest.approx(-2 / x**2, rel=1e-14)


def test_actual_webppl_lazy_branch_bindings(tmp_path):
    script = tmp_path / "lazy.wppl"
    script.write_bytes(
        _render("webppl") + b"\ndisplay([qiec_guarded(-0.2),qiec_guarded(0.2)]);\n"
    )
    result = run_webppl(script)
    assert "NaN" not in result.stdout
    import json

    assert json.loads(result.stdout) == pytest.approx([2 * math.log(0.2)] * 2)


def test_actual_scheme_lazy_branch_bindings(tmp_path):
    script = tmp_path / "lazy.scm"
    script.write_bytes(
        _render("church")
        + b"\n(display (qiec_guarded -0.2)) (newline) (display (qiec_guarded 0.2)) (newline)\n"
    )
    result = subprocess.run(
        [require_scheme(), "--script", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert [float(row) for row in result.stdout.splitlines()] == pytest.approx(
        [2 * math.log(0.2)] * 2
    )


@pytest.mark.parametrize("target", ["turing", "gen"])
def test_actual_julia_lazy_branch_bindings(target, tmp_path):
    script = tmp_path / "lazy.jl"
    # No probabilistic operation is exercised: the frontend emits an empty
    # model alongside pure QIEC functions. Admit that inert model macro only.
    macro = (
        b"macro model(expression)\n esc(expression)\nend\n"
        if target == "turing"
        else b"macro gen(expression)\n esc(expression)\nend\n"
    )
    script.write_bytes(
        macro
        + _render(target)
        + b"\nprintln(qiec_guarded(-0.2))\nprintln(qiec_guarded(0.2))\n"
    )
    result = subprocess.run(
        [require_tool("julia"), "--startup-file=no", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert [float(row) for row in result.stdout.splitlines()] == pytest.approx(
        [2 * math.log(0.2)] * 2
    )


def test_stan_keeps_branch_locals_inside_conditional_blocks_and_typechecks(tmp_path):
    source = _render("stan")
    # Every partial log is declared inside its selected branch; none is
    # evaluated ahead of the guard. Source locals use qv_, temporaries do not.
    text = source.decode()
    assert text.index("if (") < text.index("qv_partial =")
    assert "qiec_conditional_0" in text
    program = tmp_path / "lazy.stan"
    program.write_bytes(source)
    subprocess.run(
        [require_tool("stanc"), str(program), "--o", str(tmp_path / "lazy.hpp")],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
