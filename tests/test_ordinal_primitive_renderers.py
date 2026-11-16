"""Checked probability primitives retain domains, tails, and target capability."""

from __future__ import annotations

import ast
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import panproto
import pytest
from scipy.special import log_ndtr
import torch

from quivers.dsl import parse
from quivers.dsl.pure_builtins import EAGER_BUILTINS
from quivers.qiec.primitives import IMPLEMENTATIONS
from quivers.transpile import available_targets, transpile, UnsupportedConstruct
from quivers.transpile._builtin_spellings import BUILTIN_SPELLINGS, Refusal
from quivers.transpile.qiec_ir import capabilities_for_target
from quivers.transpile._runtime_sources import (
    javascript_runtime_source,
    python_runtime_source,
)
from tests.transpile._tools import require_tool, require_scheme
from tests.transpile._webppl import run_webppl


NAMES = ("log_probability", "log1mexp", "log_ndtr", "positive_parameter")
RUNTIME = Path(__file__).resolve().parents[1] / "src/quivers/transpile"
VALUES = {
    "positive_parameter": [1e-300, 0.2, 3.0],
    "log_probability": [0.0, 1e-300, 0.2, 1.0],
    "log1mexp": [-math.inf, -1000.0, -0.7, -1e-15, 0.0],
    "log_ndtr": [-math.inf, -1000.0, -40.0, -10.0, -2.5, 0.0, 2.5, 10.0, math.inf],
}
INVALID = {
    "positive_parameter": [True, 0.0, -1.0, math.inf, -math.inf, math.nan],
    "log_probability": [True, -0.1, 1.1, math.inf, math.nan],
    "log1mexp": [True, 0.1, math.inf, math.nan],
    "log_ndtr": [True, math.nan],
}
GRAMMARS = dict.fromkeys(("pyro", "numpyro", "pymc", "edward2"), "python") | {
    "stan": "stan",
    "church": "scheme",
    "webppl": "javascript",
    "turing": "julia",
    "gen": "julia",
    "bugs": "bugs",
    "jags": "jags",
}


def _expected(name, value):
    if name == "positive_parameter":
        return value
    if name == "log_ndtr":
        return float(log_ndtr(value))
    if name == "log_probability":
        return math.log(value) if value else -math.inf
    if value == 0:
        return -math.inf
    return (
        math.log(-math.expm1(value))
        if value > -math.log(2)
        else math.log1p(-math.exp(value))
    )


def _functions(file):
    """Evaluate only standalone probability helpers, without backend imports."""
    tree = ast.parse(python_runtime_source(RUNTIME / file).decode())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith(
            ("_qvr_log", "_qvr_probability_check", "_qvr_positive_parameter")
        )
    ]
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), file, "exec"), namespace
    )
    return namespace


@lru_cache
def _rendered(name, target, canonical=False):
    define = (
        f"define checked(x : Real) : Real !{{}} =\n    if x < -10000.0 then\n        checked(-x)\n    else\n        return {name}(x)\n"
        if canonical
        else ""
    )
    callee = "checked" if canonical else name
    binding = "<-" if canonical else "="
    source = f"""object Out : Real 1
{define}program p(x : Real, y : Real) : Out -> Out
    let mu {binding} {callee}(x)
    observe y <- Normal(loc=mu, scale=1.0)
    return mu
export p
"""
    return transpile(parse(source), target=target)


@pytest.mark.parametrize("name", NAMES)
def test_scalar_and_torch_probability_domains_endpoints_and_tail_oracles(name):
    namespace = _functions("runtime_pyro.py")
    helper = namespace["_qvr_" + name]
    for value in VALUES[name]:
        expected = _expected(name, value)
        assert IMPLEMENTATIONS[name](value) == pytest.approx(
            expected, rel=3e-14, abs=3e-15
        )
        tensor = torch.tensor(value, dtype=torch.float64)
        assert float(helper(tensor)) == pytest.approx(expected, rel=3e-14, abs=3e-15)
        assert float(EAGER_BUILTINS[name](tensor)) == pytest.approx(
            expected, rel=3e-14, abs=3e-15
        )
    for value in INVALID[name]:
        with pytest.raises((ValueError, TypeError)):
            helper(torch.tensor(value))


@pytest.mark.parametrize(
    "name,value",
    [
        ("log_probability", 0.2),
        ("positive_parameter", 0.2),
        ("log1mexp", -0.2),
        ("log1mexp", -20.0),
        ("log_ndtr", -40.0),
        ("log_ndtr", 0.0),
        ("log_ndtr", 8.0),
    ],
)
def test_torch_probability_first_second_derivatives_and_endpoint_masking(name, value):
    helper = _functions("runtime_pyro.py")["_qvr_" + name]
    x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(helper, (x,))
    assert torch.autograd.gradgradcheck(helper, (x,))
    endpoints = torch.tensor(VALUES[name], dtype=torch.float64, requires_grad=True)
    output = helper(endpoints)
    safe = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
    gradient = torch.autograd.grad(safe.sum(), endpoints)[0]
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("target", sorted(available_targets()))
@pytest.mark.parametrize("name", NAMES)
def test_ordinary_probability_renderers_define_helpers_or_refuse_explicitly(
    name, target
):
    if isinstance(BUILTIN_SPELLINGS[target][name], Refusal):
        with pytest.raises(UnsupportedConstruct, match=name):
            _rendered(name, target)
        return
    output = _rendered(name, target)
    parsed = panproto.AstParserRegistry().parse_with_protocol(
        GRAMMARS[target], output, name
    )
    assert parsed.vertex_count
    if GRAMMARS[target] == "python":
        compile(output, name, "exec")


@pytest.mark.parametrize("name", NAMES)
def test_probability_stan_helpers_pass_real_typechecking(name, tmp_path):
    path = tmp_path / (name + ".stan")
    path.write_bytes(_rendered(name, "stan"))
    result = subprocess.run(
        [require_tool("stanc"), "--info", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", NAMES)
def test_probability_pymc_symbolic_values_derivatives_and_domain_assertions(name):
    import pymc
    import pytensor
    import pytensor.tensor as pt

    namespace = _functions("runtime_pymc.py") | {"pymc": pymc}
    # Functions retain their defining namespace.
    helper = namespace["_qvr_" + name]
    helper.__globals__["pymc"] = pymc
    x = pt.dscalar("x")
    result = helper(x)
    evaluate = pytensor.function([x], result, mode="FAST_COMPILE")
    for value in VALUES[name]:
        assert float(evaluate(value)) == pytest.approx(
            _expected(name, value), rel=3e-14, abs=3e-15
        )
    for value in INVALID[name]:
        if isinstance(value, bool):
            with pytest.raises(ValueError):
                helper(pt.bscalar())
        else:
            with pytest.raises(AssertionError):
                evaluate(value)
    value = 0.2 if name in {"log_probability", "positive_parameter"} else -0.2
    gradient = pytensor.function([x], pt.grad(result, x), mode="FAST_COMPILE")
    h = 1e-5
    oracle = (_expected(name, value + h) - _expected(name, value - h)) / (2 * h)
    assert float(gradient(value)) == pytest.approx(oracle, rel=1e-7)
    if name == "positive_parameter":
        assert float(gradient(value)) == 1.0
        return
    first = pt.grad(result, x)
    curvature = pytensor.function([x], pt.grad(first, x), mode="FAST_COMPILE")
    if name == "log_probability":
        second = -1 / value**2
    elif name == "log1mexp":
        second = -math.exp(value) / (-math.expm1(value)) ** 2
    else:
        ratio = math.exp(
            -0.5 * value**2 - 0.5 * math.log(2 * math.pi) - _expected(name, value)
        )
        second = -ratio * (value + ratio)
    assert float(curvature(value)) == pytest.approx(second, rel=1e-10)


def test_shared_javascript_probability_runtime_values_and_checked_domains():
    driver = """
var error = function(message) { throw new Error(message); };
""" + javascript_runtime_source(RUNTIME / "runtime_qiec.js").decode()
    cases = [(name, value) for name in NAMES for value in VALUES[name]]
    calls = [
        f"_qvr_{name}_scalar({'Infinity' if value == math.inf else '-Infinity' if value == -math.inf else repr(value)})"
        for name, value in cases
    ]
    driver += (
        "\nconsole.log(JSON.stringify(["
        + ",".join(calls)
        + "].map(function(x){return String(x);})));\n"
    )
    invalid = [(name, value) for name in NAMES for value in INVALID[name]]
    for name, value in invalid:
        literal = (
            "true"
            if value is True
            else "NaN"
            if math.isnan(value)
            else "Infinity"
            if value == math.inf
            else repr(value)
        )
        driver += f'var refused = false; try {{ _qvr_{name}_scalar({literal}); }} catch (e) {{ refused = true; }} if (!refused) {{ throw new Error("domain accepted"); }}\n'
    result = subprocess.run(
        [require_tool("node"), "-e", driver], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    numbers = [float(value) for value in json.loads(result.stdout)]
    assert numbers == pytest.approx(
        [_expected(name, value) for name, value in cases], rel=3e-14, abs=3e-15
    )


def test_shared_scheme_probability_runtime_values_and_domains(tmp_path):
    runtime = (RUNTIME / "runtime_qiec.scm").read_text()
    calls = []
    expected = []
    for name in NAMES:
        for value in VALUES[name]:
            literal = (
                "+inf.0"
                if value == math.inf
                else "-inf.0"
                if value == -math.inf
                else repr(value)
            )
            calls.append(
                f"(display (_qvr-{name.replace('_', '-')}-scalar {literal})) (newline)"
            )
            expected.append(_expected(name, value))
    for name in NAMES:
        for value in INVALID[name]:
            literal = (
                "#t"
                if value is True
                else "+nan.0"
                if math.isnan(value)
                else "+inf.0"
                if value == math.inf
                else repr(value)
            )
            calls.append(
                f'(unless (guard (condition (else #t)) (_qvr-{name.replace("_", "-")}-scalar {literal}) #f) (error \'test "domain accepted"))'
            )
    path = tmp_path / "probability.scm"
    path.write_text(runtime + "\n" + "\n".join(calls) + "\n")
    result = subprocess.run(
        [require_scheme(), "--script", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    values = [
        float(line.replace("+inf.0", "inf").replace("-inf.0", "-inf"))
        for line in result.stdout.splitlines()
    ]
    assert values == pytest.approx(expected, rel=3e-14, abs=3e-15)


@pytest.mark.parametrize("target", sorted(available_targets()))
@pytest.mark.parametrize("name", NAMES)
def test_canonical_probability_capabilities_and_runtime_routing(name, target):
    assert capabilities_for_target(target).supports("probability") == (
        target not in {"bugs", "jags", "stan"}
    )
    if target in {"bugs", "jags", "stan"}:
        with pytest.raises(UnsupportedConstruct):
            _rendered(name, target, canonical=True)
        return
    output = _rendered(name, target, canonical=True)
    parsed = panproto.AstParserRegistry().parse_with_protocol(
        GRAMMARS[target], output, name
    )
    assert parsed.vertex_count
    if GRAMMARS[target] == "python":
        compile(output, name, "exec")


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "name,value",
    [
        ("log_probability", 0.2),
        ("positive_parameter", 0.2),
        ("log1mexp", -0.000001),
        ("log_ndtr", -40.0),
    ],
)
def test_emitted_pyro_probability_density_and_gradient(name, value, canonical):
    import pyro

    namespace = {"torch": torch, "pyro": pyro}
    exec(_rendered(name, "pyro", canonical), namespace)
    x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    recorded = pyro.poutine.trace(namespace["model"]).get_trace(
        torch.zeros(1), x=x, y=torch.tensor(0.3, dtype=torch.float64)
    )
    score = recorded.log_prob_sum()
    expected = torch.distributions.Normal(
        torch.tensor(_expected(name, value), dtype=torch.float64), 1.0
    ).log_prob(torch.tensor(0.3, dtype=torch.float64))
    torch.testing.assert_close(score, expected)
    derivative = torch.autograd.grad(score, x)[0]
    h = min(1e-6, abs(value) / 100) if value else 1e-6

    def f(t):
        return -0.5 * (0.3 - _expected(name, t)) ** 2 - 0.5 * math.log(2 * math.pi)

    assert float(derivative) == pytest.approx(
        (f(value + h) - f(value - h)) / (2 * h), rel=1e-4
    )


def test_shared_julia_probability_runtime_executes_values_and_domains(tmp_path):
    source = (RUNTIME / "runtime_qiec.jl").read_text()
    expected = []
    for name in NAMES:
        for value in VALUES[name]:
            literal = (
                "Inf"
                if value == math.inf
                else "-Inf"
                if value == -math.inf
                else repr(value)
            )
            source += f"\nprintln(_qvr_{name}({literal}))\n"
            expected.append(_expected(name, value))
    for name in NAMES:
        for value in INVALID[name]:
            literal = (
                "true"
                if value is True
                else "NaN"
                if math.isnan(value)
                else "Inf"
                if value == math.inf
                else repr(value)
            )
            source += f'\ntry; _qvr_{name}({literal}); error("domain accepted"); catch exception; exception isa ErrorException && exception.msg == "domain accepted" && rethrow(); end\n'
    path = tmp_path / "probability.jl"
    path.write_text(source)
    result = subprocess.run(
        [require_tool("julia"), "--startup-file=no", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert [float(x) for x in result.stdout.splitlines()] == pytest.approx(
        expected, rel=3e-14, abs=3e-15
    )


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "name,value",
    [
        ("log_probability", 0.2),
        ("positive_parameter", 0.2),
        ("log1mexp", -0.2),
        ("log_ndtr", -40.0),
    ],
)
def test_emitted_webppl_checked_probability_compiles_and_executes(
    canonical, name, value, tmp_path
):
    source = _rendered(name, "webppl", canonical).decode()
    script = tmp_path / "checked-probability.wppl"
    script.write_text(
        source
        + f'\nvar result = Infer({{method:"MCMC",samples:2,burn:0}},function(){{return model(0,{value},0);}});\ndisplay(JSON.stringify(result.support()));\n'
    )
    completed = run_webppl(script)
    assert json.loads(completed.stdout) == pytest.approx(
        [_expected(name, value)], rel=3e-14
    )


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "name,value",
    [
        ("log_probability", 0.2),
        ("positive_parameter", 0.2),
        ("log1mexp", -0.2),
        ("log_ndtr", -40.0),
    ],
)
def test_emitted_pymc_probability_density_uses_checked_helpers(name, value, canonical):
    import pymc
    import pytensor

    namespace = {"pymc": pymc, "np": np}
    exec(_rendered(name, "pymc", canonical), namespace)
    model = namespace["build_model"](out=0.0, x=value, y=0.3)
    evaluate = pytensor.function([], model.logp(), mode="FAST_COMPILE")
    expected = -0.5 * (0.3 - _expected(name, value)) ** 2 - 0.5 * math.log(2 * math.pi)
    assert float(evaluate()) == pytest.approx(expected, rel=1e-7, abs=1e-7)


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "name,value",
    [
        ("log_probability", 0.2),
        ("positive_parameter", 0.2),
        ("log1mexp", -0.2),
        ("log_ndtr", -40.0),
    ],
)
def test_emitted_church_probability_density_uses_checked_helpers(
    name, value, canonical, tmp_path
):
    from tests.transpile.probes._protocol import Point
    from tests.transpile.probes.church import ChurchProbe

    result = ChurchProbe().evaluate(
        _rendered(name, "church", canonical),
        name,
        [Point(params={}, data={"out": (0.0,), "x": value, "y": 0.3})],
        scratch=tmp_path,
    )
    expected = -0.5 * (0.3 - _expected(name, value)) ** 2 - 0.5 * math.log(2 * math.pi)
    assert result.log_densities == pytest.approx([expected], rel=3e-14, abs=3e-15)


@pytest.mark.parametrize("target", ["pyro", "numpyro", "pymc", "edward2"])
def test_probability_helpers_share_one_ordinary_and_canonical_runtime_fragment(
    target,
):
    ordinary = (
        "runtime_factor_edward2.py"
        if target == "edward2"
        else "runtime_" + target + ".py"
    )

    def definitions(file):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(python_runtime_source(RUNTIME / file).decode()).body
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith(
                ("_qvr_log", "_qvr_probability_check", "_qvr_positive_parameter")
            )
        }

    assert definitions(ordinary)
    assert definitions(ordinary) == definitions("runtime_qiec_" + target + ".py")


@pytest.mark.parametrize("literal", [False, True])
def test_pymc_select_keeps_literal_and_symbolic_boolean_conditions(literal):
    import pymc
    import pytensor
    import pytensor.tensor as pt

    namespace = {}
    tree = ast.parse((RUNTIME / "runtime_pymc.py").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_qvr_select"
    ]
    namespace["pymc"] = pymc
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "runtime_pymc.py", "exec"),
        namespace,
    )
    helper = namespace["_qvr_select"]
    condition = pt.scalar("condition", dtype="bool")
    symbolic = pytensor.function(
        [condition], helper(condition, 3.0, 7.0), mode="FAST_COMPILE"
    )
    constant = pytensor.function([], helper(literal, 3.0, 7.0), mode="FAST_COMPILE")
    assert float(symbolic(literal)) == (3.0 if literal else 7.0)
    assert float(constant()) == (3.0 if literal else 7.0)


LAZY_SOURCE = """object Out : Real 1
define guarded_probability(x : Real) : Real !{} =
    if x < 0.0 then
        return 0.0
    else
        return log_probability(x)
program p(x : Real, y : Real) : Out -> Out
    let mu <- guarded_probability(x)
    observe y <- Normal(loc=mu, scale=1.0)
    return mu
export p
"""


@pytest.mark.parametrize("value", [-0.2, 0.2, 1.2])
def test_checked_probability_conditionals_preserve_reference_and_pyro_lazy_domains(
    value,
):
    import pyro
    from quivers.dsl import loads
    from quivers.qiec.execution import ExecutionFailure

    compiled = loads(LAZY_SOURCE)
    namespace = {"torch": torch, "pyro": pyro}
    exec(transpile(parse(LAZY_SOURCE), target="pyro"), namespace)
    x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    if value > 1:
        with pytest.raises((ValueError, ExecutionFailure)):
            compiled.run("p", data={"out": (0.0,), "x": value, "y": 0.3})
        with pytest.raises(ValueError, match="log_probability"):
            namespace["model"](0.0, x=x, y=torch.tensor(0.3, dtype=torch.float64))
        return
    expected_mean = 0.0 if value < 0 else math.log(value)
    expected = -0.5 * (0.3 - expected_mean) ** 2 - 0.5 * math.log(2 * math.pi)
    reference = compiled.run("p", data={"out": (0.0,), "x": value, "y": 0.3})
    assert reference.log_joint == pytest.approx(expected, rel=1e-14)
    recorded = pyro.poutine.trace(namespace["model"]).get_trace(
        0.0, x=x, y=torch.tensor(0.3, dtype=torch.float64)
    )
    score = recorded.log_prob_sum()
    assert float(score.detach()) == pytest.approx(expected, rel=1e-14)
    if value > 0:
        gradient = torch.autograd.grad(score, x)[0]
        assert float(gradient) == pytest.approx(
            (0.3 - math.log(value)) / value, rel=1e-14
        )


@pytest.mark.parametrize("value", [-0.2, 0.2, 1.2])
def test_pymc_checked_probability_conditional_uses_known_data_branch(value):
    import pymc
    import pytensor

    namespace = {"pymc": pymc, "np": np}
    exec(transpile(parse(LAZY_SOURCE), target="pymc"), namespace)
    if value > 1:
        with pytest.raises(ValueError, match="log_probability"):
            namespace["build_model"](out=0.0, x=value, y=0.3)
        return
    model = namespace["build_model"](out=0.0, x=value, y=0.3)
    score = pytensor.function([], model.logp(), mode="FAST_COMPILE")()
    expected_mean = 0.0 if value < 0 else math.log(value)
    expected = -0.5 * (0.3 - expected_mean) ** 2 - 0.5 * math.log(2 * math.pi)
    assert float(score) == pytest.approx(expected, rel=1e-7)


def test_symbolic_checked_probability_branch_is_refused_explicitly_by_pymc():
    import pymc

    source = LAZY_SOURCE.replace(
        "program p(x : Real, y : Real)", "program p(y : Real)"
    ).replace(
        "    let mu <-", "    sample x <- Normal(loc=0.0, scale=1.0)\n    let mu <-"
    )
    namespace = {"pymc": pymc, "np": np}
    exec(transpile(parse(source), target="pymc"), namespace)
    with pytest.raises(ValueError, match="cannot branch on a symbolic Boolean"):
        namespace["build_model"](out=0.0, y=0.3)
