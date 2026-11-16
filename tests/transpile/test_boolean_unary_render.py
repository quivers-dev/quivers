"""Boolean negation survives canonical substitution and target array semantics."""

from functools import lru_cache
import ast
import json
import subprocess

import numpy as np
import pytest
import torch

from quivers.dsl import parse
from quivers.transpile import transpile
from quivers.transpile._pipeline import parser_registry
from tests.transpile._tools import require_scheme, require_tool
from tests.transpile._webppl import run_webppl


SOURCE = """object Out:Real 1
define flipped(x:Real):Real !{} =
    if not (x > 0.0) then
        return 3.0
    else
        return -2.0
program p(x:Real,y:Real):Out->Out
    let mu <- flipped(x)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""

ARRAY_SOURCE = """object Row:FinSet 3
object Out:Real 1
program p(x:Tensor[Real]([Row]),y:Tensor[Real]([Row])):Row->Out
    let flag = not (x > 0.0)
    let mu = select(flag,3.0,-2.0)
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""

NEGATIVE_SOURCE = """object Out:Real 1
program p(x:Real,y:Real):Out->Out
    let mu = -x
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""


@lru_cache
def _rendered(target: str, source: str = SOURCE) -> bytes:
    return transpile(parse(source), target=target)


@pytest.mark.parametrize(
    "target,grammar,spelling",
    [
        ("pyro", "python", "torch.logical_not"),
        ("numpyro", "python", "jnp.logical_not"),
        ("pymc", "python", "_qvr_bool_not"),
        ("edward2", "python", "tf.logical_not"),
        ("webppl", "javascript", "!"),
        ("stan", "stan", "!"),
        ("turing", "julia", "!"),
        ("gen", "julia", "!"),
        ("bugs", "bugs", "equals"),
        ("jags", "jags", "equals"),
        ("church", "scheme", "(not"),
    ],
)
def test_checked_not_call_retains_boolean_operator_and_valid_target_syntax(
    target, grammar, spelling
):
    source = _rendered(target)
    assert spelling in source.decode()
    parser_registry().parse_with_protocol(grammar, source, f"unary.{grammar}")
    if grammar == "python":
        model = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"model", "build_model"}
        )
        assert any(
            isinstance(node, ast.Attribute) and node.attr == "gt"
            for node in ast.walk(model)
        )


@pytest.mark.parametrize("source", [SOURCE, ARRAY_SOURCE])
def test_actual_pyro_scalar_and_lifted_not_match_independent_piecewise_law(source):
    import pyro

    namespace = {"torch": torch, "pyro": pyro}
    exec(_rendered("pyro", source), namespace)
    values = [-1.5, 0.0, 2.0]
    if source == ARRAY_SOURCE:
        x = torch.tensor(values)
        actual = namespace["model"](x=x, y=torch.zeros(3))
        torch.testing.assert_close(actual, torch.tensor([3.0, 3.0, -2.0]))
    else:
        for x in [-1.5, -1e-300, 0.0, 1e-300, 2.0]:
            actual = namespace["model"](torch.zeros(1, 1), x=x, y=torch.tensor(0.0))
            assert float(actual) == (3.0 if x <= 0 else -2.0)


@pytest.mark.parametrize("source", [SOURCE, ARRAY_SOURCE])
def test_actual_pymc_scalar_and_lifted_not_match_independent_piecewise_law(source):
    import pymc
    import pytensor

    namespace = {"pymc": pymc, "np": np}
    exec(_rendered("pymc", source), namespace)
    values = [-1.5, 0.0, 2.0]
    for x in [np.asarray(values)] if source == ARRAY_SOURCE else values:
        observed = np.zeros(3) if source == ARRAY_SOURCE else 0.0
        model = namespace["build_model"](x=x, y=observed)
        actual = pytensor.function(
            [], model.named_vars["mu_value"], mode="FAST_COMPILE"
        )()
        expected = np.where(np.asarray(x) <= 0, 3.0, -2.0)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("source", [SOURCE, ARRAY_SOURCE])
def test_actual_webppl_scalar_and_lifted_not_match_piecewise_law(source, tmp_path):
    x = "[-1.5,0,2]" if source == ARRAY_SOURCE else "-1.5"
    y = "[0,0,0]" if source == ARRAY_SOURCE else "0"
    script = tmp_path / "not.wppl"
    script.write_bytes(
        _rendered("webppl", source)
        + (
            "\nvar result = Infer({method:'enumerate',model:function(){"
            f"return model([0],{x},{y});"
            "}});\ndisplay(result.support());\n"
        ).encode()
    )
    values = json.loads(run_webppl(script).stdout.strip())
    assert values == ([[3, 3, -2]] if source == ARRAY_SOURCE else [3])


def test_actual_stanc_accepts_not_in_the_checked_computation_and_model(tmp_path):
    source = tmp_path / "not.stan"
    source.write_bytes(_rendered("stan"))
    result = subprocess.run(
        [require_tool("stanc"), "--name=boolean_negation", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_actual_pyro_arithmetic_negation_and_its_derivative_are_unchanged():
    import pyro

    namespace = {"torch": torch, "pyro": pyro}
    exec(_rendered("pyro", NEGATIVE_SOURCE), namespace)
    x = torch.tensor(0.7, requires_grad=True)
    actual = namespace["model"](torch.zeros(1, 1), x=x, y=torch.tensor(0.0))
    torch.testing.assert_close(actual, -x)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.tensor(-1.0))


@pytest.mark.parametrize("target", ["turing", "gen"])
@pytest.mark.parametrize("source", [SOURCE, ARRAY_SOURCE, NEGATIVE_SOURCE])
def test_actual_julia_not_computes_scalar_and_array_booleans(target, source, tmp_path):
    # The distribution/observe operations are inert here; deterministic model
    # expressions and both Julia broadcast paths execute in the actual host.
    macro = "model" if target == "turing" else "gen"
    prelude = (
        f"macro {macro}(ex)\n esc(ex)\nend\n"
        "macro trace(ex, label)\n nothing\nend\n"
        "Normal(args...) = nothing\nproduct_distribution(args...) = nothing\n"
        "Base.:~(a,b) = nothing\n"
    )
    calls = (
        "println(join(model([-1.5,0.0,2.0], zeros(3)), ','))"
        if source == ARRAY_SOURCE
        else "for x in [-1.5,0.0,2.0]\n println(model([0.0],x,0.0))\nend"
    )
    script = tmp_path / "negation.jl"
    script.write_text(prelude + _rendered(target, source).decode() + "\n" + calls)
    result = subprocess.run(
        [require_tool("julia"), "--startup-file=no", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    actual = result.stdout.strip().replace(",", "\n").splitlines()
    assert [float(value) for value in actual] == (
        [1.5, 0.0, -2.0] if source == NEGATIVE_SOURCE else [3.0, 3.0, -2.0]
    )


@pytest.mark.parametrize("source", [SOURCE, ARRAY_SOURCE, NEGATIVE_SOURCE])
def test_actual_scheme_not_computes_scalar_and_array_booleans(source, tmp_path):
    calls = (
        "(for-each (lambda (x) (display x) (newline)) (model '(0.0 0.0 0.0) '(-1.5 0.0 2.0)))"
        if source == ARRAY_SOURCE
        else "(for-each (lambda (x) (display (model '(0.0) x 0.0)) (newline)) '(-1.5 0.0 2.0))"
    )
    script = tmp_path / "negation.scm"
    script.write_bytes(
        _rendered("church", source)
        + (
            "\n(define (gaussian . args) '())\n(define (observe . args) '())\n" + calls
        ).encode()
    )
    result = subprocess.run(
        [require_scheme(), "--script", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert [float(value) for value in result.stdout.splitlines()] == (
        [1.5, 0.0, -2.0] if source == NEGATIVE_SOURCE else [3.0, 3.0, -2.0]
    )


@pytest.mark.parametrize("target", ["bugs", "jags"])
@pytest.mark.parametrize("source", [SOURCE, ARRAY_SOURCE, NEGATIVE_SOURCE])
def test_actual_jags_executes_both_graph_negation_dialects(target, source, tmp_path):
    model = tmp_path / "negation.bug"
    model.write_bytes(_rendered(target, source))
    data = tmp_path / "data.R"
    data.write_text(
        "x <- c(-1.5,0.0,2.0)\ny <- c(0.0,0.0,0.0)\n"
        if source == ARRAY_SOURCE
        else "x <- -1.5\ny <- 0.0\n"
    )
    stem = tmp_path / "negation"
    command = tmp_path / "run.cmd"
    command.write_text(
        f'model in "{model}"\ndata in "{data}"\ncompile, nchains(1)\ninitialize\nmonitor mu\nupdate 1\ncoda *, stem("{stem}")\nexit\n'
    )
    result = subprocess.run(
        [require_tool("jags"), str(command)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "ERROR" not in result.stderr.upper(), result.stderr
    values = [
        float(row.split()[1])
        for row in (tmp_path / "negationchain1.txt").read_text().splitlines()
    ]
    assert values == (
        [1.5]
        if source == NEGATIVE_SOURCE
        else [3.0, 3.0, -2.0]
        if source == ARRAY_SOURCE
        else [3.0]
    )


@pytest.mark.parametrize("comparison", ["==", "!="])
@pytest.mark.parametrize("target", ["pyro", "pymc"])
def test_equality_is_a_value_comparison_for_scalar_and_symbolic_hosts(
    target, comparison
):
    source = SOURCE.replace("x > 0.0", f"x {comparison} 0.0")
    if target == "pyro":
        import pyro

        namespace = {"torch": torch, "pyro": pyro}
    else:
        import pymc

        namespace = {"pymc": pymc, "np": np}
    exec(_rendered(target, source), namespace)
    for x in [-1.0, 0.0, 1.0]:
        condition = x == 0 if comparison == "==" else x != 0
        expected = -2.0 if condition else 3.0
        if target == "pyro":
            actual = namespace["model"](torch.zeros(1, 1), x=x, y=torch.tensor(0.0))
            assert float(actual) == expected
        else:
            import pytensor

            model = namespace["build_model"](x=x, y=0.0)
            actual = pytensor.function(
                [], model.named_vars["mu_value"], mode="FAST_COMPILE"
            )()
            assert float(actual) == expected
