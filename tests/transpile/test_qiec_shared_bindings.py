"""Shared value bindings survive every transpile path at their size.

A ``fold`` whose body reads its accumulator twice, and a lambda-bound
name applied to a computed argument, lower to ``LetValue`` bindings. The
four host runtimes bind each one once in the target language, Stan
declares each as a typed local, and a program's plan states each as a
statement of its own. The emitted code thus grows linearly in the
collection's length, and evaluates to the reference machine's value.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec.execution import run_named
from quivers.transpile import Lower, transpile
from quivers.transpile.ir import IRDeterministic
from quivers.transpile.qiec_ir import analyze_qiec_capabilities, capabilities_for_target
from tests.transpile._tools import require_scheme, require_tool
from tests.transpile._webppl import run_webppl

SCHEME_EXECUTABLE = require_scheme()

LENGTH = 24

SHARED = f"""\
define shared(v : Tensor[Real]([{LENGTH}])) : Real !{{}} =
    let total = fold(v, 0.5, acc -> x -> acc * acc * 0.25 + acc * 0.5 + x)
    let lift = y -> y * y + y
    return lift(total + 1.0)
"""

_POINT = tuple(0.05 * (k % 7) - 0.1 for k in range(LENGTH))


def _expected(v: tuple[float, ...]) -> float:
    """The reference value of ``shared``, computed in Python.

    Parameters
    ----------
    v : tuple[float, ...]
        The vector argument.

    Returns
    -------
    float
        What every runtime must produce.
    """
    acc = 0.5
    for x in v:
        acc = acc * acc * 0.25 + acc * 0.5 + x
    lifted = acc + 1.0
    return lifted * lifted + lifted


def test_the_reference_machine_runs_the_shared_bindings() -> None:
    module = lower_qvr_to_qiec(parse(SHARED), module_name="shared")
    assert run_named(module, "shared", (_POINT,)).value == pytest.approx(
        _expected(_POINT), rel=1e-12
    )


def test_the_structural_ir_carries_the_bindings() -> None:
    ir = Lower().forward(parse(SHARED)).module
    assert ir is not None
    computation = next(item for item in ir.computations if item.name == "shared")
    assert '"kind": "let"' in computation.model_dump_json()
    assert not analyze_qiec_capabilities(ir, "pyro")
    # The graph targets inline return/bind chains only, and name the
    # binding form as the feature they lack.
    assert any(
        item.feature == "let" and item.computation == "shared"
        for item in analyze_qiec_capabilities(ir, "jags")
    )


def test_python_runtime_binds_each_accumulator_once() -> None:
    source = transpile(parse(SHARED), target="pyro").decode()
    # One assignment expression per intermediate accumulator and one for
    # the shared argument: the source grows linearly in the length.
    assert source.count(":=") == LENGTH
    namespace: dict[str, object] = {}
    exec(source, namespace)
    assert namespace["qiec_shared"](_POINT) == pytest.approx(  # type: ignore[operator]
        _expected(_POINT), rel=1e-12
    )


def test_webppl_runtime_binds_each_accumulator_once(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "shared.wppl"
    script.write_bytes(
        transpile(parse(SHARED), target="webppl")
        + (
            f"display(JSON.stringify(qiec_shared(Object.freeze({list(_POINT)}), "
            "[], {}, {}, {})));\n"
        ).encode()
    )
    completed = run_webppl(script)
    assert json.loads(completed.stdout) == pytest.approx(_expected(_POINT), rel=1e-12)


def test_julia_runtime_binds_each_accumulator_once(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "shared.jl"
    script.write_text(
        "macro model(expression)\n    esc(expression)\nend\n"
        + transpile(parse(SHARED), target="turing").decode()
        + f"\nprintln(qiec_shared({_POINT}))\n"
    )
    completed = subprocess.run(
        [require_tool("julia"), "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(completed.stdout.strip()) == pytest.approx(
        _expected(_POINT), rel=1e-12
    )


def test_scheme_runtime_binds_each_accumulator_once(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "shared.scm"
    point = " ".join(str(x) for x in _POINT)
    script.write_bytes(
        transpile(parse(SHARED), target="church")
        + f"(display (qiec_shared (_qvr-qiec-tuple (list {point})))) (newline)\n".encode()
    )
    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(completed.stdout.strip()) == pytest.approx(
        _expected(_POINT), rel=1e-12
    )


SCALAR = """\
object Val : Real 1

define lifted(x : Real) : Real !{} =
    let square = y -> y * y + y
    return square(square(x + 1.0))

program prog : Val -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    let b <- lifted(a)
    score s = b
    return a

export prog
"""


def test_stan_declares_each_shared_binding_as_a_typed_local(
    tmp_path: pathlib.Path,
) -> None:
    assert capabilities_for_target("stan").supports("let")
    source = transpile(parse(SCALAR), target="stan").decode()
    function = source[source.index("real qiec_lifted") : source.index("} data")]
    assert function.count("real qv___argument_") == 2
    program = tmp_path / "lifted.stan"
    program.write_text(source)
    subprocess.run(
        [require_tool("stanc"), str(program), "--o", str(tmp_path / "lifted.hpp")],
        check=True,
        capture_output=True,
        text=True,
    )


PROGRAM = """\
object State : FinSet 2
object Step : FinSet 3
object Val : Real 1

program prog : Step -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Step <- Normal(loc=mu, scale=1.0)
    let message = fold(y, [mu, 0.5], m -> x -> map(m, v -> m[0] * 0.5 + m[1] * 0.25 + v * x))
    let rows = map(y, x -> fold(y, x * mu, acc -> z -> acc * acc * 0.1 + z))
    score s = sum(message) + sum(rows)
    return mu

export prog
"""


def test_a_plan_states_each_shared_binding_once() -> None:
    plan = Lower().forward(parse(PROGRAM))
    deterministic = [
        node.name for node in plan.body if isinstance(node, IRDeterministic)
    ]
    # Two intermediate messages, the initial message, and the rows' two
    # intermediate accumulators and initial values, each hoisted over the
    # rows' axis, precede the two named lets.
    assert len(deterministic) == 8
    assert deterministic[-1] == "rows"
    assert len(set(deterministic)) == len(deterministic)
    assert not any(name.startswith("_") for name in deterministic)
