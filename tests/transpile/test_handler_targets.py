"""Authored handlers on the transpile targets.

A target that renders no handler reads the discharged module, so a program that
calls a handled computation transpiles to the statements of the program written
by hand. A target that renders handlers renders their parameters, and a
computation it calls through its QIEC runtime has the reference machine's value.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess

import pytest

from quivers.dsl import Compiler, loads, parse
from quivers.dsl.qiec_tooling import target_diagnostics
from quivers.qiec import dumps, run_named
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile.qiec_ir import (
    analyze_qiec_capabilities,
    capabilities_for_target,
)
from tests._handler_programs import HANDLERS, random_body
from tests._target_runtimes import pymc_log_density, pyro_log_density
from tests.dsl.test_handler_fitting import DISCOURSE, STORE
from tests.dsl.test_handler_parameters import COUNTED, NESTED, SHOTS
from tests.transpile._tools import require_scheme, require_tool
from tests.transpile._webppl import run_webppl

_Y4 = [0.4, -0.3, 0.9, 0.2]


def _compiler(source: str) -> Compiler:
    """A compiled source."""
    compiler = Compiler(parse(source), module_name="targets", file_path="t.qvr")
    compiler.compile()
    return compiler


def _text(source: str, target: str) -> str:
    """An emitted program with its whitespace collapsed."""
    return re.sub(r"\s+", " ", transpile(parse(source), target=target).decode())


# ---------------------------------------------------------------------------
# Targets that render no handler
# ---------------------------------------------------------------------------


def test_checking_the_program_for_stan_accepts_it() -> None:
    compiler = _compiler(STORE)
    assert compiler.qiec_module is not None
    assert target_diagnostics(compiler.qiec_module, parse(STORE), "stan") == ()
    assert analyze_qiec_capabilities(compiler.qiec_module, "stan") == ()


def test_stan_states_the_answers() -> None:
    emitted = transpile(parse(STORE), target="stan")
    assert re.sub(r"\s+", " ", emitted.decode()).strip() == (
        "functions { real qiec_width_from_store(real qv_base) { "
        "return (2.0 + qv_base); } } "
        "data { array [4] real y; } parameters { real <lower = 0> base; }"
        "transformed parameters{ real w = 2.0+base; } model { "
        "target +=normal_lpdf(base | 0,1); for (m_Trial in 1 : 4) { "
        "target +=normal_lpdf(y[m_Trial] | 0,w); } }generated quantities{ "
        "real <lower = 0> base_value = base; }"
    )
    require_tool("stanc")
    completed = subprocess.run(
        ["stanc", "--info", "-"], input=emitted, capture_output=True, timeout=120.0
    )
    assert completed.returncode == 0, completed.stderr.decode()


@pytest.mark.parametrize("target", ["bugs", "jags"])
def test_the_graph_targets_state_the_answers(target: str) -> None:
    text = _text(STORE, target)
    assert "w <- 2+base" in text
    assert "dnorm(0,1/(w*w))" in text
    assert "handle" not in text


def test_jags_compiles_the_emitted_model(tmp_path: pathlib.Path) -> None:
    jags = require_tool("jags")
    (tmp_path / "model.txt").write_bytes(transpile(parse(STORE), target="jags"))
    (tmp_path / "data.txt").write_text('"y" <- c(0.4, -0.3, 0.9, 0.2)\n')
    (tmp_path / "run.txt").write_text(
        'model in "model.txt"\ndata in "data.txt"\ncompile\ninitialize\nexit\n'
    )
    completed = subprocess.run(
        [jags, "run.txt"], cwd=tmp_path, capture_output=True, text=True, timeout=120.0
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ERROR" not in completed.stdout.upper(), completed.stdout


def test_stan_states_a_conditional_of_the_branches() -> None:
    emitted = transpile(parse(DISCOURSE), target="stan")
    text = re.sub(r"\s+", " ", emitted.decode())
    block = text[text.index("transformed parameters") : text.index("model {")]
    assert block.count("?") == 2
    assert "w*" in block.replace(" ", "")
    completed = subprocess.run(
        ["stanc", "--info", "-"], input=emitted, capture_output=True, timeout=120.0
    )
    assert completed.returncode == 0, completed.stderr.decode()


_FOREIGN = (
    HANDLERS
    + """
handler outer for Maybe : Real -> Real [coverage=total, forwards=none, implementation=foreign]
    check(v : Real) resumes 1

define called(a : Real) : Real !{} =
    handle mb with outer in
        let x <- perform mb.check(a)
        return x

object Row : FinSet 4

program fitted : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let m <- called(a)
    observe y : Row <- Normal(loc=m, scale=1.0)
    return y

export fitted
"""
)
_FOREIGN_REASON = (
    "installs the foreign handler `outer`, whose clause `check` a runtime provider "
    "implements"
)


def test_stan_refuses_an_undischarged_call_naming_the_clause() -> None:
    with pytest.raises(UnsupportedConstruct) as raised:
        transpile(parse(_FOREIGN), target="stan")
    assert f"qiec:capability:handle:called: called {_FOREIGN_REASON}" in (
        raised.value.kinds
    )
    assert _FOREIGN_REASON in str(raised.value)


def test_checking_an_undischarged_call_names_the_clause() -> None:
    compiler = _compiler(_FOREIGN)
    assert compiler.qiec_module is not None
    messages = [
        item.message
        for item in target_diagnostics(compiler.qiec_module, parse(_FOREIGN), "stan")
    ]
    assert any(
        f"its handlers cannot be discharged: it {_FOREIGN_REASON}" in message
        for message in messages
    )


def test_the_checked_module_keeps_its_handlers() -> None:
    """Discharge is a reading of the module, not a change of it."""
    compiler = _compiler(STORE)
    assert compiler.qiec_module is not None
    before = dumps(compiler.qiec_module)
    for target in ("stan", "jags", "numpyro"):
        transpile(parse(STORE), target=target)
        target_diagnostics(compiler.qiec_module, parse(STORE), target)
    assert dumps(compiler.qiec_module) == before
    assert any(
        event.event == "handler.entered"
        for event in run_named(compiler.qiec_module, "width_from_store", (1.0,)).trace
    )


def test_a_target_with_handlers_reads_the_module_as_it_is_checked() -> None:
    assert capabilities_for_target("numpyro").supports("handle")
    assert not capabilities_for_target("stan").supports("handle")
    assert "_qvr_qiec_authored[" in transpile(parse(COUNTED), target="pyro").decode()


# ---------------------------------------------------------------------------
# Python targets score the program
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["pyro", "pymc"])
@pytest.mark.parametrize("base", [0.4, 1.7])
def test_a_python_target_scores_the_reference_joint(target: str, base: float) -> None:
    emitted = transpile(parse(STORE), target=target)
    reference = (
        loads(STORE)
        .run("uses_handler", data={"y": tuple(_Y4)}, sites={"base": base}, seed=0)
        .log_joint
    )
    assert reference is not None
    score = pyro_log_density if target == "pyro" else pymc_log_density
    assert score(emitted, {"y": _Y4}, {"base": base}) == pytest.approx(
        reference, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Host runtimes run the handlers, parameters included
# ---------------------------------------------------------------------------

#: Each computation, the host arguments it is called with, and its value.
_CALLS: dict[str, tuple[str, tuple[int, ...], int]] = {
    "counted": (NESTED, (3,), 4),
    "crossed": (NESTED, (3, 7), 307),
    "inner_state": (SHOTS, (), 1007 * 7 + 1017),
    "outer_state": (SHOTS, (), (7 * 7 + 18) * 1000 + 8),
}


@pytest.mark.parametrize("name", sorted(_CALLS))
def test_the_reference_machine_gives_each_value(name: str) -> None:
    source, arguments, value = _CALLS[name]
    compiler = _compiler(source)
    assert compiler.qiec_module is not None
    assert run_named(compiler.qiec_module, name, arguments).value == value


@pytest.mark.parametrize("target", ["pyro", "pymc", "edward2"])
def test_python_threads_the_parameters(target: str) -> None:
    for name, (source, arguments, value) in _CALLS.items():
        namespace: dict[str, object] = {}
        exec(transpile(parse(source), target=target), namespace)
        assert namespace[f"qiec_{name}"](*arguments) == value  # type: ignore[operator]


def test_webppl_threads_the_parameters(tmp_path: pathlib.Path) -> None:
    for source in (NESTED, SHOTS):
        calls = {
            name: (arguments, value)
            for name, (owner, arguments, value) in _CALLS.items()
            if owner is source
        }
        script = tmp_path / "parameters.wppl"
        called = ", ".join(
            f"qiec_{name}({', '.join(map(str, arguments))})"
            for name, (arguments, _value) in calls.items()
        )
        script.write_bytes(
            transpile(parse(source), target="webppl")
            + f"display(JSON.stringify([{called}]));\n".encode()
        )
        completed = run_webppl(script)
        assert json.loads(completed.stdout) == [value for _, value in calls.values()]


def test_julia_threads_the_parameters(tmp_path: pathlib.Path) -> None:
    require_tool("julia")
    for source in (NESTED, SHOTS):
        calls = {
            name: (arguments, value)
            for name, (owner, arguments, value) in _CALLS.items()
            if owner is source
        }
        script = tmp_path / "parameters.jl"
        called = ', " ", '.join(
            f"qiec_{name}({', '.join(map(str, arguments))})"
            for name, (arguments, _value) in calls.items()
        )
        script.write_text(
            "macro model(expression)\n    esc(expression)\nend\n"
            + transpile(parse(source), target="turing").decode()
            + f"\nprintln({called})\n"
        )
        completed = subprocess.run(
            ["julia", "--startup-file=no", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.split() == [str(value) for _, value in calls.values()]


def test_scheme_threads_the_parameters(tmp_path: pathlib.Path) -> None:
    scheme = require_scheme()
    for source in (NESTED, SHOTS):
        calls = {
            name: (arguments, value)
            for name, (owner, arguments, value) in _CALLS.items()
            if owner is source
        }
        script = tmp_path / "parameters.scm"
        called = "".join(
            f'(display (qiec_{name} {" ".join(map(str, arguments))})) (display " ")\n'
            for name, (arguments, _value) in calls.items()
        )
        script.write_bytes(
            transpile(parse(source), target="church") + f"\n{called}".encode()
        )
        completed = subprocess.run(
            [scheme, "--script", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.split() == [str(value) for _, value in calls.values()]


@pytest.mark.parametrize("seed", [2, 4, 5, 8, 10, 14])
def test_python_runs_a_random_handler_program(seed: int) -> None:
    """The host's handlers and the reference machine's answer alike."""
    source = HANDLERS + random_body(seed)
    compiler = _compiler(source)
    assert compiler.qiec_module is not None
    namespace: dict[str, object] = {}
    exec(transpile(parse(source), target="pyro"), namespace)
    for point in ((0.3137, -0.7171), (-0.9049, 0.8117)):
        expected = run_named(compiler.qiec_module, "handled", point).value
        actual = namespace["qiec_handled"](*point)  # type: ignore[operator]
        assert float(actual) == pytest.approx(expected, rel=1e-9)
