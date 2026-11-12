"""One entry-point invocation across the command line, the REPL, and Python.

A checked module's entry points are its ``program`` and ``define``
declarations. Each surface invokes them through the same function, so
the value, the log joint, the trace, and the diagnostic codes agree
however an entry is reached.
"""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from quivers.cli.repl_session import ReplSession
from quivers.cli.run import main as run_main
from quivers.dsl import Compiler, load, parse
from quivers.continuous import MonadicProgram
from quivers.qiec import (
    ExecutionFailure,
    RuntimeConfiguration,
    RuntimeSelection,
    entry_point,
    entry_points,
    invoke_entry,
    parse_bindings,
    render_entry,
)
from quivers.qiec.distributions import seed_reference_rng
from quivers.qiec.program_runtime import run_program
from quivers.qiec.types import render_static

SOURCE = """\
object Obs : FinSet 4

define shift(x : Real, k : Real) : Real !{} =
    return x + k

define noisy(x : Real) : Real !{random, score} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    perform score.add(weight(-0.25 * y * y))
    return y

instance random : Random
instance score : Score

program prog : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    let b <- shift(a, 2.0)
    let c <- noisy(b)
    observe y : Obs <- Normal(loc=c, scale=0.5)
    return c
export prog
"""

DATA = (0.1, 0.2, 0.3, 0.4)


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    path = tmp_path / "calls.qvr"
    path.write_text(SOURCE)
    return path


def _run_args(path: Path, **overrides: object) -> Namespace:
    """Arguments for ``qvr run`` as argparse builds them.

    Parameters
    ----------
    path : Path
        The source file.
    **overrides : object
        Argument values replacing the defaults.

    Returns
    -------
    Namespace
        The arguments.
    """
    values: dict[str, object] = {
        "file": str(path),
        "computation": None,
        "arguments": [],
        "list": False,
        "static": [],
        "data": [],
        "site": [],
        "seed": None,
        "runtime": None,
        "fuel": None,
        "trace": False,
        "json": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_entry_points_list_programs_and_computations() -> None:
    compiler = Compiler(parse(SOURCE))
    points = compiler.entry_points()
    assert [point.name for point in points] == ["prog", "shift", "noisy"]
    assert [point.kind for point in points] == ["program", "computation", "computation"]
    program = compiler.entry("prog")
    assert [parameter.name for parameter in program.parameters] == ["y"]
    assert program.parameters[0].role == "observation"
    assert program.sites == ("a",)
    assert render_entry(program) == "program prog(y : Tensor[Real]([4])) : Real sites a"
    noisy = compiler.entry("noisy")
    assert noisy.effects == ("random : Random", "score : Score")
    assert render_entry(compiler.entry("shift")) == (
        "computation shift(x : Real, k : Real) : Real"
    )
    with pytest.raises(ExecutionFailure) as failure:
        compiler.entry("missing")
    assert failure.value.diagnostic.code == "qiec-run-computation"


def test_a_program_entry_draws_the_sites_it_is_not_given() -> None:
    module = Compiler(parse(SOURCE)).qiec_module
    assert module is not None
    forward = invoke_entry(module, "prog", (DATA,), sites={"a": 0.5}, seed=0)
    assert forward.entry.kind == "program"
    assert forward.log_joint is not None
    # The drawn site is the value returned, and replaying it with the
    # given site scores the same joint.
    replayed = run_program(
        module, "prog", data={"y": DATA}, sites={"a": 0.5, "noise": forward.value}
    )
    assert replayed.log_joint == pytest.approx(forward.log_joint)
    assert forward.to_data()["kind"] == "program"
    assert forward.to_data()["result_type"] == "Real"
    assert forward.to_data()["value"] == forward.value
    by_name = invoke_entry(module, "prog", data={"y": DATA}, sites={"a": 0.5})
    assert by_name.entry == forward.entry


def test_a_computation_entry_runs_named() -> None:
    module = Compiler(parse(SOURCE)).qiec_module
    assert module is not None
    run = invoke_entry(module, "shift", (1.0, 2.5))
    assert run.value == 3.5
    assert run.log_joint is None
    assert run.to_data()["kind"] == "computation"
    assert "log_joint" not in run.to_data()


def test_invocation_refusals_carry_stable_codes() -> None:
    module = Compiler(parse(SOURCE)).qiec_module
    assert module is not None
    cases = [
        (dict(arguments=(1.0, 2.0), sites={"a": 1.0}), "shift", "qiec-run-config"),
        (dict(arguments=(1.0, 2.0), data={"x": 1.0}), "shift", "qiec-run-config"),
        (
            dict(arguments=(DATA,), static_arguments=("A=Int",)),
            "prog",
            "qiec-run-static",
        ),
        (
            dict(
                arguments=(DATA,),
                runtime=RuntimeConfiguration((RuntimeSelection("core", {"x": 1}),)),
            ),
            "prog",
            "qiec-run-config",
        ),
        (dict(arguments=(DATA,), data={"y": DATA}), "prog", "qiec-run-config"),
        (dict(arguments=(DATA, 1.0)), "prog", "qiec-run-arity"),
        (dict(), "prog", "qiec-run-arity"),
        (dict(data={"y": DATA, "z": 1.0}), "prog", "qiec-run-config"),
        (dict(arguments=(1.0,)), "missing", "qiec-run-computation"),
    ]
    for options, name, code in cases:
        arguments = options.pop("arguments", ())
        with pytest.raises(ExecutionFailure) as failure:
            invoke_entry(module, name, arguments, **options)  # type: ignore[arg-type]
        assert failure.value.diagnostic.code == code, (name, options)


_TEMPLATE = """\
object School : FinSet 8
object Effect : Real 1
program school_effects(spread : Real, K : FinSet) : K -> Effect
    sample z : K <- Normal(loc=0.0, scale=1.0)
    let effect = spread * z
    return effect
program pooled : School -> Effect
    sample theta <- school_effects(0.6, School)
    sample sigma <- LogNormal(loc=0.0, scale=0.5)
    observe y : School <- Normal(loc=theta, scale=sigma)
    return theta
export pooled
"""


def test_a_template_entry_takes_its_extent_as_a_static_argument() -> None:
    """A template's object parameter is an extent no data fixes, so a
    run supplies it, and the result type is instantiated at it; a
    program whose data fix its extents refuses a static argument that
    disagrees, and a template run without its extent is refused."""
    module = Compiler(parse(_TEMPLATE)).qiec_module
    assert module is not None
    entry = entry_point(module, "school_effects")
    assert entry.statics == ("K",)
    seed_reference_rng(3)
    run = invoke_entry(module, "school_effects", (0.6,), static_arguments=("K=8",))
    assert isinstance(run.value, tuple) and len(run.value) == 8
    assert render_static(run.result.result_type) == "Tensor[Real]([8])"
    assert run.log_joint is not None
    with pytest.raises(ExecutionFailure) as failure:
        invoke_entry(module, "school_effects", (0.6,))
    assert failure.value.diagnostic.code == "qiec-run-config"
    assert "supply it as a static argument" in failure.value.diagnostic.message
    y = tuple(float(v) / 10 for v in range(8))
    pooled = invoke_entry(
        module, "pooled", (), data={"y": y}, sites={"theta$z": y, "sigma": 0.8}
    )
    assert pooled.log_joint == pytest.approx(
        run_program(
            module, "pooled", data={"y": y}, sites={"theta$z": y, "sigma": 0.8}
        ).log_joint
    )


def test_parse_bindings_reads_name_json_pairs() -> None:
    assert parse_bindings(("a=1.5", "y=[1, 2]")) == {"a": 1.5, "y": (1, 2)}
    for bad in (("a",), ("a=1", "a=2"), ("a={",), ("=1",), ("a=null",)):
        with pytest.raises(ValueError):
            parse_bindings(bad)


def test_cli_lists_and_runs_entry_points(source_path: Path, capsys) -> None:
    assert run_main(_run_args(source_path)) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in listed["entries"]] == ["prog", "shift", "noisy"]

    assert run_main(_run_args(source_path, json=False)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "program prog(y : Tensor[Real]([4])) : Real sites a"

    args = _run_args(
        source_path,
        computation="prog",
        arguments=[json.dumps(list(DATA))],
        site=["a=0.5"],
        seed=0,
    )
    assert run_main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "program"
    assert payload["result_type"] == "Real"
    module = Compiler(parse(SOURCE)).qiec_module
    assert module is not None
    expected = invoke_entry(module, "prog", (DATA,), sites={"a": 0.5}, seed=0)
    assert payload["value"] == expected.value
    assert payload["log_joint"] == expected.log_joint

    by_name = _run_args(
        source_path,
        computation="prog",
        data=[f"y={json.dumps(list(DATA))}"],
        site=["a=0.5"],
        seed=0,
        json=False,
    )
    assert run_main(by_name) == 0
    line = capsys.readouterr().out
    assert line.startswith(f"prog = {expected.value!r} : Real  log_joint = ")

    assert (
        run_main(_run_args(source_path, computation="shift", arguments=["1", "2.5"]))
        == 0
    )
    assert json.loads(capsys.readouterr().out)["value"] == 3.5


def test_cli_reports_refusals_with_the_shared_codes(source_path: Path, capsys) -> None:
    failing = [
        (_run_args(source_path, computation="missing"), "qiec-run-computation"),
        (
            _run_args(source_path, computation="shift", arguments=["1"], site=["a=1"]),
            "qiec-run-config",
        ),
        (_run_args(source_path, computation="prog"), "qiec-run-arity"),
        (
            _run_args(source_path, computation="prog", data=["y"]),
            "qiec-run-config",
        ),
        (
            _run_args(source_path, computation="noisy", arguments=["1.0"]),
            "qiec-run-evaluation",
        ),
    ]
    for args, code in failing:
        assert run_main(args) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["diagnostics"][0]["code"] == code


def test_repl_run_matches_the_cli_and_python(source_path: Path) -> None:
    session = ReplSession()
    assert session.load_file(source_path).ok
    listing = session.dispatch(":run")
    assert listing.ok
    assert listing.body.splitlines()[0] == (
        "program prog(y : Tensor[Real]([4])) : Real sites a"
    )
    response = session.dispatch(
        f":run prog {json.dumps(list(DATA), separators=(',', ':'))} "
        "--site a=0.5 --seed 0"
    )
    assert response.ok, response.diagnostics
    summary = json.loads(response.body)
    module = Compiler(parse(SOURCE)).qiec_module
    assert module is not None
    expected = invoke_entry(module, "prog", (DATA,), sites={"a": 0.5}, seed=0)
    assert summary["kind"] == "program"
    assert summary["value"] == expected.value
    assert summary["log_joint"] == expected.log_joint
    assert session.last_run is not None
    assert session.last_run.computation == "prog"

    computation = session.dispatch(":run shift 1 2.5")
    assert computation.ok
    assert json.loads(computation.body) == {
        "kind": "computation",
        "value": 3.5,
        "type": "Real",
        "runtime": "core",
        "trace_events": json.loads(computation.body)["trace_events"],
    }
    for invocation, code in (
        (":run missing 1", "qiec-run-computation"),
        (":run shift 1 --site a=1", "qiec-run-config"),
        (":run prog", "qiec-run-arity"),
        (":run prog --data y", "qiec-run-config"),
    ):
        failure = session.dispatch(invocation)
        assert not failure.ok
        assert failure.diagnostics[0].code == code, invocation


def test_the_compiled_program_runs_the_callee_in_place(source_path: Path) -> None:
    """The program's call of ``noisy``, which only draws and scores, runs
    in place: the compiled joint at ``c$noise`` is the reference run's at
    the callee's own site ``noise``."""
    program = load(source_path)
    assert isinstance(program.morphism, MonadicProgram)
    assert [point.name for point in program.entry_points()] == [
        "prog",
        "shift",
        "noisy",
    ]
    assert program.entry("prog").kind == "program"
    run = program.run("prog", data={"y": DATA}, sites={"a": 0.5}, seed=0)
    joint = program.morphism.log_joint(
        torch.zeros(2, 1),
        {
            "y": torch.tensor(DATA),
            "a": torch.tensor([0.5]),
            "c$noise": torch.tensor([float(run.value)]),
        },
    )
    assert joint.shape == (2,)
    assert float(joint[0]) == pytest.approx(run.log_joint, rel=1e-5)
    assert float(joint[1]) == pytest.approx(run.log_joint, rel=1e-5)
    assert program.morphism.observed_names == {"y"}
    sampled = program.morphism.rsample(
        torch.zeros(1, 1), torch.Size([3]), observations={"y": torch.tensor(DATA)}
    )
    assert sampled.shape == (3, 1, 1)
    assert entry_point(program.qiec, "shift").kind == "computation"  # type: ignore[arg-type]
    assert len(entry_points(program.qiec)) == 3  # type: ignore[arg-type]
