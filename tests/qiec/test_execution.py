"""Named QIEC execution and explicit runtime-provider contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import didactic.api as dx
import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    INT,
    CoreRuntimeProvider,
    ExecutionTraceEvent,
    ExecutionTraceRecorder,
    ExecutionFailure,
    RuntimeAttachments,
    RuntimeConfiguration,
    RuntimeProvider,
    RuntimeSelection,
    parse_static_arguments,
    runtime_configuration_from_data,
    run_named,
)
from quivers.qiec.types import TypeExpr


def _module(source: str):
    compiler = Compiler(
        parse(source), module_name="execution", file_path="execution.qvr"
    )
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def test_named_execution_specializes_and_validates_arguments() -> None:
    module = _module(
        "define identity[A : Type](value : A) : A !{} =\n    return value\n"
    )
    statics = parse_static_arguments(module, "identity", ("A=Int",))
    result = run_named(module, "identity", (7,), static_arguments=statics)
    assert result.value == 7
    assert result.result_type == INT
    assert [event.event for event in result.trace] == [
        "run.started",
        "run.specialized",
        "runtime.attached",
        "argument.validated",
        "run.completed",
    ]

    with pytest.raises(ExecutionFailure) as error:
        run_named(module, "identity", (True,), static_arguments=statics)
    assert error.value.diagnostic.code == "qiec-run-argument"


def test_trace_observer_receives_the_same_stable_events_as_the_result() -> None:
    module = _module("define answer() : Int !{} =\n    return 42\n")
    observed = []
    result = run_named(module, "answer", observer=observed.append)
    assert observed == list(result.trace)
    assert [event.sequence for event in observed] == list(range(len(observed)))


def test_polymorphic_computation_refuses_unspecialized_execution() -> None:
    module = _module(
        "define identity[A : Type](value : A) : A !{} =\n    return value\n"
    )
    with pytest.raises(ExecutionFailure) as error:
        run_named(module, "identity", (7,))
    assert error.value.diagnostic.code == "qiec-run-static"
    assert "supply all static arguments" in str(error.value)


def test_specialization_reaches_constructed_gadt_results() -> None:
    module = _module(
        "index Nat = Z | S(Nat)\n\n"
        "family Vec[A : Type](n : Nat) : Type\n"
        "    constructor Nil : Vec[A](Z)\n"
        "    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))\n\n"
        "define singleton[A : Type](value : A) : Vec[A](S(Z)) !{} =\n"
        "    return construct Cons[A, Z](value, construct Nil[A]() as Vec[A](Z)) "
        "as Vec[A](S(Z))\n"
    )
    statics = parse_static_arguments(module, "singleton", ("A=Int",))
    result = run_named(module, "singleton", (7,), static_arguments=statics)
    assert result.to_data()["result_type"] == "Vec[Int, S(Z)]"
    assert result.to_data()["value"]["fields"][0] == 7  # type: ignore[index]


@dataclass
class _AttachmentWitness(RuntimeProvider):
    tables: list[RuntimeAttachments] = field(default_factory=list)
    name: str = "witness"

    def attach(self, module, attachments: RuntimeAttachments) -> None:  # type: ignore[no-untyped-def]
        del module
        self.tables.append(attachments)

    def validator_for(self, type_: TypeExpr):  # type: ignore[no-untyped-def]
        return CoreRuntimeProvider().validator_for(type_)


def test_each_run_receives_fresh_runtime_attachments() -> None:
    module = _module("define answer() : Int !{} =\n    return 42\n")
    witness = _AttachmentWitness()
    runtime = RuntimeConfiguration(selections=(), providers=(witness,))
    assert run_named(module, "answer", runtime=runtime).value == 42
    assert run_named(module, "answer", runtime=runtime).value == 42
    assert len(witness.tables) == 2
    assert witness.tables[0] is not witness.tables[1]


def test_runtime_records_snapshot_options_and_preserve_live_providers() -> None:
    """Record updates retain provider state and reject malformed replacements."""
    options = {"seed": 3}
    selection = RuntimeSelection(provider="core", options=options)
    options["seed"] = 4
    assert selection.options["seed"] == 3
    witness = _AttachmentWitness()
    config = RuntimeConfiguration(selections=(selection,), providers=(witness,))
    assert config.with_(selections=()).instantiate() == (witness,)
    with pytest.raises(dx.ValidationError, match="RuntimeProvider"):
        config.with_(providers=("witness",))
    with pytest.raises(dx.ValidationError, match="cannot be empty"):
        selection.with_(provider="")
    for item in ("", {"name": ""}):
        with pytest.raises(ValueError, match="cannot be empty"):
            runtime_configuration_from_data({"providers": [item]})


def test_trace_records_own_details_and_keep_collectors_independent() -> None:
    """Trace mappings are snapshots and event lists are per recorder."""
    detail = {"runtime": "core"}
    event = ExecutionTraceEvent(
        sequence=0, event="run.started", computation="answer", detail=detail
    )
    detail["runtime"] = "other"
    first = ExecutionTraceRecorder()
    second = ExecutionTraceRecorder()
    first(event)
    assert first.events == [event]
    assert second.events == []
    assert event.to_data()["detail"] == {"runtime": "core"}
    with pytest.raises(TypeError):
        event.detail["runtime"] = "other"
    with pytest.raises(dx.ValidationError, match="ExecutionTraceEvent"):
        first.with_(events=["run.started"])


def test_unknown_provider_has_stable_diagnostic() -> None:
    module = _module("define answer() : Int !{} =\n    return 42\n")
    runtime = RuntimeConfiguration(selections=(RuntimeSelection(provider="missing"),))
    with pytest.raises(ExecutionFailure) as error:
        run_named(module, "answer", runtime=runtime)
    assert error.value.diagnostic.code == "qiec-run-provider"
    assert error.value.diagnostic.to_data()["severity"] == "error"


def test_explicit_provider_name_loads_installed_entry_point(monkeypatch) -> None:
    import quivers.qiec.execution as execution

    loaded_options: list[object] = []

    class Point:
        name = "plugin-runtime"

        @staticmethod
        def load():
            def factory(options):  # type: ignore[no-untyped-def]
                loaded_options.append(dict(options))
                return CoreRuntimeProvider()

            return factory

    class Points(tuple):
        def select(self, *, group, name=None):  # type: ignore[no-untyped-def]
            assert group == "quivers.qiec_runtime"
            return Points(point for point in self if name is None or point.name == name)

    monkeypatch.setattr(execution, "entry_points", lambda: Points((Point(),)))
    module = _module("define answer() : Int !{} =\n    return 42\n")
    runtime = RuntimeConfiguration(
        selections=(RuntimeSelection(provider="plugin-runtime", options={"seed": 3}),)
    )
    assert run_named(module, "answer", runtime=runtime).value == 42
    assert loaded_options == [{"seed": 3}]
