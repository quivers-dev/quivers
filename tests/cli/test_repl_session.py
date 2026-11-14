"""ReplSession unit tests.

Drive the session directly so the meta-command surface is exercised
without spinning up a TUI or LSP server. Loading from a tmp .qvr file
covers :load / :reload; the bare statement path covers the
appended-statement evaluator; :type / :kind / :browse / :info /
:dump / :set / :help round out the surface.
"""

from __future__ import annotations

from pathlib import Path

from quivers.cli.repl_session import ReplSession


SOURCE = """\
object X : FinSet 3
object Y : FinSet 4
morphism f : X -> Y [role=parameter]
"""


def _populated() -> ReplSession:
    s = ReplSession()
    # Bare statement path; the session compiles incrementally.
    response = s.dispatch(SOURCE)
    assert response.ok, response.diagnostics
    return s


def test_help_lists_commands() -> None:
    s = ReplSession()
    r = s.dispatch(":help")
    assert ":load" in r.body
    assert ":type" in r.body
    assert r.ok


def test_unknown_command() -> None:
    s = ReplSession()
    r = s.dispatch(":nope")
    assert not r.ok
    assert "unknown command" in r.diagnostics[0].message


def test_type_rejects_type_level_name() -> None:
    s = _populated()
    # GHCi semantics: :type on a type-level binding errors and
    # redirects to :kind.
    r = s.dispatch(":type X")
    assert not r.ok
    assert "use :kind" in r.diagnostics[0].message


def test_type_of_morphism() -> None:
    s = _populated()
    r = s.dispatch(":type f")
    assert r.ok, r.diagnostics
    # GHCi-style: ``name :: dom -> cod``, no decl keyword, no
    # ``[role=...]`` annotation.
    assert r.body == "f :: X -> Y"


def test_kind_of_object() -> None:
    s = _populated()
    r = s.dispatch(":kind X")
    assert r.ok, r.diagnostics
    # :kind renders the type-level binding as a QVR object decl so
    # the grammar can classify the constructor + arguments.
    assert r.body.startswith("object X")
    assert "FinSet" in r.body


def test_kind_rejects_value_level_name() -> None:
    s = _populated()
    r = s.dispatch(":kind f")
    assert not r.ok
    assert "use :type" in r.diagnostics[0].message


def test_browse_lists_namespaces() -> None:
    s = _populated()
    r = s.dispatch(":browse")
    assert "objects:" in r.body
    assert "X" in r.body and "Y" in r.body
    assert "morphisms:" in r.body
    assert "f" in r.body


def test_browse_filters_namespace() -> None:
    s = _populated()
    r = s.dispatch(":browse objects")
    assert "objects:" in r.body
    assert "morphisms" not in r.body


def test_info_includes_signature() -> None:
    s = _populated()
    r = s.dispatch(":info f")
    assert r.ok
    assert "morphism f" in r.body
    assert "X -> Y" in r.body


def test_dump_renders_ast() -> None:
    s = _populated()
    r = s.dispatch(":dump f")
    assert "MorphismDecl" in r.body
    assert "names=('f',)" in r.body


def test_dump_json() -> None:
    s = _populated()
    r = s.dispatch(":dump f --json")
    assert r.body.lstrip().startswith("{")
    assert "morphism_decl" in r.body


def test_set_toggles_option() -> None:
    s = ReplSession()
    r = s.dispatch(":set highlight=false")
    assert r.ok
    assert s.options.highlight is False
    r = s.dispatch(":set highlight=on")
    assert s.options.highlight is True


def test_set_rejects_unknown() -> None:
    s = ReplSession()
    r = s.dispatch(":set bogus=42")
    assert not r.ok


def test_load_and_reload(tmp_path: Path) -> None:
    s = ReplSession()
    path = tmp_path / "demo.qvr"
    path.write_text(SOURCE)
    r = s.dispatch(f":load {path}")
    assert r.ok, r.diagnostics
    assert "loaded" in r.body
    assert "X" in s.env and "Y" in s.env
    # Mutate the file: drop Y and reload, expect a removal diff.
    path.write_text("object X : FinSet 3\nmorphism g : X -> X [role=parameter]\n")
    r2 = s.dispatch(":reload")
    assert r2.ok, r2.diagnostics
    assert "removed:" in r2.body and "Y" in r2.body
    assert "added:" in r2.body and "g" in r2.body


def test_load_missing_file() -> None:
    s = ReplSession()
    r = s.dispatch(":load /nonexistent/path.qvr")
    assert not r.ok


def test_parse_error_yields_diagnostic() -> None:
    s = ReplSession()
    r = s.dispatch("@@@ totally bogus @@@")
    assert not r.ok
    # The session must not crash; the error must reach diagnostics.
    assert any(d.severity == "error" for d in r.diagnostics)


def test_quit_returns_sentinel() -> None:
    s = ReplSession()
    r = s.dispatch(":quit")
    assert r.body == "__quit__"


def test_doc_no_comment() -> None:
    s = _populated()
    r = s.dispatch(":doc f")
    # f has no doc comment in the test source.
    assert r.ok
    assert "no doc comment" in r.body


def test_browse_unknown_namespace() -> None:
    s = _populated()
    r = s.dispatch(":browse bogus")
    assert not r.ok


def test_info_python_flag_returns_repr() -> None:
    s = _populated()
    r = s.dispatch(":info f --python")
    assert r.ok
    # The didactic repr always contains the dataclass name.
    assert "MorphismDecl" in r.body


def test_info_default_returns_qvr() -> None:
    s = _populated()
    r = s.dispatch(":info f")
    assert r.ok
    assert "morphism f : X -> Y" in r.body
    # No raw didactic struct in the default rendering.
    assert "MorphismDecl(" not in r.body


def test_env_kinds_classifies_bindings() -> None:
    s = _populated()
    kinds = s.env_kinds()
    assert kinds["X"] == "type"
    assert kinds["Y"] == "type"
    assert kinds["f"] == "function"
    # Unknown names are absent.
    assert "Z" not in kinds


def test_save_writes_module(tmp_path: Path) -> None:
    s = ReplSession()
    src = tmp_path / "in.qvr"
    src.write_text(SOURCE)
    s.dispatch(f":load {src}")
    out = tmp_path / "out.qvr"
    r = s.dispatch(f":save {out}")
    assert r.ok, r.diagnostics
    written = out.read_text()
    assert "object X : FinSet 3" in written
    assert "morphism f : X -> Y" in written


def test_save_without_arg_writes_to_loaded_path(tmp_path: Path) -> None:
    s = ReplSession()
    src = tmp_path / "live.qvr"
    src.write_text(SOURCE)
    s.dispatch(f":load {src}")
    r = s.dispatch(":save")
    assert r.ok
    assert src.read_text().count("object X") == 1


def test_save_without_loaded_file_errors() -> None:
    s = ReplSession()
    r = s.dispatch(":save")
    assert not r.ok


def test_watch_pins_expression() -> None:
    s = _populated()
    r = s.dispatch(":watch X")
    assert r.ok
    assert "watch X" in r.body
    results = s.watch_results()
    assert "X" in results
    assert "object X" in results["X"]


def test_watch_persists_through_module_edits() -> None:
    s = _populated()
    s.dispatch(":watch f")
    # Append another statement and confirm watch still resolves.
    s.dispatch("object Z : FinSet 5")
    results = s.watch_results()
    assert "f" in results
    assert " -> " in results["f"]


def test_unwatch_removes_pin() -> None:
    s = _populated()
    s.dispatch(":watch X")
    r = s.dispatch(":unwatch X")
    assert r.ok
    assert "X" not in s.watch_results()


def test_unwatch_empty_clears_all() -> None:
    s = _populated()
    s.dispatch(":watch X")
    s.dispatch(":watch Y")
    r = s.dispatch(":unwatch")
    assert r.ok
    assert s.watch_results() == {}


def test_unwatch_unknown_errors() -> None:
    s = _populated()
    r = s.dispatch(":unwatch nope")
    assert not r.ok


def test_kind_handles_constructor() -> None:
    s = _populated()
    # Constructor-style sized objects are a primary type-expression
    # shape in the grammar (e.g. ``FinSet 3``). :kind resolves the
    # expression and prints the resulting universe / shape.
    r = s.dispatch(":kind FinSet 3")
    assert r.ok
    assert "FinSet 3" in r.body
    assert "::" in r.body


def test_doc_unknown_name() -> None:
    s = _populated()
    r = s.dispatch(":doc missing")
    assert not r.ok


def test_dump_unknown_name() -> None:
    s = _populated()
    r = s.dispatch(":dump missing")
    assert not r.ok


def test_dump_json_is_valid_json() -> None:
    import json

    s = _populated()
    r = s.dispatch(":dump f --json")
    # Output starts with a `{` and parses as JSON.
    parsed = json.loads(r.body)
    assert parsed["names"] == ["f"]


def test_autoreload_only_when_stale(tmp_path: Path) -> None:
    s = ReplSession()
    src = tmp_path / "w.qvr"
    src.write_text(SOURCE)
    s.dispatch(f":load {src}")
    # First call should be a no-op: mtime hasn't moved.
    assert s.autoreload_if_stale() is None
    # Advance mtime and add a binding; expect reload to fire.
    src.write_text(SOURCE + "object Z : FinSet 2\n")
    import os
    import time

    future = time.time() + 5
    os.utime(src, (future, future))
    r = s.autoreload_if_stale()
    assert r is not None
    assert "Z" in s.env


def test_bare_expression_falls_through_to_type_or_kind() -> None:
    s = _populated()
    # `X` alone isn't a statement form the grammar accepts, but the
    # session's _eval_source falls back to a unified inspector that
    # tries :type first, then :kind. X is type-level, so :kind wins.
    r = s.dispatch("X")
    assert r.ok
    assert r.body.startswith("object X")
    # And for a value-level name, :type wins (GHCi-style sig line).
    r = s.dispatch("f")
    assert r.ok
    assert r.body.startswith("f ::")


def test_bare_statement_extends_module() -> None:
    s = _populated()
    s.dispatch("object Z : FinSet 7")
    assert "Z" in s.env
    assert any(
        "Z" in (getattr(stmt, "names", None) or ()) for stmt in s.module.statements
    )


def test_set_invalid_form() -> None:
    s = ReplSession()
    r = s.dispatch(":set highlight")
    assert not r.ok


def test_help_unknown_command() -> None:
    s = ReplSession()
    r = s.dispatch(":help nope")
    assert not r.ok


def test_help_known_command_renders_detail() -> None:
    s = ReplSession()
    r = s.dispatch(":help load")
    assert r.ok
    assert "Parse" in r.body or "parse" in r.body


def test_type_unknown_name() -> None:
    s = _populated()
    r = s.dispatch(":type definitely_missing")
    assert not r.ok


def test_short_aliases_work() -> None:
    s = _populated()
    assert s.dispatch(":t X").body == s.dispatch(":type X").body
    assert s.dispatch(":i f").body == s.dispatch(":info f").body
    assert s.dispatch(":b").body == s.dispatch(":browse").body
