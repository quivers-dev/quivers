"""Jupyter kernel unit tests.

The kernel is a thin adapter over :class:`ReplSession`; the bulk of
the behaviour is covered by ``tests/cli/test_repl_session.py``. Here
we exercise the kernel-side glue: cell splitting, prefix extraction
for completion, word extraction for inspect, and the do_execute /
do_complete / do_inspect surface against a directly-instantiated
``QuiversKernel`` (no jupyter_client round-trip required).
"""

from __future__ import annotations

import gc
import warnings
from unittest.mock import patch

import pytest

from quivers.kernel.quivers_kernel import (
    _prefix_at,
    _split_cell,
    _word_at,
)


SOURCE = """\
object X : FinSet 3
object Y : FinSet 4
morphism f : X -> Y [role=parameter]
"""


def _kernel():  # type: ignore[no-untyped-def]
    """Build a kernel without going through IPKernelApp.

    The Kernel base class is happy with no kwargs at construction time
    if we never start its channels; for the do_* method tests that's
    sufficient.
    """
    from quivers.kernel.quivers_kernel import QuiversKernel

    # Disable Kernel's normal session-creation by passing dummy
    # iopub_socket / shell_socket attributes after construction.
    k = QuiversKernel()
    # Capture iopub messages by monkey-patching send_response.
    k._captured = []  # type: ignore[attr-defined]
    k.send_response = lambda socket, msg_type, content: k._captured.append(  # type: ignore[assignment]
        (msg_type, content)
    )
    k.iopub_socket = None
    return k


# ----- splitter -----------------------------------------------------


def test_split_cell_groups_blank_separated_blocks() -> None:
    cell = (
        "object X : FinSet 3\nobject Y : FinSet 4\n"
        "\nmorphism f : X -> Y [role=parameter]\n"
    )
    out = _split_cell(cell)
    assert "object X : FinSet 3\nobject Y : FinSet 4" in out
    assert "morphism f : X -> Y [role=parameter]" in out


def test_split_cell_meta_lines_isolated() -> None:
    cell = ":load file.qvr\n:type X\n"
    out = _split_cell(cell)
    assert ":load file.qvr" in out
    assert ":type X" in out
    # Each meta is its own chunk so the dispatcher sees them
    # individually.
    assert len(out) == 2


def test_split_cell_empty_input() -> None:
    assert _split_cell("") == []
    assert _split_cell("\n\n") == []


def test_split_cell_preserves_multi_line_block() -> None:
    cell = "program p : X -> Y\n    z <- f\n    return z\n"
    out = _split_cell(cell)
    assert out == ["program p : X -> Y\n    z <- f\n    return z"]


# ----- helpers ------------------------------------------------------


def test_prefix_at_walks_back_through_identifier() -> None:
    assert _prefix_at(":lo", 3) == ":lo"
    assert _prefix_at("hello world", 11) == "world"
    assert _prefix_at(":info Source", 12) == "Source"


def test_word_at_extends_left_and_right() -> None:
    assert _word_at(":info Source", 8) == "Source"
    assert _word_at("foo bar baz", 0) == "foo"


# ----- do_execute ---------------------------------------------------


def test_do_execute_runs_meta_and_statement() -> None:
    k = _kernel()
    out = k.do_execute(SOURCE, silent=False)
    assert out["status"] == "ok"
    # Subsequent :type query against the freshly-loaded env.
    k._captured.clear()  # type: ignore[attr-defined]
    out2 = k.do_execute(":type f", silent=False)
    assert out2["status"] == "ok"
    streams = [c for kind, c in k._captured if kind == "stream"]  # type: ignore[attr-defined]
    body = "".join(c["text"] for c in streams)
    assert "latent f" in body or " -> " in body


def test_do_execute_silent_suppresses_stream() -> None:
    k = _kernel()
    k.do_execute(SOURCE, silent=True)
    streams = [c for kind, c in k._captured if kind == "stream"]  # type: ignore[attr-defined]
    assert streams == []


def test_do_execute_error_status() -> None:
    k = _kernel()
    out = k.do_execute("@@@ definitely not qvr @@@", silent=False)
    assert out["status"] == "error"


# ----- do_complete --------------------------------------------------


def test_do_complete_meta_prefix() -> None:
    k = _kernel()
    out = k.do_complete(":lo", 3)
    assert out["status"] == "ok"
    assert ":load" in out["matches"]
    assert out["cursor_start"] == 0
    assert out["cursor_end"] == 3


def test_do_complete_recording_options_uses_the_shared_role_catalog() -> None:
    kernel = _kernel()
    code = "observe y <- Normal(loc=0.0, scale=1.0) [cens"
    reply = kernel.do_complete(code, len(code))
    assert {"cens=", "cens_upper="} <= set(reply["matches"])
    assert reply["cursor_start"] == len(code) - 4


def test_do_complete_env_after_load() -> None:
    k = _kernel()
    k.do_execute(SOURCE, silent=True)
    out = k.do_complete("f", 1)
    assert "f" in out["matches"]


# ----- do_inspect ---------------------------------------------------


def test_do_inspect_returns_decl_info() -> None:
    k = _kernel()
    k.do_execute(SOURCE, silent=True)
    out = k.do_inspect("f", 1)
    assert out["status"] == "ok"
    assert out["found"] is True
    rendered = out["data"]["text/plain"]
    assert "morphism f : X -> Y [role=parameter]" in rendered


def test_do_inspect_unknown_name_not_found() -> None:
    k = _kernel()
    k.do_execute(SOURCE, silent=True)
    out = k.do_inspect("nope_unknown", 12)
    assert out["found"] is False


# ----- :quit --------------------------------------------------------


def test_quit_ends_the_cell_and_leaves_no_unawaited_coroutine() -> None:
    k = _kernel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = k.do_execute(":quit\n\n" + SOURCE, silent=False)
        gc.collect()
    assert out["status"] == "ok"
    assert not [w for w in caught if "never awaited" in str(w.message)]
    # The chunks after `:quit` are not dispatched: `f` is not declared.
    assert k.do_inspect("f", 1)["found"] is False
    streams = [c for kind, c in k._captured if kind == "stream"]  # type: ignore[attr-defined]
    assert [c["name"] for c in streams] == ["stderr"]
    assert "shut the kernel down from the front end" in streams[0]["text"]


def test_quit_in_a_silent_cell_writes_nothing() -> None:
    k = _kernel()
    out = k.do_execute(":quit", silent=True)
    assert out["status"] == "ok"
    assert k._captured == []  # type: ignore[attr-defined]


@pytest.mark.parametrize("silent", [False, True])
@pytest.mark.parametrize("source", ["@@@ invalid @@@", ":unknown-command"])
def test_error_reply_is_complete_and_stops_later_chunks(silent, source) -> None:
    kernel = _kernel()
    reply = kernel.do_execute(source + "\n\nobject Never : FinSet 2", silent=silent)
    assert reply["status"] == "error"
    assert reply["ename"] and reply["evalue"] and reply["traceback"]
    assert not kernel.repl.info("Never").ok
    if silent:
        assert kernel._captured == []
    else:
        assert any(kind == "error" for kind, _ in kernel._captured)


def test_kernel_versions_follow_the_installed_package() -> None:
    from quivers import __version__
    from quivers.kernel.quivers_kernel import QuiversKernel

    assert QuiversKernel.implementation_version == __version__
    assert QuiversKernel.language_info["version"] == __version__
    assert "codemirror_mode" not in QuiversKernel.language_info


@pytest.mark.parametrize("silent", [False, True])
def test_kernel_preserves_target_refusal_category_and_message(tmp_path, silent):
    from quivers.cli.check import check_file

    path = tmp_path / "recursive.qvr"
    path.write_text(
        "define count(n : Int) : Int !{} =\n"
        "    if n == 0 then\n"
        "        return 0\n"
        "    else\n"
        "        let rest <- count(n - 1)\n"
        "        return rest + 1\n"
    )
    diagnostics = check_file(path, target="bugs")
    assert diagnostics and all(
        d.code.startswith("qiec:capability:") for d in diagnostics
    )
    kernel = _kernel()
    assert kernel.repl.set_option("target=bugs").ok
    reply = kernel.do_execute(f":load {path}\n\nobject Never : FinSet 2", silent=silent)
    assert reply["status"] == "error"
    assert reply["ename"] == diagnostics[0].code
    assert reply["evalue"] == "\n".join(d.message for d in diagnostics)
    assert reply["traceback"] == [f"{d.code}: {d.message}" for d in diagnostics]
    assert not kernel.repl.info("Never").ok
    if silent:
        assert kernel._captured == []
    else:
        assert [kind for kind, _ in kernel._captured] == ["error"]


def test_notebook_preserves_blank_lines_inside_a_computation() -> None:
    kernel = _kernel()
    reply = kernel.do_execute(
        "define answer() : Int !{} =\n\n    return 42\n:run answer", silent=False
    )
    assert reply["status"] == "ok"
    assert any("42" in str(content) for _, content in kernel._captured)


def test_notebook_completes_transpile_arguments_and_inspects_scoped_names() -> None:
    kernel = _kernel()
    code = ":transpile sta"
    reply = kernel.do_complete(code, len(code))
    assert ":transpile stan" in reply["matches"]
    assert reply["cursor_start"] == 0
    assert _word_at("model::weight", 10) == "model::weight"
    assert _word_at("State.get", 7) == "State.get"


def test_object_prelude_prepares_once_without_changing_chunk_outputs():
    from quivers.cli.repl_session import ReplSession
    from quivers.dsl.qiec_tooling import analyze_module

    source = "\n\n".join(f"object X{i}:FinSet {i + 2}" for i in range(4))
    reference = ReplSession()
    expected = [reference.dispatch(chunk).body + "\n" for chunk in _split_cell(source)]
    kernel = _kernel()
    with patch(
        "quivers.cli.repl_session.analyze_module", wraps=analyze_module
    ) as analyze:
        reply = kernel.do_execute(source, silent=False)
    assert reply["status"] == "ok"
    assert analyze.call_count == 1
    assert len(analyze.call_args.args[0].statements) == 4
    actual = [content["text"] for kind, content in kernel._captured if kind == "stream"]
    assert actual == expected
    assert kernel.repl.module == reference.module


def test_object_batching_keeps_meta_commands_between_their_source_preludes():
    kernel = _kernel()
    reply = kernel.do_execute(
        "object A:FinSet 2\n\nobject B:FinSet 3\n:info A\n"
        "object C:FinSet 4\n\nobject D:FinSet 5\n:info D",
        silent=False,
    )
    assert reply["status"] == "ok"
    outputs = [
        content["text"] for kind, content in kernel._captured if kind == "stream"
    ]
    assert "1 binding(s)" in outputs[0] and "2 binding(s)" in outputs[1]
    assert "object A" in outputs[2]
    assert "3 binding(s)" in outputs[3] and "4 binding(s)" in outputs[4]
    assert "object D" in outputs[5]


@pytest.mark.parametrize("bad", ["object A:FinSet 3", "object B:Missing"])
def test_failed_object_batch_retains_exact_sequential_error_boundary(bad):
    from quivers.cli.repl_session import ReplSession

    source = f"object A:FinSet 2\n\n{bad}\n\nobject Never:FinSet 5"
    reference = ReplSession()
    expected = []
    for chunk in _split_cell(source):
        response = reference.dispatch(chunk)
        expected.append(response)
        if not response.ok:
            break
    kernel = _kernel()
    reply = kernel.do_execute(source, silent=False)
    assert reply["status"] == "error"
    assert kernel.repl.module == reference.module
    assert not kernel.repl.info("Never").ok
    assert reply["evalue"] == "\n".join(d.message for d in expected[-1].diagnostics)
    outputs = [
        content["text"] for kind, content in kernel._captured if kind == "stream"
    ]
    assert outputs == [expected[0].body + "\n"]


def test_object_batches_do_not_reorder_a_forward_reference():
    kernel = _kernel()
    reply = kernel.do_execute(
        "object Alias:Later\n\nobject Later:FinSet 2", silent=True
    )
    assert reply["status"] == "error"
    assert not kernel.repl.info("Later").ok


def test_existing_executable_model_keeps_sequential_compilation_and_rng():
    import torch
    from quivers.cli.repl_session import ReplSession
    from quivers.dsl.qiec_tooling import analyze_module

    addition = "object A:FinSet 2\n\nobject B:FinSet 3"
    with torch.random.fork_rng():
        torch.manual_seed(812)
        reference = ReplSession()
        assert reference.dispatch(SOURCE).ok
        expected = [
            reference.dispatch(chunk).body + "\n" for chunk in _split_cell(addition)
        ]
        expected_rng = torch.rand(5)
        torch.manual_seed(812)
        kernel = _kernel()
        assert kernel.do_execute(SOURCE, silent=True)["status"] == "ok"
        with patch(
            "quivers.cli.repl_session.analyze_module", wraps=analyze_module
        ) as analyze:
            assert kernel.do_execute(addition, silent=False)["status"] == "ok"
        torch.testing.assert_close(torch.rand(5), expected_rng, rtol=0, atol=0)
    assert analyze.call_count == 2
    assert [
        content["text"] for kind, content in kernel._captured if kind == "stream"
    ] == expected


@pytest.mark.parametrize("symbolic", [False, True])
def test_watches_and_symbolic_preludes_keep_per_chunk_installation(symbolic):
    from quivers.dsl.qiec_tooling import analyze_module

    kernel = _kernel()
    if symbolic:
        source = "object Rows:FinSet _\n\nobject Out:Real 1"
    else:
        assert (
            kernel.do_execute("object A:FinSet 2\n:watch A", silent=True)["status"]
            == "ok"
        )
        source = "object B:FinSet 3\n\nobject C:FinSet 4"
    with patch(
        "quivers.cli.repl_session.analyze_module", wraps=analyze_module
    ) as analyze:
        assert kernel.do_execute(source, silent=True)["status"] == "ok"
    assert analyze.call_count == 2
