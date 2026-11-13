"""Tests for the 0.12.0 program-exploration meta-commands.

Covers ``:plate``, ``:graph``, ``:where``, ``:effects``,
``:shape`` plus the modal help dialog's category structure. Each
command is exercised through ``ReplSession.dispatch()`` so the
dispatcher's argument parsing is part of the test surface.
"""

from __future__ import annotations

import pytest  # noqa: E402

from quivers.cli.repl_session import ReplSession  # noqa: E402


@pytest.fixture
def lda():
    s = ReplSession()
    s.load_file("tests/fixtures/qvr/grouped_topic_mixture.qvr")
    return s


# ---------------------------------------------------------------------------
# :plate
# ---------------------------------------------------------------------------


def test_plate_default_table_view(lda):
    r = lda.dispatch(":plate lda")
    assert r.ok
    assert "plate graph of lda" in r.body
    for needle in ("theta", "phi", "z", "w", "Dirichlet", "Categorical"):
        assert needle in r.body
    # The header carries the program signature: the domain is the
    # token plate and the codomain the value space of the returned
    # topic mixture.
    assert "(Token -> Mix)" in r.body
    # The three plates are listed in the footer.
    assert "Doc" in r.body and "Topic" in r.body and "Token" in r.body
    # Vocab is the value space the tokens range over, not a plate,
    # so it is absent from the footer.
    assert "Vocab" not in r.body


def test_plate_mermaid(lda):
    r = lda.dispatch(":plate lda --mermaid")
    assert r.ok
    assert r.body.startswith("graph TD")
    assert "subgraph" in r.body
    assert "-->" in r.body


def test_plate_dot(lda):
    r = lda.dispatch(":plate lda --dot")
    assert r.ok
    assert "digraph lda" in r.body
    assert "cluster_Doc" in r.body
    assert "cluster_Topic" in r.body


def test_plate_tikz(lda):
    r = lda.dispatch(":plate lda --tikz")
    assert r.ok
    assert "\\begin{tikzpicture}" in r.body
    assert "\\node" in r.body
    assert "\\plate" in r.body


def test_plate_daft(lda):
    r = lda.dispatch(":plate lda --daft")
    assert r.ok
    assert "import daft" in r.body
    assert "build_pgm" in r.body
    assert "add_node" in r.body
    assert "add_edge" in r.body
    assert "add_plate" in r.body


def test_plate_unknown_program(lda):
    r = lda.dispatch(":plate notaprogram")
    assert not r.ok


def test_plate_usage_error(lda):
    r = lda.dispatch(":plate")
    assert not r.ok
    assert "usage" in r.diagnostics[0].message.lower()


def test_plate_unknown_flag(lda):
    r = lda.dispatch(":plate lda --bogus")
    assert not r.ok


# ---------------------------------------------------------------------------
# :graph
# ---------------------------------------------------------------------------


def test_graph_default_step_view(lda):
    r = lda.dispatch(":graph lda")
    assert r.ok
    assert "program lda" in r.body
    # Each step kind + variable appears in the body.
    for needle in (
        "sample theta",
        "sample phi",
        "marginalize z",
        "observe w",
    ):
        assert needle in r.body


def test_graph_mermaid_matches_plate_mermaid(lda):
    # ``:graph`` and ``:plate`` produce the same Mermaid graph
    # (the structure is identical; only the in-TUI rendering
    # differs). This ensures the two commands stay aligned.
    r1 = lda.dispatch(":graph lda --mermaid")
    r2 = lda.dispatch(":plate lda --mermaid")
    assert r1.ok and r2.ok
    assert r1.body == r2.body


def test_graph_usage_error(lda):
    r = lda.dispatch(":graph")
    assert not r.ok


# ---------------------------------------------------------------------------
# :where
# ---------------------------------------------------------------------------


def test_where_finds_top_level_binding(lda):
    r = lda.dispatch(":where Doc")
    assert r.ok
    assert "Doc" in r.body
    assert "object" in r.body


def test_where_finds_scoped_binding(lda):
    r = lda.dispatch(":where theta")
    assert r.ok
    assert "lda::theta" in r.body


def test_where_missing_name(lda):
    r = lda.dispatch(":where nope")
    assert not r.ok


def test_where_usage_error(lda):
    r = lda.dispatch(":where")
    assert not r.ok


# ---------------------------------------------------------------------------
# :effects
# ---------------------------------------------------------------------------


def test_effects_reports_inferred_set(lda):
    r = lda.dispatch(":effects lda")
    assert r.ok
    # LDA has sample sites, an observe, and a marginalize, so all
    # three of Sample / Score / Marginal appear in the inferred set.
    assert "Sample" in r.body
    assert "Score" in r.body
    assert "Marginal" in r.body


def test_effects_unknown_program(lda):
    r = lda.dispatch(":effects nope")
    assert not r.ok


# ---------------------------------------------------------------------------
# :shape
# ---------------------------------------------------------------------------


def test_shape_lists_per_step_metadata(lda):
    r = lda.dispatch(":shape lda")
    assert r.ok
    assert "chain shape" in r.body
    # The header column names appear
    for col in ("depth", "kind", "name", "size"):
        assert col in r.body
    # The step kinds + names appear
    for needle in ("theta", "phi", "z", "w"):
        assert needle in r.body


# ---------------------------------------------------------------------------
# Modal help: category structure
# ---------------------------------------------------------------------------


def test_help_categories_structure():
    from quivers.cli.repl_session import HELP_CATEGORIES

    assert HELP_CATEGORIES, "no help categories registered"
    seen_cmds: set[str] = set()
    for cat, entries in HELP_CATEGORIES:
        assert cat, f"empty category name: {entries}"
        assert entries, f"category {cat} has no entries"
        for cmd, summary in entries:
            assert cmd.startswith(":") or cmd.startswith("("), (cat, cmd)
            assert summary, (cat, cmd)
            seen_cmds.add(cmd.split()[0])
    # Make sure the new exploration commands are documented.
    assert ":plate" in seen_cmds
    assert ":graph" in seen_cmds
    assert ":where" in seen_cmds
    assert ":effects" in seen_cmds
    assert ":shape" in seen_cmds


def test_key_bindings_structure():
    from quivers.cli.repl_session import KEY_BINDINGS

    assert KEY_BINDINGS
    for binding, desc in KEY_BINDINGS:
        assert binding
        assert desc


# ---------------------------------------------------------------------------
# Dispatch table coverage
# ---------------------------------------------------------------------------


def test_every_new_command_is_registered():
    from quivers.cli.repl_session import _META_COMMANDS

    for cmd in (
        "plate",
        "p",
        "graph",
        "g",
        "where",
        "effects",
        "shape",
    ):
        assert cmd in _META_COMMANDS, cmd
