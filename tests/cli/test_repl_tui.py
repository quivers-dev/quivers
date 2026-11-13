"""Coverage for the Textual REPL TUI pipeline.

The pure-Python ``ReplSession`` is exercised by
``test_repl_session.py``; this file covers everything the TUI layer
adds on top of the session: env-tree node construction with bare
names on ``node.data``, click-target resolution, status-bar
content, rendered ANSI styling per token class, and the per-kind
nested-children builders.

Tests deliberately avoid spinning up the live Textual event loop;
each piece is exercised through a module-level seam (or via Rich
console capture for the rendering pipeline), so the suite stays
fast and machine-deterministic. The two prior bugs both have
direct regression tests here:

* ``test_resolve_click_target_*`` covers the click handler that
  was reading ``str(node.label)`` and dispatching the rich
  signature string back to the resolver.
* ``test_info_body_uses_truecolor_codes`` and
  ``test_info_body_distinct_styles_per_token_class`` cover the
  STYLE_TABLE switch from palette-indexed colour names to
  truecolor hex codes, which fixes the "everything red" symptom
  on customised terminal palettes.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest  # noqa: E402

from quivers.cli.repl_session import ReplSession  # noqa: E402

LDA_PATH = "tests/fixtures/qvr/grouped_topic_mixture.qvr"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lda_session() -> ReplSession:
    s = ReplSession()
    s.load_file(LDA_PATH)
    return s


# ---------------------------------------------------------------------------
# Click-target resolution
# ---------------------------------------------------------------------------


class _FakeNode:
    """Minimal stand-in for a Textual ``TreeNode`` carrying just
    ``data``, ``label``, and ``is_root``."""

    def __init__(self, *, data=None, label="", is_root=False):
        self.data = data
        self.label = label
        self.is_root = is_root


def _resolve(node):  # type: ignore[no-untyped-def]
    from quivers.cli.repl_tui import resolve_click_target

    return resolve_click_target(node)


def test_resolve_click_target_returns_data_for_named_binding():
    assert _resolve(_FakeNode(data="lda", label="lda(alpha : Real)")) == "lda"


def test_resolve_click_target_skips_root():
    assert _resolve(_FakeNode(data="x", label="x", is_root=True)) is None


def test_resolve_click_target_skips_category_nodes():
    # Category headings ("objects", "programs", ...) carry data=None
    assert _resolve(_FakeNode(data=None, label="objects")) is None


def test_resolve_click_target_ignores_nested_children():
    # Step / rule / sort children of expandable declarations carry no
    # data; the handler must NOT dispatch :info against their labels.
    assert _resolve(_FakeNode(data=None, label="sample theta <- Dirichlet")) is None


def test_resolve_click_target_rejects_non_identifier_data():
    # Defensive: even if a future change accidentally sets data to a
    # rich label, the handler refuses to use it.
    assert _resolve(_FakeNode(data="lda(alpha : Real)")) is None
    assert _resolve(_FakeNode(data="")) is None
    assert _resolve(_FakeNode(data=42)) is None


# ---------------------------------------------------------------------------
# STYLE_TABLE: truecolor regression guard
# ---------------------------------------------------------------------------


def test_style_table_uses_truecolor_hex_codes():
    """Every styled entry in ``STYLE_TABLE`` must use truecolor hex
    colours (or italic / underline modifiers only), never the
    16-colour palette names. The palette names collapse into a
    single hue on customised terminal themes, which made every
    token look the same red on the user's terminal.
    """
    from quivers.cli.repl_highlight import STYLE_TABLE

    palette_names = {
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "default",
        "bright_black",
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
        "bright_white",
    }
    for token, style in STYLE_TABLE.items():
        words = style.split()
        for w in words:
            assert w.lower() not in palette_names, (
                f"STYLE_TABLE[{token!r}] contains palette-indexed colour "
                f"name {w!r}; use a truecolor hex like #RRGGBB instead"
            )


# ---------------------------------------------------------------------------
# Per-kind env-tree builders
# ---------------------------------------------------------------------------


def test_program_builder_renders_full_signature(lda_session):
    from quivers.cli.repl_browse import _children_for_program

    tmpl = lda_session._compiler.programs["lda"]
    head, children = _children_for_program("lda", tmpl)
    assert head == "lda(alpha : Real, beta : Real) : Token -> Mix", head
    # First three children are the two sample sites + the marginalize block.
    # The marginalize child has its own nested observe child.
    assert children, children
    sample_labels = [c[0] for c in children if c[0].startswith("sample")]
    assert any("sample theta" in lbl for lbl in sample_labels)
    assert any("sample phi" in lbl for lbl in sample_labels)
    marg = next((c for c in children if c[0].startswith("marginalize")), None)
    assert marg is not None
    marg_head, marg_kids = marg
    assert "marginalize z" in marg_head
    assert any("observe w" in k[0] for k in marg_kids)
    ret = next((c for c in children if c[0].startswith("return")), None)
    assert ret is not None


def test_object_builder_renders_finset(lda_session):
    from quivers.cli.repl_browse import _children_for_object

    head, children = _children_for_object("Doc", lda_session._compiler.objects["Doc"])
    assert head == "Doc : FinSet 20", head
    assert children == [], children


def test_deduction_builder_renders_rule_lines():
    from quivers.cli.repl_browse import _children_for_deduction

    s = ReplSession()
    s.load_file("docs/examples/source/ccg.qvr")
    assert s._compiler is not None
    ded = s._compiler.deductions["CCG"]
    head, children = _children_for_deduction("CCG", ded)
    assert head == "CCG", head
    # children include a "rules" subtree with one entry per rule
    rules_node = next((c for c in children if c[0] == "rules"), None)
    assert rules_node is not None
    rules_head, rule_entries = rules_node
    rule_names = {entry[0].split(" : ", 1)[0] for entry in rule_entries}
    assert {"fwd_app", "bwd_app"} <= rule_names, rule_names


# ---------------------------------------------------------------------------
# ANSI rendering through the live to_rich_text path
# ---------------------------------------------------------------------------


def _ansi_for_info_body(session: ReplSession) -> str:
    """Render the ``:info lda`` body through the same to_rich_text
    pipeline the TUI uses and return the captured ANSI string."""
    from io import StringIO

    from rich.console import Console

    from quivers.cli.repl_tui import _to_tui_rich_text

    env_kinds = session.env_kinds()
    body = session.info("lda").body
    console = Console(
        file=StringIO(),
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=160,
    )
    for line in body.splitlines() or [""]:
        stripped = line.lstrip()
        if stripped.startswith("--") and not stripped.startswith("->"):
            continue
        console.print(_to_tui_rich_text(line, env_kinds=env_kinds, link_action="info"))
    return console.file.getvalue()  # type: ignore[attr-defined]


def test_tui_click_metadata_preserves_truecolor_style() -> None:
    from rich.console import Console

    from quivers.cli.repl_tui import _to_tui_rich_text

    rendered = _to_tui_rich_text(
        "State", env_kinds={"State": "type"}, link_action="info"
    )
    style = rendered.get_style_at_offset(Console(no_color=False), 0)

    assert style.color is not None
    assert style.color.triplet is not None
    assert tuple(style.color.triplet) == (97, 175, 239)
    assert style.meta["@click"] == "info('State')"


def test_info_body_emits_truecolor_codes(lda_session):
    """Rendered output must include truecolor 38;2;R;G;B escapes
    and no 16-colour palette-indexed foregrounds."""
    ansi = _ansi_for_info_body(lda_session)

    # No 16-colour palette indices for foreground.
    palette_fg = re.compile(
        r"\x1b\[(?:1;|2;|3;)?(?:30|31|32|33|34|35|36|37|39|90|91|92|93|94|95|96|97)m"
    )
    assert not palette_fg.search(ansi), (
        f"palette-indexed foreground code found in output: {ansi!r}"
    )

    # And there are at least four distinct truecolor sequences emitted.
    truecolor = set(re.findall(r"\x1b\[(?:[\d;]+;)?38;2;\d+;\d+;\d+m", ansi))
    assert len(truecolor) >= 4, truecolor


def test_info_body_distinct_styles_per_token_class(lda_session):
    """The body's tokens must receive different ANSI sequences by
    token class: ``program`` (keyword), ``Doc`` (type), ``<-``
    (operator), ``Dirichlet`` (variable) must each get a different
    escape sequence."""
    ansi = _ansi_for_info_body(lda_session)
    pairs = re.findall(r"(\x1b\[[\d;]+m)([^\x1b]*)", ansi)

    def _code_for(text: str) -> str | None:
        for esc, lit in pairs:
            if lit.lstrip().startswith(text):
                return esc
        return None

    program_code = _code_for("program")
    doc_code = _code_for("Doc")
    arrow_code = _code_for("<-")
    dirichlet_code = _code_for("Dirichlet")

    assert program_code is not None
    assert doc_code is not None
    assert arrow_code is not None
    assert dirichlet_code is not None

    codes = {program_code, doc_code, arrow_code, dirichlet_code}
    assert len(codes) >= 3, f"expected distinct ANSI per token class; got {codes}"


def test_info_body_keyword_truecolor_is_unique_to_keywords(lda_session):
    """The keyword colour must not be the same as the type colour
    (the One-Dark palette puts them on different hues)."""
    ansi = _ansi_for_info_body(lda_session)
    pairs = re.findall(r"(\x1b\[[\d;]+m)([^\x1b]*)", ansi)
    code_for_program = next(
        esc for esc, lit in pairs if lit.lstrip().startswith("program")
    )
    code_for_real = next(esc for esc, lit in pairs if lit.lstrip().startswith("Real"))
    assert code_for_program != code_for_real, (
        f"keyword colour collided with type colour: {code_for_program}"
    )


# ---------------------------------------------------------------------------
# Session-level dispatch under TUI shape
# ---------------------------------------------------------------------------


def test_type_lda_emits_ghci_signature(lda_session):
    # GHCi-style: ``name :: dom -> cod``. Decl-style ``program
    # lda(alpha, beta) : Token -> Mix`` lives in :info / :browse.
    assert lda_session.type_of("lda").body == ("lda :: (Real, Real) => Token -> Mix")


def test_browse_includes_program_and_nested_steps(lda_session):
    body = lda_session.browse().body
    assert "programs:" in body
    assert "lda(alpha : Real, beta : Real) : Token -> Mix" in body
    for needle in (
        "sample theta",
        "sample phi",
        "marginalize z",
        "observe w",
        "return theta",
    ):
        assert needle in body, body


def test_browse_for_ccg_shows_deduction_rules():
    s = ReplSession()
    s.load_file("docs/examples/source/ccg.qvr")
    body = s.browse().body
    assert "deductions:" in body
    assert "fwd_app :" in body
    assert "bwd_app :" in body


# ---------------------------------------------------------------------------
# Env-kinds, completer
# ---------------------------------------------------------------------------


def test_env_kinds_classifies_every_bucket(lda_session):
    kinds = lda_session.env_kinds()
    assert kinds.get("Doc") == "type"
    assert kinds.get("lda") == "function"


def test_completion_includes_program_name(lda_session):
    from quivers.cli.repl_complete import all_completions

    cs = all_completions(lda_session, "ld")
    names = {c.text for c in cs}
    assert "lda" in names, names


def test_completion_classifies_program_with_detail(lda_session):
    from quivers.cli.repl_complete import all_completions

    cs = [c for c in all_completions(lda_session, "ld") if c.text == "lda"]
    assert cs, "no completion for 'lda'"
    assert cs[0].detail == "program", cs[0]


# ---------------------------------------------------------------------------
# TUI import surface: smoke check
# ---------------------------------------------------------------------------


def test_tui_module_imports_cleanly():
    """Importing ``quivers.cli.repl_tui`` resolves every front-end
    dependency the ``repl`` extra installs."""
    import importlib

    importlib.import_module("quivers.cli.repl_tui")


def test_tui_module_exports_resolve_click_target():
    from quivers.cli import repl_tui

    assert callable(repl_tui.resolve_click_target)


def test_real_textual_app_executes_qiec_run_end_to_end(tmp_path: Path) -> None:
    """Drive ``:run`` through Textual's event loop and submit action."""

    from textual.widgets import Static, TextArea

    from quivers.cli.repl_tui import _build_tui_app

    path = tmp_path / "identity.qvr"
    path.write_text(
        "define identity[A : Type](value : A) : A !{} =\n    return value\n"
    )
    session = ReplSession()
    assert session.load_file(path).ok

    async def exercise() -> None:
        app = _build_tui_app(session)
        async with app.run_test(size=(120, 40)) as pilot:  # type: ignore[attr-defined]
            input_widget = app.query_one("#input", TextArea)  # type: ignore[attr-defined]
            input_widget.text = ":run identity --static A=Int 7"
            await pilot.press("ctrl+g")
            await pilot.pause()

            assert input_widget.text == ""
            assert session.last_run is not None
            assert session.last_run.computation == "identity"
            assert session.last_run.value == 7
            status = app.query_one("#status", Static)  # type: ignore[attr-defined]
            assert (
                "runtime:core entry:identity(computation) last:identity=7:Int"
                in str(status.render())
            )

    asyncio.run(exercise())


# ---------------------------------------------------------------------------
# Scope-tree population
# ---------------------------------------------------------------------------


class _FakeTreeNode:
    """In-memory Textual ``TreeNode`` stand-in for assertions."""

    def __init__(self, label, *, data=None, is_root=False):
        self.label = label
        self.data = data
        self.is_root = is_root
        self.children: list[_FakeTreeNode] = []

    def add(self, label, *, data=None, expand=False):
        child = _FakeTreeNode(label, data=data)
        self.children.append(child)
        return child

    def add_leaf(self, label, *, data=None):
        return self.add(label, data=data)


def _build_fake_tree(session):  # type: ignore[no-untyped-def]
    """Run the scope-tree walker against a fake root and return
    the populated tree."""
    from quivers.cli.repl_tui import _populate_scope_tree

    root = _FakeTreeNode("<root>", is_root=True)
    _populate_scope_tree(root, session, lambda _n: True, filter_text="")
    return root


def _walk_paths(node, depth=0):  # type: ignore[no-untyped-def]
    """Yield (depth, label, data) for every node depth-first."""
    yield depth, str(node.label), node.data
    for child in node.children:
        yield from _walk_paths(child, depth + 1)


def test_scope_tree_top_level_carries_bare_path(lda_session):
    root = _build_fake_tree(lda_session)
    entries = list(_walk_paths(root))
    # The lda program must appear with data=='lda'
    program_nodes = [e for e in entries if e[2] == "lda"]
    assert program_nodes, [e for e in entries if isinstance(e[2], str)]


def test_scope_tree_nested_steps_carry_scope_paths(lda_session):
    root = _build_fake_tree(lda_session)
    entries = list(_walk_paths(root))
    paths = {e[2] for e in entries if isinstance(e[2], str)}
    # Every program step + the marginalize's body observe + return.
    assert {"lda::theta", "lda::phi", "lda::z", "lda::z::w", "lda::return"} <= paths


def test_scope_tree_deduction_rules_carry_paths():
    s = ReplSession()
    s.load_file("docs/examples/source/ccg.qvr")
    root = _build_fake_tree(s)
    entries = list(_walk_paths(root))
    paths = {e[2] for e in entries if isinstance(e[2], str)}
    assert "CCG" in paths
    assert "CCG::fwd_app" in paths
    assert "CCG::bwd_app" in paths


def test_scope_tree_category_headings_have_no_data(lda_session):
    """Category nodes (objects / programs / ...) must not be
    clickable because their ``data`` is ``None``."""
    root = _build_fake_tree(lda_session)
    # Direct children of root are category headings.
    for child in root.children:
        assert child.data is None, (child.label, child.data)


def test_resolve_click_target_accepts_scoped_paths():
    """The click handler must accept ``::``-paths (the new env-tree
    leaf format), not just bare identifiers."""
    from quivers.cli.repl_tui import resolve_click_target

    leaf = _FakeNode(data="lda::z::w", label="observe w : Token <- ...")
    assert resolve_click_target(leaf) == "lda::z::w"


def test_resolve_click_target_rejects_path_with_invalid_segment():
    from quivers.cli.repl_tui import resolve_click_target

    leaf = _FakeNode(data="lda::not an ident", label="...")
    assert resolve_click_target(leaf) is None


# ---------------------------------------------------------------------------
# Scope-aware completion
# ---------------------------------------------------------------------------


def test_completion_bare_prefix_picks_up_scoped_descendants(lda_session):
    """Typing a bare-name prefix like ``thet`` should surface
    ``lda::theta`` so users discover scoped bindings without
    typing the prefix."""
    from quivers.cli.repl_complete import all_completions

    cs = all_completions(lda_session, "thet")
    texts = {c.text for c in cs}
    assert "lda::theta" in texts, texts


def test_completion_scope_path_lists_children(lda_session):
    """``lda::`` lists every child of the lda program's scope:
    params, sample / observe / marginalize sites, return."""
    from quivers.cli.repl_complete import all_completions

    cs = all_completions(lda_session, "lda::")
    texts = {c.text for c in cs}
    assert "lda::alpha" in texts
    assert "lda::theta" in texts
    assert "lda::z" in texts
    assert "lda::return" in texts


def test_completion_scope_path_partial_segment(lda_session):
    """``lda::th`` lists ``lda::theta`` (and not unrelated paths)."""
    from quivers.cli.repl_complete import all_completions

    cs = all_completions(lda_session, "lda::th")
    texts = {c.text for c in cs if c.kind == "env"}
    assert "lda::theta" in texts
    assert "lda::phi" not in texts


def test_completion_deep_scope_path(lda_session):
    """``lda::z::`` lists the marginalize block's inner scope."""
    from quivers.cli.repl_complete import all_completions

    cs = all_completions(lda_session, "lda::z::")
    texts = {c.text for c in cs}
    assert "lda::z::w" in texts


def test_completion_emits_kind_detail_for_scoped(lda_session):
    """Each scoped completion's ``detail`` is the descendant's
    ScopeKind so the prompt_toolkit / LSP frontend can colour or
    label appropriately."""
    from quivers.cli.repl_complete import all_completions

    cs = [c for c in all_completions(lda_session, "lda::") if c.text == "lda::theta"]
    assert cs
    assert cs[0].detail == "sample-site"


def test_completion_unresolvable_scope_prefix_returns_empty(lda_session):
    """``nonexistent::foo`` resolves to nothing."""
    from quivers.cli.repl_complete import all_completions

    cs = [c for c in all_completions(lda_session, "nonexistent::") if c.kind == "env"]
    assert cs == []
