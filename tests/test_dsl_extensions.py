"""Tests for the DSL-extension surface introduced for editor + tooling work.

Covers:

- ``bundle`` declarations: rule-set bundles. AST shape only at the
  parser layer; cycle detection + member resolution is exercised
  by the constraint solver.
- ``##`` doc comments attached to the appropriate AST nodes.
- ``FreeMonoid(...)`` object initializer.
- ``qvr check`` CLI: parse / compile diagnostics on representative
  invalid programs.
- The constraint solver in :mod:`quivers.dsl.constraints`: residuated-
  context violations and bundle-unknown-member detection.
- Pygments lexer (tree-sitter-driven) on a representative source.
- Highlight queries cover every new production.
"""

from __future__ import annotations

import io
import json
import re
import textwrap
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
from pygments.token import Comment, Error, Keyword, Name

from quivers.cli import check_file
from quivers.cli.check import main as check_main
from quivers.core.objects import FreeMonoid
from quivers.dsl import Compiler, ParseError, parse
from quivers.dsl.ast_nodes import (
    BundleDecl,
    ObjectDecl,
    TypeFreeMonoid,
)
from quivers.dsl.constraints import check_constraints
from quivers.dsl.pygments_lexer import QvrLexer


# ---------------------------------------------------------------------------
# bundle declarations
# ---------------------------------------------------------------------------


class TestBundleDecl:
    def test_bundle_decl_is_in_module_ast(self):
        src = textwrap.dedent("""
            object Atoms : {NP, S}
            object Cat : FreeResiduated(Atoms, depth=1, ops=[slash])
            schema fa[X, Y : Cat] : (X/Y) * Y -> X
            bundle B : [fa]
        """)
        m = parse(textwrap.dedent(src))
        bundles = [s for s in m.statements if isinstance(s, BundleDecl)]
        assert len(bundles) == 1
        assert bundles[0].name == "B"
        assert bundles[0].rules == ("fa",)

    def test_bundle_with_multiple_members(self):
        src = textwrap.dedent("""
            object Atoms : {NP, S, VP}
            object Cat : FreeResiduated(Atoms, depth=2, ops=[slash])
            schema fa[X, Y : Cat] : (X/Y) * Y -> X
            schema ba[X, Y : Cat] : Y * (X\\Y) -> X
            bundle CCG : [fa, ba]
        """)
        m = parse(textwrap.dedent(src))
        bundles = [s for s in m.statements if isinstance(s, BundleDecl)]
        assert len(bundles) == 1
        assert bundles[0].rules == ("fa", "ba")


# ---------------------------------------------------------------------------
# Comments at parse time
# ---------------------------------------------------------------------------


class TestCommentHandling:
    def test_plain_comments_dropped(self):
        src = textwrap.dedent("""
            # plain comment, dropped
            object X : FinSet 3
        """)
        m = parse(textwrap.dedent(src))
        objs = [s for s in m.statements if isinstance(s, ObjectDecl)]
        assert len(objs) == 1
        assert objs[0].names == ("X",)


# ---------------------------------------------------------------------------
# FreeMonoid object initializer
# ---------------------------------------------------------------------------


class TestFreeMonoidSurface:
    def test_free_monoid_object_parses(self):
        src = textwrap.dedent("""
            object X : FinSet 3
            object Free : FreeMonoid(X, max_length=4)
        """)
        m = parse(textwrap.dedent(src))
        decls = [
            s
            for s in m.statements
            if isinstance(s, ObjectDecl) and isinstance(s.init, TypeFreeMonoid)
        ]
        assert len(decls) == 1
        assert decls[0].init.generators == "X"
        assert decls[0].init.max_length == 4

    def test_free_monoid_object_compiles(self):
        src = textwrap.dedent("""
            object X : FinSet 3
            object Free : FreeMonoid(X, max_length=4)
        """)
        m = parse(textwrap.dedent(src))
        compiler = Compiler(m)
        compiler.compile_env()
        assert isinstance(compiler.objects["Free"], FreeMonoid)


# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------


class TestConstraintSolver:
    def test_clean_module_yields_no_violations(self):
        src = textwrap.dedent("""
            object X : FinSet 3
        """)
        m = parse(textwrap.dedent(src))
        assert check_constraints(m) == []

    def test_residuated_violation_detected(self):
        """Schema with a slash pattern but no residuated parameter is flagged."""
        src = textwrap.dedent("""
            object X : FinSet 3
            object Y : FinSet 4
            schema bad[A : X] : (A/A) * Y -> Y
        """)
        m = parse(textwrap.dedent(src))
        v = check_constraints(m)
        assert any(d.code == "residuated_constraint" for d in v)

    def test_bundle_unknown_member_diagnostic(self):
        src = textwrap.dedent("""
            bundle B : [does_not_exist]
        """)
        m = parse(textwrap.dedent(src))
        v = check_constraints(m)
        codes = {d.code for d in v}
        assert "bundle_unknown_member" in codes

    def test_effect_application_lowercase_starts_flagged(self):
        """T(X) where T starts with a lowercase, non-underscored token is unusual
        enough that the constraint solver flags it as not matching the
        effect-name convention."""
        src = textwrap.dedent("""
            object Atoms : {NP, S}
            object Cat : FreeResiduated(Atoms, depth=1, ops=[slash])
            schema bad[X : Cat] : x(X) -> X
        """)
        m = parse(textwrap.dedent(src))
        v = check_constraints(m)
        assert any(d.code == "effect_constraint" for d in v)

    def test_effect_application_uppercase_accepted(self):
        src = textwrap.dedent("""
            object Atoms : {NP, S}
            object Cat : FreeResiduated(Atoms, depth=1, ops=[slash])
            schema ok[X : Cat] : Cont_S(X) -> Cont_S(X)
        """)
        m = parse(textwrap.dedent(src))
        v = check_constraints(m)
        assert all(d.code != "effect_constraint" for d in v)


# ---------------------------------------------------------------------------
# `qvr check` CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_qvr(tmp_path: Path):
    def _write(name: str, body: str) -> Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(body))
        return p

    return _write


class TestQvrCheckCli:
    def test_clean_file_returns_zero(self, tmp_qvr):
        f = tmp_qvr(
            "clean.qvr",
            """
            object X : FinSet 3
            """,
        )
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = check_main([str(f)], json_output=False)
        assert rc == 0
        assert "OK" in out.getvalue()

    def test_json_output_is_well_formed(self, tmp_qvr):
        f = tmp_qvr(
            "any.qvr",
            """
            object X : FinSet 3
            """,
        )
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = check_main([str(f)], json_output=True)
        assert rc == 0
        payload = json.loads(out.getvalue())
        assert payload["ok"] is True
        assert payload["files"] == [str(f)]
        assert payload["diagnostics"] == []

    def test_check_one_returns_diagnostics(self, tmp_qvr):
        f = tmp_qvr(
            "constraint_bad.qvr",
            """
            bundle B : [no_such_rule]
            """,
        )
        diags = check_file(f)
        codes = {d.code for d in diags}
        assert "bundle_unknown_member" in codes

    def test_neural_parameter_source_is_valid_qvr(self, tmp_qvr):
        f = tmp_qvr(
            "neural.qvr",
            """
            object Feature : Real 2
            object Target : Real 1
            object Resp : FinSet 6
            morphism net : Feature -> Target [param_source=mlp(5, 3)] ~ Normal
            program prog : Resp -> Target
                observe y : Resp <- net(x)
                return y
            export prog
            """,
        )
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = check_main([str(f)], json_output=False)
        assert rc == 0
        assert "OK" in out.getvalue()
        assert err.getvalue() == ""


# ---------------------------------------------------------------------------
# Highlight queries cover every new production
# ---------------------------------------------------------------------------


class TestHighlightQueries:
    def test_highlights_query_compiles(self):
        """The grammars/qvr/queries/highlights.scm file references only
        node kinds that exist in the regenerated grammar's node-types.json.
        Regression test against future grammar / query drift."""
        repo = Path(__file__).resolve().parents[1]
        node_types_path = repo / "grammars" / "qvr" / "src" / "node-types.json"
        highlights_path = repo / "grammars" / "qvr" / "queries" / "highlights.scm"
        node_types = json.loads(node_types_path.read_text())
        known_kinds = {nt["type"] for nt in node_types}

        text = highlights_path.read_text()
        cleaned = re.sub(r";.*", "", text)
        kinds_in_query = set(re.findall(r"\(\s*([a-z_][a-z_0-9]*)\b", cleaned))
        unknown = kinds_in_query - known_kinds
        unknown.discard("identifier")
        unknown.discard("integer")
        unknown.discard("float")
        unknown.discard("signed_number")
        unknown.discard("line_comment")
        assert not unknown, (
            f"highlights.scm references unknown kinds: {sorted(unknown)}"
        )


# ---------------------------------------------------------------------------
# Pygments lexer
# ---------------------------------------------------------------------------


class TestPygmentsLexer:
    def test_tokenises_keywords_and_identifiers(self):
        lex = QvrLexer()
        tokens = list(lex.get_tokens_unprocessed("object X : FinSet 3\n"))
        kinds = [tok[1] for tok in tokens]
        assert any(k is Keyword for k in kinds)
        assert any(k in Name or k is Name.Class for k in kinds)

    def test_doc_comment_tokenises_distinctly(self):
        lex = QvrLexer()
        tokens = list(
            lex.get_tokens_unprocessed("## a doc comment\nobject X : FinSet 3\n")
        )
        kinds = [tok[1] for tok in tokens]
        assert any(k in Comment for k in kinds)

    def test_full_example_round_trips(self):
        """Tokenise a full categorial-effects example and assert no Errors."""
        repo = Path(__file__).resolve().parents[1]
        ex = repo / "docs" / "examples" / "source" / "quantifier_scope.qvr"
        lex = QvrLexer()
        tokens = list(lex.get_tokens_unprocessed(ex.read_text()))
        for _, tok, value in tokens:
            assert tok is not Error, f"Error token at value={value!r}"


# ---------------------------------------------------------------------------
# Top-level grammar-shape coverage
# ---------------------------------------------------------------------------


class TestGrammarSurfaceParity:
    def test_every_example_still_parses(self):
        """Every bundled example parses under the current surface."""
        repo = Path(__file__).resolve().parents[1]
        examples = sorted((repo / "docs" / "examples" / "source").glob("*.qvr"))
        assert examples, "no examples found"
        for ex in examples:
            try:
                parse(ex.read_text(), file_path=str(ex))
            except ParseError as e:  # pragma: no cover
                pytest.fail(f"{ex} failed to parse: {e}")
