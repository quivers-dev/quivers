"""Tests for ``::``-scoped meta-commands in the REPL.

``:info`` / ``:type`` / ``:doc`` / ``:browse`` all route through
the scope-path resolver when their argument contains ``::``. This
file covers the dispatch surface plus the per-kind type lines so
the user can address program steps, deduction rules, signature
sorts, encoder op-rules, etc. with the same commands they already
use for top-level bindings.
"""

from __future__ import annotations

import pytest  # noqa: E402

from quivers.cli.repl_session import ReplSession  # noqa: E402


def _session(path: str) -> ReplSession:
    s = ReplSession()
    s.load_file(path)
    return s


@pytest.fixture
def lda():
    return _session("tests/fixtures/qvr/grouped_topic_mixture.qvr")


@pytest.fixture
def ccg():
    return _session("docs/examples/source/ccg.qvr")


# ---------------------------------------------------------------------------
# :type
# ---------------------------------------------------------------------------


def test_type_scoped_sample_site(lda):
    # GHCi-style: ``theta :: index -> value-space``.
    r = lda.type_of("lda::theta")
    assert r.ok
    assert r.body.startswith("theta ::")
    assert "Doc" in r.body
    assert "Topic" in r.body


def test_type_scoped_marginalize_site(lda):
    r = lda.type_of("lda::z")
    assert r.ok
    assert r.body.startswith("z ::")
    assert "Topic" in r.body


def test_type_scoped_observe_inside_marginalize(lda):
    r = lda.type_of("lda::z::w")
    assert r.ok
    assert r.body.startswith("w ::")
    assert "Token" in r.body


def test_type_scoped_param(lda):
    r = lda.type_of("lda::alpha")
    assert r.ok
    assert r.body == "alpha :: Real"


def test_type_scoped_return(lda):
    r = lda.type_of("lda::return")
    assert r.ok
    assert r.body == "return :: theta"


def test_type_scoped_deduction_rule(ccg):
    r = ccg.type_of("CCG::fwd_app")
    assert r.ok
    assert r.body.startswith("fwd_app :: ")
    assert "|-" in r.body


def test_type_unknown_path_returns_error(lda):
    r = lda.type_of("lda::nonexistent")
    assert not r.ok
    assert "unknown path" in r.diagnostics[0].message


def test_type_bare_name_still_works(lda):
    """The bare-name fast path emits a GHCi-style signature line for
    a program: ``name :: dom -> cod`` (no params, no decl keyword)."""
    r = lda.type_of("lda")
    assert r.ok
    # Per ``docs/semantics/programs.md §3a``, typed program
    # parameters denote a dependent family ``∏ p_i:P_i. Kern(dom, cod)``;
    # they are NOT curried with the kernel arrow. The renderer surfaces
    # them as a Haskell-style constraint context.
    assert r.body == "lda :: (Real, Real) => Token -> Mix"


# ---------------------------------------------------------------------------
# :info
# ---------------------------------------------------------------------------


def test_info_scoped_sample_site(lda):
    r = lda.info("lda::theta")
    assert r.ok
    assert "sample theta" in r.body
    assert "sample-site inside lda" in r.body


def test_info_scoped_observe_in_marginalize(lda):
    r = lda.info("lda::z::w")
    assert r.ok
    assert "observe w" in r.body
    assert "observe-site inside lda::z" in r.body


def test_info_scoped_top_level_falls_through(lda):
    """Calling ``:info lda`` (no separator) still uses the
    existing top-level renderer with verbatim source slicing."""
    r = lda.info("lda")
    assert r.ok
    # The verbatim source includes the full program body.
    assert "program lda(alpha" in r.body
    assert "return theta" in r.body


def test_info_unknown_path_error(lda):
    r = lda.info("lda::nope")
    assert not r.ok


# ---------------------------------------------------------------------------
# :doc
# ---------------------------------------------------------------------------


def test_doc_scoped_returns_step_docs_when_present(lda):
    # No doc comment on lda's sample steps in the gallery file;
    # the doc command returns the "no doc comment" placeholder
    # rather than raising.
    r = lda.doc("lda::theta")
    assert r.ok
    assert "no doc comment" in r.body or r.body.strip()


def test_doc_unknown_path_error(lda):
    r = lda.doc("lda::nope")
    assert not r.ok


# ---------------------------------------------------------------------------
# :browse
# ---------------------------------------------------------------------------


def test_browse_scoped_lists_program_children(lda):
    r = lda.browse("lda")
    assert r.ok
    assert "program lda(alpha : Real, beta : Real) : Token -> Mix" in r.body
    for needle in (
        "param alpha",
        "param beta",
        "sample theta",
        "sample phi",
        "marginalize z",
        "return theta",
    ):
        assert needle in r.body, (needle, r.body)


def test_browse_scoped_lists_marginalize_inner_scope(lda):
    r = lda.browse("lda::z")
    assert r.ok
    assert "marginalize z" in r.body
    assert "observe w" in r.body


def test_browse_scoped_lists_deduction_rules(ccg):
    r = ccg.browse("CCG")
    assert r.ok
    assert "fwd_app" in r.body
    assert "bwd_app" in r.body


def test_browse_no_arg_falls_through_to_module_view(lda):
    """Empty argument keeps the existing module-level browse
    behaviour: lists every populated bucket."""
    r = lda.browse()
    assert r.ok
    assert "objects:" in r.body
    assert "programs:" in r.body


def test_browse_unknown_path_falls_through(lda):
    """A non-resolvable scope path falls through to the
    module-level browse (which will report no matches for that
    namespace name)."""
    r = lda.browse("nonexistent")
    # Falls through to the legacy namespace dispatcher; the legacy
    # path returns an "unknown namespace" error for "nonexistent".
    assert not r.ok or "(empty" in r.body or "objects:" in r.body
