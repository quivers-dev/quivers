"""Tests for the plate-graph extractor.

The extractor's job is to take a compiled QVR program and produce
an immutable :class:`PlateGraph` capturing exactly the structure
that a plate-notation diagram should render: nodes (latent /
observed / marginalized / deterministic), plates (with parents
for nested groupings), and dependency edges. Tests use the gallery
examples (LDA, factor analysis, BNN, ...) plus targeted synthetic
fixtures to lock down every shape.
"""

from __future__ import annotations

import textwrap

import pytest  # noqa: E402

from quivers.analysis.plate_graph import (  # noqa: E402
    Edge,
    Plate,
    PlateNode,
    build_plate_graph,
)
from quivers.dsl import Compiler  # noqa: E402
from quivers.dsl.parser import parse  # noqa: E402


def _compile(src: str) -> Compiler:
    c = Compiler(parse(textwrap.dedent(src)))
    c.compile_env()
    return c


# ---------------------------------------------------------------------------
# Unknown program
# ---------------------------------------------------------------------------


def test_build_returns_none_for_unknown_program():
    c = _compile(
        """
        composition product_fuzzy [level=algebra]
        object A : FinSet 3
        """
    )
    assert build_plate_graph(c, "nope") is None


# ---------------------------------------------------------------------------
# LDA (canonical multi-plate example)
# ---------------------------------------------------------------------------


@pytest.fixture
def lda_graph():
    from quivers.cli.repl_session import ReplSession

    s = ReplSession()
    s.load_file("docs/examples/source/lda.qvr")
    g = build_plate_graph(s._compiler, "lda")
    assert g is not None
    return g


def test_lda_program_metadata(lda_graph):
    assert lda_graph.program_name == "lda"
    assert lda_graph.domain == "Token"
    assert lda_graph.codomain == "Mix"


def test_lda_plates_are_doc_topic_word(lda_graph):
    names = {p.name for p in lda_graph.plates}
    assert names == {"Doc", "Topic", "Token"}, names


def test_lda_plate_cardinalities(lda_graph):
    by_name = {p.name: p for p in lda_graph.plates}
    assert by_name["Doc"].cardinality == 20
    assert by_name["Topic"].cardinality == 3
    assert by_name["Token"].cardinality == 200


def test_lda_word_plate_is_nested_inside_doc(lda_graph):
    """An observe inside a marginalize-over-Doc inherits Doc as
    its grouping plate, so the Token plate (the observe's own
    index) records Doc as its parent."""
    by_name = {p.name: p for p in lda_graph.plates}
    assert by_name["Token"].parent == "Doc"


def test_lda_theta_is_latent_on_doc_plate(lda_graph):
    theta = next(n for n in lda_graph.nodes if n.name == "theta")
    assert theta.kind == "latent"
    assert theta.family == "Dirichlet"
    assert theta.plates == ("Doc",)
    assert theta.args == ("alpha",)


def test_lda_phi_is_latent_on_topic_plate(lda_graph):
    phi = next(n for n in lda_graph.nodes if n.name == "phi")
    assert phi.kind == "latent"
    assert phi.plates == ("Topic",)


def test_lda_z_is_marginalized_on_topic_plate(lda_graph):
    z = next(n for n in lda_graph.nodes if n.name == "z")
    assert z.kind == "marginalized"
    assert z.plates == ("Topic",)


def test_lda_w_is_observed_on_doc_x_word(lda_graph):
    """The inner observe lives on Doc (inherited from the
    enclosing marginalize's over axis) cross-product Token (its own
    index). It is NOT on Topic (the marginalized axis) and NOT on
    word_idx (an index map map, not a plate)."""
    w = next(n for n in lda_graph.nodes if n.name == "w")
    assert w.kind == "observed"
    assert w.plates == ("Doc", "Token")


def test_lda_edges_match_dependency_structure(lda_graph):
    pairs = {(e.src, e.dst) for e in lda_graph.edges}
    assert ("alpha", "theta") in pairs
    assert ("beta", "phi") in pairs
    assert ("theta", "z") in pairs
    assert ("phi", "w") in pairs
    assert ("z", "w") in pairs


def test_lda_kind_partitions_are_consistent(lda_graph):
    assert {n.name for n in lda_graph.latents} == {"theta", "phi"}
    assert {n.name for n in lda_graph.marginalized} == {"z"}
    assert {n.name for n in lda_graph.observed} == {"w"}
    assert lda_graph.deterministic == ()


# ---------------------------------------------------------------------------
# Single-plate model with a let intermediate
# ---------------------------------------------------------------------------


def test_let_step_appears_as_deterministic_node():
    c = _compile(
        """
        composition log_prob [level=algebra]

        object Resp : FinSet 50

        program reg : Resp -> Resp
            sample alpha <- Normal(loc=0.0, scale=5.0)
            sample beta  <- Normal(loc=0.0, scale=2.0)
            sample sigma <- HalfCauchy(scale=1.0)
            let mu = alpha + beta
            observe y : Resp <- Normal(loc=mu, scale=sigma)
            return y
        """
    )
    g = build_plate_graph(c, "reg")
    assert g is not None
    mu = next((n for n in g.nodes if n.name == "mu"), None)
    assert mu is not None
    assert mu.kind == "deterministic"
    # mu depends on alpha and beta
    assert "alpha" in mu.args
    assert "beta" in mu.args


def test_observe_with_index_on_response_plate():
    c = _compile(
        """
        composition log_prob [level=algebra]
        object Resp : FinSet 50

        program reg : Resp -> Resp
            sample mu : Resp <- Normal(loc=0.0, scale=1.0)
            observe y : Resp <- Normal(loc=mu, scale=0.1)
            return mu
        """
    )
    g = build_plate_graph(c, "reg")
    assert g is not None
    plates = {p.name for p in g.plates}
    assert "Resp" in plates
    y = next(n for n in g.nodes if n.name == "y")
    assert y.kind == "observed"
    assert y.plates == ("Resp",)


# ---------------------------------------------------------------------------
# Nested marginalize
# ---------------------------------------------------------------------------


def test_doubly_nested_marginalize():
    """Two stacked marginalize blocks should produce two
    marginalized nodes. Verifies the walker recurses correctly
    and inherits the outer marginalize's ``over`` axis through
    the inner one."""
    c = _compile(
        """
        composition log_prob [level=algebra]
        object Doc : FinSet 20
        object Topic : FinSet 3
        object Sense : FinSet 2
        object Word : FinSet 200

        program lda2 : Word -> Word
            sample theta : Doc <- Dirichlet(concentration=1.0) [over=Topic, iid_over=Doc]
            sample phi : Topic <- Dirichlet(concentration=1.0) [over=Word, iid_over=Topic]
            sample psi : Topic <- Dirichlet(concentration=1.0) [over=Sense, iid_over=Topic]

            marginalize z : Topic <- Categorical(probs=theta) [over=Doc, reduction=logsumexp]
                marginalize s : Sense <- Categorical(probs=psi[z]) [over=Doc, reduction=logsumexp]
                    observe w : Word <- Categorical(probs=phi[z]) [via=word_idx]

            return theta
        """
    )
    g = build_plate_graph(c, "lda2")
    assert g is not None
    assert {n.name for n in g.marginalized} == {"z", "s"}
    w = next(n for n in g.nodes if n.name == "w")
    # w is on Doc (from both marginalize over Doc) + Word (own
    # index); neither marginalized variable's plate appears.
    assert "Doc" in w.plates
    assert "Word" in w.plates
    assert "Topic" not in w.plates
    assert "Sense" not in w.plates


# ---------------------------------------------------------------------------
# Subscript dependencies
# ---------------------------------------------------------------------------


def test_subscript_args_produce_edges_to_both_base_and_index(lda_graph):
    """``Categorical(probs=phi[z])`` produces edges from both ``phi``
    and ``z`` to the observe site."""
    pairs = {(e.src, e.dst) for e in lda_graph.edges}
    assert ("phi", "w") in pairs
    assert ("z", "w") in pairs


# ---------------------------------------------------------------------------
# PlateGraph is a dx.Model: serialisable
# ---------------------------------------------------------------------------


def test_plate_graph_is_dx_model(lda_graph):
    assert hasattr(lda_graph, "__field_specs__")
    assert isinstance(lda_graph.nodes, tuple)
    assert isinstance(lda_graph.plates, tuple)
    assert isinstance(lda_graph.edges, tuple)
    for n in lda_graph.nodes:
        assert isinstance(n, PlateNode)
    for p in lda_graph.plates:
        assert isinstance(p, Plate)
    for e in lda_graph.edges:
        assert isinstance(e, Edge)
