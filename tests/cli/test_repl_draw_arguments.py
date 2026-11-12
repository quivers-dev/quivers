"""The REPL prints a draw's arguments as the source writes them.

A scoped ``:info``, ``:browse``, ``:browse`` of a marginalize site, and
``:graph`` each print a program's draws. Every one of them shows keyword,
literal, indexed, list, and nested-distribution arguments in QVR source
form, never as the parser's argument records, and ``:graph`` keeps a
literal argument it would otherwise read only as a dependency edge.
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.cli.repl_session import ReplSession
from quivers.dsl.ast_nodes import ProgramDecl, SampleStep
from quivers.dsl.emit import draw_call_to_source
from quivers.dsl.parser import parse

_SOURCE = """\
object Doc : FinSet 4
object Topic : FinSet 2
object Vocab : FinSet 5
object Token : FinSet 6
object Mix : Real 2

program lda : Token -> Mix
    sample theta : Doc <- Dirichlet(concentration=1.0) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(concentration=0.5) [over=Vocab, iid_over=Topic]
    let theta_tok = theta[doc_idx]
    marginalize z : Topic <- Categorical(probs=theta_tok) [over=Token, reduction=logsumexp]
        observe w : Token <- Categorical(probs=phi[z]) [via=token_idx]
    return theta

export lda
"""

_THETA = "Dirichlet(concentration=1.0)"
_PHI = "Dirichlet(concentration=0.5)"
_Z = "Categorical(probs=theta_tok)"
_W = "Categorical(probs=phi[z])"


@pytest.fixture
def session(tmp_path: pathlib.Path) -> ReplSession:
    path = tmp_path / "lda.qvr"
    path.write_text(_SOURCE)
    repl = ReplSession()
    repl.load_file(path)
    return repl


def _body(session: ReplSession, command: str) -> str:
    response = session.dispatch(command)
    assert response.ok, response.body
    assert "DrawArg" not in response.body, response.body
    return response.body


def test_scoped_info_prints_keyword_arguments(session: ReplSession) -> None:
    assert f"sample theta : Doc <- {_THETA}" in _body(session, ":info lda::theta")


def test_browse_prints_every_draw_as_written(session: ReplSession) -> None:
    body = _body(session, ":browse")
    for call in (_THETA, _PHI, _Z, _W):
        assert call in body


def test_scoped_browse_prints_the_marginalize_scope(session: ReplSession) -> None:
    body = _body(session, ":browse lda::z")
    assert f"marginalize z : Topic <- {_Z}" in body
    assert f"observe w : Token <- {_W}" in body


def test_graph_keeps_literal_arguments(session: ReplSession) -> None:
    body = _body(session, ":graph lda")
    for line in (
        f"sample theta : Doc <- {_THETA}",
        f"sample phi : Topic <- {_PHI}",
        "let theta_tok = …",
        f"marginalize z : Topic <- {_Z}",
        f"observe w : Token <- {_W}",
    ):
        assert line in body


@pytest.mark.parametrize(
    "call",
    [
        "Normal(loc=0.0, scale=1.0)",
        "Normal(mu, 2.5)",
        "Categorical(probs=phi[z])",
        "Categorical(probs=[0.25, 0.75])",
        "Mixture(weights=w, components=[Normal(0.0, 1.0), Normal(m, s)])",
        "kernel",
    ],
)
def test_a_draw_call_reads_back_as_its_source(call: str) -> None:
    source = f"object A : FinSet 2\nprogram p : A -> A\n    sample x <- {call}\n    return x\n"
    (program,) = (
        statement
        for statement in parse(source).statements
        if isinstance(statement, ProgramDecl)
    )
    (step,) = (step for step in program.draws if isinstance(step, SampleStep))
    assert draw_call_to_source(step.morphism, step.args) == call
