"""A grouped marginalization's prior reads its free data per group row.

A grouped `marginalize z : K <- ... [over=Token]` batches its latent over
`Token`, so a free name its prior reads through a let ranges over `Token`
as the names its scope reads do. A per-token LDA gathers each token's
topic weights by its document, `theta[doc_idx]`, and `doc_idx` is one
document per token. Both execution paths score the model against a
hand-written float64 joint; a document-level mixture, every token drawn
from the first document's weights, scores a different joint.
"""

from __future__ import annotations

import torch
import torch.distributions as D

import pytest

from quivers.dsl import loads
from quivers.qiec.canonical import tensor_shape
from quivers.qiec.types import IndexLiteral

_SOURCE = """\
object Doc : FinSet 2
object Topic : FinSet 2
object Vocab : FinSet 3
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

_THETA = [[0.75, 0.25], [0.25, 0.75]]
_PHI = [[0.5, 0.25, 0.25], [0.125, 0.125, 0.75]]
_W = [0, 1, 2, 0, 1, 2]
_DOC = [0, 0, 0, 1, 1, 1]


def _joint(doc: list[int]) -> float:
    """The float64 joint with each token reading the weights `doc` names.

    Parameters
    ----------
    doc : list[int]
        The document each token's topic is drawn from.

    Returns
    -------
    float
        The priors plus each token's marginal over its topic.
    """
    theta = torch.tensor(_THETA, dtype=torch.float64)
    phi = torch.tensor(_PHI, dtype=torch.float64)
    index = torch.tensor(doc)
    words = torch.tensor(_W)
    prior = D.Dirichlet(torch.ones(2, dtype=torch.float64)).log_prob(theta).sum()
    prior = (
        prior
        + D.Dirichlet(torch.full((3,), 0.5, dtype=torch.float64)).log_prob(phi).sum()
    )
    return float(prior + torch.log((theta[index] * phi[:, words].T).sum(-1)).sum())


def test_the_gathered_index_is_one_document_per_token() -> None:
    module = loads(_SOURCE).qiec
    assert module is not None
    entry = next(e for e in module.entries if e.name == "lda")
    (parameter,) = (p for p in entry.parameters if p.name == "doc_idx")
    assert parameter.axes == ("Token",)
    shape = tensor_shape(parameter.type)
    assert shape is not None
    (extent,) = shape[1]
    assert isinstance(extent, IndexLiteral)
    assert extent.value == 6


def test_compiled_runtime_scores_a_topic_per_token_from_its_document() -> None:
    program = loads(_SOURCE)
    joint = program.morphism.log_joint(
        torch.zeros(1, 1),
        {
            "theta": torch.tensor(_THETA),
            "phi": torch.tensor(_PHI),
            "w": torch.tensor(_W),
            "doc_idx": torch.tensor(_DOC),
            "token_idx": torch.arange(6),
        },
    )
    assert float(joint) == pytest.approx(_joint(_DOC), rel=1e-5)


def test_reference_machine_scores_a_topic_per_token_from_its_document() -> None:
    run = loads(_SOURCE).run(
        "lda",
        data={"w": tuple(_W), "doc_idx": tuple(_DOC), "token_idx": tuple(range(6))},
        sites={
            "theta": tuple(map(tuple, _THETA)),
            "phi": tuple(map(tuple, _PHI)),
        },
    )
    assert run.log_joint == pytest.approx(_joint(_DOC), rel=1e-9)


def test_a_document_level_mixture_scores_a_different_joint() -> None:
    # Every token drawn from the first document's weights, as a scalar
    # `doc_idx` would read them, misses the per-token joint.
    assert abs(_joint([0] * 6) - _joint(_DOC)) > 0.1
