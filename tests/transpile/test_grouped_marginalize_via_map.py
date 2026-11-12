"""A grouped marginalize over the observation plate reads its `via` map.

When a grouped `marginalize` is batched over the same plate its observe
rows live on, the observe's `via` map still decides which group each row
belongs to. Two readings follow from the prior's allocation, and both
execution paths and every rendering target agree on them:

- a prior allocated per group (a per-token LDA's `theta_tok`, one row of
  topic weights per token) weighs row ``n`` by its group's prior row,
  ``theta_tok[via[n]]``, each row drawing its own class;
- a prior shared by every group (one simplex over the classes) gives the
  rows a group collects one shared class, so their log-likelihoods are
  summed within the group before the reduction over the class.

A renderer that took a matching plate for the identity map paired row
``n`` with group ``n``'s prior and reduced each row on its own. The
tests here score the Pyro rendering in process, which needs no
container, against a float64 closed form, and check that the
unmapped reading scores a different joint.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import pyro
import pyro.poutine
import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.transpile import transpile

_LDA = """\
object Doc : FinSet 2
object Topic : FinSet 2
object Vocab : FinSet 3
object Token : FinSet 6
object Mix : Real 2

program lda : Token -> Mix
    sample theta : Doc <- Dirichlet(concentration=1.0) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(concentration=1.0) [over=Vocab, iid_over=Topic]
    let theta_tok = theta[doc_idx]
    marginalize z : Topic <- Categorical(probs=theta_tok) [over=Token, reduction=logsumexp]
        observe w : Token <- Categorical(probs=phi[z]) [via=token_idx]
    return theta

export lda
"""

_SHARED = """\
object Cls : FinSet 2
object Obs : FinSet 6

program shared : Obs -> Obs
    sample pi <- Dirichlet(concentration=1.0) [over=Cls]
    sample mu_low <- Normal(loc=-2.0, scale=1.0)
    sample mu_diff <- HalfNormal(scale=1.0)
    let mu = factor c : Cls in mu_low + c * mu_diff
    marginalize c : Cls <- Categorical(probs=pi) [over=Obs, reduction=logsumexp]
        observe y : Obs <- Normal(loc=mu[c], scale=0.5) [via=g]
    return pi

export shared
"""

#: The word rows in an order other than the tokens': `token_idx` is a
#: permutation, so word `n` belongs to token `token_idx[n]`.
_LDA_DATA: dict[str, list[int]] = {
    "w": [2, 0, 1, 2, 0, 0],
    "doc_idx": [0, 0, 0, 1, 1, 1],
    "token_idx": [3, 0, 5, 1, 4, 2],
}

_LDA_POINTS: list[dict[str, list[list[float]]]] = [
    {
        "theta": [[0.875, 0.125], [0.125, 0.875]],
        "phi": [[0.75, 0.125, 0.125], [0.0625, 0.125, 0.8125]],
    },
    {
        "theta": [[0.125, 0.875], [0.625, 0.375]],
        "phi": [[0.75, 0.125, 0.125], [0.25, 0.5, 0.25]],
    },
]

#: Six rows on the grouping plate itself, three of them collected by
#: group 2 and none by groups 1, 3 and 4.
_SHARED_DATA: dict[str, list[float] | list[int]] = {
    "y": [-2.1, -1.4, 0.3, -0.2, 0.6, -1.9],
    "g": [0, 0, 2, 2, 2, 5],
}

_SHARED_POINTS: list[dict[str, float | list[float]]] = [
    {"pi": [0.3, 0.7], "mu_low": -1.5, "mu_diff": 1.8},
    {"pi": [0.8, 0.2], "mu_low": -0.5, "mu_diff": 0.4},
]

#: The tolerance for a float32 rendering of six rows against the float64
#: closed form: a few float32 ulps of a joint near ten nats per term.
_ATOL = 1e-4


def _f64(value: float | list[float] | list[list[float]]) -> torch.Tensor:
    """A host value in double precision.

    Parameters
    ----------
    value : float | list[float] | list[list[float]]
        The value.

    Returns
    -------
    torch.Tensor
        A float64 tensor.
    """
    return torch.tensor(value, dtype=torch.float64)


def _lda_joint(point: dict[str, list[list[float]]], *, mapped: bool) -> float:
    """The per-token LDA joint in float64.

    Parameters
    ----------
    point : dict[str, list[list[float]]]
        `theta` and `phi`.
    mapped : bool
        Weigh word `n` by its token's document, `theta[doc_idx[token_idx[n]]]`;
        when false, by row `n`'s, `theta[doc_idx[n]]`, the joint with the
        map left unread.

    Returns
    -------
    float
        The Dirichlet priors plus each word's marginal over its topic.
    """
    theta, phi = _f64(point["theta"]), _f64(point["phi"])
    w = torch.tensor(_LDA_DATA["w"])
    rows = theta[torch.tensor(_LDA_DATA["doc_idx"])]
    if mapped:
        rows = rows[torch.tensor(_LDA_DATA["token_idx"])]
    prior = D.Dirichlet(torch.ones(2, dtype=torch.float64)).log_prob(theta).sum()
    prior = prior + D.Dirichlet(torch.ones(3, dtype=torch.float64)).log_prob(phi).sum()
    return float(prior + torch.log((rows * phi[:, w].T).sum(-1)).sum())


def _shared_joint(point: dict[str, float | list[float]], *, grouped: bool) -> float:
    """The shared-class mixture joint in float64.

    Parameters
    ----------
    point : dict[str, float | list[float]]
        `pi`, `mu_low` and `mu_diff`.
    grouped : bool
        Sum each group's rows before the reduction over the class, every
        group of the plate contributing; when false, reduce each row on
        its own, the joint with the map left unread.

    Returns
    -------
    float
        The priors plus the marginal of the rows.
    """
    pi = _f64(point["pi"])
    mu_low, mu_diff = _f64(point["mu_low"]), _f64(point["mu_diff"])
    mu = torch.stack([mu_low, mu_low + mu_diff])
    y = _f64(_SHARED_DATA["y"])
    groups = torch.tensor(_SHARED_DATA["g"])
    per_row = D.Normal(mu, _f64(0.5)).log_prob(y.unsqueeze(-1))
    if grouped:
        accumulated = torch.zeros(6, 2, dtype=torch.float64).index_add(
            0, groups, per_row
        )
    else:
        accumulated = per_row
    marginal = torch.logsumexp(torch.log(pi) + accumulated, dim=-1).sum()
    prior = (
        D.Dirichlet(torch.ones(2, dtype=torch.float64)).log_prob(pi)
        + D.Normal(_f64(-2.0), _f64(1.0)).log_prob(mu_low)
        + D.HalfNormal(_f64(1.0)).log_prob(mu_diff)
    )
    return float(prior + marginal)


def _pyro_joint(
    source: str,
    data: dict[str, torch.Tensor],
    observed: str,
    point: dict[str, torch.Tensor],
) -> float:
    """The Pyro rendering's log joint at a point.

    Parameters
    ----------
    source : str
        The QVR source.
    data : dict[str, torch.Tensor]
        The host data, observed site included.
    observed : str
        The observed site's name.
    point : dict[str, torch.Tensor]
        The latent values.

    Returns
    -------
    float
        The sum of every site's log density and every factor in the
        conditioned trace.
    """
    text = transpile(parse(source), target="pyro").decode()
    namespace: dict[str, object] = {"pyro": pyro, "torch": torch}
    exec(text, namespace)  # noqa: S102 - emitted source under test
    model = namespace["model"]
    assert callable(model)
    conditioned = pyro.poutine.condition(model, data=point)
    kwargs = {name: value for name, value in data.items() if name != observed}
    tr = pyro.poutine.trace(conditioned).get_trace(
        **kwargs, **{observed: data[observed]}
    )
    return float(tr.log_prob_sum())


def _compiled_joint(source: str, values: dict[str, torch.Tensor]) -> float:
    """The compiled runtime's log joint at a point.

    Parameters
    ----------
    source : str
        The QVR source.
    values : dict[str, torch.Tensor]
        Data and latent values together.

    Returns
    -------
    float
        The one-run log joint.
    """
    model = loads(source).morphism
    return float(model.log_joint(torch.zeros(1, 1), values).sum())


type Nested = float | int | tuple["Nested", ...]
"""A number, or nested tuples of numbers."""


def _nested(value: torch.Tensor) -> Nested:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    Nested
        A number, or nested tuples of numbers.
    """
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(item) for item in value)


def _reference_joint(
    source: str, data: dict[str, torch.Tensor], point: dict[str, torch.Tensor]
) -> float:
    """The reference machine's log joint at a point.

    Parameters
    ----------
    source : str
        The QVR source.
    data : dict[str, torch.Tensor]
        The host data, observed site included.
    point : dict[str, torch.Tensor]
        The latent values.

    Returns
    -------
    float
        The checked machine's log joint.
    """
    program = loads(source)
    entry = program.entry_points()[0]
    run = program.run(
        entry.name,
        data={name: _nested(value) for name, value in data.items()},
        sites={name: _nested(value) for name, value in point.items()},
    )
    return float(run.log_joint)


@pytest.mark.parametrize(
    "target", ["stan", "pyro", "numpyro", "pymc", "edward2", "webppl", "church"]
)
def test_every_rendering_target_reads_the_via_map(target: str) -> None:
    emitted = transpile(parse(_LDA), target=target).decode()
    compact = "".join(emitted.split())
    assert any(
        spelling in compact
        for spelling in (
            "[token_idx",
            "token_idx[",
            "theta_tok,token_idx",
            "token_idx))",
            # The rows scattered into their groups through the map.
            ",token_idx,",
            "(token_idx,tf.int32)",
            "token_idx6)",
        )
    ), emitted


def _lda_values(
    point: dict[str, list[list[float]]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The per-token LDA's data and latent values as tensors.

    Parameters
    ----------
    point : dict[str, list[list[float]]]
        `theta` and `phi`.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        The data, then the latents.
    """
    data = {name: torch.tensor(value) for name, value in _LDA_DATA.items()}
    latents = {name: torch.tensor(value) for name, value in point.items()}
    return data, latents


def _shared_values(
    point: dict[str, float | list[float]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The shared-class mixture's data and latent values as tensors.

    Parameters
    ----------
    point : dict[str, float | list[float]]
        `pi`, `mu_low` and `mu_diff`.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        The data, then the latents.
    """
    data = {
        "y": torch.tensor(_SHARED_DATA["y"]),
        "g": torch.tensor(_SHARED_DATA["g"]),
    }
    latents = {name: torch.tensor(value) for name, value in point.items()}
    return data, latents


_SCORERS: dict[
    str, Callable[[str, dict[str, torch.Tensor], dict[str, torch.Tensor]], float]
] = {
    "compiled": lambda source, data, point: _compiled_joint(source, {**data, **point}),
    "reference": _reference_joint,
}


@pytest.mark.parametrize("path", sorted(_SCORERS))
@pytest.mark.parametrize("index", range(len(_LDA_POINTS)))
def test_both_execution_paths_weigh_a_word_by_its_tokens_document(
    path: str, index: int
) -> None:
    data, point = _lda_values(_LDA_POINTS[index])
    expected = _lda_joint(_LDA_POINTS[index], mapped=True)
    assert _SCORERS[path](_LDA, data, point) == pytest.approx(expected, abs=_ATOL)


@pytest.mark.parametrize("index", range(len(_LDA_POINTS)))
def test_pyro_weighs_a_word_by_its_tokens_document(index: int) -> None:
    data, point = _lda_values(_LDA_POINTS[index])
    expected = _lda_joint(_LDA_POINTS[index], mapped=True)
    assert _pyro_joint(_LDA, data, "w", point) == pytest.approx(expected, abs=_ATOL)


@pytest.mark.parametrize("index", range(len(_LDA_POINTS)))
def test_the_lda_oracle_rejects_the_unmapped_joint(index: int) -> None:
    mapped = _lda_joint(_LDA_POINTS[index], mapped=True)
    unmapped = _lda_joint(_LDA_POINTS[index], mapped=False)
    assert not math.isclose(mapped, unmapped, abs_tol=100 * _ATOL)


@pytest.mark.parametrize("path", sorted(_SCORERS))
@pytest.mark.parametrize("index", range(len(_SHARED_POINTS)))
def test_both_execution_paths_share_a_class_within_a_group(
    path: str, index: int
) -> None:
    data, point = _shared_values(_SHARED_POINTS[index])
    expected = _shared_joint(_SHARED_POINTS[index], grouped=True)
    assert _SCORERS[path](_SHARED, data, point) == pytest.approx(expected, abs=_ATOL)


@pytest.mark.parametrize("index", range(len(_SHARED_POINTS)))
def test_pyro_shares_a_class_within_a_group(index: int) -> None:
    data, point = _shared_values(_SHARED_POINTS[index])
    expected = _shared_joint(_SHARED_POINTS[index], grouped=True)
    assert _pyro_joint(_SHARED, data, "y", point) == pytest.approx(expected, abs=_ATOL)


@pytest.mark.parametrize("index", range(len(_SHARED_POINTS)))
def test_the_shared_oracle_rejects_a_class_per_row(index: int) -> None:
    grouped = _shared_joint(_SHARED_POINTS[index], grouped=True)
    per_row = _shared_joint(_SHARED_POINTS[index], grouped=False)
    assert not math.isclose(grouped, per_row, abs_tol=100 * _ATOL)
