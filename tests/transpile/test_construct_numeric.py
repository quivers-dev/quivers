"""Construct fixtures that a target can render with the right shape but the wrong value.

Each fixture here isolates one construct whose emission depends on
the shape and origin conventions of a statically typed or one-based
target: a marginalized latent enumerated over its declared index
object, a `let` whose Stan type follows from its use, real division of
real literals, the lets of a marginalize body read at the observe's
row and the latent's atom, index arithmetic on a `factor` binder, a row
gather from a two-axis table, and a two-axis data input read one row at
a time. A program that compiles but reads the wrong entry or the wrong
atom count would still pass a syntax check, so every rendering target
is scored against the reference machine at points that vary every
site, and the reference itself is pinned to an independent
`torch.distributions` computation of the joint.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import pathlib

import pytest
import torch
import torch.distributions as D

from quivers.continuous.programs import MonadicProgram
from quivers.effects.checked_program import CheckedProgram
from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.probes import _protocol
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int]
"""A number or a flat list of numbers, as a point carries it."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint at shaped values."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

#: The image-backed targets this module scores: the statically typed
#: and one-based ones, whose emission these constructs exercise. Church
#: runs in-process.
_TARGETS: tuple[str, ...] = tuple(sorted(_BACKENDS_WITH_IMAGES))

#: The fixtures that marginalize one latent shared by every row of a
#: plated observe.
_UNGROUPED_MARGINALS: tuple[str, ...] = (
    "steps/marginalize_body_let_chain",
    "steps/marginalize_body_plated_let",
    "steps/marginalize_latent_gather",
    "steps/marginalize_latent_value",
    "steps/marginalize_let_probs",
    "steps/marginalize_let_vector_probs",
)


def _poisson_marginal(
    log_weights: torch.Tensor,
    rates: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """The log-sum over atoms of a Poisson body's joint.

    Parameters
    ----------
    log_weights : torch.Tensor
        One log prior weight per atom.
    rates : torch.Tensor
        The rate at each atom (and row, when two-dimensional).
    y : torch.Tensor
        The counts.

    Returns
    -------
    torch.Tensor
        ``logsumexp_k(log_weights[k] + sum_n Poisson(rates[k]).log_prob(y[n]))``.
    """
    per_atom = torch.stack(
        [
            log_weights[k] + D.Poisson(rates[k]).log_prob(y).sum()
            for k in range(log_weights.shape[0])
        ]
    )
    return torch.logsumexp(per_atom, 0)


def _let_probs(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Gamma rate and a uniform four-atom latent setting a Poisson rate `r * (z + 1)`.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    r, y = v["r"], v["y"]
    rates = torch.stack([r * (z + 1) for z in range(4)])
    return D.Gamma(2.0, 1.0).log_prob(r) + _poisson_marginal(
        torch.full((4,), 0.25).log(), rates, y
    )


def _let_vector_probs(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A HalfNormal scale and a uniform latent setting the mean `z - 1.5`.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    s, y = v["s"], v["y"]
    per_atom = torch.stack(
        [
            torch.tensor(0.25).log() + D.Normal(z - 1.5, s).log_prob(y).sum()
            for z in range(4)
        ]
    )
    return D.HalfNormal(1.0).log_prob(s) + torch.logsumexp(per_atom, 0)


def _body_let_chain(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Dirichlet-weighted latent whose body rate is `2 r + 1`.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    r, prior, y = v["r"], v["prior"], v["y"]
    rates = (2.0 * r + 1.0).expand(4)
    return (
        D.Gamma(2.0, 1.0).log_prob(r)
        + D.Dirichlet(torch.ones(4)).log_prob(prior)
        + _poisson_marginal(prior.log(), rates, y)
    )


def _body_plated_let(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Dirichlet-weighted latent whose body rate for row `t` is `r (t + 1)`.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    r, prior, y = v["r"], v["prior"], v["y"]
    row_rates = r * torch.arange(1.0, 4.0)
    rates = row_rates.expand(2, 3)
    return (
        D.Gamma(2.0, 1.0).log_prob(r)
        + D.Dirichlet(torch.ones(2)).log_prob(prior)
        + _poisson_marginal(prior.log(), rates, y)
    )


def _latent_gather(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Dirichlet-weighted latent selecting one sampled location.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    mu, prior, y = v["mu"], v["prior"], v["y"]
    per_atom = torch.stack(
        [prior[z].log() + D.Normal(mu[z], 1.0).log_prob(y).sum() for z in range(4)]
    )
    return (
        D.Normal(0.0, 1.0).log_prob(mu).sum()
        + D.Dirichlet(torch.ones(4)).log_prob(prior)
        + torch.logsumexp(per_atom, 0)
    )


def _latent_value(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Dirichlet-weighted latent setting a Poisson rate `r * (z + 1)`.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    r, prior, y = v["r"], v["prior"], v["y"]
    rates = torch.stack([r * (z + 1) for z in range(3)])
    return (
        D.Gamma(2.0, 1.0).log_prob(r)
        + D.Dirichlet(torch.ones(3)).log_prob(prior)
        + _poisson_marginal(prior.log(), rates, y)
    )


def _real_division(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Normal mean observed with scale `1 / 4`.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    mu, y = v["mu"], v["y"]
    return D.Normal(0.0, 1.0).log_prob(mu) + D.Normal(mu, 0.25).log_prob(y).sum()


def _factor_index_arithmetic(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Normal mean plus the score `sum_t (t + 1) y[2 - t]`.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    mu, y = v["mu"], v["y"]
    score = sum((t + 1) * y[2 - t] for t in range(3))
    return (
        D.Normal(0.0, 1.0).log_prob(mu)
        + D.Normal(mu, 1.0).log_prob(y).sum()
        + torch.as_tensor(score)
    )


def _row_gather(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Two-class rows of a sigmoid table, gathered per observation.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    a, group_idx, y = v["a"], v["group_idx"], v["y"]
    on = torch.sigmoid(a)
    table = torch.stack([1.0 - on, on], -1)
    return (
        D.Normal(0.0, 1.0).log_prob(a).sum()
        + D.Categorical(probs=table[group_idx]).log_prob(y).sum()
    )


def _row_gather_sampled(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Rows of a sampled table of simplexes, gathered per observation.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    table, group_idx, y = v["table"], v["group_idx"], v["y"]
    return (
        D.Dirichlet(torch.ones(3)).log_prob(table).sum()
        + D.Categorical(probs=table[group_idx]).log_prob(y).sum()
    )


def _host_matrix_row(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A regression whose mean for row `n` is `sum_k X[n, k] w[design_idx[n], k]`.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    sigma, w, design_idx, X, y = (
        v["sigma"],
        v["w"],
        v["design_idx"],
        v["X"],
        v["y"],
    )
    mu = (X * w[design_idx]).sum(-1)
    return (
        D.HalfNormal(1.0).log_prob(sigma)
        + D.Normal(0.0, 1.0).log_prob(w).sum()
        + D.Normal(mu, sigma).log_prob(y).sum()
    )


def _plated_list_operand(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Normal mean scaled per row by a literal, and a score over a let
    that adds a literal row by row.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    mu, y = v["mu"], v["y"]
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    total = weights * mu + torch.tensor([2.5, 0.5, 3.5, 1.0])
    return (
        D.Normal(0.0, 1.0).log_prob(mu)
        + D.Normal(mu * weights, 1.0).log_prob(y).sum()
        + (total * total).sum()
    )


def _observe_class_code(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A Dirichlet simplex scoring zero-based class codes.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    probs, y = v["probs"], v["y"]
    return (
        D.Dirichlet(torch.ones(3)).log_prob(probs)
        + D.Categorical(probs=probs).log_prob(y).sum()
    )


def _sample_class_code(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A class drawn per row, read as a subscript and as a number.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    probs, mu, z, x = v["probs"], v["mu"], v["z"], v["x"]
    return (
        D.Dirichlet(torch.ones(3)).log_prob(probs)
        + D.Normal(0.0, 1.0).log_prob(mu).sum()
        + D.Categorical(probs=probs).log_prob(z).sum()
        + D.Normal(mu[z] + z, 1.0).log_prob(x).sum()
    )


def _gather_event_vector(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Each row's class picking one entry of an event vector.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    mu, z, x = v["mu"], v["z"], v["x"]
    return (
        D.Normal(0.0, 2.0).log_prob(mu).sum()
        + D.Categorical(probs=torch.tensor([0.2, 0.3, 0.5])).log_prob(z).sum()
        + D.Normal(mu[z], 1.0).log_prob(x).sum()
    )


def _gather_matrix_row(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Each row's class picking one row of a matrix.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    return _gather_event_vector(v)


def _observe_ordinal_code(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """An ordered-logistic outcome over three zero-based classes.

    Parameters
    ----------
    v : Mapping[str, torch.Tensor]
        The point's values at their declared shapes.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    eta, y = v["eta"], v["y"]
    cut = torch.tensor([-1.0, 0.5])
    above = torch.sigmoid(eta - cut)
    probs = torch.stack([1.0 - above[0], above[0] - above[1], above[1]])
    return D.Normal(0.0, 1.0).log_prob(eta) + probs[y].log().sum()


#: Each fixture's data inputs, as flat lists.
_DATA: dict[str, dict[str, Host]] = {
    "steps/observe_class_code": {"y": [0, 2, 1, 1, 0]},
    "steps/sample_class_code": {"x": [-0.5, 4.0, 1.2, 3.3]},
    "steps/observe_gather_event_vector": {"x": [0.1, 1.7, 1.2, 2.4]},
    "steps/observe_gather_matrix_row": {
        "x": [0.1, -0.3, 1.7, 0.2, 1.2, 1.1, 2.4, -0.8]
    },
    "steps/observe_ordinal_code": {"y": [0, 2, 1, 1]},
    "let_expressions/let_plated_list_operand": {"y": [0.5, 1.0, 2.5, 2.0]},
    "steps/marginalize_let_probs": {"y": [4, 6, 5]},
    "steps/marginalize_let_vector_probs": {"y": [0.4, -1.2, 2.1]},
    "steps/marginalize_body_let_chain": {"y": [1, 3, 4]},
    "steps/marginalize_body_plated_let": {"y": [1, 3, 4]},
    "steps/marginalize_latent_value": {"y": [1, 3, 4]},
    "steps/marginalize_latent_gather": {"y": [0.4, -1.2, 2.1]},
    "let_expressions/let_real_division": {"y": [0.1, -0.2, 0.3]},
    "let_expressions/let_factor_index_arithmetic": {"y": [1.0, 2.0, 3.0]},
    "let_expressions/let_row_gather": {"group_idx": [0, 1, 1, 0], "y": [0, 1, 0, 1]},
    "let_expressions/let_row_gather_sampled": {
        "group_idx": [0, 1, 1, 0],
        "y": [0, 2, 1, 1],
    },
    "let_expressions/let_host_matrix_row": {
        "design_idx": [0, 0, 0, 0],
        "X": [0.1, 0.4, -0.3, 0.7, -0.2, 0.5, 0.0, 0.3, 0.9, -0.6, 0.2, 0.1],
        "y": [0.2, -0.4, 0.6, 0.1],
    },
}

#: Each fixture's clamped points; every site varies across them.
_POINTS: dict[str, list[dict[str, Host]]] = {
    "steps/observe_class_code": [
        {"probs": [0.2, 0.3, 0.5]},
        {"probs": [0.6, 0.3, 0.1]},
        {"probs": [0.25, 0.5, 0.25]},
    ],
    "steps/sample_class_code": [
        {"probs": [0.2, 0.3, 0.5], "mu": [-1.0, 0.5, 2.0], "z": [0, 2, 1, 2]},
        {"probs": [0.6, 0.3, 0.1], "mu": [0.3, -0.7, 1.1], "z": [2, 2, 0, 1]},
        {"probs": [0.25, 0.5, 0.25], "mu": [1.5, 1.0, -0.5], "z": [1, 0, 0, 2]},
    ],
    "steps/observe_gather_event_vector": [
        {"mu": [-1.0, 0.5, 2.0], "z": [0, 2, 1, 2]},
        {"mu": [0.3, -0.7, 1.1], "z": [2, 2, 0, 1]},
        {"mu": [1.5, 1.0, -0.5], "z": [1, 0, 0, 2]},
    ],
    "steps/observe_gather_matrix_row": [
        {"mu": [-1.0, 0.3, 0.5, 1.2, 2.0, -0.4], "z": [0, 2, 1, 2]},
        {"mu": [0.3, 0.8, -0.7, 0.1, 1.1, -1.3], "z": [2, 2, 0, 1]},
        {"mu": [1.5, -0.2, 1.0, 0.6, -0.5, 0.9], "z": [1, 0, 0, 2]},
    ],
    "steps/observe_ordinal_code": [{"eta": -0.8}, {"eta": 0.3}, {"eta": 1.6}],
    "let_expressions/let_plated_list_operand": [{"mu": -0.6}, {"mu": 0.7}, {"mu": 1.3}],
    "steps/marginalize_let_probs": [{"r": 0.7}, {"r": 1.5}, {"r": 2.6}],
    "steps/marginalize_let_vector_probs": [{"s": 0.6}, {"s": 1.0}, {"s": 2.2}],
    "steps/marginalize_body_let_chain": [
        {"r": 0.7, "prior": [0.1, 0.2, 0.3, 0.4]},
        {"r": 1.5, "prior": [0.4, 0.3, 0.2, 0.1]},
        {"r": 2.6, "prior": [0.25, 0.25, 0.25, 0.25]},
    ],
    "steps/marginalize_body_plated_let": [
        {"r": 0.7, "prior": [0.3, 0.7]},
        {"r": 1.5, "prior": [0.5, 0.5]},
        {"r": 2.6, "prior": [0.8, 0.2]},
    ],
    "steps/marginalize_latent_gather": [
        {"mu": [-1.0, 0.0, 0.5, 2.0], "prior": [0.1, 0.2, 0.3, 0.4]},
        {"mu": [0.3, -0.8, 1.2, 0.1], "prior": [0.4, 0.3, 0.2, 0.1]},
        {"mu": [1.5, 0.6, -0.4, -1.3], "prior": [0.25, 0.25, 0.25, 0.25]},
    ],
    "steps/marginalize_latent_value": [
        {"r": 0.7, "prior": [0.2, 0.3, 0.5]},
        {"r": 1.5, "prior": [0.6, 0.3, 0.1]},
        {"r": 2.6, "prior": [0.3, 0.4, 0.3]},
    ],
    "let_expressions/let_real_division": [{"mu": -0.4}, {"mu": 0.0}, {"mu": 0.7}],
    "let_expressions/let_factor_index_arithmetic": [
        {"mu": -0.4},
        {"mu": 0.5},
        {"mu": 1.9},
    ],
    "let_expressions/let_row_gather": [
        {"a": [0.3, -0.2]},
        {"a": [-1.0, 0.8]},
        {"a": [1.4, 0.1]},
    ],
    "let_expressions/let_row_gather_sampled": [
        {"table": [0.2, 0.3, 0.5, 0.6, 0.3, 0.1]},
        {"table": [0.5, 0.25, 0.25, 0.1, 0.1, 0.8]},
        {"table": [0.3, 0.4, 0.3, 0.2, 0.5, 0.3]},
    ],
    "let_expressions/let_host_matrix_row": [
        {"sigma": 0.8, "w": [0.5, -0.3, 0.2]},
        {"sigma": 1.3, "w": [-0.2, 0.9, 0.4]},
        {"sigma": 0.5, "w": [1.1, 0.0, -0.7]},
    ],
}

#: The shape of every name a point carries; a scalar's is empty.
_SHAPES: dict[str, dict[str, list[int]]] = {
    "steps/observe_class_code": {"probs": [3], "y": [5]},
    "steps/sample_class_code": {"probs": [3], "mu": [3], "z": [4], "x": [4]},
    "steps/observe_gather_event_vector": {"mu": [3], "z": [4], "x": [4]},
    "steps/observe_gather_matrix_row": {"mu": [3, 2], "z": [4], "x": [4, 2]},
    "steps/observe_ordinal_code": {"eta": [], "y": [4]},
    "let_expressions/let_plated_list_operand": {"mu": [], "y": [4]},
    "steps/marginalize_let_probs": {"r": [], "y": [3]},
    "steps/marginalize_let_vector_probs": {"s": [], "y": [3]},
    "steps/marginalize_body_let_chain": {"r": [], "prior": [4], "y": [3]},
    "steps/marginalize_body_plated_let": {"r": [], "prior": [2], "y": [3]},
    "steps/marginalize_latent_value": {"r": [], "prior": [3], "y": [3]},
    "steps/marginalize_latent_gather": {"mu": [4], "prior": [4], "y": [3]},
    "let_expressions/let_real_division": {"mu": [], "y": [3]},
    "let_expressions/let_factor_index_arithmetic": {"mu": [], "y": [3]},
    "let_expressions/let_row_gather": {"a": [2], "group_idx": [4], "y": [4]},
    "let_expressions/let_row_gather_sampled": {
        "table": [2, 3],
        "group_idx": [4],
        "y": [4],
    },
    "let_expressions/let_host_matrix_row": {
        "sigma": [],
        "w": [1, 3],
        "design_idx": [4],
        "X": [4, 3],
        "y": [4],
    },
}

#: The integer-valued names of each fixture; every other name is real.
_INTEGERS: dict[str, frozenset[str]] = {
    "steps/observe_class_code": frozenset({"y"}),
    "steps/sample_class_code": frozenset({"z"}),
    "steps/observe_gather_event_vector": frozenset({"z"}),
    "steps/observe_gather_matrix_row": frozenset({"z"}),
    "steps/observe_ordinal_code": frozenset({"y"}),
    "steps/marginalize_let_probs": frozenset({"y"}),
    "steps/marginalize_body_let_chain": frozenset({"y"}),
    "steps/marginalize_body_plated_let": frozenset({"y"}),
    "steps/marginalize_latent_value": frozenset({"y"}),
    "let_expressions/let_row_gather": frozenset({"group_idx", "y"}),
    "let_expressions/let_row_gather_sampled": frozenset({"group_idx", "y"}),
    "let_expressions/let_host_matrix_row": frozenset({"design_idx"}),
}

#: Each fixture's independent joint.
_ORACLES: dict[str, Oracle] = {
    "steps/observe_class_code": _observe_class_code,
    "steps/sample_class_code": _sample_class_code,
    "steps/observe_gather_event_vector": _gather_event_vector,
    "steps/observe_gather_matrix_row": _gather_matrix_row,
    "steps/observe_ordinal_code": _observe_ordinal_code,
    "steps/marginalize_let_probs": _let_probs,
    "steps/marginalize_let_vector_probs": _let_vector_probs,
    "steps/marginalize_body_let_chain": _body_let_chain,
    "steps/marginalize_body_plated_let": _body_plated_let,
    "steps/marginalize_latent_value": _latent_value,
    "steps/marginalize_latent_gather": _latent_gather,
    "let_expressions/let_real_division": _real_division,
    "let_expressions/let_factor_index_arithmetic": _factor_index_arithmetic,
    "let_expressions/let_row_gather": _row_gather,
    "let_expressions/let_row_gather_sampled": _row_gather_sampled,
    "let_expressions/let_host_matrix_row": _host_matrix_row,
    "let_expressions/let_plated_list_operand": _plated_list_operand,
}

#: The refusals each target states for a fixture it cannot render, by
#: the kind prefix the refusal names. A target that samples each row's
#: latent on its own scores a different measure from one latent shared
#: by the rows; BUGS / JAGS `sum` contracts every axis of a two-axis
#: operand where QVR reduces the last; and BUGS cannot score an observed
#: class code, which the data holds from zero and `dcat` reads from one,
#: having no data block to shift it in. Stan's samplers draw no discrete
#: parameter, and a target whose library has no ordinal family refuses it
#: by name.
_REFUSED: dict[tuple[str, str], str] = {
    **{
        (fixture, backend): "marginalize:ungrouped-over-plate"
        for fixture in _UNGROUPED_MARGINALS
        for backend in ("bugs", "church", "edward2", "gen", "jags", "pymc", "turing")
    },
    ("let_expressions/let_host_matrix_row", "bugs"): "let-expr:axis-reduction",
    ("let_expressions/let_host_matrix_row", "jags"): "let-expr:axis-reduction",
    ("let_expressions/let_row_gather", "bugs"): "observe:class-code",
    ("let_expressions/let_row_gather_sampled", "bugs"): "observe:class-code",
    ("steps/observe_class_code", "bugs"): "observe:class-code",
    ("steps/sample_class_code", "stan"): "sample:discrete-latent",
    **{
        (fixture, backend): kind
        for fixture in (
            "steps/observe_gather_event_vector",
            "steps/observe_gather_matrix_row",
        )
        for backend, kind in (
            ("stan", "sample:discrete-latent"),
            ("bugs", "arg:list-literal"),
            ("jags", "arg:list-literal"),
        )
    },
    ("steps/observe_gather_matrix_row", "turing"): "sample:event-axis-lift",
    ("steps/observe_gather_matrix_row", "webppl"): "batch-rank:2",
    ("steps/observe_ordinal_code", "bugs"): "arg:list-literal",
    ("steps/observe_ordinal_code", "church"): "family:OrderedLogistic",
    ("steps/observe_ordinal_code", "edward2"): "family:OrderedLogistic",
    ("steps/observe_ordinal_code", "gen"): "family:OrderedLogistic",
    ("steps/observe_ordinal_code", "jags"): "family:no-target-name:OrderedLogistic",
    ("steps/observe_ordinal_code", "turing"): "family:OrderedLogistic",
    ("steps/observe_ordinal_code", "webppl"): "family:no-webppl-target:OrderedLogistic",
}


def _source(fixture: str) -> str:
    """A fixture's QVR source.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    str
        The source text.
    """
    return (_FIXTURES / f"{fixture}.qvr").read_text()


def _dtypes(fixture: str) -> dict[str, str]:
    """Each name's wire dtype.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    dict[str, str]
        ``"int"`` for an integer-valued name, ``"float"`` otherwise.
    """
    integers = _INTEGERS.get(fixture, frozenset())
    return {name: "int" if name in integers else "float" for name in _SHAPES[fixture]}


def _shaped(fixture: str, values: Mapping[str, Host]) -> dict[str, torch.Tensor]:
    """A point's values as tensors of their declared shapes.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.
    values : Mapping[str, Host]
        Flat values by name.

    Returns
    -------
    dict[str, torch.Tensor]
        Each value reshaped, integer names as `torch.long`.
    """
    integers = _INTEGERS.get(fixture, frozenset())
    out: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        dtype = torch.long if name in integers else torch.float32
        out[name] = torch.as_tensor(value, dtype=dtype).reshape(_SHAPES[fixture][name])
    return out


def _points(fixture: str) -> list[_protocol.Point]:
    """The fixture's clamped points with its data.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    list[_protocol.Point]
        One point per entry of the fixture's point set.
    """
    return [
        _protocol.Point(params=dict(point), data=dict(_DATA[fixture]))
        for point in _POINTS[fixture]
    ]


def _reference(fixture: str) -> list[float]:
    """The reference machine's joint at each point.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    list[float]
        The log joint at each point, each checked against the
        fixture's independent oracle.
    """
    model = loads(_source(fixture)).morphism
    assert isinstance(model, MonadicProgram | CheckedProgram)
    densities: list[float] = []
    for point in _POINTS[fixture]:
        values = _shaped(fixture, {**_DATA[fixture], **point})
        sites = {
            name: value.reshape(1) if value.dim() == 0 else value
            for name, value in values.items()
        }
        joint = float(model.log_joint(torch.zeros(1, 1), sites).sum())
        assert joint == pytest.approx(float(_ORACLES[fixture](values)), rel=1e-5)
        densities.append(joint)
    return densities


def _n_obs(fixture: str) -> int:
    """The number of observed rows a fixture's joint sums over.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    int
        The number of entries the fixture's data inputs hold.
    """
    return sum(math.prod(_SHAPES[fixture][name]) for name in _DATA[fixture])


@pytest.mark.parametrize("fixture", sorted(_ORACLES))
def test_reference_matches_the_independent_joint(fixture: str) -> None:
    assert len(_reference(fixture)) == len(_POINTS[fixture])


@pytest.mark.parametrize(
    ("fixture", "backend"),
    [
        (fixture, backend)
        for fixture in sorted(_ORACLES)
        for backend in _TARGETS
        if (fixture, backend) not in _REFUSED
    ],
)
def test_target_scores_the_reference_joint(
    fixture: str, backend: str, scratch: pathlib.Path
) -> None:
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    points = _points(fixture)
    source = transpile(parse(_source(fixture)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": pt.params, "data": pt.data} for pt in points],
        scratch=scratch / fixture.replace("/", "-") / backend,
        shapes=_SHAPES[fixture],
        dtypes=_dtypes(fixture),
    )
    _equivalence.assert_log_density_match(
        _reference(fixture),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=_n_obs(fixture)),
        context=f"{backend}@{fixture}",
    )


@pytest.mark.requires_probe("church")
@pytest.mark.parametrize(
    "fixture",
    [fixture for fixture in sorted(_ORACLES) if (fixture, "church") not in _REFUSED],
)
def test_church_scores_the_reference_joint(fixture: str, scratch: pathlib.Path) -> None:
    source = transpile(parse(_source(fixture)), target="church")
    nested = {
        name: torch.as_tensor(value).reshape(_SHAPES[fixture][name]).tolist()
        for name, value in _DATA[fixture].items()
    }
    points = [
        _protocol.Point(params=dict(point), data=dict(nested))
        for point in _POINTS[fixture]
    ]
    result = ChurchProbe().evaluate(
        source,
        fixture.replace("/", "-"),
        points,
        scratch=scratch / fixture.replace("/", "-") / "church",
    )
    _equivalence.assert_log_density_match(
        _reference(fixture),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=_n_obs(fixture)),
        context=f"church@{fixture}",
    )


@pytest.mark.parametrize(("fixture", "backend"), sorted(_REFUSED))
def test_target_refuses_under_its_kind(fixture: str, backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(fixture)), target=backend)
    expected = _REFUSED[(fixture, backend)]
    assert any(kind.startswith(expected) for kind in caught.value.kinds), (
        caught.value.kinds
    )
