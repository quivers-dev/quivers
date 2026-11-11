"""A standalone scan scores the trajectory it is given.

`ScanMorphism.log_prob` is checked against a torch reconstruction of
the per-position transition densities: the trajectory it scores is the
supplied one, never one rebuilt from the final state, and a final state
alone is refused. A composition that runs a scan in its prefix scores
the scan's trajectory at the canonical coordinates.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as td

from quivers.continuous.families import ConditionalNormal
from quivers.continuous.morphisms import SampledComposition
from quivers.continuous.scan import ScanMorphism
from quivers.continuous.spaces import Euclidean, ProductSpace

_INPUT = Euclidean(name="Input", dim=2)
_HIDDEN = Euclidean(name="Hidden", dim=3)
_OUT = Euclidean(name="Out", dim=1)


def _scan() -> ScanMorphism:
    torch.manual_seed(0)
    return ScanMorphism(
        ConditionalNormal(ProductSpace(components=(_INPUT, _HIDDEN)), _HIDDEN)
    )


def _transition_density(
    scan: ScanMorphism, x: torch.Tensor, states: torch.Tensor
) -> torch.Tensor:
    """The trajectory's joint density, one Normal per position.

    Parameters
    ----------
    scan : ScanMorphism
        The scan, whose cell's parameter source gives each position's
        location and log-scale.
    x : torch.Tensor
        Inputs, shape ``(batch, seq_len, 2)``.
    states : torch.Tensor
        The trajectory, shape ``(batch, seq_len, 3)``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch,)``.
    """
    cell = scan.cell
    assert isinstance(cell, ConditionalNormal)
    previous = torch.zeros(x.shape[0], 3)
    total = torch.zeros(x.shape[0])
    for t in range(x.shape[1]):
        raw = cell.param_source(torch.cat([x[:, t], previous], dim=-1))
        loc, scale = raw[:, :3], raw[:, 3:].exp()
        total = total + td.Normal(loc, scale).log_prob(states[:, t]).sum(-1)
        previous = states[:, t]
    return total


def test_a_scan_scores_the_supplied_trajectory() -> None:
    scan = _scan()
    x = torch.randn(4, 5, 2)
    states = torch.randn(4, 5, 3)
    with torch.no_grad():
        scored = scan.log_prob(x, states)
        folded = scan.log_prob(x, states.reshape(4, 15))
        expected = _transition_density(scan, x, states)
    torch.testing.assert_close(scored, expected)
    torch.testing.assert_close(folded, expected)


def test_a_scan_scores_its_own_trajectory_draw() -> None:
    scan = _scan()
    x = torch.randn(3, 4, 2)
    with torch.no_grad():
        states = scan.rsample_trajectory(x)
        scored = scan.log_prob(x, states)
        expected = _transition_density(scan, x, states)
    assert states.shape == (3, 4, 3)
    torch.testing.assert_close(scored, expected)


def test_a_scan_refuses_a_final_state_alone() -> None:
    scan = _scan()
    x = torch.randn(2, 5, 2)
    with pytest.raises(ValueError, match="Pass the trajectory"):
        scan.log_prob(x, torch.randn(2, 3))


def test_a_one_position_trajectory_is_its_final_state() -> None:
    scan = _scan()
    x = torch.randn(2, 1, 2)
    final = torch.randn(2, 3)
    with torch.no_grad():
        scored = scan.log_prob(x, final)
        expected = _transition_density(scan, x, final.unsqueeze(1))
    torch.testing.assert_close(scored, expected)


def test_a_scan_in_a_prefix_scores_its_canonical_trajectory() -> None:
    """The prefix binds the scan to its coordinates at zero, and every
    state those coordinates select carries its transition density."""
    scan = _scan()
    head = ConditionalNormal(_HIDDEN, _OUT)
    chain = SampledComposition(scan, head)
    x = torch.randn(3, 4, 2)
    z = torch.randn(3, 1)
    with torch.no_grad():
        base = torch.zeros(3, 4 * 3)
        states = scan.trajectory_from_base(x, base)
        raw = head.param_source(states[:, -1])
        expected = _transition_density(scan, x, states) + td.Normal(
            raw[:, :1], raw[:, 1:].exp()
        ).log_prob(z).sum(-1)
        scored = chain.log_prob(x, z)
    torch.testing.assert_close(scored, expected)


def test_a_scan_pushes_coordinates_to_the_final_state_of_their_trajectory() -> None:
    scan = _scan()
    x = torch.randn(2, 3, 2)
    base = torch.randn(2, 9)
    with torch.no_grad():
        states = scan.trajectory_from_base(x, base)
        final, density = scan.push_base_scored(x, base)
    torch.testing.assert_close(scan.push_base(x, base), states[:, -1])
    torch.testing.assert_close(final, states[:, -1])
    torch.testing.assert_close(density, _transition_density(scan, x, states))
