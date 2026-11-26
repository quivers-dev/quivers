"""Deductions choose by declared properties of the algebra and the items, not by names."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from quivers.dsl.compiler._deduction_terms import TERM_ATOM_TAG
from quivers.dsl.compiler.deductions import (
    BASE_WEIGHT_KEY,
    SpanItems,
    _make_rule_weight_fn,
    derives_start,
    goal_of,
)
from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)
from tests.evidence.counters import forwarding

ATOM = TERM_ATOM_TAG


def _weight(semiring: ChartSemiring) -> torch.Tensor:
    """A learned child rule under a learned parent, over two premises.

    Parameters
    ----------
    semiring : ChartSemiring
        The semiring of the run.

    Returns
    -------
    torch.Tensor
        The conclusion's weight.
    """
    params = {
        "child": nn.ParameterDict({BASE_WEIGHT_KEY: nn.Parameter(torch.tensor(-0.5))}),
        "parent": nn.ParameterDict(
            {BASE_WEIGHT_KEY: nn.Parameter(torch.tensor(-0.25))}
        ),
    }
    weight_fn = _make_rule_weight_fn(
        rule_name="child",
        positions=frozenset(),
        rule_parent={"child": "parent"},
        rule_param_dicts=params,
        rule_learnable={"child": True, "parent": True},
        rule_bounded={"child": False, "parent": False},
    )
    premises = (torch.tensor(-1.0), torch.tensor(-2.0))
    if not semiring.log_weights:
        premises = (torch.tensor(2.0), torch.tensor(3.0))
    return weight_fn({}, premises, semiring)


@pytest.mark.parametrize("semiring", [LOG_PROB, VITERBI, BOOLEAN, COUNTING])
def test_a_forwarding_algebra_reads_weights_as_its_base(
    semiring: ChartSemiring,
) -> None:
    """A semiring that forwards to another reads learned weights identically."""
    forwarded, counts = forwarding(semiring)
    assert forwarded is not semiring
    assert forwarded.log_weights == semiring.log_weights
    assert torch.equal(_weight(forwarded), _weight(semiring))
    assert counts["products"] > 0


def test_log_semirings_read_the_learned_weights() -> None:
    """The log semirings add the child's and the parent's log weights."""
    assert float(_weight(LOG_PROB)) == pytest.approx(-1.0 - 2.0 - 0.5 - 0.25)
    assert float(_weight(VITERBI)) == pytest.approx(-1.0 - 2.0 - 0.5 - 0.25)
    assert float(_weight(COUNTING)) == pytest.approx(6.0)


def _span(head: str, start: int, end: int, category: str) -> tuple:
    """A span item.

    Parameters
    ----------
    head : str
        The span constructor.
    start, end : int
        The positions.
    category : str
        The atom.

    Returns
    -------
    tuple
        The item.
    """
    return (head, start, end, (ATOM, category))


def test_a_renamed_span_head_behaves_identically() -> None:
    """Goal, extent and category depend on the declared head, not on its name."""
    plain, renamed = SpanItems(), SpanItems(head="cell")
    for start, end, category in ((0, 3, "S"), (0, 2, "NP"), (1, 3, "S")):
        original = _span("span", start, end, category)
        moved = _span("cell", start, end, category)
        assert plain.extent(original) == renamed.extent(moved) == (start, end)
        assert plain.category(original) == renamed.category(moved)
        assert goal_of(plain, "S")(original) == goal_of(renamed, "S")(moved)
    assert goal_of(plain, "S")(_span("span", 0, 3, "S"))
    assert not goal_of(plain, "S")(_span("cell", 0, 3, "S"))
    assert renamed.extent(_span("span", 0, 3, "S")) is None


def test_a_renamed_head_keyed_item_behaves_identically() -> None:
    """A head-keyed goal follows the start symbol, whatever it is called."""
    items = SpanItems()
    assert goal_of(items, "reach")(("reach", 3)) == goal_of(items, "route")(
        ("route", 3)
    )
    assert goal_of(items, "reach")(("reach", 3))
    assert not goal_of(items, "reach")(("route", 3))
    assert goal_of(items, "S")((ATOM, "S"))
    assert not derives_start((ATOM, "S", "extra"), "S")
    assert goal_of(items, None)(("anything",))
