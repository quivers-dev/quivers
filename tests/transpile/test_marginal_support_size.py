"""Static atom counts for categorical marginalization.

The plan states a marginalized latent's support as an integer range
whose length is the atom count every target enumerates: the latent's
declared index object sizes it, and a latent declared without one takes
the width of its probability argument. The renderers read the count off
that range, whatever form the probability argument takes.
"""

from __future__ import annotations

import pytest

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.transpile.ir import (
    CSIntegerInterval,
    CSReal,
    IRArgRef,
    IRMarginalize,
    Plate,
)
from quivers.transpile.plan import checked_program_plan
from quivers.transpile.renderers._python_helpers import marginal_support_size


def _marginal(constraint: CSIntegerInterval | CSReal) -> IRMarginalize:
    """A categorical marginal over the given support.

    Parameters
    ----------
    constraint : CSIntegerInterval | CSReal
        The planned support.

    Returns
    -------
    IRMarginalize
        The block, with a reference probability argument.
    """
    return IRMarginalize(
        latent="z",
        family="Categorical",
        args=(IRArgRef(name="probs"),),
        arg_names=("probs",),
        constraint=constraint,
        plate=Plate(event_dims=(), batch_dims=()),
        reduction="logsumexp",
        scope=(),
    )


def _planned_marginal(source: str) -> IRMarginalize:
    """The marginalize block the plan derives for a program.

    Parameters
    ----------
    source : str
        A QVR program with one marginalize block.

    Returns
    -------
    IRMarginalize
        The block.
    """
    plan = checked_program_plan(loads(source).qiec, parse(source), "stan")
    (block,) = (node for node in plan.body if isinstance(node, IRMarginalize))
    return block


_LET_PROBS = """
object Step : FinSet 3
object K : FinSet 4
object Val : Real 1

program p : Step -> Val
    sample r <- Gamma(concentration=2.0, rate=1.0)
    let prior = factor k : K in 0.25
    marginalize z{index} <- Categorical(probs=prior) [reduction=logsumexp]
        let rate = r * real(z + 1)
        observe y : Step <- Poisson(rate=rate)
    return r

export p
"""


def test_the_count_is_the_length_of_the_planned_range() -> None:
    assert marginal_support_size(_marginal(CSIntegerInterval(lower=0, upper=5))) == 6


def test_a_support_that_is_not_an_integer_range_is_refused() -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        marginal_support_size(_marginal(CSReal()))
    assert any(
        kind.startswith("marginalize:unknown-cardinality:Categorical")
        for kind in caught.value.kinds
    )


def test_the_declared_index_object_sizes_a_let_probability_argument() -> None:
    block = _planned_marginal(_LET_PROBS.format(index=" : K"))
    assert block.constraint == CSIntegerInterval(lower=0, upper=3)
    assert marginal_support_size(block) == 4


def test_without_an_index_the_probability_argument_width_sizes_the_range() -> None:
    block = _planned_marginal(_LET_PROBS.format(index=""))
    assert block.constraint == CSIntegerInterval(lower=0, upper=3)


def test_stan_enumerates_every_atom_of_the_index_object() -> None:
    source = _LET_PROBS.format(index=" : K")
    program = transpile(parse(source), target="stan").decode()
    assert "vector [4] lps_z" in program
    assert "for (k in 1 : 4)" in program
