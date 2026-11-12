"""A target that loops over a scalar family's draws reads a vector parameter per draw.

``observe y <- Normal(loc=z, scale=0.5)`` with ``z`` one vector over
``Feat`` is one Normal per entry of ``z``. The targets that score such a
site one position at a time (Stan, BUGS, JAGS, Gen, WebPPL, Church) must
read ``z`` at the loop's position, and Turing, which has no loop, must
broadcast the family over ``z``; a vector that fills a parameter reading
a whole event (`Categorical`'s ``probs``) stays whole. These checks pin
the rendered text; `test_vector_location_numeric` scores the same
fixtures in each target's own runtime.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "steps"

#: The program of the report: an unplated observe at a let-bound vector
#: location, under a program over a real object.
_REPORTED = """
object Feat : Real 3
program p : Feat -> Feat
    sample a <- Normal(loc=0.0, scale=1.0) [over=Feat]
    let z = tanh(a)
    observe y <- Normal(loc=z, scale=0.5)
    return z
export p
"""


def _render(source: str, target: str) -> str:
    """A program's target source with every space removed.

    Parameters
    ----------
    source : str
        The QVR source.
    target : str
        The target.

    Returns
    -------
    str
        The rendered program, spaces dropped so the checks read the
        tokens rather than the pretty-printer's layout.
    """
    return re.sub(r"\s+", "", transpile(parse(source), target=target).decode())


def _fixture(stem: str) -> str:
    """A step fixture's source.

    Parameters
    ----------
    stem : str
        The fixture's file stem.

    Returns
    -------
    str
        The source text.
    """
    return (_FIXTURES / f"{stem}.qvr").read_text()


def test_stan_scores_the_reported_observation_at_each_entry() -> None:
    stan = _render(_REPORTED, "stan")
    assert "normal_lpdf(y[m_Feat]|z[m_Feat],0.5)" in stan
    assert "normal_lpdf(y[m_Feat]|z,0.5)" not in stan


#: Per target, the text that reads the let-bound location at the draw's
#: position, and the text that would read the whole vector instead.
_ENTRYWISE: dict[str, tuple[str, str]] = {
    "stan": ("normal_lpdf(y[m_Feat]|z[m_Feat],0.5)", "|z,0.5)"),
    "bugs": ("dnorm(z[m_Feat],", "dnorm(z[1:3],"),
    "jags": ("dnorm(z[m_Feat],", "dnorm(z,"),
    "gen": ("normal(z[m_Feat_y],0.5)", "normal(z,0.5)"),
    "turing": ("product_distribution(Normal.(z,0.5))", "filldist(Normal(z,0.5)"),
    "webppl": ("mu:z[n]", "mu:z,"),
    "church": ("(gaussian(list-refzm_Feat)0.5)", "(gaussianz0.5)"),
}


@pytest.mark.parametrize("target", sorted(_ENTRYWISE))
def test_target_reads_a_let_bound_vector_location_per_entry(target: str) -> None:
    rendered = _render(_fixture("observe_vector_location_let"), target)
    entrywise, whole = _ENTRYWISE[target]
    assert entrywise in rendered
    assert whole not in rendered


#: Per target, the text that reads the shared simplex whole, and the text
#: that would read one entry of it per row.
_WHOLE_EVENT: dict[str, tuple[str, str]] = {
    "stan": ("categorical_lpmf(y[m_K]+1|w)", "|w[m_K])"),
    "jags": ("dcat(w)", "dcat(w[m_K])"),
    "gen": ("qvr_categorical(w)", "qvr_categorical(w[m_K])"),
    "turing": ("filldist(qvr_categorical(w),3)", "qvr_categorical.(w)"),
    "webppl": ("ps:w,", "ps:w[n]"),
    "church": ("(categoricalw)", "(categorical(list-refwm_K))"),
}


@pytest.mark.parametrize("target", sorted(_WHOLE_EVENT))
def test_target_keeps_a_whole_event_parameter_whole(target: str) -> None:
    rendered = _render(_fixture("observe_shared_probs_vector"), target)
    whole, entrywise = _WHOLE_EVENT[target]
    assert whole in rendered
    assert entrywise not in rendered


def test_the_broadcast_plate_is_named_after_the_location_axis() -> None:
    # The draw is replicated over the location's own axis, so the plan
    # names the plate `Feat`, as the location's draw names it, rather
    # than after the binding `z`.
    pymc = _render(_fixture("observe_vector_location_let"), "pymc")
    assert 'dims=("Feat",),observed=y' in pymc
    assert '"z":' not in pymc
