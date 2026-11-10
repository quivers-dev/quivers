"""A run refuses a value of the wrong element type where it enters.

A value clamped at a sample site, or supplied as a host datum, is
checked against its checked type as the reference machine receives it:
a float for an `Int` site, a float vector for a one-hot `Tensor[Int]`
site, or an integer for a `Bool` site is refused with a diagnostic that
names the site or datum, renders the checked type, and shows the value.
A value of the right element type is scored.
"""

from __future__ import annotations

import pytest

from quivers.dsl import loads
from quivers.qiec.entries import HostValue
from quivers.qiec.execution import ExecutionFailure

_HEADER = "object K : FinSet 3\nobject One : FinSet 1\nobject Resp : FinSet 4\n\n"

#: Each case: the draw, a value of the wrong element type with the
#: diagnostic it draws, and a value of the right type.
_SITES: dict[str, tuple[str, HostValue, str, HostValue]] = {
    "Categorical": (
        "Categorical(probs=[0.2, 0.3, 0.5])",
        1.0,
        r"the value given for site 'v', 1\.0, does not inhabit the site's "
        r"checked type Int$",
        1,
    ),
    "OneHotCategorical": (
        "OneHotCategorical(probs=[0.2, 0.3, 0.5]) [over=K]",
        (0.0, 1.0, 0.0),
        r"the value given for site 'v', \(0\.0, 1\.0, 0\.0\), does not inhabit "
        r"the site's checked type Tensor\[Int\]\(\[3\]\)$",
        (0, 1, 0),
    ),
    "Bernoulli": (
        "Bernoulli(probs=0.3)",
        1,
        r"the value given for site 'v', 1, does not inhabit the site's "
        r"checked type Bool$",
        True,
    ),
}


def _site_program(draw: str) -> str:
    """A program that draws one site and returns it.

    Parameters
    ----------
    draw : str
        The draw's right-hand side.

    Returns
    -------
    str
        The module source.
    """
    return (
        _HEADER + f"program p : One -> One\n    sample v <- {draw}\n"
        "    return v\nexport p\n"
    )


@pytest.mark.parametrize("family", sorted(_SITES))
def test_a_clamped_site_of_the_wrong_element_type_is_refused_by_name(
    family: str,
) -> None:
    draw, wrong, message, _ = _SITES[family]
    program = loads(_site_program(draw))
    with pytest.raises(ExecutionFailure, match=message):
        program.run("p", data={}, sites={"v": wrong})


@pytest.mark.parametrize("family", sorted(_SITES))
def test_a_clamped_site_of_its_checked_type_is_scored(family: str) -> None:
    draw, _, _, right = _SITES[family]
    program = loads(_site_program(draw))
    assert program.run("p", data={}, sites={"v": right}).log_joint is not None


def test_a_host_datum_of_the_wrong_element_type_is_refused_by_name() -> None:
    program = loads(
        _HEADER + "program p : Resp -> Resp\n"
        "    observe y : Resp <- Categorical(probs=[0.2, 0.3, 0.5])\n"
        "    return y\nexport p\n"
    )
    with pytest.raises(
        ExecutionFailure,
        match=(
            r"argument 0 \(y\) does not inhabit Tensor\[Int, Shape\[4\]\]: "
            r"\(0\.0, 1\.0, 2\.0, 1\.0\)"
        ),
    ):
        program.run("p", data={"y": (0.0, 1.0, 2.0, 1.0)})
    assert program.run("p", data={"y": (0, 1, 2, 1)}).log_joint is not None
