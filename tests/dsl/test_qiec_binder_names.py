"""Generated program binders are deterministic and avoid authored names."""

import math
from pathlib import Path

import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec import Bind, QiecModule, dumps
from quivers.qiec.program_runtime import run_program


_COLLIDING_NAMES = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    let __step_1 = 2.0
    sample x <- Normal(loc=0.0, scale=1.0)
    score first = x * x
    let __step_2 = 3.0
    score second = __step_1 + __step_2
    return __step_2
export prog
"""


def _lower(source: str) -> QiecModule:
    """Lower source under a fixed module and file identity.

    Parameters
    ----------
    source : str
        The QVR source to lower.

    Returns
    -------
    QiecModule
        Its checked kernel graph.
    """
    return lower_qvr_to_qiec(parse(source), module_name="same", file_path="same.qvr")


@pytest.mark.parametrize("source", [_COLLIDING_NAMES, "fixture"])
def test_repeated_lowering_has_identical_wire_bytes(source: str) -> None:
    """Independent lowerings preserve every generated binder and wire byte.

    Parameters
    ----------
    source : str
        A program or the marker for the effectful helper fixture.
    """
    if source == "fixture":
        source = (
            Path(__file__).parents[1]
            / "fixtures/qiec/effectful_probabilistic_helper.qvr"
        ).read_text()
    modules = [_lower(source) for _ in range(3)]
    assert len({dumps(module) for module in modules}) == 1
    assert len({hash(module) for module in modules}) == 1


def test_generated_binders_avoid_earlier_and_later_authored_names() -> None:
    """Fresh binders preserve later lets and the independently scored joint."""
    module = _lower(_COLLIDING_NAMES)
    computation = next(item for item in module.computations if item.name == "prog")
    names = []
    body = computation.body
    while isinstance(body, Bind):
        names.append(body.binder.name)
        body = body.then
    assert len(names) == len(set(names))
    assert {"__step_1", "__step_2"}.issubset(names)
    result = run_program(module, "prog", data={}, sites={"x": 0.8})
    assert result.value == 3.0
    prior = -0.5 * 0.8**2 - 0.5 * math.log(2 * math.pi)
    assert result.log_joint == pytest.approx(prior + 0.8**2 + 5.0)
    assert result.log_joint != pytest.approx(prior + 0.8**2 + 4.0)
