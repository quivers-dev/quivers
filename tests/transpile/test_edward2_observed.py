"""Edward2 observations hold the observed data over their whole plate.

A plated `observe` renders as a random variable whose value is the
observation and whose shape is the plate's, so a `let` that reads the
observed name reads the data rather than a fresh draw, and a run with
no data draws one value per row.
"""

from __future__ import annotations

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from tests.transpile.fixtures import _load

READS_OBSERVATION = """
object Item : FinSet 4
program prog : Item -> Item
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Item <- Normal(loc=mu, scale=1.0)
    let resid = y - mu
    score s = sum(resid)
    return y
export prog
"""


def _line(source: str, name: str) -> str:
    rendered = transpile(parse(source), target="edward2").decode()
    return next(
        line.strip()
        for line in rendered.splitlines()
        if line.strip().startswith(f"{name} = edward2.")
    )


def test_a_plated_observation_carries_its_plate_and_its_data() -> None:
    line = _line(READS_OBSERVATION, "y")
    assert "sample_shape=[4]" in line
    assert "value=y" in line


def test_an_observation_over_a_called_value_carries_its_plate() -> None:
    fixture = next(item for item in _load.load_steps() if item.name == "call_step")
    line = _line(fixture.source, "y")
    assert "sample_shape=[4]" in line
    assert "value=y" in line


def test_a_latent_carries_no_value() -> None:
    assert "value=" not in _line(READS_OBSERVATION, "mu")
