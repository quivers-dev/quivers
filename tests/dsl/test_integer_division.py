"""Integer division and remainder truncate toward zero on every path.

The language specifies that `Int / Int` and `Int % Int` truncate toward
zero; a real division stays exact. The reference machine and the compiled
runtime agree on operands of both signs.
"""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_program

SOURCE = """
object Step : FinSet 4
object Val : Real 1

program p : Step -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    let below = factor t : Step in real((t - 3) / 2)
    let rest = factor t : Step in real((t - 3) % 2)
    let above = factor t : Step in real((t + 3) / 2)
    let quarter = mu / 4.0
    score s = sum(below) * 100.0 + sum(rest) * 10.0 + sum(above) + quarter
    return mu

export p
"""

MU = 1.0
PRIOR = -0.5 * MU * MU - 0.5 * math.log(2 * math.pi)
#: (t - 3) / 2 = -1, -1, 0, 0; (t - 3) % 2 = -1, 0, -1, 0;
#: (t + 3) / 2 = 1, 2, 2, 3.
EXPECTED = PRIOR + (-2) * 100.0 + (-2) * 10.0 + 8 + MU / 4.0


def test_the_reference_machine_truncates_toward_zero() -> None:
    module = lower_qvr_to_qiec(parse(SOURCE), module_name="division")
    run = run_program(module, "p", data={}, sites={"mu": MU})
    assert run.log_joint == pytest.approx(EXPECTED)


def test_the_compiled_runtime_truncates_toward_zero() -> None:
    model = loads(SOURCE).morphism
    value = model.log_joint(torch.zeros(1, 1), {"mu": torch.tensor([MU])})
    assert float(value.reshape(-1)[0]) == pytest.approx(EXPECTED)
