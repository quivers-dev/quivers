"""Every target spells integer division and remainder as truncating ones.

Python's `//` and `%` floor, JavaScript's `/` and Julia's `/` divide
exactly, and BUGS / JAGS hold every value as a real, so `Int / Int` and
`Int % Int` render as each target's truncating operation. The Docker
probe tier scores them on every target in `test_construct_let_numeric`.
"""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile

SOURCE = """
object Step : FinSet 2
object Val : Real 1

program p : Step -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    let q = factor t : Step in real((t - 3) / 2)
    let r = factor t : Step in real((t - 3) % 2)
    score s = mu * (sum(q) + sum(r))
    return mu

export p
"""

#: ``target -> (text of the quotient, text of the remainder)``.
_SPELLINGS: dict[str, tuple[str, str]] = {
    "pyro": ('rounding_mode="trunc"', "torch.fmod("),
    "numpyro": ("jnp.floor_divide(jnp.abs(", "jnp.subtract("),
    "pymc": ("pymc.pytensorf.pt.int_div(", "pymc.pytensorf.pt.sub("),
    "edward2": ("tf.math.floordiv(tf.math.abs(", "tf.math.subtract("),
    "turing": ("div(0 - 3, 2)", "rem(0 - 3, 2)"),
    "gen": ("div(0 - 3, 2)", "rem(0 - 3, 2)"),
    "webppl": ("quot(", "rem("),
    "church": ("(quotient", "(remainder"),
    "jags": ("trunc((0-3)/2)", "trunc((0-3)/2)"),
    "bugs": ("trunc((0-3)/2)", "trunc((0-3)/2)"),
    "stan": ("(0-3)/2", "(0-3)%2"),
}


@pytest.mark.parametrize("target", sorted(_SPELLINGS))
def test_integer_division_renders_truncating(target: str) -> None:
    rendered = transpile(parse(SOURCE), target=target).decode()
    compact = rendered.replace(" ", "").replace("\n", "")
    for text in _SPELLINGS[target]:
        assert text.replace(" ", "") in compact, (target, text)
    if target not in ("stan",):
        assert "//2" not in compact
