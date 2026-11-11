"""A scalar family drawn at vector-valued parameters.

With no plate annotation and no axis-role clause, a scalar family whose
parameters are vector-valued bindings draws one independent variate per
entry of their broadcast extent: a vector beside a scalar, or two
vectors of the same extent. The site has that extent on the checked
program, on the compiled runtime, and in a scan cell's per-position
layout; two parameters of different extents are refused at the draw.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.compiler._prelude import CompileError
from quivers.inference import trace

HEAD = """
object E : Real 2
object H : Real 4
object W : Real 3
morphism gate : E * H -> H ~ LogitNormal
morphism wide : E * H -> W ~ LogitNormal
"""


def _cell(body: str) -> str:
    return HEAD + f"program p(x_in, h_prev) : E * H -> H\n{body}\nexport p\n"


def _checked_extent(program, site: str) -> tuple[int, ...]:  # type: ignore[no-untyped-def]
    entry = next(e for e in program.qiec.entries if e.name == "p")
    found = next(s for s in entry.sites if s.name == site)
    return tuple(int(axis.size.value) for axis in (*found.batch, *found.event))


def _x() -> torch.Tensor:
    return torch.tensor([[0.3, -0.2, 0.1, 0.4, -0.6, 0.9]])


SCALE_SOURCES = {
    "scalar": ("0.5", lambda g, h: torch.tensor(0.5)),
    "vector": ("spread", lambda g, h: torch.exp(h * 0.5)),
}


@pytest.mark.parametrize("scale", sorted(SCALE_SOURCES))
def test_a_scalar_family_at_a_vector_location_draws_its_extent(scale: str) -> None:
    expression, scale_of = SCALE_SOURCES[scale]
    program = loads(
        _cell(
            "    sample g <- gate(x_in, h_prev)\n"
            "    let centre = g * h_prev\n"
            "    let spread = exp(h_prev * 0.5)\n"
            f"    sample w <- Normal(loc=centre, scale={expression})\n"
            "    return w"
        )
    )
    assert _checked_extent(program, "w") == (4,)
    x = _x()
    run = trace(program.morphism, x)
    w = run.sites["w"].value
    assert w.numel() == 4
    g = run.sites["g"].value.reshape(4)
    h = x[0, 2:]
    expected = D.Normal(g * h, scale_of(g, h)).log_prob(w.reshape(4)).sum()
    values = {name: site.value for name, site in run.sites.items()}
    joint = program.morphism.log_joint(x, values)
    gate_density = run.sites["g"].log_prob
    torch.testing.assert_close(joint.sum(), (expected + gate_density).sum())


def test_an_observation_at_a_vector_location_has_its_extent() -> None:
    program = loads(
        _cell(
            "    sample g <- gate(x_in, h_prev)\n"
            "    let centre = g * h_prev\n"
            "    observe w <- Normal(loc=centre, scale=0.5)\n"
            "    return centre"
        )
    )
    assert _checked_extent(program, "w") == (4,)
    draw = trace(program.morphism, _x()).sites["w"].value
    assert draw.numel() == 4


def test_parameters_of_different_extents_are_refused_at_the_draw() -> None:
    source = _cell(
        "    sample g <- gate(x_in, h_prev)\n"
        "    sample v <- wide(x_in, h_prev)\n"
        "    let centre = g * h_prev\n"
        "    sample w <- Normal(loc=centre, scale=v)\n"
        "    return centre"
    )
    with pytest.raises(CompileError, match="3 and 4 entries, which do not broadcast"):
        loads(source)


SCAN = (
    HEAD
    + """
object Tok : FinSet 5
morphism embed : Tok -> E [role=embed]

program cell(x_in, h_prev) : E * H -> H
    sample g <- gate(x_in, h_prev)
    let centre = g * h_prev
    sample h_cand <- Normal(loc=centre, scale=0.5)
    return h_cand

define backbone = embed >> scan(cell)

program p : Tok -> H
    sample h <- backbone
    return h

export p
"""
)


def test_a_scan_cell_draw_at_a_vector_location_is_laid_out_by_its_extent() -> None:
    program = loads(SCAN)
    x = torch.tensor([[0, 3, 1]])
    run = trace(program.morphism, x)
    cand = [
        name for name in run.sites if name.startswith("h$h_cand") or "h_cand" in name
    ]
    assert cand
    per_position = run.sites[cand[0]].value
    assert per_position.shape[-1] == 4
    sequence = torch.stack(
        [run.sites[name].value.reshape(4) for name in sorted(cand)], dim=0
    ).unsqueeze(0)
    # A whole-sequence value under the site's label is (batch, seq_len, 4).
    label = cand[0].split("@")[0]
    values = {
        name: site.value
        for name, site in run.sites.items()
        if site.morphism is not None and not name.startswith(label + "@")
    }
    values[label] = sequence
    assert torch.isfinite(program.morphism.log_joint(x, values)).all()
