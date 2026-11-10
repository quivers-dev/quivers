"""Forward sampling reads each parameter at the width its checked type gives.

An inline family stacks its variable parameters into one input and
splits them again by width. Every width comes from the type of the
parameter's expression (a site's space, a let's operands, a host
datum's checked type), never from dividing the stacked column count, so
for every program `log_joint` scores, `rsample` and `trace` draw a value
`log_joint` scores, whatever mix of literals, sampled scalars, plate
vectors, and host data fills the parameters: for a plated observation
in one run, and for an unplated draw over any number of input rows, the
row count equal to the stacked width included.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from quivers.dsl import loads
from quivers.inference import trace

ROWS = 3

PLATED = """
object Row : FinSet 3
object Val : Real 1

program p : Row -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample sig <- HalfNormal(scale=1.0)
    sample z : Row <- Normal(loc=0.0, scale=1.0)
{lets}
    observe y : Row <- Normal(loc={loc}, scale={scale})
    return y

export p
"""

#: Each source of a location: (expression, the let defining it if any,
#: host data it reads).
LOCATIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "literal": ("0.5", "", ()),
    "sampled-scalar": ("mu", "", ()),
    "plate-vector": ("z", "", ()),
    "sampled-scalar-plus-plate-vector": ("shifted", "let shifted = mu + z", ()),
    "host-vector": ("hv", "", ("hv",)),
    "host-through-let": ("host_loc", "let host_loc = hl * 2.0", ("hl",)),
}

#: Each source of a scale, as for a location.
SCALES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "literal": ("1.5", "", ()),
    "sampled-scalar": ("sig", "", ()),
    "plate-vector": ("spread", "let spread = softplus(z) + sig", ()),
    "host-vector": ("se", "", ("se",)),
    "host-through-let": ("host_scale", "let host_scale = hs + sig", ("hs",)),
}

#: The host data, one value per row of the plate.
HOST: dict[str, torch.Tensor] = {
    "hv": torch.tensor([0.3, -0.2, 0.9]),
    "hl": torch.tensor([0.1, 0.4, -0.5]),
    "se": torch.tensor([0.5, 1.0, 1.5]),
    "hs": torch.tensor([0.2, 0.3, 0.4]),
}


def _source(loc: str, loc_let: str, scale: str, scale_let: str) -> str:
    lets = "\n".join(f"    {let}" for let in (loc_let, scale_let) if let)
    return PLATED.format(lets=lets, loc=loc, scale=scale)


@pytest.mark.parametrize(
    ("loc", "scale"),
    list(itertools.product(sorted(LOCATIONS), sorted(SCALES))),
)
def test_a_plated_observe_draws_what_its_joint_scores(loc: str, scale: str) -> None:
    loc_expr, loc_let, loc_hosts = LOCATIONS[loc]
    scale_expr, scale_let, scale_hosts = SCALES[scale]
    model = loads(_source(loc_expr, loc_let, scale_expr, scale_let)).morphism
    host = {name: HOST[name] for name in (*loc_hosts, *scale_hosts)}
    x = torch.zeros(1, 1)
    draw = model.rsample(x, observations=host)
    assert draw.numel() == ROWS
    run = trace(model, x, observations=host)
    y = run.sites["y"].value
    assert y.numel() == draw.numel()
    values = {name: site.value for name, site in run.sites.items()}
    joint = model.log_joint(x, {**values, **host})
    assert torch.isfinite(joint).all()


UNPLATED = """
object Hidden : Real 2

program p : Hidden -> Hidden
    sample sigma <- HalfNormal(scale=1.0)
    let m = base * 2.0
    sample s <- Normal(loc={loc}, scale={scale})
    return s

export p
"""


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize(
    ("loc", "scale"),
    [("m", "sigma"), ("m", "1.0"), ("0.5", "sigma"), ("m", "m")],
)
def test_an_unplated_draw_reads_host_data_at_its_checked_width(
    loc: str, scale: str, rows: int
) -> None:
    # Three rows equal the stacked width of `m` and `sigma` (one column
    # each, plus the second coordinate of the codomain), which a split
    # by the stacked column count would misread.
    model = loads(UNPLATED.format(loc=loc, scale=scale)).morphism
    x = torch.zeros(rows, 2)
    base = torch.rand(rows) + 0.5
    draw = model.rsample(x, observations={"base": base})
    run = trace(model, x, observations={"base": base})
    s = run.sites["s"].value
    assert tuple(s.shape) == tuple(draw.shape)
    values = {name: site.value for name, site in run.sites.items()}
    joint = model.log_joint(x, {**values, "base": base})
    assert torch.isfinite(joint).all()


def test_host_data_wider_than_its_checked_type_is_refused_on_both_paths() -> None:
    model = loads(UNPLATED.format(loc="m", scale="sigma")).morphism
    x = torch.zeros(3, 2)
    base = torch.randn(3, 2)
    with pytest.raises(ValueError, match="host datum 'base'"):
        model.rsample(x, observations={"base": base})
    with pytest.raises(ValueError, match="host datum 'base'"):
        model.log_joint(
            x,
            {"base": base, "sigma": torch.tensor([0.5]), "s": torch.zeros(3, 2)},
        )


def test_the_reported_program_samples_and_scores() -> None:
    source = """
object Row : FinSet 3
object Val : Real 1

program shifted : Row -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample z : Row <- Normal(loc=0.0, scale=1.0)
    let m = mu + z
    observe y : Row <- Normal(loc=m, scale=se)
    return y

export shifted
"""
    model = loads(source).morphism
    x = torch.zeros(1, 1)
    se = torch.ones(3)
    draw = model.rsample(x, observations={"se": se})
    assert draw.numel() == 3
    run = trace(model, x, observations={"se": se})
    values = {name: site.value for name, site in run.sites.items()}
    assert torch.isfinite(model.log_joint(x, {**values, "se": se})).all()
