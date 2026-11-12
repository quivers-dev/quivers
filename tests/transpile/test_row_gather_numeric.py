"""A gathered row or slice of a table is read whole on every target.

`let row_table = table[idx]` reads, for each observation row, the row of a
two-axis table at that row's index: its axes are the index's followed by
the table's after the first. A target that subscripts the gathered value
with the index alone reads the wrong entries, or none. Three fixtures
gather by a host index (a row, and a two-axis slice of a three-axis
table) and by a drawn index; each is scored on the compiled runtime and
the reference machine, both pinned to an independent
`torch.distributions` joint, and on every target that renders it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import pathlib

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.probes import _protocol
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

type Host = float | int | list[float] | list[int]
"""A number or a flat list of numbers."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""An independent computation of a fixture's joint."""

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_HOST_ROW = "let_expressions/let_row_gather"
_DRAWN_ROW = "let_expressions/let_row_gather_drawn"
_HOST_SLICE = "let_expressions/let_slice_gather"


def _host_row(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    a, y, idx = v["a"], v["y"].long(), v["group_idx"].long()
    k = torch.arange(2, dtype=torch.float64)
    table = k * torch.sigmoid(a)[:, None] + (1.0 - k) * (
        1.0 - torch.sigmoid(a)[:, None]
    )
    prior = D.Normal(0.0, 1.0).log_prob(a).sum()
    return prior + D.Categorical(probs=table[idx]).log_prob(y).sum()


def _drawn_row(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    a, y, group = v["a"], v["y"].long(), v["group"].long()
    k = torch.arange(2, dtype=torch.float64)
    table = k * torch.sigmoid(a)[:, None] + (1.0 - k) * (1.0 - torch.sigmoid(a)[:, None])
    prior = D.Normal(0.0, 1.0).log_prob(a).sum()
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    return (
        prior
        + D.Categorical(probs=weights).log_prob(group).sum()
        + D.Categorical(probs=table[group]).log_prob(y).sum()
    )


def _host_slice(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    a, y, idx = v["a"], v["y"], v["group_idx"].long()
    part = torch.arange(2, dtype=torch.float64)[None, :, None]
    cls = torch.arange(3, dtype=torch.float64)[None, None, :]
    table = a[:, None, None] * (part + 1.0) + cls
    prior = D.Normal(0.0, 1.0).log_prob(a).sum()
    return prior + D.Normal(table[idx], 1.0).log_prob(y).sum()


_SLICE_Y = [
    [[0.3, 1.2, 2.5], [0.1, 0.9, 1.7]],
    [[-0.4, 0.8, 2.2], [-1.0, 0.2, 1.1]],
    [[0.6, 1.4, 2.9], [0.9, 2.1, 2.8]],
    [[0.0, 1.0, 2.0], [0.2, 1.3, 2.4]],
]

#: Each case: its fixture, its program, its data, its clamped points, and
#: its oracle.
_CASES: dict[str, tuple[str, str, dict[str, Host], list[dict[str, Host]], Oracle]] = {
    "host-row": (
        _HOST_ROW,
        "rows",
        {"y": [1, 0, 1, 1], "group_idx": [0, 1, 1, 0]},
        [{"a": [0.4, -1.3]}, {"a": [-0.9, 2.0]}, {"a": [1.7, 0.2]}],
        _host_row,
    ),
    "drawn-row": (
        _DRAWN_ROW,
        "rows",
        {"y": [1, 0, 1, 1]},
        [
            {"a": [0.4, -1.3], "group": [0, 1, 1, 0]},
            {"a": [-0.9, 2.0], "group": [1, 1, 0, 0]},
            {"a": [1.7, 0.2], "group": [0, 0, 1, 1]},
        ],
        _drawn_row,
    ),
    "host-slice": (
        _HOST_SLICE,
        "slices",
        {"y": _SLICE_Y, "group_idx": [0, 1, 1, 0]},
        [{"a": [0.4, -1.3]}, {"a": [-0.9, 2.0]}, {"a": [1.7, 0.2]}],
        _host_slice,
    ),
}

#: The observed site each fixture scores.
_OBSERVED = "y"

#: The names each fixture reads as integers.
_INTEGER_DATA: dict[str, frozenset[str]] = {
    _HOST_ROW: frozenset({"y", "group_idx"}),
    _DRAWN_ROW: frozenset({"y", "group"}),
    _HOST_SLICE: frozenset({"group_idx"}),
}

#: The refusals each target states for a fixture it cannot render.
_REFUSED: dict[tuple[str, str], str] = {
    # BUGS has no data block to shift an observed zero-based class code
    # to its one-based `dcat`.
    (_HOST_ROW, "bugs"): "observe:class-code:y",
    # BUGS and JAGS take no list-literal probability vector; Stan draws no
    # discrete parameter.
    (_DRAWN_ROW, "bugs"): "arg:list-literal",
    (_DRAWN_ROW, "jags"): "arg:list-literal",
    (_DRAWN_ROW, "stan"): "sample:discrete-latent",
    # WebPPL carries one batch axis per observation; Gen and Turing hold
    # the table as a nested tower of vectors, which a site cannot read
    # one subscript per axis.
    (_HOST_SLICE, "webppl"): "batch-rank:3",
    (_HOST_SLICE, "gen"): "let-expr:nested-slice-gather:row_slice",
    (_HOST_SLICE, "turing"): "let-expr:nested-slice-gather:row_slice",
}

_TARGETS: tuple[str, ...] = tuple(sorted(_BACKENDS_WITH_IMAGES))


def _source(fixture: str) -> str:
    """A fixture's QVR source.

    Parameters
    ----------
    fixture : str
        ``<category>/<stem>``.

    Returns
    -------
    str
        The source text.
    """
    return (_FIXTURES / f"{fixture}.qvr").read_text()


def _tensor(fixture: str, name: str, value: Host) -> torch.Tensor:
    """A host value as a tensor of its dtype.

    Parameters
    ----------
    fixture : str
        The fixture, which decides which names are integers.
    name : str
        The value's name.
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    dtype = torch.long if name in _INTEGER_DATA[fixture] else torch.float32
    return torch.as_tensor(value, dtype=dtype)


type Nested = float | int | tuple["Nested", ...]
"""A number, or nested tuples of numbers."""


def _nested(value: torch.Tensor) -> Nested:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    Nested
        A number, or nested tuples of numbers.
    """
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(item) for item in value)


def _flat(value: Host) -> Host:
    """A value flattened in row-major order, as a probe point carries it.

    Parameters
    ----------
    value : Host
        The value.

    Returns
    -------
    Host
        A number, or the flat list of its entries.
    """
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.dim() == 0:
        return tensor.item()
    return tensor.reshape(-1).tolist()


def _reference(case: str) -> list[float]:
    """The joint at each point, agreed by both execution paths.

    Parameters
    ----------
    case : str
        The key of the case in `_CASES`.

    Returns
    -------
    list[float]
        The compiled runtime's log joint at each point, each equal to the
        reference machine's and to the case's oracle.
    """
    fixture, entry, data, points, oracle = _CASES[case]
    program = loads(_source(fixture))
    densities: list[float] = []
    for point in points:
        merged = {**data, **point}
        values = {name: _tensor(fixture, name, value) for name, value in merged.items()}
        expected = float(oracle({k: v.double() for k, v in values.items()}))
        joint = float(
            program.morphism.log_joint(torch.zeros(1, dtype=torch.long), values).sum()
        )
        assert joint == pytest.approx(expected, rel=1e-5)
        run = program.run(
            entry,
            data={name: _nested(values[name]) for name in data},
            sites={name: _nested(values[name]) for name in point},
        )
        assert run.log_joint == pytest.approx(expected, rel=1e-9)
        densities.append(joint)
    return densities


@pytest.mark.parametrize("case", sorted(_CASES))
def test_both_execution_paths_score_the_independent_joint(case: str) -> None:
    assert len(_reference(case)) == len(_CASES[case][3])


def test_the_oracle_separates_the_gathered_row_from_its_first_entry() -> None:
    # Reading only the first entry of each gathered row, as a gather
    # subscripted by its index alone would, is a joint the oracle tells
    # apart.
    _, _, data, points, oracle = _CASES["host-slice"]
    values = {
        name: torch.as_tensor(value, dtype=torch.float64)
        for name, value in {**data, **points[0]}.items()
    }
    a, y, idx = values["a"], values["y"], values["group_idx"].long()
    first = a[idx][:, None, None] + torch.zeros_like(y)
    wrong = (
        D.Normal(0.0, 1.0).log_prob(a).sum() + D.Normal(first, 1.0).log_prob(y).sum()
    )
    assert abs(float(oracle(values)) - float(wrong)) > 1.0


@pytest.mark.parametrize(
    ("case", "backend"),
    [
        (case, backend)
        for case in sorted(_CASES)
        for backend in _TARGETS
        if (_CASES[case][0], backend) not in _REFUSED
    ],
)
def test_target_scores_the_reference_joint(
    case: str, backend: str, scratch: pathlib.Path
) -> None:
    fixture, _, data, points, _ = _CASES[case]
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    source = transpile(parse(_source(fixture)), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    shapes: dict[str, list[int]] = {
        name: list(torch.as_tensor(value).shape)
        for name, value in {**data, **points[0]}.items()
    }
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[
            {
                "params": {name: _flat(value) for name, value in point.items()},
                "data": {name: _flat(value) for name, value in data.items()},
            }
            for point in points
        ],
        scratch=scratch / case / backend,
        shapes=shapes,
        dtypes={
            name: "int" if name in _INTEGER_DATA[fixture] else "float"
            for name in shapes
        },
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=torch.as_tensor(data[_OBSERVED]).numel()),
        context=f"{backend}@{case}",
    )


@pytest.mark.requires_probe("church")
@pytest.mark.parametrize(
    "case",
    [case for case in sorted(_CASES) if (_CASES[case][0], "church") not in _REFUSED],
)
def test_church_scores_the_reference_joint(case: str, scratch: pathlib.Path) -> None:
    fixture, _, data, points, _ = _CASES[case]
    source = transpile(parse(_source(fixture)), target="church")
    result = ChurchProbe().evaluate(
        source,
        case,
        [_protocol.Point(params=dict(point), data=dict(data)) for point in points],
        scratch=scratch / case / "church",
    )
    _equivalence.assert_log_density_match(
        _reference(case),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=torch.as_tensor(data[_OBSERVED]).numel()),
        context=f"church@{case}",
    )


@pytest.mark.parametrize(("fixture", "backend"), sorted(_REFUSED))
def test_target_refuses_under_its_kind(fixture: str, backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source(fixture)), target=backend)
    expected = _REFUSED[(fixture, backend)]
    assert any(kind.startswith(expected) for kind in caught.value.kinds), (
        caught.value.kinds
    )
