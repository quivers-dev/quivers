"""A gathered family argument reads the element its checked type leaves.

`mu[z]` with `z` one class per row reads, per row, what remains of `mu`
after its gathered axis: one entry of an event vector or of a plated
site, one row of a matrix-valued site. Each program is scored on the
compiled runtime and on the reference machine, both pinned to an
independent `torch.distributions` joint, with the gather inside an
observe and inside a sample.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.continuous.inline import MixedInlineDistribution, make_inline_distribution
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.program_steps import Indexed, StepArgument
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.dsl.ast_nodes import DrawArgIndex, DrawArgNamed, DrawArgScalar
from quivers.qiec.entries import HostValue

_HEAD = (
    "object Class : FinSet 3\nobject Dim : FinSet 2\nobject Row : FinSet 4\n"
    "object Val : Real 1\nprogram p : Row -> Val\n"
)
_CLASSES = "    sample z : Row <- Categorical(probs=[0.2, 0.3, 0.5])\n"
_PROBS = torch.tensor([0.2, 0.3, 0.5])
_Z = torch.tensor([0, 2, 1, 2])
_VECTOR = torch.tensor([-1.0, 0.5, 2.0])
_MATRIX = torch.tensor([[-1.0, 0.3], [0.5, 1.2], [2.0, -0.4]])
_X = torch.tensor([0.1, 1.7, 1.2, 2.4])
_X_ROWS = torch.tensor([[0.1, -0.3], [1.7, 0.2], [1.2, 1.1], [2.4, -0.8]])

#: Each source of the gather: its declaration, its value, and whether a
#: gathered element is a row of two entries.
_SOURCES: dict[str, tuple[str, torch.Tensor, bool]] = {
    "event-vector": (
        "    sample mu <- Normal(loc=0.0, scale=2.0) [over=Class]\n",
        _VECTOR,
        False,
    ),
    "plated-site": (
        "    sample mu : Class <- Normal(loc=0.0, scale=2.0)\n",
        _VECTOR,
        False,
    ),
    "plated-rows": (
        "    sample mu : Class <- Normal(loc=0.0, scale=2.0) [over=Dim]\n",
        _MATRIX,
        True,
    ),
    "matrix-over-two-axes": (
        "    sample mu <- Normal(loc=0.0, scale=2.0) [over=[Class, Dim]]\n",
        _MATRIX,
        True,
    ),
}


def _nested(value: torch.Tensor) -> HostValue:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The value.

    Returns
    -------
    HostValue
        A number for a scalar, a tuple of the entries otherwise.
    """
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(entry) for entry in value)


def _case(source: str, step: str) -> tuple[str, torch.Tensor, torch.Tensor, float]:
    """A program gathering from one source in one kind of step.

    Parameters
    ----------
    source : str
        The key of the source in `_SOURCES`.
    step : str
        ``"observe"`` or ``"sample"``.

    Returns
    -------
    tuple[str, torch.Tensor, torch.Tensor, float]
        The program source, the gathered-from value, the value of `x`,
        and the independent joint.
    """
    declaration, mu, rows = _SOURCES[source]
    over = " [over=Dim]" if rows else ""
    x = _X_ROWS if rows else _X
    program = (
        _HEAD
        + declaration
        + _CLASSES
        + f"    {step} x : Row <- Normal(loc=mu[z], scale=1.0){over}\n"
        + "    return z\nexport p\n"
    )
    expected = (
        D.Normal(0.0, 2.0).log_prob(mu).sum()
        + D.Categorical(probs=_PROBS).log_prob(_Z).sum()
        + D.Normal(mu[_Z], 1.0).log_prob(x).sum()
    )
    return program, mu, x, float(expected)


_CELLS = [
    (source, step) for source in sorted(_SOURCES) for step in ("observe", "sample")
]


@pytest.mark.parametrize(("source", "step"), _CELLS)
def test_compiled_runtime_scores_the_gathered_element(source: str, step: str) -> None:
    program, mu, x, expected = _case(source, step)
    model = loads(program).morphism
    assert isinstance(model, MonadicProgram)
    joint = model.log_joint(torch.zeros(1, 1), {"mu": mu, "z": _Z, "x": x})
    assert float(joint.sum()) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(("source", "step"), _CELLS)
def test_reference_machine_scores_the_gathered_element(source: str, step: str) -> None:
    program, mu, x, expected = _case(source, step)
    loaded = loads(program)
    (entry,) = loaded.entry_points()
    sites = {"mu": _nested(mu), "z": _nested(_Z)}
    data: dict[str, HostValue] = {}
    (data if step == "observe" else sites)["x"] = _nested(x)
    run = loaded.run(entry.name, data=data, sites=sites)
    assert run.log_joint == pytest.approx(expected, rel=1e-5)


def _gathered_normal(
    extents: tuple[int, ...], indices: tuple[str, ...]
) -> tuple[ContinuousMorphism, tuple[StepArgument, ...] | None]:
    """An inline Normal whose loc gathers ``mu`` at ``indices``.

    Parameters
    ----------
    extents : tuple[int, ...]
        The checked extents of ``mu``.
    indices : tuple[str, ...]
        The index bindings of the gather.

    Returns
    -------
    tuple[ContinuousMorphism, tuple[StepArgument, ...] | None]
        The inline morphism and its step arguments.
    """
    args = (
        DrawArgNamed(
            parameter="loc",
            value=DrawArgIndex(name="mu", indices=indices, line=0, col=0),
            line=0,
            col=0,
        ),
        DrawArgNamed(
            parameter="scale",
            value=DrawArgScalar(value=1.0, line=0, col=0),
            line=0,
            col=0,
        ),
    )
    return make_inline_distribution(
        "Normal",
        args,
        Euclidean(name="x", dim=1),
        variable_types={"mu": Euclidean(name="mu", dim=6)},
        value_extents={"mu": extents},
    )


@pytest.mark.parametrize(
    ("extents", "width"), [((6,), 1), ((3, 2), 2), ((2, 3), 3), ((1, 2, 3), 6)]
)
def test_a_gather_is_as_wide_as_the_axes_it_leaves(
    extents: tuple[int, ...], width: int
) -> None:
    morphism, arguments = _gathered_normal(extents, ("z",))
    assert isinstance(morphism, MixedInlineDistribution)
    assert arguments == (Indexed(name="mu", indices=("z",)),)
    assert morphism._param_spec[0] == ("var", width)


def test_a_gather_past_the_value_s_axes_is_refused() -> None:
    with pytest.raises(
        ValueError, match=r"mu\[i, j\] indexes 2 axes of a value with 1"
    ):
        _gathered_normal((6,), ("i", "j"))
