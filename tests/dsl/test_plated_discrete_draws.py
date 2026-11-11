"""A plated discrete draw holds one class index per row on every path.

`sample z : Row <- Categorical(...)` draws one class per row of `Row`, so
its density is one family term per row, whatever the class count and
whether the family's parameters are literals or earlier draws. Each
program is scored on the compiled runtime (without and with a leading
run axis) and on the reference machine, all pinned to an independent
`torch.distributions` joint, and sampled on the compiled runtime.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
import torch.distributions as D

from quivers.continuous.programs import MonadicProgram
from quivers.dsl import loads
from quivers.qiec.entries import HostValue

_HEAD = (
    "object Class : FinSet 3\nobject Row : FinSet 4\nobject Val : Real 1\n"
    "program p : Row -> Val\n"
)
_PROBS_PRIOR = "    sample probs <- Dirichlet(concentration=1.0) [over=Class]\n"

_PROBS = torch.tensor([0.2, 0.3, 0.5])
_LOGITS = torch.tensor([0.1, -0.4, 0.9])
_Z = torch.tensor([0, 2, 1, 2])
_B = torch.tensor([True, False, False, True])
_X = torch.tensor([0.1, 1.7, 1.2, 2.4])
_W = torch.tensor([-1.0, 0.5, 2.0])

_DIRICHLET = D.Dirichlet(torch.ones(3)).log_prob(_PROBS)
_CATEGORICAL = D.Categorical(probs=_PROBS).log_prob(_Z).sum()

#: Each program: its body, its site values, the observed data among
#: them, and the independent joint.
_CASES: dict[str, tuple[str, dict[str, torch.Tensor], frozenset[str], torch.Tensor]] = {
    "categorical-sampled-probs": (
        _PROBS_PRIOR + "    sample z : Row <- Categorical(probs=probs)\n    return z\n",
        {"probs": _PROBS, "z": _Z},
        frozenset(),
        _DIRICHLET + _CATEGORICAL,
    ),
    "categorical-literal-probs": (
        "    sample z : Row <- Categorical(probs=[0.2, 0.3, 0.5])\n    return z\n",
        {"z": _Z},
        frozenset(),
        _CATEGORICAL,
    ),
    "categorical-literal-logits": (
        "    sample z : Row <- Categorical(logits=[0.1, -0.4, 0.9])\n    return z\n",
        {"z": _Z},
        frozenset(),
        D.Categorical(logits=_LOGITS).log_prob(_Z).sum(),
    ),
    "categorical-sampled-logits": (
        "    sample l <- Normal(loc=0.0, scale=1.0) [over=Class]\n"
        "    sample z : Row <- Categorical(logits=l)\n    return z\n",
        {"l": _LOGITS, "z": _Z},
        frozenset(),
        D.Normal(0.0, 1.0).log_prob(_LOGITS).sum()
        + D.Categorical(logits=_LOGITS).log_prob(_Z).sum(),
    ),
    "bernoulli-sampled-probs": (
        "    sample q <- Beta(concentration1=2.0, concentration0=2.0)\n"
        "    sample b : Row <- Bernoulli(probs=q)\n    return b\n",
        {"q": torch.tensor(0.3), "b": _B},
        frozenset(),
        D.Beta(2.0, 2.0).log_prob(torch.tensor(0.3))
        + D.Bernoulli(probs=0.3).log_prob(_B.float()).sum(),
    ),
    "bernoulli-literal-probs": (
        "    sample b : Row <- Bernoulli(probs=0.3)\n    return b\n",
        {"b": _B},
        frozenset(),
        D.Bernoulli(probs=0.3).log_prob(_B.float()).sum(),
    ),
    "class-read-as-number": (
        _PROBS_PRIOR + "    sample z : Row <- Categorical(probs=probs)\n"
        "    let loc = real(z)\n    return z\n",
        {"probs": _PROBS, "z": _Z},
        frozenset(),
        _DIRICHLET + _CATEGORICAL,
    ),
    "class-read-as-number-observed": (
        _PROBS_PRIOR + "    sample z : Row <- Categorical(probs=probs)\n"
        "    let loc = real(z)\n"
        "    observe x : Row <- Normal(loc=loc, scale=1.0)\n    return z\n",
        {"probs": _PROBS, "z": _Z, "x": _X},
        frozenset({"x"}),
        _DIRICHLET + _CATEGORICAL + D.Normal(_Z.float(), 1.0).log_prob(_X).sum(),
    ),
    "class-as-subscript": (
        _PROBS_PRIOR + "    sample w : Class <- Normal(loc=0.0, scale=2.0)\n"
        "    sample z : Row <- Categorical(probs=probs)\n"
        "    let m = w[z]\n    return z\n",
        {"probs": _PROBS, "w": _W, "z": _Z},
        frozenset(),
        _DIRICHLET + D.Normal(0.0, 2.0).log_prob(_W).sum() + _CATEGORICAL,
    ),
    "class-as-subscript-observed": (
        _PROBS_PRIOR + "    sample w : Class <- Normal(loc=0.0, scale=2.0)\n"
        "    sample z : Row <- Categorical(probs=probs)\n"
        "    observe x : Row <- Normal(loc=w[z], scale=1.0)\n    return z\n",
        {"probs": _PROBS, "w": _W, "z": _Z, "x": _X},
        frozenset({"x"}),
        _DIRICHLET
        + D.Normal(0.0, 2.0).log_prob(_W).sum()
        + _CATEGORICAL
        + D.Normal(_W[_Z], 1.0).log_prob(_X).sum(),
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


def _model(case: str) -> MonadicProgram:
    """A case's program on the compiled runtime.

    Parameters
    ----------
    case : str
        The key of the case in `_CASES`.

    Returns
    -------
    MonadicProgram
        The compiled program.
    """
    model = loads(_HEAD + _CASES[case][0] + "export p\n").morphism
    assert isinstance(model, MonadicProgram)
    return model


@pytest.mark.parametrize("case", sorted(_CASES))
def test_compiled_runtime_scores_one_term_per_row(case: str) -> None:
    _, sites, _, expected = _CASES[case]
    joint = _model(case).log_joint(torch.zeros(1, 1), sites)
    assert joint.shape == (1,)
    assert float(joint) == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_compiled_runtime_scores_one_term_per_row_under_a_run_axis(case: str) -> None:
    _, sites, _, expected = _CASES[case]
    batched: Mapping[str, torch.Tensor] = {
        name: value.unsqueeze(0) for name, value in sites.items()
    }
    joint = _model(case).log_joint(torch.zeros(1, 1), batched)
    assert joint.shape == (1,)
    assert float(joint) == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_reference_machine_scores_one_term_per_row(case: str) -> None:
    body, sites, observed, expected = _CASES[case]
    program = loads(_HEAD + body + "export p\n")
    (entry,) = program.entry_points()
    run = program.run(
        entry.name,
        data={name: _nested(sites[name]) for name in observed},
        sites={
            name: _nested(value)
            for name, value in sites.items()
            if name not in observed
        },
    )
    assert run.log_joint == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize(
    ("case", "classes"),
    [
        ("categorical-literal-probs", 3),
        ("categorical-literal-logits", 3),
        ("categorical-sampled-logits", 3),
        ("categorical-sampled-probs", 3),
        ("bernoulli-literal-probs", 2),
        ("bernoulli-sampled-probs", 2),
    ],
)
def test_a_plated_discrete_draw_samples_one_class_per_row(
    case: str, classes: int
) -> None:
    torch.manual_seed(0)
    value = _model(case).rsample(torch.zeros(1, 1))
    assert isinstance(value, torch.Tensor)
    assert value.numel() == 4
    # A Boolean support is drawn as floating 0 / 1; every draw is a class.
    assert torch.equal(value, value.round())
    assert int(value.min()) >= 0
    assert int(value.max()) < classes
