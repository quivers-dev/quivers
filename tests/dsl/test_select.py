"""``select`` and the numbers of a Boolean in let expressions.

``select(c, a, b)`` takes ``a`` where ``c`` holds and ``b`` elsewhere,
entry by entry, broadcasting a number among the three; ``real(c)`` and
``int(c)`` read a Boolean as one or zero. Each program is checked, run on
the reference machine, and run on the compiled runtime with one and with
two input rows, all against an independent ``torch.where`` oracle; the
compiled gradient is held to the oracle's and is finite where the
untaken branch is finite; swapping the branches is shown to miss the
joint; and the refusals are located at the call.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.inference import trace
from quivers.qiec.entries import HostValue

_SOURCE = """
object Row : FinSet 1
object H : FinSet 4
object Pos : FinSet 3
program p : Row -> Row
    sample v : H <- Normal(loc=0.0, scale=1.0)
    sample s <- Normal(loc=0.0, scale=1.0)
    let clipped = select(v > 0.0, v, 0.1 * v)
    let hinge = select(s < 0.5, s, 1.0 - s)
    let positive = real(v > 0.2)
    let mask = factor i : Pos, j : Pos in real(j <= i)
    let count = int(s > 0.0) + 1
    let total = sum(clipped) + hinge * sum(positive) + sum(sum(mask)) * s + real(count)
    observe y <- Normal(loc=total, scale=1.0)
    return y
export p
"""

_V = torch.tensor([[0.4, -1.2, 0.1, 0.9], [-0.5, 0.3, 1.7, -0.2]], dtype=torch.float64)
_S = torch.tensor([0.3, 0.8], dtype=torch.float64)
_Y = torch.tensor([1.1, -0.6], dtype=torch.float64)


def _parts(v: torch.Tensor, s: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every let of one run, computed with ``torch`` alone.

    Parameters
    ----------
    v : torch.Tensor
        The vector, shape ``(4,)``.
    s : torch.Tensor
        The scalar.

    Returns
    -------
    dict[str, torch.Tensor]
        Each let by name.
    """
    clipped = torch.where(v > 0.0, v, 0.1 * v)
    hinge = torch.where(s < 0.5, s, 1.0 - s)
    positive = (v > 0.2).to(v.dtype)
    mask = torch.tril(torch.ones(3, 3, dtype=v.dtype))
    count = (s > 0.0).to(v.dtype) + 1.0
    total = clipped.sum() + hinge * positive.sum() + mask.sum() * s + count
    return {
        "clipped": clipped,
        "hinge": hinge,
        "positive": positive,
        "mask": mask,
        "total": total,
    }


def _joint(v: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One run's log joint, computed without the DSL.

    Parameters
    ----------
    v : torch.Tensor
        The vector.
    s : torch.Tensor
        The scalar.
    y : torch.Tensor
        The observation.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    standard = D.Normal(0.0, 1.0)
    return (
        standard.log_prob(v).sum()
        + standard.log_prob(s)
        + D.Normal(_parts(v, s)["total"], 1.0).log_prob(y)
    )


def _nested(value: torch.Tensor) -> HostValue:
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(entry) for entry in value)


@pytest.mark.parametrize("runs", [1, 2])
def test_compiled_runtime_matches_the_oracle(runs: int) -> None:
    model = loads(_SOURCE).morphism
    sites = {
        "v": _V[:runs].float(),
        "s": _S[:runs].float().reshape(runs, 1),
        "y": _Y[:runs].float(),
    }
    recorded = trace(model, torch.zeros(runs, 1), observations=sites)
    for run in range(runs):
        expected = _parts(_V[run], _S[run])
        for name in ("clipped", "positive", "mask"):
            value = recorded.sites[name].value.reshape(runs, *expected[name].shape)
            torch.testing.assert_close(value[run].double(), expected[name])
    joints = model.log_joint(torch.zeros(runs, 1), sites)
    for run in range(runs):
        assert float(joints[run]) == pytest.approx(
            float(_joint(_V[run], _S[run], _Y[run])), rel=1e-5
        )


def test_reference_machine_matches_the_oracle() -> None:
    run = loads(_SOURCE).run(
        "p",
        data={"y": float(_Y[0])},
        sites={"v": _nested(_V[0]), "s": float(_S[0])},
        seed=0,
    )
    assert run.log_joint == pytest.approx(float(_joint(_V[0], _S[0], _Y[0])), rel=1e-9)


def test_gradient_matches_the_oracle() -> None:
    model = loads(_SOURCE).morphism
    v = _V[0].float().clone().requires_grad_(True)
    s = _S[0].float().reshape(1).clone().requires_grad_(True)
    joint = model.log_joint(
        torch.zeros(1, 1),
        {"v": v.unsqueeze(0), "s": s.unsqueeze(0), "y": _Y[:1].float()},
    ).sum()
    joint.backward()
    ov = _V[0].clone().requires_grad_(True)
    os_ = _S[0].clone().requires_grad_(True)
    _joint(ov, os_, _Y[0]).backward()
    assert v.grad is not None and s.grad is not None
    assert ov.grad is not None and os_.grad is not None
    torch.testing.assert_close(v.grad.double(), ov.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        s.grad.double().reshape(()), os_.grad, rtol=1e-4, atol=1e-5
    )


def test_the_gradient_is_finite_where_the_untaken_branch_is_finite() -> None:
    # `log(v)` is finite at every entry here; the untaken entries take no
    # gradient, so the gradient is the taken branch's alone.
    source = """
object Row : FinSet 1
object H : FinSet 3
program p : Row -> Row
    sample v : H <- Normal(loc=2.0, scale=1.0)
    let g = select(v > 2.0, log(v), v * v)
    let total = sum(g)
    observe y <- Normal(loc=total, scale=1.0)
    return y
export p
"""
    model = loads(source).morphism
    v = torch.tensor([[1.5, 2.5, 3.0]], requires_grad=True)
    model.log_joint(
        torch.zeros(1, 1), {"v": v, "y": torch.tensor([0.0])}
    ).sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    ov = torch.tensor([1.5, 2.5, 3.0], dtype=torch.float64, requires_grad=True)
    total = torch.where(ov > 2.0, torch.log(ov), ov * ov).sum()
    (
        D.Normal(2.0, 1.0).log_prob(ov).sum()
        + D.Normal(total, 1.0).log_prob(torch.tensor(0.0, dtype=torch.float64))
    ).backward()
    assert ov.grad is not None
    torch.testing.assert_close(v.grad[0].double(), ov.grad, rtol=1e-4, atol=1e-5)


def test_swapped_branches_miss_the_pinned_joint() -> None:
    swapped = _SOURCE.replace(
        "select(v > 0.0, v, 0.1 * v)", "select(v > 0.0, 0.1 * v, v)"
    )
    model = loads(swapped).morphism
    sites = {
        "v": _V[:1].float(),
        "s": _S[:1].float().reshape(1, 1),
        "y": _Y[:1].float(),
    }
    joint = float(model.log_joint(torch.zeros(1, 1), sites).sum())
    assert joint != pytest.approx(float(_joint(_V[0], _S[0], _Y[0])), rel=1e-3)


_REFUSALS = """
object Row : FinSet 1
object H : FinSet 4
object G : FinSet 3
program p : Row -> Row
    sample v : H <- Normal(loc=0.0, scale=1.0)
    sample w : G <- Normal(loc=0.0, scale=1.0)
    LET
    observe y <- Normal(loc=0.0, scale=1.0)
    return y
export p
"""


@pytest.mark.parametrize(
    ("let", "fragment"),
    [
        ("let z = select(v, v, 0.0)", "not defined at (Real, Real, Real)"),
        ("let z = select(v > 0.0, v, 0)", "not defined at (Bool, Real, Int)"),
        ("let z = select(v > 0.0, v)", "not defined at (Bool, Real)"),
        ("let z = select(v > 0.0, v, w)", "tensors of differing shapes"),
        ("let z = real(true, 1.0)", "not defined at (Bool, Real)"),
    ],
)
def test_a_misfit_is_refused_at_the_call(let: str, fragment: str) -> None:
    with pytest.raises(QiecDiagnosticError) as raised:
        loads(_REFUSALS.replace("LET", let))
    assert fragment in str(raised.value)
    assert "<source>:8:12:" in str(raised.value)
