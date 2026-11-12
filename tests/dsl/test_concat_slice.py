"""``concat`` and ``slice`` rebuild the last axis of their tensors.

``concat(a, b, ...)`` joins tensors of one element type along their last
axis, the leading axes agreeing; ``slice(v, start, length)`` keeps
``length`` entries of the last axis from ``start``, both bounds integer
literals. Each program is checked, run on the reference machine, and run
on the compiled runtime with one and with two input rows, all against
an independent ``torch`` oracle (``torch.cat`` and Python slicing); the
gradient of the compiled joint is held to the oracle's, a misread
(operands swapped, a window shifted by one) is shown to miss the joint,
and every refusal is located at its argument.
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
object H : FinSet 3
object K : FinSet 2
object Packed : Real 7
program p : Row -> Packed
    sample h : H <- Normal(loc=0.0, scale=1.0)
    sample c : H <- Normal(loc=0.0, scale=1.0)
    sample s <- Normal(loc=0.0, scale=1.0)
    let packed = concat(h, c, [s])
    let back = slice(packed, 3, 3)
    let table = factor k : K, j : H in real(k + 1) * h[j]
    let wide = concat(table, table)
    let window = slice(wide, 2, 2)
    let total = sum(packed * packed) + sum(back) + sum(sum(window))
    observe y <- Normal(loc=total, scale=1.0)
    return packed
export p
"""

_H = torch.tensor([[0.4, -1.2, 0.7], [1.1, 0.3, -0.5]], dtype=torch.float64)
_C = torch.tensor([[-0.8, 0.2, 1.5], [0.6, -0.9, 0.1]], dtype=torch.float64)
_S = torch.tensor([0.9, -0.3], dtype=torch.float64)
_Y = torch.tensor([1.3, -0.4], dtype=torch.float64)


def _parts(
    h: torch.Tensor, c: torch.Tensor, s: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Every let of one run, computed with ``torch`` alone.

    Parameters
    ----------
    h, c : torch.Tensor
        The two vectors, shape ``(3,)``.
    s : torch.Tensor
        The scalar, shape ``()``.

    Returns
    -------
    dict[str, torch.Tensor]
        Each let by name.
    """
    packed = torch.cat([h, c, s.reshape(1)])
    back = packed[3:6]
    table = torch.stack([(k + 1.0) * h for k in range(2)])
    wide = torch.cat([table, table], dim=-1)
    window = wide[..., 2:4]
    total = (packed * packed).sum() + back.sum() + window.sum()
    return {
        "packed": packed,
        "back": back,
        "table": table,
        "wide": wide,
        "window": window,
        "total": total,
    }


def _joint(
    h: torch.Tensor, c: torch.Tensor, s: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """One run's log joint, computed without the DSL.

    Parameters
    ----------
    h, c : torch.Tensor
        The two vectors.
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
        standard.log_prob(h).sum()
        + standard.log_prob(c).sum()
        + standard.log_prob(s)
        + D.Normal(_parts(h, c, s)["total"], 1.0).log_prob(y)
    )


def _nested(value: torch.Tensor) -> HostValue:
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(entry) for entry in value)


@pytest.mark.parametrize("runs", [1, 2])
def test_compiled_runtime_matches_the_oracle(runs: int) -> None:
    model = loads(_SOURCE).morphism
    sites = {
        "h": _H[:runs].float(),
        "c": _C[:runs].float(),
        "s": _S[:runs].float().reshape(runs, 1),
        "y": _Y[:runs].float(),
    }
    recorded = trace(model, torch.zeros(runs, 1), observations=sites)
    for run in range(runs):
        expected = _parts(_H[run], _C[run], _S[run])
        for name in ("packed", "back", "wide", "window"):
            # A let read from plated sites alone holds the runs' values
            # one after another; with one run it is that run's value.
            value = recorded.sites[name].value.reshape(runs, *expected[name].shape)
            torch.testing.assert_close(
                value[run].double(), expected[name], rtol=1e-6, atol=1e-6
            )
    joints = model.log_joint(torch.zeros(runs, 1), sites)
    for run in range(runs):
        assert float(joints[run]) == pytest.approx(
            float(_joint(_H[run], _C[run], _S[run], _Y[run])), rel=1e-5
        )


def test_reference_machine_matches_the_oracle() -> None:
    run = loads(_SOURCE).run(
        "p",
        data={"y": float(_Y[0])},
        sites={"h": _nested(_H[0]), "c": _nested(_C[0]), "s": float(_S[0])},
        seed=0,
    )
    expected = _parts(_H[0], _C[0], _S[0])
    assert run.value == pytest.approx(tuple(expected["packed"].tolist()))
    assert run.log_joint == pytest.approx(
        float(_joint(_H[0], _C[0], _S[0], _Y[0])), rel=1e-9
    )


def test_gradient_matches_the_oracle() -> None:
    model = loads(_SOURCE).morphism
    h = _H[0].float().clone().requires_grad_(True)
    c = _C[0].float().clone().requires_grad_(True)
    s = _S[0].float().reshape(1).clone().requires_grad_(True)
    joint = model.log_joint(
        torch.zeros(1, 1),
        {
            "h": h.unsqueeze(0),
            "c": c.unsqueeze(0),
            "s": s.unsqueeze(0),
            "y": _Y[:1].float(),
        },
    ).sum()
    joint.backward()
    oh = _H[0].clone().requires_grad_(True)
    oc = _C[0].clone().requires_grad_(True)
    os_ = _S[0].clone().requires_grad_(True)
    _joint(oh, oc, os_, _Y[0]).backward()
    assert h.grad is not None and c.grad is not None and s.grad is not None
    assert oh.grad is not None and oc.grad is not None and os_.grad is not None
    torch.testing.assert_close(h.grad.double(), oh.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(c.grad.double(), oc.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        s.grad.double().reshape(()), os_.grad, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    ("original", "misread"),
    [
        ("concat(h, c, [s])", "concat(c, h, [s])"),
        ("slice(packed, 3, 3)", "slice(packed, 4, 3)"),
        ("slice(wide, 2, 2)", "slice(wide, 1, 2)"),
    ],
)
def test_the_pinned_joint_rejects_a_misread(original: str, misread: str) -> None:
    # The oracle stays fixed; the program is changed, and its joint must
    # leave the oracle's, so the comparison above can tell them apart.
    model = loads(_SOURCE.replace(original, misread)).morphism
    sites = {
        "h": _H[:1].float(),
        "c": _C[:1].float(),
        "s": _S[:1].float().reshape(1, 1),
        "y": _Y[:1].float(),
    }
    joint = float(model.log_joint(torch.zeros(1, 1), sites).sum())
    assert joint != pytest.approx(float(_joint(_H[0], _C[0], _S[0], _Y[0])), rel=1e-3)


_REFUSALS = """
object Row : FinSet 1
object H : FinSet 3
object G : FinSet 2
program p : Row -> Row
    sample h : H <- Normal(loc=0.0, scale=1.0)
    sample s <- Normal(loc=0.0, scale=1.0)
    LET
    observe y <- Normal(loc=0.0, scale=1.0)
    return y
export p
"""


@pytest.mark.parametrize(
    ("let", "fragment", "column"),
    [
        ("let z = concat(h, s)", "argument 2 is a Real; a number x joins", 22),
        ("let z = concat(h)", "two or more tensors", 12),
        ("let z = concat(h, [1, 2])", "argument 2 holds Int, not Real", 12),
        ("let z = slice(h, 1, 3)", "reads entries 1 to 3, past the last entry", 12),
        ("let z = slice(h, 0, 0)", "keeps at least one entry, not 0", 24),
        ("let z = slice(h, 1.0, 2)", "start of a slice is a non-negative integer", 21),
        ("let z = slice(h, 0, h)", "length of a slice is a non-negative integer", 24),
        ("let z = slice(h, 1)", "slice(v, start, length)", 12),
        ("let z = slice(s, 0, 1)", "argument 1 is a Real", 18),
    ],
)
def test_a_misfit_is_refused_at_its_argument(
    let: str, fragment: str, column: int
) -> None:
    with pytest.raises(QiecDiagnosticError, match="qiec-primitive") as raised:
        loads(_REFUSALS.replace("LET", let))
    assert fragment in str(raised.value)
    assert f"<source>:8:{column}:" in str(raised.value)


def test_concat_refuses_disagreeing_leading_axes_at_the_call() -> None:
    let = "let m = factor i : G, j : H in h[j]\n    let z = concat(m, h)"
    with pytest.raises(
        QiecDiagnosticError, match="axes before the last agree"
    ) as raised:
        loads(_REFUSALS.replace("LET", let))
    assert "<source>:9:12:" in str(raised.value)
