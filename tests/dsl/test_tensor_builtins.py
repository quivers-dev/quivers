"""``matmul``, ``layer_norm``, and ``rms_norm`` in let expressions.

``matmul(a, b)`` sums the last axis of ``a`` against the first of ``b`` (a
vector or a matrix); ``layer_norm`` and ``rms_norm`` normalize each row
along the last axis, with an optional real ``eps``. Each program is
checked, run on the reference machine, and run on the compiled runtime
with one and with two input rows, all against an independent ``torch``
oracle (``torch.matmul``, ``torch.nn.functional.layer_norm``, and the RMS
formula); the compiled gradient is held to the oracle's; a transposed
product and a normalization without centering are shown to miss the
joint; and the refusals are located.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D
import torch.nn.functional as F

from quivers.dsl import loads
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.inference import trace
from quivers.qiec.entries import HostValue

_SOURCE = """
object Row : FinSet 1
object In : FinSet 3
object Out : FinSet 2
object Seq : FinSet 4
program p : Row -> Row
    sample W : Out <- Normal(loc=0.0, scale=1.0) [over=In]
    sample x : In <- Normal(loc=0.0, scale=1.0)
    sample X : Seq <- Normal(loc=0.0, scale=1.0) [over=In]
    let Wt = factor i : In, o : Out in W[o, i]
    let z = matmul(W, x)
    let Z = matmul(X, Wt)
    let d = matmul(x, x)
    let ln = layer_norm(x)
    let LN = layer_norm(X, 0.001)
    let rn = rms_norm(x)
    let RN = rms_norm(Z)
    let total = sum(z) + sum(sum(Z)) * 0.1 + d * 0.01 + sum(ln * x) + sum(sum(LN)) * 0.5 + sum(rn) + sum(sum(RN * Z))
    observe y <- Normal(loc=total, scale=1.0)
    return y
export p
"""

_W = torch.tensor(
    [[[0.3, -1.1, 0.8], [1.2, 0.4, -0.6]], [[-0.7, 0.5, 0.2], [0.1, -0.3, 0.9]]],
    dtype=torch.float64,
)
_X1 = torch.tensor([[0.4, -1.2, 0.7], [1.1, 0.3, -0.5]], dtype=torch.float64)
_XS = torch.linspace(-1.3, 1.6, 24, dtype=torch.float64).reshape(2, 4, 3)
_Y = torch.tensor([1.3, -0.4], dtype=torch.float64)


def _parts(
    w: torch.Tensor, x: torch.Tensor, xs: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Every let of one run, computed with ``torch`` alone.

    Parameters
    ----------
    w : torch.Tensor
        The ``(2, 3)`` matrix.
    x : torch.Tensor
        The ``(3,)`` vector.
    xs : torch.Tensor
        The ``(4, 3)`` sequence.

    Returns
    -------
    dict[str, torch.Tensor]
        Each let by name.
    """
    z = torch.matmul(w, x)
    big = torch.matmul(xs, w.T)
    d = torch.dot(x, x)
    ln = F.layer_norm(x, (3,), eps=1e-5)
    seq_ln = F.layer_norm(xs, (3,), eps=1e-3)
    rn = x / torch.sqrt((x * x).mean() + 1e-6)
    seq_rn = big / torch.sqrt((big * big).mean(dim=-1, keepdim=True) + 1e-6)
    total = (
        z.sum()
        + big.sum() * 0.1
        + d * 0.01
        + (ln * x).sum()
        + seq_ln.sum() * 0.5
        + rn.sum()
        + (seq_rn * big).sum()
    )
    return {
        "z": z,
        "Z": big,
        "d": d,
        "ln": ln,
        "LN": seq_ln,
        "rn": rn,
        "RN": seq_rn,
        "total": total,
    }


def _joint(
    w: torch.Tensor, x: torch.Tensor, xs: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """One run's log joint, computed without the DSL.

    Parameters
    ----------
    w, x, xs : torch.Tensor
        The sites.
    y : torch.Tensor
        The observation.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    standard = D.Normal(0.0, 1.0)
    return (
        standard.log_prob(w).sum()
        + standard.log_prob(x).sum()
        + standard.log_prob(xs).sum()
        + D.Normal(_parts(w, x, xs)["total"], 1.0).log_prob(y)
    )


def _nested(value: torch.Tensor) -> HostValue:
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(entry) for entry in value)


def _sites(runs: int) -> dict[str, torch.Tensor]:
    return {
        "W": _W[:runs].float(),
        "x": _X1[:runs].float(),
        "X": _XS[:runs].float(),
        "y": _Y[:runs].float(),
    }


@pytest.mark.parametrize("runs", [1, 2])
def test_compiled_runtime_matches_the_oracle(runs: int) -> None:
    model = loads(_SOURCE).morphism
    recorded = trace(model, torch.zeros(runs, 1), observations=_sites(runs))
    for run in range(runs):
        expected = _parts(_W[run], _X1[run], _XS[run])
        for name in ("z", "Z", "ln", "LN", "rn", "RN"):
            value = recorded.sites[name].value.reshape(runs, *expected[name].shape)
            torch.testing.assert_close(
                value[run].double(), expected[name], rtol=1e-5, atol=1e-5
            )
    joints = model.log_joint(torch.zeros(runs, 1), _sites(runs))
    for run in range(runs):
        assert float(joints[run]) == pytest.approx(
            float(_joint(_W[run], _X1[run], _XS[run], _Y[run])), rel=1e-5
        )


def test_reference_machine_matches_the_oracle() -> None:
    run = loads(_SOURCE).run(
        "p",
        data={"y": float(_Y[0])},
        sites={"W": _nested(_W[0]), "x": _nested(_X1[0]), "X": _nested(_XS[0])},
        seed=0,
    )
    assert run.log_joint == pytest.approx(
        float(_joint(_W[0], _X1[0], _XS[0], _Y[0])), rel=1e-9
    )


def test_gradient_matches_the_oracle() -> None:
    model = loads(_SOURCE).morphism
    leaves = {
        name: value[:1].float().clone().requires_grad_(True)
        for name, value in (("W", _W), ("x", _X1), ("X", _XS))
    }
    model.log_joint(torch.zeros(1, 1), {**leaves, "y": _Y[:1].float()}).sum().backward()
    oracle = {
        name: value[0].clone().requires_grad_(True)
        for name, value in (("W", _W), ("x", _X1), ("X", _XS))
    }
    _joint(oracle["W"], oracle["x"], oracle["X"], _Y[0]).backward()
    for name, leaf in leaves.items():
        assert leaf.grad is not None and oracle[name].grad is not None
        torch.testing.assert_close(
            leaf.grad[0].double(), oracle[name].grad, rtol=1e-3, atol=1e-4
        )


@pytest.mark.parametrize(
    ("original", "misread", "name"),
    [
        (
            "matmul(X, Wt)",
            "matmul(X, factor i : In, o : Out in W[o, (i + 1) % 3])",
            "Z",
        ),
        ("layer_norm(x)", "rms_norm(x)", "ln"),
        ("rms_norm(Z)", "layer_norm(Z)", "RN"),
    ],
)
def test_the_oracle_rejects_a_misread(original: str, misread: str, name: str) -> None:
    # The oracle stays fixed and the program is changed: the let it
    # computes must leave the oracle's value, so the comparison above
    # can tell the two apart.
    model = loads(_SOURCE.replace(original, misread)).morphism
    recorded = trace(model, torch.zeros(1, 1), observations=_sites(1))
    expected = _parts(_W[0], _X1[0], _XS[0])[name]
    value = recorded.sites[name].value.reshape(expected.shape).double()
    assert not torch.allclose(value, expected, rtol=1e-3, atol=1e-3)


_REFUSALS = """
object Row : FinSet 1
object In : FinSet 3
object Out : FinSet 2
program p : Row -> Row
    sample W : Out <- Normal(loc=0.0, scale=1.0) [over=In]
    sample x : In <- Normal(loc=0.0, scale=1.0)
    sample u : Out <- Normal(loc=0.0, scale=1.0)
    sample s <- Normal(loc=0.0, scale=1.0)
    LET
    observe y <- Normal(loc=0.0, scale=1.0)
    return y
export p
"""


@pytest.mark.parametrize(
    ("let", "fragment", "column"),
    [
        (
            "let z = matmul(W, u)",
            "of extent 3, against the first axis of its second, of extent 2",
            12,
        ),
        ("let z = matmul(W, s)", "takes a Tensor[Real] at argument 2, not Real", 22),
        (
            "let z = matmul(x, W)",
            "of extent 3, against the first axis of its second, of extent 2",
            12,
        ),
        ("let z = matmul(W)", "matmul(a, b)", 12),
        ("let z = layer_norm(s)", "takes a Tensor[Real] at argument 1, not Real", 23),
        ("let z = rms_norm(x, 0)", "eps of rms_norm is a positive real literal", 24),
        (
            "let z = layer_norm(x, s)",
            "eps of layer_norm is a positive real literal",
            26,
        ),
    ],
)
def test_a_misfit_is_refused_at_its_argument(
    let: str, fragment: str, column: int
) -> None:
    with pytest.raises(QiecDiagnosticError, match="qiec-primitive") as raised:
        loads(_REFUSALS.replace("LET", let))
    assert fragment in str(raised.value)
    assert f"<source>:10:{column}:" in str(raised.value)
