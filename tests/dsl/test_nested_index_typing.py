"""A host name gathered from in an index position holds integers.

`E[tokens[q], k]` inside a `factor` reads `tokens` entrywise and uses each
entry as a row of `E`, so `tokens` is a table of `Int`, exactly as when it
is written as an index itself. Both paths score the program against an
independent `torch.distributions` joint with the ids given as integers,
and a joint over misordered ids is shown to differ.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.qiec.entries import HostValue
from quivers.qiec.types import INT

_SOURCE = """
object Token : FinSet 6
object Pos : FinSet 5
object Emb : FinSet 4
program p : Pos -> Pos
    sample E : Token <- Normal(loc=0.0, scale=1.0) [over=Emb, iid_over=Token]
    let X = factor q : Pos, k : Emb in E[tokens[q], k]
    let s = sum(X)
    observe y : Pos <- Normal(loc=s, scale=1.0)
    return y
export p
"""

_TOKENS = torch.tensor([0, 3, 2, 5, 1])
_Y = torch.tensor([0.4, -1.2, 0.3, 2.1, -0.5], dtype=torch.float64)
_E = torch.linspace(-1.5, 1.7, 24, dtype=torch.float64).reshape(6, 4)


def _oracle(tokens: torch.Tensor) -> float:
    """The joint at ``_E``, computed without the DSL.

    Parameters
    ----------
    tokens : torch.Tensor
        The ids, one per position.

    Returns
    -------
    float
        The log joint.
    """
    # `sum` collapses the last axis of the (Pos, Emb) table: one total
    # per position.
    totals = torch.tensor(
        [sum(float(_E[int(tokens[q]), k]) for k in range(4)) for q in range(5)],
        dtype=torch.float64,
    )
    return float(
        D.Normal(0.0, 1.0).log_prob(_E).sum() + D.Normal(totals, 1.0).log_prob(_Y).sum()
    )


def _nested(value: torch.Tensor) -> HostValue:
    if value.dim() == 0:
        return value.item()
    return tuple(_nested(entry) for entry in value)


def test_the_gathered_name_is_typed_as_integers() -> None:
    (entry,) = loads(_SOURCE).entry_points()
    types = {parameter.name: parameter.type for parameter in entry.parameters}
    assert str(types["tokens"]).count(INT.constructor.name) == 1
    assert "Real" not in str(types["tokens"])


def test_compiled_runtime_reads_integer_ids() -> None:
    model = loads(_SOURCE).morphism
    sites = {"E": _E.float(), "y": _Y.float(), "tokens": _TOKENS}
    joint = float(model.log_joint(torch.zeros(1, 1), sites).sum())
    assert joint == pytest.approx(_oracle(_TOKENS), rel=1e-5)
    assert joint != pytest.approx(_oracle(_TOKENS.flip(0)), rel=1e-3)


def test_reference_machine_reads_integer_ids() -> None:
    run = loads(_SOURCE).run(
        "p",
        data={"tokens": _nested(_TOKENS), "y": _nested(_Y)},
        sites={"E": _nested(_E)},
        seed=0,
    )
    assert run.log_joint == pytest.approx(_oracle(_TOKENS), rel=1e-9)
