"""Row vectors stacked along an axis keep their rows and coordinates."""

from __future__ import annotations

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.dsl.ast_nodes import ProgramDecl
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.formulas._ast import _add, _var, row_zero, stack_rows
from quivers.inference.trace import trace

SOURCE = """
object Resp : FinSet 4
object K : FinSet 3
program model(a : Tensor[Real]([Resp]), b : Tensor[Real]([Resp])) : Resp -> Resp
    sample s <- Normal(loc=0.0, scale=1.0)
    let loc = factor response : Resp in m[response, 0] + 10.0 * m[response, 1] + 100.0 * m[response, 2]
    observe y : Resp <- Normal(loc=loc, scale=1.0)
    return y
export model
"""


def test_stacked_rows_hold_each_vector_at_its_coordinate():
    """A scalar is broadcast by the zero vector, and cases may come unordered."""
    module = parse(SOURCE)
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    zero = "z"
    steps = (
        program.draws[0],
        row_zero(zero),
        *stack_rows(
            "m",
            (
                (2, _add(_var(zero), _var("s"))),
                (0, _var("a")),
                (1, _var("b")),
            ),
            ("k", "K"),
            stem="by_coordinate",
        ),
        *program.draws[1:],
    )
    module = module.with_(
        statements=tuple(
            program.with_(draws=steps) if isinstance(item, ProgramDecl) else item
            for item in module.statements
        )
    )
    compiled = Compiler(module).compile().morphism
    assert isinstance(compiled, MonadicProgram)
    a, b = torch.arange(4.0), torch.tensor([0.5, -1.0, 2.0, 0.0])
    s, y = torch.tensor(0.3), torch.tensor([1.0, 2.0, 3.0, 4.0])
    value = trace(
        compiled, torch.zeros(1, 1), {"a": a, "b": b, "s": s, "y": y}
    ).log_joint
    assert value is not None
    normal = torch.distributions.Normal
    expected = (
        normal(0.0, 1.0).log_prob(s)
        + normal(a + 10.0 * b + 100.0 * s, 1.0).log_prob(y).sum()
    )
    torch.testing.assert_close(value.sum().float(), expected)
