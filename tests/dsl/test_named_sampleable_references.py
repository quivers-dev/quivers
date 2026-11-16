"""Native nested constructors retain declared sampleables and lexical precedence."""

from contextlib import contextmanager

import pytest
import torch
from quivers.dsl import loads
from quivers.inference.trace import trace


@pytest.mark.parametrize("reference", ["base"])
def test_fixed_sampleable_reference_and_alias_keep_named_initializer_order(reference):
    source = f"""object Out:Real 1
morphism base:Out->Out ~ Normal(scale=2.0,loc=0.25)
define alias = base
program model:Out->Out
    sample b <- Restrict(high=1.0,base={reference},low=-1.0)
    return b
export model
"""
    with _double_context():
        program = loads(source)
        point = torch.tensor(0.4, dtype=torch.double)
        normal = torch.distributions.Normal(
            point.new_tensor(0.25), point.new_tensor(2.0)
        )
        expected = (
            normal.log_prob(point)
            - (
                normal.cdf(point.new_tensor(1.0)) - normal.cdf(point.new_tensor(-1.0))
            ).log()
        )
        actual = trace(
            program.morphism, point.new_zeros(1, 1), {"b": point}
        ).log_joint.sum()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        assert program.run(
            "model", data={"out": (0.0,)}, sites={"b": 0.4}
        ).log_joint == pytest.approx(expected.item(), abs=1e-12)


@contextmanager
def _double_context():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def test_lexical_value_shadowing_global_morphism_is_not_expanded_in_numeric_position():
    source = """object Out:Real 1
morphism base:Out->Out ~ Normal(scale=2.0,loc=0.25)
program model(base:Real):Out->Out
    sample b <- Restrict(base=Normal(loc=base,scale=1.0),low=-1.0,high=1.0)
    return b
export model
"""
    with _double_context():
        program = loads(source)
        point = torch.tensor(0.4, dtype=torch.double)
        loc = point.new_tensor(0.7).requires_grad_()
        normal = torch.distributions.Normal(loc, point.new_tensor(1.0))
        oracle = (
            normal.log_prob(point)
            - (
                normal.cdf(point.new_tensor(1.0)) - normal.cdf(point.new_tensor(-1.0))
            ).log()
        )
        actual = trace(
            program.morphism, point.new_zeros(1, 1), {"base": loc, "b": point}
        ).log_joint.sum()
        torch.testing.assert_close(actual, oracle)
        torch.testing.assert_close(
            torch.autograd.grad(actual, loc, retain_graph=True)[0],
            torch.autograd.grad(oracle, loc)[0],
        )


def test_sampleable_alias_resolution_uses_canonical_source_origin_without_bypassing_checker():
    from quivers.dsl import parse
    from quivers.dsl.compiler.core import Compiler
    from quivers.dsl.ast_nodes import DrawArgDist, DrawArgName, DrawArgNamed
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec, QiecDiagnosticError

    source = """object Out:Real 1
morphism base:Out->Out ~ Normal(scale=2.0,loc=0.25)
define alias = base
program model:Out->Out
    sample b <- Restrict(base=alias,low=-1.0,high=1.0)
    return b
export model
"""
    module = parse(source)
    compiler = Compiler(module)
    args = (
        next(
            statement
            for statement in module.statements
            if getattr(statement, "name", None) == "model"
        )
        .draws[0]
        .args
    )
    expanded = compiler._inline_sampleable_references("Restrict", args, set())
    base = next(
        argument.value
        for argument in expanded
        if isinstance(argument, DrawArgNamed) and argument.parameter == "base"
    )
    assert isinstance(base, DrawArgDist) and base.family == "Normal"
    assert tuple(argument.parameter for argument in base.args) == ("scale", "loc")
    shadowed = compiler._inline_sampleable_references("Restrict", args, {"alias"})
    assert isinstance(shadowed[0].value, DrawArgName)
    with pytest.raises(QiecDiagnosticError, match="takes a Sampleable"):
        lower_qvr_to_qiec(module)
