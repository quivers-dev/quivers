"""Static substitutions traverse every categorical object constructor."""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import Compiler, loads, parse
from quivers.dsl.ast_nodes import (
    ContinuousConstructor,
    DiscreteConstructor,
    ObjectCoproduct,
    ObjectEffectApply,
    ObjectProduct,
    ObjectSlash,
    TypeName,
)
from quivers.dsl.program_templates import TemplateError, substitute_program_object
from quivers.inference.trace import trace


def test_substitution_crosses_nested_products_coproducts_slashes_and_effects() -> None:
    source = ObjectProduct(
        components=(
            TypeName(name="A", line=7, col=8),
            ObjectCoproduct(
                components=(
                    DiscreteConstructor(constructor="FinSet", args=("n",)),
                    ObjectSlash(
                        result=ContinuousConstructor(
                            constructor="Real",
                            args=("n",),
                            kwargs={"low": "n", "high": 8.0},
                            line=9,
                            col=10,
                        ),
                        argument=ObjectEffectApply(
                            effect="E", args=(TypeName(name="A"),)
                        ),
                        direction="\\",
                    ),
                )
            ),
        ),
    )
    substitutions = {
        "A": TypeName(name="Row"),
        "n": TypeName(name="3"),
        "E": TypeName(name="Random"),
    }
    expected = ObjectProduct(
        components=(
            TypeName(name="Row", line=7, col=8),
            ObjectCoproduct(
                components=(
                    DiscreteConstructor(constructor="FinSet", args=("3",)),
                    ObjectSlash(
                        result=ContinuousConstructor(
                            constructor="Real",
                            args=("3",),
                            kwargs={"low": 3, "high": 8.0},
                            line=9,
                            col=10,
                        ),
                        argument=ObjectEffectApply(
                            effect="Random", args=(TypeName(name="Row"),)
                        ),
                        direction="\\",
                    ),
                )
            ),
        )
    )
    assert substitute_program_object(source, substitutions) == expected
    assert (
        Compiler(parse("object Row : FinSet 3\n"))._rename_type(source, substitutions)
        == expected
    )
    assert source.components[0] == TypeName(name="A", line=7, col=8)


@pytest.mark.parametrize(
    "constructor",
    [
        "Real",
        "Simplex",
        "Sphere",
        "Ball",
        "CholeskyFactor",
        "Covariance",
        "Correlation",
        "Orthogonal",
        "Stiefel",
        "LowerTriangular",
        "Diagonal",
    ],
)
def test_each_continuous_constructor_substitutes_all_static_dimensions(
    constructor,
) -> None:
    source = ContinuousConstructor(
        constructor=constructor, args=("n", "m"), line=4, col=5
    )
    rewritten = substitute_program_object(
        source, {"n": TypeName(name="Row"), "m": TypeName(name="2")}
    )
    assert isinstance(rewritten, ContinuousConstructor)
    assert rewritten.args == ("Row", "2")
    assert (rewritten.line, rewritten.col) == (4, 5)


def test_a_constructor_dimension_requires_a_named_or_literal_actual() -> None:
    source = DiscreteConstructor(constructor="FinSet", args=("n",))
    with pytest.raises(TemplateError, match="requires a named object or literal"):
        substitute_program_object(
            source,
            {"n": ObjectProduct(components=(TypeName(name="A"), TypeName(name="B")))},
        )


def test_static_dimensions_in_constructor_syntax_preserve_density_and_gradient() -> (
    None
):
    source = """object Row : FinSet 3
program sub[n : Nat](scale : Real) : FinSet n -> Real n
    sample z : FinSet n <- Normal(loc=0.0, scale=scale)
    return z
define applied = sub[3](sigma)
export applied
"""
    program = loads(source)
    sigma = torch.tensor(0.7, requires_grad=True)
    values = torch.tensor((0.1, -0.3, 0.5), requires_grad=True)
    result = trace(
        program.morphism, torch.zeros(1, 1), observations={"z": values, "sigma": sigma}
    )
    assert result.log_joint is not None
    actual = result.log_joint.sum()
    expected = torch.distributions.Normal(0.0, sigma).log_prob(values).sum()
    torch.testing.assert_close(actual, expected)
    for received, wanted in zip(
        torch.autograd.grad(actual, (values, sigma), retain_graph=True),
        torch.autograd.grad(expected, (values, sigma)),
        strict=True,
    ):
        torch.testing.assert_close(received, wanted)
    assert program.morphism.domain.cardinality == 3
    assert program.morphism.codomain.dim == 3
