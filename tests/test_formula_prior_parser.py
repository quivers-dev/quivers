"""Prior strings use QVR's AST, schemas, and contextual numeric carriers."""

from unittest.mock import patch

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.dsl.ast_nodes import DrawArgDist, DrawArgNamed, DrawArgScalar, ProgramDecl
from quivers.dsl.emit import module_to_source
from quivers.formulas import FormulaToQVRModule, families, formula_from_data
from quivers.formulas.compile import _autoscaled_prior_step, _auxiliary_draw
from quivers.inference.trace import trace


@pytest.fixture(autouse=True)
def double_literals():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _program(prior):
    module = parse("""object Out : Real 1
program model(center : Real, spread : Real) : Out -> Out
    sample b <- Normal(loc=0,scale=1)
    return b
export model
""")
    return Compiler(
        module.with_(
            statements=tuple(
                statement.with_(draws=(_auxiliary_draw("b", prior),))
                if isinstance(statement, ProgramDecl)
                else statement
                for statement in module.statements
            )
        )
    ).compile()


@pytest.mark.parametrize(
    "prior",
    [
        "Normal(0,1)",
        " Normal ( scale = 1 , loc = 0 ) ",
        "Normal(\n0,\n1,\n)",
        "Independent(Normal(0,1),0)",
    ],
)
def test_integer_prior_lexemes_check_in_their_real_or_integer_context(prior):
    model = _program(prior)
    value = torch.tensor(0.4, requires_grad=True)
    actual = trace(
        model.morphism,
        torch.zeros(1, 1),
        {"center": torch.tensor(0.2), "spread": torch.tensor(0.7), "b": value},
    ).log_joint.sum()
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(value)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual, value, create_graph=True)[0],
        torch.autograd.grad(expected, value, create_graph=True)[0],
        atol=1e-12,
        rtol=1e-12,
    )
    reference = model.run(
        "model", data={"center": 0.2, "spread": 0.7, "out": (0.0,)}, sites={"b": 0.4}
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=1e-12)


@pytest.mark.parametrize(
    ("prior", "kind"),
    [
        ("Normal(center,spread)", "normal"),
        ("Normalize(Restrict(Normal(center,spread),-0.5,0.8))", "restricted"),
        (
            "Normalize(Mixture([0.3,0.7],[Normal(center,spread),Normal(-1,2)]))",
            "mixture",
        ),
        ("LogitNormal(sigma=spread,mu=center)", "logit_normal"),
    ],
)
def test_nested_named_and_alias_priors_preserve_density_and_every_derivative(
    prior, kind
):
    model = _program(prior)
    point = torch.tensor([0.2, 0.7, 0.4], requires_grad=True)

    def actual(theta):
        return trace(
            model.morphism,
            theta.new_zeros(1, 1),
            dict(zip(("center", "spread", "b"), theta.unbind(), strict=True)),
        ).log_joint.sum()

    def expected(theta):
        center, spread, value = theta.unbind()
        base = torch.distributions.Normal(center, spread)
        if kind == "restricted":
            return (
                base.log_prob(value)
                - (
                    base.cdf(theta.new_tensor(0.8)) - base.cdf(theta.new_tensor(-0.5))
                ).log()
            )
        if kind == "mixture":
            return torch.logsumexp(
                torch.stack(
                    (
                        theta.new_tensor(0.3).log() + base.log_prob(value),
                        theta.new_tensor(0.7).log()
                        + torch.distributions.Normal(-1.0, 2.0).log_prob(value),
                    )
                ),
                0,
            )
        if kind == "logit_normal":
            return base.log_prob(torch.logit(value)) - value.log() - torch.log1p(-value)
        return base.log_prob(value)

    torch.testing.assert_close(actual(point), expected(point), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(expected, point),
        atol=2e-10,
        rtol=2e-10,
    )
    assert torch.autograd.gradcheck(actual, (point,))
    reference = model.run(
        "model", data={"center": 0.2, "spread": 0.7, "out": (0.0,)}, sites={"b": 0.4}
    )
    assert reference.log_joint == pytest.approx(expected(point).item(), abs=2e-12)


def test_large_integer_parameter_is_preserved_before_runtime_domain_validation():
    prior = _auxiliary_draw("count", "Binomial(total_count=9007199254740993,probs=0.5)")
    count = next(
        argument.value
        for argument in prior.args
        if isinstance(argument, DrawArgNamed) and argument.parameter == "total_count"
    )
    assert isinstance(count, DrawArgScalar) and type(count.value) is int
    assert count.value == 9007199254740993
    independent = _auxiliary_draw("b", "Independent(Normal(0,1),0)")
    base, dimensions = independent.args
    assert isinstance(base, DrawArgNamed) and isinstance(base.value, DrawArgDist)
    assert isinstance(dimensions, DrawArgNamed) and isinstance(
        dimensions.value, DrawArgScalar
    )
    assert type(dimensions.value.value) is int and dimensions.value.value == 0


@pytest.mark.parametrize(
    "prior",
    [
        "Normal(0,1",
        "Normal(0,scale=1)",
        "Normal(loc=0,loc=1,scale=2)",
        "Normal(mu=0,scale=1)",
        "Normal()",
        "Normal(0,1,2)",
        "NotAFamily(0,1)",
        "Restrict(Normal(0,1),low=-1,high=1)",
        "Normal(0,1)\nsample extra <- Normal(0,1)",
    ],
)
def test_malformed_prior_is_refused_by_the_parser_or_shared_schema(prior):
    with pytest.raises(ValueError):
        _auxiliary_draw("b", prior)


def _frame(rows, groups):
    return pd.DataFrame(
        {
            "y": [0.1 + i / 10 for i in range(rows)],
            "x": [i + 1.0 for i in range(rows)],
            "z": [(-1.0) ** i for i in range(rows)],
            "g": groups,
        }
    )


def test_default_templates_parse_once_per_forward_and_never_leak_across_formulas():
    first = formula_from_data("y~x+z+(1|g)", _frame(4, ["a", "b", "a", "b"]))
    second = formula_from_data("y~x+z+(1|g)", _frame(5, ["c", "d", "e", "c", "d"]))
    lens = FormulaToQVRModule(families["gaussian"])
    with patch("quivers.dsl.parser.parse", wraps=parse) as parser:
        first_module, first_complement = lens.forward(first)
        retained_source = module_to_source(first_module)
        first_templates = lens._prior_templates
        second_module, second_complement = lens.forward(second)
        assert lens._prior_templates is not first_templates
        third_module, third_complement = lens.forward(first)
    calls = [call.args[0] for call in parser.call_args_list]
    assert sum("<- Normal(0.0, 5.0)\n" in source for source in calls) == 3
    assert (
        module_to_source(first_module)
        == retained_source
        == module_to_source(third_module)
    )
    assert lens.backward(first_module, first_complement) == first
    assert lens.backward(second_module, second_complement) == second
    assert lens.backward(third_module, third_complement) == first
    assert "FinSet 2" in retained_source and "FinSet 3" in module_to_source(
        second_module
    )


def test_explicit_integer_scale_stays_literal_and_default_scale_uses_the_column_rms():
    formula = formula_from_data("y~0+x", _frame(4, ["a", "b", "a", "b"]))
    explicit = FormulaToQVRModule(
        families["gaussian"], user_priors={"beta_x": "Normal(0,1)"}
    )
    module, _ = explicit.forward(formula)
    assert "sample beta_x <- Normal(loc=0, scale=1)" in module_to_source(module)
    default = FormulaToQVRModule(families["gaussian"], fixed_prior="Normal(0,1)")
    module, _ = default.forward(formula)
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    beta = next(
        step for step in program.draws if getattr(step, "vars", ()) == ("beta_x",)
    )
    scale = next(arg.value for arg in beta.args if arg.parameter == "scale")
    assert isinstance(scale, DrawArgScalar)
    assert scale.value == pytest.approx(
        1 / (sum(x * x for x in (1, 2, 3, 4)) / 4) ** 0.5
    )


def test_symbolic_default_scale_needs_no_predictor_data_pass():
    import numpy as np

    prior = _auxiliary_draw("b", "Normal(0,spread)")
    with patch(
        "quivers.formulas.compile.np.mean", side_effect=AssertionError("unused RMS")
    ):
        assert _autoscaled_prior_step(prior, np.array([1.0, 2.0])) is prior
