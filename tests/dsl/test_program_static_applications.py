"""Static program application uses the same checked interface in QVR and Python."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import CompileError, loads, parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.inference.trace import trace

HEAD = "object Row : FinSet 3\nobject Out : Real 1\n"
NAT = """program sub[n : Nat](scale : Real) : n -> Out
    sample z : n <- Normal(loc=0.0, scale=scale)
    return z
"""


def _density(program, values, **data):
    recorded = trace(
        program.morphism, torch.zeros(1, 1), observations={"z": values, **data}
    )
    assert recorded.log_joint is not None
    return recorded.log_joint.sum()


def test_static_natural_define_application_has_correct_density_and_gradient() -> None:
    program = loads(HEAD + NAT + "define applied = sub[3](sigma)\nexport applied\n")
    sigma = torch.tensor(0.7, requires_grad=True)
    values = torch.tensor((0.1, -0.3, 0.5), requires_grad=True)
    actual = _density(program, values, sigma=sigma)
    expected = torch.distributions.Normal(0.0, sigma).log_prob(values).sum()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, (values, sigma), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (values, sigma))
    for received, wanted in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(received, wanted)
    assert program.morphism.input_contract.telescope == ()
    assert tuple(p.name for p in program.morphism.input_contract.parameters) == (
        "sigma",
    )


def test_static_natural_python_factory_retains_partial_runtime_inputs() -> None:
    family = loads(HEAD + NAT + "export sub\n")
    program = family.sub(n=3)
    sigma = torch.tensor(0.7, requires_grad=True)
    values = torch.tensor((0.1, -0.3, 0.5))
    actual = _density(program, values, scale=sigma)
    expected = torch.distributions.Normal(0.0, sigma).log_prob(values).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, sigma, retain_graph=True)[0],
        torch.autograd.grad(expected, sigma)[0],
    )
    assert tuple(p.name for p in program.morphism.input_contract.parameters) == (
        "scale",
    )
    assert float(_density(family.sub(n=4, scale=1.0), torch.zeros(4))) == pytest.approx(
        -2 * math.log(2 * math.pi)
    )
    assert float(_density(family.sub(n=3, scale=1.0), torch.zeros(3))) == pytest.approx(
        -1.5 * math.log(2 * math.pi)
    )


def test_type_application_preserves_input_contract_and_autograd() -> None:
    source = (
        HEAD
        + """program typed_identity[A : Type](x : A) : Row -> Out
    return x
"""
    )
    explicit = loads(
        source + "define applied = typed_identity[Real](features)\nexport applied\n"
    )
    partial = loads(source + "export typed_identity\n").typed_identity(A="Real")
    for program, name in ((explicit, "features"), (partial, "x")):
        value = torch.tensor(2.5, requires_grad=True)
        result = trace(program.morphism, torch.zeros(1, 1), observations={name: value})
        torch.testing.assert_close(result.output, value)
        assert torch.autograd.grad(result.output.sum(), value)[0] == 1.0
        assert tuple(p.name for p in program.morphism.input_contract.parameters) == (
            name,
        )


def test_typed_tensor_define_application_has_correct_density_and_gradients() -> None:
    source = (
        HEAD
        + """program sub[A : FinSet](x : Tensor[Real]([A]), s : Real) : Row -> Out
    sample z : A <- Normal(loc=x, scale=s)
    return z

define applied = sub[Row](features, sigma)
export applied
"""
    )
    program = loads(source)
    features = torch.tensor((0.2, -0.4, 0.6), requires_grad=True)
    sigma = torch.tensor(0.5, requires_grad=True)
    values = torch.tensor((0.1, -0.3, 0.5))
    actual = _density(program, values, features=features, sigma=sigma)
    expected = torch.distributions.Normal(features, sigma).log_prob(values).sum()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, (features, sigma), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (features, sigma))
    for received, wanted in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(received, wanted)


def test_typed_application_preserves_labeled_tuple_outputs() -> None:
    source = (
        HEAD
        + """program sub[n : Nat] : n -> Out * Out
    sample a : n <- Normal(loc=0.0, scale=1.0)
    sample b : n <- Normal(loc=0.0, scale=1.0)
    return (first: a, second: b)
define applied = sub[3]()
export applied
"""
    )
    program = loads(source)
    a, b = torch.ones(3), torch.zeros(3)
    result = trace(program.morphism, torch.zeros(1, 1), observations={"a": a, "b": b})
    assert set(result.output) == {"first", "second"}
    torch.testing.assert_close(result.output["first"], a)
    torch.testing.assert_close(result.output["second"], b)


@pytest.mark.parametrize(
    "application", ["sub(Row, sigma)", "sub[Row, 0.5]()", "sub[Row](Row)"]
)
def test_top_level_application_rejects_misplaced_static_and_runtime_arguments(
    application: str,
) -> None:
    source = (
        HEAD
        + """program sub[A : FinSet](s : Real) : Row -> Out
    sample z : A <- Normal(loc=0.0, scale=s)
    return z
"""
        + f"define applied = {application}\nexport applied\n"
    )
    with pytest.raises(CompileError, match="static|runtime"):
        loads(source)


@pytest.mark.parametrize("side", ["domain", "codomain"])
def test_an_unused_morphism_actual_still_checks_its_declared_signature(
    side: str,
) -> None:
    head = HEAD + "object Other : FinSet 7\nobject Wide : Real 2\n"
    domain = "Other" if side == "domain" else "Row"
    codomain = "Wide" if side == "codomain" else "Out"
    declaration = f"morphism wrong : {domain} -> {codomain} [role=kernel] ~ Normal\n"
    template = """program sub[prior : Mor[Row, Out]] : Row -> Out
    let z = 0.0
    return z
"""
    source = (
        head
        + declaration
        + template
        + "program main : Row -> Out\n    sample result <- sub[wrong]()\n    return result\nexport main\n"
    )
    with pytest.raises(QiecDiagnosticError, match=f"incompatible {side}"):
        lower_qvr_to_qiec(parse(source))
    with pytest.raises(CompileError, match=f"incompatible {side}"):
        loads(source)
    top = (
        head
        + declaration
        + template
        + "define applied = sub[wrong]()\nexport applied\n"
    )
    with pytest.raises(CompileError, match=f"incompatible {side}"):
        loads(top)


def test_a_program_morphism_actual_uses_its_declared_signature() -> None:
    source = (
        HEAD
        + """program prior : Row -> Out
    sample z <- Normal(loc=0.0, scale=1.0)
    return z
program sub[p : Mor[Row, Out]] : Row -> Out
    sample z <- p
    return z
define applied = sub[prior]()
export applied
"""
    )
    program = loads(source)
    result = trace(
        program.morphism, torch.zeros(1, 1), observations={"z": torch.tensor(0.2)}
    )
    assert result.log_joint is not None
    assert float(result.log_joint.sum()) == pytest.approx(
        -0.5 * (0.2**2 + math.log(2 * math.pi))
    )


def test_core_and_repl_telescopes_keep_the_authored_static_order() -> None:
    from quivers.cli.repl_browse import _program_parameter_groups
    from quivers.dsl.ast_nodes import ProgramDecl
    from quivers.qiec.program_runtime import program_entry

    source = (
        HEAD
        + """program sub[n : Nat, A : Type, K : FinSet, E : Effect](x : A) : Row -> Out
    return x
"""
    )
    authored = parse(source)
    template = next(
        statement
        for statement in authored.statements
        if isinstance(statement, ProgramDecl)
    )
    entry = program_entry(lower_qvr_to_qiec(authored), "sub")
    assert [binder.name for binder in entry.telescope] == ["n", "A", "K", "E"]
    static, _ = _program_parameter_groups(template)
    assert [parameter.split(":")[0].strip() for parameter in static] == [
        "n",
        "A",
        "K",
        "E",
    ]


def test_bound_scalar_factory_parameters_are_removed_from_external_contract() -> None:
    source = (
        HEAD
        + """program sub(scale : Real) : Row -> Out
    sample z <- Normal(loc=0.0, scale=scale)
    return z
export sub
"""
    )
    program = loads(source).sub(scale=0.7)
    assert program.morphism.input_contract.parameters == ()
    assert program.morphism.host_data_types == {}
    actual = _density(program, torch.tensor(0.1))
    expected = -math.log(0.7 * math.sqrt(2 * math.pi)) - 0.5 * (0.1 / 0.7) ** 2
    assert float(actual) == pytest.approx(expected)


def test_scalar_source_actual_renames_the_external_input_contract() -> None:
    source = (
        HEAD
        + """program sub(scale : Real) : Row -> Out
    sample z <- Normal(loc=0.0, scale=scale)
    return z
define applied = sub(sigma)
export applied
"""
    )
    program = loads(source)
    assert tuple(
        parameter.name for parameter in program.morphism.input_contract.parameters
    ) == ("sigma",)
    assert set(program.morphism.host_data_types) == {"sigma"}
    actual = _density(program, torch.tensor(0.1), sigma=torch.tensor(0.7))
    expected = -math.log(0.7 * math.sqrt(2 * math.pi)) - 0.5 * (0.1 / 0.7) ** 2
    assert float(actual) == pytest.approx(expected)


def test_shared_scalar_actual_appears_once_in_the_external_contract() -> None:
    source = (
        HEAD
        + """program sub(mu : Real, scale : Real) : Row -> Out
    sample z <- Normal(loc=mu, scale=scale)
    return z
define applied = sub(shared, shared)
export applied
"""
    )
    program = loads(source)
    assert tuple(
        parameter.name for parameter in program.morphism.input_contract.parameters
    ) == ("shared",)
    assert float(
        _density(program, torch.tensor(1.0), shared=torch.tensor(1.0))
    ) == pytest.approx(-0.5 * math.log(2 * math.pi))


def test_one_runtime_actual_cannot_fill_incompatible_unused_parameter_types() -> None:
    source = (
        HEAD
        + """program sub(mu : Real, count : Nat) : Row -> Out
    let z = 0.0
    return z
define applied = sub(shared, shared)
export applied
"""
    )
    with pytest.raises(CompileError, match="incompatible types"):
        loads(source)


def test_factory_requires_tensor_values_to_be_bound_as_data() -> None:
    source = (
        HEAD
        + """program sub[A : Type](x : A) : Row -> Out
    return x
export sub
"""
    )
    with pytest.raises(TypeError, match="Model.bind"):
        loads(source).sub(A="Real", x=torch.tensor(1.0))
