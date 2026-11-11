"""A kernel's ``~ Family(...)`` arguments are literals or absent.

A kernel declaration either fixes its family with literal arguments or
leaves the argument list empty and takes every parameter from its
learned parameter source. An argument that reads a name has nothing to
read in a declaration; every path refuses it at the argument, and
``qvr check`` reports it as a diagnostic. Literal vectors and matrices
fix the family on the compiled runtime as on the reference machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quivers.cli import check_file
from quivers.dsl import CompileError, loads
from quivers.dsl.parser import parse
from quivers.inference.trace import trace
from quivers.transpile import UnsupportedConstruct, transpile

_STEP = """\
object Driver : Real 2
object State : Real 3
morphism transition : Driver * State -> State {options}~ {family}
program step : Driver * State -> State
    sample s <- transition(u_t, s_prev)
    return s
export step
"""

_FIXED = """\
object X : Real 2
object V : Real 3
morphism k : X -> V ~ {family}
program p : X -> V
    sample s <- k
    return s
export p
"""

# The value every density below is scored at.
_VALUE = (0.2, 0.3, 0.5)


def _source(family: str, options: str = "") -> str:
    """The transition kernel source with the given initializer and options."""
    return _STEP.format(family=family, options=f"{options} " if options else "")


def _argument_column(source: str, argument: str) -> int:
    """The zero-based column of an argument on the kernel's line."""
    return source.splitlines()[2].index(argument)


@pytest.mark.parametrize(
    ("family", "argument"),
    [
        ("Normal(scale=q)", "scale=q"),
        ("Normal(loc=0.0, scale=q)", "scale=q"),
        ("Normal(loc=[0.0, q, 0.0], scale=1.0)", "loc=[0.0, q, 0.0]"),
        ("Normal(loc=0.0, scale=s[i])", "scale=s[i]"),
        ("Normal(loc=0.0, scale=exp(q))", "scale=exp(q)"),
    ],
)
@pytest.mark.parametrize("options", ["", "[param_source=linear]"])
def test_a_kernel_argument_that_reads_a_name_is_refused_at_the_argument(
    family: str, argument: str, options: str
) -> None:
    source = _source(family, options)
    with pytest.raises(CompileError) as caught:
        loads(source)
    error = caught.value
    assert f"family argument `{argument}`" in str(error)
    assert "is not a literal" in str(error)
    assert (error.line, error.col) == (3, _argument_column(source, argument))


def test_the_compiled_runtime_no_longer_drops_a_partial_argument_list() -> None:
    """The partial list is refused rather than compiled to a learned kernel."""
    source = _source("Normal(scale=q)", "[param_source=linear]")
    with pytest.raises(CompileError, match="`scale=q`"):
        loads(source)
    # Without the argument list the kernel is the learned one.
    learned = loads(_source("Normal", "[param_source=linear]")).morphism
    assert [name for name, _ in learned.named_parameters()]


def test_the_checked_path_refuses_a_kernel_argument_that_reads_a_name() -> None:
    source = _FIXED.format(family="Dirichlet(concentration=[1.0, c, 3.0])")
    with pytest.raises(UnsupportedConstruct, match="`concentration=\\[1.0, c, 3.0\\]`"):
        transpile(parse(source), target="stan")


@pytest.mark.parametrize("options", ["", "[param_source=linear]"])
def test_qvr_check_reports_the_refusal_without_a_traceback(
    tmp_path: Path, options: str
) -> None:
    source = _source("Normal(scale=q)", options)
    path = tmp_path / "transition.qvr"
    path.write_text(source)
    diagnostics = check_file(path)
    compile_errors = [d for d in diagnostics if d.code == "compile"]
    assert len(compile_errors) == 1
    (error,) = compile_errors
    assert error.severity == "error"
    assert "`scale=q`" in error.message
    assert (error.line, error.col) == (3, _argument_column(source, "scale=q"))
    schema = [d for d in diagnostics if d.code == "family-arg-parameterization"]
    assert [d.message for d in schema] == [
        "family 'Normal' requires exactly one complete parameter schema "
        "(loc, scale); morphism 'transition' init clause supplied (scale)"
    ]


def test_qvr_check_reads_a_kernel_with_literal_arguments(tmp_path: Path) -> None:
    path = tmp_path / "fixed.qvr"
    path.write_text(_FIXED.format(family="Normal(loc=0.0, scale=1.0)"))
    assert [d for d in check_file(path) if d.severity == "error"] == []


def test_qvr_check_names_a_plural_kernel_in_a_family_diagnostic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plural.qvr"
    path.write_text("object X : Real 1\nmorphism a, b : X -> X ~ Normal(loc=0.0)\n")
    messages = [
        d.message for d in check_file(path) if d.code == "family-arg-parameterization"
    ]
    assert messages == [
        "family 'Normal' requires exactly one complete parameter schema "
        "(loc, scale); morphism 'a, b' init clause supplied (loc)"
    ]


def _scored(family: str) -> tuple[torch.Tensor, float, list[str]]:
    """Score ``_VALUE`` under a kernel declared with ``family`` on both paths.

    Returns
    -------
    tuple[torch.Tensor, float, list[str]]
        The compiled runtime's log joint, the reference machine's, and
        the compiled program's parameter names.
    """
    program = loads(_FIXED.format(family=family))
    recorded = trace(
        program.morphism,
        torch.zeros(1, 2),
        observations={"s": torch.tensor([_VALUE])},
    )
    reference = program.run(
        "p", data={"x": (0.0, 0.0)}, sites={"s": _VALUE}, seed=0
    ).log_joint
    names = [name for name, _ in program.morphism.named_parameters()]
    return recorded.log_joint, float(reference), names


_SCALE_TRIL = torch.tensor([[1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 0.5]])


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        (
            "Dirichlet(concentration=[1.0, 2.0, 3.0])",
            torch.distributions.Dirichlet(torch.tensor([1.0, 2.0, 3.0])),
        ),
        (
            "MultivariateNormal(loc=[0.0, 1.0, 0.0], scale_tril=[[1.0, 0.0, 0.0], "
            "[0.5, 2.0, 0.0], [0.0, 0.0, 0.5]])",
            torch.distributions.MultivariateNormal(
                torch.tensor([0.0, 1.0, 0.0]), scale_tril=_SCALE_TRIL
            ),
        ),
        (
            "Normal(loc=[0.0, 1.0, 2.0], scale=2.0)",
            torch.distributions.Independent(
                torch.distributions.Normal(torch.tensor([0.0, 1.0, 2.0]), 2.0), 1
            ),
        ),
    ],
)
def test_literal_vector_and_matrix_arguments_fix_the_family_on_both_paths(
    family: str, expected: torch.distributions.Distribution
) -> None:
    compiled, reference, names = _scored(family)
    target = float(expected.log_prob(torch.tensor(_VALUE)))
    # Float32 density arithmetic on the compiled runtime; float64 on the
    # reference machine.
    assert names == []
    assert compiled.shape == (1,)
    assert float(compiled[0]) == pytest.approx(target, abs=1e-5)
    assert reference == pytest.approx(target, abs=1e-5)


def test_the_literal_check_distinguishes_the_declared_concentration() -> None:
    """A Dirichlet with other literals scores the same value differently."""
    compiled, _, _ = _scored("Dirichlet(concentration=[1.0, 2.0, 3.0])")
    flat = torch.distributions.Dirichlet(torch.ones(3)).log_prob(torch.tensor(_VALUE))
    assert abs(float(compiled[0]) - float(flat)) > 0.5
