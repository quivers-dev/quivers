"""The source a program's formulas emit, and the composer that emits it.

Oracles: the expected program text, written by hand from the program and the
links of its parameters' supports; and the syntax of the composer, which must
not ask what kind of kernel it composes.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from quivers.formulas import _kernel, compile as compiler, formula_to_qvr

from tests.formulas._program_cases import BOUNDED, HALFWIDTH, SIMPLEX, frame

_KIND_NAMES = frozenset(
    {"kernel_law", "ProgramKernelSpec", "mixture", "MixtureSpec", "program_family"}
)
"""Names a branch on the kind of kernel would read."""


def _kind_tests(source: str) -> list[str]:
    """Return the kind names a function's body reads."""
    found = []
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        name = (
            node.attr
            if isinstance(node, ast.Attribute)
            else node.id
            if isinstance(node, ast.Name)
            else None
        )
        if name in _KIND_NAMES:
            found.append(name)
    return found


@pytest.mark.parametrize(
    "function",
    [
        _kernel.compose,
        _kernel.KernelEmission.emit_slot,
        _kernel.KernelEmission.bind_slot,
        compiler.FormulaToQVRModule._build_module,
    ],
)
def test_the_composer_does_not_ask_what_kind_of_kernel_it_composes(function) -> None:
    assert _kind_tests(inspect.getsource(function)) == []


def test_the_kind_check_sees_a_branch() -> None:
    mutated = "def compose(kernel):\n    if kernel.family.kernel_law is not None:\n        pass\n"
    assert _kind_tests(mutated) == ["kernel_law"]


def test_a_bounded_rate_takes_its_scaled_logit() -> None:
    source = formula_to_qvr(
        program=BOUNDED, parameters={"lapse": "~ 1"}, response="accept", data=frame()
    )
    assert source == (
        "object Resp : FinSet 12\n"
        "\n"
        "program model : Resp -> Resp\n"
        "    sample dpar_lapse_intercept <- Normal(loc=0.0, scale=5.0)\n"
        "    let dpar_lapse_eta = dpar_lapse_intercept\n"
        "    let lapse = 0.0 + 0.5 * sigmoid(dpar_lapse_eta)\n"
        "    sample theta <- Normal(loc=0.0, scale=1.0)\n"
        "    let p = lapse + (1.0 - 2.0 * lapse) * sigmoid(theta)\n"
        "    observe accept : Resp <- Bernoulli(probs=p)\n"
        "    return accept\n"
        "\n"
        "export model\n"
    )


def test_a_simplex_takes_a_softmax_with_a_reference_coordinate() -> None:
    source = formula_to_qvr(
        program=SIMPLEX,
        parameters={"weights": "~ 1"},
        response="y",
        data=frame(),
        data2={"halfwidth": HALFWIDTH},
    )
    assert "object dpar_weights_Res : FinSet 2" in source
    assert (
        "    let dpar_weights_eta = factor category : Res in "
        "{0 -> 0.0, 1 -> dpar_weights_eta_free[0], 2 -> dpar_weights_eta_free[1]}\n"
    ) in source
    assert "    let weights = softmax(dpar_weights_eta)\n" in source
    assert "Dirichlet" not in source
