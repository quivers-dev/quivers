"""PyMC selection preserves symbolic conditions after factor unrolling."""

import ast
from typing import Any

import numpy as np
import pymc
import pytest
import pytensor
import pytensor.tensor as pt

from quivers.dsl import parse
from quivers.transpile import transpile


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("integer", [False, True])
def test_shared_select_preserves_values_and_branch_gradients(checked, integer):
    carrier = "Int" if integer else "Real"
    source = (
        f"define choose(x:{carrier}):{carrier} !{{}} =\n"
        "    return select(1 < 2, x, -x)\n"
        if checked
        else "object Out:Real 1\nprogram p:Out->Out\n"
        "    sample x <- Normal(loc=0.0,scale=1.0)\n"
        "    let chosen = select(1 < 2,x,-x)\n"
        "    return chosen\nexport p\n"
    )
    emitted = transpile(parse(source), target="pymc")
    definitions = [
        node
        for node in ast.parse(emitted).body
        if isinstance(node, ast.FunctionDef) and node.name == "_qvr_select"
    ]
    assert len(definitions) == 1
    namespace: dict[str, Any] = {"pymc": pymc, "np": np, "pt": pt}
    exec(emitted, namespace)
    choose = namespace["_qvr_select"]
    values = pt.vector("values", dtype="int64" if integer else "float64")
    point = np.array([-2, 1, 3], dtype="int64" if integer else "float64")
    for condition, mask in (
        (True, True),
        (False, False),
        (np.bool_(True), True),
        (np.array([True, False, True]), np.array([True, False, True])),
        (values > 0, point > 0),
    ):
        actual = choose(condition, 2 * values, -3 * values)
        outputs = [actual] if integer else [actual, pt.grad(actual.sum(), values)]
        evaluate = pytensor.function([values], outputs)
        result = evaluate(point)
        assert result is not None
        np.testing.assert_array_equal(result[0], np.where(mask, 2 * point, -3 * point))
        if not integer:
            np.testing.assert_array_equal(
                result[1], np.where(mask, 2.0, -3.0) * np.ones(3)
            )


def test_unrolled_comparisons_build_and_score_the_expected_density():
    source = """object Row:FinSet 3
program p(y:Tensor[Real]([Row])):Row->Row
    sample x <- Normal(loc=0.0,scale=1.0)
    let selected = factor j:Row in select(j <= 1, x, -x)
    observe y:Row <- Normal(loc=selected,scale=1.0)
    return y
export p
"""
    namespace: dict[str, Any] = {"pymc": pymc, "np": np, "pt": pt}
    exec(transpile(parse(source), target="pymc"), namespace)
    observed = np.array([0.2, -0.6, 0.7])
    model = namespace["build_model"](y=observed)
    logp = model.compile_logp()
    from scipy.stats import norm

    for value in (-1.3, 0.0, 0.8):
        expected = (
            norm.logpdf(value) + norm.logpdf(observed, [value, value, -value]).sum()
        )
        assert logp({"x": value}) == pytest.approx(expected, rel=1e-12, abs=1e-12)
