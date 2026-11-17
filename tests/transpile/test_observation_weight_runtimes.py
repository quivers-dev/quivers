"""Every host runtime validates likelihood weights and preserves absorbing zero."""

from __future__ import annotations

from collections.abc import Callable
import json
import math
from pathlib import Path
import subprocess
from typing import cast

import pytest

from quivers.dsl import parse
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile._runtime_sources import javascript_runtime_source
from tests.transpile._tools import require_scheme, require_tool
from tests.transpile._webppl import run_webppl

SOURCE = """define checked_weight(w : Real) : Real !{} =
    return observation_weight(w)
define scaled_density(w : Real, density : Real) : Real !{} =
    return weight_value(scale_weight(observation_weight(w), weight(density)))
"""

WEIGHTED_OBSERVE = """object Row : FinSet 3
object Out : Real 1
program main : Row -> Out
    observe y : Row <- Normal(loc=0.0, scale=1.0) [weights=w]
    return y
export main
"""


def test_generated_python_runtime_validates_scalar_and_tensor_weights() -> None:
    namespace: dict[str, object] = {}
    exec(transpile(parse(SOURCE), target="pyro"), namespace)
    primitive = cast(Callable, namespace["_qvr_qiec_primitive"])
    checked = cast(Callable, namespace["qiec_checked_weight"])
    scaled = cast(Callable, namespace["qiec_scaled_density"])
    for value in (0.0, -0.0, 0.5, 2.0):
        assert checked(value) == value
    for value in (-1.0, math.inf, -math.inf, math.nan, True, False):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            primitive("observation_weight", (value,))
    assert primitive("observation_weight", ((0.0, 2.0, 0.5),)) == (0.0, 2.0, 0.5)
    for value in ((0.0, -1.0), (0.0, math.nan), (0.0, True)):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            primitive("observation_weight", (value,))
    assert scaled(0.0, -math.inf) == 0.0
    assert primitive("scale_weight", ((0.0, 2.0, 0.5), (-math.inf, -3.0, -4.0))) == (
        0.0,
        -6.0,
        -2.0,
    )


def test_javascript_runtime_checks_values_and_tensor_broadcasting(
    tmp_path: Path,
) -> None:
    runtime = Path(__file__).parents[2] / "src/quivers/transpile/runtime_qiec.js"
    script = tmp_path / "weights.js"
    script.write_text(
        "var globalStore = {};\n"
        "var map = (fn, xs) => xs.map(fn);\n"
        "var filter = (fn, xs) => xs.filter(fn);\n"
        "var any = (fn, xs) => xs.some(fn);\n"
        "var mapN = (fn, n) => Array.from({length: n}, (_, i) => fn(i));\n"
        "var _ = {has: (value, key) => Object.prototype.hasOwnProperty.call(value, key)};\n"
        "var apply = (fn, xs) => fn.apply(null, xs);\n"
        "var error = message => { throw new Error(message); };\n"
        + javascript_runtime_source(runtime).decode()
        + """
var invalid = [-1, Infinity, -Infinity, NaN, true, false, [0, -1], [0, NaN], [0, true]];
var rejected = invalid.map(function(value) {
  try { _qvr_qiec_primitive("observation_weight", [value]); return false; }
  catch (failure) { return /finite and nonnegative/.test(failure.message); }
});
console.log(JSON.stringify({
  valid: _qvr_qiec_primitive("observation_weight", [[0, 2, 0.5]]),
  rejected: rejected,
  scalarZero: _qvr_qiec_primitive("scale_weight", [0, -Infinity]),
  scaled: _qvr_qiec_primitive("scale_weight", [[0, 2, 0.5], [-Infinity, -3, -4]])
}));
"""
    )
    completed = subprocess.run(
        [require_tool("node"), str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "valid": [0.0, 2.0, 0.5],
        "rejected": [True] * 9,
        "scalarZero": 0.0,
        "scaled": [0.0, -6.0, -2.0],
    }


def test_webppl_compiles_weight_validation_and_absorbing_zero(tmp_path: Path) -> None:
    script = tmp_path / "weights.wppl"
    script.write_bytes(
        transpile(parse(SOURCE), target="webppl")
        + b'\ndisplay(JSON.stringify([qiec_checked_weight(0.5), qiec_scaled_density(0, -Infinity), _qvr_qiec_primitive("observation_weight", [[0, 2, 0.5]]), _qvr_qiec_primitive("scale_weight", [[0, 2, 0.5], [-Infinity, -3, -4]])]));\n'
    )
    completed = run_webppl(script)
    assert json.loads(completed.stdout) == [
        0.5,
        0.0,
        [0.0, 2.0, 0.5],
        [0.0, -6.0, -2.0],
    ]


@pytest.mark.parametrize(
    "value",
    [
        "-1",
        "Infinity",
        "-Infinity",
        "NaN",
        "true",
        "false",
        "[0, -1]",
        "[0, NaN]",
        "[0, true]",
    ],
)
def test_webppl_rejects_invalid_weights_without_host_exception_syntax(
    tmp_path: Path, value: str
) -> None:
    script = tmp_path / "invalid-weights.wppl"
    script.write_bytes(
        transpile(parse(SOURCE), target="webppl")
        + f'\ndisplay(_qvr_qiec_primitive("observation_weight", [{value}]));\n'.encode()
    )
    completed = run_webppl(script, check=False)
    assert completed.returncode != 0
    assert "finite and nonnegative" in completed.stdout + completed.stderr


def test_generated_julia_runtime_validates_values_and_zero_products(
    tmp_path: Path,
) -> None:
    script = tmp_path / "weights.jl"
    script.write_text(
        "macro model(expression)\n    esc(expression)\nend\n"
        + transpile(parse(SOURCE), target="turing").decode()
        + """
function refused(value)
    try
        _qvr_qiec_primitive("observation_weight", Any[value])
        return false
    catch failure
        return occursin("finite and nonnegative", sprint(showerror, failure))
    end
end
invalid = Any[-1.0, Inf, -Inf, NaN, true, false, (0.0, -1.0), (0.0, NaN), (0.0, true)]
@assert all(refused(value) for value in invalid)
@assert qiec_checked_weight(0.5) == 0.5
@assert qiec_scaled_density(0.0, -Inf) == 0.0
@assert _qvr_qiec_primitive("observation_weight", Any[(0.0, 2.0, 0.5)]) == (0.0, 2.0, 0.5)
@assert _qvr_qiec_primitive("scale_weight", Any[(0.0, 2.0, 0.5), (-Inf, -3.0, -4.0)]) == (0.0, -6.0, -2.0)
println("validated")
"""
    )
    completed = subprocess.run(
        [require_tool("julia"), "--startup-file=no", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.stdout.strip() == "validated"


def test_generated_scheme_runtime_validates_values_and_zero_products(
    tmp_path: Path,
) -> None:
    script = tmp_path / "weights.scm"
    script.write_bytes(
        transpile(parse(SOURCE), target="church")
        + b"""
(define (refused value)
  (guard (failure (else #t))
    (_qvr-qiec-primitive "observation_weight" (list value))
    #f))
(define invalid (list -1.0 +inf.0 -inf.0 +nan.0 #t #f
  (_qvr-qiec-tuple (list 0.0 -1.0))
  (_qvr-qiec-tuple (list 0.0 +nan.0))
  (_qvr-qiec-tuple (list 0.0 #t))))
(if (not (for-all refused invalid)) (error 'test "invalid weights accepted"))
(if (not (= (qiec_checked_weight 0.5) 0.5)) (error 'test "identity changed value"))
(if (not (= (qiec_scaled_density 0.0 -inf.0) 0.0)) (error 'test "zero not absorbing"))
(if (not (equal?
    (_qvr-qiec-value (_qvr-qiec-primitive "observation_weight" (list (_qvr-qiec-tuple (list 0.0 2.0 0.5)))))
    (list 0.0 2.0 0.5)))
  (error 'test "tensor identity changed values"))
(if (not (equal?
    (_qvr-qiec-value (_qvr-qiec-primitive "scale_weight" (list
      (_qvr-qiec-tuple (list 0.0 2.0 0.5))
      (_qvr-qiec-tuple (list -inf.0 -3.0 -4.0)))))
    (list 0.0 -6.0 -2.0)))
  (error 'test "tensor scaling changed values"))
(display "validated") (newline)
"""
    )
    completed = subprocess.run(
        [require_scheme(), "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.stdout.strip() == "validated"


@pytest.mark.parametrize("target", tuple(available_targets()))
def test_direct_weighted_observation_is_explicitly_refused_without_losing_site_metadata(
    target: str,
) -> None:
    with pytest.raises(UnsupportedConstruct) as raised:
        transpile(parse(WEIGHTED_OBSERVE), target=target)
    assert raised.value.target == f"qvr-{target}"
    assert raised.value.kinds == ["qiec:weighted-observe:main"]
