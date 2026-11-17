"""Portable static metadata preserves Nat sums through host specialization."""

from pathlib import Path
import json
import subprocess

from quivers.dsl import Compiler, parse
from quivers.transpile.qiec_ir import IRQiecIndexSum, lower_qiec_ir
from quivers.transpile._runtime_sources import javascript_runtime_source
from tests.transpile._tools import require_scheme, require_tool
from tests.transpile._webppl import run_webppl

RUNTIME = Path(__file__).parents[2] / "src/quivers/transpile"


def literal(value):
    return {"kind": "index-literal", "value": value, "sort": {"kind": "nat"}}


def variable(name, identity=None):
    return {
        "kind": "index-variable",
        "name": name,
        "identity": identity,
        "sort": {"kind": "nat"},
    }


def index_sum(*terms, constant=0):
    return {"kind": "index-sum", "terms": list(terms), "constant": constant}


def test_checked_ir_retains_nominal_sum_structure():
    compiler = Compiler(
        parse(
            "define f[n : Nat](x : Tensor[Real]([nat_add(n, 1)])) : Tensor[Real]([nat_add(n, 1)]) !{} =\n    return x\n"
        )
    )
    compiler.compile()
    ir = lower_qiec_ir(compiler.qiec_module)
    dimension = ir.computations[0].parameters[0].type.arguments[1].dimensions[0]
    assert isinstance(dimension, IRQiecIndexSum)
    assert dimension.constant == 1
    assert dimension.terms[0].name == "n"


def test_shared_python_specialization_folds_and_flattens_without_name_capture():
    namespace = {}
    exec((RUNTIME / "runtime_qiec.py").read_text(), namespace)
    specialize = namespace["_qvr_qiec_specialize"]
    n = variable("n")
    assert specialize(index_sum(n, constant=1), {"n": literal(3)}) == literal(4)
    assert specialize(
        index_sum(n, constant=1), {"n": index_sum(variable("m"), constant=2)}
    ) == index_sum(variable("m"), constant=3)
    assert specialize(
        index_sum(variable("a"), variable("b")),
        {"a": variable("z"), "b": variable("c")},
    ) == index_sum(variable("c"), variable("z"))
    rigid = variable("n", "qiec:static-var:" + "f" * 64)
    assert specialize(index_sum(rigid, constant=1), {"n": literal(3)}) == index_sum(
        rigid, constant=1
    )


def test_actual_webppl_specialization_uses_functional_arithmetic(tmp_path):
    cases = [
        (index_sum(variable("n"), constant=1), {"n": literal(3)}),
        (
            index_sum(variable("n"), constant=1),
            {"n": index_sum(variable("m"), constant=2)},
        ),
    ]
    calls = ",".join(
        f"_qvr_qiec_specialize({json.dumps(term)}, {json.dumps(environment)})"
        for term, environment in cases
    )
    path = tmp_path / "index-sums.wppl"
    path.write_text(
        javascript_runtime_source(RUNTIME / "runtime_qiec.js").decode()
        + f"\ndisplay(JSON.stringify([{calls}]));\n"
    )
    result = run_webppl(path)
    decoded = json.loads(result.stdout.strip())
    assert decoded == [literal(4), index_sum(variable("m"), constant=3)]


def test_actual_scheme_specialization_preserves_static_extent_arithmetic(tmp_path):
    script = (
        (RUNTIME / "runtime_qiec.scm").read_text()
        + r"""
(define term '(("kind" . "index-sum") ("constant" . 1)
               ("terms" . ((("kind" . "index-variable") ("name" . "n"))))))
(define closed '(("n" . (("kind" . "index-literal") ("value" . 3) ("sort" . (("kind" . "nat")))))))
(define nested '(("n" . (("kind" . "index-sum") ("constant" . 2)
                         ("terms" . ((("kind" . "index-variable") ("name" . "m"))))))))
(define first (_qvr-qiec-specialize term closed))
(unless (and (equal? (_qvr-qiec-get first "kind") "index-literal")
             (= (_qvr-qiec-get first "value") 4)) (error 'test "closed sum"))
(define second (_qvr-qiec-specialize term nested))
(unless (and (= (_qvr-qiec-get second "constant") 3)
             (equal? (_qvr-qiec-get (car (_qvr-qiec-get second "terms")) "name") "m"))
        (error 'test "nested sum"))
(display "ok")
"""
    )
    path = tmp_path / "index-sums.scm"
    path.write_text(script)
    result = subprocess.run(
        [require_scheme(), "--script", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_actual_julia_specialization_preserves_static_extent_arithmetic(tmp_path):
    script = (
        (RUNTIME / "runtime_qiec.jl").read_text()
        + """
term = Dict("kind"=>"index-sum", "constant"=>1, "terms"=>[Dict("kind"=>"index-variable", "name"=>"n")])
closed = Dict("n"=>Dict("kind"=>"index-literal", "value"=>3, "sort"=>Dict("kind"=>"nat")))
@assert _qvr_qiec_specialize(term, closed)["value"] == 4
nested = Dict("n"=>Dict("kind"=>"index-sum", "constant"=>2, "terms"=>[Dict("kind"=>"index-variable", "name"=>"m")]))
result = _qvr_qiec_specialize(term, nested)
@assert result["constant"] == 3 && result["terms"][1]["name"] == "m"
println("ok")
"""
    )
    path = tmp_path / "index-sums.jl"
    path.write_text(script)
    result = subprocess.run(
        [require_tool("julia"), "--startup-file=no", str(path)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
