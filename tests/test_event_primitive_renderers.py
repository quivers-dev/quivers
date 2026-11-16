"""Recorded-event validators keep checked domains on external runtimes."""

import ast
from functools import lru_cache
import json
import math
import re
from pathlib import Path
import subprocess

import panproto
import pytest
import torch

from quivers.dsl import parse
from quivers.transpile import transpile, UnsupportedConstruct
from tests.transpile._tools import require_tool, require_scheme
from tests.transpile._webppl import run_webppl

RUNTIME = Path(__file__).resolve().parents[1] / "src/quivers/transpile"
TARGETS = {
    "pyro": "python",
    "pymc": "python",
    "numpyro": "python",
    "edward2": "python",
    "church": "scheme",
    "webppl": "javascript",
    "turing": "julia",
    "gen": "julia",
    "stan": "stan",
    "bugs": "bugs",
    "jags": "jags",
}


def _helpers(host):
    tree = ast.parse((RUNTIME / f"runtime_probability_{host}.py").read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), host, "exec"), namespace
    )
    return namespace


@lru_cache
def _rendered(name, target, guarded=False):
    if name == "observation_status":
        definition = (
            """define checked(status:Int) : Int !{} =
    if status < -1 then
        return 0
    else
        return observation_status(status)
"""
            if guarded
            else ""
        )
        header = "status:Int,y:Real"
        body = (
            "let code <- checked(status)"
            if guarded
            else "let code = observation_status(status)"
        )
        body += "\n    let mu = real(code)"
    else:
        definition = (
            """define checked(left:Real,right:Real) : Real !{} =
    if left == right then
        return 0.0
    else
        return interval_upper(left,right)
"""
            if guarded
            else ""
        )
        header = "left:Real,right:Real,y:Real"
        body = (
            "let mu <- checked(left,right)"
            if guarded
            else "let mu = interval_upper(left,right)"
        )
    source = f"""object Out:Real 1
{definition}program model({header}):Out->Out
    {body}
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export model
"""
    return transpile(parse(source), target=target)


def test_torch_event_validator_domains_broadcasting_and_derivatives():
    namespace = _helpers("pyro")
    status = namespace["_qvr_observation_status"]
    values = torch.tensor([-1, 0, 1, 2])
    assert status(values) is values
    for bad in [True, 1.0, 3, -2, math.nan, math.inf, complex(1, 0)]:
        with pytest.raises(ValueError):
            status(bad)
    upper = namespace["_qvr_interval_upper"]
    low = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    high = torch.tensor([0.1, 1.2], dtype=torch.float64, requires_grad=True)
    assert upper(low, high) is high
    torch.testing.assert_close(
        torch.func.jacrev(upper, argnums=(0, 1))(low, high)[0],
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.func.jacrev(upper, argnums=(0, 1))(low, high)[1],
        torch.eye(2, dtype=torch.float64),
    )
    assert torch.equal(
        upper(torch.tensor(-math.inf), torch.tensor(1.0)), torch.tensor(1.0)
    )
    for low_value, high_value in [
        (0.0, 0.0),
        (2.0, 1.0),
        (math.nan, 1.0),
        (0.0, math.inf),
        (0.0, -math.inf),
        (True, 1.0),
        (0.0, True),
    ]:
        with pytest.raises(ValueError):
            upper(low_value, high_value)


def test_pymc_event_validators_preserve_integer_carriers_and_real_gradients():
    import pymc
    import pytensor
    import pytensor.tensor as pt

    namespace = _helpers("pymc")
    for helper in list(namespace.values()):
        if callable(helper) and hasattr(helper, "__globals__"):
            helper.__globals__["pymc"] = pymc
    status = pt.lscalar("status")
    evaluate = pytensor.function(
        [status], namespace["_qvr_observation_status"](status), mode="FAST_COMPILE"
    )
    for value in [-1, 0, 1, 2]:
        assert evaluate(value) == value
    for value in [-2, 3]:
        with pytest.raises(AssertionError):
            evaluate(value)
    for value in [True, pt.dscalar(), pt.constant(True, dtype="bool")]:
        with pytest.raises(ValueError):
            namespace["_qvr_observation_status"](value)
    low, high = pt.dscalars("low", "high")
    value = namespace["_qvr_interval_upper"](low, high)
    evaluate = pytensor.function(
        [low, high],
        [
            value,
            pt.grad(value, low, disconnected_inputs="ignore"),
            pt.grad(value, high),
        ],
        mode="FAST_COMPILE",
    )
    assert [float(x) for x in evaluate(-0.2, 1.3)] == pytest.approx([1.3, 0.0, 1.0])
    for a, b in [(0.0, 0.0), (2.0, 1.0), (math.nan, 1.0), (0.0, math.inf)]:
        with pytest.raises(AssertionError):
            evaluate(a, b)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("name", ["observation_status", "interval_upper"])
def test_ordinary_event_validator_renderers_define_helpers_or_refuse(target, name):
    if target in {"bugs", "jags"}:
        with pytest.raises(UnsupportedConstruct, match=name):
            _rendered(name, target)
        return
    source = _rendered(name, target)
    assert (
        panproto.AstParserRegistry()
        .parse_with_protocol(TARGETS[target], source, name)
        .vertex_count
    )
    if TARGETS[target] == "python":
        compile(source, name, "exec")


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("name", ["observation_status", "interval_upper"])
def test_pyro_emitted_event_validators_keep_inactive_branches_lazy(name, guarded):
    import pyro

    namespace = {"torch": torch, "pyro": pyro}
    exec(_rendered(name, "pyro", guarded), namespace)

    def score(*values):
        recorded = pyro.poutine.trace(namespace["model"]).get_trace(
            torch.zeros(1), *values, torch.tensor(0.3, dtype=torch.float64)
        )
        return recorded.log_prob_sum()

    args = (
        (torch.tensor(2),)
        if name == "observation_status"
        else (
            torch.tensor(0.1, dtype=torch.float64),
            torch.tensor(1.2, dtype=torch.float64, requires_grad=True),
        )
    )
    mean = 2.0 if name == "observation_status" else 1.2
    expected = -0.5 * (0.3 - mean) ** 2 - 0.5 * math.log(2 * math.pi)
    assert float(score(*args).detach()) == pytest.approx(expected, abs=1e-6)
    if name == "interval_upper":
        torch.testing.assert_close(
            torch.autograd.grad(score(*args), args[1])[0],
            torch.tensor(0.3 - mean, dtype=torch.float64),
        )
    if guarded:
        inactive = (
            (torch.tensor(-3),)
            if name == "observation_status"
            else (torch.tensor(1.0), torch.tensor(1.0))
        )
        assert torch.isfinite(score(*inactive))
    invalid = (
        (torch.tensor(3),)
        if name == "observation_status"
        else (torch.tensor(2.0), torch.tensor(1.0))
    )
    with pytest.raises(ValueError):
        score(*invalid)


@pytest.mark.parametrize("name", ["observation_status", "interval_upper"])
def test_stan_event_validator_helpers_typecheck(name, tmp_path):
    path = tmp_path / "validator.stan"
    path.write_bytes(_rendered(name, "stan"))
    completed = subprocess.run(
        [require_tool("stanc"), "--info", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("name", ["observation_status", "interval_upper"])
@pytest.mark.parametrize("guarded", [False, True])
def test_webppl_event_validator_model_compiles_and_runs(name, guarded, tmp_path):
    expected = 2.0 if name == "observation_status" else 1.2
    source = _rendered(name, "webppl", guarded).decode()
    signature = re.search(r"var model\s*=\s*function\s*\(([^)]*)\)", source)
    assert signature is not None
    bindings = {"out": "0", "status": "2", "left": "0.1", "right": "1.2", "y": "0.3"}
    args = ",".join(
        bindings[parameter.strip()] for parameter in signature[1].split(",")
    )
    path = tmp_path / "validator.wppl"
    path.write_text(
        source
        + f'\nvar out=Infer({{method:"MCMC",samples:2,burn:0}},function(){{return model({args});}});\ndisplay(JSON.stringify(out.support()));\n'
    )
    result = run_webppl(path)
    assert json.loads(result.stdout) == pytest.approx([expected])


def test_julia_scalar_validation_helpers_execute_without_optional_model_packages(
    tmp_path,
):
    text = (RUNTIME / "runtime_qiec.jl").read_text()
    helpers = text[text.index("function _qvr_observation_status") :]
    source = (
        helpers
        + """\nfor value in (-1,0,1,2); @assert _qvr_observation_status(value)==value; end
for value in (true,1.0,3,-2,NaN,Inf)
    refused=false
    try; _qvr_observation_status(value); catch; refused=true; end
    @assert refused
end
@assert _qvr_interval_upper(-Inf,1.)==1.
for pair in ((0.,0.),(2.,1.),(NaN,1.),(0.,Inf),(true,1.))
    refused=false
    try; _qvr_interval_upper(pair...); catch; refused=true; end
    @assert refused
end
println("ok")
"""
    )
    path = tmp_path / "validators.jl"
    path.write_text(source)
    result = subprocess.run(
        [require_tool("julia"), "--startup-file=no", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_scheme_scalar_validators_reject_noninteger_codes_and_bad_intervals(tmp_path):
    text = (RUNTIME / "runtime_qiec.scm").read_text()
    helpers = text[text.index("(define (_qvr-observation-status-scalar") :]
    script = (
        "(import (chezscheme))\n"
        + helpers
        + """
(for-each (lambda (x) (assert (= x (_qvr-observation-status-scalar x)))) '(-1 0 1 2))
(for-each (lambda (x) (assert (guard (exception (else #t)) (_qvr-observation-status-scalar x) #f))) '(#t 1.0 -2 3 +nan.0 +inf.0))
(assert (= 1.0 (_qvr-interval-upper-scalar -inf.0 1.0)))
(for-each (lambda (p) (assert (guard (exception (else #t)) (apply _qvr-interval-upper-scalar p) #f))) '((0.0 0.0) (2.0 1.0) (+nan.0 1.0) (0.0 +inf.0) (#t 1.0)))
(display "ok")
"""
    )
    path = tmp_path / "validators.scm"
    path.write_text(script)
    result = subprocess.run(
        [require_scheme(), "--script", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("name", ["observation_status", "interval_upper"])
@pytest.mark.parametrize("guarded", [False, True])
def test_emitted_pymc_event_validators_score_independent_normal_density(name, guarded):
    import pymc
    import pytensor
    import numpy as np

    namespace = {"pymc": pymc, "np": np}
    exec(_rendered(name, "pymc", guarded), namespace)
    bindings = (
        {"status": 2} if name == "observation_status" else {"left": 0.1, "right": 1.2}
    )
    model = namespace["build_model"](out=0.0, y=0.3, **bindings)
    score = pytensor.function([], model.logp(), mode="FAST_COMPILE")()
    mean = 2.0 if name == "observation_status" else 1.2
    assert float(score) == pytest.approx(
        -0.5 * (0.3 - mean) ** 2 - 0.5 * math.log(2 * math.pi), abs=1e-6
    )


@pytest.mark.parametrize("name", ["observation_status", "interval_upper"])
@pytest.mark.parametrize("guarded", [False, True])
def test_emitted_church_event_validators_score_independent_normal_density(
    name, guarded, tmp_path
):
    from tests.transpile.probes._protocol import Point
    from tests.transpile.probes.church import ChurchProbe

    bindings = (
        {"status": 2} if name == "observation_status" else {"left": 0.1, "right": 1.2}
    )
    result = ChurchProbe().evaluate(
        _rendered(name, "church", guarded),
        "model",
        [Point(params={}, data={"out": (0.0,), "y": 0.3, **bindings})],
        scratch=tmp_path,
    )
    mean = 2.0 if name == "observation_status" else 1.2
    assert result.log_densities == pytest.approx(
        [-0.5 * (0.3 - mean) ** 2 - 0.5 * math.log(2 * math.pi)], abs=1e-10
    )
