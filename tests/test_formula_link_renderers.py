"""Formula links survive source emission and actual downstream tool checks."""

from __future__ import annotations

from functools import lru_cache
import json
import math
import re
import subprocess

import numpy as np
import pandas as pd
import panproto
import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.dsl import loads
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.formulas import families, formula_to_qvr
from quivers.inference import trace
from quivers.transpile import available_targets, transpile
from tests.transpile._tools import require_tool
from tests.transpile._webppl import run_webppl
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.church import ChurchProbe


LINKS = ("squareplus", "softit")
X = [-1.0, 0.0, 1.0]
Y = [0.1, 0.4, 0.8]
POINTS = ((0.2, 0.7), (-8.0, 2.0), (8.0, -2.0), (0.0, 0.0))
TARGET_GRAMMARS = {
    "bugs": "bugs",
    "church": "scheme",
    "edward2": "python",
    "gen": "julia",
    "jags": "jags",
    "numpyro": "python",
    "pymc": "python",
    "pyro": "python",
    "stan": "stan",
    "turing": "julia",
    "webppl": "javascript",
}


@lru_cache
def _source(link: str) -> str:
    return formula_to_qvr(
        "y ~ x",
        data=pd.DataFrame({"y": Y, "x": X}),
        family=families["gaussian"].with_links(mu=link),
        fixed_parameters={"sigma": 0.7},
        priors={"intercept": "Normal(0.0, 1.2)", "beta_x": "Normal(0.0, 0.8)"},
    )


@lru_cache
def _rendered(link: str, target: str) -> bytes:
    return transpile(parse(_source(link)), target=target)


def _mean(link: str, eta: float) -> float:
    if link == "squareplus":
        radius = math.hypot(eta, 2.0)
        return (eta + radius) / 2 if eta >= 0 else 2 / (radius - eta)
    positive = max(eta, 0.0) + math.log1p(math.exp(-abs(eta)))
    return positive / (1 + positive)


def _normal_logpdf(value: float, mean: float, scale: float) -> float:
    return (
        -0.5 * ((value - mean) / scale) ** 2
        - math.log(scale)
        - math.log(2 * math.pi) / 2
    )


def _joint(link: str, intercept: float, slope: float) -> float:
    return (
        _normal_logpdf(intercept, 0.0, 1.2)
        + _normal_logpdf(slope, 0.0, 0.8)
        + math.fsum(
            _normal_logpdf(y, _mean(link, intercept + slope * x), 0.7)
            for x, y in zip(X, Y, strict=True)
        )
    )


@pytest.mark.parametrize("link", LINKS)
def test_formula_link_roundtrip_preserves_native_and_reference_density(link):
    source = _source(link)
    canonical = module_to_source(parse(source))
    assert module_to_source(parse(canonical)) == canonical
    compiled = loads(canonical)
    program = compiled.morphism
    assert isinstance(program, MonadicProgram)
    for intercept, slope in POINTS:
        observations = {
            "x": torch.tensor(X, dtype=torch.float64),
            "y": torch.tensor(Y, dtype=torch.float64),
            "intercept": torch.tensor(intercept, dtype=torch.float64),
            "beta_x": torch.tensor(slope, dtype=torch.float64),
        }
        recorded = trace(program, torch.zeros(1, 1), observations)
        assert recorded.log_joint is not None
        assert recorded.log_joint.item() == pytest.approx(
            _joint(link, intercept, slope), rel=2e-7
        )
        reference = compiled.run(
            "model",
            data={"x": tuple(X), "y": tuple(Y)},
            sites={"intercept": intercept, "beta_x": slope},
        )
        assert reference.log_joint == pytest.approx(
            _joint(link, intercept, slope), rel=1e-10
        )


@pytest.mark.parametrize("target", sorted(available_targets()))
@pytest.mark.parametrize("link", LINKS)
def test_formula_link_renders_and_reparses_on_every_external_target(link, target):
    output = _rendered(link, target)
    assert output
    assert b"std::" not in output
    assert b"squareplus(" not in output and b"softit(" not in output
    schema = panproto.AstParserRegistry().parse_with_protocol(
        TARGET_GRAMMARS[target], output, f"{link}.{target}"
    )
    assert schema.vertex_count > 0
    if TARGET_GRAMMARS[target] == "python":
        compile(output, f"{link}.{target}.py", "exec")


@pytest.mark.parametrize("link", LINKS)
def test_formula_link_stan_passes_real_typechecking(link, tmp_path):
    path = tmp_path / f"{link}.stan"
    path.write_bytes(_rendered(link, "stan"))
    result = subprocess.run(
        [require_tool("stanc"), "--info", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("link", LINKS)
def test_formula_link_pyro_executes_density_and_predictor(link):
    import pyro

    namespace: dict[str, object] = {"pyro": pyro, "torch": torch}
    exec(_rendered(link, "pyro"), namespace)
    model = namespace["model"]
    assert callable(model)
    for intercept, slope in POINTS:
        conditioned = pyro.condition(
            model,
            data={
                "intercept": torch.tensor(intercept, dtype=torch.float64),
                "beta_x": torch.tensor(slope, dtype=torch.float64),
            },
        )
        recorded = pyro.poutine.trace(conditioned).get_trace(
            x=torch.tensor(X, dtype=torch.float64),
            y=torch.tensor(Y, dtype=torch.float64),
        )
        expected = torch.tensor(
            [_mean(link, intercept + slope * x) for x in X], dtype=torch.float64
        )
        distribution = recorded.nodes["y"].get("fn")
        assert isinstance(distribution, torch.distributions.Normal)
        torch.testing.assert_close(distribution.loc, expected)
        assert float(recorded.log_prob_sum()) == pytest.approx(
            _joint(link, intercept, slope), rel=2e-7
        )


@pytest.mark.parametrize("link", LINKS)
def test_formula_link_pymc_builds_and_executes_log_density(link):
    import pymc

    namespace: dict[str, object] = {"pymc": pymc, "np": np}
    exec(_rendered(link, "pymc"), namespace)
    builder = namespace["build_model"]
    assert callable(builder)
    model = builder(x=np.asarray(X), y=np.asarray(Y))
    assert isinstance(model, pymc.Model)
    density = model.compile_logp()
    for intercept, slope in POINTS:
        assert float(
            density({"intercept": intercept, "beta_x": slope})
        ) == pytest.approx(_joint(link, intercept, slope), rel=1e-10)


@pytest.mark.parametrize("link", LINKS)
def test_formula_link_church_executes_joint_density(link, tmp_path):
    result = ChurchProbe().evaluate(
        _rendered(link, "church"),
        link,
        [
            Point(
                params={"intercept": intercept, "beta_x": slope}, data={"x": X, "y": Y}
            )
            for intercept, slope in POINTS
        ],
        scratch=tmp_path,
    )
    assert result.log_densities == pytest.approx(
        [_joint(link, intercept, slope) for intercept, slope in POINTS], rel=1e-10
    )


@pytest.mark.parametrize("link", LINKS)
def test_formula_link_webppl_compiles_and_executes_model(link, tmp_path):
    source = _rendered(link, "webppl").decode()
    signature = re.search(r"var model\s*=\s*function\s*\(([^)]*)\)", source)
    assert signature is not None
    data = {"x": X, "y": Y}
    arguments = ",".join(
        json.dumps(data[name.strip()]) for name in signature.group(1).split(",")
    )
    script = tmp_path / f"{link}.wppl"
    script.write_text(
        source
        + f'\nvar fitted = Infer({{method:"MCMC",samples:2,burn:0}},function(){{return model({arguments});}});\n'
        + "display(JSON.stringify(fitted.support()));\n"
    )
    completed = run_webppl(script)
    assert json.loads(completed.stdout) == [Y]


@pytest.mark.parametrize("target", ["turing", "gen"])
@pytest.mark.parametrize("link", LINKS)
def test_formula_link_julia_source_passes_language_parser(link, target):
    result = subprocess.run(
        [
            require_tool("julia"),
            "--startup-file=no",
            "--quiet",
            "-e",
            "let src=read(stdin,String), pos=1; while pos<=ncodeunits(src); "
            "expr,pos=Meta.parse(src,pos;raise=true); expr===nothing && break; end; end",
        ],
        input=_rendered(link, target),
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("target", ["jags", "bugs"])
@pytest.mark.parametrize("link", LINKS)
def test_formula_link_jags_compiles_and_evaluates_deterministic_means(
    link, target, tmp_path
):
    model = tmp_path / f"{link}.{target}"
    model.write_bytes(_rendered(link, target))
    data = tmp_path / "data.R"
    data.write_text(
        "x <- c(-1,0,1)\ny <- c(0.1,0.4,0.8)\nintercept <- 0.2\nbeta_x <- 0.7\n"
    )
    commands = tmp_path / "run.cmd"
    stem = tmp_path / "samples"
    commands.write_text(
        f'model in "{model}"\ndata in "{data}"\ncompile, nchains(1)\ninitialize\n'
        f'monitor mu\nupdate 1\ncoda *, stem("{stem}")\nexit\n'
    )
    result = subprocess.run(
        [require_tool("jags"), str(commands)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "error" not in (result.stdout + result.stderr).lower(), (
        result.stdout + result.stderr
    )
    recorded = np.loadtxt(str(stem) + "chain1.txt", ndmin=2)
    assert recorded[:, 1].tolist() == pytest.approx(
        [_mean(link, 0.2 + 0.7 * x) for x in X], rel=1e-5
    )
