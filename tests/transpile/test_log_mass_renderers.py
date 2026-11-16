"""Mass queries use distribution bridges and explicit target capabilities."""

import ast
from pathlib import Path
import subprocess

import pytest
import torch

from quivers.dsl import parse
from quivers.transpile import transpile, UnsupportedConstruct
from quivers.transpile.qiec_ir import capabilities_for_target
from tests.transpile._tools import require_scheme, require_tool
from tests.transpile._webppl import run_webppl

ROOT = Path(__file__).parents[2] / "src/quivers/transpile"
DYNAMIC = ("pyro", "numpyro", "pymc", "edward2", "turing", "gen", "webppl", "church")
STATIC = ("stan", "bugs", "jags")
SOURCE = "define mass() : LogWeight !{} =\n    return log_mass(Normal(0.0, 1.0))\n"


@pytest.mark.parametrize("target", DYNAMIC)
def test_dynamic_target_renders_structural_mass_query(target):
    rendered = transpile(parse(SOURCE), target=target)
    assert b"log_mass" in rendered or b"log-mass" in rendered
    assert capabilities_for_target(target).supports("log-mass")
    if target in ("pyro", "numpyro", "pymc", "edward2"):
        ast.parse(rendered)


@pytest.mark.parametrize("target", STATIC)
def test_static_targets_refuse_mass_query_by_name(target):
    with pytest.raises(UnsupportedConstruct, match="log-mass"):
        transpile(parse(SOURCE), target=target)
    assert not capabilities_for_target(target).supports("log-mass")


@pytest.mark.parametrize("target", (*DYNAMIC, *STATIC))
def test_unmapped_raw_measure_is_an_explicit_refusal(target):
    raw = SOURCE.replace("Normal(0.0, 1.0)", "Restrict(Normal(0.0, 1.0), -1.0, 2.0)")
    with pytest.raises(UnsupportedConstruct):
        transpile(parse(raw), target=target)


def test_pyro_genuine_host_unit_mass_and_raw_measure_ad_delegation():
    import pyro

    namespace = {"pyro": pyro, "torch": torch}
    exec(transpile(parse(SOURCE), target="pyro"), namespace)
    assert float(namespace["qiec_mass"]()) == 0.0
    query = namespace["_qvr_qiec_log_mass"]
    loc = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    normal = torch.distributions.Normal(loc, 1.0)
    mass = query(normal, 1)
    torch.testing.assert_close(mass, torch.zeros_like(loc))
    torch.testing.assert_close(
        torch.autograd.grad(mass.sum(), loc)[0], torch.zeros_like(loc)
    )
    from quivers.continuous.measure import Restrict

    restricted = Restrict(normal, low=torch.tensor(-0.5), high=torch.tensor(1.0))
    actual = query(restricted, 1)
    expected = torch.log(torch.special.ndtr(1 - loc) - torch.special.ndtr(-0.5 - loc))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), loc)[0],
        torch.autograd.grad(expected.sum(), loc)[0],
    )


def test_actual_webppl_unit_mass(tmp_path):
    script = tmp_path / "mass.wppl"
    script.write_bytes(
        transpile(parse(SOURCE), target="webppl") + b"\ndisplay(qiec_mass());\n"
    )
    result = run_webppl(script)
    assert float(result.stdout.strip()) == 0.0


def test_actual_scheme_unit_mass(tmp_path):
    script = tmp_path / "mass.scm"
    script.write_bytes(
        transpile(parse(SOURCE), target="church")
        + b"\n(display (qiec_mass))\n(newline)\n"
    )
    result = subprocess.run(
        [require_scheme(), "--script", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert float(result.stdout.strip()) == 0.0


@pytest.mark.parametrize("target", ("turing", "gen"))
def test_actual_julia_bridge_retains_replication_prefix(target, tmp_path):
    # The bridge itself has no package requirement until distribution methods
    # are called; this query delegates a raw normalizer without a scoring point.
    script = tmp_path / "mass.jl"
    script.write_text(
        (ROOT / f"runtime_qiec_{target}.jl").read_text()
        + '\nstruct RawMass\n log_normalizer\nend\n@assert _qvr_qiec_log_mass(RawMass(() -> log(0.5)), 1, (2, 3)) == fill(3log(0.5), 2)\nprintln("ok")\n'
    )
    result = subprocess.run(
        [require_tool("julia"), "--startup-file=no", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
