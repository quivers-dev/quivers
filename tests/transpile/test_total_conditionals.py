"""Graph targets eagerly select only branches with a closed domain proof."""

from pathlib import Path
import subprocess

import pytest

from quivers.dsl import parse
from quivers.qiec.primitives import primitive
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile._tools import require_tool


FIXTURE = Path(__file__).parent / "fixtures/steps/call_lowered_pure.qvr"


@pytest.mark.parametrize(
    "name", ["add_real", "mul_real", "neg_int", "min_real", "gt_real", "select_real"]
)
def test_elementary_totality_contracts_are_explicit(name):
    assert primitive(name).total


@pytest.mark.parametrize(
    "name",
    [
        "div_real",
        "div_int",
        "mod_int",
        "log",
        "sqrt",
        "pow_real",
        "observation_weight",
        "positive_parameter",
        "log_probability",
        "interval_upper",
    ],
)
def test_primitives_with_domain_restrictions_remain_unproved(name):
    assert not primitive(name).total


@pytest.mark.parametrize("a,b", [(0.7, -1.3), (-0.6, 0.2), (1.4, 0.9), (-2.0, 0.1)])
def test_actual_jags_total_nested_conditionals_match_independent_piecewise_oracle(
    a, b, tmp_path
):
    model = tmp_path / "total.jags"
    model.write_bytes(transpile(parse(FIXTURE.read_text()), target="jags"))
    data = tmp_path / "data.R"
    # Conditioning every stochastic node makes monitored deterministic values
    # exact evaluations of the emitted graph rather than posterior averages.
    data.write_text(f"a <- {a}\nb <- {b}\ny <- c(0.1,0.2)\n")
    stem = tmp_path / "draws"
    command = tmp_path / "run.cmd"
    command.write_text(
        f'model in "{model}"\ndata in "{data}"\ncompile, nchains(1)\ninitialize\nmonitor mean\nupdate 1\ncoda *, stem("{stem}")\nexit\n'
    )
    result = subprocess.run(
        [require_tool("jags"), str(command)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "ERROR" not in result.stderr.upper(), result.stderr
    actual = float((tmp_path / "drawschain1.txt").read_text().split()[1])
    expected = (a if a > 0 else 0.1 * a) + 0.5 + min(max(b, -0.5), 0.5)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "expression", ["log(x)", "1.0 / x", "sqrt(x)", "log_probability(x)"]
)
@pytest.mark.parametrize("target", ["bugs", "jags"])
def test_graph_targets_refuse_partial_branches_instead_of_evaluating_them_eagerly(
    expression, target
):
    source = f"""object Out : Real 1
define guarded(x:Real):Real !{{}} =
    if x > 0.0 then
        return {expression}
    else
        return 0.0
program p : Out -> Out
    sample x <- Normal(loc=0.0,scale=1.0)
    let mu <- guarded(x)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""
    with pytest.raises(UnsupportedConstruct) as error:
        transpile(parse(source), target=target)
    assert error.value.kinds == ["call:graph:guarded"]
