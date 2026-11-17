"""Recorded source observations have the same native and canonical scores."""

import math

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec, QiecDiagnosticError
from quivers.inference.trace import trace
from quivers.qiec.program_runtime import run_program, program_entry
from quivers.transpile import transpile, UnsupportedConstruct


NORMAL = """object Row : FinSet 4
program p(y:Tensor[Real]([Row]),c:Tensor[Int]([Row]),u:Tensor[Real]([Row])) : Row -> Row
    sample a <- Normal(loc=0.0,scale=2.0)
    observe y:Row <- Normal(loc=a,scale=1.0) [cens=c,cens_upper=u]
    return a
export p
"""


@pytest.mark.parametrize("truncated", [False, True])
def test_normal_events_and_truncation_agree_between_native_reference_and_math(
    truncated,
):
    source = (
        NORMAL.replace(
            "cens_upper=u]", "cens_upper=u,trunc_lower=-1.0,trunc_upper=2.0]"
        )
        if truncated
        else NORMAL
    )
    module = parse(source)
    checked = lower_qvr_to_qiec(module)
    data = {"y": (-0.2, 0.3, 0.8, 0.1), "c": (-1, 0, 1, 2), "u": (0.0, 0.0, 0.0, 1.1)}
    actual = run_program(checked, "p", data=data, sites={"a": 0.2}).log_joint
    normal = torch.distributions.Normal(torch.tensor(0.2, dtype=torch.float64), 1.0)

    def cdf(x):
        return torch.special.ndtr(torch.tensor(x, dtype=torch.float64) - 0.2)

    low, high = (-1.0, 2.0) if truncated else (-math.inf, math.inf)
    logmass = (cdf(high) - cdf(low)).log()
    oracle = (
        (cdf(-0.2) - cdf(low)).log()
        + normal.log_prob(torch.tensor(0.3, dtype=torch.float64))
        + (cdf(high) - cdf(0.8)).log()
        + (cdf(1.1) - cdf(0.1)).log()
        - 4 * logmass
        + torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
        ).log_prob(torch.tensor(0.2, dtype=torch.float64))
    )
    assert actual == pytest.approx(float(oracle), abs=2e-12)
    inputs = {
        name: torch.tensor(value, dtype=torch.long if name == "c" else torch.float64)
        for name, value in data.items()
    } | {"a": torch.tensor(0.2, dtype=torch.float64)}
    native = trace(Compiler(module).compile().morphism, torch.zeros(1, 4), inputs)
    # The pre-existing fixed prior stores its scale in the native default
    # dtype. Event scoring itself retains the Float64 response parameters.
    prior = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(2.0, dtype=torch.float64)
    ).log_prob(torch.tensor(0.2, dtype=torch.float64))
    assert float(
        native.sites["y"].sampleable.raw_log_prob_pointwise(inputs["y"]).sum()
    ) == pytest.approx(actual - float(prior), abs=2e-12)
    assert float(native.log_joint.sum()) == pytest.approx(actual, abs=2e-7)


@pytest.mark.parametrize("code", [3, -2])
def test_reference_invalid_runtime_status_does_not_fall_through_to_interval(code):
    checked = lower_qvr_to_qiec(parse(NORMAL))
    with pytest.raises((ValueError, RuntimeError), match="observation_status"):
        run_program(
            checked,
            "p",
            data={
                "y": (0.1, 0.2, 0.3, 0.4),
                "c": (0, code, 0, 0),
                "u": (0.0, 0.0, 0.0, 0.0),
            },
            sites={"a": 0.2},
        )


def test_optional_event_roles_are_inferred_in_the_same_input_contract():
    source = """object Row : FinSet 2
program p : Row -> Row
    observe y:Row <- Poisson(rate=1.2) [cens=c,cens_upper=u]
    return y
export p
"""
    entry = program_entry(lower_qvr_to_qiec(parse(source)), "p")
    assert {parameter.name for parameter in entry.parameters} == {"y", "c", "u"}
    from quivers.qiec.types import render_static

    assert {p.name: render_static(p.type) for p in entry.parameters} == {
        "y": "Tensor[Int]([2])",
        "c": "Tensor[Int]([2])",
        "u": "Tensor[Real]([2])",
    }


@pytest.mark.parametrize(
    "option", ["cens=1.5", "cens=true", "cens_upper=u", "cens=c,cens=c"]
)
def test_invalid_event_option_contracts_fail_at_compile_time(option):
    with pytest.raises(QiecDiagnosticError):
        lower_qvr_to_qiec(parse(NORMAL.replace("cens=c,cens_upper=u", option)))


@pytest.mark.parametrize(
    "target",
    [
        "pyro",
        "numpyro",
        "pymc",
        "edward2",
        "turing",
        "gen",
        "webppl",
        "church",
        "stan",
        "bugs",
        "jags",
    ],
)
def test_renderers_refuse_recorded_site_metadata_until_the_target_has_interval_observe_support(
    target,
):
    with pytest.raises(UnsupportedConstruct, match="recorded-event-observe"):
        transpile(parse(NORMAL), target=target)
