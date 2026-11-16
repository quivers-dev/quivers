"""Compiled counting observations use the same inclusive-event probability law."""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from quivers.dsl import Compiler, parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.inference.trace import trace
from quivers.qiec.program_runtime import run_program


_BASES = {
    "Bernoulli": ("probs=0.25", stats.bernoulli(0.25)),
    "Categorical": (
        "probs=[0.125,0.125,0.25,0.25,0.25]",
        stats.rv_discrete(values=(np.arange(5), [0.125, 0.125, 0.25, 0.25, 0.25])),
    ),
    "Binomial": ("total_count=8,probs=0.25", stats.binom(8, 0.25)),
    "NegativeBinomial": ("total_count=2.0,probs=0.25", stats.nbinom(2, 0.75)),
    "Poisson": ("rate=2.0", stats.poisson(2)),
    "Geometric": ("probs=0.25", stats.geom(0.25, loc=-1)),
    "BetaBinomial": (
        "total_count=8,concentration1=0.5,concentration0=2.0",
        stats.betabinom(8, 0.5, 2),
    ),
}
_CASES = list(_BASES)
_CASES += [
    "ZeroInflated" + name
    for name in ["Poisson", "NegativeBinomial", "Binomial", "BetaBinomial"]
]
_CASES += ["HurdlePoisson", "HurdleNegativeBinomial"]


def _case(name):
    if name.startswith("ZeroInflated"):
        args, base = _BASES[name.removeprefix("ZeroInflated")]
        return (
            "zero_prob=0.25," + args,
            lambda k: 0.25 * (np.asarray(k) == 0) + 0.75 * base.pmf(k),
        )
    if name.startswith("Hurdle"):
        args, base = _BASES[name.removeprefix("Hurdle")]
        return "zero_prob=0.25," + args, lambda k: np.where(
            np.asarray(k) == 0, 0.25, 0.75 * base.pmf(k) / base.sf(0)
        )
    args, base = _BASES[name]
    return args, base.pmf


@pytest.mark.parametrize("name", _CASES)
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checked_count_events_match_finite_probability_oracle(name, truncated, dtype):
    args, pmf = _case(name)
    # Left-censored at zero, exact zero, right-censored above zero, and
    # interval-censored (0, 2] exercise atom inclusion independently.
    options = "cens=c,cens_upper=u"
    if truncated:
        options += ",trunc_lower=0.0,trunc_upper=4.0"
    carrier = "Bool" if name == "Bernoulli" else "Int"
    source = f"""object Row : FinSet 4
program p(y:Tensor[{carrier}]([Row]),c:Tensor[Int]([Row]),u:Tensor[Real]([Row])) : Row -> Row
    observe y:Row <- {name}({args}) [{options}]
    return y
export p
"""
    module = parse(source)
    checked = lower_qvr_to_qiec(module)
    data = {
        "y": (False, False, False, False) if name == "Bernoulli" else (0, 0, 0, 0),
        "c": (-1, 0, 1, 2),
        "u": (0.0, 0.0, 0.0, 2.0),
    }
    probabilities = np.asarray(pmf(np.arange(5)))
    normalizer = probabilities.sum() if truncated else 1.0
    masses = [
        probabilities[0],
        probabilities[0],
        normalizer - probabilities[0],
        probabilities[1:3].sum(),
    ]
    oracle = sum(math.log(mass / normalizer) for mass in masses)
    reference = run_program(checked, "p", data=data, sites={}).log_joint
    assert reference == pytest.approx(oracle, abs=2e-11)
    tensors = {
        key: torch.tensor(
            value,
            dtype=torch.float64
            if key == "u"
            else torch.bool
            if key == "y" and name == "Bernoulli"
            else torch.int64,
        )
        for key, value in data.items()
    }
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        native = trace(Compiler(module).compile().morphism, torch.zeros(1, 4), tensors)
    finally:
        torch.set_default_dtype(old_dtype)
    # Fixed source factories create parameters using the runtime default dtype.
    assert float(native.log_joint.sum()) == pytest.approx(
        oracle, abs=1e-6 if dtype == torch.float32 else 2e-11
    )
