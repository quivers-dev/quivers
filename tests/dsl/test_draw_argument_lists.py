"""List literal arguments to inline families on the compiled runtime.

A list literal fills a vector or matrix family parameter, alone or beside
arguments read from the program's bindings. The compiled runtime reads
every draw argument kind through one classification, so each family that
takes a vector or matrix parameter scores a list literal as a
`torch.distributions` oracle does, and as the reference machine does. A
kind no value parameter takes is refused by name.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
import torch
import torch.distributions as D

from quivers import Program
from quivers.continuous._ordered import OrderedLogistic
from quivers.dsl import loads
from quivers.dsl.compiler import CompileError

_HEADER = (
    "object K : FinSet 3\nobject Two : FinSet 2\nobject One : FinSet 1\n"
    "object Resp : FinSet 4\n\n"
)

_L = [[1.0, 0.0], [0.5, 1.0]]

type Value = float | int | list[float] | list[int] | list[list[float]]
"""A site or datum value as the case writes it."""

type Oracle = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
"""The independent joint at the case's values."""

#: Each case: the program body, its data, its site values, and its oracle.
CASES: dict[str, tuple[str, dict[str, Value], dict[str, Value], Oracle]] = {
    "Categorical probs": (
        "program p : One -> One\n"
        "    observe c <- Categorical(probs=[0.2, 0.3, 0.5])\n    return c",
        {"c": 2},
        {},
        lambda v: D.Categorical(probs=torch.tensor([0.2, 0.3, 0.5])).log_prob(v["c"]),
    ),
    "Categorical logits": (
        "program p : One -> One\n"
        "    observe c <- Categorical(logits=[0.2, -0.3, 0.5])\n    return c",
        {"c": 1},
        {},
        lambda v: D.Categorical(logits=torch.tensor([0.2, -0.3, 0.5])).log_prob(v["c"]),
    ),
    "Dirichlet": (
        "program p : One -> One\n"
        "    sample v <- Dirichlet(concentration=[1.0, 2.0, 3.0])\n    return v",
        {},
        {"v": [0.2, 0.3, 0.5]},
        lambda v: D.Dirichlet(torch.tensor([1.0, 2.0, 3.0])).log_prob(v["v"]),
    ),
    "MultivariateNormal": (
        "program p : One -> One\n"
        "    sample v <- MultivariateNormal(loc=[0.0, 1.0], scale_tril="
        "[[1.0, 0.0], [0.5, 1.0]]) [over=Two]\n    return v",
        {},
        {"v": [0.3, 0.4]},
        lambda v: D.MultivariateNormal(
            torch.tensor([0.0, 1.0]), scale_tril=torch.tensor(_L)
        ).log_prob(v["v"]),
    ),
    "MultivariateNormal beside a sampled mean": (
        "program p : One -> One\n"
        "    sample m <- Normal(loc=0.0, scale=1.0) [over=Two]\n"
        "    sample v <- MultivariateNormal(loc=m, scale_tril="
        "[[1.0, 0.0], [0.5, 1.0]]) [over=Two]\n    return v",
        {},
        {"m": [0.1, -0.2], "v": [0.3, 0.4]},
        lambda v: (
            D.Normal(0.0, 1.0).log_prob(v["m"]).sum()
            + D.MultivariateNormal(v["m"], scale_tril=torch.tensor(_L)).log_prob(v["v"])
        ),
    ),
    "LowRankMVN": (
        "program p : One -> One\n"
        "    sample v <- LowRankMVN(loc=[0.0, 1.0], cov_factor=[[1.0], [0.5]], "
        "cov_diag=[1.0, 2.0]) [over=Two]\n    return v",
        {},
        {"v": [0.3, 0.4]},
        lambda v: D.LowRankMultivariateNormal(
            torch.tensor([0.0, 1.0]),
            torch.tensor([[1.0], [0.5]]),
            torch.tensor([1.0, 2.0]),
        ).log_prob(v["v"]),
    ),
    "MixtureNormal": (
        "program p : One -> One\n"
        "    sample v <- MixtureNormal(weights=[0.3, 0.7], loc=[-1.0, 1.0], "
        "scale=[1.0, 0.5])\n    return v",
        {},
        {"v": 0.4},
        lambda v: D.MixtureSameFamily(
            D.Categorical(probs=torch.tensor([0.3, 0.7])),
            D.Normal(torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 0.5])),
        ).log_prob(v["v"]),
    ),
    "MixtureNormal beside sampled locations": (
        "program p : One -> One\n"
        "    sample m <- Normal(loc=0.0, scale=1.0) [over=Two]\n"
        "    sample v <- MixtureNormal(weights=[0.3, 0.7], loc=m, "
        "scale=[1.0, 0.5])\n    return v",
        {},
        {"m": [-1.0, 1.0], "v": 0.4},
        lambda v: (
            D.Normal(0.0, 1.0).log_prob(v["m"]).sum()
            + D.MixtureSameFamily(
                D.Categorical(probs=torch.tensor([0.3, 0.7])),
                D.Normal(v["m"], torch.tensor([1.0, 0.5])),
            ).log_prob(v["v"])
        ),
    ),
    "Mixture": (
        "program p : One -> One\n"
        "    sample v <- Mixture(weights=[0.3, 0.7], components="
        "[Normal(loc=-1.0, scale=1.0), Normal(loc=1.0, scale=0.5)])\n    return v",
        {},
        {"v": 0.4},
        lambda v: D.MixtureSameFamily(
            D.Categorical(probs=torch.tensor([0.3, 0.7])),
            D.Normal(torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 0.5])),
        ).log_prob(v["v"]),
    ),
    "OrderedLogistic": (
        "program p : Resp -> Resp\n"
        "    observe y : Resp <- OrderedLogistic(predictor=0.3, "
        "cutpoints=[-1.5, -0.5, 0.5, 1.5])\n    return y",
        {"y": [0, 1, 4, 2]},
        {},
        lambda v: (
            OrderedLogistic(torch.full((4,), 0.3), torch.tensor([-1.5, -0.5, 0.5, 1.5]))
            .log_prob(v["y"])
            .sum()
        ),
    ),
    "OrderedLogistic beside a sampled predictor": (
        "program p : Resp -> Resp\n"
        "    sample eta <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : Resp <- OrderedLogistic(predictor=eta, "
        "cutpoints=[-1.5, -0.5, 0.5, 1.5])\n    return y",
        {"y": [0, 1, 4, 2]},
        {"eta": 0.3},
        lambda v: (
            D.Normal(0.0, 1.0).log_prob(v["eta"])
            + OrderedLogistic(v["eta"].expand(4), torch.tensor([-1.5, -0.5, 0.5, 1.5]))
            .log_prob(v["y"])
            .sum()
        ),
    ),
}

#: Cases the compiled runtime scores whose reference-machine score is not
#: the oracle's; the compiled runtime is pinned to the oracle alone.
RUNTIME_ONLY: dict[str, tuple[str, dict[str, Value], Oracle]] = {
    "OneHotCategorical": (
        "program p : One -> One\n"
        "    sample v <- OneHotCategorical(probs=[0.2, 0.3, 0.5]) [over=K]\n"
        "    return v",
        {"v": [0.0, 1.0, 0.0]},
        lambda v: D.OneHotCategorical(probs=torch.tensor([0.2, 0.3, 0.5])).log_prob(
            v["v"]
        ),
    ),
    "Wishart": (
        "program p : One -> One\n"
        "    sample v : Two <- Wishart(df=4.0, scale_tril=[[1.0, 0.0], [0.5, 1.0]])\n"
        "    return v",
        {"v": [[2.0, 0.3], [0.3, 1.0]]},
        lambda v: D.Wishart(df=torch.tensor(4.0), scale_tril=torch.tensor(_L)).log_prob(
            v["v"]
        ),
    ),
}


def _program(body: str) -> Program:
    """Compile a case's program.

    Parameters
    ----------
    body : str
        The program declaration.

    Returns
    -------
    Program
        The compiled module.
    """
    return loads(_HEADER + body + "\nexport p\n")


def _runtime_values(
    data: Mapping[str, Value], sites: Mapping[str, Value]
) -> dict[str, torch.Tensor]:
    """Values for the compiled runtime, a site's event under the run axis.

    Parameters
    ----------
    data : Mapping[str, Value]
        The data.
    sites : Mapping[str, Value]
        The site values.

    Returns
    -------
    dict[str, torch.Tensor]
        Each value; a vector- or matrix-valued site leads with the run axis.
    """
    values = {name: torch.tensor(value) for name, value in data.items()}
    for name, value in sites.items():
        tensor = torch.tensor(value)
        values[name] = tensor.unsqueeze(0) if isinstance(value, list) else tensor
    return values


def _nested(value: Value) -> object:
    """A value as the reference machine's nested tuples.

    Parameters
    ----------
    value : Value
        The value.

    Returns
    -------
    object
        A number, or nested tuples of numbers.
    """
    if isinstance(value, list):
        return tuple(_nested(item) for item in value)
    return value


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_list_literal_scores_the_oracle_on_both_paths(case: str) -> None:
    body, data, sites, oracle = CASES[case]
    program = _program(body)
    expected = float(
        oracle({name: torch.tensor(value) for name, value in {**data, **sites}.items()})
    )
    joint = program.morphism.log_joint(torch.zeros(1, 1), _runtime_values(data, sites))
    assert float(joint.sum()) == pytest.approx(expected, rel=1e-5)
    run = program.run(
        "p",
        data={name: _nested(value) for name, value in data.items()},
        sites={name: _nested(value) for name, value in sites.items()},
    )
    assert run.log_joint == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("case", sorted(RUNTIME_ONLY))
def test_a_list_literal_scores_the_oracle_on_the_compiled_runtime(case: str) -> None:
    body, sites, oracle = RUNTIME_ONLY[case]
    program = _program(body)
    expected = float(
        oracle({name: torch.tensor(value) for name, value in sites.items()})
    )
    joint = program.morphism.log_joint(torch.zeros(1, 1), _runtime_values({}, sites))
    assert float(joint.sum()) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "program p : One -> One\n"
            "    sample v <- MultivariateNormal(loc=Normal(loc=0.0, scale=1.0), "
            "scale_tril=[[1.0, 0.0], [0.5, 1.0]]) [over=Two]\n    return v",
            r"a nested distribution \(DrawArgDist\) fills a parameter that takes "
            r"a value",
        ),
        (
            "program p : One -> One\n"
            "    sample a <- Normal(loc=0.0, scale=1.0)\n"
            "    sample v <- Dirichlet(concentration=[a, 2.0, 3.0])\n    return v",
            r"a list literal argument holds numbers or lists of numbers, not a "
            r"DrawArgName",
        ),
    ],
)
def test_a_draw_argument_no_value_parameter_takes_is_refused_by_name(
    body: str, message: str
) -> None:
    with pytest.raises(CompileError, match=message):
        _program(body)
