"""The shape of a free host name is inferred from every use of it.

A name no step binds is a program input whose checked type every path
reads it at: the compiled runtime refuses data that contradicts it, the
reference machine validates against it, and every target declares it so.
A use combining the name elementwise with a value of known shape fixes
the name to that shape, in a ``score`` as in a ``let``; a reduction of a
name nothing else fixes is ambiguous and refused where it is written.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.transpile import transpile

_TEMPLATE = """
object Item : FinSet 4
object Val : Real 1

program weighted : Item -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Item <- Normal(loc=mu, scale=1.0)
BODY
    return mu

export weighted
"""

_WEIGHTED = _TEMPLATE.replace(
    "BODY", "    let resid = y - mu\n    score weighting = sum(weight * resid)"
)

Y = [0.5, -1.0, 2.0, 0.3]
WEIGHT = [1.0, 0.0, 1.0, 0.0]
MU = 0.2


def _base_joint() -> float:
    """The joint of the prior and the likelihood at the test point.

    Returns
    -------
    float
        ``log Normal(mu; 0, 1) + sum_i log Normal(y_i; mu, 1)``.
    """
    normal = torch.distributions.Normal
    return float(
        normal(0.0, 1.0).log_prob(torch.tensor(MU))
        + normal(MU, 1.0).log_prob(torch.tensor(Y)).sum()
    )


def _rendered_score(target: str, weight: list[float]) -> float:
    """Evaluate the rendered body's score statements on the test point.

    Parameters
    ----------
    target : str
        ``"numpyro"`` or ``"pyro"``.
    weight : list[float]
        The weight datum.

    Returns
    -------
    float
        The value the rendered body binds to ``weighting``.
    """
    tree = ast.parse(transpile(parse(_WEIGHTED), target=target).decode())
    statements = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"resid", "weighting"}
    ]
    if target == "numpyro":
        namespace: dict[str, object] = {
            "jnp": np,
            "y": np.asarray(Y),
            "mu": MU,
            "weight": np.asarray(weight),
        }
    else:
        namespace = {
            "torch": torch,
            "y": torch.tensor(Y),
            "mu": torch.tensor(MU),
            "weight": torch.tensor(weight),
        }
    for statement in statements:
        exec(statement, namespace)
    return float(namespace["weighting"])  # type: ignore[arg-type]


def test_a_name_a_score_combines_with_a_plated_value_takes_its_plate() -> None:
    program = loads(_WEIGHTED)
    (entry,) = program.entry_points()
    types = {parameter.name: parameter.type for parameter in entry.parameters}
    assert types["weight"] == "Tensor[Real]([4])"
    expected = _base_joint() + sum(w * (v - MU) for w, v in zip(WEIGHT, Y))
    compiled = program.morphism.log_joint(
        torch.zeros(1, 1),
        {"mu": torch.tensor(MU), "y": torch.tensor(Y), "weight": torch.tensor(WEIGHT)},
    )
    assert float(compiled.sum()) == pytest.approx(expected, rel=1e-6)
    reference = program.run(
        "weighted",
        data={"y": tuple(Y), "weight": tuple(WEIGHT)},
        sites={"mu": MU},
    )
    assert reference.log_joint == pytest.approx(expected, rel=1e-6)
    score = sum(w * (v - MU) for w, v in zip(WEIGHT, Y))
    for target in ("numpyro", "pyro"):
        assert _rendered_score(target, WEIGHT) == pytest.approx(score, rel=1e-6)


def test_a_scalar_for_a_per_row_name_is_refused() -> None:
    program = loads(_WEIGHTED)
    with pytest.raises(ValueError, match=r"host datum 'weight' has shape \(\)"):
        program.morphism.log_joint(
            torch.zeros(1, 1),
            {"mu": torch.tensor(MU), "y": torch.tensor(Y), "weight": torch.tensor(1.0)},
        )


def test_a_name_only_scalars_meet_stays_a_scalar() -> None:
    program = loads(_TEMPLATE.replace("BODY", "    score s = scale * mu"))
    (entry,) = program.entry_points()
    types = {parameter.name: parameter.type for parameter in entry.parameters}
    assert types["scale"] == "Real"
    joint = program.morphism.log_joint(
        torch.zeros(1, 1),
        {"mu": torch.tensor(MU), "y": torch.tensor(Y), "scale": torch.tensor(2.0)},
    )
    assert float(joint.sum()) == pytest.approx(_base_joint() + 2.0 * MU, rel=1e-6)
    reference = program.run(
        "weighted", data={"y": tuple(Y), "scale": 2.0}, sites={"mu": MU}
    )
    assert reference.log_joint == pytest.approx(_base_joint() + 2.0 * MU, rel=1e-6)
    with pytest.raises(ValueError, match=r"host datum 'scale' has shape \(4,\)"):
        program.morphism.log_joint(
            torch.zeros(1, 1),
            {"mu": torch.tensor(MU), "y": torch.tensor(Y), "scale": torch.ones(4)},
        )


@pytest.mark.parametrize(
    "body",
    (
        "    score s = sum(weight)",
        "    score s = sum(weight * other)",
        "    let w = softmax(weight)\n    score s = w * mu",
    ),
)
def test_a_reduced_name_nothing_shapes_is_refused_where_it_is_read(body: str) -> None:
    with pytest.raises(QiecDiagnosticError) as captured:
        loads(_TEMPLATE.replace("BODY", body))
    message = captured.value.message
    assert "host datum 'weight' is reduced here, but nothing fixes its shape" in message
    assert "factor i : <Object> in weight[i]" in message
    assert str(captured.value).startswith("<source>:8:")


def test_reading_a_name_entrywise_in_a_factor_states_its_extent() -> None:
    program = loads(
        _TEMPLATE.replace(
            "BODY", "    score s = sum(factor i : Item in weight[i] * (y[i] - mu))"
        )
    )
    (entry,) = program.entry_points()
    types = {parameter.name: parameter.type for parameter in entry.parameters}
    assert types["weight"] == "Tensor[Real]([weight_extent])"
    joint = program.morphism.log_joint(
        torch.zeros(1, 1),
        {"mu": torch.tensor(MU), "y": torch.tensor(Y), "weight": torch.tensor(WEIGHT)},
    )
    expected = _base_joint() + sum(w * (v - MU) for w, v in zip(WEIGHT, Y))
    assert float(joint.sum()) == pytest.approx(expected, rel=1e-6)


def test_uses_disagreeing_on_a_name_extent_are_refused() -> None:
    # `z` reaches the `Item`-plated observe through `m`, and so would `x`,
    # while `x * v` fixes `x` to the two entries of `Pair`.
    source = """
object Item : FinSet 4
object Pair : FinSet 2
object Val : Real 1

program clash : Item -> Val
    sample v : Pair <- Normal(loc=0.0, scale=1.0)
    let m = sum(x * v) + z
    observe y : Item <- Normal(loc=m, scale=1.0)
    return y

export clash
"""
    with pytest.raises(QiecDiagnosticError, match="disagree on its extent"):
        loads(source)
