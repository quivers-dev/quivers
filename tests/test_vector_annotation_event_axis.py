"""A vector family's index annotation is its event axis unless its arguments fix another.

``sample w : K <- Dirichlet(concentration=2.0)`` is one point of the
``K``-simplex on the checked path and on the compiled runtime; a list
argument of another length makes the annotation a plate. A draw whose
event size nothing determines is refused at compile time with its
location, never sized by a default.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.compiler._prelude import CompileError
from quivers.inference import trace

HEAD = """
object A : FinSet 2
object K : FinSet 5
object Item : FinSet 4
object Val : Real 1
program p : A -> Val
    {step}
    return w
export p
"""


def _program(step: str):  # type: ignore[no-untyped-def]
    return loads(HEAD.format(step=step))


def _checked_shape(program) -> tuple[int, ...]:  # type: ignore[no-untyped-def]
    entry = next(e for e in program.qiec.entries if e.name == "p")
    site = next(s for s in entry.sites if s.name == "w")
    return tuple(int(axis.size.value) for axis in (*site.batch, *site.event))


@pytest.mark.parametrize(
    "concentration", ["2.0", "[2.0, 2.0, 2.0, 2.0, 2.0]"], ids=["scalar", "list"]
)
def test_an_annotated_dirichlet_is_one_point_of_the_annotated_simplex(
    concentration: str,
) -> None:
    program = _program(f"sample w : K <- Dirichlet(concentration={concentration})")
    assert _checked_shape(program) == (5,)
    w = trace(program.morphism, torch.zeros(1, 1)).sites["w"].value
    assert w.numel() == 5
    torch.testing.assert_close(w.sum(), torch.tensor(1.0))
    point = w.reshape(5)
    expected = D.Dirichlet(torch.full((5,), 2.0)).log_prob(point)
    joint = program.morphism.log_joint(torch.zeros(1, 1), {"w": point})
    torch.testing.assert_close(joint.sum(), expected)


def test_a_list_of_another_length_makes_the_annotation_a_plate() -> None:
    program = _program("sample w : Item <- Dirichlet(concentration=[1.0, 2.0, 3.0])")
    w = trace(program.morphism, torch.zeros(1, 1)).sites["w"].value
    assert tuple(w.shape) == (4, 3)
    torch.testing.assert_close(w.sum(-1), torch.ones(4))


def test_an_annotated_one_hot_categorical_is_one_vector_over_the_classes() -> None:
    program = _program(
        "sample w : K <- OneHotCategorical(probs=[0.1, 0.2, 0.3, 0.2, 0.2])"
    )
    w = trace(program.morphism, torch.zeros(1, 1)).sites["w"].value
    assert w.numel() == 5
    assert float(w.sum()) == 1.0


def test_explicit_one_coordinate_codomain_fixes_the_singleton_simplex() -> None:
    # The declared Real 1 codomain fixes extent 1 on both checked and native
    # paths; it is not an unsized vector or an invented minimum dimension.
    program = _program("sample w <- Dirichlet(concentration=2.0)")
    point = torch.ones(1)
    torch.testing.assert_close(
        program.morphism.log_joint(torch.zeros(1, 1), {"w": point}).sum(),
        D.Dirichlet(torch.tensor([2.0])).log_prob(point),
    )
    checked = program.run("p", data={}, sites={"w": (1.0,)})
    torch.testing.assert_close(
        checked.value, torch.ones(1, dtype=torch.float64), rtol=0, atol=0
    )
    assert checked.log_joint == 0.0


def test_a_vector_family_the_runtime_cannot_size_is_refused_at_its_draw() -> None:
    with pytest.raises(CompileError, match="one variate over 'K'") as raised:
        _program("sample w : K <- LogisticNormal(loc=0.0, scale=1.0)")
    assert "line 7" in str(raised.value)


@pytest.mark.parametrize(
    "step",
    [
        "sample w <- Dirichlet(concentration=[2.0])",
        "sample w : K <- Dirichlet(concentration=2.0)",
        "sample w <- Dirichlet(concentration=2.0) [over=K]",
    ],
)
def test_explicit_singleton_evidence_retains_the_unique_simplex_point(step):
    program = loads(HEAD.replace("FinSet 5", "FinSet 1").format(step=step))
    if "[2.0]" not in step:
        assert _checked_shape(program) == (1,)
    recorded = trace(program.morphism, torch.zeros(1, 1))
    torch.testing.assert_close(recorded.sites["w"].value.reshape(-1), torch.ones(1))
    assert recorded.log_joint.item() == pytest.approx(0.0, abs=1e-6)
