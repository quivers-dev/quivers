"""A latent morphism's `~ Family(...)` prior is the law a draw from it has.

`morphism W : A -> B [role=latent] ~ Normal(loc=0.0, scale=2.0)` declares
the prior of a draw `sample v <- W`. The reference machine and every
transpile target score such a draw under the declared family; the compiled
runtime has to as well, rather than reading the morphism as its learned
table (which has no density, so the draw could not be scored at all). The
compiled log joint is pinned to the closed-form Normal density, to the
reference machine, and to the Pyro rendering executed in process.
"""

from __future__ import annotations

import pyro
import pytest
import torch
import torch.distributions as D

from quivers.dsl import CompileError, loads
from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile

_SOURCE = """\
object A : FinSet 3
object B : FinSet 2
object Row : FinSet 4
morphism W : A -> B [role=latent] ~ Normal(loc=0.5, scale=2.0)
program p : Row -> Row
    sample v : Row <- W
    return v
export p
"""

#: Draws exact in single precision, so both paths read the same numbers.
_V = [0.125, -1.0, 2.0, 0.5]


def _closed_form() -> float:
    """The four draws' log density under ``Normal(0.5, 2)``, in float64.

    Returns
    -------
    float
        The log density.
    """
    values = torch.tensor(_V, dtype=torch.float64)
    law = D.Normal(
        torch.tensor(0.5, dtype=torch.float64), torch.tensor(2.0, dtype=torch.float64)
    )
    return float(law.log_prob(values).sum())


def test_both_execution_paths_score_the_declared_prior() -> None:
    program = loads(_SOURCE)
    compiled = float(
        program.morphism.log_joint(torch.zeros(1, 1), {"v": torch.tensor([_V])})[0]
    )
    reference = program.run("p", data={}, sites={"v": tuple(_V)}).log_joint
    assert compiled == pytest.approx(_closed_form(), rel=1e-6)
    assert reference == pytest.approx(_closed_form(), rel=1e-12)


def test_the_pyro_rendering_scores_the_declared_prior() -> None:
    text = transpile(parse(_SOURCE), target="pyro").decode()
    namespace: dict[str, object] = {"pyro": pyro, "torch": torch}
    exec(text, namespace)  # noqa: S102 - emitted source under test
    model = namespace["model"]
    assert callable(model)
    conditioned = pyro.poutine.condition(
        model, data={"v": torch.tensor(_V, dtype=torch.float64)}
    )
    with pyro.poutine.trace() as tracer:
        conditioned()
    assert float(tracer.trace.log_prob_sum()) == pytest.approx(_closed_form(), rel=1e-6)


def test_the_oracle_rejects_the_unit_normal() -> None:
    values = torch.tensor(_V, dtype=torch.float64)
    unit = float(D.Normal(0.0, 1.0).log_prob(values).sum())
    assert abs(unit - _closed_form()) > 1e-3 * abs(_closed_form())


def test_a_draw_that_passes_arguments_to_the_prior_is_refused() -> None:
    source = _SOURCE.replace("sample v : Row <- W", "sample v : Row <- W(x)")
    with pytest.raises(CompileError, match="draw from it without arguments"):
        loads(source)


@pytest.mark.parametrize("init", ["~ Normal(loc=0.0, scale=1.0)", "~ Normal"])
def test_an_embedding_with_an_initializer_is_refused(init: str) -> None:
    source = (
        "object Word : FinSet 3\n"
        "object Vec : Real 2\n"
        f"morphism e : Word -> Vec [role=embed] {init}\n"
    )
    with pytest.raises(CompileError, match="takes no ``~`` initializer"):
        loads(source)
    # Without one the embedding compiles.
    loads(
        "object Word : FinSet 3\nobject Vec : Real 2\nmorphism e : Word -> Vec [role=embed]\n"
    )


def test_every_target_refuses_an_embedding_with_an_initializer() -> None:
    source = (
        "object Word : FinSet 3\n"
        "object Vec : Real 2\n"
        "object Row : FinSet 2\n"
        "morphism e : Word -> Vec [role=embed] ~ Normal(loc=0.0, scale=1.0)\n"
        "program p : Row -> Row\n"
        "    sample v : Row <- e(w)\n"
        "    return v\n"
        "export p\n"
    )
    for target in ("stan", "pyro", "numpyro"):
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(parse(source), target=target)
        kinds = {kind.partition(": ")[0] for kind in caught.value.kinds}
        assert "morphism:embed-initializer:e" in kinds, caught.value.kinds
