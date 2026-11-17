"""The checked ``cholesky`` builtin: eager and reference values, derivatives, refusals.

Tolerances are fixed from floating-point error before comparison: the
matrices here have condition numbers below 1e2, so double-precision factors,
their first and second derivatives, and the reference machine's recursion
agree to 1e-10.
"""

import pytest
import torch

from quivers.dsl import loads, parse
from quivers.dsl.pure_builtins import BUILTIN_REGISTRY, EAGER_BUILTINS
from quivers.qiec.canonical import tensor_type
from quivers.qiec.kinds import NAT
from quivers.qiec.types import IndexLiteral
from quivers.qiec.serialization import from_data, to_data
from quivers.qiec.terms import CholeskyFactor, Local, Var
from quivers.qiec.types import REAL
from quivers.transpile import UnsupportedConstruct, available_targets, transpile

_SOURCE = """
object Resp : FinSet 4

object U : FinSet 3

program model(left : Tensor[Real]([U, U]), idx : Tensor[Int]([Resp])) : Resp -> Resp
    sample scale <- HalfNormal(scale=1.0)
    sample z : U <- Normal(loc=0.0, scale=1.0)
    let covariance = scale * scale * left
    let lower = cholesky(covariance)
    let latent = matmul(lower, z)
    let per_row = latent[idx]
    observe y : Resp <- Normal(loc=per_row, scale=0.5)
    return y

export model
"""

_MATRIX = torch.tensor(
    [[2.0, 0.6, 0.2], [0.6, 1.5, 0.3], [0.2, 0.3, 1.2]], dtype=torch.float64
)
_IDX = torch.tensor([0, 2, 1, 2])
_Y = torch.tensor([0.3, -0.4, 0.8, 0.1], dtype=torch.float64)


def _hand_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Differentiable Cholesky-Banachiewicz recursion written out entry by entry."""
    size = matrix.shape[-1]
    rows: list[list[torch.Tensor]] = [[] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            partial = matrix[i, j] - sum(
                (rows[i][k] * rows[j][k] for k in range(j)),
                start=matrix.new_zeros(()),
            )
            rows[i].append(partial.sqrt() if i == j else partial / rows[j][j])
    zero = matrix.new_zeros(())
    return torch.stack([torch.stack(row + [zero] * (size - len(row))) for row in rows])


def test_eager_factor_equals_an_entrywise_recursion_with_its_derivatives():
    cholesky = EAGER_BUILTINS["cholesky"]
    point = _MATRIX.clone().requires_grad_()

    # Derivatives are taken along symmetric matrices, where the factor is
    # defined; torch spreads a gradient over both triangles.
    def actual(m):
        symmetric = (m + m.T) / 2
        return (cholesky(symmetric) * torch.arange(1.0, 10.0).reshape(3, 3)).sum()

    def oracle(m):
        symmetric = (m + m.T) / 2
        return (_hand_cholesky(symmetric) * torch.arange(1.0, 10.0).reshape(3, 3)).sum()

    torch.testing.assert_close(
        cholesky(_MATRIX), _hand_cholesky(_MATRIX), rtol=0, atol=1e-10
    )
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(oracle, point),
        rtol=0,
        atol=1e-10,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(oracle, point),
        rtol=0,
        atol=1e-10,
    )


def test_eager_factor_batches_and_marks_indefinite_matrices_undefined():
    cholesky = EAGER_BUILTINS["cholesky"]
    batch = torch.stack((_MATRIX, -_MATRIX))
    result = cholesky(batch)
    torch.testing.assert_close(result[0], torch.linalg.cholesky(_MATRIX))
    assert torch.isnan(result[1]).all()
    spec = BUILTIN_REGISTRY["cholesky"]
    assert spec.qiec_form == "matrix" and spec.qiec_arities == frozenset({1})
    assert spec.targets == frozenset({"eager", "qiec"})


def test_compiled_density_equals_the_reference_machine_and_a_closed_form():
    program = loads(_SOURCE)
    scale = torch.tensor(1.3, dtype=torch.float64)

    def compiled(z: torch.Tensor) -> torch.Tensor:
        values = {"left": _MATRIX, "idx": _IDX, "y": _Y, "scale": scale, "z": z}
        return program.morphism.log_joint(
            torch.zeros(1, dtype=torch.long), values
        ).sum()

    def closed(z: torch.Tensor) -> torch.Tensor:
        latent = torch.linalg.cholesky(scale**2 * _MATRIX) @ z
        return (
            torch.distributions.HalfNormal(1.0).log_prob(scale)
            + torch.distributions.Normal(0.0, 1.0).log_prob(z).sum()
            + torch.distributions.Normal(latent[_IDX], 0.5).log_prob(_Y).sum()
        )

    z = torch.tensor([0.4, -1.1, 0.7], dtype=torch.float64)
    other = torch.tensor([-0.3, 0.2, 1.5], dtype=torch.float64)
    expected = closed(z)
    # The compiled runtime scores its built-in families with constants that
    # agree with torch's to 1e-8; that offset is the same at every z, so the
    # difference isolates the factor to double precision.
    torch.testing.assert_close(compiled(z), expected, rtol=0, atol=1e-8)
    torch.testing.assert_close(
        compiled(z) - compiled(other), expected - closed(other), rtol=0, atol=1e-10
    )
    run = program.run(
        "model",
        data={
            "left": tuple(tuple(row) for row in _MATRIX.tolist()),
            "idx": tuple(_IDX.tolist()),
            "y": tuple(_Y.tolist()),
        },
        sites={"scale": float(scale), "z": tuple(z.tolist())},
    )
    assert run.log_joint == pytest.approx(float(expected), abs=1e-10)


@pytest.mark.parametrize(
    ("let", "message"),
    [
        ("cholesky(z)", "last two axes"),
        ("cholesky(left, left)", "last two axes"),
    ],
)
def test_checked_lowering_refuses_non_matrices(let, message):
    source = _SOURCE.replace("cholesky(covariance)", let)
    with pytest.raises(Exception, match=message):
        loads(source)


def test_the_checked_term_serializes_and_reads_back_as_the_builtin():
    three = IndexLiteral(value=3, sort=NAT)
    matrix = tensor_type(REAL, (three, three))
    term = CholeskyFactor(
        value=Var(local=Local(name="m", type=matrix)), result_type=matrix
    )
    assert from_data(to_data(term)) == term


def test_every_transpile_target_refuses_the_factorization():
    module = parse(_SOURCE)
    for target in available_targets():
        with pytest.raises(UnsupportedConstruct) as refusal:
            transpile(module, target=target)
        assert any("cholesky" in kind for kind in refusal.value.kinds), target
