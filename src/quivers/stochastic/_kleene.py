"""The Kleene star of a unary-rule matrix in a chart semiring.

Unary rules derive a category over a span from another category over the
same span, so a chain of them never consumes input and may cycle. The
weight they add to a chart cell is the cell times the reflexive,
transitive closure of the unary-rule matrix ``U``,

    U* = I ⊕ U ⊕ U² ⊕ U³ ⊕ ...,

the sum over every chain of unary rules of the product of its weights
(Stolcke 1995, definition 6, where for probabilities ``U* = (I - U)⁻¹``;
Goodman 1999 for the semiring generalization). The closure is applied
once to the cell the binary and lexical rules produce; applying ``U``
repeatedly to an accumulating cell instead counts a chain once per
ordering of its prefixes and overstates every weight.

The series need not converge. For probabilities it converges exactly
when the spectral radius of ``U`` is below one; in the max-plus Viterbi
semiring when no cycle of unary rules has positive log weight; in the
Boolean semiring always. A closure that diverges has no value, and the
functions here raise rather than return a truncated sum.

References
----------
Stolcke, A. (1995). An efficient probabilistic context-free parsing
algorithm that computes prefix probabilities. *Computational
Linguistics* 21(2), 165-201, section 4.5.2.
<https://aclanthology.org/J95-2002/>

Goodman, J. (1999). Semiring parsing. *Computational Linguistics*
25(4), 573-605. <https://aclanthology.org/J99-4004/>
"""

from __future__ import annotations

import torch

from quivers.stochastic.semiring import (
    BooleanSemiring,
    ChartSemiring,
    CountingSemiring,
    LogProbSemiring,
    ViterbiSemiring,
)


class DivergentClosureError(ValueError):
    """A path closure has no finite value in the declared semiring."""


def _check_square(matrix: torch.Tensor) -> int:
    """Return the order of a square matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix.

    Returns
    -------
    int
        Its number of rows.

    Raises
    ------
    ValueError
        If the tensor is not a square matrix.
    """
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"a unary-rule matrix is square; got shape {tuple(matrix.shape)}"
        )
    return int(matrix.shape[0])


def _reachable(support: torch.Tensor) -> torch.Tensor:
    """The reflexive, transitive closure of a Boolean relation.

    Parameters
    ----------
    support : torch.Tensor
        A ``(C, C)`` Boolean matrix; ``support[b, a]`` says a rule
        derives ``a`` from ``b``.

    Returns
    -------
    torch.Tensor
        ``(C, C)`` Boolean: whether some chain (possibly empty) of
        rules derives ``a`` from ``b``.
    """
    n = support.shape[0]
    reach = torch.eye(n, dtype=torch.bool, device=support.device) | support
    for _ in range(max(n - 1, 0).bit_length() + 1):
        reach = reach | ((reach.float() @ reach.float()) > 0)
    return reach


def _linear_star(matrix: torch.Tensor) -> torch.Tensor:
    """``(I - U)⁻¹`` for a nonnegative matrix of spectral radius below one.

    Parameters
    ----------
    matrix : torch.Tensor
        ``(C, C)`` nonnegative weights.

    Returns
    -------
    torch.Tensor
        The closure ``Σ_k Uᵏ``, with exact zeros where no chain exists.

    Raises
    ------
    DivergentClosureError
        If the spectral radius of ``U`` is at least one, so the series
        diverges.
    """
    n = _check_square(matrix)
    if n == 0:
        return matrix
    radius = float(torch.linalg.eigvals(matrix.detach().double()).abs().max())
    if not radius < 1.0:
        raise DivergentClosureError(
            "the unary rules' closure I + U + U² + ... diverges: the "
            f"spectral radius of the unary-rule weight matrix is {radius:.6g}, "
            "and the series converges only below 1; lower the weights of the "
            "unary cycles"
        )
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    star = torch.linalg.solve(eye - matrix, eye)
    reach = _reachable(matrix.detach() > 0)
    return torch.where(reach, star.clamp(min=0.0), torch.zeros_like(star))


def _log_star(matrix: torch.Tensor) -> torch.Tensor:
    """Sum all paths by state elimination in log space.

    Parameters
    ----------
    matrix : torch.Tensor
        Square log-weight matrix, with negative infinity for absent edges.

    Returns
    -------
    torch.Tensor
        Log weights of all paths, including empty paths.

    Raises
    ------
    DivergentClosureError
        If a state's effective return weight is at least one.
    ValueError
        If the matrix contains NaN or positive infinity.
    """
    n = _check_square(matrix)
    if bool((torch.isnan(matrix) | torch.isposinf(matrix)).any()):
        raise ValueError("log closure requires finite weights or negative infinity")
    star = matrix
    indices = torch.arange(n, device=matrix.device)
    for k in range(n):
        loop = star[k, k]
        if not bool(loop < 0):
            raise DivergentClosureError(
                "the unary rules' closure diverges: an effective return weight "
                "is at least one"
            )
        repeat = -torch.log(-torch.expm1(loop))
        through = star[:, k, None] + repeat + star[None, k, :]
        empty = torch.isneginf(star) & torch.isneginf(through)
        combined = torch.logaddexp(
            torch.where(empty, torch.zeros_like(star), star),
            torch.where(empty, torch.zeros_like(through), through),
        )
        combined = torch.where(empty, -torch.inf, combined)
        row = indices[:, None] == k
        column = indices[None, :] == k
        star = torch.where(row | column, star + repeat, combined)
        star = torch.where(row & column, repeat, star)
    return star


def _max_plus_star(matrix: torch.Tensor) -> torch.Tensor:
    """The closure in the max-plus semiring: best chain log weights.

    Parameters
    ----------
    matrix : torch.Tensor
        ``(C, C)`` log weights, ``-inf`` where there is no rule.

    Returns
    -------
    torch.Tensor
        ``star[b, a]``: the largest total log weight of a chain deriving
        ``a`` from ``b`` (``0`` on the diagonal, the empty chain).

    Raises
    ------
    DivergentClosureError
        If some cycle of unary rules has positive log weight, so chains
        of unbounded weight exist.
    """
    n = _check_square(matrix)
    star = torch.full_like(matrix, float("-inf"))
    star = star.fill_diagonal_(0.0) if n else star
    # Chains of at most C - 1 rules visit every acyclic path; one more
    # round changes the closure only if a cycle has positive weight.
    for step in range(n + 1):
        extended = (star.unsqueeze(2) + matrix.unsqueeze(0)).amax(dim=1)
        updated = torch.maximum(star, extended)
        if torch.equal(updated, star):
            return star
        if step == n:
            break
        star = updated
    raise DivergentClosureError(
        "the unary rules' closure diverges in the Viterbi semiring: a cycle "
        "of unary rules has positive log weight, so a chain around it can "
        "be made arbitrarily heavy"
    )


def _boolean_star(matrix: torch.Tensor) -> torch.Tensor:
    """The closure in the Boolean semiring: reachability.

    Parameters
    ----------
    matrix : torch.Tensor
        ``(C, C)`` truth values in ``{0, 1}``.

    Returns
    -------
    torch.Tensor
        ``(C, C)`` truth values: whether a chain derives ``a`` from ``b``.
    """
    _check_square(matrix)
    return _reachable(matrix > 0).to(matrix.dtype)


def kleene_star(matrix: torch.Tensor, semiring: ChartSemiring) -> torch.Tensor:
    """The reflexive, transitive closure ``U*`` of a unary-rule matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        ``(C, C)`` rule weights in the semiring's representation, indexed
        ``[derived-from, derived]``: ``matrix[b, a]`` is the weight of the
        rule deriving ``a`` from ``b`` over one span; the semiring's zero
        where there is no rule.
    semiring : ChartSemiring
        The semiring the weights live in.

    Returns
    -------
    torch.Tensor
        ``(C, C)`` closure in the same representation and indexing.

    Raises
    ------
    DivergentClosureError
        If the closure's series diverges in the semiring.
    TypeError
        If the semiring is not one whose closure this module computes.
    """
    if isinstance(semiring, LogProbSemiring):
        return _log_star(matrix)
    if isinstance(semiring, CountingSemiring):
        return _linear_star(matrix)
    if isinstance(semiring, ViterbiSemiring):
        return _max_plus_star(matrix)
    if isinstance(semiring, BooleanSemiring):
        return _boolean_star(matrix)
    raise TypeError(
        f"no unary-rule closure is defined for {semiring!r}; the closure is "
        "computed for LogProb, Counting, Viterbi, and Boolean"
    )


def apply_star(
    cell: torch.Tensor, star: torch.Tensor, semiring: ChartSemiring
) -> torch.Tensor:
    """Close a chart cell under the unary rules: ``cell ⊗ U*``.

    Parameters
    ----------
    cell : torch.Tensor
        ``(batch, C)`` weights of each category over one span.
    star : torch.Tensor
        ``(C, C)`` closure from `kleene_star`.
    semiring : ChartSemiring
        The semiring of both.

    Returns
    -------
    torch.Tensor
        ``(batch, C)``: ``out[:, a] = ⊕_b cell[:, b] ⊗ star[b, a]``.
    """
    products = semiring.times(cell.unsqueeze(2), star.unsqueeze(0))
    return semiring_sum(products, dim=1, semiring=semiring)


def semiring_sum(
    scores: torch.Tensor, dim: int, semiring: ChartSemiring
) -> torch.Tensor:
    """The semiring sum over an axis, with a zero gradient on empty sums.

    Under ``LogProb`` a sum whose terms are all ``-inf`` is ``-inf``;
    ``torch.logsumexp`` returns that value but differentiates it to
    ``nan``. Such terms are bounded at the dtype's floor for the
    reduction and the empty sums restored to ``-inf``, so they carry no
    gradient. Other semirings reduce with their own ``plus``.

    Parameters
    ----------
    scores : torch.Tensor
        The terms.
    dim : int
        The axis to sum over.
    semiring : ChartSemiring
        The semiring.

    Returns
    -------
    torch.Tensor
        The sums, with ``dim`` removed.
    """
    if not isinstance(semiring, LogProbSemiring):
        return semiring.plus(scores, dim=dim)
    empty = torch.isneginf(scores)
    bounded = torch.where(
        empty, torch.full_like(scores, torch.finfo(scores.dtype).min), scores
    )
    reduced = torch.logsumexp(bounded, dim=dim)
    return torch.where(
        empty.all(dim=dim), torch.full_like(reduced, float("-inf")), reduced
    )
