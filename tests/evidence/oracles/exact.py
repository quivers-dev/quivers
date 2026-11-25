"""Recurrences in exact rational arithmetic.

These compute what the enumeration oracles compute by a different
method, so the two witness each other.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from fractions import Fraction

import torch

type Matrix = list[list[Fraction]]
"""A dense rational matrix by rows."""


class Singular(ValueError):
    """The linear system has no unique solution."""


def rational(value: str | int | Fraction) -> Fraction:
    """A rational from a string such as ``"3/7"``.

    Parameters
    ----------
    value : str or int or Fraction
        The value.

    Returns
    -------
    Fraction
        The rational.
    """
    return Fraction(value)


def solve(matrix: Matrix, right: Matrix) -> Matrix:
    """Solve ``matrix X = right`` by Gauss-Jordan elimination.

    Parameters
    ----------
    matrix : Matrix
        A square matrix.
    right : Matrix
        The right-hand sides.

    Returns
    -------
    Matrix
        The solution.

    Raises
    ------
    Singular
        If the matrix is singular.
    """
    size = len(matrix)
    work = [[*row, *extra] for row, extra in zip(matrix, right, strict=True)]
    for column in range(size):
        pivot = next((r for r in range(column, size) if work[r][column] != 0), None)
        if pivot is None:
            raise Singular("the matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        head = work[column][column]
        work[column] = [value / head for value in work[column]]
        for row in range(size):
            if row != column and work[row][column] != 0:
                factor = work[row][column]
                work[row] = [
                    value - factor * base
                    for value, base in zip(work[row], work[column], strict=True)
                ]
    return [row[size:] for row in work]


def identity(size: int) -> Matrix:
    """The identity matrix.

    Parameters
    ----------
    size : int
        The order.

    Returns
    -------
    Matrix
        The matrix.
    """
    return [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]


def linear_star(matrix: Matrix) -> Matrix:
    """The inverse of ``I - matrix``.

    For nonnegative entries and spectral radius below one it is the sum
    of all powers of the matrix.

    Parameters
    ----------
    matrix : Matrix
        A square matrix.

    Returns
    -------
    Matrix
        The inverse.
    """
    size = len(matrix)
    base = [
        [Fraction(int(i == j)) - matrix[i][j] for j in range(size)] for i in range(size)
    ]
    return solve(base, identity(size))


def determinant(matrix: Matrix) -> Fraction:
    """The determinant by the sum over permutations.

    Parameters
    ----------
    matrix : Matrix
        A square matrix.

    Returns
    -------
    Fraction
        The determinant; one for the empty matrix.
    """
    size = len(matrix)
    answer = Fraction(0)
    for order in itertools.permutations(range(size)):
        inversions = sum(
            order[i] > order[j] for i in range(size) for j in range(i + 1, size)
        )
        product = Fraction(-1 if inversions % 2 else 1)
        for row, column in enumerate(order):
            product *= matrix[row][column]
        answer += product
    return answer


def tree_partition(
    nodes: int, root: int, weights: Mapping[tuple[int, int], Fraction]
) -> Fraction:
    """The directed matrix-tree determinant.

    Parameters
    ----------
    nodes : int
        The number of nodes.
    root : int
        The root.
    weights : Mapping[tuple of (int, int), Fraction]
        The weight of each edge from parent to child.

    Returns
    -------
    Fraction
        The determinant of the Laplacian without the root.
    """
    rest = [node for node in range(nodes) if node != root]
    laplacian: Matrix = []
    for child in rest:
        row = []
        for other in rest:
            if child == other:
                row.append(
                    sum(
                        (
                            weights.get((parent, child), Fraction(0))
                            for parent in range(nodes)
                            if parent != child
                        ),
                        Fraction(0),
                    )
                )
            else:
                row.append(-weights.get((other, child), Fraction(0)))
        laplacian.append(row)
    return determinant(laplacian)


def forward(
    initial: Sequence[Fraction],
    transition: Sequence[Sequence[Fraction]],
    emission: Sequence[Sequence[Fraction]],
    symbols: Sequence[int],
) -> Fraction:
    """The likelihood of symbols when each is emitted after one transition.

    Parameters
    ----------
    initial : Sequence of Fraction
        The distribution of the state before the first symbol.
    transition : Sequence of Sequence of Fraction
        The transition rows.
    emission : Sequence of Sequence of Fraction
        The emission rows.
    symbols : Sequence of int
        The observed symbols.

    Returns
    -------
    Fraction
        The likelihood.
    """
    message = list(initial)
    for symbol in symbols:
        message = [
            sum(
                (message[i] * transition[i][j] for i in range(len(message))),
                Fraction(0),
            )
            * emission[j][symbol]
            for j in range(len(message))
        ]
    return sum(message, Fraction(0))


def inside(
    binary: Mapping[tuple[int, int, int], Fraction],
    lexical: Mapping[tuple[int, int], Fraction],
    unary: Mapping[tuple[int, int], Fraction],
    tokens: Sequence[int],
    categories: int,
) -> list[Fraction]:
    """Inside weights of a whole string by the closed cubic recurrence.

    Parameters
    ----------
    binary : Mapping[tuple of (int, int, int), Fraction]
        Weight of ``a -> b c`` at ``(a, b, c)``.
    lexical : Mapping[tuple of (int, int), Fraction]
        Weight of ``a -> t`` at ``(a, t)``.
    unary : Mapping[tuple of (int, int), Fraction]
        Weight of ``a -> b`` at ``(a, b)``.
    tokens : Sequence of int
        The string.
    categories : int
        The number of categories.

    Returns
    -------
    list of Fraction
        The weight of each category over the whole string.
    """
    step = [
        [unary.get((b, a), Fraction(0)) for b in range(categories)]
        for a in range(categories)
    ]
    star = linear_star(step)

    def close(cell: Sequence[Fraction]) -> list[Fraction]:
        """Close a cell under the unary rules.

        Parameters
        ----------
        cell : Sequence of Fraction
            The weight of each category before unary rules.

        Returns
        -------
        list of Fraction
            The weight after every chain.
        """
        return [
            sum((cell[b] * star[b][a] for b in range(categories)), Fraction(0))
            for a in range(categories)
        ]

    size = len(tokens)
    cells: dict[tuple[int, int], list[Fraction]] = {}
    for i, token in enumerate(tokens):
        cells[(i, i + 1)] = close(
            [lexical.get((a, token), Fraction(0)) for a in range(categories)]
        )
    for width in range(2, size + 1):
        for i in range(size - width + 1):
            j = i + width
            cell = [Fraction(0)] * categories
            for (a, b, c), weight in binary.items():
                for k in range(i + 1, j):
                    cell[a] += weight * cells[(i, k)][b] * cells[(k, j)][c]
            cells[(i, j)] = close(cell)
    return cells[(0, size)]


def contraction[N: (Fraction, float, torch.Tensor)](
    inputs: Sequence[str],
    output: str,
    sizes: Mapping[str, int],
    tables: Sequence[Mapping[tuple[int, ...], N]],
    zero: N,
    one: N,
) -> dict[tuple[int, ...], N]:
    """A sum of products over every assignment of indices.

    Parameters
    ----------
    inputs : Sequence of str
        The index letters of each table.
    output : str
        The index letters of the result.
    sizes : Mapping[str, int]
        The extent of each letter.
    tables : Sequence of Mapping[tuple of int, N]
        The entries of each table.
    zero : N
        The additive unit.
    one : N
        The multiplicative unit.

    Returns
    -------
    dict[tuple of int, N]
        The entries of the result.
    """
    letters = sorted(set("".join(inputs)))
    answer = {
        key: zero for key in itertools.product(*(range(sizes[x]) for x in output))
    }
    for values in itertools.product(*(range(sizes[x]) for x in letters)):
        at = dict(zip(letters, values, strict=True))
        product = one
        for letters_in, table in zip(inputs, tables, strict=True):
            product = product * table[tuple(at[x] for x in letters_in)]
        key = tuple(at[x] for x in output)
        answer[key] = answer[key] + product
    return answer


def simple_path_star(matrix: Sequence[Sequence[Fraction]]) -> Matrix:
    """The largest product of a path between two states, with no repeated state.

    For weights whose cycles have product at most one, it is the closure
    in the algebra of maxima and products.

    Parameters
    ----------
    matrix : Sequence of Sequence of Fraction
        Nonnegative weights.

    Returns
    -------
    Matrix
        The best path weights; one on the diagonal.
    """
    size = len(matrix)
    answer = identity(size)
    for length in range(2, size + 1):
        for states in itertools.permutations(range(size), length):
            product = Fraction(1)
            for left, right in itertools.pairwise(states):
                product *= matrix[left][right]
            if product > answer[states[0]][states[-1]]:
                answer[states[0]][states[-1]] = product
    return answer
