"""Contractions of tensor networks by variable elimination.

A contraction ``x_1, ..., x_k -> y`` names each axis of each input by a
letter. Its value at an index of the output letters is the sum, over
every index of the letters that are not in the output, of the product of
the inputs' entries at that index. Evaluated as written, the product of
all inputs is formed over every letter before anything is summed, so the
largest tensor has the product of all extents as its size.

In a commutative semiring the sum distributes over the product, and both
may be regrouped and reordered. A summed letter can thus be removed by
multiplying only the factors that contain it and summing it away, which
replaces them by one factor over the other letters of their bag
(variable elimination; Dechter 1999, and InsideOut of Abo Khamis, Ngo and
Rudra 2016 for semirings). The work is bounded by the elements of the
largest bag of the chosen order instead of the product of all extents.
Finding an order of least width is NP-hard (Arnborg, Corneil and
Proskurowski 1987), so `elimination_order` chooses one greedily, and the
plan records the order and its widths without claiming them least.

The laws are read from the algebra's claims, never from its name: an
algebra that establishes every law of `ELIMINATION_LAWS` is eliminated,
and any other is evaluated in the written order, with its products taken
left to right in the order of the inputs.

References
----------
Dechter, R. (1999). Bucket elimination: a unifying framework for
reasoning. *Artificial Intelligence* 113(1-2), 41-85.
[doi:10.1016/S0004-3702(99)00059-4](https://doi.org/10.1016/S0004-3702(99)00059-4)

Abo Khamis, M., Ngo, H. Q., and Rudra, A. (2016). FAQ: questions asked
frequently. *PODS*, 13-28.
[doi:10.1145/2902251.2902280](https://doi.org/10.1145/2902251.2902280)

Arnborg, S., Corneil, D. G., and Proskurowski, A. (1987). Complexity of
finding embeddings in a k-tree. *SIAM Journal on Algebraic and Discrete
Methods* 8(2), 277-284.
[doi:10.1137/0608024](https://doi.org/10.1137/0608024)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal, Protocol

import didactic.api as dx
import torch

from quivers.core.laws import (
    DISTRIBUTIVE,
    JOIN_ITERATED,
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    PLUS_UNIT,
    TIMES_ASSOCIATIVE,
    TIMES_COMMUTATIVE,
    TIMES_UNIT,
    ZERO_ANNIHILATES,
    Law,
)

__all__ = [
    "ELIMINATION_LAWS",
    "ContractionAlgebra",
    "ContractionCounts",
    "ContractionPlan",
    "ContractionStep",
    "elimination_order",
    "plan_contraction",
    "run_contraction",
]


ELIMINATION_LAWS: tuple[Law, ...] = (
    PLUS_ASSOCIATIVE,
    PLUS_COMMUTATIVE,
    PLUS_UNIT,
    TIMES_ASSOCIATIVE,
    TIMES_COMMUTATIVE,
    TIMES_UNIT,
    DISTRIBUTIVE,
    ZERO_ANNIHILATES,
    JOIN_ITERATED,
)
"""The laws elimination reorders by: a commutative semiring on one domain.

Sums are regrouped and reordered, products are regrouped and reordered,
a sum is moved inside the product of the factors that do not contain
its letter, an absent term is the zero and annihilates its product, a
sum over several axes is taken one axis at a time, and a product of no
factors is the unit."""


class ContractionAlgebra(Protocol):
    """The operations and law claims a contraction uses."""

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The product of two broadcastable tensors, elementwise.

        Parameters
        ----------
        a : torch.Tensor
            The left factor.
        b : torch.Tensor
            The right factor.

        Returns
        -------
        torch.Tensor
            ``a ⊗ b``.
        """
        ...

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """The sum over axes.

        Parameters
        ----------
        t : torch.Tensor
            The terms.
        dim : int or tuple of int
            The axes.

        Returns
        -------
        torch.Tensor
            The sum with the axes removed.
        """
        ...

    def establishes(self, *laws: Law) -> bool:
        """Whether every law is claimed and proved or decided.

        Parameters
        ----------
        *laws : Law
            The laws.

        Returns
        -------
        bool
            True when each is established.
        """
        ...


def elimination_order(
    variables: Sequence[str],
    edges: Sequence[Collection[str]],
    kept: Collection[str],
    cost: Callable[[frozenset[str]], float] = len,
) -> tuple[str, ...]:
    """A greedy order of the variables to sum out: cheapest bag first.

    At each step the variable whose bag (the variables of the edges that
    contain it, itself included) has the least cost is removed, ties in
    the order of ``variables``; its edges are merged into one without it.
    Finding an order of least width is NP-hard (Arnborg, Corneil and
    Proskurowski 1987), so this order is recorded, not claimed least.

    Parameters
    ----------
    variables : Sequence of str
        Every variable, in the order that breaks ties.
    edges : Sequence of Collection of str
        The variables of each factor, and of any constraint that has to
        see its variables together.
    kept : Collection of str
        The variables that are not summed out.
    cost : Callable
        The cost of a bag; its number of variables by default.

    Returns
    -------
    tuple of str
        The variables not in ``kept``, in elimination order.
    """
    current = [frozenset(edge) for edge in edges]
    position = {name: index for index, name in enumerate(variables)}
    left = [name for name in variables if name not in kept]
    chosen: list[str] = []
    while left:

        def bag(name: str) -> frozenset[str]:
            """The bag of a variable among the remaining edges.

            Parameters
            ----------
            name : str
                The variable.

            Returns
            -------
            frozenset of str
                The variables of the edges that contain it, and itself.
            """
            return frozenset({name}).union(*(edge for edge in current if name in edge))

        best = min(left, key=lambda name: (cost(bag(name)), position[name]))
        joined = bag(best)
        current = [edge for edge in current if best not in edge] + [joined - {best}]
        left.remove(best)
        chosen.append(best)
    return tuple(chosen)


class ContractionStep(dx.Model):
    """One step of an elimination: factors multiplied, letters summed out.

    Parameters
    ----------
    letters : tuple of str
        The letters summed out at this step: the chosen one and every
        other summed letter whose factors are all joined here.
    factors : tuple of int
        The factors joined: inputs are numbered from zero in order, and
        the result of step ``s`` is factor ``k + s`` for ``k`` inputs.
    bag : tuple of str
        The letters of the joined factors, in order of first mention.
    elements : int
        Elements of the product over the bag.
    """

    letters: tuple[str, ...] = ()
    factors: tuple[int, ...] = ()
    bag: tuple[str, ...] = ()
    elements: int = 0


class ContractionPlan(dx.Model):
    """How a contraction is evaluated, fixed before any value is read.

    Parameters
    ----------
    inputs : tuple of str
        The letters of each input.
    output : str
        The letters of the output.
    extents : dict of str to int
        The extent of each letter, in order of first mention.
    algorithm : str
        ``"variable elimination"``, or ``"written order"`` for an algebra
        that does not establish `ELIMINATION_LAWS`.
    order : tuple of str
        The summed letters in the order `elimination_order` chose; empty
        in the written order.
    steps : tuple of ContractionStep
        The steps of an elimination; empty in the written order.
    width : int
        Letters of the largest bag less one: the induced width of the
        order. In the written order, every letter less one.
    largest : int
        Elements of the largest tensor the plan forms: the largest bag
        or the output, and in the written order the product of every
        extent.
    """

    inputs: tuple[str, ...] = ()
    output: str = ""
    extents: dict[str, int] = {}
    algorithm: Literal["variable elimination", "written order"] = "written order"
    order: tuple[str, ...] = ()
    steps: tuple[ContractionStep, ...] = ()
    width: int = 0
    largest: int = 0

    @property
    def summed(self) -> tuple[str, ...]:
        """The letters summed out, in order of first mention.

        Returns
        -------
        tuple of str
            Every letter not in the output.
        """
        return tuple(name for name in self.extents if name not in self.output)


class ContractionCounts(dx.Model):
    """The work of one contraction, counted in elements of the algebra.

    Parameters
    ----------
    products : int
        Binary products, one per element of each product formed.
    sums : int
        Binary sums: a sum of ``m`` terms is ``m - 1`` of them.
    allocated : int
        Elements of the tensors formed.
    peak : int
        Elements of the tensors the plan formed that were held at once,
        besides the inputs.
    """

    products: int = 0
    sums: int = 0
    allocated: int = 0
    peak: int = 0


def _universe(inputs: Sequence[str]) -> tuple[str, ...]:
    """Every letter, in order of first mention.

    Parameters
    ----------
    inputs : Sequence of str
        The letters of each input.

    Returns
    -------
    tuple of str
        The letters.
    """
    return tuple(dict.fromkeys("".join(inputs)))


def _elements(letters: Collection[str], extents: Mapping[str, int]) -> int:
    """The elements of a tensor over letters.

    Parameters
    ----------
    letters : Collection of str
        The letters.
    extents : Mapping of str to int
        The extent of each.

    Returns
    -------
    int
        The product of their extents.
    """
    return math.prod(extents[name] for name in letters)


def plan_contraction(
    algebra: ContractionAlgebra,
    inputs: Sequence[str],
    output: str,
    extents: Mapping[str, int],
) -> ContractionPlan:
    """Choose how to evaluate a contraction, from its letters and extents.

    Parameters
    ----------
    algebra : ContractionAlgebra
        The algebra, whose claims decide between elimination and the
        written order.
    inputs : Sequence of str
        The letters of each input; no letter repeats within one.
    output : str
        The output letters, each in some input and none repeated.
    extents : Mapping of str to int
        The extent of every letter.

    Returns
    -------
    ContractionPlan
        The plan. Elimination orders its letters by the elements of their
        bags, ties in order of first mention.

    Raises
    ------
    ValueError
        If there is no input, a letter repeats within an input or the
        output, an output letter is in no input, or a letter has no
        extent or a negative one.
    """
    if not inputs:
        raise ValueError("a contraction has at least one input")
    universe = _universe(inputs)
    for name in universe:
        if name not in extents or extents[name] < 0:
            raise ValueError(f"letter {name!r} has no extent of at least zero")
    return _planned(
        algebra.establishes(*ELIMINATION_LAWS),
        tuple(inputs),
        output,
        tuple(int(extents[name]) for name in universe),
    )


@functools.lru_cache(maxsize=4096)
def _planned(
    eliminable: bool,
    inputs: tuple[str, ...],
    output: str,
    sizes: tuple[int, ...],
) -> ContractionPlan:
    """The plan of a contraction, which depends on the algebra only by its laws.

    Parameters
    ----------
    eliminable : bool
        Whether the algebra establishes `ELIMINATION_LAWS`.
    inputs : tuple of str
        The letters of each input.
    output : str
        The output letters.
    sizes : tuple of int
        The extent of each letter, in order of first mention.

    Returns
    -------
    ContractionPlan
        The plan; the same object for the same arguments, which is sound
        because a plan holds letters and extents only.

    Raises
    ------
    ValueError
        If a letter repeats within an input or the output, or an output
        letter is in no input.
    """
    for index, letters in enumerate(inputs):
        if len(set(letters)) != len(letters):
            raise ValueError(f"input {index} repeats a letter of {letters!r}")
    if len(set(output)) != len(output):
        raise ValueError(f"the output repeats a letter of {output!r}")
    universe = _universe(inputs)
    for name in output:
        if name not in universe:
            raise ValueError(f"output letter {name!r} is in no input")
    extents = dict(zip(universe, sizes, strict=True))
    ordered = {name: int(extents[name]) for name in universe}
    if not eliminable:
        return ContractionPlan(
            inputs=tuple(inputs),
            output=output,
            extents=ordered,
            algorithm="written order",
            width=max(len(universe) - 1, 0),
            largest=_elements(universe, extents),
        )
    order = elimination_order(
        universe,
        [set(letters) for letters in inputs],
        set(output),
        cost=lambda bag: _elements(bag, extents),
    )
    factors: dict[int, tuple[str, ...]] = {
        key: tuple(letters) for key, letters in enumerate(inputs)
    }
    pending = list(order)
    steps: list[ContractionStep] = []
    for name in order:
        if name not in pending:
            continue
        joined = [key for key, letters in factors.items() if name in letters]
        bag = tuple(
            letter
            for letter in universe
            if any(letter in factors[key] for key in joined)
        )
        letters = tuple(
            letter
            for letter in pending
            if letter in bag
            and all(key in joined for key, held in factors.items() if letter in held)
        )
        for letter in letters:
            pending.remove(letter)
        steps.append(
            ContractionStep(
                letters=letters,
                factors=tuple(joined),
                bag=bag,
                elements=_elements(bag, extents),
            )
        )
        for key in joined:
            del factors[key]
        factors[len(inputs) + len(steps) - 1] = tuple(
            letter for letter in bag if letter not in letters
        )
    largest = max(
        [step.elements for step in steps] + [_elements(output, extents)], default=0
    )
    return ContractionPlan(
        inputs=tuple(inputs),
        output=output,
        extents=ordered,
        algorithm="variable elimination",
        order=order,
        steps=tuple(steps),
        width=max((len(step.bag) - 1 for step in steps), default=0),
        largest=largest,
    )


class _Tally:
    """Running counts of a contraction, in elements of the algebra."""

    def __init__(self) -> None:
        self.products = 0
        self.sums = 0
        self.allocated = 0
        self.live = 0
        self.peak = 0

    def made(self, elements: int) -> None:
        """Record a tensor formed and now held.

        Parameters
        ----------
        elements : int
            Its elements.
        """
        self.allocated += elements
        self.live += elements
        self.peak = max(self.peak, self.live)

    def freed(self, elements: int) -> None:
        """Record a formed tensor no longer held.

        Parameters
        ----------
        elements : int
            Its elements.
        """
        self.live -= elements

    def counts(self) -> ContractionCounts:
        """The counts so far.

        Returns
        -------
        ContractionCounts
            The counts.
        """
        return ContractionCounts(
            products=self.products,
            sums=self.sums,
            allocated=self.allocated,
            peak=self.peak,
        )


class _Factor:
    """A tensor over letters, and whether the contraction formed it.

    Parameters
    ----------
    letters : tuple of str
        Its index letters, in its axis order.
    tensor : torch.Tensor
        The tensor, its element axes last.
    formed : bool
        Whether the contraction formed it, rather than received it.
    """

    def __init__(self, letters: tuple[str, ...], tensor: torch.Tensor, formed: bool):
        self.letters = letters
        self.tensor = tensor
        self.formed = formed


def _indices(tensor: torch.Tensor, rank: int) -> int:
    """The elements of the algebra a tensor holds.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor, its element axes last.
    rank : int
        The number of element axes.

    Returns
    -------
    int
        The product of its index extents.
    """
    return math.prod(tensor.shape[: tensor.dim() - rank])


def _aligned(factor: _Factor, target: Sequence[str], rank: int) -> torch.Tensor:
    """A view of a factor over target letters, extent one where it has none.

    Parameters
    ----------
    factor : _Factor
        The factor; its letters are among ``target``.
    target : Sequence of str
        The letters, in the order of the view's index axes.
    rank : int
        The number of element axes, kept last.

    Returns
    -------
    torch.Tensor
        A view, never a copy.
    """
    present = [name for name in target if name in factor.letters]
    order = [factor.letters.index(name) for name in present]
    tensor = factor.tensor.permute(
        (*order, *range(len(factor.letters), len(factor.letters) + rank))
    )
    index: list[slice | None] = [
        slice(None) if name in factor.letters else None for name in target
    ]
    return tensor[(*index, *([slice(None)] * rank))]


def _product(
    algebra: ContractionAlgebra,
    factors: Sequence[_Factor],
    target: Sequence[str],
    rank: int,
    tally: _Tally,
) -> torch.Tensor:
    """The product of factors over target letters, left to right.

    One factor is returned as a view; each product formed is counted.

    Parameters
    ----------
    algebra : ContractionAlgebra
        The algebra.
    factors : Sequence of _Factor
        The factors, at least one.
    target : Sequence of str
        The letters of the product.
    rank : int
        The number of element axes.
    tally : _Tally
        The counts.

    Returns
    -------
    torch.Tensor
        The product.
    """
    total = _aligned(factors[0], target, rank)
    formed = False
    for factor in factors[1:]:
        result = algebra.tensor_op(total, _aligned(factor, target, rank))
        elements = _indices(result, rank)
        tally.products += elements
        tally.made(elements)
        if formed:
            tally.freed(_indices(total, rank))
        total, formed = result, True
    return total


def _sum(
    algebra: ContractionAlgebra,
    terms: torch.Tensor,
    axes: tuple[int, ...],
    rank: int,
    tally: _Tally,
) -> torch.Tensor:
    """The sum of a tensor over index axes.

    Parameters
    ----------
    algebra : ContractionAlgebra
        The algebra.
    terms : torch.Tensor
        The terms.
    axes : tuple of int
        The index axes to sum out, at least one.
    rank : int
        The number of element axes.
    tally : _Tally
        The counts.

    Returns
    -------
    torch.Tensor
        The sum.

    Raises
    ------
    ValueError
        If the algebra's sum does not have the shape of its terms without
        the summed axes.
    """
    result = algebra.join(terms, axes)
    wanted = tuple(size for axis, size in enumerate(terms.shape) if axis not in axes)
    if tuple(result.shape) != wanted:
        raise ValueError(
            f"the sum over axes {axes} of terms of shape {tuple(terms.shape)} has "
            f"shape {tuple(result.shape)}, not {wanted}"
        )
    elements = _indices(result, rank)
    tally.sums += max(_indices(terms, rank) - elements, 0)
    tally.made(elements)
    return result


def _check(plan: ContractionPlan, tensors: Sequence[torch.Tensor], rank: int) -> None:
    """Refuse tensors that do not fit a plan.

    Parameters
    ----------
    plan : ContractionPlan
        The plan.
    tensors : Sequence of torch.Tensor
        The inputs.
    rank : int
        The number of element axes.

    Raises
    ------
    TypeError
        If an input is not a tensor.
    ValueError
        If the number of inputs, an input's number of axes, or an extent
        differs from the plan.
    """
    if len(tensors) != len(plan.inputs):
        raise ValueError(
            f"the contraction takes {len(plan.inputs)} inputs, got {len(tensors)}"
        )
    extents = plan.extents
    for index, (letters, tensor) in enumerate(zip(plan.inputs, tensors, strict=True)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"input {index} of the contraction is a {type(tensor).__name__}, "
                "not a tensor"
            )
        if tensor.dim() != len(letters) + rank:
            raise ValueError(
                f"input {index} has {tensor.dim()} axes; its letters {letters!r} "
                f"and {rank} element axes need {len(letters) + rank}"
            )
        found = tuple(tensor.shape[: len(letters)])
        wanted = tuple(extents[name] for name in letters)
        if found != wanted:
            raise ValueError(
                f"input {index} has extents {found}, the plan {wanted} for {letters!r}"
            )


def run_contraction(
    plan: ContractionPlan,
    algebra: ContractionAlgebra,
    tensors: Sequence[torch.Tensor],
    element_rank: int = 0,
) -> tuple[torch.Tensor, ContractionCounts]:
    """Evaluate a contraction by its plan.

    In the written order every input is aligned to all letters, the
    inputs are multiplied left to right, and the summed letters are
    summed at once. In an elimination each step multiplies the factors
    it joins over its bag, in the order they were made, and sums its
    letters away; the factors left are multiplied over the output. At a
    tie of a selective sum the derivative goes to the first best term of
    each sum the plan forms, so to the first best index in the order of
    elimination.

    Parameters
    ----------
    plan : ContractionPlan
        The plan.
    algebra : ContractionAlgebra
        The algebra the plan was made for.
    tensors : Sequence of torch.Tensor
        The inputs, each with its index axes in the order of its letters
        and then ``element_rank`` element axes.
    element_rank : int
        The number of trailing axes of one element of the algebra.

    Returns
    -------
    tuple of torch.Tensor and ContractionCounts
        The output, its index axes in the order of the output letters,
        and the work.

    Raises
    ------
    TypeError
        If an input is not a tensor.
    ValueError
        If the inputs do not fit the plan.
    """
    _check(plan, tensors, element_rank)
    tally = _Tally()
    universe = tuple(plan.extents)
    received = [
        _Factor(letters=tuple(letters), tensor=tensor, formed=False)
        for letters, tensor in zip(plan.inputs, tensors, strict=True)
    ]
    if plan.algorithm == "written order":
        total = _product(algebra, received, universe, element_rank, tally)
        axes = tuple(
            index for index, name in enumerate(universe) if name not in plan.output
        )
        if axes:
            total = _sum(algebra, total, axes, element_rank, tally)
        kept = [name for name in universe if name in plan.output]
        result = _Factor(letters=tuple(kept), tensor=total, formed=True)
        return _aligned(result, plan.output, element_rank), tally.counts()
    factors: dict[int, _Factor] = dict(enumerate(received))
    for number, step in enumerate(plan.steps):
        joined = [factors.pop(key) for key in step.factors]
        product = _product(algebra, joined, step.bag, element_rank, tally)
        for factor in joined:
            if factor.formed:
                tally.freed(_indices(factor.tensor, element_rank))
        axes = tuple(step.bag.index(name) for name in step.letters)
        reduced = _sum(algebra, product, axes, element_rank, tally)
        if len(joined) > 1:
            tally.freed(_indices(product, element_rank))
        factors[len(plan.inputs) + number] = _Factor(
            letters=tuple(name for name in step.bag if name not in step.letters),
            tensor=reduced,
            formed=True,
        )
    left = list(factors.values())
    total = _product(algebra, left, plan.output, element_rank, tally)
    return total, tally.counts()
