"""Checks of wiring contractions against sums over index assignments."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import partial

import torch

from quivers.core.algebras import (
    BooleanAlgebra,
    CompositionRule,
    GODEL,
    CountingAlgebra,
    GodelAlgebra,
    LogProbAlgebra,
    MarkovAlgebra,
    MaxPlusAlgebra,
    ProductFuzzyAlgebra,
    RealAlgebra,
    StandardMVAlgebra,
    TropicalAlgebra,
)
from quivers.core.wiring import EinsumWiring
from quivers.enriched.day_convolution import day_convolution
from quivers.categorical.monoidal import CartesianProductMonoidal
from quivers.core.objects import FinSet
from quivers.qiec.contracts import Phase, measured_total
from tests.evidence.checks._registry import add
from tests.evidence.counters import TensorCounter
from tests.evidence.compare import (
    differentiate,
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_gradients,
    compare_tensor,
)
from tests.evidence.oracles import exact
from tests.evidence.problems import logarithm

type Table = dict[tuple[int, ...], Fraction]
"""The entries of one tensor."""

NETWORKS: dict[str, tuple[str, dict[str, int]]] = {
    "product": ("ij,jk->ik", {"i": 2, "j": 3, "k": 2}),
    "chain": ("ij,jk,kl,lm->im", {"i": 2, "j": 3, "k": 2, "l": 3, "m": 2}),
    "triangle": ("ij,jk,ki->", {"i": 3, "j": 3, "k": 3}),
    "star": ("ai,bi,ci->abc", {"a": 2, "b": 2, "c": 2, "i": 3}),
    "transposed": ("ij,jk->ki", {"i": 2, "j": 3, "k": 4}),
    "no_output": ("i,i->", {"i": 4}),
    "unary": ("ij->j", {"i": 3, "j": 2}),
}
"""Contraction networks and their extents."""


def random_networks(count: int, seed: int) -> dict[str, tuple[str, dict[str, int]]]:
    """Contraction networks drawn at random, of at most 216 assignments.

    Each has two to four inputs of one to three letters from three to six
    letters of extent one to three, and an output of up to two of them.

    Parameters
    ----------
    count : int
        How many.
    seed : int
        The seed.

    Returns
    -------
    dict of str to tuple of str and dict of str to int
        Each network's wiring and extents, by name.
    """
    chooser = random.Random(seed)
    found: dict[str, tuple[str, dict[str, int]]] = {}
    while len(found) < count:
        letters = "abcdef"[: chooser.randint(3, 6)]
        inputs = [
            "".join(chooser.sample(letters, chooser.randint(1, 3)))
            for _ in range(chooser.randint(2, 4))
        ]
        universe = sorted(set("".join(inputs)))
        sizes = {x: chooser.randint(1, 3) for x in universe}
        if math.prod(sizes.values()) > 216:
            continue
        output = "".join(
            chooser.sample(universe, chooser.randint(0, min(2, len(universe))))
        )
        found[f"random_{len(found)}"] = (",".join(inputs) + "->" + output, sizes)
    return found


NETWORKS.update(random_networks(12, 2026))
"""Twelve random networks join the named ones."""

RULES: dict[str, CompositionRule] = {
    "real": RealAlgebra(),
    "counting": CountingAlgebra(),
    "markov": MarkovAlgebra(),
    "logprob": LogProbAlgebra(),
    "boolean": BooleanAlgebra(),
    "maxplus": MaxPlusAlgebra(),
    "tropical": TropicalAlgebra(),
    "godel": GodelAlgebra(),
    "product_fuzzy": ProductFuzzyAlgebra(),
    "standard_mv": StandardMVAlgebra(),
}
"""The composition rules a contraction is checked in."""


def table(letters: str, sizes: Mapping[str, int], salt: int, *, zeros: bool) -> Table:
    """A rational tensor with entries in the open unit interval.

    Parameters
    ----------
    letters : str
        Its index letters.
    sizes : Mapping[str, int]
        The extent of each letter.
    salt : int
        A number that makes tensors different.
    zeros : bool
        Whether every fourth entry is exactly zero.

    Returns
    -------
    Table
        The entries.
    """
    entries: Table = {}
    for position, key in enumerate(
        itertools.product(*(range(sizes[x]) for x in letters))
    ):
        numerator = 1 + (7 * position + 5 * salt) % 12
        entries[key] = Fraction(numerator, 13)
        if zeros and position % 4 == salt % 4:
            entries[key] = Fraction(0)
    return entries


def decode(rule: str, value: Fraction) -> float:
    """A rational in a rule's representation.

    Parameters
    ----------
    rule : str
        The rule's name.
    value : Fraction
        The value.

    Returns
    -------
    float
        Its representation.
    """
    if rule in ("logprob", "maxplus"):
        return logarithm(value)
    if rule == "tropical":
        return -logarithm(value)
    return float(value)


def tensor(
    rule: str, letters: str, sizes: Mapping[str, int], entries: Table
) -> torch.Tensor:
    """A rational tensor in a rule's representation.

    Parameters
    ----------
    rule : str
        The rule's name.
    letters : str
        The index letters.
    sizes : Mapping[str, int]
        The extents.
    entries : Table
        The entries.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    shape = [sizes[x] for x in letters]
    values = [decode(rule, value) for value in entries.values()]
    return torch.tensor(values, dtype=torch.float64).reshape(shape)


def read(rule: str, entries: Table) -> Table:
    """Round a tensor's entries into a rule's carrier.

    Parameters
    ----------
    rule : str
        The rule's name.
    entries : Table
        Rational entries.

    Returns
    -------
    Table
        Zero and one for the Boolean rule, whole numbers for the counting
        rule, the entries otherwise.
    """
    if rule == "boolean":
        return {
            key: Fraction(int(value > Fraction(1, 2))) for key, value in entries.items()
        }
    if rule == "counting":
        return {key: Fraction(int(13 * value)) for key, value in entries.items()}
    return entries


def reduce_terms(rule: str, terms: Sequence[Fraction]) -> Fraction:
    """Combine the terms of one output entry by the rule's definition.

    Parameters
    ----------
    rule : str
        The rule's name.
    terms : Sequence of Fraction
        The products, in the order of the contracted assignments.

    Returns
    -------
    Fraction
        The result.
    """
    if rule in ("boolean", "godel", "maxplus", "tropical"):
        return max(terms)
    if rule == "standard_mv":
        return min(Fraction(1), sum(terms, Fraction(0)))
    if rule == "product_fuzzy":
        remainder = Fraction(1)
        for term in terms:
            remainder *= 1 - term
        return 1 - remainder
    return sum(terms, Fraction(0))


def multiply(rule: str, left: Fraction, right: Fraction) -> Fraction:
    """Combine two entries by the rule's definition.

    Parameters
    ----------
    rule : str
        The rule's name.
    left : Fraction
        The first entry.
    right : Fraction
        The second entry.

    Returns
    -------
    Fraction
        The product: the minimum for the Godel rule, the truncated sum
        for the Lukasiewicz rule, the product of numbers otherwise.
    """
    if rule == "godel":
        return min(left, right)
    if rule == "standard_mv":
        return max(Fraction(0), left + right - 1)
    return left * right


def reference(
    rule: str, spec: str, sizes: Mapping[str, int], tables: Sequence[Table]
) -> Table:
    """A contraction from its definition, one index assignment at a time.

    Parameters
    ----------
    rule : str
        The rule's name.
    spec : str
        The wiring.
    sizes : Mapping[str, int]
        The extents.
    tables : Sequence of Table
        The inputs.

    Returns
    -------
    Table
        The entries of the result.
    """
    left, output = spec.split("->")
    inputs = left.split(",")
    letters = list(dict.fromkeys("".join(inputs)))
    terms: dict[tuple[int, ...], list[Fraction]] = {
        key: [] for key in itertools.product(*(range(sizes[x]) for x in output))
    }
    for values in itertools.product(*(range(sizes[x]) for x in letters)):
        at = dict(zip(letters, values, strict=True))
        product: Fraction | None = None
        for names, entries in zip(inputs, tables, strict=True):
            entry = entries[tuple(at[x] for x in names)]
            product = entry if product is None else multiply(rule, product, entry)
        assert product is not None
        terms[tuple(at[x] for x in output)].append(product)
    return {key: reduce_terms(rule, found) for key, found in terms.items()}


def inputs_of(
    network: str, rule: str, *, zeros: bool
) -> tuple[str, dict[str, int], list[Table], list[torch.Tensor]]:
    """The data of one contraction.

    Parameters
    ----------
    network : str
        The network.
    rule : str
        The rule's name.
    zeros : bool
        Whether some entries are exactly zero.

    Returns
    -------
    tuple
        The wiring, the extents, the rational inputs, and the tensors.
    """
    spec, sizes = NETWORKS[network]
    names = spec.split("->")[0].split(",")
    tables = [
        read(rule, table(letters, sizes, salt, zeros=zeros))
        for salt, letters in enumerate(names)
    ]
    tensors = [
        tensor(rule, letters, sizes, entries)
        for letters, entries in zip(names, tables, strict=True)
    ]
    return spec, sizes, tables, tensors


def accuracy(network: str, rule: str) -> Accuracy:
    """The accuracy of a contraction.

    One output entry multiplies one entry per input for every contracted
    assignment and combines the assignments. The noisy-or of a rule of
    complements subtracts from one, so its condition number is the
    reciprocal of the smallest result.

    Parameters
    ----------
    network : str
        The network.
    rule : str
        The rule's name.

    Returns
    -------
    Accuracy
        The accuracy.
    """
    spec, sizes = NETWORKS[network]
    inputs = spec.split("->")[0].split(",")
    assignments = math.prod(sizes.values())
    log = rule in ("logprob", "maxplus", "tropical")
    condition = 1.0
    if rule in ("product_fuzzy", "standard_mv"):
        _, _, tables, _ = inputs_of(network, rule, zeros=False)
        smallest = min(
            (x for x in reference(rule, spec, sizes, tables).values() if x != 0),
            default=Fraction(1),
        )
        condition = len(inputs) * float(1 / smallest)
    return Accuracy(
        domain="log" if log else "linear",
        operations=2 * assignments * (len(inputs) + 2 * ELEMENTARY),
        magnitude=8.0 * len(inputs) if log else 1.0,
        condition=condition,
    )


def check_contraction(network: str, rule: str, zeros: bool) -> None:
    """Compare a contraction with the sum over assignments.

    Parameters
    ----------
    network : str
        The network.
    rule : str
        The rule's name.
    zeros : bool
        Whether some entries are exactly zero.
    """
    spec, sizes, tables, tensors = inputs_of(network, rule, zeros=zeros)
    output = spec.split("->")[1]
    wanted = reference(rule, spec, sizes, tables)
    compare_tensor(
        EinsumWiring(RULES[rule], spec).apply(*tensors),
        tensor(rule, output, sizes, wanted),
        accuracy(network, rule),
        f"{rule} {network}",
    )


def check_contraction_gradient(network: str, rule: str) -> None:
    """Compare a contraction's gradient at exact zeros with the oracle's.

    The scalar is the sum of the output entries. The derivative in an
    entry is the sum over the assignments that use it of the other
    factors, so it is nonzero where the entry itself is zero.

    Parameters
    ----------
    network : str
        The network.
    rule : str
        ``"real"`` or ``"markov"``.
    """
    spec, sizes, tables, tensors = inputs_of(network, rule, zeros=True)
    leaves = [value.requires_grad_() for value in tensors]
    total = EinsumWiring(RULES[rule], spec).apply(*leaves).sum()
    found = torch.autograd.grad(total, leaves, allow_unused=True)
    names = spec.split("->")[0].split(",")
    gradients: dict[str, torch.Tensor | None] = {}
    wanted: dict[str, Fraction] = {}
    for position, (letters, entries) in enumerate(zip(names, tables, strict=True)):
        gradient = found[position]
        for key in entries:
            label = f"{position}{letters}{key}"
            gradients[label] = None if gradient is None else gradient[key]
            unit: Table = {other: Fraction(int(other == key)) for other in entries}
            changed = [*tables[:position], unit, *tables[position + 1 :]]
            wanted[label] = sum(
                exact.contraction(
                    names, "", sizes, changed, Fraction(0), Fraction(1)
                ).values(),
                Fraction(0),
            )
    bound = accuracy(network, rule)
    compare_gradients(
        gradients,
        wanted,
        Accuracy(domain="linear", operations=5 * bound.operations),
        f"gradient of {rule} {network}",
    )


def check_empty(rule: str) -> None:
    """Require a contraction over no assignment to be the rule's zero.

    Parameters
    ----------
    rule : str
        The rule's name.

    Raises
    ------
    Mismatch
        If the value differs, or the contraction refuses a sum whose shape
        is not the shape of its terms without the summed axes.
    """
    zero = {"logprob": -math.inf, "maxplus": -math.inf, "tropical": math.inf}.get(
        rule, 0.0
    )
    for dtype in (torch.float32, torch.float64):
        left = torch.zeros(2, 0, dtype=dtype, requires_grad=True)
        right = torch.zeros(0, 3, dtype=dtype, requires_grad=True)
        try:
            value = EinsumWiring(RULES[rule], "ij,jk->ik").apply(left, right)
        except ValueError as refused:
            raise Mismatch("shape", math.inf, str(refused)) from refused
        compare_tensor(
            value.detach().abs() if zero == 0.0 else value,
            torch.full((2, 3), zero, dtype=dtype),
            Accuracy(domain="linear", operations=1),
            f"{rule} over no assignment",
        )


def check_impossible_row() -> None:
    """Compare a gradient of a contraction with an impossible row.

    The first row of the left factor is impossible. The objective reads
    the second row, whose derivative in a log entry of the left factor is
    that entry's share of each output, and is zero in the first row.
    """
    left = [[Fraction(0), Fraction(0)], [Fraction(1, 3), Fraction(2, 5)]]
    right = [[Fraction(1, 2), Fraction(1, 7)], [Fraction(3, 4), Fraction(2, 9)]]
    leaves = [
        torch.tensor(
            [[logarithm(x) for x in row] for row in matrix], dtype=torch.float64
        ).requires_grad_()
        for matrix in (left, right)
    ]
    value = EinsumWiring(RULES["logprob"], "ij,jk->ik").apply(*leaves)[1].sum()
    (found,) = differentiate(value, leaves[:1], "impossible row")
    reference: dict[str, Fraction] = {}
    for row in range(2):
        for column in range(2):
            reference[f"{row}{column}"] = Fraction(0)
            if row == 1:
                for k in range(2):
                    total = sum(
                        (left[1][m] * right[m][k] for m in range(2)), Fraction(0)
                    )
                    reference[f"{row}{column}"] += (
                        left[1][column] * right[column][k] / total
                    )
    compare_gradients(
        {key: found[int(key[0]), int(key[1])] for key in reference},
        reference,
        Accuracy(domain="linear", operations=40 * ELEMENTARY, condition=4.0),
        "impossible row",
    )


def check_refuses_diagonal() -> None:
    """Require a repeated letter in one input to be refused by name.


    Raises
    ------
    Mismatch
        If the wiring is built or the diagnostic differs.
    """
    wanted = (
        "EinsumWiring: input 0 of spec 'ii->i' repeats an axis letter ('ii'); "
        "diagonal/trace contractions are not supported by this wiring"
    )
    try:
        EinsumWiring(RealAlgebra(), "ii->i")
    except ValueError as refusal:
        if str(refusal) == wanted:
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, "a diagonal wiring was built")


def check_order() -> None:
    """Require the factors of a wiring to be multiplied in written order.

    The rule of left projection is not commutative, so the result names
    the first input.
    """
    matrix = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    other = torch.tensor([[5.0], [7.0]], dtype=torch.float64)
    compare_tensor(
        EinsumWiring(_LeftProjection(), "ij,jk->ik").apply(matrix, other),
        torch.tensor([[5.0]], dtype=torch.float64),
        Accuracy(domain="linear", operations=2),
        "left projection",
    )


def chain_of_matrices(
    length: int, size: int
) -> tuple[EinsumWiring, list[torch.Tensor]]:
    """A line of log-weight matrices and its contraction to the end points.

    Parameters
    ----------
    length : int
        The number of matrices.
    size : int
        Their order.

    Returns
    -------
    tuple of EinsumWiring and list of torch.Tensor
        The wiring and the matrices.
    """
    letters = [chr(0x100 + k) for k in range(length + 1)]
    spec = (
        ",".join(letters[k] + letters[k + 1] for k in range(length))
        + "->"
        + letters[0]
        + letters[length]
    )
    generator = torch.Generator().manual_seed(length)
    matrices = [
        torch.rand(size, size, dtype=torch.float64, generator=generator).log()
        for _ in range(length)
    ]
    return EinsumWiring(RULES["logprob"], spec), matrices


def check_chain_peak() -> None:
    """Require a line of matrices to be contracted within three of its axes.

    A line of ``k`` matrices of order six, contracted to its end points,
    is eliminated one inner letter at a time; the largest tensor formed
    is a bag of three letters, ``6^3`` elements, for every ``k``, as the
    dispatch counter sees it and as the plan and its work report record.

    Raises
    ------
    Mismatch
        If a larger tensor is formed or recorded.
    """
    for length in (3, 4, 5):
        wiring, matrices = chain_of_matrices(length, 6)
        with TensorCounter() as counter:
            _, report = wiring.apply_with_work(*matrices)
        plan = wiring.plan({name: 6 for name in "".join(wiring.input_specs)})
        for what, found in (
            ("formed", counter.largest),
            ("planned", plan.largest),
        ):
            if found != 6**3:
                raise Mismatch(
                    "work",
                    found / 6**3,
                    f"{length} matrices: the largest tensor {what} has {found} elements",
                )
        if report.peak_live_elements > 2 * 6**3:
            raise Mismatch(
                "work",
                report.peak_live_elements / 6**3,
                f"{length} matrices: {report.peak_live_elements} elements held",
            )


def check_counts_are_measured() -> None:
    """Require the work a contraction reports to be the work it dispatches.

    Over the reals a product is one multiplication, and a sum over ``a``
    axes of a bag is ``a`` sums of one axis, each reading at most the
    bag. So the dispatch counter's products are the reported ones, and
    its sums lie between the reported binary sums and the bags' elements
    times their summed letters. Every named and random network is
    checked.

    Raises
    ------
    Mismatch
        If the counts disagree.
    """
    for network, (spec, sizes) in NETWORKS.items():
        _, _, _, tensors = inputs_of(network, "real", zeros=False)
        wiring = EinsumWiring(RULES["real"], spec)
        with TensorCounter() as counter:
            value, report = wiring.apply_with_work(*tensors)
        plan = wiring.plan(sizes)
        bags = sum(step.elements * len(step.letters) for step in plan.steps)
        if counter.counts["products"] != report.scalar_products:
            raise Mismatch(
                "work",
                math.inf,
                f"{network}: dispatched {counter.counts['products']} products, "
                f"reported {report.scalar_products}",
            )
        if not report.scalar_sums <= counter.counts["sums"] <= max(bags, value.numel()):
            raise Mismatch(
                "work",
                math.inf,
                f"{network}: dispatched {counter.counts['sums']} sums, reported "
                f"{report.scalar_sums} over bags of {bags} elements",
            )
        bound = next(
            bound
            for bound in wiring.complexity.bounds
            if bound.phase is Phase.EXECUTION and bound.resource == "time"
        )
        terms = len(plan.inputs) + len(plan.summed)
        if measured_total(bound, report) > 2 * terms * max(plan.largest, 1):
            raise Mismatch(
                "work",
                measured_total(bound, report) / (2 * terms * max(plan.largest, 1)),
                f"{network}: work above twice (k + s) W",
            )


def _day_reference(
    f: Sequence[Fraction],
    g: Sequence[Fraction],
    table: Mapping[tuple[int, int], tuple[int, Fraction]],
) -> list[Fraction]:
    """Day convolution over a product table in the Godel algebra, by its definition.

    Parameters
    ----------
    f : Sequence of Fraction
        The first presheaf.
    g : Sequence of Fraction
        The second presheaf.
    table : Mapping
        The object and weight of each pair.

    Returns
    -------
    list of Fraction
        ``max`` over each object's pairs of ``min(h, f, g)``.
    """
    out = [Fraction(0)] * len(f)
    for (a, b), (target, hom) in table.items():
        out[target] = max(out[target], min(hom, f[a], g[b]))
    return out


def check_day_convolution() -> None:
    """Compare Day convolution with its definition, and its work with ``n^2``.

    Over ``Z/5`` with weights on the pairs, in the Godel algebra, each
    object's value is the largest ``min(h, f(a), g(b))`` over its pairs.
    The products are two per pair and no tensor of ``n^3`` elements is
    formed.

    Raises
    ------
    Mismatch
        If a value differs, or a tensor of more than ``n^2`` elements is
        formed.
    """
    n = 5
    f = [Fraction(1 + (3 * a) % 7, 8) for a in range(n)]
    g = [Fraction(1 + (5 * b + 2) % 7, 8) for b in range(n)]
    table = {
        (a, b): ((a + b) % n, Fraction(1 + (a * b) % 6, 7))
        for a in range(n)
        for b in range(n)
    }
    objects = [FinSet(name=f"g{i}", cardinality=1) for i in range(n)]
    with TensorCounter() as counter:
        value = day_convolution(
            torch.tensor([float(x) for x in f], dtype=torch.float64),
            torch.tensor([float(x) for x in g], dtype=torch.float64),
            objects,
            CartesianProductMonoidal(),
            GODEL,
            product_table={
                key: (target, torch.tensor(float(hom), dtype=torch.float64))
                for key, (target, hom) in table.items()
            },
        )
    compare_tensor(
        value,
        torch.tensor(
            [float(x) for x in _day_reference(f, g, table)], dtype=torch.float64
        ),
        Accuracy(domain="linear", operations=4),
        "Day convolution over Z/5",
    )
    if counter.largest > n * n + 1:
        raise Mismatch(
            "work",
            counter.largest / (n * n),
            f"a tensor of {counter.largest} elements for {n} objects",
        )


class _LeftProjection(CompositionRule):
    """A rule whose product is its first factor and whose join is the sum."""

    @property
    def name(self) -> str:
        """The rule's name.

        Returns
        -------
        str
            The name.
        """
        return "left_projection"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Keep the first factor.

        Parameters
        ----------
        a : torch.Tensor
            The first factor.
        b : torch.Tensor
            The second factor, which gives only its shape.

        Returns
        -------
        torch.Tensor
            The first factor over the joint shape.
        """
        return a.expand(torch.broadcast_shapes(a.shape, b.shape))

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Sum.

        Parameters
        ----------
        t : torch.Tensor
            The terms.
        dim : int or tuple of int
            The axes.

        Returns
        -------
        torch.Tensor
            The sums.
        """
        return t.sum(dim=dim)


def _register() -> None:
    """Register every check of this module."""
    row = "einsum-wiring"
    for rule in RULES:
        for network in NETWORKS:
            for zeros in (False, True):
                add(
                    f"contraction/value/{rule}/{network}/{'zeros' if zeros else 'positive'}",
                    row,
                    "Q6",
                    partial(check_contraction, network, rule, zeros),
                )
        add(f"contraction/empty/{rule}", row, "Q6", partial(check_empty, rule))
    for rule in ("real", "markov"):
        for network in NETWORKS:
            add(
                f"contraction/gradient/{rule}/{network}",
                row,
                "Q1g",
                partial(check_contraction_gradient, network, rule),
            )
    add("contraction/order", row, "Q6", check_order)
    add("contraction/chain-peak", row, "Q6", check_chain_peak)
    add("contraction/counts-measured", row, "Q6", check_counts_are_measured)
    add("contraction/day-convolution", row, "Q6", check_day_convolution)
    add("contraction/impossible_row", row, "Q1g", check_impossible_row)
    add("contraction/refuses_diagonal", row, "Q6", check_refuses_diagonal)


_register()
