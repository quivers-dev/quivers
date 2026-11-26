"""N-ary tensor contractions specified by shared axis names.

A `CompositionRule` defines binary composition:
``(f >> g)[i, k] = ⋁_j f[i, j] ⊗ g[j, k]``. A wiring rule can instead
combine several tensors in one contraction, for instance:

    out[s, c] = ⋁_{p, q} (
        arg1[s, p] ⊗ arg2[s, q] ⊗ kernel[p, q, c]
    )

Here ``s`` remains shared by both arguments, while ``p`` and ``q`` are
eliminated. Expressing this through binary composition would also require
operations that arrange, copy, or pair the interfaces; plain chaining alone
does not describe that wiring.

`WiringRule` pairs a `CompositionRule`, which supplies ``tensor_op`` and
``join``, with an einsum-style specification of the axes to retain or reduce.
`EinsumWiring` evaluates it by the plans of `quivers.core.elimination`:
variable elimination when the rule establishes the laws of a commutative
semiring, the written order otherwise. `CONTRACTION_WORK` and
`WRITTEN_ORDER_WORK` are the cost claims of the two, and
`EinsumWiring.apply_with_work` measures the counters they bound.
Wiring diagrams and operads provide a mathematical account of how compatible
interfaces connect [Vagner et al. (2015)](<https://www.tac.mta.ca/tac/volumes/30/51/30-51abs.html>). Interpreting these contractions as
operations of an operad requires the relevant composition laws, including
the compatibility of tensor products and reductions. A runtime wiring
specification alone does not establish those laws.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal

import torch

from quivers.core.algebras import Carrier, CompositionRule
from quivers.core.elimination import (
    ELIMINATION_LAWS,
    ContractionCounts,
    ContractionPlan,
    plan_contraction,
    run_contraction,
)
from quivers.core.laws import Citation
from quivers.qiec.contracts import (
    ALGEBRA_OPERATIONS,
    ComplexityBound,
    ComplexityClaim,
    ComplexityParameter,
    Conditional,
    Phase,
    PhaseInapplicable,
    Unresolved,
    WorkField,
    WorkReport,
    measuring,
)

CONTRACTION_MEASURES: tuple[WorkField, ...] = measuring(
    {
        WorkField.SCALAR_SUMS,
        WorkField.SCALAR_PRODUCTS,
        WorkField.SCALAR_DIVISIONS,
        WorkField.STAR_PIVOTS,
        WorkField.EXACT_OPERATIONS,
        WorkField.KEY_COMPARISONS,
        WorkField.HASH_OPERATIONS,
        WorkField.ELEMENTS_ALLOCATED,
        WorkField.PEAK_LIVE_ELEMENTS,
    }
)
"""The counters a contraction measures."""

_FAQ = Citation(
    reference="Abo Khamis, Ngo and Rudra 2016, FAQ: Questions Asked Frequently",
    locator="InsideOut and the width of an order",
    doi="10.1145/2902251.2902280",
)
_DECHTER = Citation(
    reference=("Dechter 1999, Bucket elimination: a unifying framework for reasoning"),
    locator="the cost of an order is exponential in its induced width",
    doi="10.1016/S0004-3702(99)00059-4",
)
_MARX = Citation(
    reference="Marx 2010, Can you beat treewidth?",
    locator="lower bound under the exponential time hypothesis",
    doi="10.4086/toc.2010.v006a005",
)
_BAUR_STRASSEN = Citation(
    reference="Baur and Strassen 1983, The complexity of partial derivatives",
    locator="the gradient costs a constant times the function",
    doi="10.1016/0304-3975(83)90110-X",
)

CONTRACTION_WORK = ComplexityClaim(
    query="contraction of a tensor network over a commutative semiring",
    parameters=(
        ComplexityParameter(name="k", meaning="inputs"),
        ComplexityParameter(name="s", meaning="letters summed out"),
        ComplexityParameter(
            name="W",
            meaning=(
                "elements of the largest bag of the chosen order, or of the "
                "output when that is larger"
            ),
        ),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="(k + s) W",
            status=Conditional(
                hypothesis=(
                    "the exponential time hypothesis: for classes of bounded "
                    "arity no algorithm beats treewidth by more than a "
                    "logarithmic factor in the exponent"
                ),
                sources=(_FAQ, _DECHTER, _MARX),
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="(s + 1) W",
            status=Conditional(hypothesis="as for time", sources=(_FAQ, _DECHTER)),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="time",
            expression="(k + s) W",
            status=Conditional(
                hypothesis="as for execution; a gradient costs a constant times it",
                sources=(_FAQ, _BAUR_STRASSEN),
            ),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="(k + s) W",
            status=Conditional(
                hypothesis="the autograd tape holds the operands of every product",
                sources=(_FAQ,),
            ),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason=(
                "the order is chosen from the letters and their extents, in time "
                "polynomial in the letters and independent of the values"
            ),
        ),
        PhaseInapplicable(
            phase=Phase.REFIT,
            reason="a refit is one more contraction; the plan of the extents is kept",
        ),
    ),
    conditions=(
        "every law of ELIMINATION_LAWS established by the algebra's own claims",
        "the order is greedy by the elements of each bag and is recorded with "
        "its widths; an order of least width is NP-hard to find and is not "
        "claimed",
        "at a tie of a selective sum the derivative goes to the first best "
        "index in the order of elimination",
    ),
)
"""The cost of a contraction by variable elimination."""

_WRITTEN_BOUNDS: tuple[tuple[Phase, Literal["time", "space"], str], ...] = (
    (Phase.EXECUTION, "time", "k U"),
    (Phase.EXECUTION, "space", "U"),
    (Phase.DIFFERENTIATION, "time", "k U"),
    (Phase.DIFFERENTIATION, "space", "k U"),
)
"""The bounds of the written order, by phase and resource."""

WRITTEN_ORDER_WORK = ComplexityClaim(
    query="contraction of a tensor network over an algebra without those laws",
    parameters=(
        ComplexityParameter(name="k", meaning="inputs"),
        ComplexityParameter(
            name="U", meaning="the product of the extents of every letter"
        ),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=tuple(
        ComplexityBound(
            phase=phase,
            resource=resource,
            expression=expression,
            status=Unresolved(
                obligation=(
                    "whether fewer operations suffice: the algebra establishes no "
                    "law that moves a sum inside a product, so the written "
                    "expression is evaluated as it is defined"
                )
            ),
        )
        for phase, resource, expression in _WRITTEN_BOUNDS
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason="the written order is read off the specification",
        ),
        PhaseInapplicable(
            phase=Phase.REFIT,
            reason="a refit is one more contraction",
        ),
    ),
    conditions=(
        "the inputs are multiplied left to right in written order, then every "
        "contracted letter is summed at once, as the contraction is defined",
    ),
)
"""The cost of a contraction in the written order."""


class WiringRule(ABC):
    """An n-ary morphism contraction under a fixed
    `CompositionRule`.

    Generalizes binary composition ``>>`` to arbitrary-arity
    tensor networks. A ``WiringRule`` carries:

    * a ``composition_rule`` whose ``tensor_op`` and ``join``
      drive the contraction;
    * a wiring specification that names which axes of which
      inputs participate in which contractions, and which axes
      survive to the output.

    Subclasses implement `apply` to perform the contraction
    given a list of input tensors.
    """

    @property
    @abstractmethod
    def composition_rule(self) -> CompositionRule:
        """The underlying binary composition rule of this contraction.

        Returns
        -------
        CompositionRule
            The enriching algebra.
        """

    @property
    @abstractmethod
    def input_arity(self) -> int:
        """Number of input tensors this wiring expects.

        Returns
        -------
        int
            The number of inputs.
        """

    @abstractmethod
    def apply(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Run the contraction on the given input tensors.

        Parameters
        ----------
        *tensors : torch.Tensor
            The inputs.

        Returns
        -------
        torch.Tensor
            The contraction.
        """

    @property
    def name(self) -> str:
        """The class name.

        Returns
        -------
        str
            The name.
        """
        return type(self).__name__

    def __repr__(self) -> str:
        """The class name and arity.

        Returns
        -------
        str
            The representation.
        """
        return f"{type(self).__name__}(arity={self.input_arity})"


class EinsumWiring(WiringRule):
    """A multi-input contraction specified by an einsum-style string.

    Spec syntax::

        "input_1, input_2, ..., input_n -> output"

    where each ``input_i`` and ``output`` is a sequence of axis
    letters identifying which dimensions of that tensor align.
    Axes that appear in some input but not in the output are
    contracted (reduced via the composition rule's ``join``);
    axes that appear in the output are preserved (broadcast as
    needed).

    Examples::

        EinsumWiring(product_fuzzy, "ij, jk -> ik")
            # binary composition: contract over j, preserve (i, k)
        EinsumWiring(product_fuzzy, "sp, sq, pqo -> so")
            # ternary contraction: two argument tensors over
            # (s, *) and a kernel contracted over (p, q),
            # preserving (s, o)

    How the contraction is evaluated is fixed by
    `quivers.core.elimination.plan_contraction` from the rule's law
    claims and the extents of the inputs, and recorded as a
    `quivers.core.elimination.ContractionPlan`. A rule that establishes
    every law of `quivers.core.elimination.ELIMINATION_LAWS` is
    contracted by variable elimination, with the cost of
    `CONTRACTION_WORK`; any other rule is evaluated in the written order,
    with the cost of `WRITTEN_ORDER_WORK`: every input is aligned to all
    letters, the inputs are folded left to right with ``tensor_op``, and
    the contracted letters are reduced at once with ``join``. A rule
    whose elements have axes of their own, such as a
    `quivers.core.algebras.MatrixSemiring`, keeps them last in every
    tensor.

    Parameters
    ----------
    composition_rule : CompositionRule
        The rule supplying ``tensor_op`` and ``join``.
    spec : str
        The einsum specification.

    Raises
    ------
    TypeError
        If ``composition_rule`` is not a `CompositionRule`.
    ValueError
        If the specification is malformed.
    """

    def __init__(self, composition_rule: CompositionRule, spec: str) -> None:
        """Parse and check a specification.

        Parameters
        ----------
        composition_rule : CompositionRule
            The rule supplying ``tensor_op`` and ``join``.
        spec : str
            The einsum specification.

        Raises
        ------
        TypeError
            If ``composition_rule`` is not a `CompositionRule`.
        ValueError
            If the specification has no arrow, an empty input, a letter
            repeated within an input or the output, or an output letter in
            no input.
        """
        if not isinstance(composition_rule, CompositionRule):
            raise TypeError(
                f"EinsumWiring: composition_rule must be a CompositionRule, "
                f"got {type(composition_rule).__name__}"
            )
        if "->" not in spec:
            raise ValueError(f"EinsumWiring: spec must contain '->', got {spec!r}")
        lhs, rhs = spec.split("->", maxsplit=1)
        input_specs = tuple(s.strip() for s in lhs.split(","))
        output_spec = rhs.strip()
        # A letter repeated within one input would denote a diagonal or a
        # trace, which no contraction here evaluates, so it is refused when
        # the wiring is built.
        for i, ispec in enumerate(input_specs):
            if not ispec:
                raise ValueError(f"EinsumWiring: input {i} of spec {spec!r} is empty")
            if len(set(ispec)) != len(ispec):
                raise ValueError(
                    f"EinsumWiring: input {i} of spec {spec!r} repeats "
                    f"an axis letter ({ispec!r}); diagonal/trace "
                    f"contractions are not supported by this wiring"
                )
        if len(set(output_spec)) != len(output_spec):
            raise ValueError(
                f"EinsumWiring: output {output_spec!r} of spec "
                f"{spec!r} repeats an axis letter"
            )
        letters = set("".join(input_specs))
        for ch in output_spec:
            if ch not in letters:
                raise ValueError(
                    f"EinsumWiring: output letter {ch!r} not present "
                    f"in any input of spec {spec!r}"
                )
        self._rule = composition_rule
        self._spec = spec
        self._input_specs = input_specs
        self._output_spec = output_spec
        self._rank = (
            len(composition_rule.element_shape)
            if isinstance(composition_rule, Carrier)
            else 0
        )
        self._plans: dict[tuple[tuple[str, int], ...], ContractionPlan] = {}

    @property
    def composition_rule(self) -> CompositionRule:
        """The rule of the contraction.

        Returns
        -------
        CompositionRule
            The rule.
        """
        return self._rule

    @property
    def input_arity(self) -> int:
        """The number of inputs.

        Returns
        -------
        int
            One per input specification.
        """
        return len(self._input_specs)

    @property
    def spec(self) -> str:
        """The specification as written.

        Returns
        -------
        str
            The specification.
        """
        return self._spec

    @property
    def input_specs(self) -> tuple[str, ...]:
        """The letters of each input.

        Returns
        -------
        tuple of str
            One string per input.
        """
        return self._input_specs

    @property
    def output_spec(self) -> str:
        """The letters of the output.

        Returns
        -------
        str
            The output letters.
        """
        return self._output_spec

    @property
    def complexity(self) -> ComplexityClaim:
        """The cost claim of the contraction under its rule.

        Returns
        -------
        ComplexityClaim
            `CONTRACTION_WORK` when the rule establishes every law of
            `quivers.core.elimination.ELIMINATION_LAWS`, else
            `WRITTEN_ORDER_WORK`.
        """
        if self._rule.establishes(*ELIMINATION_LAWS):
            return CONTRACTION_WORK
        return WRITTEN_ORDER_WORK

    def plan(self, extents: Mapping[str, int]) -> ContractionPlan:
        """The plan for given extents, made once per extents.

        Parameters
        ----------
        extents : Mapping of str to int
            The extent of every letter.

        Returns
        -------
        ContractionPlan
            The plan.

        Raises
        ------
        ValueError
            If a letter has no extent of at least zero.
        """
        key = tuple(
            (name, int(extents[name]) if name in extents else -1)
            for name in dict.fromkeys("".join(self._input_specs))
        )
        found = self._plans.get(key)
        if found is None:
            found = plan_contraction(
                self._rule, self._input_specs, self._output_spec, extents
            )
            self._plans[key] = found
        return found

    def _extents(self, tensors: tuple[torch.Tensor, ...]) -> dict[str, int]:
        """The extent of every letter, read from the inputs.

        Parameters
        ----------
        tensors : tuple of torch.Tensor
            The inputs.

        Returns
        -------
        dict of str to int
            The extents.

        Raises
        ------
        TypeError
            If an input is not a tensor.
        ValueError
            If the number of inputs or of an input's axes is wrong, or two
            inputs give one letter different extents.
        """
        if len(tensors) != len(self._input_specs):
            raise ValueError(
                f"EinsumWiring.apply: spec expects "
                f"{len(self._input_specs)} inputs, got {len(tensors)}"
            )
        extents: dict[str, int] = {}
        for index, (tensor, ispec) in enumerate(
            zip(tensors, self._input_specs, strict=True)
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"EinsumWiring.apply: input {index} is a "
                    f"{type(tensor).__name__}, not a tensor"
                )
            if tensor.dim() != len(ispec) + self._rank:
                raise ValueError(
                    f"EinsumWiring: input spec {ispec!r} declares "
                    f"{len(ispec)} axes; tensor has {tensor.dim()} "
                    f"dims (shape {tuple(tensor.shape)})"
                    + (
                        f" and the rule's elements have {self._rank} more"
                        if self._rank
                        else ""
                    )
                )
            for name, extent in zip(ispec, tensor.shape, strict=False):
                if extents.setdefault(name, int(extent)) != int(extent):
                    raise ValueError(
                        f"EinsumWiring.apply: letter {name!r} has extent "
                        f"{extents[name]} and {int(extent)}"
                    )
        return extents

    def apply_with_work(
        self, *tensors: torch.Tensor
    ) -> tuple[torch.Tensor, WorkReport]:
        """Run the contraction and count its work.

        Parameters
        ----------
        *tensors : torch.Tensor
            The inputs, in the order of the specification.

        Returns
        -------
        tuple of torch.Tensor and WorkReport
            The output, its axes in the order of the output letters, and
            the work of execution in elements of the algebra.

        Raises
        ------
        TypeError
            If an input is not a tensor.
        ValueError
            If the inputs do not fit the specification.
        """
        plan = self.plan(self._extents(tensors))
        value, counts = run_contraction(plan, self._rule, tensors, self._rank)
        return value, contraction_report(counts)

    def apply(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Run the contraction.

        Parameters
        ----------
        *tensors : torch.Tensor
            The inputs, in the order of the specification.

        Returns
        -------
        torch.Tensor
            The output, its axes in the order of the output letters.

        Raises
        ------
        TypeError
            If an input is not a tensor.
        ValueError
            If the inputs do not fit the specification.
        """
        return self.apply_with_work(*tensors)[0]


def contraction_report(counts: ContractionCounts) -> WorkReport:
    """The work of a contraction as a work report of its execution.

    Parameters
    ----------
    counts : ContractionCounts
        The counts.

    Returns
    -------
    WorkReport
        Sums, products, elements formed and the largest number held, each
        measured; a contraction divides, pivots, compares and hashes
        nothing, which is measured as zero.
    """
    return WorkReport(
        phase=Phase.EXECUTION,
        scalar_sums=counts.sums,
        scalar_products=counts.products,
        elements_allocated=counts.allocated,
        peak_live_elements=counts.peak,
        measured=CONTRACTION_MEASURES,
    )


def einsum_wiring(composition_rule: CompositionRule, spec: str) -> EinsumWiring:
    """Convenience constructor for `EinsumWiring`.

    Parameters
    ----------
    composition_rule : CompositionRule
        The rule supplying ``tensor_op`` and ``join``.
    spec : str
        An einsum specification such as ``"ij,jk->ik"``.

    Returns
    -------
    EinsumWiring
        The wiring rule for ``spec``.

    Raises
    ------
    TypeError
        If ``composition_rule`` is not a `CompositionRule`.
    ValueError
        If ``spec`` is malformed.
    """
    return EinsumWiring(composition_rule, spec)


def contract(
    rule: WiringRule,
    *tensors: torch.Tensor,
) -> torch.Tensor:
    """Apply a `WiringRule` to a sequence of input tensors.

    Thin wrapper around `WiringRule.apply` for call-site
    readability::

        result = contract(my_rule, arg1, arg2, kernel)

    is an alternative to the method-call form.

    Parameters
    ----------
    rule : WiringRule
        The wiring rule to apply.
    *tensors : torch.Tensor
        The input tensors, in the order the rule expects.

    Returns
    -------
    torch.Tensor
        The contracted tensor.
    """
    return rule.apply(*tensors)


__all__ = [
    "CONTRACTION_MEASURES",
    "CONTRACTION_WORK",
    "EinsumWiring",
    "WRITTEN_ORDER_WORK",
    "WiringRule",
    "contract",
    "contraction_report",
    "einsum_wiring",
]
