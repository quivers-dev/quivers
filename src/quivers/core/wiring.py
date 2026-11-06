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
Wiring diagrams and operads provide a mathematical account of how compatible
interfaces connect [Vagner et al. (2015)](<https://www.tac.mta.ca/tac/volumes/30/51/30-51abs.html>). Interpreting these contractions as
operations of an operad requires the relevant composition laws, including
the compatibility of tensor products and reductions. A runtime wiring
specification alone does not establish those laws.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.algebras import CompositionRule


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
        """The underlying binary composition rule used as the
        enriching algebra for this n-ary contraction."""

    @property
    @abstractmethod
    def input_arity(self) -> int:
        """Number of input tensors this wiring expects."""

    @abstractmethod
    def apply(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Run the contraction on the given input tensors."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
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

    The contraction algorithm:

    1. Determine the universe of axis letters from inputs and output.
    2. Broadcast each input tensor against the full axis universe
       (singleton dims for axes the input doesn't carry).
    3. Fold all broadcast inputs with ``tensor_op`` (left fold).
    4. Reduce over the contracted axes using ``join``.
    5. Permute to the output's letter order.

    Steps 3-4 correspond to the categorical content; steps 1, 2,
    and 5 are the einsum bookkeeping that places axes in a common
    shape so the binary ``tensor_op`` can fold over the n inputs.
    """

    def __init__(self, composition_rule: CompositionRule, spec: str) -> None:
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
        # Validate that every input is non-empty and uses distinct
        # letters.  Duplicate letters within one input express a
        # trace / diagonal contraction that this implementation
        # does not realise; refusing them at construction time
        # surfaces the limitation honestly rather than producing
        # wrong shapes at apply time.
        for i, ispec in enumerate(input_specs):
            if not ispec:
                raise ValueError(f"EinsumWiring: input {i} of spec {spec!r} is empty")
            if len(set(ispec)) != len(ispec):
                raise ValueError(
                    f"EinsumWiring: input {i} of spec {spec!r} repeats "
                    f"an axis letter ({ispec!r}); diagonal/trace "
                    f"contractions are not supported by this wiring"
                )
        # Output letters must also be distinct (a repeated output
        # letter would require axis-duplication which einsum does
        # not realise).
        if len(set(output_spec)) != len(output_spec):
            raise ValueError(
                f"EinsumWiring: output {output_spec!r} of spec "
                f"{spec!r} repeats an axis letter"
            )
        # Build the axis universe (preserving first-mention order).
        seen: dict[str, None] = {}
        for ispec in input_specs:
            for ch in ispec:
                seen.setdefault(ch, None)
        for ch in output_spec:
            if ch not in seen:
                raise ValueError(
                    f"EinsumWiring: output letter {ch!r} not present "
                    f"in any input of spec {spec!r}"
                )
        self._rule = composition_rule
        self._spec = spec
        self._input_specs = input_specs
        self._output_spec = output_spec
        self._axis_universe: tuple[str, ...] = tuple(seen)
        # Indexed mapping for fast access.
        self._axis_index = {ch: i for i, ch in enumerate(self._axis_universe)}
        # Axes that get reduced (in universe but not in output).
        out_set = set(output_spec)
        self._reduce_axes: tuple[int, ...] = tuple(
            i for i, ch in enumerate(self._axis_universe) if ch not in out_set
        )

    @property
    def composition_rule(self) -> CompositionRule:
        return self._rule

    @property
    def input_arity(self) -> int:
        return len(self._input_specs)

    @property
    def spec(self) -> str:
        return self._spec

    @property
    def input_specs(self) -> tuple[str, ...]:
        return self._input_specs

    @property
    def output_spec(self) -> str:
        return self._output_spec

    def apply(self, *tensors: torch.Tensor) -> torch.Tensor:
        if len(tensors) != len(self._input_specs):
            raise ValueError(
                f"EinsumWiring.apply: spec expects "
                f"{len(self._input_specs)} inputs, got {len(tensors)}"
            )
        # 1. Broadcast each input to the full axis universe.
        broadcast = [
            self._broadcast_to_universe(t, ispec)
            for t, ispec in zip(tensors, self._input_specs)
        ]
        # 2. Fold with tensor_op (left fold).
        acc = broadcast[0]
        for nxt in broadcast[1:]:
            acc = self._rule.tensor_op(acc, nxt)
        # 3. Reduce over contracted axes via join.
        if self._reduce_axes:
            acc = self._rule.join(acc, self._reduce_axes)
        # 4. Permute to output letter order. If letters drop, the
        #    contracted axes are gone after the join; what remains
        #    is the slice of the universe whose letters are in the
        #    output, in universe order. We just need to reorder
        #    those to match the output_spec.
        surviving_letters = [
            ch for ch in self._axis_universe if ch in self._output_spec
        ]
        if surviving_letters != list(self._output_spec):
            permutation = [surviving_letters.index(ch) for ch in self._output_spec]
            acc = acc.permute(*permutation).contiguous()
        return acc

    def _broadcast_to_universe(self, tensor: torch.Tensor, spec: str) -> torch.Tensor:
        """Insert singleton dimensions so ``tensor`` aligns with the
        full ``self._axis_universe`` axis order.

        Each axis letter present in ``spec`` keeps its dim from
        ``tensor`` (in the universe's slot); axes not in ``spec``
        get a singleton.
        """
        if tensor.dim() != len(spec):
            raise ValueError(
                f"EinsumWiring: input spec {spec!r} declares "
                f"{len(spec)} axes; tensor has {tensor.dim()} "
                f"dims (shape {tuple(tensor.shape)})"
            )
        # Permute the input's axes into universe-order (only those
        # the input carries), then unsqueeze singletons for the
        # missing universe axes.
        # First permute: order spec letters by their universe index.
        sorted_letters = sorted(
            range(len(spec)), key=lambda i: self._axis_index[spec[i]]
        )
        permuted = tensor.permute(*sorted_letters)
        # Now permuted's axes are spec-letters in universe order.
        # Build the broadcast shape: full universe length with 1s
        # for letters the input doesn't carry.
        present = sorted(spec, key=lambda ch: self._axis_index[ch])
        target_shape: list[int] = []
        p = 0
        for ch in self._axis_universe:
            if p < len(present) and present[p] == ch:
                target_shape.append(permuted.shape[p])
                p += 1
            else:
                target_shape.append(1)
        return permuted.reshape(target_shape)


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
    "EinsumWiring",
    "WiringRule",
    "contract",
    "einsum_wiring",
]
