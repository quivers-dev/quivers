"""Algebraic effects + handlers.

Following Plotkin & Power and Bauer & Pretnar, exposes a free-monad-
over-signature construction so any user-defined effect signature
induces a `Monad` instance automatically. Handlers translate
a free-monad computation into a target monad's effects.

The free-monad carrier ``Free_Σ^{≤d}(A)`` is realised as a flat
`FinSet` whose flat-index layout is:

- Indices ``[0, |A|)`` represent leaf nodes ``leaf(a)``.
- For each operation ``op_i`` (in declared order), a contiguous block
  of ``|op_i.parameter| · |Free^{d-1}(A)|^{|op_i.result|}`` indices
  represents operation nodes ``op_i(p, k)`` where ``p`` is a
  parameter and ``k`` is a deterministic continuation
  ``op_i.result → Free^{d-1}(A)``.

The flat-FinSet representation avoids the `CoproductSet`
auto-flattening collision that would otherwise dissolve the
recursive leaf-vs-operation distinction when ``A`` is itself a
coproduct. Structural decomposition is recovered from
``signature + depth`` rather than from the carrier object.

Handler interpretation walks the flat-index decomposition in
post-order: leaves are routed through ``return_clause`` and
operation nodes through ``operation_clauses``.

References
----------
- Plotkin, G. and Power, J. (2003). *Algebraic operations and generic
  effects*. Applied Categorical Structures, 11(1), 69–94.
  doi:10.1023/A:1023064908962.
- Bauer, A. and Pretnar, M. (2015). *Programming with algebraic
  effects and handlers*. Journal of Logical and Algebraic Methods in
  Programming, 84(1), 108–123. doi:10.1016/j.jlamp.2014.02.001.
"""

from __future__ import annotations

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.core.morphisms import Morphism, observed
from quivers.core.morphisms import identity as id_morph
from quivers.core.objects import FinSet, ProductSet, SetObject
from quivers.core.algebras import PRODUCT_FUZZY
from quivers.monadic.typeclasses import Monad


class Operation(dx.Model):
    """One operation in an effect signature.

    Attributes
    ----------
    name : str
        Operation name, used as the key in handler clauses.
    parameter : SetObject
        Operation parameter type.
    result : SetObject
        Operation result type given to the continuation.
    """

    name: str
    parameter: SetObject
    result: SetObject


class EffectSignature(dx.Model):
    """A signature of effect operations.

    Realisable both as a Python value (a tuple of operations) and as
    a `panproto.Theory` whose sorts are `SetObject` and
    whose operations are the listed `Operation` instances.

    Attributes
    ----------
    name : str
        Signature name; appears in panproto theory naming.
    operations : tuple of Operation
        The operations comprising the signature.
    """

    name: str
    operations: tuple[Operation, ...] = ()

    def to_theory(self) -> object:
        """Realise this signature as a panproto Theory.

        Returns
        -------
        panproto.Theory
            The theory whose single sort is ``Carrier``, whose
            operations are one per `operations` entry with
            input ``parameter * (result -> Carrier)`` and output
            ``Carrier``, and whose equations are the three monad
            laws (left/right unit, associativity).
        """
        import panproto

        op_specs: list[dict] = []
        for op in self.operations:
            op_specs.append(
                {
                    "name": op.name,
                    "inputs": [
                        ("parameter", "Carrier"),
                        ("continuation", "Carrier"),
                    ],
                    "output": "Carrier",
                }
            )
        spec = {
            "id": f"quivers.effect.{self.name}",
            "description": f"Free monad over the {self.name} signature.",
            "theories": [
                {
                    "theory": f"Th{self.name}",
                    "sorts": [{"name": "Carrier"}],
                    "ops": op_specs,
                }
            ],
        }
        return panproto.create_theory(spec)


# ---------------------------------------------------------------------------
# Bounded-depth signature-tree carrier (flat-FinSet representation)
# ---------------------------------------------------------------------------


def _carrier_size(signature: EffectSignature, leaf_size: int, depth: int) -> int:
    """Total cardinality of ``Free_Σ^{≤depth}(A)`` for ``|A| = leaf_size``."""
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if depth == 0 or not signature.operations:
        return leaf_size
    sub_size = _carrier_size(signature, leaf_size, depth - 1)
    total = leaf_size
    for op in signature.operations:
        total += op.parameter.size * (sub_size**op.result.size)
    return total


def _tree_carrier(
    signature: EffectSignature, leaf_type: SetObject, depth: int
) -> SetObject:
    """The bounded-depth free-monad carrier as a flat `FinSet`.

    The carrier's flat-index layout is structural: leaves first, then
    one contiguous block per operation. The decomposition is
    reconstructed by `_decompose_carrier_index`.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if depth == 0 or not signature.operations:
        return leaf_type
    size = _carrier_size(signature, leaf_type.size, depth)
    leaf_name = leaf_type.name if hasattr(leaf_type, "name") else str(leaf_type)
    return FinSet(name=f"Free<{signature.name}>^{depth}({leaf_name})", cardinality=size)


def _decompose_carrier_index(
    signature: EffectSignature,
    leaf_size: int,
    depth: int,
    flat: int,
) -> tuple[str, int, int, int]:
    """Decompose a flat carrier index into structural components.

    Returns ``(kind, op_index, p_idx, cont_flat)`` where:

    - ``kind`` is ``"leaf"`` or ``"op"``.
    - For leaves: ``op_index = -1``, ``p_idx`` holds the leaf-flat
      index ``flat`` itself, ``cont_flat = 0``.
    - For ops: ``op_index`` is the operation's index in
      ``signature.operations``; ``p_idx`` is the parameter's flat
      index; ``cont_flat`` is the continuation's flat function-space
      index in ``Free^{d-1}(A)^{op.result.size}``.
    """
    if depth == 0 or not signature.operations:
        if not 0 <= flat < leaf_size:
            raise ValueError(f"flat index {flat} out of carrier range [0, {leaf_size})")
        return ("leaf", -1, flat, 0)
    if flat < leaf_size:
        return ("leaf", -1, flat, 0)
    rest = flat - leaf_size
    sub_size = _carrier_size(signature, leaf_size, depth - 1)
    for i, op in enumerate(signature.operations):
        cont_card = sub_size**op.result.size
        block = op.parameter.size * cont_card
        if rest < block:
            p_idx = rest // cont_card
            cont_flat = rest % cont_card
            return ("op", i, p_idx, cont_flat)
        rest -= block
    raise ValueError(
        f"flat index {flat} out of carrier range; cardinality "
        f"= {_carrier_size(signature, leaf_size, depth)}"
    )


def _compose_carrier_index(
    signature: EffectSignature,
    leaf_size: int,
    depth: int,
    kind: str,
    op_index: int,
    p_idx: int,
    cont_flat: int,
) -> int:
    """Inverse of `_decompose_carrier_index`."""
    if kind == "leaf":
        return p_idx
    if depth == 0 or not signature.operations:
        raise ValueError("cannot compose an op-kind index at depth 0")
    sub_size = _carrier_size(signature, leaf_size, depth - 1)
    offset = leaf_size
    for i, op in enumerate(signature.operations):
        cont_card = sub_size**op.result.size
        block = op.parameter.size * cont_card
        if i == op_index:
            return offset + p_idx * cont_card + cont_flat
        offset += block
    raise ValueError(f"op_index {op_index} out of range")


def _decode_continuation(flat: int, sub_size: int, result_size: int) -> tuple[int, ...]:
    """Decode a function-space flat index into the output tuple.

    The encoding is row-major: ``flat = ∑_i outputs[i] · sub_size^(result_size-1-i)``.
    """
    outs = [0] * result_size
    rem = flat
    for i in range(result_size - 1, -1, -1):
        outs[i] = rem % sub_size
        rem //= sub_size
    return tuple(outs)


def _encode_continuation(outputs: tuple[int, ...], sub_size: int) -> int:
    """Inverse of `_decode_continuation`."""
    flat = 0
    for v in outputs:
        flat = flat * sub_size + v
    return flat


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class Handler(ValueModel):
    """Interpretation of a free-monad computation in a target monad.

    A handler supplies a clause for each operation in its signature
    plus a return-clause that lifts plain values into the target
    monad. `run` folds a bounded-depth signature tree by
    interpreting each leaf through `return_clause` and each
    operation node through `operation_clauses`.

    Operation-clause shape: for an operation ``op`` whose result feeds
    a continuation ``k : op.result → target(B)``, the clause has
    signature ``op.parameter × [op.result → target(B)] → target(B)``.
    The implementation translates each free-monad operation node's
    continuation (a function-space flat index into the inner-tree
    carrier) into the corresponding target-side function-space
    continuation by recursive descent, then routes through the clause.

    Handlers compare by value over ``signature`` and by identity over
    ``target``, ``return_clause``, and ``operation_clauses``: two
    handlers are equal exactly when they share a signature and hold
    the same target monad, return-clause morphism, and clause mapping
    objects.

    Attributes
    ----------
    signature : EffectSignature
        The effect signature this handler interprets.
    target : Monad
        The target monad in which the operations are realised.
    return_clause : Morphism
        ``A → target(A)``: lifts plain values into the target monad.
    operation_clauses : dict[str, Morphism]
        For each `signature.operations` entry, a morphism
        ``parameter × [result → target(B)] → target(B)`` (encoded
        with the function-space `FinSet`).
    """

    signature: EffectSignature
    target: object = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    return_clause: object = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    operation_clauses: object = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def as_theory_morphism(self) -> object:
        """Realise this handler as a panproto theory morphism.

        The morphism's domain is ``signature.to_theory()``; its
        codomain is the target monad's panproto theory.
        """
        import panproto

        spec = {
            "id": f"quivers.handler.{self.signature.name}",
            "description": (
                f"Handler from {self.signature.name} into the registered target monad."
            ),
            "theories": [
                {
                    "theory": f"Th{self.signature.name}Target",
                    "sorts": [{"name": "Carrier"}],
                    "ops": [
                        {
                            "name": "return_clause",
                            "inputs": [("value", "Carrier")],
                            "output": "Carrier",
                        },
                        *[
                            {
                                "name": op.name,
                                "inputs": [
                                    ("parameter", "Carrier"),
                                    ("continuation", "Carrier"),
                                ],
                                "output": "Carrier",
                            }
                            for op in self.signature.operations
                        ],
                    ],
                }
            ],
        }
        return panproto.create_theory(spec)

    def run(self, A: SetObject, depth: int = 3) -> Morphism:
        """Apply this handler to a free-monad computation up to ``depth``.

        Returns a morphism ``Free_Σ^{≤depth}(A) → target(A)``.

        The construction proceeds by induction on the tree depth:
        leaves are routed through `return_clause`; each
        operation node is decomposed into its parameter and
        continuation components, the continuation is recursively
        interpreted into the target monad, and the result is fed
        through the corresponding operation-clause morphism.
        """
        target_A = self.target.fmap_obj(A)
        carrier = _tree_carrier(self.signature, A, depth)
        leaf_size = A.size
        if depth == 0 or not self.signature.operations:
            # The carrier coincides with A; the result is the
            # return-clause itself.
            return self.return_clause
        # Recursive case: build the per-summand interpretations and
        # combine them through a single tensor.
        sub_morph = self.run(A, depth - 1)
        sub_size = _carrier_size(self.signature, leaf_size, depth - 1)
        target_A_size = target_A.size
        data = torch.full((carrier.size, target_A_size), PRODUCT_FUZZY.zero)
        # Leaf branch.
        return_t = self.return_clause.tensor.reshape(leaf_size, target_A_size)
        for a_flat in range(leaf_size):
            data[a_flat] = return_t[a_flat]
        # Operation branches.
        offset = leaf_size
        for op in self.signature.operations:
            clause = self.operation_clauses[op.name]
            # Clause has source = op.parameter × [op.result → target(A)]
            # and codomain = target(A). Encode the target-side function-
            # space cardinality:
            target_cont_card = target_A_size**op.result.size
            target_cont_fs = FinSet(
                name=f"_target_cont_{op.name}",
                cardinality=target_cont_card,
            )
            clause_src = ProductSet(components=(op.parameter, target_cont_fs))
            if clause.domain != clause_src or clause.codomain != target_A:
                raise TypeError(
                    f"handler clause {op.name!r} has shape "
                    f"{clause.domain!r} → {clause.codomain!r}; expected "
                    f"{clause_src!r} → {target_A!r}"
                )
            clause_t = clause.tensor.reshape(
                op.parameter.size, target_cont_card, target_A_size
            )
            sub_t = sub_morph.tensor.reshape(sub_size, target_A_size)
            cont_card = sub_size**op.result.size
            block = op.parameter.size * cont_card
            for p_idx in range(op.parameter.size):
                for cont_flat in range(cont_card):
                    outer_flat = offset + p_idx * cont_card + cont_flat
                    # Decode the continuation as a deterministic function
                    # op.result → sub-carrier; translate each output
                    # through sub_morph to obtain a distribution over
                    # target(A); enumerate the joint cartesian product
                    # across result slots and translate back to the
                    # target_cont function-space encoding.
                    cont_outs = _decode_continuation(
                        cont_flat, sub_size, op.result.size
                    )
                    per_slot: list[list[tuple[int, float]]] = []
                    for s in cont_outs:
                        nonzero = [
                            (j, float(sub_t[s, j].item()))
                            for j in range(target_A_size)
                            if sub_t[s, j].item() > 0
                        ]
                        per_slot.append(nonzero)
                    if any(not slot for slot in per_slot):
                        continue
                    # Enumerate cartesian product across slots.
                    stack: list[tuple[int, float, list[int]]] = [(0, 1.0, [])]
                    while stack:
                        slot_idx, w, choices = stack.pop()
                        if slot_idx == op.result.size:
                            tgt_cont_flat = _encode_continuation(
                                tuple(choices), target_A_size
                            )
                            # Route through the clause.
                            row = clause_t[p_idx, tgt_cont_flat]
                            for b in range(target_A_size):
                                bw = w * float(row[b].item())
                                if bw > 0:
                                    cur = data[outer_flat, b].item()
                                    # Algebra join: noisy-OR-like
                                    # aggregation across cartesian
                                    # branches.
                                    data[outer_flat, b] = max(cur, bw)
                            continue
                        for j, jw in per_slot[slot_idx]:
                            stack.append((slot_idx + 1, w * jw, choices + [j]))
            offset += block
        return observed(carrier, target_A, data)


# ---------------------------------------------------------------------------
# FreeMonad
# ---------------------------------------------------------------------------


class FreeMonad(dx.Model):
    """The free monad over an effect signature with a depth bound.

    ``FreeMonad(sig, depth)`` carries the bounded-depth signature-tree
    construction. Its typeclass instance methods are concrete
    V-relation morphisms over the flat-FinSet carrier.

    Attributes
    ----------
    signature : EffectSignature
        The signature whose free monad this is.
    depth : int
        The depth bound for the operation-tree carrier.
    """

    signature: EffectSignature
    depth: int = 3

    def _carrier(self, A: SetObject) -> SetObject:
        return _tree_carrier(self.signature, A, self.depth)

    def fmap_obj(self, A: SetObject) -> SetObject:
        return self._carrier(A)

    def pure(self, A: SetObject) -> Morphism:
        """``pure : A → Free^d(A)`` is the leaf injection."""
        carrier = self._carrier(A)
        if carrier == A:
            return id_morph(A)
        data = torch.full((A.size, carrier.size), PRODUCT_FUZZY.zero)
        for a in range(A.size):
            data[a, a] = PRODUCT_FUZZY.unit
        # Reshape to (*A.shape, *carrier.shape).
        data = data.reshape(*A.shape, *carrier.shape)
        return observed(A, carrier, data)

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        """Lift ``f : A → B`` to ``Free^d(f) : Free^d(A) → Free^d(B)``.

        Acts as ``f`` on the leaf summand and recursively as
        ``Free^{d-1}(f)`` on each operation's continuation slots.
        Each operation node's continuation function-space is re-encoded
        through the recursive ``fmap``'s deterministic image.
        """
        carrier_A = self._carrier(A)
        carrier_B = self._carrier(B)
        if carrier_A == A and carrier_B == B:
            return f
        sub_A_size = _carrier_size(self.signature, A.size, self.depth - 1)
        sub_B_size = _carrier_size(self.signature, B.size, self.depth - 1)
        sub_recur = FreeMonad(signature=self.signature, depth=self.depth - 1).fmap(
            A, B, f
        )
        sub_t = sub_recur.tensor.reshape(sub_A_size, sub_B_size)
        data = torch.full((carrier_A.size, carrier_B.size), PRODUCT_FUZZY.zero)
        # Leaf branch: route through f.
        f_t = f.tensor.reshape(A.size, B.size)
        for a in range(A.size):
            for b in range(B.size):
                v = float(f_t[a, b].item())
                if v > 0:
                    data[a, b] = v
        # Op branches.
        offset_A = A.size
        offset_B = B.size
        for op in self.signature.operations:
            cont_A_card = sub_A_size**op.result.size
            cont_B_card = sub_B_size**op.result.size
            block_A = op.parameter.size * cont_A_card
            block_B = op.parameter.size * cont_B_card
            for p_idx in range(op.parameter.size):
                for cont_A_flat in range(cont_A_card):
                    outs_A = _decode_continuation(
                        cont_A_flat, sub_A_size, op.result.size
                    )
                    # Build per-slot distributions over sub_B.
                    per_slot: list[list[tuple[int, float]]] = []
                    for s in outs_A:
                        nonzero = [
                            (j, float(sub_t[s, j].item()))
                            for j in range(sub_B_size)
                            if sub_t[s, j].item() > 0
                        ]
                        per_slot.append(nonzero)
                    if any(not slot for slot in per_slot):
                        continue
                    src = offset_A + p_idx * cont_A_card + cont_A_flat
                    stack: list[tuple[int, float, list[int]]] = [(0, 1.0, [])]
                    while stack:
                        slot_idx, w, choices = stack.pop()
                        if slot_idx == op.result.size:
                            cont_B_flat = _encode_continuation(
                                tuple(choices), sub_B_size
                            )
                            tgt = offset_B + p_idx * cont_B_card + cont_B_flat
                            cur = data[src, tgt].item()
                            data[src, tgt] = max(cur, w)
                            continue
                        for j, jw in per_slot[slot_idx]:
                            stack.append((slot_idx + 1, w * jw, choices + [j]))
            offset_A += block_A
            offset_B += block_B
        return observed(carrier_A, carrier_B, data)

    def join(self, A: SetObject) -> Morphism:
        """``join : Free^d(Free^d(A)) → Free^d(A)`` via tree splicing.

        Splices outer trees: an outer-leaf carrying an inner tree
        produces that inner tree directly; an outer-operation node
        ``op(p, k)`` produces ``op(p, k')`` where each continuation
        slot is the recursive join of the corresponding outer
        subtree. Truncation at depth ``d`` is enforced by recursive
        descent: outer ops at depth ``d`` produce inner ops whose
        sub-trees come from depth-``d-1`` recursion.
        """
        carrier_A = self._carrier(A)
        carrier_FA = _tree_carrier(self.signature, carrier_A, self.depth)
        FA_size = carrier_FA.size
        A_carrier_size = carrier_A.size
        data = torch.full((FA_size, A_carrier_size), PRODUCT_FUZZY.zero)

        def splice(outer_flat: int, depth_remaining: int) -> dict[int, float]:
            """Compute join image of an outer index at the given depth."""
            if depth_remaining == 0 or not self.signature.operations:
                # The outer tree is a single leaf in carrier_A; identity.
                return {outer_flat: 1.0}
            # Outer-leaf summand: indices [0, A_carrier_size).
            if outer_flat < A_carrier_size:
                return {outer_flat: 1.0}
            # Outer-op summand.
            sub_outer_size = _carrier_size(
                self.signature, A_carrier_size, depth_remaining - 1
            )
            rest = outer_flat - A_carrier_size
            offset = 0
            for i, op in enumerate(self.signature.operations):
                cont_card = sub_outer_size**op.result.size
                block = op.parameter.size * cont_card
                if rest < block:
                    p_idx = rest // cont_card
                    cont_flat = rest % cont_card
                    cont_outs = _decode_continuation(
                        cont_flat, sub_outer_size, op.result.size
                    )
                    # Recursively join each slot.
                    per_slot: list[dict[int, float]] = [
                        splice(co, depth_remaining - 1) for co in cont_outs
                    ]
                    # Map each slot's image (in Free^{depth_remaining - 1}(A))
                    # back into the inner-carrier's op-summand cont
                    # function-space. The inner carrier is
                    # Free^{self.depth}(A); its sub-carrier at depth-1
                    # corresponds to Free^{self.depth - 1}(A). But here
                    # the recursive `splice` produces values in
                    # `_tree_carrier(sig, A, depth_remaining - 1)` —
                    # which has size sub_inner_size.
                    sub_inner_size = _carrier_size(
                        self.signature, A.size, depth_remaining - 1
                    )
                    # The image's inner-cont function-space cardinality
                    # is sub_inner_size ** op.result.size.
                    # Build the offset for this op in carrier_A's
                    # op-summand block (at depth = self.depth).
                    inner_offset_at_depth = _carrier_op_offset(
                        self.signature, A.size, self.depth, i
                    )
                    inner_cont_card = sub_inner_size**op.result.size
                    out: dict[int, float] = {}
                    stack: list[tuple[int, float, list[int]]] = [(0, 1.0, [])]
                    while stack:
                        slot_idx, w, choices = stack.pop()
                        if slot_idx == op.result.size:
                            inner_cont_flat = _encode_continuation(
                                tuple(choices), sub_inner_size
                            )
                            target_idx = (
                                inner_offset_at_depth
                                + p_idx * inner_cont_card
                                + inner_cont_flat
                            )
                            if target_idx < A_carrier_size:
                                out[target_idx] = max(out.get(target_idx, 0.0), w)
                            continue
                        for choice, choice_w in per_slot[slot_idx].items():
                            if choice_w > 0 and choice < sub_inner_size:
                                stack.append(
                                    (slot_idx + 1, w * choice_w, choices + [choice])
                                )
                    return out
                rest -= block
                offset += block
            return {}

        for outer_flat in range(FA_size):
            dist = splice(outer_flat, self.depth)
            for inner_idx, w in dist.items():
                data[outer_flat, inner_idx] = w
        return observed(carrier_FA, carrier_A, data)

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        """``bind(m, k) = join_B ∘ Free^d(k)(m)``."""
        return self.fmap(A, self._carrier(B), k) >> self.join(B)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        """``apply : Free^d([A → B]) ⊗ Free^d(A) → Free^d(B)``.

        Derived from `lift_a2` and the function-space evaluator.
        """
        from quivers.monadic.instances import _evaluation_morphism, _function_space

        fn_space = _function_space(A, B)
        return self.lift_a2(fn_space, A, B, _evaluation_morphism(A, B))

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        """``lift_a2(f) : Free^d(A) ⊗ Free^d(B) → Free^d(C)``.

        Recursive definition (the standard free-monad applicative):

        - ``liftA2(f)(leaf(a), leaf(b)) = leaf(f(a, b))``.
        - ``liftA2(f)(op(p, k), n) = op(p, λr. liftA2(f)(k(r), n))``.
        - ``liftA2(f)(leaf(a), op(p, k)) = op(p, λr. liftA2(f)(leaf(a), k(r)))``.

        The recursion descends one depth at a time on the side carrying
        the operation, while the other side stays at its current
        depth. Sub-results are produced in
        :math:`\\mathrm{Free}^{\\max(d_a, d_b) - 1}(C)` and re-encoded
        into the parent's op-summand continuation slots through the
        appropriate sub-carrier function-space cardinality. Truncation
        is enforced uniformly when the output depth would exceed the
        bound ``d``: such sub-results are dropped, preserving the
        bounded-depth contract.
        """
        carrier_A = self._carrier(A)
        carrier_B = self._carrier(B)
        carrier_C = self._carrier(C)
        f_t = f.tensor.reshape(A.size, B.size, C.size)
        data = torch.full(
            (carrier_A.size, carrier_B.size, carrier_C.size),
            PRODUCT_FUZZY.zero,
        )
        sig = self.signature

        def recur(
            da: int, db: int, dc: int, a_flat: int, b_flat: int
        ) -> dict[int, float]:
            """Distribution over flat ``Free^dc(C)`` indices.

            ``da`` is the depth at which ``a_flat`` lives in
            ``Free^da(A)``; ``db`` likewise for ``b_flat``; ``dc`` is
            the target output depth (``dc = max(da, db)`` keeps the
            recursion well-typed against the bounded-depth target).
            """
            a_kind, a_op, a_pay, a_cont = _decompose_carrier_index(
                sig, A.size, da, a_flat
            )
            b_kind, b_op, b_pay, b_cont = _decompose_carrier_index(
                sig, B.size, db, b_flat
            )
            if a_kind == "leaf" and b_kind == "leaf":
                # leaf(a) ★ leaf(b) = leaf(f(a, b))
                out: dict[int, float] = {}
                for c in range(C.size):
                    v = float(f_t[a_pay, b_pay, c].item())
                    if v > 0:
                        out[c] = v
                return out
            if a_kind == "op":
                # op(p_a, k_a) ★ n = op(p_a, λr. recur(k_a(r), n))
                if dc < 1:
                    return {}
                op = sig.operations[a_op]
                sub_a_size = _carrier_size(sig, A.size, da - 1)
                cont_outs = _decode_continuation(a_cont, sub_a_size, op.result.size)
                per_slot: list[dict[int, float]] = [
                    recur(da - 1, db, dc - 1, k_r, b_flat) for k_r in cont_outs
                ]
                if any(not slot for slot in per_slot):
                    return {}
                sub_c_size = _carrier_size(sig, C.size, dc - 1)
                offset_c = _carrier_op_offset(sig, C.size, dc, a_op)
                cont_card_c = sub_c_size**op.result.size
                out_dist: dict[int, float] = {}
                stack: list[tuple[int, float, list[int]]] = [(0, 1.0, [])]
                while stack:
                    slot_idx, w, choices = stack.pop()
                    if slot_idx == op.result.size:
                        cont_c_flat = _encode_continuation(tuple(choices), sub_c_size)
                        c_idx = offset_c + a_pay * cont_card_c + cont_c_flat
                        out_dist[c_idx] = max(out_dist.get(c_idx, 0.0), w)
                        continue
                    for cc, cw in per_slot[slot_idx].items():
                        if cc < sub_c_size:
                            stack.append((slot_idx + 1, w * cw, choices + [cc]))
                return out_dist
            # a is leaf, b is op
            if dc < 1:
                return {}
            op = sig.operations[b_op]
            sub_b_size = _carrier_size(sig, B.size, db - 1)
            cont_outs = _decode_continuation(b_cont, sub_b_size, op.result.size)
            per_slot = [recur(da, db - 1, dc - 1, a_flat, k_r) for k_r in cont_outs]
            if any(not slot for slot in per_slot):
                return {}
            sub_c_size = _carrier_size(sig, C.size, dc - 1)
            offset_c = _carrier_op_offset(sig, C.size, dc, b_op)
            cont_card_c = sub_c_size**op.result.size
            out_dist = {}
            stack = [(0, 1.0, [])]
            while stack:
                slot_idx, w, choices = stack.pop()
                if slot_idx == op.result.size:
                    cont_c_flat = _encode_continuation(tuple(choices), sub_c_size)
                    c_idx = offset_c + b_pay * cont_card_c + cont_c_flat
                    out_dist[c_idx] = max(out_dist.get(c_idx, 0.0), w)
                    continue
                for cc, cw in per_slot[slot_idx].items():
                    if cc < sub_c_size:
                        stack.append((slot_idx + 1, w * cw, choices + [cc]))
            return out_dist

        for a_flat in range(carrier_A.size):
            for b_flat in range(carrier_B.size):
                dist = recur(self.depth, self.depth, self.depth, a_flat, b_flat)
                for c_idx, w in dist.items():
                    if c_idx < carrier_C.size:
                        data[a_flat, b_flat, c_idx] = w
        source = ProductSet(components=(carrier_A, carrier_B))
        data = data.reshape(*source.shape, *carrier_C.shape)
        return observed(source, carrier_C, data)


Monad.register(FreeMonad)


def _carrier_op_offset(
    signature: EffectSignature, leaf_size: int, depth: int, op_index: int
) -> int:
    """Flat offset of the ``op_index``-th operation's summand in the carrier."""
    if depth == 0 or not signature.operations:
        raise ValueError("no op summands at depth 0")
    sub_size = _carrier_size(signature, leaf_size, depth - 1)
    offset = leaf_size
    for i, op in enumerate(signature.operations):
        if i == op_index:
            return offset
        cont_card = sub_size**op.result.size
        offset += op.parameter.size * cont_card
    raise ValueError(f"op_index {op_index} out of range")


__all__ = [
    "Operation",
    "EffectSignature",
    "Handler",
    "FreeMonad",
]
