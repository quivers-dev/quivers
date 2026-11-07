# Decoders

A decoder uses a vector to define a distribution over structured values. It chooses a
root constructor, generates its fields, and continues through its children. [Sampling
produces a reconstruction; scoring evaluates the probability or density assigned to a
supplied structure](../../guides/structural-compression-decoders-and-losses.md#decoder-blocks). [Reconstruction training](../../tutorials/qvr/12-structural-autoencoders.md#3-invoke-the-checked-loss) increases the score of the original
structure.

`Decoder` is a `torch.nn.Module` that realizes the Kleisli coalgebra
`Vec_D → Kern(T_Σ)`. Given an input vector, it defines a distribution
over terms of a signature through two operations:

- `sample(vec, ctx, sort)` draws a single `Term`.
- `log_prob(term, vec, ctx, sort)` scores an observed term
  under the same distribution.

Corecursion over a signature Σ proceeds as follows:

1. At each sort position, the decoder produces logits over its
   *choice set*, every constructor and binder whose codomain is
   that sort, plus the built-in `BoundVar` whenever the context
   contains at least one in-scope variable of that sort.
2. For the chosen op, the parent vector is split into per-child
   sub-vectors by the per-(sort, arity) `factor` function, and
   the decoder recurses on each child.
3. Data-sorted children are sampled from a closed vocabulary via
   the per-sort `primitive` head; index-sorted children are
   sampled via `binder_select` over the in-scope variables.
4. Binder ops extend Γ before recursing on their scoped arguments, as
   in the encoder.

Construction sets a recursion-depth bound. At that bound, the decoder
restricts the choice set to terminating ops. If a sort has no such op,
the decoder raises an error that identifies the sort.

::: quivers.structural.decoder
