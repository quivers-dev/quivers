# Decoders

A [decoder](../../guides/structural-compression-decoders-and-losses.md#decoder-blocks) maps a vector to
a distribution over terms of a signature. `Decoder` is a `torch.nn.Module` that implements a
[Kleisli arrow](https://ncatlab.org/nlab/show/Kleisli+category) $\mathrm{Vec}_D \to \mathcal{G}(T_\Sigma)$
of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) $\mathcal{G}$, from vectors of
dimension $D$ to distributions over the terms $T_\Sigma$ of the signature $\Sigma$. It chooses a root
constructor, generates its fields, and continues through the children. It has two operations, and
[reconstruction training](../../tutorials/qvr/12-structural-autoencoders.md#3-invoke-the-checked-loss)
raises the score of the original term:

- `sample(vec, ctx, sort)` draws a single `Term`;
- `log_prob(term, vec, ctx, sort)` scores an observed term
  under the same distribution.

Corecursion over a signature Σ proceeds as follows:

1. At each sort position, the decoder produces logits over its
   *choice set*, every constructor and binder whose codomain is
   that sort, plus the built-in `BoundVar` whenever the context
   contains at least one in-scope variable of that sort.
2. For the chosen operation, the decoder's `factor` head for that sort and arity splits the
   parent vector into one vector per child, and the decoder recurses on each child.
3. Children of a data sort are sampled from a closed vocabulary by that sort's `primitive` head,
   and children of an index sort by `binder_select` over the variables in scope.
4. Binder ops extend Γ before recursing on their scoped arguments, as
   in the encoder.

Construction sets a recursion-depth bound. At that bound, the decoder
restricts the choice set to terminating ops. If a sort has no such op,
the decoder raises an error that identifies the sort.

::: quivers.structural.decoder
