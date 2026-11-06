# Structural Compression: Decoders and Losses

This page covers the decoder dual to the [encoder](structural-compression-signatures-and-encoders.md#encoder-blocks) (the [Kleisli arrow](monadic.md#kleisli-category) back from
vectors to terms), loss declarations, and integration with chart parsers and
Bayesian programs. The signature and encoder side lives
in
[Signatures and Encoders](structural-compression-signatures-and-encoders.md).

## Decoder blocks

Decoding grows a structure from the top down. Given a vector, first choose
a constructor, such as a leaf or an application; then produce the values
or child vectors that constructor needs; finally decode recursive
children in the same way. Scoring an observed term follows its known
constructor choices and adds their log probabilities. *Corecursion*
names this repeated expansion of a structure.

The same sampling and scoring distinction appears in [probabilistic programs](programs.md#rsample-and-log_joint). The [generated computation reference](../reference/qvr/generated-computations.md#structural-signatures-and-autoencoders) shows how a decoder becomes a callable entry.

Several terms can have similar codes. The decoder represents this
ambiguity by assigning probabilities to alternative reconstructions;
repeated calls to `sample(vec)` can return different terms. The Kleisli
arrow below has this distribution as its result.


A decoder declares a [Kleisli
arrow](https://ncatlab.org/nlab/show/Kleisli+category) $\mathrm{Vec}_D
\to \mathrm{Kern}(T_\Sigma)$. Per-sort `structure` / `primitive` /
`factor` / `binder_select` heads are scaffolded as learnable neural
networks; the corecursion (op choice, factor split, recursive
descent, `BoundVar` fallback to in-scope variables) is supplied by
the framework.

<!-- compile: false -->
```qvr
decoder D : LF [depth=8]
    body |-> recursive
```

The two runtime operations:

- `sample(vec) -> Term`: draw a single term.
- `log_prob(term, vec) -> Tensor`: score an observed term.

**Depth-bounded termination.** At the depth limit the choice set is
restricted to ops whose every child sort is `data` or `index`
(never `object`); if no such op exists at a sort, the decoder
raises with a precise diagnostic.

## Loss declarations

A reconstruction loss asks the decoder to give the input term high
probability after encoding it. If its probability is `0.2`, the negative
log likelihood is about `1.609`; raising it to `0.8` lowers the loss
to about `0.223`. At probability `0.8`, repeated decoding still assigns
`0.2` probability to other terms. A loss weight sets the reconstruction
term’s contribution relative to the other training criteria.


Losses are first-class declarations attachable at any site in the
training graph:

<!-- compile: false -->
```qvr
loss reconstruction [weight=1.0, on="encoder C"]
    -D(C(input)).log_prob(input)

loss type_coherence [weight=0.1, on="rule combine in Parse"]
    cross_entropy(parent_type_dist, combined_children_type_dist)

loss completed [weight=0.01, on="chart of Parse"]
    -chart.goal_weight()

loss nli [weight=1.0, on="program nli_predict"]
    bce_with_logits(predicted_logit, true_label)
```

Attachment kinds: `global` (no `on` clause), `program <name>`,
`deduction <name>`, `encoder <name>`, `decoder <name>`,
`rule <name> in <D>` (fires on every application of that rule
during chart construction), and `chart of <D>` (fires once on the
completed chart).

`prog.losses.evaluate(env)` sums every loss;
`evaluate_on(kind, target, env)` filters by attachment. Inside a
[deduction](deduction.md#the-framework), the runtime accumulates rule-attached and chart-attached
losses into `ChartView.attached_loss` for the training driver to
read.

## Deduction integration

A chart item can carry both a symbolic identity and a learned vector.
The symbolic part determines which rules apply and how derivation
weights combine; the encoder computes features of that item for
learnable scoring or prediction. For instance, an encoder can turn a
partial parse into a vector used to score its next rule application; the
chart then aggregates the resulting derivation weights with its semiring.


A `deduction` block may attach an item signature and an encoder;
the chart's `embedding(item)` operation then returns a
differentiable vector computed by the attached encoder's
algebra-homomorphism recursion over the chart-item term.

<!-- compile: false -->
```qvr
signature ChartItem
    sorts
        Item : object [dim=128]
        Idx : data [dim=8]
        Type : object [dim=32]
    constructors
        span : Idx, Idx, Type -> Item

deduction Parse : Sentence -> Tree [semiring=LogProb, signature=ChartItem, encoder=InsideC]
    atoms S, NP, span
    rule combine : span(i, k, Fwd(X, Y)), span(k, j, X) |- span(i, j, Y)
```

The attention-weighted aggregation that distinguishes the
vector-inside-outside parser from semiring parsing lives entirely
inside the encoder's per-op function, outside the chart's role, so
the semiring abstraction is not broken.

## Bayesian integration

A decoder is a Kleisli arrow into a structured type: a distribution
over terms. This gives the same sampling/scoring interface as other distributions,
though actual inference and target support depend on the operations
implemented for structured values. The intended modeling patterns are:

- **Structured latents.** `latent_term <- D(prior_vec)` inside a
  program body draws a random term from the decoder.
- **Observations of structured data.**
  `observe known_term <- D(some_vec)` scores under the decoder's
  log-prob.
- **[Marginalization](dsl-programs-and-lets.md#scoped-marginalize) over structured latents.**
  `marginalize t : Term <- D(v)` with an indented body
  `observe y <- ...`   expresses a sum over terms when a finite enumeration is available.
  This requires an execution route that enumerates the finite term set;
  a route exposing only `sample` and `log_prob` needs a sampling approximation.
- **Variational decoders.**
  `program q : Sentence -> Term`
  defines a variational decoder over LFs given a sentence.

## Stdlib decoders

The depth bound is part of the represented model. At the limit, only
constructors that can finish without another object-valued child are
eligible. Increasing the bound admits more structures and changes the
support. For instance, a term requiring nine recursive expansions is
excluded by a bound of eight and can become eligible with a larger bound.


The same `quivers.structural.shapes` registry that ships the
canonical encoders also ships matching decoders:

- `ar_decoder(sig, dim, vocab, depth)`: autoregressive decoder for
  sequence signatures; `depth` caps the unfolded structure size.
- `tree_decoder(sig, dim, leaf_vocab, label_vocab, depth)`:
  top-down structural decoder for tree signatures.

Both are realized on top of the generic
[`Decoder`](../api/structural/decoder.md) runtime, the same Kleisli
coalgebra pattern, no special-casing.

## See also

- [Signatures and Encoders](structural-compression-signatures-and-encoders.md):
  the encoder side of the autoencoder pair.
- [Weighted Deduction Systems](deduction.md): the chart representation to
  which decoders attach.
- [Compositional Effects](effects.md): the algebraic-effects
  framework over which the decoder's stochasticity is realized.
- [Structural autoencoders tutorial](../tutorials/qvr/12-structural-autoencoders.md):
  invoking a reconstruction loss through `Program.run`, preserving autograd,
  and inspecting registered checkpoint parameters.
- [Generated computation graphs](../reference/qvr/generated-computations.md):
  the typed `Compute` attachment and target-capability contract.
