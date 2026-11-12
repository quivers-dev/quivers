# Tree and Graph Autoencoders: Decoders and Losses

A decoder is the second half of the autoencoders of
[Signatures and Encoders](structural-compression-signatures-and-encoders.md): given a
vector, it defines a probability distribution over the terms of a signature. It is
thus a [Markov kernel](https://en.wikipedia.org/wiki/Markov_kernel)
$\mathrm{Vec}_D \to \mathcal G(T_\Sigma)$ from vectors to terms, an arrow of the
[Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the
[Giry monad](stochastic.md#the-giry-monad-and-finstoch) $\mathcal G$.

## Decoder blocks

A decoder grows a term from the top down. At a position of sort $s$ it scores the
choices available there: every constructor and binder that builds an $s$, and, when
a variable of sort $s$ is in scope, `BoundVar`. For the chosen constructor it splits
the parent vector into one vector per child and decodes each child in the same way.
This repeated expansion is [corecursion](https://en.wikipedia.org/wiki/Corecursion),
the dual of the encoder's fold. Scoring an observed term follows that term's
constructor choices and adds their log probabilities, as `log_joint` does for a
[probabilistic program](programs.md#rsample-and-log_joint).

Each step uses one of four learned functions, which the library builds by default as
small neural networks:

- the `structure` head gives the logits over the constructors of a sort;
- the `factor` function splits a parent vector into its children's vectors, one
  function for each sort and number of children;
- the `primitive` head gives the logits over a data sort's vocabulary;
- the `binder_select` head gives the logits over the variables in scope, for an
  child of an index sort.

```qvr
signature LF
    sorts
        Term : object [dim=64]
        Type : object [dim=32]
        Name : data [dim=32, vocab=["dog", "cat", "every", "some"]]
    constructors
        Const : Name -> Term
        App : Term, Term -> Term
        Base : Name -> Type
    binders
        Lam : binds (x : Term : ty : Type) in (body : Term) -> Term

encoder C : LF
    dim Term = 64
    dim Type = 32
    op App(f, x) |-> gelu(f + x)

decoder D : LF [depth=8]
    body |-> recursive

loss reconstruction [weight=1.0, on=decoder(D)]
    -D.log_prob(term, C(term))
```

The body `body |-> recursive` selects this default corecursion. A decoder has two
operations:

- `sample(vec)` draws one term;
- `log_prob(term, vec)` scores an observed term.

Different terms can have similar codes, and the decoder expresses this by spreading
probability over them, so repeated calls to `sample(vec)` can return different terms.

The option `depth` bounds the depth of decoded terms. At the bound, only
constructors whose children are all of data or index sort remain available, and if
a sort has no such constructor the decoder raises an error. The bound is part of the
model: a term that needs nine nested expansions has probability zero under
`depth=8` and positive probability under a larger bound.
[Generated computations](../reference/qvr/generated-computations.md#structural-signatures-and-autoencoders)
shows how a decoder becomes an entry point that programs can call.

## Loss declarations

A `loss` declaration is a named expression added to a training objective, attached
to a part of the module with the `on` option. The reconstruction loss above is the
negative log probability that the decoder gives the input term after encoding it:
probability `0.2` gives a loss of about `1.609`, and probability `0.8` about `0.223`.
`weight` scales a loss relative to the others.

The attachment points are:

| `on=` | Where the loss applies |
|---|---|
| omitted | globally |
| `program(P)`, `deduction(D)`, `encoder(E)`, `decoder(D)` | the named declaration |
| `rule(R, D)` | every application of rule `R` while deduction `D` builds its chart |
| `chart(D)` | once, on the completed chart of deduction `D` |

`prog.losses.evaluate(env)` sums every loss, and `evaluate_on(kind, target, env)` sums
those with one attachment. Inside a [deduction](deduction.md#the-framework), the
engine adds the losses attached to its rules and its chart to
`ChartView.attached_loss`.

## Deduction integration

A chart item can carry a learned vector as well as its symbolic form. The symbolic
part decides which rules apply and how derivation weights combine, and an attached
encoder computes a vector for the item, for instance to score the next rule
application. A `deduction` takes a signature for its items and an encoder with the
options `signature=` and `encoder=`, and its chart's `embedding(item)` then returns
the encoder's vector for that item, with gradients.

```qvr
object Sentence : FinSet 4
object Tree : FinSet 4

signature ChartItem
    sorts
        Item : object [dim=128]
        Idx : data [dim=8]
        Cat : object [dim=32]
    constructors
        span : Idx, Idx, Cat -> Item

encoder InsideC : ChartItem
    dim Item = 128
    dim Cat = 32

deduction Parse : Sentence -> Tree [semiring=LogProb, start=S, signature=ChartItem, encoder=InsideC]
    atoms S, NP, Fwd, span
    rule combine : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X)

loss completed [weight=0.01, on=chart(Parse)]
    -chart.goal_weight()
```

`combine` is forward application in Steedman notation: `Fwd(X, Y)` is `X/Y`, which
takes a `Y` on its right and gives an `X` ([Weighted Deduction Systems](deduction.md#qvr-syntax)).
Any weighting of items by their vectors, as in the vector inside-outside parsers of
Le and Zuidema [@le2014inside] and Drozdov et al. [@drozdov2019diora], happens inside
the encoder's functions, and the chart still combines derivation weights with its
semiring.

## Bayesian integration

A decoder has the interface of a distribution, `sample` and `log_prob`. A program
can thus add a decoder's log probability of a term to its log joint density with
`score`:

```qvr
signature LF2
    sorts
        Term : object [dim=8]
        Name : data [dim=8, vocab=["dog", "cat"]]
    constructors
        Const : Name -> Term
        App : Term, Term -> Term

decoder D2 : LF2 [depth=3]
    body |-> recursive

object Code : Real 8
object Out : Real 1

program score_term : Code -> Out
    score s = D2.log_prob(t, x)
    return s
```

Here `x` is the program's input vector and `t` a term given with the observations.
A decoder cannot appear on the right of `sample`, `observe`, or `marginalize` in a
program, so a program cannot draw a latent term from it or sum over its terms.

## Stdlib decoders

`quivers.structural.shapes` provides decoders to match its encoders:

- `ar_decoder(sig, dim, vocab, depth)`: an autoregressive decoder for sequence
  signatures, with `depth` bounding the length;
- `tree_decoder(sig, dim, leaf_vocab, label_vocab, depth)`: a top-down decoder for
  tree signatures.

Both are instances of the same [`Decoder`](../api/structural/decoder.md) class as a
declared decoder.

## See also

- [Signatures and Encoders](structural-compression-signatures-and-encoders.md): the
  encoder half of the autoencoder.
- [Weighted Deduction Systems](deduction.md): the chart parser to which an encoder
  attaches.
- [Tree autoencoders tutorial](../tutorials/qvr/12-structural-autoencoders.md):
  running a reconstruction loss through `Program.run`, gradients, and checkpointed
  parameters.
- [Generated computation graphs](../reference/qvr/generated-computations.md): the
  entry points a signature, encoder, and decoder add.
