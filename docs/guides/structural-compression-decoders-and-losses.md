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
The right side of a checked `sample` or `observe` needs a typed law. For a
first-order decoder with a discrete carrier,
[`FiniteDecoderSupport`][quivers.qiec.structural_runtime.FiniteDecoderSupport]
builds such a law over an explicit, bounded set of supplied terms. For distinct
terms in the supplied set $S$, its probability is

$$
p_S(t\mid z) = \frac{p_D(t\mid z)}{\sum_{u\in S} p_D(u\mid z)},
\qquad t\in S.
$$

The denominator conditions the decoder on the supplied set. Duplicate entries
are a single term in this set. The prepared support checks the structural term
types once, and each call to `distribution` scores them at the new code and
parameters, retaining gradients through the normalizer. The consuming module
uses the support's rule-presentation declarations, and its
[`FiniteSampleable`][quivers.qiec.structured_sampleables.FiniteSampleable]
accepts draws, scores, and exact enumeration on that finite set.
The supplied candidates and their tensor weights enter as typed runtime
values. A source decoder name on its own remains a structural computation.

### Complete depth-bounded support

`FiniteDecoderSupport.complete(decoder, sort=..., max_candidates=...)`
starts from the declared root sort at the decoder's depth. It follows
every constructor that the decoder can choose at each reachable sort
and remaining depth. At depth zero the decoder admits only terminating
constructors. Each data position is filled from its reachable finite
vocabulary, with one alternative per distinct payload. Every reachable
sort and depth must have a productive expansion. A missing vocabulary
or a reachable choice that can exhaust the depth budget is refused.

Let $N(s,d)$ count the terms at object sort $s$ and depth $d$. For each
admitted constructor, multiply the counts of its ordered children,
using their object counts at depth $d-1$ or the sizes of their data
vocabularies. Sum those products over the constructors. The product
of no child counts is one. A constructor and its ordered children
identify one structural term, so the recurrence counts each term once.
Counting saturates at one above the candidate limit. The check visits
the reachable sort and depth pairs and their ordered constructor
positions before it builds any terms. If it accepts, every reachable
state also fits the limit, since its terms extend to distinct root
terms along a chosen path with productive siblings.

With maximum constructor arity $a$ and root depth $d$, a term has at
most $\sum_{j=0}^{d+1} a^j$ object and data nodes. Each data payload is
one of the stored vocabulary values, so its size is bounded by the
largest corresponding vocabulary entry. Preparing the support can
still require exponential work in depth, and the caller supplies the
candidate limit. Scoring the resulting law visits every prepared term.

If all reachable decoder heads produce valid local probabilities and
child embeddings, the complete support has original decoder mass one.
Inductively, each constructor's ordered product of complete child
laws has mass one, so summing over the locally normalized constructor
choices also has mass one. The finite law then agrees with the original
depth-bounded decoder. A supplied candidate set has the conditional
semantics above. The signature, depth and vocabulary stay fixed between
calls on a prepared support, while code and learned parameters can change.

### Drawing from a declared decoder

An authored decoder can declare `max_candidates` to prepare its complete
support before execution. This exposes the checked computation
`D__law(code)` for decoder `D`. Its term type uses the signature name
and the root object sort, for instance `Colors__TermValue`. Its
constructors use the signature's prefix. The checked law uses the
same neural heads as the decoder and keeps their gradients.

<!-- compile: qiec -->
```qvr
signature Colors
    sorts
        Term : object [dim=2]
    constructors
        Red : -> Term
        Blue : -> Term

decoder D : Colors [depth=1, max_candidates=2]
    body |-> recursive

instance random : Random

define draw_color(code : Tensor[Real]([2])) : Colors__TermValue !{random} =
    let law <- D__law(code)
    perform random.sample[Colors__TermValue](site("color"), law)

define read_color(color : Colors__TermValue) : Real !{} =
    case color motive => Real
        Colors__Red =>
            return 2.0
        Colors__Blue =>
            return 4.0

object Input : FinSet 1
object Out : Real 1

program measure(code : Tensor[Real]([2])) : Input -> Out
    let color <- draw_color(code)
    let center <- read_color(color)
    observe response <- Normal(loc=center, scale=1.0)
    return center

export measure
```

The call takes one code and returns a `Sampleable[Colors__TermValue]`,
which also admits `log_prob(law, color)` and replay at the same site.
The bound is carried in the checked request. Running serialized QIEC
requires the compiled decoder attachment with the same carrier and
bound. The namespaced values coexist with the opaque term type used
by the original structural decoder and encoder computations.

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
