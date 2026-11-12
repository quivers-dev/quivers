# DSL Declarations

A `.qvr` module is a sequence of declarations. `program` blocks and many-input
contractions have their own pages ([Programs and Let-Expressions](dsl-programs-and-lets.md),
[Contractions](dsl-contractions.md)), and the [Overview](dsl-overview.md#grammar)
summarizes the grammar.

## Algebra

`composition NAME [level=...]` selects the module's composition rule, the operations
$\otimes$ and $\bigoplus$ that composition uses
([Core Types](core.md#algebras-enrichment-algebras)). The choice follows what the
entries mean: `boolean` asks whether a path exists, `real` is ordinary sum-product
[contraction](dsl-contractions.md#type-driven-wiring-inference), and `markov` composes
conditional probabilities. Without a declaration the module uses `product_fuzzy`,
which combines soft relation scores and does not normalize rows.

```qvr
composition log_prob [level=algebra]
object A : FinSet 3
morphism f : A -> A [role=latent]
export f
```

The built-in rules at the `algebra` level are `product_fuzzy`, `boolean`,
`lukasiewicz`, `godel`, `tropical`, `max_plus`, `log_prob`, `markov`, `real`,
`probability`, and `counting`, and [Core Types](core.md#algebras-enrichment-algebras)
gives each one's operations and standard name. The value of `level` names the class
the rule must belong to, and the compiler checks the registered rule against it:

| Level | Class | Laws the rule claims |
|---|---|---|
| `algebra` | `Algebra` | associativity and an identity |
| `semigroupoid` | `Semigroupoid` | associativity, no identity |
| `bilinear_form` | `BilinearForm` | neither |
| `rule` | any `CompositionRule` | none required |

No law is checked ([The composition-rule hierarchy](transformations.md#the-composition-rule-hierarchy)).
The operations
[`identity(A)`](../api/core/morphisms.md#quivers.core.morphisms.identity),
[`f.dagger`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.dagger),
[`f.trace(A)`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.trace),
[`cup(A)`](../api/core/morphisms.md#quivers.core.morphisms.cup), and
[`cap(A)`](../api/core/morphisms.md#quivers.core.morphisms.cap) compile only when the
module's rule is an `Algebra`. [Composition Rules](../semantics/composition-rules.md)
gives the denotation of each level.

A rule can also be defined inline in an indented body, each entry a `let`-expression.
An `algebra` body requires `tensor_op`, `join`, `unit`, and `zero`, and may add
`negation` and `meet`, while `semigroupoid` and `bilinear_form` bodies require only
`tensor_op` and `join`. The compiler checks the entries against the level.

```qvr
composition my_sum_product [level=algebra]
    tensor_op(a, b) = a * b
    join(t)         = sum(t)
    unit            = 1.0
    zero            = 0.0

object A : FinSet 3
morphism f : A -> A [role=latent]
define g = f >> f
export g
```

## Object

An `object` declaration names a type. Its right side is an object expression:

```qvr
object X : FinSet 3
object Y : FinSet 4
object XY : X * Y                              # ProductSet(components=(X, Y))
object Sum : X + Y                             # CoproductSet(components=(X, Y))
object Strings : FreeMonoid(X, max_length=4)   # words over X of length at most 4
object Atoms : {np, s, n}                      # a finite set of named elements
object Cat : FreeResiduated(Atoms, depth=2, ops=[slash])

morphism f : XY -> X [role=latent]
export f
```

`FreeMonoid(X, max_length=n)` is the set of words over `X` of length at most `n`, the
[Kleene star](https://en.wikipedia.org/wiki/Kleene_star) of `X` truncated at `n`.
`FreeResiduated(Atoms, depth=d, ops=[...])` is the finite set of categorial
(Lambek) types over the given atoms of depth at most `d`, built with the listed
connectives (`slash` gives `/` and `\`) [@lambek1958mathematics]. A name declared as an
object can stand for its definition anywhere a type is expected, so
`morphism f : XY -> X` above has the product as its domain.

## Morphism

A morphism's role, in its option block, decides what is stored and learned:

| Role | What the declaration holds | Typical use |
| --- | --- | --- |
| `latent` without a prior | a table of learnable parameters | a learned soft relation or weight table |
| `latent` with `~ Family(...)` | a random tensor with a prior | Bayesian uncertainty about weights |
| `observed` | fixed data | a known relation |
| `kernel`, or no role | a distribution for each input | random outputs given inputs |
| `function` | a learned deterministic map | a neural network layer |
| `discretize`, `embed` | a map between finite and continuous objects | binning, embedding |

The role `latent` names a learnable table, not a latent random variable. A kernel's
learned parameters and a value drawn from the kernel are different things: changing a
parameter changes the distribution, and a draw uses the current distribution.

```qvr
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 2

# A learnable table
morphism f : X -> Y [role=latent]

# The same with a given initialization scale
morphism g : Y -> Z [role=latent, scale=0.3]

# Fixed data
morphism h : X -> X [role=observed] ~ identity(X)

define fg = f >> g
export fg
```

### Latent morphism prior

A `latent` declaration may end with `~ Family(args)`, which places a prior with the
given literal hyperparameters on the morphism's tensor. The compiler turns the prior
into a sample site whose value is the tensor, so the morphism becomes a deterministic
function of a random tensor. (A kernel's `~ Family` clause differs: there a network
computes the family's parameters from each input.) The
[`over` clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over) and other
options go in the option block, before the `~`, and the whole declaration is one line:

```qvr
object D : FinSet 3
object K : FinSet 2

# A matrix-normal prior on the 3 x 2 tensor of W
morphism W : D -> K [role=latent, over=[dom, cod]] ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0)
export W
```

Without the prior, `W` is a free parameter, estimated by maximum likelihood or as a
maximum a posteriori point under whatever objective trains the model.

### Init recipes (`[init=auto]`) {#init-recipes-initauto}

By default a learnable table starts from `randn * scale`. The option `[init=auto]`
uses instead the initialization recipe of the module's `Algebra`, from
[`Algebra.init_spec(depth, intermediate_size)`](../api/core/algebras.md). For
`product_fuzzy`, `boolean`, `godel`, `lukasiewicz`, and `probability` the recipe sets
the value after the sigmoid of
[`LatentMorphism`](../api/core/morphisms.md#quivers.core.morphisms.LatentMorphism),
so the raw parameter is `logit(value)`. For the other algebras it sets the raw
parameter directly.

```qvr
object D : FinSet 3
object K : FinSet 2
morphism W : D -> K [role=latent, init=auto]
export W
```

The recipes start intermediate values near the `Algebra`'s neutral element, which
avoids saturated sigmoids for truth values and exploding products for multiplicative
algebras. `recommend_init` and `saturation_warnings` in
[Fitting and Diagnostics](analysis-fitting-and-diagnostics.md#algebra-guided-training-tooling)
inspect a trained module.

## Space

Continuous types are objects too. `Real n` is $\mathbb R^n$, bounds go in braces, and
`Simplex n` is the [probability simplex](https://en.wikipedia.org/wiki/Simplex) of
dimension `n`:

```qvr
object R3 : Real 3
object R2_bounded : Real 2 {low=0.0, high=1.0}
object U : Real 1 {low=0.0, high=1.0}
object P2 : Real 2 {low=0.0}
object S3 : Simplex 3

# Product space
object RU : R3 * U

object X : FinSet 2
morphism k : X -> RU ~ Normal
export k
```

They resolve to subclasses of [`ContinuousSpace`](../api/continuous/spaces.md)
([Continuous Spaces](continuous-spaces.md)).

### Named composites

`object` is the only declaration that introduces a type. Finite sets,
[continuous spaces](continuous-spaces.md#continuousspace-hierarchy), products,
coproducts, and categorial types all enter the module as `object` declarations, and a
later declaration can use the name wherever a type is expected.

```qvr
object Hidden : Real 64
object Output : Real 32
object Combined : Hidden * Output

# Categorial (Lambek-style) types are object expressions too.
object Sentence : S \ NP
```

## Kernel

A kernel morphism declares a [Markov kernel](https://en.wikipedia.org/wiki/Markov_kernel)
`A -> B`. It is the default role, so a `morphism` without `role=` is a kernel. There
are two forms:

- **Without `~`**, a kernel between finite sets is a lookup table, a learnable matrix
  of conditional probabilities. Each row is a distribution over the codomain, an
  element of the distribution monad $\mathcal D(B)$ of [Stochastic Morphisms](stochastic.md#the-giry-monad-and-finstoch).
- **With `~ Family`**, a kernel into a continuous space draws from the family, whose
  parameters a network computes from each input. Each input is thus sent to a
  probability measure on `B`, an element of $\mathcal G(B)$ for the Giry monad
  $\mathcal G$. The optional [`over` clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over)
  sets which codomain axes form one joint event and which are independent copies.

A family call whose arguments are all literals fixes the family:
`morphism prior : A -> R ~ Normal(loc=0.0, scale=1.0)` ignores its input, draws from
that Normal for every element of `A`, and has no learned parameters. Writing `~ Normal`
without arguments lets the input choose the parameters through the kernel's
parameter source. A literal may be a number, a vector, or a matrix, as in
`~ Dirichlet(concentration=[1.0, 2.0, 3.0])`. A family argument that reads a name,
such as `~ Normal(scale=q)`, is refused, since a declaration binds no value for it to
read and a kernel does not mix learned and given parameters. A draw with a given scale
around a computed mean belongs in a program step, `sample s <- Normal(loc=m, scale=q)`,
where both values are bound.

```qvr
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 2
object R3 : Real 3
object S3 : Simplex 3

# Lookup-table kernels on finite sets
morphism s : X -> Y
morphism cat : X -> Y * Z

# Parametric kernels into continuous spaces
morphism f : X -> R3 ~ Normal
morphism g : R3 -> R3 ~ Normal
morphism k : X -> S3 ~ Dirichlet

# A conditional normalizing flow; its options go before the ~
morphism flow : R3 -> R3 [n_layers=6, hidden_dim=32] ~ Flow()

export s
```

More than fifty families are registered ([Continuous Families](continuous-families.md)).
`Flow` is not one of them: it builds a conditional
[normalizing flow](https://en.wikipedia.org/wiki/Flow-based_generative_model)
[@rezende2015flows] with `n_layers` coupling layers (4 by default) and hidden width
`hidden_dim` (64 by default). The runtime chooses the lookup-table or the parametric
form by whether the codomain is a [`FinSet`](../api/core/objects.md) or a
[`ContinuousSpace`](../api/continuous/spaces.md).

### Latent weights

The weights of a network, whether a network kernel's
(`[param_source=mlp(...)] ~ Normal`) or a [learned function's](#function), are point
estimates by default. With `[weights=latent]` every weight and bias becomes a latent
sample site with an independent `Normal(0, prior_scale)` prior (`prior_scale` is 1 by
default), drawn once per run and shared by every row and position that applies the
network. A Bayesian neural network regression [@mackay1992bayesian] is then one
kernel:

```qvr
object Feature : Real 2
object Target : Real 1
object Row : FinSet 6

morphism net : Feature -> Target [param_source=mlp(8), weights=latent, prior_scale=1.0] ~ Normal

program bnn : Row -> Target
    observe y : Row <- net(x)
    return y

export bnn
```

- **Sites.** A program without value parameters that uses the network draws its
  weights at its start, under the names `net_param_layer0_weight` (rows by columns),
  `net_param_layer0_bias`, and so on, through `net_param_weight` and `net_param_bias`
  for the output layer. A scan's cell and a called program read the weights their
  caller draws. Guides, MCMC, and `Predictive` treat these sites like any other latent
  site, and the compiled program has no parameters of its own for them.
- **Scope.** The option applies to a kernel whose family a network parameterizes and
  to a learned function, with a `linear` or `mlp` network. A draw through such a
  kernel names its inputs (`net(x)`), and the morphism composes in program steps or a
  scan, not in a `define` chain.
- **Transpilation.** No transpile target supports latent network weights
  ([Transpiler support](../transpile-support.md)). Writing the layers as
  [factor expressions](dsl-programs-and-lets.md#factor-expressions-assembling-indexed-tensors)
  over sampled weight matrices gives a model that every target accepts.

## Discretize

A `discretize` morphism maps a bounded one-dimensional space to a finite set of
equal-width bins:

```qvr
object U : Real 1 {low=0.0, high=1.0}

# Twenty bins on the unit interval
morphism d : U -> FinSet 20 [role=discretize, bins=20]
export d
```

Its runtime class is
[`Discretize`](../api/continuous/boundaries.md#quivers.continuous.boundaries.Discretize),
which accepts one-dimensional spaces only.

## Embed

An `embed` morphism maps a finite set into a continuous space:

```qvr
object X : FinSet 3
object R3 : Real 3

morphism e : X -> R3 [role=embed]
export e
```

Its runtime class
[`Embed`](../api/continuous/boundaries.md#quivers.continuous.boundaries.Embed) gives
each element $i$ a Normal distribution with a learnable center and a learnable scale,
$p(y \mid i) = \mathcal N(y; \mu_i, \sigma_i)$.

## Function

A neural network layer is a learned function with no noise: a projection
$z = Wx + b$, a recurrent update $h_t = \tanh(W [x_t; h_{t-1}] + b)$, a feature map.
`role=function` declares one between continuous spaces:

```qvr
object Feature : Real 3
object Hidden : Real 4
object Score : Real 2

# Affine by default: W x + b.
morphism proj : Feature -> Score [role=function]
# Two hidden layers of 16 under ReLU, then a tanh on the output.
morphism cell : Feature * Hidden -> Hidden [role=function, param_source=mlp(16, 16), activation=relu, output=tanh]
```

- **Network.** `param_source` selects it: `linear` (the default), `mlp` with optional
  hidden widths (`mlp(16, 16)`, and bare `mlp` is two layers of 64), or `identity`,
  which has no weights and needs a domain of one factor as wide as the codomain.
  `activation` names the nonlinearity between an `mlp`'s hidden layers (`tanh` by
  default, or `relu`, `sigmoid`, `softplus`), and `output` the one applied to the
  network's output (`identity` by default, or `tanh`, `sigmoid`, `softplus`, `relu`).
- **No site.** The function is the [Dirac kernel](https://en.wikipedia.org/wiki/Dirac_measure)
  at its output: every input goes to a point mass. A program calls it as a value,
  `let z = proj(x)`, with one argument per factor of its domain, and never draws from
  it, so `sample z <- proj(x)` is refused. In a chain it composes like any morphism,
  and as the cell of a scan it gives a deterministic recurrence.
- **Point-estimated weights.** Its weights are parameters of the compiled program,
  fit by whatever objective trains the model, as a network kernel's are. The checked
  core reads them as typed program inputs (`proj_param_weight`, `proj_param_bias`, and
  `cell_param_layer<k>_weight` and `_bias` for hidden layers), as the
  [learned functions reference](../reference/qvr/probabilistic-programs.md#learned-functions)
  states.

The runtime class is
[`LearnedFunction`](../api/continuous/deterministic.md#quivers.continuous.deterministic.LearnedFunction).
A linear Gaussian [state-space model](https://en.wikipedia.org/wiki/State-space_representation)
draws each state around a learned function of the previous state and the input. The
cell `step` below reads the noise scales `q` and `r`, which the calling program `ssm`
draws before the scan, since a program cell reads the bindings its caller makes
before the scan:

```qvr
object Driver : Real 2
object State : Real 3
object Obs : Real 2

morphism transition : Driver * State -> State [role=function]
morphism emission : State -> Obs [role=function]

program step(u_t, s_prev) : Driver * State -> State
    let mean = transition(u_t, s_prev)
    sample s <- Normal(loc=mean, scale=q)
    let center = emission(s)
    observe o <- Normal(loc=center, scale=r)
    return s

define run_all = scan(step)

program ssm : Driver -> State
    sample q <- HalfNormal(scale=1.0)
    sample r <- HalfNormal(scale=1.0)
    sample last <- run_all
    return last

export ssm
```

### Sequence functions: self-attention and position tables

Two networks read a whole sequence at once instead of one row at a time. A function
declared with `param_source=self_attention` is multi-head scaled dot-product
self-attention [@vaswani2017attention]: it maps a sequence of `T` rows to a sequence of
`T` rows, and the row at position `i` is a weighted sum of value projections of every
row it attends to. A function declared with `param_source=positions(N)` adds a learned
vector to each row, the `t`-th row of an `N`-row table to position `t`, which is how a
model that attends tells positions apart.

```qvr
object Model : Real 16

# Two heads of width 8; position i attends to positions 0 through i.
morphism attn : Model -> Model [role=function, param_source=self_attention, heads=2, causal=true]
# One learned vector per position, for sequences of up to 64 positions.
morphism positions : Model -> Model [role=function, param_source=positions(64)]
```

- **Options.** `heads` (one by default) must divide the model width $m$, the width of
  the codomain, and each head reads $m / \mathrm{heads}$ coordinates of the query, key,
  and value projections. `causal` is required, since the mask decides what the model may
  read: `causal=true` lets position `i` attend to positions `j <= i` only, as a language
  model scoring the next token needs, and `causal=false` lets every position attend to
  every position. The domain is one space, not a product, and a position table's
  codomain is its domain. Neither network takes `activation` or `output`, and both
  refuse `weights=latent`, so their weights are point estimates.
- **Sequences, not rows.** A program calls a sequence function in a `let` on a table
  `Tensor[Real]([T, d])`: the every-state output of a scan with `outputs=all`, a `let`
  computed from one, or a table given as data. The value is a table of the same length.
  `T` is read from the data, so one program handles sequences of any length, up to `N`
  for a position table. A sequence function cannot be a link of a chain or the cell of
  a scan, which apply their morphisms one position at a time, and the compiler refuses
  such a chain where it is drawn.
- **Composition by `let`.** A residual connection adds tables,
  `let h = u + attn(layer_norm(u))`, and
  [`layer_norm`](../reference/qvr/types-and-indexed-families.md#pure-expressions)
  normalizes each row [@ba2016layer].

A scan whose cell ignores its state runs a morphism at every position. A decoder-only
language model thus embeds each token in one scan, attends across the sequence in a
`let`, and scores each next token in a second scan:

```qvr
object Vocab : Real 8
object Model : Real 16
object Logits : Real 8

morphism embed : Vocab -> Model [role=function]
morphism positions : Model -> Model [role=function, param_source=positions(32)]
morphism attn : Model -> Model [role=function, param_source=self_attention, heads=2, causal=true]
morphism ff : Model -> Model [role=function, param_source=mlp(32), activation=relu]
morphism lm_head : Model -> Logits [role=function]

program embed_step(x_t, s_prev) : Vocab * Model -> Model
    let e = embed(x_t)
    return e

program head_step(h_t, s_prev) : Model * Model -> Model
    let z = h_t + ff(layer_norm(h_t))
    let logits = lm_head(layer_norm(z))
    observe next_token <- Categorical(logits=logits)
    return z

define embedded = scan(embed_step, outputs=all)
define scored = scan(head_step, outputs=all)

program transformer_lm : Vocab -> Model
    sample e <- embedded
    let u = positions(e)
    let h = u + attn(layer_norm(u))
    sample out <- scored(h)
    return out

export transformer_lm
```

Each input row is one sequence of one-hot tokens, of shape `(batch, T, 8)`, and the
observation `next_token`, of shape `(batch, T)`, holds the token after each position.
With the source above saved as `transformer_lm.qvr`:

```python
import torch
from quivers.dsl import load

model = load("transformer_lm.qvr").morphism
tokens = torch.randint(0, 8, (4, 9))
x = torch.nn.functional.one_hot(tokens, 8).float()
next_token = torch.randint(0, 8, (4, 9))
log_joint = model.log_joint(x, {"next_token": next_token})  # shape (4,)
```

The runtime class is
[`SequenceFunction`](../api/continuous/deterministic.md#quivers.continuous.deterministic.SequenceFunction)
over a
[`SelfAttentionSource`](../api/continuous/param_source.md#quivers.continuous.param_source.SelfAttentionSource)
or a
[`PositionSource`](../api/continuous/param_source.md#quivers.continuous.param_source.PositionSource).
The checked core builds the same map from comprehensions over the open sequence length,
with the weights as typed program inputs, as the
[sequence functions reference](../reference/qvr/probabilistic-programs.md#sequence-functions)
states.

## Deduction

`deduction NAME : Domain -> Codomain [options]` with an indented body declares a
[weighted deduction system](deduction.md#the-framework) that the agenda engine
evaluates. The body gives the item constructors and the rules, and the option block
gives the semiring, the goal, and the evaluation bounds:

```qvr
object Term : FinSet 16

deduction CCG : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms NP, S, N, VP, PP, Fwd, Bwd, span
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X)
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X)
```

- **`atoms NAME, NAME, ...`** lists the constructor symbols. In a rule pattern an
  identifier that is neither a declared atom nor a numeral is a pattern variable,
  whatever its spelling, and a variable that occurs twice in a rule must match the same
  term at both occurrences. A declared atom in a chart-position slot, such as the first
  two slots of `span`, is a compile error naming the rule and the identifier.
- **`rule NAME : premises |- conclusion`** is an inference rule with any number of
  comma-separated premises. A trailing `#[learnable]` pragma gives the rule a learnable
  weight, one for each distinct binding of its category variables and shared across
  chart positions, so a rule whose variables are all positions has one weight.
- **`semiring=`** selects `LogProb`, `Viterbi`, `Boolean`, or `Counting`.
- **`start=`** names the category of the goal items.
- **`depth=`** bounds the constructor depth of the categories in rule conclusions, so
  the agenda terminates on cyclic rules.
- **`tolerance=`** (optional) stops the iteration of a cyclic chart once its weights
  change by less than the tolerance, which computes the Kleene closure of a cycle
  [@mohri2002semiring]. It is used with rule weights marked `#[learnable, bounded]`,
  which keeps cyclic weights below one.

Slash, product, and modal type constructors are declared atoms, not built-in syntax.
In Steedman notation [@steedman2000syntactic], `Fwd(X, Y)` is `X/Y`, which takes a `Y`
on its right to give `X`, and `Bwd(X, Y)` is `X\Y`, which takes a `Y` on its left. The
usual further names are `Tns(X, Y)` for a product, `Dia(X)` and `Box(X)` for the
modalities ◇ and □, and `Cont(X)` for a continuation-typed category, and a grammar may
declare any others.

### Axiom sources

The axioms are the weighted items the input contributes. A deduction obtains them in
one of three ways:

| Form | Use |
|------|----------|
| a `lexicon` block of `"word" : Cat = lf` entries, optionally `#[learnable]` | a lookup table written in the body |
| `lexicon from "path.tsv" [learnable=true]` | a lookup table read from a TSV file at compile time |
| `[axioms=some_morphism]` in the option block | a declared morphism from the input to weighted items |

```qvr
object Term : FinSet 16

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4]
    atoms S, NP, VP, Det, N, V, span, the, cat, sleeps
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S) #[learnable]
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP) #[learnable]
    rule vp_v : span(I, J, V) |- span(I, J, VP) #[learnable]
    lexicon
        "the"    : Det = the    #[learnable]
        "cat"    : N   = cat    #[learnable]
        "sleeps" : V   = sleeps #[learnable]
```

A `#[learnable]` lexicon entry gets a log weight, a PyTorch
[`nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Parameter.html)
initialized to `0.0`.

### Chart-query expressions

A program parses its input with `parse`, and the result is a chart that answers
queries:

<!-- compile: cumulative -->
```qvr
object Term : FinSet 16
object Weight : Real 1

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4]
    atoms S, NP, VP, Det, N, V, span, the, cat, sleeps
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S) #[learnable]
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP) #[learnable]
    rule vp_v : span(I, J, V) |- span(I, J, VP) #[learnable]
    lexicon
        "the"    : Det = the     #[learnable]
        "cat"    : N   = cat     #[learnable]
        "sleeps" : V   = sleeps  #[learnable]

program parse_score : Term -> Weight
    let chart = parse(PCFG, sentence)
    score w = chart.goal_weight()
    return w

export parse_score
```

Here `sentence` is a [data input](inference-foundations.md#host-data-per-row-covariates-and-index-arrays),
given with the observations. The chart's
[`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) methods
are:

- `chart.weight(item)`: the weight of one item;
- `chart.enumerate(pattern)`: the `(item, weight)` pairs matching a pattern with
  variables;
- `chart.derivations(item)`: the items that contributed to `item`'s weight, as a list;
- `chart.goal_weight()`: the weight of the goal.

The weights are tensors, and their gradients reach every learnable rule and lexicon
weight through the semiring's operations.

## Replicated declarations

`[replicate=N]` declares `N` copies of a morphism with independent parameters, named
`NAME_0`, ..., `NAME_{N-1}`. The base name then refers to the group, which
[`fan`](#fan-out-diagonal-morphism) expands:

```qvr
object Latent : Real 4
object HeadOut : Real 2
object State : FinSet 3
object Obs : FinSet 2
object Token : FinSet 5
object Hidden : Real 3

# head_0, head_1, head_2, head_3, each with its own parameters
morphism head : Latent -> HeadOut [replicate=4] ~ Normal

# Replication works for lookup-table kernels and embeddings as well
morphism T : State -> Obs [replicate=3]
morphism tok : Token -> Hidden [role=embed, replicate=2]

define multi_head = fan(head)
export multi_head
```

## Combinators on morphisms

The combinators below build morphisms from simpler ones in `define` declarations at the
top level of a module, outside `program` blocks.

### Fan-out (diagonal morphism)

`fan(f, g, ...)` sends one input to several morphisms with the same domain and returns
all their outputs together. Its codomain is the product of their codomains, laid out
by concatenating their outputs, so a vector output's width is the sum of the widths.
`f @ g`, in contrast, expects a pair of inputs ([Morphisms](morphisms.md#tensor-product-the-operator)).
`fan` takes morphism names or the name of a replicated group:

```qvr
object Latent : Real 4
object HeadOut : Real 2
object Both : Real 8

morphism head : Latent -> HeadOut [replicate=4] ~ Normal

# fan(head) expands to fan(head_0, head_1, head_2, head_3)
define multi_head = fan(head)

# often followed by a projection that recombines the outputs
morphism proj : Both -> Latent ~ Normal
define recombined = fan(head) >> proj
export recombined
```

The morphisms keep their own parameters.

### Repeat (iterated composition)

`repeat(f, n)` composes an endomorphism with itself `n` times using the same
parameters at every step, as a transition applied several times. `stack` uses a fresh
copy at each step, and a scan also reads a new input at each step. The three differ
in which parameters the steps share.

```qvr
object State : FinSet 3
object Obs : FinSet 2

morphism transition : State -> State
morphism emission : State -> Obs

# transition >> transition >> transition, one set of parameters
define deep = repeat(transition, 3)

# repeat(f, 1) is f
define same = repeat(transition, 1)

# the number of steps set at run time
define n_step = repeat(transition) >> emission

export n_step
```

Without a count, `repeat(f)` builds a
[`RepeatMorphism`](../api/core/morphisms.md#quivers.core.morphisms.RepeatMorphism)
whose number of steps is set by `Program.forward(n_steps=N)` and computed by repeated
squaring in $O(\log n)$ compositions:

<!-- python: skip -->
```python
prog = load("hmm.qvr")
obs_3 = prog(n_steps=3)    # T^3 >> E
obs_50 = prog(n_steps=50)  # T^50 >> E, same model, different length
```

### Stack (independent multi-layer)

`stack(f, n)` composes `n` copies of an endomorphism `f`, each with its own
parameters:

```qvr
object State : FinSet 3

morphism transition : State -> State

# 3 layers, each with its own parameters
define deep = stack(transition, 3)

# 3 iterations sharing one set of parameters
define tied = repeat(transition, 3)

export deep
```

### Scan (temporal recurrence)

`scan(cell)` threads a hidden state through a sequence. The cell has a product
domain `A * H -> H`, where `A` is the type of one input and `H` the type of the state,
and `scan(cell) : A -> H`, with the sequence on the second axis of the input tensor,
`[batch, seq_len, input_dim]`.

```qvr
object Token : FinSet 256
object Embedded : Real 64
object Hidden : Real 128
object Output : Real 64

morphism tok_embed : Token -> Embedded [role=embed]
morphism cell : Embedded * Hidden -> Hidden ~ Normal
morphism output_proj : Hidden -> Output ~ Normal

# the last state, read by a projection
define rnn = tok_embed >> scan(cell) >> output_proj

# a learned initial state (zeros by default)
define rnn_learned = tok_embed >> scan(cell, init=learned) >> output_proj

# every state rather than the last, and a scan from the last position to the first
define states = tok_embed >> scan(cell, outputs=all)
define backward = tok_embed >> scan(cell, reverse=true)

export rnn
```

For a sequence $x_0, \dots, x_{T-1}$ and an initial state $h_{\mathrm{init}}$, the scan
computes

$$
h_{-1} = h_{\mathrm{init}}, \qquad h_t = \mathrm{cell}(x_t, h_{t-1}) \quad (t = 0, \dots, T - 1),
$$

so $h_t$ is the state after reading $x_t$.

- **Initial state.** $h_{\mathrm{init}}$ is zeros by default, a learned vector with
  `init=learned`, or, for a scan inside a program, a value the program binds before
  the scan (`init=h0`).
- **Output.** `outputs=last` (the default) returns $h_{T-1}$, and `outputs=all` returns
  the sequence $h_0, \dots, h_{T-1}$.
- **Direction.** `reverse=true` visits the positions from last to first,
  $h_t = \mathrm{cell}(x_t, h_{t+1})$ with $h_T = h_{\mathrm{init}}$, and returns $h_0$.
  Its sequence of states is still laid out in input order, entry $t$ being the state
  after reading $x_t$.
- **Options.** `init`, `outputs`, and `reverse` may appear in any order, each at most
  once.
- **Cells.** A cell may be a parametric kernel (`morphism cell : A * H -> H ~ Normal`),
  a [learned function](#function) (`[role=function]`, a deterministic recurrence with
  no site), or a program (`program cell(x, h) : A * H -> H`). Inside a program a scan's
  step runs once per position. A kernel cell is drawn at the scan's binding, with sites
  `h@0`, `h@1`, ..., for `sample h <- scan(cell)`. A program cell reads the bindings its
  caller makes before the scan, its sites are addressed `s@0`, `s@1`, ..., and an
  `observe` in the cell scores one observation per position. The
  [scans reference](../reference/qvr/probabilistic-programs.md#scans) states the rules.

Scans stack. Each scan but the last returns every state with `outputs=all`, and the
next one reads that sequence one state per position:

```qvr
object Token : FinSet 16
object Embedded : Real 4
object H1 : Real 3
object H2 : Real 3
object Output : Real 2

morphism tok_embed : Token -> Embedded [role=embed]
morphism cell_1 : Embedded * H1 -> H1 ~ Normal
morphism cell_2 : H1 * H2 -> H2 ~ Normal
morphism output_proj : H2 -> Output ~ Normal

define deep_rnn = tok_embed >> scan(cell_1, outputs=all) >> scan(cell_2) >> output_proj
export deep_rnn
```

Each scan threads its own state. A morphism between two scans,
`scan(cell_1, outputs=all) >> proj >> scan(cell_2)`, runs at every position inside the
second scan's step. A scan over the final state of another, `scan(cell_1) >> scan(cell_2)`,
is refused, since one state is not a sequence, and so is a morphism applied to a whole
sequence of states outside a scan.

`fan` and `@` pair two sequences of states position by position, concatenating the
two states at each position. A bidirectional recurrence runs a forward and a reversed
scan over the same tokens and reads both at every position with a third scan:

```qvr
object Token : FinSet 16
object Embedded : Real 4
object Forward, Backward : Real 3
object Both : Real 6
object Tag : Real 2

morphism tok_embed : Token -> Embedded [role=embed]
morphism forward_cell : Embedded * Forward -> Forward ~ Normal
morphism backward_cell : Embedded * Backward -> Backward ~ Normal
morphism tag_cell : Both * Tag -> Tag ~ Normal

define forward_states = tok_embed >> scan(forward_cell, outputs=all)
define backward_states = tok_embed >> scan(backward_cell, outputs=all, reverse=true)
define tagger = fan(forward_states, backward_states) >> scan(tag_cell, outputs=all)

program tag_sequence : Token -> Tag
    sample tags <- tagger
    return tags

export tag_sequence
```

At position $t$ the tag cell reads the forward state after $x_0, \dots, x_t$ and the
backward state after $x_{T-1}, \dots, x_t$. The two sequences pair by input position,
so a reversed scan's states need no reindexing. Sites are numbered in the order the
run reaches them, so the reversed scan's occurrence `@0` is its draw at the last
position.

**Marginalized chains.** A program cell whose state is a finite latent variable that
it marginalizes and returns is one step of a discrete Markov chain, and
`scan(cell, init=p)` sums out every path of states with the
[forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm)
[@rabiner1989hmm], in $O(T K^2)$ time for $T$ positions and $K$ states. With
emissions that observe the scanned symbols, this is a hidden Markov model:

```qvr
object State : FinSet 2
object Symbol : FinSet 4
object StartProbs : Real 2

program hmm_step(symbol, z_prev) : Symbol * State -> State
    marginalize z : State <- Categorical(probs=transition[z_prev])
        observe symbol <- Categorical(probs=emission[z])
    return z

define chain = scan(hmm_step, init=start)

program hmm : Symbol -> StartProbs
    sample start <- Dirichlet(concentration=1.0) [over=State]
    sample transition : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission : State <- Dirichlet(concentration=1.0) [over=Symbol, iid_over=State]
    sample path <- chain
    return start

export hmm
```

`start` is the distribution of the state before the first position,
`transition[z_prev]` moves the chain, and the block scores each position's symbol at
the new state. The chain adds its log marginal likelihood to the joint under the name
`path`, which holds no state and is not read again. The cell's `let` steps may read
the scanned input, as an input-output model's transitions do, and an observation
under a name other than the input's is one observation per position. The
[marginalized chains reference](../reference/qvr/probabilistic-programs.md#marginalized-chains)
states the rules and the refused forms, and
[chapter 5 of the tutorial](../tutorials/qvr/05-time-series.md#hidden-markov-models)
compares the marginal with a forward recursion written in PyTorch.

### Backward composition

`<<` writes composition right to left: `g << f` is `f >> g`. Both operators compose in
the operands' shared `Algebra` and reject a mismatch, and an explicit `.change_base`
between segments crosses algebras ([Transformations](transformations.md)).

```qvr
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 2

morphism f : X -> Y [role=latent]
morphism g : Y -> Z [role=latent]

define fg = f >> g
define gf = g << f    # the same composite as fg
export gf
```

### Curry combinators (residuation witnesses)

Currying passes the arguments of a function of two arguments one at a time: a function
of a pair `(x, y)` becomes a function that takes `x` and returns a function awaiting
`y`. In a [residuated](https://en.wikipedia.org/wiki/Residuated_lattice) structure the
two directions have their own types, and for $f : X \otimes Y \to Z$ the postfix
combinators give

| Postfix | Result |
|---------|--------|
| `f.curry_right` | a morphism `X -> Z/Y` |
| `f.curry_left`  | a morphism `Y -> X\Z` |

where `Z/Y` and `X\Z` are categorial types over the codomain in the argument-under
(Lambek) notation [@lambek1958mathematics]. The tensor is unchanged, and only its
split into domain and codomain axes is reread, so the domain must be a product of at
least two factors.

```qvr
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 5
morphism f : X * Y -> Z [role=latent]
define g = f.curry_right    # g : X -> Z/Y
export g
```

## Top-level `define` declarations

`define` binds a morphism expression at the top level of a module:

<!-- compile: false -->
```qvr
define fg = f >> g
define par = f @ g
define marg = fg.marginalize(Y)
define composed = f >> g >> h
```

`let` binds values inside a `program` block
([Programs and Let-Expressions](dsl-programs-and-lets.md#let-expressions-arithmetic-primitives-and-collections)).

### `where` clauses

`where` attaches local `define` bindings, visible only in the enclosing definition:

```qvr
object Token : FinSet 4
object State : FinSet 3
object Obs : FinSet 2

morphism tok_embed : Token -> State
morphism transition : State -> State
morphism output_proj : State -> Obs

define model = embed >> layers >> output_proj where
    define layers = stack(transition, 3)
    define embed = tok_embed

export model
```

## Indexed families and effect declarations

Six declarations belong to the Quivers Indexed Effect Core (QIEC), the typed core
language of QVR ([language reference](../reference/qvr/index.md)). They state
contracts that a tensor shape cannot. An `index` introduces a finite enumerated index
type and its constructors. A `family` declares an
[indexed family](https://ncatlab.org/nlab/show/inductive+family) [@dybjer1994inductive],
a type such as vectors of a given length whose constructors fix the index, with type
parameters in square brackets and indices in parentheses. An `effect` declares the
operations of an [effect interface](effects.md#qiec-effect-interfaces-and-rows), and
an `instance` creates a named occurrence of one. A `handler` interprets the
operations of an instance, and a typed `define` names a computation that a `program`
or another computation calls.

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

effect Abort[E : Type]
    abort : E -> Unit

instance stop : Abort[String]

handler ignore[E : Type, A : Type] for Abort[E] : A -> A [coverage=partial, implementation=foreign]
    abort resumes 0
```

A constructor's result names its family with explicit indices, so a case analysis on a
`Vec[A](S(m))` knows the vector is nonempty. A handler's options give its
`coverage` (`total` or `partial`), its forwarding of uncovered operations
(`forwards=unknown|none`), and the effects it introduces (`introduces=!{...}`). Each
operation clause has a
[resumption grade](../reference/qvr/effects-and-handlers.md#resumption-grades): `0`
never resumes, `aff` resumes at most once, `1` exactly once, and `omega` any number of
times when the captured context can be copied. A handler's clause bodies are either
written in QVR (`[implementation=authored]`, each clause a `=>` body that may
`resume`) or given at run time by a Python provider
(`[implementation=foreign]`). [Types and indexed families](../reference/qvr/types-and-indexed-families.md),
[effects and handlers](../reference/qvr/effects-and-handlers.md), and
[computations](../reference/qvr/computations.md) specify these forms.

## Structural compression: `signature`, `encoder`, `decoder`, `loss`

These four declarations define autoencoders over terms. A `signature` declares sorts
and constructors, an `encoder` folds a term into a vector with one learnable function
per constructor, a `decoder` defines a distribution over terms given a vector, and a
`loss` attaches a training objective to part of the module.
[Signatures and Encoders](structural-compression-signatures-and-encoders.md) and
[Decoders and Losses](structural-compression-decoders-and-losses.md) describe them.

## Export

`export` names a morphism or program as an output of the compiled module. A module
may have several `export` declarations, and each is compiled into its own output:

<!-- compile: false -->
```qvr
export f
export fg
export my_prog
```
