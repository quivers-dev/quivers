# DSL Declarations

This page covers every declaration form a `.qvr` module can carry
apart from `program` blocks and operadic contractions, which have
their own pages ([Programs and Let-Expressions](dsl-programs-and-lets.md),
[Contractions](dsl-contractions.md)). The grammar summary lives in
the [Overview](dsl-overview.md#grammar).

## Algebra

Choose the rule by the meaning of the entries you want to combine.
Boolean asks whether a path exists, `real` performs ordinary numeric
sum-product [contraction](dsl-contractions.md#type-driven-wiring-inference), and `markov` combines conditional probabilities.
The default fuzzy rule combines soft relation scores and does not
normalize probability rows. See [Core Types](core.md) for examples and
for the distinction between the API's levels and mathematical lawfulness.


Choose the enriching algebra (optional, defaults to `product_fuzzy`):

<!-- compile: false -->
```qvr
composition product_fuzzy [level=algebra]
composition boolean [level=algebra]
composition lukasiewicz [level=algebra]
composition godel [level=algebra]
composition tropical [level=algebra]
composition max_plus [level=algebra]
composition log_prob [level=algebra]
composition markov [level=algebra]
composition real [level=algebra]
composition probability [level=algebra]
composition counting [level=algebra]
```

The keyword `algebra` resolves a name against the built-in
composition-rule registry and verifies the registered rule is at
the [`Algebra`](../api/core/algebras.md) level. Three weaker levels
can also be declared in QVR:

<!-- compile: false -->
```qvr
composition material_impl [level=semigroupoid]
composition some_bf [level=bilinear_form]
composition any_rule [level=rule]
```

A [semigroupoid](https://ncatlab.org/nlab/show/semigroupoid) level
promises associativity but not an identity; `bilinear_form` promises
neither; `rule` is permissive and accepts any rule in the registry.
Operations that need the identity element
([`identity(A)`](../api/core/morphisms.md#quivers.core.morphisms.identity),
[`f.dagger`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.dagger),
[`f.trace(A)`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.trace),
[`cup(A)`](../api/core/morphisms.md#quivers.core.morphisms.cup),
[`cap(A)`](../api/core/morphisms.md#quivers.core.morphisms.cap))
compile only under `algebra`. See [Composition
Rules](../semantics/composition-rules.md) for the algebraic
hierarchy and operadic contraction syntax.

A composition rule can also be defined inline via an indented body,
with each entry a `let`-expression:

<!-- compile: false -->
```qvr
composition my_sum_product [level=algebra]
    tensor_op(a, b) = a * b
    join(t)         = sum(t)
    unit            = 1.0
    zero            = 0.0

composition my_semi [level=semigroupoid]
    tensor_op(a, b) = a * b
    join(t)         = sum(t)

composition my_bf [level=bilinear_form]
    tensor_op(a, b) = (a + b) * 0.5
    join(t)         = sum(t)
```

`algebra` bodies require `tensor_op`, `join`, `unit`, `zero` (with
optional `negation` and `meet`); `semigroupoid` and `bilinear_form`
bodies require only `tensor_op` and `join`. The compiler verifies
the entry set against the declared level.

## Object

Three forms are available:

<!-- compile: false -->
```qvr
# 1. anonymous-element FinSet of given cardinality, or a TypeExpr
object X : FinSet 3          # FinSet [name="X", cardinality=3]
object Y : FinSet 4
object XY : X * Y     # ProductSet(components=(X, Y))
object Sum : X + Y    # CoproductSet(components=(X, Y))

# 2. FreeMonoid, bounded Kleene closure over a FinSet of generators.
object Strings : FreeMonoid(X, max_length=4)

# 3. Residuated universe.
object Cat : FreeResiduated(Atoms, depth=4, ops=[slash])
```

Object-shaped aliases (resolvable to a
[`SetObject`](../api/core/objects.md)) are interchangeable with the
underlying object, so `morphism f : Pair -> X [role=latent]` is well-formed when
`Pair = X * Y`.

## Morphism

The role determines what is stored and learned:

| Role | What the declaration supplies | Typical use |
| --- | --- | --- |
| `latent` without a prior | a table of optimizable parameters | a learned soft relation or weight table |
| `latent` with `~ Family(...)` | a random representing tensor | Bayesian uncertainty about weights |
| `observed` | fixed data | a supplied relation |
| `kernel` or omitted role | an input-dependent distribution | random outputs given inputs |

A kernel's learned parameters and a sample drawn from that kernel are
different objects. Changing a parameter changes the distribution; drawing
a sample uses the current distribution.


Declare a learnable or fixed morphism:

<!-- compile: false -->
```qvr
# Latent (learnable)
morphism f : X -> Y [role=latent]

# With init scale
morphism g : Y -> Z [role=latent, scale=0.3]

# Observed (fixed)
morphism h : X -> X [role=observed] ~ identity(X)
```

### Latent morphism prior

A `latent` morphism declaration accepts a trailing
`~ Family(args)` clause that puts a prior on its representing
tensor. The literal `(args)` carry the prior's hyperparameters at
declaration time (in contrast to `kernel`'s `~ Family` clause,
whose parameters come from a neural network at sample time). The
axis-role configuration (`over=`, `iid_over=`) and any other
modifiers live in the morphism's option block, before the `~`. The
compiler desugars the prior into a fresh sample site whose value
is the tensor representing the morphism, making the morphism a
deterministic wrap of that random tensor.

<!-- compile: false -->
```qvr
# Free-parameter morphism (point estimate; MLE / MAP target).
morphism W : Real D -> Real K [role=latent]

# The same morphism with a Matrix-Normal prior on its tensor.
morphism W : Real D -> Real K [role=latent, over=[dom, cod]]
    ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0)
```

The [axis-role clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over)
is fully detailed alongside the program-step bind and observe forms.

### Init recipes (`[init=auto]`)

The `[init=auto]` option overrides the default `randn * scale` init
with the active algebra's initialization recipe, derived from
[`Algebra.init_spec(depth, intermediate_size)`](../api/core/algebras.md).
For ProductFuzzy, Boolean, Gödel, Łukasiewicz, and Probability the
recipe is applied through
[`LatentMorphism`](../api/core/morphisms.md#quivers.core.morphisms.LatentMorphism)'s
sigmoid bijector (the raw parameter is set via `logit(value)`); for
the other algebras the recipe is applied to the raw parameter
directly.

<!-- compile: false -->
```qvr
morphism W : Real D -> Real K [role=latent, init=auto]
```

The per-algebra recipes ensure that a fresh model with `[init=auto]`
on every latent starts with intermediate activations close to the
algebra's neutral element, avoiding saturated sigmoids on
truth-valued algebras and exploding products on multiplicative
algebras. See `recommend_init` and `saturation_warnings` in the
[fitting and diagnostics guide](analysis-fitting-and-diagnostics.md#algebra-guided-training-tooling)
for the dynamic counterpart that audits a trained module.

## Space

Declare a continuous space:

<!-- compile: false -->
```qvr
object R3 : Real 3
object R2_bounded : Real 2 {low=0.0, high=1.0}
object U : Real 1 {low=0.0, high=1.0}
object P2 : Real 2 {low=0.0}
object S3 : Simplex 3

# Product space
object RU : R3 * U
```

Spaces resolve to subclasses of
[`ContinuousSpace`](../api/continuous/spaces.md); see the
[continuous spaces guide](continuous-spaces.md) for the full
hierarchy and constraints.

### Named composites

`object` is the only top-level type-introduction keyword: discrete
sets, [continuous spaces](continuous-spaces.md#continuousspace-hierarchy), and any composite (products, coproducts,
residuated patterns) all enter the environment as `object`
declarations whose RHS is an object expression.

```qvr
object Hidden : Real 64
object Output : Real 32
object Combined : Hidden * Output

# Residuated / Lambek-style patterns are object expressions too.
object Sentence : S \ NP
```

A declared object can be referenced by name in any subsequent
declaration; the compiler substitutes it wherever the name appears
in a domain or codomain position.

## Kernel

A kernel morphism declares a [Markov
kernel](https://ncatlab.org/nlab/show/Markov+kernel) `A -> B`. It is
the default role, so a `morphism` with no `role=` key is a kernel.
Two shapes:

- **Without a `~` clause**: a finite-set lookup-table kernel
  `A -> D(B)`, realized as a learnable matrix of conditional
  probabilities.
- **With a `~ Family [options] [axis_role_clause]` clause**: a
  parametric kernel `A -> G(B)` whose family parameters come from
  the input by a parameter network at sample time. The optional
  [axis-role clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over)
  configures the family's event / batch decomposition over codomain
  factors.

An explicit all-literal family call fixes the family instead:
`morphism prior : A -> R ~ Normal(loc=0.0, scale=1.0)` ignores its input and draws
from that Normal for every element of `A`. It has no learned parameter
table. Leave the arguments off, as in `~ Normal`, when the input should
select parameters through the kernel's parameter source. A literal may be a
number, a vector, or a matrix, as in
`~ Dirichlet(concentration=[1.0, 2.0, 3.0])`. Every path refuses a family
argument that reads a name, such as `~ Normal(scale=q)`, at that argument: a
declaration binds no value for it to read, and a kernel does not mix learned
and given parameters. A draw that needs a given scale around a computed mean
says so in a program step, `sample s <- Normal(loc=m, scale=q)`, where both
values are bound.

<!-- compile: false -->
```qvr
# Lookup-table kernel on finite sets.
morphism s : X -> Y
morphism cat : X -> Y * Z

# Parametric kernel: input-conditional Normal on R^3.
morphism f : X -> R3 ~ Normal
# Family options control the parameter network.
morphism g : R3 -> R3 ~ Normal
morphism k : X -> S3 ~ Dirichlet
# More than forty families are registered; see the families guide.
# `Flow` is a special-case constructor (not in the family registry)
# that compiles to a conditional normalizing flow.
morphism flow : R3 -> R3 ~ Flow [n_layers=6, hidden_dim=32]
```

The kernel role unifies the discrete (finite-set lookup) and
continuous or stochastic (parametric `Family`) cases under one declaration;
the runtime branches on whether the codomain is a
[`FinSet`](../api/core/objects.md) or a
[`ContinuousSpace`](../api/continuous/spaces.md).

### Latent weights

By default a network's weights, whether a network kernel's
(`[param_source=mlp(...)] ~ Normal`) or a [learned function's](#function),
are point estimates. `[weights=latent]` makes every weight and bias a latent
sample site with an independent `Normal(0, prior_scale)` prior
(`prior_scale` defaults to 1), drawn once per run and shared by every row and
position that applies the network. A Bayesian neural network regression is
then one kernel:

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

- **Sites.** Each program without value parameters that uses the network
  draws its weights at its head, named as the checked elaboration names a
  network's inputs: `net_param_layer0_weight` (rows by columns),
  `net_param_layer0_bias`, ..., `net_param_weight`, `net_param_bias`. A
  scan's cell and a called program read the weights their caller draws.
  Guides, MCMC, and `Predictive` treat them as they treat any latent site,
  and the compiled program has no parameters of its own for them.
- **Where it applies.** A kernel over a family a network parameterizes, or a
  learned function, with a `linear` or `mlp` network. A draw through a
  latent-weight kernel names its inputs (`net(x)`), and a latent-weight
  morphism composes in program steps or a scan, not in a `define` chain
  without one.
- **Transpilation.** Every target refuses it as `param-source:<kind>:latent`:
  the weights are sites, but no target has a form for the network's layers
  applied to them. Writing the layers as `factor` comprehensions over
  sampled weight matrices gives a model every target renders.

## Discretize

Convert a continuous space to a finite set via binning:

<!-- compile: false -->
```qvr
# Discretize a UnitInterval `U` into 20 bins.
morphism d : U -> FinSet 20 [role=discretize, bins=20]

# Discretize an R^3 box into 100 bins.
morphism d2 : R3 -> FinSet 100 [role=discretize, bins=100]
```

The runtime realization is
[`Discretize`](../api/continuous/boundaries.md#quivers.continuous.boundaries.Discretize),
which produces uniform-width bins on bounded supports.

## Embed

Embed a discrete object into a continuous space:

<!-- compile: false -->
```qvr
# Embed a finite-set value into R^3 (treated as uniform).
morphism e : X -> R3 [role=embed]
```

The runtime realization is
[`Embed`](../api/continuous/boundaries.md#quivers.continuous.boundaries.Embed).

## Function

A neural network layer is a learned function with no noise: a projection
`z = W x + b`, a recurrent update `h_t = tanh(W [x_t; h_{t-1}] + b)`, a
feature map. `role=function` declares one between continuous spaces:

```qvr
object Feature : Real 3
object Hidden : Real 4
object Score : Real 2

# Affine by default: W x + b.
morphism proj : Feature -> Score [role=function]
# Two hidden layers of 16 under ReLU, then a tanh on the output.
morphism cell : Feature * Hidden -> Hidden [role=function, param_source=mlp(16, 16), activation=relu, output=tanh]
```

- **Network.** `param_source` selects it: `linear` (the default), `mlp`
  with optional hidden widths (`mlp(16, 16)`; bare `mlp` is two layers of
  64), or `identity`, which has no weights and needs a one-factor domain as
  wide as the codomain. `activation` names the nonlinearity between an
  `mlp`'s hidden layers (`tanh` by default, or `relu`, `sigmoid`,
  `softplus`), and `output` the one applied to the network's output
  (`identity` by default, or `tanh`, `sigmoid`, `softplus`, `relu`).
- **No site.** The function is the Dirac kernel at its output. A program
  calls it as a value, `let z = proj(x)`, with one argument per factor of
  its domain, and never draws from it: `sample z <- proj(x)` is refused.
  In a chain it composes like any morphism, and as the cell of a scan it
  is a deterministic recurrence.
- **Point-estimated weights.** Its weights are parameters of the compiled
  program, fit by whatever objective trains the model around it, as a
  network kernel's are. The checked elaboration reads them as typed program
  inputs (`proj_param_weight`, `proj_param_bias`, and
  `cell_param_layer<k>_weight` and `_bias` for hidden layers), as the
  [learned functions reference](../reference/qvr/probabilistic-programs.md#learned-functions)
  states.

The runtime realization is
[`LearnedFunction`](../api/continuous/deterministic.md#quivers.continuous.deterministic.LearnedFunction).
A constant-noise transition around a learned mean, the form a linear
Gaussian state-space model takes, draws from a family around the
function's value, with the noise scale a site of the scanning program:

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

## Deduction

A `deduction NAME : Domain -> Codomain [options]` declaration with
an indented body declares an agenda-based weighted [deduction](deduction.md#the-framework). See the
[weighted deduction systems guide](deduction.md) for the framework
and the [semiring layer](../api/stochastic/semiring.md) for the
scoring algebra. The seven irreducible parameters of an
agenda-driven deduction (item algebra, rule set, semiring, axiom
source, goal predicate, start symbol, depth bound) become named
fields in the block:

<!-- compile: false -->
```qvr
deduction CCG : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms NP, S, N, VP, PP, Fwd, Bwd, span

    rule fwd_app
        : span(I, K, Fwd(X, Y)), span(K, J, Y)
        |- span(I, J, X)

    rule bwd_app
        : span(I, K, Y), span(K, J, Bwd(X, Y))
        |- span(I, J, X)
```

- **`atoms NAME, NAME, ...`** declares the closed constructor universe.
  In a rule pattern, a declared atom's name always denotes that atom;
  every other identifier (other than a numeral) is a wildcard variable,
  conventionally a single uppercase letter (`X`, `Y`, `I`, `J`, `K`,
  ...). A variable must not share its spelling with an atom: in a
  chart-position slot such as the first two slots of `span`, an atom
  is a compile error naming the rule and the identifier.
- **`rule NAME : premises |- conclusion`** is a sequent. Premises
  are comma-separated; arity is arbitrary (unary rules fire on a
  single chart cell, binary on a pair of adjacent cells, etc.). A
  trailing `#[learnable]` pragma gives the rule a learnable weight:
  one per distinct binding of its category variables, shared across
  every chart position, so a rule whose variables are all positions
  has exactly one weight.
- **`semiring=...`** in the option block selects the scoring
  algebra: `LogProb`, `Viterbi`, `Boolean`, `Counting`.
- **`start=...`** declares the goal-item predicate (the start atom).
- **`depth=...`** bounds the conclusion-category constructor depth
  so the agenda terminates on cyclic rule graphs.
- **`tolerance=...`** (optional) enables Kleene-star convergence
  detection in the chart's aggregation, paired with rule-level
  `#[learnable, bounded]` for contractive-cycle log-weights.

Pattern variables are single uppercase identifiers; every other
identifier in a rule pattern must be listed in `atoms`. A variable
appearing more than once in the same rule must unify across
occurrences.

Slash, tensor, and modal type constructors are *user-declared
atoms*, not built-in syntax: `Fwd(X, Y) ≡ X/Y`, `Bwd(X, Y) ≡ X\Y`,
`Tns(X, Y) ≡ X⊗Y`, `Dia(X) ≡ ◇X`, `Box(X) ≡ □X`,
`Cont(X) ≡ continuation-typed X`. The user is free to introduce
additional constructors for any algebra the deduction reasons over.

### Axiom sources

A deduction needs an axiom-injection kernel that maps an input into
initial weighted chart items. The block admits three forms:

<!-- compile: false -->
```qvr
deduction PCFG : Term -> Term [semiring=LogProb, start=S]
    atoms S, NP, VP, Det, N, V, the, cat, sleeps, span, leaf
    rule branch : span(I, K, B), span(K, J, C) |- span(I, J, A) #[learnable]
    rule anchor : leaf(I, T)                    |- span(I, J, A) #[learnable]

    # Inline lexicon: label-indexed lookup.
    lexicon
        "the"    : Det = the    #[learnable]
        "cat"    : N   = cat    #[learnable]
        "sleeps" : V   = sleeps #[learnable]
```

The three alternatives:

| Form | Use when |
|------|----------|
| `lexicon` block with `"word" : Cat = lf #[learnable]` entries | label-indexed lookup table inline in the body |
| `lexicon from "path.tsv" [learnable=true]` | label-indexed lookup loaded from a TSV at compile time |
| `[axioms=some_morphism]` in the deduction's option block | general kernel `Input -> List(Item × K)` defined as a declared morphism |

Marking a lexicon entry `#[learnable]` allocates an
[`nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Parameter.html)
log-weight initialized to `0.0`.

### Chart-query expressions

The runtime view of a compiled deduction exposes the chart as a
first-class differentiable value with four query methods. Inside a
`program` block (Kleisli-bind sigil `<-`), a deduction call yields
a `chart` value:

<!-- compile: false -->
```qvr
program parse_score : Sentence -> Real [effects=[Sample, Score]]
    sample chart <- CCG(input)
    let w = chart.goal_weight()
    observe valid <- Bernoulli(probs=sigmoid(w))
    return w
```

- `chart.weight(item)`: log-weight of a fully-determined item.
- `chart.enumerate(pattern)`: list of `(item, weight)` pairs
  matching a pattern with wildcards.
- `chart.derivations(item)`: flat enumeration of chart items that
  contributed to `item`'s weight (returned as a `list[Item]`).
- `chart.goal_weight()`: log-weight of the goal predicate.

Each returns a
[`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html)
whose gradients flow back through the agenda's semiring operations
to any `learnable` axiom or rule weight.

## Replicated declarations

Declare N independent copies of a morphism. Each copy has
independent parameters; the base name becomes a group that can be
referenced by [`fan`](#fan-out-diagonal-morphism):

<!-- compile: false -->
```qvr
# creates head_0, head_1, head_2, head_3 with independent parameters
morphism head : Latent -> HeadOut [replicate=4] ~ Normal

# works on every morphism whose role permits replication:
morphism T    : State -> Obs [replicate=3]               # lookup-table kernel
morphism emit : State -> Obs [replicate=3] ~ Normal      # parametric kernel
morphism tok  : Token -> Hidden [role=embed, replicate=2]              # finite-to-Real
```

## Combinators on morphisms

These combinators appear inside `let` declarations to assemble
morphisms from simpler pieces. They live at the expression level,
outside `program` blocks.

### Fan-out (diagonal morphism)

Use fan-out when several operations should see the same input. `f @ g`
expects two inputs as a pair; `fan(f, g)` explicitly supplies one input
to both. The outputs are paired, so a later operation can use them
together. Both calls retain their existing learned parameters.


Copy a single input to N morphisms and concatenate their outputs.
Accepts explicit morphism names or a group name from a replicated
declaration:

<!-- compile: false -->
```qvr
# explicit: fan-out to three named morphisms
define parallel = fan(f, g, h)

# group expansion: fan(head) expands to fan(head_0, head_1, head_2, head_3)
morphism head : Latent -> HeadOut [replicate=4] ~ Normal

define multi_head = fan(head)

# commonly followed by a projection to recombine
morphism proj : Combined -> Latent ~ Normal
define attention = fan(head) >> proj
```

All component morphisms must have the same domain. The output
dimension is the sum of all component codomain dimensions.

### Repeat (iterated composition)

Repeating a transition asks what happens after several steps governed
by the same rule. Its parameters are shared across those steps. Stacking
uses a separate copy at each layer; scanning additionally receives a
new sequence element at each step. These are distinct assumptions about
where a model shares information.


Compose a morphism (or composed expression) with itself N times.
Two forms are available.

**Static repeat.** Count known at compile time, unrolled into a
fixed composition chain:

<!-- compile: false -->
```qvr
# transition >> transition >> transition
define deep = repeat(transition, 3)

# works with composed expressions too
define layer = attn >> residual >> ffn >> residual
define deep_model = repeat(layer, 6)

# repeat(f, 1) = f
define same = repeat(f, 1)
```

**Runtime-variable repeat.** Count omitted, creates a
[`RepeatMorphism`](../api/core/morphisms.md#quivers.core.morphisms.RepeatMorphism)
whose step count is set via `Program.forward(n_steps=N)`. Uses
repeated squaring for $O(\log n)$ compositions:

<!-- compile: false -->
```qvr
morphism transition : State -> State
morphism emission : State -> Obs

# runtime-variable: no count specified
define n_step = repeat(transition) >> emission

export n_step
```

<!-- python: skip -->
```python
prog = load("hmm.qvr")
obs_3 = prog(n_steps=3)    # T^3 >> E
obs_50 = prog(n_steps=50)  # T^50 >> E, same model, different length
```

The morphism's codomain must match its domain (endomorphism) for
repeat to work.

### Stack (independent multi-layer)

Create N independent deep copies of a morphism, each with its own
parameters (no weight-tying):

<!-- compile: false -->
```qvr
# stack creates independent parameters per layer
define deep = stack(transition, 3)  # 3 layers, each with own params

# repeat reuses the same parameters (weight-tying)
define tied = repeat(transition, 3)  # 3 iterations, shared params
```

Unlike `repeat`, which composes a morphism with itself using the
same parameters, `stack(f, N)` creates N fresh deep copies of f,
each with independent learnable parameters. Use `stack` when the
layers should not share weights.

### Scan (temporal recurrence)

Thread hidden state across a sequence using a recurrent cell:

<!-- compile: false -->
```qvr
# Basic syntax: cell has product domain A * H -> H
morphism cell : Embedded * Hidden -> Hidden ~ Normal
define rnn = tok_embed >> scan(cell) >> output_proj

# With learned initial state (default is zeros)
define rnn_learned = tok_embed >> scan(cell, init=learned) >> output_proj
```

The `scan` combinator implements temporal recurrence by threading
hidden state `H` across a sequence:

- **Cell signature.** The morphism passed to `scan` must have a
  product domain `A * H -> H`, where `A` is the input type at each
  step and `H` is the hidden state type. The codomain must equal
  `H` (an endomorphism on the hidden state).
- **Execution.** Given a sequence of inputs `[x_0, x_1, ..., x_T]`
  (implicit in batch-first tensor shape `[batch, seq_len,
  input_dim]`), scan computes:
  - `h_0 = zeros(H)`, a learned initial state (if `init=learned`), or,
    for a scan inside a program, a binding the program makes before the
    scan (`init=h0`)
  - `h_t = cell(x_t, h_{t-1})` for `t = 1..T`
  - Returns the final hidden state `h_T`
- **Type.** If `cell : A * H -> H`, then `scan(cell) : A -> H`.
  The sequence dimension is implicit in the tensor's second
  dimension.
- **Works with every form.** Parametric kernels (`morphism cell : A *
  H -> H ~ Normal`), [learned functions](#function) (`morphism cell : A
  * H -> H [role=function]`, a deterministic recurrence with no site), and
  programs (`program cell(x, h) : A * H -> H`).
  Inside a program, a scan's step runs once per position: a morphism
  cell is drawn at the scan's binding, `h@0`, `h@1`, ..., for
  `sample h <- scan(cell)`, and a program cell's steps read the
  program's bindings made before the scan, their sites are addressed
  `s@0`, `s@1`, ..., and an `observe` in the cell scores one
  observation per position. The
  [scans reference](../reference/qvr/probabilistic-programs.md#scans)
  states the rules.

**Example: vanilla RNN.**

<!-- compile: false -->
```qvr
object Token : FinSet 256
object Embedded : Real 64
object Hidden : Real 128
object Output : Real 64

morphism tok_embed : Token -> Embedded [role=embed]

morphism cell : Embedded * Hidden -> Hidden ~ Normal
morphism output_proj : Hidden -> Output ~ Normal
define rnn = tok_embed >> scan(cell) >> output_proj

export rnn
```

For deeper temporal models, stack multiple scans:

<!-- compile: false -->
```qvr
define deep_rnn = tok_embed >> scan(cell_1) >> scan(cell_2) >> output_proj
```

Each `scan` threads its own hidden state independently.

### Backward composition

Bind a composite morphism with `define`, and write the pipeline in
either direction:

<!-- compile: false -->
```qvr
# forward composition:
define fg = f >> g

# backward composition:
define gf = g << f    # equivalent to f >> g
```

`<<` writes the same [Kleisli composition](https://ncatlab.org/nlab/show/Kleisli+category)
right to left: `g << f` is `f >> g`. Both operators compose in the
operands' shared algebra and reject a mismatch; to cross algebras,
insert an explicit `.change_base` between segments.

### Curry combinators (residuation witnesses)

Currying changes how arguments are supplied. A function of a pair
`(x, y)` can be viewed as a function that takes `x` now and leaves a
place for `y` later. The slash types below record which argument remains
unsupplied and on which side it combines. In this tensor representation
the operation retains the tensor’s entries and reassigns its axes between
the domain and codomain.


The `.curry_right` and `.curry_left` postfix combinators witness
the right- and left-[residuation](https://ncatlab.org/nlab/show/residuated+category)
isomorphisms. For an inner morphism `f : X * Y -> Z`:

| Postfix | Result |
|---------|--------|
| `f.curry_right` | morphism `X -> Z/Y` |
| `f.curry_left`  | morphism `Y -> X\Z` |

Underlying tensor data is unchanged; only the domain / codomain
factoring is reinterpreted. Validity requires the inner morphism's
domain to factor as a non-trivial product.

<!-- compile: false -->
```qvr
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 5
morphism f : X * Y -> Z [role=latent]
define g = f.curry_right    # g : X -> Z/Y
export g
```

The [Lambek calculus](https://ncatlab.org/nlab/show/Lambek+calculus)
inference rules (forward / backward application) become *theorems*
derivable from `identity` + `curry`.

## Top-level `define` declarations

Compose morphisms and bind:

<!-- compile: false -->
```qvr
define fg = f >> g
define par = f @ g
define marg = fg.marginalize(Y)
define composed = f >> g >> h
```

For arithmetic and family-argument `let` expressions inside a
`program` block, see
[Programs and Let-Expressions](dsl-programs-and-lets.md#let-expressions-arithmetic-primitives-and-collections).

### `where` clauses

Attach local `define` bindings to a top-level `define` declaration
using `where`:

<!-- compile: false -->
```qvr
define model = embed >> layers >> output_proj where
    define layers = stack(transition, 3)
    define embed = tok_embed
```

The `where` keyword introduces a block of local definitions scoped
to the parent `define` binding.

## Indexed families and effect declarations

These forms describe stronger contracts than a plain tensor shape.
An indexed family can record that a list is nonempty, permitting a
checked `head` operation. An [effect row](../reference/qvr/effects-and-handlers.md#effect-rows) records requests a computation
may make, such as random sampling or state access. A handler supplies
the interpretation of those requests. The detailed references explain
the notation; ordinary model authors can introduce these forms as
their models need them.


QVR has six QIEC declaration forms. An `index` introduces a closed
index sort and its constructors; a `family` separates type parameters in square
brackets from refinable indices in parentheses; an `effect` declares a
parameterized operation interface; an `instance` gives an applied interface a
lexical identity; a `handler` records coverage and [resumption](../reference/qvr/effects-and-handlers.md#resumption-grades) contracts and
may author its clauses; and a `define` names a typed computation over a row
of instances, which a `program` or another computation calls.

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

The heterogeneous square-bracket telescope admits type, index, and effect-interface
binders; the binder annotation determines its kind. Constructor results must
name their family explicitly, which makes the refined indices available to
case checking. An effect declaration consists of its name, static telescope,
and operation signatures. QVR package releases and serialized QIEC formats
carry their own compatibility identifiers outside the authored effect.

A handler option block may specify `coverage=total|partial`,
`forwards=unknown|none`, and an `introduces=!{...}` row. Each operation clause
assigns a resumption grade: `0` forbids resumption, `aff` permits at most one,
`1` requires exactly one, and `omega` permits multiple resumptions when the
captured context is duplicable. A handler's signature is stable data; its
clause bodies are authored in the source (`[implementation=authored]`, each
clause a `=>` body that may `resume`) or supplied as process-local runtime
attachments (`[implementation=foreign]`). See
[QVR types](../reference/qvr/types-and-indexed-families.md), [effects and
handlers](../reference/qvr/effects-and-handlers.md), and
[computations](../reference/qvr/computations.md) for the user contract; the
[QIEC developer note](../developer/qiec.md) specifies the kernel ABI.

## Structural compression: `signature`, `encoder`, `decoder`, `loss`

The structural-compression declarations represent transformers, tree LSTMs,
GNNs, RNNs, autoregressive language models, and the vector
inside-outside parser as instances of one interface. A
`signature { ... }` block declares sorts and constructors over a
free term algebra. An `encoder { ... }` block declares a
per-constructor parametric function realizing the
[F-algebra homomorphism](https://ncatlab.org/nlab/show/algebra+over+an+endofunctor)
from terms to vectors. A `decoder { ... }` block declares the dual
Kleisli arrow from vectors back to terms. A `loss { ... }` block
attaches a training objective at any site in the graph.

The four blocks are detailed in
[Structural Compression: Signatures and Encoders](structural-compression-signatures-and-encoders.md)
and
[Structural Compression: Decoders and Losses](structural-compression-decoders-and-losses.md).

## Export

Export a morphism as a compiled program output. Any number of
`export` declarations may appear per module; each is compiled into
a separate output:

<!-- compile: false -->
```qvr
export f
export fg
export my_prog
```
