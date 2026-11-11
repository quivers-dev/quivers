# Programs

A probabilistic program specifies random choices and the factors used to
score them. Consider this body fragment, with an observed value `y` supplied
by the caller:

```text
sample location <- Normal(loc=0.0, scale=1.0)
observe y <- Normal(loc=location, scale=0.5)
return location
```

Writing the sampled location as $z$, its unnormalized weight at data $y$ is

$$
p(z)\,p(y\mid z)
=\mathcal N(z;0,1)\,\mathcal N(y;z,0.5^2).
$$

`sample` introduces $z$; `observe` evaluates the likelihood of the supplied
$y$ at that $z$. The supplied $y$ stays fixed while candidate values of $z$ receive different scores. `return`
selects the reported value without discarding the likelihood factor. The
posterior is this weight divided by its integral over $z$, provided that
integral is finite and positive. A forward execution and posterior inference
are thus different operations on the same program.

The following sections make this account compositional: each statement has
a meaning, and the meaning of a block follows from how statements compose.

The `program` block is the monadic sublanguage of QVR. A `program` declaration

```
program P (params) : τ₁ -> τ₂
    s₁
    s₂
    ⋮
    sₙ
    return e
```

denotes an unnormalized weighted kernel (the typing rules for program steps and
their QIEC elaboration live in [Typing §3](typing.md#3-program-contexts) and
[§8](typing.md#8-program-elaboration))

$$
\llbracket P \rrbracket : \llbracket \tau_1 \rrbracket \to \mathcal{M}_+\bigl(\llbracket \tau_2 \rrbracket\bigr),
$$

where $\mathcal M_+$ records the nonnegative weight contributed by
observations and explicit scores. If the body contains neither operation, the
denotation factors through the probability-measure subspace $\mathcal G$ and
is a Markov kernel. A scored program becomes a probability kernel only after
normalization, when its total mass is finite and nonzero.

The block body is a sequence of weighted-kernel operations. `sample`, `let`,
and `return` enter through the probability-kernel embedding; `observe` and
`score` multiply the current measure by a nonnegative factor; and finite
`marginalize` pushes that measure forward along a projection.

For the statistical reading of prior, likelihood, and posterior, see
[@gelman2013bayesian]. For a programming-oriented explanation of monadic
sequencing, see [@wadler1992essence], and for its semantic formulation,
[@moggi1991notions].

## 1. The Giry monad

The Giry monad packages two familiar operations on probability distributions,
[concentrating mass at a value and averaging distributions](../getting-started/concepts.md#6-monads-and-handlers-describe-how-computations-continue).
First, a certain value $s$ can be treated as a distribution that puts all its
mass at $s$; this is the Dirac distribution $\delta_s$, written as the unit
$\eta(s)$. Second, a distribution over distributions can be averaged into
one distribution; this is multiplication $\mu$. For instance, choosing a
coin with probabilities $0.25$ and $0.75$, whose head probabilities are
$0.2$ and $0.8$, gives head probability $0.25(0.2)+0.75(0.8)=0.65$.

A *Kleisli arrow* is a function whose result is a distribution. Composing two
such arrows means passing each possible first result into the second and
averaging its distribution with the first result's weight. The integral
below is the continuous version of the coin calculation. It describes all
possible executions together, even though one runtime execution draws just
one sequence of values.

Let $\mathcal{G}$ denote the Giry monad on $\mathbf{SBor}$, with unit $\eta_S : S \to \mathcal{G}(S)$ given by $s \mapsto \delta_s$ (Dirac at $s$) and multiplication $\mu_S : \mathcal{G}(\mathcal{G}(S)) \to \mathcal{G}(S)$ given by integration. The Kleisli category $\mathbf{Kern}$ of $\mathcal{G}$ has the same objects as $\mathbf{SBor}$ and morphisms $S \to T$ given by Markov kernels $S \to \mathcal{G}(T)$.

Sampling stays in $\mathcal G$. Observation and explicit scoring instead use
nonnegative weighted measures, written $\mathcal M_+$ here. Statements about
finite measures require the relevant integrability conditions in addition
to finite log weights at individual executions. This larger space is necessary because a
continuous density, or an explicit factor $\exp(w)$, may exceed one. Calling
the scored result a sub-probability measure would thus be incorrect.
Probability kernels embed into $\mathcal M_+$, and normalization remains an
inference-layer operation.

A QVR program without scoring is a morphism in $\mathbf{Kern}$, built
compositionally by interpreting each statement as a *Kleisli arrow* and
composing via Kleisli composition $\diamond$. For $k_1 : S \to
\mathcal{G}(T)$ and $k_2 : T \to \mathcal{G}(U)$:

$$
(k_1 \diamond k_2)(s, C) \;=\; \mu_U \bigl( \mathcal{G}(k_2)(k_1(s)) \bigr)(C)
\;=\; \int_T k_2(t, C) \, k_1(s, \mathrm{d}t).
$$

The same composition formula applies to weighted kernels whenever the
resulting integrals are defined. We extend a program-body environment to track
*random variables*: for a pre-fixed program domain $\Gamma$ and a current
statement-context $\Phi = (X_1, \dots, X_k)$, every random variable bound
earlier in the body has a weighted kernel

$$
\rho_{\mathrm{rv}}(v) : \Gamma \to \mathcal{M}_+(\Phi).
$$

The body-level denotation function is

$$
\mathcal{B}\llbracket s_1 \,;\, \cdots \,;\, s_n \,;\, \mathsf{return}\ e \rrbracket : \Gamma \to \mathcal{M}_+(\llbracket \tau_2 \rrbracket).
$$

The probability-measure construction originates with Giry
[@giry1982categorical]. The use of monadic structure to interpret computations
compositionally is developed by Moggi [@moggi1991notions].

## 2. Statements

We give the denotation of each statement form as a Kleisli arrow on the program's accumulated random-variable context $\Phi$. Concretely, the body is interpreted as the Kleisli composite

$$
\mathcal{B}\llbracket s_1; \cdots; s_n; \mathsf{return}\ e \rrbracket
\;=\; \mathcal{S}\llbracket s_1 \rrbracket \diamond \mathcal{S}\llbracket s_2 \rrbracket \diamond \cdots \diamond \mathcal{S}\llbracket s_n \rrbracket \diamond \mathsf{ret}_e,
$$

where each $\mathcal{S}\llbracket s_i \rrbracket : \Phi_{i-1} \to
\mathcal{M}_+(\Phi_i)$ is the weighted arrow assigned to statement $s_i$
(with $\Phi_0 = \Gamma$), and $\mathsf{ret}_e$ is the deterministic Dirac
arrow projecting onto the components named by the `return` clause.

### 2.1 Bind

A bind statement

```
sample v <- F(args)
```

denotes the Kleisli arrow extending the context with a fresh random variable distributed according to family $F$:

$$
\mathcal{S}\llbracket v \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{G}\bigl(\Phi \times \llbracket \mathsf{cod}(F) \rrbracket\bigr),
$$

defined on measurable rectangles $B \times C$ (with $B \subseteq \Phi$, $C \subseteq \llbracket \mathsf{cod}(F) \rrbracket$) by

$$
\mathcal{S}\llbracket v \leftarrow F(\bar a) \rrbracket(\phi,\, B \times C)
\;=\;
\mathbf{1}_B(\phi) \cdot \int_C p_F\bigl( y \,;\, \theta_F(\bar a, \phi) \bigr)\, \mathrm{d}y,
$$

where $\theta_F$ is the family's parameter map (which may depend on previously-bound variables in $\phi$). The operation keeps the current trace $\phi$ and appends a fresh sample from $F$ conditioned on it. The induced action on measures over $\Phi$ is $\mu_{\Phi \times \mathsf{cod}(F)} \circ \mathcal{G}\bigl(\mathcal{S}\llbracket \mathsf{bind} \rrbracket\bigr)$.

A family application supplies one complete keyword schema for $\theta_F$.
Thus `Normal(loc=m, scale=s)` fixes both normal coordinates explicitly, while
`Categorical(probs=p)` selects the probability schema and
`Categorical(logits=l)` selects the logit schema. Positional family arguments,
incomplete schemas, duplicate names, and mixtures of alternative schemas are
undefined and rejected statically. Computation and morphism application remain
positional because their argument order is declared by their signatures.

#### 2.1.1 Destructuring bind

The bind statement admits a *tuple pattern* on the left-hand side:

```
sample (v_1, …, v_m) <- F(args)
```

with denotation identical to the scalar bind above except that the codomain $\llbracket \mathsf{cod}(F) \rrbracket = K_1 \times \cdots \times K_m$ must be an $m$-fold product and the trace is extended with $m$ named coordinates rather than a single one. Subsequent statements may reference each $v_i$ as if it had been bound separately by $v_i \leftarrow \pi_i \circ F(\bar a)$. The two forms have the same denotation when the family's codomain is a product type; the destructuring form gives the components names.

### 2.2 Observe

The [rain posterior](../getting-started/concepts.md#4-a-probabilistic-program-describes-how-data-could-arise) calculates the likelihood update numerically. The [observation-binding guide](../guides/programs.md#the-observations-dict) shows how to supply the fixed values used by QVR.

An observation changes the weight attached to the current choices while
leaving those choices and the supplied observation fixed. In the preceding
Normal example, locations near `y` receive a larger likelihood than distant
locations. Multiplying by this likelihood is what the formula below records;
an inference procedure can use this function to sample or approximate the posterior location.

An observe statement

```
observe v <- F(args)
```

denotes a *score* update against an externally-supplied observed value $v_{\mathrm{obs}}$. As a Kleisli arrow in the finite nonnegative-measure monad $\mathcal{M}_+$,

$$
\mathcal{S}\llbracket \mathsf{observe}\ v \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{M}_+(\Phi),
\qquad
\mathcal{S}\llbracket \mathsf{observe}\ v \leftarrow F(\bar a) \rrbracket(\phi,\, B) \;=\; \mathbf{1}_B(\phi) \cdot p_F\bigl( v_{\mathrm{obs}} \,;\, \theta_F(\bar a, \phi)\bigr).
$$

The trace context is preserved, but the measure is multiplied by the density
or mass of $v_{\mathrm{obs}}$ at $\phi$. That factor need not be bounded by
one. Normalization and posterior inference are deferred to the inference
layer (see [`quivers.inference`](../api/inference/svi.md)).

### 2.3 Let

A let statement

```
let v = expr
```

denotes a *deterministic* extension of the context. The right-hand side `expr` is an arithmetic / function-application expression over previously-bound names; it denotes a measurable map $h : \Phi \to T$, and the let statement is the Kleisli arrow

$$
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket : \Phi \to \mathcal{G}(\Phi \times T),
\qquad
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket(\phi) \;=\; \delta_{(\phi,\, h(\phi))},
$$

a Dirac kernel. Equivalently, on rectangles $B \times C$:

$$
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket(\phi,\, B \times C)
\;=\; \mathbf{1}_B(\phi) \cdot \mathbf{1}_C\bigl(h(\phi)\bigr),
$$

i.e.\ pushforward by $\mathrm{id}_{\Phi} \times h$ realized through the *strength* of the Giry monad.

The arithmetic sublanguage is interpreted standardly: $\mathbb{R}$-valued and $\mathbb{N}$-valued operators denote the corresponding measurable functions on the relevant space, and built-in functions denote the corresponding total measurable maps.

#### 2.3.1 Built-in primitives

The let-expression call form `f(arg, ...)` resolves first against the closed
QIEC primitive registry. Each application is selected by argument type; a
name without a matching signature is a static error. A reduction collapses
the last axis of its argument: a vector reduces to one number, and a tensor of
shape $(n_1, \ldots, n_k)$ to the tensor of shape $(n_1, \ldots, n_{k-1})$
whose entries reduce its rows. A rowwise operation acts along the same last
axis and keeps the shape. Nested reductions therefore collapse one axis each,
innermost first, and `sum(sum(t))` means the two lets `u = sum(t)` and
`s = sum(u)`. The checked elaboration, the reference machine, the compiled
runtime, and every transpile target read these builtins so; a target with no
spelling for a last-axis reduction of a given argument refuses the program
rather than reducing more axes. Reductions over a *named* axis go through the
typed
[`contraction`](../api/dsl/compiler.md) syntax.

| Category | Primitives |
| --- | --- |
| Conversions | `real`, `int`, `weight`, `weight_value` |
| Arithmetic calls | `pow`, `abs`, `min`, `max` |
| Activations | `relu`, `relu6`, `elu`, `selu`, `gelu`, `silu`, `mish`, `softplus`, `logsigmoid`, `softsign`, `sigmoid`, `tanh` |
| Probability-simplex | `softmax`, `log_softmax`, `normalize` |
| Transcendentals | `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `rsqrt`, `square`, `sign`, `reciprocal` |
| Trigonometric / hyperbolic | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh` |
| Rounding | `floor`, `ceil`, `round`, `trunc` |
| Special functions | `erf`, `erfc`, `erfinv`, `lgamma`, `digamma` |
| Reductions | `sum`, `mean`, `min`, `max`, `prod`, `logsumexp` |
| Rowwise | `cumsum`, `sort` |

<!-- compile: false -->
```qvr
# Illustrative: each name below is a let-expression primitive,
# not a top-level morphism, so this block is not standalone-compilable.
softmax(x)         # softmax over the last axis
gelu(x)            # smooth gate
sum(x)             # dim=-1 reduction
```

The eager PyTorch compiler has a larger native table for attachment-backed
workflows. It does not widen the checked semantics: a native-only call makes
QIEC elaboration fail with `qiec-program-gap`. The exact split is listed in
the [let-expression guide](../guides/dsl-programs-and-lets.md#checked-primitive-reference).

#### 2.3.1.1 Finite collection expressions

Let $x = (x_0, \ldots, x_{n-1})$ be a tensor with statically known leading
extent $n$. The pure collection operations have the following denotations:

$$
\begin{aligned}
\llbracket \mathsf{map}(x, f) \rrbracket
  &= (\llbracket f \rrbracket(x_0), \ldots, \llbracket f \rrbracket(x_{n-1})), \\
\llbracket \mathsf{fold}(x, z, f) \rrbracket
  &= f(\cdots f(f(z, x_0), x_1) \cdots, x_{n-1}), \\
\llbracket \mathsf{length}(x) \rrbracket &= n, \\
\llbracket \mathsf{logsumexp\_over}(x, f) \rrbracket
  &= \log \sum_{i=0}^{n-1} \exp(\llbracket f \rrbracket(x_i)).
\end{aligned}
$$

`map` lowers to a typed QIEC comprehension. `fold` is a left fold, written
`fold(xs, init, acc -> item -> body)`, and is elaborated at its fixed extent.
A body that reads the accumulator more than once, or inside a lambda or
`factor`, has each intermediate accumulator bound once by a QIEC
[`LetValue`][quivers.qiec.LetValue] and read through that binding, so the elaborated term is linear in
$n$ however often the body reads the accumulator. A `map` or `fold`
source that is not a variable, and the argument of a lambda-bound name
whose body reads its parameter more than once, are bound the same way.
`filter` remains available to the eager evaluator but has no checked
denotation because its result length depends on values.

If $k : A \to \mathcal M_+(B)$ is the computation named by a `traverse`
lambda, then

$$
\llbracket \mathsf{traverse}(x, k) \rrbracket
  = k(x_0) \mathbin{\diamond} \cdots \mathbin{\diamond} k(x_{n-1})
    \mathbin{\diamond} \eta_{B^n}\langle b_0, \ldots, b_{n-1}\rangle.
$$

The elaborator expands this expression into ordinary calls and binds in source
order. Thus the callee's effects join the enclosing row, and every call keeps
a distinct structural path. The [collection reference](../reference/qvr/collection-expressions.md)
states the surface restrictions and backend boundary.

#### 2.3.2 User-defined callables

Calls in let bodies also resolve against the program's own [`_morphisms`](../api/dsl/compiler.md), [`_encoders`](../api/dsl/compiler.md), [`_decoders`](../api/dsl/compiler.md), [`_deductions`](../api/dsl/compiler.md), and [`_signatures`](../api/dsl/compiler.md) tables. A deterministic [`program`](programs.md) is a [Dirac](https://en.wikipedia.org/wiki/Dirac_measure) [Kleisli arrow](https://en.wikipedia.org/wiki/Kleisli_category) embedding [Smooth](https://en.wikipedia.org/wiki/Smooth_manifold) into [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category)$(\mathcal{G})$; calling it from an encoder rule body composes the two Smooth pieces and stays in Smooth.

```text
signature Seq {
    sorts {
        Seq : object dim 64
        A   : data   dim 64
    }
    constructors {
        Nil  :        -> Seq
        Cons : A, Seq -> Seq
    }
}

encoder C over Seq {
    dim Seq = 64
    Nil                              |-> 0.0
    Cons(head, tail) recurrent state |-> gelu(head + state)
}
```

The dispatcher consults builtins, then user-defined callables, then the user-declared constructor set (for free-term algebra construction). Builtins shadow user-injected names with the same identifier.

#### 2.3.3 Arity and shape checking

At compile time, the let-expression compiler checks the positional arity of every user-defined callable against the call site:

* a [`Morphism`](../api/core/morphisms.md) is unary (takes the domain tensor);
* a [`MonadicProgram`](../api/continuous/programs.md) with named `params` is `len(params)`-ary, otherwise unary;
* any other callable is introspected through [`inspect.signature`](https://docs.python.org/3/library/inspect.html#inspect.signature), counting positional parameters without defaults; `*args` makes the arity unknowable, in which case the check is skipped.

Tensor-shape mismatches inside a user-defined callable initially raise a
PyTorch `RuntimeError`. The dispatcher wraps these errors, together with any
`TypeError`, in a [`CompileError`](../api/dsl/compiler.md) that names the call
site. The result is `call to 'L' failed: ...` rather than a bare PyTorch trace.

### 2.4 Indexed Bind (Plate)

A plate repeats a draw over a [named finite axis](typing.md#32-plates-events-and-fibrations). If `Group` has eight
members, an indexed scalar Normal draw produces eight values, one per group.
They are [independent conditional on the parameters passed to the family](https://pyro.ai/examples/tensor_shapes.html#Declaring-independent-dims-with-plate).
A shared random scale can still make the eight values dependent in the
joint model: a large shared scale makes large magnitudes more likely in
all eight draws. Conditioning on the scale removes that source of dependence.

An indexed bind

```
v : A <- F(args)
```

declares $v$ as an $A$-indexed plate of independent $F$-draws. The per-fiber codomain $K = \mathsf{cod}(F)$ is taken from the family. For a fixed context, the family of coordinate distributions determines the independent product

$$
\mathbf{Kern}(A,K)\longrightarrow\mathbf{Kern}(\mathbf{1},K^A),
\qquad (p_a)_{a\in A}\longmapsto\bigotimes_{a\in A}p_a.
$$

Taking coordinate marginals recovers the $p_a$. Arbitrary joint distributions can have these same marginals with different dependence: two fair bits may be independent or always equal. The plate selects the independent product. The statement thus denotes the context-extending Kleisli arrow

$$
\mathcal{S}\llbracket v : A \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{G}\bigl(\Phi \times K^A\bigr),
$$

with density $\prod_{a \in A} p_F\bigl(v(a) \,;\, \theta_F(\bar a, \phi)\bigr)$ on the appended coordinate.

### 2.5 Indexed Observe

An indexed-observe statement

```
observe r : N <- F(args) [via = idx]
```

denotes a weighted Kleisli arrow in $\mathcal{M}_+$,

$$
\mathcal{S}\llbracket \mathsf{observe}\ r : N \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{M}_+(\Phi),
\qquad
\phi \;\longmapsto\; \mathbf{1}_{(\cdot)}(\phi) \cdot \prod_{n \in N} p_F\bigl( r_{\mathrm{obs}}(n) \,;\, \theta_F(\bar a, n, \phi) \bigr).
$$

Bracket-indexed family arguments `theta[N]` in $\bar a$ pick out the $N$-section of a previously-bound plate variable. The response buffer $r_{\mathrm{obs}} : N \to \llbracket \mathsf{cod}(F) \rrbracket$ is supplied externally by the inference layer; the trace context is preserved and the measure is multiplied by the batched likelihood factor.

The optional `via = idx` entry in the unified option block names a fibration into a grouping plate ([§2.7](#27-grouped-marginalize-with-multi-observe-fibration)); the bare-statement form omits it.

### 2.6 Marginalize

Finite marginalization [evaluates every possible value of a hidden choice](https://pyro.ai/examples/enumeration.html#Mechanics-of-enumeration) [@pyroEnumeration]
and [adds its weighted contribution](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#summing-out-the-responsibility-parameter) [@stanFiniteMixtures]. If $c\in\{0,1\}$ has prior weights
$(0.3,0.7)$ and an observed event has likelihoods $(0.2,0.8)$ under those
choices, its marginal likelihood is $0.3(0.2)+0.7(0.8)=0.62$.
The log contribution is
$\operatorname{logsumexp}(\log0.3+\log0.2,\log0.7+\log0.8)=\log0.62$.
The result no longer contains a value for $c$: both possibilities have
contributed. This is the probability interpretation of the default
`logsumexp` reduction; other reductions select different aggregations.

A scoped marginalize statement

```
marginalize c : A <- F(args)
    s₁
    s₂
    ⋮
    sₖ
```

introduces the coordinate $c$ bound to $F(\bar a)$, optionally $A$-indexed,
with $s_1; \ldots; s_k$ as its scope. For a finite-support latent, the
accumulated weighted measure on $\Phi \times C$ is pushed forward through the
projection $\pi_{\Phi} : \Phi \times C \to \Phi$:

$$
\mathcal{S}\llbracket \mathsf{marginalize}\ c \rrbracket : \mathcal{M}_+(\Phi \times C) \to \mathcal{M}_+(\Phi),
\qquad
\nu \;\longmapsto\; \pi_{\Phi *} \nu.
$$

Operationally, QVR implements this pushforward exactly for enumerable families
such as `Categorical` and Bernoulli-family supports. It combines prior and
scope weights across the support with the requested reduction. For a family
without finite support, the current elaborator samples once and runs the
scope with that value. This path contributes the score of one sampled
configuration. Computing a continuous marginal would require integrating
these weighted scores across the hidden variable. The elaborator rejects
`logsumexp`, `sum`, and `mean` on this path. After the scope closes, $c$ falls out of scope in either path.

The four bind variants, scalar, indexed, scored, marginalized, are uniformly a single underlying step with a `mode ∈ {sample, score, marginal}` tag and an optional index `A`. The scalar/plate axis is orthogonal to the probability/weighted-measure distinction.

The statistical operation is summing out a discrete latent variable, as in
mixture models [@bishop2006pattern]. The grouping below specifies which
likelihood factors must be combined before that sum.

### 2.7 Grouped marginalize with multi-observe fibration

Grouping determines [which observations share a hidden choice](https://pyro.ai/examples/enumeration.html#Dependencies-among-plates) [@pyroEnumeration].
[Transpilation must preserve this grouping](transpile-correctness/index.md#5-plates-marginalization-and-via). Suppose
three rows map to groups through `idx = [0, 0, 1]`. The first two rows share
one latent class, while the third has its own group's class. For group zero,
the required likelihood is

$$
\sum_k \pi_{0k}\,p(y_0\mid k)\,p(y_1\mid k).
$$

Marginalizing each row separately would instead give
$(\sum_k\pi_{0k}p(y_0\mid k))(\sum_k\pi_{0k}p(y_1\mid k))$,
which allows different classes for the two rows. The `via` map prevents that
change of model: first add row log likelihoods within each group and class,
then apply `logsumexp` over classes. A *fiber* of the map is the set of
rows assigned to one group.

The marginalize step admits a *grouping clause* that turns the body into a fibered scoring problem over a shared plate. The clause lives in the marginalize step's option block, alongside an optional reduction selector; the body's observe steps each carry their own `via = idx` fibration in their option blocks:

```
marginalize c : K <- F(args) [over = G, reduction = R]
    observe r_1 : N_1 <- F_1(...) [via = idx_1]
    observe r_2 : N_2 <- F_2(...) [via = idx_2]
    ⋮
```

with a single `over = G` entry on the marginalize step (or `over = product(G_1, G_2, …)` / `over = [G_1, G_2, …]` for a product grouping plate) and a `via = idx_m` entry on each observe naming a fibration $\iota_m : N_m \to G$ from that observe's response plate into the shared grouping plate.

The row maps $\iota_m:N_m\to G$ determine which likelihoods share a group. After accumulating those likelihoods for each pair $(g,k)$, the operation reduces the class coordinate. In particular, the default log-sum-exp reduction implements the finite pushforward that sums weighted alternatives. Its log-scale contribution is:

$$
\Sigma_g \;\mathrm{aggr}_R\!\!\bigl[\log \pi(g, k) + \textstyle\sum_m \sum_{n \,:\, \iota_m(n) = g} \ell_m(n, k)\bigr],
$$

where $\ell_m(n, k) = \log p_{F_m}\bigl(r^{\mathrm{obs}}_m(n);\, \theta_m(n, k, \phi)\bigr)$ is the per-row per-class log-likelihood of observe $m$, $\pi$ is the per-group per-class prior weight, and $\mathrm{aggr}_R \in \{\mathrm{logsumexp}, \mathrm{sum}, \mathrm{mean}\}$ is the reduction selected by the optional `reduction = R` annotation (default `logsumexp`, the canonical mixture-marginalization form).

The product-grouping case `over G_1 * G_2 * …` paired with `via product(idx_1, idx_2, …)` on each observe extends the grouping target to a flat plate of cardinality $\prod_i |G_i|$; the source arity must match. The flat position of a row is the row-major combination of its factor indices, $\iota(n) = \sum_i \iota_i(n) \prod_{j > i} |G_j|$, so the last factor varies fastest.

A grouped block may nest inside another. The inner block then contributes one aggregate per position of its own group to the outer group's accumulator rather than a single number, and its group must stand in one of two relations to the outer group $G$: it is an axis of the same extent, identified with $G$ position by position, or it is a product $G \times H$ with $G$ as a factor, in which case the inner per-position aggregates are summed along the projection $G \times H \to G$ and any inner argument indexed by $G$ (a prior `theta[z]` selected by the outer latent) is pulled back along that projection. A `sample` inside a block that reads nothing the block binds is drawn once before the block, since every value of the latent shares it; a draw whose arguments read the latent would be a draw per class and is rejected.

### 2.8 Effect signatures

A `program` declaration may carry an *effect signature* via the `effects` entry of its option block:

```
program P (params) : τ₁ -> τ₂ [effects = [E₁, E₂, …]]
    body
```

where each $E_i$ is one of $\{\mathsf{Sample}, \mathsf{Score}, \mathsf{Marginal}, \mathsf{Pure}\}$. The signature is a *static type* over the program: a subset of an *effect algebra* $\mathcal{E}$ that the body's statements collectively produce.

Each statement form contributes an effect:

| Statement form | Effect produced |
|---|---|
| `sample v <- F(args)` | $\mathsf{Sample}$ |
| `sample v : A <- F(args)` | $\mathsf{Sample}$ |
| `observe v <- F(args)` | $\mathsf{Score}$ |
| `observe r : N <- F(args)` | $\mathsf{Score}$ |
| `marginalize c <- F(args) [over = G, reduction = R]` (with indented scope body) | $\mathsf{Marginal}$ |
| `let v = expr` | no capability |
| `score v = expr` | $\mathsf{Score}$ |
| `return e` | no capability |

The compiler computes the *actual* capability set $\mathcal{E}(P)$ of the
lowered body and verifies $\mathcal{E}(P) \subseteq
\mathcal{E}_{\mathrm{decl}}$. `Pure` is a declaration sentinel rather than an
effect contributed by `let` or `return`; it requires the actual set to be
empty and thus rejects every `sample`, `observe`, `score`, or
`marginalize` step.

Categorically, the summary distinguishes deterministic maps, probability
kernels, weighted kernels in $\mathcal M_+$, and finite-support pushforwards.
The elaborated QIEC row is the precise effect type: it records the lexical
`Random` and `Score` instances and any effects of reachable calls. The
four-name effect summary provides a coarser check on the program body.

The `over = <model>` entry in a program's option block marks the program as consuming the named model's latents: the consumed coordinates appear as data parameters and the body is restricted to $\mathsf{Pure}$ (a *posterior consumer*, the deterministic Kleisli arrow $\Theta \to \tau_2$ that lifts to $\mathrm{Data} \to \mathcal{G}(\tau_2)$ by post-composition with the model's posterior kernel).

### 2.9 Indexed Gather (Let-Pullback)

A gather uses an index array to reuse already-bound values. If group effects
are `v = [10, 20]` and row-to-group indices are `[0, 0, 1]`, then `v[idx]`
is `[10, 10, 20]`. The first two rows share the same realized group effect;
they retrieve its already realized value. The pullback notation
below describes this reindexing of the shared random quantity.

A `let` right-hand side of the form `arr[idx]` is the *Kleisli pullback*. For a plate variable $v : A \to \mathcal{G}(B)$ bound earlier in the body, and a finite fibration $\iota : N \to A$ named in the context, the gather $\iota^* v$ is the composite

$$
\iota^* v \;=\; v \circ \iota \;:\; N \to \mathcal{G}(B).
$$

Interpreted as a deterministic measurable map on the accumulated context (because $v$ has already been realized as a tensor $A \to B$ in the trace), the let-step denotes the Dirac extension

$$
\mathcal{S}\llbracket \mathsf{let}\ w = \mathit{arr}[\mathit{idx}] \rrbracket(\phi)
\;=\;
\delta_{(\phi,\, \phi.\mathit{arr}[\phi.\mathit{idx}])}.
$$

### 2.10 Score (factor)

A score is supplied on the log scale. [Adding $w$ to the log joint](https://docs.webppl.org/en/master/inference/conditioning.html#factor) multiplies
the ordinary weight by $\exp(w)$; a score of zero leaves it unchanged.
This permits factors that do not arrive through a named observation family,
including [a chart’s goal weight](grammar.md#9-chart-access-from-program-bodies).
It also means an arbitrary score need not define a normalized probability
distribution, or even one with a finite normalizing integral.

A score statement

```
score v = expr
```

denotes a Kleisli arrow that simultaneously (i) binds the value of `expr` to a fresh name `v` in the program trace, and (ii) adds that value to the program's running log-joint. As a weighted Kleisli arrow in $\mathcal{M}_+$,

$$
\mathcal{S}\llbracket \mathsf{score}\ v = \mathit{expr} \rrbracket : \Phi \to \mathcal{M}_+(\Phi \times T),
\qquad
\mathcal{S}\llbracket \mathsf{score}\ v = \mathit{expr} \rrbracket(\phi,\, B \times C) \;=\; \mathbf{1}_B(\phi) \cdot \mathbf{1}_C\bigl(h(\phi)\bigr) \cdot \exp\bigl(h(\phi)\bigr),
$$

where $h : \Phi \to \mathbb{R}$ is the measurable map denoted by `expr`
(which must denote a scalar log weight). The trace context is extended with
the named coordinate, and the resulting finite measure is multiplied by
$\exp(h(\phi))$, a factor that may exceed one.

Score is the Kleisli pendant of [observe](#22-observe), with the log-density supplied directly by an expression rather than via a family's `log_prob` against an externally-supplied observed value. The canonical use is to lift an arbitrary differentiable tensor expression (typically a deduction's chart goal weight, see [Weighted Deduction Fragment §9](grammar.md#9-chart-access-from-program-bodies)) into the program's log-joint. In particular, with $\mathit{expr} = \mathit{chart}.\mathsf{goal\_weight}()$ the program's log-joint matches the sentence's inside log-marginal under the referenced deduction.

The effect contributed by a score step is $\mathsf{Score}$ (the same as `observe`), and the soundness condition $\mathcal{E}(P) \subseteq \mathcal{E}_{\mathrm{decl}}$ in [§2.8](#28-effect-signatures) applies unchanged.

### 2.11 Return

A return statement

```
return e
```

closes the body. If $e = (v_1, \dots, v_m)$ is a tuple of bound names, the return clause is the deterministic Kleisli arrow

$$
\mathsf{ret}_e : \Phi_n \to \mathcal{G}\bigl(\llbracket \tau_2 \rrbracket\bigr),
\qquad
\mathsf{ret}_e(\phi) \;=\; \delta_{\pi_{v_1, \dots, v_m}(\phi)},
$$

where $\pi_{v_1, \dots, v_m} : \Phi_n \to \llbracket \tau_2 \rrbracket$
projects the trace onto the named coordinates. Composing with the body chain
pushes the joint weighted measure onto those coordinates.

A bare-tuple return `return (x, y)` projects the trace onto the named coordinates; the resulting product space's components are ordered by tuple position.

## 3. Data parameters

A program declared with bare-identifier parameters

```
program P (q₁, …, qₖ) : τ₁ -> τ₂
    body
```

names the components of the domain $\tau_1$: when $\tau_1 = \sigma_1 \times \cdots \times \sigma_k$ is a $k$-fold product, each $q_i$ binds to the projection $\pi_i$ of the input. The denotation is unchanged from the unparameterized form,

$$
\llbracket P \rrbracket : \llbracket \tau_1 \rrbracket \to \mathcal{M}_+(\llbracket \tau_2 \rrbracket),
$$

that is, a single weighted kernel whose input components can be referenced
by the names $q_i$. Typed parameters, covered in
§3a below, extend this to dependent kernel families.

## 3a. Parametric programs

A program whose parameter list contains *typed* parameters denotes a *dependent* family of Kleisli arrows. With parameters $p_i : P_i$ drawn from the universes

| Parameter kind | Universe $P_i$ |
|---|---|
| `FinSet`, `Space`, `Object` | an object of the relevant subcategory of $\mathbf{Kern}$ |
| `Real`, `Nat` | a hom-object of scalar type (a hyperparameter) |
| `Mor[A, B]` | the hom-set $\mathbf{Kern}(A, B)$ |

the denotation lives in the dependent kernel space

$$
\llbracket P \rrbracket \;:\; \prod_{p_1 : P_1} \cdots \prod_{p_k : P_k} \mathbf{Kern}\bigl(\mathrm{dom}(p), \mathrm{cod}(p)\bigr),
$$

an object of the indexed family of Kleisli arrows over the parameter category. The domain and codomain may themselves mention the formal parameters, so each fiber is a kernel between possibly-different objects of $\mathbf{Kern}$.

### Inline expansion as substitution

A call site `v <- P(a₁, …, aₖ)` inside another program is interpreted by *substitution* on the dependent denotation: the actual arguments $a_i$ are substituted for the formal parameters $p_i$ in the body of $P$, yielding a closed Kleisli arrow which is then inlined as a sequence of statements into the caller's body. Internal latents are α-renamed under a fresh prefix $v\$$, and the return-variable is renamed to $v$ directly; the result is a well-typed sequence of caller-level Kleisli arrows.

This is sound by a standard substitution lemma: because each formal parameter is bound at the top of the body and the body interprets to a Kleisli arrow built compositionally from its statements, substitution commutes with the body's denotation function $\mathcal{B}\llbracket \cdot \rrbracket$. Consistent α-renaming preserves the body's binding structure, so references continue to denote the same bound values. Two call sites of the same template thus contribute *distinct* factors to the caller's joint kernel, fresh latents per use, giving each call its own latent variables in a hierarchical model.

## 4. Composition of programs

Two programs $P : X \to Y$ and $Q : Y \to Z$ compose by Kleisli composition:

$$
\llbracket P \mathbin{>\!\!>} Q \rrbracket(x, C) \;=\; \int_Y \llbracket Q \rrbracket(y, C) \, \llbracket P \rrbracket(x, \mathrm{d}y).
$$

The DSL exposes this through a top-level `define` binding and an `export` declaration naming the composite:

<!-- compile: false -->
```qvr
define pq = p >> q
export pq
```

`export` is the public-binding form. A top-level `define N = e` declaration is the corresponding private binding: it extends the morphism environment with $N \mapsto \llbracket e \rrbracket_\rho$, but $N$ is not part of the module's compiled output unless additionally exported. A `where` block may contain nested `define` declarations; the compiler processes those bindings before the outer expression. Formally, for a module $M$ with $\mathrm{export}\ E_1, \dots, \mathrm{export}\ E_k$ declarations,

$$
\llbracket M \rrbracket_{\mathrm{exports}} \;=\; \bigl(\llbracket E_1 \rrbracket,\, \dots,\, \llbracket E_k \rrbracket\bigr),
$$

a tuple of compiled morphisms / posteriors / deductions. The expression $E_i$ may be any value-level expression: a top-level morphism name, a program name, a deduction name, an encoder / decoder name, or a `define`-bound composite. The denotation of each export is the denotation of the underlying expression; `export` adds that value to the module output.

## 5. Algebraic laws and their scope

The interpretations above satisfy the standard monadic equations:

| Equation | Statement |
|---------|-----------|
| Left unit | $\eta \diamond k = k$ |
| Right unit | $k \diamond \eta = k$ |
| Associativity | $(k_1 \diamond k_2) \diamond k_3 = k_1 \diamond (k_2 \diamond k_3)$ |
| Strength and symmetry | $\mathcal G(\sigma_{S,T})\circ\mathrm{str}_{S,T}=\mathrm{str}\prime_{T,S}\circ\sigma_{S,\mathcal G(T)}$ |

These equations apply directly to the unscored fragment interpreted in
$\mathcal G$. The corresponding equations for scored programs apply to
well-defined weighted-kernel composites. Normalization additionally requires
finite, positive total mass. The
order of independent draws is irrelevant to the denotation by symmetry of the
product measure; dependent statements and effectful calls remain ordered.

**Theorem (Monad laws for the generative fragment).** *The Kleisli
composition $\diamond$ of [Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels)
on the probability kernels produced by `sample`, `let`, and `return` satisfies
the four equations above. Finite marginalization is ordinary pushforward and
thus respects equality of measures.*

**Proof.** The first three equations are the Kleisli laws of the Giry monad; the fourth relates its strength to the symmetry of products [@giry1982categorical; @kock1972strong]. $\mathcal{G}$ is a monad on $\mathbf{SBor}$ with unit $\eta_S(s) = \delta_s$ (the Dirac at $s$) and multiplication $\mu_S(M)(B) = \int_{\mathcal{G}(S)} m(B)\, M(\mathrm{d}m)$ (integration of measures over measures). Kleisli composition $k_1 \diamond k_2 = \mu \circ \mathcal{G}(k_2) \circ k_1$ satisfies the unit and associativity equations induced by the monad laws. Concretely:

* **Left unit.** $\eta_S \diamond k = \mu_T \circ \mathcal{G}(k) \circ \eta_S = \mu_T \circ \eta_{\mathcal{G}(T)} \circ k = k$, where the second equality is the naturality of $\eta$ and the third is the monad's left-unit law $\mu \circ \eta_{\mathcal{G}} = \mathrm{id}$.
* **Right unit.** $k \diamond \eta_T = \mu_T \circ \mathcal{G}(\eta_T) \circ k = \mathrm{id}_{\mathcal{G}(T)} \circ k = k$, by the monad's right-unit law $\mu \circ \mathcal{G}(\eta) = \mathrm{id}$.
* **Associativity.** A direct calculation: $((k_1 \diamond k_2) \diamond k_3)(s) = \mu \circ \mathcal{G}(k_3) \circ (\mu \circ \mathcal{G}(k_2) \circ k_1)(s)$. Naturality of $\mu$ rewrites this as $\mu \circ \mu_{\mathcal{G}} \circ \mathcal{G}(\mathcal{G}(k_3)) \circ \mathcal{G}(k_2) \circ k_1$, and the monad's associativity law $\mu \circ \mu_{\mathcal{G}} = \mu \circ \mathcal{G}(\mu)$ rewrites it to $\mu \circ \mathcal{G}(\mu) \circ \mathcal{G}(\mathcal{G}(k_3)) \circ \mathcal{G}(k_2) \circ k_1 = \mu \circ \mathcal{G}(\mu \circ \mathcal{G}(k_3) \circ k_2) \circ k_1 = k_1 \diamond (k_2 \diamond k_3)(s)$.
* **Strength coherence.** $\mathcal{G}$ is a *commutative strong* monad on $\mathbf{SBor}$. The right strength $\mathrm{str}_{S, T} : S \times \mathcal{G}(T) \to \mathcal{G}(S \times T)$ is given on rectangles by $(s, m) \mapsto \delta_s \otimes m$, and the left strength $\mathrm{str}'_{S, T} : \mathcal{G}(S) \times T \to \mathcal{G}(S \times T)$ symmetrically by $(m, t) \mapsto m \otimes \delta_t$; both are measurable. Here $\sigma_{S,T}$ swaps the two coordinates. Pushing $\delta_s\otimes m$ forward along this swap gives $m\otimes\delta_s$, which proves the row relating strength and symmetry directly. The deeper *commutativity* property of $\mathcal{G}$ — the equality, as maps $\mathcal{G}(S) \times \mathcal{G}(T) \to \mathcal{G}(S \times T)$, of the two double-strength composites

$$
\mathrm{dst}_1 \;=\; \mu_{S \times T} \,\circ\, \mathcal{G}(\mathrm{str}_{S, T}) \,\circ\, \mathrm{str}'_{S, \mathcal{G}(T)}
\qquad\text{and}\qquad
\mathrm{dst}_2 \;=\; \mu_{S \times T} \,\circ\, \mathcal{G}(\mathrm{str}'_{S, T}) \,\circ\, \mathrm{str}_{\mathcal{G}(S), T}
$$

— is what licenses reordering of independent draws in a QVR program body without changing the denotation. For $\mathcal{G}$ on $\mathbf{SBor}$, $\mathrm{dst}_1 = \mathrm{dst}_2$ reduces exactly to Fubini–Tonelli on the product $\sigma$-algebra. Kock [@kock1972strong] provides the general categorical background for this notion of commutativity.

$\square$

## 6. Inference and conditioning

In the opening example, fixing `y` defines a function of the unknown
location $z$. Inference uses that function to characterize the distribution
of $z$ after observing `y`. Exact integration, finite enumeration, MCMC,
and variational approximation are different ways of answering this question.
A scored program supplies an unnormalized weight for each candidate latent
configuration. If these weights have finite, positive total mass, dividing
by that mass defines the posterior. The
[`quivers.inference`](../api/inference/svi.md) module uses trace scores to
approximate posterior expectations or fit a variational distribution.

For variational approximation and stochastic optimization of its objective,
see [@blei2017variational; @hoffman2013svi]. For Hamiltonian Monte Carlo and
its adaptive No-U-Turn variant, see [@neal2011mcmc; @hoffman2014nuts]. These
methods operate on the model density described above.

## 7. Bayesian lifts

A `program` whose body lacks explicit `sample` priors on its
learnable parameters still has a well-defined weighted-kernel denotation
(it is a parameterized arrow, with the parameters held
fixed). To pass such a program to the SVI / NUTS layer, which
operates on programs with explicit priors, the
[`quivers.inference.lifts`](../api/inference/lifts.md) module exposes four lifts. Each lifts a
parameter-bearing artifact into a [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) of the standard
sample-then-score shape and admits a precise semantic statement.

Throughout this section let
$\theta \in \mathbb{R}^{D}$ denote the inner artifact's
learnable parameters (the flattened, concatenated tensor running
over [`nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Parameter.html) leaves), let $\sigma_{\theta} > 0$ be a fixed
prior scale, and let $\mathbf{z}$ name an optional collection of
intermediate latents lifted into the program by the caller.

### 7.1 Parameter-only Bayesian lift

[`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) takes a parameter-bearing
[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) `inner_model` exposing
$\log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta)$ via a
`log_joint(x, observations)` method, plus an optional
`additional_latents` map naming intermediate latents
$\mathbf{z}$ together with their shapes. It returns a lifted
program whose log-joint equals

$$
\log \pi_{\mathrm{lift}}(\theta, \mathbf{z})
\;=\;
\log \mathcal{N}(\theta;\, 0,\, \sigma_{\theta}^{2} I_{D})
\;+\;
\log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta).
$$

**Proposition (parameter lift, exactness of placeholder cancellation).**
Let $\sigma_{z} > 0$ be the placeholder scale supplied via
`latent_placeholder_scale`. The lifted program declares one
sample site $\theta_{i} \sim \mathcal{N}(0, \sigma_{\theta}^{2})$
per parameter and one sample site
$\mathbf{z}_{j} \sim \mathcal{N}(0, \sigma_{z}^{2} I)$ per
declared additional latent. Its score step computes

$$
S(\theta, \mathbf{z})
\;=\;
\log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta)
\;-\;
\sum_{j} \log \mathcal{N}\bigl(\mathbf{z}_{j};\, 0,\, \sigma_{z}^{2} I\bigr).
$$

The total log-density assembled by the inference layer (sum of
all sample-site log-priors plus the score step) is

$$
\log \pi_{\mathrm{lift}}(\theta, \mathbf{z})
\;=\;
\underbrace{\log \mathcal{N}(\theta; 0, \sigma_{\theta}^{2} I_{D})}_{\text{parameter prior}}
\;+\;
\underbrace{\sum_{j} \log \mathcal{N}(\mathbf{z}_{j}; 0, \sigma_{z}^{2} I)}_{\text{placeholder priors}}
\;+\;
S(\theta, \mathbf{z}),
$$

which by inspection cancels the placeholder priors and leaves
$\log \mathcal{N}(\theta; 0, \sigma_{\theta}^{2} I_{D}) + \log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta)$ pointwise.

*Proof.* The score step subtracts exactly the sum of placeholder
log-priors that the sample-site declarations add, so the algebra
is an identity. Gradient flow back to $\theta$ is realized by
the [`_swap_named_parameters`](../api/inference/lifts.md) context manager, which
temporarily writes the override tensor into the parent module's
`_parameters` dict so that downstream attribute reads return the
override; because the override is a non-Parameter tensor with
its own autograd graph, gradients propagate back through it to
the surrounding NUTS / SVI latents. $\square$

**Corollary (target distribution).** Under the bijection
$\theta \mapsto \theta$, $\mathbf{z} \mapsto \mathbf{z}$
([`NUTS`](../api/inference/mcmc.md) operates on the unconstrained
$\mathbb{R}^{D + \sum_{j} \dim \mathbf{z}_{j}}$),

$$
\pi_{\mathrm{lift}}(\theta, \mathbf{z})
\;\propto\;
p(\theta) \, p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta),
$$

the [Bayesian posterior](https://en.wikipedia.org/wiki/Posterior_probability)
$p(\theta, \mathbf{z} \mid x, y)$ under the
$\mathcal{N}(0, \sigma_{\theta}^{2} I)$ prior on $\theta$. The
placeholder scale $\sigma_{z}$ is denotationally irrelevant (it
cancels exactly) and operationally affects only NUTS
mass-matrix adaptation during warmup.

### 7.2 Deterministic-morphism lift via an observation family

[`lift_to_bayesian_program`](../api/inference/lifts.md#quivers.inference.lifts.lift_to_bayesian_program) takes a parameter-bearing
deterministic [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), a
location-extraction callable
$\mu : \mathcal{X} \to \mathcal{M}$, an observation family
$\mathcal{F} \subset \mathbf{Dist}$ (any
[`torch.distributions.Distribution`](https://docs.pytorch.org/docs/stable/distributions.html) subclass), and a key
$k$ naming the observed coordinate in the observations dict.
The lift composes (i) the parameter-only Bayesian lift of §7.1
against (ii) a synthetic `log_joint` that scores the data under
$\mathcal{F}\bigl(\mu(x);\, \boldsymbol{\eta}\bigr)$ at the
observed value $y = \mathrm{observations}[k]$, where
$\boldsymbol{\eta}$ collects the family's remaining parameters
(scale, concentration, total_count, etc.) from the caller's
`observation_kwargs`.

**Proposition (observation-family lift).** The lifted program's
log-joint equals

$$
\log \pi_{\mathrm{lift}}(\theta)
\;=\;
\log \mathcal{N}(\theta;\, 0,\, \sigma_{\theta}^{2} I_{D})
\;+\;
\log p_{\mathcal{F}}\bigl(y \,\big|\, \mu(x;\, \theta),\, \boldsymbol{\eta}\bigr),
$$

i.e. the [Bayesian posterior](https://en.wikipedia.org/wiki/Posterior_probability)
of $\theta$ under the chosen
likelihood, modulo the
$\theta$-independent normalizer of $\mathcal{F}$.

*Proof.* The lift is implemented as a delegating wrapper whose
`log_joint(x, obs)` returns $\log p_{\mathcal{F}}(y \mid \mu(x), \boldsymbol{\eta})$ (reduced over event axes), then
delegates to [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) without additional latents.
By §7.1, the lifted log-density equals
$\log p(\theta) + \log p_{\mathrm{wrapper}}(y \mid x, \theta)$,
and the wrapper's `log_joint` is exactly the family's
log-density. $\square$

### 7.3 Direct log-prob lift

[`lift_from_log_prob`](../api/inference/lifts.md#quivers.inference.lifts.lift_from_log_prob) is the variant of §7.2 for
modules that already expose
$\log p_{\mathrm{inner}}(y \mid x, \theta)$ directly (without
going through a family / location split). Given a
parameter-bearing module and a callable
$\ell : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ that
returns the conditional log-density, the lift produces a program
whose log-joint equals

$$
\log \pi_{\mathrm{lift}}(\theta)
\;=\;
\log \mathcal{N}(\theta;\, 0,\, \sigma_{\theta}^{2} I_{D})
\;+\;
\ell(x, y;\, \theta).
$$

The proof is the §7.1 argument with the wrapper's `log_joint`
defined as $\ell$ directly. The lift exists so callers that have
already written a `log_prob`-style method (a [`SampledComposition`](../api/continuous/morphisms.md)
over a Normal kernel, an encoder-decoder composition with a
closed-form ELBO term, etc.) do not have to repackage it through
a family interface.

### 7.4 Monte-Carlo conditional-likelihood wrapper

[`monte_carlo_log_joint`](../api/inference/lifts.md#quivers.inference.lifts.monte_carlo_log_joint) wraps a `MonadicProgram` whose
body contains one or more named intermediate `sample` steps
$\mathbf{z}$ that the caller does not want to enumerate as
NUTS latents. The wrapper's `log_joint(x, observations)`
forward-draws each $\mathbf{z}_{j}$ from its declared family at
$x$, merges the draw into the observations dict, calls the
inner's `log_joint`, and subtracts the drawn latents' prior
contributions so the residual is the *conditional* data
likelihood.

**Definition.** Let
$q_{j}(\mathbf{z}_{j} \mid x, \theta) = p_{\mathrm{inner}}(\mathbf{z}_{j} \mid x, \theta)$ denote the
sample-step family of the inner program at site $j$. The
wrapper realizes the random map

$$
\widetilde{\ell}(x, y;\, \theta)
\;=\;
\log p_{\mathrm{inner}}\bigl(y \,\big|\, \mathbf{z}_{*},\, x,\, \theta\bigr),
\qquad
\mathbf{z}_{j*} \sim q_{j}(\cdot \mid x, \theta).
$$

**Proposition (single-sample MC bias).** The wrapper's output is
a single-sample [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimator of the conditional
log-likelihood, *not* of the marginal log-likelihood
$\log p_{\mathrm{inner}}(y \mid x, \theta) = \log \mathbb{E}_{\mathbf{z} \sim q}[p_{\mathrm{inner}}(y \mid \mathbf{z}, x, \theta)]$. By [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)
applied to the concave logarithm,

$$
\mathbb{E}_{\mathbf{z}_{*} \sim q}\bigl[\widetilde{\ell}(x, y;\, \theta)\bigr]
\;\le\;
\log p_{\mathrm{inner}}(y \mid x, \theta),
$$

with equality only when $p_{\mathrm{inner}}(y \mid \mathbf{z}, x, \theta)$ is constant in $\mathbf{z}$. The
wrapper's output is thus biased downward as an estimator
of the marginal log-likelihood.

*Proof.* Jensen applied pointwise in $(x, y, \theta)$ to the
random variable $p_{\mathrm{inner}}(y \mid \mathbf{z}, x, \theta)$ under $q$:
$\log \mathbb{E}_{\mathbf{z}}[p(y \mid \mathbf{z})] \ge \mathbb{E}_{\mathbf{z}}[\log p(y \mid \mathbf{z})]$. $\square$

**Soundness for SVI; unsoundness for NUTS.** The reparameterized
families ([`Normal`](https://docs.pytorch.org/docs/stable/distributions.html#normal), [`MultivariateNormal`](https://docs.pytorch.org/docs/stable/distributions.html#multivariatenormal),
[`LowRankMultivariateNormal`](https://docs.pytorch.org/docs/stable/distributions.html#lowrankmultivariatenormal), etc.) yield a pathwise-
differentiable $\mathbf{z}_{*}(\xi; x, \theta)$ in an auxiliary
noise $\xi$. If differentiation can be interchanged with expectation, the
reparameterization identity [@kingma2014autoencoding] gives

$$
\nabla_{\theta} \mathbb{E}_{\mathbf{z}}[\log p(y \mid \mathbf{z}, \theta)]
= \mathbb{E}_{\xi}[\nabla_{\theta} \log p(y \mid \mathbf{z}_{*}(\xi; \theta), \theta)].
$$

Under these regularity conditions, one independent Monte Carlo draw supplies
an unbiased gradient of the wrapper’s expected objective. This objective is
a lower bound on the marginal log likelihood by Jensen’s inequality. A
convergence argument additionally needs a specified optimization schedule
and assumptions controlling the objective and gradient noise
[@robbins1951stochastic].

For NUTS, the wrapper is unsuitable: re-drawing
$\mathbf{z}_{*}$ on every leapfrog evaluation makes the
Hamiltonian energy stochastic, invalidates the usual deterministic Hamiltonian proposal and its acceptance calculation [@hoffman2014nuts; @neal2011mcmc]. For
NUTS, enroll $\mathbf{z}$ as an additional NUTS latent via
[`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters)
with `additional_latents={'<name>': <shape>}` (§7.1), so that
$(\theta, \mathbf{z})$ are jointly sampled from a joint target with a deterministic
log density at each evaluation.

### 7.5 Bayesian lift for weighted deduction systems

[`nuts_program_from_deduction`](../api/stochastic/deduction/bayes.md#quivers.stochastic.deduction.bayes.nuts_program_from_deduction) is the
deduction-system specialization of §7.1. Given a
[`DeductionSystem`](../api/stochastic/deduction.md#quivers.stochastic.deduction.DeductionSystem)
$D$ with learnable log-weights $\mathbf{w} \in \mathbb{R}^{D}$
and a corpus $\{s_{n}\}_{n=1}^{N}$, the lift produces a program
whose log-joint equals

$$
\log \pi(\mathbf{w})
\;=\;
-\frac{1}{2 \sigma^{2}} \lVert \mathbf{w} \rVert_{2}^{2}
\;+\;
\sum_{n = 1}^{N} \log Z(s_{n};\, \mathbf{w}),
$$

where $Z(s_{n}; \mathbf{w})$ is the chart's goal weight (the
sentence's inside log-partition under $D$). Whether this joint
*is* the Bayesian posterior $p(\mathbf{w} \mid \{s_{n}\})$
depends on the modeling reading:

* **Unnormalized log-linear reading.** If $Z(s;\mathbf w)$ is an
  unnormalized sentence potential, a likelihood also contains the global
  partition term $-N\log\sum_{s'}Z(s';\mathbf w)$. The lift does not compute
  that term. Thus, it targets a weighted pseudo-posterior unless the
  omitted normalizer is independent of $\mathbf w$ or the caller adds it as a
  score.

* **Locally normalized generative reading.** For a proper, terminating PCFG
  whose expansion probabilities lie on local simplexes,
  $Z(s;\mathbf w)$ is the sentence probability obtained by summing its
  derivations; no second sentence-level normalizer is needed. The lift samples
  unconstrained Gaussian rule weights, however, and does not impose those
  simplex constraints. A caller that needs this reading must parameterize the
  local rule probabilities explicitly, for instance with Dirichlet draws or a
  softmax, and score the resulting normalized grammar.

The denotational claim of §7.1 (that the lifted program's
log-density equals the parameter prior plus the wrapped
log-joint, pointwise and with exact placeholder cancellation)
applies unchanged to [`nuts_program_from_deduction`](../api/stochastic/deduction/bayes.md#quivers.stochastic.deduction.bayes.nuts_program_from_deduction);
the grammar's parameterization determines whether that density has the
normalized likelihood required by the intended Bayesian model.

## References

For probability kernels and categorical composition, see [@giry1982categorical; @fritz2020synthetic]; for strong monads, see [@kock1972strong]. Reparameterization estimators are developed in [@kingma2014autoencoding]. For convergence conditions on stochastic updates, see [@robbins1951stochastic].
