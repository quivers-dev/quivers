# Programs as kernels

A `program` declaration

```
program P (params) : τ₁ -> τ₂
    s₁
    ⋮
    sₙ
    return e
```

denotes an [s-finite kernel](setting.md#4-finite-and-s-finite-kernels)
$\llbracket P \rrbracket$ from $\llbracket \tau_1 \rrbracket$ to $\llbracket \tau_2 \rrbracket$
[@staton2017commutative]. Each step denotes a kernel from the values bound before it to the
values bound after it, and the body denotes the composite of these kernels. A body with no
`observe` or `score` step denotes a Markov kernel [@kozen1981semantics]. Otherwise the kernel
multiplies the probability of each run by its likelihood, so its total mass need not be one, and
normalizing it, when the mass is finite and positive, gives the posterior. The typing rules for
steps and their elaboration to the Quivers Indexed Effect Core (QIEC) are on
[Typing and checking §3](typing.md#3-program-contexts) and [§8](typing.md#8-program-elaboration).

For the program

```text
sample z <- Normal(loc=0.0, scale=1.0)
observe y <- Normal(loc=z, scale=0.5)
return z
```

with a given value of `y`, the denotation is the measure on the value $z$ returned with density
$\mathcal N(z; 0, 1)\, \mathcal N(y; z, 0.5^2)$ with respect to Lebesgue measure, the prior times
the likelihood. Its total mass is the marginal likelihood of $y$, and dividing by it gives the
posterior of $z$.

## 1. Contexts and the composition of steps {#1-the-giry-monad}

Write $\Gamma = \llbracket \tau_1 \rrbracket$ for the input space. After the steps
$s_1, \dots, s_i$ have bound the variables $v_1, \dots, v_i$ with spaces $X_1, \dots, X_i$, the
*context* is the product space

$$
\Phi_i = \Gamma \times X_1 \times \cdots \times X_i,
\qquad \Phi_0 = \Gamma,
$$

and a point $\phi \in \Phi_i$ assigns a value to each name in scope. This is the environment
$\rho_{\mathrm{var}}$ of [Setting §6](setting.md#5-environments). Each step denotes a kernel
$\mathcal S\llbracket s_i \rrbracket$ from $\Phi_{i-1}$ to $\Phi_i$, and kernels compose by
integrating over the intermediate value, as in
[Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels):

$$
(k ; l)(\phi, C) = \int l(\psi, C)\, k(\phi, \mathrm{d}\psi).
$$

The body denotes

$$
\mathcal{B}\llbracket s_1; \cdots; s_n; \mathsf{return}\ e \rrbracket
= \mathcal{S}\llbracket s_1 \rrbracket ; \mathcal{S}\llbracket s_2 \rrbracket ; \cdots ; \mathcal{S}\llbracket s_n \rrbracket ; \mathsf{ret}_e,
$$

where $\mathsf{ret}_e$ is the deterministic kernel that sends $\phi$ to the value of $e$. A step that
draws a value is a Markov kernel, the Kleisli composite of the [Giry monad](setting.md#3-standard-borel-spaces-and-markov-kernels)
[@giry1982categorical], and Moggi's account of programs as Kleisli morphisms applies to the body
without `observe` and `score` [@moggi1991notions]. A step that conditions multiplies the measure
by a likelihood and is an s-finite kernel. Staton shows that s-finite kernels are closed under this
composition and that it is associative [@staton2017commutative], so the composite is defined for
every body.

## 2. Steps

### 2.1 Draws {#21-bind}

A draw

```
sample v <- F(args)
```

extends the context with a value drawn from the family $F$:

$$
\mathcal{S}\llbracket \mathsf{sample}\ v \leftarrow F(\bar a) \rrbracket(\phi, B \times C)
= \mathbf{1}_B(\phi) \int_C p_F\bigl(y ;\, \theta_F(\bar a, \phi)\bigr)\, \nu_F(\mathrm{d}y),
$$

a Markov kernel from $\Phi$ to $\Phi \times \llbracket \mathrm{cod}(F) \rrbracket$. Here
$\theta_F(\bar a, \phi)$ is the parameter of $F$ computed from the arguments $\bar a$, which may
read the names bound in $\phi$, $p_F$ is the family's density, and $\nu_F$ is its reference measure,
Lebesgue measure or counting measure. The kernel keeps $\phi$ and adds the new value.

The arguments of a family are named, and they choose one complete parameterization:
`Normal(loc=m, scale=s)` gives both Normal parameters, while `Categorical(probs=p)` and
`Categorical(logits=l)` choose the probability and the logit parameterization of the categorical.
Positional arguments to a family, an incomplete parameterization, a repeated name, and a mixture of
two parameterizations are compile errors. Calls of computations and morphisms remain positional,
since their signatures fix the order of their arguments.

A draw may name the components of a product value: `sample (v_1, ..., v_m) <- F(args)` requires
$\llbracket \mathrm{cod}(F) \rrbracket = K_1 \times \cdots \times K_m$ and has the same denotation as
the draw of one value $v$, with $v_i$ the $i$-th projection of $v$.

### 2.2 Observations {#22-observe}

An observation

```
observe v <- F(args)
```

reads the given value $v_{\mathrm{obs}}$ of $v$ and multiplies the measure by its likelihood:

$$
\mathcal{S}\llbracket \mathsf{observe}\ v \leftarrow F(\bar a) \rrbracket(\phi, B)
= \mathbf{1}_B(\phi)\, p_F\bigl(v_{\mathrm{obs}} ;\, \theta_F(\bar a, \phi)\bigr),
$$

a kernel from $\Phi$ to $\Phi$ that keeps the context and has total mass
$p_F(v_{\mathrm{obs}} ; \theta_F(\bar a, \phi))$. A density may exceed one, so this kernel is in
general not a subprobability kernel. The observed values are given at run time with the program's
inputs ([observations](../guides/programs.md#the-observations-dict)), and normalization is left to
[inference](#6-inference-and-conditioning).

### 2.3 Deterministic bindings

A binding

```
let v = expr
```

extends the context deterministically. The expression denotes a measurable function
$h : \Phi \to T$ of the names in scope, and the step denotes the deterministic kernel

$$
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket(\phi) = \delta_{(\phi, h(\phi))},
$$

the pushforward along $\phi \mapsto (\phi, h(\phi))$. Arithmetic operators and built-in functions
denote the corresponding measurable functions.

#### 2.3.1 Built-in functions

A call `f(arg, ...)` in an expression resolves first against the closed set of QIEC primitives,
selected by the types of the arguments, and a name with no matching signature is a compile error.
A reduction (`sum`, `mean`, `min`, `max`, `prod`, `logsumexp`) collapses the last axis of its
argument: a vector reduces to a number, and a tensor of shape $(n_1, \ldots, n_k)$ to the tensor of
shape $(n_1, \ldots, n_{k-1})$ of its row reductions. A rowwise operation (`cumsum`, `sort`,
`softmax`, `log_softmax`, `normalize`) acts along the last axis and keeps the shape. Nested
reductions collapse one axis each, so `sum(sum(t))` sums the last two axes. The checker, the
[reference machine](../developer/qiec.md#reference-evaluator) (the library's interpreter for QIEC), the
[compiled runtime](../guides/dsl-overview.md#compilation-pipeline) (its PyTorch implementation), and every transpilation target read the primitives this
way, and a target that cannot express a reduction along the last axis of a given argument refuses the
program. A reduction over a named axis is written as a [`contraction`](composition-rules.md#4-operadic-contractions).
The [let-expression guide](../guides/dsl-programs-and-lets.md#checked-primitive-reference) lists
every primitive. The compiled runtime accepts further functions for programs that use PyTorch
attachments, but a call to one of them makes elaboration to QIEC fail with `qiec-program-gap`.

#### 2.3.2 Collection expressions

For a tensor $x = (x_0, \ldots, x_{n-1})$ with a leading extent $n$ known statically,

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

`map` elaborates to a QIEC comprehension. `fold`, written `fold(xs, init, acc -> item -> body)`, is
a left fold elaborated at its fixed extent. When the body reads the accumulator more than once, or
inside a lambda or a factor expression, each intermediate accumulator is bound once by a QIEC
[`LetValue`](../api/qiec/terms.md) and read through that binding, so the elaborated term grows
linearly in $n$. A source of `map` or `fold` that is not a variable, and the argument of a lambda
whose body reads its parameter more than once, are bound the same way. `filter` has no checked
denotation, because the length of its result depends on values, and only the compiled runtime
evaluates it.

For a computation $k$ from $A$ to $B$ given by a lambda, `traverse(x, k)` runs $k$ on each element
in order and collects the results. Its denotation is the composite of the kernels
$k(x_0), \ldots, k(x_{n-1})$ in that order, each extending the context with its result $b_i$,
followed by the deterministic kernel that returns $(b_0, \ldots, b_{n-1})$. This is the traversal of an applicative
functor over a list [@mcbride2008applicative]. The elaborator expands it into calls and draws in
source order, so the callee's effects join the enclosing row and each call has its own site
addresses. The [collection reference](../reference/qvr/collection-expressions.md) gives the
restrictions on the form.

#### 2.3.3 Calls of morphisms and other declarations

A call in an expression may also name a morphism, an encoder, a decoder, a deduction, or a
signature declared in the module. Built-in primitives take precedence over these names, which take
precedence over the constructors of a declared signature. A `program` with no `sample`, `observe`,
or `score` step denotes a deterministic kernel, one that sends each input to a
[Dirac measure](https://en.wikipedia.org/wiki/Dirac_measure), and an encoder may call it.

The compiler checks the number of positional arguments of each such call: a morphism takes one
argument, a [`MonadicProgram`](../api/continuous/programs.md) with named parameters takes one per
parameter, and another callable takes as many as its positional parameters without defaults (the
check is skipped when it accepts `*args`). A shape mismatch or a `TypeError` inside the callee is
reported as a [`CompileError`](../api/dsl/compiler.md) that names the call.

### 2.4 Plates {#24-indexed-bind-plate}

A [plate](https://en.wikipedia.org/wiki/Plate_notation) repeats a variable over an index set as
conditionally independent copies [@koller2009probabilistic]. A draw over a plate

```
sample v : A <- F(args)
```

draws one value $v(a)$ for each $a$ in the finite set $A$, independently given the context:

$$
\mathcal{S}\llbracket v : A \leftarrow F(\bar a) \rrbracket(\phi, -)
= \delta_\phi \otimes \bigotimes_{a \in A} F\bigl(\theta_F(\bar a, a, \phi)\bigr),
$$

a Markov kernel from $\Phi$ to $\Phi \times K^A$, with $K = \llbracket \mathrm{cod}(F) \rrbracket$ and
density $\prod_{a \in A} p_F(v(a) ; \theta_F(\bar a, a, \phi))$. The family of marginals
$(p_a)_{a \in A}$ does not determine a joint distribution on $K^A$, since other couplings have the
same marginals, and the plate chooses the product. The values may still be dependent in the joint
model when their parameters share a latent variable, such as a common scale, and they are
independent given it. Pyro calls the same construction `plate` [@pyroTensorShapes].

### 2.5 Plated observations

A plated observation

```
observe r : N <- F(args) [via = idx]
```

multiplies the measure by the product of the likelihoods of the observed values
$r_{\mathrm{obs}} : N \to \llbracket \mathrm{cod}(F) \rrbracket$:

$$
\mathcal{S}\llbracket \mathsf{observe}\ r : N \leftarrow F(\bar a) \rrbracket(\phi, B)
= \mathbf{1}_B(\phi) \prod_{n \in N} p_F\bigl(r_{\mathrm{obs}}(n) ;\, \theta_F(\bar a, n, \phi)\bigr).
$$

An argument written `theta[N]` reads the value of the plate variable `theta` at $n$. The optional
`via = idx` names an *index map*, a function from the rows to the groups of an enclosing grouped
`marginalize`
([§2.7](#27-grouped-marginalize-with-multi-observe-fibration)).

### 2.6 Summing out a discrete latent variable {#26-marginalize}

A block

```
marginalize c : A <- F(args)
    s₁
    ⋮
    sₖ
```

binds $c$ to a draw from $F(\bar a)$, plated over $A$ when $A$ is given, with the steps
$s_1, \ldots, s_k$ as its scope. When $F$ has finite support, the block denotes the measure of its
scope pushed forward along the projection $\pi_\Phi : \Phi \times C \to \Phi$ that forgets $c$:

$$
\nu \mapsto (\pi_\Phi)_* \nu,
\qquad
(\pi_\Phi)_* \nu(B) = \nu(B \times C).
$$

In log space, the scope's weight for each value $c$, added to the log prior of $c$, is reduced
over the support by log-sum-exp, which is the
[marginal likelihood](https://en.wikipedia.org/wiki/Marginal_likelihood) with $c$ summed out. This
is the enumeration of Pyro [@pyroEnumeration] and the marginalization of a mixture in Stan
[@stanFiniteMixtures]. The option `reduction = R` replaces log-sum-exp by `sum` or `mean` of the log
weights, which is not a marginalization. After the block, $c$ is out of scope.

When $F$ does not have finite support, the elaborator draws one value of $c$ and runs the scope
with it. The block then contributes the weight of one sampled configuration, not the integral over
$c$, and the elaborator rejects `reduction` on this path.

A draw, an observation, and a marginalized draw are one kind of step with a mode (draw, condition,
or sum out) and an optional plate. Summing out a discrete latent variable is the usual way to fit a
mixture model by its marginal likelihood [@bishop2006pattern].

#### 2.6.1 A latent variable carried by a scan

A scan cell `cell(x, z') : X * C -> C` that ends with a block over $z \in C$ and returns $z$ does not
close the block at the end of the cell: every later position reads $z$ as its previous state, so the
pushforward is taken only after the last position. For $T$ positions with first state
$z_0 \sim p$, the scan contributes the sum over every sequence of states,

$$
\sum_{z_0, \ldots, z_T} p(z_0) \prod_{t=1}^{T} w_t(z_{t-1}, z_t),
\qquad
w_t(i, j) = p_{F}\bigl(j ;\, \theta(x_t, i)\bigr) \exp \sigma_t(j),
$$

where $F(\theta(x_t, i))$ is the block's family at input $x_t$ and previous state $i$, and
$\sigma_t(j)$ is the log weight of the scope at $z_t = j$. The sum is computed from the left by the
[forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm),
$\alpha_t(j) = \sum_i \alpha_{t-1}(i)\, w_t(i, j)$ with $\alpha_0 = p$, and $\sum_j \alpha_T(j)$ is the
marginal likelihood of the chain [@rabiner1989hmm]. Every $z_t$ is summed out, so the name the scan
binds has no value. The
[marginalized chains reference](../reference/qvr/probabilistic-programs.md#marginalized-chains)
gives the forms the compiler accepts.

### 2.7 Grouped observations sharing one latent variable {#27-grouped-marginalize-with-multi-observe-fibration}

When several rows belong to one group, they share the group's latent class, and the likelihoods of
the rows must be multiplied before the class is summed out [@pyroEnumeration]. Let an *index map*
$\iota : N \to G$ send each row $n$ to its group, and call $\iota^{-1}(g)$, the set of rows of group
$g$, the [fiber](https://en.wikipedia.org/wiki/Fiber_(mathematics)) of $\iota$ over $g$. For the
rows $0, 1, 2$ with groups $\iota = (0, 0, 1)$, the likelihood of group $0$ is

$$
\sum_k \pi_{0k}\, p(y_0 \mid k)\, p(y_1 \mid k),
$$

whereas summing out a class for each row separately gives
$\bigl(\sum_k \pi_{0k} p(y_0 \mid k)\bigr)\bigl(\sum_k \pi_{0k} p(y_1 \mid k)\bigr)$, a model in which the two
rows may have different classes. Transpilation preserves this grouping
([Transpilation correctness §5](transpile-correctness/index.md#5-plates-marginalization-and-via)).

The grouping is declared on the block, and each observation in it names its index map:

```
marginalize c : K <- F(args) [over = G, reduction = R]
    observe r_1 : N_1 <- F_1(...) [via = idx_1]
    observe r_2 : N_2 <- F_2(...) [via = idx_2]
    ⋮
```

`over = G` names the group set (or `over = product(G_1, G_2, ...)`, also written
`over = [G_1, G_2, ...]`, for a product of groups), and `via = idx_m` names the index map
$\iota_m : N_m \to G$ of the $m$-th observation, given as an integer array (role `fibration` in the
API). The block contributes the log weight

$$
\sum_{g \in G} \operatorname{aggr}_{R,\, k}\Bigl[\log \pi(g, k) + \sum_m \sum_{n \,:\, \iota_m(n) = g} \ell_m(n, k)\Bigr],
$$

where $\ell_m(n, k) = \log p_{F_m}(r^{\mathrm{obs}}_m(n) ; \theta_m(n, k, \phi))$ is the log likelihood
of row $n$ of observation $m$ under class $k$, $\pi(g, k)$ is the prior weight of class $k$ in group
$g$, and $\operatorname{aggr}_R$ is the reduction over $k$: log-sum-exp by default, which sums the
class out, or `sum` or `mean`.

The compiled runtime and the transpilation targets compute this formula when the prior $\pi$ is the
same for every group. When the prior is itself a plate over $G$, as in
`marginalize z : K <- Categorical(probs=theta) [over=G]` with `theta : G`, they compute instead
$\sum_n \log \sum_k \theta(\iota(n), k)\, p(r_n \mid k)$, the model in which every row has its own class
drawn from its group's prior, and the grouping of the likelihoods is lost.

For a product of groups $G_1 \times G_2 \times \cdots$ paired with `via product(idx_1, idx_2, ...)`
on each observation, the groups are the tuples, ordered with the last factor varying fastest: a row
goes to position $\iota(n) = \sum_i \iota_i(n) \prod_{j > i} |G_j|$.

A grouped block may contain another. The inner block then contributes one value for each of its
own groups to the outer group's sum, and its group set must be either the outer set $G$ itself or a
product $G \times H$. In the second case the inner values are summed along the projection
$G \times H \to G$, and an inner argument indexed by $G$, such as a prior `theta[z]` selected by the
outer latent, is read at the $G$ component. A `sample` inside a block that reads nothing the block
binds is drawn once before the block, since every value of the latent shares it. A draw whose
arguments read the latent would be one draw for each class, and the compiler rejects it.

### 2.8 Effect annotations {#28-effect-signatures}

A program may declare which effects its body performs:

```
program P (params) : τ₁ -> τ₂ [effects = [E₁, E₂, ...]]
    body
```

where each $E_i$ is `Sample`, `Score`, `Marginal`, or `Pure`. This is the effect annotation of an
[effect system](https://en.wikipedia.org/wiki/Effect_system): a static bound on what a computation
may do [@lucassen1988polymorphic]. Each step contributes an effect:

| Step | Effect |
|---|---|
| `sample v <- F(args)`, `sample v : A <- F(args)` | `Sample` |
| `observe v <- F(args)`, `observe r : N <- F(args)`, `score v = expr` | `Score` |
| `marginalize c <- F(args) ...` | `Marginal` |
| `let v = expr`, `return e` | none |

The compiler computes the set $\mathcal E(P)$ of effects the body performs and checks
$\mathcal E(P) \subseteq \mathcal E_{\mathrm{decl}}$. `Pure` requires $\mathcal E(P)$ to be empty and
so rejects every `sample`, `observe`, `score`, and `marginalize`. A program whose body is `Pure`
denotes a deterministic kernel, one with only `Sample` a Markov kernel, and one with `Score` an
s-finite kernel. The QIEC effect row after elaboration is a finer type: it names the built-in
effect instances `random` and `score` and the effects of every reachable call
([Typing and checking §6](typing.md#6-computations-and-rows)).

A program with `over = M` in its options is a program over the posterior draws of the model $M$,
which Stan calls generated quantities. Its parameters include the latent variables of $M$, and its
body must be `Pure`, so it denotes a deterministic kernel $f$ from the latent space $\Theta$ of $M$
to $\llbracket \tau_2 \rrbracket$. Given the posterior kernel $\mathrm{post}$ from the data to
$\Theta$, the distribution of its output is the composite $\mathrm{post} ; f$, the
[pushforward](https://en.wikipedia.org/wiki/Pushforward_measure) of the posterior along $f$: the
posterior kernel comes first.

### 2.9 Reindexing a plate variable

A binding whose right side is `arr[idx]` *reindexes* a plate variable. Let $v : A \to B$ be the
values already drawn for a plate over $A$, and let $\iota : N \to A$ be an index map that assigns
each row to a member of $A$. The binding denotes the precomposition $v \circ \iota : N \to B$, a
deterministic kernel:

$$
\mathcal{S}\llbracket \mathsf{let}\ w = \mathit{arr}[\mathit{idx}] \rrbracket(\phi)
= \delta_{(\phi,\, v \circ \iota)}.
$$

For $v = (10, 20)$ and $\iota = (0, 0, 1)$ it gives $(10, 10, 20)$: rows of one group read the same
value, and no new value is drawn. This is a gather in PyTorch. Several index arrays,
$\mathit{arr}[\iota_1, \ldots, \iota_k]$ for $v : A_1 \times \cdots \times A_k \to B$ and
$\iota_j : N \to A_j$ on one $N$, read $v$ along the map $n \mapsto (\iota_1(n), \ldots, \iota_k(n))$,
so the entry at $n$ is $v(\iota_1(n), \ldots, \iota_k(n))$. A fixed index stands for the constant
map at its value. Index arrays on different row sets have no common $N$, and the compiler refuses
them.

### 2.10 Scores {#210-score-factor}

A score step

```
score v = expr
```

binds the value $h(\phi)$ of `expr` to $v$ and multiplies the measure by $\exp(h(\phi))$:

$$
\mathcal{S}\llbracket \mathsf{score}\ v = \mathit{expr} \rrbracket(\phi, B \times C)
= \mathbf{1}_B(\phi)\, \mathbf{1}_C\bigl(h(\phi)\bigr) \exp\bigl(h(\phi)\bigr),
$$

a kernel from $\Phi$ to $\Phi \times \mathbb R$, where $h : \Phi \to \mathbb R$ is the measurable
function `expr` denotes. `score` generalizes `observe`: where `observe` multiplies by the density
of a given value under a family, `score` multiplies by $\exp(w)$ for any expression $w$, as
`factor` does in WebPPL [@goodman2014design]. Since $h$ need not be bounded, the kernel is s-finite
but in general not finite [@staton2017commutative], and a program with a score need not have a
finite normalizing constant. Its main use is to add a differentiable quantity computed elsewhere,
such as the goal weight of a [deduction chart](grammar.md#9-chart-access-from-program-bodies), to the
program's log joint density. With $\mathit{expr} = \mathit{chart}.\mathsf{goal\_weight}()$ under the
log semiring, the score is the log inside weight of the sentence. A score step contributes the
effect `Score` (§2.8).

### 2.11 Return

`return e` ends the body. For $e = (v_1, \ldots, v_m)$, $\mathsf{ret}_e(\phi) = \delta_{(\phi(v_1), \ldots, \phi(v_m))}$,
the deterministic kernel that keeps the named values in the order written. Composed after the
body, it pushes the measure on the context forward to these values.

## 3. Parameters

A program whose parameters are bare names

```
program P (q₁, ..., qₖ) : τ₁ -> τ₂
    body
```

names the components of its domain: when $\tau_1 = \sigma_1 \times \cdots \times \sigma_k$, $q_i$ is the
$i$-th projection of the input. The denotation is the kernel from $\llbracket \tau_1 \rrbracket$ to
$\llbracket \tau_2 \rrbracket$ of §1.

### 3.1 Typed parameters

Typed parameters make $P$ a family of kernels, one for each value of the parameters:

| Parameter kind | Ranges over |
|---|---|
| `FinSet`, `Space`, `Object` | finite sets or standard Borel spaces |
| `Real`, `Nat` | real numbers or natural numbers (a hyperparameter) |
| `Mor[A, B]` | morphisms from $A$ to $B$ |

$$
\llbracket P \rrbracket \in \prod_{p_1 : P_1} \cdots \prod_{p_k : P_k} \mathbf{K}\bigl(\mathrm{dom}(p), \mathrm{cod}(p)\bigr),
$$

where $\mathbf K(X, Y)$ is the set of s-finite kernels from $X$ to $Y$. The domain and codomain may
mention the parameters, so different parameter values may give kernels between different spaces.

### 3.2 Calls of programs

A call `v <- P(a₁, ..., aₖ)` in another program substitutes the arguments $a_i$ for the parameters
$p_i$ in the body of $P$ and inlines the result as steps of the caller. The latent variables of the
callee are renamed under the fresh prefix `v$`, and its return value is bound to $v$. Substitution
commutes with the denotation of the body, because each parameter is bound at the top of the body
and the body's denotation is built from the denotations of its steps, and renaming bound variables
consistently does not change a denotation. Two calls of one program thus contribute separate
factors with their own latent variables, as in a hierarchical model.

### 3.3 Calls of computations

A step `y <- f(a₁, ..., aₖ)` that calls a computation $f$ declared with `define` has the denotation
of the body of $f$ at the arguments, sequenced into the caller. The compiled runtime expresses this
denotation as steps of the caller in four cases.

1. **Pure callees.** A body that makes no request denotes a value $\llbracket f \rrbracket(a)$, so the
   call is the deterministic kernel $\delta_{\llbracket f \rrbracket(a)}$ and its steps are `let`
   bindings. A pure computation that the body calls is substituted in turn. A call that reaches its
   own callee again has no finite unfolding and is refused.
2. **Conditionals between pure branches.** For a Boolean $c$ and pure branches,

    $$
    \llbracket \mathsf{if}\ c\ \mathsf{then}\ e_1\ \mathsf{else}\ e_2 \rrbracket
      = \mathsf{select}(c, \llbracket e_1 \rrbracket, \llbracket e_2 \rrbracket),
    $$

    a value of the same type, differentiable in each branch away from the points where $c$ changes.
    When a branch draws or scores, the set of sites, and with it the space on which the joint
    density is defined, depends on $c$. The compiled runtime fits on a fixed space, so it refuses such
    a conditional.
3. **Callees that request only `Random` and `Score`.** Such a body denotes an s-finite kernel built
   from the caller's own two effects, and inlining it is the substitution of §3.2: its site `s`
   becomes the caller's site `y$s`, and each `Score.add` adds its weight to the caller's joint. A
   callee that handles an effect inside its body is refused, since its denotation is the handler's
   fold over the body, which the caller's steps do not express.
4. **`traverse` over a lambda.** When the runs $k(x_0), \ldots, k(x_{n-1})$ read only their own
   items and names bound before the traverse, the composite of §2.3.2 is the product

    $$
    \llbracket \mathsf{traverse}(x, k) \rrbracket = \bigotimes_{i < n} \llbracket k \rrbracket(x_i),
    $$

    the kernel of a plate over the axis of $x$ (§2.4). A pure $k$ gives a deterministic tensor, and a
    $k$ that draws gives one plated site `y$s` for each site `s` of $k$. A recurrence, in which a run
    reads its predecessor, is not a product and is written with `scan`.

The [reference machine](../reference/qvr/index.md), the library's interpreter for QIEC, runs every
call as written, recursion and handlers included, so a program refused by the compiled runtime
still has its joint density there.

## 4. Composition of programs

Programs $P$ from $X$ to $Y$ and $Q$ from $Y$ to $Z$ compose as kernels:

$$
\llbracket P \mathbin{>\!\!>} Q \rrbracket(x, C) = \int \llbracket Q \rrbracket(y, C)\, \llbracket P \rrbracket(x, \mathrm{d}y).
$$

A composite is named with `define` and made public with `export`:

<!-- compile: false -->
```qvr
define pq = p >> q
export pq
```

`define N = e` extends the morphism environment with $N \mapsto \llbracket e \rrbracket_\rho$, and
`export N` adds $N$ to the module's output. A `where` block may contain nested `define` declarations,
which the compiler processes before the outer expression. For a module $M$ with the declarations
`export E_1`, ..., `export E_k`,
$\llbracket M \rrbracket_{\mathrm{exports}} = (\llbracket E_1 \rrbracket, \ldots, \llbracket E_k \rrbracket)$,
where each $E_i$ names a morphism, a program, a deduction, an encoder, a decoder, or a composite
bound by `define`.

## 5. Equations {#5-algebraic-laws-and-their-scope}

Write $\eta$ for the deterministic kernel $\phi \mapsto \delta_\phi$ and $;$ for the composition of
§1. The denotations satisfy

| Equation | Statement |
|---------|-----------|
| Left unit | $\eta ; k = k$ |
| Right unit | $k ; \eta = k$ |
| Associativity | $(k_1 ; k_2) ; k_3 = k_1 ; (k_2 ; k_3)$ |
| Commutativity | $\mathsf{sample}\ x \leftarrow t;\ \mathsf{sample}\ y \leftarrow u;\ s \;=\; \mathsf{sample}\ y \leftarrow u;\ \mathsf{sample}\ x \leftarrow t;\ s$ when $t$ does not read $y$ and $u$ does not read $x$ |

For the steps `sample`, `let`, and `return`, which denote Markov kernels, the first three equations
are the laws of the Kleisli category of the Giry monad, and commutativity holds because the Giry
monad is commutative, which is Fubini's theorem for probability measures
[@giry1982categorical; @kock1972strong]. The commutation is what lets two independent draws be
reordered without changing the denotation.

For bodies with `observe` and `score`, all four equations hold for s-finite kernels: Staton proves
associativity of their composition and the commutativity equation, including for steps that score
with unbounded weights [@staton2017commutative]. Without s-finiteness, interchanging the order of
integration can fail for kernels into infinite measures, which is why the setting is the s-finite
kernels. Steps that depend on each other, and calls with effects other than `Random` and `Score`,
stay in the order written. A finite `marginalize` is a pushforward, so it respects equality of
measures. Normalizing a scored program is a separate operation, defined when the total mass is
finite and positive.

## 6. Inference {#6-inference-and-conditioning}

A program with observations gives, for each configuration of its latent variables, an unnormalized
weight. When the total weight is finite and positive, dividing by it gives the posterior. Exact
integration, enumeration, Markov chain Monte Carlo, and variational inference approximate this
posterior or its expectations. The module [`quivers.inference`](../api/inference/svi.md) uses the
log joint density of a run to fit a variational family [@blei2017variational; @hoffman2013svi] or to
sample with [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) and the
No-U-Turn Sampler [@neal2011mcmc; @hoffman2014nuts].

## 7. Wrappers that place priors on parameters {#7-bayesian-lifts}

A model whose learnable parameters have no prior still denotes a kernel, with the parameters held
fixed. The module [`quivers.inference.lifts`](../api/inference/lifts.md) turns such a model into a
[`MonadicProgram`](../api/continuous/programs.md) with an explicit prior, so that stochastic variational inference (SVI) and NUTS can
treat the parameters as latent variables, as in a Bayesian neural network [@mackay1992bayesian]. The
library calls these functions *Bayesian lifts*. Below, $\theta \in \mathbb R^D$ is the vector of all
learnable parameters of the wrapped model, $\sigma_\theta > 0$ is a fixed prior scale, and
$\mathbf z$ names optional intermediate latent variables of the wrapped model.

### 7.1 Priors on parameters

[`bayesian_lift_parameters`](../api/inference/lifts.md) takes a PyTorch module `inner_model` that
computes $\log p_{\mathrm{inner}}(\mathbf z, y \mid x, \theta)$ through a method
`log_joint(x, observations)`, and an optional map `additional_latents` from names of latent
variables $\mathbf z$ to their shapes. The returned program draws $\theta_i \sim \mathcal N(0, \sigma_\theta^2)$
for each parameter and $\mathbf z_j \sim \mathcal N(0, \sigma_z^2 I)$ for each additional latent, where
$\sigma_z$ is the argument `latent_placeholder_scale`. Its score step computes

$$
S(\theta, \mathbf{z}) = \log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta) - \sum_{j} \log \mathcal{N}\bigl(\mathbf{z}_{j} ;\, 0, \sigma_{z}^{2} I\bigr),
$$

so the log joint density of the program, the sum of the log priors of its draws and the score, is

$$
\log \pi(\theta, \mathbf z) = \log \mathcal{N}(\theta ;\, 0, \sigma_{\theta}^{2} I_{D}) + \log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta).
$$

The placeholder priors on $\mathbf z$ cancel, so $\sigma_z$ does not change the target density. It
can change the warmup of NUTS, which adapts its mass matrix to the draws. The target is proportional
to the posterior $p(\theta, \mathbf z \mid x, y)$ under the prior $\mathcal N(0, \sigma_\theta^2 I)$ on
$\theta$. To let gradients reach $\theta$, the program temporarily replaces each parameter of the
wrapped module by the drawn value while it evaluates `log_joint`.

### 7.2 Deterministic models with an observation family

[`lift_to_bayesian_program`](../api/inference/lifts.md) takes a deterministic module, a function $\mu$
that reads a location from the module's output, a family $\mathcal F$ (a subclass of
[`torch.distributions.Distribution`](https://docs.pytorch.org/docs/stable/distributions.html)), and the
key $k$ of the observed value in the observations. It wraps the module in a `log_joint` that scores
$y = \mathrm{observations}[k]$ under $\mathcal F(\mu(x) ; \boldsymbol\eta)$, where $\boldsymbol\eta$ holds
the family's other parameters from `observation_kwargs`, and applies §7.1 to the wrapper with no
additional latents. The log joint density is

$$
\log \pi(\theta) = \log \mathcal{N}(\theta ;\, 0, \sigma_{\theta}^{2} I_{D}) + \log p_{\mathcal{F}}\bigl(y \mid \mu(x ; \theta), \boldsymbol{\eta}\bigr),
$$

the log posterior of $\theta$ under this likelihood up to a constant.

### 7.3 Models with a log density

[`lift_from_log_prob`](../api/inference/lifts.md) does the same for a module and a function
$\ell(x, y ; \theta)$ that returns a conditional log density directly, giving
$\log \pi(\theta) = \log \mathcal N(\theta ; 0, \sigma_\theta^2 I_D) + \ell(x, y ; \theta)$. It serves
modules that already have a `log_prob` method, such as a
[`SampledComposition`](../api/continuous/morphisms.md) of Normal kernels.

### 7.4 A Monte Carlo likelihood

[`monte_carlo_log_joint`](../api/inference/lifts.md) wraps a `MonadicProgram` with intermediate
`sample` steps $\mathbf z$ that the caller does not want as latent variables of the sampler. Its
`log_joint(x, observations)` draws each $\mathbf z_j$ from its family $q_j(\mathbf z_j \mid x, \theta)$,
adds the draws to the observations, evaluates the inner `log_joint`, and subtracts the log priors
of the draws, which leaves

$$
\widetilde{\ell}(x, y ;\, \theta) = \log p_{\mathrm{inner}}\bigl(y \mid \mathbf{z}_{*}, x, \theta\bigr),
\qquad
\mathbf{z}_{j*} \sim q_{j}(- \mid x, \theta).
$$

This is a [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimate of
$\mathbb E_{\mathbf z \sim q}[\log p(y \mid \mathbf z, x, \theta)]$, not of the log marginal likelihood
$\log p(y \mid x, \theta) = \log \mathbb E_{\mathbf z \sim q}[p(y \mid \mathbf z, x, \theta)]$. By
[Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) the first is at most the
second, with equality only when $p(y \mid \mathbf z, x, \theta)$ does not depend on $\mathbf z$, so as an
estimate of the log marginal likelihood it is biased downward [@burda2016iwae].

For SVI, with a family that can be reparameterized, $\mathbf z_*$ is a differentiable function of
fixed noise $\xi$, and when differentiation and expectation may be exchanged, one draw gives an
unbiased gradient of $\mathbb E_{\mathbf z}[\log p(y \mid \mathbf z, x, \theta)]$
[@kingma2014autoencoding]. Stochastic gradient ascent then optimizes this lower bound on the log
marginal likelihood, under the usual conditions on step sizes [@robbins1951stochastic].

For NUTS the wrapper is unsuitable. A new $\mathbf z_*$ at every evaluation makes the target density
random, so the Hamiltonian and the acceptance step of NUTS no longer refer to one fixed target
[@neal2011mcmc; @hoffman2014nuts]. Using a random log density inside MCMC is the pseudo-marginal
method, which is valid only for an unbiased estimate of the density itself, not of its logarithm
[@andrieu2009pseudo]. For NUTS, declare $\mathbf z$ as an additional latent with
`bayesian_lift_parameters(..., additional_latents={'<name>': <shape>})` (§7.1), so that
$(\theta, \mathbf z)$ are sampled from one joint target with a deterministic log density.

### 7.5 Weighted deduction systems

[`nuts_program_from_deduction`](../api/stochastic/deduction/bayes.md) applies the construction of §7.1
to a [`DeductionSystem`](../api/stochastic/deduction.md) $D$ with learnable log weights
$\mathbf w \in \mathbb R^{d}$ and a corpus $s_1, \ldots, s_N$. It draws each $w_i \sim \mathcal N(0, \sigma^2)$
and adds the score $\sum_{n=1}^{N} \mathrm{goal}_D(s_n ; \mathbf w)$, where $\mathrm{goal}_D(s ; \mathbf w)$
is the chart's goal weight for the sentence $s$. When $D$ uses the log semiring, the goal weight is the
logarithm $\log Z(s ; \mathbf w)$ of the inside weight $Z$, and the log joint density is

$$
\log \pi(\mathbf{w}) = -\frac{1}{2 \sigma^{2}} \lVert \mathbf{w} \rVert_{2}^{2} + \sum_{n=1}^{N} \log Z(s_{n} ;\, \mathbf{w}) + \mathrm{const}.
$$

Whether this is the posterior $p(\mathbf w \mid s_1, \ldots, s_N)$ depends on what $Z$ is:

- If $Z(s ; \mathbf w)$ is an unnormalized weight of the sentence, the likelihood also contains the
  term $-N \log \sum_{s'} Z(s' ; \mathbf w)$ of the global normalizer, which the function does not
  compute. The target is then the posterior only when that normalizer does not depend on $\mathbf w$,
  or when the caller adds it as a score.
- If the rule weights are the probabilities of a consistent PCFG, normalized for each left-hand side,
  $Z(s ; \mathbf w)$ is the probability of the sentence, the sum over its derivations, and no further
  normalizer is needed [@chi1998estimation]. The function draws unconstrained Normal log weights,
  which do not satisfy these constraints. A caller who wants this reading parameterizes the local rule
  probabilities explicitly, for instance by Dirichlet draws [@johnson2007bayesian] or a softmax for
  each left-hand side, and scores the normalized grammar.
