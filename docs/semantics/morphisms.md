# Morphisms

This page assigns denotations to the three QVR morphism strata (discrete $\mathcal{V}$-enriched, stochastic, and continuous) and to the structural transitions between them. Throughout, we fix an algebra $\mathcal{V}$ as in [Setting and notation](setting.md). The implementation checks these classic morphism expressions during source compilation; the separate QIEC judgment and its guarantees are summarized in [Typing and checking](typing.md#10-guarantees-and-limits).

A morphism has an input boundary and an output boundary. What crosses those
boundaries depends on the kind of morphism:

| Representation | What one input determines | How to read the result |
|---|---|---|
| Weighted relation | a row of weights | compatibility, cost, count, or another algebra-specific value |
| Stochastic matrix | a row of nonnegative numbers summing to one | probabilities of the possible finite outputs |
| Conditional family | a distribution with parameters computed from the input | samples and scores for outputs that need not be finitely enumerable |

For instance, the row $(0.2,0.8)$ can be a distribution over two states.
The row $(2,8)$ can be a valid nonnegative weighted relation, but it is not
that distribution until a normalization is supplied. A Normal kernel needs
no infinite table: it can compute a mean and scale from the input, draw an
output, and evaluate a log density at a supplied output.

## 0. The unified `morphism` declaration

Every morphism, kernel, latent, observed, embed, and discretize binding uses a single declaration form

```
morphism f : DOM -> COD [k = v, ...] [~ INIT]
```

with the *role* selected by the option block; `kernel` is the default, so a `morphism` with no `role=` key is a kernel and the other roles are named explicitly:

| `role=...` | Stratum                                    | Initialiser admitted | Default initializer |
|------------|--------------------------------------------|----------------------|----------------------|
| `latent`   | learnable parameter on the source category | `~ Family(...)`      | Normal prior on raw logits |
| `kernel`   | parameter-driven distribution-family kernel | `~ Family(...)`      | family-specific      |
| `observed` | fixed-data morphism                         | `~ expr`             | required             |
| `embed`    | section of a discretization                 | `~ Family(...)` (sample-from-cell) | family-specific |
| `discretize` | quotient kernel                           | `~ expr` (partition) | uniform-quantile     |
| `let`      | deterministic morphism (alias for `~ expr`) | `~ expr`             | required             |

Other option-block keys configure individual roles: `scale` and `init` set a `latent` lowering's initial parameter scale and named initialization regime; `bins` configures `discretize`; `param_source` and `hidden_dim` configure the `kernel` lowering's parameter map $\theta$ (§2.1); `replicate=N` allocates $N$ independently parameterized copies under names `f_0, …, f_{N-1}` and binds the group as $f$; and the axis-role keys (`over`, `iid`) configure the family priors in §6.

A key outside the resolved role's set is rejected. For instance, `scale` initializes a `latent`'s tensor; a `kernel` obtains its distribution parameters from the input-dependent map $\theta$.

The remainder of this page uses the legacy keyword form (`latent`, `kernel`, `embed`) when illustrating individual strata; every snippet desugars to the unified form by `morphism f : … [role=KIND, …]`.

## 1. Discrete V-enriched morphisms

A discrete morphism declaration

```
morphism f : τ₁ -> τ₂ [role=latent]
```

denotes a morphism of $\mathcal{V}\text{-}\mathbf{Rel}$:

$$
\llbracket f \rrbracket : \llbracket \tau_1 \rrbracket \to \llbracket \tau_2 \rrbracket,
\qquad \llbracket f \rrbracket \in V^{|{\llbracket \tau_1 \rrbracket}| \times |{\llbracket \tau_2 \rrbracket}|}.
$$

For `morphism f : τ₁ -> τ₂ [role=latent]`, the entries are free parameters. `LatentMorphism.tensor` currently applies `sigmoid` to the raw parameter for every algebra, so the realized entries lie in $(0,1)$. This same constraint map applies across the algebra registry.

For `morphism g : τ₁ -> τ₂ [role = observed] ~ data`, the entries are *fixed*: $\llbracket g \rrbracket(x, y) = \mathrm{data}[x, y]$.

Both cases denote arrows in $\mathcal{V}\text{-}\mathbf{Rel}$. Training updates the latent tensor's parameters while keeping the observed tensor fixed.

### 1.1 Composition, tensor, identity

The primer computes [a complete two-table composition](../getting-started/concepts.md#2-composition-eliminates-an-intermediate-value). The [tensor-product guide](../guides/morphisms.md#tensor-product-the-operator) shows the axis order when two arrows run on separate inputs.

Composition connects boundaries in sequence: $A\to B\to C$.
Tensor product places two boundaries side by side:
$(A\to B)\otimes(C\to D)$ has input $A\times C$ and output $B\times D$.
The identity leaves the boundary value unchanged. In the discrete formulas,
that means an identity table has the algebra's unit on the matching diagonal
and its zero elsewhere. For the Markov algebra on a two-element object,
this is the matrix $\left(\begin{smallmatrix}1&0\\0&1\end{smallmatrix}\right)$.

Composition $;$, tensor $\boxtimes$, and identity $1_X$ in $\mathcal{V}\text{-}\mathbf{Rel}$ are defined as in [Setting and notation §2](setting.md#2-v-enriched-relations). The DSL operators correspond directly:

| Syntax | Denotation | Definition |
|--------|-----------|------------|
| `f >> g` | $\llbracket g \rrbracket \circ \llbracket f \rrbracket$ | $(f; g)(x, z) = \bigoplus_y f(x, y) \otimes g(y, z)$ |
| `f @ g` | $\llbracket f \rrbracket \boxtimes \llbracket g \rrbracket$ | $(f \boxtimes g)((x_1, x_2), (y_1, y_2)) = f(x_1, y_1) \otimes g(x_2, y_2)$ |
| `identity(X)` | $1_{\llbracket X \rrbracket}$ | $1_X(x, x') = \mathbf{1}$ if $x = x'$, $\bot$ otherwise |

**Proposition (Categorical structure).** *Assume $\mathcal{V}$ is a commutative unital quantale ([Setting §1](setting.md#1-algebras-as-enrichment-bases)): $\otimes$ distributes over arbitrary joins $\bigoplus$ on both sides. Then $\mathcal{V}\text{-}\mathbf{Rel}$ is a symmetric monoidal category, with $;$ as composition, $1_X$ as identities, $\boxtimes$ as monoidal product, $\mathbf{1}$ (the singleton) as monoidal unit, and the braid $\sigma_{X, Y}(x, y) = \mathbf{1}$ iff coordinates swap. Under the same hypothesis $\mathcal{V}\text{-}\mathbf{Rel}$ is also compact closed with every object self-dual. The product-fuzzy and Łukasiewicz operation pairs do not satisfy these hypotheses ([Algebras §2.1](algebras.md#21-a-note-on-the-product-fuzzy-and-ukasiewicz-pairs)). The cited counterexamples show the failed distributivity step.*

*Proof.* Associativity of $;$ unfolds to

$$
((f; g); h)(x, w) \;=\; \bigoplus_z \Bigl( \bigoplus_y f(x, y) \otimes g(y, z) \Bigr) \otimes h(z, w).
$$

The quantale distributivity law of [Setting §1](setting.md#1-algebras-as-enrichment-bases) commutes the outer $\otimes h(z, w)$ past the inner join over $y$, giving $\bigoplus_z \bigoplus_y f(x, y) \otimes g(y, z) \otimes h(z, w)$; the associativity of $\otimes$ (standing assumption on $\mathcal{V}$ as a commutative monoid, [Setting §1](setting.md#1-algebras-as-enrichment-bases)) means the bracketing of the threefold tensor is immaterial. The join's universal-colimit property collapses $\bigoplus_z \bigoplus_y = \bigoplus_{(y, z)}$ and the two-variable joins commute by the same property (a colimit cone is determined by its components in any order), so we can re-bracket as $\bigoplus_y f(x, y) \otimes \bigoplus_z g(y, z) \otimes h(z, w)$, applying distributivity once more to absorb $f(x, y)$ into the inner join. This is $(f; (g; h))(x, w)$. This step is where the proof uses distributivity. The identity laws use $1_X(x, x') = \mathbf{1}$ iff $x = x'$ and $\bot$ otherwise to collapse one join to a single term, using $\bot \otimes a = \bot$ (absorption) for the off-diagonal entries: this requires $\bot$ to be the bottom of the lattice, which holds in every unital quantale. Symmetry of $\boxtimes$ follows from commutativity of $\otimes$ (a standing assumption on $\mathcal{V}$, [Setting §1](setting.md#1-algebras-as-enrichment-bases)). Compact closure is established in [Expressions §2.9](expressions.md#29-compact-closed-structure) under the quantale hypothesis. $\square$

### 1.2 Marginalization

`Morphism.marginalize(*sets)` removes named factors from the *codomain*. For $f:A\to X\otimes Y$, `f.marginalize(X)` reduces the $X$ axis with the active algebra's `join`:

$$
\llbracket f.\mathrm{marginalize}(X) \rrbracket(a, y) \;=\; \bigoplus_{x \in \llbracket X \rrbracket} \llbracket f \rrbracket\bigl(a,(x,y)\bigr),
$$

a morphism $A\to Y$. Passing a set that is absent from the codomain is an error.

## 2. Stochastic morphisms

A default-role `morphism` declaration with finite-set codomain and no family initializer,

```
morphism kern : τ₁ -> τ₂
```

denotes a morphism of $\mathbf{Stoch}$, i.e.\ a row-stochastic $|{\llbracket \tau_1 \rrbracket}| \times |{\llbracket \tau_2 \rrbracket}|$ matrix (each row $\llbracket \mathrm{kern} \rrbracket(x, \cdot)$ is a probability distribution over $\llbracket \tau_2 \rrbracket$):

$$
\llbracket \mathrm{kern} \rrbracket(x, \cdot) \in \mathcal{G}_{\mathrm{fin}}\bigl(\llbracket \tau_2 \rrbracket\bigr)
\quad \text{for every } x \in \llbracket \tau_1 \rrbracket.
$$

The implementation realizes $\llbracket \mathrm{kern} \rrbracket$ via softmax over a parameter tensor; this is the canonical surjection $\mathbb{R}^{|Y|} \twoheadrightarrow \Delta^{|Y|-1}$ from raw logits onto the simplex.

Composition is the Kleisli composition for $\mathcal{G}_{\mathrm{fin}}$:

$$
(k_1; k_2)(x, z) \;=\; \sum_{y} k_1(x, y) \cdot k_2(y, z).
$$

Note that $\mathbf{Stoch}$ is *not* a sub-category of $\mathcal{V}_{\mathrm{pf}}\text{-}\mathbf{Rel}$: their composition operations differ. $\mathcal{V}_{\mathrm{pf}}\text{-}\mathbf{Rel}$ uses noisy-OR aggregation (the t-conorm $\bigoplus a_i = 1 - \prod (1 - a_i)$) while $\mathbf{Stoch}$ uses ordinary summation, which is the correct aggregation for events partitioning the sample space. The two categories share an underlying tensor representation in $[0, 1]^{|X| \times |Y|}$, but the categorical structure is supplied by different binary operations.

### 2.1 Distribution-family morphisms

Continuous distribution families *parameterized* by a finite set, declared with

```
morphism f : τ₁ -> σ ~ Family(...)
```

denote morphisms of $\mathbf{Kern}$:

$$
\llbracket f \rrbracket : \iota\bigl(\llbracket \tau_1 \rrbracket\bigr) \to \llbracket \sigma \rrbracket,
\qquad
\llbracket f \rrbracket(x, B) \;=\; \int_B p_{\mathrm{Family}}(y \,;\, \theta(x))\, \mathrm{d}y,
$$

where $\theta : \llbracket \tau_1 \rrbracket \to \Theta$ is the family's parameter map and $p_{\mathrm{Family}}(\cdot \,;\, \theta)$ is the density of the family at parameter $\theta$. The QVR-supplied family registry catalogs the pairs $(\Theta, p)$ for each name (Normal, Beta, Dirichlet, …).

The kernel thus factors through $\Theta$, and $\theta$ is what the `param_source` option selects: a single affine map by default, an MLP under `[param_source=mlp]`, a lookup table whenever $\llbracket \tau_1 \rrbracket$ is finite. A finite $\Theta$-factorization is a real restriction on which kernels a declaration can denote, since $\llbracket f \rrbracket$ ranges only over the image of

$$
\mathbf{Meas}\bigl(\llbracket \tau_1 \rrbracket,\, \Theta\bigr) \longrightarrow \mathbf{Kern}\bigl(\llbracket \tau_1 \rrbracket,\, \llbracket \sigma \rrbracket\bigr),
\qquad
\theta \longmapsto p_{\mathrm{Family}}(\,\cdot\;;\theta(-)),
$$

For `~ Normal`, each input thus receives a Gaussian distribution. Learnable weights $\varphi$ in the parameter map give a family of kernels $\varphi \mapsto p_{\mathrm{Family}}(\,\cdot\;;\theta_\varphi(-))$. Fitting selects the weights and hence one kernel from this family.

The type `f : τ₁ -> σ ~ Family` fixes the input and output spaces. Changing $\theta$ changes the kernel within that hom-set. The `param_source` option records whether the parameter map is affine, an MLP, or a lookup table.

#### The density is a product for most families

When $\llbracket \sigma \rrbracket$ has dimension $d > 1$, the display above hides a choice, because most names in the registry have [independent coordinates](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape) [@pyroTensorShapes]: their density factors,

$$
p_{\mathrm{Family}}\bigl(y \,;\, \theta(x)\bigr) \;=\; \prod_{i=1}^{d} p\bigl(y_i \,;\, \theta_i(x)\bigr),
\qquad\text{so}\qquad
\log \llbracket f \rrbracket(x, \cdot) \;=\; \sum_{i=1}^{d} \log p\bigl(y_i \,;\, \theta_i(x)\bigr).
$$

Structurally this is a [conditional independence](https://en.wikipedia.org/wiki/Conditional_independence), and it has a presentation that mentions neither coordinates nor Gaussians. In a [Markov category](https://ncatlab.org/nlab/show/Markov+category) a morphism $f : A \to X \otimes Y$ exhibits the independence of $X$ and $Y$ given $A$ exactly when it factors through the copy map as the tensor of its marginals, $f = (f_X \otimes f_Y) \circ \Delta_A$. An independent family is that factorization iterated over the codomain's coordinates:

$$
\llbracket f \rrbracket \;=\; \bigl(f_1 \otimes \cdots \otimes f_d\bigr) \circ \Delta_{\llbracket \tau_1 \rrbracket},
\qquad
f_i(x) \;=\; p\bigl(\,\cdot\;;\theta_i(x)\bigr).
$$

`Normal`, `Beta`, `Gamma`, and the other scalar families use this independent product over coordinates. `MultivariateNormal`, `LowRankMVN`, and `MatrixNormal` additionally support covariance structures that couple coordinates. Their parameterizations enforce the relevant covariance constraints; a [Cholesky factor](https://en.wikipedia.org/wiki/Cholesky_decomposition), for instance, represents a positive-definite covariance as $LL^\top$. Use `~ MultivariateNormal` when a Gaussian output needs residual correlation.

The coordinates can share a parameter map while remaining independent given $x$. For instance, let $y_1=x+\epsilon_1$ and $y_2=x+\epsilon_2$, with independent Normal noise terms. Averaging over a random $x$ makes the outputs correlated because both contain $x$. Holding $x$ fixed leaves independent residuals. To model correlation between those residuals, the output family must carry a joint covariance. For specified coordinate means and variances, the independent Gaussian also maximizes joint differential entropy; this Gaussian result does not apply unchanged to every scalar family.

## 3. Continuous morphisms

The [sampled-composition guide](../guides/continuous-spaces.md#composition-sampledcomposition) exercises this operation in Python. [Implementation correspondence](adequacy.md#3-probabilistic-composition) explains how its Monte Carlo density estimator relates to the integral below.

For a Normal kernel, think of two steps: compute $(\mu(x),\sigma(x))$ from
$x$, then use the distribution $\mathcal N(\mu(x),\sigma(x)^2)$.
The parameter map can be deterministic and learnable while the kernel's
output remains random. Sampling and scoring answer different questions:
`rsample(x)` produces an output, whereas `log_prob(x, y)` evaluates how the
kernel weights a supplied output $y$. A continuous density can exceed one;
probabilities are obtained by integrating it over a set of outputs.

A family-initialized `morphism` whose source is itself a space, e.g.

```
morphism g : σ₁ -> σ₂ ~ Family(...)
```

denotes a Markov kernel between standard Borel spaces:

$$
\llbracket g \rrbracket : \llbracket \sigma_1 \rrbracket \to \llbracket \sigma_2 \rrbracket,
\qquad
\llbracket g \rrbracket(s, B) \;=\; \int_B p_{\mathrm{Family}}(t \,;\, \theta(s))\, \mathrm{d}t.
$$

Composition is the Chapman–Kolmogorov integral

$$
(g_1; g_2)(s, C) \;=\; \int_{\llbracket \sigma_2 \rrbracket} g_2(t, C) \, g_1(s, \mathrm{d}t),
$$

realized numerically by Monte-Carlo or sampled-composition approximation in the implementation.

Composing does not always buy expressiveness, and whether it does is a fact about $\theta$. Gaussian kernels with affine $\theta$ are closed under the integral above: for $g_1(s, \cdot) = \mathcal{N}(As + a,\, \Sigma_1)$ and $g_2(t, \cdot) = \mathcal{N}(Bt + b,\, \Sigma_2)$,

$$
(g_1; g_2)(s, \cdot) \;=\; \mathcal{N}\bigl(BA\,s + Ba + b,\; B \Sigma_1 B^{\top} + \Sigma_2\bigr),
$$

again a Gaussian kernel with affine $\theta$. A chain of them is one kernel of the same family, and the intermediate spaces contribute only a rank bound on $BA$. A nonlinear $\theta$ breaks the closure, which is what makes a composite of `~ Normal` kernels deeper than its factors rather than equal to their product.

The kernel interpretation follows the categorical account of probability
[@giry1982categorical; @fritz2020synthetic]. For the statistical distinction
between conditional distributions and joint models, see
[@bishop2006pattern; @gelman2013bayesian].

## 4. Tensor product across strata

The `@` combinator extends to all three strata via the canonical monoidal structures:

| Strata of $f$ and $g$ | Ambient category of $f \otimes g$ |
|------------------------|------------------------------------|
| Both discrete | $\mathcal{V}\text{-}\mathbf{Rel}$ |
| Both stochastic | $\mathbf{Stoch}$ |
| Both continuous | $\mathbf{Kern}$ |
| Mixed | The smallest enclosing category, via the canonical embeddings $\mathcal{V}_{\mathbb{B}}\text{-}\mathbf{Rel}_{\mathrm{fun}} \hookrightarrow \mathbf{Stoch} \hookrightarrow \mathbf{Kern}$ of [Setting §4](setting.md#4-the-three-semantic-strata) |

The denotation is the parallel product of kernels:

$$
\llbracket f \otimes g \rrbracket\bigl((x, y), B \times C\bigr) \;=\; \llbracket f \rrbracket(x, B) \cdot \llbracket g \rrbracket(y, C),
$$

extended uniquely (by the standard product-measure construction) to the product $\sigma$-algebra.

## 4a. Replicated kernels and option blocks

### 4a.1 Replication

A declaration with `[replicate=n]` introduces $n$ independently parameterized morphisms with the same signature. The compiler registers synthesized members `f_0`, …, `f_{n-1}` and a group binding under `f`:

$$
\llbracket f\_i \rrbracket \;:\; \llbracket A \rrbracket \to \llbracket B \rrbracket \quad (i = 0, \dots, n-1),
$$

each a separate morphism in the appropriate stratum with its own learnable parameters. The bare identifier $f$ is *not* itself a morphism; it is a group reference accepted by expressions that admit splicing (notably $\mathsf{fan}$, [Expressions §3.1](expressions.md#31-fan)), where it expands to the comma-separated list of its members. Outside a splice site, referring to $f$ alone is a compile-time error.

The same option is admitted for roles that support replication.

### 4a.2 Option blocks

A `morphism` declaration may carry a bracketed *option block*

$$
\bigl[\,k_1 = v_1,\ \dots,\ k_m = v_m\,\bigr]
$$

after the codomain, listing family- or boundary-specific keyword overrides. The option block is denotationally inert at the categorical level: its sole effect is to override default values of the family's parameter map (`bins` for `discretize`, distribution-specific clipping bounds for kernels, etc.). Two declarations with identical signatures and different option blocks denote *different* morphisms whose tensors are computed from differently-configured parameter networks but are otherwise of the same categorical shape.

## 5. Stratum transitions

The `discretize` and `embed` declarations witness the canonical functors between strata.

### 5.1 Discretization

```
morphism d : σ -> n [role=discretize, bins=n]
```

denotes the *quotient* kernel induced by a measurable partition of $\llbracket \sigma \rrbracket$ into $n$ Borel cells $B_0, \dots, B_{n-1}$:

$$
\llbracket d \rrbracket(s, \{i\}) \;=\; \mathbf{1}_{B_i}(s),
$$

i.e.\ the deterministic kernel sending each $s$ to the (unique) cell containing it. As a morphism of $\mathbf{Kern}$ it factors through $\iota(\{0, \dots, n-1\})$. The choice of partition is supplied by the family annotation (e.g.\ uniform quantiles, learned thresholds).

### 5.2 Embedding

```
morphism e : n -> σ [role=embed] ~ Family(...)
```

denotes a *section* of a discretization: a kernel $e : \iota(\{0, \dots, n-1\}) \to \llbracket \sigma \rrbracket$ such that the composite $e ; d = \mathrm{id}$. In the implementation this is typically realized by sampling from a per-cell *embedding family* $p_{\mathrm{Family}}(\cdot \,;\, \theta_i)$.

Under the stated section condition, $(d,e)$ is a *retraction*. Discretization keeps the cell index and discards the original position within the cell. Embedding samples a new position from that cell's family, so the opposite composite generally changes the input.

## 6. Axis-role specifications

An *event* is one outcome of a distribution; a *batch* is a collection of
distributions. One three-coordinate multivariate Normal draw is one event
whose coordinates may be correlated. Ten such draws have a batch axis of
length ten and an event axis of length three. Their tensor has shape
$(10,3)$, but `log_prob` first scores each three-coordinate event and leaves
ten batch scores. Treating the event coordinates as ten-times-three
independent scalar draws would describe a different model.

Each registered family carries a declared *event rank* $r_F \in \mathbb{N}$.

| Event rank | Family names | Event shape |
|---|---|---|
| 0 | `Normal`, `LogitNormal`, `Beta`, `TruncatedNormal`, `Uniform`, `Bernoulli`, `RelaxedBernoulli`, `Binomial`, `Categorical`, `OneHotCategorical`, `GeneralizedPareto`, `HalfCauchy`, `HalfNormal`, `LogNormal`, `Exponential`, `Gamma` | $\mathbb{R}$ (scalar) |
| 1 | `MultivariateNormal`, `LowRankMVN`, `Dirichlet`, `LogisticNormal`, `RelaxedOneHotCategorical`, `GP`, `Horseshoe` | $\mathbb{R}^{d}$ for a single named axis |
| 2 | `Wishart`, `LKJCholesky` | $\mathbb{R}^{d_1 \times d_2}$ for two named axes |

Every family in the table is installed in the unified family catalog by [`_register_family`](../api/continuous/families.md) (directly for the bespoke families, via [`_make_family`](../api/continuous/families.md) for the auto-generated wrappers around `torch.distributions`). Each is thus equally usable as a conditional morphism (a kernel or `[role=latent]` morphism with a `~ Family(args)` initializer) and as an inline draw site (`sample x <- Family(args)`). The parameter map, support, and `log_prob` semantics are uniform across the two call paths.

A distribution clause `~ F(args) over <axes> [iid over <axes>]` *configures* the event–batch decomposition of a $F$-valued draw. Concretely, for a morphism $f : A \to B$ whose representing tensor has shape $\prod_{i} d_i$ indexed by the named factors $\{a_1, \dots, a_m\}$ of $A$ and $\{b_1, \dots, b_n\}$ of $B$, the clause names a sub-multiset $E \subseteq \{a_i\} \cup \{b_j\}$ of cardinality $|E| = r_F$ and declares:

$$
\llbracket f \rrbracket \;\in\; F\bigl(\textstyle\prod_{e \in E} d_e\bigr)^{\prod_{i \notin E} d_i},
$$

an iid product over the *batch axes* $\{a_i\} \cup \{b_j\} \setminus E$ of an $F$-distributed draw on the *event axes* $E$. Categorically, the iid axes are the product structure of the kernel's domain, while the event axes carry the family's joint (possibly correlated) distribution.

**Arity contract.**  The axis-role clause is well-typed only when $|E| = r_F$.  This preserves a critical categorical distinction: a flat $\mathrm{MVN}_{d_1 d_2}$ with dense covariance over $\mathbb{R}^{d_1 d_2}$ (event rank 1, single named axis whose dim equals $d_1 \cdot d_2$) is a different morphism from a $\mathrm{MatrixNormal}(d_1, d_2)$ with Kronecker covariance $V \otimes U$ over $\mathbb{R}^{d_1 \times d_2}$ (event rank 2, two named axes); no auto-substitution between them occurs.

**Positional binding.**  For families with two distinguishable event axes (row and column for `MatrixNormal`; the two correlation indices for `LKJCholesky`), the ordering of `over (e_1, e_2)` corresponds positionally to the family's declared event-axis ordering.

**Naturality.**  Refactoring a morphism's dom or cod (e.g. $B \mapsto B_1 \otimes B_2$) invalidates the old axis references at type-check time rather than silently rebinding them. This rejection preserves the declared product structure instead of guessing how the old axes map to the new factors.

The `dom` and `cod` shortcuts are legal in $E$ only when the corresponding side of the morphism is a single unfactored object; for a product-typed side, every factor must be named explicitly, since silently flattening a categorical product into an opaque single axis would erase the product structure.

## 7. Latent morphism priors

The discrete morphism declaration of §1 extends with a *parameter prior* via the `~ Family(args) [axis_role_clause]` clause:

```
morphism f : τ₁ -> τ₂ [role=latent] ~ Family(args) [over <axes> [iid over <axes>]]
```

The clause declares that the representing tensor of $f$ is itself a random variable drawn from the named family at the requested axis-role configuration. Concretely, the denotation desugars to the composite

$$
\llbracket f \rrbracket \;=\; (\text{wrap as morphism}) \circ \mathrm{rsample}_{F(\mathrm{args})},
$$

where `rsample` is the reparameterized sample of the family at its declared event/batch shape, and "wrap as morphism" is the canonical identification of a tensor of shape $|\tau_1| \times |\tau_2|$ (or the equivalent under the axis specification) with a morphism $\tau_1 \to \tau_2$ in $\mathcal{V}\text{-}\mathbf{Rel}$.

**Independence default.**  Without a prior clause, the morphism's representing tensor is a free parameter (a point in $V^{|\tau_1| \times |\tau_2|}$); the latent / observed distinction of §1 controls only whether the gradient flows. With a prior clause, $f$ becomes a sample site whose value is drawn afresh at each forward pass; the inference layer integrates $f$ as if it were an explicit `<-` step at the top of every enclosing program.

**Composition with marginalize.**  When the prior is on a $\mathcal{V}\text{-}\mathbf{Rel}$ morphism and the surrounding program's `marginalize` block integrates over a related coordinate, the marginalization composes with the prior in the standard way: $\mathrm{marg}(\nu) = \pi_* \nu$ where $\nu$ is the joint over $(W_f, \mathrm{rest})$ and $\pi$ projects away $W_f$ if the prior is the integration target, or leaves $W_f$ in scope otherwise.

## 8. Equivariance and naturality

Replacing an algebra changes how weights combine; replacing a family changes the available distributions. [Naturality](../guides/categorical.md#natural-transformations) requires agreement between applying an arrow and converting its representation, in either order. To preserve a denotation under replacement, check the corresponding operations, support, parameterization, and normalization. [Algebras](algebras.md#3-base-change) gives the preservation equations for discrete conversions.
