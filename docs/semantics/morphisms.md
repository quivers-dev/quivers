# Morphisms

A `morphism` declaration denotes a morphism in one of the three fragments of
[Setting §5](setting.md#5-fragments): a $\mathcal V$-matrix in $\mathbf{Mat}(\mathcal V)$, a
stochastic matrix in $\mathbf{FinStoch}$, or a Markov kernel in $\mathbf{BorelStoch}$. Throughout,
$\mathcal V$ is the module's composition rule ([Composition rules](composition-rules.md)).
Composition, tensor product, and marginalization are interpreted in the fragment of their
operands, and `discretize` and `embed` declarations move between the finite and the continuous
fragments. The checks the source compiler applies to these declarations are on
[Typing and checking](typing.md#10-guarantees-and-limits).

## 0. The `morphism` declaration

Every morphism is declared in one form,

```
morphism f : DOM -> COD [k = v, ...] [~ INIT]
```

and its `role` option selects the fragment and the runtime class. A declaration without `role`
has `role=kernel`.

| `role` | Denotes | Initializer | Runtime class |
|------------|--------------------------------------------|----------------------|----------------------|
| `latent` | a learnable $\mathcal V$-matrix | none, or `~ Family(...)` | `LatentMorphism` |
| `observed` | a fixed $\mathcal V$-matrix | `~ expr`, required | the morphism `expr` denotes |
| `kernel` | a stochastic matrix, or a Markov kernel with `~ Family(...)` | none, or `~ Family(...)` | `StochasticMorphism`, or a conditional family |
| `embed` | a Markov kernel from a finite set to a space | none | `Embed` |
| `discretize` | a Markov kernel from a space to a finite set | none, with `bins=n` required | `Discretize` |
| `let` | the morphism `expr` denotes | `~ expr`, required | the morphism `expr` denotes |
| `function` | a learned deterministic map between spaces | none | `LearnedFunction` |

Other options configure single roles. `scale` and `init` set the initial scale and the
initialization scheme of a `latent` table. `bins` sets the number of cells of a `discretize`.
`param_source` and `hidden_dim` select the parameter map $\theta$ of a family kernel (§2.1).
`replicate=n` declares $n$ copies with separate parameters (§4.1). `over` and `iid` give the event
and batch axes of a family (§6). The compiler rejects an option that the declared role does not
read.

## 1. Finite weighted morphisms

A declaration

```
morphism f : τ₁ -> τ₂ [role=latent]
```

denotes a morphism of $\mathbf{Mat}(\mathcal V)$, that is, a matrix

$$
\llbracket f \rrbracket \in V^{\llbracket \tau_1 \rrbracket \times \llbracket \tau_2 \rrbracket}.
$$

The entries of a `latent` table are learnable. The runtime stores an unconstrained real parameter
for each entry, drawn at initialization from a Normal distribution with mean zero and standard
deviation `scale` (default $0.5$), and its entries are the
[sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) of the parameters, so they lie in
$(0, 1)$ whatever the composition rule. The name `latent` is the library's keyword for this
learnable table. It is not a latent variable: without a prior the entries are point estimates.

A declaration `morphism g : τ₁ -> τ₂ [role=observed] ~ data` denotes the fixed matrix
$\llbracket g \rrbracket(x, y) = \mathrm{data}[x, y]$.

### 1.1 Composition, tensor product, identity {#11-composition-tensor-identity}

Composition $;$, tensor product $\boxtimes$, and identities $1_X$ are those of
[Setting §2](setting.md#2-v-enriched-relations):

| Syntax | Denotation | Entries |
|--------|-----------|------------|
| `f >> g` | $\llbracket f \rrbracket ; \llbracket g \rrbracket$ | $(f ; g)(x, z) = \bigoplus_y f(x, y) \otimes g(y, z)$ |
| `f @ g` | $\llbracket f \rrbracket \boxtimes \llbracket g \rrbracket$ | $(f \boxtimes g)((x_1, x_2), (y_1, y_2)) = f(x_1, y_1) \otimes g(x_2, y_2)$ |
| `identity(X)` | $1_{\llbracket X \rrbracket}$ | $1_X(x, x') = \mathbf 1$ if $x = x'$, and $\mathbf 0$ otherwise |

**Proposition.** If $\mathcal V$ is a commutative semiring, then $\mathbf{Mat}(\mathcal V)$ with
these operations is a symmetric monoidal category, with the one-element set as monoidal unit and
the swap $\sigma_{X, Y}((x, y), (y', x')) = \mathbf 1$ if $(y', x') = (y, x)$ and $\mathbf 0$
otherwise.

*Proof.* For associativity,

$$
((f ; g) ; h)(x, w) = \bigoplus_z \Bigl(\bigoplus_y f(x, y) \otimes g(y, z)\Bigr) \otimes h(z, w)
= \bigoplus_z \bigoplus_y f(x, y) \otimes g(y, z) \otimes h(z, w)
$$

by right distributivity and associativity of $\otimes$. Since $\bigoplus$ is associative and
commutative, the two finite reductions may be exchanged, and left distributivity gives
$\bigoplus_y f(x, y) \otimes \bigoplus_z g(y, z) \otimes h(z, w) = (f ; (g ; h))(x, w)$. For the
identity laws, $(1_X ; f)(x, y) = \bigoplus_{x'} 1_X(x, x') \otimes f(x', y)$ has one term
$\mathbf 1 \otimes f(x, y) = f(x, y)$, and the others are $\mathbf 0 \otimes f(x', y) = \mathbf 0$,
the unit of $\bigoplus$. Functoriality of $\boxtimes$, the interchange law
$(f ; f') \boxtimes (g ; g') = (f \boxtimes g) ; (f' \boxtimes g')$, follows by expanding both sides
into $\bigoplus_{y, y'} f(x_1, y) \otimes f'(y, z_1) \otimes g(x_2, y') \otimes g'(y', z_2)$, which
uses distributivity and commutativity of $\otimes$ to reorder the four factors. The associativity
and unit isomorphisms of $\boxtimes$ are those of the Cartesian product of finite sets, and they
and the swap are natural by the same expansion [@maclane1998categories]. $\square$

The built-in instances that are not semirings, among them the default `ProductFuzzyAlgebra`, do not
satisfy the hypothesis ([Semirings, t-norms, and change of base §2](algebras.md#2-algebraic-scope)),
and for them composites of three or more morphisms may depend on the parenthesization.
$\mathbf{Mat}(\mathcal V)$ for a commutative semiring is also compact closed, with each object its
own dual ([Expressions §2.9](expressions.md#29-compact-closed-structure)).

### 1.2 Marginalization

`f.marginalize(X)` removes the factor $X$ from the codomain by reducing its axis with
$\bigoplus$. For $f : A \to X \times Y$,

$$
\llbracket f.\mathrm{marginalize}(X) \rrbracket(a, y) = \bigoplus_{x \in \llbracket X \rrbracket} \llbracket f \rrbracket\bigl(a, (x, y)\bigr),
$$

a morphism $A \to Y$. Naming a factor that the codomain does not have is an error.

## 2. Finite stochastic morphisms

A declaration with the default role, a finite codomain, and no family initializer,

```
morphism k : τ₁ -> τ₂
```

denotes a morphism of $\mathbf{FinStoch}$: a matrix whose row $\llbracket k \rrbracket(x, -)$ is a
probability distribution on $\llbracket \tau_2 \rrbracket$ for each $x$, that is, an element of
$\mathcal D(\llbracket \tau_2 \rrbracket)$. The runtime stores a matrix of real logits and applies
the [softmax](https://en.wikipedia.org/wiki/Softmax_function) to each row. Softmax maps
$\mathbb R^{n}$ onto the interior of the simplex, the distributions with every entry positive, so a
learned stochastic matrix never has an exact zero.

Composition is Kleisli composition for $\mathcal D$, the product of stochastic matrices:

$$
(k ; l)(x, z) = \sum_{y} k(x, y)\, l(y, z).
$$

This is the matrix product over the probability semiring, not over the module's composition rule.
For the default rule, which reduces by noisy-OR, the composite of two stochastic matrices in
$\mathbf{Mat}(\mathcal V)$ would in general not be stochastic. The two products share a
representation as matrices with entries in $[0, 1]$ and differ in their reduction.

### 2.1 Distribution families

A declaration with a family initializer,

```
morphism f : τ -> σ ~ Family(...)
```

denotes a Markov kernel in $\mathbf{BorelStoch}$:

$$
\llbracket f \rrbracket(x, B) = \int_B p_{\mathrm{Family}}\bigl(y ;\, \theta(x)\bigr)\, \nu(\mathrm{d}y),
$$

where $\theta$ maps an input to the family's parameters and $p_{\mathrm{Family}}(- ; \theta)$ is the
family's density at parameters $\theta$ with respect to a reference measure $\nu$, Lebesgue measure
for a continuous family and counting measure for a discrete one
([density](https://en.wikipedia.org/wiki/Probability_density_function)). The family registry gives
the parameter space and the density of each family.

The kernel factors through the parameter space $\Theta$. The `param_source` option chooses
$\theta$: an affine map by default, a multilayer perceptron with `param_source=mlp`, and a table
when the input set is finite. The declaration can denote only kernels of the form
$x \mapsto p_{\mathrm{Family}}(- ; \theta(x))$, so for `~ Normal` every input receives a Normal
distribution. The learnable weights $\varphi$ of $\theta$ index a family of such kernels, and
fitting selects one.

### 2.2 Independent coordinates

Most families have a density that factors over the coordinates of the output: for an output in
$\mathbb R^d$,

$$
p_{\mathrm{Family}}\bigl(y ;\, \theta(x)\bigr) = \prod_{i=1}^{d} p\bigl(y_i ;\, \theta_i(x)\bigr),
$$

so the coordinates of the output are
[conditionally independent](https://en.wikipedia.org/wiki/Conditional_independence) given $x$
[@pyroTensorShapes]. In a Markov category, $f : A \to X \boxtimes Y$ makes $X$ and $Y$ independent
given $A$ exactly when $f$ equals the composite of the copy map $A \to A \boxtimes A$ with the
tensor product of its two marginals [@fritz2020synthetic]. A family with independent coordinates is
this factorization applied to each coordinate. `Normal`, `Beta`, `Gamma`, and the other scalar
families have independent coordinates. `MultivariateNormal`, `LowRankMVN`, and `MatrixNormal`
have a covariance that couples them. The coordinates may share a parameter map and still be
independent given $x$: they are correlated only after $x$ is averaged out.

## 3. Continuous morphisms {#3-continuous-morphisms}

A family declaration whose domain is a space,

```
morphism g : σ₁ -> σ₂ ~ Family(...)
```

denotes the Markov kernel

$$
\llbracket g \rrbracket(s, B) = \int_B p_{\mathrm{Family}}\bigl(t ;\, \theta(s)\bigr)\, \nu(\mathrm{d}t).
$$

Kernels compose by the Chapman–Kolmogorov integral of
[Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels):

$$
(g_1 ; g_2)(s, C) = \int g_2(t, C)\, g_1(s, \mathrm{d}t).
$$

The runtime samples from the composite exactly, by drawing the intermediate value and then the
output. Its density is another matter: for a finite intermediate set the runtime sums over it
exactly, but for a continuous intermediate space it does not compute the integral. It returns the
density of one fixed path through the intermediate values instead, which is not the density of the
composite ([Implementation correspondence §3](adequacy.md#3-probabilistic-composition)).

Normal kernels with affine parameter maps are closed under composition. For
$g_1(s, -) = \mathcal{N}(As + a, \Sigma_1)$ and $g_2(t, -) = \mathcal{N}(Bt + b, \Sigma_2)$,

$$
(g_1 ; g_2)(s, -) = \mathcal{N}\bigl(BAs + Ba + b,\; B \Sigma_1 B^{\top} + \Sigma_2\bigr)
$$

[@bishop2006pattern]. A chain of such kernels is thus a single affine Normal kernel. With a
nonlinear $\theta$ the composite is in general not a Normal kernel.

## 4. Tensor product across fragments {#4-tensor-product-across-fragments}

`f @ g` denotes the tensor product in the fragment of its operands:

| Fragments of $f$ and $g$ | Category of $f \boxtimes g$ |
|------------------------|------------------------------------|
| both finite weighted | $\mathbf{Mat}(\mathcal{V})$ |
| both finite stochastic | $\mathbf{FinStoch}$ |
| both continuous | $\mathbf{BorelStoch}$ |

For kernels, the tensor product is the product of measures:

$$
\llbracket f \boxtimes g \rrbracket\bigl((x, y), B \times C\bigr) = \llbracket f \rrbracket(x, B) \cdot \llbracket g \rrbracket(y, C),
$$

which extends to a measure on the product σ-algebra as the
[product measure](https://en.wikipedia.org/wiki/Product_measure). When one operand is continuous
and the other is a finite matrix, the runtime wraps the finite matrix as a kernel
(`DiscreteAsContinuous`) and forms the product in $\mathbf{BorelStoch}$. This is the embedding of
[Setting §5](setting.md#5-fragments) only when the matrix is stochastic: the wrapper samples from
each row after normalizing it, but scores an output by the unnormalized entry, so for a matrix
whose rows do not sum to one the two disagree.

### 4.1 Replication

A declaration with `[replicate=n]` declares $n$ morphisms $f_0, \dots, f_{n-1}$ with the same
signature and separate parameters, and binds the name $f$ to the group. The name $f$ alone is not
a morphism. It may appear only where an expression accepts a list of morphisms, as in
[`fan`](expressions.md#31-fan), where it stands for $f_0, \dots, f_{n-1}$. Anywhere else it is a
compile error.

### 4.2 Options

The option block configures the runtime object, such as the number of cells of a `discretize` or
the parameter map of a family kernel. It can change the denotation. Two declarations with the same
signature and different options may denote different morphisms, because they build their matrices
or kernels with different parameter maps.

## 5. Moving between fragments {#5-fragment-transitions}

### 5.1 Discretization

```
morphism d : σ -> B [role=discretize, bins=n]
```

requires $\sigma$ to be a bounded interval $[l, h] \subset \mathbb R$ and denotes a Markov kernel
from $[l, h]$ to the set $\{0, \dots, n-1\}$ of bins. Let $c_0, \dots, c_{n-1}$ be $n$ evenly spaced
points from $l$ to $h$. The kernel sends $s$ to the categorical distribution

$$
\llbracket d \rrbracket(s, \{i\}) = \frac{\exp\bigl(-(s - c_i)^2 / \tau\bigr)}{\sum_j \exp\bigl(-(s - c_j)^2 / \tau\bigr)},
\qquad \tau = 0.1,
$$

a softmax over negative squared distances to the bin centers. It is not deterministic: it puts
most of its mass on the nearest center, and all of it there only in the limit $\tau \to 0$. The
runtime class `Discretize` also has a deterministic nearest-center mode, which a QVR declaration
does not select.

### 5.2 Embedding

```
morphism e : B -> σ [role=embed]
```

requires $B$ to be a finite set and $\sigma$ a Euclidean space $\mathbb R^d$, possibly bounded, and
denotes the Markov kernel that sends $i$ to a Normal distribution with a learnable mean $m_i$ and learnable
coordinate scales $s_i$:

$$
\llbracket e \rrbracket(i, -) = \mathcal N\bigl(m_i, \operatorname{diag}(s_i^2)\bigr).
$$

The means start evenly spaced for a bounded interval, uniformly at random in a bounded box, and
near zero otherwise, and the scales start at one. Neither composite of $d$ and $e$ is an identity:
$e ; d$ returns $i$ only with some probability below one, and $d ; e$ replaces a point by a draw
near a learned center. The compiler accepts a `~ Family(...)` initializer on an `embed`
declaration and does not use it.

## 6. Event and batch axes {#6-axis-role-specifications}

A family's draw has an *event shape* and a *batch shape*: the event axes belong to one joint draw,
whose coordinates may be dependent, and the batch axes index independent draws
[@pyroTensorShapes]. Ten draws from a three-dimensional multivariate Normal have batch shape $(10)$
and event shape $(3)$, and `log_prob` returns ten numbers, one for each event. Each family has a
number of event dimensions $r_F$, read from the event dimension of its support in PyTorch:

| $r_F$ | Families (examples) | One event |
|---|---|---|
| 0 | `Normal`, `Beta`, `Gamma`, `Bernoulli`, `Binomial`, `Categorical`, `Poisson`, `Horseshoe`, `TruncatedNormal`, `Uniform` | a scalar in the family's support, real or integer |
| 1 | `MultivariateNormal`, `LowRankMVN`, `Dirichlet`, `LogisticNormal`, `OneHotCategorical`, `RelaxedOneHotCategorical`, `GP` | a vector along one named axis |
| 2 | `Wishart`, `InverseWishart`, `LKJCholesky`, `MatrixNormal` | a matrix along two named axes |

A family may be used both as the initializer of a declaration and as an inline draw
(`sample x <- Family(args)`), with the same parameters, support, and density.

In a declaration `morphism f : A -> B [over=E, iid=I] ~ F(args)`, the option `over` states which
named factors of the domain and codomain are event axes, and `iid` names batch axes. Each takes one
axis name or a bracketed list, as in `[over=cod, iid=dom]` or `[over=[dom, cod]]`. For
$f : A \to B$ whose factors are $a_1, \dots, a_m$ and $b_1, \dots, b_n$, with sizes $d_i$, `over`
names a set $E$ of $r_F$ factors as event axes. All other factors, on either side, are batch axes,
and the declaration states

$$
\llbracket f \rrbracket \sim \prod_{\text{batch indices}} F\Bigl(\prod_{e \in E} d_e\Bigr),
$$

independent draws of $F$ over the event axes, one for each index of the batch axes.

**Number of event axes.** In a declaration the clause is accepted only when $|E| = r_F$. A
multivariate Normal over one axis of size $d_1 d_2$ with a dense covariance and a matrix Normal over
two axes with the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) covariance
$V \otimes U$ are different families with different numbers of event axes, and the compiler does not
substitute one for the other. In a program's `sample` or `observe` step, a family with
$r_F \le 1$ may list $n > r_F$ axes $e_1, \dots, e_n$. The last $r_F$ are event axes and the
others are batch axes, in the order written. A matrix family in a step still needs $|E| = r_F$.

**Order of event axes.** For a family with two event axes (rows and columns for `MatrixNormal`,
the two indices for `LKJCholesky`), `over=[e_1, e_2]` binds them in the family's order.

**Changing a signature.** If a domain or codomain is refactored, for instance $B$ into
$B_1 \times B_2$, an axis name that no longer exists is a type error, not silently rebound.
The shorthand names `dom` and `cod` may be used in $E$ only when that side is a single factor. For a
product, each factor must be named.

## 7. Priors on learnable tables

A `latent` declaration accepts a family clause, with the `over` and `iid` options of §6:

```
morphism f : τ₁ -> τ₂ [role=latent, over=E, iid=I] ~ Family(args)
```

The clause states a prior on the entries of the table: the table is a draw from the family over
the stated event and batch axes, so that a model using $f$ treats the table as one more latent
variable.

In the [compiled runtime](../guides/dsl-overview.md#compilation-pipeline), the PyTorch implementation of a module,
the prior is not applied. The compiler checks the clause's axes against
the family and then builds the same `LatentMorphism` as for a declaration without a clause: a free
table initialized from a Normal with standard deviation `scale` and passed through the sigmoid. The
compiled runtime thus fits a point estimate of the table where the clause describes a
posterior over it.
