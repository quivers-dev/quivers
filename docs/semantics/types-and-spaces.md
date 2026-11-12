# Types and spaces

An object expression denotes a finite set, and a space expression denotes a
[standard Borel space](setting.md#3-standard-borel-spaces-and-markov-kernels). Both are
interpreted by structural recursion on the syntax tree of
[`quivers.dsl.ast_nodes`](../api/dsl/ast_nodes.md): the denotation of a product is the product of
the denotations of its factors, and so on for each constructor. The finite sets are objects of
$\mathbf{FinSet}$ and the spaces are objects of $\mathbf{SBor}$
([Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels)).

## 1. Syntax

The object and space expressions are given by the grammar below. The names in parentheses are
the classes of the syntax tree.

$$
\begin{array}{rcl}
\tau & ::= & \mathrm{FinSet}\ n \;\big|\; X \;\big|\; \tau_1 \times \tau_2 \;\big|\; \tau_1 + \tau_2 \;\big|\; \tau_1 / \tau_2 \;\big|\; \tau_1 \backslash \tau_2 \;\big|\; T(\bar \tau)
\\[2pt]
& & \quad (\textsf{DiscreteConstructor} \mid \textsf{TypeName} \mid \textsf{ObjectProduct} \mid \textsf{ObjectCoproduct} \mid \textsf{ObjectSlash} \mid \textsf{ObjectEffectApply})
\\[6pt]
\sigma & ::= & C(\bar a; \bar k) \;\big|\; S \;\big|\; \sigma_1 \times \sigma_2
\\[2pt]
& & \quad (\textsf{ContinuousConstructor} \mid \textsf{TypeName} \mid \textsf{ObjectProduct})
\end{array}
$$

Here $n$ is a positive integer literal, $X$ ranges over object names, $S$ over space names, and
$T$ over declared effects. $C$ ranges over the space constructors `Real`, `Simplex`, `Sphere`,
`Ball`, `CholeskyFactor`, `Covariance`, `Correlation`, `Orthogonal`, `Stiefel`,
`LowerTriangular`, and `Diagonal`, and $C(\bar a; \bar k)$ is a constructor applied to positional
arguments $\bar a$ and keyword arguments $\bar k$. In source, `*` writes $\times$, positional
arguments are separated by spaces (`Real 3`), and keyword arguments go in braces
(`Ball 3 {radius=2}`).

The slashes $/$ and $\backslash$ and the effect application $T(\bar\tau)$ build
[categorial types](schemas.md#3-residuated-type-formers), formulas such as `S / NP` that serve as
syntactic categories in a grammar. They are elements of a finite set of formulas, not finite sets
themselves, and the compiler accepts them only inside rule schemata. Outside that context the
compiler rejects them with "residuated / effect-typed expressions do not resolve to a concrete
object or space".

## 2. Denotation of object expressions

The denotation of object expressions is a function

$$
\llbracket \cdot \rrbracket_{\mathrm{Ty}} : \textsf{ObjectExpr} \times \mathrm{Env}_{\mathrm{obj}} \to \mathrm{Ob}(\mathbf{FinSet}),
$$

from an expression and an object environment $\rho_{\mathrm{obj}}$
([Setting §6](setting.md#5-environments)) to a finite set. We drop the environment argument when
it is clear. The clauses are

$$
\begin{array}{rcl}
\llbracket \mathrm{FinSet}\ n \rrbracket
& = & \{0, 1, \dots, n - 1\}
\\[2pt]
\llbracket X \rrbracket_{\rho}
& = & \rho_{\mathrm{obj}}(X) \quad \text{if } X \in \mathrm{dom}(\rho_{\mathrm{obj}})
\\[2pt]
\llbracket \tau_1 \times \tau_2 \rrbracket_{\rho}
& = & \llbracket \tau_1 \rrbracket_{\rho} \times \llbracket \tau_2 \rrbracket_{\rho}
\\[2pt]
\llbracket \tau_1 + \tau_2 \rrbracket_{\rho}
& = & \llbracket \tau_1 \rrbracket_{\rho} \sqcup \llbracket \tau_2 \rrbracket_{\rho}
\end{array}
$$

where $\sqcup$ is the [disjoint union](https://en.wikipedia.org/wiki/Disjoint_union), whose
elements remember which summand they came from. The product and the disjoint union are the
[product](https://ncatlab.org/nlab/show/product) and
[coproduct](https://ncatlab.org/nlab/show/coproduct) of $\mathbf{FinSet}$
[@maclane1998categories]. The function is undefined on the slashes and on effect application, as
§1 states.

### 2.1 Object declarations

The declaration `object X : value` binds $X$ in $\rho_{\mathrm{obj}}$, or in $\rho_{\mathrm{spc}}$
when the value is a space. Besides an object or space expression, the value may take three other
forms:

| Value | Denotation |
|---|---|
| `{e_1, ..., e_n}` | the finite set of the labels $e_1, \dots, e_n$, in that order (class `EnumSet`) |
| `FreeMonoid(G, max_length = n)` | the words over $G$ of length at most $n$ (§3) |
| `FreeResiduated(G, depth = d, ops = [...])` | the categorial types over the atoms $G$ of depth at most $d$ ([Schemas §4](schemas.md#4-the-free-residuated-universe)) |

A set given by enumeration is determined by its labels: two such sets with the same number of
elements and different labels are different objects. A `category C_1, ..., C_n` declaration binds
each name to a set with one label ([Schemas §1](schemas.md#1-category-atoms)). There is one
declaration keyword, `object`, for finite sets and spaces alike. The value decides which
environment receives the name.

## 3. Words of bounded length

The [free monoid](https://en.wikipedia.org/wiki/Free_monoid) $G^*$ on a finite set $G$ is the set
of all finite words over $G$, with concatenation as its operation. `FreeMonoid(G, max_length = n)`
denotes its finite subset of words of length at most $n$:

$$
\llbracket \mathrm{FreeMonoid}(G, n) \rrbracket = \coprod_{k = 0}^{n} \rho_{\mathrm{obj}}(G)^{k},
$$

a finite set of $\sum_{k=0}^{n} |\rho_{\mathrm{obj}}(G)|^{k}$ words. For $G = \{a, b\}$ and $n = 2$
it is $\{\epsilon, a, b, aa, ab, ba, bb\}$, where $\epsilon$ is the empty word. This set is not
closed under concatenation ($aa$ followed by $b$ has length three), so it is not a monoid, and a
statement that uses the universal property of $G^*$ does not apply to it without a separate
treatment of the length bound. The generator $G$ must be a declared object name. The runtime class
is [`FreeMonoid`](../api/core/objects.md), which stores the set as the coproduct
$\mathbf 1 + G + G^2 + \dots + G^n$.

## 4. Denotation of space expressions

The denotation of space expressions is a function

$$
\llbracket \cdot \rrbracket_{\mathrm{Sp}} : \textsf{SpaceExpr} \times \mathrm{Env}_{\mathrm{spc}} \times \mathrm{Env}_{\mathrm{obj}} \to \mathrm{Ob}(\mathbf{SBor}),
$$

and we write $\rho = (\rho_{\mathrm{spc}}, \rho_{\mathrm{obj}})$. Each constructor is resolved to a
class of [`quivers.continuous.spaces`](../api/continuous/spaces.md):

| Source | Class | Denotation |
|---|---|---|
| `Real d` | [`Euclidean`](../api/continuous/spaces.md#quivers.continuous.spaces.Euclidean) | $\mathbb{R}^d$ |
| `Real d {low=l, high=h}` | `Euclidean` | $[l, h]^d$ |
| `Simplex d` | [`Simplex`](../api/continuous/spaces.md#quivers.continuous.spaces.Simplex) | $\Delta^{d-1}$ |
| `Sphere d` | [`Sphere`](../api/continuous/spaces.md#quivers.continuous.spaces.Sphere) | $S^{d-1} \subset \mathbb{R}^d$ |
| `Ball d {radius=r}` | [`Ball`](../api/continuous/spaces.md#quivers.continuous.spaces.Ball) | closed ball of radius $r$ in $\mathbb{R}^d$, $r = 1$ by default |
| `CholeskyFactor d` | [`CholeskyFactor`](../api/continuous/spaces.md#quivers.continuous.spaces.CholeskyFactor) | Cholesky factors of $d \times d$ correlation matrices |
| `Covariance d` | [`Covariance`](../api/continuous/spaces.md#quivers.continuous.spaces.Covariance) | symmetric positive definite $d \times d$ matrices |
| `Correlation d` | [`Correlation`](../api/continuous/spaces.md#quivers.continuous.spaces.Correlation) | correlation matrices of size $d$ |
| `Orthogonal d` | [`Orthogonal`](../api/continuous/spaces.md#quivers.continuous.spaces.Orthogonal) | the orthogonal group $O(d)$ |
| `Stiefel(N, K)` | [`Stiefel`](../api/continuous/spaces.md#quivers.continuous.spaces.Stiefel) | the Stiefel manifold $V_K(\mathbb{R}^N)$, $K \le N$ |
| `LowerTriangular d` | [`LowerTriangular`](../api/continuous/spaces.md#quivers.continuous.spaces.LowerTriangular) | lower triangular $d \times d$ matrices |
| `Diagonal d` | [`Diagonal`](../api/continuous/spaces.md#quivers.continuous.spaces.Diagonal) | diagonal $d \times d$ matrices, stored as $\mathbb{R}^d$ |

The sets in the last column are

$$
\begin{array}{rcl}
\Delta^{d-1} & = & \bigl\{ x \in \mathbb{R}^{d} : x_i \ge 0,\ \textstyle\sum_i x_i = 1 \bigr\}
\\[2pt]
S^{d-1} & = & \bigl\{ x \in \mathbb{R}^{d} : \lVert x \rVert_2 = 1 \bigr\}
\\[2pt]
\mathrm{Sym}^{+}_{d} & = & \bigl\{ M \in \mathbb{R}^{d \times d} : M = M^{\top},\ x^\top M x > 0 \text{ for all } x \ne 0 \bigr\}
\\[2pt]
\mathrm{Corr}_{d} & = & \bigl\{ R \in \mathrm{Sym}^{+}_{d} : R_{ii} = 1 \text{ for all } i \bigr\}
\\[2pt]
\mathrm{Chol}_{d} & = & \bigl\{ L \in \mathbb{R}^{d \times d} : L_{ij} = 0 \text{ for } j > i,\ L_{ii} > 0,\ \textstyle\sum_{j \le i} L_{ij}^{2} = 1 \bigr\}
\\[2pt]
O(d) & = & \bigl\{ Q \in \mathbb{R}^{d \times d} : Q^{\top} Q = I_{d} \bigr\}
\\[2pt]
V_{K}(\mathbb{R}^{N}) & = & \bigl\{ X \in \mathbb{R}^{N \times K} : X^{\top} X = I_{K} \bigr\}
\end{array}
$$

$\Delta^{d-1}$ is the [probability simplex](https://en.wikipedia.org/wiki/Simplex),
$O(d)$ the [orthogonal group](https://en.wikipedia.org/wiki/Orthogonal_group), and
$V_K(\mathbb{R}^N)$ the [Stiefel manifold](https://en.wikipedia.org/wiki/Stiefel_manifold) of
matrices with orthonormal columns. $\mathrm{Chol}_d$ is the set of Cholesky factors of
correlation matrices: $L \mapsto L L^\top$ is a bijection from $\mathrm{Chol}_d$ onto
$\mathrm{Corr}_d$, and it is the support of the
[LKJ distribution](https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution) on
Cholesky factors [@lewandowski2009lkj]. It corresponds to Stan's `cholesky_factor_corr`, not to
the Cholesky factors of arbitrary covariance matrices.

Each of these sets is a Borel subset of $\mathbb{R}^d$ or $\mathbb{R}^{d \times d}$ with the
inherited σ-algebra, and a Borel subset of a standard Borel space is standard Borel
[@kechris1995classical]. The remaining clauses are

$$
\begin{array}{rcl}
\llbracket S \rrbracket_{\rho}
& = & \rho_{\mathrm{spc}}(S) \quad \text{if } S \in \mathrm{dom}(\rho_{\mathrm{spc}})
\\[2pt]
\llbracket S \rrbracket_{\rho}
& = & \iota\bigl(\rho_{\mathrm{obj}}(S)\bigr) \quad \text{if } S \notin \mathrm{dom}(\rho_{\mathrm{spc}}) \text{ and } S \in \mathrm{dom}(\rho_{\mathrm{obj}})
\\[2pt]
\llbracket \sigma_1 \times \sigma_2 \rrbracket_{\rho}
& = & \llbracket \sigma_1 \rrbracket_{\rho} \times \llbracket \sigma_2 \rrbracket_{\rho}
\end{array}
$$

where $\iota : \mathbf{FinSet} \hookrightarrow \mathbf{SBor}$ gives a finite set all its subsets as
events. The second clause lets a product mix a finite set with a space, as in `Real 3 * Token`
for an object `Token : FinSet 256`. The product of two standard Borel spaces, with the product
σ-algebra, is standard Borel [@kechris1995classical], so the third clause always denotes an
object of $\mathbf{SBor}$.

## 5. Regrouping and order of factors

The product and the disjoint union are associative and commutative only up to isomorphism: the
map $((a, b), c) \mapsto (a, (b, c))$ and the swap $(a, b) \mapsto (b, a)$ are bijections, not
identities. By the [coherence theorem](https://ncatlab.org/nlab/show/coherence+theorem+for+monoidal+categories)
for monoidal categories, any two composites of associativity and unit isomorphisms between the
same two groupings are equal, so a regrouping has one meaning whichever way it is derived
[@maclane1998categories]. The swap is not covered by that theorem and changes the order of the
factors.

The runtime makes regrouping literal. It flattens nested `ProductSet` and `CoproductSet` values
while keeping the order of the components, so `(A * B) * C` and `A * (B * C)` have the same
representation, while `A * B` and `B * A` stay different until a symmetry morphism swaps them. The
empty product denotes a one-element set $\mathbf 1$, the unit of $\times$, and the empty
coproduct `EmptySet` denotes $\emptyset$, the unit of $+$.

## 6. Implementation {#6-resolution-in-code}

One recursive walk over `ObjectExpr` in
[`quivers.dsl.compiler.resolution`](../api/dsl/resolution.md) realizes both denotations. It
returns a [`SetObject`](../api/core/objects.md) for an object expression and a
[`ContinuousSpace`](../api/continuous/spaces.md) for a space expression, including a mixed
`ProductSpace` such as `Real 3 * Token`. Callers that need a finite set reject a continuous
result with a compile error.

The matrix spaces store a $d \times d$ (or $N \times K$) matrix as a flat tensor in row major
order, and the sampling family is responsible for producing values in the set. Membership tests
are approximate. `Covariance` and `Correlation` accept a matrix that is symmetric to within
$10^{-5}$ and whose eigenvalues are at least $-10^{-7}$, so they also accept singular matrices
that lie outside $\mathrm{Sym}^+_d$. `Orthogonal`, `Stiefel`, and `Sphere` test their defining
equations to within $10^{-5}$. `CholeskyFactor` has no membership test: its `contains` raises
`NotImplementedError`.
