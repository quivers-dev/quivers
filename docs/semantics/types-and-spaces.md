# Types and Spaces

This page gives the denotation of the syntactic categories `ObjectExpr` (finite-set types) and `SpaceExpr` (continuous spaces). Both are interpreted compositionally by structural recursion on the syntactic constructors of [`quivers.dsl.ast_nodes`](../api/dsl/ast_nodes.md), and both are realized by the unified resolution walk in [`quivers.dsl.compiler.resolution`](../api/dsl/resolution.md).

An object describes possible values and the axes used to represent them.
If `Color` has three values and `Shape` has two, `Color * Shape` describes
six pairs. `Color + Shape` instead describes five tagged alternatives: a
color or a shape, with the tag preserving which side it came from.
`Real 3` describes vectors with three real coordinates; each coordinate can
take infinitely many values. Cardinality and dimension determine how a
morphism's inputs and outputs are represented.

The unit object has one possible value. Pairing with it adds no information:
$(a,*)$ carries exactly the information in $a$. The empty set has no possible
values; it is a different object, used as the unit for tagged alternatives.

## 1. Syntactic categories

We assume the following grammar fragment, whose constructors are realized one-to-one by the AST classes named in parentheses.

$$
\begin{array}{rcl}
\tau & ::= & n \;\big|\; X \;\big|\; \tau_1 \times \tau_2 \;\big|\; \tau_1 + \tau_2 \;\big|\; \tau_1 / \tau_2 \;\big|\; \tau_1 \backslash \tau_2 \;\big|\; T(\bar \tau)
\\[2pt]
& & \quad (\textsf{TypeName} \mid \textsf{ObjectProduct} \mid \textsf{ObjectCoproduct} \mid \textsf{ObjectSlash} \mid \textsf{ObjectEffectApply})
\\[6pt]
\sigma & ::= & C(\bar a; \bar k) \;\big|\; S \;\big|\; \sigma_1 \times \sigma_2
\\[2pt]
& & \quad (\textsf{ContinuousConstructor} \mid \textsf{TypeName} \mid \textsf{ObjectProduct})
\end{array}
$$

Here $n \in \mathbb{N}$ is an integer literal, $X$ ranges over object names, $S$ over space names, $T$ over declared effect identifiers, and $C \in \{\mathrm{Real}, \mathrm{Simplex}, \mathrm{Sphere}, \mathrm{Ball}, \mathrm{CholeskyFactor}, \mathrm{Covariance}, \mathrm{Correlation}, \mathrm{Orthogonal}, \mathrm{Stiefel}, \mathrm{LowerTriangular}, \mathrm{Diagonal}\}$ over constructor names. The notation $C(\bar a; \bar k)$ stands for a constructor invocation with positional arguments $\bar a$ and keyword arguments $\bar k$. The two slash directions (`/` and `\`) and the effect-apply form are residuated / effect-extended formers whose denotations live in a residuated-monoidal universe ([Schemas §3](schemas.md#3-residuated-type-formers)); they have no denotation in the bare $\mathbf{FinSet}$ stratum, and well-typedness restricts them to expressions whose every named atom resolves in a residuated universe such as `FreeResiduated`.

## 2. Denotation of types

The function signature below says that interpreting a type expression takes
its syntax and a table of declared objects, and returns an object.
$\mathrm{Ob}(\mathbf{FinSet})$ means “the objects of the category of finite
sets.” Each line then explains one syntax constructor: look up a name, form
a [pair, or form a tagged alternative](../guides/core.md#productset-and-coproductset). The final three lines require the
richer grammar-specific universe described after the display.

Type denotation is a function

$$
\llbracket \cdot \rrbracket_{\mathrm{Ty}} : \textsf{ObjectExpr} \times \mathrm{Env}_{\mathrm{obj}} \to \mathrm{Ob}(\mathbf{FinSet}).
$$

We omit the environment argument when it is clear.

$$
\begin{array}{rcl}
\llbracket n \rrbracket
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
\\[2pt]
\llbracket \tau_1 / \tau_2 \rrbracket_{\rho}
& = & \llbracket \tau_1 \rrbracket_{\rho} \,/\, \llbracket \tau_2 \rrbracket_{\rho}
\quad \text{(right residual; see [Schemas §3.1](schemas.md#31-residuation-slashes))}
\\[2pt]
\llbracket \tau_1 \backslash \tau_2 \rrbracket_{\rho}
& = & \llbracket \tau_1 \rrbracket_{\rho} \,\backslash\, \llbracket \tau_2 \rrbracket_{\rho}
\quad \text{(left residual; see [Schemas §3.1](schemas.md#31-residuation-slashes))}
\\[2pt]
\llbracket T(\tau_1, \dots, \tau_k) \rrbracket_{\rho}
& = & T\bigl(\llbracket \tau_1 \rrbracket_{\rho}, \dots, \llbracket \tau_k \rrbracket_{\rho}\bigr)
\quad \text{(effect application; see [Schemas §3.2](schemas.md#32-effect-application))}
\end{array}
$$

The last three cases require the surrounding object universe to be residuated and/or effect-extended; otherwise the denotation is undefined and well-typedness rejects the expression.

Finite cardinalities use the explicit constructor `FinSet n`, as in `morphism f : FinSet 3 -> FinSet 4`. Bare integer type expressions are not part of the current grammar.

The cartesian product $\times$ in $\mathbf{FinSet}$ is associative and commutative up to isomorphism. The implementation flattens nested `ProductSet` and `CoproductSet` values while preserving component order. Thus reassociated parses have the same structural representation, but `A * B` and `B * A` remain distinct values unless an explicit symmetry morphism swaps them.

Products, coproducts, and their universal properties are treated in
[@maclane1998categories]. The finite representations here choose concrete
carriers for those constructions; they also retain labels and component
order needed by the implementation.

## 2a. Object initializers

The surface form

```
object X : value
```

admits five *initializer* shapes for the right-hand side `value`, each binding $X$ to a concrete object of $\mathbf{FinSet}$ or $\mathbf{SBor}$ built from explicit data rather than from a syntactic `ObjectExpr`. Each is interpreted at the value layer and contributes its denotation directly to $\rho_{\mathrm{obj}}(X)$ (or $\rho_{\mathrm{spc}}(X)$ for the continuous-space initializer):

| Initialiser | Denotation |
|---|---|
| `FinSet n` | a finite set of cardinality $n$ |
| `Real d` (or any other [`ContinuousSpace`](#4-denotation-of-spaces) family invocation) | a standard Borel space of the named family at the supplied parameters |
| `{e_1, …, e_n}` | the enum set on the named labels (§2a.1) |
| `FreeMonoid(X, max_length = n)` | the bounded Kleene closure (§2a.2) |
| `FreeResiduated(G, depth = d, ops = O)` | the depth-bounded free residuated category over $G$ (§2a.3) |

The five initialisers share a single declaration form: there is no `space X : …` keyword variant. Whether $X$ lands in $\mathbf{FinSet}$ or $\mathbf{SBor}$ is decided by the family invoked on the right-hand side; the surrounding `object` keyword is uniform.

### 2a.1 Enum sets

```
object Atoms : {NP, S, VP}
```

denotes the finite set whose elements are exactly the named labels, ordered by declaration position:

$$
\llbracket \{e_1, \dots, e_n\} \rrbracket \;=\; \{e_1, \dots, e_n\},
\qquad |{\cdot}| = n.
$$

The label identity is part of the object: two enum sets with the same cardinality but different labels are *different* objects of $\mathbf{FinSet}$ under the `dx.Model` structural equality used throughout the codebase.

A `category C_1, …, C_n` declaration is the singleton-cardinality special case ([Schemas §1](schemas.md#1-category-atoms)).

### 2a.2 Free monoids

The `FreeMonoid(X, max_length = n)` initializer binds $X$ to the bounded Kleene closure of the generator set. See §3 below for the denotation.

### 2a.3 Free residuated categories

The `FreeResiduated(G, depth = d, ops = O)` initializer binds $X$ to a finite enumeration of category expressions over the generators, closed under the chosen residuation operations up to a depth bound. The denotation is given in [Schemas §4](schemas.md#4-the-free-residuated-universe).

## 3. Free monoids

The runtime value layer exposes a [`FreeMonoid` object class](../guides/core.md#freemonoid), which arises in the deduction fragment as the carrier of strings over an alphabet (see [Weighted Deduction Fragment](grammar.md)). At the surface, `FreeMonoid(X, max_length = n)` appears only as an *object initializer* (§2a.2 above), not as a `ObjectExpr` constructor: a free monoid must be bound to an `object` name via `object Words : FreeMonoid(...)` before it can be referenced as a type.

The denotation of `FreeMonoid(generators = X, max_length = n)` is the bounded Kleene star

$$
\llbracket \mathrm{FreeMonoid}(X, n) \rrbracket \;=\; \coprod_{k = 0}^{n} \rho_{\mathrm{obj}}(X)^{k},
$$

a finite set of size $\sum_{k=0}^{n} |\rho_{\mathrm{obj}}(X)|^{k}$. The bound $n$ is supplied by the grammar's depth parameter; the unbounded free monoid $\rho_{\mathrm{obj}}(X)^{*}$ is countably infinite and is not realized as a `SetObject`.

For an alphabet $\{a,b\}$ and maximum length two, the bounded word set is
$\{\epsilon,a,b,aa,ab,ba,bb\}$, with $\epsilon$ the empty word. Its seven
members match $1+2+4$. This is a finite inventory drawn from the unbounded
free monoid; it is not closed under arbitrary concatenation, since `aa`
followed by `b` exceeds the bound. Claims about the unbounded universal
property thus need an additional treatment of truncation.

## 4. Denotation of spaces

The [continuous-space hierarchy](../guides/continuous-spaces.md#continuousspace-hierarchy) gives examples of Euclidean, simplex, and positive-real values. [Event and batch axes](morphisms.md#6-axis-role-specifications) explain how these spaces determine distribution shapes.

The continuous-space sublanguage is the family-registry surface
`Real d`, `Simplex d`, plus the matrix-manifold and submanifold
variants below, combined under product via `*`. Each of these is
a *family invocation* against the [`ContinuousSpace`](../api/continuous/spaces.md) family registry:

| Surface | Family identifier | Carrier | Parameters |
|---|---|---|---|
| `Real d`              | [`Euclidean`](../api/continuous/spaces.md#quivers.continuous.spaces.Euclidean) | $\mathbb{R}^d$ | dimension $d$, with optional constructor bounds such as `Real 1 {low=0, high=1}`; bounds are inclusive |
| `Simplex d`           | [`Simplex`](../api/continuous/spaces.md#quivers.continuous.spaces.Simplex) | $\Delta^{d-1}$ | dimension $d$ |
| `Sphere d`            | [`Sphere`](../api/continuous/spaces.md#quivers.continuous.spaces.Sphere) | $S^{d-1} \subset \mathbb{R}^d$ | ambient dimension $d$ |
| `Ball d {radius=r}` | [`Ball`](../api/continuous/spaces.md#quivers.continuous.spaces.Ball) | $\{x \in \mathbb{R}^d : \lVert x \rVert_2 \le r\}$ | ambient dim $d$, radius $r$ |
| `CholeskyFactor d`    | [`CholeskyFactor`](../api/continuous/spaces.md#quivers.continuous.spaces.CholeskyFactor) | unit-row-norm lower-triangular $d \times d$ | $d$ |
| `Covariance d`        | [`Covariance`](../api/continuous/spaces.md#quivers.continuous.spaces.Covariance) | $\mathrm{Sym}^+_d$ (SPD cone) | $d$ |
| `Correlation d`       | [`Correlation`](../api/continuous/spaces.md#quivers.continuous.spaces.Correlation) | SPD with unit diagonal | $d$ |
| `Orthogonal d`        | [`Orthogonal`](../api/continuous/spaces.md#quivers.continuous.spaces.Orthogonal) | $O(d) \subset \mathbb{R}^{d \times d}$ | $d$ |
| `Stiefel(rows, cols)` | [`Stiefel`](../api/continuous/spaces.md#quivers.continuous.spaces.Stiefel) | $V_K(\mathbb{R}^N)$ with $K \le N$ | rows $N$, cols $K$ |
| `LowerTriangular d`   | [`LowerTriangular`](../api/continuous/spaces.md#quivers.continuous.spaces.LowerTriangular) | $d \times d$ lower-triangular | $d$ |
| `Diagonal d`          | [`Diagonal`](../api/continuous/spaces.md#quivers.continuous.spaces.Diagonal) | $\mathbb{R}^d$ via diagonal embedding | $d$ |

The surface `Real d` resolves to [`Euclidean`](../api/continuous/spaces.md#quivers.continuous.spaces.Euclidean). Positional arguments are space-separated, and constructor keyword arguments use braces. Adding a runtime class alone does not add surface syntax: the grammar and compiler constructor tables must also recognize its name.

For the matrix-manifold variants (`CholeskyFactor`, `Covariance`,
`Correlation`, `Orthogonal`, `Stiefel`, `LowerTriangular`) the
carrier is a flat $d \times d$ (or $N \times K$) tensor laid out
row-major; the sampling family is responsible for producing values on
the intended support. A membership check may return a per-batch mask
for classes that implement one. For instance,
[`Covariance`](../api/continuous/spaces.md#quivers.continuous.spaces.Covariance)
checks symmetry and nonnegative eigenvalues up to tolerance, and
[`Stiefel`](../api/continuous/spaces.md#quivers.continuous.spaces.Stiefel)
checks orthogonal columns. The base implementation is permissive, so
the type alone does not validate every manifold constraint.

Space denotation is a function

$$
\llbracket \cdot \rrbracket_{\mathrm{Sp}} : \textsf{SpaceExpr} \times \mathrm{Env}_{\mathrm{spc}} \times \mathrm{Env}_{\mathrm{obj}} \to \mathrm{Ob}(\mathbf{SBor}).
$$

For brevity write $\rho = (\rho_{\mathrm{spc}}, \rho_{\mathrm{obj}})$. The constructor cases are:

$$
\begin{array}{rcl}
\llbracket \mathrm{Euclidean}(d) \rrbracket
& = & \mathbb{R}^{d}
\\[2pt]
\llbracket \mathrm{Euclidean}(d; \mathrm{low} = \ell, \mathrm{high} = h) \rrbracket
& = & \prod_{i=1}^{d} [\ell, h]
\\[2pt]
\llbracket \mathrm{Simplex}(d) \rrbracket
& = & \Delta^{d-1} \;=\; \Bigl\{ x \in \mathbb{R}^{d}_{\ge 0} \,\Big|\, \textstyle\sum_i x_i = 1 \Bigr\}
\\[2pt]
\llbracket \mathrm{Sphere}(d) \rrbracket
& = & S^{d-1} \;=\; \bigl\{ x \in \mathbb{R}^{d} : \lVert x \rVert_2 = 1 \bigr\}
\\[2pt]
\llbracket \mathrm{Ball}(d; \mathrm{radius} = r) \rrbracket
& = & \bigl\{ x \in \mathbb{R}^{d} : \lVert x \rVert_2 \le r \bigr\}
\\[2pt]
\llbracket \mathrm{Covariance}(d) \rrbracket
& = & \mathrm{Sym}^{+}_{d} \;=\; \bigl\{ M \in \mathbb{R}^{d \times d} : M = M^{\top},\ M \succ 0 \bigr\}
\\[2pt]
\llbracket \mathrm{Correlation}(d) \rrbracket
& = & \bigl\{ R \in \mathrm{Sym}^{+}_{d} : R_{ii} = 1 \text{ for all } i \bigr\}
\\[2pt]
\llbracket \mathrm{CholeskyFactor}(d) \rrbracket
& = & \bigl\{ L \in \mathbb{R}^{d \times d} : L_{ij} = 0 \text{ if } j > i,\ \textstyle\sum_{j \le i} L_{ij}^{2} = 1 \bigr\}
\\[2pt]
\llbracket \mathrm{Orthogonal}(d) \rrbracket
& = & O(d) \;=\; \bigl\{ Q \in \mathbb{R}^{d \times d} : Q^{\top} Q = I_{d} \bigr\}
\\[2pt]
\llbracket \mathrm{Stiefel}(N, K) \rrbracket
& = & V_{K}(\mathbb{R}^{N}) \;=\; \bigl\{ X \in \mathbb{R}^{N \times K} : X^{\top} X = I_{K} \bigr\}
\\[2pt]
\llbracket \mathrm{LowerTriangular}(d) \rrbracket
& = & \bigl\{ L \in \mathbb{R}^{d \times d} : L_{ij} = 0 \text{ if } j > i \bigr\}
\\[2pt]
\llbracket \mathrm{Diagonal}(d) \rrbracket
& = & \mathbb{R}^{d}
\quad \text{(diagonal embedding into } \mathbb{R}^{d \times d}\text{)}
\end{array}
$$

each carrying its standard Borel $\sigma$-algebra (inherited as a
Borel submanifold of the ambient $\mathbb{R}^{d}$ or $\mathbb{R}^{d \times d}$). The matrix carriers ([`CholeskyFactor`](../api/continuous/spaces.md#quivers.continuous.spaces.CholeskyFactor),
[`Covariance`](../api/continuous/spaces.md#quivers.continuous.spaces.Covariance), [`Correlation`](../api/continuous/spaces.md#quivers.continuous.spaces.Correlation),
[`Orthogonal`](../api/continuous/spaces.md#quivers.continuous.spaces.Orthogonal), [`Stiefel`](../api/continuous/spaces.md#quivers.continuous.spaces.Stiefel),
[`LowerTriangular`](../api/continuous/spaces.md#quivers.continuous.spaces.LowerTriangular)) are stored as flat
row-major tensors; the manifold equations cut out a [smooth submanifold](https://en.wikipedia.org/wiki/Submanifold) of the ambient Euclidean space and the
Borel structure descends. The remaining cases are:

$$
\begin{array}{rcl}
\llbracket S \rrbracket_{\rho}
& = & \rho_{\mathrm{spc}}(S) \quad \text{if } S \in \mathrm{dom}(\rho_{\mathrm{spc}})
\\[2pt]
\llbracket S \rrbracket_{\rho}
& = & \iota\bigl(\rho_{\mathrm{obj}}(S)\bigr) \quad \text{otherwise, if } S \in \mathrm{dom}(\rho_{\mathrm{obj}})
\\[2pt]
\llbracket \sigma_1 \times \sigma_2 \rrbracket_{\rho}
& = & \llbracket \sigma_1 \rrbracket_{\rho} \times \llbracket \sigma_2 \rrbracket_{\rho}
\end{array}
$$

where $\iota : \mathbf{FinSet} \hookrightarrow \mathbf{SBor}$ is the canonical inclusion (every finite set as a discrete standard Borel space). The fallback rule for `SpaceName` allows mixed-domain `ProductSpace` instances such as `Real 3 * Token` where `Token : FinSet 256` is a previously declared object. The denotation is the product in $\mathbf{SBor}$, which is well-defined precisely because $\iota$ is a faithful functor preserving finite products.

## 5. Coherence

Coherence concerns changes of grouping that preserve the same components.
The nested pair `((color, shape), size)` and ` (color, (shape, size))` can be
converted without losing information. Swapping `color` and `shape` also has
an inverse, but changes their positions. QVR flattens the grouping in its
representation while retaining order; equal cardinalities alone do not
make differently ordered axes interchangeable.

The two type-formers $\times$ and $+$ on `ObjectExpr` are interpreted by cartesian product and disjoint union in $\mathbf{FinSet}$. Both are associative up to canonical isomorphism; their symmetries are isomorphisms, not structural equalities. Concretely:

- The flattening converters on `ProductSet` and `CoproductSet` realize the *coherence isomorphisms* $((\tau_1 \times \tau_2) \times \tau_3) \cong (\tau_1 \times (\tau_2 \times \tau_3))$ as identities on the chosen representative.
- Empty products denote $\mathbf{1}$ (a singleton; `EmptySet` for coproducts denotes $\emptyset$).
- The `EmptySet` constructor in `categorical.monoidal` is the unit for $+$.

Flattening makes reassociation literal in the runtime representation. Factor and summand order remains part of that representation.

## 6. Resolution in code

Both denotations are realized by a single unified walk on `ObjectExpr` in [`quivers.dsl.compiler.resolution`](../api/dsl/resolution.md). The mixin `_ResolutionMixin` exposes `_resolve_any_space`, which dispatches on the AST `kind` field and returns either a `SetObject` or a `ContinuousSpace`, including mixed-domain `ProductSpace` instances such as `Real 3 * Token` where `Token : FinSet 256`. The narrowing wrapper `_resolve_type` constrains the result to the discrete stratum for callers that require a `SetObject`.
