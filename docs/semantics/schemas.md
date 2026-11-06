# Schemas, Rules, Categories, and Bundles

This page covers the grammar-supporting declarations `category`, `rule`, `schema`, and `bundle`, together with `object` bindings used by the deduction surface. The current grammar has no separate top-level `alias` or `type` declaration.

These declarations describe reusable grammar patterns. A `category` names a
basic grammatical type, a `rule` connects premise patterns to a conclusion,
a `schema` supplies a parameterized morphism pattern, and a `bundle` collects
rules or schemas for use together. Here *category* often means a grammatical
type such as `NP` or `S`. In category theory, a category is the larger
structure of objects and arrows equipped with composition laws.

For instance, a rule with pattern `X / Y, Y |- X` says that something seeking
a `Y` to its right can combine with that `Y` and produce an `X`. Substituting
`S` for `X` and `NP` for `Y` yields one concrete instance. The formal
variables let one rule describe many such applications.

## 1. Category atoms

A `category` declaration introduces one or more atomic category names:

```
category NP, S, VP
```

Category names are registered in a *category-atom set* $\rho_{\mathrm{cat}}$ which is *separate* from the object environment $\rho_{\mathrm{obj}}$ and the space environment $\rho_{\mathrm{spc}}$:

$$
\llbracket \mathrm{category}\ C_1, \dots, C_n \rrbracket
\;:\;\;
\rho_{\mathrm{cat}}\ \longmapsto\ \rho_{\mathrm{cat}}\ \cup\ \{C_1, \dots, C_n\}.
$$

The atom set serves as the generator alphabet of the residuated category universe consumed by chart parsers (§4) and by rule / schema declarations (§5 / §6). Category atoms are *not* finite-set objects: they cannot be used directly as morphism domains or codomains. To make their residuated closure available as a type, declare a `FreeResiduated` object over them (§4) or pass them via `parser(categories=[…])` at the call site.

## 2. Object-level name bindings

A single declaration form binds a name to an object or space expression:

$$
\llbracket \mathrm{object}\ N : \tau \rrbracket
\;:\;\;
\rho_{\mathrm{obj}}\ \longmapsto\ \rho_{\mathrm{obj}}\bigl[N \mapsto \llbracket \tau \rrbracket_{\rho_{\mathrm{obj}}}\bigr].
$$

The behavior depends on whether $\tau$ resolves as a `SetObject`:

- *Finite-set binding.* When $\tau$ has no residuated slashes or effect applications and resolves cleanly to a finite-set object (for instance `FinSet 16`, `A * B`, `A + B`, or a previously-bound finite-set name), $N$ is bound in $\rho_{\mathrm{obj}}$ to the resulting object. It may appear as the domain or codomain of any later morphism.

- *Continuous-space binding.* When $\tau$ is a supported continuous constructor such as `Real D`, `Simplex K`, or `Covariance D`, $N$ is bound in $\rho_{\mathrm{spc}}$ to the corresponding continuous space.

- *Syntactic binding.* When $\tau$ contains a residuated slash, an effect application, or otherwise fails `SetObject` and `ContinuousSpace` resolution, the binding is recorded as a textual substitution rule $\rho_{\mathrm{alias}}[N \mapsto \tau]$ that the schema-pattern matcher inlines at every use site. Syntactic bindings are not first-class objects: they cannot appear as morphism domains or codomains, only inside `schema` / `rule` patterns.

Bindings are transparent: every later occurrence of $N$ resolves to the underlying object / space, and the schema-level interpretation of [The Program Theory](program-theory.md) records no new vertex for the binding itself.

## 3. Residuated type formers

The `ObjectExpr` grammar admits two formers beyond products / coproducts:

$$
\tau \;::=\; \dots \;\big|\; \tau_1 \,/\, \tau_2 \;\big|\; \tau_1 \,\backslash\, \tau_2 \;\big|\; T(\tau_1, \dots, \tau_k).
$$

### 3.1 Residuation slashes

Milewski’s [“Adjunctions and Hom-Sets” explanation](https://bartoszmilewski.com/2016/04/18/adjunctions/#:~:text=Adjunctions%20and%20Hom-Sets) motivates the correspondence between the two arrow types below [@milewski2016adjunctions]. The [residuation witnesses](expressions.md#210-residuation-witnesses) implement the corresponding rearrangement of arguments.

Read `S / NP` as “produces `S` when an `NP` arrives on the right.” In QVR,
`S \ NP` similarly produces `S` when an `NP` arrives on the left. Thus both
forms put the result type first; the slash direction selects the side of
the missing argument. The adjunctions below express that supplying the
missing argument and treating it as a pending input are two equivalent ways
to describe the same composition, under the residuated-category hypotheses.

QVR uses a *biclosed* slash convention: in both $\tau_1 / \tau_2$ and $\tau_1 \backslash \tau_2$, the lexical-left operand $\tau_1$ is the *result* and the lexical-right operand $\tau_2$ is the *argument*. The direction indicates *where the argument is sought* in a chart firing: $/$ seeks the argument on the right, $\backslash$ seeks it on the left. This is opposite to the standard Lambek convention (where `A\B` reads "needs A on the left to give B"); see the runtime documentation of `SlashCategory` for confirmation.

Under this convention, the right and left residuals are the unique objects characterized by the adjunctions

$$
\mathcal{C}(X \otimes \llbracket \tau_2 \rrbracket,\ \llbracket \tau_1 \rrbracket)
\;\cong\;
\mathcal{C}(X,\ \llbracket \tau_1 / \tau_2 \rrbracket),
\qquad
\mathcal{C}(\llbracket \tau_2 \rrbracket \otimes X,\ \llbracket \tau_1 \rrbracket)
\;\cong\;
\mathcal{C}(X,\ \llbracket \tau_1 \backslash \tau_2 \rrbracket).
$$

In a *free* residuated category over an atom set (the standard case for QVR grammar declarations; see §4), the slashes are *constructors*: $\llbracket \tau_1 / \tau_2 \rrbracket$ and $\llbracket \tau_1 \backslash \tau_2 \rrbracket$ are formal terms, not derived objects. The witnesses of the adjunctions are the schema instances of `curry_right` / `curry_left` on $\mathcal{V}\text{-}\mathbf{Rel}$ ([Expressions §2.10](expressions.md#210-residuation-witnesses)).

### 3.2 Effect application

The form $T(\tau_1, \dots, \tau_k)$ is the application of an effect functor $T$ to a tuple of object arguments; the denotation is $T(\llbracket \tau_1 \rrbracket, \dots, \llbracket \tau_k \rrbracket)$ in the effect-extended universe of [Effects §1](effects.md#1-setting). The effect $T$ must be a fully-instantiated effect name; parametric effects $T[\pi]$ must be fully baked into a concrete identifier before they may appear in a type expression.

## 4. The free residuated universe

A `FreeResiduated(G, depth = d, ops = O)` initializer, used in `object Cat : FreeResiduated(Atoms, …)`, denotes the *bounded free residuated category*

$$
\llbracket \mathrm{FreeResiduated}(G, d, O) \rrbracket
\;=\;
\Bigl\{\, \text{well-formed expressions over $\llbracket G \rrbracket$ using ops from $O$, of formation depth $\le d$}\,\Bigr\},
$$

where $\llbracket G \rrbracket$ is the underlying generator set (typically an `EnumSet` from a `category` declaration or an `object Atoms : {…}` form), and $O \subseteq \{\,/,\ \backslash,\ \otimes,\ \diamondsuit\,\}$ is the chosen operator pool. Formation depth is the height of the syntax tree. As a finite-set object this is a finite enumeration of category expressions. A depth bound limits further type formation: if $A/B$ already has the maximum depth, forming $(A/B)/C$ leaves the inventory. The universal property of the unbounded free residuated category applies to its full syntax; the finite enumeration contains only the terms within the chosen bound.

The operator choice $O = \{/, \backslash\}$ supplies the slash syntax used in *bi-closed* presentations of Lambek calculus [@lambek1958mathematics]; when $O = \{/, \backslash, \otimes\}$ it also supplies product syntax; the selected rules determine the associated deductive behavior; the diamond modality $\diamondsuit$ adds the multimodal extension.

## 5. Rule declarations

A `rule` declaration

```
rule R(X_1, …, X_k) : π_1, …, π_m |- π
```

introduces a typed inference rule over the residuated universe. The variables $X_1, \dots, X_k$ are universally quantified meta-variables ranging over $\llbracket \mathrm{Cat} \rrbracket$; the patterns $\pi_1, \dots, \pi_m, \pi$ are `ObjectExpr`s built from the meta-variables and any atoms in scope. The compiler accepts $m \in \{1, 2\}$ (unary and binary chart rules); declaring a rule with $m \ge 3$ raises `CompileError`.

The denotation is a family of hyperedges in the multicategory of items: for each substitution $\sigma : \{X_1, \dots, X_k\} \to \llbracket \mathrm{Cat} \rrbracket$, the rule contributes the hyperedge

$$
\llbracket R \rrbracket_\sigma \;:\; \llbracket \pi_1 \rrbracket_\sigma,\, \dots,\, \llbracket \pi_m \rrbracket_\sigma \;\longrightarrow\; \llbracket \pi \rrbracket_\sigma,
$$

a generic element of the item algebra's multicategory in the sense of [Weighted Deduction Fragment §2](grammar.md#2-rules). Concretely, in a chart parser whose item algebra is a residuated category $\mathcal{C}$, the rule fires whenever the chart contains cells whose categories unify with the premise patterns; the conclusion cell is the corresponding substituted category.

Standard CCG / Lambek combinators are direct instances under QVR's biclosed slash convention (§3.1):

| Rule | Surface | Denotation |
|---|---|---|
| Forward application | `fwd_app(X, Y) : X / Y, Y |- X` | $(X / Y) \otimes Y \to X$, the counit of the right residuation |
| Backward application | `bwd_app(X, Y) : X, Y \ X |- Y` | $X \otimes (Y \backslash X) \to Y$, the counit of the left residuation |
| Forward composition | `fwd_comp(X, Y, Z) : X / Y, Y / Z |- X / Z` | $(X / Y) \otimes (Y / Z) \to X / Z$, composition of right residuals |

## 6. Schema declarations

The [schema-lifting guide](../guides/effects.md#class-driven-schema-lifting) shows a concrete rule being lifted through an effect. Its [formal lifted denotation](effects.md#3-effect-typed-schema-denotations) states the typeclass laws assumed by that construction.

A schema separates a structural pattern from its instantiation. Instead of
writing a separate application morphism for every pair of grammatical types,
the author writes one pattern and the matcher supplies the types at a firing
site. The parameter space $\Theta$ below is the collection of all
allowed assignments to those pattern variables.

A `schema` declaration

```
schema R(X_{1,1}, …, X_{1,k_1} : τ_1, …, X_{g,1}, …, X_{g,k_g} : τ_g) : Dom -> Cod
```

is the type-polymorphic morphism analogue of a `rule`. Where a rule is a *generic* hyperedge (no further structure), a schema is a *family of morphisms* parameterized by typed meta-variables. Each parameter group $X_{i,1}, \dots, X_{i,k_i} : \tau_i$ declares $k_i$ meta-variables ranging over $\llbracket \tau_i \rrbracket$ (typically $\tau_i = \mathrm{Cat}$).

Let $\bar X = (X_{i,j})_{i, j}$ be the flat tuple of parameters, and let

$$
\Theta \;=\; \prod_i \llbracket \tau_i \rrbracket^{k_i}
$$

be the parameter-space product. The denotation is the parameterized family

$$
\llbracket R \rrbracket \;:\;\; \prod_{\sigma \in \Theta}\;\mathcal{V}\text{-}\mathbf{Rel}\bigl(\llbracket \mathrm{Dom} \rrbracket_\sigma,\, \llbracket \mathrm{Cod} \rrbracket_\sigma\bigr),
$$

i.e. for every concrete substitution $\sigma \in \Theta$ of the parameters, a $\mathcal{V}$-relation between the substituted domain and codomain types. The family is *uniform* in $\sigma$: the morphism at each $\sigma$ is determined by the same syntactic specification, and substitution commutes with composition and tensor.

The arity of a schema (binary vs. unary at the chart level) is derived from the domain shape:

- A top-level `ObjectProduct` $\mathrm{Dom} = D_1 \times D_2$ yields a *binary* schema: a chart hyperedge $D_1 \times D_2 \to \mathrm{Cod}$ that fires on adjacent cells at substituted categories $\sigma D_1, \sigma D_2$.
- Any other domain shape yields a *unary* schema: a hyperedge on a single cell at the substituted domain category.

Schemas are first-class citizens of the chart-parser API: `parser(rules = [R_1, …, R_n])` accepts schemas by name; the chart's joint dispatch ([Effects §4](effects.md#4-joint-type-and-effect-dispatch)) treats them as type-polymorphic firing entries with substitutions inferred from the cells at the firing site.

## 7. Bundles

A `bundle B : [r_1, …, r_n]` declaration is a first-class set binding:

$$
\llbracket B \rrbracket \;=\; \{\,\llbracket r_1 \rrbracket,\, \dots,\, \llbracket r_n \rrbracket\,\},
$$

an ordered collection of schema / rule references. Bundles are accepted at parser and chart-fold rule positions; expansion splices the members at the call site.

The denotation is invariant under bundling: a schema set $\{r_1, \dots, r_n\}$ may be passed as a literal list `[r_1, …, r_n]` or via a bundle name with the same parse result.

## 8. Vertex and edge kinds in graph signatures

A `signature` block may, alongside its `sorts` / `constructors` / `binders`, declare typed vertex and edge kinds for graph-shaped algebras:

```
signature Mol
    vertex_kinds
        Atom : data [dim=32]
        Bond : data [dim=32]
    edge_kinds
        bonded : Atom -- Atom
        in_bond : Atom -> Bond
```

These declarations extend the signature's *theory*. The QVR protocol records the declaration syntax, while the signature theory records its vertex and edge kinds. A `vertex_kind_decl` $K : k\,[\mathrm{dim}\,d]$ extends the signature's kind set with a typed vertex kind whose sort role is $k$ and whose declared embedding dimension is $d$; an `edge_kind_decl` $\varepsilon : K_s\,\mathrm{arrow}\,K_t$ extends the signature's edge set with a typed binary relation between vertex kinds, directed iff `arrow == '->'`. The denotation of a graph signature equipped with these declarations is the category $\mathbf{Graph}_\Sigma$ of finite typed graphs of the prescribed shape; see [Structural Compression §1.3](structural.md#13-graph-signatures) for the formal treatment and §2.4 of the same page for the message-passing encoder that consumes them.

## References


The connection between grammatical composition and directed implication originates in Lambek calculus [@lambek1958mathematics]. QVR’s operand convention is stated in §3.1.
