# Categorial types, rules, schemas, and bundles

The declarations `category`, `rule`, `schema`, and `bundle` give the grammar of a chart parser in
the style of [categorial grammar](https://en.wikipedia.org/wiki/Categorial_grammar), in which a
syntactic category is either atomic, such as `NP` or `S`, or built with slashes, such as `S / NP`
[@lambek1958mathematics]. A `category` declaration adds atomic categories, a `rule` is an inference
rule over categories with variables, a `schema` is a family of morphisms indexed by categories, and
a `bundle` names a list of rules and schemata. On these pages "category" alone means a category in
category theory, and the grammatical sense is always "syntactic category".

A rule with the pattern `X / Y, Y |- X` combines a category that seeks a `Y` on its right with an
adjacent `Y` to give an `X`. Substituting `S` for `X` and `NP` for `Y` gives one instance of it.

## 1. Atomic categories {#1-category-atoms}

```
category NP, S, VP
```

adds atomic syntactic categories to the environment $\rho_{\mathrm{cat}}$, which is separate from
the object environment $\rho_{\mathrm{obj}}$ and the space environment $\rho_{\mathrm{spc}}$:

$$
\llbracket \mathrm{category}\ C_1, \dots, C_n \rrbracket : \rho_{\mathrm{cat}} \longmapsto \rho_{\mathrm{cat}} \cup \{C_1, \dots, C_n\}.
$$

The atoms generate the categorial types of chart parsers (§4) and of rule and schema declarations
(§5, §6). An atom is not a finite set and cannot be the domain or codomain of a morphism: such a
use is the compile error "undefined object or space". To use the categorial types over some atoms
as a finite set, declare a `FreeResiduated` object over them (§4), or pass the atoms to
`parser(categories=[...])`.

## 2. Object declarations with categorial types

The declaration `object N : τ` binds $N$ according to what $\tau$ is:

- When $\tau$ is a finite set expression, such as `FinSet 16`, `A * B`, `A + B`, or a name bound to
  a finite set, $N$ is bound in $\rho_{\mathrm{obj}}$ and may be the domain or codomain of a later
  morphism.
- When $\tau$ is a space, such as `Real D`, `Simplex K`, or `Covariance D`, $N$ is bound in
  $\rho_{\mathrm{spc}}$.
- When $\tau$ contains a slash or an effect application, and is neither a finite set nor a space,
  $N$ is bound as an abbreviation for the expression $\tau$, which the matcher of rule and schema
  patterns expands wherever $N$ occurs. Such an abbreviation is not an object and may appear only in
  `rule` and `schema` patterns.

In every case later occurrences of $N$ mean what $N$ abbreviates, and the
[program-shape protocol](program-theory.md) records no separate entry for the binding.

## 3. Slashes and effect application {#3-residuated-type-formers}

Object expressions have two forms beyond products and coproducts:

$$
\tau ::= \dots \;\big|\; \tau_1 / \tau_2 \;\big|\; \tau_1 \backslash \tau_2 \;\big|\; T(\tau_1, \dots, \tau_k).
$$

### 3.1 Slashes {#31-residuation-slashes}

QVR writes slashes in Steedman's notation, with the result first
[@steedman2000syntactic]: `S / NP` yields `S` from an `NP` on its right, and `S \ NP` yields `S`
from an `NP` on its left. In $\tau_1 / \tau_2$ and in $\tau_1 \backslash \tau_2$, $\tau_1$ is the
result and $\tau_2$ the argument, and the slash says on which side the argument is found. Lambek's
notation writes the backward slash the other way, with the argument under the slash, so that his
`NP\S` is QVR's `S \ NP` [@lambek1958mathematics]. The runtime class `SlashCategory` stores the result
and the argument as separate fields.

A [*biclosed monoidal category*](https://ncatlab.org/nlab/show/closed+monoidal+category) has, for
each object, a right and a left internal hom, characterized by natural bijections

$$
\mathcal{C}(X \otimes B, A) \cong \mathcal{C}(X, A / B),
\qquad
\mathcal{C}(B \otimes X, A) \cong \mathcal{C}(X, A \backslash B),
$$

written here in QVR's notation, with $A$ the result and $B$ the argument
[@lambek1969deductive]. These are the two residuations of the
[residuated lattice](https://en.wikipedia.org/wiki/Residuated_lattice) that the Lambek calculus
describes when only provability, and not proofs, is kept. In the free biclosed monoidal category
on a set of atoms, the slashes are constructors: $A / B$ and $A \backslash B$ are formal terms. Its
morphisms are the proofs of the Lambek calculus [@lambek1969deductive]. QVR uses only its objects,
the categorial types, as syntactic categories, and the rules of §5 play the part of its proofs.

### 3.2 Effect application

$T(\tau_1, \dots, \tau_k)$ applies an effect $T$, such as a continuation or a set of alternatives,
to categorial types, giving the categorial type of an expression with that effect
([Effects and type classes §1](effects.md#1-setting)) [@charlow2025scope]. $T$ must be a declared
effect with all its parameters filled in: a parametric effect $T[\pi]$ must be instantiated to a
concrete name before it appears in a type.

## 4. Categorial types of bounded depth {#4-the-free-residuated-universe}

The declaration `object Cat : FreeResiduated(G, depth = d, ops = O)` binds `Cat` to the finite set

$$
\llbracket \mathrm{FreeResiduated}(G, d, O) \rrbracket = \bigl\{\text{categorial types over the atoms } \llbracket G \rrbracket \text{ built with the connectives } O \text{, of depth at most } d\bigr\},
$$

where $\llbracket G \rrbracket$ is the set of atoms, usually from a `category` declaration or an
`object Atoms : {...}` declaration, and $O$ names connectives among `slash` (both $/$ and
$\backslash$), `product` ($\otimes$), `unit`, `diamond` ($\diamondsuit$), and `box` ($\square$). The depth
of a type is the height of its syntax tree, so atoms have depth $0$. The runtime class is
[`FreeResiduated`](../api/core/objects.md), which enumerates the types as strings.

This set is a finite set of objects of the free biclosed monoidal category of §3.1, not a category:
it has no morphisms, and it is not closed under the connectives, since $(A / B) / C$ is outside it
when $A / B$ already has depth $d$. A statement that uses the universal property of the free
category holds for the unbounded set of types, not for this truncation. With
$O = \{/, \backslash\}$ the types are those of the Lambek calculus without product
[@lambek1958mathematics], with $\otimes$ they include products, and with $\diamondsuit$ and $\square$ they
include the unary modalities of multimodal type-logical grammar [@moortgat1997categorial]. Which
inferences hold among the types is fixed by the rules a parser is given, not by $O$.

## 5. Rule declarations {#5-rule-declarations}

```
rule R(X_1, ..., X_k) : π_1, ..., π_m |- π
```

declares an inference rule over categorial types. The variables $X_1, \dots, X_k$ range over the
categorial types $\llbracket \mathrm{Cat} \rrbracket$, and the patterns $\pi_1, \dots, \pi_m, \pi$ are object
expressions built from the variables and the atoms in scope. The compiler accepts one or two
premises, and a rule with three or more is a compile error.

For each substitution $\sigma : \{X_1, \dots, X_k\} \to \llbracket \mathrm{Cat} \rrbracket$, the rule
contributes the rule instance

$$
\llbracket R \rrbracket_\sigma : \llbracket \pi_1 \rrbracket_\sigma, \dots, \llbracket \pi_m \rrbracket_\sigma \longrightarrow \llbracket \pi \rrbracket_\sigma,
$$

a hyperedge in the sense of [Weighted deduction §2](grammar.md#2-rules). A chart parser fires it
whenever cells adjacent in the input have categories that unify with the premises, and the new
cell has the conclusion under the unifying substitution.

The rules of [combinatory categorial grammar](https://en.wikipedia.org/wiki/Combinatory_categorial_grammar)
(CCG) [@steedman2000syntactic] and of the Lambek calculus are instances, in QVR's notation:

| Rule | Declaration | Biclosed reading |
|---|---|---|
| Forward application | `fwd_app(X, Y) : X / Y, Y |- X` | $(X / Y) \otimes Y \to X$, evaluation of the right internal hom |
| Backward application | `bwd_app(X, Y) : X, Y \ X |- Y` | $X \otimes (Y \backslash X) \to Y$, evaluation of the left internal hom |
| Forward composition | `fwd_comp(X, Y, Z) : X / Y, Y / Z |- X / Z` | $(X / Y) \otimes (Y / Z) \to X / Z$, composition of right internal homs |

## 6. Schema declarations {#6-schema-declarations}

The [schema lifting guide](../guides/effects.md#class-driven-schema-lifting) lifts a rule through an
effect, and [Effects and type classes §3](effects.md#3-effect-typed-schema-denotations) states its
denotation. A schema states one pattern for many categories, and the matcher fills in the categories
where it fires.

```
schema R(X_{1,1}, ..., X_{1,k_1} : τ_1, ..., X_{g,1}, ..., X_{g,k_g} : τ_g) : Dom -> Cod
```

declares a family of morphisms indexed by its typed variables. Each group
$X_{i,1}, \dots, X_{i,k_i} : \tau_i$ declares $k_i$ variables ranging over $\llbracket \tau_i \rrbracket$, usually
$\tau_i = \mathrm{Cat}$. With $\Theta = \prod_i \llbracket \tau_i \rrbracket^{k_i}$ the set of assignments
to all variables, the schema denotes

$$
\llbracket R \rrbracket \in \prod_{\sigma \in \Theta} \mathbf{Mat}(\mathcal{V})\bigl(\llbracket \mathrm{Dom} \rrbracket_\sigma, \llbracket \mathrm{Cod} \rrbracket_\sigma\bigr),
$$

a $\mathcal V$-matrix from the substituted domain to the substituted codomain for each assignment
$\sigma$. One declaration gives the matrix at every $\sigma$.

The number of premises of a schema in a chart follows from the shape of its domain:

- a product domain $\mathrm{Dom} = D_1 \times D_2$ gives a binary rule $D_1 \times D_2 \to \mathrm{Cod}$,
  which fires on adjacent cells with categories $\sigma D_1$ and $\sigma D_2$;
- any other domain gives a unary rule on one cell with category $\sigma \mathrm{Dom}$.

`parser(rules = [R_1, ..., R_n])` accepts schemata by name, and the chart treats each as a rule whose
substitution is found from the cells where it fires
([Effects and type classes §4](effects.md#4-joint-type-and-effect-dispatch)).

## 7. Bundles {#7-bundles}

```
bundle B : [r_1, ..., r_n]
```

binds $B$ to the list of rules and schemata $r_1, \dots, r_n$, in order. A bundle may be given wherever
`parser` and `chart_fold` take rules, and it is replaced by its members there, so passing `B` and
passing the list `[r_1, ..., r_n]` give the same parser.

## 8. Vertex and edge kinds in graph signatures

A `signature` may declare kinds of vertices and edges for graphs, beside its sorts, constructors, and
binders:

```
signature Mol
    vertex_kinds
        Atom : data [dim=32]
        Bond : data [dim=32]
    edge_kinds
        bonded : Atom -- Atom
        in_bond : Atom -> Bond
```

A vertex kind $K : k\,[\mathrm{dim}\ d]$ has the sort role $k$ and an embedding of dimension $d$. An edge
kind $\varepsilon : K_s \to K_t$ or $\varepsilon : K_s \mathbin{-\!-} K_t$ is a binary relation between
vertex kinds, directed when written with `->` and undirected with `--`. Graph signatures and the
encoders that read them are described on
[Tree and graph autoencoders §1.3](structural.md#13-graph-signatures).
