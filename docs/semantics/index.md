# Denotational semantics of QVR

QVR is the modeling language of quivers. Its
[denotational semantics](https://en.wikipedia.org/wiki/Denotational_semantics) assigns each
phrase a mathematical object, its *denotation*, so that the denotation of a compound phrase is
determined by the denotations of its parts [@moggi1991notions]. The semantics determines when two
QVR programs denote the same Markov kernel or weighted relation, which rewrites a compiler may
apply without changing a program's meaning, and which runtime operations each phrase must
implement.

Category theory and measure theory enter through the definitions on
[Setting and notation](setting.md), which the other pages link to. On the grammar pages, a
[*syntactic category*](https://en.wikipedia.org/wiki/Syntactic_category) is a grammatical type
such as `NP` or `S/NP`, and "category" alone means a category in the sense of
[category theory](https://ncatlab.org/nlab/show/category).

Every executable QVR declaration elaborates to a term of the
[Quivers Indexed Effect Core (QIEC)](../reference/qvr/index.md), the typed core language of
quivers, which has indexed families, effect rows, and effect handlers.

## Pages

1. **[Setting and notation](setting.md):** the categories, kernels, and monads in which phrases
   are interpreted, and the notation the other pages use.
2. **[Semirings, t-norms, and change of base](algebras.md):** the operations that weight
   composition, the built-in instances of `Algebra`, and maps between them.
3. **[Composition rules](composition-rules.md):** the levels of laws a composition rule may
   claim, rules declared in QVR, einsum contractions, and morphism transformations.
4. **[Types and spaces](types-and-spaces.md):** finite sets and standard Borel spaces as the
   denotations of object and space expressions.
5. **[Morphisms](morphisms.md):** weighted relations, stochastic matrices, and Markov kernels,
   with their composition, tensor product, and marginalization.
6. **[Expressions](expressions.md):** the expression combinators (`>>`, `@`, `.marginalize`,
   `fan`, `repeat`, `stack`, `scan`, the parser combinators, and others).
7. **[Programs as kernels](programs.md):** `program` blocks and their steps (`sample`,
   `observe`, `let`, `score`, `marginalize`, `return`) as kernels into probability and
   s-finite measures.
8. **[Weighted deduction](grammar.md):** `deduction` declarations as weighted deduction systems,
   their chart denotation, and the conditions under which agenda evaluation computes it.
9. **[Categorial types, rules, schemas, and bundles](schemas.md):** syntactic categories,
   rule declarations, rule schemata, and bundles of rules.
10. **[Tree and graph autoencoders](structural.md):** signatures, encoders as folds over terms,
    decoders as kernels into terms, and losses.
11. **[Effects and type classes](effects.md):** the type classes of `quivers.monadic`, algebraic
    effects and handlers, and how a parser lifts rules into a type class.
12. **[The program-shape protocol](program-theory.md):** the panproto schema extracted from a
    compiled module, and what equality of such schemas does and does not show.
13. **[Typing and checking](typing.md):** the source checker and the QIEC checker, with the
    judgments each enforces.
14. **[Implementation correspondence and limits](adequacy.md):** which equations the runtime
    realizes directly, which tests exercise them, and where the implementation is approximate.
15. **[Transpilation architecture](transpile-architecture.md):** the pipeline from a compiled
    module to source text for another probabilistic programming language.
16. **[Transpilation correctness](transpile-correctness/index.md):** the correctness criterion
    for each target and the evidence for it, with one page per target:
    [Stan](transpile-correctness/stan.md), [NumPyro](transpile-correctness/numpyro.md),
    [Pyro](transpile-correctness/pyro.md), [PyMC](transpile-correctness/pymc.md),
    [Edward2](transpile-correctness/edward2.md), [Turing.jl](transpile-correctness/turing.md),
    [Gen.jl](transpile-correctness/gen.md), [Church](transpile-correctness/church.md),
    [WebPPL](transpile-correctness/webppl.md), [BUGS](transpile-correctness/bugs.md), and
    [JAGS](transpile-correctness/jags.md).

## Declaration syntax

Most categorical and probabilistic declarations share one shape:

```
KIND NAME : SIGNATURE [k = v, ...] [~ INIT] [BODY]
```

- `KIND` is one of `composition`, `category`, `object`, `morphism`, `bundle`, `program`,
  `contraction`, `export`, `deduction`, `signature`, `encoder`, `decoder`, `loss`, `schema`, or
  `rule`. A `define` declaration introduces a QIEC computation or names an expression such as a
  parser or a composed morphism. `let` binds a deterministic value inside a computation or a
  program body.
- `SIGNATURE` follows the colon: an object value for `object`, a `dom -> cod` arrow for a
  declaration that denotes a morphism, and an `inputs / codomain` shape for `contraction`.
- `[k = v, ...]` is the option block. Each declaration and each step accepts a fixed set of keys
  and rejects any other key.
- `~ INIT` is the optional initializer. A `~ Family(args)` initializer gives a parametric
  family, and a `~ expr` initializer gives a deterministic value. Which forms a `morphism`
  accepts depends on its `role` option
  ([QVR reference](../reference/qvr/index.md)).
- `BODY` is an optional indented block: a rule list for `composition`, `deduction`, or
  `signature`, a step list for `program`, and the body the grammar fixes for each other kind.

The QIEC declarations `index`, `family` with its constructors, `effect` with its operations,
`instance`, `handler`, and `define` have their own forms, given in
[Typing and checking](typing.md) and in the [grammar](grammar.md#11-qiec-fragment).

A pragma is an attribute statement. The outer form `#[k = v, ...]` decorates the next
declaration, and the inner form `#![k = v, ...]` decorates the enclosing module. Lexicon entries
and `rule` lines in a deduction take a trailing `#[k = v, ...]` that sets options for that entry
or rule, such as `learnable`, `bounded`, `parent`, and `weight = expr`
([Weighted deduction](grammar.md#21-learnable-rule-weights-learnable)).

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{V} = (V, \otimes, \bigoplus, \mathbf{1})$ | A set $V$ of weights with a product $\otimes$, a reduction $\bigoplus$, and a unit $\mathbf 1$ ([Setting §1](setting.md#1-weights-and-their-operations)) |
| $\mathbf{FinSet}$ | Finite sets and functions |
| $\mathbf{SBor}$ | Standard Borel spaces and measurable maps |
| $\mathbf{Mat}(\mathcal{V})$ | Finite sets and $\mathcal V$-matrices ([Setting §2](setting.md#2-v-enriched-relations)) |
| $\mathbf{FinStoch}$ | Finite sets and stochastic matrices |
| $\mathbf{BorelStoch}$ | Standard Borel spaces and Markov kernels |
| $\mathcal{D}$, $\mathcal{G}$ | The distribution monad on finite sets and the Giry monad on standard Borel spaces |
| $\llbracket \phi \rrbracket_\rho$ | The denotation of the phrase $\phi$ in the environment $\rho$ |
| $\Gamma \vdash \phi : \tau$ | In the typing context $\Gamma$, the phrase $\phi$ has type $\tau$ |

In the categorical part of the language, a *type* denotes a finite set and a *space* denotes a
standard Borel space. QIEC also has a `Type` kind for primitive, product, distribution, and
types of indexed families, which [Typing and checking](typing.md#4-qiec-kinds-and-types) keeps
separate from the first sense. We write $|X|$ for the cardinality of a finite set $X$ and
$\dim(S)$ for the dimension of a continuous space $S$.
