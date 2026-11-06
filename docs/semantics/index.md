# Denotational Semantics of the QVR DSL

This section gives a mathematical interpretation of the QVR DSL and records how that interpretation relates to the current implementation. Each categorical result states its algebraic hypotheses; the implementation sections identify the operations and tests that realize it.

A *denotation* is what an expression means mathematically. For instance, a
finite transition table denotes a rule assigning a probability distribution
to each input state. Composing two such tables means averaging over the
intermediate state. The notation records this calculation so that we can
ask which program transformations preserve it.

There are two useful routes through these pages:

- **From programming to the mathematics:** read the
  [conceptual introduction](../getting-started/concepts.md), then
  [types and spaces](types-and-spaces.md), [morphisms](morphisms.md), and
  [programs](programs.md). Their examples explain what the formal objects do.
- **From the mathematics to implementation:** begin with
  [setting and notation](setting.md) and [algebras](algebras.md), then use
  [typing](typing.md), [implementation correspondence](adequacy.md), and
  [transpilation correctness](transpile-correctness/index.md) to locate the
  assumptions, executable checks, and limits of each claim.

The following pages develop the definitions and connect them to the implementation.

1. **[Setting and notation](setting.md).** The semantic universe: $\mathcal{V}$-enriched symmetric monoidal closed categories, finite sets, standard Borel spaces, and the global parameters that distinguish discrete, stochastic, and continuous fragments.
2. **[Algebras and base change](algebras.md).** The algebraic structure underlying $\mathcal{V}$-enriched composition, the eleven built-in algebras, and the base-change functors that mediate between them.
3. **[Composition rules](composition-rules.md).** The class hierarchy $\mathbf{CompositionRule} \supseteq \{\mathbf{BilinearForm},\ \mathbf{Semigroupoid} \supseteq \mathbf{Algebra}\}$; user-defined inline composition rules; operadic n-ary contractions via flat wirings; first-class transformations with sequential composition.
4. **[Types and spaces](types-and-spaces.md).** The denotation of `ObjectExpr` and `SpaceExpr` syntactic categories as objects in $\mathbf{FinSet}$ and $\mathbf{SBor}$ respectively.
5. **[Morphisms](morphisms.md).** $\mathcal{V}$-relations, stochastic kernels, and continuous conditional families as the three morphism strata; their composition, tensor product, marginalization, and trace.
6. **[Expressions](expressions.md).** Compositional semantics of expression-level combinators: `>>`, `>>>`, `@`, `.marginalize`, `.change_base`, `fan`, `repeat`, `stack`, `scan`, and the parser combinators.
7. **[Programs](programs.md).** Probability- and weighted-kernel semantics of `program` blocks; the denotation of bind (`<-`), `observe`, `let`, `score`, finite `marginalize`, and `return`; grouped marginalization with multi-observe fibrations; and the boundary between exact finite enumeration and ordinary sampling.
8. **[Typing and checking](typing.md).** The current two-layer checker: source-level objects, spaces, axes, and family shapes; QIEC kinds, indexed families, computations, lexical effect rows, handlers, and program elaboration; plus the guarantees and dynamic limits of that implementation.
9. **[Weighted deduction fragment](grammar.md).** Item algebras, rule systems as hyperedges in a multicategory, semiring-weighted chart enumeration, axiom injectors, strategy independence, and differentiable charts. Includes learnable per-rule and bindings-keyed weights, hierarchical (`parent=`) and bounded (`bounded`) rule parameterizations, alpha-renamed binder blocks, convergent-cycle (`tolerance=`) chart evaluation, and the chart API (`parse(D, x)`, `compose(D_1, D_2)`, `subst(t, v, w)`) that lifts a chart's goal weight into a `program`'s log-joint via a [`score`](programs.md#210-score-factor) step.
10. **[Schemas, rules, categories, bundles](schemas.md).** Category atoms; rule declarations as universally-quantified hyperedges; pattern-polymorphic schema declarations; bundles as first-class rule sets; the residuated type formers $/$, $\backslash$, $T(\cdot)$; free residuated category universes; object- and space-level aliases.
11. **[Structural compression](structural.md).** Signatures as generalized algebraic theories; encoders as initial-algebra catamorphisms into a vector carrier; decoders as Kleisli coalgebras of the Giry monad; losses as attached scalar functionals on training-site traces.
12. **[Compositional effects](effects.md).** Typeclass + algebraic-effects framework over a residuated category universe; class-driven schema lifting; joint type-and-effect dispatch in the chart parser; conservativity over the bare deduction fragment.
13. **[The program-shape protocol](program-theory.md).** Extraction of a compiled environment into a validating panproto `Schema`, plus the limits of schema equality.
14. **[Implementation correspondence and limits](adequacy.md).** Direct correspondences, test evidence, Monte Carlo qualifications, and open gaps.
15. **[Transpilation architecture](transpile-architecture.md).** The `Module -> IRProgram -> panproto.Schema -> bytes` pipeline, its shared family metadata, and backend renderers.
16. **[Transpilation correctness](transpile-correctness/index.md).** Structural, re-emission, external-syntax, and numeric-equivalence evidence, with per-target support notes: [Stan](transpile-correctness/stan.md), [NumPyro](transpile-correctness/numpyro.md), [Pyro](transpile-correctness/pyro.md), [PyMC](transpile-correctness/pymc.md), [Edward2](transpile-correctness/edward2.md), [Turing.jl](transpile-correctness/turing.md), [Gen.jl](transpile-correctness/gen.md), [Church](transpile-correctness/church.md), [WebPPL](transpile-correctness/webppl.md), [BUGS](transpile-correctness/bugs.md), and [JAGS](transpile-correctness/jags.md).

For the categorical background, see [@maclane1998categories; @kelly1982basic].
For probability kernels and their compositional interpretation, see
[@giry1982categorical; @fritz2020synthetic].

## Declaration syntax

The categorical and probabilistic declarations use a shared option-block and
initializer vocabulary. Their common schematic shape is:

```
KIND NAME : SIGNATURE [k = v, ...] [~ INIT] [BODY]
```

QIEC declarations have separate checked forms: `index`, `family` and `constructor`, `effect` and
operations, `instance`, `handler`, and computation-valued `define`. Their
static telescopes, runtime telescopes, indexed results, and effect rows are
described in [Typing and checking](typing.md) and summarized in the
[grammar](grammar.md#11-qiec-fragment).

For the categorical/probabilistic shape above:

* `KIND` is one of `composition`, `category`, `object`, `morphism`, `bundle`, `program`, `contraction`, `export`, `deduction`, `signature`, `encoder`, `decoder`, `loss`, `schema`, or `rule`. A `define` may instead introduce a QIEC computation or one of the established expression aliases such as a parser or composed morphism. `let` is reserved for deterministic bindings inside computations and program bodies.
* `SIGNATURE` is a colon-prefixed phrase (an object value for `object`, a `dom -> cod` arrow for everything that denotes a morphism, an `inputs / codomain` shape for `contraction`, etc.).
* `[k = v, ...]` is the shared option block. Each declaration or step accepts a closed set of keys and rejects ignored or misspelled entries. Options select a role, shape, family, reduction, or elaboration policy.
* `~ INIT` is the optional *initializer*: either a `Family(args)` clause (sampled stochastic kernel, evaluated through the family registry) or an arbitrary `expr` (deterministic morphism, evaluated through the expression denotation). On a `morphism` declaration the initializer interacts with `role` in the option block: `role=latent` admits only `~ Family(...)`; `role=let` / `role=observed` admits only `~ expr`; `role=kernel` admits both.
* `[BODY]` is an optional indented block: a rule list for `composition`, `deduction`, or `signature`; a step list for `program`; or the declaration-specific body defined by the grammar.

Pragmas are top-level *attribute* statements that decorate the *next* declaration (outer form `#[k = v, ...]`) or the *enclosing module* (inner form `#![k = v, ...]`). They carry the same `pragma_entry` shape as the unified option block; the compiler attaches the entries to the decorated declaration or module-level environment.

The deduction fragment adds a parallel form: lexicon entries and individual `rule` lines admit a trailing `#[k = v, ...]` pragma that controls per-entry / per-rule behavior (`learnable`, `bounded`, `parent`, `weight = expr`, etc.; see [Weighted deduction fragment §2.1–§2.3](grammar.md#21-learnable-rule-weights-learnable)).

## Conventions

Throughout, we use the following conventions.

| Symbol | Meaning |
|--------|---------|
| $\mathcal{V}$ | A QVR algebra (see [Algebras §1](algebras.md)). |
| $\otimes,\ \mathbf{1}$ | Monoidal product and unit of $\mathcal{V}$. |
| $\bigoplus$ | The reduction combining alternatives; a lattice join only where the algebra supplies one. |
| $\mathbf{FinSet}$ | The category of finite sets. |
| $\mathbf{SBor}$ | The category of standard Borel spaces with measurable maps. |
| $\mathcal{V}\text{-}\mathbf{Rel}$ | The $\mathcal{V}$-enriched category of $\mathcal{V}$-relations on finite sets. |
| $\mathbf{Stoch}$ | Finite sets and stochastic matrices, viewed as a subcategory of kernels. |
| $\mathbf{Kern}$ | The category of standard Borel spaces and Markov kernels. |
| $\mathcal{G}$ | The Giry monad (discrete or continuous, disambiguated by context). |
| $\llbracket \phi \rrbracket$ | The denotation of phrase $\phi$. |
| $\Gamma \vdash \phi : \tau$ | $\phi$ is well-typed of type $\tau$ under environment $\Gamma$. |
| $\rho$ | A semantic environment (assignment of denotations to free names). |

In the categorical fragment, a *type* is a finite-set object and a
*space* is a standard Borel object. QIEC additionally has a `Type` kind for
primitive, product, distribution, and indexed-family types; [Typing and
checking](typing.md#4-qiec-kinds-and-types) keeps those two uses separate. We
write $|X|$ for the cardinality of a finite set $X$, and $\dim(S)$ for the
dimension of a continuous space $S$.

## Audience

The [notation table](../getting-started/concepts.md#7-reading-the-notation-and-finding-the-next-explanation) decodes the symbols used across these pages. For additional background, Milewski’s [functor example using optional values](https://bartoszmilewski.com/2015/01/20/functors/) [@milewski2015functors] and DIPPL’s [coin-flip program](https://dippl.org/chapters/02-webppl.html#and-inference) [@goodman2014design] connect the categorical and probabilistic vocabulary to executable calculations.

This document is written for users who wish to reason formally about QVR programs: to verify that two `.qvr` files denote the same morphism, to prove that an optimization pass preserves meaning, or to understand the precise relationship between the syntactic AST, the `dx.Model` value-type layer, and the underlying tensor computations. The examples and notation explanations provide an entry point without prior category-theory or probabilistic-programming training. Enriched categories [@kelly1982basic] and probability kernels [@giry1982categorical; @fritz2020synthetic] provide the mathematical background.


## References


For enriched categories, see [@kelly1982basic]. For probability measures and kernels, see [@giry1982categorical; @fritz2020synthetic]. The [concepts primer](../getting-started/concepts.md) introduces the notation used here.
