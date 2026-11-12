# API Reference

The public API of quivers, by subpackage, with each module page generated from its docstrings.

| Task | Start here | Result |
|---|---|---|
| Read the basic concepts | [Concepts](../getting-started/concepts.md) | Composition of weighted relations, the choice of `Algebra`, and the steps of a program |
| Load a `.qvr` model | [DSL entry points](dsl/index.md) | Parsed or compiled source, depending on the entry point |
| Build a weighted computation in Python | [Objects](core/objects.md), [morphisms](core/morphisms.md), and [algebras](core/algebras.md) | Finite sets, weighted tables between them, and the rules that combine their weights |
| Fit a model to data | [Stochastic variational inference](inference/svi.md) (SVI) or [Markov chain Monte Carlo](inference/mcmc.md) (MCMC) | An approximation to the posterior, with different computational tradeoffs |
| Fit a regression formula | [Formulas](formulas/index.md) | A generated model and fitted result from tabular data |
| Inspect the results | [Diagnostics](diagnostics/index.md) | Checks of inference behaviour and predictive adequacy |
| Extend checking or execution | [Quivers Indexed Effect Core](qiec/index.md) | Explicit types, computations, and effect interfaces |
| Translate to another probabilistic programming language (PPL) | [Transpilation](transpile/index.md) | Target source, or a diagnostic naming the unsupported feature |

## Core Package

The `core` package defines the objects, morphisms, composition rules, and tensor operations
shared by the other packages.

- **[Objects](core/objects.md)**: finite and enumerated sets, products, coproducts, free monoids, and the categorial (Lambek) types over a set of atoms up to a fixed depth
- **[Algebras](core/algebras.md)**: the `Algebra` class and its eleven built-in instances, each a [semiring](https://en.wikipedia.org/wiki/Semiring), a [t-norm](https://en.wikipedia.org/wiki/T-norm) with its t-conorm, or a product with capped sum, plus duals and user-defined instances
- **[Transformations](core/transformations.md)**: entrywise maps between `Algebra` instances and row normalizations, applied by [change of base](https://ncatlab.org/nlab/show/change+of+enriching+category)
- **[Morphisms](core/morphisms.md)**: base morphism classes and their composition
- **[Tensor Operations](core/tensor_ops.md)**: noisy-OR contraction and reduction, noisy-AND reduction, and componentwise lifting of tensors
- **[Utilities](core/util.md)**: clamping and logarithm helpers for fuzzy computations
- **[Wiring](core/wiring.md)**: [einsum](https://en.wikipedia.org/wiki/Einstein_notation) contractions of many morphisms under one composition rule

## Categorical Package

The `categorical` package contains functors, natural transformations, adjunctions, and monoidal
structures.

- **[Functors](categorical/functors.md)**: [functors](https://ncatlab.org/nlab/show/functor) between categories
- **[Natural Transformations](categorical/natural_transformations.md)**: [natural transformations](https://ncatlab.org/nlab/show/natural+transformation) between functors
- **[Adjunctions](categorical/adjunctions.md)**: [adjoint](https://ncatlab.org/nlab/show/adjunction) pairs of functors
- **[Monoidal](categorical/monoidal.md)**: [monoidal](https://ncatlab.org/nlab/show/monoidal+category) structures on morphisms
- **[Change of Base](categorical/base_change.md)**: change of base between Boolean and fuzzy weights
- **[Traced](categorical/traced.md)**: [traced monoidal](https://ncatlab.org/nlab/show/traced+monoidal+category) structure

## Monadic Package

The `monadic` package contains the [type class](https://en.wikipedia.org/wiki/Type_class)
hierarchy, monad instances and transformers, comonads, algebras, distributive laws, and algebraic
effects.

- **[Typeclass Hierarchy](monadic/typeclasses.md)**: `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, `MonadTrans` abstract base classes
- **[Stdlib Effect Instances](monadic/instances.md)**: `Identity`, `Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`, `List`
- **[Monad Transformers](monadic/transformers.md)**: `StateT`, `ReaderT`, `MaybeT`, `ContT`, `WriterT`
- **[Algebraic Effects](monadic/algebraic.md)**: `Operation`, `EffectSignature`, `Handler`, `FreeMonad`
- **[Bridges](monadic/bridges.md)**: `Kleisli` and `ArrowMonad`, which connect the monad and arrow hierarchies
- **[Typeclass Theories](monadic/theories.md)**: the type classes stated as panproto theories
- **[Typeclass Laws](monadic/laws.md)**: the laws of each type class as checkable records
- **[Concrete Monads](monadic/monads.md)**: `KleisliCategory`, `FuzzyPowersetMonad`, `FreeMonoidMonad`
- **[Comonads](monadic/comonads.md)**: [comonads](https://ncatlab.org/nlab/show/comonad) and their co-Kleisli categories
- **[Algebras and Coalgebras](monadic/algebras.md)**: [algebras of a monad](https://ncatlab.org/nlab/show/algebra+over+a+monad) and coalgebras of a comonad
- **[Distributive Laws](monadic/distributive_laws.md)**: [distributive laws](https://ncatlab.org/nlab/show/distributive+law) between monads and comonads

## Arrows Package

The `arrows` package contains the hierarchy of [arrows](https://en.wikipedia.org/wiki/Arrow_(computer_science))
[@hughes2000arrows], which parallels the monad hierarchy.

- **[Typeclass Hierarchy](arrows/typeclasses.md)**: `Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, `ArrowPlus`
- **[Arrow Theories](arrows/theories.md)**: the arrow classes stated as panproto theories
- **[Arrow Instances](arrows/instances.md)**: arrows of functions, weighted relations, and stochastic transitions

## Enriched Package

The `enriched` package contains constructions on categories
[enriched](https://ncatlab.org/nlab/show/enriched+category) over a chosen `Algebra`.

- **[Ends & Coends](enriched/ends_coends.md)**: [ends and coends](https://ncatlab.org/nlab/show/end)
- **[Kan Extensions](enriched/kan_extensions.md)**: left and right [Kan extensions](https://ncatlab.org/nlab/show/Kan+extension) along functions between finite sets
- **[Weighted Limits](enriched/weighted_limits.md)**: [weighted limits](https://ncatlab.org/nlab/show/weighted+limit) and colimits
- **[Profunctors](enriched/profunctors.md)**: [profunctors](https://ncatlab.org/nlab/show/profunctor) and their composition
- **[Yoneda](enriched/yoneda.md)**: presheaves and the [Yoneda embedding](https://ncatlab.org/nlab/show/Yoneda+lemma)
- **[Day Convolution](enriched/day_convolution.md)**: the [Day convolution](https://ncatlab.org/nlab/show/Day+convolution) product of presheaves
- **[Optics](enriched/optics.md)**: [lenses](https://ncatlab.org/nlab/show/lens+%28in+computer+science%29) and other optics

## Stochastic Package

The `stochastic` package contains finite stochastic morphisms, finite distribution families, and
weighted deduction systems.

- **[Morphisms](stochastic/morphisms.md)**: [Markov kernels](https://en.wikipedia.org/wiki/Markov_kernel) between finite sets, stored as stochastic matrices
- **[Families](stochastic/families.md)**: parametric families of finite distributions
- **[Transforms](stochastic/transforms.md)**: conditioning, mixing, and factoring of stochastic morphisms
- **[Queries](stochastic/queries.md)**: probabilities, marginals, and expectations
- **[Giry Monad](stochastic/giry.md)**: the [distribution monad](https://ncatlab.org/nlab/show/distribution+monad) on finite sets (class `GiryMonad`) and its Kleisli category `FinStoch`
- **[Weighted Deduction](stochastic/deduction.md)**: rule schemas, chart semirings, and chart parsers

## Continuous Package

The `continuous` package defines continuous spaces, distribution families, parameter sources, and
continuous morphisms.

- **[Spaces](continuous/spaces.md)**: typed carriers and support constraints for continuous values
- **[Morphisms](continuous/morphisms.md)**: kernels with log densities and sampling methods
- **[Families](continuous/families.md)**: families of continuous distributions
- **[Programs](continuous/programs.md)**: probabilistic programs over continuous spaces
- **[Boundaries](continuous/boundaries.md)**: discretization and embedding between finite and continuous spaces
- **[Flows](continuous/flows.md)**: [normalizing flows](https://en.wikipedia.org/wiki/Flow-based_generative_model)

## DSL Package

The `dsl` package implements the QVR language. A [tree-sitter](https://tree-sitter.github.io/)
grammar, driven through [panproto](https://github.com/panproto/panproto), parses source, and
[didactic](https://github.com/panproto/didactic) `Model` types hold the AST nodes and value
types.

- **[Parser](dsl/parser.md)**: `parse`, `parse_file`, and `ParseError`
- **[AST Nodes](dsl/ast_nodes.md)**: the `Model` class of every syntactic node
- **[Compiler](dsl/compiler.md)**: compilation from AST to `Program` (`Compiler`, `CompileError`)
- **[Resolution](dsl/resolution.md)**: the [lenses](https://ncatlab.org/nlab/show/lens+%28in+computer+science%29) (`dx.Lens`) from `ObjectExpr` to runtime values and back
- **[Program Theory](dsl/program_theory.md)**: `QVR_PROGRAM_PROTOCOL` and `extract_program_schema`, which emit a panproto `Schema` for a compiled program

## Inference Package

The `inference` package contains variational inference, MCMC, and posterior predictive sampling.

- **[Trace](inference/trace.md)**: execution traces of programs
- **[Conditioning](inference/conditioning.md)**: conditioning on observations
- **[Guides](inference/guide.md)**: variational guides
- **[ELBO](inference/elbo.md)**: the evidence lower bound and related objectives
- **[SVI](inference/svi.md)**: stochastic variational inference
- **[Predictive](inference/predictive.md)**: posterior predictive sampling
- **[MCMC](inference/mcmc.md)**: Hamiltonian Monte Carlo (HMC) and No-U-Turn Sampler (NUTS) kernels, and the runner for several chains

## Data, Diagnostics, and Formulas

These packages connect model definitions to tabular data and Bayesian analysis.

- **[Data](data/index.md)**: schemas for data frames and column encoders
- **[Diagnostics](diagnostics/index.md)**: conversion of traces for ArviZ, model comparison, and posterior predictive checks
- **[Formulas](formulas/index.md)**: R-style formula parsing, QVR generation, and the `fit` entry point

## Analysis Package

The `analysis` package derives metadata from checked programs without changing them.

- **[Chain shapes](analysis/chain_shape.md)**: step shapes and intermediate dimensions, located in the source
- **[Initialization](analysis/init_spec.md)**: initialization recipes for each `Algebra`
- **[Saturation diagnostics](analysis/saturation.md)**: warnings tied to source locations
- **[Program inspection](analysis/plate_graph.md)**: plate graphs, their renderers, and lexical scope queries

## Structural and Effects Packages

The `structural` package defines signatures, encoders, decoders, and attached losses for tree and
graph autoencoders over terms. The `effects` package gives handler stacks for conditioning,
intervention, reweighting, tracing, replay, and reparameterization.

- **[Signatures](structural/signature.md)**, **[Encoders](structural/encoder.md)**, **[Decoders](structural/decoder.md)**, **[Loss Registry](structural/losses.md)**, **[Shapes](structural/shapes.md)**
- **[Effects overview](effects/index.md)**, **[Handler base](effects/base.md)**, **[Handlers](effects/handlers.md)**, **[Traces](effects/trace.md)**, **[Checked programs](effects/checked_program.md)**

## Transpilation and QIEC

The `transpile` package lowers checked programs through an IR that no target depends on and
renders them in eleven probabilistic programming languages. The `qiec` package holds the
[Quivers Indexed Effect Core](qiec/index.md) (QIEC), the typed core language that program
elaboration, execution, tooling, and structural computations share.

- **[Transpilation overview](transpile/index.md)**, **[Plan](transpile/plan.md)**, **[Renderers](transpile/renderers.md)**
- **[QIEC overview](qiec/index.md)**, **[Checking](qiec/checking.md)**, **[Evaluator](qiec/evaluator.md)**, **[Execution](qiec/execution.md)**

## Root Module

- **[Program](program.md)**: `Program`, the container a compiled QVR module returns
