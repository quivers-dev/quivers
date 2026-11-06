# API Reference

This reference groups the public API by subpackage. Module pages obtain
class, function, and property documentation from the source docstrings.

Choose an entry point by the task you want to perform:

| Task | Start here | What it gives you |
|---|---|---|
| Read the basic concepts | [Concepts by example](../getting-started/concepts.md) | Worked table composition and Bayesian inference before formal terminology |
| Load a `.qvr` model | [DSL entry points](dsl/index.md) | Parsed or compiled source, depending on the entry point |
| Build a weighted computation in Python | [Objects](core/objects.md), [morphisms](core/morphisms.md), and [algebras](core/algebras.md) | Named axes, weighted tables, and their combination rules |
| Fit a model to data | [SVI](inference/svi.md) or [MCMC](inference/mcmc.md) | An approximation to the posterior, with different computational tradeoffs |
| Fit a regression formula | [Formulas](formulas/index.md) | A generated model and fitted result from tabular data |
| Inspect the results | [Diagnostics](diagnostics/index.md) | Checks of inference behavior and predictive adequacy |
| Extend checking or execution | [Indexed effect core](qiec/index.md) | Explicit types, computations, and effect interfaces |
| Translate to another PPL | [Transpilation](transpile/index.md) | Target source or a specific unsupported-capability diagnostic |

Each module page introduces an operation, then gives its signatures, shapes,
return values, and exceptions. The linked guides explain the mathematical
assumptions and work through examples of the operations.

## Core Package

The `core` package defines the categorical objects, algebras,
morphisms, and tensor operations shared by the other packages.

- **[Objects](core/objects.md)**: Finite and enumerated sets, products, coproducts, free monoids, and free residuated universes
- **[Algebras](core/algebras.md)**: Enrichment algebras for $\mathcal{V}$-enriched composition (the eleven built-in cases plus duals and user-defined extensions)
- **[Algebra Morphisms](core/transformations.md)**: Homomorphisms between algebras, used for base-change
- **[Morphisms](core/morphisms.md)**: Base morphism classes and composition operations
- **[Tensor Operations](core/tensor_ops.md)**: Vectorized operations on tensors
- **[Utilities](core/util.md)**: Helper functions for core functionality

## Categorical Package

The `categorical` package contains functors, natural transformations,
adjunctions, and monoidal structures.

- **[Functors](categorical/functors.md)**: Mappings between categories preserving structure
- **[Natural Transformations](categorical/natural_transformations.md)**: Morphisms between functors
- **[Adjunctions](categorical/adjunctions.md)**: Adjoint functor pairs
- **[Monoidal](categorical/monoidal.md)**: Monoidal category structures
- **[Base Change](categorical/base_change.md)**: Change of enriching category
- **[Traced](categorical/traced.md)**: Traced monoidal categories

## Monadic Package

The `monadic` package contains the typeclass hierarchy, concrete monad
instances, comonads, algebras, distributive laws, transformers, and
algebraic effects.

- **[Typeclass Hierarchy](monadic/typeclasses.md)**: `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, `MonadTrans` ABCs
- **[Stdlib Effect Instances](monadic/instances.md)**: `Identity`, `Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`, `List`
- **[Monad Transformers](monadic/transformers.md)**: `StateT`, `ReaderT`, `MaybeT`, `ContT`, `WriterT`
- **[Algebraic Effects](monadic/algebraic.md)**: `Operation`, `EffectSignature`, `Handler`, `FreeMonad`
- **[Bridges](monadic/bridges.md)**: `Kleisli` / `ArrowMonad` connecting the monad and arrow towers
- **[Typeclass Theories](monadic/theories.md)**: panproto-theory mirrors
- **[Concrete Monads](monadic/monads.md)**: `KleisliCategory`, `FuzzyPowersetMonad`, `FreeMonoidMonad`
- **Comonads**, **Algebras**, **Distributive Laws**

## Arrows Package

The `arrows` package contains the Hughes-style hierarchy parallel to
the monad hierarchy.

- **[Typeclass Hierarchy](arrows/typeclasses.md)**: `Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, `ArrowPlus`
- **[Arrow Theories](arrows/theories.md)**: panproto-theory mirrors

## Enriched Package

The `enriched` package contains constructions for categories enriched
over a selected algebra.

- **[Ends & Coends](enriched/ends_coends.md)**: End and coend computations in enriched categories
- **[Kan Extensions](enriched/kan_extensions.md)**: Left and right Kan extensions
- **[Weighted Limits](enriched/weighted_limits.md)**: Limits and colimits weighted by enrichment
- **[Profunctors](enriched/profunctors.md)**: Profunctor (bimodule) definitions
- **[Yoneda](enriched/yoneda.md)**: Yoneda embeddings and lemmas
- **[Day Convolution](enriched/day_convolution.md)**: Day convolution product
- **[Optics](enriched/optics.md)**: Optics and lens constructions

## Stochastic Package

The `stochastic` package contains stochastic morphisms, finite
distribution families, and weighted deduction systems.

- **[Morphisms](stochastic/morphisms.md)**: Stochastic relations and kernels
- **[Families](stochastic/families.md)**: Parametric families of distributions
- **[Transforms](stochastic/transforms.md)**: Operations on stochastic morphisms
- **[Queries](stochastic/queries.md)**: Probabilistic queries and computations
- **[Giry Monad](stochastic/giry.md)**: The Giry monad construction
- **[Weighted Deduction](stochastic/deduction.md)**: Rule schemas, chart semirings, and CKY parsers

## Continuous Package

The `continuous` package defines continuous spaces, distribution
families, parameter sources, and continuous morphisms.

- **[Spaces](continuous/spaces.md)**: Typed carriers and support constraints for continuous values
- **[Morphisms](continuous/morphisms.md)**: Operational kernels with log-density and sampling methods
- **[Families](continuous/families.md)**: Families of continuous distributions
- **[Programs](continuous/programs.md)**: Probabilistic programs in continuous domains
- **[Boundaries](continuous/boundaries.md)**: Discretization and embedding between finite and continuous carriers
- **[Flows](continuous/flows.md)**: Normalizing flows and transformations

## DSL Package

The `dsl` package implements the QVR language. Panproto parses the
`qvr` tree-sitter grammar, while didactic Models represent AST nodes
and value types.

- **[Parser](dsl/parser.md)**: panproto-driven parser walker (`parse`, `parse_file`, `ParseError`)
- **[AST Nodes](dsl/ast_nodes.md)**: didactic Model definitions for every syntactic node
- **[Compiler](dsl/compiler.md)**: lowering from AST to `Program` (`Compiler`, `CompileError`)
- **[Resolution](dsl/resolution.md)**: bidirectional `dx.Lens` family from `ObjectExpr` to runtime values
- **[Program Theory](dsl/program_theory.md)**: `QVR_PROGRAM_PROTOCOL` and `extract_program_schema` for emitting a panproto `Schema` from a compiled program

## Inference Package

The `inference` package contains variational inference, MCMC, and
posterior-predictive sampling.

- **[Trace](inference/trace.md)**: Program trace data structures
- **[Conditioning](inference/conditioning.md)**: Conditioning and observations
- **[Guides](inference/guide.md)**: Variational guide distributions
- **[ELBO](inference/elbo.md)**: Evidence lower bound computation
- **[SVI](inference/svi.md)**: Stochastic variational inference
- **[Predictive](inference/predictive.md)**: Predictive inference and sampling
- **[MCMC](inference/mcmc.md)**: HMC and NUTS kernels plus the multi-chain runner

## Data, Diagnostics, and Formulas

These packages connect model definitions to tabular data and Bayesian analysis.

- **[Data](data/index.md)**: data-frame schemas and column encoders
- **[Diagnostics](diagnostics/index.md)**: ArviZ-compatible trace conversion, model comparison, and posterior-predictive checks
- **[Formulas](formulas/index.md)**: R-style formula parsing, QVR generation, and the `fit` entry point

## Analysis Package

The `analysis` package derives metadata from checked programs without changing
them.

- **[Chain shapes](analysis/chain_shape.md)**: source-aware step shapes and intermediate dimensions
- **[Initialization](analysis/init_spec.md)**: algebra-specific initialization recipes
- **[Saturation diagnostics](analysis/saturation.md)**: warnings tied to source locations
- **[Program inspection](analysis/plate_graph.md)**: plate graphs, renderers, and lexical scope queries

## Structural and Effects Packages

The `structural` package defines signatures, encoders, decoders, and attached
losses. The `effects` package supplies checked handler stacks for conditioning,
intervention, reweighting, tracing, replay, and reparameterization.

## Transpilation and QIEC

The `transpile` package lowers checked programs through a target-independent IR
and renders eleven probabilistic-programming languages. The `qiec` package is
the typed indexed-effect core used by program elaboration, execution, tooling,
and host-backed structural computations.

## Root Module

- **[Program](program.md)**: Top-level probabilistic program definitions
