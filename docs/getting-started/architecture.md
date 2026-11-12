# Architecture

The library has twenty subpackages and one public top-level module, `program.py`.

## Follow one model through the library

A regression begins as QVR source or as Python objects. The compiler in `dsl/`
resolves names, checks types, and elaborates every executable declaration to the
[Quivers Indexed Effect Core (QIEC)](../developer/qiec.md), a typed core
language in which sampling and scoring are explicit effects (`qiec/`). The
compiled runtime, the PyTorch implementation built on `continuous/` and
`program.py`, runs the program with its
[data inputs](../reference/qvr/probabilistic-programs.md#host-data), and `effects/` provides the
handlers that trace, condition, and replay its sample sites. `inference/` fits
it by evaluating its log joint density while it adjusts a variational guide or
moves a Markov chain, and `diagnostics/` summarizes the result.

To fit a model, start in `dsl/` or `formulas/` and follow the result into
`inference/` and `diagnostics/`. To add a distribution family, start in
`continuous/families.py` and the family registry in `qiec/families.py`. To
change how finite morphisms compose, start in `core/`. To trace an
[effect request](../tutorials/qvr/10-effects-and-handlers.md#1-declare-the-request)
or a [target that refuses a program](../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing),
read `qiec/`, `effects/`, and `transpile/`.

## Package structure

```text
quivers/
├── core/           # objects, algebras, finite morphisms, composition
├── categorical/    # functors, natural transformations, adjunctions,
│                   # monoidal and traced structure, change of base
├── enriched/       # ends and coends, Kan extensions, weighted limits,
│                   # profunctors, Yoneda, Day convolution, optics
├── monadic/        # functor, applicative, and monad type classes,
│                   # monad instances, comonads,
│                   # monad algebras, distributive laws, free monads
├── arrows/         # arrow type classes and their instances
├── stochastic/     # FinStoch, discretized families, conditioning,
│                   # weighted deduction and chart parsing
├── continuous/     # continuous spaces, distribution families,
│                   # probabilistic programs, plates, scans, flows
├── structural/     # signatures, tree and graph encoders and decoders
├── qiec/           # the QIEC data model, checker, serialization,
│                   # reference machine, distribution family registry
├── dsl/            # QVR parser, AST, compiler, QIEC lowering,
│                   # Pygments lexer
├── effects/        # effect handlers and reparameterizations
├── inference/      # traces, guides, objectives, SVI, MCMC, SMC,
│                   # annealed importance sampling, predictive draws
├── analysis/       # static analysis of programs
├── data/           # data frame encoding and dataset schemas
├── diagnostics/    # ArviZ conversion, convergence, calibration,
│                   # predictive checks, model comparison
├── formulas/       # R-style formulas compiled to QVR
├── transpile/      # intermediate representation and eleven renderers
├── cli/            # the qvr command, REPL, grammar migrations
├── kernel/         # Jupyter kernel
├── lsp/            # language server
└── program.py      # Program: a morphism as an nn.Module
```

The QVR tree-sitter grammar is in `grammars/qvr/` at the repository root. The
package includes its generated parser and installs it in place of the QVR
grammar that `panproto-grammars-all` carries, so parsing and highlighting use the
same grammar whatever that dependency provides.

## Module descriptions

### `core/`

Objects, algebras, and morphisms between finite sets ([API](../api/core/objects.md)).

- **`objects.py`:** objects as `dx.TaggedUnion` variants: `SetObject` (root), `FinSet`, `ProductSet`, `CoproductSet`, `FreeMonoid`, `EnumSet`, `FreeResiduated`, and the one-element set `Unit`.
- **`algebras.py`:** the classes that fix how weights combine. `CompositionRule` is the root. `Semigroupoid` is a composition rule declared associative, whose morphisms form a [semicategory](https://ncatlab.org/nlab/show/semicategory). `Magmoid` is a composition rule with no associativity or identity guarantee. `Algebra` adds a unit, a zero, a meet, and a negation, so composition has identities. `Semiring` is both an `Algebra` and a `Semigroupoid`, the class of the instances whose operations satisfy the semiring laws, and `IdempotentSemiring` adds an idempotent $\bigoplus$. The concrete classes include `ProductFuzzyAlgebra`, `BooleanAlgebra`, `StandardMVAlgebra`, `GodelAlgebra`, `TropicalAlgebra`, `MaxPlusAlgebra`, `LogProbAlgebra`, `RealAlgebra`, `BoundedSumProductAlgebra`, `CountingAlgebra`, and `MarkovAlgebra`, each with a shared instance (`PRODUCT_FUZZY`, `BOOLEAN`, `MARKOV`, and so on).
- **`algebra_morphisms.py`:** `CarrierMap`, the entrywise maps from one `Algebra`'s weights to another's, and its subclass `AlgebraHomomorphism` for the maps that preserve both operations and both units, with instances such as `Embedding`, `Threshold`, and `ProbabilityToReal`.
- **`morphism_transformations.py`:** `MorphismTransformation` and its row normalizations `Softmax`, `L1Normalize`, and `L2Normalize`, and Bayesian inversion `BayesInvert`. These and the entrywise maps are what `Morphism.change_base` applies.
- **`morphisms.py`:** `Morphism` and its subclasses `ObservedMorphism` (fixed tensor), `LatentMorphism` (learnable tensor), `ComposedMorphism`, `ProductMorphism`, `MarginalizedMorphism`, `TransformedMorphism`, `FunctorMorphism`, `RepeatMorphism`, and `CurriedMorphism`, with the factories `morphism()`, `observed()`, `identity()`, `cup()`, and `cap()`.
- **`tensor_ops.py`, `trans.py`, `wiring.py`:** noisy-OR contraction and reduction, sequences of transformations (`TransSeq`), and einsum contractions over several inputs.

### `categorical/`

[Functors](https://ncatlab.org/nlab/show/functor), natural transformations, and adjunctions between categories of finite morphisms (`Functor`, `NaturalTransformation`, `Adjunction`); monoidal structures (`CartesianProductMonoidal`, `CoproductMonoidal`); [traced monoidal](https://ncatlab.org/nlab/show/traced+monoidal+category) structure (`TracedMonoidal`, `CompactClosedTrace`, `trace()`); and change of base between Boolean and product fuzzy weights (`BaseChange`).

### `enriched/`

Constructions from [enriched category theory](https://ncatlab.org/nlab/show/enriched+category) [@kelly1982basic] computed on finite tensors over an `IdempotentSemiring`: ends and coends, left and right [Kan extensions](https://ncatlab.org/nlab/show/Kan+extension), weighted limits and colimits, profunctors, the Yoneda embedding, Day convolution, and optics (lenses, prisms, adapters, grates).

### `monadic/`

The [type classes](https://en.wikipedia.org/wiki/Type_class) `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, and `MonadTrans` [@mcbride2008applicative; @wadler1992essence], with instances (`Identity`, `Maybe`, `Alternative_`, `State`, `Reader`, `Writer`, `Continuation`), the monad `FuzzyPowersetMonad` with `KleisliCategory`, comonads, algebras of a monad and the Eilenberg-Moore category, distributive laws, free monads with handlers (`algebraic.py`), bridges to the arrow classes (`bridges.py`), and law checks (`laws.py`). `theories.py` describes each type class as a panproto theory.

### `arrows/`

The [arrow](https://en.wikipedia.org/wiki/Arrow_(computer_science)) type classes `Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, and `ArrowPlus` [@hughes2000arrows], with three instances: `VRel` (weighted relations), `Function` (relations whose rows have exactly one nonzero entry), and `Stochastic` (Markov kernels).

### `stochastic/`

Markov kernels between finite sets and weighted deduction.

- **`morphisms.py`, `giry.py`:** `StochasticMorphism` (learnable row-stochastic matrices, factory `stochastic()`), the distribution monad on finite sets, represented by its stochastic matrices (`GiryMonad`), and its Kleisli category `FinStoch`.
- **`families.py`:** discretized families such as `DiscretizedNormal` and `DiscretizedBeta`.
- **`transforms.py`, `queries.py`:** `condition()`, `mix()`, `factor()`, `normalize()`, and the queries `prob()`, `marginal_prob()`, and `expectation()`.
- **`agenda.py`, `inside.py`, `semiring.py`, `span.py`, `categories.py`, `schema.py`, `rules.py`, `ccg.py`, `lambek.py`, `parsers.py`, `productions.py`, `effect_lifts.py`:** [weighted deduction](../guides/deduction.md) [@shieber1995principles; @goodman1999semiring]: syntactic categories, rule schemata, CCG and Lambek rule sets, the agenda algorithm, the inside algorithm, and the `ChartParser`, `CCGParser`, and `LambekParser` drivers.
- **`deduction/`:** deduction primitives, sampling, and fitting of rule weights, including placing priors on them.
- **`stdlib/`:** packaged rule sets.

### `continuous/`

Continuous spaces, conditional distribution families, and probabilistic programs.

- **`spaces.py`:** continuous spaces as `dx.TaggedUnion` variants, among them `Euclidean`, `Simplex`, `PositiveReals`, `CholeskyFactor`, `Covariance`, `Correlation`, `Sphere`, `Stiefel`, and `ProductSpace`.
- **`families.py`, `family_spec.py`, `param_source.py`, `param_transforms.py`:** conditional distribution families such as `ConditionalNormal`, `ConditionalDirichlet`, and `ConditionalMultivariateNormal`, most generated from declarative specifications, with pluggable sources and constraint transforms for their parameters.
- **`ordered.py`, `log_space.py`, `_lkj.py`, `_matrix.py`, `_zip_hurdle.py`:** ordinal families, the logit of a Beta and the log of a Gamma variable, LKJ, matrix-valued, and zero-inflated and hurdle families.
- **`morphisms.py`, `deterministic.py`, `boundaries.py`:** continuous morphisms and their composition (`SampledComposition`), deterministic maps, and the boundaries `Discretize` and `Embed` between finite and continuous spaces.
- **`programs.py`, `program_steps.py`, `plate.py`, `scan.py`, `inline.py`:** `MonadicProgram`, the steps it sequences (`Draw`, `Observe`, `Score`, and the rest), plates, scans, and inline composition for the compiler.
- **`measure.py`, `bijectors.py`, `flows.py`:** measures of total mass at most one with their transformations, bijectors, and normalizing flows.

### `structural/`

Tree and graph autoencoders over terms of a binding signature: `Signature`, `Sort`, `Constructor`, `Binder`, and `Term` (`signature.py`); `Encoder`, which folds a term bottom-up into a vector; `Decoder`, which scores or samples terms from a vector; reconstruction losses (`LossRegistry`); and one shape backend each for sequences, trees, and graphs (`shapes/`).

### `qiec/`

The data model and checker of QIEC: kinds and finite enumerated index types, telescopes, indexed families and constructors, effect interfaces and rows, handlers, values, and computations (`kinds.py`, `declarations.py`, `effects.py`, `types.py`, `terms.py`), checked by `checking.py` and `coverage.py`. `serialization.py` defines a deterministic JSON format for checked modules. `evaluator.py` and `execution.py` implement the [reference machine](../developer/qiec.md#reference-evaluator), the library's interpreter for QIEC, and `program_runtime.py` runs programs on it. `families.py` is the distribution family registry that the compiler, the reference machine, and the transpilers share. `builtins.py` declares the built-in effect interfaces (`Random`, `Score`, `Compute`, `Param`).

Values that can be serialized cross module and process boundaries as QIEC data. Python callables, samplers, state, and handler implementations are runtime attachments: values bound to a declared identifier in the running Python process and never serialized. The transpile layer reads the serialized data. Eight targets receive the whole computation through a shared runtime, and Stan, BUGS, and JAGS accept a restricted subset of programs. A program that uses a feature a target lacks receives a diagnostic at that feature's source location. [Quivers Indexed Effect Core](../developer/qiec.md) gives the details.

### `dsl/`

The QVR front end.

- **`parser/`:** builds `dx.Model` AST nodes from the parse tree that panproto produces with the packaged QVR grammar.
- **`ast_nodes/`:** the AST, one `dx.Model` per node, with recursive sums as `dx.TaggedUnion` roots.
- **`compiler/`:** `Compiler`, which walks the AST and produces a `quivers.Program`. Its mixins handle declarations, expressions, programs, deductions, and structural declarations, and `resolution.py` maps each object expression to a `SetObject` or a `ContinuousSpace`.
- **`qiec_lowering.py`, `program_elaboration.py`, `structural_elaboration.py`, `deduction_elaboration.py`, `qiec_tooling.py`:** elaborate a module to QIEC, check it, and report diagnostics to the CLI, the REPL, and the language server.
- **Parametric programs:** a program with parameters, `program name (G : FinSet, scale : Real) : dom -> cod`, is stored on the `Compiler` as a template and instantiated at each call site with fresh names, so each call contributes its own factors to the joint density. `program_templates.py` instantiates a program at a draw from it.
- **`program_theory.py`:** extracts a panproto schema from a compiled module (`extract_program_schema`), so that panproto's diff and migration tools apply to `.qvr` programs.
- **`constraints.py`, `emit.py`, `pygments_lexer.py`, `_grammar_build.py`, `_grammar_introspection.py`:** checks run after parsing, printing an AST back to source, the Pygments lexer, loading the packaged parser, and syntax metadata shared by the editors.

`quivers.dsl` exports the entry points `load()`, `loads()`, `parse()`, and `parse_file()`, the `Compiler` and `Module` classes, and the QIEC lowering (`lower_qvr_to_qiec()`). The [probabilistic programs reference](../reference/qvr/probabilistic-programs.md) specifies the program syntax.

### `effects/`

Effect handlers over the reference machine's `Random`, `Score`, `Compute`, and `Param` interfaces: tracing (`trace_handler.py`), conditioning on observed values (`clamp.py`), intervention (`do.py`), masking, scaling, blocking, replay, subsampling, placing priors on learned parameters (`lift.py`), and integrating a conjugate parent out of the joint density (`collapse.py`). `interpreter.py` runs a program under the active handlers, `joint_plan.py` prepares a program's log joint once per data layout, and `reparam/` holds the loc-scale, transform, NeuTra, and conjugate reparameterizations.

### `inference/`

Inference for probabilistic programs.

- **`trace.py`, `conditioning.py`, `registry.py`:** execution traces, conditioning on observed values, and `LatentRegistry`, the list of a program's latent sites that every guide and MCMC kernel reads.
- **`guides/`:** variational families: `AutoNormalGuide`, `AutoMultivariateNormalGuide`, `AutoLowRankMultivariateNormalGuide`, `AutoDeltaGuide`, `AutoLaplaceApproximation`, `AutoNormalizingFlow`, `AutoIAFGuide`, `AutoNeuralSplineGuide`, `AutoMixtureGuide`, `AutoGuideList`, `AutoStructured`, and `AmortizedGuide`.
- **`estimators.py`, `objectives.py`, `svi.py`:** gradient estimators, the objectives `ELBO`, `IWAEBound`, `RenyiBound`, `VRIWAEBound`, `ChiVI`, `RWS`, and `DReGsBound`, and the `SVI` driver.
- **`mcmc/`:** `HMCKernel`, `NUTSKernel`, `MetropolisKernel`, and `GibbsKernel`, kernels that combine them, step size and mass matrix adaptation, checkpoints, and the `MCMC` driver.
- **`smc.py`, `dais.py`, `warmup.py`:** sequential Monte Carlo, differentiable annealed importance sampling (`AutoDAIS`), and warmup followed by Hamiltonian Monte Carlo (HMC).
- **`predictive.py`:** `Predictive`, draws from a fitted guide or an MCMC result.
- **`lifts.py`:** `lift_to_bayesian_program`, which places a prior on every parameter of a deterministic model, and `monte_carlo_log_joint`, which estimates a likelihood by drawing named latent variables from their priors.
- **`transforms.py`:** the invertible transforms of the normalizing flow guides.

### `analysis/`

Static analysis of programs: scope traversal, shape inference for chain rules, plate diagrams (`plate_graph.py`, `plate_render.py`), initializer specifications, and saturation checks.

### `data/`

`DatasetSchema` pairs a program's declared columns with their roles, types, and plates. `encoding.py` converts narwhals, pandas, polars, or pyarrow frames into the tensors the runtime reads.

### `diagnostics/`

`arviz_io.py` converts an `MCMCResult` to an ArviZ data tree. `convergence.py` computes R-hat and effective sample size, `calibration.py` runs simulation-based calibration of an inference procedure, `predictive_checks.py` runs posterior predictive checks, and `comparison.py` compares models by approximate leave-one-out cross-validation.

### `formulas/`

R-style formulas. `formula.py` parses a formula against a data frame, `compile.py` turns it into a QVR module, `family.py` declares response families and links, and `fit()` fits the result by stochastic variational inference (SVI) or MCMC.

### `transpile/`

`lower.py` and `plan.py` produce a representation that does not depend on the target, and `renderers/` emits Stan, NumPyro, Pyro, PyMC, Edward2, Turing, Gen, Church, WebPPL, BUGS, or JAGS source from it. The `runtime_*` files are the helper code that emitted programs include.

### `cli/`

The `qvr` command and its subcommands `check`, `run`, `repl`, `lsp`, `kernel`, `migrate`, and `transpile`. `migrations/` holds one module for each grammar change that `qvr migrate` applies.

### `kernel/`

The Jupyter kernel: `install.py` registers it and `quivers_kernel.py` runs it.

### `lsp/`

The language server, built on pygls: `server.py` handles requests and `document.py` tracks open documents.

### Top-level module {#top-level-modules}

- **`program.py`:** `Program`, which wraps a finite or continuous morphism as an `nn.Module`.

## Dependencies between packages

The packages do not form a strict layering: several import one another at
module level. Grouping each set of mutually dependent packages together gives
this order, in which each group imports only from the groups before it:

1. `core/` and `categorical/`, which import each other. `structural/` imports no other package.
2. `enriched/`, on `core/` and `categorical/`.
3. `monadic/` and `arrows/`, which import each other, on `core/` and `categorical/`.
4. `stochastic/`, `continuous/`, `qiec/`, `dsl/`, `effects/`, `inference/`, `analysis/`, `diagnostics/`, `transpile/`, and `program.py`, which import one another and every group above.
5. `data/` and `formulas/`, on group 4.
6. `cli/`, on group 4, and then `lsp/` and `kernel/`, on `cli/`.

Imports inside functions add further edges, among them `categorical/` to
`enriched/`, `core/` to `stochastic/`, and `cli/` to `lsp/` and `kernel/`.

## Design principles

- **Tensors as morphisms:** a morphism between finite sets is a PyTorch tensor, so automatic differentiation and GPU execution come from PyTorch.
- **Values are didactic models:** record values (AST nodes, `SetObject` and `ContinuousSpace` variants, syntactic categories, rule systems, the arrow instances) are frozen [didactic](https://pypi.org/project/didactic/) `dx.Model` values, with structural equality, JSON round trips, and checks that relate several fields. Recursive sums are `dx.TaggedUnion` roots with a `kind` discriminator.
- **Mutable tensor holders:** `Presheaf`, `Weight`, `SampleSite`, and `Trace` hold tensors that inference updates in place, and are Python dataclasses.
- **Lazy composition:** composing morphisms builds a graph, and the tensor is computed when it is read.
- **Checked types:** constructors and the compiler check domains and codomains before any tensor is evaluated. Constraints that depend on runtime data are checked when the data arrive.
- **PyTorch modules:** every morphism has a `.module()` method that returns an `nn.Module`, so it trains in an ordinary PyTorch loop.
- **One implementation for every `Algebra`:** the morphism classes do not depend on the `Algebra`, but the laws composition satisfies do ([concepts](concepts.md#3-an-algebra-chooses-how-weights-combine)).
- **Programs as panproto schemas:** parsing uses [panproto](https://github.com/panproto/panproto)'s tree-sitter support, and each compiled program can be extracted as a panproto schema, so panproto's diff and migration tools apply to `.qvr` programs.
