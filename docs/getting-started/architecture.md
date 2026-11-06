# Architecture

The quivers library is organized into twenty subpackages plus two
top-level modules. The categorical core (`core/`, `categorical/`,
`monadic/`, `arrows/`, `enriched/`, `stochastic/`, `continuous/`) is
surrounded by a DSL layer (`dsl/`), an inference layer (`inference/`),
a data layer (`data/`, `diagnostics/`, `formulas/`), structural and
analysis utilities (`structural/`, `analysis/`), effect handlers
(`effects/`), transpilers (`transpile/`), and the interactive tools
(`cli/`, `kernel/`, `lsp/`). This document describes the
package hierarchy, dependencies, and key abstractions in each module.

## Follow one model through the library

A regression begins as source text or Python objects. The compiler resolves
names and checks the declared structure, then represents sampling and scoring
as explicit computations in QIEC. A runtime supplies data and executes those
computations. Inference repeatedly evaluates their [joint score](../tutorials/python/05-variational-inference.md#tracing-the-model) while adjusting
a guide or moving a sampling chain. The model specifies the target; the
inference algorithm specifies how to approximate it.

For ordinary model fitting, start in `dsl/` or `formulas/` and follow results
into `inference/` and `diagnostics/`. To add a distribution, follow
`continuous/`. To change typed tensor composition, follow `core/`. To understand
an [effect request](../tutorials/qvr/10-effects-and-handlers.md#1-declare-the-request) or a [deployment refusal](../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing), follow `qiec/`, `effects/`, and
`transpile/`. The [conceptual introduction](concepts.md) explains the objects,
arrows, distributions, and handlers these packages manipulate.

Two representations recur in the inventory. A finite relation is a tensor
whose entries connect named input and output elements. A conditional kernel
instead maps an input to a distribution over outputs. Both compose, but only
the first is generally a finite table you can inspect directly. A handler
supplies an interpretation for a computation's requests; an attachment is a
runtime value, such as a PyTorch module, associated with a checked identifier.

## Package Structure

```
quivers/
├── core/              # foundational types and operations
├── categorical/       # functors, transformations, adjunctions, monoidal
├── monadic/           # typeclass tower (Functor → Applicative →
│                      # Monad → MonadPlus, plus Traversable /
│                      # MonadTrans), stdlib effect instances,
│                      # transformers, algebraic effects,
│                      # Kleisli/ArrowMonad/CoKleisli bridges,
│                      # panproto theory stubs
├── arrows/            # Hughes-style arrow tower (Category_ → Arrow →
│                      # ArrowApply / ArrowChoice / ArrowLoop /
│                      # ArrowZero / ArrowPlus) plus VRel / Function /
│                      # Stochastic instances and panproto theories
├── enriched/          # ends/coends, Kan, weighted limits, profunctors,
│                      # Yoneda, Day convolution, optics
├── stochastic/        # FinStoch (Markov kernels), families, transforms
│                      # (condition / mix / factor / normalize), Giry
│                      # monad, deductive chart parsers (CCG, Lambek),
│                      # semirings, schema, span, effect_lifts
├── continuous/        # parameterized families, spaces, boundaries,
│                      # flows, monadic programs, plates, scans,
│                      # deterministic transforms, inline composition
├── dsl/               # panproto-driven tree-sitter parser package,
│                      # didactic AST node package, compiler package,
│                      # program/deduction theory schemas, Pygments
│                      # lexer, grammar introspection helpers
├── inference/         # trace, conditioning, latent registry,
│                      # variational guides (mean-field, full-rank,
│                      # low-rank, IAF, neural-spline, mixture,
│                      # Laplace, delta), gradient estimators,
│                      # objectives (ELBO / IWAE / Rényi / VR-IWAE /
│                      # ChiVI / RWS / DReGs),
│                      # SVI, predictive, MCMC (HMC, NUTS), DAIS,
│                      # warmup, Bayesian lifts
├── analysis/          # static analysis of programs: scope, chain
│                      # shape, plate graph rendering, initialiser
│                      # specifications, saturation
├── data/              # data-frame encoding helpers and observation
│                      # schemas
├── diagnostics/       # ArviZ-backed trace I/O and predictive checks
├── formulas/          # R-style formula DSL → quivers program
│                      # compilation (formulae integration)
├── structural/        # encoder/decoder shape inference, structural
│                      # signatures, losses
├── effects/           # algebraic effect handlers and reparameterizers
├── qiec/              # indexed-family/effect core, checker, codec,
│                      # reference evaluator, built-in handlers
├── transpile/         # target-independent IR and eleven PPL renderers
├── cli/               # the `qvr` command, REPL, language-server
│                      # entry-point, migration driver
├── kernel/            # Jupyter kernel install + runtime
├── lsp/               # pygls-based language server (document model,
│                      # server)
├── program.py         # nn.Module wrapper for morphisms (discrete
│                      # or continuous)
└── giry.py            # backwards-compat re-exports of GiryMonad and
                       # FinStoch from stochastic.giry
```

The QVR tree-sitter grammar lives at the repo root under `grammars/qvr/`.
Quivers packages the generated current source and overrides Panproto's `qvr`
protocol at runtime, so ordinary parsing and Python-side highlighting load the
same grammar independently of the QVR revision in `panproto-grammars-all`.
First-party editor assets are synchronized with the in-tree source.

## Module Descriptions

### `core/`

Foundational categorical types and operations. Defines the objects and morphisms that all other modules build on.

- **`objects.py`:** Categorical objects as `dx.TaggedUnion` variants: `SetObject` (root), `FinSet`, `ProductSet`, `CoproductSet`, `FreeMonoid`, `EnumSet`, `FreeResiduated`, plus the module-level `Unit: FinSet = FinSet(name="1", cardinality=1)` singleton.
- **`algebras.py`:** Composition rules organized as a three-level hierarchy: `CompositionRule` (abstract) → `Semigroupoid` (composition only) → `Algebra` (composition + tensor unit + bilinear pair). `BilinearForm` sits beside `Semigroupoid` for the tensor-product side. Concrete variants: `ProductFuzzyAlgebra`, `BooleanAlgebra`, `LukasiewiczAlgebra`, `GodelAlgebra`, `TropicalAlgebra`, `MaxPlusAlgebra`, `LogProbAlgebra`, `RealAlgebra`, `ProbabilityAlgebra`, `CountingAlgebra`, `MarkovAlgebra`, plus `DualAlgebra`, `CustomAlgebra`, `CustomSemigroupoid`, `CustomBilinearForm`. Module-level singletons: `PRODUCT_FUZZY`, `BOOLEAN`, `LUKASIEWICZ`, `GODEL`, `TROPICAL`, `MAX_PLUS`, `LOG_PROB`, `REAL`, `PROBABILITY`, `COUNTING`, `MARKOV`, `REICHENBACH`, `BOOLEAN_DUAL`, `DUAL_LUKASIEWICZ`, `DUAL_GODEL`.
- **`algebra_morphisms.py`:** `AlgebraHomomorphism` (abstract) and its concrete variants (`IdentityHom`, `Embedding`, `Expectation`, `LogProb`, `MaxPlus`, `Threshold`, `MaterialImplication`, `ProbabilityClamp`, `CountingFromReal`, `ProbabilityToReal`, `CountingToReal`), plus the `threshold()`, `embedding()`, and `lookup_homomorphism()` helpers.
- **`morphism_transformations.py`:** `MorphismTransformation` (abstract step type) plus the `Softmax`, `L1Normalize`, `L2Normalize`, and `BayesInvert` concrete steps and their `softmax()`/`l1_normalize()`/`l2_normalize()` factory wrappers. These steps drive `Morphism.change_base` and the `TransformedMorphism` carrier.
- **`morphisms.py`:** $\mathcal{V}$-enriched relations as tensors: `Morphism` (abstract), `ObservedMorphism` (fixed tensor data), `LatentMorphism` (learnable), `TransformedMorphism`, `ComposedMorphism`, `ProductMorphism`, `MarginalizedMorphism`, `FunctorMorphism`, `RepeatMorphism`, `CurriedMorphism`, plus the factory functions `morphism()`, `observed()`, `identity()`, `cup()`, `cap()`, and helpers `as_torch_module()`, `extract_morphism()`.
- **`tensor_ops.py`:** Tensor operations: `noisy_or_contract`, `noisy_or_reduce`, `noisy_and_reduce`, `componentwise_lift`.
- **`trans.py`:** `TransSeq`, a composable sequence of `MorphismTransformation` steps.
- **`wiring.py`:** Wiring helpers for tensor-product composition.
- **`_factories.py`, `_util.py`:** Internal factory helpers and numerical utilities (`EPS`, `clamp_probs`, `safe_log1p_neg`).

### `categorical/`

Categorical structures built from objects and morphisms.

- **`functors.py`:** Functors and functor composition: `Functor`, `IdentityFunctor`, `ComposedFunctor`, `FreeMonoidFunctor`.
- **`natural_transformations.py`:** Natural transformations: `NaturalTransformation`, `ComponentwiseNT`.
- **`adjunctions.py`:** Adjunctions and free-forgetful pairs: `Adjunction`, `ForgetfulFunctor`, `FreeForgetfulAdjunction`.
- **`monoidal.py`:** Monoidal structures: `MonoidalStructure`, `CartesianMonoidal`, `CoproductMonoidal`, `EmptySet`, `EMPTY`.
- **`base_change.py`:** Base change between algebras: `BaseChange`, `BoolToFuzzy`, `FuzzyToBool`.
- **`traced.py`:** Traced monoidal categories: `TracedMonoidal`, `CartesianTrace`, `IterativeTrace`, functions `trace()`, `partial_trace()`.

### `monadic/`

A Haskell-style typeclass hierarchy with concrete monad instances
plus comonads, algebras, distributive laws, transformers, and
algebraic effects. Each typeclass is mirrored by a panproto theory.

Modules:

- **`typeclasses.py`:** `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, `MonadTrans` ABCs.
- **`monads.py`:** `FuzzyPowersetMonad`, `FreeMonoidMonad` (concrete Monad instances) plus `KleisliCategory`.
- **`comonads.py`:** `Comonad`, `CoKleisliCategory`, `DiagonalComonad`, `CofreeComonad`.
- **`algebras.py`:** Eilenberg–Moore: `Algebra`, `FreeAlgebra`, `ObservedAlgebra`, `Coalgebra`, `CofreeCoalgebra`, `ObservedCoalgebra`, `EilenbergMooreCategory`.
- **`distributive_laws.py`:** `DistributiveLaw`, `FreeMonoidPowersetLaw`.
- **`instances.py`:** stdlib effect instances: `Identity`, `Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`, `List`. Each registers against the appropriate ABC(s) via `ABC.register(...)`.
- **`transformers.py`:** monad transformers: `StateT`, `ReaderT`, `MaybeT`, `ContT`, `WriterT`.
- **`algebraic.py`:** algebraic effects + handlers: `Operation`, `EffectSignature`, `Handler`, `FreeMonad`.
- **`bridges.py`:** `Kleisli`, `ArrowMonad`, `CoKleisli` plus `kleisli` / `arrow_monad` / `cokleisli` factory helpers connecting the monad, arrow, and comonad towers.
- **`theories.py`:** panproto-theory stubs for each typeclass.
- **`laws.py`:** runtime-checkable typeclass-law scaffolding.

### `arrows/`

Hughes-style arrow tower (parallel to the monad-side hierarchy).

- **`typeclasses.py`:** `Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, `ArrowPlus` ABCs.
- **`instances.py`:** concrete arrow carriers: `VRel` (the $\mathcal{V}$-enriched relation arrow), `Function` (the deterministic restriction of `VRel` to 0/1-valued function-graph tensors), `Stochastic` (Markov-kernel arrow).
- **`theories.py`:** panproto-theory stubs for each arrow typeclass.

### `enriched/`

Enriched category theory: the calculus of $\mathcal{V}$-valued hom-functors.

- **`ends_coends.py`:** Limits and colimits in enriched categories: `coend()`, `end()`.
- **`kan_extensions.py`:** Kan extensions: `ObjectMap`, `Projection`, `Inclusion`, `left_kan()`, `right_kan()`.
- **`weighted_limits.py`:** Weighted (co)limits: `Weight`, `Diagram`, `weighted_limit()`, `weighted_colimit()`, `representable_weight()`, `terminal_weight()`.
- **`profunctors.py`:** Profunctors (distributor, bimodules): `Profunctor`.
- **`yoneda.py`:** Yoneda lemma and embedding: `Presheaf`, `representable_profunctor()`, `corepresentable_profunctor()`, `yoneda_embedding()`, `yoneda_lemma()`, `yoneda_density()`, `verify_yoneda_fully_faithful()`.
- **`day_convolution.py`:** Day convolution for monoidal categories: `day_convolution()`, `day_unit()`, `day_convolution_profunctors()`.
- **`optics.py`:** Optics and their composition: `Optic`, `Lens`, `Prism`, `Adapter`, `Grate`, `compose_optics()`.

### `stochastic/`

Stochastic morphisms and the category **FinStoch** of Markov kernels on finite sets.

- **`morphisms.py`:** Row-stochastic learnable Markov kernels: `StochasticMorphism` and its alias `CategoricalMorphism`, plus the `stochastic()` factory.
- **`families.py`:** Discretized parameterized distributions: `DiscretizedNormal`, `DiscretizedLogitNormal`, `DiscretizedBeta`, `DiscretizedTruncatedNormal`.
- **`transforms.py`:** Transforms on stochastic morphisms and the carrier classes they produce: `condition()`, `mix()`, `factor()`, `normalize()`, with `ConditionedMorphism`, `MixtureMorphism`, `FactoredMorphism`, `NormalizedMorphism`.
- **`queries.py`:** Query functions: `prob()`, `marginal_prob()`, `expectation()`.
- **`giry.py`:** The Giry monad on FinStoch: `GiryMonad`, `FinStoch`.
- **`categories.py`, `effect_lifts.py`, `inside.py`, `agenda.py`, `span.py`, `semiring.py`:** chart-parser infrastructure: deduction categories, agenda-based search, span representations, semiring weights, the inside algorithm, and effect lifts.
- **`schema.py`:** `RuleSchema`, a functor from `CategorySystem` to `RuleSystem`. Provides the binary/unary schema combinators (`BinaryRuleSchema`, `UnaryRuleSchema`, `UnionSchema`, `WeightedSchema`) and the concrete CCG/Lambek schema instances (`ForwardApplication`, `BackwardApplication`, `ForwardComposition`, `BackwardComposition`, the crossed and commutative variants, `TensorIntroduction`, `LeftUnitElimination`, …). Schemas compose via `|` (union) and `.weighted(w)`.
- **`deduction/`:** deduction-system primitives (atoms, axioms, rules).
- **`_rule_system.py`, `rules.py`, `ccg.py`, `lambek.py`, `parsers.py`:** the `RuleSystem` core, the `ccg_rules` / `lambek_rules` / `custom_rules` libraries, and the `ChartParser` / `CCGParser` / `LambekParser` drivers.
- **`stdlib/`:** packaged grammars and rule sets.

### `continuous/`

Continuous and hybrid discrete-continuous morphisms and monadic programs.

- **`spaces.py`:** Continuous sample spaces as `dx.TaggedUnion` variants: `ContinuousSpace`, `Euclidean`, `UnitInterval` (factory wrapping `Euclidean`), `Simplex`, `PositiveReals`, `CholeskyFactor`, `ProductSpace`, `Sphere`, `Ball`, `Covariance`, `Correlation`, `Orthogonal`, `Stiefel`, `LowerTriangular`, `Diagonal`.
- **`morphisms.py`:** Continuous morphism base classes and combinators.
- **`families.py`:** Conditional distribution families, defined either as explicit subclasses (e.g. `ConditionalNormal`, `ConditionalBeta`, `ConditionalDirichlet`, `ConditionalMultivariateNormal`, `ConditionalLowRankMVN`, `ConditionalWishart`, `ConditionalInverseWishart`, `ConditionalMatrixNormal`, `ConditionalRelaxedBernoulli`, `ConditionalRelaxedOneHotCategorical`, `ConditionalLKJCholesky`, `ConditionalGaussianProcess`, `ConditionalHorseshoe`, `ConditionalMixture`, `ConditionalIndependent`, `ConditionalTransformed`, `ConditionalBernoulli`, `ConditionalCategorical`, `ConditionalBinomial`, `ConditionalLogisticNormal`, `ConditionalOneHotCategorical`) or generated by the `_make_family` factory (e.g. `ConditionalCauchy`, `ConditionalLaplace`, `ConditionalGumbel`, `ConditionalLogNormal`, `ConditionalStudentT`, `ConditionalExponential`, `ConditionalGamma`, `ConditionalChi2`, `ConditionalHalfCauchy`, `ConditionalHalfNormal`, `ConditionalInverseGamma`, `ConditionalWeibull`, `ConditionalPareto`, `ConditionalKumaraswamy`, `ConditionalContinuousBernoulli`, `ConditionalFisherSnedecor`, `ConditionalPoisson`, `ConditionalGeometric`). Also exports the `LKJCorrelationFactor` and `Truncated` helpers.
- **`family_spec.py`:** Declarative specification objects backing `_make_family`.
- **`programs.py`:** Monadic programs: `MonadicProgram` (compositional probabilistic computations) plus the internal `_StepSpec` / `_LetSpec` / `_ScoreSpec` step records.
- **`plate.py`, `scan.py`:** `plate(...)` and `scan(...)` constructs for vectorized and sequential program fragments.
- **`inline.py`:** inline-composition helpers used by the DSL compiler.
- **`deterministic.py`:** deterministic continuous morphisms (pure functions on continuous spaces).
- **`boundaries.py`:** Boundaries between discrete and continuous: `Discretize`, `Embed`.
- **`flows.py`:** Normalizing flows: `AffineCouplingLayer`, `ConditionalFlow`.

### `dsl/`

Domain-specific language for quiver expressions in `.qvr` files. Parsing is
delegated to Panproto after Quivers installs its packaged current `qvr`
tree-sitter grammar; Quivers does not run a hand-written lexer. The AST
carries distinct QIEC nodes, and `qiec_lowering.py` constructs Didactic's
public GADT representation before completing the exact route from
`qvr-source/v0.20` to `qiec-core/v1alpha1`. The reference `.qvr`
programs that drive the test suite and the tree-sitter grammar fixtures live at
`docs/examples/source/`, not under the package itself.

- **`parser/`:** A package that walks the panproto-produced parse tree and builds a tree of `dx.Model` AST nodes. The top-level module re-exports `parse()`, `parse_file()`, `ParseError`; `qiec.py` owns indexed-family, effect, handler, and computation forms, while the remaining submodules partition the established declarations and expressions.
- **`ast_nodes/`:** A package whose every node is a `dx.Model`. Recursive sums (`ObjectExpr`, `Expr`, `LetExprNode`, `ProgramStep`, `Statement`) are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field. `qiec.py` defines the complete QIEC node set; the other submodules split the probabilistic and structural nodes by category.
- **`compiler/`:** A package that walks the AST and produces a `quivers.Program`. `core.py` defines `Compiler` (composed via mixins from `declarations.py`, `deductions.py`, `expressions.py`, `programs.py`, `structural.py`); `resolution.py` provides the `_ResolutionMixin` whose `_resolve_any_space` routes an `ObjectExpr` to either a `SetObject` (discrete) or a `ContinuousSpace`; `_options.py` parses option blocks and `_prelude.py` defines `CompileError` plus shared compiler-state primitives.
- **`program_theory.py`:** Defines `QVR_PROGRAM_PROTOCOL` (a panproto protocol whose vertex kinds enumerate every `SetObject` and `ContinuousSpace` variant plus the QVR declaration variants), `extract_program_schema(compiler)`, and `extract_deduction_schema(compiler)`. These make every `.qvr` program a schema in panproto's sense, diff, migrate, and lens-generation workflows apply.
- **`constraints.py`:** post-parse axiom checks: `check_constraints(module)` returns a list of `Violation` records (used by the LSP, the REPL, and `qvr check`).
- **`emit.py`:** AST → `.qvr` source emission (the print direction of the parser).
- **`pygments_lexer.py`:** The tree-sitter-backed Pygments lexer shared by documentation and notebook tooling, registered as the `qvr` lexer entry point.
- **`_grammar_build.py`, `_grammar_data/`, `_grammar_introspection.py`:** loads
  the generated checkout grammar during development or the packaged
  source/native/manifest trio from a platform wheel, builds a content-addressed
  parser only for editable grammar work, and derives shared syntax metadata.
  Immutable migration snapshots remain separate under `quivers.cli.migrations`.
- **`qiec_lowering.py`, `program_elaboration.py`, `qiec_tooling.py`:** project
  indexed declarations into Didactic's public `GADT` API, elaborate every
  `program` to a named computation over the module's canonical `random` and
  `score` instances, check and lower the complete module, negotiate the exact
  route, and expose one diagnostic and symbol model to the CLI, REPL, and LSP.

Top-level DSL API (re-exported from `quivers.dsl`): `parse()`, `parse_file()`,
`loads()`, `load()`, `Compiler`, `Module`, `QVR_PROGRAM_PROTOCOL`,
`extract_program_schema()`, `QVR_SOURCE_VERSION`, `QvrQiecLowerer`,
`lower_qvr_to_qiec()`, `has_qiec_surface()`, `non_qiec_projection()`, plus
`ParseError`, `CompileError`, and `QiecDiagnosticError`.

Parametric programs, `program name (G : FinSet, scale : Real, prior : Mor[A, B]) : dom -> cod`, are stored as templates on the compiler (`Compiler._program_templates`) and inlined per call site by substitution + α-renaming; the runtime `MonadicProgram` is built from the inlined step list, so distinct call sites contribute distinct factors to the parent's joint kernel.

Program bodies use a single Kleisli-bind sigil `<-`. The sampling shapes are `sample v <- F(args)` (scalar draw), `sample v : A <- F(args)` (A-indexed plate), `observe v <- F(args)` (scored observation), `observe r : N <- F(args)` (N-indexed batched score), and `marginalize c : A <- F(args) [over = G, reduction = R]` with an indented scope body (scoped marginalization). A program declaration carries an effect signature in its option block, `[effects = [Sample, Score, Marginal, Pure]]`; the compiler verifies that the body's actual effects are a subset of the declared set. A posterior block is an `[effects = [Pure]]` program with an `[over = M]` option consuming the latents of model `M` as data parameters.

### `inference/`

Inference primitives for monadic probabilistic programs: tracing, observation conditioning, automatic variational guides, gradient estimators, variational objectives, Markov-chain Monte Carlo, and posterior-predictive sampling.

- **`trace.py`:** Execution tracing: `SampleSite`, `Trace`, `trace()`.
- **`conditioning.py`:** Observation marking: `Conditioned`, `condition()`.
- **`registry.py`:** `LatentRegistry` and `LatentSite`, the per-site introspection helper every guide and MCMC kernel consumes.
- **`guides/`:** A package of variational guide families. Public classes: `Guide` (base), `AutoNormalGuide`, `AutoMultivariateNormalGuide`, `AutoLowRankMultivariateNormalGuide`, `AutoDeltaGuide`, `AutoLaplaceApproximation`, `AutoNormalizingFlow`, `AutoIAFGuide`, `AutoNeuralSplineGuide`, `AutoMixtureGuide`, `AutoGuideList`, and `AutoStructured`.
- **`estimators.py`:** Gradient estimators: `GradientEstimator`, `Reparameterized`, `StickingTheLanding`, `DoublyReparameterized`, `ScoreFunction`.
- **`objectives.py`:** Variational objectives: `Objective`, `ELBO`, `IWAEBound`, `RenyiBound`, `VRIWAEBound`, `ChiVI`, `RWS`, and `DReGsBound`.
- **`svi.py`:** Stochastic variational inference driver: `SVI`.
- **`predictive.py`:** Posterior-predictive sampling: `Predictive`.
- **`mcmc/`:** Markov-chain Monte Carlo: `MCMCKernel` (base), `HMCKernel`, `NUTSKernel`, plus the `MCMC` driver and `MCMCResult`.
- **`dais.py`:** Differentiable annealed importance sampling: `AutoDAIS`.
- **`warmup.py`:** `WarmupThenHMC`, a warmup-then-sampling composite kernel.
- **`lifts.py`:** Bayesian lifts that turn a deterministic program into a sampled one: `bayesian_lift_parameters`, `lift_to_bayesian_program`, `lift_from_log_prob`, `monte_carlo_log_joint`.
- **`transforms.py`:** Bijective flow transforms used by the normalizing-flow guides: `TransformModule` (base), `AffineCouplingTransform`, `MaskedAutoregressiveTransform`, `InverseAutoregressiveTransform`, `NeuralSplineCouplingTransform`, `LULinearTransform`, `BatchNormTransform`, plus the `MADE` autoregressive net and the `alternating_mask` / `half_mask` mask helpers.

### `analysis/`

Static analysis of compiled programs: `scope.py` (scope-path traversal), `chain_shape.py` (chain-rule shape inference), `plate_graph.py` and `plate_render.py` (plate-diagram extraction and rendering), `init_spec.py` (initializer specifications), `saturation.py` (saturation checks for the type/effect system).

### `data/`

Data-frame ingestion. `schema.py` defines `DatasetSchema` (a frozen `dx.Model` pairing a program's declared columns with their roles, dtypes, and plate indices) plus the `compose()` helper that splices a `DatasetSchema` into a `.qvr` body. `encoding.py` defines `ColumnRole`, `MissingPolicy`, and `encode_column()`, which convert narwhals / pandas / polars / pyarrow frames into the tensor dictionaries the runtime consumes.

### `diagnostics/`

ArviZ-backed posterior diagnostics. `arviz_io.py` converts an `MCMCResult` into an `xarray.DataTree` (the ArviZ 1.x container) via `az.from_dict`, repackaging sample tensors under the canonical ArviZ group names with user-supplied coords and dims; `comparison.py` and `predictive_checks.py` wrap the standard cross-validation and posterior-predictive-check workflows.

### `formulas/`

R-style formula DSL. `formula.py` defines the `Formula`, `FormulaData`, `FixedColumn`, and `RandomTerm` data classes plus `formula_from_data()`, the entry point from a formula string + data frame to a parsed formula. `compile.py` defines `FormulaToQVRModule` (a `dx.Lens[Formula, Module, FormulaData]`) that lowers a parsed formula into a `quivers.dsl.Module`. `family.py` declares the conjugate response families (`Family`, `Link`, `AuxParam`). `_fit.py` provides the `BayesianFit` orchestration record and the `fit()` convenience entry point (dispatches between SVI and MCMC).

### `structural/`

Structural compression: a uniform algebraic interface for encoding arbitrary structured terms to fixed-length vectors and decoding them back under a learned distribution.

- **`signature.py`:** Term-level data model. `Sort`, `Constructor`, `Binder`, `BinderArgSpec`, `BinderVarSpec`, `Term`, `VertexKind`, `EdgeKind`, `Signature`, `Context`, `EMPTY_CONTEXT`, plus the `make_term` / `bound_var` factories.
- **`encoder.py`:** `Encoder` (`nn.Module`) plus the `make_default_op_fn` / `make_default_var_init` defaults.
- **`decoder.py`:** `Decoder` (`nn.Module`), the inverse direction.
- **`losses.py`:** `LossEntry`, `LossRegistry` for plug-in reconstruction objectives.
- **`shapes/`:** shape backends (`seq.py`, `tree.py`, `graph.py`), one per supported shape category.

### `qiec/`

The Quivers Indexed Effect Core (QIEC) is the stable kernel every executable
declaration of a module elaborates to: indexed families, algebraic effects,
and probabilistic programs. Its records represent kinds,
heterogeneous telescopes, indexed constructors, lexical effect instances,
effect rows, handler signatures, values, computations, and branch evidence.
`serialization.py` defines the deterministic `qiec-json/v1` envelope;
`checking.py` and `row_solver.py` validate indexed terms and row constraints;
`evaluator.py` supplies the reference handler semantics; and `builtins.py`
provides the standard effect interfaces and process-local runtime attachments.

The package boundary is deliberate. Stable QIEC data may cross a process or
Panproto boundary, while host callables, state cells, samplers, and handler
implementations remain runtime attachments. The transpile layer projects that
stable data into a typed `IRQiecModule`: eight host-language targets consume
the complete computation tree through a common runtime ABI, while Stan, BUGS,
and JAGS accept an analyzer-proven closed scalar subset. Unsupported features
receive source-located capability diagnostics rather than being erased.
See [Quivers Indexed Effect Core](../developer/qiec.md) for the source and ABI
contracts.

### `effects/`

Algebraic handlers for tracing, clamping observations, interventions,
masking, scaling, blocking, replay, lifting, collapsing, and
reparameterization, each a lexical handler of the kernel's `Random`, `Score`,
`Compute`, or `Param` interface. `interpreter.py` runs a program on the
reference machine under the active handler stack through the kernel
computation `program_module.py` encodes it to; `checked_program.py` binds a
program whose steps call a computation to the module's entry point; `reparam/`
contains the loc-scale, transform, NeuTra, and conjugate strategies.

### `transpile/`

Cross-language emission for compiled QVR modules. `lower.py` produces a
target-independent IR, and `renderers/` emits Stan, NumPyro, Pyro,
PyMC, Edward2, Turing, Gen, Church, WebPPL, BUGS, or JAGS source.

### `cli/`

The `qvr` command (`__init__.py`'s `main`) and its subcommands:
`check.py` (validate a program), `lsp.py` (launch the LSP), `migrate.py` and `migrations/` (run grammar/AST migrations), `repl.py` plus the `repl_*` modules (REPL session, prompt, highlighting, Textual TUI, tab completion).

### `kernel/`

Jupyter kernel: `install.py` registers the kernelspec; `quivers_kernel.py` is the in-process kernel; `__main__.py` is the launcher.

### `lsp/`

pygls-based language server: `server.py` is the LSP entry point; `document.py` models open buffers and incremental edits.

### Root-level modules

- **`program.py`:** `Program`: wraps a morphism (discrete or continuous) as a differentiable `nn.Module`.
- **`giry.py`:** Backwards-compat re-export of `GiryMonad` and `FinStoch` from `quivers.stochastic.giry`.

## Dependency Graph

The package has a roughly layered dependency structure. Layers higher
in the list depend only on layers above them, with one documented
exception: `dsl.compiler.declarations` pulls a single helper
(`_algebra_init_spec`) out of `analysis.init_spec`, so the strict
analysis-then-dsl ordering bends at that one call site.

1. **`core/`:** foundational (no internal dependencies).
2. **`categorical/`, `monadic/`, `arrows/`:** build on `core/`.
3. **`enriched/`:** builds on `core/` and `categorical/`.
4. **`stochastic/`:** builds on `core/`, `categorical/`, `monadic/`.
5. **`continuous/`:** builds on `core/`, `categorical/`, `monadic/`, `stochastic/`.
6. **`analysis/`:** builds on `core/` and `dsl.ast_nodes` / `dsl.compiler._prelude` (uses AST types and the compiler's algebra registry for static analysis).
7. **`dsl/`:** synthesizes `core/`, `categorical/`, `monadic/`, `enriched/`, `stochastic/`, `continuous/`, plus `analysis.init_spec` for one initializer-checking helper.
8. **`inference/`:** builds on `core/`, `categorical/`, `stochastic/`, `continuous/`, and `program.py`.
9. **`data/`:** builds on `dsl/`. **`diagnostics/`:** builds on `inference/`.
10. **`formulas/`:** builds on `continuous/`, `dsl/`, and `inference/`.
11. **`structural/`:** standalone PyTorch-only shape inference; no in-package dependencies.
12. **`effects/`:** builds on `continuous/` and provides the handler-aware interpreter used by inference.
13. **`transpile/`:** builds on `dsl/` and lowers compiled modules to target-specific schemas and source.
14. **`cli/`:** consumes `analysis/`, `dsl/`, `kernel/`, `lsp/`, and `transpile/`. **`kernel/`:** uses `cli/` to start the in-process REPL session. **`lsp/`:** uses `cli/` and `dsl/`.
15. **`program.py`:** depends on `core/` and `continuous/`.
16. **`giry.py`:** thin backwards-compat re-export of `stochastic/`.

Users can enter at the layer their task requires: core algebra,
categorical structures, probabilistic computations, or the full DSL.

## Design Principles

- **Tensors as morphisms**: $\mathcal{V}$-relations are represented as PyTorch tensors, making them amenable to automatic differentiation and GPU acceleration.
- **Value types are didactic Models**: every record-shaped value (AST nodes, `SetObject` variants, `ContinuousSpace` variants, deduction categories, `RuleSystem`, `Kleisli` / `ArrowMonad` / `CoKleisli` bridges, `VRel` / `Function` / `Stochastic` arrows) is a frozen `dx.Model`. This buys uniform construction, structural equality, JSON round-trips via `model_dump_json` / `model_validate_json`, and cross-field axioms via `__axioms__`. Recursive sums are `dx.TaggedUnion` roots with a `kind: Literal[...]` discriminator.
- **Tensor-bearing classes stay mutable**: `Presheaf`, `Weight`, `SampleSite`, and `Trace` accumulate `torch.Tensor` fields and are mutated in place during inference; they remain `@dataclass`.
- **Lazy composition**: morphism composition creates a DAG; the final tensor is materialized only on evaluation.
- **Type safety**: constructors and compilation check declared domain/codomain compatibility before tensor evaluation; constraints that depend on runtime data are checked at that boundary.
- **Module transparency**: all morphisms expose a `.module()` method returning an `nn.Module`, enabling integration with PyTorch training loops.
- **Algebra flexibility**: the same morphism structure works with any algebra; swapping algebras changes semantics (fuzzy, Boolean, tropical, etc.) without code duplication.
- **DSL pipeline as a panproto theory morphism**: parsing is delegated to panproto's tree-sitter–driven AST registry; resolution is a compiler mixin that walks `ObjectExpr` trees into `SetObject` or `ContinuousSpace` values; each compiled program extracts to a panproto `Schema` against `QVR_PROGRAM_PROTOCOL`. The diff, migration, and lens-generation tools operate directly on `.qvr` programs.
