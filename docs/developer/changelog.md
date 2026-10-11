# Changelog

All notable changes to the quivers library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.25.1] - 2026-10-09

### Added

- **Explicit MCMC random streams.** `MCMC` accepts either a reproducible `seed` or a caller-owned CPU `torch.Generator`, derives an independent local generator for every chain, and routes initialization, momentum, NUTS tree direction, multinomial subtree selection, and HMC acceptance through that stream without mutating Torch's global RNG state.

### Fixed

- **MCMC adaptation is independent across chains and runs.** Every chain now starts with fresh dual-averaging, mass-matrix, and adaptation state instead of inheriting the preceding chain's tuned state; reusing an `MCMC` driver also starts a fresh adaptation lifecycle.
- **NUTS reports every divergent subtree.** U-turn termination and divergence are represented separately, divergence propagates from either child at every tree depth, and an ordinary U-turn is not counted as a divergence.
- **MCMC initialization and diagnostics preserve their contracts.** Generated and guide-provided initial positions must be finite before sampling begins, and `MCMCResult.divergence_counts` is an integer tensor containing one post-warmup count per chain.

## [0.25.0] - 2026-10-07

### Added

- **A written, enforced export policy.** The new *Public API and Exports* page of the developer guide states what Quivers exports and from where: every public module declares `__all__`, a public package re-exports the names of each of its modules as the same objects, public signatures name only public types, every class the DSL compiler places in a compiled program is public, and every public name renders in the API reference. `tests/test_public_api.py` checks the import-time rules over every module, and `tools/check_api_reference.py` checks the built site in a new CI docs job, which also builds the documentation in strict mode on every pull request.
- **A supported Python surface for building programs without the DSL.** `quivers.continuous` exports the typed step records a `MonadicProgram` is built from (`Draw`, `Observe`, `Let`, `Score`, and the `Indexed` argument form for gathers such as `mu[group]`), together with every builder the compiler emits: `FixedDistribution`, `MixedInlineDistribution`, the `make_fixed_*` factories, `PlateDraw`, `VectorisedObserve`, `FanOutMorphism`, `MarginalizedFactor`, `reading`, and the family registry. `MonadicProgram.steps` returns a program's records and `MonadicProgram.step_module` the module a draw applies, so a compiled program can be inspected in the same vocabulary it is built from. The programs guide works through one program built both ways and checks that the two score identically.
- **A literal-parameter inline `LKJCholesky`.** `sample L <- LKJCholesky(concentration=2.0)` compiles through the new `make_fixed_lkj_cholesky` factory, and a `PlateDraw` over either it or `LKJCorrelationFactor` draws one Cholesky factor per group.
- The module-level names newly exported under the policy, among them the bijectors, the measure algebra, the deterministic morphisms, the agenda and category types of `quivers.stochastic`, the structural encoder helpers, the transformation and wiring morphisms of `quivers.core`, the arrow instances, the flow transforms and MCMC adaptation helpers of `quivers.inference`, `loo_pit` in `quivers.diagnostics`, and the effect-handler trace, conjugate-solver, and checked-program types of `quivers.effects`.

### Changed

- **`MonadicProgram` takes step records rather than tuples.** A draw is `Draw(names=("x",), morphism=f, args=("a",))` rather than `(("x",), f, ("a",))`, an observe is `Observe(...)` rather than a four-tuple ending in `True`, a let is `Let(name="w", value=fn)`, and a score step is `Score(name="s", score=fn)`. Two draws or observes that bind the same name are an error rather than a silent replacement of the first morphism.
- **Names that collided under a package's namespace were renamed.** The bijectors carry a `Bijector` suffix (`ExpBijector`, `SoftplusBijector`, ...); the family registry functions are `register_family`, `get_family_spec`, and `family_names`; the deterministic builders are `cumsum_morphism`, `softmax_morphism`, and `cholesky_quad_form_morphism`, built on the public `DeterministicMorphism`; the homomorphism singletons of `quivers.core.algebra_morphisms` are `LOG_PROB_HOM` and `MAX_PLUS_HOM`; `quivers.monadic.instances.List` is `ListMonad`; the CLI check result types are `CheckDiagnostic` and `CheckSeverity`, with `check_file` the per-file entry point; `quivers.kernel.install.install` is `install_kernelspec`; the QIEC execution trace recorder is `ExecutionTraceRecorder`, and the static scope checking functions take is the public `StaticScope`; and the transpile renderer context is the public `RenderContext`; and the ordered-logistic morphism a `~ OrderedLogistic` declaration compiles to, which learns its cutpoints, is `ConditionalLearnedOrderedLogistic`, distinct from the `ConditionalOrderedLogistic` that takes them explicitly.
- **Renderer helpers are private.** Each renderer's `render_list` and `render_matrix` helpers are internal, `PyroRenderer.marginalize` takes no private context keyword, and `format_stan` and `turing.render_module` are no longer exported. Renderer methods annotate only public IR types, and the full `ir` and `qiec_ir` vocabularies are exported for renderer authors.
- **`check_monad_laws` checks the monad laws.** It verifies left unit, right unit, and associativity through `pure`, `join`, and `fmap`, and takes the instance and a carrier.
- **`LKJCorrelationFactor` is the normalized LKJ density.** It scores and samples through `torch.distributions.LKJCholesky`, honours `sample_shape`, and accepts flat or square factors.

### Fixed

- **Stacked parameters keep their dtype.** Multi-argument inline parameters, and the log joint that accumulates their densities, stay in the widest floating dtype of their inputs instead of being cast to float32, so a float64 program agrees with a float64 reference to rounding.
- **`OrderedLogistic` is exact in both tails.** A category's log probability is computed as a sum of log-sigmoids and a `log1mexp` term rather than as the log of a difference of sigmoids floored at the smallest normal float, so log likelihoods below about -708 are no longer clamped.
- **`MixedInlineDistribution` checks its declared widths.** The stacked columns must match the widths the parameter specification declares; at most one vector parameter may leave its width unresolved (`None`) and take the remaining columns. An inline `Categorical` or `Dirichlet` reads its parameter as a vector per row, which corrects categorical formula likelihoods that had scored only the first logit column, and an observe whose logits are a plate of per-category scalars, which had scored a one-category distribution.
- **Inline parameters of unknown width are read off their values.** A family argument whose binding has no known type leaves its width unresolved, and the stacked arguments settle it, so a vector-valued `let` passed to a scalar family parameter supplies one parameter per coordinate. Literal parameters broadcast against such a vector, and its per-coordinate densities sum to one per row. The GRU, LSTM, and vanilla-RNN language models had drawn every candidate-state coordinate around the first coordinate of its mean; a gather such as `emission_rows[state]` is now one plate row wide rather than the whole plate.
- **A plate over a matrix-valued family sizes its rows by the matrix's element count**, so a `PlateDraw` over a Cholesky-factor family draws and scores whole factors.
- **`loo_pit` matches ArviZ 1.x**, passing `var_names` and returning a dataset.

### Removed

- `check_applicative_laws` and `check_alternative_laws`, which asserted nothing.
- The unused `DirectBernoulli`, `DirectNormal`, and `DirectTruncatedNormal` classes.
- Exports of internal machinery: parser walkers, compiler mixins and their private re-exports, elaboration and lowering helpers, program-encoding constants, the CLI subcommand `main` functions, and `commit_id_for`.

## [0.24.0] - 2026-10-03

### Changed

- **Distribution-family construction is keyword-only.** Every QVR family application now names one complete parameter schema, such as `Normal(loc=mu, scale=sigma)`, `Dirichlet(concentration=alpha)`, or either `Categorical(probs=weights)` or `Categorical(logits=scores)`. Parameter names are structural syntax-tree nodes, are preserved through parsing, emission, validation, eager execution, QIEC elaboration, and transpilation, and are reordered only at positional target-API boundaries. Ordinary computation and morphism application remains positional. `qvr migrate` names legacy family arguments and gathers formerly splatted categorical and Dirichlet values into their vector parameter.

## [0.23.0] - 2026-09-23

### Added

- **Identified categorical formula models.** `family="categorical"` now infers contiguous response labels, takes category zero as the reference, and emits class-specific fixed and non-centered random effects over the remaining `K - 1` logits. External neural predictors may return either reference logits or a full `K`-logit matrix.
- **Configurable finite-mixture regressions.** `family="mixture"` accepts any integer `mixture_components=K` with `K >= 2`, shifts all Gaussian component locations by the formula predictor, and learns shared softmax weights, sum-zero component offsets, and positive component scales. An orthonormal Helmert basis maps `K - 1` free contrasts into the sum-zero `K`-dimensional subspace without a redundant direction; under the default isotropic Normal priors, the induced distributions are permutation symmetric. The likelihood integrates component assignments rather than adding discrete latent sites.

### Fixed

- **Scalar formula families match their documented parameters.** Gamma converts mean and shape to concentration and rate, Beta converts mean and precision to two concentrations, Student-t uses `(df, loc, scale)`, and zero-inflated and hurdle Poisson use `(zero_probability, rate)`.
- **Derived mixture vectors retain their component width.** An indexed component draw now supplies the width of transformed weight and location vectors to the multi-vector inline likelihood, so formula-generated mixtures execute under SVI and MCMC.
- **Python transpilers lower softmax on the category axis.** Pyro, NumPyro, PyMC, and Edward2 emit their native softmax with `dim=-1` or `axis=-1`, allowing generated categorical and mixture formulas to cross those target boundaries.
- **Fitted QVR source preserves prior configuration.** `BayesianFit.qvr_source` and `dump_qvr` now re-emit the fixed, random-scale, and named prior overrides used for the fit.

## [0.22.0] - 2026-09-23

### Added

- **Formula-compiled ordinal mixed models.** `family="cumulative"` now learns an identified ordered cutpoint vector from positive spacings. `thresholds_by="group"` gives each level partially pooled spacing deviations, gathers its centered cutpoints per response row, and composes with the formula's fixed and random effects.
- **Differentiable external predictors.** `fit(..., predictor=module, predictor_data=tensor, method="svi")` adds a PyTorch module's per-row output to the formula linear predictor and optimizes its parameters jointly with the QVR program and variational guide. Emitted QVR exposes the attachment as the explicit `neural_eta` host input; frozen predictors also work with HMC and NUTS.
- **True binomial formula likelihoods.** `binomial_trials` accepts either one positive integer or a per-row data-column name and emits `Binomial(total_count, probs)` rather than treating binomial as a Bernoulli alias.

### Fixed

- **Formula-family parameter semantics.** Cumulative formulas now supply ordered cutpoints to `OrderedLogistic`. Negative-binomial formulas convert their mean/concentration surface to QVR's `(total_count, probs)` convention with `probs = mu / (mu + disp)`.
- **Impossible log-probability cells have finite gradients.** `LogProbAlgebra.join` retains `-inf` for an all-`-inf` reduction and defines its gradient as zero instead of inheriting `NaN` from `torch.logsumexp`.
- **The ArviZ dependency matches the DataTree API.** The `diagnostics` and `formulas` extras now require ArviZ 1.0 or newer, the first stable line matching the adapter's `xarray.DataTree` contract.

### Changed

- **Binomial's inline family signature is explicit.** Runtime and generated programs share the two-parameter `Binomial(total_count, probs)` surface; the conditional morphism continues to learn probabilities for its configured trial count.

## [0.21.0] - 2026-09-22

### Fixed

- **Nullary logical-form constants match deduction patterns.** Inline and TSV-backed lexicons now encode a declared nullary constant as `("atom", name)`, the same representation the rule compiler emits. A rule such as `span(I, J, QNP, App(every_q, P))` therefore matches an LF constructed from the lexical constant `every_q`.
- **Complex TSV logical forms parse through the current grammar.** File-backed lexicons use a valid synthetic QVR module before applying the same constant tagging, binder alpha-renaming, and scope validation as inline lexicons.

### Changed

- **Deduction terms use discriminated public leaves.** A bound LF variable is now `("var", canonical_name)` rather than `(canonical_name,)`; a nullary LF constant is `("atom", name)` rather than `(name,)`; and applications remain `(constructor, *arguments)`. Consumers that inspect chart tuples should migrate one-tuple leaf checks to the corresponding tag. `atom` and `var` are reserved in deduction `atoms` declarations.
- **The Montague NLI example factors syntax from quantificational force.** `every` and `some` share one `Det` category and one determiner-composition rule; sentence rules distinguish them by matching their nullary LF constants.

## [0.20.1] - 2026-09-22

### Fixed

- **Finite traversals cross the probabilistic target boundary.** The program-plan walker reassociates the nested QIEC binds produced by a fixed `traverse`, so sequenced collection calls inside `program` bodies transpile to all eight dynamic targets. The lexical-uncertainty RSA example now transpiles completely to Pyro and NumPyro rather than stopping at its state-likelihood traversal.
- **Checked integer values retain their target type.** `int(real_value)` renders as a real-to-integer cast on Pyro, NumPyro, PyMC, and Edward2 instead of failing during Python target emission, and integer-typed QIEC literals render as integer tokens rather than floating-point tokens. Tensor indices produced by finite traversal therefore remain valid target-language indices.
- **Target-aware tooling validates the executable plan.** `qvr check --target`, the REPL, and the LSP now derive the probabilistic plan after QIEC capability analysis and report plan-level refusals such as a non-log-sum-exp marginalization before `qvr transpile` is run.

## [0.20.0] - 2026-09-21

### Added

- **Checked finite collection expressions.** `map`, `fold`, `length`, and `logsumexp_over` now lower from the shared pure-expression surface to typed QIEC terms. `traverse` sequences a named computation across a tensor with a fixed leading extent, preserves source order and call provenance, and joins the callee's effects into the enclosing row. A local pure binding may name a lambda reused by these operations. `filter` remains available to the eager PyTorch evaluator and is rejected under `qiec-collection:filter:dynamic-shape` because its value-dependent result length is not a fixed QIEC tensor shape.
- **Collection documentation and executable examples.** The QVR reference, program guide, formal semantics, and a finite-grid Bayesian tutorial state the pure/effectful distinction, shape rules, diagnostic boundary, and backend support. The tutorial source is checked and runnable through the reference machine.

### Changed

- **Multiline delimiters use a hanging-indent form.** Nonempty parenthesized, square-bracketed, and brace-delimited lists may place one trailing-comma-terminated item per indented line. Calls, QIEC static applications and requests, constructor values, list literals, and indices share this grammar. The source protocol advances to `qvr-source/v0.20`; the QIEC and JSON ABIs remain `qiec-core/v1alpha1` and `qiec-json/v1`.
- **Let-expression analysis is exhaustive and binder-aware.** Free-name collection, recursive walks, and capture-avoiding substitution share one implementation across program elaboration, native compilation, QIEC lowering, and dependency resolution. Newly introduced AST variants therefore fail explicitly instead of disappearing from one of those passes.
- **Incomplete program elaboration is an error.** `qvr check`, the LSP, and entry lookup report `qiec-program-gap` when a source program cannot enter the checked module. Native-only builtins no longer make the checked path appear successful.
- **Builtin capabilities have one registry.** Eager implementations and arities, purity, shape action, checked QIEC form, and execution routes are recorded together. Native-only calls report `qiec-builtin:host-only`; eager execution no longer drops additional call arguments.
- **Collection calls highlight consistently.** Pygments, the REPL/TUI, tree-sitter queries, and the packaged VS Code grammar classify the checked collection operations as builtins.
- **The v0.19 grammar is frozen in the migration chain.** Its released parser and schema identity now sit between v0.18 and `HEAD`; the new hanging-delimiter grammar has its own verified `HEAD` snapshot and a validating, byte-preserving v0.19 migration hop.

## [0.19.0] - 2026-09-19

### Added

- **QVR v0.19 indexed families and algebraic effects.** The typed QIEC surface adds closed index sorts, indexed-family constructors, heterogeneous static telescopes, parameterized effect interfaces, lexical instances, open effect rows with `lacks` constraints, handler contracts with resumption grades, and checked `return`, `perform`, `bind`, `handle`, and indexed-case computations. Type binders shadow same-spelled primitive types, and effect-interface applications instantiate `Effect` binders under the same lexical kinding rules.
- **A stable indexed-effect kernel.** QIEC modules lower through Didactic's exact `qvr-source/v0.19` to `qiec-core/v1alpha1` route, serialize through the deterministic `qiec-json/v1` allowlist, and execute under a reference evaluator with deep lexical handlers. Deserialization validates every allowlisted field recursively, and module validation checks closed effect instances, request provenance, and escaped static variables. Runtime callables and mutable resources remain process-local attachments.
- **QIEC-aware developer tooling.** `qvr run` executes named computations with checked JSON arguments, closed static specializations, explicit runtime-provider configurations, and stable traces. The REPL and TUI expose the same boundary through `:run`, `:runtime`, and `:detach`, and the TUI status bar shows the active runtime and last result. `qvr check --target`, `qvr lsp --target`, `workspace/didChangeConfiguration`, and the VS Code `qvr.transpileTarget` setting provide live target-capability diagnostics. These surfaces also share QIEC symbols, completions, source ranges, and stable diagnostic codes. The Pygments and tree-sitter highlighters recognize the same v0.19 surface. The panproto migration chain gains a validating, byte-preserving `v0.18.0 -> HEAD` hop because the new grammar is additive.
- **A complete v0.19 documentation track.** The QVR reference now covers indexed families, effects and handlers, computations, probabilistic-program elaboration, generated parser and structural graphs, entry execution, runtime providers, LSP behavior, transpilation, and grammar ownership. Five executable tutorials develop the new features from indexed recursion through target-aware release checks. The examples gallery gains a feature map, checked-runtime paths for the schema parser and term autoencoder, and an integrated Amortized Bayesian Semantics case study combining parser search, structural attachments, indexed missingness, the measure algebra, an authored handler, hierarchical effects, and grouped exact marginalization.

- **Programs are computations.** A `program` elaborates to a named computation over the module's canonical `random` and `score` instances: its data, observations, and fibrations are typed parameters, `sample` and `observe` steps perform `Random.sample` and `Score.add`, plates are typed tensor shapes, and a `marginalize` block is a helper computation enumerated under the prelude's enumeration handler. The module records each program as a `ProgramEntry`, `run_program` scores a program's log joint on the reference machine, and a program body binds a named computation's result with `let x <- helper(args)`, draws from another program with `sample x <- sub` or `sample (a, b) <- sub` (each name one component of the pair the program returns), while a computation calls a program by name.
- **Literal family arguments fix a kernel.** A declaration such as `morphism prior : A -> R ~ Normal(0.0, 1.0)` now denotes that same input-independent family on the torch runtime and the QIEC reference machine. It carries no learned lookup table; a bare `~ Normal` continues to request an input-conditional parameter source.
- **Schema chart parsers are computations.** A `parser(...)` definition over declared schemas and bundles elaborates to a finite category item family, lexical and rule weights, a goal-directed recursive derivation, and a `Search`/`Weight` entry computation. `run_deduction` gives the same inside score as the classic `InsideAlgorithm`, including the bounded unary closure, and transpile targets refuse the executable parser under `qiec:capability:search` rather than treating its declarations as inert metadata.
- **Structural autoencoders are typed call graphs.** A structural signature becomes an opaque term family indexed by its declared sorts. Encoders, decoders, and decoder negative log likelihoods become typed `Compute` computations, while a reconstruction loss calls the encoder and decoder computations and returns a `LogWeight`. `Program.run` supplies the compiled PyTorch modules at the provider boundary, validates their results, preserves their autograd graph, and agrees exactly with the classic loss registry. The modules are registered under stable source-derived names so optimizers and checkpoints see every parameter. Targets without those attachments refuse the named computation under `qiec:capability:neural-attachment`.
- **Open input extents are static parameters.** A program input whose shape no plate or annotation fixes, such as the concentration vector of `sample probs <- Dirichlet(alpha)`, is typed over an index variable the program's computation binds, so the entry point records one `Nat` binder per open extent, a run reads each extent off the data, a call step from another program takes the extent as its own, and an authored computation names it with an index literal: `prog[3](alpha)`. Integer literals are admitted wherever a static argument fills a `Nat` binder.
- **The reference machine scores every family.** The reference distribution backend implements all fifty-two families of the semantic registry in plain Python, the matrix and compositional ones included, with each density held to the torch definition the transpile probes use; a program using any registry family runs end to end on the reference machine.
- **The program protocol carries the computation graph.** `QVR_PROGRAM_PROTOCOL` gains vertex kinds for effect interfaces and their operations, lexical instances, handlers and their clauses, named computations and their parameters, and program entry points with their parameters and sites, joined by `instance_of`, `handles`, `row`, `calls`, `performs`, `installs`, `body`, `random`, and `score` edges. `extract_program_schema` writes the checked module beside the object and morphism declarations, so a mixed module diffs as one call graph.
- **Effect handlers are lexical kernel handlers.** A `MonadicProgram` runs on the reference machine as the kernel computation `program_kernel` encodes it to, whose draws are `Random.sample` requests on the program's `random` instance and whose densities are `Score.add` requests on its `score` instance, with torch tensors as host values. Every `EffectHandler` on the stack installs a prelude handler of one of those instances, so `clamp`, `do`, `scale`, `mask`, `block`, `replay`, `lift`, `collapse`, the reparameterizations, and the trace compose by the kernel's own rules: a handler of `random` sees a site before any handler outside it and sees the outer answer flow back, and every handler of `score` transforms every density on its way to the accumulator. A trace reads each site's density from the contributions that reached the accumulator under the site's provenance, and records the morphism the program declares beside the sampleable actually drawn from. The prelude gains the `Compute` and `Param` interfaces and the `draw_scoring`, `intervene`, `reweight`, `block`, `compute`, and `param` handler factories the stack is built from.
- **The measure algebra is a family.** `Restrict` (also spelled `Truncate`), `Normalize`, `PointMass`, `Mixture` over a heterogeneous component list, `Transformed` (also `Pushforward`) over a named bijector chain, and `Independent` are registry families with reference densities and log masses, and the classic operator forms elaborate through them; an integer atom beside an integer family, `Mixture([0.3, 0.7], [PointMass(0), Poisson(rate)])`, samples an `Int`. `ZeroInflatedPoisson`, `HurdlePoisson`, and `ZeroOneInflatedBeta` join the registry.
- **Grouped marginalization in full.** A `marginalize` block elaborates with its `reduction` (`logsumexp`, `sum`, or `mean`), a product group `[over=[G, H]]` with the product fibration `[via=[g, h]]` flattened row-major, and nesting: a grouped block inside a grouped block adds one aggregate per group position to the enclosing group's weights, its group either identified with the enclosing one position by position or a product refining it, whose aggregates and enclosing-group-shaped arguments are carried along the projection. A `sample` inside a block that reads nothing the block binds is drawn once before it. A vector family's spread literals, `Dirichlet(1.0, 2.0, 3.0)`, gather into its one parameter, and a single literal is the symmetric parameter at the dimension the plate or the program's codomain fixes; a `: A` annotation beside an `over` clause is the batch plate. The classic compiler accepts an indexed class prior, `Categorical(theta[z])`, in a grouped block and host-data product fibrations, and reports its own diagnostics before the elaboration's.
- **One lowered root.** `Lower` derives a program's target plan from its checked computation rather than from the source steps: a `Random.sample` request is a draw, a scored density at an observation an observation with its fibration, a pure binding a deterministic node, another scored weight a score, and a marginalization helper's scored call an integration scope, each with the support and plate read off the module's types, and the inputs are the entry point's parameters, named by the axes the elaboration recorded. `IRProgram` carries the checked `IRQiecModule` as its `module` beside the plan, the renderers place the module's other computations with `render_computations_dynamic` and `render_computations_static`, and the transpile boundary is the lowering itself. A program the elaboration has no computation for is refused under the kind its gap names; an unknown call in a `let` is refused (`let:call:unknown:<name>`) rather than emitted as an unresolved name, and a lambda binding leaves no node since the elaboration inlines its uses. Program parameters record their axis names. `sum` and the other reductions of a tensor of rank two or more reduce along the last axis, as the torch runtime does; a `marginalize` over `ContinuousBernoulli` or `RelaxedBernoulli` enumerates the hard atoms under the Bernoulli masses, one latent per element of an annotated axis, as the torch runtime does; `Pareto` binds `scale` before `alpha` and the relaxed families take their `temperature` first, as the torch constructors do; a zero-rate `Poisson` is a point mass at zero.
- **Scans are recurrences.** A draw from a chain with `scan(cell)` expands to a step program that draws the chain's elements before the scan at one position and applies the cell, and to a let `scan(step, xs)` over the chain's input, which elaborates to a recursive helper computation over the positions of an open extent the program binds; the step's sites are reached once per position and `run_program` replays a site's `n`-th occurrence from the value keyed `"<site>@<n>"`, so the reference machine scores the trajectory the torch runtime's `ScanMorphism` scores. A kernel morphism's heads follow the torch runtime's parameterization for every family it supports, a scalar family applied to a vector draws one coordinate per entry, and a program called with its domain arguments alone passes its remaining inputs through the caller. Every gallery model but the program templates elaborates in full; the transpile boundary keeps refusing scans.
- **Network kernels are computations.** A draw through a morphism carrying `[param_source=mlp]` elaborates to the family with its heads read off the last of a chain of affine maps under `tanh`, the hidden widths taken from `mlp(a, b)`, `hidden_dim`, or the default two layers of sixty-four; the layers' weights and biases are typed program inputs whose shapes match the torch runtime's `MLPSource` layer by layer. An embedding morphism, `[role=embed]`, elaborates to a Normal kernel over its finite domain reading a table of centres and log scales. A kernel under a plate applies its map at every row, so an unbound conditioning name becomes a per-row data input. A step drawing from a composite expands into one site per factor: `fan` over a `[replicate=k]` morphism draws through each replica and bundles the draws, a kernel conditioned on a bundle reads its rows in order, `stack` composes copies declared afresh with parameters of their own, and a tensor product's branches at the head of a chain read their own factors of the program's domain, and the chain's head conditions on the row the step wrote. The neural gallery models elaborate and run on the reference machine, whose site densities agree with the torch runtime's morphisms.
- **Deductions are computations.** A `deduction` elaborates to a closed item family, a recursive derivation computation, and an entry that answers the inside weight of the goal in the declared semiring. Rules become branches of the derivation: it chooses an axiom or a group of rules of one premise shape on the module's `Search` effect, derives each premise by recursion, matches it by `case` analysis with the family's structural equality for shared pattern variables, instantiates the conclusion, and adds the rule's weight on a `Weight[K]` instance; a failed match performs an empty choice. Learned weights are read through a `Param` instance under the names the agenda engine keys its parameters by, a span-shaped system derives by span, and any other takes its axioms and their weights as input. The entry handles the weight instance with a collecting handler and the search with a handler that resumes once per alternative and combines the shots by the semiring's addition, so the reference machine gives the number the agenda engine tabulates, in every semiring. A program calls a deduction with `let chart = parse(D, sentence)` and scores `chart.goal_weight()`; `run_deduction` runs one on the reference machine and `deduction_item` builds its items. A foreign clause may resume in tail position with `TailResume`, which the collecting handler does so a search inside its scope may resume any number of times, and a collecting handler configured `forkable` gives each shot its own total. An integer operand beside a real one is converted in a program or computation body, the kernel term carrying the conversion. The agenda engine's non-learnable entries and rules contribute the semiring's one, so its Boolean and counting charts hold.
- **Programs call computations on every target.** A program that binds a computation's result, `let c <- noisy(b)`, transpiles to each of the eight host-language targets as a call in the model body: the plan inlines a pure callee whose body is a chain of bindings, and renders every other call against the callee's generated entry point with the program's `Random` and `Score` instances bound to a native operation table built from the host's own primitives, so a draw inside the helper is a site of the model under the helper's label and a scored weight is a term of the joint. Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church score a model calling an effectful helper to the reference machine's joint, the probe images checking each. The Python bridges spell the scalar primitives in the host's own library, so a helper's arithmetic on a drawn value stays on the host's array. Stan defines a needed pure computation as a user-defined function, with scalar primitives, conditionals, calls, and recursion, and calls it as a transformed parameter; BUGS and JAGS refuse a call as `call:graph:<callee>`. A static target is judged by the computations the program's calls reach, and defines every computation it can represent; one the program never calls and the target cannot represent is left out of its output.
- **The WebPPL runtime is WebPPL.** The QIEC runtime a WebPPL program carries is written in WebPPL's functional subset, with recursion for its loops and the global store for its serials, ambient handler stack, and handler cells, so a generated program compiles and runs under the `webppl` command line, inference included, and the runtime tests run it there. A WebPPL host attachment holds a mutable binding in a global-store cell named by its `cell` field.
- **Every target scores a `score` step.** Edward2 and Gen, which have no factor primitive, trace a scored weight as a choice of a one-point distribution whose log density is the weight, Gen under the `:qvr_factor` address namespace, and their probes read it as a term of the joint. Gen also marginalizes: its emission is the Turing renderer's, the per-row draws stacked into matrices and the reduced weight traced as a factor, and `lda` and `zip_regression` score the reference joint on Gen. Church binds a scored weight under its name before adding it.
- **Julia shapes follow the plan.** Turing and Gen read whether a binding is per-element off the plate the plan gives it, which the elaboration reads off the binding's checked type, rather than off the sites that read it, so a scalar binding a plated observation reads, `cell0 = cell_score[0, 0]`, is one value shared across the rows; BUGS and JAGS write a plated list binding, such as a case table over a finite object, one cell per relation.
- **One entry-point invocation.** A checked module's executable entry points are its `define` computations and its programs, listed by `entry_points` and invoked by `invoke_entry`, which `qvr run`, the REPL's `:run`, `Program.run`, and `Compiler.entry` share, so an entry validates its arguments, selects its providers, traces, and fails with the same `qiec-run-*` codes however it is reached. A program runs forward on the reference machine through `sample_program`: the sites given with `--site` are replayed and scored, every other site is drawn and scored under the core provider's `draw-scoring` handler kind, its parameters are given positionally or by name with `--data`, and `--seed` seeds the reference generator the draws use; the run reports the program's value and its log joint. `qvr run FILE` with no entry name, or `:run` alone, lists the entry points with their signatures. Each invocation owns its observations, parameters, generator state, score accumulator, and trace, and no handler stack outlives a call.
- **Programs run on the reference machine.** `MonadicProgram.rsample` and `MonadicProgram.log_joint` run the program's kernel encoding under the active effect handlers, so the joint `log_joint` reports is the one `trace` accumulates; each host step is encoded over the values it reads, a step closure declaring its names through `reading`, and every handler a run installs resumes in tail position, so a program's encoding and its evaluation grow linearly with its length. A program a host step runs in turn is a separate invocation the enclosing run's handlers neither observe nor condition. A program that calls a named computation compiles to a `CheckedProgram`, whose `rsample` and `log_joint` are the module's own `sample_program` and `run_program` on its entry point. A deterministic morphism drawn as a step is contracted against its input as an `apply` step of the encoding.
- **Nested grouped marginalization scores as stated.** An inner grouped block contributes one aggregate per position of its plate to the enclosing block, identified with the enclosing plate position by position or summed along the projection from a product plate onto it, as `programs.md` §2.7 states; `marginalize_grouped` returns the per-group aggregates under `per_group=True`. A nested plate of another extent is refused at compile time.
- **Every target keys a grouped block by group.** A `marginalize` block whose observed rows fibre into its groups through a `via` index sums each atom's per-row log-likelihoods within the group before the reduction over the atoms, as `programs.md` §2.7 states, on Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, Church, and JAGS, which selects a group's rows through an identity matrix subscripted by the fibration and adds the per-group density through the zeros trick. BUGS refuses such a block (`marginalize:grouped-fibration`) and every scope that is no categorical mixture (`marginalize:no-collapse`), since the language has no statement that adds a free log-density to the joint and a declared latent would denote a measure on a larger space than the source integrates. Church integrates a marginalized latent out through its runtime's `log-sum-exp`, `group-sums`, and `factor`, binds each observe and factor under a name so a definition may follow it in the body, and binds a call's static environment and local aliases once per body.
- **Static arguments parse in the kernel's form.** `parse_static_arguments` reads `n=S(S(Z))` as the index term a checked type carries, a nullary constructor being the sort's literal, so a specialization at a user index sort validates the values the module's own constructors build; a diagnostic names an escaping local instance by its source name.
- **A program draws from a program.** `sample theta <- school_effects(0.6, School)` and `sample (u, v) <- sub` run the program in place: the objects and scalars the draw names substitute for the program's template parameters, each name the program binds is renamed under the binder (`z` becomes `theta$z`, the returned name becomes `theta`, a pattern's names take the returned names positionally), and every target spells a renamed site with `__` (`theta__z`), since no target admits `$` in an identifier. A program over object parameters is itself a computation whose telescope binds one index variable per object, listed and invocable as an entry point with its statics; `let h <- cell(x, h_prev)` calls a program as the computation it is, which the plan refuses (`qiec:call:program:<name>`) since the callee's sites are its own. A morphism parameter takes a declared morphism at the draw, `with_prior(Subj, my_prior)`, the template's steps drawing through it; such a template is no entry point of its own, a morphism being no static argument. A plate over a finite object drawn through a kernel on that object conditions each row on its own element. A `score` step on JAGS binds its zeros-trick carrier in the emitted `data` block, and the BUGS probe supplies the carrier the BUGS emission leaves to the host.
- **One check path on every surface.** `qvr check`, a REPL `:load`, the TUI's diagnostics panel, and the language server report call-graph, row, handler-body, recursion, and target-capability errors with the same codes and positions, and a row or type in a diagnostic renders in the surface spelling with the instances' source names (`!{random : Random}`) rather than as a dataclass. The REPL's `:type` resolves an operation through its interface or an instance with the instance's static arguments substituted, `:kind` and `:effects` reach the prelude's interfaces, `:effects` prints a computation's inferred row after call expansion and a program's checked row, completion offers a prelude instance's operations, and the TUI status bar names the last entry point with its kind, value, and log joint, or a failed run's diagnostic code. `:set target=` and `qvr repl --target` report a transpile target's capability diagnostics on load, as `qvr check --target` does.
- **Calls and clauses in the language server.** Hovering a call shows the signature the call instantiates its callee at and the callee's inferred row; hovering a request's operation shows the instance's instantiated signature above the interface's; a computation's hover carries its inferred row. `textDocument/rename` and `prepareRename` rename a declaration at every reference and refuse prelude names; document symbols nest a computation's parameters, locals, scoped instances, branch binders, and call sites, a handler clause's binders and calls, and a program's steps; a clause's binders resolve inside the clause alone. A configuration change re-runs only the capability pass over the already checked module. The REPL and TUI highlighters take their token classes from the Pygments lexer, which marks a clause's operation and a call's callee as functions, a resumption grade as a number, and a handler option key as a keyword; the tree-sitter query and the TextMate grammar agree on the same corpus.
- **Distributions, tensors, and sites are values.** Applying a family of the semantic registry builds a typed `Sampleable`, a list literal is a `Tensor`, `site("x")` names a sample site, and `log_prob(d, x)` is a `LogWeight`; a construction may be plated over batch and event axes with parameters broadcast along them. The pure expression layer gains elementwise math over tensors, whole-tensor reductions, last-axis operations, `factor` comprehensions, gathers, segment sums, kernel matrices, affine parameter maps, and table-indexed parameter maps, which a kernel over a finite domain reads its heads from at the element the program is applied to. Every dynamic transpile target spells a first-class distribution in its own library with that target's parameterization conventions, and probe images check each host's log density against the registry.

### Changed

- **`Horseshoe(scale)` is refused by every transpile target.** The family denotes the marginal of a normal draw over a half-Cauchy local scale, which has no closed form and no library distribution on any target. Each target refuses the family with a diagnostic naming the explicit `tau * lambda * z` decomposition, which every target expresses and the gallery already uses. The torch runtime and the reference machine score the marginal itself.
- **Effect declarations are unversioned source contracts.** An `effect`
  contains only its name, static telescope, and operation signatures. QIEC ABI
  and package compatibility remain external to authored QVR declarations.
- **Indexed declarations use Didactic's GADT API directly.** QVR families and constructors are projected through Didactic's public `GADT` API and compiled by Panproto before the QIEC-specific checker completes the exact lowering route. QIEC retains stable identities, effect rows, and the distinction between uniform family parameters and refinable indices.
- **The current QVR grammar ships with Quivers and loads fail-closed.** Editable checkouts use their generated `grammars/qvr/src`; wheels carry the same generated parser source. `panproto-grammars-all` continues to provide the eleven transpiler-target grammars but is no longer the QVR source of truth. Panproto validates persisted historical objects against the enum payload read from disk, so the migration chain retains its content-addressed fixture identities without rewriting stored hashes. The floors move to `didactic>=0.17.1`, `panproto>=0.74.4`, and `panproto-grammars-all>=0.74.4`.
- **Platform wheels load QVR without a host compiler.** Each Linux, macOS, and Windows wheel contains a native current parser plus native libraries for every packaged migration snapshot. Source-binding manifests cover the generated parser and tree-sitter metadata as well as the library bytes; a missing or inconsistent pair fails closed. Source distributions and editable grammar development still compile during build or into the development cache, but an installed platform wheel does not compile on first parse, highlight, or migration.
- **Probabilistic transpilers execute the QIEC boundary.** Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church emit named QIEC functions through corresponding stable-ID ABI implementations, including lexical handlers, effect requests, indexed cases, attachments, and checked resumption grades. Target reparse, external-syntax, and available-runtime execution tests exercise these implementations without claiming an equivalence proof. Stan admits closed, monomorphic, effect-free scalar computations of bindings, primitives, conditionals, calls, and recursion, with named `Bool`, `Int`, and `Real` value parameters; BUGS and JAGS admit parameterless `Return`/`Bind` computations. A computation the target lacks a feature for receives feature-level diagnostics.
- **The transpiler has one renderer path.** The unused PyMC-only AST adapter and the public `SchemaTransform` / `realize` exports are removed; every target enters through the registered Module → IR → renderer boundary.

## [0.18.0] - 2026-09-10

### Added

- **Transpilation to eleven probabilistic programming languages.** `transpile(module, target=...)` emits a parsed QVR module as source for BUGS, Church, Edward2, Gen, JAGS, NumPyro, PyMC, Pyro, Stan, Turing or WebPPL; `available_targets()` lists them. A previous release shipped the target-agnostic scaffold, the IR and the lowering, and left the per-target renderers out; this release supplies them, so `available_targets()` returns the eleven names rather than nothing. The emitted program denotes the same measure as the QVR module, up to a point-independent additive constant that reflects the target's base measure, and [`docs/transpile-support.md`](https://github.com/quivers-dev/quivers/blob/main/docs/transpile-support.md) records the measured constant for every cell that carries one.
- **A construct the target cannot express is refused rather than approximated.** `UnsupportedConstruct` names the construct, says why the target has no form for it, and gives the ways to write the model that do transpile. The message is prose assembled from the module's own AST; the structured tag that identifies the rule is still carried for programmatic dispatch. Two classes of refusal appear. Some constructs no backend expresses, `scan(...)` among them: a scan denotes one draw per sequence position over intermediate states the program never names, over an axis that arrives with the data, so there is no loop bound to emit and no name to bind the states to. Others are target-specific, and each turns on a measure the target cannot represent rather than on effort: Gen has no way to add a free log-density term to a trace, so it cannot integrate a `marginalize` at all, and BUGS, Church, Edward2, JAGS, PyMC and Turing cannot express the accumulated reduction an ungrouped `marginalize` denotes.
- **`qvr transpile` on the command line and `:transpile` in the REPL.** The subcommand takes a `.qvr` file with `--to <target>` or `--to-all`, and the REPL command completes on the available target names.
- **`ConditionalBetaBinomial` and `ConditionalHalfStudentT`.** Conditional forms of two families whose unconditional versions were already available, for counts with overdispersion and for positive scales with heavy tails.
- **Measured transpilation support for every backend.** [`docs/transpile-support.md`](https://github.com/quivers-dev/quivers/blob/main/docs/transpile-support.md) reports all 46 gallery programs and all 46 construct fixtures across all 11 backends. It distinguishes constructs no backend supports from target-specific refusals and records each raised message. A test regenerates the page and requires the committed copy to match byte for byte, so changes to measured support require a matching page.
- **Per-backend correctness notes and an architecture guide.** [`docs/semantics/transpile-correctness/`](https://github.com/quivers-dev/quivers/blob/main/docs/semantics/transpile-correctness/) states, for each target, the measure its emitted program denotes and the base measure that separates it from the QVR reference. [`docs/semantics/transpile-architecture.md`](https://github.com/quivers-dev/quivers/blob/main/docs/semantics/transpile-architecture.md) describes the lowering and the renderer surface.
- **Five gallery examples.** A beta-binomial A/B test, a half-Student-t hierarchical model, a Kumaraswamy bounded-outcome model, an LKJ-Cholesky correlation model, and a logistic-noise regression.

### Changed

- **An ungrouped `marginalize` reduces over the body, not over each row.** A block that carries no index and no `over =` clause declares one latent, and the rows of a plated `observe` inside it are conditioned on that single draw, so its density accumulates the rows before reducing over the latent ([`programs.md` §2.6](https://github.com/quivers-dev/quivers/blob/main/docs/semantics/programs.md)). The compiler reduced each row on its own and summed, which gives every row a draw the source never declares; on the gallery's `hmm` the two differ by 1.5 nats at the ground-truth point and the difference moves with the data. Log densities of ungrouped marginalize models change accordingly. A block that carries a grouping plate is unaffected.
- **The vanilla-RNN transition is explicit.** `vanilla_rnn_lm` previously hid its transition in a morphism carrying `[param_source=mlp]`; the model-internal weights prevented every backend from transpiling it. The cell is now a program with a declared step. Its update uses one linear parameter map followed by an affine transform of a `LogitNormal` draw; the previous version used a two-hidden-layer network.

## [0.17.0] - 2026-07-22

### Added

- **A zero-one-inflated beta family for proportions that pile up at their endpoints.** Rates, shares, and probabilities are frequently recorded with exact zeros and ones. A beta component has no point mass at either endpoint, so it cannot assign positive probability to these boundary observations. `ZeroOneInflatedBeta(mu, phi, zoi, coi)` mixes point masses at 0 and 1 with a beta on the open interval, taking `zoi` as the probability that an observation is an endpoint and `coi` as the probability that such an endpoint is 1. The continuous part uses the mean-precision parameterisation, so `mu` remains the mean of the beta component and `phi` its precision. The formula frontend names it `zero_one_inflated_beta`, with a logit link on the mean and on both inflation probabilities and a log link on the precision.

## [0.16.0] - 2026-07-15

### Changed

- **A kernel's parameter source defaults to a linear map.** A morphism declared `f : X -> Y ~ Family` over continuous spaces reads its distribution parameters from a single `nn.Linear`, which is how its arrow reads. A model whose parameters depend nonlinearly on the input requests that with `[param_source=mlp]`, so the fact appears in the source instead of in a default: previously every such kernel carried a two-hidden-layer tanh MLP, and a `Real 1 -> Real 1` kernel silently held 4418 weights. The gallery is now what it says it is. `linear_gaussian_ssm` is linear, and an LSTM, GRU or vanilla-RNN gate is one matrix under a cell whose nonlinearity is the sigmoid its program body applies. `deep_markov`, the VAE's encoder and decoder bodies, the transformer and seq2seq feed-forward sublayers, and the recurrent cells that carry their own tanh now name the MLP they rely on.
- **Morphism options are checked against the role that reads them.** Options were checked against the union of every role's keys and then read per role, so an option belonging to another role passed validation and was dropped in silence. `[scale=]` on a family-backed kernel is the case that bit: nothing reads it there, and all 34 uses across the examples configured an init that never existed. Each role now declares the keys its lowering consumes, and a key outside that set is rejected with an error naming the role that does read it.
- **A default coefficient prior is autoscaled to its column.** A column enters a formula's linear predictor as `beta * column`, so a prior on the coefficient alone states the scale of its contribution, and one fixed `Normal(0, 5)` thus means something different for every column. The default's scale is now divided by the column's root-mean-square, which states it in contribution space; an explicit per-name prior is emitted as written. Coefficients stay on their own column's scale, so nothing is transformed back.

### Added

- **Inline distributions take any number of vector-typed parameters.** A family may declare vector parameters in any position, and the stacked input is split at each one's declared dim rather than by a single trailing parameter eating the remaining columns. `MixtureNormal(weights, locations, scales)`, whose three per-component vectors are each shared across the response plate, is expressible.
- **`[param_source=<kind>(...)]` carries its arguments.** The grammar has always parsed the call form and `param_source_from_option` has always read parenthesised hidden widths, but the compiler decoded the option as a bare identifier and raised on anything else, so `[param_source=mlp(16, 8)]` died at compile time and the width parser was unreachable from QVR.
- **`hidden_dim` takes a sequence of widths.** One entry per hidden layer, so `[param_source=mlp, hidden_dim=[64, 32, 16]]` builds those three; a bare `hidden_dim=64` is the one-layer case. A single number could say how wide an MLP's layers are but not how many of them there were.

### Fixed

- **Every family that has a parameter source can select it.** `[param_source=...]` is accepted by the compiler for any family-backed kernel, and exactly one of the 28 conditional families could receive it: `_make_continuous_morphism` passed the keyword into a constructor that only `ConditionalNormal` declared, so the other 27 died on a `TypeError` raised inside `__init__`, with no line or column on it. Every family whose parameters come from a source now takes `param_source` and `param_source_option` and hands them to the one seam that builds it. The four whose parameters do not (`Horseshoe`, `GaussianProcess`, `Independent`, `Transformed`) refuse the option at compile time, naming the family.
- **`hidden_dim` is read or refused, not dropped.** It was ignored whenever `param_source` was given, so `[param_source=mlp, hidden_dim=32]` silently built the default width. It now reaches the source that has layers to apply it to, and a width aimed at a source with none is an error: asking a single matrix how wide its hidden layers should be has no answer, and silence would read as one.
- **A scalar response is scored against its codomain's event axis.** A conditional family's parameters for a `d`-dimensional codomain arrive as `(N, d)`. An `(N,)` response against a `d = 1` codomain, which is the shape the response placeholder implies, broadcast to `(N, N)`, and because the observe step sums its score the result was a finite log-joint wrong by a factor of `N`. The axis is restored, and a response that cannot carry the codomain's event shape is rejected rather than broadcast into one. An inline family keeps the plate layout its parameters already align with.
- **An unsupplied `via` fibration index names itself.** A grouped `marginalize` reads its index from the observations dict like any covariate, so omitting it is a user error; it surfaced as a bare `KeyError`.
- **`quivers.__version__` reads the installed distribution's metadata.** It was a literal, and it had drifted a release behind.

### Documentation

- **The Bayesian neural network is a neural network.** The example composed four latent morphisms under an algebra whose composition is matrix multiplication, so with no pointwise nonlinearity the chain collapsed to a single linear map; its fit drove a learnable input morphism at pure noise and reported a residual worse than predicting zero, its NUTS block named an `x` and an `observations` no earlier block defined, over a composite carrying no likelihood to sample, and its walkthrough placed matrix-normal priors through a syntax that does not parse. It is now a Bayesian MLP for nonlinear regression whose conditional-Normal kernel emits both a mean and a log-scale, fit to a sine wave that no linear model can represent and scored against the best least-squares line under a matching Gaussian.
- **The Gaussian mixture is scored per row.** The example grouped a per-row component assignment under a `marginalize` block keyed by a fibration index, which read as an item-grouped mixture and carried a discrete latent no guide could biject. `MixtureNormal` integrates the assignment out in closed form.
- **`ParamSource` is documented.** Its eight sources and the `[param_source=...]` option had no API page, despite being where a kernel's dependence on its input is computed.

## [0.15.0] - 2026-07-02

### Changed

#### Grammar surface

- **Optional option blocks and a default role.** `morphism` and `contraction` declarations no longer require an option block, and a morphism without `role=` is a kernel. The other roles (`latent`, `observed`, `embed`, `discretize`, `let`) remain explicit. `morphism f : A -> B` is now a complete declaration.
- **Brace-delimited constructor options.** Continuous-space constructors take keyword options in braces: `Real 1 {low=-1.0, high=1.0}`. A trailing `[...]` always belongs to the enclosing declaration, so the constructor-versus-declaration attachment of an option block is decided by syntax rather than by parse-order luck, and the grammar is now conflict-free.
- **Signed and scientific numeric literals.** Option values, option-list items, option-call arguments, and constructor keyword values accept signed numbers; floats admit trailing-dot (`1.`), leading-dot (`.5`), and exponent (`1e-3`, `2.5e-3`) forms.
- **One turnstile.** Top-level `rule` declarations state their conclusion with `|-` (or `⊢`), the same marker deduction rules use.
- **Homogenized declaration headers.** `bundle NAME : [...]` and `decoder NAME : SIG` use the same `:` connective as every other declaration; the composition level moves into the option block (`composition NAME [level=algebra]`).
- **`define` binds morphism expressions.** The top-level value binding is `define NAME = EXPR`, with `where` blocks of nested defines scoped to the binding; the program-step `let` binds tensor arithmetic. The two binding forms no longer share a keyword.
- **Plural-name declarations.** `object A, B : V`, `morphism f, g : A -> B`, and lexicon entries `"a", "an" : CAT = LF` declare one item per name with shared shape and independent parameters, matching the existing `category` list form.
- **Parenthesized variable patterns.** `sample (a, b) <- f` and `return (a, b)` share one tuple shape; `observe` accepts the same pattern and reports a clear arity error for tuples, which its runtime does not support.
- **Keyword-led encoder rules.** Encoder constructor rewrites carry a leading `op` (`op App(fun, arg) |-> ...`), so an operator named `dim` or `init` cannot shadow the sibling entry keywords.
- **Composition operators.** Sequential composition is `>>` and `<<` (the same pipeline written right to left), transformation composition is `>>>`, and the tensor product is `@`. The algebra-tagged operator family (`>=>`, `*>`, `~>`, `||>`, `?>`, `&&>`, `+>`, `$>`, `%>`) is gone; algebra-tagged composition is expressed with `.change_base(...)` or a `composition` declaration.
- **Migration.** `qvr migrate --from v0.14.0 --to v0.15.0` rewrites sources across all of the above with byte-preserving span edits; the hop is registered on the migration chain with full coverage of the removed rules, and every repository `.qvr` file and fenced doc block is migrated. When a source cannot be migrated (a removed compose operator has no rewrite), the CLI reports the location and exits non-zero rather than raising.

#### Packaging

- **The QVR grammar ships vendored.** Parsing goes through the `qvr` grammar bundled in `panproto-grammars-all`; the floor moves to `panproto>=0.58.0` and `panproto-grammars-all>=0.58.0`, which vendor the current surface and surface tree-sitter's inserted (MISSING) tokens to the walker.

#### Diagnostics

- **Malformed input is rejected, never reinterpreted.** Parsing fails loudly on any damaged span anywhere in the tree, with the innermost offending token's line, column, and source snippet. Inputs that previously parsed to a silently different model, such as `[low=-1.0]` dropping its sign, `[scale=.5]` reading as `5.0`, a mis-bracketed `sample` step vanishing from the program, or a declaration truncated by an unbalanced `{` or `[`, now raise `ParseError` at the exact position.
- **Closed option-key sets with suggestions.** Every declaration and step kind validates its option keys; an unknown key raises at the entry's own position with a did-you-mean suggestion and the valid set, and a constructor key such as `low` on a declaration adds the hint to attach it with braces on the codomain.
- **Named failure for a missing `return`.** A program body without a return step reports that directly instead of leaking an internal registry message, and a `Program` holding only parametric templates raises a typed error naming the template and its instantiation call when `.domain` is touched.

### Added

- **A complete source emitter.** [`module_to_source`](https://quivers.dev/api/dsl/emit) covers every AST node kind, and a round-trip suite asserts that parsing, emitting, and re-parsing reaches a byte-identical canonical fixed point over every repository `.qvr` file and fenced doc block. LSP formatting uses it directly.
- **Four gallery examples with end-to-end tests.** A schema-bundled categorial chart parser (`schema`, `bundle`, `parser(...)`, `chart_fold(...)`), a bilinear tensor contraction (`contraction` with operadic three-way wiring), a term autoencoder (`signature`, `encoder`, `decoder`, `loss`), and parametric partial pooling (typed program parameters, labeled return tuples, `score` steps, `export` selection). Tests also cover file-loaded and plural-word lexicons, `define ... where`, doc comments, `.curry_left`/`.curry_right`/`.trace`, and `from_data` tensors flowing through inference.
- **A diagnostics regression suite** pinning the rejected-input catalogue and message quality, including line and column accuracy.

- **Editor and highlighting support.** The tree-sitter corpus (fifty cases), the VS Code and Zed extensions, and the Pygments lexer that renders `qvr` blocks in the docs all track the current surface.

### Fixed

- **`parser(...)` and `chart_fold(...)` keyword arguments.** The walkers read argument keywords from anonymous-token field constraints; every argument was previously dropped, so no surface form of either expression could compile.
- **`<<` composition** compiles as the reversed pipeline of `>>`, and reverse chains expand right to left in the transpiler.
- **Cyclic deductions converge in linearly many agenda pops.** The FIFO agenda merges contributions pushed for an item that is already pending via the semiring's plus, so a contractive cycle reaches its `tolerance` fixed point with at most one queue entry per item per wavefront; previously every contribution was enqueued separately and the pending-entry count grew geometrically with derivation depth.
- **`bounded` rule weights carry a joint sub-stochastic cap.** Each bounded rule's per-firing factor is strictly below one over the count of the deduction's bounded rules, so the total mass an item can push through them stays below 1 and interlocking cycles (e.g. an introduction / elimination pair over nested constructors) stay contractive for every parameter value; the previous per-rule cap of 1 left such systems free to diverge under fitting.

## [0.14.1] - 2026-07-01

### Fixed

- **v0.11.0 -> v0.14.0 grammar migration hop.** The `CHAIN` in `src/quivers/cli/migrations/__init__.py` now includes `"v0.14.0"`, with the accompanying hop module `v0_11_0_to_v0_14_0.py` and a matching entry in `MIGRATORS` and `COVERAGE`. `qvr migrate --from v0.11.0 --to v0.14.0` composes cleanly; `qvr migrate --check` reports `added_rules: family_call_arg, list_arg` and `uncovered_removed: []` for the pair. The hop is byte-identity: v0.14.0 adds two productions (`family_call_arg`, `list_arg`) to the `_draw_arg` choice as strict grammar extensions, so every source that parses under v0.11.0 also parses under v0.14.0. The `grammars/qvr/vcs/parsers/v0.14.0/` tree-sitter parser snapshot and the panproto VCS object store under `grammars/qvr/vcs/.panproto/` are rebuilt via `python grammars/qvr/vcs/build_schemas.py --reset` and `python grammars/qvr/vcs/build_parsers.py`.

## [0.14.0] - 2026-06-30

### Added

#### Distribution families

- **`OrderedLogistic` distribution + DSL inline observe surface.** PyTorch ships no `OrderedLogistic`. The hand-rolled [`OrderedLogistic`](https://quivers.dev/api/continuous/_ordered) implements the cumulative-link form $P(Y=k \mid \eta, c) = \sigma(c_k - \eta) - \sigma(c_{k-1} - \eta)$ via sigmoid-difference probabilities gathered at the observed category. Broadcasting handles three cutpoint shapes uniformly: shared `(K-1,)`, per-row `(batch, K-1)`, and arbitrary leading batch dimensions. Wired into the inline path via `_FAMILY_BUILDERS["OrderedLogistic"]` with a `(predictor, cutpoints)` param schema and `_FAMILY_SUPPORTS["OrderedLogistic"] = nonnegative_integer`. The canonical ordinal-mixed-model program (per-participant cutpoints gathered through a participant index) compiles and traces:

      program ord : Resp -> Resp
          sample eta <- Normal(0.0, 1.0)
          let row_cuts = cutpoints[participant_idx]
          observe y : Resp <- OrderedLogistic(eta, row_cuts)
          return y

  The host supplies `cutpoints` as a `(num_participants, K-1)` tensor and `participant_idx` as a per-row long tensor; the inline log-prob broadcasts the gathered per-row cutpoints against the predictor.
- **`ZeroInflatedPoisson` and `HurdlePoisson` distributions.** Two count families brms / Stan users routinely reach for, neither shipped by `torch.distributions`. [`ZeroInflatedPoisson(pi, lambda)`](https://quivers.dev/api/continuous/_zip_hurdle) implements the mixture $P(Y=0) = \pi + (1-\pi)\,e^{-\lambda}$, $P(Y=k) = (1-\pi)\,\text{Poisson}(k\mid\lambda)$ for $k > 0$. [`HurdlePoisson(pi, lambda)`](https://quivers.dev/api/continuous/_zip_hurdle) implements the two-stage hurdle $P(Y=0) = \pi$, $P(Y=k\mid Y>0) = \text{Poisson}(k\mid\lambda)/(1 - e^{-\lambda})$ for $k > 0$. Both ship `log_prob`, `sample`, and `mean` over arbitrary batch shapes; both register inline and conditional (`ConditionalZeroInflatedPoisson`, `ConditionalHurdlePoisson` over `MLP(x) -> (sigmoid pi, softplus rate)`).
- **`MixtureNormal` distribution.** Finite Gaussian-mixture with per-row weights, locations, and scales. `log_prob` uses `torch.logsumexp` over `log w_k + log Normal(loc_k, scale_k)`, sample dispatches through a Categorical / Normal pair. Conditional path `ConditionalMixtureNormal(num_components=K, hidden_dim=...)` parameterises all three slots through a single MLP head with softmax / identity / softplus on the appropriate slice.
- **`Poisson`, `NegativeBinomial`, `Binomial` registered as conditional families.** `ConditionalPoisson`, `ConditionalNegativeBinomial`, `ConditionalBinomial` are now wired into [`_FAMILY_REGISTRY`](https://quivers.dev/api/dsl/compiler/_prelude); `morphism f : A -> B [role=kernel] ~ Poisson` and the like compile. The formula frontend's `family="poisson" | "negative_binomial" | "binomial"` paths route through them cleanly, and matching `zero_inflated_poisson`, `hurdle_poisson`, and `mixture` entries land in [`quivers.formulas.family.families`](https://quivers.dev/api/formulas/family).

#### Compositional measure algebra

- **Compositional measure algebra at the DSL and Python surfaces.** Distribution families are built by composing five primitive operators from the (sub-)Giry monad rather than by enumerating a Cartesian product of (operator x base family):
  - `PointMass(x)` — Dirac measure at `x`, the unit $\eta$ of the Giry monad.
  - `Restrict(D, low, high)` — restriction of `D` to `[low, high]`, the sub-Giry monad's natural operation. Does not renormalise.
  - `Pushforward(D, b)` — pushforward through a `Bijector`, the Giry monad's functoriality on measurable isomorphisms.
  - `Mixture(weights, components)` — n-ary convex combination, the unique algebra structure on the Giry monad's Eilenberg-Moore category.
  - `Independent(D, n)` — declare the last `n` batch dims as event dims (strong monoidal product).
  - `Normalize(D)` — collapse a sub-measure to a probability measure; right adjoint to inclusion.

  Lazy normalisation: every operator returns a `Measure` whose symbolic `log_normalizer()` propagates through composition, and `Normalize` (or the implicit observe / sample boundary) collapses it. The discipline is the [partial Markov categories axiomatisation](https://arxiv.org/abs/2502.03477) made operational; the closure-under-composition story is the [Hakaru](https://hakaru-dev.github.io/lang/rand/) / [MeasureTheory.jl](https://arxiv.org/abs/2110.00602) / [Scibior et al. 2018](https://doi.org/10.1145/3236778) insight that unweighted measures provide the universal type.

- **`Bijector` library** at `quivers.continuous.bijectors`. Each bijector exposes `forward` / `inverse` / `forward_log_det_jacobian` / `inverse_log_det_jacobian`, all in log space and stable in tails. `Identity`, `Exp`, `Log`, `Sigmoid`, `Logit`, `Softplus`, `Affine`, `StickBreaking`, `Compose`, `Inverse`. Engineering pattern follows [TensorFlow Probability's `Bijector`](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector).

- **QVR grammar extension** for distribution-valued draw args. The `_draw_arg` rule grows two productions:
  - `family_call_arg` for nested `Family(...)` expressions:

        observe y <- Mixture([0.3, 0.7], [PointMass(0.0), Poisson(2.0)])
        observe y <- Restrict(Normal(0.0, 1.0), 0.0, 1.0)
        observe y <- Pushforward(Normal(0.0, 1.0), Exp)

  - `list_arg` for `[item, item, ...]` literals in mixture weights and component lists.

  The parser walks these to new tagged `DrawArg` AST variants (`DrawArgDist`, `DrawArgList`, `DrawArgName`, `DrawArgScalar`) so the compiler recurses on call-shaped args and builds the inner measure before passing it to the outer family's constructor.

- **Compiler sugar table** at `quivers.dsl.compiler.sugar` that desugars the brms-style named families to the canonical operator form at parse time when every argument is a constant literal:
  - `TruncatedNormal(μ, σ, a, b)` → `Restrict(Normal(μ, σ), a, b)`
  - `HalfNormal(σ)` → `Restrict(Normal(0, σ), 0)`
  - `HalfCauchy(σ)` → `Restrict(Cauchy(0, σ), 0)`
  - `HalfLaplace(σ)` → `Restrict(Laplace(0, σ), 0)`
  - `HalfStudentT(ν, σ)` → `Restrict(StudentT(ν, 0, σ), 0)`

  Sugar calls with free-variable arguments route through their dedicated inline family entries (`ZeroInflatedPoisson`, `HurdlePoisson`, `MixtureNormal`, etc.), which compose the same `Mixture` / `Restrict` / `PointMass` operators internally. Source can be either form; the compiler canonicalises to the operator form when it can and the pretty-printer re-sugars on the way out.

- **Four compiler rewrite rules** on `Mixture`, each backed by a literature result:
  - `Mixture.flatten()` — Giry monad associativity ($\mathrm{Mix}(\alpha, \mathrm{Mix}(\beta, A, B), C) \equiv \mathrm{Mix}(\alpha\beta, \alpha(1-\beta), 1-\alpha; A, B, C)$). Closes the [PyMC nested-mixture gap](https://github.com/pymc-devs/pymc/issues/5533).
  - `Mixture.pushforward_inside(b)` — functoriality of the Giry monad on isos: $g_*(\sum_k \pi_k\,\mu_k) = \sum_k \pi_k\,g_*\mu_k$.
  - `Mixture.restrict_to(low, high)` — the correct Mixture-Restrict non-commutation identity, reweighting by per-component truncation mass and wrapping the per-component restrictions in `Normalize`. Surfaces the modelling distinction [Welsh et al. 1996](https://doi.org/10.1016/0304-3800(95)00113-1) flag at the operator level.
  - `Mixture.lift_point_masses()` — surfaces `PointMass` components as the Bernoulli-style branch the ZIP / hurdle canonical factorisation expects (the [Lambert 1992](https://doi.org/10.2307/1269547) / [Mullahy 1986](https://doi.org/10.1016/0304-4076(86)90002-3) shape derived from the operator combination rather than baked per family).

#### Pluggable parameter sources

- **`ParamSource` ABC** at `quivers.continuous.param_source`. Every `ConditionalX` family accepts a `param_source=` override in place of the default two-layer MLP. Concrete sources: `LinearSource` (matches the transpile backends' single-matmul emit exactly), `MLPSource(hidden_dims, activation)` (the parameterised default), `LookupSource(n_entries, param_dim)` (discrete-domain table), `EmbeddingSource(n_entries, embed_dim, head)` (embedding + downstream head), `AttentionSource(num_heads)` (self-attention over the input dimension), `IdentitySource` (parameters supplied as data), `FunctionSource(fn, param_dim)` (wraps any `nn.Module` or callable), `ComposeSource(outer, inner)` (categorical composition). The DSL surface reads `[param_source=<kind>[(...)]]` on `morphism` declarations:

      morphism trans : State -> State [role=kernel, param_source=linear] ~ Normal
      morphism trans : State -> State [role=kernel, param_source=mlp(64, 64)] ~ Normal
      morphism attn  : Token -> Hidden [role=kernel, param_source=attention(heads=4)] ~ Normal

- **Parameter transforms unified with the bijector library.** `quivers.continuous.param_transforms` ships `TRANSFORM_TO_BIJECTOR` and `INLINE_CLAMP_TO_BIJECTOR` registries mapping every historical string key (`"id"`, `"sigmoid"`, `"softplus"`, `"softplus_shifted"`, `"exp"`) to a `Bijector` instance. `ParamSpec.transform` accepts a string key or a `Bijector` directly. `_make_family` routes raw parameter tensors through `bijector.forward`, so `inverse`, `forward_log_det_jacobian`, and `inverse_log_det_jacobian` are available on the guide and pushforward paths.

#### Variational objectives

- Three new objectives on the `Objective` ABC. Each is a subclass alongside `ELBO`, `IWAEBound`, `RenyiBound`, `VRIWAEBound`; each accepts the same `GradientEstimator` strategy attribute as the existing objectives.
    - `ChiVI` minimises the chi-squared upper bound on `log p(y)` via the CUBO surrogate ([Dieng et al. 2017](https://doi.org/10.48550/arXiv.1611.00328)). Useful for posterior calibration and for sandwich estimates with the ELBO.
    - `RWS` implements reweighted wake-sleep with a wake-theta phase on the model and a wake-phi phase on the guide ([Bornschein and Bengio 2015](https://doi.org/10.48550/arXiv.1406.2751)). Handles discrete latents where the reparameterization trick does not apply.
    - `DReGsBound` pairs the IWAE bound with the doubly-reparameterised gradient surrogate ([Tucker et al. 2019](https://doi.org/10.48550/arXiv.1810.04152)). Removes the score-function term whose signal-to-noise ratio collapses with `K` in naive reparameterised IWAE. Distinct from the `DoublyReparameterized` gradient-estimator strategy: the estimator is the scalar gradient rule, `DReGsBound` is the (bound + estimator) pair for callers that want to switch both together.

#### Autoguide combinators

- `AutoGuideList` concatenates disjoint per-block guides so a single model can hold, for instance, `AutoNormal` on local variables and `AutoMultivariateNormal` on a small global block.
- `AutoStructured` admits per-site conditional choice (delta / normal / mvn) plus per-edge dependencies (linear affine or a user-supplied callable) walked in ancestral order.
- Pyro-style short-name aliases (`AutoNormal`, `AutoMultivariateNormal`, `AutoLowRankMVN`, `AutoDelta`, `AutoLaplace`, `AutoIAFNormal`) re-exported from `quivers.inference.guides` as thin aliases for the pre-existing `Auto*Guide` classes.

#### Algebraic effect handlers

- `quivers.effects` package: `EffectHandler` ABC with a thread-local handler stack and three-phase `apply_stack(msg, default=...)` dispatch, plus `run_program` that walks a `MonadicProgram` through the active stack. The design mirrors the Pyro `poutine` / NumPyro `handlers` `Messenger` protocol ([Plotkin and Pretnar 2009](https://doi.org/10.1007/978-3-642-00590-9_7), [Scibior et al. 2018](https://doi.org/10.1145/3236778)).
- Handlers: `TraceHandler`, `clamp(data)` (poutine-style pin of sample sites; the name avoids the collision with the top-level `condition` factory that returns a `Conditioned` model wrapper), `do(data)` (Pearl intervention), `mask(mask_tensor)`, `scale(factor)`, `block(names)`, `replay(trace)`, `lift(prior_scale)`, `collapse()`. The handlers compose: `clamp + scale + mask` is the subsampled-mini-batch likelihood pattern; `do + trace` is intervention-then-record.
- The `EffectHandler` ABC is distinct from `quivers.monadic.algebraic.Handler`. `EffectHandler` is a mutable-message dispatcher for runtime interception of a `MonadicProgram`; `monadic.algebraic.Handler` is a free-monad-over-signature interpreter that folds a bounded-depth signature tree into a target monad. Different territories; both remain.

- **Reparameterisation strategies** at `quivers.effects.reparam` wrap sample sites and rewrite their distributions: `LocScaleReparam` for the centred / non-centred rewrite of Normal(loc, scale) sites, `TransformReparam(bijector)` driving change-of-variables through the bijector library, `NeuTraReparam(autoguide)` warping HMC geometry through a trained `AutoIAFGuide`, `ConjugateReparam` for analytic conjugate collapse. The `reparam({name: strategy})` orchestrator wires per-site strategies into the effect stack:

      with reparam({"theta": LocScaleReparam(), "z": NeuTraReparam(guide)}):
          samples = nuts.run(model, x, observations)

#### Diagnostics for any fit type

- **`to_datatree_from_svi`** and **`to_datatree_any`** at `quivers.diagnostics` dispatch on fit type. Every ArviZ downstream (`loo`, `waic`, `plot_trace`, `compare`) works uniformly on `MCMCResult`, `Guide` / `Predictive` draws, or `(samples, log_densities)` tuples.

#### Extensible transpile IR

- **`quivers.transpile`** ships the target-agnostic scaffold: the `IRNode` tagged-union IR with `ConstraintSpec` taxonomy, `IRArg` variants, `Dim` / `Plate` shape metadata; the `Lower` AST-to-IR mapping; the `RendererBase` shared machinery; the `FAMILY_META` per-family transpile metadata registry; and the `renderer_registry` emit-hook table. Target-specific backend renderers live on their own branches; they register per-`(backend, IRNode)` emits via `@emit_hook(backend_name, node_type)`.

### Changed

- **`SampleStep.args` / `ObserveStep.args` type signatures** widen from `tuple[str | float, ...]` to `tuple[DrawArg, ...]` over the new tagged AST union. The compiler's inline path unwraps `DrawArgName` -> `str` and `DrawArgScalar` -> `float` at entry so the existing arg-shape machinery continues to handle them; `DrawArgDist` and `DrawArgList` are recognised by the operator-family dispatch.

- **`ZeroInflatedPoisson`, `HurdlePoisson`, `MixtureNormal` internals** route through the measure algebra: each user-facing class delegates `log_prob` / `sample` to a `Mixture` of `PointMass` and (for hurdle) `Normalize(Restrict(...))`. The user-facing API is unchanged; the renderer-side IR sees a single Mixture-with-PointMass shape rather than per-family hand-rolled emit logic.

- **`_make_source`** in `quivers.continuous.morphisms` accepts a `param_source=` override; every family that constructs its source through `_make_source` gains the same kwarg plus a `param_source_option=` string form.

- **`quivers.inference.trace.trace`** becomes a thin wrapper that pushes a `TraceHandler` and delegates to `run_program`. The API is unchanged.

### Fixed

- **`Restrict.log_normalizer()` for continuous bases** uses `cdf(low)` for continuous bases and `cdf(low - 1)` (or the survival-sum path for distributions without a `cdf` method) for integer-supported bases, matching the standard truncation-mass formula for each support type.

- **`HurdlePoisson` log-density** handles the survival-sum formula for `Poisson`-restricted-to-`{1, 2, ...}` via `1 - e^{-\lambda}` plus a general pmf-sum for arbitrary integer intervals. Works around `torch.distributions.Poisson`'s absent `cdf`.

- **`OrderedLogistic.sample` accepts any `Sequence[int]` for `sample_shape`.** `torch.distributions.Distribution.sample`'s contract admits tuple, list, or `torch.Size`; `sample_shape` is coerced through `torch.Size` at the top of the method so every shape form works.

- **Inline DSL `observe ... <- OrderedLogistic(predictor, cutpoints)` routes a shared cutpoints vector as a distribution parameter.** `quivers.continuous.inline` ships a `_PARAM_EVENT_RANKS` table declaring each family parameter's event rank (0 = per-row scalar, 1 = shared-or-per-row vector). `_resolve_input` reads the table through the wrapping `VectorisedObserve._param_event_ranks` property and broadcasts rank-1 shared vectors to per-row shape before stacking; `MixedInlineDistribution` takes a `param_event_ranks` argument and lets the trailing vector-typed slot consume the remaining columns of the stacked input. Covers `OrderedLogistic(predictor:(N,), cutpoints:(K-1,) or (N,K-1))` plus `MixtureNormal`, `ZeroInflatedPoisson`, and `HurdlePoisson`.

- **DSL quick-start docstring uses grammar keywords.** [`quivers.dsl`](https://quivers.dev/api/dsl)'s module docstring's `loads(...)` example compiles against the canonical `KIND NAME : SIGNATURE` surface (`program NAME : A -> B` with `sample` / `observe` / `return` steps and a top-level `export NAME`), no `latent` / `output` / `alias` (none of which are grammar keywords).

- **REPL/LSP guide examples use the current morphism surface.** [`docs/guides/repl-and-lsp.md`](https://quivers.dev/guides/repl-and-lsp) shows `morphism NAME : ... [role=latent]` for latent-role morphisms and `object NAME : FinSet N` for finite-set declarations.

## [0.13.0] - 2026-05-22

### Added

- **GHCi-style `:type` and `:kind` commands in the REPL.** `:type` accepts an expression (morphism, program, deduction, scoped sample / observe / let / marginalize / return site, contraction, rule, encoder / decoder / loss, bundle, transformation) and prints the GHCi-shaped one-line signature `name :: A -> B` (or `name :: (Real, Real) => Word -> Word` for parametric programs, with the Π-bound parameter universes in a Haskell-style constraint context). `:kind` accepts a type-level binding (object, space, signature, category) or a bare type expression (`FinSet 3`, `A * B`) and prints the kind. Each command rejects the wrong sort with a redirect: `:type Doc` → "Doc is a type, not an expression; use :kind Doc". The bare-expression fallback tries `:type` first, then `:kind`, so users can keep typing a name on its own. Aliases `:t` and `:k` work as before.
- **User-given type aliases preserved across the REPL.** The compiler internally resolves `object Item : FinSet 200` to `FinSet(name='_FinSet_200', cardinality=200)`. The REPL now builds an `id(obj) → user_name` map per session and renders morphism signatures through it, so `:type X` prints `X :: Item -> H_in` rather than `X :: FinSet 200 -> FinSet 4`. The map recurses into product / coproduct components, so a mixed `Doc * Topic` still surfaces both user names.
- **`render_signature(compiler, name)` shared helper in `quivers.cli.repl_session`.** Single entry point for the GHCi-style signature line, used by both the REPL's `:type` / `:kind` / bare-expression fallback and the LSP's hover panel — so a binding looks the same in the TUI and in an editor.
- **LSP hover leads with a `**Type**` / `**Kind**` block.** Hover now shows the GHCi signature first (same string `:type` / `:kind` would print), then the doc comment, then the verbatim QVR source slice, then the collapsed AST. Editor users get the one-glance signature that REPL users already had.
- **New semantics chapter: [`docs/semantics/typing.md`](https://quivers.dev/semantics/typing).** Chapter 8 in the semantics development, presenting the QVR type system as a proof calculus: syntactic categories (kinds, type expressions, families, term expressions, statements, programs); contexts ($\Gamma$, $\Delta$, $\Phi$); judgments ($\Gamma \vdash \tau : \kappa$, $\Gamma; \Phi \vdash e : A \rightsquigarrow B$, $\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B]$, $\Gamma; \Phi \vdash s \dashv \Phi'$, $\Gamma \vdash p : (\Delta) \Rightarrow A \rightsquigarrow B$); the full inference-rule set; the four structural lemmas (Weakening, Substitution, Type Uniqueness, Inversion); the Soundness theorem against the denotation with proof sketches for the representative Compose / Bind / Marginalize / Prog cases; the Subject-Reduction theorem for parametric instantiation; a Completeness theorem (every phrase with a defined raw denotation has a derivation) with a Decidable-Type-Membership corollary on the non-residuated fragment; the bidirectional algorithm underlying the compiler's typechecker; and a worked example deriving the LDA program's full judgment.

### Changed

- **Constraint context in parametric programs drops binder names.** A program with typed parameters renders as `lda :: (Real, Real) => Word -> Word`, not `lda :: (alpha : Real, beta : Real) => Word -> Word` — Haskell convention. Parameter names are accessible through `:info lda`.
- **`:type` of a parametric program no longer curries the parameters with the kernel arrow.** Per `docs/semantics/programs.md §3a`, typed parameters denote a dependent family $\prod_{p:P} \mathbf{Kern}(\mathrm{dom}, \mathrm{cod})$ — they live in a $\Pi$, not in the kernel's domain. Rendering them as `Real -> Real -> Word -> Word` (the previous "GHCi-curried" form) conflates the $\Pi$-context with the kernel arrow.
- **Compiler-driven expression typing in `:type`.** The expression-fallback path now drives `Compiler._compile_expr` directly on the already-compiled session state instead of re-running the full module compile for every `:type` query.
- **Auto-generated placeholder names stripped from output.** The compiler assigns synthetic names like `_FinSet_20` and `_Real_8` to anonymous FinSets and Euclidean spaces. They no longer leak into `:type` / `:kind` output as `Doc : _FinSet_20`; the stripper renders them as `Doc : FinSet 20`.
- **Help-text rewrite for `:type` and `:kind`.** Both commands' inline `:help` entries and detailed `:help :type` / `:help :kind` text now match the GHCi-style semantics.

### Documentation

- **`docs/semantics/typing.md`** added as chapter 8; index renumbered.
- **Semantics corpus audit (substantive).**  After threading `typing.md` into the development:
  - Cross-link `setting.md §6`, `morphisms.md`, and `programs.md` into the new typing chapter at their natural touch-points.
  - `morphisms.md §1.1` gets a "Categorical structure" proposition under the strict-quantale hypothesis, proving $\mathcal{V}\text{-}\mathbf{Rel}$ is a symmetric monoidal category using the join's universal colimit property, $\otimes$ associativity, $\bot$ absorption, and the distributivity law; flags the lax cases of $\mathcal{V}_{\mathrm{pf}}$ / $\mathcal{V}_{\mathrm{L}}$ explicitly.
  - `programs.md §5` upgrades the monad-law table to a "Monad laws for QVR programs" theorem with proofs derived from the Giry monad's universal property (Giry 1982, Kock 1972, Fritz 2020). The commutativity equation is now stated with an explicit double-strength pair $\mathrm{dst}_1$ / $\mathrm{dst}_2$ rather than the previous ambiguous one-line formula, identifying Fubini–Tonelli as its substance.
  - `adequacy.md §3.4` upgrades the one-liner "the proofs are straightforward" to a full "Adequacy of `@`" proposition with term-by-term unfolding; the remaining five tensor combinators get a table whose every row makes the proof template explicit.
  - `effects.md §4a` adds explicit inference rules for the four chart firings (`Base`, `Lift_T`, `Handle_{T→S}`, `Eliminate_T`, `Swap_{T|U}`), turning the previously prose-only joint type-and-effect dispatch into a proof system.
  - Stale class-name references across the corpus updated: `TypeExpr` → `ObjectExpr`, `TypeProduct` → `ObjectProduct`, `TypeEffectApply` → `ObjectEffectApply` (the AST class was renamed and the docs had not tracked it).
  - `adequacy.md §3.7` references the real method names (`_compile_program`, not `_compile_program_body`; dispatch key `compose`, not `compose_deductions`).
  - `adequacy.md §4`: drop references to `tests/test_resolution_lenses.py`, which no longer exists.
- **British → American spelling sweep across all 14 semantics chapters.** marginalisation → marginalization, realise → realize, behaviour → behavior, catalogue → catalog, modelling → modeling, analyse → analyze, etc.

### Fixed

- **`Compiler.signatures`, `Compiler.categories` now exposed to the REPL's bare-name lookup.** Previously, `:type SomeSignature` returned "unknown name" because signatures and categories were not in either the value-level or type-level buckets the REPL consulted. They are now classified as type-level (a signature denotes a generalised algebraic theory; a category denotes an atom universe) and respond to `:kind`.
- **Contraction and rule signatures render correctly under `:type`.** Contractions get an operadic `c :: (A_1, …, A_k) -> B` signature; rules get the `r :: prem |- concl` schema form.

## [0.12.0] - 2026-05-21

### Added

- **`::` scope paths everywhere a binding name is taken.** `:info`, `:type`, `:doc`, `:browse`, `:plate`, `:graph`, `:where`, `:effects`, `:shape` all accept a `::`-separated scope path. The leftmost segment is a top-level binding; each subsequent segment looks up a named child in that binding's scope. Coverage across every container kind: `program` (typed parameters + sample / observe / let / marginalize / return steps; recursing into nested marginalize bodies), `deduction` (rules / atoms / lexicon entries), `signature` (sorts / constructors / binders / vertex_kinds / edge_kinds), `encoder` (op_rules / init_rules / message_rules / update_rules / var_inits), `decoder` (per-constructor heads), `bundle` (member rule names), `contraction` (input parameter names). Examples: `lda::theta` resolves to the sample site inside the lda program; `lda::z::w` to the observe site inside the lda program's marginalize block; `LF::Term` to the Term sort inside the LF signature; `CCG::fwd_app` to the forward-application rule inside the CCG deduction.
- **`quivers.analysis.scope` module.** New `ScopedRef` value type, `resolve_scoped_path(compiler, path)` resolver, `scope_children(ref)` view, `find_all_references(compiler, name)` reverse lookup. Foundation for every scope-aware REPL feature.
- **Env-tree click navigation on every nested node.** Each env-tree node now carries its full `::` path on `node.data`. Clicking any node (top-level binding, program step, deduction rule, signature sort, bundle member) fires `:info PATH` against that exact binding. Previously only top-level bindings were clickable; nested children either did nothing or raised "undefined morphism" errors when the click handler tried to dispatch against the rich signature label.
- **Scope-aware tab completion.** `lda::` completes against the lda program's children; `lda::z::` completes against the marginalize block's inner scope; a bare-name prefix like `thet` surfaces every scoped descendant whose final segment matches (`lda::theta`).
- **`quivers.analysis.plate_graph` module.** Extractor that walks a compiled QVR program and builds a `PlateGraph` capturing nodes (latent / observed / marginalized / deterministic), plates (with nested parents), and dependency edges. The grouping plate of a marginalize block propagates correctly to nested observes via the `[over=G]` axis; the `[via=idx]` fibration index is correctly treated as a routing label rather than an additional plate. Handles nested marginalize blocks recursively.
- **`:plate PROGRAM` meta-command** with five output formats: default Rich table (kind / variable / family / plates / parents), `--mermaid` (Mermaid `graph TD` source with `subgraph` clusters per plate), `--dot` (Graphviz DOT with `cluster_*` subgraphs), `--tikz` (LaTeX TikZ + bayesnet snippet), `--daft` (Python script using the `daft` library), and `--open` (renders via daft or graphviz to a temp PNG and launches the system default viewer; falls back to Mermaid source if no renderer is installed).
- **`:graph PROGRAM` meta-command.** Vertical step-flow view of a program: one row per sample / observe / let / marginalize / return step with its dependency parents on the side. No 2D edge routing in the in-TUI view so the output is robust on any terminal width. Same `--mermaid` / `--dot` / `--open` flags as `:plate`.
- **`:where NAME` meta-command.** Lists every scope path in the loaded module whose final segment is `NAME`, sorted by depth. Useful for finding cross-references between programs, deductions, and other declarations.
- **`:effects PROGRAM` meta-command.** Prints the program's declared `[effects=[...]]` set alongside the effect set inferred from the body (Sample / Score / Marginal / Pure based on the step kinds present). Flags leak (body uses effects not declared) and unused (declared but not used) divergences.
- **`:shape PROGRAM` meta-command.** Pretty-prints the program's [`ChainShape`](https://quivers.dev/api/analysis/chain_shape): per-step depth, kind, intermediate axis size, source location. Useful for auditing chain depth before fitting.
- **Modal `:help` dialog.** Promotes `:help` from "write to the output log" to a proper Textual `ModalScreen` overlay with its own focus, ESC-to-close, and a live filter input. Lists every meta-command grouped by category (loading / inspection / program exploration / live editing / control) plus the full keybinding table. The TUI's `F1` binding pushes the same modal.

### Changed

- **TUI env-tree builder unified.** Replaced the 12 per-kind `_children_for_*` builders with a single uniform walker that consults `scope_children()`. Each node's display label comes from a single per-kind type-line renderer; each node's `data` is its full scope path. Result: every container kind expands to its declared children in a single layered tree without duplicate per-kind logic.
- **`Compiler.compile_env()` returns every populated bucket.** Adds `programs`, `deductions`, `signatures`, `encoders`, `decoders`, `losses`, `bundles`, `contractions`, `transformations` to the union, so consumers that walk the env dict see every declared atom.

### Fixed

- **`Compiler.losses` and `compile_env()` called `reg.entries()` (parentheses).** `LossRegistry.entries` is a plain list attribute, not a method. The misplaced call broke every module that declared a `loss`. The accessors now read the attribute directly.

## [0.11.5] - 2026-05-21

### Fixed

- **`qvr repl` rendered every token in the same colour on customised terminal palettes.** The `STYLE_TABLE` used palette-indexed colour names (`"bold magenta"`, `"bold blue"`, `"bold cyan"`, `"default"`). Terminals whose palettes mapped those four into the same warm hue (common in custom themes) collapsed every span into a single colour, so `:info`'s body looked uniformly red. Switched the table to truecolor `#RRGGBB` codes (One-Dark inspired) so palette mapping does not apply and the keyword / type / operator / variable distinctions remain visible regardless of the user's terminal theme.
- **Clicking a program / deduction / signature / encoder / decoder node in the env tree raised a "compile error: undefined morphism" or did nothing at all.** The click handler read the node's display label (`lda(alpha : Real, beta : Real) : Word -> Word`) and dispatched `:info` against that whole string. After 0.11.3 added nested-children expansion, expandable nodes also stopped firing the handler entirely because of an `allow_expand` gate. The handler now reads the bare binding name from a dedicated `node.data` slot populated at tree-build time, and the `allow_expand` gate is gone, so a click on any named declaration fires `:info NAME` correctly.

### Added

- **Comprehensive TUI regression suite at `tests/cli/test_repl_tui.py`.** 20 tests covering: click-target resolution under every node shape (root / category heading / named binding / nested step / non-identifier data); the `STYLE_TABLE` truecolor invariant; each per-kind nested-children builder; the rendered ANSI for a real `:info` body (truecolor codes only, distinct codes per token class, keyword / type colour distinct); the full `:type` and `:browse` signature lines including typed parameter lists; the completion engine surfacing program names with the correct kind detail; and the TUI module's import-cleanliness on a minimal install.

## [0.11.4] - 2026-05-21

### Fixed

- **`:type` on a parametric program dropped its parameter list.** Typed program parameters (`alpha : Real`, `beta : Real`, `G : FinSet`, `k : Mor[A, B]`) live on the AST's `type_params` slot, not on `params` (which holds bare names). The REPL session and the TUI env-tree builder were both reading only `params`, so loading `lda.qvr` and running `:type lda` printed `program lda : Word -> Word` instead of `program lda(alpha : Real, beta : Real) : Word -> Word`; the env tree's program node had the same problem. Both call sites now walk both slots and render the typed parameter list with its `ScalarParam` / `ObjectParam` / `MorphismParam` annotation.

## [0.11.3] - 2026-05-21

### Fixed

- **REPL environment was missing eight declaration categories.** `qvr repl`, `:browse`, and the TUI env browser only surfaced `objects` / `spaces` / `morphisms` / `rules` — every other top-level declaration (`program`, `composition`, `deduction`, `signature`, `encoder`, `decoder`, `loss`, `bundle`, `contraction`) compiled successfully but was invisible at the user-facing surface. Loading [`lda.qvr`](https://quivers.dev/examples/lda) showed three objects and "0 programs" despite the declared `program lda`. The compiler now exposes public `@property` accessors (`programs`, `deductions`, `signatures`, `encoders`, `decoders`, `losses`, `bundles`, `contractions`, `transformations`) and `compile_env()` returns the full set; the REPL session, `:browse`, the completer, and the TUI status bar / env tree all surface every populated bucket.

### Added

- **Nested env-tree rendering.** The TUI env pane now expands structural declarations to their internal AST shape on click: a `program` opens to its `sample` / `observe` / `let` / `marginalize` / `return` step list (with `marginalize` recursively opening to its body); a `deduction` opens to its `rules` (each labelled with its `premises |- conclusion` line) plus `semiring` / `tolerance` metadata; a `signature` opens to its `sorts` / `constructors` / `binders` / `vertex_kinds` / `edge_kinds` tables; a `bundle` opens to its constituent rule names. The plain-text `:browse` meta-command renders the same hierarchy as indented text so plain-mode users (`qvr repl --plain`, Jupyter kernel) see the same structure the TUI shows.

## [0.11.2] - 2026-05-21

### Changed

- **`panproto>=0.48.4` and `panproto-grammars-all>=0.48.4`.** Upstream now vendors the homogenized QVR grammar (the unified `KIND NAME : SIGNATURE [k = v, ...] [~ INIT] [BODY]` surface, the `composition NAME as <level>` keyword, the `[role=...]` option-block roles, the indented-body deductions / signatures / encoders / decoders / marginalize blocks). The `QVR_USE_LOCAL_GRAMMAR` shim is no longer required for the runtime parser: a fresh wheel install parses every gallery example out of the box.

### Fixed

- **`qvr repl`, `qvr-lsp`, `qvr-kernel` parse-error on every `.qvr` load.** 0.11.0 / 0.11.1 against `panproto-grammars-all 0.48.3` (the latest at release time) used a pre-homogenization grammar whose `object_decl` node had no `init` field; the in-tree parser walker raised `ParseError: object_decl missing value` on the first homogenized declaration. With `panproto-grammars-all>=0.48.4` the vendored grammar matches the walker's expectations and every example in `docs/examples/source/` loads cleanly.

## [0.11.1] - 2026-05-20

### Fixed

- **Wheel shipped without `grammar.json`.** `_grammar_introspection` walks the tree-sitter `grammar.json` to derive the keyword / punctuation / builtin sets the Pygments lexer, the REPL completer, the TUI highlighter, and the LSP semantic-tokens legend all consume. The canonical file lives at `grammars/qvr/src/grammar.json` (regenerated by `tree-sitter generate`), which sits outside `src/quivers/` and so was excluded from the wheel build. The 0.11.0 install crashed on first import of `quivers.dsl.pygments_lexer` with `FileNotFoundError: grammars/qvr/src/grammar.json not found above ...`, which in turn blocked `qvr repl`, `qvr-lsp`, and `qvr-kernel`. A package-local copy at `src/quivers/dsl/_grammar_data/grammar.json` is now committed and included in the wheel; `_grammar_json_path()` prefers the package-local copy and falls back to the in-tree source for editable installs. `tools/sync_grammar_data.py` keeps the two copies in lockstep, with a `--check` mode CI can gate on.

## [0.11.0] - 2026-05-20

### Added

- **Homogenized DSL surface.** Every declaration follows the unified `KIND NAME : SIGNATURE [k = v, ...] [~ INIT] [BODY]` skeleton: `morphism`, `object`, `composition`, `program`, `deduction`, `signature`, `encoder`, `decoder`, `loss`, `category`, `bundle`, `schema`, `rule`, `let`, `export`. Per-role morphism keywords (`latent`, `kernel`, `observed`, `embed`, `discretize`) collapse into `morphism X : ... [role=ROLE]`. Brace-bodied declarations (`algebra X { ... }`, `deduction X : ... { ... }`, `signature X { sorts { ... } }`, `marginalize ... in { ... }`, `atoms { ... }`, `lexicon { ... }`) become indented bodies under the unified header. Effect signatures live in the option block as `[effects = [Sample, Score, Marginal, Pure]]` rather than `! E1, E2, ...`. Observation fibrations are `[via = idx]` and grouping plates are `[over = G, reduction = R]` in the option block instead of keyword clauses. Morphism replication is `[replicate = N]`. The pragma surface is `#[k = v, ...]` (outer, decorates the next declaration) and `#![k = v, ...]` (inner, decorates the enclosing module); per-line rule and lexicon pragmas (`#[learnable]`, `#[bounded]`, `parent = ...`) replace the legacy `@ learnable` marker.
- **`composition NAME as <level>` keyword.** A single surface for algebraic composition rules (algebra / semigroupoid / bilinear_form / rule), with either a registry lookup (`composition product_fuzzy as algebra`) or an indented inline body (`composition my_godel as algebra` followed by `tensor_op(a, b) = ...`).
- **`ContinuousSpace` family registry.** Eleven continuous-space constructors (`Real`, `Simplex`, `PositiveReals`, `UnitInterval`, `Sphere`, `Ball`, `Covariance`, `Correlation`, `CholeskyFactor`, `Orthogonal`, `Stiefel`, `LowerTriangular`, `Diagonal`) ship as `ContinuousSpace` subclasses dispatched by the homogenized `DiscreteConstructor` / `ContinuousConstructor` AST nodes. Discrete and continuous space initialisers share the same `object X : <init>` surface.
- **`quivers.inference.lifts` module.** Four general-purpose Bayesian-lift helpers:
  - [`bayesian_lift_parameters`](https://quivers.dev/api/inference/lifts) lifts every learnable `nn.Parameter` into a Normal-prior sample site, returning a [`MonadicProgram`](https://quivers.dev/api/continuous/programs) that SVI and NUTS consume uniformly. The new `additional_latents` argument lifts intermediate program latents as joint NUTS variables via an exact placeholder-cancellation construction; the lifted log-density equals `log p(theta) + log p_inner(z, y | x, theta)` pointwise, with no Monte-Carlo noise across leapfrog steps. The module docstring includes a methodological-notes section justifying the Normal-prior choice (maximum entropy on R^n; equivalence to L2 regularization; computational fit with NUTS) and the placeholder-cancellation theorem with proof.
  - [`lift_to_bayesian_program`](https://quivers.dev/api/inference/lifts) lifts a parameter-only morphism plus any `torch.distributions.Distribution` observation family into a Bayesian program, with a user-supplied `location_fn` so the same lift works for `rsample(x)`-style, `tensor`-attribute, and `prog(x)`-forward shapes.
  - [`lift_from_log_prob`](https://quivers.dev/api/inference/lifts) lifts a parameter-only model whose forward is already a `log_prob(x, y)` function (the VAE / encoder-decoder pattern).
  - [`monte_carlo_log_joint`](https://quivers.dev/api/inference/lifts) wraps a program so its `log_joint` MC-draws named intermediate `sample` sites and returns the conditional data likelihood, with the prior-cancellation correction applied automatically. Documented honestly as a single-sample MC estimator valid for SVI as a stochastic gradient estimator; the function's docstring notes that for rigorous NUTS the joint lift via `additional_latents` is the correct route.
- **`quivers.stochastic.deduction` package.** Reorganised the deduction-system surface from the previous flat `quivers.deduction_fit` module into a coherent package with one canonical import path. Subpackages: `primitives` (the abstract `Axiom`, `Deduction`, `Goal`, `Schedule`, `DeductiveSystem` building blocks), `fit` (point-estimate gradient fitting via `adam_fit_deduction`), `bayes` (posterior wrapping via `nuts_program_from_deduction`), `sample` (exact length-conditional forward sampling via `sample_corpus`). The model type `DeductionSystem` is re-exported from the package root alongside the operations and `bayesian_lift_parameters` (re-exported from `quivers.inference.lifts`).
- **`ScanMorphism.log_joint` accepts a dict signature.** The recurrent-cell scan morphism's `log_joint` now accepts the hidden-state trajectory either positionally as a tensor or via a `{state_key: tensor}` dict (default `state_key = "h"`), so the standard inference contract `log_joint(x, observations: dict)` works without an adapter at the call site.
- **`qvr migrate` subcommand.** Forward-migrates user `.qvr` source files across the QVR grammar release chain (v0.2.0 through HEAD). The CLI composes the adjacent-pair hop migrators in `quivers.cli.migrations` automatically; `--from VER` and `--to VER` pin the boundary, `--dry-run` reports without writing, `--output DIR` writes migrated copies. `--check` mode validates the migration chain against the panproto VCS schema diff and reports any rule removed in an adjacent grammar revision that the corresponding hop migrator has no converter for.
- **In-tree panproto VCS for the grammar.** `grammars/qvr/vcs/.panproto/` holds a content-addressed schema chain with one tagged commit per grammar release (`v0.2.0` through `v0.11.0`). `grammars/qvr/vcs/build_schemas.py` walks tagged grammar revisions, builds a `panproto.Schema` keyed by rule name, and commits each as a tagged VCS commit. The migration system uses these commits to derive coverage reports and detect uncovered removed-rule gaps in any adjacent-pair migrator.
- **Comprehensive `## Try it` sections in every example doc.** All 28 gallery examples (regressions, latent-variable models, state-space and time-series, language models, encoder-decoder, weighted deductions) ship a verified three-block runnable Try-it section: synthetic-data generation under known true parameters, an SVI fit, and a Bayesian posterior block. Every block executes in CI via `test_gallery_try_it_blocks_execute`. Illustrative-budget caveat at the top of each section explicitly notes the SVI / NUTS budgets are sized for tens-of-seconds CI runs, not production convergence.
- **Example walkthrough alignment.** All 28 example docs' `## Overview` and `## Walkthrough` prose audited against the actual `.qvr` source; the 13 docs with prose-vs-source drift (most severely `pcfg.md` and `montague-nli.md`, whose embedded `## QVR Source` blocks described entirely different models than the actual source) were rewritten in place.
- **Examples gallery index restructured** as a flat grouped list of one-line example descriptions linking to each gallery doc.
- **API reference structure expanded.** New pages for `api/inference/lifts`, `api/inference/mcmc`, `api/inference/elbo`, `api/continuous/scan`, `api/continuous/inline`, `api/stochastic/agenda`, `api/stochastic/deduction` (+ four subpages: `primitives`, `fit`, `bayes`, `sample`). The `qvr migrate`, `qvr repl`, and `qvr lsp` subcommands are documented in `api/cli.md`.
- **Denotational semantics extensions.** `docs/semantics/programs.md` gains a §7 "Bayesian lifts" with five propositions and proofs (placeholder-cancellation for the parameter lift, the observation-family lift, the log-prob lift, the single-sample MC bias bound via Jensen, the CRF-vs-PCFG distinction for `nuts_program_from_deduction`). `docs/semantics/adequacy.md` gains a §3.7b on adequacy of the lift theorems. `docs/semantics/types-and-spaces.md` §1 and §4 cover all 13 `ContinuousSpace` variants with set-builder denotations. `docs/semantics/grammar.md` rewrites `composition_rule_decl` EBNF to match the unified surface. `docs/semantics/expressions.md` §3.4 notes the dict-signature route on `ScanMorphism.log_joint`.
- **`qvr migrate` documentation.** `docs/guides/migration.md` covers the `--from` / `--to` / `--dry-run` / `--output` / `--check` flag set with worked examples.

### Changed

- **`quivers.deduction_fit` module removed.** Imports move to `quivers.stochastic.deduction` (or `quivers.inference.lifts` for `bayesian_lift_parameters`). Pre-1.0 clean break, no compat shim. The 20+ docs and tests that referenced the old path are updated.
- **Docstring cross-references converted to mkdocstrings autorefs.** All 1103 Sphinx-style `:func:`X`` / `:class:`X`` / `:mod:`X`` / `:meth:`X`` / `:attr:`X`` / `:data:`X`` / `:exc:`X`` role markers in source docstrings (132 files) are now either mkdocstrings autorefs (for internal symbols the docs build resolves) or plain inline code (for external / unresolved references). Sphinx role markers no longer render as ugly grey code spans in the API docs.
- **DSL surface drift across the entire test and doc corpus.** Every fixture, every embedded source string, every walkthrough prose reference is on the current homogenized surface. `mkdocs build --strict` produces zero warnings; `pytest tests/` reports 2009 passing, 140 skipped, zero failing.

### Fixed

- **Grammar parser-recovery bugs.** `encoder_decl` and `composition_decl` previously used `optional(body), $._newline`, which let the trailing `$._newline` swallow the next top-level statement after an indented body. Replaced with `choice(body, $._newline)` so the `$._dedent` properly terminates and the following statement parses. `parser.c`, `grammar.json`, and `node-types.json` regenerated.
- **`ScanMorphism.log_joint` dict-vs-positional adapter** unified into the morphism itself so call sites no longer need a `DictWrap` shim.
- **`_TruncNorm` distribution shim** in `quivers.continuous.inline` now carries `event_shape` and `batch_shape` attributes so the generic family-`log_prob` path treats it uniformly with stock `torch.distributions` families.
- **Formula compiler** (`quivers.formulas.compile`) emits `object Resp : FinSet N` via the homogenized `DiscreteConstructor` initialiser and accepts both old and new shapes on the reverse-pass decoder.
- **Dataset schema** (`quivers.data.schema`) emits `object X : FinSet N` in the prelude returned from `DatasetSchema.declarations_qvr` and the `compose` wrapper.
- **Compiler template-locals walker** (`quivers.dsl.compiler.programs._collect_template_local_names`) now walks the surface `SampleStep` / `ObserveStep` / `MarginalizeStep` nodes (was only walking the legacy `BindStep` IR), so locally-bound names in parametric program templates are correctly captured during alpha-renaming.
- **Compiler structural pass** (`quivers.dsl.compiler.structural._compile_signature`) reads `dim` and `vocab` from each sort and vertex-kind's `options` block (the homogenized surface) rather than from removed direct attributes. `_compile_loss` accepts numeric-literal `weight = N` in the loss option block.
- **`docs/developer/homogenization-plan.md` removed.** Internal design-tracking doc with phase counts and version anchors; not appropriate for shipped documentation.

### Editor packages

- **VS Code extension `editors/vscode-qvr` bumped to 0.6.0.** Syntax highlighting refreshed for the homogenized grammar (the unified declaration keywords, the option-block surface, the new pragma markers, the `composition NAME as <level>` keyword, the `sample` / `observe` / `score` / `marginalize` / `binders` program-step keywords). Tracks the new shared `STYLE_TABLE` so VS Code, the REPL TUI, the Pygments lexer, and the LSP semantic-tokens legend all paint the same source the same colour.
- **Zed extension `editors/zed-extension-qvr` bumped to 0.2.0.** Tree-sitter `highlights.scm` regenerated from the homogenized grammar via the shared `grammars/qvr/queries/_generate.py` script, so the Zed mirror tracks the canonical highlights file automatically.

## [0.10.0] - 2026-05-17

### Added

- **`qvr repl` — GHCi-style interactive type explorer.** Four-pane [Textual](https://textual.textualize.io/) TUI (top status bar; input editor with bracket pairing and auto-indent; eval log with live syntax highlighting; environment browser with namespace tree, fuzzy filter, and click-to-`:info`; auto-hiding watches and diagnostics strips), plus a [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/) single-line fallback (`--plain` / non-TTY). Meta-commands `:load`, `:reload`, `:type`, `:kind`, `:info` (`--python` opts in to the didactic AST), `:doc`, `:browse`, `:dump` (`--json` opts in to `model_dump_json`), `:edit` (opens `$EDITOR` and splices the result back), `:trace`, `:save`, `:watch`, `:unwatch`, `:set`, `:help`, `:quit`, with the short aliases GHCi users expect (`:l`, `:r`, `:t`, `:k`, `:i`, `:b`, `:s`, `:w`, `:h`, `:q`). Bare lines compile as appended statements into the live module or fall through to `:type`. Persistent input history at `~/.config/quivers/history` (`Ctrl-Up` / `Ctrl-Down` walks it). `Tab` cycles through env, grammar-keyword, builtin, and path completions. `Ctrl-P` opens a Textual command palette over every meta-command. `Ctrl-G` / `Ctrl-O` / `F8` evaluate; the bindings are picked to clear every layer of OS, terminal, XON/XOFF, and `TextArea` keymap interference on macOS, Linux, and Windows. Install with `pip install 'quivers[repl]'`.
- **Live syntax highlighting shared with the editor.** A single `STYLE_TABLE` and tree-sitter-driven tokenizer (`quivers.cli.repl_highlight`) feeds the TUI's `RichLog`, the prompt_toolkit lexer, the Jupyter kernel's mimetype hint, and the LSP's `semanticTokens/full` legend. The tokenizer is env-aware: identifiers the grammar can't classify in an ERROR context are upgraded to the env-known kind (`type` / `function` / `namespace`), so the same name reads the same colour across every surface. Adding a keyword to `grammars/qvr/grammar.js` paints it everywhere automatically.
- **File-watcher auto-reload.** The TUI polls the loaded file's mtime once per second and re-runs `:reload` when it advances, printing the diff. Pinned `:watch` expressions re-evaluate against the new env. Off via `:set autoload_on_save=false`.
- **Pinned watches.** `:watch EXPR` adds EXPR to a dedicated panel (auto-hidden when empty) and re-evaluates after every recompile; `:unwatch EXPR` removes one, `:unwatch` alone clears all.
- **Click-to-info and click-to-edit.** Identifiers in rendered eval-log output (anything the env classified as a type / function / namespace) are clickable links that fire `:info NAME`. The `-- declared at PATH:LINE:COL` footer underneath each `:info` body is also clickable; it launches `$EDITOR +LINE PATH`.
- **Inline diagnostic markers.** Compile, parse, and constraint errors that carry a known location move the input pane's cursor selection to the offending span so the user sees where to look without leaving the editor.
- **Jupyter kernel.** `qvr-kernel install` (or `qvr kernel install`) registers a `quivers` kernelspec whose cell evaluator is the same `ReplSession` driving the REPL. Cells with a leading `:` dispatch to meta-commands; blank lines split independent statement chunks within a cell. Tab-completion (`do_complete`) and Shift-Tab inspect (`do_inspect` → `:info`) share the REPL's completer and renderer. The kernelspec advertises `pygments_lexer: qvr` so notebook frontends highlight cells via the bundled Pygments lexer.
- **`qvr-lsp` — Language Server (LSP 3.17).** [pygls](https://github.com/openlawlibrary/pygls) 2.x server with incremental document sync, `publishDiagnostics` (parser + constraint solver + compiler with source ranges), `semanticTokens/full` (driven by the shared `STYLE_TABLE`), `hover` (QVR declaration sliced verbatim from the original source plus the didactic AST stacked beneath under a collapsed `<details>`, pretty-printed one field per indented line), `definition`, `references`, `documentSymbol` (`Class` for objects/spaces, `Function` for morphisms, `Variable` otherwise), `completion` (same engine as the REPL), and `formatting` (canonical re-emission via [`module_to_source`](https://quivers.dev/api/dsl/emit)). Install with `pip install 'quivers[lsp]'`. New console scripts `qvr-lsp` and `qvr-kernel`; new subcommands `qvr repl`, `qvr lsp`, `qvr kernel`.
- **First-party VS Code / Cursor extension.** `editors/vscode-qvr` (v0.5.0) bundles a [`vscode-languageclient`](https://github.com/microsoft/vscode-languageclient-node) bridge to `qvr-lsp` plus the existing TextMate grammar. Auto-discovers the server in (1) `qvr.lsp.path` (with `${workspaceFolder}` expansion), (2) `<workspace>/.venv/bin/qvr-lsp`, (3) `<workspace>/.venv/Scripts/qvr-lsp.exe`, (4) `$VIRTUAL_ENV/bin/qvr-lsp`, (5) `$PATH`. Configurable via `qvr.lsp.enabled`, `qvr.lsp.path`, `qvr.lsp.args`.
- **First-party Zed extension wiring.** `editors/zed-extension-qvr` adds `qvr-lsp` as a `language_server` entry; Zed spawns it automatically on `.qvr` files.
- **`:save FILE` meta-command.** Writes the live module (including any statements appended in the session) back to disk via `module_to_source`. Defaults to the loaded path when no argument is given.
- **Optional dependency groups.** `[repl]` (Textual, prompt_toolkit, rich, ipykernel, jupyter_client) and `[lsp]` (pygls, lsprotocol). `pip install quivers` alone keeps the core dependency list unchanged — none of the interactive tooling is mandatory.
- **Comprehensive interactive guide.** New [`guides/repl-and-lsp.md`](https://quivers.dev/guides/repl-and-lsp) covers every meta-command, every keybinding (with the cross-platform reasoning), the layout, the file watcher, semantic highlighting, themes, the Jupyter kernel, the LSP capabilities, and per-editor wiring instructions. Cross-linked from the README, `docs/index.md`, `getting-started/installation.md`, `getting-started/quickstart.md`, and `getting-started/highlighting.md`.

### Changed

- **panproto bumped to >=0.47.3 and panproto-grammars-all to >=0.47.3.** The current QVR tree-sitter grammar (with the `alias`, `export`, `program`, kernel-decl forms) is now vendored upstream; the local-grammar override (`QVR_USE_LOCAL_GRAMMAR=1`) is no longer necessary for either the runtime or the CI test suite.

### Fixed

- **pygls 2.x API migration.** `LanguageServer.publish_diagnostics` was removed upstream in favour of `text_document_publish_diagnostics(PublishDiagnosticsParams(...))`. The previous call surfaced as an `AttributeError` dialog in VS Code on every save.

## [0.9.1] - 2026-05-16

### Fixed

- **`MonadicProgram.log_joint` rank-promotes upstream continuous values for single-arg steps.** When a draw step's `args` was a single name, `_resolve_input` returned the env tensor verbatim, so a 1-D continuous draw (the shape `Guide.rsample` emits for a 1-D `Euclidean` codomain) fed `nn.Linear` as a `(B,)` row vector and crashed the downstream `_NeuralSource` matmul with `mat1 and mat2 shapes cannot be multiplied (1xB and 1xH)`. Single-arg continuous inputs are now promoted to `(B, 1)` via the same rule `_stack_tensors` already applied for the multi-arg branch, with integer-dtype inputs (which feed `nn.Embedding` lookups) left untouched. `_NeuralSource.forward` also unsqueezes a stray rank-1 input as a belt-and-braces guard against custom `ContinuousMorphism`s regressing the same way. Affects every shipped `Conditional*` family sitting downstream of a 1-D continuous step; surfaced by direct `MonadicProgram(steps=[...])` users on the first batched `svi.step`.

## [0.9.0] - 2026-05-15

### Added

- **Comprehensive PyTorch primitive surface for let-expression bodies.** The let-expression compiler now resolves 86 named primitives drawn from `torch.nn.functional` and `torch`: every standard activation (`relu`, `gelu`, `silu`/`swish`, `mish`, `elu`, `selu`, `leaky_relu`, `prelu`, `rrelu`, `hardtanh`, `hardshrink`, `softplus`, `softshrink`, `softsign`, `hardsigmoid`, `hardswish`, `sigmoid`, `logsigmoid`, `tanh`, `tanhshrink`, `threshold`, `glu`), simplex maps (`softmax`, `log_softmax`, `softmin`, `normalize`), pointwise transcendentals (`exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `rsqrt`, `square`, `abs`, `neg`, `sign`, `reciprocal`, `clamp`), trig / hyperbolic (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`), rounding (`floor`, `ceil`, `round`, `trunc`), special functions (`erf`, `erfc`, `erfinv`, `lgamma`, `digamma`), `dim=-1` reductions (`sum`, `mean`, `var`, `std`, `min`, `max`, `argmin`, `argmax`, `prod`, `amax`, `amin`, `logsumexp`, `norm`), cumulative / ordering ops (`cumsum`, `cumprod`, `cummax`, `cummin`, `flip`, `sort`), and training-mode primitives (`dropout`, `alpha_dropout`, `layer_norm`, `rms_norm`). Reductions take `dim=-1` by convention; reductions over a named axis go through the typed `contraction` surface.
- **User-defined programs callable from encoder bodies.** Encoder / decoder / loss rule bodies (and any let-expression context) can now invoke top-level `program` declarations, `let`-bound morphisms, `encoder`s, `decoder`s, and `deduction`s by name. The call form `f(arg, ...)` resolves first against the primitive table, then against the structural-globals dict, then against the user-declared constructor set. A deterministic program is a Dirac Kleisli arrow embedding Smooth into Kleisli(Giry); calling it from an encoder body composes the two Smooth pieces and stays in Smooth.
- **Compile-time arity checking on let-expression calls.** For user-defined callables (`Morphism`, `MonadicProgram`, plain Python callables) the compiler verifies the positional argument count at compile time and raises `CompileError("call to 'f': expected N positional argument(s), got M")`. Variadic callables (`*args`) skip the check. `Morphism` is unary; `MonadicProgram` arity equals `len(params)` when present, else 1.
- **Runtime tensor-shape errors wrapped at call sites.** A `RuntimeError` or `TypeError` raised inside a user-defined callable surfaces as `CompileError("call to 'f' failed: ...")`, so shape mismatches name the call site instead of producing a bare PyTorch trace.
- **DSL `[init=auto]` annotation on latent declarations.** `latent f : A -> B [init=auto]` overrides the default `randn * scale` init with the algebra's initialization recipe. Through `LatentMorphism`'s sigmoid bijector (for ProductFuzzy / Boolean / Gödel / Łukasiewicz / Probability) the raw parameter is set via `logit(value)`; for other algebras the recipe is applied to the raw parameter directly. The annotation rides on the existing `option_block` grammar; no new keywords.
- **Encoder factory form: `encoder NAME over SIG using FACTORY [k=v, ...]`.** An alternative to the explicit `{ ... }` body that invokes a shipped builder from `quivers.structural.shapes` (`rnn_encoder`, `transformer_encoder`, `bow_encoder`, `tree_lstm_encoder`, `gnn_encoder`) with optional kwarg overrides. The two forms are mutually exclusive on one declaration and can coexist in the same module. The registry mapping factory name → import path lives in `quivers.dsl.compiler.structural._ENCODER_FACTORY_REGISTRY`.
- **`quivers.analysis` package** — algebra-guided training tooling.
  - [`ChainShape.from_module(module)`](https://quivers.dev/api/analysis/chain_shape) walks a compiled QVR `Module` and produces a per-step record (`StepShape`) with bound variable name, source line / col, chain depth (1-indexed at the first stochastic bind), governing algebra, and inferred intermediate axis size.
  - [`Algebra.init_spec(depth, intermediate_size) -> InitSpec`](https://quivers.dev/api/analysis/init_spec) returns the closed-form initialization recipe per algebra (product-fuzzy `p ≈ ln(2)/k`, Łukasiewicz `p ≈ 1/k`, log-prob `-ln k`, Markov logits at 0, tropical/max-plus/real `1/√k`, Boolean/Gödel idempotent at 1/2, probability/counting `1/k`). [`recommend_init`](https://quivers.dev/api/analysis/init_spec) maps every latent in a module to its `InitSpec`; [`apply_init_spec`](https://quivers.dev/api/analysis/init_spec) materializes the recipe onto a learnable tensor via `torch.nn.init`.
  - [`saturation_warnings(module)`](https://quivers.dev/api/analysis/saturation) returns source-keyed `SaturationWarning` diagnoses for every latent whose recipe differs materially (> 20% in location or scale) from a default `Normal(0, 1)`.
- **Type-driven contraction wiring.** `contraction NAME(args) : DOM -> COD` blocks no longer require a hand-written einsum string. The compiler infers the einsum from the typed signature: axes that appear in the output propagate; axes that appear in ≥ 2 inputs but not the output get contracted via the rule's join; axes that appear in exactly one input but not the output are flagged with a source-keyed error pointing at the disambiguators. Two opt-in clauses for the cases inference can't derive: `share T1, T2, ...` keeps the listed axes element-wise even when shared across inputs; `wiring "<einsum>"` remains as the explicit escape hatch (diagonal extraction, reorderings, etc.).

## [0.8.0] - 2026-05-15

### Changed

- **BREAKING: `fit(..., sampler=...)` renamed to `fit(..., method=...)`.** SVI is not a sampler (it fits a variational `Guide` by optimization); the parameter selects an inference algorithm. The literal values (`"nuts"`, `"hmc"`, `"svi"`) are unchanged. Update call sites `sampler="..."` → `method="..."`.
- **BREAKING: random-effect terms emitted in non-centered form by default.** `(slope | g)` formulae now compile to a standard-Normal plate draw `z_g_<slope> : g <- Normal(0, 1)` plus a deterministic `let alpha_g_per_row = sigma_g * z_g_<slope>[g_idx]`, rather than the textbook centered `alpha_g : g <- Normal(0, sigma_g)`. Same posterior mathematically; the funnel that wrecks centered hierarchical sampling under HMC / NUTS and under-shrinks variance components under VI is gone. Pass `fit(..., reparameterize="centered")` to recover the previous behaviour.

### Added

- **`fit(..., guide=...)`** exposes guide selection for SVI (previously hard-coded to [`AutoNormalGuide`](https://quivers.dev/api/inference/guides/normal)). Accepts any [`Guide`](https://quivers.dev/api/inference/guides/base) subclass: `AutoMultivariateNormalGuide`, `AutoLowRankMultivariateNormalGuide`, `AutoLaplaceApproximation`, `AutoIAFGuide`, etc. The class is constructed internally with `(program, observed_names=...)`.
- **`fit(..., reparameterize=...)`** controls the parameterization of random effects. `"noncentered"` (default; recommended by Stan / brms) or `"centered"`. `formula_to_qvr` takes the same parameter; `BayesianFit.reparameterize` records the choice so `qvr_source` echoes the actual emitted form.
- **[Tutorial 8: Analysis pipelines](https://quivers.dev/tutorials/python/08-analysis-pipelines).** Step-by-step walkthrough of the formula → fit → diagnostics surface on synthetic data: Gaussian SVI fit, hierarchical SVI, NUTS Bernoulli + PSIS-LOO model comparison, ArviZ posterior-predictive checks, the lens round-trip. Every Python block verified end-to-end against the running runtime.

### Fixed

- **NUTS / HMC on hierarchical formula models** no longer dies with `Sizes of tensors must match except in dimension 1`. `MonadicProgram._stack_tensors` broadcasts size-1 leading dims against the maximum batch in the input list, so a scalar latent (`sigma`) and a per-row tensor (`mu`) can be concatenated ahead of an indexed observe.

## [0.7.1] - 2026-05-15

### Changed

- **`FormulaToQVRModule` complement is `FormulaData`, not `Formula`.** The lens was declared as `Lens[Formula, Module, Formula]` — the lazy function-lens idiom in which the entire source rides as complement and `backward` is `const`. The honest setup is `Lens[Formula, Module, FormulaData]`: the structural fields of the formula (which columns exist, intercept flag, random-effect group / slope structure, family choice, response identifier) are recoverable from the emitted `Module` via a new `_decode_module` walker, and only the un-decodable fields ride in the complement (per-row data arrays, original pre-[`_qvr_name`](https://quivers.dev/api/formulas/formula) identifier names, `term` / `name` presentation labels, the original formula string). `backward(module, complement)` now performs a real decode + fuse pass. GetPut holds for every formula. New public symbol: [`FormulaData`](https://quivers.dev/api/formulas/formula#quivers.formulas.formula.FormulaData) in `quivers.formulas.formula`.

## [0.7.0] - 2026-05-15

### Changed

- **BREAKING: `quantale` → `algebra` across DSL, Python API, and docs.** The `quantale` keyword and every `Quantale`-named symbol overclaimed the algebraic structure: seven of the eleven built-in cases fail strict quantale-distributivity in [Kelly's sense](https://ncatlab.org/nlab/show/enriched+category). The new term *algebra* names the structure $(V, \otimes, \oplus, \mathbf{1})$ honestly; the subclass that does satisfy the strict laws keeps the name *strict quantale* and is identified inline in [Algebras §2](https://quivers.dev/semantics/algebras/#2-algebraic-scope). Migration:

  | Old | New |
  |-----|-----|
  | `quantale X` (DSL keyword) | `algebra X` |
  | `Quantale` (abstract base) | `Algebra` |
  | `BooleanQuantale`, `LukasiewiczQuantale`, `GodelQuantale`, `TropicalQuantale`, `MaxPlusQuantale`, `LogProbQuantale`, `RealQuantale`, `ProbabilityQuantale`, `CountingQuantale`, `MarkovQuantale` | `BooleanAlgebra`, `LukasiewiczAlgebra`, `GodelAlgebra`, `TropicalAlgebra`, `MaxPlusAlgebra`, `LogProbAlgebra`, `RealAlgebra`, `ProbabilityAlgebra`, `CountingAlgebra`, `MarkovAlgebra` |
  | `ProductFuzzy` | `ProductFuzzyAlgebra` |
  | `DualQuantale`, `CustomQuantale`, `QuantaleFromCallables` | `DualAlgebra`, `CustomAlgebra`, `AlgebraFromCallables` |
  | `QuantaleHomomorphism` | `AlgebraHomomorphism` |
  | `Algebra.compatible_quantales()` | `Algebra.compatible_algebras()` |
  | `_QUANTALE_REGISTRY`, `_register_extra_quantales` | `_ALGEBRA_REGISTRY`, `_register_extra_algebras` |
  | compiler env binding `__quantale__` | `__algebra__` |
  | `QuantaleDecl` AST node | `AlgebraDecl` |
  | `quivers.core.quantales` module | `quivers.core.algebras` |
  | `quivers.core.quantale_morphisms` module | `quivers.core.algebra_morphisms` |
  | `quivers.stochastic.quantale` re-export | **deleted**; import `MarkovAlgebra`, `MARKOV` from `quivers.core.algebras` |
  | `quivers.Algebra` (Eilenberg-Moore monad algebra) | `quivers.MonadAlgebra` (the public re-export; inside `quivers.monadic.algebras` the class is still named `Algebra`) |

  Singletons (`PRODUCT_FUZZY`, `BOOLEAN`, `LUKASIEWICZ`, `GODEL`, `TROPICAL`, `MAX_PLUS`, `LOG_PROB`, `REAL`, `PROBABILITY`, `COUNTING`, `MARKOV`, `REICHENBACH`, `MATERIAL_IMPLICATION`) keep their names. Registry key strings (`"product_fuzzy"`, `"boolean"`, ...) keep their names.

### Added: analysis pipelines (formula → fit → diagnostics)

- **`quivers.formulas`: brms-style formula frontend.** A typed AST lens compiles `y ~ x + (1 | g)` formulas through the existing QVR DSL without touching source strings: the lens emits a [`quivers.dsl.ast_nodes.Module`](https://quivers.dev/api/dsl/ast_nodes) directly and the existing Compiler consumes it. R / brms behaviour is faithful: orthogonal polynomials by default (`poly(x, k)` matches `stats::poly`), one coefficient per design-matrix column (`poly(x, 2)` produces `beta_poly_x_2_1` / `beta_poly_x_2_2`), R-style numeric transforms wired in (`log`, `exp`, `sqrt`, ...), and the family registry (`gaussian`, `bernoulli`, `binomial`, `categorical`, `poisson`, `negative_binomial`, `gamma`, `beta`, `student_t`, `cumulative`) drives the inverse-link + observe step. One-line entry is `fit("y ~ x + (1|g)", data=df, family=, sampler=, ...)` returning a [`BayesianFit`](https://quivers.dev/api/formulas/fit) `dx.Model`; `formula_to_qvr` emits the same module as canonical `.qvr` source. Gated behind the `formulas` extra.
- **`quivers.data`: DataFrame schema bridge.** [`DatasetSchema`](https://quivers.dev/api/data/schema) turns a pandas / polars / Narwhals-compatible dataframe into the object cardinalities, observations dict, and plate-index codes a QVR program needs. `compose(qvr_body, schema)` prepends inferred `object X : N` declarations to a user-written program body so the user only writes the program body. Four missing-data policies: `raise`, `drop`, `impute`, `mask`. Gated behind the `data` extra.
- **`quivers.diagnostics`: ArviZ adapter.** Glue layer between quivers' inference records and the ArviZ 1.x `xarray.DataTree` surface. [`to_datatree`](https://quivers.dev/api/diagnostics/arviz_io) wraps quivers' per-site sample tensors, log-densities, acceptance rates, and divergences into the canonical ArviZ groups (consumable by `plot_trace`, `plot_forest`, `plot_ppc`, `loo`, `compare`, `hdi`, ...). [`compare`](https://quivers.dev/api/diagnostics/comparison) delegates to ArviZ's PSIS-LOO ranking ([Yao et al. 2018](https://doi.org/10.1214/17-BA1091)). [`posterior_predictive_check`](https://quivers.dev/api/diagnostics/predictive_checks) computes the canonical posterior-predictive p-value for a user-chosen test statistic. Gated behind the `diagnostics` extra.
- **`quivers.dsl.emit`: AST → `.qvr` source serializer.** [`module_to_source`](https://quivers.dev/api/dsl/emit) walks a `Module` AST and produces canonical `.qvr` source; the round-trip `loads(module_to_source(m)) == m` is exercised on every formula in the test suite.
- **`TransformedMorphism`.** `Morphism.change_base`, `.dagger`, `.trace`, and `.refactor` now return a [`TransformedMorphism`](https://quivers.dev/api/core/morphism_transformations) whose `.tensor` is recomputed from the source morphism's tensor on every access, keeping autograd graphs alive across transforms.
- **Posterior-recovery test marker.** Ten end-to-end recovery tests under `tests/test_formulas.py::TestPosteriorRecovery` fit synthetic data with known true coefficients (Gaussian / Bernoulli / Poisson / Gamma, hierarchical random intercepts, multi-predictor regression, log transforms, polynomial terms, interactions, random slopes); marked `@pytest.mark.slow` and deselected by default (`pytest --runslow tests/` to include).

### Changed: regression examples

- Drop the spurious `x : Resp <- Normal(0, 1)` declarations from every regression example (`bayesian_regression`, `beta_regression`, `dirichlet_regression`, `horseshoe_regression`, `negbin_regression`, `zip_regression`): predictors are exogenous data and flow through the host-data channel as free variables in `let` expressions, not as latent draws. Gallery markdown explainers updated to match.

## [0.6.0] - 2026-05-14

### Added

- **Factor expressions: `let f = factor v_1 : I_1, ..., v_n : I_n in <body>`.** A new let-expression form that assembles an indexed tensor of shape `(|I_1|, ..., |I_n|, *body_shape)` by evaluating `<body>` once per cell of the Cartesian product of binder indices. The categorical content is the left adjoint of multi-axis indexing: factor is to indexing as a (co)limit cone is to its components. A single-binder pattern-match form `factor v : I in { 0 -> e_0, 1 -> e_1, ... }` is also supported for cell-structured priors (label coverage and in-range labels are checked at compile time). Closes [#19](https://github.com/quivers-dev/quivers/issues/19).

## [0.5.0] - 2026-05-13

### Headline additions

- **Axis-role surface on every distribution clause.** Kernel
  declarations, latent priors, sample steps, and observe steps
  accept an optional `over <axes> [iid over <axes>]` post-clause.
  `over` names the **event axes** of the family; remaining axes are
  iid (categorically a product of independent distributions).  Axis
  names resolve against the named factors of the surrounding
  morphism's dom/cod; `dom`/`cod` are shortcuts when that side is a
  single unfactored object.  The axis count must match the family's
  declared `event_rank` (0 / 1 / 2 for scalar / vector / matrix
  families); mismatch is a compile-time error rather than a silent
  reinterpretation, preserving the categorical distinction between
  a dense MVN over `dim(A)*dim(B)` and a `MatrixNormal` with
  Kronecker structure `V (X) U`.
- **Unified `kernel` keyword.** The `continuous` and `stochastic`
  declaration keywords are removed.  A `kernel f : A -> B`
  declaration without a `~` clause is a finite-set lookup-table
  kernel; with `~ Family [options] [axes]` it is a parametric kernel
  whose family parameters come from the input by a neural parameter
  network at sample time.
- **Latent morphism priors.** A `latent f : A -> B ~ Family(args)
  [options] [over <axes>]` clause puts a prior on the morphism's
  representing tensor (factor-analysis / PPCA / BNN idiom).
- **Four new conditional families.**
  [`ConditionalMatrixNormal`](https://quivers.dev/api/continuous/families#quivers.continuous.families.ConditionalMatrixNormal)
  (Kronecker covariance, event_rank 2);
  [`ConditionalInverseWishart`](https://quivers.dev/api/continuous/families#quivers.continuous.families.ConditionalInverseWishart)
  (conjugate covariance prior, event_rank 2);
  [`ConditionalGaussianProcess`](https://quivers.dev/api/continuous/families#quivers.continuous.families.ConditionalGaussianProcess)
  (RBF / Matern 5/2 / linear kernels with learnable length scale
  and amplitude, event_rank 1); and
  [`ConditionalHorseshoe`](https://quivers.dev/api/continuous/families#quivers.continuous.families.ConditionalHorseshoe)
  (Carvalho-Polson-Scott global-local shrinkage with a 16-point
  Gauss-Legendre quadrature giving the exact marginal log-prob,
  event_rank 1).
- **Example gallery, 36 examples.** Regression (Bayesian, beta,
  Dirichlet, negative-binomial, horseshoe, ZIP); latent-variable
  (factor analysis, PPCA, LDA, IRT-2PL, PMF, BNN, GMM, VAE);
  time-series and state-space (HMM discrete, continuous HMM,
  linear-Gaussian SSM, deep Markov, AR(1), stochastic volatility,
  changepoint, Weibull survival); sequence architectures (vanilla
  RNN, LSTM, GRU, bidirectional RNN, transformer; all as language
  models; one seq2seq with encoder + decoder); formal grammars
  (PCFG, CCG, Lambek calculus, multimodal TLG, custom rules,
  Montague NLI, quantifier scope, event-structure latent class).
  Each example uses at least one distinguishing quivers feature
  (axis-role priors, marginalize, change_base, encoder block,
  scan, deduction declarations) and runs end-to-end on synthetic
  data.

### Refactors

- `src/quivers/continuous/bayesian.py` renamed to `plate.py` and
  further split: `_DeterministicMorphism`, `cumsum`, `softmax`,
  `cholesky_quad_form` moved to `quivers.continuous.deterministic`;
  `CholeskyFactor` continuous space moved to
  `quivers.continuous.spaces`; `LKJCorrelationFactor` and
  `Truncated` moved to `quivers.continuous.families`; `plate.py`
  now strictly holds the plate / vectorized-observe /
  grouped-marginalize machinery.
- `src/quivers/dsl/compiler.py` (~6300 LOC) split into a package
  `src/quivers/dsl/compiler/` with one module per concern
  (`_prelude`, `core`, `declarations`, `programs`, `structural`,
  `deductions`, `resolution`, `expressions`).  Public import paths
  unchanged.

A substantive expansion of the inference layer plus the full
implementation of scoped grouped `marginalize` blocks (issue #9),
a unified distribution family registry with every torch.distributions
family exposed inline, the Tier-1 through Tier-6 benchmark grid
emitting `docs/developer/inference-benchmarks.md`, and the
categorical refinements from issues #15-#17 (algebra duality,
shape-aware change-of-base, storage-level shape compatibility).
Pre-1.0 clean break: no backwards-compatibility shims; user code
that relied on the `SVI(loss=...)` keyword should switch to
`SVI(objective=...)`, `Predictive(guide=...)` should use the
positional `posterior` argument, and imports from
`quivers.core.extra_algebras` move to `quivers.core.algebras`.

### Added: algebra duality and user-defined algebras

- **`Algebra.dual()`** returns a `DualAlgebra` whose
  `tensor_op` and `join` swap roles under the de-Morgan
  involution. `ProductFuzzy.dual` gives the canonical
  Reichenbach-flavor probabilistic-implication composition
  (`⊗ = noisy-OR`, `⋁ = product reduction`); `Boolean.dual`
  gives `(OR, AND)`; `Lukasiewicz.dual` gives the bounded-sum
  t-conorm pair; `Godel.dual` gives `(max, min)`.
- **Named singletons** `REICHENBACH`, `BOOLEAN_DUAL`,
  `DUAL_LUKASIEWICZ`, `DUAL_GODEL` exported from
  `quivers.core.algebras` and registered with the DSL algebra
  catalog so users can write `algebra reichenbach`.
- **`CustomAlgebra`** accepts user-supplied `tensor_op` / `join`
  / `unit` / `zero` / `negate` callables for arbitrary
  user-defined algebras, with construction-time sanity checks
  on the identity / absorbing axioms.

### Added: functorial change-of-base

- **`MorphismTransformation` ABC** in
  `quivers.core.morphism_transformations`: shape-aware
  change-of-base for transformations that don't factor pointwise
  through a algebra homomorphism. Concrete subclasses:
  `Softmax(axis_object)`, `L1Normalize(axis_object)`,
  `L2Normalize(axis_object)`, `BayesInvert(prior)`.
- **`Morphism.change_base`** now dispatches on either a
  `AlgebraHomomorphism` (pointwise) or a `MorphismTransformation`
  (shape-aware). The latter may swap the morphism's domain and
  codomain (used by `BayesInvert`).

### Added: first-class transformations in the DSL

- **Trans values** are now full DSL values: a transformation (a
  `AlgebraHomomorphism` or `MorphismTransformation`) can be
  let-bound, composed with `>>>`, and passed to `change_base`.
  The transformation namespace is disjoint from the morphism
  namespace; let-binding routes based on the RHS shape.
- **Constructors** are bare-verb names (`softmax(B)`,
  `l1_normalize(B)`, `l2_normalize(B)`, `bayes_invert(prior)`).
  The Python-side factory names in
  `quivers.core.morphism_transformations` follow the same
  spelling, `softmax`, `l1_normalize`, `l2_normalize`,
  `bayes_invert`, replacing the previous `*_over` suffixes.
- **Singletons** (`expectation`, `log_prob`, `material_implication`,
  `boolean_embedding`, `probability_clamp`, `probability_to_real`,
  `counting_from_real`, `counting_to_real`, `threshold`, `max_plus`)
  resolve by bare name, both directly inside `change_base` and as
  let-bindings (`let phi = expectation`).
- **`t1 >>> t2`** composes two transformations; nested chains
  flatten and each adjacent `target` / `source` boundary is
  type-checked at compile time. Inline or via let:
  ```
  let phi = softmax(B) >>> expectation
  let g = f.change_base(phi)
  ```

  Pre-1.0 clean break: the previous mini-surface
  `change_base(softmax_over(B))` is removed. Users on 0.4.x
  rename `softmax_over` → `softmax`, `l1_normalize_over` →
  `l1_normalize`, `l2_normalize_over` → `l2_normalize` (DSL and
  Python API both).

### Added: multi-observe grouped `marginalize` blocks

A grouped `marginalize` block now supports multiple `observe`
steps in the same body, each carrying its own `via <idx>` clause.
The runtime scatter-sums each observe's `(N_m, K)` per-row
per-class log-likelihood into the same `(|G|, K)` accumulator
before the log-sum-exp; categorically the right Kan extension
along the coproduct fibration $\coprod_m r_m : \coprod_m
\mathrm{Resp}_m \to G$. The single-observe case is the unary
slice; the multi-observe case unblocks hierarchical-Bayes
models where multiple heterogeneous response axes share a
per-group class indicator (the canonical pattern in the Stan
likelihoods that motivated the work).

- Surface change: the `via <idx>` clause moves from the
  `marginalize` header to each `observe` step. The marginalize
  header now declares only the grouping plate (`over G` or
  `over G * H`); each observe inside the body carries its own
  `via <idx>` (or `via product(idx_a, idx_b)`) clause. A
  grouped body whose observe lacks a `via` is rejected at
  compile time with a typed error.
- Runtime: `quivers.continuous.bayesian.marginalize_grouped`
  accepts either a single `(N, K)` log-likelihood tensor and
  fibration (single-observe path) or a parallel list of
  `(N_m, K)` tensors and fibrations (multi-observe path).
- Pre-1.0 clean break: the previous `marginalize ... over G via
  idx in { ... }` header form is removed. Users on 0.4.x with
  a grouped marginalize block move the `via <idx>` clause from
  the header onto the observe inside the body; a single
  per-block fibration becomes a single observe carrying that
  fibration. Multi-block patterns where two marginalize blocks
  shared a class prior collapse to a single block with two
  observes (per the canonical pattern in the issue that
  motivated this).
- `src/quivers/dsl/examples/event_structure.qvr` updated to the
  single-block-with-two-observes form.

### Added: documentation overhaul

- Two-track tutorial structure: a new **QVR DSL track** (seven
  chapters, model development to inference, side-by-side with
  PyMC / NumPyro / Stan) and a refreshed **Python API track**
  (two new chapters covering first-class transformations and
  the composition-rule hierarchy).
- New denotational-semantics page,
  `docs/semantics/composition-rules.md`, formalizing the
  `CompositionRule → Semigroupoid → Algebra` hierarchy,
  operadic n-ary contractions via flat wirings, and the sort
  `Trans[V, W]` for first-class transformations.
- New conceptual guide `docs/guides/transformations.md` covering
  the transformation surface and the composition-rule hierarchy
  end-to-end.
- Updated `docs/semantics/grammar.md` and `docs/guides/dsl.md`
  EBNF productions to reflect 0.5.0 surface: `>>>`, `change_base`,
  composition-rule keywords, contraction declarations, grouped
  `marginalize`, the full compose-operator family.
- `README.md` and `docs/index.md` rewritten with separate
  framings (probabilistic-programming user vs library
  developer); architecture and inference-stack diagrams render
  in Mermaid.
- New module `quivers.core.trans` exposing the `TransSeq` class
  and `compose_trans` function as the Python-side surface for
  the DSL's `>>>` operator; `Morphism.change_base` accepts
  `TransSeq` natively.
- QVR syntax highlighting refreshed across all three rendering
  surfaces: the Pygments lexer
  (`src/quivers/dsl/pygments_lexer.py`), the tree-sitter
  highlights query (`grammars/qvr/queries/highlights.scm`), and
  the VSCode TextMate grammar
  (`editors/vscode-qvr/syntaxes/qvr.tmLanguage.json`).
- Restored the MkDocs hook
  `docs/hooks/register_qvr_lexer.py` and wired it through
  `mkdocs.yml`; Mermaid rendering enabled via `pymdownx.superfences`
  custom-fence classifier plus a `mermaid-init.js` themer.

### Added: shape compatibility under product / factored codomains

- **Storage-level init check**: `observed f : A -> B = from_data("KEY")`
  now matches the init tensor against the declared codomain on
  *numel* rather than per-axis shape. A flat init whose total
  cardinality matches a declared product codomain is accepted
  and reshaped to the factored layout.
- **`Morphism.refactor(domain=..., codomain=...)`** exposes the
  reshape as a user-facing method: switch a morphism's view
  between flat and product factorings of isomorphic objects.

### Changed: module consolidation

- `quivers.core.extra_algebras` is **deleted**. All its classes
  (Lukasiewicz, Godel, Tropical, MaxPlus, LogProb, Real,
  Probability, Counting) and singletons now live in
  `quivers.core.algebras` alongside ProductFuzzy and Boolean.
  Internal imports updated; user code that imported from
  `extra_algebras` must update to import from `algebras`.

### Added: distribution-family wrappers

- **`ConditionalMixture`** wraps a `ConditionalX` family in a
  K-component mixture with learnable mixture logits.
- **`ConditionalIndependent`** reinterprets a base distribution's
  trailing batch dim as an event dim (analogue of
  `torch.distributions.Independent`).
- **`ConditionalTransformed`** pushes a base through a chain of
  bijectors, applying the log-det-Jacobian correction in `log_prob`.

### Fixed: HMC at constrained-support boundaries

- HMC's potential function now catches `torch.distributions`'
  support-validation errors and returns `-inf`; the kernel reads
  non-finite log-densities as divergent transitions and rejects
  them in the Metropolis step. Eliminates the prior ERROR cells
  in the benchmark matrix at the Gamma / InverseGamma /
  HalfNormal support boundaries.

### Changed: codebase-wide American spelling

- All British spellings (centered / normalize / parameterize /
  optimize / reparameterise / recognize / factorize / initialise
  / specialize / organize / minimize / maximize / discretize and
  their derived forms) converted to American spellings.
  `eight_schools_centred.qvr` renamed to `eight_schools_centered.qvr`
  with the program names updated to match.

### Added: unified distribution family registry

- **`FamilySpec` and `ParamSpec`** in
  `quivers.continuous.family_spec`, a single source of truth per
  family. Both the conditional path
  (`_IndependentConditional` and the hand-written `ConditionalX`
  classes) and the inline path (`FixedDistribution` /
  `MixedInlineDistribution`) read from the same record. Replaces
  the previous duplication between `families.py` and `inline.py`.
- **Inline coverage for 11 previously registry-only families**:
  Cauchy, Laplace, Gumbel, StudentT, Chi2, InverseGamma, Weibull,
  Pareto, Kumaraswamy, ContinuousBernoulli, FisherSnedecor. Every
  family declared via `_make_family` now auto-registers its
  fixed-inline factory, mixed-inline builder, and support
  constraint from the spec.
- **New families**: ConditionalPoisson, ConditionalGeometric,
  ConditionalNegativeBinomial, ConditionalVonMises plus their
  inline factories. Brings the registry to 34 distinct
  distribution families.
- **`_IndependentConditional` is shape- and discreteness-aware**:
  continuous-reparameterised, continuous-non-reparameterised
  (VonMises), and discrete (Poisson / Geometric / Bernoulli /
  Categorical) all share the same generic conditional class via
  a `discrete` flag plus a fall-back to `.sample()` when the
  underlying distribution lacks `rsample`.

### Added: benchmark suite expansion

- **Tier 1 conjugate**: normal_inverse_gamma (joint mean / variance
  recovery), gamma_exponential, bayes_linear_regression
  (well-conditioned).
- **Tier 2 hierarchical**: eight_schools_centred and
  eight_schools_noncentred (Rubin 1981 / Gelman et al. 2013) with
  cached NUTS-derived posterior moments.
- **Tier 3 hard geometry**: neal_funnel (scale-of-scale dependency,
  Neal 2003), ill_conditioned_mvn (per-dim posteriors with
  eigenvalues across four decades).
- **Tier 6 constrained support**: half_normal_scale (positive
  support), truncated_normal_recovery (unit interval).
- **Metrics extension**: gaussian_kl (closed-form KL between two
  Gaussians), wasserstein_2_1d (sorted-quantile W₂),
  total_variation_grid (TV against a quadrature-supplied
  density), split_r_hat and effective_sample_size (Vehtari et al.
  2021).
- **Runner** (`tests/benchmarks/runner.py`) drives an algorithm ×
  problem grid across AutoNormalGuide, AutoMultivariateNormalGuide,
  AutoLaplaceApproximation, HMCKernel, NUTSKernel and emits
  `docs/developer/inference-benchmarks.md`. Capture problems
  document expected failure modes (mean-field underfit on
  correlated MVN / funnel posteriors).

### Added: earlier 0.5.0 work


### Added: inference layer

- **`LatentRegistry`**: per-site introspection helper that every
  guide and MCMC kernel consumes. Walks the model's `_step_specs`
  once at construction and exposes per-site (support, dims, plate
  vs scalar, bijector, flat-vector offsets) plus flatten /
  unflatten between site dicts and a single flat unconstrained
  vector.
- **Transforms / flows**: `TransformModule` cooperative base
  inheriting both `torch.distributions.transforms.Transform` and
  `torch.nn.Module`. Primitives: `AffineCouplingTransform` (RealNVP),
  `MaskedAutoregressiveTransform` (MAF), `InverseAutoregressiveTransform`
  (IAF), `NeuralSplineCouplingTransform` (NSF rational-quadratic
  spline coupling), `LULinearTransform` (Glow), `BatchNormTransform`,
  plus `MADE` and helper masks.
- **Variational objectives**: `Objective` ABC and four
  implementations: `ELBO`, `IWAEBound` (Burda-Grosse-Salakhutdinov
  2016), `RenyiBound` (Li-Turner 2016), `VRIWAEBound`
  (Daudel-Douc-Roueff 2023). Each accepts a pluggable
  `GradientEstimator` strategy: `Reparameterised` (default),
  `StickingTheLanding` (Roeder-Wu-Duvenaud 2017),
  `DoublyReparameterised` (Tucker-Lawson-Gu-Maddison 2019; default
  for IWAE), `ScoreFunction`.
- **Guide zoo**: `AutoMultivariateNormalGuide` (full-rank Cholesky
  Gaussian), `AutoLowRankMultivariateNormalGuide` (Σ = W W^T +
  diag(σ²) via Woodbury), `AutoNormalizingFlow` (user-supplied
  transform stack), `AutoIAFGuide` (preconfigured IAF stack with
  reverse permutations), `AutoNeuralSplineGuide` (NSF coupling
  stack), `AutoLaplaceApproximation` (two-phase: SVI MAP then
  Hessian-derived Gaussian), `AutoMixtureGuide` (Gumbel-Softmax
  mixture of component guides).
- **MCMC**: `HMCKernel` and `NUTSKernel` operating on the flat
  unconstrained latent vector via the `LatentRegistry`. Both
  support identity / diagonal / dense mass matrices, Nesterov
  dual-averaging step-size adaptation (Hoffman-Gelman 2014 Alg 6),
  and Welford-online mass-matrix adaptation. `MCMC` driver orchestrates
  parallel chains with warmup-then-sample phases and produces
  `MCMCResult` carrying per-site posterior draws + split-R̂ + ESS
  diagnostics.
- **Hybrid samplers**: `AutoDAIS` (differentiable annealed
  importance sampling guide; Geffner-Domke 2021 / Zhang et al.
  2021) and `WarmupThenHMC` (SVI warmup then MCMC seeded from the
  guide's posterior mean).
- **`Predictive`** now accepts either a `Guide` or an `MCMCResult`
  as its posterior; the MCMC overload iterates the recorded
  posterior draws.

### Added: issue #9: scoped grouped `marginalize` blocks

- **`marginalize <v> : K <- F(probs) over G via idx in { ... }`**
  with the full feature surface from issue #9:
  - **Arbitrary nesting** of grouped marginalize blocks. The
    runtime distinguishes the innermost level (which has a row
    axis and a fibration to scatter along) from outer levels
    (which consume already-reduced contributions broadcast over
    the surviving outer class axes), so a stack of N blocks
    composes to the correct hierarchical log-mixture. Tested at
    depths 3-7.
  - **Product fibrations** via `via product(idx_a, idx_b, ...)`
    paired with `over G * H` product grouping plates.
  - **`reduction = logsumexp | sum | mean`** per block.
  - **Continuous latents in scope** inside the body; the compiler
    binds the latent name to `torch.arange(K)` so let-arithmetic
    referencing the latent broadcasts across the class axis.
  - **Body vectorization**: the body's terminal `observe` step is
    captured by the compiler (`GroupedBodyObserveStep` IR) and its
    per-row per-class log-likelihood is stored at the latent's
    environment slot for the marginalize callable to consume. No
    user-side `let` boilerplate.
- `quivers.continuous.bayesian.marginalize_grouped` generalized
  to accept multi-axis broadcast inputs, product fibrations, and
  the three reduction modes.
- `dsl/examples/event_structure.qvr` now uses real `observe`
  steps inside the grouped blocks.

### Added: synthetic benchmark suite

- `tests/benchmarks/` with Tier-1 conjugate (Beta-Bernoulli,
  Normal-Normal) and Tier-3 hard-geometry (correlated MVN)
  benchmarks. Models live in `tests/benchmarks/models/*.qvr` as
  canonical didactic models; dataset / reference modules are plain
  functions returning `(model, observations, true_params)`. The
  Tier-3 suite includes a capture test for the mean-field failure
  mode on a correlated posterior.

### Added: V-Cat categorical surface

- **Backend-agnostic morphism → `nn.Module` adapter**
  (`quivers.core.morphisms.as_torch_module`). Every binding site
  that calls `add_module` (`MonadicProgram` step list,
  `FanOutMorphism`, parametric programs) funnels through the
  adapter so non-Module morphisms (`LatentMorphism`,
  `ComposedMorphism`, `ProductMorphism`, …) bind without crashing.
  The wrapper attaches the original categorical morphism on
  `_morphism` for runtime recovery; `MonadicProgram.rsample`
  detects V-Cat steps and computes them as deterministic tensor
  applications.
- **Multi-algebra composition** with one operator per algebra.
  `>>` (ProductFuzzy default), `<<` (reverse), `>=>` (Kleisli),
  `*>` (Markov sum-product), `~>` (LogProb log-space),
  `||>` (Gödel min/max + Heyting), `?>` (Viterbi max-plus),
  `&&>` (Boolean), `+>` (Łukasiewicz), `$>` (Real sum-product),
  `%>` (Probability sum-product). Each operator carries its
  algebra; cross-operator chains require explicit
  `.change_base(φ)`. Five new algebra classes:
  `MaxPlusAlgebra` (Viterbi / MAP), `LogProbAlgebra`
  (log-space sum-product), `RealAlgebra` (sum-product on ℝ),
  `ProbabilityAlgebra` (sum-product on [0, 1]), `CountingAlgebra`
  (sum-product on non-negative integers). The last three mirror
  the corresponding arcweight weight-set semirings (`RealWeight`,
  `ProbabilityWeight`, `IntegerWeight`) and round out the
  built-in algebra catalog to eleven distinct algebras.
- **Algebra homomorphisms** (`quivers.core.algebra_morphisms`)
  for change-of-base: `Expectation`, `LogProb`, `MaxPlus`,
  `Threshold`, `MaterialImplication`, `Embedding`, `IdentityHom`,
  `ProbabilityClamp` (Real → Probability), `CountingFromReal`
  (Real → Counting via floor), `ProbabilityToReal` and
  `CountingToReal` (sub-algebra inclusions). Module-level
  singletons (`EXPECTATION`, `LOG_PROB_HOM`, `MAX_PLUS_HOM`,
  `MATERIAL_IMPLICATION`, `PROBABILITY_CLAMP`,
  `COUNTING_FROM_REAL`, `PROBABILITY_TO_REAL`,
  `COUNTING_TO_REAL`); factory helpers `threshold(tau)` /
  `embedding(src, tgt)`; a `HOMOMORPHISM_REGISTRY` keyed by
  `(source.name, target.name)`; and `lookup_homomorphism()`.
  Morphisms expose `.change_base(phi)` to apply a homomorphism;
  the DSL surface is `f.change_base(name)` with the catalog
  wired into the compiler (`expectation`, `log_prob`,
  `max_plus`, `material_implication`, `threshold`,
  `boolean_embedding`, `probability_clamp`,
  `probability_to_real`, `counting_from_real`,
  `counting_to_real`).
- **Compact-closed surface** on V-Cat morphisms: `f.dagger`
  (transpose), `f.trace(A)` (categorical trace), `cup(A)` and
  `cap(A)` (unit / counit). Each operation is well-defined for
  every algebra; the semantic interpretation depends on the
  active algebra (ProductFuzzy: tensor transpose; Markov:
  Bayes inversion; Viterbi: max-plus reversal; Boolean:
  relational converse).
- **Data-derived and expression-derived initialisers** for
  morphism declarations. `observed f : A -> B = from_data("KEY")`
  binds the morphism's tensor from a runtime-supplied data
  dictionary (passed via the `data=` keyword on
  `quivers.dsl.loads` / `load`, or `Compiler.bind_data(...)`).
  `inner.freeze` is a postfix that materialises an expression's
  tensor with `.detach().clone()` and wraps the result as a
  parameter-free `ObservedMorphism`. Expression-derived
  initialisers without `.freeze` propagate gradient lineage; the
  declaration's type-check accepts shape-compatible inits and
  re-tags them with the user-declared domain / codomain.

### Changed

- **`SVI`** takes an `objective: Objective` instead of `loss: ELBO`.
- **`Predictive`** takes a positional `posterior: Guide | MCMCResult`
  instead of a `guide:` keyword.
- The variational machinery moved off `PlateDraw` (`kl_to_prior`
  removed; the deeper migration of `_mean`/`_log_scale` happens
  in v0.5.0 alongside the rest of the refactor).
- `MonadicProgram` binding sites no longer crash on V-Cat
  morphisms; the runtime applies them as deterministic tensor
  steps.
- `FanOutMorphism` accepts both ContinuousMorphism and V-Cat
  morphism components; non-continuous components are wrapped in
  `DiscreteAsContinuous` so the fan-out loop dispatches uniformly.

### Removed

- `PlateDraw.kl_to_prior` (no callers; the Guide hierarchy owns
  variational distributions now).
- `quivers.inference.elbo` (re-exported from
  `quivers.inference.objectives` as `ELBO`).

## [0.4.1] - 2026-05-12

Three connected bug fixes that unblock hierarchical-Bayesian
workflows under the 0.4 surface. No grammar or parser changes;
panproto does not need to revendor `panproto-grammars-all` for this
release.

### Fixed

- **Variational guides respect constrained supports.**
  `quivers.inference.AutoNormalGuide` and
  `quivers.inference.AutoDeltaGuide` previously sampled in
  unconstrained real space for every latent and fed the result
  into the prior's `log_prob`, which raised
  `ValueError: Expected value to be within the support of the
  distribution` for any constrained family (`HalfNormal`, `Beta`,
  `Uniform`, `Exponential`, `Gamma`, `LogNormal`, `LogitNormal`,
  `HalfCauchy`, `Dirichlet`, …). Every `ContinuousMorphism` now
  exposes a `support: Constraint` property; the inline
  distributions (`FixedDistribution`, `MixedInlineDistribution`,
  `DirectBernoulli`, `DirectTruncatedNormal`) and the
  family-conditional distributions in `quivers.continuous.families`
  override it with the correct constraint. The auto-guides sample
  `z ~ Normal(loc, scale)` in unconstrained space, push through
  `biject_to(support)` to land on the constrained side, and
  evaluate `log_prob` with the Jacobian correction
  (`log N(z) + log|det J_{T^{-1}}(v)|`). The simplex case routes
  through `StickBreakingTransform` and accounts for the d ↔ d-1
  dimension reduction. This is the same construction Pyro's
  `AutoNormal` uses.

- **`condition(model, data)` exposes host data to `let`-expression
  gather, and plate latents are batch-invariant.** Keys in the
  conditioning data dict that do not match a declared sample /
  observe site are pre-populated into the trace environment as
  deterministic values, visible to `let`-expression evaluation.
  Free variables in `let` expressions (variables not bound by any
  sample / observe / let / lambda step) are no longer rejected at
  compile time; the runtime resolves them against the
  conditioning data dict. Together with the batch-invariant plate
  semantics below, this unlocks the canonical crossed-random-
  effects idiom:
  ```
  program p : Resp -> Resp
      by_subj : Subj <- Normal(0.0, 1.0)
      let mu = sigmoid(by_subj[subj_idx])
      observe r : Resp <- Bernoulli(mu)
      return mu

  cond = condition(p.morphism, {"subj_idx": idx, "r": y})
  ```
  Plate draws (`v : A <- F(args)`) are now batch-invariant: the
  latent is a single shared tensor of shape `(|A|, *B.shape)` :
  the standard Pyro / NumPyro semantic, instead of being
  replicated against the program input's leading batch axis. The
  gather `by_subj[subj_idx]` along the plate axis then produces a
  per-row predictor of shape `(N_resp,)` that broadcasts cleanly
  against an observed `Resp`-plate kernel. Scalar-per-row plates
  (`Normal`, `HalfNormal`, …) drop the trailing length-1 axis so
  the latent has the natural `(|A|,)` shape. Both
  :class:`AutoNormalGuide` and :class:`AutoDeltaGuide` were
  updated to advertize the same shape on the variational side:
  plate latents are stored as `(|A|, unconstrained_dim)` parameter
  tensors and sampled batch-invariant so ELBO substitution into
  the model's log-joint env aligns shape-by-shape with the
  model's :class:`PlateDraw` output. (Without this, the SVI step
  ran into an `IndexError` from the plate-axis gather even though
  `cond.trace(...)` on the same model succeeded.)

- **Inline `Dirichlet` accepted as a prior with scalar or vector
  concentration.** `pc <- Dirichlet(α)` and
  `pc <- Dirichlet([α_1, …, α_K])` both compile. For scalar `α`,
  the simplex dimension is inferred from the program's declared
  codomain (`dim` for a `ContinuousSpace`, `cardinality` for a
  `SetObject`, or 2 as a minimum); for a per-component vector,
  the simplex dimension is the number of literals. The
  `make_fixed_dirichlet` factory accepts both single-element
  sequences (treated as symmetric) and per-component sequences.
  Previous behavior was to raise
  `distribution family 'Dirichlet' is not supported as an inline
  distribution; declare it as a continuous morphism instead`, or
  to crash on the vector form with
  `TypeError: make_fixed_dirichlet() takes 2 positional arguments
  but 4 were given`.

### Internal

- `ContinuousMorphism.support` defaults to `constraints.real`;
  every constrained-output subclass overrides it. Variational
  guides consume this through
  `torch.distributions.constraint_registry.biject_to`.
- `_FAMILY_SUPPORTS` in `quivers.continuous.inline` maps each
  inline family to its support, applied by
  `make_inline_distribution` when constructing a
  `MixedInlineDistribution`. `Uniform` and `TruncatedNormal`
  specialize to the actual interval when both bounds are literal.
- `trace()` pre-populates `env` with the non-site keys of the
  observations dict before the program's steps run.
- `_validate_let_expr_vars` treats unbound names as deferred host
  references; the eval-time evaluator raises a clear `KeyError` if
  the value is missing.
- `PlateDraw.rsample` is batch-invariant: returns
  `(|A|, *B.shape)` regardless of the program input's batch axis;
  scalar-per-row plates squeeze the trailing length-1 dimension.
  `PlateDraw.log_prob` accepts either the natural plate-latent
  shape or the legacy flat `(batch, |A| · prod(B))` shape for
  back-compat.
- `_VECTOR_PARAM_FAMILIES` in `quivers.continuous.inline` lists
  the inline families whose all-literal factory takes a single
  vector argument rather than splatting positional floats; the
  parser's flattened literal sequence is re-bundled into a list
  before the factory call. Currently `{"Dirichlet"}`; the
  mechanism is ready for `MultivariateNormal`, `Wishart`, and
  `LKJCorrelationFactor` as those land.
- `make_fixed_dirichlet` treats a single-element concentration
  sequence as a scalar (symmetric Dirichlet), broadcast to the
  codomain's simplex dimension; multi-element sequences must
  match the codomain dimension exactly.

### Tests

`tests/test_inference_constrained.py` (21 cases): every supported
constrained family under both auto-guides, host-data passing
through `condition`, the end-to-end Bernoulli hierarchical-
regression observation kernel that exercises the plate-gather →
observe-plate composition, an SVI-step regression that verifies
the guide / model plate shapes align and the ELBO descends with
``loc_by_subj`` driven negative by all-zero responses, and inline
Dirichlet with both scalar and vector concentrations.

## [0.4.0] - 2026-05-12

This release lands three deeply-interconnected bodies of work in a
single minor bump, all motivated by the goal of making quivers a
unified surface for probabilistic, weighted-deductive, and
neural-symbolic programs:

1. **DSL surface homogenization**, one Kleisli-bind sigil `<-`,
   type-annotated indexed binds, scoped `marginalize`, and
   `!`-prefixed effect signatures.
2. **Agenda-based weighted-deduction framework**, a single
   engine subsuming CKY, Earley, Viterbi, inside-outside,
   semi-naive Datalog, A\* parsing, Knuth's algorithm, and MLTT
   proof search, declared via `deduction { … }` blocks with
   first-class differentiable charts.
3. **Hierarchical-Bayesian + arrow / algebraic-effects substrate**
  , plate draws, vectorized observations, marginalization,
   LKJ priors, Cholesky factor spaces, the Hughes arrow tower
   (`Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`,
   `ArrowZero`, `ArrowPlus`), stdlib monads / monad transformers,
   algebraic effects + handlers via `FreeMonad`.
4. **Structural compression: signatures, encoders, decoders,
   losses**, a uniform algebraic interface for compressing
   arbitrary structured objects (sequences, trees, graphs, charts,
   typed lambda terms) to fixed-length vectors and decoding them
   back under a learned distribution. Realizes transformers,
   tree-LSTMs, graph-NNs, autoregressive LMs, VAEs, and
   vector-inside-outside parsers as instances of one F-algebra /
   F-coalgebra pattern.

### Changed (breaking, pre-1.0 clean cut)

Surface DSL homogenization. The program-block surface is reorganized
around a single Kleisli-bind sigil `<-`, type-annotated indexing on
the binder, scoped marginalization, and `!`-prefixed effect
signatures. The categorical denotation is unchanged; only the surface
forms shift to a Haskell-PPL aesthetic.

- **`draw v ~ F(args)` → `v <- F(args)`**: the `draw` keyword is
  retired in favor of the unique Kleisli-bind sigil `<-`. The
  surface-arrow alternative (the v0.4 do-notation `v <- F` form) is
  unified with the new sigil; there is now one and only one way to
  introduce a random variable.
- **`draw v : A -> K ~ F(args)` → `v : A <- F(args)`**: indexed
  (plate) binds use a type annotation on the binder rather than a
  separate keyword family. The per-fiber codomain is taken from the
  family; the `: A` annotation declares the index set.
- **`observe v ~ F(args)` → `observe v <- F(args)`**: scored binds
  retain the `observe` prefix; the rest of the line matches the
  Kleisli-bind shape.
- **`observe r[n] ~ F(args) for n in N` → `observe r : N <- F(args)`** :
  the vectorized-observe `for n in N` shape collapses into the
  type-annotated form. Bracket-indexed family arguments `theta[N]`
  annotate that an argument is a section of an N-indexed family.
- **`marginalize c` (trailing) → `marginalize c : A <- F(args) in { … }`** :
  marginalization is always scoped; the integration target and the
  scope are visible at the binding site. The categorical pushforward
  becomes visually local to its binding.
- **`posterior name (model) [params] : dom -> cod` → `program name (params) : dom -> cod ! Pure over model`** :
  posterior blocks are encoded as `program` declarations with a
  `! Pure` effect signature and an `over model` modifier. The
  parser routes such programs to the posterior registry and the
  compiler enforces the determinism constraint.
- **`!` effect signature on programs**: `program P : X -> Y ! Sample, Score`
  declares the body's capability set. Effects: `Sample`,
  `Score`, `Marginal`, `Pure`. The compiler verifies the body's
  actual effects are a subset of the declared set (or rejects
  any effect when `Pure` is declared).
- **`output X` → `export X`**: module-level exports replace
  `output`; multiple `export` declarations per module are allowed.
- **Drop labeled-tuple return form**: the v0.4 `return (a: x, b: y)`
  form was purely syntactic rebinding without semantic effect; it
  is removed as dead surface.

### Internal

- `BindStep` unifies the four old step shapes (`DrawStep`,
  `PlateDrawStep`, `VectorisedObserveStep`, `MarginalizeStep`) at
  the AST level; the old shapes remain as compiler-internal IR
  consumed by the runtime step-builder. The parser emits only
  `BindStep` and `LetStep`; the compiler's `_expand_bind_steps`
  pass translates to the internal IR.
- `ProgramDecl` gains `effects: frozenset[str] | None` and
  `over_model: str | None` fields.
- `MonadicProgram` gains an `effect_set: frozenset[str] | None`
  field for introspection by downstream inference / dispatch code.
- `ExportDecl` replaces `OutputDecl`. The first declared export
  is the module's primary output; subsequent exports are
  accessible as additional module bindings.
- The DSL's `<-` arrow now binds with `prec.right`, eliminating
  the previous GLR ambiguity between draw-args and following
  steps.

### Migration guide

For most programs the mechanical translation is:

| pre-0.4                                      | 0.4                                                 |
|----------------------------------------------|-----------------------------------------------------|
| `draw v ~ F(args)`                           | `v <- F(args)`                                      |
| `draw v : A -> K ~ F(args)`                  | `v : A <- F(args)`                                  |
| `observe v ~ F(args)`                        | `observe v <- F(args)`                              |
| `observe r[n] ~ F(args) for n in N`          | `observe r : N <- F(args)`                          |
| `marginalize c`                              | `marginalize c <- F(args) in { … }`                 |
| `posterior P (M) [v] : … ` body              | `program P(v) : … ! Pure over M ` body              |
| `output E`                                   | `export E`                                          |

### Tests

- All example `.qvr` files migrated to the 0.4 surface.
- Tree-sitter corpus covers the new surface shapes (20/20 parses).

### Agenda-based weighted-deduction framework

0.4 also lands the full agenda-engine substrate
underneath a declarative `deduction { … }` block. The framework
subsumes CKY, Earley, Viterbi, inside-outside, semi-naive Datalog
evaluation, A* parsing, Knuth's algorithm, depth-first MLTT
proof search, and edit-distance dynamic programming as parameter
settings on a single engine.

- **Surface form**:
  ```qvr
  deduction CG : Atom -> Atom [semiring=LogProb, start=S, depth=4]
      atoms NP, S, VP
      rule fwd_app : X/Y, Y |- X
      rule bwd_app : Y, Y\X |- X
  ```
  Single-uppercase-letter pattern names (`X`, `Y`) bind as
  wildcards; non-wildcard atoms match literally. The block
  declares the seven irreducible parameters of an agenda-based
  deduction (item algebra via atoms, rule set, semiring, axiom
  source, goal predicate, start symbol, depth bound). Concrete
  parsing strategies are selected by the compiler from these
  parameters.

- **Charts as first-class differentiable values**: each
  deduction's runtime view exposes
  `chart.weight(item)`, `chart.enumerate(pattern)`,
  `chart.derivations(item)`, `chart.goal_weight()`, all returning
  `torch.Tensor` values whose gradients flow back through the
  agenda's semiring operations to any `requires_grad=True`
  axiom / rule weight, enabling end-to-end gradient-based
  learning over deduction systems (the Goodman 1999 semiring
  framework lifted to PyTorch tensors).

- **Pre-registered stdlib deductions**
  (`quivers.stochastic.stdlib`): `CCG`, `Lambek`, `STLC`,
  `MLTT`, `Datalog`, `Dijkstra`, `HMM`, `ViterbiHMM`,
  `EditDistance`. Users import and run them directly:
  ```python
  from quivers.stochastic.stdlib import Datalog
  view = Datalog(edge_axioms)
  reaches = view.enumerate(("reach", source, Wildcard("Y")))
  ```

- **Agenda strategies**: `cky_agenda()`, `earley_agenda()`,
  `viterbi_agenda(priority_fn)`, `astar_agenda(g_plus_h)`,
  `knuth_agenda()`, `depth_first_agenda()`,
  `semi_naive_agenda()`. Strategy independence
  (Goodman 1999 §3) is verified in `test_agenda.py`: under
  idempotent semirings, chart values agree across all
  strategies.

- **panproto integration**: `QVR_DEDUCTION_PROTOCOL` and
  `extract_deduction_schema(compiler)` make deduction systems
  first-class panproto schemas. Schema morphisms over the
  protocol correspond to deduction-system specializations
  (e.g., CCG ⊂ Lambek ⊂ MultimodalLambek).

### Structural compression: signatures, encoders, decoders, losses

The release lands a uniform algebraic interface for compressing
arbitrary structured objects to fixed-length vectors and decoding
them back under a learned distribution. Categorically: every
constructor algebra a user declares is the initial Σ-algebra `T_Σ`
of a multi-sorted signature; a **encoder** is a Σ-algebra
homomorphism `T_Σ → Vec_D`, and a **decoder** is a Kleisli
coalgebra `Vec_D → Kern(T_Σ)`. The recursion / corecursion is
supplied by the framework; the analyst supplies only the
per-operation parametric functions. This single abstraction
subsumes RNN / transformer / tree-LSTM / graph-NN encoders,
autoregressive LM and variational decoders, and the proposal's
vector inside-outside parser (the chart's item signature is Σ, the
parser's `combine` / `split^L,R` are the per-operation encoder
functions, the attention-weighted aggregation lives entirely
inside the encoder, outside the chart's role, so the
semiring abstraction is not broken).

- **Signature blocks** declare sorts, constructors, binders, and
  (for graph signatures) vertex / edge kinds:
  <!-- compile: false -->
  ```qvr
  signature LF
      sorts
          Term : object [dim=64]
          Type : object [dim=32]
          Name : data   [dim=32, vocab=["dog", "cat", "every"]]
      constructors
          Const : Name      -> Term
          App   : Term, Term -> Term
      binders
          Lam : binds (x : Term : ty : Type) in (body : Term) -> Term
  ```
  Three sort kinds: `object` (recursively decoded), `data`
  (opaque raw values; data sorts may declare a closed vocabulary
  via `vocab { … }` of string / integer / float literals),
  `index` (de-Bruijn slots). The reserved `BoundVar` op is a
  built-in de-Bruijn reference; binders thread a typed context Γ
  carrying `(var_sort, embedding, type_term)` per scope entry.
  Binder variables may carry an annotation sort via
  `binds (x : Term : ty : Type)`, the variable's type is
  structurally tracked through the de-Bruijn context.
- **Encoder blocks** declare an F-algebra homomorphism
  `T_Σ → Vec_D`. Per-constructor bodies are user-supplied or
  scaffolded as 2-layer MLPs by the compiler with correct
  per-arg dimensions. Sequence sugar, `Cons(head, tail) recurrent
  state |-> body` for left-folds and `Cons(head, tail) attention
  prefix |-> body` for iterative outside-in walks that thread a
  running prefix list, handles RNN / transformer-shaped
  encoders uniformly. Graph signatures use a
  `message_passing`-shaped body (`init[V]`, `message[E]`,
  `update[V]`, `readout`, `iterations N`).
- **Decoder blocks** are Kleisli coalgebras `Vec_D → Kern(T_Σ)`
  with `sample(vec) -> Term` and `log_prob(term, vec) -> Tensor`.
  Per-sort `structure` / `primitive` / `factor` / `binder_select`
  heads are scaffolded as learnable neural networks; the
  corecursion (structure choice, factor split, recursive descent
  with extended Γ at binders, BoundVar fallback to in-scope
  variables, depth-bounded termination) is supplied by the
  framework. No silent type coercion or sentinel value: an
  observed term whose shape doesn't match the canonical form
  raises with a typed diagnostic.
- **`var_init` per (var_sort, annot_sort) pair**: multiple
  `var_init Term from Type as ty |-> body` declarations per
  encoder, one per pair of sorts the signature's binders
  introduce; the compiler scaffolds defaults for omitted pairs.
- **Stdlib shapes** (`quivers.structural.shapes`): `Seq[A]` with
  `rnn_encoder`, `transformer_encoder`, `bow_encoder`,
  `ar_decoder`; `Tree[L, B]` with `tree_lstm_encoder` and
  `tree_decoder`; `Graph[V, E]` with `graph_signature` and
  `gnn_encoder` (per-edge-kind message MLP, per-vertex-kind
  GRU update, mean / sum / max readout).
- **Deduction integration**: a `deduction` block may declare an
  item `signature` and attach a `encoder`; the chart's
  `embedding(item)` query returns a differentiable vector
  computed by the attached encoder's algebra-homomorphism
  recursion over the chart-item term.
- **Loss attachments** (`loss <name> [weight ...] [on <site>] { body }`):
  attachable at `global`, `program <name>`, `deduction <name>`,
  `encoder <name>`, `decoder <name>`, `rule <name> in <D>`,
  and `chart of <D>` sites. Rule-attached losses fire on every
  rule application during chart construction (the agenda's
  `_fire` path invokes a registered `rule_callback` with the
  full antecedent list); chart-attached losses fire once on the
  completed chart. `LossRegistry.evaluate_on(kind, target, env)`
  returns the weighted partial sum for a given attachment site;
  `ChartView.attached_loss` exposes the accumulated rule + chart
  losses fired during a deduction's run.
- **Optional `export`**: a module with only signatures /
  encoders / decoders / losses (no top-level morphism) now
  compiles into a `Program(None)` container; the artifacts are
  reachable through `prog.signatures` / `prog.encoders` /
  `prog.decoders` / `prog.losses`. The previous `no export
  declaration found` hard error is replaced by a precise
  diagnostic on `forward()`.
- **Strict declaration discipline**: every sort referenced in a
  constructor's domain or a binder's variables / scoped
  arguments / codomain must be declared in the signature's
  `sorts { … }` block, no silent auto-registration. Every sort
  with no inline `dim` must have its dim supplied by every
  encoder / decoder over the signature. Reserved op names
  (`BoundVar`, `Data`) are rejected as user-declared constructors
  or binders. `vocab` clauses are only valid on `data` sorts;
  duplicate vocabulary entries are rejected.
- **Public surface**: `quivers.structural` exports `Signature`,
  `Sort`, `Constructor`, `Binder`, `BinderVarSpec`,
  `BinderArgSpec`, `VertexKind`, `EdgeKind`, `Term`, `Context`,
  `EMPTY_CONTEXT`, `DataLeaf`, `Encoder`, `Decoder`,
  `LossEntry`, `LossRegistry`, `bound_var`, `make_term`.
  `quivers.structural.shapes` exports the canonical sequence /
  tree / graph factories.
- **27 new tests** in `tests/test_structural.py` cover every
  surface form, every strict-rule diagnostic, the typed-binder
  discipline, end-to-end compression and decoding for sequences /
  trees / graphs, rule-attached and chart-attached loss firing,
  per-pair `var_init` overrides, recurrent / attention modes, and
  the data-sort vocabulary pipeline.

### Hierarchical-Bayesian primitives and arrow / effects tower

### Added

- Hierarchical-Bayesian modeling primitives in `quivers.continuous.bayesian`, each carrying its categorical denotation in **Kern**:
  - `PlateDraw(index_size, family, domain)`, finite-domain-indexed draw realized as a Kern-morphism `A → B` by the natural isomorphism `Kern(1, B^A) ≅ Kern(A, B)`; subclass of `ContinuousMorphism` so it threads through the existing `MonadicProgram` step machinery.
  - `VectorisedObserve(family, response)`, batched-observation kernel `Φ → G_{≤1}(Φ)` with score `∏_n p_F(r_obs(n); θ(n, φ))`.
  - `marginalize_categorical(log_probs)`, program-level pushforward through `π_{Φ\C}` realized as `log_sum_exp` over the class axis.
  - `LKJCorrelationFactor(K, eta)`, LKJ prior on `CholeskyFactor(K)` via the Lewandowski-Kurowicka-Joe onion method; analytic `log_prob` matches Stan's `lkj_corr_cholesky_lpdf`.
  - `Truncated(base, lower, upper)`, generic interval-truncation combinator (rejection sampling with Monte-Carlo truncation-mass estimation).
  - `cumsum(K)`, `softmax(K)`, deterministic morphisms for monotone splines and simplex projection.
  - `cholesky_quad_form(K)`, covariance reconstruction `Σ = diag(s) L L^T diag(s)`.
  - `CholeskyFactor(K)` `ContinuousSpace`, manifold of K×K lower-triangular factors of correlation matrices.
- Surface syntax for hierarchical-Bayesian models in `.qvr`:
  - `draw v : A -> K ~ Family(args)`, finite-domain-indexed plate draw.
  - `observe r[n] ~ Family(args) for n in N`, vectorized observation.
  - `marginalize c`, program-level discrete-latent marginalization.
  - `arr[idx]`, Kleisli pullback gather expression inside `let`-bodies.
  - `posterior name (model) : domain -> codomain { steps return ... }`, deterministic post-conditioning block whose body consumes posterior latents.
  - Parametric programs: `program name (G : FinSet, scale : Real, prior : Mor[A, B]) : dom -> cod ...`, programs polymorphic over objects, scalars, and morphisms; denote dependent kernels `Π(p:P).Kern(dom(p), cod(p))`. Instantiated at each call site `draw v ~ name(args)` by parameter substitution + α-renaming, so each call contributes fresh latent factors to the caller's joint kernel. Supports the random-effects reuse story without tying latents across call sites.
  - Let-expression builtins: `cumsum`, `softmax`, `cholesky_quad_form` join the existing `sigmoid` / `exp` / `log` / `abs` / `softplus`.
- AST nodes in `quivers.dsl.ast_nodes`: `PlateDrawStep`, `VectorisedObserveStep`, `MarginalizeStep`, `LetExprIndex`, `PosteriorDecl`; each docstring carries the Kern denotation.
- Stan-model port at `src/quivers/dsl/examples/event_structure.qvr`, a faithful translation of the four-class telicity × durativity latent-class model from `~/Projects/supertelicity/analysis/event-structure-induction/models/event-structure-model.stan`, demonstrating crossed random effects, ordinal monotone splines, vectorized observations, and `marginalize` over the discrete latent class.
- `tests/test_bayesian.py`: 15 tests covering every new primitive and every new AST node's parse / compile round-trip, plus a compile-time smoke test on the Stan-model port.

### Changed

- `_walk_program_step` return type widened from `DrawStep | LetStep` to the `ProgramStep` union root.
- Tree-sitter grammar regenerated (`grammars/qvr/src/parser.c`, `grammar.json`, `node-types.json`) to recognize the new program steps and top-level declarations.

### Added

- Pattern-polymorphic `schema` declarations: `schema r[X, Y : Cat] : (X/Y) * Y -> X`. Subsumes `rule` with explicit parameter types and a unified domain/codomain shape.
- New SetObject variants: `EnumSet(name, elements)` for named-element finite sets, `FreeResiduated(generators, depth, ops)` for residuated category universes. New surface syntax `object Atoms = {NP, S, VP}` and `object Cat = FreeResiduated(Atoms, depth=4, ops=[slash])`.
- `FreeMonoid` surface form: `object Free = FreeMonoid(X, max_length=4)`.
- `TypeSlash` and `TypeEffectApply` `TypeExpr` variants, residuated patterns and effect-typed types are first-class. The `CatPattern` AST family is removed (folded into `TypeExpr`).
- `chart_fold(lex=, binary=, unary=, start=, depth=, effect_depth=, handlers=)` primitive expression, desugared form of `parser(rules=...)`. `unary=` is wired through the inside algorithm's reflexive-transitive unary-rule closure. `handlers=` post-composes effect handlers on the parser output as log-space transition morphisms.
- `.curry_right` / `.curry_left` postfix combinators witnessing the residuation isomorphisms; backed by `quivers.core.morphisms.CurriedMorphism`.
- Typeclass tower in `quivers.monadic.typeclasses`: `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Foldable`, `Traversable`, `MonadTrans`. Concrete monads (`FuzzyPowersetMonad`, `FreeMonoidMonad`, `GiryMonad`) subclass `Monad` directly.
- Stdlib effect instances in `quivers.monadic.instances`: `Identity`, `Maybe`, `Alternative_`, `Continuation`, `State`, `Reader`, `Writer`, `List`. All operations (`pure`, `fmap`, `apply`, `join`, `bind`, `lift_a2`, `empty`, `alt`, `foldr`, `traverse`) are concrete V-relation realizations; function-space-dependent operations encode `[A → B]` as a finite `FinSet` of cardinality `|B|^|A|`. Monad transformers in `quivers.monadic.transformers`: `StateT`, `ReaderT`, `MaybeT`, `ContT`, `WriterT`.
- Algebraic effects + handlers in `quivers.monadic.algebraic`: `Operation`, `EffectSignature`, `Handler`, `FreeMonad`. `FreeMonad` carrier is the bounded-depth signature-tree set realized as a flat `FinSet` with structural decomposition via `_decompose_carrier_index` / `_compose_carrier_index`; `pure`, `fmap`, `join`, `bind`, `lift_a2` satisfy the monad laws up to truncation. `Handler.run` is the post-order tree fold interpreting each leaf through `return_clause` and each operation node through `operation_clauses`. `EffectSignature.to_theory()` and `Handler.as_theory_morphism()` realize the panproto-side theory and theory morphism.
- `quivers.monadic.bridges`: `Kleisli`, `ArrowMonad`, `CoKleisli`, `kleisli`, `arrow_monad`, `cokleisli` connecting the monad and arrow towers. `Kleisli.compose` is fmap-then-join with structural recovery of the underlying B; `Kleisli.first` is realized via the canonical monad strength `σ = (pure × id) >> lift_a2(id_{A⊗B})`; `Kleisli.app` routes through the Applicative apply. `ArrowMonad` provides `fmap/pure/apply/join/bind/lift_a2` via the underlying arrow's `arr`/`id_arr`/`app`/`compose`. `CoKleisli` is registered as `Category_`; promoting to `Arrow` requires an explicit comonad costrength supplied via `first_via_costrength(f, C, costrength)`.
- `quivers.arrows` package, Hughes-style arrow hierarchy (`Category_`, `Arrow`, `ArrowChoice`, `ArrowApply`, `ArrowLoop`, `ArrowZero`, `ArrowPlus`) with panproto-theory mirrors. New `quivers.arrows.instances` with `VRel`, `Function`, `Stochastic` arrow instances; `loop_arr` realized via the V-quantale iterative trace (Joyal-Street-Verity 1996, §3).
- `quivers.stochastic.effect_lifts.class_directed_lifts`: class-driven schema lifting for effect-typed parsers. `make_swap_schema` / `swap_rule_set` emit `swap_TU` schemas from registered `DistributiveLaw` instances for commutation firings.
- `quivers.core._factories` module, concrete morphism constructors `inj`, `case`, `pi`, `pair`, `parallel`, `terminal`, `constant`, `distrib_right`, `coproduct_map`. The algebra on which the stdlib monads, arrows, and algebraic-effects layer are built.
- New tree-sitter grammar at `grammars/qvr/` with regenerated parser; the unified `_type_expr` family subsumes the prior `_cat_pattern` productions.
- Local-grammar override at `quivers.dsl._dev_grammar` (activated by `QVR_USE_LOCAL_GRAMMAR=1`) using panproto 0.47's first-class `AstParserRegistry.override_grammar()` API. Compiles the in-tree grammar and installs it into the standard registry when the upstream `panproto-grammars-all` bundle hasn't yet vendored the latest grammar source.
- `docs/guides/effects.md` and `docs/semantics/effects.md`, user guide and formal denotational layer for the typeclass + algebraic-effects framework.
- `quantifier_scope.qvr` example demonstrating Charlow-style scope-taking via `Continuation`.

### Changed

- The `Monad` ABC is the typeclass-tower one (`quivers.monadic.typeclasses.Monad`); the previous parallel ABC in `quivers.monadic.monads` is removed. Concrete monads provide both the typeclass operations (`pure`, `apply`, `join`) and the Eilenberg–Moore aliases (`unit`, `multiply`).
- `RuleDecl` premises and conclusion are typed at `TypeExpr` (previously `CatPattern`).
- `ObjectDecl` admits both `: type_expr` and `= initializer` forms.
- `parser(...)` infers category atoms from a uniquely-declared `FreeResiduated` in scope when no `categories=` argument is supplied.
- `QVR_PROGRAM_PROTOCOL` extended with `enum_set`, `free_residuated`, `schema_decl` vertex kinds.
- `InsideAlgorithm` accepts an optional `unary` morphism; the chart fills with reflexive-transitive unary-rule closure at each cell.
- `chart_fold(effect_depth>0)` no longer raises `CompileError`; the parameter flows through as informational metadata and the caller-supplied `binary` morphism (typically built via `lift_rule_set` over declared effects) provides the lifted firings. `handlers=` are post-composed via `_ChartHandlerComposite` log-space transitions.
- Denotational-semantics docs: corrected marginalization formulas (proper handling of residual input `Y`), Kleisli composition ordering in `programs.md` (`s_1 ⋄ ⋯ ⋄ s_n ⋄ ret`, not the reverse), scan formula typing in `expressions.md`, profunctor typing in `grammar.md`, the row-stochastic/`column-stochastic` distinction in `morphisms.md`, the `arrow_monad ∘ kleisli ≅ id` natural isomorphism in `effects.md`.

### Fixed

- `FreeMonad` carrier no longer collapses under `CoproductSet` auto-flattening when the leaf type is itself a coproduct; the flat-`FinSet` representation preserves the recursive leaf-vs-operation structure.
- `FreeMonad.lift_a2` is the correct free-monad applicative recursion (bi-depth `(d_a, d_b, d_c)` tracking with proper continuation re-encoding), replacing a prior block-identity rule that misrepresented op-summand handling.
- `FreeMonad.join` splices outer trees correctly through `_carrier_op_offset`, replacing a prior block-identity that mapped op-summand indices without accounting for the differing inner/outer continuation cardinalities.
- `CoKleisli.first` was type-incorrect (`W(A) × C → B × C` vs the required `W(A × C) → B × C`); now registered as `Category_` only, with `first_via_costrength` for promotion to `Arrow` when a comonad costrength is supplied.
- `List.fmap_obj` accepts non-`FinSet` inputs by re-encoding via cardinality; `List(List(A))` now type-checks and the monad laws hold on its own image.
- `TypeCoproduct` was already correctly handled at `resolution.py:119`; documented as such.
- Bridge round-trip claim in `effects.md` §7 corrected from `=` to `≅` (Hughes 2000 proves natural isomorphism via the `1 ⊗ A ≅ A` unitor, not equality).

### Upstream

- panproto/panproto#89 closed and shipped in panproto 0.47.0, first-class runtime grammar override via `AstParserRegistry.override_grammar()`.
- panproto/didactic#38 closed and shipped in didactic 0.7.0, `tuple[Model, ...]` field types accept any `dx.Model` element directly (the workaround tuple-of-`TaggedUnion`-roots is no longer needed).
- panproto/didactic#39 closed and shipped in didactic 0.7.0, `dx.field(opaque=True)` for fields typed at typeclass ABCs (`Monad`, `ArrowApply`, etc.); opaque fields preserve in-process identity through `with_` but drop to `None` on JSON round-trip.

### Dependencies

- `panproto >= 0.47.0` (was `>= 0.45.0`).
- `panproto-grammars-all >= 0.47.0` (was `>= 0.45.0`).
- `didactic >= 0.7.1` (was `>= 0.6.0`).

## [0.3.0] - 2026-04-12

This release provides effects integration: a typeclass +
algebraic-effects substrate underneath the DSL, joint type-and-effect
schema lifting in the chart parser, and a substantially expanded
surface for categorial grammars driven by it.

### Surface

- **Unified type-expression family.** The categorial-pattern
  sublanguage (`CatPattern`, `CatPatternSlash`, `CatPatternProduct`)
  is folded into `TypeExpr`. Slash patterns (`X/Y`, `X\Y`) parse as
  `TypeSlash`; effect-typed applications (`T(X)`, `Cont_S(NP)`) parse
  as `TypeEffectApply`. Rule and schema premises / conclusions are
  typed at `TypeExpr` uniformly.
- **`schema` declarations.** Pattern-polymorphic morphism schemas
  with explicit parameter types and a unified domain / codomain
  shape: `schema forward_app[X, Y : Cat] : (X/Y) * Y -> X`. Arity is
  derived from the domain shape, a 2-component `TypeProduct`
  produces a binary chart-rule, otherwise unary.
- **`EnumSet` and `FreeResiduated` object initializers.**
  `object Atoms = {NP, S, VP}` declares an `EnumSet`;
  `object Cat = FreeResiduated(Atoms, depth=2, ops=[slash])` declares
  the residuated category universe over an `EnumSet` of generators,
  closed under the listed connectives up to a bounded nesting depth.
- **`FreeMonoid` object surface.** `object Strings = FreeMonoid(X, max_length=4)`
  parses, walks, and compiles to a runtime `FreeMonoid` carrier.
- **`chart_fold(...)` primitive.** `chart_fold(lex=…, binary=…,
  unary=…, start=…, depth=…, effect_depth=…)` exposes the inside
  algorithm as a first-class morphism expression. Accepts unary
  binaries via the reflexive-transitive closure of unary chart cells.
- **Residuation-witness combinators.** `.curry_right` and
  `.curry_left` postfix methods realize the right / left
  residuation isomorphism on a binary morphism. Forward / backward
  application become *theorems* derivable from identity + curry
  once joint type-and-effect dispatch fires.
- **`alias` declarations.** Object-shaped aliases (`alias Pair = X *
  Y`) bind a resolved `SetObject` in the compiler environment;
  residuated aliases (`alias VP = S \ NP`) are stored for syntactic
  substitution at schema-pattern use sites. Duplicate-declaration
  and shadowing diagnostics included.
- **`bundle` declarations.** `bundle CCG = [forward_app,
  backward_app, harmonic_composition]` names a tuple of rule
  references that `parser(rules=…)` and `chart_fold(binary=…)`
  splice into the rule list. Bundles can reference other bundles;
  the expander detects cycles.
- **Doc comments.** Lines starting with `##` attach to the next
  declaration that carries a docs field (`object`, `morphism`,
  `schema`, `program`, `alias`, `bundle`). Plain `#` line comments
  continue to be dropped at parse time.

### Categorial-effects substrate

- **Typeclass tower** (`quivers.monadic.typeclasses`). `Functor` /
  `Applicative` / `Monad` / `Alternative` / `MonadPlus` /
  `Foldable` / `Traversable` / `MonadTrans`. Each ABC documents its
  laws; the runtime law-check scaffold lives in `laws.py`.
- **Arrow tower** (`quivers.arrows`). `Category_` / `Arrow` /
  `ArrowChoice` / `ArrowApply` / `ArrowLoop` / `ArrowZero` /
  `ArrowPlus` (Hughes 2000 §3). `ArrowApply` is bridged to `Monad`;
  `ArrowLoop` is the denotational target of `chart_fold`'s loop.
- **Stdlib effect instances** (`quivers.monadic.instances`).
  `Identity`, `Maybe`, `Alternative_` (Hamblin powerset),
  `Continuation(answer)`, `State(state)`, `Reader(env)`,
  `Writer(monoid)`, `List(max_length)`. Each is a `dx.Model`
  registered against its appropriate ABC via `ABC.register(...)`,
  with concrete V-relation realizations of `pure` / `fmap` /
  `apply` / `join` / `bind` / `lift_a2`.
- **Monad transformers** (`quivers.monadic.transformers`).
  `StateT(state)`, `ReaderT(env)`, `MaybeT`, `ContT(answer)`,
  `WriterT(monoid)`.
- **Algebraic effects + handlers** (`quivers.monadic.algebraic`).
  `Operation`, `EffectSignature`, `Handler`, `FreeMonad(signature)`
  with a bounded-depth flat-FinSet carrier that preserves the
  recursive leaf-vs-operation structure. `Handler.run` is a
  post-order tree fold; a handler is a panproto theory morphism
  from `sig.to_theory()` into the target monad's theory.
- **Bridges** (`quivers.monadic.bridges`). `Kleisli(monad)` wraps a
  `Monad` as `Arrow` / `ArrowApply` with concrete strength and app;
  `ArrowMonad(arrow)` wraps an `ArrowApply` as a `Monad`. The
  Joyal–Street–Verity iterative trace realizes `loop_arr`.

### Chart parser

- **Class-directed lifts** (`quivers.stochastic.effect_lifts.class_directed_lifts`).
  Given a base `SchemaDecl` and an effect, returns the tuple of
  lifted schemas keyed by the effect's typeclass interface:
  `Applicative` → `pure_T` / `apply_T`; `Monad` → adds `bind_T`
  (Charlow's scope-extruding lift); `Alternative` → adds `alt_T`;
  `MonadPlus` is their union. Dispatch is on the typeclass, never
  the effect's identity, so adding a new effect or a new typeclass
  extends the lifting machinery automatically.
- **`lift_rule_set(base, effects)`** applies the dispatch matrix
  pairwise across a rule-set and an effect-stack, returning the
  union of base + lifted schemas; consumed uniformly by the
  chart-fold runtime.
- **`chart_fold` end-to-end.** The compiler path drops the previous
  `effect_depth > 0` guard, composes handlers as log-space
  transitions on the parser output, and emits `swap_TU` schemas
  from registered `DistributiveLaw` instances.

### Tooling

- **`qvr check` CLI** (`quivers.cli.{__init__,check}`). Parse +
  constraint-solver + compile pipeline over a list of `.qvr` files;
  structured JSON output via `--json`; exit codes `0` / `1` / `2`
  for clean / error / usage. Registered as the `qvr` console script.
- **Constraint solver** (`quivers.dsl.constraints.check_constraints`).
  Walks the parsed AST and reports `residuated_constraint`,
  `effect_constraint`, and `bundle_unknown_member` violations
  without invoking the full compiler. Surfaced via `qvr check`.
- **Pygments lexer** (`quivers.dsl.pygments_lexer`). Driven by the
  in-tree tree-sitter parser via the `_dev_grammar` shim, so the
  highlighter always reflects the authoritative grammar.
- **Highlight queries** (`grammars/qvr/queries/highlights.scm`)
  refreshed for `schema_decl`, `type_slash`, `type_effect_apply`,
  `chart_fold`, `curry_right` / `curry_left`, `EnumSet` /
  `FreeResiduated` / `FreeMonoid`, `alias_decl`, `bundle_decl`.
- **GitHub linguist** (`.gitattributes`). `.qvr` classified as
  detectable; tree-sitter generated artefacts marked vendored /
  generated.

### Examples + documentation

- **`quantifier_scope.qvr`**: Charlow-style scope-taking grammar
  using the `Continuation` effect, exercising the new surface
  (`EnumSet`, `FreeResiduated`, `schema`, `Cont_S(X)`) end-to-end.
- **`docs/guides/effects.md`**: typeclass + algebraic-effects
  framework: monad and arrow towers, stdlib effect instances,
  class-driven schema lifting, joint type-and-effect dispatch,
  bridges between the two towers.
- **`docs/semantics/effects.md`**: denotational layer: panproto-
  theory mirrors of each typeclass, effect-typed schema
  denotations as natural transformations, joint type-and-effect
  chart dispatch, lifting-adequacy theorem, conservativity over
  the bare-grammar fragment.
- **`docs/getting-started/architecture.md`**: `quivers/arrows/`
  section added; `quivers/monadic/` rewritten to cover the
  typeclass spine, transformers, algebraic effects, bridges,
  theories, laws.
- **mkdocstrings stubs** for every new public module:
  `quivers.monadic.{typeclasses,instances,transformers,algebraic,bridges,theories,laws}`,
  `quivers.arrows.{typeclasses,theories}`,
  `quivers.stochastic.effect_lifts`.

### Internal

- **Program theory** (`quivers.dsl.program_theory`). Vertex kinds
  extended with `enum_set`, `free_residuated`, `schema_decl`.
  `write_set_object` branches for `EnumSet` (name + element
  constraints) and `FreeResiduated` (depth + op constraints +
  `generators` edge to an `enum_set`).
- **Local-grammar override** (`quivers.dsl._dev_grammar`). Adopts
  panproto 0.47's first-class `AstParserRegistry.override_grammar`
  API, replacing the previous ctypes-pinned-buffer shim. Activated
  by `QVR_USE_LOCAL_GRAMMAR=1` until `panproto-grammars-all` vendors
  the new QVR grammar.
- **`quivers.core._factories`**: the concrete morphism alphabet
  (coproduct injection / case eliminator, product projection /
  pairing, parallel pair, distributivity, terminal, constant,
  coproduct functorial action) on which the typeclass realizations
  are built.
- **Legacy `Monad` ABC removed.** `quivers.monadic.monads` no
  longer ships a parallel `Monad`; `FuzzyPowersetMonad`,
  `FreeMonoidMonad`, and `GiryMonad` subclass the typeclass `Monad`
  directly. The Eilenberg–Moore aliases (`unit`, `multiply`,
  `kleisli_compose`) are kept as convenience methods.
  `FreeMonoidMonad.join` now does the full word-concatenation
  tensor with truncation beyond `max_length`.

### Dependencies

- `panproto >= 0.47.0` (was `>= 0.45.0`).
- `panproto-grammars-all >= 0.47.0` (was `>= 0.45.0`).
- `didactic >= 0.7.0` (was `>= 0.6.0`).

### Tests

- 986 tests pass under `QVR_USE_LOCAL_GRAMMAR=1`. New surface
  coverage: 32 cases in `tests/test_dsl_extensions.py` (aliases,
  bundles, doc comments, FreeMonoid surface, constraint solver,
  `qvr check` CLI, highlight-query parity, Pygments lexer
  round-trips, grammar-shape parity across every example).
- Monad unit laws hold on `Identity`, `Maybe`, `Alternative_`,
  `State`, `Reader`, `Writer`; `FreeMonad` left / right unit laws
  hold up to truncation; trace yanking and identity verified on
  `VRel`.

### Key citations

- Charlow, S. (2025). [*Static and dynamic exceptional scope*](https://doi.org/10.1093/jos/ffad012). Journal of Semantics, 42(4), 353–398.
- Bumford, D. and Charlow, S. (forthcoming, 2026). [*Effect-Driven Interpretation: Functors for Natural Language Composition*](https://www.cambridge.org/core/elements/abs/effectdriven-interpretation/56671E539160AAA1DACF8555B82A2FE4). Elements in Semantics, Cambridge University Press. Online ISBN 9781009285377; preprint [arXiv:2504.00316](https://arxiv.org/abs/2504.00316).
- Hughes, J. (2000). [*Generalising monads to arrows*](https://doi.org/10.1016/S0167-6423(99)00023-4). Science of Computer Programming 37(1–3):67–111.
- Bauer, A. and Pretnar, M. (2015). [*Programming with algebraic effects and handlers*](https://doi.org/10.1016/j.jlamp.2014.02.001). Journal of Logical and Algebraic Methods in Programming 84(1):108–123.
- Plotkin, G. and Power, J. (2003). [*Algebraic operations and generic effects*](https://doi.org/10.1023/A:1023064908962). Applied Categorical Structures 11(1):69–94.
- Joyal, A., Street, R., and Verity, D. (1996). [*Traced monoidal categories*](https://doi.org/10.1017/S0305004100074338). Mathematical Proceedings of the Cambridge Philosophical Society 119(3):447–468.
- McBride, C. and Paterson, R. (2008). [*Applicative programming with effects*](https://doi.org/10.1017/S0956796807006326). Journal of Functional Programming 18(1):1–13.

## [0.2.0] - 2026-05-06

### Changed

- Every record-shaped value type (AST nodes, `FinSet`, `ProductSet`, `CoproductSet`, `ContinuousSpace` variants, `Category` variants, `RuleSystem`) is now a `didactic.api.Model`. Recursive sums are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field. JSON round-trips via `model_dump_json` / `model_validate_json` are available on every value type.
- Resolution from `TypeExpr` / `SpaceExpr` AST trees to runtime `SetObject` / `ContinuousSpace` values is expressed as a `dx.Lens` family in `quivers.dsl.resolution`.
- Variadic constructors `ProductSet(A, B, C)` and `CoproductSet(A, B, C)` are replaced by keyword form `ProductSet(components=(A, B, C))` and `CoproductSet(components=(A, B, C))`. The flattening converter preserves the previous flattening behavior.
- Continuous spaces (`Euclidean`, `Simplex`, `PositiveReals`, `ProductSpace`) expose public `name` and `dim` fields (no longer private with property accessors).
- Minimum supported Python is now 3.14.

### Added

- A tree-sitter grammar for the QVR DSL at `grammars/qvr/`, registered with panproto's `panproto-grammars-all` distribution.
- `quivers.dsl.parser` delegates parsing to panproto via the `qvr` tree-sitter grammar.
- `quivers.dsl.program_theory` defines `QVR_PROGRAM_PROTOCOL` and `extract_program_schema`, lifting every compiled program to a panproto `Schema` for use with `panproto schema diff`, `panproto lens generate`, and related tooling.
- A `Denotational Semantics` documentation section giving a formal, compositional semantics for the DSL across the discrete, stochastic, and continuous strata, plus an adequacy theorem connecting the compiler implementation to the denotation.
- `RuleSystem` carries cross-field axioms (`__axioms__`) ensuring `binary_weights`/`unary_weights` lengths match `binary_rules`/`unary_rules` when supplied.
- `.github/workflows/release.yml` builds an sdist + wheel on tag push and publishes to PyPI via the quivers-dev/quivers OIDC trusted publisher.
- Pull request template under `.github/PULL_REQUEST_TEMPLATE.md` and issue templates under `.github/ISSUE_TEMPLATE/`.

### Removed

- `quivers.dsl.lexer` and `quivers.dsl.tokens`: the hand-written lexer is replaced by panproto's tree-sitter–integrated lexing.
- `LexError`: lexical errors now surface as `ParseError`.

## [0.1.0] - 2026-03-26

### Added

#### Core Categorical Algebra

- Fundamental category types and morphisms
- Object declarations and morphism composition
- Support for latent and observed morphisms
- Basic categorical operations and abstractions

#### Stochastic Morphisms

- Stochastic morphism declarations and semantics
- Integration with probability theory
- Support for morphism composition in stochastic settings

#### Continuous Distributions (30+ Families)

- Normal distribution and variants (LogitNormal, TruncatedNormal)
- Beta, Dirichlet for probability simplices
- Exponential family: Exponential, Gamma, Chi2
- Heavy-tailed: Cauchy, StudentT, Pareto
- Bounded: Uniform, Kumaraswamy
- Half-variants: HalfCauchy, HalfNormal
- Transformed: LogNormal, Gumbel, Laplace, Weibull
- Multivariate: MultivariateNormal, LowRankMVN, Wishart
- Bernoulli variants: Bernoulli, ContinuousBernoulli, RelaxedBernoulli
- Advanced: RelaxedOneHotCategorical, FisherSnedecor
- Normalized flows: Flow
- Categorical and discrete approximations

#### Monadic Programs

- Draw statements for sampling from morphisms
- Observe statements for conditioning and likelihood
- Return statements with optional labeled outputs
- Variable binding and destructuring in patterns
- Program parameters and composition

#### QVR DSL

- Complete lexer with token recognition for all language constructs
- Recursive descent parser with full grammar support
- Abstract syntax tree (AST) node definitions
- Program block execution with proper scoping
- Let bindings for expression computation
- Built-in let functions: sigmoid, exp, log, abs, softplus
- Comment support (#)
- Type expressions: products (*), coproducts (+)
- Expression operators: composition (>>), tensor product (@), marginalization
- Indentation-aware program body parsing
- Specialized handling for draw/observe arguments

#### Variational Inference Layer

- Inference interface for probabilistic programs
- Support for approximate posterior computation
- Integration with continuous distribution families
