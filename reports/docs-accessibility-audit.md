# Documentation accessibility audit

November 6, 2026

Three agents audited the entry points and learning material, the guides and QVR reference, and the semantics and API reference. Their initial findings appear below. The revisions add concrete calculations, explain notation where it first appears, and retain formal definitions and implementation limits. The historical changelog is retained as a record of earlier releases.

The main changes are:

1. A concepts primer works from finite probability tables to composition, conditioning, monads, and handlers.
2. Tutorials, guides, and examples explain what each operation computes and how to interpret its output. Source/model mismatches are documented precisely, including the document-level topic mixture and the negative-binomial parameterization.
3. Semantics pages state the assumptions needed for each result. Corrections cover finite versus measurable distribution spaces, distributivity, symmetric monoidal structure, normalized distributions, and the scope of finite tensor constructions.
4. API pages introduce the operation before the generated signatures. A final prose pass replaces repeated generic cautions with concrete equations, examples, and checks.
5. Citations render from `docs/references.bib`. Each cited page includes linked references; unknown citation keys fail the build. The companion citation verification report identifies publisher evidence and unresolved access limits.

The source-module edits change docstrings only. A comparison of Python abstract syntax trees with docstrings removed confirms that runtime definitions are unchanged.

## Initial audit reports

These reports preserve the findings that guided the revisions. Statements describing a defect refer to the documentation before this revision, except where a remaining limitation is explicitly identified in the verification report.

## Accessibility audit: entry points, tutorials, examples, developer docs

Scope: README.md, docs/index.md, four getting-started pages, all 24 tutorial pages, all 48 example Markdown pages, five developer pages, and transpile-support.md. Changelog is historical; generated measurements must be changed through their generator. Audit covers explanatory prose, equations, code context, and selected implementation checks; it is not a full execution/citation audit.

### Priorities

1. Establish the problem before naming its categorical or inference machinery. Give readers a concrete input, intermediate value, output, and reason for each operation. Link a common primer, but do not substitute a link for local explanation.
2. Preserve distinctions that intuitive shortcuts currently blur: tensors versus distribution kernels, learned parameters versus sampled latents, draws versus inference, deterministic dependence versus a constant, row conditioning versus Bayes inversion, and typed operations versus proved categorical laws.
3. Explain what output can establish. Short SVI/MCMC demonstrations establish execution, not convergence; chart weights are not automatically normalized sentence probabilities; a model's name may describe inspiration rather than implemented architecture.
4. Replace explicit PPL/category prerequisites with basic Python/tensor/probability prerequisites and reading routes. Retain equations and technical terms after the concrete example.
5. Remove unnecessary invented acronyms (LIS, RPS, DPV, TAP, CDC, FGP) per the user's clarification. Keep standard names and actual product identifiers.

### Entry points and getting started

| Page | Findings and revision needed |
|---|---|
| README.md | First paragraph and table front-load typed functional PPL, indexed effects, enrichment, and SVI. Explain modeling unknown quantities and relating typed computations; separate ordinary fitting from extension work. The first model is a stochastic GRU, far too advanced as an unqualified quick start; provide a short regression route and explain which part is simulation/checking/inference without deleting the existing showcase. |
| docs/index.md | Strong navigation but hero and composition paragraph assume denotation, enriched categories, signatures, and kernels. Give a familiar regression or typed-table story before those terms, connect each of the three levels to one concrete question, and mark finite tensor morphisms as one component rather than every program. |
| getting-started/quickstart.md | Opens with V-enriched relation, proceeds through monadic steps, then jumps to coends/Yoneda. Explain row/column meanings and numerical two-step composition before operators; introduce monadic program as sample-then-use sequencing; distinguish trainable tensor loss from Bayesian inference. Make enriched structures an optional extension and show what outputs mean. |
| getting-started/installation.md | Mostly accessible. State which install path a model user needs, what extras buy them, that parser dependencies are automatic, and that full tests require a checkout/development context. Keep detailed packaging provenance for contributors. |
| getting-started/highlighting.md | Explain syntax colors versus semantic checking and why a highlighted model can still be invalid. Give reader-facing choice of editor first; preserve grammar provenance and setup details. Advanced token-check example needs purpose, not implied prerequisite. |
| getting-started/architecture.md | Inventory is comprehensive but opens with subpackages instead of a model's path. Add source → checked representation → runtime/inference trace, explain distinction between compiler and inference, and provide task-to-package navigation. Define tensor arrow, kernel, handler, and attachment before module catalogs. |

### Tutorials

| Page | Findings and revision needed |
|---|---|
| tutorials/index.md | Explicitly requires familiarity with another PPL and working category knowledge. Remove those gates; give newcomer routes and say comparisons are optional. Explain QVR versus Python choice by task rather than jargon. |
| python/01-first-quiver.md | Starts with lattice/t-norm. Explain a relation table and one entry, why typed dimensions matter even when sizes coincide, and noisy-OR with two numerical routes. Distinguish fuzzy scores from probabilities and warn that a runtime Algebra label alone does not justify reassociation. |
| python/02-stochastic-relations.md | Add weather→activity two-route total-probability calculation. Correct claim that conditioning rows and pushing a prior yields posterior over input; input posterior needs prior times evidence likelihood and normalization. Zero-evidence rows remain zeros under implementation clamp, not distributions. DiscretizedNormal uses normalized center densities, not exact interval masses. |
| python/03-probabilistic-programs.md | Explain sample then use its realized value; deterministic let can still be random across executions. Define trace joint versus output marginal using hidden z. Preserve log_prob unavailability and host-data distinction; disambiguate two showcased steps from the broader step vocabulary. |
| python/04-fuzzy-factorization.md | Existing some/through explanation is good. Add concrete users/features/items or object/property example, why bottleneck encourages reusable structure, and interpret BCE and nonunique factors. Independence is an interpretation of noisy-OR, not inferred evidence. |
| python/05-variational-inference.md | Define posterior by comparing candidate z values and explain guide as an adjustable distribution, ELBO as joint-fit minus guide log density, negative sign, particles, and posterior versus replicated response. Preserve existing warning that conditioned forward draws are not posterior draws. |
| python/06-first-class-trans.md | Lead with same numbers interpreted differently, row normalization versus entrywise conversion, and one numeric softmax/Bayes example. Strong opening that all algebra homomorphisms are lax lattice maps overclaims runtime contract. Round trip restores algebra tag, not values. |
| python/07-composition-rules.md | Explain what a law permits (rebracketing/identity) before hierarchy. Reichenbach statement that no left identity exists is false: 1→b=b, but no two-sided identity and nonassociative. Separate generic library BilinearForm from mathematical bilinearity; attention is generally nonlinear. Preserve smoke-check limitation. |
| python/08-analysis-pipelines.md | Formula notation and random effects need interpretation before fitting; introduce uncertainty on coefficients, posterior draw axes, partial pooling, and predictive comparison separately. Replace categorical claims about 'fastest', universal recovery, and 'empirical truth table' with measured-scope wording. |
| python/09-debugging.md | Organize failures by parse/check/runtime/statistical inference, explain log density and invalid support, negative ELBO sign, and diagnostic symptoms as evidence not unique diagnoses. Free host-data names are not necessarily compile errors. |
| qvr/01-first-model.md | Model starts in equations and assumes PyMC familiarity. Explain slope/intercept/noise, prior/likelihood/posterior, host covariate versus response, sample versus inference, and what a posterior interval says. Annotate SVI components and Predictive shapes/results. |
| qvr/02-glms.md | Explain why binary/count outcomes require constrained distribution parameters and inverse links, with sigmoid(0)=0.5 and exp(0)=1. Link function acts on a response parameter (often mean), not always the natural parameter. Deterministic let may define a derived random variable. |
| qvr/03-hierarchical.md | Explain partial pooling as noisy group estimates borrowing population information; use tau near zero/large to explain funnel and noncentering. Replace 'Non-centered fixes it' and exact-posterior NUTS claims with conditional statements; define diagnostics in plain terms. |
| qvr/04-marginalize.md | Start with two candidate component explanations, weighted likelihood sum, and log-sum-exp. Explain grouped sum-inside-mixture versus per-row mixture as different models. Current grouped Stan analogy can imply wrong order; show formula directly. Retain exactness limited to finite latent, not whole posterior. |
| qvr/05-time-series.md | Define dependence across time versus independent indexed repetition, what state stores, scan final-state output, and observed versus hidden lag. Preserve prominent limitations: fuzzy powering not HMM forward; separate recognition path not analytic Kalman filter. |
| qvr/06-inference-zoo.md | Counts/catalog before intuition overwhelms. Explain optimization approximation versus correlated sampling, guide capacity versus objective versus gradient estimator. NUTS is not automatic cure for multimodality; benchmark passes are not universal algorithm advice. |
| qvr/07-categorical.md | Dense setting paragraph tries to teach all enrichment at once. Start typed arrows and function composition, then relation tables with alternative-path arithmetic; give numeric Boolean/probability/fuzzy examples. Distinguish categories satisfying laws from runtime interfaces. Bayes normalization and change_base need operational interpretation. |
| qvr/08-diagnostics-and-comparison.md | Explain three distinct questions: sampler reliability, model adequacy, predictive comparison. Define chains/ESS/MCSE in ordinary language; DataTree groups need concrete purposes. Stacking weights are not posterior model probabilities; predictive p-values are not hypothesis-test p-values. |
| qvr/09-indexed-data.md | Good motivating invariant but constructor/motive/skolem syntax arrives unexplained. Trace empty/two-element vector through constructor and case; explain type index versus runtime length. Remove LIS coinage. |
| qvr/10-effects-and-handlers.md | Explain perform as request and resume as supplying answer before open rows/grades; walk shrink(8)→4→continuation. Explain interface versus lexical instance versus handler. Remove RPS coinage; preserve deep-handler and dynamic-grade caveats. |
| qvr/11-parsing-and-search.md | Define chart/span/derivation/inside score with 'the dog sleeps'; explain why two derivations add and why log weights need not normalize sentences. Explain optimized chart versus reference search role. Remove DPV coinage. |
| qvr/12-structural-autoencoders.md | Explain tree→vector→distribution over trees before typed attachment path. Define constructor/binder and how reconstruction NLL rewards original tree; parameter registration makes training/checkpoint complete. Remove TAP coinage. |
| qvr/13-release-workflow.md | Explain each guarantee separately: checks, execution, emission, target parsing, density agreement. Authored handler is tutorial 10, not previous tutorial. Remove CDC coinage; preserve exact capability constraints. |
| qvr/14-collection-programs.md | Strong small example. Add explicit Bayes normalized three-weight calculation, log-likelihood summation intuition, grid exactness versus continuous approximation, and map/traverse rationale. Remove FGP coinage. |

### Examples

Every example should identify what the reader supplies, what remains unknown, what output to inspect, and which parts are illustrative. Keep source/code and precise limitations; add local explanations rather than repetitive generic prefatory prose.

| Page | Page-specific findings and revision needed |
|---|---|
| examples/index.md | Add beginner progression from regression to mixtures/grouping and optional structural path; translate feature-map jargon. Correct LDA/shared-topic and ZIP wording to match actual pages. |
| amortized-bayesian-semantics.md | Explain one trial's participant/item/group lookup, why ambiguity is summed, and each component's distinct role. Do not imply missing branch is posterior predictive without conditioned parameters. Remove unnecessary model acronym; preserve mixed-runtime limits. |
| ar1.md | Explain phi as carry-over fraction and intercept/stationary mean distinction. Supplied lag means conditional regression, first-row boundary matters. Replace promise ELBO approaches oracle likelihood and short run yields usable posterior. Translate Kleisli paragraph into sequential draws/score. |
| bayesian-regression.md | Put model story before source; explain coefficient/noise uncertainty and output interpretation. Observe adds likelihood weight, does not itself compute normalized posterior. Weighted-measure qualification needed in categorical perspective. |
| beta-binomial-ab-test.md | Explain arm mean c1/(c1+c0), concentration controlling batch variation, and comparison distribution of arm differences. Keep no shared hyperprior/no cross-arm pooling caveat. |
| beta-regression.md | Explain fraction response, endpoint exclusion, mean versus precision, and flat row→output lookup with one cell. Posterior slope is on log-odds of mean, not raw response. Translate pullback into gathering. |
| bidirectional-rnn-lm.md | Explain token→embedding→two summaries→merge→prediction; preserve both branches left-to-right caveat. Distinguish fixed-hidden synthetic scoring from marginal hidden inference. Categorical output is a distribution, scoring yields weighted computation. |
| bnn.md | Useful prior distinction already present. Explain uncertainty in weights versus observation noise, learned mean/scale, and why MLP layers matter. Preserve point fitting versus lifted Bayesian run and very short sampling caveat. |
| ccg.md | Work one forward application (the:N P/N + dog:N), explain chart interval and alternate derivations; separate unconstrained weights from normalized probabilities and standalone rule-count prior from grammar fitting. |
| changepoint.md | Explain before/after rate weighted by sigmoid, smooth transition rather than hard switch, and posterior on switch time. Hard shared change-point must sum likelihood over time before marginalizing one common switch. Remove 'usable posterior' promise. |
| continuous-hmm.md | Explain state versus measurement, transition versus emission, and what clamped true-state fit learns. Preserve separately learned filter caveat; ELBO is a bound, not parameter-marginal log likelihood itself. |
| custom-rules.md | Walk substitution on one concrete span pair, repeated variable means equality. Explain semiring combination with two derivations. Resolve arbitrary-arity versus unary/binary contradiction; qualify strategy/fixed-point claims by convergence assumptions. |
| deep-markov.md | Explain neural transition/emission as nonlinear conditional parameters and why intermediary draws matter. Preserve recognition not connected and training uses true states; avoid claiming composite marginal density tractable. |
| factor-analysis.md | Explain shared hidden factors with one reconstructed observation; replace very long axis-role paragraph with compact mapping and example. Preserve isotropic noise distinction and rotation-invariant interpretation. |
| gamma-regression.md | Explain positive totals, shape/rate mean and variance, category lookup and optional normalization. Positive support excludes zero; normalizing observed totals changes modeled data and is not automatic Dirichlet regression fitting. |
| gru-lm.md | Explain gates as proportions retained/replaced; stochastic gates are sampled, table deterministic equalities are only rough motivation. Preserve simplified candidate and terminal prediction scope. |
| half-student-t-hierarchical.md | Explain group scale versus residual scale and heavy-tailed prior versus heavy-tailed likelihood. Existing paragraph falsely suggests scale prior absorbs outliers, contradicted later; correct. Explain partial pooling and remove oracle-loss promise. |
| hmm.md | Explain paths through finite states, fuzzy score versus normalized transition probability, and distinguish two exported constructions. Preserve unused transition_rows and single shared state in probabilistic body. |
| horseshoe-regression.md | Explain global shrinkage and local escape with numeric factor interpretation; coefficients are near zero, not exactly sparse. Preserve one-coefficient-per-row design caveat and interpret posterior predictions instead of raw local scales. |
| irt-2pl.md | Explain ability=difficulty→p=0.5; discrimination controls curve steepness. Replace axis-role wall with two concrete respondent/item indices; discuss relative scale anchored by priors and uncertainty. |
| kumaraswamy-bounded-outcome.md | Explain bounded outcomes and quantile as inverse probability-to-value map, uniform a=b=1 sanity check, slope affects shape not mean directly. Preserve open interval and mean formula. |
| lda.md | Serious contradiction: overview/source comments say token-specific LDA, actual over=Doc marginal shares one topic per document. Present source as document-level topic mixture, contrast sum_k product_n with product_n sum_k, and interpret theta accordingly. Correct later per-word claims; retain canonical LDA background only as contrast. |
| linear-gaussian-ssm.md | Explain transition/emission/driver story and terminal scan output; distinguish classical equations from input-dependent-scale implementation and clamped-state fit. Retain not Kalman filter and remove short usable-posterior promise. |
| lkj-cholesky-correlation.md | Explain why arbitrary pairwise correlations can be invalid and L L^T guarantees valid correlation structure. Distinguish correlation from covariance, prior draws from data-informed posterior, eta concentration. |
| logistic-noise-regression.md | Clarify continuous logistic noise versus binary logistic regression; explain scale versus SD and quantile simulation. Avoid unqualified 1.6 shoulder claim and guaranteed loss ordering against oracle. |
| lstm-lm.md | Explain gate products and local memory update; table deterministic gate equations conflict sampled LogitNormal. Preserve missing separately threaded cell state, final-token scope, and parameter-lift role. |
| mixture-model.md | Work weighted two-component density; explain weights sum one, component versus row, exact label integration versus approximate parameter inference, label switching. Correct categorical Resp→Resp value-space confusion. |
| montague-nli.md | Walk one sentence→Every(DOG,BARK)→Claim and one proof before lambda/alpha machinery; explain no proof versus low weight. Preserve existential-import premise and no beta reduction. 'Calibrated probability' not established by toy fit. |
| multimodal-tlg.md | Define modality as typed wrapper restricting allowed rules; source's two rules merely add/remove it. Keep distinction from full multimodal calculus, explain cycles/depth and weighted-score interpretation. |
| negbin-regression.md | Explain count overdispersion and effect of dispersion. Verify parameter convention: source uses dispersion/(dispersion+mu), while torch NegativeBinomial mean uses probs/(1-probs), so current mean/variance claims may be reversed. Do not silently change model in prose revision; document actual behavior or coordinate correction. |
| parametric-pooling.md | Explain reusable template as repeated local model with fresh sites, spread controls pooling, and score as soft preference rather than exact constraint. Scored measures need not be subprobabilities in general. |
| pcfg.md | Work simple sentence parse/inside sum, distinguish productions from weighted chart and normalized PCFG. Learnable log weights not automatically production probabilities; generic agenda strategies need assumptions. |
| pmcfg.md | Existing gap illustration helpful; explain tuple components as separated substring pieces, rank versus number of premises. Derivation log weights add (not product of log weights); keep probability-versus-weight distinction. |
| pmf.md | Explain user/item features and one dot-product predicted rating. Show transpose only aligns axes, not Bayes inverse. Dense demo scores all cells; sparse missing ratings require gather/mask. Retain factor nonidentifiability/prior separation. |
| ppca.md | Explain compression/reconstruction and same noise in every observed dimension; split axis-role wall; explain covariance recovery and rotations. Composite is ComposedMorphism, not LatentMorphism. |
| quantifier-scope.md | Explain two readings of every...a... before continuation notation; actual vocabulary includes one quantifier and does not demonstrate scope ambiguity end-to-end. Explain Cont as pending context; qualify theorem claims for finite rule fragment. |
| schema-chart-parser.md | Explain schema as reusable rule template, bundle as list, depth as inventory bound; one NP/N+N match. Inside weight unnormalized; output scalar interpretation needed. Preserve reference/classic comparison and parameter-store caveat. |
| seq2seq.md | Explain two inputs, independent branches and merge, flattened paired-token computation versus sequence attention; retain no masks/no attention. Clarify what hidden h is conditioned on in each fit. |
| stochastic-volatility.md | Explain changing spread around zero and exp(h/2), supplied latent lags, target conditional on those lags. Preserve not recursively inferred state chain; interpret intervals on volatility not just phi. |
| survival-weibull.md | Define survival time and hazard plainly, shape controls duration dependence, positive beta raises hazard. Preserve uncensored-only likelihood; censored event contributes survival probability rather than exact-time density. |
| tensor-contraction.md | Explain one score from all pairs of feature coordinates; annotate surviving versus summed axes with a tiny numerical calculation. Separate bare tensor and added probabilistic model; retain invariance caveat. |
| term-autoencoder.md | Explain recursive tree compression and probabilistic reconstruction with tiny App tree; define binder/de Bruijn index in ordinary language. NLL rewards original tree, decoder not exact inverse; preserve lazy parameter materialization and attachment limits. |
| transformer-lm.md | Explain same-input parallel branches, independent stack layers, one-token prediction; preserve no attention/causal modeling. Avoid presenting short falling ELBO as posterior recovery. |
| tree-categorical.md | Explain branch path probability products and deterministic tensor assembly; clarify factor construct versus density factor. Observations depend only cell[0,0], so most latents remain uninformed, not all recoverable. |
| type-logical.md | Explain slash as need for neighbor, tensor as paired resource, and one application. Preserve weakening projection caveat; weighted scores not automatically normalized probabilities. |
| vae.md | Explain generation from prior versus reconstruction from encoded observation and amortization; maintain current fit does not train encoder guide. Remove final claim ELBO here measures encoder reconstruction, contradicting actual path. |
| vanilla-rnn-lm.md | Explain hidden state as running summary and terminal next-token prediction, stochastic update versus deterministic RNN. Existing overview incorrectly says head reads each hidden state; actual scan returns terminal state. Preserve runtime trajectory/export limits. |
| zip-regression.md | Explain excess-zero branch plus ordinary Poisson zeros with p(0)=(1-pi)+pi exp(-rate); pi is active rather than zero probability. Preserve unusual reference endpoint-enumeration contract and distinguish forward relaxed draw. |

### Developer and generated documentation

| Page | Findings and revision needed |
|---|---|
| developer/contributing.md | Explain compiler stages by what each validates, AST as syntax tree, elaboration as explicit core expansion, and density probe purpose. Give task-to-test map before module detail; preserve commands/contracts. |
| developer/public-api.md | Already precise and comparatively accessible. Add user/contributor interpretation and one public-package/leaf example; explain re-export identity and why generated API coverage matters. |
| developer/inference-benchmarks.md | Explain mean recovery versus full distribution recovery, tolerance and capture-PASS inversion with a worked row. Throughput units incomparable across SVI/MCMC; benchmark grid covers five listed algorithms not all shipped choices. Generated results and thresholds should stay untouched. |
| developer/qiec.md | Needs vocabulary ladder for stable core, attachment, telescope, refinable index, row, continuation, lexical/deep handlers; trace one sample/observe program through lowering. Preserve full ABI detail. Last paragraph falsely says no recursive calls/scoped instances/authored clauses, contradicting working examples in same document; remove stale limits. |
| developer/changelog.md | Historical generated/synchronized record; no accessibility rewrite. |
| transpile-support.md | Generated file: improve generator prose only. Explain transpile as translating program to another PPL, yes=bytes emitted (not executed/verified density), fixture versus user's call graph, and decode one capability code. Preserve every measurement/refusal. |

### Source checks / rigor flags for revision

- `src/quivers/stochastic/transforms.py` normalizes each conditioned row with denominator clamped at EPS. No input posterior computed; zero mass row remains zero.
- `docs/examples/source/lda.qvr` explicitly sets `[over=Doc]` and `[via=word_idx]`: one class per document is executable semantics despite comments describing canonical LDA.
- `src/quivers/continuous/families.py` builds `ConditionalNegativeBinomial` directly from `torch.distributions.NegativeBinomial`; verify inline registry behavior before describing source's probs as NB2 mean link.
- Reichenbach 1−a+ab has left identity a=1 but no two-sided identity; runtime class names do not prove associative categorical composition.
- `docs/developer/qiec.md` explicitly describes recursive helpers and authored handlers, so its final missing-feature paragraph is internally stale.

## Guides and QVR reference accessibility audit

Read all 27 guide pages and all 8 QVR reference pages. Main finding: the guides frequently substitute a categorical name for an explanation, while the reference provides useful concrete syntax but introduces unnecessary coined acronyms. Retain precise contracts, equations, code, and operational limits; introduce motivation and a small trace/calculation before formal descriptions. Use established terminology only.

### Highest priorities

1. Distinguish relation scores from normalized probabilities and continuous densities. Core says all morphisms are finite; noisy-OR is called a least upper bound and categorical composition despite missing distributivity. Distribution families and default latent tensor parameters need explicit distinctions.
2. Teach the modeling/inference boundary: sampling and conditioning do not compute a posterior; a trace is a record of one execution; an observation contributes a likelihood. Introduce latent, prior, likelihood, posterior, guide, ELBO, warmup, chain, and predictive draw before their technical details.
3. Explain categorical constructions as operations on a small representation, then state laws and limits. Correct the adjunction summary (adjunctions are not inverse functors up to natural isomorphism), qualify finite/discrete implementations of enriched operations, and avoid claiming categorical laws from names or numerical spot checks.
4. Replace unexplained type-system vocabulary with a running typed example and plain definitions: telescope is a parameter list, residual row lists still-unhandled requests, resumption continues the suspended computation. Remove FCC/CC/LID/EB/RPC/GCC/OCM/PCC/ICI invented labels from the reference.
5. Correct pedagogically damaging mathematical claims: indexed-independent products are a restricted subclass of joint distributions, densities can exceed one, LogProb log-sum-exp is not idempotent, and deterministic gradient descent on a hard threshold needs a differentiable surrogate (current fitting example claims otherwise).

### Page-by-page findings and requested changes

| Page | Findings and revision |
|---|---|
| guides/index.md | Assumes category familiarity. Give entry routes for ordinary Python/tensor users, statistical modelers, and category experts; link new concepts primer. |
| guides/core.md | Begins with false all-finite claim; terminal singleton and free monoid unexplained. Explain finite index sets, pairs versus tagged alternatives, empty string; show two-path composition calculation. Separate software join from lattice supremum, mark ProductFuzzy/noisy-OR non-distributivity, fix strongest/weakest t-norm reversal and max-min terminology. |
| guides/morphisms.md | Formula precedes question being answered. Explain row=input, column=output and compose via unobserved intermediate. Compute one noisy-OR entry; explain sigmoid is not row normalization, latent tensor is optimized parameter, @ is parallel inputs, marginalization depends on algebra. |
| guides/categorical.md | Functor/naturality/adjunction definitions assume all background. Show mapping strings characterwise, naturality as two routes to same result, adjunction as correspondence between maps with structure and maps of generators; correct summary. Explain associator/braiding as regroup/swap, trace as matching a feedback index rather than a general solver for recursion. |
| guides/enriched.md | End definition starts with wedge/equalizer/contravariance; every subsequent section repeats this difficulty. Add practical purpose and concrete operation per section (diagonal meet/join, relation matrices, lookup profiles, fibers, weighted constraints, convolution decompositions, record/variant optics). Explicitly distinguish discrete finite realizations from general enriched universal constructions; tests do not establish general isomorphism. |
| guides/monadic.md | Starts with natural transformations and whiskered equations. Begin computations returning alternatives; pure/map/join worked nested-list example and Kleisli sequential interpretation. Explain laws in words, comonad contextual input, algebra evaluating structured inputs, distributive law order. Truncated words and noisy-OR need bounded/representation caveats. |
| guides/effects.md | Towers, lifting, residuation, Hamblin, coindexing, scope extrusion all precede ordinary effect explanation. Start request/handler/resume and linguistic ambiguity example, separate semantic effects from source-level lexical rows, define lifting and typeclass; explain each table operation by information carried. |
| guides/transformations.md | Change of base sounds like guaranteed structure preservation. Explain score-to-probability normalization with numeric row, distinction entrywise vs whole-row action, >>> vs >>. Explain level declarations as API capability/smoke checks rather than proofs. Several listed homomorphisms need preservation qualifications. Wiring examples conflict with contractions page diagonal rule; avoid teaching unsupported example. |
| guides/stochastic.md | Good stochastic matrix definition but no worked probability. Compute weather/observation mixture row, explain sum over mutually exclusive intermediate states, row normalization and evidence reweighting. Last sentence contradicts Kleisli identification; replace. Mark sums on arbitrary [0,1] inputs need not stay in carrier. |
| guides/continuous-spaces.md | Space list is jargon-heavy; no density versus probability explanation. Explain real value vs finite index, support, conditional distribution, why density can exceed one, ancestral sampling versus integration, product domains (two inputs, not same shared x), flow Jacobian volume correction. |
| guides/continuous-families.md | Catalogue of families precedes choosing by data/support. Add support/parameter/event/batch/sample distinction; walk shape (100,8,3) and event rank vs event extent, conditional learned network vs fixed distributions. Explain matrix covariance, GP similarity, shrinkage in ordinary terms before technical formulas. |
| guides/programs.md | Opens Kleisli bind and presents uncertain learned conditional networks as fixed Normal examples. Explain a generative story and accumulated log joint. Prior sampling/observed clamping/posterior inference separate. Flag learnable ConditionalNormal does not automatically mean Normal(x,1). |
| guides/programs-hierarchical.md | First model draws random intercepts but never uses them; uses unconstrained Normal intercept as Bernoulli probability. Replace with actual logistic gather-and-sum model. Explain partial pooling and template copies, work small group/class likelihood explaining sum over classes outside product over group observations. |
| guides/inference-foundations.md | Architecture acronym diagram overwhelms before purpose; conditioning implies inference. Add one latent/one observed story, trace score table, parameter versus latent, flatten/bijector explanation. |
| guides/inference-svi.md | Posterior assumed known, options overwhelm and guide names overpromise multimodality etc. Explain adjustable approximating distribution and ELBO log p minus log q using fixed-model identity; variance/correlation tradeoffs, particles as replicate estimates, objective vs estimator. Full regression example does not multiply w by x and reports predicted y as w; replace with explicit actual model. |
| guides/inference-mcmc.md | Misnamed Variational Inference; MCMC framed fallback only; HMC/NUTS/ESS/Rhat unexplained. Explain correlated posterior draws, warmup vs retained draws, momentum gradient trajectories, NUTS trajectory selection, practical diagnostic meaning and limits. |
| guides/deduction.md | Hypergraph/presheaf/fixed point preempts parser story. Work two premises -> conclusion and alternatives under Boolean/logsumexp/max. Correct LogProb idempotence and qualify strategy independence/convergence. |
| guides/structural-compression-signatures-and-encoders.md | Starts Sigma algebra homomorphism and huge architecture catalogue. Work Const/App recursive encoding equation; signature as allowed tree shapes, sort as kind of child, embedding compression not invertibility. Explain bound-variable index by nesting, preserve technical binder order. |
| guides/structural-compression-decoders-and-losses.md | Kleisli coalgebra/corecursion before purpose. Walk root choice, child vectors, recursive term sampling, sum of log-probs; reconstruction is likelihood, not inverse. Explain depth bound changes model support; existing Bayesian integration overstates generic supported structured marginalization. |
| guides/dsl-overview.md | Familiarity with existing PPL assumed; implementation internals prominent, claims universal adequate symmetric monoidal closed denotation. Add model reading example and stages in plain language; qualify formal assumptions. |
| guides/dsl-declarations.md | Reference density, role=latent/kernel differences unclear. Add role selection table and parameter-versus-random draw explanation, fan vs @, repeat vs stack vs scan, curry as defer one argument. Some heading names still top-level let despite define source. |
| guides/dsl-programs-and-lets.md | Every everyday construct translated immediately into category jargon. Add sample/observe/let/marginalize walkthrough, lookup example, event/batch numerical shape; fix joint-kernel isomorphism, subprobability density claim, unsupported continuous marginalization scope. Rename prior>>likelihood posterior to predictive. |
| guides/dsl-contractions.md | Operadic/einsum defined too abstractly. Work ij,jk->ik with named axes and one numeric entry; explain shared retained axis vs reduced intermediate. Fix tropical/min vs max-plus distinction and invalid/anomalous initial inferred example. |
| guides/analysis-data-and-formulas.md | brms/lme4 literacy assumed. Read y~x+(1|g) in ordinary terms; explain intercept, slope, grouped shift, link, indexed lookup and category ordering. Implementation lens terminology can follow user purpose. |
| guides/analysis-fitting-and-diagnostics.md | Runnable fit then diagnostics name dump without interpretation. Separate convergence, approximation, model adequacy, predictive comparison. Explain Rhat/ESS/Pareto-k/PPP and no posterior model probabilities from stacking. Fix noisy-OR init explanation (not a K-way product); hard threshold gradient example problematic. |
| guides/migration.md | Operationally strong, low category burden. Add reader-facing explanation of grammar/source revision and difference parse-valid vs statistically equivalent execution. |
| guides/repl-and-lsp.md | Mostly accessible operational reference but 1200 lines; add short first-use route and expand REPL/TUI/LSP/AST; explain :run forward/replay vs fit inference and :trace elaboration vs probabilistic execution trace. |
| reference/qvr/index.md | Removes OCM label; explain checked computation as common typed representation; identify simplest entry for user task before mixed module. |
| reference/qvr/types-and-indexed-families.md | Remove ICI; explain Vec length tags, static vs runtime and telescope before notation; compare constructor to tagged record, motive to result type with branch evidence. |
| reference/qvr/computations.md | Remove CC; annotate signature with ordinary words; explain = vs <- even empty effect rows, static specialization and residual row; execution trace of clamp. |
| reference/qvr/effects-and-handlers.md | Remove LID; add request/handler/resume worked half_weight(8)->4 explanation; effect row capability not count/order. Existing run_state example always returns 7 despite put(3), must label fixed responder rather than actual mutable-state implementation. |
| reference/qvr/collection-expressions.md | Remove FCC; explain map vs traverse result identical for pure helper but sequencing required; numeric fold and log-normalization example; expand curried accumulator lambda. |
| reference/qvr/probabilistic-programs.md | Remove PCC; explain regression generative story, prior/likelihood/latent, observe scoring vs inferred posterior, grouping lookup and why assignments integrated per item. |
| reference/qvr/generated-computations.md | Remove GCC; explain elaboration like expanding shorthand, generated helpers versus public entry, portable descriptor vs runtime object; preserve exact capability boundaries. |
| reference/qvr/execution-and-tooling.md | Remove EB/RPC; distinguish check/run/inference/transpile, explain provider as runtime implementation, replay vs new posterior sampling. |

### Verification plan for revision

Preserve all checked QVR snippets unless intentionally corrected and compile modified examples. Check internal links. Run docs build and existing snippet tests where feasible. Numeric explanatory examples should be manually or programmatically verified. Check new prose against ASW voice without invented names; retain existing sources, verify any newly introduced external attribution. Several findings above are existing accuracy defects that should be corrected as part of making the explanations reliable, not excused with additional jargon.

## Semantics and API accessibility audit

The main barrier is sequencing: pages commonly introduce a categorical classification before the problem it solves, and API pages mostly identify a module without helping a reader choose or use it. Preserve the equations and autogenerated signatures, but introduce each construction through a small concrete calculation, state how to read its notation, and explicitly delimit the analogy. No new terminology is needed.

### Priority 1: contradictions that undermine rigorous explanations

- `composition-rules.md` §1–2: associative scalar multiplication does not alone imply associative matrix composition; distributivity and compatible reductions are required. The statement that all eleven algebras occupy the strongest mathematical level contradicts `algebras.md`. Separate runtime interface membership from laws assumed for a result.
- `setting.md` §3: a full probability-distribution functor on finite sets does not land in finite sets. Describe stochastic matrices as the finite-object subcategory of kernels, avoiding the claimed Giry monad endofunctor on FinSet.
- `algebras.md` §1.7: log-sum-exp corresponds to ordinary sum-product, not noisy-OR product-fuzzy aggregation. Nonpositive individual log probabilities are not closed under unrestricted log-sum-exp.
- `effects.md` §3/5: ABC registration is not evidence that the laws hold; qualify naturality/lifting adequacy by the stated laws and distinguish mathematical proof sketches from implementation conformance.
- `effects.md` §6: coproduct over depth-at-most-n trees duplicates the same tree at every larger depth. Use a union/colimit or exact-depth indexing.
- `types-and-spaces.md` / `schemas.md`: bounded FreeMonoid/FreeResiduated inventories do not automatically inherit an unbounded universal property or closure under arbitrary concatenation/type formation.
- `transpile-correctness` per-target pages mostly describe explicit latent rewrites for marginalization, contradicting the head page's enumeration/refusal contract. Verify renderer code before replacing.
- `transpile-correctness/stan.md` references absent Theorem 6.1 and Lemma 5.1.1, mislabels the transformation direction, and claims exact arithmetic. `numpyro.md` also references absent Theorem 6.1. `transpile-architecture.md` references a natural-isomorphism theorem not present in the contract. Replace with scoped contract/evidence statements.
- `transpile-correctness/church.md`: a joint log density is a sum of per-site log contributions, not a product.

### Semantics: page-by-page findings and revision targets

- `index.md`: explicitly assumes Kelly/Giry/Fritz familiarity. Replace with layered reading routes; define denotation, category, morphism and kernel in plain terms; link primer. Keep formal roadmap.
- `setting.md`: start with a 2×2 weighted table and route-through-intermediate-state calculation. Explain V as the entry values, tensor as combining stages, reduction as combining alternatives. Motivate measurable spaces using interval events before standard Borel definition. Translate environment/judgment notation.
- `algebras.md`: add one shared pair-of-paths calculation under Boolean, Markov, Tropical, LogProb, and ProductFuzzy. Explain why reassociation and thresholding can change results; retain exact counterexample. Clarify algebra interface versus algebraic theorem.
- `composition-rules.md`: explain a contraction as matching and eliminating internal axes; annotate i,j,k formula. Add operational-versus-lawful table and an einsum-style example. Transformations change weight conventions; composition changes paths.
- `types-and-spaces.md`: introduce product as paired values, coproduct as tagged alternatives, Unit as one possible value; work cardinalities. Explain bounded word inventories and flattening versus permutation. Separate support mathematics from flat storage and runtime membership checks.
- `morphisms.md`: distinguish weighted relation, normalized transition matrix, and conditional distribution before formal strata. Contrast a row of scores with a row summing to one; explain family arguments versus learned parameter maps. Explain event axes versus independent batches and mixed-stratum restrictions.
- `expressions.md`: add a plain operator choice table before grammar. Annotate composition/marginalization with index examples; distinguish hidden-coordinate summation, trace, and recurrence. Preserve repeat-versus-stack parameter sharing and stochastic fan's conditional independence.
- `programs.md`: replace monad-first entry with a sample/observe/return example and joint factorization. Explain observe as scoring supplied data and posterior as normalization. Explain Giry unit/bind by deterministic distributions and weighted averaging, then formal definition. Add worked finite/grouped marginalization and gather example; explain score scale and event/batch shapes. Separate finite exact marginalization from continuous single draws and Monte Carlo estimators.
- `typing.md`: translate judgments and premises/conclusions before using them. Decode static telescope, kind, index, effect row and continuation in plain language. Explain indexed Vec through length tracking and impossible branches. Distinguish runtime input validation from compile-time proof.
- `grammar.md`: introduce chart as memoized table, rules as dependencies, agenda as work list; work a two-edge path with alternative derivations. Explain semiring controls what answer is accumulated. Explain cycle convergence and depth bounds separately. Large grammar listing needs a route for readers who need meaning rather than syntax inventory.
- `schemas.md`: use grammar application X/Y,Y→X before residual adjunctions; explain rule/schema/bundle roles. Highlight nonstandard backslash convention with explicit noun-phrase/sentence example. Qualify bounded free construction.
- `structural.md`: introduce tree folding to vectors and unfolding to generated trees before catamorphism/coalgebra. Work Nil/Cons or small arithmetic tree. Explain bound-variable context as scope stack; encoder-decoder is not necessarily inversion.
- `effects.md`: show request/handler intuition, e.g. same random request drawn, replayed, or enumerated. Explain functor/applicative/monad dependency differences. Distinguish parser effect wrappers from QIEC lexical instances; give two separate state-cell example. Qualify theorem scope and explain handler resumptions.
- `program-theory.md`: introduce schema as inspectable graph of declarations, not executable semantics. A same-shape/different-weights example makes non-equivalence concrete. Explain protocol as allowed graph vocabulary.
- `adequacy.md`: already well scoped; add simple worked distinction between sampling a composition and calculating its marginal density. Explain finite-sample log bias and evidence levels without requiring adequacy terminology.
- `transpile-architecture.md`: define intermediate representation as structured common description; explain check→plan→render→print through a single Normal call before LDA. Motivate scale/precision conversion and preservation of axes. Remove stale theorem claim.
- `transpile-correctness/index.md`: introduce same model as same relative weights, not same random draws; numeric additive-constant example; explain Jacobian as density correction under coordinate change. Define all per-target notation (Psi, pi, c) here or remove unexplained symbols in target intros.
- `transpile-correctness/stan.md`: explain density-program versus forward simulator; Jacobian direction and one-variable exp example; fix phantom theorem references and exact-arithmetic claim; retain support/refusal and finite evidence.
- `transpile-correctness/numpyro.md`: explain trace and clamping before probe; distinguish JAX from torch; fix marginalization and phantom theorem.
- `transpile-correctness/pyro.md`: explain trace as sample-site record and conditioning as fixing values; fix marginalization and incorrect assertion NumPyro uses torch distributions.
- `transpile-correctness/pymc.md`: explain graph-building and compiled logp; dimensions name axes without creating independence; fix marginalization.
- `transpile-correctness/edward2.md`: explain tape records for density evaluation; scope native score restrictions to actual renderer; fix marginalization/refusal.
- `transpile-correctness/turing.md`: explain model macro and supplied observation values; fix marginalization; recheck claim normalized truncation adds log 2 discrepancy.
- `transpile-correctness/gen.md`: explain choice-map addresses and fixed-value density; fix marginalization.
- `transpile-correctness/church.md`: explain continuation without assuming lambda calculus; correct sum of log densities; preserve runtime-coverage limitation; fix marginalization.
- `transpile-correctness/webppl.md`: explain score-at-fixed-values probe and continuation as remaining computation; fix marginalization; distinguish JavaScript parse success from WebPPL execution.
- `transpile-correctness/bugs.md`: explain precision with scale=2→precision=1/4; explain zero trick and requirement nonnegative Poisson rate; fix marginalization.
- `transpile-correctness/jags.md`: orient inherited BUGS semantics and support; qualify same-shape claims by actual implementation; explain inference algorithm does not determine marginalization semantics.

### API: shared design

Retain every `:::` directive and options. Before autogenerated reference, add 2–4 sentences answering: what concrete problem does this module solve, what enters/leaves it, when should I use it versus its neighbor, and what formal assumptions remain? Add links to the relevant guide/semantics. Advanced leaf pages need a concrete mental model, not another catalogue of formal names. Module docstrings were inspected for categorical/enriched/monadic pages; they contain precise equations but generally still begin with advanced vocabulary, so generated content alone does not supply this bridge.

`index.md`: add task routes (load/run model; build weighted composition; fit model; inspect compiler; extend categorical machinery) before package taxonomy. `program.md`: distinguish loaded module container from a single probabilistic program. `cli.md`: add check→list→run→transpile workflow; explain forward run/log joint is not posterior inference and --target checking is not rendering.

#### analysis

Explain that these inspect shapes, scopes, initialization, and diagnostics without fitting; each leaf should name its input and returned report/graph.

- `analysis/chain_shape.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `analysis/index.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `analysis/init_spec.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `analysis/plate_graph.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `analysis/plate_render.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `analysis/saturation.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `analysis/scope.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### arrows

Explain arrows as typed composable computations; use parallel input processing to motivate first/choice/loop. Separate class interfaces, concrete instances, and theory records.

- `arrows/instances.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `arrows/theories.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `arrows/typeclasses.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
#### categorical

Functor maps objects and computations; natural transformation converts wrapped representations consistently; monoidal product combines interfaces; adjunction relates two construction tasks; trace closes matching wires. Base change needs law qualifications.

- `categorical/adjunctions.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `categorical/base_change.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `categorical/functors.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `categorical/monoidal.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `categorical/natural_transformations.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `categorical/traced.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### continuous

Distinguish support space, family, conditional parameter source, sample/log_prob interface, program step, plate, scan and flow. Bijectors need inverse/Jacobian intuition; scan returns final state and log_prob is trajectory score.

- `continuous/bijectors.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/boundaries.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `continuous/deterministic.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/families.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `continuous/family_spec.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/flows.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `continuous/inline.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/measure.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/morphisms.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/ordered.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/param_source.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/plate.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/program_steps.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/programs.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `continuous/scan.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `continuous/spaces.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### core

Objects name axes, morphisms store weighted tables, algebras select combination/reduction, wiring eliminates internal axes, transformations alter weights/conventions. Existing introductions mostly enumerate names.

- `core/algebras.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `core/morphisms.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `core/objects.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `core/tensor_ops.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `core/transformations.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `core/util.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `core/wiring.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### data

Explain table columns→encoded tensors and schema→declared axes/observations; describe category codes, missing policy and input/output roles.

- `data/encoding.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `data/index.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `data/schema.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### diagnostics

Explain posterior draws→analysis container, predictive discrepancy checks, and leave-one-out comparison; expand PSIS-LOO/LOO-PIT and distinguish diagnostics from guarantees.

- `diagnostics/arviz_io.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `diagnostics/comparison.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `diagnostics/index.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `diagnostics/predictive_checks.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### dsl

Present source→AST→resolution/checking→checked module. Distinguish parser from compiler and internal implementation pages from public load/loads. Explain elaboration as making implicit operations explicit.

- `dsl/ast_nodes.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `dsl/compiler.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `dsl/composite_lets.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/constraints.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/deduction_elaboration.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/draw_args.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/emit.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/family_schemas.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `dsl/index.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/let_expr_traversal.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `dsl/parser.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `dsl/program_elaboration.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/program_theory.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/pure_builtins.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/pygments_lexer.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `dsl/qiec_diagnostics.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `dsl/qiec_lowering.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `dsl/qiec_tooling.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/resolution.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `dsl/step_resolution.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
#### effects

Introduce sample request interception and trace recording; distinguish conditioning from intervention, replay from fresh sampling, scaling from masking, analytic collapse from approximate inference. Explain lexical handler order.

- `effects/base.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/checked_program.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/collapse.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/conjugate.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/handlers.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `effects/index.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/interpreter.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/program_module.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/reparam.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/sites.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/trace.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `effects/trace_types.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### enriched

Supply finite-table interpretations: ends/coends match and reduce axes; Kan extensions regroup rows; profunctors are weighted relationships; Yoneda describes an object through maps into it; optics focus/reassemble; Day convolution combines structured contributions; weighted limits require their law assumptions.

- `enriched/day_convolution.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `enriched/ends_coends.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `enriched/kan_extensions.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `enriched/optics.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `enriched/profunctors.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `enriched/weighted_limits.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `enriched/yoneda.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### formulas

Explain formula plus table→design matrices→QVR→fit, response family versus link, fixed versus group effects, and source inspection before inference.

- `formulas/compile.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `formulas/family.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `formulas/fit.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `formulas/formula.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `formulas/index.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### inference

Explain model/guide/objective/estimator/optimizer distinction, MCMC versus variational approximation, trace versus posterior draws, predictions versus latent samples, and why lifts add priors.

- `inference/conditioning.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `inference/elbo.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `inference/estimators.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `inference/guide.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `inference/lifts.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `inference/mcmc.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `inference/predictive.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `inference/svi.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `inference/trace.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
#### monadic

Explain context-bearing results, fmap/pure/join/bind, dependent sequencing, and transformer order with concrete Maybe/State/Writer examples. Distinguish numerical law checks from proofs and theory data from implementations.

- `monadic/algebraic.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `monadic/algebras.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `monadic/bridges.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `monadic/comonads.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `monadic/distributive_laws.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `monadic/instances.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `monadic/laws.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `monadic/monads.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `monadic/theories.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `monadic/transformers.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `monadic/typeclasses.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
#### qiec

Provide phase map: identifiers/kinds/types/terms→declarations/module→checking/coverage/substitution→execution/evaluator/runtime→serialization. Explain lexical rows, static indices, equality evidence and program metadata; leaf intros currently only restate titles.

- `qiec/builtins.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/canonical.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/checking.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/coverage.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/declarations.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/distributions.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/effects.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/entries.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/evaluator.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/evidence.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/execution.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/families.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/identifiers.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/index.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/kinds.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/module.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/primitives.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/program_runtime.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/programs.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/reference_families.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/reference_support.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/serialization.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/structural_runtime.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/substitution.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/terms.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `qiec/types.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### stochastic

Distinguish categorical probabilities, weighted deductions, grammar categories, chart semirings and inference queries. Concrete derivation example and normalizing/finite-enumeration caveats needed; existing parser/rule/biclosed pages are stronger than stubs.

- `stochastic/agenda.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/biclosed.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/categories.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/deduction/bayes.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/deduction/fit.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/deduction/primitives.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/deduction/sample.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/deduction.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/effect_lifts.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/families.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `stochastic/giry.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `stochastic/inside.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `stochastic/morphisms.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/parsers.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/queries.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `stochastic/rules.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/semiring.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/stdlib.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `stochastic/transforms.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
#### structural

Explain signature as permitted tree forms, encoder as fold, decoder as stochastic unfold, losses as scalar training penalties; scope stack before de Bruijn context. Preserve depth/termination and no-inverse caveats.

- `structural/decoder.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `structural/encoder.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `structural/losses.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `structural/shapes.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `structural/signature.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
#### transpile

Explain plan and two IR levels, family semantics versus target spelling, renderer registry versus renderer implementation; user entry point before internal tables.

- `transpile/family_meta.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `transpile/family_spelling.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `transpile/index.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `transpile/ir.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `transpile/lower.md`: Existing orientation is terse or assumes specialist vocabulary; add the concrete use/inputs/outputs distinction above and preserve detailed reference.
- `transpile/plan.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `transpile/qiec_ir.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `transpile/renderer_registry.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.
- `transpile/renderers.md`: Autodoc-only page: add module-specific orientation and a practical cross-link.

## Final prose review

After the first revision, an additional pass removed formulaic contrasts and generic cautions throughout the guides, tutorials, semantics, and API introductions. A separate reviewer identified remaining examples, which were revised. The resulting explanations use calculations, implementation behavior, or concrete checks where the earlier prose merely announced a limitation.

## Validation

- The executable documentation suite passed all 241 cases: QVR compilation throughout the docs and Python execution in the tutorials.
- The new primer's Python calculation was executed separately and passed.
- All 11 bibliography rendering tests passed, including unknown keys, grouped citations, code preservation, repeated authors and years, and escaped URL characters.
- The generated support page matched a fresh measurement. This check used `QUIVERS_SKIP_DOCKER=1`; it measures translation support without running backend containers.
- Ruff passed for the new bibliography hook and its tests.
- The 20 changed source modules have identical executable Python ASTs after removing docstrings. The nine changed gallery source files differ only in comments.
- The bibliography contains 136 cited works and 499 citation occurrences. Each key has a verification record in `docs-citation-verification.md`.
- The final strict MkDocs build passed. The rendered math audit passed, and all 499 rendered citation links across 186 pages resolved to their local bibliography entries. The link audit found no unexpanded citation syntax or escaped URL characters.
- `git diff --check` passed.

## Follow-up cross-linking revision

The [cross-linking audit](docs-link-audit.md) records the linking pass and its editorial follow-up, including current link counts, source verification, build results, and checks that executable examples are unchanged.
