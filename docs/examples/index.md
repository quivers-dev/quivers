# Examples Gallery

`.qvr` programs spanning probabilistic regressions, latent-variable
models, state-space models, language models, encoder-decoder networks,
weighted deductions, and QIEC computations. Most full pages include a
`## Try it` section; CI executes runnable documentation blocks and skips blocks
marked as illustrative. Pages with current model limitations identify them
directly.

The measured transpilation corpus lives under `docs/examples/source/`. Mixed
QIEC case studies whose purpose is reference-machine execution live under
`docs/examples/qiec/`, so adding one does not silently expand the promised
backend matrix.

## Choose a first example

Start with [Bayesian regression](bayesian-regression.md) to connect unknown
[coefficients, observations, and posterior draws](../tutorials/qvr/01-first-model.md#what-inference-will-answer). Then try the
[Beta-Binomial example](beta-binomial-ab-test.md) for variation across batches,
[partial pooling](parametric-pooling.md) for shared group structure, and a
[mixture](mixture-model.md) for alternative explanations of an observation.
The [conceptual introduction](../getting-started/concepts.md) defines the
probability and composition vocabulary along that route.

For structured language models, begin with [PCFG](pcfg.md) to learn chart
items and weighted derivations, then move to [rule schemas](schema-chart-parser.md)
or the [term autoencoder](term-autoencoder.md). Read each page's limitations:
some sources illustrate a model's wiring or one conditional component rather
than implementing every feature suggested by a historical filename.

## Language feature map

Use this table when you want an example of a language feature rather than a
particular statistical family.

| Feature | Primary example | What to inspect |
| --- | --- | --- |
| indexed families, authored handlers, computation calls, parser search, structural attachments, grouped marginalization | [Bayesian Semantics Case Study](bayesian-semantics-case-study.md) | mixed entry list, effect trace, target-capability diagnostics |
| schema and bundle elaboration, generated search and weight effects | [Schema-Bundled Chart Parser](schema-chart-parser.md) | classic chart against `run_deduction` |
| direct deduction, learned rule weights, program-to-deduction call | [PCFG](pcfg.md) and [CCG](ccg.md) | inside weights, parameter store, scored parser call |
| structural signature, encoder, decoder, loss, typed PyTorch attachments | [Term Autoencoder](term-autoencoder.md) | `Program.run("reconstruct", term)` and autograd |
| grouped and fibred exact marginalization | [Document-level topic mixture](lda.md) | `over`, `via`, and the per-group reduction |
| program templates and subprogram calls | [Parametric Partial Pooling](parametric-pooling.md) | static object arguments, prefixed sites, labeled tuple return |
| open sequence extent and recursive scan lowering | [Vanilla RNN LM](vanilla-rnn-lm.md) | generated scan step and trajectory site addresses |
| network kernel computation | [Bayesian Neural Network](bnn.md) | MLP parameter source and reference-machine boundary |
| measure-algebra families | [Bayesian Semantics Case Study](bayesian-semantics-case-study.md) | heterogeneous `Mixture` value used for scoring and prediction |
| finite collection traversal, indexed effects, nested computation calls, lexical uncertainty | [Lexical-Uncertainty RSA](qiec/lexical-uncertainty-rsa.qvr) | `map`/`fold` normalization, `traverse` over worlds and lexica, lexical marginalization, observed-or-predicted interpretation |

The generated [transpilation support matrix](../transpile-support.md) records
whether each measured source reaches each backend. A successful
reference-machine run and a successful transpile are distinct claims; the
examples state both where they differ.

## Integrated QIEC case study

- [Bayesian Semantics Case Study](bayesian-semantics-case-study.md): four views
  of one reading experiment in a checked module: a schema-backed categorial
  parser summing over derivations, a structural logical-form autoencoder, an
  indexed family separating observed from missing reading times under a
  contamination measure, and a hierarchical reading-time model with grouped
  exact marginalization.
- [Lexical-Uncertainty RSA](qiec/lexical-uncertainty-rsa.qvr): the specificity
  model of [@bergen2016pragmatic],
  with a literal listener nested inside a pragmatic speaker and a listener
  that traverses and marginalizes over three lexica.

## Regressions

- [Bayesian Linear Regression](bayesian-regression.md): Normal likelihood with Normal-prior coefficients and a HalfCauchy noise scale, recovered by SVI and NUTS.
- [Multi-Output Beta Regression](beta-regression.md): Beta response on the unit interval with a logit-link mean and a per-output precision.
- [Gamma Regression](gamma-regression.md): log-link Gamma GLM with a per-category intercept, slope, and shape.
- [Horseshoe Sparse Regression](horseshoe-regression.md): dense multiple regression whose coefficients carry the horseshoe's global-local shrinkage, recovering a sparse truth.
- [Linear Regression with Logistic Noise](logistic-noise-regression.md): Logistic observation noise in place of Normal.
- [Hierarchical regression with a HalfStudentT scale prior](half-student-t-hierarchical.md): group intercepts with heavy-tailed priors on both scales.
- [Multi-Output Negative Binomial Regression](negbin-regression.md): the NB2 model, with a log-link mean and a per-output dispersion.
- [Multi-output zero-inflated Poisson regression](zip-regression.md): a logistic regression for the structural-zero probability and a log-linear one for the rate, with the zero branch marginalized exactly.
- [Bounded-Outcome Regression via Kumaraswamy](kumaraswamy-bounded-outcome.md): a Kumaraswamy likelihood for a response on the unit interval.
- [Beta-Binomial A/B Test](beta-binomial-ab-test.md): per-arm Beta-Binomial conversion counts.
- [Correlation-Matrix Prior via LKJCholesky](lkj-cholesky-correlation.md): an LKJ prior over a correlation matrix's Cholesky factor.
- [Two-Parameter Logistic IRT](irt-2pl.md): respondent abilities with item difficulties and positive item discriminations.
- [Weibull Survival Regression](survival-weibull.md): a proportional-hazards Weibull model with right-censored follow-up, censored rows scored by their survival function.

## Latent-variable models

- [Bayesian Factor Analysis](factor-analysis.md): latent factors and loadings with a free uniqueness per observed coordinate.
- [Probabilistic Principal Component Analysis](ppca.md): the factor model with one isotropic noise scale, checked against its principal-subspace solution.
- [Document-level topic mixture](lda.md): Dirichlet priors with one topic shared by every word of a document, summed out exactly; contrasted with token-level LDA.
- [Bayesian Gaussian Mixture Model](mixture-model.md): mixing weights, locations, and scales inferred under a closed-form mixture likelihood.
- [Variational Autoencoder](vae.md): a neural decoder fitted with an amortized encoder guide.
- [Bayesian Neural Network](bnn.md): a one-hidden-layer tanh regression network whose weights are sample sites.
- [Parametric Partial Pooling](parametric-pooling.md): Rubin's parallel-experiments model through a program template, with a labeled return tuple and export selection.
- [Probabilistic Matrix Factorization](pmf.md): low-rank Bayesian completion of a sparse rating matrix, evaluated on held-out ratings.
- [Bilinear Tensor Contraction](tensor-contraction.md): neural-tensor-layer scoring of predicate-argument pairs through an operadic three-way contraction.
- [Tree-Structured Categorical Model](tree-categorical.md): class probabilities from a binary tree of splits, with per-verb random effects on each split.

## State-space and time-series models

- [AR(1) Time Series](ar1.md): the first-order autoregressive model under its exact stationary likelihood.
- [Bayesian Change-Point Detection](changepoint.md): a Poisson rate switching at a discrete change point that is marginalized exactly.
- [Stochastic Volatility](stochastic-volatility.md): AR(1) log-volatility conditionals with externally supplied lag values.
- [Gaussian-kernel state-space model](linear-gaussian-ssm.md): learned transition, emission, and filtering kernels; not a closed-form Kalman filter.
- [Finite-state path composition](hmm.md): product-fuzzy transition and emission relations, plus a partial discrete-program sketch.
- [Continuous State-Space Model](continuous-hmm.md): Gaussian transition, emission, and separately learned recognition kernels.
- [Deep Markov Model](deep-markov.md): nonlinear-MLP transition and emission cells.

## Language models

- [Vanilla RNN Language Model](vanilla-rnn-lm.md): single-cell Elman RNN scored by a Categorical head.
- [GRU-shaped language model](gru-lm.md): stochastic reset and update gates with a simplified candidate update.
- [LSTM-shaped language model](lstm-lm.md): stochastic gates without a separately threaded cell state.
- [Dual-RNN masked-token model](bidirectional-rnn-lm.md): two left-to-right scans combined for a masked-token target.
- [Transformer-shaped language model](transformer-lm.md): parallel MLP-normal branches with a Categorical head; no dot-product self-attention or causal mask.

## Encoder-decoder and structured outputs

- [Sequence-to-Sequence](seq2seq.md): parallel encoder-decoder branches joined by a learned merge, without attention.

## Weighted deductions

- [Probabilistic Context-Free Grammar](pcfg.md): learnable production weights over CKY chart items.
- [Weighted Combinatory Categorial Grammar](ccg.md): forward and backward application, harmonic and crossed composition, each needed by some sentence of the lexicon.
- [Type-Logical Grammar (Lambek calculus)](type-logical.md): a chart parser whose every rule is a theorem of the associative Lambek calculus.
- [Weighted Multiple Context-Free Grammar](wmcfg.md): rank-2 nonterminals deriving Swiss German cross-serial verb clusters.
- [Schema-Bundled Chart Parser](schema-chart-parser.md): pattern-polymorphic rule schemas bundled into a differentiable CKY chart parser.
- [Term Autoencoder](term-autoencoder.md): a signature, encoder, decoder, and loss compressing typed lambda terms.
- [Multimodal Type-Logical Grammar](multimodal-tlg.md): the Lambek calculus with residuated diamond and box modalities and one controlled structural postulate.
- [Custom Sequent Rules](custom-rules.md): an AB grammar with harmonic composition declared as sequents in the DSL.
- [Quantifier Scope](quantifier-scope.md): surface and inverse readings from the continuation monad's two applicative orders.
- [Montague NLI](montague-nli.md): Montague-style lambda-term logical forms plus a syllogistic entailment prover.
