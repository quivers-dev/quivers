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
[coefficients, observations, and posterior draws](../tutorials/qvr/01-first-model.md#the-model). Then try the
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
| indexed families, authored handlers, computation calls, parser search, structural attachments, grouped marginalization | [Amortized Bayesian Semantics](amortized-bayesian-semantics.md) | mixed entry list, effect trace, target-capability diagnostics |
| schema and bundle elaboration, generated search and weight effects | [Schema-Bundled Chart Parser](schema-chart-parser.md) | classic chart against `run_deduction` |
| direct deduction, learned rule weights, program-to-deduction call | [PCFG](pcfg.md) and [CCG](ccg.md) | inside weights, parameter store, scored parser call |
| structural signature, encoder, decoder, loss, typed PyTorch attachments | [Term Autoencoder](term-autoencoder.md) | `Program.run("reconstruct", term)` and autograd |
| grouped and fibred exact marginalization | [Latent Dirichlet Allocation](lda.md) | `over`, `via`, and the per-group reduction |
| program templates and subprogram calls | [Parametric Partial Pooling](parametric-pooling.md) | static object arguments, prefixed sites, labeled tuple return |
| open sequence extent and recursive scan lowering | [Vanilla RNN LM](vanilla-rnn-lm.md) | generated scan step and trajectory site addresses |
| network kernel computation | [Bayesian Neural Network](bnn.md) | MLP parameter source and reference-machine boundary |
| measure-algebra families | [Amortized Bayesian Semantics](amortized-bayesian-semantics.md) | heterogeneous `Mixture` value used for scoring and prediction |
| finite collection traversal, indexed effects, nested computation calls, lexical uncertainty | [Lexical-Uncertainty RSA](qiec/lexical-uncertainty-rsa.qvr) | `map`/`fold` normalization, `traverse` over worlds and lexica, lexical marginalization, observed-or-predicted interpretation |

The generated [transpilation support matrix](../transpile-support.md) records
whether each measured source reaches each backend. A successful
reference-machine run and a successful transpile are distinct claims; the
examples state both where they differ.

## Integrated QIEC case study

- [Amortized Bayesian Semantics](amortized-bayesian-semantics.md): a learnable
  categorial parser, structural logical-form autoencoder, indexed missing-data
  computation with a contamination measure, authored robustification handler,
  and hierarchical model with grouped exact marginalization.
- [Lexical-Uncertainty RSA](qiec/lexical-uncertainty-rsa.qvr): the specificity
  model of [@bergen2016pragmatic],
  with a literal listener nested inside a pragmatic speaker and a listener
  that traverses and marginalizes over three lexica.

## Regressions

- [Bayesian Linear Regression](bayesian-regression.md): Normal likelihood with Normal-prior weights.
- [Beta Regression](beta-regression.md): Beta-distributed response with logit link.
- [Gamma Regression](gamma-regression.md): Gamma response with a log-shape predictor and unit rate.
- [Horseshoe Regression](horseshoe-regression.md): sparse linear regression under the horseshoe prior.
- [Negative Binomial Regression](negbin-regression.md): overdispersed count response.
- [Zero-Inflated Poisson Regression](zip-regression.md): the scored likelihood exactly marginalizes the two gate endpoints.
- [Item Response Theory (2PL)](irt-2pl.md): logistic IRT with item difficulty and item-specific discrimination.
- [Weibull Survival](survival-weibull.md): uncensored Weibull event-time regression; censored rows require an explicit survival-score term.

## Latent-variable models

- [Isotropic Bayesian Factor Model](factor-analysis.md): low-rank factors and loadings with one shared observation scale.
- [Probabilistic PCA](ppca.md): the isotropic factor model interpreted through its principal-component solution.
- [Document-Level Topic Mixture](lda.md): Dirichlet priors with one shared topic assignment per document; contrasted with token-level LDA.
- [Gaussian Mixture Model](mixture-model.md): finite mixture with grouped marginalization over the cluster label.
- [Latent Decoder and Encoder Paths](vae.md): prior-decoder program plus a separately exported encoder-decoder path; the current SVI block does not use the encoder as its guide.
- [Bayesian Neural Network](bnn.md): nonlinear regression through an MLP-parameterized Normal kernel, made Bayesian after lifting priors over its weights.
- [Parametric Partial Pooling](parametric-pooling.md): random effects from a parametric program template, with a labeled return tuple, a score-step sum-to-zero factor, and export selection.
- [Probabilistic Matrix Factorization](pmf.md): low-rank Bayesian completion of a sparse rating matrix.
- [Bilinear Tensor Contraction](tensor-contraction.md): neural-tensor-layer scoring of predicate-argument pairs via an operadic three-way contraction.

## State-space and time-series models

- [AR(1)](ar1.md): first-order autoregressive Normal scalar series.
- [Changepoint Model](changepoint.md): a smooth transition between Poisson rates at a sampled change location.
- [Conditionally Factorized Stochastic Volatility](stochastic-volatility.md): AR(1) log-volatility conditionals with externally supplied lag values.
- [Gaussian-Kernel SSM](linear-gaussian-ssm.md): learned transition, emission, and filtering kernels; not a closed-form Kalman filter.
- [Finite-State Path Composition](hmm.md): product-fuzzy transition and emission relations, plus a partial discrete-program sketch.
- [Continuous-State Sequence Model](continuous-hmm.md): Gaussian transition, emission, and separately learned recognition kernels.
- [Deep Markov Model](deep-markov.md): nonlinear-MLP transition and emission cells.

## Language models

- [Vanilla RNN LM](vanilla-rnn-lm.md): single-cell Elman RNN scored by a Categorical head.
- [GRU-shaped LM](gru-lm.md): stochastic reset and update gates with a simplified candidate update.
- [LSTM-shaped LM](lstm-lm.md): stochastic gates without a separately threaded cell state.
- [Dual-RNN Masked-Token Model](bidirectional-rnn-lm.md): two left-to-right scans combined for a masked-token target.
- [Transformer-shaped LM](transformer-lm.md): parallel MLP-normal branches with a Categorical head; no dot-product self-attention or causal mask.

## Encoder-decoder and structured outputs

- [Sequence-to-Sequence](seq2seq.md): parallel encoder-decoder branches joined by a learned merge, without attention.
- [Tree-Structured Score Tensor](tree-categorical.md): recursive additive scores over a parametric tree.

## Weighted deductions

- [PCFG](pcfg.md): probabilistic context-free grammar with learnable production weights.
- [CCG](ccg.md): combinatory categorial grammar with forward / backward application and composition.
- [Type-Logical Grammar (Lambek)](type-logical.md): Lambek calculus with residuated slashes and tensor.
- [PMCFG](pmcfg.md): probabilistic multiple context-free grammar with WH-movement via a rank-2 non-terminal.
- [Schema-Bundled Chart Parser](schema-chart-parser.md): pattern-polymorphic rule schemas bundled into a differentiable CKY chart parser.
- [Term Autoencoder](term-autoencoder.md): a signature, encoder, decoder, and loss compressing typed lambda terms.
- [Multimodal TLG](multimodal-tlg.md): Lambek calculus extended with diamond and box modalities.
- [Custom Sequent Rules](custom-rules.md): user-defined sequents over a free residuated category.
- [Quantifier Scope](quantifier-scope.md): continuation-monad lift for generalized quantifiers.
- [Montague NLI](montague-nli.md): Montague-style lambda-term LFs plus modus-ponens NLI prover.
