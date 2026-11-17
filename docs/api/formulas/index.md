# `quivers.formulas`

The formula front end compiles a
[regression formula](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) in the
style of brms [@burkner2017brms] to QVR source and
[fits](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line)
the resulting Bayesian model. In `y ~ x + (1 | group)`, `x` contributes a common slope and each
group receives its own intercept adjustment [@gelman2013bayesian]. The family and the priors
complete the statistical specification.

Joint count and compositional responses use `cbind(a, b, c) ~ x`.
A retained `ResponseLayout` distinguishes component axes from scalar factor
codes, and `Family.with_reference` selects predictor coordinates without
reordering response components. The
[vector response guide](../../guides/analysis-data-and-formulas.md#factor-and-vector-responses)
gives count totals, compositional means, and new-row predictions.

Response-side declarations such as `rt | dec(choice) + weights(importance) ~ x`
are retained in the immutable `ResponseModifiers` value. For the Wiener family,
the compiler emits a visible map from positive reaction times and binary choices
to signed joint responses. `BayesianFit.posterior_predict` retains those signed
responses, while `posterior_response` returns a `ResponseDraws` record with
positive times and matching choices. `posterior_epred` gives expected positive
time; `posterior_choice_probs` gives lower/upper probabilities. These methods
share the fitted prediction plan and the existing posterior replay machinery;
`PredictionOptions` and `ExpectationOptions` describe their shared keywords.
See the [joint response example](../../guides/analysis-data-and-formulas.md#joint-reaction-times-and-decisions)
and [formal response law](../../semantics/programs.md#62-joint-reaction-time-and-decision-responses).

::: quivers.formulas
