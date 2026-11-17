# Multivariate formulas

Joint (multivariate) formulas are built with `bf`, combined with `+` or
`mvbf`, and fitted by the ordinary [`fit`](fit.md) entry point, which
returns a `MultivariateFit` for a `MultivariateFormula`. The
[multivariate guide](../../guides/formula-multivariate.md) gives the brms
correspondence, parameter names, and prediction units. The
[missing-data guide](../../guides/formula-missing-data.md) covers `mi()`,
`me()`, and multiple imputation, and the
[semantics page](../../semantics/programs.md#joint-multivariate-formula-responses)
states the joint density.

## Entries

- [`bf`](index.md#quivers.formulas.bf)
- [`mvbf`](index.md#quivers.formulas.mvbf)
- [`set_rescor`](index.md#quivers.formulas.set_rescor)
- [`lf`](index.md#quivers.formulas.lf)
- [`nlf`](index.md#quivers.formulas.nlf)
- [`set_nl`](index.md#quivers.formulas.set_nl)
- [`ParameterFormulas`](index.md#quivers.formulas.ParameterFormulas)
- [`ResponseModel`](index.md#quivers.formulas.ResponseModel)
- [`MultivariateFormula`](index.md#quivers.formulas.MultivariateFormula)
- [`RescorSetting`](index.md#quivers.formulas.RescorSetting)
- [`MultivariateFit`](index.md#quivers.formulas.MultivariateFit)
- [`MecorSetting`](index.md#quivers.formulas.MecorSetting)
- [`set_mecor`](index.md#quivers.formulas.set_mecor)
- [`fit_multiple`](index.md#quivers.formulas.fit_multiple)
- [`combine_models`](index.md#quivers.formulas.combine_models)
- [`MultipleFit`](index.md#quivers.formulas.MultipleFit)
