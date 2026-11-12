# Ordered and Zero-Inflated Families

[Ordered families](../../guides/analysis-data-and-formulas.md#ordinal-mixed-models-and-neural-predictors)
model ranked outcomes, such as low, medium, and high, through thresholds on a linear predictor
[@mccullagh1980regression]. The zero-inflated and hurdle families model
[extra zero outcomes](../../examples/zip-regression.md) [@mullahy1986specification;
@mclachlan2000finite]: their parameters give the probability of the zero branch and the
distribution on the other branch. Their log probabilities are computed with the stable methods of
[@maechler2012log1mexp].

For a proportion that may equal zero or one, an inflated beta distribution
combines point masses at the endpoints with a density between them
[@ospina2010inflated]. An endpoint probability and an interior density are
different quantities, and each contribution is scored against its own measure, discrete or
continuous.

The module gives ordered categorical families, conditional and inline, and the zero-inflated,
hurdle, and mixture distributions that PyTorch does not include.

::: quivers.continuous.ordered

::: quivers.continuous._ordered

::: quivers.continuous._zip_hurdle
