# Ordered and Zero-Inflated Families

[Ordered families](../../guides/analysis-data-and-formulas.md#ordinal-mixed-models-and-neural-predictors) model ranked outcomes, such as low, medium, and high, through
thresholds on a predictor [@mccullagh1980regression]. The zero-inflated and hurdle families in this module model
[extra zero outcomes](../../examples/zip-regression.md) [@mullahy1986specification; @mclachlan2000finite]. Their parameters specify the probability of taking the zero branch
and the distribution used on the other branch.

The stable computation of log probabilities uses the numerical methods
explained in [@maechler2012log1mexp].

For a proportion that may equal zero or one, an inflated beta distribution
combines point masses at the endpoints with a density between them
[@ospina2010inflated]. An endpoint probability and an interior density are
different quantities; their contributions are scored against the appropriate
discrete or continuous measure.

Ordered-categorical families, conditional and inline, and the
zero-inflated, hurdle, and mixture distributions PyTorch does not ship.

::: quivers.continuous.ordered

::: quivers.continuous._ordered

::: quivers.continuous._zip_hurdle
