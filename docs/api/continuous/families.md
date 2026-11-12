# Continuous Families

The module gives the families of continuous distributions
([family registry](../../guides/continuous-families.md#family-registry)), each chosen by the values
a response can take: a Normal for real values, and other families for positive or bounded
supports. A [conditional family](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families)
computes its parameters from its input through a learned parameter map, while a fixed call such as
`Normal(loc=0, scale=1)` has constant parameters.

::: quivers.continuous.families
