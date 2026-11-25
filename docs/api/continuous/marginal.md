# Marginalized responses

The score of a `marginalize` block holds the density of the block's responses with the latent summed
out ([response laws](../../semantics/programs.md#272-laws-of-marginalized-responses)).
`MarginalizedResponse` is the law such a response is drawn from when a run is given no value for it,
bound to the values of one run. `draw_classes` draws the latents of nested grouped blocks jointly
from the `GroupLevel` of each.

::: quivers.continuous.marginal
