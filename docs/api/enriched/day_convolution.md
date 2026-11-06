# Day Convolution

Ordinary convolution combines all ways of splitting an output index into two inputs. Day
convolution extends this idea to a monoidal combination of objects: combine
contributions from pairs of objects that can produce the requested output. Here presheaf
values and morphisms are represented by finite tensors, and the selected algebra
determines how those contributions aggregate.

See [the finite implementation of Day convolution](../../guides/enriched.md#day-convolution) and
[the weighted relations used by these constructions](../../semantics/setting.md#2-v-enriched-relations).

For the general enriched constructions behind this finite interface, see [@kelly1982basic].

Day convolution product for enriched presheaves.

::: quivers.enriched.day_convolution
