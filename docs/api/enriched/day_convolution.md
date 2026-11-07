# Day Convolution

Ordinary convolution combines all ways of splitting an output index into two inputs. [Day
convolution](../../guides/enriched.md#day-convolution) extends this idea to a monoidal combination of objects: combine
contributions from pairs of objects that can produce the requested output [@kelly1982basic]. Here presheaf
values and morphisms are represented by [finite tensors](../../semantics/setting.md#2-v-enriched-relations), and the selected algebra
determines how those contributions aggregate.

Day convolution product for enriched presheaves.

::: quivers.enriched.day_convolution
