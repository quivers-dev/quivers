# Day Convolution

[Day convolution](https://ncatlab.org/nlab/show/Day+convolution) combines two presheaves over a
monoidal category by summing, for each requested output object, the contributions of every pair of
objects whose tensor product maps to it [@kelly1982basic]
([Day convolution](../../guides/enriched.md#day-convolution)). Ordinary convolution, which sums
over every way of splitting an index, is the case of a discrete monoid. Here presheaves and
morphisms are [finite tensors](../../semantics/setting.md#2-v-enriched-relations), and the chosen
`Algebra` fixes how the contributions combine.

::: quivers.enriched.day_convolution
