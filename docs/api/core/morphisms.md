# Morphisms

A finite morphism stores one weight for each pair of an input and an output, as a tensor.
[Composing](../../guides/morphisms.md#composition-the-operator) a table from users to features with a
table from features to items sums out the feature axis under the chosen `Algebra` and gives
weights from users to items. An `ObservedMorphism` holds given values, and a `LatentMorphism` holds
learnable parameters. The module defines the base morphism classes, their composition, and shared
morphism utilities.

::: quivers.core.morphisms
