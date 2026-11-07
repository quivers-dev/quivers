# Morphisms

A finite morphism stores a weight for each input–output pair. [Composing a user-to-feature table with a feature-to-item table](../../guides/morphisms.md#composition-the-operator) eliminates the feature axis and produces
user-to-item weights. An observed morphism holds supplied values; a latent morphism
holds trainable parameters. The chosen algebra determines how those values combine.

Base morphism classes, composition operations, and shared morphism utilities.

::: quivers.core.morphisms
