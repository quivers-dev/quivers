# Morphisms

A finite morphism stores a weight for each input–output pair. Composing a user-to-feature table with a feature-to-item table eliminates the feature axis and produces
user-to-item weights. An observed morphism holds supplied values; a latent morphism
holds trainable parameters. The chosen algebra determines how those values combine.

Milewski's [introduction to composition](https://bartoszmilewski.com/2014/11/04/category-the-essence-of-composition/) begins with ordinary functions and the laws for connecting them [@milewski2014composition].

See [composing morphisms in Python](../../guides/morphisms.md#composition-the-operator) and
[a complete composition example](../../tutorials/python/01-first-quiver.md#composition).

For a calculation before the general theory, see [@milewski2014composition].

Base morphism classes, composition operations, and shared morphism utilities.

::: quivers.core.morphisms
