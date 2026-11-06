# Optics

An optic focuses on part of a structured value while retaining what is needed to rebuild
the whole. A lens selects and updates a component of a product; a prism handles one
alternative of a tagged union. Their laws constrain how reading and updating interact.
The tensor interfaces below implement these operations in the library’s finite
representation.

See [selecting and updating with optics](../../guides/enriched.md#optics) and
[the weighted relations used by these constructions](../../semantics/setting.md#2-v-enriched-relations).

For the general enriched constructions behind this finite interface, see [@kelly1982basic].

Optics and lenses in enriched category theory.

::: quivers.enriched.optics
