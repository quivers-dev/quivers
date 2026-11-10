# Optics

An optic focuses on part of a structured value while retaining what is needed to rebuild
the whole. A lens [selects and updates a component of a product](../../guides/enriched.md#optics); a prism handles one
alternative of a tagged union. Their laws constrain how reading and updating interact.
The tensor interfaces below implement these operations in the library’s finite
representation.

Optics and lenses in enriched category theory.

::: quivers.enriched.optics
