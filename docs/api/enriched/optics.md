# Optics

An [optic](https://ncatlab.org/nlab/show/lens+%28in+computer+science%29) focuses on part of a
structured value and keeps what is needed to rebuild the whole. A `Lens`
[reads and updates a component of a product](../../guides/enriched.md#optics), a `Prism` handles one
alternative of a tagged union, `Adapter` is a pair of mutually inverse morphisms, which its constructor checks, and `Grate`
completes the family. `Lens.put` writes a component back, and the lens laws, which constrain how
reading and writing interact, hold for every `Lens` [@foster2007combinators]. `compose_optics`
composes two optics into a `ComposedOptic`. The classes
implement these operations on finite tensors.

::: quivers.enriched.optics
