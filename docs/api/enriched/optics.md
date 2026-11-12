# Optics

An [optic](https://ncatlab.org/nlab/show/lens+%28in+computer+science%29) focuses on part of a
structured value and keeps what is needed to rebuild the whole. A `Lens`
[reads and updates a component of a product](../../guides/enriched.md#optics), a `Prism` handles one
alternative of a tagged union, and `Adapter` and `Grate` complete the family. `compose_optics`
composes two optics, and their laws constrain how reading and updating interact. The classes
implement these operations on finite tensors.

::: quivers.enriched.optics
