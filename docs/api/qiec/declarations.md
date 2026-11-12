# Declarations

A [family declaration](../../reference/qvr/types-and-indexed-families.md#family-declarations)
declares an [inductive family](https://ncatlab.org/nlab/show/inductive+family) of types
[@dybjer1994inductive], and its constructor declarations say how values are formed. A
[vector family](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check) may be indexed by
its element type and its length. Each constructor declaration records the indices and fields its
values have, so that checking recovers those facts when a value is matched.

::: quivers.qiec.declarations
