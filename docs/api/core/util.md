# Utilities

These helpers keep fuzzy computations away from numerical singularities, such as taking
a logarithm of zero. A numerical margin can change exact boundary values, so consult
each helper’s clipping convention when a model distinguishes impossible events from very
unlikely events. Clipping also affects [gradients](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#computing-gradients)
[@pytorchAutogradTutorial] through these operations; this matters when
[learning morphism parameters](../../guides/morphisms.md#learning-and-gradients).

Numerical helpers for stable fuzzy-logic operations, importable from
`quivers.core`. The default margin `EPS` is documented on the
[core overview](objects.md#package-constants).

::: quivers.core._util
    options:
      filters:
        - "!^_"
        - "!^EPS$"
