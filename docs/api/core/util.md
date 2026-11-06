# Utilities

These helpers keep fuzzy computations away from numerical singularities, such as taking
a logarithm of zero. A numerical margin can change exact boundary values, so consult
each helper’s clipping convention when a model distinguishes impossible events from very
unlikely events. Most users encounter these functions through the higher-level algebra
operations.

PyTorch's [computing gradients example](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#computing-gradients) shows how `backward()` populates parameter gradients [@pytorchAutogradTutorial].

See [differentiating through morphisms](../../guides/morphisms.md#learning-and-gradients) and
[a complete composition example](../../tutorials/python/01-first-quiver.md#composition).

Numerical helpers for stable fuzzy-logic operations, importable from
`quivers.core`. The default margin `EPS` is documented on the
[core overview](objects.md#package-constants).

::: quivers.core._util
    options:
      filters:
        - "!^_"
        - "!^EPS$"
