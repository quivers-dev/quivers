# Utilities

These helpers keep fuzzy computations away from numerical singularities, such as the logarithm of
zero, by clamping to a margin. The clamp moves exact boundary values and changes
[gradients](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#computing-gradients)
[@pytorchAutogradTutorial] near them, which matters when a model distinguishes impossible events
from very unlikely ones or [learns morphism parameters](../../guides/morphisms.md#learning-and-gradients).
They are importable from `quivers.core`, and the default margin `EPS` is documented on the
[core overview](objects.md#package-constants).

::: quivers.core._util
    options:
      filters:
        - "!^_"
        - "!^EPS$"
