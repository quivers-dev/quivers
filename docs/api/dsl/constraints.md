# Constraint Solver

`check_constraints(module)` walks a parsed `Module` and reports violations of well-formedness, such
as an invalid residuated context, an effect name that breaks the naming convention, or a bundle
member that does not resolve, without running the compiler
([error handling](../../guides/dsl-overview.md#error-handling)). Elaboration and type checking
perform further checks on the resolved declarations.

```python
from quivers.dsl import Violation, check_constraints
```

::: quivers.dsl.constraints
