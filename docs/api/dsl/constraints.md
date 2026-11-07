# Constraint Solver

[Constraint checking reports problems](../../guides/dsl-overview.md#error-handling) visible in the parsed module, including invalid
effect conventions and unresolved bundle members. It supplies diagnostics before runtime
objects are constructed. Subsequent elaboration and type checking use the resolved
declarations to perform further checks.

`check_constraints(module)` walks a parsed `Module` and reports
well-formedness violations (residuated context, effect-name
convention, bundle-member resolvability) without invoking the full
compiler.

```python
from quivers.dsl import Violation, check_constraints
```

::: quivers.dsl.constraints
