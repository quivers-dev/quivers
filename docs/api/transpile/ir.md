# Structural IR

The structural [intermediate representation](../../semantics/transpile-architecture.md#2-the-ir) (IR)
that renderers consume records a program's structure without the syntax of any target. A
distribution call keeps its parameters and support, and each renderer decides how its target
spells it, which separates a modeling operation from a choice of format.

::: quivers.transpile.ir
