# Structural IR

The structural [intermediate representation](../../semantics/transpile-architecture.md#2-the-ir) (IR)
that renderers consume records a program's structure without the syntax of any target. A
distribution call keeps its parameters and support, and each renderer decides how its target
spells it, which separates a modeling operation from a choice of format.

Deterministic bindings and computation calls retain their checked scalar carrier
through `ConstraintSpec`: `CSBoolean` represents Boolean values, `CSInteger`
represents unrestricted integers, and real-valued bindings use `CSReal`.
A runtime `Nat` input additionally retains its nonnegative boundary restriction;
arithmetic on that input still has the integer carrier.

::: quivers.transpile.ir
