# Structural IR

An [intermediate representation (IR)](../../semantics/transpile-architecture.md#2-the-ir) records program structure without committing to one
target language’s syntax. A distribution call can retain its parameters and support
while renderers decide how that target spells it. This separation helps distinguish a
modeling operation from a formatting choice.

The structural intermediate representation renderers consume.

::: quivers.transpile.ir
