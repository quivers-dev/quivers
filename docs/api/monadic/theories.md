# Typeclass Theories (panproto mirror)

A monad theory records [operations and equations](../../guides/effects.md#typeclass-hierarchy) describing [how effectful computations
compose](../../semantics/effects.md#2-typeclass-denotations) [@moggi1991notions; @wadler1992essence]. Tooling uses this representation to inspect the expected interface.
Implementations must satisfy its equations for the values they support.

`TypeclassTheory` records of the sorts, operations, and laws of each
typeclass: `ThFunctor`,
`ThApplicative`, `ThMonad`, `ThAlternative`, `ThMonadPlus`,
`ThMonadTrans`, `ThFoldable`, and `ThTraversable`.

::: quivers.monadic.theories
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The theory constants are documented on the
[monadic overview](typeclasses.md#package-constants).
