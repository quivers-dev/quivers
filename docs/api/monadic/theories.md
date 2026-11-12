# Typeclass Theories (panproto mirror)

Each `TypeclassTheory` records the sorts, operations, and laws of one
[type class](https://en.wikipedia.org/wiki/Type_class) as a panproto theory
([type class hierarchy](../../guides/effects.md#typeclass-hierarchy)), so that tooling can inspect
the interface an implementation must satisfy [@moggi1991notions; @wadler1992essence]. The theories
are `ThFunctor`, `ThApplicative`, `ThMonad`, `ThAlternative`, `ThMonadPlus`, `ThMonadTrans`,
`ThFoldable`, and `ThTraversable`, and their
[denotations](../../semantics/effects.md#2-typeclass-denotations) are given in the semantics.

::: quivers.monadic.theories
    options:
      filters:
        - "!^_"
        - "!^[A-Z][A-Z0-9_]*$"

The theory constants are documented on the
[monadic overview](typeclasses.md#package-constants).
