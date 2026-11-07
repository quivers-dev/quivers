# Typeclass Hierarchy

The typeclasses group operations for [mapping, combining, and sequencing computations](../../guides/effects.md#typeclass-hierarchy). An
abstract base class identifies the methods an instance supplies. Its accompanying laws
specify how those methods must agree, such as whether two successive mappings equal one
mapping of the composite function [@moggi1991notions; @wadler1992essence].

ABCs for `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`,
`Foldable`, `Traversable`, and `MonadTrans`. An applicative interface
combines computations whose structure is fixed before their results are
available [@mcbride2008applicative]; monadic bind can choose the next
computation using an earlier result.

::: quivers.monadic.typeclasses

## Package constants

The typeclass theories of `quivers.monadic`, each a `TypeclassTheory`.

::: quivers.monadic
    options:
      members:
        - ThFunctor
        - ThApplicative
        - ThMonad
        - ThAlternative
        - ThMonadPlus
        - ThMonadTrans
        - ThFoldable
        - ThTraversable
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
