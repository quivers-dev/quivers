# Categories

A category here is a [syntactic category](https://en.wikipedia.org/wiki/Syntactic_category) of a
[categorial grammar](https://en.wikipedia.org/wiki/Categorial_grammar), such as a noun phrase or a
function from a noun phrase to a sentence [@lambek1958mathematics; @moortgat1997categorial], not a
category in the mathematical sense. The module gives atomic, slash, product, unit, and modal
categories, whose [slash and product types](../../semantics/schemas.md#3-residuated-type-formers) say
how constituents combine. `CategorySystem.from_generators()` enumerates the categories built from
the atoms and the type constructors up to a given depth, which gives the finite inventories that
CCG, Lambek, multimodal type-logical, and custom rule systems use.

::: quivers.stochastic.categories
    options:
      filters: ["!^_", "!^[A-Z][A-Z_]+$"]

## Constructor names

::: quivers.stochastic
    options:
      members:
        - BUILTIN_CONSTRUCTOR_NAMES
      show_root_heading: false
      show_root_toc_entry: false
