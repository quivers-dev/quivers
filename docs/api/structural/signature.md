# Signatures

A signature declares the shapes of valid structured values. An expression tree might
allow a constant constructor and an application constructor with two expression
children. Sorts distinguish kinds of child values; binders introduce variables with a
defined scope. The signature specifies valid structure before an encoder assigns
numerical vectors.

Milewski's [F-algebras chapter](https://bartoszmilewski.com/2017/02/28/f-algebras/) builds recursive datatypes from their constructors before introducing folds [@milewski2017falgebras].

See [constructors, sorts, and signatures](../../guides/structural-compression-signatures-and-encoders.md#signature-blocks) and
[constructing values for an autoencoder](../../tutorials/qvr/12-structural-autoencoders.md#2-construct-typed-host-data).

Multi-sorted algebra signatures with constructors, typed binders, and
optional vertex and edge kinds for graph-shaped signatures. These
objects are the runtime carriers of `signature { … }` DSL declarations.

The de Bruijn discipline is enforced structurally: a
`BoundVar(i)` term carries an integer index; binders push
a fresh `ContextEntry(var_sort, embedding, type_term)` onto
an implicit context Γ tracked by the encoder / decoder
runtime. Binder variables may carry an annotation sort:
the variable's type is stored alongside its embedding in Γ.

::: quivers.structural.signature
    options:
      filters: ["!^_", "!^(DataLeaf|TermArg|SortKind|EMPTY_CONTEXT)$"]

## Type aliases and constants

::: quivers.structural
    options:
      members:
        - DataLeaf
        - TermArg
        - SortKind
        - EMPTY_CONTEXT
      show_root_heading: false
      show_root_toc_entry: false
