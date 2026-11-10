# Signatures

A signature declares the [shapes of valid structured values](../../guides/structural-compression-signatures-and-encoders.md#signature-blocks). An [expression tree](../../tutorials/qvr/12-structural-autoencoders.md#2-construct-typed-host-data) might
allow a constant constructor and an application constructor with two expression
children. Sorts distinguish kinds of child values; binders introduce variables with a
defined scope. The signature specifies valid structure before an encoder assigns
numerical vectors.

Multi-sorted algebra signatures with constructors, typed binders, and
optional vertex and edge kinds for graph-shaped signatures. These
objects are the runtime carriers of `signature { … }` DSL declarations.

The de Bruijn discipline is enforced structurally: a
`BoundVar(i)` term carries an integer index; binders push
a fresh `ContextEntry(var_sort, embedding, type_term)` onto
an implicit context Γ tracked by the encoder / decoder
runtime. Binder variables may carry an annotation sort:
the variable's type is stored alongside its embedding in Γ.

Terms are immutable values with structural equality. Two terms are
equal exactly when their operators agree and their arguments agree
position by position, recursively through nested subterms, data
leaves, binder annotations, and de Bruijn indices; hashing agrees with
equality, so terms work as dictionary keys and set members. Data
leaves compare by type as well as value: `make_term("C", 1)`,
`make_term("C", 1.0)`, and `make_term("C", True)` are three distinct
terms. A reconstruction check such as `decoded == original` therefore
holds only when the decoder recovered every part of the term.
Arguments outside the `TermArg` union are rejected when the term is
constructed.

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
