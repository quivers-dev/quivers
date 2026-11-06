# Pygments lexer

The lexer classifies source spans as keywords, identifiers, literals, and other token
kinds for syntax highlighting. The compiler and `qvr check` perform parsing and type
checking; editor highlighting uses these token classifications to display the source.

See [the grammar used by editor tooling](../../reference/qvr/execution-and-tooling.md#grammar-ownership-and-panproto) and
[using diagnostics during development](../../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing).

The Pygments lexer for `.qvr` source, registered under the `qvr` alias
through the `pygments.lexers` entry point.

::: quivers.dsl.pygments_lexer
