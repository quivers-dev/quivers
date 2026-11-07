# Pygments lexer

The lexer classifies source spans as keywords, identifiers, literals, and other token
kinds for [syntax highlighting](../../reference/qvr/execution-and-tooling.md#grammar-ownership-and-panproto). The compiler and [`qvr check`](../../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing) perform parsing and type
checking; editor highlighting uses these token classifications to display the source.

The Pygments lexer for `.qvr` source, registered under the `qvr` alias
through the `pygments.lexers` entry point.

::: quivers.dsl.pygments_lexer
