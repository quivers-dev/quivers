# Pygments lexer

The [Pygments](https://pygments.org/) lexer for `.qvr` source, registered under the `qvr` alias
through the `pygments.lexers` entry point, classifies spans as keywords, identifiers, literals, and
other tokens for [syntax highlighting](../../reference/qvr/execution-and-tooling.md#grammar-ownership-and-panproto).
Parsing and type checking belong to the compiler and to
[`qvr check`](../../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing).

::: quivers.dsl.pygments_lexer
