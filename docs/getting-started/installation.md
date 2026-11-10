# Installation

For a first model, install the base package from PyPI and verify the import.
The additional packages listed below are installed automatically; you do not
need to learn their APIs. Add `formulas` for dataframe regressions,
`diagnostics` for posterior assessment, or `repl,lsp` for interactive editing.
A development checkout is needed only when changing Quivers itself or running
its repository tests.

## Prerequisites

- **Python** >= 3.14
- **PyTorch** >= 2.0
- **didactic** >= 0.17.1
- **panproto** >= 0.74.4 (provides schema and lens operations plus historical-object validation)
- **panproto-grammars-all** >= 0.74.4 (provides the eleven transpiler-target grammars)
- **Pygments** >= 2.10, **tree-sitter** >= 0.21

The didactic, panproto, panproto-grammars-all, Pygments, and
tree-sitter packages are pulled in automatically by
`pip install quivers`. Release wheels for CPython 3.14 on manylinux x86-64,
macOS ARM64, and Windows AMD64 include the native QVR parsers and do not
require a C compiler at installation time or first parse. Other platforms
install from the source distribution and need a C compiler.

## From PyPI

```bash
pip install quivers
```

## From Source

To install directly from source:

```bash
git clone https://github.com/quivers-dev/quivers
cd quivers
pip install -e .
```

A source or editable build requires a C compiler (`cc`, Clang, GCC, or MSVC).
The build compiles the generated current parser; editable checkouts may also
rebuild it into a content-addressed development cache after grammar changes.

## Development Installation

If you intend to run tests and contribute:

```bash
pip install -e ".[dev]"
```

This adds:

- `pytest >= 7.0`: test runner
- `pytest-cov`: coverage reporting
- `ruff`: linter and formatter
- `pyright`: static type checker
- `numpy`, `pandas`, `polars`, `pyarrow`: data-frame fixtures used by the data-encoding tests

The `[docs]` extra (MkDocs, Material for MkDocs, mkdocstrings[python],
PyMdown Extensions, and Pygments) is needed to build the documentation
site locally. Material provides navigation and search; the Quivers styles in
`docs/css/custom.css` define the visual system.

## Optional feature groups

These extras install dependencies for features that the core package does not
require:

```bash
pip install 'quivers[repl]'         # qvr repl + Jupyter kernel
pip install 'quivers[lsp]'          # qvr-lsp language server
pip install 'quivers[data]'         # narwhals + scikit-learn data adapters
pip install 'quivers[diagnostics]'  # ArviZ / xarray / netCDF4 trace I/O
pip install 'quivers[formulas]'     # data + diagnostics + formulae DSL
pip install 'quivers[repl,lsp]'     # any combination is fine
```

`[repl]` pulls in [Textual](https://textual.textualize.io/),
[prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/),
[rich](https://rich.readthedocs.io/),
[ipykernel](https://ipykernel.readthedocs.io/), and
[jupyter_client](https://jupyter-client.readthedocs.io/).

`[lsp]` pulls in [pygls](https://github.com/openlawlibrary/pygls) and
`lsprotocol`.

`[data]` pulls in [narwhals](https://narwhals-dev.github.io/narwhals/)
and [scikit-learn](https://scikit-learn.org/); `[diagnostics]` adds
[ArviZ](https://python.arviz.org/), [xarray](https://docs.xarray.dev/),
and [netCDF4](https://unidata.github.io/netcdf4-python/). `[formulas]`
is a superset that also brings in
[formulae](https://bambinos.github.io/formulae/) for the R-style
formula DSL.

After installing the extras you have these new console scripts:

| Command | Provided by | What it does |
| --- | --- | --- |
| `qvr repl` | `[repl]` | Four-pane Textual TUI (or prompt_toolkit fallback) |
| `qvr-lsp` | `[lsp]` | LSP 3.17 server over stdio |
| `qvr lsp` | `[lsp]` | The same, as a `qvr` subcommand |
| `qvr-kernel install` | `[repl]` | Register the Jupyter kernelspec |
| `qvr kernel install` | `[repl]` | Same, as a subcommand |

The [interactive tools](../guides/repl-and-lsp.md) support program inspection,
entry execution, and editor diagnostics after installation.

## Verify Installation

Check that the import works:

```python
import quivers
print(quivers.__version__)
```

Run the test suite:

```bash
pytest tests/
```

## Dependencies

Quivers depends on:

- **torch** (>= 2.0): differentiable tensors and [automatic differentiation](../tutorials/python/01-first-quiver.md#working-as-a-differentiable-module)
- **didactic** (>= 0.17.1): typed-data, indexed-family checking, and exact extension-lowering boundary used by QVR-to-QIEC checking
- **panproto** (>= 0.74.4): schema and theory operations used to check indexed declarations and extract a `Schema` from each `.qvr` program for diff/migrate workflows
- **panproto-grammars-all** (>= 0.74.4): supplies the eleven target-language grammars used by the transpiler pipeline; it does not define QVR syntax
- **Pygments** (>= 2.10): in-tree `qvr` lexer for documentation and notebooks
- **tree-sitter** (>= 0.21): runtime bindings for the QVR grammar

Quivers' platform wheels ship the generated QVR parser source together with a
native library whose manifest binds it to that source and its tree-sitter
metadata. The same source-bound pairing is included for every parser snapshot
used by `qvr migrate`. Installed wheels load these verified libraries directly,
so neither parsing, highlighting, nor migration launches a compiler. An
editable checkout instead prefers its current `grammars/qvr/src` and may build a
content-addressed library for development. Quivers refuses to load a parser if
the wheel's source, library, or manifest is absent or inconsistent; it does not
substitute the QVR parser from `panproto-grammars-all`.

The optional capability
extras (`[repl]`, `[lsp]`, `[data]`, `[diagnostics]`, `[formulas]`)
install the interactive tools and data or diagnostics integrations
described above.
