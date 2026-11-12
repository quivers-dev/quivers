# Installation

quivers requires Python 3.14 or later.

## From PyPI

```bash
pip install quivers
```

This installs PyTorch, [didactic](https://pypi.org/project/didactic/),
[panproto](https://pypi.org/project/panproto/), `panproto-grammars-all`,
Pygments, and the tree-sitter bindings as dependencies, along with the `qvr`
command. To run on a GPU, install the [PyTorch build](https://pytorch.org/get-started/locally/)
for your platform before installing quivers.

Wheels for Linux on x86-64, macOS on ARM64, and Windows on AMD64 include the
compiled QVR parser. On other platforms pip builds the package from the source
distribution, which needs a C compiler (`cc`, Clang, GCC, or MSVC).

Check the installation:

```python
import quivers
print(quivers.__version__)
```

## Optional features

Each extra installs the dependencies of one feature that the base package does
not need. Extras combine, as in `pip install 'quivers[repl,lsp]'`.

| Extra | Feature | Packages it adds |
| --- | --- | --- |
| `repl` | `qvr repl` and the Jupyter kernel | [Textual](https://textual.textualize.io/), [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/), [rich](https://rich.readthedocs.io/), [ipykernel](https://ipykernel.readthedocs.io/), [jupyter_client](https://jupyter-client.readthedocs.io/) |
| `lsp` | the `qvr-lsp` language server | [pygls](https://github.com/openlawlibrary/pygls), `lsprotocol` |
| `data` | data frame adapters | [narwhals](https://narwhals-dev.github.io/narwhals/), [scikit-learn](https://scikit-learn.org/) |
| `diagnostics` | posterior diagnostics and trace files | NumPy, [ArviZ](https://python.arviz.org/), [xarray](https://docs.xarray.dev/), [netCDF4](https://unidata.github.io/netcdf4-python/) |
| `formulas` | R-style formulas | everything in `data` and `diagnostics`, plus [formulae](https://bambinos.github.io/formulae/) |
| `targets` | running programs transpiled to Pyro or PyMC | [Pyro](https://pyro.ai/), [PyMC](https://www.pymc.io/), SciPy |

For instance:

```bash
pip install 'quivers[formulas]'
```

The `repl` and `lsp` extras enable these commands:

| Command | Extra | What it does |
| --- | --- | --- |
| `qvr repl` | `repl` | Starts the REPL: a terminal interface built on Textual, or a prompt_toolkit prompt when Textual is not installed |
| `qvr-kernel install`, `qvr kernel install` | `repl` | Registers the `quivers` Jupyter kernel |
| `qvr-lsp`, `qvr lsp` | `lsp` | Runs the language server over standard input and output |

[Interactive tools](../guides/repl-and-lsp.md) describes each one.

## From source

```bash
git clone https://github.com/quivers-dev/quivers
cd quivers
pip install -e .
```

A source build compiles the QVR parser and needs a C compiler. An editable
checkout parses with the grammar in `grammars/qvr/src` and rebuilds its parser
after that grammar changes. An installed wheel loads only the parser packaged
with it, for parsing, highlighting, and `qvr migrate` alike. If those files are
missing or do not match one another, quivers raises an error instead of using
the QVR grammar in `panproto-grammars-all`, which provides only the grammars of
the transpilation targets.

The `dev` extra adds the test and lint tools, and the `docs` extra adds what is
needed to build the documentation site:

```bash
pip install -e ".[dev]"
pytest tests/
```

[Contributing](../developer/contributing.md) describes the test tiers and the
documentation build.
