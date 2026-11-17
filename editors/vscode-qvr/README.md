# QVR for VS Code and Cursor

The first-party QVR extension provides language support for Visual Studio Code
and Cursor. It combines a TextMate grammar for immediate colorization with
`qvr-lsp` for typed language features.

The extension recognizes the complete current surface: categorical and
probabilistic declarations, indexed families, constructors and motive-checked
cases, `define` computations, lexical effect instances, row-polymorphic
effects, handlers, graded resumptions, deductions, schema parsers, and
structural encoder/decoder graphs. Program headers separate static parameters
in `[...]` from typed runtime inputs in `(...)`. Typed parameters receive
completion, hover, navigation, and renaming in their lexical scope; tensor shapes
may refer to preceding static binders. Schemas and rules also bind static
parameters in `[...]`. `object Row : FinSet _` remains checkable before data
fix the extent; the environment browser and scoped inspection in the shared
REPL/kernel session retain the symbolic declaration.

## Install

First install Quivers with its language-server dependencies:

```sh
pip install 'quivers[lsp]'
qvr-lsp --help
```

Then install `vscode-qvr-0.8.1.vsix` from the corresponding Quivers GitHub
release:

```sh
code --install-extension vscode-qvr-0.8.1.vsix
# or
cursor --install-extension vscode-qvr-0.8.1.vsix
```

To package the extension from a checkout:

```sh
cd editors/vscode-qvr
npm ci
npm run package
code --install-extension vscode-qvr-0.8.1.vsix
```

## Language features

TextMate supplies highlighting while the server starts. The LSP then adds:

- source-ranged parser, checker, QIEC, and transpile-target diagnostics;
- hover, go to definition, references, and safe QVR-symbol renaming;
- nested document symbols and context-sensitive completion;
- semantic tokens derived from the same vocabulary as the compiler;
- canonical formatting; and
- live capability checking for any of the eleven transpilation targets.

The server uses the same checked module as `qvr check`, `qvr run`, the REPL,
and the transpilers. An editor diagnostic is thus not a separate approximation
of the language's type-and-effect rules.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `qvr.lsp.enabled` | `true` | Start the QVR language server. |
| `qvr.lsp.path` | empty | Executable path; `${workspaceFolder}` is expanded. |
| `qvr.lsp.args` | `[]` | Extra arguments passed when the server starts. |
| `qvr.transpileTarget` | none | Publish live capability diagnostics for one target. |

The extension resolves the server from the configured path, a workspace
`.venv`, the active `VIRTUAL_ENV`, and finally `PATH`. Changing
`qvr.transpileTarget` rechecks open files without reparsing them. Valid targets
are `bugs`, `church`, `edward2`, `gen`, `jags`, `numpyro`, `pymc`, `pyro`,
`stan`, `turing`, and `webppl`.

For instance, this workspace configuration checks whether every reachable
computation can lower to Pyro:

```json
{
  "qvr.transpileTarget": "pyro"
}
```

An empty `qvr.lsp.path` enables environment discovery. An explicit value names
an executable path or a command on `PATH`; a missing explicit executable produces
a launch error. Changing the path, arguments, or enabled setting restarts the
client. Saved and untitled QVR documents receive language-server features.
VS Code and compatible editors must support the extension's declared minimum
version, which follows the language-client dependency.

## Verify the language surface

Open
[`docs/tutorials/qvr/source/authored-handler.qvr`](https://github.com/quivers-dev/quivers/blob/main/docs/tutorials/qvr/source/authored-handler.qvr).
`effect`, `instance`, `handler`, `resumes`, `define`, `handle`, and `perform`
should be colored before the server starts. Once it attaches, hovering
`robustify` should show a pure residual effect row, while `robust_request`
retains its open `Robust` effect.

If plain syntax colors appear but typed features do not, inspect the QVR output
channel and run `qvr-lsp --help` in the editor's environment. The full launch
order, capability table, and setup for other editors are documented in
[Editor support and syntax highlighting](https://quivers.dev/getting-started/highlighting/)
and the [REPL and LSP guide](https://quivers.dev/guides/repl-and-lsp/).

## Release maintenance

After changing `package.json`, the TextMate grammar, language configuration,
extension source, compiled JavaScript, or this README, run:

```sh
cd editors/vscode-qvr
npm ci
npm run package
cd ../..
python tools/check_editor_release.py --full
```

The release gate compares the checked-in VSIX with these source files and its
production dependencies. A Quivers release must never attach a stale VSIX.
