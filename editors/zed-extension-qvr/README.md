# QVR for Zed

This [Zed](https://zed.dev) extension provides QVR parsing, syntax highlighting,
bracket/comment behavior, and language-server integration for `.qvr` files.
The parser recognizes categorical and probabilistic
declarations, indexed families, constructors and cases, computations, lexical
effect instances, open rows, handlers, and graded resumptions.

With `qvr-lsp` attached, Zed also receives parser and type-and-effect
diagnostics, hovers, definitions, references, document symbols, completion,
formatting, semantic tokens, and QIEC capability diagnostics.

## Prerequisite

Install Quivers with its LSP dependencies and confirm that the executable is
available to Zed:

```sh
pip install 'quivers[lsp]'
qvr-lsp --help
```

The Rust extension adapter launches `qvr-lsp` from the worktree environment. If Quivers is installed in a
project virtual environment, start Zed from that activated environment or set
Zed's language-server binary path to the environment's `qvr-lsp` executable.

For an explicit interpreter environment and target diagnostics, configure Zed:

```json
{
  "lsp": {
    "qvr-lsp": {
      "binary": {
        "path": "/path/to/environment/bin/qvr-lsp",
        "arguments": ["--target", "pyro"]
      },
      "settings": {"qvr": {"transpileTarget": "pyro"}}
    }
  }
}
```

The adapter uses Zed's supported
[language-server extension API](https://zed.dev/docs/extensions/languages#language-servers)
and forwards the configured workspace settings. It does not install Python or
Quivers. Install the LSP extra in the environment selected above.

## Install (dev / local)

In Zed's Extensions view, choose **Install Dev Extension** and select
`editors/zed-extension-qvr` from this checkout. Reinstall the dev
extension after changing its manifest or packaged queries. `.qvr`
files then use the tree-sitter grammar declared by the extension.

## Install (when published)

Once this extension is submitted to the public Zed extension registry,
it will be installable from Zed's `extensions:` panel by name (`QVR`).
Until then, use the dev-extension route above.

## Layout

```
editors/zed-extension-qvr/
├── extension.toml                     extension manifest
├── Cargo.toml / Cargo.lock            Rust adapter dependencies
├── src/lib.rs                         language-server launch adapter
├── languages/qvr/
│   ├── config.toml                    Zed language config (file types, comments)
│   └── highlights.scm                 tree-sitter highlight queries
└── README.md
```

The grammar manifest pins an immutable Quivers commit. The `highlights.scm`
file mirrors the canonical
[`grammars/qvr/queries/highlights.scm`](../../grammars/qvr/queries/highlights.scm).
Do not edit that mirror by hand: running
`python grammars/qvr/queries/_generate.py` from the repository root rewrites
both copies. CI fetches the pinned revision, compares its parser inputs and queries with
the current grammar, and builds the Rust adapter for `wasm32-wasip2`. Tests fail
if the pin is mutable, if either query drifts, or if
the QIEC vocabulary diverges across tree-sitter, TextMate, Pygments, the REPL,
and LSP semantic tokens.

## Verify the language surface

Open
[`docs/tutorials/qvr/source/authored-handler.qvr`](../../docs/tutorials/qvr/source/authored-handler.qvr).
`effect`, `instance`, `handler`, `resumes`, `define`, `handle`, and `perform`
should receive syntax colors immediately. Once `qvr-lsp` starts, hovering
`robustify` should show a pure residual row, while `robust_request` retains its
open `Robust` effect.

For the full editor setup and troubleshooting guide, see
[Editor support and syntax highlighting](../../docs/getting-started/highlighting.md).

## Build checks

```sh
rustup target add wasm32-wasip2
cargo build --locked --release --target wasm32-wasip2
```

From the repository root, run `python tools/check_editor_release.py --grammar-pin`
after fetching the `rev` recorded in `extension.toml`. A grammar change requires
an immutable revision containing those parser and query bytes before updating
the extension pin.
