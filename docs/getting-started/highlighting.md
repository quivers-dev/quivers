# Editor support and syntax highlighting

`.qvr` files are highlighted by two complementary systems, depending on what reads the file:

| Consumer | Engine | Comes from |
|---|---|---|
| Python tools: mkdocs, Sphinx, Jupyter, IPython, `pygmentize` | [Pygments](https://pygments.org/) | bundled with the `quivers` PyPI package |
| Editors: Neovim, Helix, Emacs, Zed | [tree-sitter](https://tree-sitter.github.io/) | the grammar and editor assets shipped by `quivers` |
| VS Code and Cursor | TextMate for initial colorization; QVR LSP for semantic tokens | the first-party `vscode-qvr` extension |
| GitHub.com source view, GitHub gists | [Linguist](https://github.com/github-linguist/linguist) | language registration (not yet upstream) |

This page documents how to wire each one up.

Syntax highlighting identifies words and punctuation; semantic checking
asks whether the declarations and computations fit together. A colorful
`Normal(...)` call can still have an invalid argument or a missing runtime
input. The language server adds type-aware feedback, while [`qvr check`](../tutorials/qvr/13-release-workflow.md#1-check-the-source-and-one-target) gives
the corresponding explicit check from a terminal.

Choose the setup for the editor or notebook you use. The parser provenance
details below matter when those tools disagree about valid syntax; they are
not additional mathematical prerequisites for writing a model.

## Pygments (Python doc and notebook tools)

Quivers ships a Pygments lexer registered as an entry point. After

```bash
pip install quivers
```

every Pygments-based tool finds `.qvr` files automatically. No further configuration is needed for mkdocs, Sphinx, `pygmentize`, IPython / Jupyter cell magic (`%%highlight qvr`), or any other consumer that asks Pygments to choose a lexer by filename or by `name="qvr"`.

To confirm the lexer is visible:

```bash
python -c "from pygments.lexers import get_lexer_by_name; print(get_lexer_by_name('qvr').name)"
# QVR
```

To highlight a file to a terminal:

```bash
pygmentize -l qvr -O style=monokai docs/examples/source/bayesian_regression.qvr
```

The lexer source lives at [`src/quivers/dsl/pygments_lexer.py`](https://github.com/quivers-dev/quivers/blob/main/src/quivers/dsl/pygments_lexer.py); the highlight rules and keyword list are kept in sync with the [tree-sitter grammar](https://github.com/quivers-dev/quivers/tree/main/grammars/qvr) by `tests/test_dsl_extensions.py`.

## Tree-sitter (editors)

The authoritative tree-sitter grammar lives at
[`grammars/qvr/`](https://github.com/quivers-dev/quivers/tree/main/grammars/qvr)
in the quivers repository. The Python package includes its generated source;
platform wheels pair that source with a manifest-verified native parser. The
Pygments lexer, CLI, REPL, and language server load this pair directly and fail
closed if either component is absent or inconsistent. Editable checkouts may
compile a content-addressed development parser after grammar changes.
`panproto-grammars-all` remains a dependency for the eleven transpiler-target
grammars, but it does not define QVR syntax.

Editor integrations use the first-party extensions below or clone the quivers
repository and build against `grammars/qvr/`. The grammar follows standard
tree-sitter conventions; if an editor has a `:TSInstall qvr` or equivalent
command and the grammar is not on its upstream list yet, the manual paths below
also work.

### Neovim

With [nvim-treesitter](https://github.com/nvim-treesitter/nvim-treesitter):

```lua
-- init.lua  (or any config file loaded after nvim-treesitter)
local parsers = require("nvim-treesitter.parsers").get_parser_configs()
parsers.qvr = {
  install_info = {
    url = "https://github.com/quivers-dev/quivers",
    files = { "grammars/qvr/src/parser.c" },
    location = "grammars/qvr",
    branch = "main",
    generate_requires_npm = false,
    requires_generate_from_grammar = false,
  },
  filetype = "qvr",
}
vim.filetype.add({ extension = { qvr = "qvr" } })
```

Then `:TSInstall qvr` from inside Neovim. The repository ships `queries/highlights.scm` at `grammars/qvr/queries/highlights.scm`; nvim-treesitter picks it up automatically once the parser is registered.

### Helix

Helix has native tree-sitter support. Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "qvr"
scope = "source.qvr"
file-types = ["qvr"]
roots = []
comment-token = "#"
indent = { tab-width = 4, unit = "    " }

[[grammar]]
name = "qvr"
source = { git = "https://github.com/quivers-dev/quivers", subpath = "grammars/qvr" }
```

Then:

```bash
hx --grammar fetch
hx --grammar build
```

Place the highlight queries at `~/.config/helix/runtime/queries/qvr/highlights.scm`; copy the file from `grammars/qvr/queries/highlights.scm` in this repository, or symlink to a checkout.

### Emacs (tree-sitter-langs)

With [tree-sitter-langs](https://github.com/emacs-tree-sitter/tree-sitter-langs) and built-in `treesit` (Emacs 29+):

```elisp
(add-to-list 'treesit-language-source-alist
             '(qvr . ("https://github.com/quivers-dev/quivers"
                      "main"  ; branch
                      "grammars/qvr/src")))
(define-derived-mode qvr-mode prog-mode "QVR"
  "Major mode for editing .qvr files."
  (when (treesit-ready-p 'qvr)
    (treesit-parser-create 'qvr)
    (setq-local treesit-font-lock-settings
                (treesit-font-lock-rules
                 :language 'qvr
                 :feature 'comment
                 '((comment) @font-lock-comment-face)
                 :language 'qvr
                 :feature 'keyword
                 '(["morphism" "object" "program"
                    "composition" "category" "contraction" "bundle"
                    "deduction" "atoms" "start" "depth"
                    "lexicon" "let" "in" "sample" "observe"
                    "marginalize" "return" "factor" "score"
                    "index" "family" "constructor" "effect"
                    "instance" "handler" "perform" "handle"
                    "case" "motive" "resumes" "construct"
                    "for" "with" "lacks"
                    "coverage"
                    "forwards" "introduces" "unknown" "none"
                    "total" "partial"
                    "aff" "omega" "true" "false" "unit"
                    "export" "as" "over"
                    "algebra" "semigroupoid" "bilinear_form"
                    "rule"]
                   @font-lock-keyword-face)))
    (treesit-major-mode-setup)))
(add-to-list 'auto-mode-alist '("\\.qvr\\'" . qvr-mode))
```

Run `M-x treesit-install-language-grammar RET qvr RET` once after Emacs starts to fetch the parser.

### Zed

The quivers repository ships a Zed extension at
[`editors/zed-extension-qvr/`](https://github.com/quivers-dev/quivers/tree/main/editors/zed-extension-qvr).
Until the extension is published to Zed's public registry, install it
locally from a checkout:

```bash
git clone https://github.com/quivers-dev/quivers
mkdir -p ~/.config/zed/extensions
ln -s "$(pwd)/quivers/editors/zed-extension-qvr" ~/.config/zed/extensions/qvr
```

Reload extensions (`zed: reload extensions` from the command palette).
The extension's manifest pins the grammar against `grammars/qvr/` in the
same repository, so a single checkout drives both the grammar source
and the editor packaging.

### VS Code / Cursor

The repository ships a first-party VS Code extension at
[`editors/vscode-qvr/`](https://github.com/quivers-dev/quivers/tree/main/editors/vscode-qvr).
It provides:

- a TextMate grammar for the initial render,
- a [`vscode-languageclient`](https://github.com/microsoft/vscode-languageclient-node)
  bridge to [`qvr-lsp`](../guides/repl-and-lsp.md) for hover,
  go-to-definition, semantic tokens, document symbols, completion,
  formatting, and live diagnostics.

Install the packaged extension into VS Code (or Cursor):

```bash
cd editors/vscode-qvr
npm install
npx tsc -p .
npx @vscode/vsce package --allow-missing-repository --no-yarn
code --install-extension vscode-qvr-*.vsix
# or: cursor --install-extension vscode-qvr-*.vsix
```

The extension auto-discovers `qvr-lsp` in (1) the `qvr.lsp.path`
setting (with `${workspaceFolder}` expansion), (2)
`<workspace>/.venv/bin/qvr-lsp`, (3) `$VIRTUAL_ENV/bin/qvr-lsp`, or
(4) plain `qvr-lsp` on `$PATH`. Install the LSP extra with
`pip install 'quivers[lsp]'`.

The full configuration options and launch resolution order are documented in
the [Interactive guide](../guides/repl-and-lsp.md#vs-code-cursor).

## GitHub source view (Linguist)

GitHub's syntax highlighting on `.qvr` files in repository source view and gists is driven by [github-linguist](https://github.com/github-linguist/linguist). QVR is not yet upstream; until the language is registered, repositories can override locally with a `.gitattributes` entry pointing at the closest existing language for at least *some* color:

```gitattributes
*.qvr linguist-language=R
*.qvr linguist-detectable=true
```

This repository uses R as the temporary stand-in, which highlights common
declaration names, calls, numbers, strings, and comments. It is not a QVR
grammar: effect rows, categorial slashes, the `~ Family` clause, and axis roles
remain approximate until Linguist has a native QVR entry.

Native GitHub highlighting requires a QVR language entry in
[github-linguist](https://github.com/github-linguist/linguist/blob/main/CONTRIBUTING.md#adding-a-language).
Panproto's separate refresh of its vendored QVR grammar is tracked in
[panproto/panproto#321](https://github.com/panproto/panproto/issues/321); that
vendor update does not register QVR with GitHub Linguist.

## Verifying your setup

A baseline probabilistic-program check is:

<!-- compile: false -->
```qvr
object Item : FinSet 32

program regression : Item -> Item [effects=[Sample, Score]]
    sample sigma <- HalfNormal(scale=1.0)
    sample beta <- Normal(loc=0.0, scale=2.0)
    let mu = beta * x
    observe y : Item <- Normal(loc=mu, scale=sigma)
    return y

export regression
```

When the highlighting is wired up, keywords (`object`, `program`,
`sample`, `let`, `observe`, `return`, `export`), the `effects` option,
the `<-` bind punctuation, family identifiers (`HalfNormal`, `Normal`),
and comments carry distinct colors.

Use this second block to verify the indexed-effect tokens:

<!-- compile: qiec -->
```qvr
index Availability = Observed | Missing

family Measurement(s : Availability) : Type
    constructor Present : Real -> Measurement(Observed)
    constructor Absent : Measurement(Missing)

effect Robust
    shrink : Real -> Real

instance robust : Robust

handler half_weight for Robust : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    shrink(x : Real) resumes 1 =>
        resume(0.5 * x)

define robustify(x : Real) : Real !{} =
    handle robust with half_weight in
        let adjusted <- perform robust.shrink(x)
        return adjusted
```

The additional keywords (`index`, `family`, `constructor`, `effect`,
`instance`, `handler`, `resumes`, `define`, `handle`, `perform`) and handler
option keys should receive their semantic classes once the LSP attaches. The
[QVR language reference](../reference/qvr/index.md) contains tested blocks for
the remaining computation and program forms.

If a token is unhighlighted (rendered as default-foreground text), the corresponding rule in the editor's highlight query is the place to look; the canonical reference is [`grammars/qvr/queries/highlights.scm`](https://github.com/quivers-dev/quivers/blob/main/grammars/qvr/queries/highlights.scm).
