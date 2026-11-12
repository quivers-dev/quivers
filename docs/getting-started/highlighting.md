# Editor support and syntax highlighting

Each tool that reads `.qvr` files highlights them with one of these engines:

| Consumer | Engine | Comes from |
|---|---|---|
| Python tools: mkdocs, Sphinx, Jupyter, IPython, `pygmentize` | [Pygments](https://pygments.org/) | bundled with the `quivers` PyPI package |
| Editors: Neovim, Helix, Emacs, Zed | [tree-sitter](https://tree-sitter.github.io/) | the grammar and editor files in the quivers repository |
| VS Code and Cursor | TextMate grammar for the first render, then semantic tokens from the QVR language server | the `vscode-qvr` extension in the repository |
| GitHub.com source view, GitHub gists | [Linguist](https://github.com/github-linguist/linguist) | no QVR entry ([below](#github-source-view-linguist)) |

Highlighting colors tokens and does not check a program. The language server
reports type errors in the editor, and
[`qvr check`](../tutorials/qvr/13-release-workflow.md#1-check-the-source-and-one-target)
reports them from a terminal.

## Pygments (Python doc and notebook tools)

quivers registers a Pygments lexer as an entry point. After

```bash
pip install quivers
```

every tool built on Pygments highlights `.qvr` files with no further configuration: mkdocs, Sphinx, `pygmentize`, the IPython and Jupyter cell magic `%%highlight qvr`, and any other tool that asks Pygments for a lexer by file name or by `name="qvr"`.

To confirm the lexer is visible:

```bash
python -c "from pygments.lexers import get_lexer_by_name; print(get_lexer_by_name('qvr').name)"
# QVR
```

To highlight a file to a terminal:

```bash
pygmentize -l qvr -O style=monokai docs/examples/source/bayesian_regression.qvr
```

The lexer is [`quivers.dsl.pygments_lexer`](../api/dsl/pygments_lexer.md). It tokenizes with the QVR tree-sitter parser.

## Tree-sitter (editors)

The QVR tree-sitter grammar is in
[`grammars/qvr/`](https://github.com/quivers-dev/quivers/tree/main/grammars/qvr)
in the quivers repository, and the package includes its compiled parser
([Installation](installation.md#from-source) says which parser an installation
loads). Editors build the parser from that directory, either through the
extensions in the repository or through the configurations below.

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

Then run `:TSInstall qvr` inside Neovim. nvim-treesitter finds the highlight queries at `grammars/qvr/queries/highlights.scm` once the parser is registered.

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

Place the highlight queries at `~/.config/helix/runtime/queries/qvr/highlights.scm`, as a copy of `grammars/qvr/queries/highlights.scm` or a symbolic link to it in a checkout.

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

The quivers repository contains a Zed extension at
[`editors/zed-extension-qvr/`](https://github.com/quivers-dev/quivers/tree/main/editors/zed-extension-qvr).
Install it from a checkout:

```bash
git clone https://github.com/quivers-dev/quivers
mkdir -p ~/.config/zed/extensions
ln -s "$(pwd)/quivers/editors/zed-extension-qvr" ~/.config/zed/extensions/qvr
```

Reload extensions (`zed: reload extensions` from the command palette).
The extension's manifest points at `grammars/qvr/` in the same repository, so
one checkout provides both the grammar and the extension.

### VS Code / Cursor

The repository contains a VS Code extension at
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

The extension looks for `qvr-lsp` in (1) the `qvr.lsp.path`
setting (with `${workspaceFolder}` expansion), (2)
`<workspace>/.venv/bin/qvr-lsp`, (3) `$VIRTUAL_ENV/bin/qvr-lsp`, or
(4) plain `qvr-lsp` on `$PATH`. Install the LSP extra with
`pip install 'quivers[lsp]'`.

[Interactive tools](../guides/repl-and-lsp.md#vs-code-cursor) lists the
extension's settings.

## GitHub source view (Linguist)

GitHub highlights repository source and gists with
[github-linguist](https://github.com/github-linguist/linguist), which has no
QVR entry. A `.gitattributes` entry can map `.qvr` files to the closest
language Linguist knows:

```gitattributes
*.qvr linguist-language=R
*.qvr linguist-detectable=true
```

R colors declaration names, calls, numbers, strings, and comments. Effect rows,
categorial slashes, the `~ Family` clause, and axis options remain uncolored or
miscolored. Native highlighting requires a QVR entry in Linguist
([adding a language](https://github.com/github-linguist/linguist/blob/main/CONTRIBUTING.md#adding-a-language)).

## Verifying your setup

Open a file containing this program:

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

With highlighting in place, keywords (`object`, `program`,
`sample`, `let`, `observe`, `return`, `export`), the `effects` option,
the `<-` bind punctuation, family identifiers (`HalfNormal`, `Normal`),
and comments carry distinct colors.

This second program uses indexed families, effects, and handlers:

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

A token rendered in the default foreground color has no matching rule in the editor's highlight query. The reference query is [`grammars/qvr/queries/highlights.scm`](https://github.com/quivers-dev/quivers/blob/main/grammars/qvr/queries/highlights.scm).
