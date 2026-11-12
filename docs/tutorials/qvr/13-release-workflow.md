# 13. Check, inspect, and transpile a module

Before releasing a QVR module, check that every public entry point runs on the reference machine and that every promised transpile target supports what the module reaches. The example is the authored handler of [chapter 10](10-effects-and-handlers.md), which has an effect, a handler, a resumption, an open row, and one entry point with an empty row.

Each check answers a separate question. A source check says that the module is well formed and well typed. A target check says that each feature the module reaches is one the target supports (the target's *capabilities*). Transpiling produces source in the target language, whose own parser and runtime check it further. Whether the target computes the same scores as the original needs a separate numerical comparison.

## 1. Check the source and one target

```bash
qvr check docs/tutorials/qvr/source/authored-handler.qvr
qvr check --target pyro docs/tutorials/qvr/source/authored-handler.qvr
```

The first command parses the module, resolves its names, elaborates it to QIEC, type-checks the result, checks handler coverage, and validates the module. The second also checks the module against the capabilities of the Pyro target. Repeat the target check for every target named in the release notes or the package metadata. Then run `qvr transpile` for each of them, because the capability check does not exercise every renderer.

## 2. Inspect the public entry points {#2-inspect-the-public-boundary}

```bash
qvr run docs/tutorials/qvr/source/authored-handler.qvr --list
qvr run docs/tutorials/qvr/source/authored-handler.qvr robustify 8.0 --json
```

The listing shows `robust_request` and `robustify`. Only `robustify` runs without outside help, because its total handler removes `robust` from its row. If `robust_request` is meant to be public, the provider that handles `robust` for it is part of the release and should be documented with it.

## 3. Inspect one trace

```bash
qvr run docs/tutorials/qvr/source/authored-handler.qvr robustify 8.0 --trace
```

Confirm that each request reaches the intended handler and resumes the intended number of times. For a probabilistic program, also fix a few sites with `--site` and inspect the reported log joint density.

## 4. Use target diagnostics while editing

Start the language server for one target:

```bash
qvr lsp --target pyro
```

In VS Code or Cursor, set:

```json
{
  "qvr.transpileTarget": "pyro"
}
```

The language server reports `qiec:capability:*` diagnostics again whenever the call graph changes. Hovers show the instantiated signature and inferred row of a call, and the document outline lists handler clauses, local instances, and generated program scopes.

## 5. Transpile after the checks pass

```bash
qvr transpile --to pyro docs/tutorials/qvr/source/authored-handler.qvr \
  -o authored_handler.py
qvr transpile --to-all docs/tutorials/qvr/source/authored-handler.qvr \
  --out-dir generated
```

`--to-all` tries each target and reports the result. A target that cannot represent a construct the module reaches returns a capability diagnostic. The [support matrix](../../transpile-support.md) gives the results on the gallery programs, and running the command on the module itself checks the operations it uses.

## 6. Editor grammar and migrations

Editors should load the tree-sitter grammar shipped in `grammars/qvr/` of the installed release, as [editor support](../../getting-started/highlighting.md) describes, so that their highlighting agrees with `qvr check`. When a release promises compatibility with source written for an earlier grammar, test the migration on a copy, with `OLD` the git tag of the earlier grammar:

```bash
qvr migrate --from OLD --to HEAD old-model.qvr
qvr check old-model.qvr
```

## 7. Automate the checks

A release job should, in order:

1. Check every source file without a target.
2. Check every promised pair of source file and target.
3. Run each public entry point on a small fixture.
4. Compare a stable result or a property of the trace.
5. Transpile to each target.
6. Run the target's own parser or compiler when its toolchain is installed.

In this order each failure is reported at the earliest step that can detect it. A source or QIEC error appears before any transpiling, and a capability refusal before any target toolchain starts, so a syntax error from a target's toolchain cannot be confused with either.

The [execution reference](../../reference/qvr/execution-and-tooling.md) documents the command line, providers, language server, diagnostics, and grammar files.
