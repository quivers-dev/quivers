# Check, inspect, and transpile a module

This tutorial turns the QVR tooling into a short pre-release workflow. The
goal is to check a deployment claim: every public entry is exercised
on the reference machine, and every advertised target is checked against the
reachable computation graph before artifacts are emitted.

We will use the authored handler from [tutorial 10](10-effects-and-handlers.md) because it
contains a custom effect, a handler, a resumption, an open row, and one pure
public wrapper.

A successful source check says the declared computation is well formed.
A successful target check says its required features pass that target's
capability checks. Emission then produces source in the target language, whose
own parser and runtime provide further checks. Matching [numerical scores](../../developer/contributing.md#test-tiers)
requires a separate comparison with the original model. Each step answers a
different question, which is why the workflow retains all of them.

## 1. Check the source and one target

```bash
qvr check docs/tutorials/qvr/source/authored-handler.qvr
qvr check --target pyro docs/tutorials/qvr/source/authored-handler.qvr
```

The first command establishes that parsing, surface resolution, elaboration,
QIEC typing, handler coverage, and module validation succeed. The second adds
the Pyro QIEC-capability pass. Repeat the target check for every backend named
in your release notes or package metadata. Then run `qvr transpile` for each
promised target: the capability pass does not exercise every
surface-to-target renderer path.

## 2. Inspect the public boundary

```bash
qvr run docs/tutorials/qvr/source/authored-handler.qvr --list
qvr run docs/tutorials/qvr/source/authored-handler.qvr robustify 8.0 --json
```

The listing exposes `robust_request` and `robustify`. Only `robustify` is
self-contained: its total handler removes `robust` from the row. If
`robust_request` is intentionally public, its runtime provider is part of the
deployment contract and should be documented beside it.

## 3. Inspect one trace

```bash
qvr run docs/tutorials/qvr/source/authored-handler.qvr robustify 8.0 --trace
```

Confirm that the expected request reaches the expected handler and resumes the
expected number of times. For a probabilistic program, also replay a small set
of sites with `--site` and inspect the reported log joint.

## 4. Use target diagnostics while editing

Launch the language server with the intended backend:

```bash
qvr lsp --target pyro
```

In VS Code or Cursor, set:

```json
{
  "qvr.transpileTarget": "pyro"
}
```

The LSP republishes `qiec:capability:*` diagnostics as the call graph changes.
Hovers show a call's instantiated signature and inferred row, while document
symbols expose nested handler clauses, local instances, and generated program
scopes.

## 5. Emit only after the check passes

```bash
qvr transpile --to pyro docs/tutorials/qvr/source/authored-handler.qvr \
  -o authored_handler.py
qvr transpile --to-all docs/tutorials/qvr/source/authored-handler.qvr \
  --out-dir generated
```

`--to-all` tries each target and reports its result. A target that cannot
represent a reachable construct returns a capability diagnostic. Use the
[support matrix](../../transpile-support.md) to inspect release-corpus results,
then run the command on your own module to check the operations it uses.

## 6. Verify editor grammar provenance

The current QVR grammar ships from this repository, including the tree-sitter
queries and first-party editor extensions. For an editor built manually, point
the parser at `grammars/qvr/`, not at a possibly older aggregate Panproto
grammar package. `qvr-lsp` then supplies semantic tokens from the same typed
source model used by `qvr check`.

Panproto remains part of the migration and generic-tree path. When a release
promises compatibility with an earlier source version, test the corresponding
migration on a copy. For instance:

```bash
qvr migrate --from 0.18.0 --to HEAD old-model.qvr
qvr check old-model.qvr
```

The v0.18-to-current hop is validating and byte-preserving because that
grammar extension was additive.

## 7. Automate the checks

A release job should, in order:

1. check every source without a target;
2. check every promised source-target pair;
3. invoke each public entry on a small fixture;
4. compare a stable result or trace invariant;
5. emit target artifacts; and
6. run the target's own parser or compiler when its toolchain is available.

This ordering attributes failures to the narrowest boundary. A source or QIEC
error appears before target emission; a capability refusal appears before a
host toolchain starts; and a host syntax failure is not confused with either.

The [execution reference](../../reference/qvr/execution-and-tooling.md) gives
the complete CLI, provider, LSP, diagnostic, and grammar-ownership contract.
