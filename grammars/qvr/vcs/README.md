# QVR grammar VCS

This directory holds a panproto VCS (`.panproto/`) that tracks the
QVR grammar's evolution as a chain of panproto schema objects. It
drives batch migration of `.qvr` files in the
repository (examples, doc code blocks, regression fixtures) whenever
the grammar or AST shape changes.

## Layout

```
grammars/qvr/vcs/
  .panproto/        # panproto repository (content-addressed store)
  parsers/          # immutable generated parser-source snapshots
    v0.15.0/source/src/
      parser.c
      scanner.c
      grammar.json
      node-types.json
    HEAD/source/src/
      # current parser sources and metadata
  build_schemas.py  # append/tag distinct authored grammar revisions
  build_parsers.py  # materialize parser snapshots from git tags
  README.md         # this file
```

## Release workflow

Grammar work is prepared under the explicit `HEAD` revision. A release then
freezes that revision under its git tag. The complete workflow is:

1. While developing, append the working-tree schema with
   `python grammars/qvr/vcs/build_schemas.py --revision HEAD`. This preserves
   immutable tagged history. Do not use `--reset` or rewrite historical
   `SCHEMA_COMMITS` entries. Regenerate the current parser with `python
   grammars/qvr/vcs/build_parsers.py --revision HEAD --force`. Update the
   migration manifest and test the previous release to `HEAD` path.
2. Commit the tested feature changes, then make a separate version and
   changelog preparation commit and open the pull request. Tag the git release
   only after merging and successful main-branch CI, as part of the authorized
   release. Development preparation creates no release tag.
3. Run `python grammars/qvr/vcs/build_schemas.py`. The script tags an existing
   untagged content commit when the release freezes the prior `HEAD`, preserves
   existing tagged schema identities, appends
   only genuinely new authored grammars, and is idempotent. History is keyed
   by `grammar.js`; regenerated `grammar.json` serialization alone does not
   constitute a language revision.
4. Run `python grammars/qvr/vcs/build_parsers.py --revision <tag>`.
   The generated C source, grammar metadata, and node types form the
   platform-neutral immutable snapshot. Platform wheels carry a matching
   native library and a manifest bound to these source bytes. An installed
   package requires that verified trio and fails closed if any part is absent
   or inconsistent. A source checkout may compile the snapshot into its local
   cache for development.
5. Run the batch migration over `.qvr` sources:
   `qvr migrate --from <prior-tag> --to HEAD <paths>`. The CLI
   composes the registered one-hop panproto migrations. The chain
   may name multiple releases separately even when they share an authored
   grammar. An additive hop may be byte-preserving, but it is still validated
   and is not declared as an identity edge. The chain always ends in an
   explicit `HEAD`.

The tagged snapshot produced in steps 3–4 is normally committed as preparation
for the next Quivers release. The release being tagged already contains the
tested `HEAD` snapshot, so this bookkeeping does not make runtime parsing
depend on a post-tag commit.

## Why a VCS instead of hand-editing files

A grammar/AST change typically touches every `.qvr` source in the
tree. Hand-editing each one re-introduces drift, misses fenced doc
blocks, and forks the migration logic across files. A single
panproto migration centralizes the schema delta and reduces drift.
Each non-identity migration parses and validates its complete output
before the CLI performs an atomic sibling-file replacement. With
`--output`, directory inputs retain their relative subtree rather than
flattening every `.qvr` file into one directory.

## Out of scope

This VCS migrates `.qvr` source files only. Grammar-derived assets such as the
generated parser and tree-sitter highlight queries are regenerated from the
authored grammar tooling. Semantic mappings in the TUI, CLI, LSP server,
Pygments lexer, and TextMate grammar still require deliberate updates. The
migration engine does not modify those surfaces.
