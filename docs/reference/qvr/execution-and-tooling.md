# Execution and tooling

Every executable QVR declaration uses the same entry-point interface.
The command line, [REPL](../../guides/repl-and-lsp.md#the-repl), TUI, and Python API all list the checked entry, validate
static and value arguments, attach runtime providers, run the evaluator, and
return stable `qiec-run-*` diagnostics through this boundary.

## Check before you run

`check` validates a model's structure; `run` executes it on supplied
inputs; inference fits a posterior; `transpile` writes a representation
for another backend. Run `check` while editing to catch incompatible
types, then `run` with representative inputs to inspect returned values.
For posterior inference, use the [inference drivers](../../guides/inference-foundations.md)
and their diagnostics.


Parse, compile, and validate one or more files with:

```bash
qvr check model.qvr
qvr check model.qvr other.qvr
qvr check --target stan model.qvr
```

Without `--target`, checking reports parser, constraint, compiler, and QIEC
diagnostics. With a target, it also analyzes the reachable QIEC computation
graph and reports unsupported core capabilities as `qiec:capability:*` or a
more specific QIEC code. Run `qvr transpile --target TARGET` next to
exercise the renderer and produce target source; renderer-specific
restrictions can appear at that stage. Executing the generated source
then tests it with the target toolchain.

Use the same target in the [language server](../../guides/repl-and-lsp.md#the-language-server):

```bash
qvr lsp --target numpyro
```

The VS Code/Cursor extension exposes it as `qvr.transpileTarget`. LSP clients
may update `transpileTarget` with `workspace/didChangeConfiguration`; the
server rechecks open documents without reparsing them.

## List entry points

```bash
qvr run model.qvr --list
```

The listing distinguishes `computation` entries declared with `define` from
`program` entries and shows value parameters, static binders, residual
effects, result type, and program sites. Omitting both the entry name and
`--list` produces the same listing.

The Python equivalent is:

```python
from quivers.dsl import load

compiled = load("docs/examples/source/bayesian_regression.qvr")
for entry in compiled.entry_points():
    print(entry.name, entry.kind, entry.result)
```

## Run an entry

### Computation arguments

Pass runtime arguments as JSON literals in declaration order:

```bash
qvr run functions.qvr robustify 8.0
qvr run functions.qvr pair 1 '[2.0, 3.0]'
```

Close static binders by name:

```bash
qvr run vectors.qvr total 1.0 2.0 3.0 --static n=3
```

The static value must be a closed type, index, effect application, or integer
for a `Nat` binder. No source variable may remain free.

### Program data and sites

Program parameters may be positional JSON arguments or named `--data`
bindings. `--site` replays one sample site; every unmentioned site is drawn:

```bash
qvr run model.qvr regression \
  --data x='[0.0, 1.0, 2.0]' \
  --data y='[0.1, 1.2, 1.9]' \
  --site slope=0.9 \
  --site intercept=0.1 \
  --site sigma=0.3 \
  --seed 7 \
  --json
```

Each invocation owns its data, observations, sites, generator state, parameter
store, score accumulator, and [trace](../../guides/inference-foundations.md#trace-and-sample-sites). No handler or mutable provider state is
shared implicitly with the next invocation.

The Python form uses the same names:

<!-- python: skip -->
```python
run = compiled.run(
    "regression",
    data={"x": [0.0, 1.0, 2.0], "y": [0.1, 1.2, 1.9]},
    sites={"slope": 0.9, "intercept": 0.1, "sigma": 0.3},
    seed=7,
)
print(run.value, run.log_joint)
```

`EntryRun.result.trace` contains the same stable events emitted by
`qvr run --trace`; `EntryRun.to_data()` returns the JSON-compatible payload.

## Execution controls

| Option | Meaning |
| --- | --- |
| `--seed N` | seed the reference generator for unspecified program sites |
| `--fuel STEPS` | stop a recursive computation after a bounded number of evaluator steps |
| `--trace` | write stable trace events to stderr in plain mode |
| `--json` | emit result, type, runtime, diagnostics, trace, entry kind, and program log joint as JSON |
| `--runtime FILE.json` | select an explicit provider configuration |

Fuel is especially useful for a recursive `define` or generated [deduction](../../guides/deduction.md#the-framework).
The default is appropriate for ordinary terminating entries; a failed bound
uses `qiec-run-fuel`.

## Runtime providers

A provider supplies executable behavior for a typed request, such as
drawing from a distribution or calling an attached [encoder](../../guides/structural-compression-signatures-and-encoders.md#encoder-blocks). The
serialized module records what is needed and under which stable
identifier; the current process supplies the implementation. This
separation permits inspection and exchange of typed modules without
embedding arbitrary host-language objects in their serialized form.


The stable core contains identifiers and typed attachment descriptors, never
host callables or mutable objects. A runtime provider configuration selects process-local implementations. A configuration has this shape:

```json
{
  "providers": [
    {"name": "core", "options": {}}
  ]
}
```

`core` supplies the reference evaluator, distributions, standard [handlers](effects-and-handlers.md#handler-declarations),
search, weights, and parameter lookup. Installed packages may expose another
provider through the `quivers.qiec_runtime` entry-point group. Structural
programs automatically compose the core provider with attachments for their
compiled encoders and decoders when invoked through `Program.run`.

A configuration is non-executable data. It may name a provider and options,
but cannot contain a Python callable. The module and provider are matched by
stable attachment, handler, instance, and operation identifiers before
evaluation begins.

## REPL and TUI

Start an interactive session with `qvr repl model.qvr`. The commands specific
to the entry boundary are:

| Command | Purpose |
| --- | --- |
| `:run` | list entries |
| `:run NAME JSON ...` | invoke a computation |
| `:run NAME --data x=JSON --site z=JSON` | invoke a program |
| `:runtime` | show the active provider configuration |
| `:runtime core` | select a registered provider |
| `:runtime FILE.json` | load a provider configuration |
| `:detach` | detach all runtime providers |
| `:effects NAME` | inspect the inferred row and call contributions |
| `:dump NAME --json` | inspect the checked declaration |

The Textual status bar shows the active runtime and last result. The complete
command and key-binding reference is in [Interactive tools: REPL, kernel, and
language server](../../guides/repl-and-lsp.md).

## Language-server features

The QVR LSP uses the same typed source model as the compiler. It provides:

- source-ranged parser, checker, QIEC, and target-capability diagnostics;
- hovers for declarations, calls, operations, instances, and inferred rows;
- definitions, references, and safe renaming for QVR-owned symbols;
- nested document symbols for handler clauses, locals, calls, and program
  scopes;
- context-sensitive completion for operations, names, keywords, built-ins,
  and paths;
- semantic tokens classified by the same rules as the Pygments lexer; and
- canonical formatting through the QVR emitter.

The first-party VS Code/Cursor and Zed extensions start `qvr-lsp`. Neovim and
other clients can launch it over stdio. See [Editor support](../../getting-started/highlighting.md)
for installation and [the LSP capability table](../../guides/repl-and-lsp.md#capabilities)
for method-level details.

## Transpilation

List the exact CLI surface with `qvr transpile --help`, then emit a target with
the target and output options shown there. Check the target first, then run
the transpiler itself:

```bash
qvr check --target pyro model.qvr
qvr transpile --to pyro model.qvr
```

Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church emit reachable
QIEC computations through the common runtime ABI. Stan emits the closed,
monomorphic, effect-free scalar fragment as user-defined functions. BUGS and
JAGS accept their measured program subset and refuse unsupported call graphs.
The generated [transpilation support matrix](../../transpile-support.md) is the
authoritative per-feature and per-example report.

QVR indices count from zero, and so does every value the program holds that a
subscript reads: a `factor` binder, a marginalized latent, an index covariate
such as `group_idx`, and a class code (a Categorical, OrderedLogistic, or
OrderedProbit outcome, or a drawn class). Stan, BUGS, JAGS, Turing, and Gen
count from one; their programs keep every value in QVR's origin and read it one
higher only where it subscripts an array or where the target's categorical
family scores it, so a subscript such as `y[2 - t]` reads the same entry it
reads in QVR, and a caller passes every target the same zero-based data. A drawn
class is returned zero-based as well. The one exception is BUGS, whose language
has neither a data block to shift an observed class code nor a free log-density
statement to score it otherwise: it refuses an observed class code with
`observe:class-code:<name>` (JAGS binds the shifted code in its `data` block).
Stan has no discrete parameters and refuses a drawn class with
`sample:discrete-latent:<name>`; marginalize it instead.

## Stable diagnostic families

| Prefix | Boundary |
| --- | --- |
| `parse` | source grammar |
| `compile` and constraint-specific codes | surface resolution and elaboration |
| `qiec-*` | kernel kinding, typing, rows, coverage, and module validation |
| `qiec-run-*` | entry selection, arguments, providers, evaluation, and results |
| `qiec:capability:*` | target cannot represent a reachable checked feature |
| `call:graph:*` | target cannot preserve a reachable call |

Prefer matching the stable code in tests and automation. Human-readable
messages explain the particular source form and may become more specific.

## Grammar ownership and Panproto

Quivers is the source of truth for the current QVR grammar:

- `grammars/qvr/grammar.js` defines the tree-sitter grammar;
- generated parser sources and wheel-native libraries ship with Quivers;
- a manifest verifies that the grammar, generated source, queries, and native
  parser agree; and
- Quivers installs this parser into Panproto's registry before ordinary QVR
  parsing.

`panproto-grammars-all` supplies the other language grammars used by the eleven
transpile targets. Its vendored QVR grammar may lag the Quivers release without
changing how Quivers parses `.qvr` files. Panproto still carries the generic
tree and composes the versioned migration chain. The migration from v0.18 to
the current source protocol is validating and byte-preserving because that
grammar extension was additive.

For editor highlighting, build against the Quivers `grammars/qvr/` directory
or install a first-party extension. Pointing an editor directly at
Panproto's aggregate grammar package may select an older QVR grammar and omit
current tokens.

## Release checklist

Before publishing a module or package:

1. run `qvr check` on every `.qvr` source;
2. run `qvr check --target` for every promised backend;
3. invoke each public entry through `qvr run` or `Program.run` with a small
   fixture;
4. inspect a trace for computations with handlers or random sites;
5. verify editor semantic highlighting with `qvr-lsp` attached; and
6. migrate a copy from the previous source revision when compatibility is part
   of the release claim.
