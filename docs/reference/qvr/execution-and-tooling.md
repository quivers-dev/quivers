# Execution and tooling

Every executable QVR declaration is run through one interface. The command line, the
[REPL](../../guides/repl-and-lsp.md#the-repl) with its terminal interface (TUI), and the Python
API all list the checked entry points, validate static and value arguments, attach runtime
providers, run the reference machine, and return stable `qiec-run-*` diagnostics. The *reference
machine* is the library's interpreter for QIEC, the Quivers Indexed Effect Core
([reference machine](../../developer/qiec.md#reference-evaluator)).

## Check before you run

`qvr check` parses, compiles, and validates one or more files:

```bash
qvr check model.qvr
qvr check model.qvr other.qvr
qvr check --target stan model.qvr
```

Without `--target`, it reports parser, constraint, compiler, and QIEC diagnostics. With a target,
it also analyzes the reachable QIEC computation graph and reports each feature the target lacks as
`qiec:capability:*` or as a more specific QIEC code. `qvr transpile --to TARGET` then runs the
renderer and writes target source, and it can report restrictions particular to that renderer.
Running the generated source tests it with the target's own toolchain. `qvr check --json` emits the
diagnostics as JSON. Posterior inference is run through the
[inference drivers](../../guides/inference-foundations.md), not through `qvr run`.

The [language server](../../guides/repl-and-lsp.md#the-language-server) and the REPL take the same
option:

```bash
qvr lsp --target numpyro
qvr repl --target numpyro model.qvr
```

The VS Code and Cursor extension exposes the target as the setting `qvr.transpileTarget`. A client
may change it with `workspace/didChangeConfiguration`, and the server then rechecks every open
document without reparsing it.

## List entry points

An *entry point* is a named computation or program that a module exposes to `qvr run`,
[`Program.run`][quivers.program.Program.run], and the transpilers.

```bash
qvr run model.qvr --list
```

The listing marks each entry as a `computation` (declared with `define`) or a `program`, and shows
its value parameters, static binders, residual effects, result type, and, for a program, its sample
sites. Omitting both the entry name and `--list` prints the same listing. In Python:

```python
from quivers.dsl import load

compiled = load("docs/examples/source/bayesian_regression.qvr")
for entry in compiled.entry_points():
    print(entry.name, entry.kind, entry.result)
```

## Run an entry

### Computation arguments

Value arguments are JSON literals in declaration order:

```bash
qvr run functions.qvr robustify 8.0
qvr run functions.qvr pair 1 '[2.0, 3.0]'
```

Static binders are closed by name:

```bash
qvr run vectors.qvr total 1.0 2.0 3.0 --static n=3
```

A static value is a closed type, index, effect application, or, for a `Nat` binder, an integer. No
source variable may remain free in it.

### Program data and sites

A program's parameters are given as positional JSON arguments or as named `--data` bindings. Each
`--site NAME=JSON` conditions one sample site on a value, and every site not named is drawn:

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

Each invocation owns its data, observations, sites, generator state, parameter store, score
accumulator, and [execution trace](../../guides/inference-foundations.md#trace-and-sample-sites),
the record of every site's value and log density. No handler or mutable provider state is shared
with the next invocation. [`Program.run`][quivers.program.Program.run] takes the same names:

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

It returns an [`EntryRun`][quivers.qiec.entries.EntryRun]. `EntryRun.result.trace` holds the
events that `qvr run --trace` prints, and `EntryRun.to_data()` returns the payload that
`qvr run --json` prints.

## Execution controls

| Option | Meaning |
| --- | --- |
| `--seed N` | seed the reference generator that draws the sites not given with `--site` |
| `--fuel STEPS` | stop after `STEPS` evaluation steps with `qiec-run-fuel` |
| `--trace` | print the execution trace events to stderr (without `--json`) |
| `--json` | print, on success, the value, result type, runtime label, trace, entry kind, and for a program the log joint; on failure, the diagnostics |
| `--runtime FILE.json` | use the provider configuration in `FILE.json` |

*Fuel* is a bound on the number of evaluation steps. Without `--fuel` a run has no bound, so a
recursive `define` or a generated [deduction](../../guides/deduction.md#the-framework) that does
not terminate runs until it is stopped.

## Runtime providers

A *runtime provider* is an object that binds handlers and type validators, held only in the
running Python process, into a run ([`RuntimeProvider`][quivers.qiec.execution.RuntimeProvider]).
It gives the behavior for a typed request, such as drawing from a distribution or calling an
attached [encoder](../../guides/structural-compression-signatures-and-encoders.md#encoder-blocks).
A serialized module records only what it needs and under which stable identifier. It never holds a
Python callable or a mutable object, so a checked module can be inspected and exchanged without
executable content.

A provider configuration names providers and their options:

```json
{
  "providers": [
    {"name": "core", "options": {}}
  ]
}
```

The `core` provider gives the reference machine, the distribution families, the standard
[handlers](effects-and-handlers.md#handler-declarations), search, weights, and parameter lookup. An
installed package may register another provider under the `quivers.qiec_runtime` entry-point group.
When a structural program is invoked through `Program.run`, the core provider is composed with the
attachments for its compiled encoders and decoders.

A configuration is data and cannot contain a Python callable. Before evaluation begins, the module
and the provider are matched by their stable attachment, handler, instance, and operation
identifiers.

## REPL and TUI

`qvr repl model.qvr` starts an interactive session, in the TUI unless `--plain` is given. The
commands that run entry points are:

| Command | Purpose |
| --- | --- |
| `:run` | list entry points |
| `:run NAME JSON ...` | invoke a computation |
| `:run NAME --data x=JSON --site z=JSON` | invoke a program |
| `:runtime` | show the active provider configuration |
| `:runtime core` | select a registered provider |
| `:runtime FILE.json` | load a provider configuration |
| `:detach` | detach every runtime provider |
| `:effects NAME` | show the declared and inferred effects |
| `:dump NAME --json` | print the declaration's syntax tree as JSON |

The TUI status bar shows the active runtime and the last result. Every command and key binding is
listed in [Interactive tools: REPL, kernel, and language server](../../guides/repl-and-lsp.md).

## Language-server features

The QVR language server (LSP) uses the compiler's typed source model. It provides:

- parser, checker, QIEC, and target-capability diagnostics, each with a source range;
- hovers for declarations, calls, operations, instances, and inferred rows;
- definitions, references, and renaming for symbols the QVR source declares;
- nested document symbols for handler clauses, locals, calls, and program scopes;
- completion of operations, names, keywords, built-ins, and paths by context;
- semantic tokens classified by the rules of the Pygments lexer; and
- formatting through the QVR emitter.

The VS Code, Cursor, and Zed extensions start `qvr-lsp`. Neovim and other clients launch it over
stdio. Installation is covered in [Editor support](../../getting-started/highlighting.md), and the
supported methods are listed in the [LSP capability table](../../guides/repl-and-lsp.md#capabilities).

## Transpilation

`qvr transpile --list-targets` prints the targets, and `qvr transpile --help` prints the options:
`--to TARGET` for one target, `--to-all` for every target, `-o FILE` for the output of one target,
and `-d DIR` for the files of `--to-all`. Check the target first, then transpile:

```bash
qvr check --target pyro model.qvr
qvr transpile --to pyro model.qvr
```

Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church render every reachable QIEC
computation as a function in the target language, called through one convention for all eight
([target calling convention](../../developer/qiec.md#target-runtime-abi)). Stan emits a closed, monomorphic scalar
computation with no effects as a user-defined function. BUGS and JAGS accept a measured subset of
programs and refuse calls they cannot express (`call:graph:*`). The generated
[transpilation support matrix](../../transpile-support.md) reports, feature by feature and example
by example, what each target accepts.

QVR counts indices from zero, and so does every value a program holds that a subscript reads: a
`factor` binder, a marginalized latent, an index covariate such as `group_idx`, and a class code (a
`Categorical`, `OrderedLogistic`, or `OrderedProbit` outcome, or a drawn class). A caller passes
every target the same zero-based data, and a drawn class is returned zero-based.

| Target | Origin of its arrays | What the generated program does |
| --- | --- | --- |
| Pyro, NumPyro, PyMC, Edward2, WebPPL, Church | zero | reads every value as QVR does |
| Stan, Turing, Gen | one | keeps each value zero-based and adds one where it subscripts an array or where a categorical family scores it, so `y[2 - t]` reads the entry it reads in QVR |
| JAGS | one | as Stan, and binds an observed class code, shifted by one, in its `data` block |
| BUGS | one | as Stan, but refuses an observed class code with `observe:class-code:<name>`, since BUGS has no `data` block to shift it in and no statement that adds an arbitrary log density to score it |

Stan has no discrete parameters and refuses a drawn class with `sample:discrete-latent:<name>`.
Marginalize the class instead ([exact marginalization](probabilistic-programs.md#exact-marginalization)).

## Stable diagnostic families

| Prefix | Raised by |
| --- | --- |
| `parse` | the source grammar |
| `compile` and the codes of particular constraints | surface resolution and elaboration |
| `qiec-*` | the QIEC checker: kinds, types, rows, coverage, and module validation |
| `qiec-run-*` | entry selection, arguments, providers, evaluation, and results |
| `qiec:capability:*` | a target that cannot represent a reachable checked feature |
| `call:graph:*` | a target that cannot keep a reachable call |

Tests and scripts should match the stable code. The message explains the particular source form,
and its wording may change.

## Grammar ownership and Panproto

quivers holds the grammar that defines QVR. [tree-sitter](https://tree-sitter.github.io/) is the
parser generator in which the grammar is written, and
[panproto](https://github.com/panproto/panproto) is the schema and migration toolkit that versions
it.

- `grammars/qvr/grammar.js` defines the tree-sitter grammar.
- The generated parser sources and the compiled parser libraries ship in the quivers wheel.
- A manifest checks that the grammar, the generated sources, the queries, and the compiled parser
  agree.
- quivers installs this parser into panproto's parser registry before it parses any QVR source.

The package `panproto-grammars-all` gives the grammars of the other languages that the eleven
transpile targets emit. Its copy of the QVR grammar may lag behind the one in quivers, and this does
not change how quivers parses `.qvr` files. panproto gives the generic parse tree and composes the
chain of grammar migrations that `qvr migrate` applies.

For editor highlighting, build against `grammars/qvr/` in the quivers repository or install one of
the quivers editor extensions. An editor pointed at panproto's grammar package may load an older
QVR grammar that lacks some tokens.

### Format identifiers

Three identifiers name three formats, and each can change without the others:

| Identifier | Names |
| --- | --- |
| `qvr-source/v0.20` | the source grammar and its elaboration to QIEC |
| `qiec-core/v1alpha1` | the typed core that checked modules are written in |
| `qiec-json/v1` | the deterministic JSON serialization of a checked module |

New syntax that elaborates to existing core terms changes the first identifier and not the second,
and a new serialization changes the third without changing how `.qvr` files parse. Code that
exchanges checked modules tests the second and third, and editors and migration tools test the
first.

## Before a release

The [release workflow tutorial](../../tutorials/qvr/13-release-workflow.md) checks every source and
promised target, runs each public entry point, and inspects a trace before a module is published.
