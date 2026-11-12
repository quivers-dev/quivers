# Contributing to Quivers

A change to quivers usually touches one of three things: what source programs can express, what
those programs compute, or how their results reach the user. Syntax work is tested by the parser
and compiler tests. Distribution and inference work is tested by comparing numerical scores and
[posterior behaviour](../tutorials/qvr/08-diagnostics-and-comparison.md#convergence-diagnostics).
A transpiler is tested on both its emitted syntax and its agreement with the source model's
[log joint density](../tutorials/python/05-variational-inference.md#tracing-the-model) at the
same points, since valid target syntax can still encode the wrong distribution. The
[architecture overview](../getting-started/architecture.md) maps each kind of change to the
packages it touches.

## Development Environment Setup

### Prerequisites

- Python 3.14 or later
- pip or conda
- git
- A C toolchain (`cc`, Clang, GCC, or MSVC) for editable parser builds

### Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/quivers-dev/quivers
cd quivers
pip install -e ".[dev]"
```

This installs the package in editable mode with its development dependencies.

### Running Tests

Run the full test suite:

```bash
python -m pytest tests/ -x
```

The `-x` flag stops at the first failure. Other common options:

```bash
python -m pytest tests/ -v                  # verbose
python -m pytest tests/ -k test_name        # run specific tests
python -m pytest tests/path/to/test_file.py # run specific file
```

### Test tiers

Two independent markers govern runtime. Tests marked `probe` drive a real
probabilistic programming language inside a Docker container to compare its
log density against the reference. Recovery and benchmark tests are marked
`slow` and are deselected by the default `pyproject.toml` configuration.

Use these commands for the common tiers:

```bash
python -m pytest tests/ -m "not slow and not probe"  # fast, no Docker
python -m pytest tests/ -m "not slow and probe"      # container probes
python -m pytest tests/ -m slow                       # recovery + benchmarks
python -m pytest tests/ --runslow                     # the complete suite
```

The `probe` marker is applied by module, from the registry in `tests/transpile/conftest.py`. A
module that calls `run_probe` without appearing in that registry fails
`test_probe_marker_registry.py`, so a new probe module cannot land in the fast tier by omission.

Probe cells need Docker. The session fixture starts the daemon and builds any missing images, so
the requested checks run rather than being skipped. Setting `QUIVERS_SKIP_DOCKER=1` turns this
off, which suits a run of `-m "not slow and not probe"` only.

### Running tests in parallel

Both tiers can run in parallel. Each probe call gets its own scratch directory, containers are
anonymous, and a file lock serializes the image build across workers.

```bash
python -m pytest tests/ -m "not slow and not probe" -n auto --dist loadfile
python -m pytest tests/ -m "not slow and probe" -n auto
```

Use `--dist loadfile` for the fast tier. Each of its tests takes far less time than handing it
to a worker costs, so whole files are distributed rather than single tests.

To divide the probe tier across machines, `pytest-split` takes a share:

```bash
python -m pytest tests/ -m "not slow and probe" --splits 4 --group 1
```

### Caching probe measurements

Loading Turing into a fresh Julia session costs seconds, while the container
starts in a fraction of one. Several cells measure the same density before
checking different properties of it. Point `QUIVERS_PROBE_CACHE` at a directory
to memoize those measurements and avoid repeating the runtime startup:

```bash
QUIVERS_PROBE_CACHE=.probe-cache python -m pytest tests/ -m "not slow and probe"
```

The key covers the image ID, the emitted source, the points, the shape
and dtype tables, and the probe script together with its reshape
helpers, all hashed by content. Rebuilding an image changes its ID and
invalidates the cached result even when its tag stays the same.

Caching is off unless the variable is set. A cached measurement is only as sound as its key, so
a run that gates a release should execute every container.

## Project Structure

```
quivers/
├── docs/                          # Site sources for mkdocs
│   ├── api/                       # API reference (mkdocstrings directives)
│   ├── developer/                 # Contributor notes
│   ├── examples/                  # Example gallery
│   ├── getting-started/           # Installation, quickstart, concepts
│   ├── guides/                    # Guides by feature area
│   ├── hooks/                     # mkdocs hooks (bibliography, changelog mirror, lexer)
│   ├── reference/qvr/             # QVR language reference
│   ├── semantics/                 # Denotational semantics of QVR
│   ├── tutorials/                 # Python and QVR tutorials
│   └── references.bib             # Bibliography
├── editors/                       # Editor extensions (VS Code, Zed)
├── grammars/qvr/                  # Tree-sitter grammar for QVR
│   ├── grammar.js                 # Grammar source of truth
│   ├── src/                       # Generated grammar.json, parser.c, node-types.json
│   ├── test/corpus/               # Tree-sitter fixtures
│   ├── queries/                   # Editor highlight queries
│   └── vcs/                       # panproto schema history of the grammar
├── src/quivers/                   # Main package
│   ├── analysis/                  # Static analysis of chains and plates
│   ├── arrows/                    # Arrow type classes
│   ├── categorical/               # Functors, natural transformations, monoidal structure
│   ├── cli/                       # qvr command line: check, run, transpile, migrate, REPL
│   ├── continuous/                # Continuous families, programs (MonadicProgram), scans
│   ├── core/                      # Objects, morphisms, Algebra and composition rules
│   ├── data/                      # Data-frame schemas and encoders
│   ├── diagnostics/               # Convergence, calibration, comparison (ArviZ I/O)
│   ├── dsl/                       # QVR front end
│   │   ├── parser/                # Parse tree to AST
│   │   ├── ast_nodes/             # didactic Model AST nodes
│   │   ├── compiler/              # AST to Program, with resolution and call lowering
│   │   ├── qiec_lowering.py       # Elaboration and checking into QIEC
│   │   ├── emit.py                # AST to QVR source printer
│   │   ├── program_theory.py      # QVR_PROGRAM_PROTOCOL and schema extraction
│   │   └── pygments_lexer.py      # Pygments lexer for highlighting
│   ├── effects/                   # Effect handlers for probabilistic programs
│   ├── enriched/                  # Ends, Kan extensions, profunctors, optics
│   ├── formulas/                  # Regression formulas compiled to QVR
│   ├── inference/                 # SVI, guides, MCMC, SMC, DAIS, lifts, posterior predictive
│   ├── kernel/                    # Jupyter kernel
│   ├── lsp/                       # Language server
│   ├── monadic/                   # Monad and comonad type classes, algebras, distributive laws
│   ├── qiec/                      # QIEC data model, checker, serialization, reference machine
│   ├── stochastic/                # Finite stochastic morphisms, semirings, weighted deduction
│   ├── structural/                # Encoders, decoders, and losses over term signatures
│   ├── transpile/                 # Lowering and renderers for target PPLs
│   ├── giry.py                    # Re-export of quivers.stochastic.giry
│   └── program.py                 # Program, the compiled module container
├── tests/                         # Test suite; subdirectories for dsl, qiec, transpile, and others
├── tools/                         # API reference, docstring, and release checks
├── scripts/                       # Rendered math check for the built site
├── pyproject.toml                 # Package metadata and pytest configuration
└── mkdocs.yml                     # Site configuration
```

## Code Style Conventions

### Type Hints

Every function signature carries type hints, written in the syntax of Python 3.14:

- `dict[K, V]`, not `Dict[K, V]`
- `list[T]`, not `List[T]`
- `X | None`, not `Optional[X]` or `Union[X, None]`
- `tuple[T, ...]` for tuples of any length

```python
def process_data(values: list[float], multiplier: float = 1.0) -> dict[str, float]:
    """Process numeric data."""
    return {str(i): v * multiplier for i, v in enumerate(values)}
```

Annotate local variables only where the type is not evident from the code.

### Docstrings

Public modules, classes, and functions have numpy-style docstrings with `Parameters`, `Returns`,
and `Raises` sections:

```python
def calculate_entropy(probabilities: list[float]) -> float:
    """Calculate Shannon entropy of a probability distribution.

    Parameters
    ----------
    probabilities : list[float]
        Probabilities that sum to 1.0.

    Returns
    -------
    float
        Shannon entropy in nats.

    Raises
    ------
    ValueError
        If probabilities do not sum to approximately 1.0.

    Examples
    --------
    >>> entropy([0.5, 0.5])
    0.6931471805599453
    """
```

### Comments

Inline comments are lowercase and explain why the code does something, not what it does:

```python
# compute sufficient statistics for exponential family
sufficient_stats = compute_stats(data)

# handle edge case where prior is uniform
if prior_strength == 0:
    posterior = likelihood
```

### Python Version and Language Features

The package targets Python 3.14 and later. Code uses union syntax (`X | None`), positional-only
parameters (`def func(a, /, b)`), and `Literal[...]` discriminators for `dx.TaggedUnion` variants.

### Value types are didactic Models

[didactic](https://github.com/panproto/didactic) is the library whose `Model` types hold the
data values of quivers. Values with named fields (AST nodes, `FinSet`, `ProductSet`, `CoproductSet`,
the `ContinuousSpace` and `Category` variants, `RuleSystem`) subclass `didactic.api.Model`.
Recursive sums are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field.
`dx.field(..., converter=...)` normalizes a field (for instance, flattening nested `ProductSet`
components), `@dx.derived` declares a computed field, and `__axioms__` states invariants across
fields.

Some classes that hold mutable tensors or runtime state are plain dataclasses, for instance
`Presheaf`, `Weight`, `SampleSite`, and `Trace`.

### Exports

What a module exports, and where it is imported from, follows the [export policy](public-api.md).
Every public module declares `__all__`, each package re-exports its modules' public names, public
signatures name only public types, and every public name renders in the API reference.
`tests/test_public_api.py` and `tools/check_api_reference.py` enforce the policy.

## The DSL Pipeline

`quivers.dsl.load(path)` and `quivers.dsl.loads(source)` take a `.qvr` file through four stages.
Parsing builds an abstract syntax tree (AST). Resolution determines what each name and object
expression refers to. Elaboration rewrites every executable declaration into explicit terms of the
[Quivers Indexed Effect Core](qiec.md) (QIEC), the typed core language, and checks their types
and effects. Compilation builds the PyTorch objects that run the model.

For instance, `observe y <- Normal(mu, sigma)` names a distribution and an observation. Its QIEC
term evaluates the density at `y` and asks the `score` instance to add the log weight. The
reference machine, the compiled runtime, and the transpilers all read that one operation.

### 1. Parsing

`quivers.dsl.parser.parse(source)` and `parse_file(path)` register the generated QVR grammar and
its native parser with the `AstParserRegistry` of [panproto](https://github.com/panproto/panproto),
which drives [tree-sitter](https://tree-sitter.github.io/). Platform wheels load the verified
native parser directly. An editable checkout may compile a development parser, cached under a
hash of its source. The parser walker then converts the parse tree into a tree of `dx.Model` AST
nodes. Lexical and syntactic errors both raise `ParseError`.

Each syntax construct is a `dx.Model`, and recursive sums (`ObjectExpr`, `Expr`, `LetExprNode`,
`ProgramStep`, `Statement`) are `dx.TaggedUnion` roots:

<!-- python: skip -->
```python
from typing import Literal
from quivers.dsl.ast_nodes import ObjectExpr, OptionEntry, ProgramParam, ProgramStep, Statement

class ProgramDecl(Statement):
    name: str
    params: tuple[str, ...] | None = None
    type_params: tuple[ProgramParam, ...] | None = None
    domain: ObjectExpr
    codomain: ObjectExpr
    options: tuple[OptionEntry, ...] = ()
    draws: tuple[ProgramStep, ...] = ()
    return_vars: tuple[str, ...] = ()
    return_labels: tuple[str, ...] | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["program_decl"] = "program_decl"
```

### 2. Resolution

`quivers.dsl.compiler.resolution._ResolutionMixin` resolves an `ObjectExpr` to either a discrete
`SetObject` or a `ContinuousSpace`. `_resolve_any_space` performs the shared dispatch, and the
narrower helpers reject a result of the wrong kind.

### 3. Elaboration and checking

`quivers.dsl.qiec_lowering.lower_qvr_to_qiec(module)` produces the checked `QiecModule`:

1. it expands the weights of networks declared with `weights=latent` into explicit draws;
2. it projects the module's indexed declarations through didactic's GADT API, and didactic's
   `lower_checked` checks that projection against its first-order theory;
3. it elaborates every executable declaration, programs included, into QIEC terms;
4. it checks the lowered module again with the QIEC checker.

A rejected module raises `QiecDiagnosticError` with a stable code and a source position. A program
whose elaboration is not defined raises the code `qiec-program-gap`. The compiler then lowers the
module without its programs and records the gap on the module. The [QIEC developer note](qiec.md)
describes what each declaration elaborates to.

### 4. Compilation

`quivers.dsl.compiler.Compiler(ast).compile()` compiles each statement in order. It resolves
object expressions, checks that domains and codomains agree, lowers a program's calls of
computations into compiled steps, and builds the morphism graph. It then wraps the exported
morphism in a `quivers.Program` and attaches the checked QIEC module from stage 3 as
`program.qiec`. A program whose calls the compiler cannot turn into steps compiles to a
`CheckedProgram`, which runs on the reference machine.

### Schema extraction

`quivers.dsl.program_theory.extract_program_schema(compiler)` walks the resolved environment and
emits a `panproto.Schema` over `QVR_PROGRAM_PROTOCOL`. `panproto schema diff` can then compare two
programs, and panproto can generate migration lenses between them.

## Adding a New Distribution Family

Most families with independent coordinates use the single registry in
`src/quivers/continuous/family_spec.py`. If PyTorch already provides the distribution, add an
`_make_family(...)` call in `src/quivers/continuous/families.py` with the parameter names and the
bijectors that map unconstrained parameters onto their supports. That call creates the
`Conditional*` class and registers both inline forms in the DSL.

<!-- python: skip -->
```python
ConditionalMyFamily = _make_family(
    "ConditionalMyFamily",
    MyTorchDistribution,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional MyFamily(loc(x), scale(x)).",
)
```

A family with vector or matrix events, a custom constructor, or nonstandard sampling needs a
hand-written `ContinuousMorphism` and a `FamilySpec` whose override fields point to it. An
entirely new distribution may also need a `torch.distributions.Distribution` implementation.

Test the registry entry, the fixed and mixed inline forms, conditional sampling, and `log_prob`.
`tests/test_family_registry.py` holds the shared registry checks, and behaviour specific to one
family belongs under `tests/continuous/`. Document the family in
`docs/guides/continuous-families.md` and `docs/api/continuous/families.md`.

## Checks a change must pass

Continuous integration runs the following on every pull request, and each can be run locally:

| Check | Command |
|---|---|
| Lint and formatting | `ruff check src/ tests/` and `ruff format --check src/ tests/` |
| Docstring sections in `quivers.qiec` | `python tools/docstring_audit.py --check src/quivers/qiec` |
| Types | `pyright src/quivers` |
| Site build | `mkdocs build --strict` |
| API reference coverage | `python tools/check_api_reference.py site` |
| Rendered math | `python scripts/check_rendered_math.py site` |
| Tests | the tiers above |

`tests/test_doc_blocks.py` compiles every `qvr` block under `docs/` and in `README.md`, and runs
every Python block under `docs/tutorials/`. An HTML comment on the line before a fence chooses the
mode: `<!-- compile: false -->` and `<!-- python: skip -->` exclude a block,
`<!-- compile: cumulative -->` joins it to the earlier cumulative blocks of the file, and
`<!-- compile: qiec -->` lowers it through QIEC as well. A change to a documented behaviour thus
changes its documentation in the same commit.

## Commits and pull requests

Commit messages follow the conventional form `type(scope): summary`. The type is one of `feat`,
`fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `build`, or `ci`, and the scope names a module
or package (`inference`, `continuous`, `transpile`, `dsl`). Work on a branch and open a pull
request against `main`.

## Questions and Issues

Bugs, feature requests, and questions go to the
[issue tracker](https://github.com/quivers-dev/quivers/issues).
