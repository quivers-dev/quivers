# Contributing to Quivers

This guide covers setting up a development environment, understanding the project structure, and contributing code to the quivers library.

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

This installs the package in editable mode along with development dependencies.

### Running Tests

Run the full test suite:

```bash
python -m pytest tests/ -x
```

The `-x` flag stops at the first failure, which is useful for iterative development. For more options:

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

The marker is applied by module, from the registry in
`tests/transpile/conftest.py`, and audited against the source: a module
that calls `run_probe` without appearing there fails
`test_probe_marker_registry.py`, so a new probe module cannot land in
the cheap tier by omission.

Probe cells need Docker. The session fixture starts the daemon and
builds any missing images rather than skipping, since a skipped check
is a silent gap. Set `QUIVERS_SKIP_DOCKER=1` to opt out, which is
appropriate when running only `-m "not slow and not probe"`.

### Running tests in parallel

Both tiers are parallel-safe: probe scratch directories are unique per
call, containers are anonymous, and the image build is serialised
across workers by a file lock.

```bash
python -m pytest tests/ -m "not slow and not probe" -n auto --dist loadfile
python -m pytest tests/ -m "not slow and probe" -n auto
```

`--dist loadfile` is the right mode for the cheap tier. Its tests are
individually far shorter than the cost of handing one to a worker, so
distributing whole files wins where distributing single tests loses.

To divide the probe tier across machines, `pytest-split` takes a share:

```bash
python -m pytest tests/ -m "not slow and probe" --splits 4 --group 1
```

### Caching probe measurements

A probe pays for a language runtime rather than for the container.
Loading Turing into a fresh Julia session costs seconds, while the
container itself starts in a fraction of one, and many cells measure
the same density and then assert different things about it. Point
`QUIVERS_PROBE_CACHE` at a directory to memoise those measurements:

```bash
QUIVERS_PROBE_CACHE=.probe-cache python -m pytest tests/ -m "not slow and probe"
```

The key covers the image ID, the emitted source, the points, the shape
and dtype tables, and the probe script together with its reshape
helpers, all hashed by content. Keying on the image ID rather than the
tag is what makes a rebuilt image miss instead of answering from
before it changed.

Caching is off unless the variable is set. A memoised measurement is
only as sound as its key, so a run that gates a release should execute
every container for real.

## Project Structure

```
quivers/
├── docs/                          # Documentation
│   ├── api/                       # API reference
│   ├── developer/                 # Developer documentation
│   ├── getting-started/           # User guides
│   ├── guides/                    # Long-form guides
│   └── tutorials/                 # Tutorials
├── grammars/qvr/                  # Authoritative QVR grammar and editor queries
├── src/quivers/                   # Main package
│   ├── __init__.py
│   ├── analysis/                  # Static program and plate analysis
│   ├── arrows/                    # Hughes-style arrow hierarchy
│   ├── categorical/               # Categorical algebra
│   ├── continuous/                # Continuous distributions (more than 40 families)
│   ├── core/                      # Core types and utilities
│   ├── data/                      # Data-frame schemas and encoders
│   ├── diagnostics/               # ArviZ-compatible diagnostics
│   ├── dsl/                       # QVR DSL (grammar walker, compiler, emitter)
│   ├── effects/                   # Algebraic effect handlers
│   ├── enriched/                  # Enriched categories
│   ├── formulas/                  # Formula parsing and model construction
│   ├── inference/                 # Variational inference
│   ├── kernel/                    # Jupyter kernel
│   ├── lsp/                       # Language Server Protocol implementation
│   ├── monadic/                   # Monadic programs (draw, observe, return)
│   ├── qiec/                      # Typed indexed-effect core and evaluator
│   ├── stochastic/                # Stochastic morphisms
│   ├── structural/                # Structural encoders, decoders, and losses
│   └── transpile/                 # Cross-language PPL emitters
├── tests/                         # Test suite (mirrors src structure)
├── pyproject.toml                 # Package metadata
└── mkdocs.yml                     # Documentation config
```

## Code Style Conventions

### Type Hints

Include type hints in all function signatures. Use modern Python 3.14+ syntax:

- Use `dict[K, V]` not `Dict[K, V]`
- Use `list[T]` not `List[T]`
- Use `X | None` not `Optional[X]`
- Use `tuple[T, ...]` for variable-length tuples

```python
def process_data(values: list[float], multiplier: float = 1.0) -> dict[str, float]:
    """Process numeric data."""
    return {str(i): v * multiplier for i, v in enumerate(values)}
```

Do not use type hints in function bodies or variable assignments unless necessary for clarity in complex code sections.

### Docstrings

Use numpy-style docstrings for all public modules, classes, and functions:

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

Use lowercase inline comments to clarify non-obvious logic:

```python
# compute sufficient statistics for exponential family
sufficient_stats = compute_stats(data)

# handle edge case where prior is uniform
if prior_strength == 0:
    posterior = likelihood
```

Avoid stating the obvious. Comments should explain "why," not "what."

### Python Version and Modern Features

Maintain compatibility with Python 3.14 and later. Use modern features:

- Type union syntax: `X | None` instead of `Union[X, None]`
- Positional-only parameters: `def func(a, /, b)`

### Exports

What a module exports, and where users import it from, follows the
[export policy](docs/developer/public-api.md): every public module declares `__all__`, each
package re-exports its modules' public names, public signatures name
only public types, and every public name renders in the API reference.
`tests/test_public_api.py` and `tools/check_api_reference.py` enforce it.

### Opaque didactic fields

didactic's `Model.__eq__` and `Model.__hash__` read only a model's
encoded storage, which an opaque field (`dx.field(opaque=True)`) does
not enter, so a plain `dx.Model` treats values that differ in an opaque
field as equal, and its pickle hooks drop the field. Prefer a typed
field whenever didactic can encode the type (`dict[str, str]`,
`dict[str, tuple[int, ...]]`, nested models). When a field must stay
opaque, the model subclasses `quivers._values.ValueModel` and the field
declares how it enters equality, hashing, copying, and pickling:

- `extras=OPAQUE_BY_VALUE` for data: compared by content (arrays
  exactly by dtype, shape, and bytes; mappings by key set; tuples
  elementwise), and frozen at construction (arrays become read-only
  copies, mappings read-only proxies);
- `extras=OPAQUE_BY_IDENTITY` for live runtime objects (tensors,
  modules, callables, AST nodes, dataframes): compared by object
  identity.

State the policy in the class docstring. `tests/test_opaque_field_policies.py`
fails on any opaque field without a declared policy and on any model
with an opaque field that is not a `ValueModel`.

## The DSL Pipeline

The QVR DSL processes `.qvr` files through these stages:

### 1. Grammar and parsing (`grammars/qvr/`, `dsl/parser/`)

There is no hand-written lexer or recursive-descent parser. The
grammar is a [tree-sitter](https://tree-sitter.github.io/) grammar
(`grammars/qvr/grammar.js`). Platform wheels include a native parser bound to
the generated source by an integrity manifest; editable checkouts compile that
source into a content-addressed development cache. The resulting parser is
registered directly with panproto's `AstParserRegistry`. Parsing a `.qvr`
source yields a panproto schema
(the parse tree as vertices, edges, and field constraints), and the
walkers in `src/quivers/dsl/parser/` turn that schema into AST nodes.
The walker rejects any `ERROR` or missing node, so a malformed source
fails loudly with a line and column rather than parsing to a silently
different tree.

Editing the grammar means editing `grammar.js`, regenerating with
`tree-sitter generate`, updating the package mirrors and editor queries, and
recording the new schema and immutable parser source in `grammars/qvr/vcs/`.
The `qvr migrate` chain carries `.qvr` sources across grammar releases.

### 2. AST nodes (`dsl/ast_nodes/`)

Each grammar production maps to a [didactic](https://github.com/panproto/didactic)
model, not a dataclass. Statement variants live under a `dx.TaggedUnion`
keyed on `kind`; leaf records are `dx.Model` subclasses:

```python
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

### 3. Compilation (`dsl/compiler/`)

The compiler walks the AST and builds executable programs:

- Resolves object and morphism references against the module's declarations.
- Reads each declaration's option block against a closed key set, reporting an unknown key with a did-you-mean suggestion.
- Expands surface program steps (`sample`, `observe`, `marginalize`, `let`, `score`) into the internal bind IR.
- Emits a [`Program`][quivers.program.Program] whose morphism is a Kleisli arrow in the (discrete or continuous) Giry monad.

## Adding a New Distribution Family

Most independent families use the single registry in
`src/quivers/continuous/family_spec.py`. If PyTorch already provides
the distribution, add an `_make_family(...)` call in
`src/quivers/continuous/families.py` with the parameter names and
bijectors that map unconstrained parameters onto their supports.
That call creates the `Conditional*` class and registers both inline
DSL paths.

<!-- python: skip -->
```python
ConditionalMyFamily = _make_family(
    "ConditionalMyFamily",
    MyTorchDistribution,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional MyFamily(loc(x), scale(x)).",
)
```

Families with vector or matrix events, custom constructors, or
nonstandard sampling behavior need a hand-written
`ContinuousMorphism` plus a `FamilySpec` whose override fields point
to that implementation. An entirely new underlying distribution may
also require a `torch.distributions.Distribution` implementation.

Add registry, fixed-inline, mixed-inline, conditional sampling, and
`log_prob` coverage. `tests/test_family_registry.py` contains the
shared registry checks; specialized behavior belongs under
`tests/continuous/`. Document the family in
`docs/guides/continuous-families.md` and
`docs/api/continuous/families.md`.

## Testing Philosophy

- Write tests for all public APIs
- Test both happy paths and edge cases
- Use pytest fixtures for common setup
- Organize tests to mirror the source tree structure
- Aim for clear, descriptive test names: `test_<function>_<condition>_<expected>`

## Git Workflow

1. Create a feature branch: `git checkout -b feature/description`
2. Make focused commits with clear messages
3. Push to your fork and open a pull request
4. Ensure all tests pass before requesting review
5. Respond to feedback and update as needed

## Questions and Issues

Open an issue on the [GitHub repository](https://github.com/quivers-dev/quivers) for bugs, feature requests, or questions.
