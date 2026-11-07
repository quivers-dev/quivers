# Transpilation architecture

The transpile layer in `quivers.transpile`
converts supported QVR modules $M$ to source bytes for a
target probabilistic programming language $\mathsf{T}$. This page
describes the architecture of that realization: the intermediate
representation, the family-metadata registry, the per-target
renderer interface, and the dispatch pattern that lets one walker
serve eleven backends without family-name special-casing.

The companion [transpilation-correctness contract](transpile-correctness/index.md)
states the evidence available for supported programs. This page
describes how the pieces fit together.

A transpiler translates a model description into another language. It must
preserve the model's factors, shapes, and support, even when the destination
uses different notation. For instance, a Normal family with `scale=2`
may become a constructor with standard deviation 2 or one with precision
$1/2^2=0.25$. Copying the number without translating its meaning would change
the model.

The intermediate representation (IR) is a structured common description of
the model. Checking establishes its types and effects; planning identifies
model operations; rendering expresses them in a target grammar; printing
turns that structure into source text. Keeping these steps separate lets an
axis or distribution argument retain its meaning while its spelling changes.
The examples below expose each boundary before the larger LDA walkthrough.

## 1. The shared pipeline

The transpile pipeline elaborates and checks the module first; the program's
plan is derived from its checked computation; the selected renderer then
renders the plan and the module's other computations, analyzing each
computation's capabilities before it is placed:

$$
\mathrm{Module}
\;\xrightarrow{\;\mathsf{Lower}=\mathsf{Check};\mathsf{Plan}\;}\;
\mathrm{IRProgram}
\;\xrightarrow{\;\mathsf{Render}_{\mathsf{T}}[
  \mathsf{AnalyzeQIEC}_{\mathsf{T}}]\;}\;
\mathrm{panproto.Schema}
\;\xrightarrow{\;\mathsf{Pretty}_{\mathsf{T}}\;}\;
\mathrm{bytes}
$$

Each arrow is a small transformation; the composition is
the correctness framework's first structural handle,
because each arrow's correctness lemma is local to its file.

* **`Check`** elaborates the module to its checked core: it compiles the
  indexed-family signature through Didactic's public `GADT` API, negotiates
  the exact route, elaborates every entry-point program to a named
  computation, validates the complete module, and returns its stable kernel
  module. This step is target-independent.
* **`Plan`** is
  target-independent. It reads the program's computation and recognizes
  the statements a probabilistic-programming target has: one `IRSample`
  per `Random.sample` request, one `IRObserve` per scored density at an
  observation, one `IRDeterministic` per pure binding, one `IRScore` per
  other weight, and one `IRMarginalize` per marginalization helper whose
  answer is scored. Each node carries the support and plate shape read off
  the module's types, `FAMILY_META`, and
  `torch.distributions.Distribution.arg_constraints`; the plan's inputs are
  the entry point's parameters. The result is one `IRProgram` carrying the
  complete module as the typed `IRQiecModule` tree beside the plan. A
  QIEC-only source thus lowers to its module with an empty plan.
* **`Render[T]`**
  is one subclass per backend
  (`StanRenderer`,
  `NumPyroRenderer`,
  `PyMCRenderer`,
  ...). It consumes the plan and emits a target-specific
  `panproto.Schema` using the support predicates of §2.3 and the `FAMILY_META`
  entries, then renders the module's other computations with
  `render_computations_dynamic` or `render_computations_static`, which call
  **`AnalyzeQIEC[T]`** over each named computation first. The analysis records
  open or effectful rows, static polymorphism, indexed cases, transport,
  attachments, and resumption grades, then compares that set with the target's
  declared capabilities. A mismatch has the stable form
  `qiec:capability:<feature>:<computation>`, retains the computation's source
  origin, and stops the rendering.
* **Pretty[T]** is
  `panproto.AstParserRegistry.emit_pretty`
  for the target's tree-sitter grammar. It renders the schema as
  the canonical source-byte serialization.

The ordinary compiler checks the same module and attaches its `QiecModule` to
the compiler environment and produced `Program`. Transpilation does not depend
on that runtime compiler object: `transpile()` classifies the surface, and
`Lower.forward()` elaborates and checks the source AST itself, so direct
lowering is validated the same way. Target capability analysis remains inside
the renderer, which consumes `IRQiecModule` after source processing.

## 2. The IR

The IR lives in
`src/quivers/transpile/ir.py`. Every entry
is a `dx.Model` or
`dx.TaggedUnion`. The IR is purely
structural: no target-language strings, no schema vertices, no
panproto types.

### 2.1 Typed QIEC module

`IRProgram.module` is the `IRQiecModule`, a lossless
structural projection of the checked kernel module: every declaration, stable
identifier, source origin, row and lacks constraint, value, equality witness,
transport, computation node, handler clause, and resumption grade has its own
`dx.Model` or `dx.TaggedUnion` representation. Thus the IR remains
Didactic/Panproto-translatable without asking a renderer to parse a
`qiec-json/v1` string.

The host-language targets Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL,
and Church consume this tree through corresponding implementations of the QIEC
runtime ABI. These implementations provide free computations, lexical
handlers, stable request addresses, constructors, indexed case dispatch,
attachments, and resumption-grade checks. Conformance tests reparse every
target's emitted source and exercise the host ABI where its runtime is
available, comparing request dispatch and returned values with the reference execution.

Stan, BUGS, and JAGS emit the analyzer-approved static subset. Each computation
must have a closed, empty effect row and an empty static telescope, return a
scalar, and contain only scalar `Return` and `Bind` forms. These restrictions do
not rule out ordinary value parameters: Stan admits named `Bool`, `Int`, and
`Real` parameters. BUGS and JAGS require parameterless computations because
their graph-level definitions have no callable parameter ABI. Other QIEC
features receive precise capability diagnostics; no renderer may treat an
executable term as ignorable metadata.

### 2.2 `Plate`: event versus batch axes

Ten three-dimensional multivariate Normal draws have [batch size ten and
event size three](https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes#multivariate_distributions). A target can express them as a vectorized distribution or
ten loop iterations, but must preserve the joint three-coordinate event
inside each draw. With non-diagonal covariance, the score includes cross
terms relating coordinates; three independent scalar scores would omit
those terms. The two fields below record the [event dimensions and the batch dimensions](morphisms.md#6-axis-role-specifications).

A draw's plate annotation decomposes into the event axes (the
family's joint structure) and the batch axes (replication).

```python
class Plate(dx.Model):
    event_dims: tuple[Dim, ...]
    batch_dims: tuple[Dim, ...]


class Dim(dx.TaggedUnion, discriminator="kind"):
    name: str


class DimStatic(Dim):
    size: int
    kind: Literal["static"] = "static"


class DimDynamic(Dim):
    size_name: str
    kind: Literal["dynamic"] = "dynamic"
```

`event_dims` comes from `AxisSpec.over` (or the deprecated
`step.index` shorthand); `batch_dims` from `AxisSpec.iid_over`.
`Lower` preserves the source-declaration order; each renderer
walks `batch_dims` to emit nested `for` loops (Stan), nested
plate contexts (NumPyro / Pyro),
`dims=(...)` declarations (PyMC), or
`filldist` / `arraydist` wrappers
(Turing.jl / Gen.jl) per its native idiom.

### 2.3 Support classification

`src/quivers/transpile/ir.py` exports a small set of predicates
over
`torch.distributions.constraints.Constraint`:

```python
def is_real_scalar(c: Constraint) -> bool: ...
def is_real_positive(c: Constraint) -> bool: ...
def is_real_unit_interval(c: Constraint) -> bool: ...
def is_real_vector(c: Constraint) -> bool: ...
def is_real_simplex(c: Constraint) -> bool: ...
def is_real_cov_matrix(c: Constraint) -> bool: ...
def is_real_corr_chol(c: Constraint) -> bool: ...
def is_real_matrix(c: Constraint) -> bool: ...
def is_real_one_hot(c: Constraint) -> bool: ...
def is_int_bit(c: Constraint) -> bool: ...
def is_int_category(c: Constraint) -> bool: ...
def is_int_count(c: Constraint) -> bool: ...
```

These are the only typeclass operations a renderer performs.
Renderers never `isinstance(c, _Simplex)` directly; they call
`is_real_simplex`. Adding
a new support kind (ordered vectors, say) means adding one
predicate. The predicates dispatch on torch's existing
`Constraint`
taxonomy.

Because Torch does not survive
didactic's tagged-union encode / decode round trip, the IR
actually stores constraints as a structural mirror
`ConstraintSpec` that
materializes to the underlying `Constraint` via
`.to_constraint()` when a renderer needs the real value. The
mirror has one variant per kind the predicates distinguish; the
`from_constraint`
converter goes the other way at Lower time.

### 2.3 `IRArg`: typed argument tree

The parser produces stringly-typed bracket args like `"phi[z]"`;
Lower parses them into a typed tree so renderers don't re-parse
strings.

```python
class IRArg(dx.TaggedUnion, discriminator="kind"):
    ...

class IRArgNumber(IRArg):
    value: float
    kind: Literal["number"] = "number"

class IRArgRef(IRArg):
    name: str
    indices: tuple[IRArg, ...] = ()
    kind: Literal["ref"] = "ref"

class IRArgBroadcast(IRArg):
    """A scalar broadcast to satisfy an arg's expected constraint."""
    value: IRArg
    target_shape: tuple[int, ...]
    kind: Literal["broadcast"] = "broadcast"

class IRArgList(IRArg):
    elements: tuple[IRArg, ...]
    kind: Literal["list"] = "list"

class IRArgMatrix(IRArg):
    rows: tuple[IRArgList, ...]
    kind: Literal["matrix"] = "matrix"

class IRArgFamilyRef(IRArg):
    """A reference to a morphism whose ``~ Family(...)`` init clause
    names the wrapped distribution (used by Truncated, Mixture,
    Independent, Transformed, LKJCorrelationFactor)."""
    name: str
    kind: Literal["family_ref"] = "family_ref"
```

Lower wraps a scalar arg in `IRArgBroadcast` when the matched
`arg_constraints[name]`
is `IndependentConstraint(base, n>=1)`. Each renderer translates
the broadcast to its native op:
`rep_vector(x, K)` in Stan,
`jnp.full((K,), x)` in NumPyro,
`torch.full((K,), x)` in Pyro,
`np.full((K,), x)` in PyMC,
`fill(x, K)` in Turing / Gen,
`repeat(K, function() { return x; })` in WebPPL,
`(make-list K x)` in Church. The translation lives in each
renderer's `broadcast(value, target_shape)` method.

### 2.4 `IRNode`: program-body statements

```python
class IRNode(dx.TaggedUnion, discriminator="kind"):
    ...

class IRDataInput(IRNode):
    name: str
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["data_input"] = "data_input"

class IRSample(IRNode):
    name: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["sample"] = "sample"

class IRObserve(IRNode):
    name: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    via: str | None
    kind: Literal["observe"] = "observe"

class IRDeterministic(IRNode):
    name: str
    expr: IRExpr
    constraint: ConstraintSpec
    plate: Plate
    kind: Literal["deterministic"] = "deterministic"

class IRScore(IRNode):
    name: str
    expr: IRExpr
    kind: Literal["score"] = "score"

class IRMarginalize(IRNode):
    """A discrete-latent integration scope. Each renderer decides
    how to emit this: Stan as `log_sum_exp` per-group enumeration,
    every other backend by inline lowering to `IRSample(latent) +
    scope body`."""
    latent: str
    family: str
    args: tuple[IRArg, ...]
    arg_names: tuple[str, ...]
    constraint: ConstraintSpec
    plate: Plate
    reduction: Literal["logsumexp"]
    scope: tuple[IRNode, ...]
    kind: Literal["marginalize"] = "marginalize"

class IRCall(IRNode):
    """A call of a module computation the plan did not inline. The
    dynamic targets render it against the callee's generated entry
    point, with the program's `Random` and `Score` instances bound to
    the host's own sample and factor primitives; Stan calls the
    callee as a user-defined function when it can define one, and
    BUGS and JAGS refuse it."""
    name: str
    callee: str
    static_arguments: tuple[IRQiecStatic, ...]
    arguments: tuple[IRExpr, ...]
    random_instance: str
    score_instance: str
    plate: Plate
    kind: Literal["call"] = "call"

class IRReturn(IRNode):
    names: tuple[str, ...]
    kind: Literal["return"] = "return"
```

`arg_names` parallels `args` and carries the keyword names from
torch's `arg_constraints` (`"loc"`, `"scale"`,
`"concentration"`, ...). Renderers that prefer keyword calls
(NumPyro, Pyro, PyMC, Edward2) read from `arg_names`; positional
renderers (Stan, BUGS, JAGS) ignore it.

```python
class IRProgram(dx.Model):
    name: str
    inputs: tuple[IRDataInput, ...]
    body: tuple[IRNode, ...]
    module: IRQiecModule
    cards: dict[str, int]
```

## 3. `FamilyMeta`: the registry for transpile-only facts

Distribution metadata answers translation questions rather than inferring a
model from a distribution name. A name such as `Normal` identifies a family;
the argument mapping says whether a target calls its location `loc` or `mu`
and whether its spread is a scale or precision. Shape and support remain
part of the typed distribution contract. The renderer needs both kinds of
information to produce the intended density.

One registry, in
`src/quivers/transpile/family_meta.py`:

```python
class FamilyMeta(dx.Model):
    qvr_name: str
    distribution_class: type[Distribution]
    quivers_class: type[ContinuousMorphism] | None
    target_names: dict[str, str]
    arg_aliases: dict[str, dict[str, str]]
```

* `qvr_name`: the DSL-facing family name (`"Normal"`,
  `"Dirichlet"`).
* `distribution_class`: the underlying
  `torch.distributions.Distribution`
  subclass (or a thin shim exposing the right `arg_constraints` +
  `.support` surface for families with no native torch
  counterpart, like `OrderedLogistic` and `HalfStudentT`). Source
  of truth for the family's argument constraints and output
  support.
* `quivers_class`: the
  [`ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism]
  subclass the inference layer instantiates at runtime
  ([`ConditionalNormal`][quivers.continuous.families.ConditionalNormal],
  etc.). Empty for wrapper families whose runtime morphism is
  constructed from a referenced inner morphism.
* `target_names`: per-backend distribution-name mapping. The
  single source of truth for backend-to-distribution-name
  resolution. Renderers look up
  `FAMILY_META[family].target_names[backend]`; no per-renderer
  `_FAMILIES` dict exists.
* `arg_aliases`: per-backend per-arg renames. Most families have
  empty `arg_aliases`. Renderers that apply parameterization-
  converting arithmetic (BUGS / JAGS Normal mean+scale to
  mean+precision; PyMC's `concentration → a` rename for
  Dirichlet) key the arithmetic on the alias's target name.

The marginalize-eligibility check is a per-call function rather
than a per-family flag:

```python
def finite_enumerable_at_call_site(
    family_meta: FamilyMeta,
    args: tuple[IRArg, ...],
) -> bool: ...
```

Returns `True` for Bernoulli, Categorical, OrderedLogistic, and
OrderedProbit unconditionally. For Binomial returns `True` only
when `args[0]` (total_count) is a literal `IRArgNumber`; the
Stan renderer's `marginalize` raises
`UnsupportedConstruct`
when the check returns `False`.

### 3.1 What `FAMILY_META` does not carry

* **Argument shapes / constraints.** Lives in
  `distribution_class.arg_constraints`. Lower reads from there.
* **Output support.** Lives in `distribution_class.support` (the
  class-level support, or its evaluation on a sentinel parameter
  set for instance-dependent supports like `Uniform(low, high)`).
* **Event rank.** Derived from
  `distribution_class().event_shape` on the sentinel.

This separation keeps `FAMILY_META` small (under a hundred lines
per family) and ties the structural classification to torch's
existing implementation. Adding a new family is one
[`Conditional*`][quivers.continuous.families.ConditionalNormal]
class plus one `FamilyMeta` entry; no renderer touches.

## 4. `Lower`: Module → IR

[Program elaboration](typing.md#8-program-elaboration) describes the checked computation that lowering consumes. The [transpilation pipeline](transpile-correctness/index.md#1-pipeline) states the structural comparisons used to test this stage and the renderer that follows it.

`Lower` is a single class in `src/quivers/transpile/plan.py`
implementing `dx.Mapping[Module, IRProgram]`. Its `forward`:

1. Elaborates and checks the module (`checked_module`), so a direct
   `Lower.forward()` call cannot bypass validation; a program the
   elaboration has no computation for is refused under the kind the gap
   names. A module without a program lowers to its `IRQiecModule` with an
   empty plan.
2. Runs `expand_composite_lets` on the program, so the site each step of
   the plan corresponds to is one step of the expanded source, whose
   morphism fixes a class-index draw's alphabet and a structured family's
   wire form.
3. Walks the program's computation. A `Random.sample` request becomes an
   `IRSample` (the site label naming it); a `Score.add` of a density at an
   observation becomes an `IRObserve`, with the fibration a
   `SegmentSum` re-indexes the density by stated as `via`; a pure binding
   becomes an `IRDeterministic`, absorbed into an `IRScore` when the next
   request scores its value; a marginalization helper's call whose answer
   is scored becomes an `IRMarginalize` whose scope is the helper's
   collected body; the final `Return` becomes an `IRReturn`.
4. Reads each construction's family, plate, and arguments off its
   `DistributionValue`: a literal, a binding, an indexed binding, a list,
   a matrix, a spread literal as `IRArgBroadcast`, a kernel matrix as
   `IRArgKernel`, and a parameter map's head as a reference to the
   `IRDeterministic` bindings that compute it. Arguments are ordered as the
   torch constructor binds them, scalars are wrapped in `IRArgBroadcast`
   where the constraint is `IndependentConstraint(base, n>=1)`, and the
   support is resolved through `FAMILY_META`'s sentinel instances and
   narrowed by the declared bounds of the site's axes.
5. Reads pure values back as let expressions: primitives as the operators
   and builtins they implement, gathers as subscripts, comprehensions as
   factors, and a last-axis reduction as its builtin call.
6. Derives the inputs from the entry point's parameters, typed by the
   module and named by the axes the elaboration recorded, in the groups
   the targets declare them: the program's domain, its scalars,
   fibrations, observations, parameter maps, and data.
7. Spells every name for the targets. A draw from a program runs the
   program in place under names of the caller's own, a local `z` of a
   program drawn under `theta` being the site `theta$z` on the reference
   machine and the torch runtime; no target language admits `$` in an
   identifier, so the plan spells such a name `theta__z` (`target_name`),
   and the probe harness spells its points the same way.

Lower is target-independent. It never imports any renderer or
backend-specific module.

## 5. `Renderer[T]`: IR → panproto.Schema

Each backend implements a
`Renderer`
subclass with one public method `render(ir: IRProgram) ->
panproto.Schema` and four private dispatch points:

```python
class Renderer(Protocol):
    @abstractmethod
    def render(self, ir: IRProgram) -> panproto.Schema: ...

    @abstractmethod
    def declare(self, name, constraint, plate, *, block) -> SchemaFragment: ...

    @abstractmethod
    def sample(self, name, family, args, arg_names, constraint,
               plate, observed) -> SchemaFragment: ...

    @abstractmethod
    def marginalize(self, node: IRMarginalize) -> SchemaFragment: ...

    @abstractmethod
    def broadcast(self, value, target_shape) -> SchemaFragment: ...
```

`BlockKind` is the renderer-side notion of where a declaration
lands (`"data"`, `"parameters"`, `"transformed_parameters"`,
`"generated_quantities"`, `"function_body"`). Each backend
interprets it per its own program structure: Stan has actual
blocks; NumPyro's "block" is the function body; PyMC's is the
`with pymc.Model() as model:` scope; BUGS / JAGS have only a
single `model { ... }` enclosure.

`RendererBase`
provides the IR walk (`IRDataInput → declare`, `IRSample
(non-observed) → declare + sample`, `IRObserve → declare +
sample(observed=True)`, `IRDeterministic → declare + assignment`,
`IRScore → declare scalar + log-density increment`, `IRMarginalize
→ marginalize`, `IRReturn → backend return idiom`), index-
substitution helpers consumed by both `sample` and `marginalize`,
and the atom enumeration (`marginal_atoms`) every backend's
`marginalize` scores one copy of the scope under.

`declare` dispatches on the predicates of §2.3. The Stan
renderer's table:

| predicate | event | batch | declaration |
|---|---|---|---|
| `is_real_scalar(c)` | () | () | `real <name>;` |
| `is_real_scalar(c)` | () | (B,) | `vector[B] <name>;` |
| `is_real_positive(c)` | () | () | `real<lower=0> <name>;` |
| `is_real_positive(c)` | () | (B,) | `vector<lower=0>[B] <name>;` |
| `is_real_unit_interval(c)` | () | () | `real<lower=0, upper=1> <name>;` |
| `is_real_vector(c)` | (E,) | () | `vector[E] <name>;` |
| `is_real_vector(c)` | (E,) | (B,) | `array[B] vector[E] <name>;` |
| `is_real_simplex(c)` | (E,) | () | `simplex[E] <name>;` |
| `is_real_simplex(c)` | (E,) | (B,) | `array[B] simplex[E] <name>;` |
| `is_real_cov_matrix(c)` | (E,) | () | `cov_matrix[E] <name>;` |
| `is_real_corr_chol(c)` | (E,) | () | `cholesky_factor_corr[E] <name>;` |
| `is_real_matrix(c)` | (R,C) | () | `matrix[R, C] <name>;` |
| `is_int_bit(c)` | () | () | `int<lower=0, upper=1> <name>;` |
| `is_int_bit(c)` | () | (B,) | `array[B] int<lower=0, upper=1> <name>;` |
| `is_int_category(c)` | () | () | `int<lower=1, upper=K> <name>;` |
| `is_int_count(c)` | () | () | `int<lower=0> <name>;` |

Other backends have analogous tables. The grammar of the table is
the same: `(predicate, event_dims, batch_dims) → target-language
declaration`. No row references a family name.

### 5.1 Backend idioms

The eleven backends fall into three idiomatic families:

* **Block-structured static-type** (Stan):
  `data { ... } parameters { ... } model { ... } generated
  quantities { ... }`. The renderer threads the per-block
  declarations through panproto schema vertices for each block.
* **Trace-based** (NumPyro, Pyro, Turing.jl, Gen.jl, Church,
  WebPPL). The renderer emits a `def model(...)` (or `@model
  function`, or `(define (model ...))`) and uses the target's
  native plate primitive
  (`numpyro.plate`,
  `pyro.plate`,
  `filldist`,
  `@trace`,
  `map` over `iota`, `repeat`)
  to express batch dimensions.
* **Graphical-model relational** (PyMC, Edward2, BUGS, JAGS).
  PyMC and Edward2 use named-distribution constructors with
  `dims=(...)` / `sample_shape=[...]` carrying the batch shape;
  BUGS and JAGS use `for (m in 1:N) { name[m] ~ d<family>(args)
  }` row-loops. The BUGS Normal mean+scale → mean+precision
  conversion (`tau = 1 / (scale * scale)`) lives in
  `FAMILY_META.arg_aliases["bugs"]` plus a renderer-internal
  arithmetic-transform table keyed on the alias target name.

A `score` step, and a called helper's scored weight, is a term of
the joint on every trace-based and graphical target: Stan's
`target +=`, NumPyro's and Pyro's `factor`, PyMC's `Potential`,
Turing's `@addlogprob!`, WebPPL's and Church's `factor`, and on
Edward2 and Gen, which have no factor primitive of their own, a
traced choice of a one-point distribution whose log density is the
weight. Gen traces such choices under the `:qvr_factor` address
namespace, and its marginalize emission is the Turing renderer's,
writing into the Gen body with the reduced weight traced the same
way; BUGS and JAGS write a score through the zeros trick.

Each backend's renderer is roughly one file of 700 to 1400 lines.
The Gen renderer reuses the Turing renderer's marginalize emission,
since both emit Julia over Distributions.jl; no other imports from
another.

## 6. LDA end-to-end

Latent Dirichlet Allocation (LDA) supplies several axes in one example:
documents, topics, and words. `theta` assigns topic weights to documents;
`phi` assigns word weights to topics. The finite latent `z` selects a topic,
and `word_idx` maps observed rows to documents. Follow one observation
through the pipeline: the target must use its mapped document, evaluate the
possible topic contributions, and combine them with the grouping prescribed
by the source. The axes and `via` map thus carry model meaning, not
only array-layout information.

The canonical Latent Dirichlet Allocation source:

```qvr
object Doc : FinSet 20
object Topic : FinSet 3
object Word : FinSet 200
program lda(alpha : Real, beta : Real) : Word -> Word
    sample theta : Doc <- Dirichlet(concentration=alpha) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(concentration=beta) [over=Word, iid_over=Topic]
    marginalize z : Topic <- Categorical(probs=theta) [over=Doc, reduction=logsumexp]
        observe w : Word <- Categorical(probs=phi[z]) [via=word_idx]
    return theta
```

After `Lower`, the IR carries:

* Inputs: `alpha` and `beta` as `IRDataInput`s with `CSReal()`
  constraints; `word_idx` and `w` as `IRDataInput`s with
  `IntegerInterval` constraints and `DimDynamic(size_name="N_w")`
  batch dimensions.
* Body: `IRSample(theta)` with `support=CSSimplex(event_dim=3)`
  and `plate=Plate(event_dims=(DimStatic(3, "Topic"),),
  batch_dims=(DimStatic(20, "Doc"),))`. Its only arg is
  `IRArgBroadcast(value=IRArgRef("alpha"), target_shape=(3,))`.
* `IRSample(phi)` analogous, transposed dims.
* `IRMarginalize(z)` with `args=(IRArgRef("theta"),)`, scope
  containing one `IRObserve(w)` whose args are
  `IRArgRef("phi", indices=(IRArgRef("z"),))` and whose
  `via="word_idx"`.
* `IRReturn(names=("theta",))`.

`StanRenderer.render` produces:

```stan
data {
  real alpha;
  real beta;
  int N_w;
  array[N_w] int<lower=1, upper=20> word_idx;
  array[N_w] int<lower=1, upper=200> w;
}
parameters {
  array[20] simplex[3] theta;
  array[3] simplex[200] phi;
}
model {
  for (m_Doc in 1:20)
    theta[m_Doc] ~ dirichlet(rep_vector(alpha, 3));
  for (m_Topic in 1:3)
    phi[m_Topic] ~ dirichlet(rep_vector(beta, 200));
  {
    array[20] vector[3] lps_z;
    for (g_Doc in 1:20)
      for (k in 1:3)
        lps_z[g_Doc, k] = categorical_lpmf(k | theta[g_Doc]);
    for (n in 1:N_w)
      for (k in 1:3)
        lps_z[word_idx[n], k] += categorical_lpmf(w[n] | phi[k]);
    for (g_Doc in 1:20)
      target += log_sum_exp(lps_z[g_Doc]);
  }
}
generated quantities {
  array[20] simplex[3] theta_value = theta;
}
```

`NumPyroRenderer.render` produces:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions

def model(alpha, beta, word_idx, w=None):
    with numpyro.plate("Doc", 20):
        theta = numpyro.sample(
            "theta",
            numpyro.distributions.Dirichlet(jnp.full((3,), alpha)),
        )
    with numpyro.plate("Topic", 3):
        phi = numpyro.sample(
            "phi",
            numpyro.distributions.Dirichlet(jnp.full((200,), beta)),
        )
    with numpyro.plate("Doc_z", 20):
        z = numpyro.sample(
            "z",
            numpyro.distributions.Categorical(theta),
        )
    with numpyro.plate("Word_obs", w.shape[0]):
        numpyro.sample(
            "w",
            numpyro.distributions.Categorical(phi[z[word_idx]]),
            obs=w,
        )
    return theta
```

Same IR, different renderer. The Stan renderer's `marginalize`
emits the `log_sum_exp` enumeration; the NumPyro renderer's
`marginalize` lowers the construct to `IRSample(z) + scope` and
the scope's `IRObserve(w)` becomes a `numpyro.sample(..., obs=w)`
inside a per-word
`plate`. Neither renderer's code
references the family name `Dirichlet` or `Categorical`; both
dispatch on `is_real_simplex` (for the Dirichlet declaration) and
`is_int_category` (for the Categorical observation).

## 7. Adding a new family

1. Implement a [`ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism]
   subclass in
   [`src/quivers/continuous/families.py`][quivers.continuous.families]
   (or a new file under `src/quivers/continuous/` if the family
   has its own structural shape, like the cutpoint-parameterized
   ordered families in `src/quivers/continuous/ordered.py`).
2. Add a
   `FamilyMeta`
   entry to `FAMILY_META`. Populate `qvr_name`,
   `distribution_class` (the torch class or a thin shim
   exposing the right `arg_constraints` and `.support`),
   `quivers_class`, `target_names`, and `arg_aliases`.

Every backend's renderer picks the new family up automatically
via the constraint-predicate dispatch on the family's torch
`.support`. No per-backend edit is needed unless the family
requires a backend-specific arithmetic transform or wrapper
shape.

## 8. Adding a new backend

1. Choose the target tree-sitter grammar (`stan`, `python`,
   `julia`, `scheme`, `javascript`, `bugs`, `jags`).
2. Implement a
   `RendererBase`
   subclass under `src/quivers/transpile/renderers/<backend>.py`.
   Override `declare`, `sample`, `marginalize`, and `broadcast`.
3. Add a `target_names[<backend>] = ...` entry to every
   `FamilyMeta` in `FAMILY_META` for the families the backend
   supports. Omit the entry for unsupported families; the
   renderer's call-site lookup raises
   `UnsupportedConstruct`
   with a precise kind.
4. Register the renderer in `src/quivers/transpile/__init__.py`'s
   `_RENDERERS` table, with the appropriate grammar string.

The IR walk, `FAMILY_META` consultation, and constraint-
predicate dispatch are inherited from `RendererBase`. A typical
backend implementation is one file, 700 to 1400 lines, with no
imports from any other backend's renderer.

## 9. The five rules

The architecture enforces five structural invariants:

1. **Single source of truth per concept.** Family metadata
   (event rank, support, argument constraints, per-target
   distribution name, argument aliases) lives in one place.
   Walkers query it; they never duplicate or override.
2. **No `if family == "X"` in any renderer.** Renderer behavior
   dispatches on the support predicates of §2.3 and on
   `FAMILY_META.target_names[backend]`.
3. **No silent drops of AST fields.** Every `AxisSpec.over`,
   `AxisSpec.iid_over`, `ObserveStep.via`,
   `MarginalizeStep.reduction`, and `MarginalizeStep.scope` is
   consumed by Lower or raised on by a renderer with a precise
   `UnsupportedConstruct` kind.
4. **Backend-symmetric abstractions.** Each renderer reads from
   the same `FAMILY_META`, the same `Lower` output, and the
   same `RendererBase` helpers. No backend is more privileged
   than another.
5. **No fallbacks, no placeholders.** When a renderer cannot
   lower a construct, it raises `UnsupportedConstruct` with a
   precise kind. Never emits `__placeholder__` or "tracked
   later" or broken code.

The IR shape, the `FamilyMeta` schema, and the `Renderer`
Protocol jointly make these invariants structural: a renderer
that violates one of them produces a schema that fails the
structural matrix test, or fails to compile against the
`Renderer` Protocol, or raises a typed `UnsupportedConstruct`
rather than emitting wrong bytes.

## References

* [Transpilation correctness](transpile-correctness/index.md). The
  supported-program contract and the structural, syntax, and numerical
  evidence used to check it.
* [QVR programs](programs.md). The source-language `Program`
  structure that `Lower` consumes.
* [Continuous families](../guides/continuous-families.md). The
  catalog of
  [`ContinuousMorphism`][quivers.continuous.morphisms.ContinuousMorphism]
  subclasses
  ([`ConditionalNormal`][quivers.continuous.families.ConditionalNormal],
  [`ConditionalDirichlet`][quivers.continuous.families.ConditionalDirichlet],
  [`ConditionalBetaBinomial`][quivers.continuous.families.ConditionalBetaBinomial],
  ...) the inference layer instantiates at run time.

The backend papers explain the language designs that the renderers target:
Stan [@carpenter2017stan], NumPyro [@phan2019numpyro], Pyro
[@bingham2019pyro], Turing [@ge2018turing], Gen
[@cusumanotowner2019gen], and Church [@goodman2008church]. For an
implementation-oriented introduction to execution traces and inference, see
the author-published textbook [@goodman2014design]. For worked Bayesian models
using JAGS and Stan, see [@kruschke2014doing]. These sources supply background;
current backend behavior is documented on each target’s correctness page.
