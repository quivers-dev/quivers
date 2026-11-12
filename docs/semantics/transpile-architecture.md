# Transpilation architecture

The package `quivers.transpile` translates a QVR module $M$ into source text for another probabilistic
programming language $\mathsf T$: Stan, NumPyro, Pyro, PyMC, Edward2, Turing.jl, Gen.jl, Church,
WebPPL, BUGS, or JAGS. A translation must keep the model's factors, shapes, and supports while the
notation changes. A Normal with `scale=2` may become a constructor that takes a standard deviation of
$2$ or one that takes a precision of $1/2^2 = 0.25$, and copying the number without converting it
would change the model. The pipeline below keeps meaning and spelling apart: an intermediate
representation (IR) records what each statement means, and a renderer for each target decides how to
write it. What the evidence for a correct translation is, and how it is checked, is on
[Transpilation correctness](transpile-correctness/index.md).

## 1. The pipeline {#1-the-shared-pipeline}

$$
\mathrm{Module}
\xrightarrow{\ \mathsf{Lower} = \mathsf{Check} ; \mathsf{Plan}\ }
\mathrm{IRProgram}
\xrightarrow{\ \mathsf{Render}_{\mathsf{T}}[\mathsf{AnalyzeQIEC}_{\mathsf{T}}]\ }
\mathrm{panproto.Schema}
\xrightarrow{\ \mathsf{Pretty}_{\mathsf{T}}\ }
\mathrm{bytes}
$$

- **`Check`** elaborates the module to the Quivers Indexed Effect Core (QIEC): it compiles the indexed
  family declarations, elaborates every entry-point program to a named computation, and checks the whole
  module ([Typing and checking](typing.md)). It does not depend on the target.
- **`Plan`** reads each program's computation and recognizes the statements a probabilistic programming
  language has: an `IRSample` for each `Random.sample` request, an `IRObserve` for each density scored at
  an observation, an `IRDeterministic` for each pure binding, an `IRScore` for each other weight, and an
  `IRMarginalize` for each summed-out latent variable. Each node carries the support and plate shape read
  from the module's types, from `FAMILY_META`, and from `arg_constraints` of the PyTorch distribution
  class. The inputs of the plan are the parameters of the entry point. The result is one `IRProgram`,
  which also holds the whole checked module as a typed tree, `IRQiecModule`. A module with no program
  lowers to its module and an empty plan. `Plan` does not depend on the target either.
- **`Render[T]`** is one class for each target (`StanRenderer`, `NumPyroRenderer`, `PyMCRenderer`, and so
  on). It turns the plan into a [panproto](https://github.com/panproto/panproto) schema of the target's
  syntax tree, using the support predicates of §2.3 and the entries of `FAMILY_META`, and then renders
  the module's other computations. Before it places a computation, **`AnalyzeQIEC[T]`** lists the QIEC
  features the computation uses (open or effectful rows, static polymorphism, indexed cases, transport,
  attachments, resumption grades) and compares them with the features the target supports. A missing
  feature stops the rendering with a diagnostic `qiec:capability:<feature>:<computation>` that keeps the
  computation's source location.
- **`Pretty[T]`** is `panproto.AstParserRegistry.emit_pretty` for the target's
  [tree-sitter](https://tree-sitter.github.io/) grammar. It prints the schema as source text.

The compiler of the runtime checks the same module and attaches its `QiecModule` to the compiled
`Program`, but transpilation does not use that object. `transpile()` classifies the source, and
`Lower.forward()` elaborates and checks the syntax tree itself, so a direct call of `Lower` is checked
the same way. The analysis of target features happens inside the renderer.

## 2. The IR {#2-the-ir}

The IR is defined in [`quivers.transpile.ir`](../api/transpile/ir.md). Every entry is a didactic
`Model` or `TaggedUnion`, and the IR holds structure only: no strings of a target language, no schema
vertices, and no panproto types.

### 2.1 The typed QIEC module

`IRProgram.module` is an `IRQiecModule`, a structural copy of the checked module without loss: every
declaration, stable identifier, source location, row and `lacks` constraint, value, equality witness,
transport, computation node, handler clause, and resumption grade has its own typed representation, so
a renderer never parses a serialized string.

Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church run this tree through an implementation of
the QIEC runtime written for each target: free computations, handlers bound to instances, stable
addresses of requests, constructors, case analysis on indexed families, attachments, and checks of
resumption grades. Tests parse the source emitted for every target and, where the target's runtime is
installed, compare its requests and results with the [reference machine](../reference/qvr/index.md).

Stan, BUGS, and JAGS accept a static subset, which the analysis checks: each computation has a closed,
empty row and no static parameters, returns a scalar, and uses only scalar `Return` and `Bind`. Stan
admits value parameters of types `Bool`, `Int`, and `Real`. BUGS and JAGS require computations without
parameters, since their models have no callable functions with parameters. Every other QIEC feature gets
a capability diagnostic, and no renderer treats an executable term as metadata it may ignore.

### 2.2 `Plate`: event and batch axes

Ten draws from a three-dimensional multivariate Normal have batch shape $(10)$ and event shape $(3)$
([event and batch axes](morphisms.md#6-axis-role-specifications)) [@pyroTensorShapes]. A target may write
them as one vectorized draw or as ten iterations of a loop, but it must keep the three coordinates of
each draw joint: with a covariance that is not diagonal, the density has terms that couple coordinates,
which three scalar densities would lose. A draw's plate records the two kinds of axes:

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

`event_dims` comes from the draw's `over` axes and `batch_dims` from its `iid_over` axes, in source
order. A renderer writes the batch axes in the target's own form: nested `for` loops in Stan, nested
`plate` contexts in NumPyro and Pyro, `dims=(...)` in PyMC, and `filldist` or `arraydist` in Turing and
Gen.

### 2.3 Classifying supports

`quivers.transpile.ir` defines predicates on the support constraints of PyTorch
(`torch.distributions.constraints.Constraint`):

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

A renderer decides how to declare a value only through these predicates, never by testing the class of a
constraint or the name of a family, so a new kind of support needs one new predicate. A PyTorch
constraint does not survive didactic's encoding and decoding, so the IR stores a structural copy,
`ConstraintSpec`, with one variant for each kind the predicates tell apart. `from_constraint` builds it
when lowering, and `.to_constraint()` rebuilds the PyTorch constraint when a renderer needs it.

### 2.4 `IRArg`: arguments as a typed tree {#24-irarg-typed-argument-tree}

The parser leaves indexed arguments as strings such as `"phi[z]"`. `Lower` parses them into a tree, so
that no renderer parses strings:

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

The IR also has `IRArgKernel` for a kernel matrix given as an argument. `Lower` wraps a scalar argument in
`IRArgBroadcast` when the argument's constraint is `IndependentConstraint(base, n)` with $n \ge 1$, that is,
when the family expects a vector there. Each renderer writes the broadcast in its target:
`rep_vector(x, K)` in Stan, `jnp.full((K,), x)` in NumPyro, `torch.full((K,), x)` in Pyro,
`np.full((K,), x)` in PyMC, `fill(x, K)` in Turing and Gen, `repeat(K, function() { return x; })` in
WebPPL, and `(make-list K x)` in Church.

### 2.5 `IRNode`: the statements of a program

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

class IRProgram(dx.Model):
    name: str
    inputs: tuple[IRDataInput, ...]
    body: tuple[IRNode, ...]
    module: IRQiecModule
    cards: dict[str, int]
```

`arg_names` gives, for each entry of `args`, its keyword name in PyTorch's `arg_constraints` (`"loc"`,
`"scale"`, `"concentration"`, ...). NumPyro, Pyro, PyMC, and Edward2 call families by keyword and read
it, and Stan, BUGS, and JAGS call positionally and ignore it. An `IRCall` is a call of a computation that
the plan did not inline: the dynamic targets call the callee's generated entry point with the program's
`Random` and `Score` instances bound to the target's own draw and factor primitives, Stan calls it as a
user-defined function when it can define one, and BUGS and JAGS refuse it.

## 3. `FamilyMeta`: facts for translation {#3-familymeta-the-registry-for-transpile-only-facts}

A family name such as `Normal` identifies a family. A target may call its location `loc` or `mu`, and its
spread a scale or a precision, and the registry records these translations. The shapes and supports of
arguments belong to the typed distribution. The renderer needs both to write the intended density.
The registry is `FAMILY_META` in [`quivers.transpile.family_meta`](../api/transpile/family_meta.md):

```python
class FamilyMeta(dx.Model):
    qvr_name: str
    distribution_class: type[Distribution]
    target_names: dict[str, str]
    arg_aliases: dict[str, dict[str, str]]
    quivers_class: type[ContinuousMorphism] | None
    structured_lowering: StructuredSampleLowering | None
    event_rank: int
```

- `qvr_name` is the family's name in QVR (`"Normal"`, `"Dirichlet"`).
- `distribution_class` is the PyTorch `Distribution` subclass, or a small class that gives the right
  `arg_constraints` and `support` for a family PyTorch lacks, such as `OrderedLogistic` and
  `HalfStudentT`. It is the source of the family's argument constraints and support.
- `target_names` maps each target to the family's name there. Renderers read
  `FAMILY_META[family].target_names[target]` and keep no table of their own.
- `arg_aliases` renames arguments for a target. Renderers that convert a parameterization, such as the
  mean and precision of a BUGS or JAGS Normal, or PyMC's `a` for the Dirichlet concentration, key the
  arithmetic on the alias.
- `quivers_class` is the [`ContinuousMorphism`](../api/continuous/morphisms.md) subclass the runtime
  builds, such as `ConditionalNormal`, and is empty for a family that wraps another morphism.
- `event_rank` is the number of event dimensions.

`finite_enumerable_at_call_site(family_meta, args)` answers whether a `marginalize`'s index object is the
family's support, so that summing over it removes the axis. It is true for `Bernoulli`, `Categorical`,
`OrderedLogistic`, and `OrderedProbit`, and for `Binomial` when its `total_count` is a literal number.

The registry does not hold the shapes and constraints of arguments, which `Lower` reads from
`distribution_class.arg_constraints`, or the support, which comes from `distribution_class.support`,
evaluated on fixed parameters for a support that depends on them, as for `Uniform(low, high)`. A new
family needs one `ContinuousMorphism` subclass and one `FamilyMeta` entry, and no renderer changes.

## 4. `Lower`: from a module to the IR {#4-lower-module-ir}

[Program elaboration](typing.md#8-program-elaboration) gives the checked computation that lowering reads,
and the [transpilation pipeline](transpile-correctness/index.md#1-pipeline) the comparisons that test this
stage. `Lower`, in [`quivers.transpile.plan`](../api/transpile/plan.md), implements
`dx.Mapping[Module, IRProgram]`. Its `forward`:

1. elaborates and checks the module, so that no direct call bypasses checking, refuses a program that has
   no computation under the diagnostic kind the elaboration reports, and lowers a module without a program
   to its `IRQiecModule` with an empty plan;
2. expands composite `let` bindings, so that each step of the plan corresponds to one step of the expanded
   source, whose morphism fixes the values a categorical draw ranges over and the form of a structured
   family's arguments;
3. walks the computation: a `Random.sample` becomes an `IRSample` named by its site label, a `Score.add` of
   a density at an observation becomes an `IRObserve` whose `via` is the index map by which a `SegmentSum`
   regroups the density, a pure binding becomes an `IRDeterministic`, or is absorbed into an `IRScore` when
   the next request scores its value, the call of a helper for summing out a latent variable, whose result is
   scored, becomes an `IRMarginalize` whose scope is the helper's body, and the final `Return` becomes an
   `IRReturn`;
4. reads each family, plate, and argument from its `DistributionValue`: a literal, a binding, an indexed
   binding, a list, a matrix, a broadcast scalar as `IRArgBroadcast`, a kernel matrix as `IRArgKernel`, and
   the output of a parameter map as a reference to the `IRDeterministic` bindings that compute it, in the
   order PyTorch's constructor takes them, with the support resolved through `FAMILY_META` and narrowed by
   the declared bounds of the site's axes;
5. reads pure values back as `let` expressions: primitives as their operators and builtins, gathers as
   subscripts, comprehensions as factor expressions, and last-axis reductions as builtin calls. A shared
   [`LetValue`](../api/qiec/terms.md) becomes a deterministic statement of its own, under a fresh name,
   before the statement that reads it, and a binding inside a comprehension is moved out as a comprehension
   of its value over the same axis, read at the index, so each shared value is computed once;
6. derives the inputs from the parameters of the entry point, typed by the module and named by the axes the
   elaboration recorded, grouped as the targets declare them: the domain, scalars, index maps, observations,
   parameter maps, and data;
7. spells every name for the targets. A draw from a program runs the program in place under names of the
   caller: a local `z` of a program drawn as `theta` is the site `theta$z` on the reference machine and the
   compiled runtime, and since no target admits `$` in a name, the plan spells it `theta__z`.

`Lower` does not depend on the target and imports no renderer.

## 5. `Renderer[T]`: from the IR to a schema {#5-renderert-ir-panprotoschema}

Each target implements the `Renderer` protocol, with one public method and four points where targets
differ:

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

`BlockKind` says where a declaration goes (`"data"`, `"parameters"`, `"transformed_parameters"`,
`"generated_quantities"`, `"function_body"`), and each target reads it in its own structure: Stan has these
blocks, NumPyro's block is the body of the model function, PyMC's is the `with pymc.Model() as model:`
scope, and BUGS and JAGS have one `model { ... }`.

`RendererBase` walks the IR: an `IRDataInput` is declared, an `IRSample` that is not observed is declared and
sampled, an `IRObserve` is declared and sampled with `observed=True`, an `IRDeterministic` is declared and
assigned, an `IRScore` adds to the log density, an `IRMarginalize` goes to `marginalize`, and an `IRReturn`
becomes the target's return. It also provides the substitution of indices that `sample` and `marginalize`
share, and `marginal_atoms`, the values of the latent at which every target's `marginalize` scores one copy
of the scope.

`declare` decides by the predicates of §2.3. For Stan:

| Predicate | Event | Batch | Declaration |
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
| `is_int_category(c)` | () | () | `int<lower=0, upper=K-1> <name>;` |
| `is_int_count(c)` | () | () | `int<lower=0> <name>;` |

The other targets have tables of the same form, from a predicate, event axes, and batch axes to a
declaration, and no row names a family.

### 5.1 Kinds of target

Van de Meent and coauthors distinguish first-order languages, whose programs have a fixed finite graph of
random variables, from higher-order languages, which evaluate arbitrary programs and record the random
choices of each run [@vandemeent2018introduction]. The eleven targets fall into three groups by how a model
is written:

- **Blocks with static types** (Stan [@carpenter2017stan]): `data { ... } parameters { ... } model { ... }
  generated quantities { ... }`, with declarations placed in the schema vertices of each block.
- **Model functions that record a trace** (NumPyro [@phan2019numpyro], Pyro [@bingham2019pyro], Turing.jl
  [@ge2018turing], Gen.jl [@cusumanotowner2019gen], Church [@goodman2008church], WebPPL
  [@goodman2014design]): the renderer writes a model function (`def model(...)`, `@model function`,
  `(define (model ...))`) and expresses batch axes with the target's own form (`numpyro.plate`,
  `pyro.plate`, `filldist`, `@trace`, `map` over `iota`, `repeat`).
- **Declared graphs of random variables** (PyMC [@salvatier2016pymc3], Edward2 [@tran2018simple], BUGS
  [@lunn2009bugs], JAGS [@plummer2003jags]): PyMC and Edward2 call named distribution constructors with the
  batch shape in `dims=(...)` or `sample_shape=[...]`, and BUGS and JAGS loop over rows,
  `for (m in 1:N) { name[m] ~ d<family>(args) }`. The conversion of a Normal's scale to the precision BUGS
  takes, `tau = 1 / (scale * scale)`, is the alias in `FAMILY_META.arg_aliases["bugs"]` with a table of
  arithmetic in the renderer.

A `score` step, and the scored weight of a called helper, is a term of the log joint density on every
target: Stan's `target +=`, NumPyro's and Pyro's `factor`, PyMC's `Potential`, Turing's `@addlogprob!`,
and WebPPL's and Church's `factor`. Edward2 and Gen have no factor primitive, and on them the weight is a
traced choice of a distribution with one point whose log density is the weight. Gen traces such choices
under the address `:qvr_factor`, and its `marginalize` reuses the Turing renderer's, since both write Julia
with Distributions.jl. BUGS and JAGS add a weight with the zeros trick, observing a zero from a Poisson
distribution whose rate carries the weight [@lunn2012bugs]. No other renderer imports another.

## 6. An example from source to Stan and NumPyro

The gallery source `docs/examples/source/lda.qvr` has documents, topics, and a vocabulary. `theta` gives
each document its topic weights, `phi` gives each topic its word weights, `z` is a summed-out topic, and
`word_idx` is the index map from tokens to documents:

```qvr
composition log_prob [level=algebra]

object Doc : FinSet 20
object Topic : FinSet 3
object Vocab : FinSet 50
object Token : FinSet 200
object Mix : Real 3

program lda(alpha : Real, beta : Real) : Token -> Mix
    sample theta : Doc <- Dirichlet(concentration=alpha) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(concentration=beta) [over=Vocab, iid_over=Topic]

    marginalize z : Topic <- Categorical(probs=theta) [over=Doc, reduction=logsumexp]
        observe w : Token <- Categorical(probs=phi[z]) [via=word_idx]

    return theta

export lda
```

The grouping `over=Doc` states one topic per document, shared by its tokens, a mixture of unigrams
[@nigam2000text]. Since the prior `theta` is a plate over `Doc`, the evaluators instead give each token its
own topic drawn from its document's weights ([Programs as kernels §2.7](programs.md#27-grouped-marginalize-with-multi-observe-fibration)),
which is the per-token topic of latent Dirichlet allocation (LDA) [@blei2003lda]. The output below shows it.

After `Lower`, the IR has:

- inputs `alpha` and `beta`, real scalars, and `word_idx` and `w`, integer arrays over `Token`;
- `IRSample(theta)` with a simplex support and the plate `Plate(event_dims=(Topic,), batch_dims=(Doc,))`,
  whose one argument is `IRArgBroadcast(value=IRArgRef("alpha"), target_shape=(3,))`, and `IRSample(phi)`
  likewise over `Vocab` and `Topic`;
- `IRMarginalize(z)` with argument `IRArgRef("theta")` and a scope with one `IRObserve(w)`, whose argument is
  `IRArgRef("phi", indices=(IRArgRef("z"),))` and whose `via` is `"word_idx"`;
- `IRReturn(names=("theta",))`.

`StanRenderer` writes:

```stan
data {
  real alpha; real beta; array [200] int <lower = 0 , upper = 199> word_idx; array [200] int <lower = 0 , upper = 49> w;
} parameters {
  array [20] simplex [3] theta; array [3] simplex [50] phi;
} model {
  for (m_Doc in 1 : 20) {
    target +=dirichlet_lpdf(theta[m_Doc] | rep_vector(alpha,3));
  } for (m_Topic in 1 : 3) {
    target +=dirichlet_lpdf(phi[m_Topic] | rep_vector(beta,50));
  } {
    array [200] vector [3] lps_z = rep_array(rep_vector(0,3),200); for (n_Token in 1 : 200) {
      for (k in 1 : 3) {
        lps_z[n_Token,k] = categorical_lpmf(k | theta[word_idx[n_Token]+1]);
      }
    } for (n_Token in 1 : 200) {
      for (k in 1 : 3) {
        lps_z[n_Token,k] += categorical_lpmf(w[n_Token]+1 | phi[k]);
      }
    } for (n_Token in 1 : 200) {
      target +=log_sum_exp(lps_z[n_Token]);
    }
  }
}generated quantities{
  array [20] simplex [3] theta_value = theta;
}
```

and `NumPyroRenderer` writes:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions
import jax.scipy.special as jsp
def model(alpha,beta,word_idx,w=None) :
  with numpyro.plate("Doc" ,20) :
    theta = numpyro.sample("theta" ,numpyro.distributions.Dirichlet(concentration=jnp.full((3,),alpha)))
  with numpyro.plate("Topic" ,3) :
    phi = numpyro.sample("phi" ,numpyro.distributions.Dirichlet(concentration=jnp.full((50,),beta)))
  __marg_z_0 = numpyro.distributions.Categorical(probs=phi[0]).log_prob(w)
  __marg_z_1 = numpyro.distributions.Categorical(probs=phi[1]).log_prob(w)
  __marg_z_2 = numpyro.distributions.Categorical(probs=phi[2]).log_prob(w)
  __marg_z_w = jnp.log(theta[word_idx])
  __marg_z = jnp.stack([__marg_z_0,__marg_z_1,__marg_z_2],axis=- 1)
  numpyro.factor("z" ,jnp.sum(jsp.logsumexp(__marg_z_w+__marg_z,axis=- 1)))
  return theta
```

Both sum the topic out for each token: Stan by a table `lps_z` and `log_sum_exp`, NumPyro by the stacked log
densities at each topic, `logsumexp`, and `numpyro.factor`, so no sample site remains for `z`. Both keep the
zero-based codes of the data: Stan declares `w` with `lower = 0` and adds one where it indexes. Neither
renderer names `Dirichlet` or `Categorical` in its code: each declares through `is_real_simplex` and
`is_int_category`. Stan bounds `word_idx` by `Token` (`upper = 199`), not by `Doc` (`upper = 19`).

## 7. Adding a family {#7-adding-a-new-family}

1. Write a [`ContinuousMorphism`](../api/continuous/morphisms.md) subclass in
   [`quivers.continuous.families`](../api/continuous/families.md), or in a new module of
   `quivers.continuous` when the family has a structure of its own, as the ordered families with cutpoints
   do.
2. Add an entry to `FAMILY_META` with `qvr_name`, `distribution_class` (the PyTorch class, or a small class
   with the right `arg_constraints` and `support`), `quivers_class`, `target_names`, and `arg_aliases`.

Every renderer then handles the family through the predicates on its support. A target needs its own change
only when the family needs a conversion of its parameters or a wrapper there.

## 8. Adding a target {#8-adding-a-new-backend}

1. Choose the target's tree-sitter grammar (`stan`, `python`, `julia`, `scheme`, `javascript`, `bugs`,
   `jags`).
2. Write a `RendererBase` subclass in `quivers.transpile.renderers.<target>` that overrides `declare`,
   `sample`, `marginalize`, and `broadcast`.
3. Add `target_names[<target>]` to each `FamilyMeta` entry whose family the target supports. A family without
   one is refused at its use with `UnsupportedConstruct` and a precise kind.
4. Register the renderer, with its grammar, in the table of renderers of `quivers.transpile`.

The walk of the IR, the use of `FAMILY_META`, and the dispatch on predicates are inherited from
`RendererBase`, and no renderer imports another, except Gen's use of Turing's `marginalize`.

## 9. Rules the architecture keeps

1. **One place for each fact.** The facts about a family (number of event dimensions, support, argument
   constraints, name and argument names for each target) are stored once, and renderers read them there.
2. **No test of a family's name in a renderer.** A renderer decides by the predicates of §2.3 and by
   `FAMILY_META.target_names`.
3. **No field of the syntax tree is dropped.** Each draw's `over` and `iid_over`, each observation's `via`,
   and each `marginalize`'s reduction and scope is used by `Lower` or refused by a renderer with a precise
   `UnsupportedConstruct` kind.
4. **The same interface for every target.** Every renderer reads the same `FAMILY_META`, the same output of
   `Lower`, and the same helpers of `RendererBase`.
5. **A refusal, not a guess.** A renderer that cannot write a construct raises `UnsupportedConstruct` with a
   precise kind and does not emit placeholder or approximate code.

A renderer that breaks one of these either produces a schema that the structural tests reject, fails the
`Renderer` protocol, or raises `UnsupportedConstruct`.
