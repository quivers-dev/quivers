# Quickstart

This guide walks through the core concepts of quivers with concrete examples: creating objects, building morphisms, composing them, and running them as differentiable programs.

If you want to fit a statistical model first, begin with the
[first regression tutorial](../tutorials/qvr/01-first-model.md). This quickstart
instead introduces the Python building blocks. The
[conceptual introduction](concepts.md) supplies the probability and composition
background used by both routes.

Read a finite morphism as a table with named row and column sets. A table from
positions to colors says how each position relates to each color. Composing
it with a table from colors to outcomes produces a table from positions to
outcomes by combining the routes through each possible color. The names
identify compatible connections; the [algebra](../guides/core.md#algebras-enrichment-algebras) specifies the arithmetic.
For this composition to satisfy the laws of an enriched category, its algebra
must satisfy the corresponding preservation and associativity requirements
[@kelly1982basic].

## 1. Basic Morphisms

A **morphism** is a $\mathcal{V}$-enriched relation represented by a tensor.
The selected composition algebra determines how its entries combine and how a
[shared axis is reduced](../tutorials/python/01-first-quiver.md#composition).

Create finite sets:

```python
from quivers import FinSet, morphism, observed
import torch

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
```

Create a **latent** (learnable) morphism from X to Y. Its tensor entries are parameters:

```python
f = morphism(X, Y)
print(f.tensor.shape)  # torch.Size([3, 4])
print(f.domain.name, "->", f.codomain.name)  # X -> Y
```

Create an **observed** morphism from a fixed tensor:

```python
data = torch.tensor([
    [1.0, 0.5, 0.0, 0.2],
    [0.0, 1.0, 0.8, 0.1],
    [0.3, 0.2, 1.0, 0.9],
])
g = observed(X, Y, data)
print(g.tensor)  # your data tensor
```

Get the underlying `nn.Module` for use in training:

```python
mod = f.module()
params = list(mod.parameters())
print(len(params))  # 1 parameter matrix
```

## 2. Composition

Compose morphisms with the `>>` operator. The domain of the second must match the codomain of the first:

```python
Z = FinSet(name="Z", cardinality=2)
h = morphism(Y, Z)

# Compose f: X -> Y and h: Y -> Z to get X -> Z
composed = f >> h
print(composed.tensor.shape)  # torch.Size([3, 2])
```

Composition is **lazy**: it builds a computation graph. The final tensor is only materialized on evaluation.

For one position and one outcome, suppose two color routes contribute
`0.4` and `0.3`. The default product-fuzzy rule combines them as
`1 - (1 - 0.4) * (1 - 0.3) = 0.58`. Ordinary matrix multiplication would
instead add them to `0.7`. Thus matching tensor shapes is only part of
composition: you also need to know what the entries mean.

## 3. Programs

Wrap a morphism as a differentiable `nn.Module` with `Program`:

```python
from quivers import FinSet, morphism, Program

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=2)
composed = morphism(X, Y) >> morphism(Y, Z)

program = Program(composed)
output = program()
print(output.shape)  # torch.Size([3, 2])
```

Integrate with a training loop:

```python
import torch.optim as optim

optimizer = optim.Adam(program.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    output = program()
    # Define a loss and backpropagate
    loss = output.sum()
    loss.backward()
    optimizer.step()
```

Here `latent` identifies tensor entries that the optimizer can adjust.
The displayed loss, `output.sum()`, pushes that sum downward; a fitting task
would replace it with a discrepancy between predictions and observations.
The next section adds probability distributions and observation likelihoods.

## 4. Monadic Programs (Continuous)

For probabilistic computations with continuous and discrete variables, use [**monadic programs**](../api/continuous/programs.md). A program is a tuple `(domain, codomain, steps, return_vars)` where each step binds a sample, observation, or let-expression into a named variable; the final `return_vars` names the variables returned in the program's output tuple.

```python
from quivers import FinSet
from quivers.continuous.spaces import Euclidean
from quivers.continuous.families import ConditionalNormal
from quivers.continuous import Draw, MonadicProgram
import torch

# Define spaces.
X = FinSet(name="context", cardinality=3)
R = Euclidean(name="response", dim=2)

# Conditional normal family: each input x ∈ X yields a 2D normal on R.
family = ConditionalNormal(X, R)

# A one-step program: bind y ∼ family(x), return y.
program = MonadicProgram(
    domain=X,
    codomain=R,
    steps=[Draw(names=("y",), morphism=family)],
    return_vars=("y",),
)

x = torch.tensor([0, 1, 2])
output = program.rsample(x)        # reparameterized samples, shape (3, 2)
```

A sampling step produces one realized value, and a later step can use that
value as an input to its distribution. This is the operational meaning of
monadic sequencing here. An observation step instead evaluates how well a
supplied value fits its distribution. Running these steps forward produces a
draw and a score; recovering a posterior requires inference in addition.

## 5. The DSL

Write categorical programs declaratively in `.qvr` files:

```qvr
composition product_fuzzy [level=algebra]
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 2
morphism f : X -> Y [role=latent]
morphism g : Y -> Z [role=latent]
define composed = f >> g
export composed
```

Load and run:

```python
from quivers.dsl import load

prog = load("docs/examples/source/bayesian_regression.qvr")
```

Or use `loads` for inline strings:

```python
from quivers.dsl import loads

source = """
composition product_fuzzy [level=algebra]
object X : FinSet 3
object Y : FinSet 4
object Z : FinSet 2
morphism f : X -> Y [role=latent]
morphism g : Y -> Z [role=latent]
define composed = f >> g
export composed
"""

prog = loads(source)
output = prog()
print(output.shape)  # torch.Size([3, 2])
```

Supported DSL operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `>>` | composition | `f >> g` |
| `>>>` | transformation composition | `softmax(B) >>> expectation` |
| `@` | tensor product | `f @ g` |
| `.marginalize(X)` | marginalize over object | `f.marginalize(X)` |
| `.change_base(t)` | change of base under transformation `t` | `f.change_base(softmax(B))` |
| `identity(X)` | identity morphism | `morphism id : X -> X [role=observed] ~ identity(X)` |

The DSL also supports monadic probabilistic programs (`program ... [effects = [Sample, Score]]`), composition rules at four algebraic levels via `composition NAME [level=<algebra | semigroupoid | bilinear_form | rule>]`, and operadic contractions (`contraction op : ... [rule = R, wiring = "..."]`). The [QVR tutorial](../tutorials/qvr/01-first-model.md) introduces the complete language.

## 6. Stochastic Morphisms

The **FinStoch** category models Markov kernels on finite sets:

```python
from quivers import (
    stochastic, FinSet, DiscretizedNormal,
    condition, ConditionedMorphism
)
import torch

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# Create a stochastic morphism (Markov kernel)
kern = stochastic(X, Y)
print(kern.tensor.shape)  # [3, 4], rows sum to 1
print(kern.tensor.sum(dim=-1))  # all 1.0

# Condition on an observation
obs_data = torch.tensor([0.1, 0.8, 0.05, 0.05])  # P(Y)
conditioned = condition(kern, obs_data)
print(conditioned.tensor.shape)  # [3, 4]
```

## 7. Enriched Structures

This optional section points to advanced categorical structures. You can skip
it while learning to build and fit models. These constructions describe how
whole families of typed relations interact:

```python
from quivers import (
    FuzzyPowersetMonad, KleisliCategory, FinSet, morphism,
)

# Create a monad
monad = FuzzyPowersetMonad()

# Work in its Kleisli category
kleisli = KleisliCategory(monad)

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

f = morphism(X, Y)
g = morphism(Y, X)

# Kleisli composition
comp = kleisli.compose(f, g)
print(comp.tensor.shape)  # [3, 3]
```

## 8. Interactive exploration

`pip install 'quivers[repl,lsp]'` adds a REPL, a Jupyter kernel, and a
language server. All three use the compiler and type checker from the library.

### The REPL

```bash
qvr repl docs/examples/source/seq2seq.qvr
```

The four-pane Textual TUI contains an input editor, an output log, and an
environment browser whose leaves open `:info`. Watch and diagnostic strips
appear at the bottom when needed, and the status bar reports the loaded file,
active algebra, and binding counts.

GHCi-style meta-commands:

```
qvr> :type seq2seq
seq2seq :: Source * Target -> Target

qvr> :info backbone
define backbone = (encoder @ decoder) >> cross
-- declared at docs/examples/source/seq2seq.qvr:59:0

qvr> :watch backbone           # pinned re-eval after every recompile
qvr> :save my_edits.qvr        # write the live module to disk
```

`Ctrl-G` (or `Ctrl-O`, `F8`) evaluates; `Tab` cycles completions;
`Ctrl-P` opens a fuzzy command palette; the loaded file auto-reloads
on save. The full reference is in
[Interactive tools](../guides/repl-and-lsp.md).

### The language server

```bash
pip install 'quivers[lsp]'    # provides `qvr-lsp` on your PATH
```

The
[`vscode-qvr`](https://github.com/quivers-dev/quivers/tree/main/editors/vscode-qvr)
and
[`zed-extension-qvr`](https://github.com/quivers-dev/quivers/tree/main/editors/zed-extension-qvr)
extensions discover it automatically. It supports hover information (the QVR
declaration plus a collapsible AST), go-to-definition, references, document
symbols, semantic highlighting, completion, formatting, and live diagnostics.
The REPL and language server share `STYLE_TABLE`, so a symbol receives the same
color in both.

### The Jupyter kernel

```bash
qvr-kernel install --user
jupyter console --kernel quivers
```

Cell semantics match the REPL: leading-`:` lines run meta-commands,
other lines append statements to the live module.

## Next Steps

- **[Interactive tools](../guides/repl-and-lsp.md):** REPL, language server, and Jupyter kernel reference.
- **[Architecture](architecture.md):** learn the package structure and design principles.
- **[API Reference](../api/index.md):** detailed documentation of all classes and functions.
- **Guides:** core types, morphisms, categorical structures, stochastic and continuous morphisms, the DSL, and variational inference.
- **Tutorials:** end-to-end examples including probabilistic programs and inference.
