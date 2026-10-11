# 9. Debugging quivers programs

When a fit doesn't converge or a compile fails, you need to know what to look at. This chapter walks the tools for inspecting a quivers program: reading `CompileError` messages, dumping a compiled `Program`, tracing intermediate tensors, and watching per-site log-densities during SVI.

## Reading `CompileError`

Every DSL compile failure raises a [`CompileError`](../../api/dsl/compiler.md) with a `line` and `col` field pointing at the offending source position. Common error categories:

```python
from quivers.dsl import loads
from quivers.dsl.compiler._prelude import CompileError

src = """
object A : FinSet 3
program p : A -> A
    sample x <- Normal(loc=0, scale=1)  # Sample used in a Pure body
    return x

export p
"""
try:
    loads(src)
except CompileError as e:
    print(f"line {e.line}, col {e.col}: {e.args[0]}")
```

The five error classes you'll see most often:

| Category | Example trigger | Where to look |
|---|---|---|
| Effect mismatch | `[effects=[Pure]]` body contains `<-` or `observe` | Loosen the effect tag or remove the offending step |
| Free-name error | A name in the body is neither bound, declared, nor in host-data scope | Add to `observed_names` or to a `<-` / `let` |
| Algebra mismatch | The two operands of `>>` carry different algebras | Insert a `change_base` on one operand so both share an algebra |
| Shape mismatch | Tensor argument has the wrong cardinality for an object | Check object declarations against the data |
| Unknown identifier | A family name (`Norma`, typo) isn't in the prelude | Check the [family catalogue](../../api/continuous/families.md) |

## Inspecting a compiled `Program`

`loads` returns a [`Program`](../../api/program.md), which is a `torch.nn.Module` wrapping a [`MonadicProgram`](../../api/continuous/programs.md). Both are inspectable:

```python
import torch

REGRESSION_SRC = """
object Item : FinSet 100

program regression : Item -> Item
    sample sigma  <- HalfNormal(scale=1.0)
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)
    let mu = beta_0 + beta_1 * x_design
    observe y : Item <- Normal(loc=mu, scale=sigma)
    return y

export regression
"""

program = loads(REGRESSION_SRC)
print(type(program).__name__)        # Program
print(program.morphism.steps)  # the compiled step records
print(program.morphism.domain, "->", program.morphism.codomain)
for name, p in program.named_parameters():
    print(name, p.shape)

torch.manual_seed(0)
x_data = torch.randn(100)
y_data = 1.5 + 2.7 * x_data + 0.3 * torch.randn(100)
```

Each step has a `name`, an `inputs` tuple (names it depends on), and a `family` or `transform` it dispatches to. Reading this list is the fastest way to see what the compiler actually built.

## Tracing intermediate values

[`trace`](../../api/inference/trace.md) runs the program forward once and records the value and log-density at every sample site:
<!-- python: skip -->
```python
from quivers.inference import trace

x = torch.zeros(1, 1)
observations = {"x_design": x_data, "y": y_data}
tr = trace(program.morphism, x, observations)

for name, site in tr.sites.items():
    print(f"{name:12s} value.shape={tuple(site.value.shape)} "
          f"log_prob={site.log_prob.sum().item():.2f}")
```

If a log-density comes back `nan` or `-inf`, you have a numerical failure at that site. The usual causes:

- A scale parameter that drifted to zero (`HalfNormal` posterior collapsing).
- A `let` that produces invalid arguments to the next family (negative variance, out-of-simplex probabilities).
- An observation that's outside the support of the likelihood (a negative count under `Poisson`, for instance).

`trace` is the right place to confirm which.

## Watching SVI training

During the SVI loop, print per-step ELBO and gradient norms:
<!-- python: skip -->
```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

model = program.morphism
guide = AutoNormalGuide(model, observed_names={"y", "x_design"})
elbo = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)
x_tensor = torch.zeros(1, 1)

for step in range(20):                # bump to ~2000 for a real fit
    loss = svi.step(x_tensor, observations)
    if step % 5 == 0:
        total_grad = sum(
            p.grad.norm().item() ** 2
            for p in guide.parameters() if p.grad is not None
        ) ** 0.5
        print(f"step {step:4d}  ELBO={-loss:.3f}  grad_norm={total_grad:.3f}")
```

A healthy SVI run often shows an upward ELBO trend and decreasing gradient norms, but minibatch and Monte Carlo noise make neither trajectory monotonic. Failure modes:

- ELBO plateaus while posterior checks remain poor: the guide or optimizer may be inadequate. Try a richer guide ([`AutoMultivariateNormalGuide`](../../api/inference/guide.md), [`AutoIAFGuide`](../../api/inference/guide.md)) and check the learning rate.
- ELBO going to `nan`: a sample-site log-density blew up. Drop in a `trace` call at the same step and find the offending site.
- ELBO oscillating with grad_norm growing: learning rate too large. Halve it.

## NUTS diagnostics that point at root causes

[`MCMCResult`](../../api/inference/predictive.md) carries three first-class diagnostic fields:

- `result.r_hat[site]`: maximum of rank-normalised and folded split-R-hat per site.
- `result.ess[site]`: rank-normalised cross-chain bulk effective sample size per site.
- `result.total_divergences`: integer count of divergent transitions across all chains.

The mapping from observation to fix:

| Symptom | Likely cause | Fix |
|---|---|---|
| R-hat > 1.01 on one site | Chain hasn't mixed | More warmup, longer chains |
| ESS < 100 on one site | Highly correlated samples | More samples, or richer step adaptation |
| Divergences > 0 | Integrator missed posterior curvature | Raise `target_accept` to 0.99, or reparameterise |
| All chains stuck at init | Initial point in a bad region | Switch `init_strategy` to `"prior"` or supply a `"value"` dict |

The full diagnostic semantics live in the [inference guide](../../guides/inference-foundations.md).

## A debugging recipe

When a fit is misbehaving, follow this order:

1. Run `trace(model, inputs, observations)` and confirm no site returns a `nan` or `-inf` log-density.
2. Print `program.morphism.steps` and verify the compiled steps match what you wrote.
3. Run a short SVI for 100 steps with prints every 10 steps; the ELBO trajectory tells you whether the guide can fit at all.
4. If SVI looks reasonable but the posterior means are off, switch to NUTS and check `result.r_hat` and `result.total_divergences`.
5. If NUTS reports divergences on a hierarchical model, reparameterise centered to non-centered ([QVR chapter 3](../qvr/03-hierarchical.md)).

## Next

Continue to the [examples gallery](../../examples/index.md) for fully-worked models, or jump back to the [QVR DSL track](../qvr/01-first-model.md) and the chapter that matches the model shape you're working on.
