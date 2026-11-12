# 9. Debugging quivers programs

A failing model fails at one of four stages. A syntax error means the source cannot be parsed. A type, effect, or algebra error means the declarations do not fit together. A runtime error concerns the data given at run time or the support of a distribution. And a model that passes all three can still fit poorly, because its [posterior](../qvr/06-inference-zoo.md#step-1-vi-mcmc-or-hybrid) is hard to approximate or its assumptions do not describe the data. The tools, one for each stage, are `CompileError` messages, the compiled steps of a `Program`, execution traces, the SVI loss and gradients, and the NUTS diagnostics.

## Reading `CompileError`

A compile failure raises a [`CompileError`](../../api/dsl/compiler.md) whose `line` and `col` give the source position and whose message names the problem. A program declared `Pure` that draws:

```python
from quivers.dsl import CompileError, loads

src = """
object A : FinSet 3
program p : A -> A [effects=[Pure]]
    sample x <- Normal(loc=0, scale=1)  # Sample used in a Pure body
    return x

export p
"""
try:
    loads(src)
except CompileError as e:
    print(e)
```

```text
line 3, col 0: program 'p' is declared as Pure but body uses effects ['Sample']
```

The common kinds of error:

| Kind | Example | Remedy |
|---|---|---|
| Effect mismatch | A body under `[effects=[Pure]]` draws or observes | Widen the declared effects or remove the step. |
| Undefined name | `Norma(...)`: `undefined morphism or distribution family 'Norma'` | Check the spelling against the [family catalog](../../api/continuous/families.md) and the declarations in scope. |
| Algebra mismatch | The operands of `>>` have different algebras | Apply `change_base` to one operand ([QVR chapter 7](../qvr/07-categorical.md#change-of-base)). |
| Shape mismatch | A tensor has the wrong size for a declared object | Compare the declared sizes with the data. |

A [data input](../qvr/01-first-model.md#data-inputs-and-the-observations-dict) missing from the observations is not a compile error: `loads` records free names, and the run that needs the value raises.

## Inspecting a compiled `Program`

`loads` returns a [`Program`](../../api/program.md), a `torch.nn.Module` whose `morphism` is the compiled [`MonadicProgram`](../../api/continuous/programs.md):

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

`steps` holds the step records of [chapter 3](03-probabilistic-programs.md#building-a-monadicprogram): a `Draw` has `names`, a `morphism` (the distribution it draws from), and `args` (the bindings it reads), and `Let`, `Observe`, and `Score` steps follow. Comparing this list with the source shows what the compiler built. This program has no learnable parameters, so the loop over `named_parameters` prints nothing.

## Tracing intermediate values

An execution trace records, for each site, its value and its log density, so a surprising log joint density can be traced to the site that causes it [@pyroEffects]. [`trace`](../../api/inference/trace.md) runs the program once:

```python
from quivers.inference import trace

x = torch.zeros(1, 1)
observations = {"x_design": x_data, "y": y_data}
tr = trace(program.morphism, x, observations)

for name, site in tr.sites.items():
    print(f"{name:12s} value.shape={tuple(site.value.shape)} "
          f"log_prob={site.log_prob.sum().item():.2f}")
```

```text
sigma        value.shape=(1, 1) log_prob=-0.45
beta_0       value.shape=(1, 1) log_prob=-3.94
beta_1       value.shape=(1, 1) log_prob=-1.78
mu           value.shape=(1, 100) log_prob=0.00
y            value.shape=(100,) log_prob=-11345.67
```

The latent variables are drawn from their priors, so the observed `y` has a very low log likelihood under them. A log density of `nan` or `-inf` marks a numerical failure at that site, usually for one of these reasons:

- A scale parameter has gone to zero.
- A `let` produces an invalid argument for the next family, such as a negative scale or probabilities off the simplex.
- An observation lies outside the support of its likelihood, such as a negative count under `Poisson`.

## Watching SVI training

Print the ELBO and the norm of the guide's gradient during the loop:

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

`SVI.step` returns the negative ELBO, which should tend downward, with fluctuations from Monte Carlo and minibatch noise. Typical failures:

- The ELBO levels off while the fit still looks wrong: the guide may be too restrictive or the learning rate unsuitable. Try a richer guide ([`AutoMultivariateNormalGuide`](../../api/inference/guide.md), [`AutoIAFGuide`](../../api/inference/guide.md)) and another learning rate.
- The ELBO becomes `nan`: some site's log density failed. Trace the model at the current guide draw to find it.
- The ELBO oscillates while the gradient norm grows: the learning rate is too large.

## NUTS diagnostics that point at root causes

An `MCMCResult` has three diagnostic fields:

- `result.r_hat[site]`: the rank-normalized split R-hat of each site [@vehtari2021rhat].
- `result.ess[site]`: the bulk effective sample size of each site.
- `result.total_divergences`: the number of divergent transitions over all chains.

Their thresholds and remedies are in the [diagnostics table of QVR chapter 8](../qvr/08-diagnostics-and-comparison.md#convergence-diagnostics). One further symptom: when every chain stays at its starting point, change `init_strategy` (`"prior"`, `"zero"`, or `"guide"` with a fitted guide passed to `MCMC.run`, as in [QVR chapter 3](../qvr/03-hierarchical.md#initialization-and-mass-matrix)).

## A debugging recipe

1. Run `trace(model, inputs, observations)` and confirm that no site has a `nan` or `-inf` log density.
2. Print `program.morphism.steps` and check that the steps match the source.
3. Run SVI for 100 steps, printing every 10. A loss that does not fall at all points to the model or the guide, not to the length of the run.
4. If SVI runs but the posterior means look wrong, fit with NUTS and check `result.r_hat` and `result.total_divergences`.
5. If NUTS reports divergences on a hierarchical model, switch to the non-centered parameterization ([QVR chapter 3](../qvr/03-hierarchical.md#non-centered-parameterization)).

## Next

The [examples gallery](../../examples/index.md) has complete models. The [QVR track](../qvr/01-first-model.md) covers each class of model in turn.
