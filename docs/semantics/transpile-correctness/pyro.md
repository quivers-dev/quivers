# Pyro

A Pyro model is a Python function whose calls `pyro.sample` register named sites, which Pyro's effect
handlers, the module [`pyro.poutine`](https://docs.pyro.ai/en/stable/poutine.html), record in an execution
trace [@bingham2019pyro; @pyroEffects]. Conditioning the model on given latent values lets the test probe
evaluate the point QVR scores ([Transpilation correctness](index.md)).

## How Pyro evaluates a log density

The probe conditions the model on the parameters $\theta$ and the data $y$, records its trace, and sums
the log densities of its sites:

```python
trace = pyro.poutine.trace(pyro.condition(model, data={**θ, **y})).get_trace(x)
log_p = trace.log_prob_sum()
```

## Change of variables

The emitted model is in constrained coordinates, so $\Psi_{\mathsf{Pyro}} = \mathrm{id}$. A probe of a
guide that transforms coordinates would also need that guide's Jacobian.

## Family parameterizations

Pyro's distributions are those of
[`torch.distributions`](https://pytorch.org/docs/stable/distributions.html), the hierarchy QVR's families
are built on, so the map is the identity for every QVR family with a `pyro.distributions` class:
$\pi_{F, \mathsf{Pyro}} = \mathrm{id}$ and $c_{F, \mathsf{Pyro}} = 0$.

## Emitted constructs

**`sample` and `observe`.** `pyro.sample("x", <dist>)` for a latent variable and
`pyro.sample("y", <dist>, obs=y_data)` for an observation, each a site of the trace.

**Plates.** Nested `with pyro.plate(name, B):` contexts
([`pyro.plate`](https://docs.pyro.ai/en/stable/primitives.html#pyro.plate)) [@pyroTensorShapes], whose density
is the product over the $B$ conditionally independent draws.

**`marginalize`.** The renderer enumerates the values of the latent variable, reduces with
`torch.logsumexp`, and adds the result with `pyro.factor`. With `via` it sums the rows of each group before
the reduction. The emitted density has the variable summed out.

**`score`, `let`, `return`.** `pyro.factor("name", expr)` for a score, an assignment for a `let`, and
Python's `return`.

## Evidence

- **Level 1.** Structural tests check the emitted `model` function and the pipeline `Lower`,
  `PyroRenderer`, and `EmitPretty(python)`.
- **Level 2.** Python's `ast` parser accepts the modules of the syntax matrix.
- **Level 3.** On selected fixtures, `trace.log_prob_sum()` is compared with the QVR density on the shared
  grid and tolerance ([evidence](index.md#6-evidence-tiers)).
