# Pyro

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Pyro}$.

A Pyro trace records the random-choice sites reached during a model run.
Conditioning the model on supplied latent values lets the probe evaluate the
same point that QVR scores. The sum of site log probabilities is then a
joint log density, including the observations. The test compares these
scores at the same supplied latent values.

## Semantics

Pyro’s [effect-handler tutorial](https://pyro.ai/examples/effect_handlers.html#Introduction) explains how tracing and conditioning compose. The [QVR trace guide](../../guides/inference-foundations.md#trace-and-sample-sites) identifies the site values and scores used by the reference evaluation.

Pyro's denotational semantics is the trace semantics of Bingham,
Chen, Jankowiak, Obermeyer, Pradhan, Karaletsos, Singh, Szerlip,
Horsfall, and Goodman
[@bingham2019pyro] implemented via
[`pyro.poutine`](https://docs.pyro.ai/en/stable/poutine.html)
handlers. The log-density probe is

```python
trace = pyro.poutine.trace(pyro.condition(model, data={**θ, **y})).get_trace(x)
log_p = trace.log_prob_sum()
```

returning the joint log-probability of the trace.

## Unconstrained-space change of variables

The renderer uses the model's constrained coordinates, so the map is the
identity. A probe of a transformed guide would additionally need to account
for that guide's change of variables.

## Family parameterizations

Pyro consumes the same
[`torch.distributions`](https://pytorch.org/docs/stable/distributions.html)
hierarchy. NumPyro instead implements distributions with JAX arrays and a
similar interface. The QVR ↔ Pyro mapping is identity for every
QVR family that has a corresponding `pyro.distributions.*` class.
$\pi_{F, \mathsf{Pyro}} = \mathrm{id}$ and $c_{F, \mathsf{Pyro}} =
0$.

## Per-construct emit

**Sample / observe.** `pyro.sample("x", <dist>)` for latents;
`pyro.sample("y", <dist>, obs=y_data)` for observations. Each
registers a trace site with the documented log-density
contribution.

**Plate.** Nested `with pyro.plate(name, B):` contexts per
[`pyro.plate`](https://docs.pyro.ai/en/stable/primitives.html#pyro.plate)
[@bingham2019pyro]. Semantics is the product measure of
$B$ i.i.d. draws.

**Marginalize.** The renderer enumerates support values and adds the integrated
log density through `pyro.factor`, using `torch.logsumexp`. It groups row
contributions before reducing alternatives when `via` is present. The
hidden coordinate is summed out of the emitted density.

**Score / let / return.** `pyro.factor("name", expr)` for score;
deterministic assignment for let; native `return`.

## Acceptance

* **Tier 1 pipeline composition.** Structural checks cover the
  emitted `model` function and the `Lower >> PyroRenderer >>
  EmitPretty(python)` pipeline.
* **Tier 2 external syntax.** Python's AST parser accepts the
  generated modules in the external-validation matrix.
* **Tier 3 numeric equivalence.** For selected fixtures,
  `trace.log_prob_sum()` is compared with the QVR reference on
  the shared finite grid and tolerance described in
  [the test contract](index.md#6-evidence-tiers).
