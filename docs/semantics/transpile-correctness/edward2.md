# Edward2

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Edward2}$.

Edward2 distribution constructors register random-variable objects. A tape
records those objects so a probe can retrieve their distributions and values
and add their log probabilities. For a mixture, the tape should contain one
random variable with the mixture density. Registering its component
distributions as additional variables would multiply extra factors into
the joint density.

## Semantics

Edward2 represents distributions as random variables recorded by a tracing
context [@tran2018simple]. The renderer instantiates this design on top of
[TensorFlow Probability](https://www.tensorflow.org/probability)'s
[`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions)
hierarchy. A program is a Python function whose
`edward2.<Family>(...)` calls register
[`RandomVariable`](https://www.tensorflow.org/probability/api_docs/python/tfp/edward2/RandomVariable)
sites in an `ed.tape` context; the log-density probe is

```python
with ed.tape() as t:
    model(...)
log_p = sum(rv.distribution.log_prob(rv.value) for rv in t.values())
```

returning the joint log-density of the trace.

## Unconstrained-space change of variables

Identity at the renderer level. TFP handles bijector-based
reparameterizations at inference time;
$\Psi_{\mathsf{Edward2}} = \mathrm{id}$ for the model-side
denotation.

## Family parameterizations

Edward2 families are TFP `Distribution` classes exported via
[`edward2`](https://www.tensorflow.org/probability/api_docs/python/tfp/edward2).
The QVR ↔ Edward2 arg mapping is identity for every family that has
a corresponding TFP class. $\pi_{F, \mathsf{Edward2}} = \mathrm{id}$
and $c_{F, \mathsf{Edward2}} = 0$.

## Per-construct emit

**Sample / observe.** `<name> = edward2.<Family>(<args>,
sample_shape=[B0, B1, ...], name="<name>")` for latents and
observed variables. For QVR’s [indexed binds](../programs.md#24-indexed-bind-plate), the `sample_shape` argument carries the
[batch-axis shape](https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes#basics). An observed variable also passes
`value=<name>`, the model argument holding the observation. Its random
variable therefore holds the data over the whole plate, and a later `let`
that reads the observed name reads the data. A run with no data draws one
value per row. When the arguments' broadcast shape is not statically
known, as for an argument gathered through an index, the observation
takes no `sample_shape`, and the data's shape stands.

**Plate.** Captured via `sample_shape=[B]`. By TFP's documented
broadcasting semantics, `sample_shape=[B]` produces a $B$-fold
i.i.d. product (TFP user guide on "Shapes and Broadcasting").

**Marginalize.** For supported mixture-shaped scopes, the renderer uses
`edward2.MixtureSameFamily` with bare TFP component and mixing distributions.
Supported grouped scopes instead accumulate atom scores by group and register
a factor variable carrying the reduced log weight. Both routes remove the
latent coordinate; unsupported scope/family combinations are rejected.

**Score / let / return.** Edward2 has no native `factor`
primitive; the renderer raises
`UnsupportedConstruct(["construct:ScoreStep"])` if a score step is
encountered. Deterministic let-bindings emit as Python
assignments. The function returns the named random variables.

## Acceptance

* **Tier 1 pipeline composition.** Structural checks cover the
  emitted `model` function and the renderer pipeline.
* **Tier 2 external syntax.** Python's AST parser accepts the
  generated modules in the external-validation matrix.
* **Tier 3 numeric equivalence.** For selected fixtures, the
  `ed.tape`-based probe is compared with the QVR reference on the
  shared finite grid and tolerance described in
  [the test contract](index.md#6-evidence-tiers).

### References

For the Edward2 design, see [@tran2018simple]. The target library and renderer determine which distribution constructors and marginalizations are supported.
