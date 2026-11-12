# Edward2

Edward2 writes a model as a Python function whose calls of distribution constructors create random
variables, which a tracing context, the tape, records [@tran2018simple]. Its distributions are those of
[TensorFlow Probability](https://www.tensorflow.org/probability). A test probe reads the random variables
on the tape and adds their log densities ([Transpilation correctness](index.md)). A mixture must be one
random variable with the mixture's density: recording its components as further random variables would
multiply extra factors into the joint density.

## How Edward2 evaluates a log density

Each call `edward2.<Family>(...)` creates a
[`RandomVariable`](https://www.tensorflow.org/probability/api_docs/python/tfp/edward2/RandomVariable),
recorded inside an `ed.tape()` context. The probe is

```python
with ed.tape() as t:
    model(...)
log_p = sum(rv.distribution.log_prob(rv.value) for rv in t.values())
```

which sums the log densities of the recorded random variables at their values.

## Change of variables

The emitted model is in constrained coordinates, and TensorFlow Probability's bijectors change coordinates
only during inference, so $\Psi_{\mathsf{Edward2}} = \mathrm{id}$.

## Family parameterizations

Edward2's families are the `Distribution` classes of
[TensorFlow Probability](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions), and the
map is the identity for every family with a TensorFlow Probability class:
$\pi_{F, \mathsf{Edward2}} = \mathrm{id}$ and $c_{F, \mathsf{Edward2}} = 0$.

## Emitted constructs

**`sample` and `observe`.** `<name> = edward2.<Family>(<args>, sample_shape=[B0, B1, ...], name="<name>")`
for a latent variable or an observation. For a [draw over a plate](../programs.md#24-indexed-bind-plate),
`sample_shape` gives the [batch shape](https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes). An observation also passes `value=<name>`, the
model argument that holds the data, so its random variable holds the data over the whole plate, and a later
`let` that reads the observed name reads the data. Without data, a run draws one value for each row. When the
broadcast shape of the arguments is not known statically, as for an argument gathered through an index, the
observation has no `sample_shape`, and the shape of the data stands.

**Plates.** `sample_shape=[B]` draws $B$ independent copies, so its density is the product of $B$ factors.

**`marginalize`.** When the scope has the form of a mixture, the renderer writes one
`edward2.MixtureSameFamily` of TensorFlow Probability component and mixing distributions. For other supported
grouped blocks, it sums the scores of each value by group and records the reduced log weight as a factor (see
below). Both remove the latent variable, and other combinations of scope and family are refused.

**`score`.** Edward2 has no factor primitive. The renderer adds to the module a distribution with one point,
`_QvrFactorDistribution`, whose log density is the weight, and records a score as a random variable of that
distribution with `_qvr_qiec_factor(name, weight)`, so the tape carries the weight like any other site.

**`let`, `return`.** An assignment for a `let`, and the function returns the named random variables.

## Evidence

- **Level 1.** Structural tests check the emitted `model` function and the pipeline.
- **Level 2.** Python's `ast` parser accepts the modules of the syntax matrix.
- **Level 3.** On selected fixtures, the probe on the tape is compared with the QVR density on the shared grid
  and tolerance ([evidence](index.md#6-evidence-tiers)).
