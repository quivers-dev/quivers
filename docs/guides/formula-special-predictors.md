# Structured predictor terms

A penalized smooth has coupled coefficient coordinates. Its penalty identifies
which directions receive Gaussian regularization and which require ordinary
population priors. QVR retains these directions as a `GaussianDesignTerm`,
separately from `FixedColumn` values, and emits them through the existing formula
compiler, inference methods, and prediction evaluator.

## Penalized B-splines

Use `s(x, bs='ps', k=6)` for a centered P-spline. `k` is the number of original
B-spline coordinates, `degree` is their polynomial degree (three by default),
and `m` is the coefficient-difference penalty order (two by default). The
training range determines equally spaced interior knots once.

```python
import numpy as np
import pandas as pd
from quivers.formulas import PriorSpec, fit

x = np.linspace(-1, 1, 20)
data = pd.DataFrame({"x": x, "y": 0.2 + 0.5 * x**2})
model = fit(
    "y ~ s(x, bs='ps', k=6)",
    data=data,
    family="gaussian",
    method="svi",
    num_steps=20,
    priors=PriorSpec(distribution="HalfNormal(0.7)", class_="sds"),
)
means = model.posterior_epred(
    new_data=pd.DataFrame({"x": [-0.25, 0.25]}),
    num_samples=10,
    seed=3,
)
assert means.shape == (1, 10, 2)
```

The compiler imposes zero average smooth contribution on the training rows.
It separates the penalty's null space from its positive spectrum. The null
space receives the configured population coefficient prior, including the
existing default column autoscaling. Each penalized block receives independent
standard-Normal innovations multiplied by a positive scale. The scale uses
`random_scale_prior` by default. A typed `PriorSpec(class_='sds')` replaces it
through the common prior catalogue. `coef` selects an authored smooth term and
`dpar` selects an auxiliary predictor.

Smooths can appear in location and distributional formulas. Coordinate-valued
responses retain a separate coefficient block and scale for each free response
component. The existing SVI, HMC, and NUTS workflows consume the same generated
program. Thus a smooth introduces no separate fitting or posterior replay loop.

Prediction retains the training knots, centering constraint, and penalty
transformation. New rows cannot redefine the fitted coefficient coordinates.
Evaluation outside the training range uses the retained B-spline polynomial
continuation. Repeated requests preserve the original formula and evaluator.

The explicit `bs='ps'` identifies this constructor. Default thin-plate `s(x)`,
other basis families, tensor-product smooths, and smooth interactions require
their own retained basis and penalty contracts. They are currently refused
instead of becoming unpenalized fixed B-spline columns. Gaussian-process,
monotonic, and multi-membership frontend constructors remain separate follow-up
work. This page does not claim complete brms special-term parity.

The released brms smooth emitter likewise distinguishes null-space population
coefficients from scaled standard-Normal penalized coefficients. QVR's explicit
P-spline constructor specifies its own knot and prior defaults. It does not
claim numerical identity to every `mgcv` smooth basis. See the
[released smooth emitter](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/R/stan-predictor.R#L780)
and [retained prediction transform](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/R/data-predictor.R#L1115).
