# Model binding and fitting

The [model interface](../guides/model-data-and-fitting.md) binds external values
using the types and roles checked from QVR, then applies a reusable inference method.

Declare explicit inputs in the runtime telescope of a program header, as in
`program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out`.
Static objects, morphisms, and extents use a preceding bracketed telescope.
Annotations constrain values; `observe` determines which inputs are responses.
The [parameter reference](../reference/qvr/index.md#parameter-telescopes) explains
the convention across language constructs.

::: quivers.modeling.model

::: quivers.modeling.methods
