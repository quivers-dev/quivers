# Checking

The core checker validates types, indices, and effect requirements using the information
available statically. A constructor match can refine an index, allowing later operations
to use the resulting equality. Runtime providers supply the implementations needed to
execute accepted computations.

See [checking a case using its index](../../reference/qvr/types-and-indexed-families.md#indexed-case-analysis) and
[using a vector length index in a fold](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check).

Validation and type and effect inference for the reference kernel.

::: quivers.qiec.checking
