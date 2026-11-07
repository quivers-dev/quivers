# Checking

The [core checker](../../reference/qvr/types-and-indexed-families.md#indexed-case-analysis) validates types, indices, and effect requirements using the information
available statically. A [constructor match can refine an index](../../tutorials/qvr/09-indexed-data.md#2-fold-without-a-shape-check), allowing later operations
to use the resulting equality. Runtime providers supply the implementations needed to
execute accepted computations.

Validation and type and effect inference for the reference kernel.

::: quivers.qiec.checking
