# Wiring

A [wiring](../../guides/transformations.md#wiring-spec) is an
[einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) string that says which
tensor axes denote the same index and which remain in the result. In
[`ij,jk->ik`](../../guides/dsl-contractions.md#worked-example-bilinear-scoring), `j` is summed out and
`i` and `k` survive. The wiring fixes the shape, and the composition rule fixes whether summing out
means a sum, a minimum, or another reduction. Einsum wiring thus extends binary composition to
contractions of any number of tensors. These contractions form an algebra of wiring diagrams only
when that algebra's equations hold [@spivak2013operad; @vagner2015algebras].

::: quivers.core.wiring
