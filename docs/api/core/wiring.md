# Wiring

[Wiring says which tensor axes denote the same index](../../guides/transformations.md#wiring-spec) and which remain in the result. The notation follows
[PyTorch’s `einsum` convention](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html). In
[`ij,jk->ik`](../../guides/dsl-contractions.md#worked-example-bilinear-scoring), `j` is the intermediate index to eliminate, while `i` and `k` survive. The
wiring determines the shape; the composition rule determines whether elimination means
summing, taking a minimum, or another reduction.

Einsum wiring rules that generalize binary composition to contractions
over any number of tensors under a composition rule. Interpreting these
operations as an algebra of wiring diagrams also requires the equations
of that algebra to hold [@vagner2015algebras].

::: quivers.core.wiring
