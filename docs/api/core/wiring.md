# Wiring

Wiring says which tensor axes denote the same index and which remain in the result. In
`ij,jk->ik`, `j` is the intermediate index to eliminate, while `i` and `k` survive. The
wiring determines the shape; the composition rule determines whether elimination means
summing, taking a minimum, or another reduction.

See [specifying tensor wiring](../../guides/transformations.md#wiring-spec) and
[a worked bilinear contraction](../../guides/dsl-contractions.md#worked-example-bilinear-scoring).

For tensor contraction using explicit axis labels, consult the [PyTorch einsum reference](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html); Quivers additionally allows a choice of reduction algebra.

Einsum wiring rules that generalize binary composition to contractions
over any number of tensors under a composition rule.

For a mathematical treatment of connecting interfaces through wiring diagrams, see [@vagner2015algebras]. The particular algebra used here must still satisfy the laws required by any claimed operadic interpretation.

::: quivers.core.wiring
