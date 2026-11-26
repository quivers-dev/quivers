# Spanning trees

A directed spanning tree of nodes $0, \dots, n-1$ rooted at $r$ (an
[arborescence](https://en.wikipedia.org/wiki/Arborescence_(graph_theory))) gives every node except
$r$ one parent and has no cycle. `weights[..., x, y]` is the weight $W[x, y]$ of the edge from
parent $x$ to child $y$; self edges, edges into the root and edges a mask removes are absent. The
partition function is

$$
Z(W) = \sum_{T} \prod_{(x \to y) \in T} W[x, y],
$$

and [`tree_partition`][quivers.core.spanning.tree_partition] computes it in a zero-sum-free
[`Semifield`][quivers.core.algebras.Semifield]: in the log semifield the log partition, in the
linear one the partition, in the max-plus one the weight of the best tree, in the Boolean one
whether a tree exists. [`directed_tree`][quivers.core.spanning.directed_tree] wraps the weights,
root, algebra and mask as a [`DirectedTrees`][quivers.core.spanning.DirectedTrees].

## Elimination without subtraction

For a node $k \ne r$ let $\mathrm{in}(k) = \bigoplus_{x} W[x, k]$, its total incoming weight. On the
nodes other than $k$ define

$$
W'[x, y] = W[x, y] \oplus W[x, k] \otimes W[k, y] \oslash \mathrm{in}(k) \qquad (x \ne y).
$$

**Theorem.** In every commutative semifield, if $\mathrm{in}(k)$ is not zero, then
$Z(W) = \mathrm{in}(k) \otimes Z(W')$.

*Proof.* Over the real numbers, let $L$ be the Laplacian with $L[y, y] = \mathrm{in}(y)$ and
$L[x, y] = -W[x, y]$ for $x \ne y$, and $\hat L$ the matrix without the root's row and column. The
matrix-tree theorem for directed graphs (Tutte 1984) gives $Z(W) = \det \hat L$. Pivoting on
$L[k, k] = \mathrm{in}(k)$ gives $\det \hat L = \mathrm{in}(k) \det S$, where the Schur complement
has entries $S[x, y] = L[x, y] - L[x, k] L[k, y] / \mathrm{in}(k)$. Off the diagonal this is
$-W'[x, y]$. On the diagonal it is
$\mathrm{in}(y) - W[y, k] W[k, y] / \mathrm{in}(k)$, and the incoming weight of $y$ under $W'$ is

$$
\sum_{x \ne y, k} \Big( W[x, y] + \frac{W[x, k] W[k, y]}{\mathrm{in}(k)} \Big)
= \mathrm{in}(y) - W[k, y] + W[k, y] \frac{\mathrm{in}(k) - W[y, k]}{\mathrm{in}(k)}
= \mathrm{in}(y) - \frac{W[y, k] W[k, y]}{\mathrm{in}(k)}.
$$

So $S$ is the Laplacian of $W'$ without the root, and $Z(W) = \mathrm{in}(k) Z(W')$ for positive
real weights. Now $Z$ is homogeneous of degree $n - 1$ in the weights of a tree on $n$ nodes, so
$\mathrm{in}(k)^{n-2} Z(W') = Z(W'')$ with $W''[x, y] = W[x, y]\,\mathrm{in}(k) + W[x, k] W[k, y]$.
The identity $\mathrm{in}(k)^{n-2} Z(W) = \mathrm{in}(k)\, Z(W'')$ is then an equality of
polynomials with nonnegative integer coefficients that holds on the positive orthant, an open set,
so the two polynomials have the same coefficients. It therefore holds in every commutative
semiring. In a semifield with $\mathrm{in}(k)$ not zero, multiplying by the inverse of
$\mathrm{in}(k)^{n-2}$ and using homogeneity again gives $Z(W) = \mathrm{in}(k) \otimes Z(W')$.
QED

Eliminating every node but the root leaves the empty tree, whose weight is the unit, so $Z(W)$ is
the product of the pivots. Each step costs $O(n^2)$ operations, the whole $O(n^3)$, and no step
subtracts. Subtraction-free circuits of this size exist and circuits with sums and products alone
need exponential size (Fomin, Grigoriev and Koshevoy 2016); whether a subtraction-free circuit below
$n^3$ exists is open. With subtraction, a determinant costs $O(n^\omega)$, but it cancels: for edge
log weights $-60$ and $-61$ from the root and a unit cycle between the two other nodes, the log
partition is $-59.6867$, which elimination computes exactly, while the log determinant of the
Laplacian in float64 is $-\infty$, since $1 + e^{-60}$ rounds to $1$.

**Zero pivots.** The algebra must be zero-sum-free, so a pivot is zero exactly when every incoming
weight of its node is. A node with a nonzero pivot is eliminated first. If the nodes left all have
zero pivots and two or more remain, no tree exists, every monomial of $Z$ holds an incoming weight
of each of them, and every first derivative vanishes: the result is the zero with derivative zero.
If one remains, the partition is the product of the pivots, the last zero; where the algebra's zero
is a finite number its derivative is that of the product, which keeps the derivative at a weight
that is zero but differentiable, and where the zero lies at infinity (log space) the
representation is not differentiable there and its derivative is zero.

## Edge marginals

[`tree_marginals`][quivers.core.spanning.tree_marginals] is the derivative of the partition with
respect to each weight: in the log semifield,
$\partial \log Z / \partial \log W[x, y]$, the probability of the edge under the normalized tree
weights; in the max-plus semifield, the indicator of the best tree. It is the derivative of the
elimination, so it costs a constant times the partition (Baur and Strassen 1983), and it can be
differentiated again.

**Erratum.** Andrews, Eisner and Dredze (2012, section 6) define the Laplacian by their
equation 12, $L[x, y] = -\delta(y \mid x)$ for $x \ne y$ and $L[y, y] = \sum_{x'} \delta(y \mid x')$,
and give the marginals in equation 14 as
$\delta(y \mid x)(\hat L^{-1}_{xx} - \hat L^{-1}_{xy})$ for $x \ne y$. The derivative of
$\log \det \hat L$ under equation 12 is $\delta(y \mid x)(\hat L^{-1}_{yy} - \hat L^{-1}_{yx})$,
since $\partial \log \det \hat L / \partial \hat L_{ij} = \hat L^{-1}_{ji}$ and $\delta(y \mid x)$
enters $\hat L_{xy}$ with sign $-1$ and $\hat L_{yy}$ with sign $+1$. This module implements the
derivative of equation 12; on a four-node graph the printed form of equation 14 differs from
enumeration of every tree by more than $0.1$, and the derivative agrees with it to rounding.

## References

- Tutte, W. T. (1984). *Graph Theory*. Addison-Wesley.
- Koo, T., Globerson, A., Carreras, X., and Collins, M. (2007). Structured prediction models via
  the matrix-tree theorem. *EMNLP-CoNLL*. [D07-1015](https://aclanthology.org/D07-1015/)
- Andrews, N., Eisner, J., and Dredze, M. (2012). Name phylogeny: a generative model of string
  variation. *EMNLP-CoNLL*, 344-355. [D12-1032](https://aclanthology.org/D12-1032/)
- Fomin, S., Grigoriev, D., and Koshevoy, G. (2016). Subtraction-free complexity, cluster
  transformations, and spanning trees. *Foundations of Computational Mathematics*.
  [doi:10.1007/s10208-014-9231-y](https://doi.org/10.1007/s10208-014-9231-y)
- Baur, W., and Strassen, V. (1983). The complexity of partial derivatives. *Theoretical Computer
  Science* 22, 317-330. [doi:10.1016/0304-3975(83)90110-X](https://doi.org/10.1016/0304-3975(83)90110-X)

::: quivers.core.spanning
