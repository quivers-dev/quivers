# Contractions

A contraction $x_1, \dots, x_k \to y$ names each axis of each input by a letter. With $\mathcal L$
the letters, $Y$ the output letters and $\mathcal L_c = \mathcal L \setminus Y$ the summed ones, its
value is

$$
\Phi(T_1, \dots, T_k)[y] = \bigoplus_{c} \bigotimes_{i=1}^{k} T_i\bigl[(y, c)|_{x_i}\bigr].
$$

[`plan_contraction`][quivers.core.elimination.plan_contraction] fixes how it is evaluated from the
letters, their extents and the algebra's law claims, and records the result as a
[`ContractionPlan`][quivers.core.elimination.ContractionPlan];
[`run_contraction`][quivers.core.elimination.run_contraction] evaluates a plan and counts its work.
[`EinsumWiring`][quivers.core.wiring.EinsumWiring], binary composition
([`CompositionRule.compose`][quivers.core.algebras.CompositionRule.compose]) and
[`day_convolution`][quivers.enriched.day_convolution.day_convolution] all run through them.

| algebra | algorithm | cost | status |
|---|---|---|---|
| establishes every law of [`ELIMINATION_LAWS`][quivers.core.elimination.ELIMINATION_LAWS] | variable elimination | $O((k + s)\,W)$ time, $O((s + 1)\,W)$ space ([`CONTRACTION_WORK`][quivers.core.wiring.CONTRACTION_WORK]) | conditional on the exponential time hypothesis |
| any other | the written order | $O(k\,U)$ time, $O(U)$ space ([`WRITTEN_ORDER_WORK`][quivers.core.wiring.WRITTEN_ORDER_WORK]) | unresolved |

Here $s = |\mathcal L_c|$, $W$ is the number of elements of the largest bag of the chosen order (or
of the output, when that is larger), and $U$ is the product of every extent.

## Elimination

For a summed letter $v$ let $F_v$ be the factors that contain it and $B_v$ the letters of $F_v$, its
bag. One step replaces $F_v$ by the single factor

$$
g[B_v \setminus \{v\}] = \bigoplus_{v} \bigotimes_{f \in F_v} f,
$$

summing at the same time every other summed letter whose factors all lie in $F_v$.

**Theorem.** In a commutative semiring, a step leaves the value of the contraction unchanged.

*Proof.* Write the product of all factors as $P \otimes Q$, with $P$ the product of $F_v$ and $Q$ that
of the others. $Q$ does not depend on $v$, so by commutativity and associativity of $\oplus$ the sum
over $\mathcal L_c$ is a sum over $\mathcal L_c \setminus \{v\}$ of a sum over $v$, and by
distributivity $\bigoplus_v P \otimes Q = (\bigoplus_v P) \otimes Q$. Reordering the product needs
$\otimes$ commutative and associative. An index range of size zero makes a sum empty, which is the
zero, and the zero annihilates the product it enters; a product of no factors is the unit. These
are the laws of `ELIMINATION_LAWS`. QED.

Each law is read from the algebra's claims through `Lawful.establishes`, never from its name, so an
inline rule whose laws are only declared is evaluated in the written order. The order is chosen by
[`elimination_order`][quivers.core.elimination.elimination_order], cheapest bag first by its
elements, ties in order of first mention. Finding an order of least width is NP-hard
([Arnborg, Corneil and Proskurowski 1987](https://doi.org/10.1137/0608024)), so the plan records the
order, its bags and their widths and claims nothing about them. The same function orders the
variables of a rule body in [`eliminate`][quivers.stochastic.joins.eliminate], with the number of
variables of a bag as its cost.

A step multiplies $|F_v| - 1$ times over $B_v$ and sums $|B_v|$ terms into $|B_v| / n_v$ results.
Every factor is joined once, and $k + s$ factors exist at most, so the products are at most
$(k + s)\,W$ and the sums at most $s\,W$; the factors held between steps are at most one per step.
This gives the bounds above ([Dechter 1999](https://doi.org/10.1016/S0004-3702(99)00059-4);
[Abo Khamis, Ngo and Rudra 2016](https://doi.org/10.1145/2902251.2902280) for semirings). For a line
of $k$ matrices of order $n$, $W = n^3$, where the written order forms a tensor of $n^{k+1}$
elements.

At a tie of a selective sum, each sum gives the whole derivative to its first best term, so the
contraction gives it to the first best index in the order of elimination.

## The written order

An algebra that does not establish the laws, such as
[`ProductFuzzyAlgebra`][quivers.core.algebras.ProductFuzzyAlgebra], is evaluated as the formula is
written: every input is aligned to all letters, the inputs are multiplied left to right, and the
summed letters are summed at once. No law moves a sum inside a product, so whether fewer operations
suffice is open for such an algebra. A rule whose elements have axes of their own, such as a
[`MatrixSemiring`][quivers.core.algebras.MatrixSemiring], keeps them last and keeps the order of
its factors.

## Work

[`EinsumWiring.apply_with_work`][quivers.core.wiring.EinsumWiring.apply_with_work] returns a
[`WorkReport`][quivers.qiec.contracts.WorkReport] of the execution: products, binary sums (a sum of
$m$ terms is $m - 1$ of them), elements formed, and the largest number of formed elements held at
once, each in elements of the algebra; divisions, pivots, comparisons and hashing are measured as
zero. The evidence harness checks these counts against a dispatch counter and the claims on growing
families.

::: quivers.core.elimination
