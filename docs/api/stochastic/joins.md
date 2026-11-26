# Joins

A rule body $r_1(\bar x_1), \dots, r_m(\bar x_m)$ is a [conjunctive query](https://en.wikipedia.org/wiki/Conjunctive_query) over $n$ variables.
[`body_query`][quivers.stochastic.joins.body_query] reads it off a
[`Rule`][quivers.qiec.rules.Rule]: each premise is an atom whose attributes are the variables of its
pattern, and [`relations`][quivers.stochastic.joins.relations] makes each atom's relation from the
items of its form that match the pattern, keyed by the values of those variables. Matching flattens
nested constructor patterns and literals; it determines the item from its tuple, since the item's
arguments are the pattern evaluated there. A premise with a sum of two variables, such as
$u(i + k)$, is refused: its matches are not a relation of its items alone.

Two questions are asked of a body, and they have different algorithms.

| question | algorithm | cost | status |
|---|---|---|---|
| every instance (a forest, requested evidence) | [`generic_join`][quivers.stochastic.joins.generic_join] | $O(N + m\,n\,\mathrm{AGM})$ | worst-case optimal in data complexity ([`GENERIC_JOIN_WORK`][quivers.stochastic.joins.GENERIC_JOIN_WORK]) |
| sum of products, grouped by the conclusion | [`eliminate`][quivers.stochastic.joins.eliminate] | $O(N + n\,m\,W + F)$ for the chosen order | conditional on the exponential time hypothesis ([`ELIMINATION_WORK`][quivers.stochastic.joins.ELIMINATION_WORK]) |
| the same, for weights that keep their order | [`enumerate_aggregate`][quivers.stochastic.joins.enumerate_aggregate] | that of `generic_join` | the products need every instance in premise order |

## Listing instances

For weights $x_e \ge 0$ on the atoms with $\sum_{e \ni v} x_e \ge 1$ for every variable $v$ (a
fractional edge cover), the join has at most $\mathrm{AGM} = \prod_e N_e^{x_e}$ tuples, and for
every query and sizes there are relations attaining it up to a constant factor (Atserias, Grohe
and Marx 2013). [`agm_bound`][quivers.stochastic.joins.agm_bound] solves the linear program that
minimizes $\sum_e x_e \log N_e$, rounds its solution to rationals and repairs any variable left
uncovered, so the recorded [`AGMBound`][quivers.stochastic.joins.AGMBound] is an exact cover and
its bound holds whether or not it is the least. For the triangle at equal sizes it is
$x = (\tfrac12, \tfrac12, \tfrac12)$ and $N^{3/2}$.

Generic Join (Ngo, Re and Rudra 2014) binds the variables in a fixed order. With the variables
before $v$ bound to $t$, the values of $v$ are the intersection, over the atoms containing $v$, of
the values their tries allow below $t$; it iterates the smallest set and probes the others in
expected constant time.

**Theorem.** Generic Join lists each instance once, and iterates at most $n\,\mathrm{AGM}$ values
and makes at most $m\,n\,\mathrm{AGM}$ probes.

*Proof.* A binding is extended at depth $d$ exactly when its first $d$ variables agree with a
tuple of every atom on the variables they share, so each complete binding is an instance and each
instance is reached by one path, the values of its variables in order. Let $v$ be the variable at
depth $d$ and $E_v$ the atoms containing it. For a prefix $t$ of the first $d - 1$ variables, the
values iterated are $\min_{e \in E_v} |R_e[t]|$, where $R_e[t]$ is the set of values of $v$ in
the tuples of $R_e$ that agree with $t$. Since $\sum_{e \in E_v} x_e \ge 1$,
$\min_e |R_e[t]| \le \prod_{e \in E_v} |R_e[t]|^{x_e}$, and summing over the prefixes that
survive, [Hölder's inequality](https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality) bounds the total by the AGM bound of the first $d$ variables under
the restricted cover, which is at most $\mathrm{AGM}$ because restriction and projection do not
grow a relation (Ngo, Re and Rudra 2014, the analysis of Generic Join; Ngo, Porat, Re and Rudra
2018, theorem 5.1). Over $n$ depths this gives $n\,\mathrm{AGM}$ values, each probed in at most
$m - 1$ other sets. QED

The bound holds under the conditions stated in the claim: the rule is fixed and the relations grow
(data complexity); attributes are flat after matching; hashing costs expected constant time; and
the side condition, evaluated on each complete binding, filters the output and is outside the
bound. Generic Join is worst-case optimal, not instance optimal: on some inputs another algorithm
does less (Ngo, Porat, Re and Rudra 2018, section 7). The indexed chart of a plan joins one premise
at a time, and on the triangle with a hub value or a skewed triangle that does $\Theta(N^2)$ work
against the $N^{3/2}$ bound, which the checks measure.

## Aggregates by elimination

For a commutative semiring the aggregate of a body, grouped by the conclusion's variables $F$, is
$\bigoplus_{\bar y} w \otimes \bigotimes_e \phi_e$, the sum over the other variables $\bar y$ of the
products of the premise weights. Eliminating $y$ joins the factors that contain it and sums it
away:

$$
\bigoplus_{y} \bigotimes_{e} \phi_e = \Big(\bigotimes_{e \not\ni y} \phi_e\Big) \otimes
\bigoplus_{y} \bigotimes_{e \ni y} \phi_e .
$$

**Theorem.** If $\oplus$ is associative and commutative, $\otimes$ is associative and commutative
with unit, and $\otimes$ distributes over $\oplus$, every order of eliminating the variables of
$\bar y$ gives the aggregate.

*Proof.* The display is distributivity applied to the factors free of $y$, after commuting them to
the front; the result is a function of the remaining variables. Repeating it for each variable of
the order, and joining the remaining factors over $F$, gives the sum over $\bar y$ of the products.
Since $\oplus$ is associative and commutative, summing the variables in one order or another, or
all at once, gives the same value. QED

There is one aggregate, the sum, over the variables not in the conclusion; so every order of them
is semantically equivalent, the condition that FAQ needs for mixed aggregates (Abo Khamis, Ngo and
Rudra 2016) holds trivially, and no product aggregate arises. The laws used are
[`ELIMINATION_LAWS`][quivers.stochastic.joins.ELIMINATION_LAWS]; each must be proved or decided by
the algebra's own claims, so a [`MatrixSemiring`][quivers.core.algebras.MatrixSemiring], whose
product does not commute, is refused, and its aggregate is taken by `enumerate_aggregate`, which
multiplies in premise order and needs only
[`ENUMERATION_LAWS`][quivers.stochastic.joins.ENUMERATION_LAWS].

A side condition reads some variables. When the first of them is eliminated, the factors covering
all of them are joined into one bag, and the condition filters the bag before the variable is
summed out; this is the elimination of the product of the factors with the condition's
indicator, which is a factor over those variables. A condition on conclusion variables only
filters the output.

Each [`EliminationStep`][quivers.stochastic.joins.EliminationStep] records its bag and a fractional
cover of the bag by the body's atoms. The bag's join has at most that cover's bound of tuples,
since it lies in the projection of the body's join; the largest is $W$ of the claim. The
[`Elimination`][quivers.stochastic.joins.Elimination] records the order, the induced width (the
largest bag less one) and the fractional width (the largest cover weight). The order is chosen by
[`elimination_order`][quivers.stochastic.joins.elimination_order], smallest bag first; finding an
order of least width is NP-hard (Arnborg, Corneil and Proskurowski 1987), so the widths are of the
chosen order and are not claimed least. Under the [exponential time hypothesis](https://en.wikipedia.org/wiki/Exponential_time_hypothesis), for classes of
bounded arity no algorithm beats [treewidth](https://en.wikipedia.org/wiki/Treewidth) by more than a logarithmic factor in the exponent (Marx
2010), which is why the claim's status is conditional.

On the star $r_1(x, y_1), r_2(x, y_2), r_3(x, y_3)$ with one value of $x$ of degree $N$, there are
$N^3$ instances; elimination removes each $y_i$ in a bag $\{x, y_i\}$ and does linear work.

## References

- Abo Khamis, M., Ngo, H. Q., and Rudra, A. (2016). FAQ: questions asked frequently. *PODS*, 13-28.
  [doi:10.1145/2902251.2902280](https://doi.org/10.1145/2902251.2902280)
- Arnborg, S., Corneil, D. G., and Proskurowski, A. (1987). Complexity of finding embeddings in a
  k-tree. *SIAM Journal on Algebraic Discrete Methods* 8(2), 277-284.
  [doi:10.1137/0608024](https://doi.org/10.1137/0608024)
- Atserias, A., Grohe, M., and Marx, D. (2013). Size bounds and query plans for relational joins.
  *SIAM Journal on Computing* 42(4), 1737-1767.
  [doi:10.1137/110859440](https://doi.org/10.1137/110859440)
- Marx, D. (2010). Can you beat treewidth? *Theory of Computing* 6, 85-112.
  [doi:10.4086/toc.2010.v006a005](https://doi.org/10.4086/toc.2010.v006a005)
- Ngo, H. Q., Porat, E., Re, C., and Rudra, A. (2018). Worst-case optimal join algorithms.
  *Journal of the ACM* 65(3), 1-40. [doi:10.1145/3180143](https://doi.org/10.1145/3180143)
- Ngo, H. Q., Re, C., and Rudra, A. (2014). Skew strikes back: new developments in the theory of
  join algorithms. *SIGMOD Record* 42(4), 5-16.
  [doi:10.1145/2590989.2590991](https://doi.org/10.1145/2590989.2590991)

::: quivers.stochastic.joins
