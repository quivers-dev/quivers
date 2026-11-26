# Cyclic systems

A plan whose recursion contract is [`LinearCyclic`][quivers.qiec.contracts.LinearCyclic] and whose
algebra is a [`Star`][quivers.core.algebras.Star] solves its system component by component.
[`solve`][quivers.stochastic.recursion.solve] builds the derived hypergraph as a
[packed forest](forest.md), splits it into
[strongly connected components](https://en.wikipedia.org/wiki/Strongly_connected_component) by
[`components`][quivers.stochastic.recursion.components], visits them so that every premise outside
a component is final first, and solves each one through the algebra's closure. Each
[`Component`][quivers.stochastic.recursion.Component] records the side it was solved on.

## The system of a component

Inside a component every edge must have at most one premise in the component; an edge with two is
refused as nonlinear. With the products read in premise order and the rule weight last, an edge
concluding $x$ whose premise in the component is $y$, at position $k$, contributes

$$
a \otimes v(y) \otimes c, \qquad a = v(p_1) \otimes \cdots \otimes v(p_{k-1}), \qquad
c = v(p_{k+1}) \otimes \cdots \otimes v(p_n) \otimes w_r,
$$

where every $v(p_i)$ other than $v(y)$ is final. Whether a coefficient is the unit is decided by
the rule, not by the numbers: $a$ is the unit when $y$ is the first premise, and $c$ when $y$ is the
last and the rule has no weight. The remaining edges and the inputs give the constant term $b(x)$.

**Theorem (one-sided solutions).** Let the algebra be a [`Star`][quivers.core.algebras.Star], whose
closure $A^* = I \oplus A \oplus A^2 \oplus \cdots$ is the sum of the weights of all paths and the
least solution of $X = I \oplus A X$ on its declared domain (Lehmann 1977). If every edge of a
component has $c$ the unit, the value of each item of the component, the sum over its derivations
of their products, is $(M^* b)_x$ with $M_{xy} = \bigoplus a$ over its edges from $y$ to $x$. If
every edge has $a$ the unit, it is $(b N^*)_x$ with $N_{yx} = \bigoplus c$. Neither needs the
product to commute.

*Proof.* In the right-linear case a derivation of $x$ is a path
$x = x_0 \leftarrow x_1 \leftarrow \cdots \leftarrow x_m$ of edges inside the component, followed
by a derivation of a term of $b(x_m)$. Each edge's coefficient $a$ stands to the left of the value it
multiplies and its $c$ is the unit, so the product of the derivation is
$a_1 \otimes \cdots \otimes a_m \otimes \beta$ with $\beta$ that term's product, the coefficients in
the order of the path. By distributivity, summing over the edges of each step and the terms of
$b$ gives $(M^m b)_x$ for the paths of length $m$, and over every length
$\bigoplus_m (M^m b)_x = (M^* b)_x$, the closure's series. It solves $v = b \oplus M v$, since
$M^* = I \oplus M M^*$. In the left-linear case each $c$ stands to the right, the product is
$\beta \otimes c_m \otimes \cdots \otimes c_1$, and the sum is $(b N^*)_x$ by the mirror
argument. QED

Outside the closure's domain the series has no value; the algebra raises, and the run is refused.

**Mixed sides.** A component in which some edge has $c$ not the unit and another (or the same) has
$a$ not the unit multiplies its unknowns on both sides. With a commutative product,
$a \otimes v \otimes c = (a \otimes c) \otimes v$, and the component is right linear with
$M_{xy} = \bigoplus a \otimes c$; the record says `"commuted"`. Without a claim that the product
commutes, established by the algebra's own evidence, no one-sided system describes it, and the run
is refused with `mixed-sides`, naming a rule multiplying on each side.

## Refusals

| [`PlanRefusal`][quivers.stochastic.planning.PlanRefusal] | when |
|---|---|
| `inadmissible` | the algebra does not establish the closure's least-solution law |
| `nonlinear-cycle` | an edge has two premises in its own component |
| `mixed-sides` | a component multiplies on both sides of a product not established to commute |
| `divergent` | the closure lies outside the algebra's domain, or rounding cannot decide whether it does |

A divergent closure has no value: no partial sum is returned. The cost is
[`CYCLIC_WORK`][quivers.stochastic.planning.CYCLIC_WORK]: the edges, plus $S^3$ operations of the
algebra for a component of $S$ items, which a ring would lower to $S^\omega$.

## Monotone polynomial systems

Under [`MonotonePolynomial`][quivers.qiec.contracts.MonotonePolynomial] a component may have
edges with several premises inside it. In an algebra with a
[`RealEmbedding`][quivers.core.algebras.RealEmbedding], such as
[`MarkovAlgebra`][quivers.core.algebras.MarkovAlgebra] or
[`LogProbAlgebra`][quivers.core.algebras.LogProbAlgebra], each item's real value is then a
polynomial with nonnegative coefficients in the component's items, $x = f(x)$, whose coefficients
carry the values of the components below. Its least fixed point $\mu$ lies in $[0, \infty]^n$
(Etessami and Yannakakis 2009). [`solve_polynomial`][quivers.stochastic.recursion.solve_polynomial]
solves the components one after another by Newton's method, the decomposed method of Etessami and
Yannakakis (2009, section 6), whose convergence Esparza, Kiefer and Luttenberger (2010) bound.

**Theorem (lower bound).** Let $0 \le l \le \mu$, let $J$ be the Jacobian of $f$ at $l$ with
spectral radius below one, and let $d$ satisfy $(I - J)\,d \le f(l) - l$. Then $l + d \le \mu$.

*Proof.* Write $\mu = l + e$ with $e \ge 0$. A polynomial with nonnegative coefficients has
$f(l + e) \ge f(l) + J e$ (Taylor's expansion, every further term nonnegative), so
$(I - J)\,e \ge f(l) - l \ge (I - J)\,d$. Since the spectral radius of $J \ge 0$ is below one,
$(I - J)^{-1} = \sum_k J^k \ge 0$, so $e \ge d$. QED.

Each step solves $(I - J)\,d = f(l) - l$ in floating point, shortens $d$, and then checks the two
hypotheses in outward rounded arithmetic: $J v < v$ for $v = (I - J)^{-1} \mathbf 1 > 0$, which
bounds the spectral radius below one (Collatz and Wielandt), and each row of
$(I - J)\,d \le f(l) - l$, compared exactly in rationals. The coefficients enter by certified lower
bounds, so the theorem holds of a system below $f$ whose least fixed point is below $\mu$. The
iteration starts at $l = 0$ and stops when no step moves $l$, or when the budget's iterations are
spent.

**Upper bound.** A point $u$ with $f(u) \le u$ is a pre-fixed point, and the least fixed point is
the least pre-fixed point (Knaster and Tarski), so $\mu \le u$. The candidate is
$u = e + t\,(I - J)^{-1}\mathbf 1$ above the floating estimate $e$, for growing $t$, checked with
the coefficients' certified upper bounds. A component's interval $[l, u]$ thus holds $\mu$, and
the interval of an item above it holds its value because its coefficients' bounds come from the
intervals below. The value returned is the estimate, inside its interval.

At a critical system, where the Jacobian at $\mu$ has spectral radius one, the only pre-fixed
points may be $\mu$ itself: $x = x^2/2 + 1/2$ has $f(u) - u = (u - 1)^2/2$. No upper end then
certifies, the run's report has no guarantee and names the spectral radius and the condition
number $\|(I - J)^{-1}\|_\infty$ at the estimate, and the contract refuses the value or returns it
unestablished. A component with $f(l) > l$ in every item has $l < \mu$; below a finite least fixed
point of a strongly connected clean system the spectral radius is below one (Esparza, Kiefer and
Luttenberger 2010, section 4), so a spectral radius certified at least one there shows $\mu$
infinite, and the plan refuses it as divergent.

**Derivative.** At the fixed point $x = F(x, w)$ the implicit derivative is
$dx = (I - J)^{-1}\,\partial F / \partial w \; dw$, which exists when the spectral radius at
$\mu$ is below one; the solver certifies it from $J(u)\,v < v$, since $J$ grows with its argument.
The value is returned as $x + (I - J)^{-1}(F - F_{\text{detached}})$, which is $x$ and has that
derivative.

## References

- Etessami, K., and Yannakakis, M. (2009). Recursive Markov chains, stochastic grammars, and
  monotone systems of nonlinear equations. *Journal of the ACM* 56(1), 1-66.
  [doi:10.1145/1462153.1462154](https://doi.org/10.1145/1462153.1462154)
- Esparza, J., Kiefer, S., and Luttenberger, M. (2010). Computing the least fixed point of
  positive polynomial systems. *SIAM Journal on Computing* 39(6), 2282-2335.
  [doi:10.1137/090749591](https://doi.org/10.1137/090749591)
- Lehmann, D. J. (1977). Algebraic structures for transitive closure. *Theoretical Computer
  Science* 4(1), 59-76.
  [doi:10.1016/0304-3975(77)90056-1](https://doi.org/10.1016/0304-3975(77)90056-1)

::: quivers.stochastic.recursion
