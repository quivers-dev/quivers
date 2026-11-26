# Carriers

The carriers here compute statistics of all derivations at the cost of computing their weight.
Each is an [`OrderPreservingSemiring`][quivers.core.algebras.OrderPreservingSemiring] of the one
algebra hierarchy whose product commutes, with its elements packed on one trailing axis, so it
composes, sums, and reduces over empty axes like any other member, and its law claims carry
evidence: the laws about the carrier by the arguments below, the laws about the implementation by
the law grid.

## Expectation {#proof-expectation}

An element of [`ExpectationSemiring`][quivers.core.carriers.ExpectationSemiring] is $\langle p, r
\rangle$ with $p$ in a commutative semiring $K$ and $r$ in a $K$-semimodule $M$
([`Semimodule`][quivers.core.algebras.Semimodule]), with

$$
\langle p_1, r_1 \rangle \otimes \langle p_2, r_2 \rangle = \langle p_1 p_2,\ p_1 r_2 \oplus p_2 r_1 \rangle,
\qquad
\langle p_1, r_1 \rangle \oplus \langle p_2, r_2 \rangle = \langle p_1 \oplus p_2,\ r_1 \oplus r_2 \rangle,
$$

zero $\langle 0, 0 \rangle$ and unit $\langle 1, 0 \rangle$ (Eisner 2002; Li and Eisner 2009,
section 3).

**Theorem.** If $K$ is a commutative semiring and $M$ a semimodule over it, these operations form a
commutative semiring.

*Proof.* The sum is componentwise, so it is associative and commutative with unit
$\langle 0, 0 \rangle$. The product is commutative because $p_1 p_2 = p_2 p_1$ and $\oplus$
commutes. It is associative: both bracketings of three factors give
$\langle p_1 p_2 p_3,\ p_1 p_2 r_3 \oplus p_1 p_3 r_2 \oplus p_2 p_3 r_1 \rangle$ by the action's
associativity and distributivity. $\langle 1, 0 \rangle$ is a unit since $1 \cdot r = r$ and
$p \cdot 0 = 0$. Distributivity:
$\langle p, r \rangle (\langle a, b \rangle \oplus \langle c, d \rangle) =
\langle p(a \oplus c),\ p(b \oplus d) \oplus (a \oplus c) r \rangle$, which equals the sum of the
two products by distributivity of $K$ and of the action over both sums. The zero annihilates since
$0 \cdot r = 0$ and $p \cdot 0 = 0$. QED

Over derivations $d$ whose factors $e$ carry $\langle p_e, r_e \rangle$, the sum of products is
$\langle Z,\ \bigoplus_d \bigoplus_{e \in d} p(d \setminus e)\, r_e \rangle$, where
$p(d \setminus e)$ multiplies the weights of the other factors of $d$. With
$p_e = \exp(\theta \cdot f_e)$ and $r_e = p_e f_e$ the statistic is
$\sum_d p(d) \sum_{e \in d} f_e = \nabla_\theta Z$, the gradient that reverse mode computes
through the sum. Both are checked against enumeration of every path and against autograd.

**Star.** Where $p^* = 1 \oplus p p^*$ exists,
$\langle p, r \rangle^* = \langle p^*,\ p^* (p^* r) \rangle$: the statistic $q$ of a solution of
$X = 1 \oplus \langle p, r \rangle X$ must satisfy $q = p q \oplus p^* r$, and
$p (p^* p^* r) \oplus p^* r = p^* (p p^* \oplus 1) r = p^* p^* r$.

## Second-order expectation {#proof-second-order}

An element of [`SecondOrderExpectationSemiring`][quivers.core.carriers.SecondOrderExpectationSemiring]
is $\langle p, r, s, t \rangle$ with $r \in K^m$, $s \in K^k$ and $t \in K^{m \times k}$:

$$
\langle p_1, r_1, s_1, t_1 \rangle \otimes \langle p_2, r_2, s_2, t_2 \rangle =
\langle p_1 p_2,\ p_1 r_2 \oplus p_2 r_1,\ p_1 s_2 \oplus p_2 s_1,\
p_1 t_2 \oplus p_2 t_1 \oplus r_1 s_2^{\mathsf T} \oplus r_2 s_1^{\mathsf T} \rangle
$$

(Li and Eisner 2009, section 4). With each factor $e$ carrying
$\langle p_e, p_e r_e, p_e s_e, p_e r_e s_e^{\mathsf T} \rangle$, the sum over derivations has last
part $\bigoplus_d p(d)\, r(d)\, s(d)^{\mathsf T}$, where $r(d)$ and $s(d)$ sum the statistics of
the factors of $d$; this is checked against enumeration of every path.

**Theorem.** Over a commutative semiring $K$ these operations form a commutative semiring.

*Proof.* Write an element as $p + \sum_i r_i \varepsilon_i + \sum_j s_j \eta_j +
\sum_{ij} t_{ij} \varepsilon_i \eta_j$, a polynomial over $K$ in commuting indeterminates whose
monomials are among $1$, $\varepsilon_i$, $\eta_j$ and $\varepsilon_i \eta_j$. The product above
is the polynomial product followed by dropping every other monomial: the $\varepsilon_i \eta_j$
coefficient of the product collects $p_1 t_2$, $p_2 t_1$, $r_{1i} s_{2j}$ and $r_{2i} s_{1j}$. The
dropped monomials are closed under multiplication by any monomial, so dropping them from a product
gives the same result whether or not they were dropped from the factors first. Dropping is
therefore a homomorphism of sums and products from the commutative polynomial semiring onto these
elements, and the elements inherit associativity, commutativity, distributivity and the units. QED

**Star.** Where $p^*$ exists,
$\langle p, r, s, t \rangle^* = \langle p^*, p^{*2} r, p^{*2} s, p^{*2} t \oplus 2 p^{*3} r s^{\mathsf T}
\rangle$, with $2 = 1 \oplus 1$: substituting into $X = 1 \oplus x X$ and using only
$p^* = 1 \oplus p p^*$ and commutativity gives each part.

## Products {#proof-product}

[`ProductSemiring`][quivers.core.carriers.ProductSemiring] runs several commutative semirings side by
side, operation by operation. A direct product of commutative semirings is a commutative semiring,
since every law holds in each factor. The star of a product is the product of the stars, where each
exists.

## References

- Eisner, J. (2002). Parameter estimation for probabilistic finite-state transducers. *ACL*.
  [doi:10.3115/1073083.1073085](https://doi.org/10.3115/1073083.1073085)
- Li, Z., and Eisner, J. (2009). First- and second-order expectation semirings with applications to
  minimum-risk training on translation forests. *EMNLP*.
  [doi:10.3115/1699510.1699517](https://doi.org/10.3115/1699510.1699517)

::: quivers.core.carriers
