# Contracts

A query over a rule presentation asks for one observable of the derivations of a judgment, under
one interpretation of the rules. What an executor may do to answer it depends on the laws of the
[weight algebra](https://en.wikipedia.org/wiki/Semiring), on how recursion is solved, on how
accurate the answer has to be, and on what provenance is kept. The contract records state each of
these as data that crosses the [wire codec](serialization.md), and the checkers refuse every
combination that does not fit, with the reason.

| record | what it states |
|---|---|
| [`QuerySpec`][quivers.qiec.contracts.QuerySpec] | presentation, interpretation, observable, input types, inputs of the gradient, batch axes, retention |
| [`LawClaim`][quivers.core.laws.LawClaim] | a law of a subject, with evidence that is proved, decided, tested or assumed |
| [`Recursion`][quivers.qiec.contracts.Recursion] | acyclic, linear cyclic, monotone polynomial, bounded by the model, or unrolled |
| [`AccuracyContract`][quivers.qiec.contracts.AccuracyContract] | numerics, target and metric, guarantee, controls, budget, failure policy, derivative |
| [`ComplexityClaim`][quivers.qiec.contracts.ComplexityClaim] | bounds per phase and resource, each with its optimality status |
| [`WorkReport`][quivers.qiec.contracts.WorkReport] | counters of the work of one phase |
| [`Retention`][quivers.qiec.contracts.Retention] | no provenance, parent counts, requested evidence, or the packed forest |
| [`AccuracyReport`][quivers.qiec.contracts.AccuracyReport] | what a run reached, why it stopped, what it cost, which claims it used |

## Observables

Write $[\![j]\!] = \sum_{d \in D(j)} w(d) \cdot \mathrm{fold}(d)$ for the detailed meaning of a
judgment $j$: the formal sum over its derivations $D(j)$, with $w(d)$ the product of the axiom
occurrence and rule weights. An observable is a map $h$ from detailed meanings to results. A
summary map $s$ licenses aggregation when it preserves the sum of alternatives and commutes with
every rule, and $h$ factors through it. Each observable states what it needs:

| observable | laws of the algebra | other needs |
|---|---|---|
| [`Support`][quivers.qiec.contracts.Support] | semiring, zero-sum-free, no zero divisors | no derivative |
| [`Weight`][quivers.qiec.contracts.Weight] | semiring | |
| [`Best`][quivers.qiec.contracts.Best] | semiring, selective sum | a tie rule for its derivative |
| [`KBest`][quivers.qiec.contracts.KBest] | semiring, selective sum, superior rules | a tie rule for its derivative |
| [`Outcome`][quivers.qiec.contracts.Outcome] | semiring | a key computation |
| [`WeightedProofRows`][quivers.qiec.contracts.WeightedProofRows] | ordered product with unit and absorbing zero, and associative commutative sum and division when normalized | finitely many derivations for a guaranteed result |
| [`Statistic`][quivers.qiec.contracts.Statistic] | semiring, semimodule action | a statistic computation |
| [`Marginal`][quivers.qiec.contracts.Marginal] | semiring, and division when normalized | |
| [`Sample`][quivers.qiec.contracts.Sample] | semiring, division | no derivative |
| [`Fold`][quivers.qiec.contracts.Fold] | semiring, and linearity of the fold | the packed forest |
| [`NonlinearFold`][quivers.qiec.contracts.NonlinearFold] | semiring | the packed forest; finitely many derivations for a guaranteed result |
| [`Forest`][quivers.qiec.contracts.Forest] | none | the packed forest; no derivative |

A nonlinear fold has no summary in general: the alternatives $-1$ and $1$ sum to $0$, whose
square is $0$, while their squares sum to $2$.

## Law claims

A claim's status is read from its evidence, so the two cannot disagree.
[`Proved`][quivers.core.laws.Proved] cites a written argument and
[`Decided`][quivers.core.laws.Decided] names a procedure that decides the law on the carrier:
every tuple of a finite carrier, a normal form of polynomial terms, or quantifier elimination over
the reals. These two are *established*. [`Tested`][quivers.core.laws.Tested] evaluates
finitely many points of a larger carrier, which can refute a law and cannot establish it, and
[`Assumed`][quivers.core.laws.Assumed] is a declaration. An exact, certified or
probabilistic result needs every law it uses to be established; a heuristic result needs every
law it uses to be claimed. A claim is about the mathematical carrier: a floating point
implementation satisfies no algebraic law exactly, and its rounding belongs to
[`Numerics`][quivers.qiec.contracts.Numerics].

## Recursion

| contract | solution | exact by a finite method | derivative |
|---|---|---|---|
| [`Acyclic`][quivers.qiec.contracts.Acyclic] | unique, one relaxation per hyperedge | yes | reverse accumulation |
| [`LinearCyclic`][quivers.qiec.contracts.LinearCyclic] | least solution $A^* b$, needs the star law | yes | $A^* \, dA \, A^*$ on the open domain of convergence |
| [`MonotonePolynomial`][quivers.qiec.contracts.MonotonePolynomial] | least fixed point in the omega-continuous completion; Newton needs a real embedding (certified) or an idempotent star (exact) | no: at best certified | implicit, if the spectral radius of the Jacobian at the solution is below one; or unrolled |
| [`Bounded`][quivers.qiec.contracts.Bounded] | the bound is part of the model | yes | exact for the bounded model |
| [`Unrolled`][quivers.qiec.contracts.Unrolled] | the first $k$ iterates, a lower bound in an ordered algebra | no | of the $k$ iterates and of nothing else |

A bound on constructor depth or length changes the model. An iteration limit, a fuel count or a
tolerance does not, and it never changes the value a query denotes.

## Accuracy

The fields of an [`AccuracyContract`][quivers.qiec.contracts.AccuracyContract] are separate: a
method parameter ([`Control`][quivers.qiec.contracts.Control]) is never read as a guarantee, a
[`ResourceBudget`][quivers.qiec.contracts.ResourceBudget] is never an error bound, and the
derivative has its own semantics. The default is [`Exact`][quivers.qiec.contracts.Exact] with
[`FailurePolicy.REFUSE`][quivers.qiec.contracts.FailurePolicy].

A guarantee is one of: [`Exact`][quivers.qiec.contracts.Exact], the mathematical value up to the
rounding in `Numerics`; [`Certified`][quivers.qiec.contracts.Certified], a bound that holds on
every run; [`Probabilistic`][quivers.qiec.contracts.Probabilistic], a bound that fails with
probability at most $\delta$; or [`Heuristic`][quivers.qiec.contracts.Heuristic]. A certified
bound that a run reports carries its [`Certificate`][quivers.qiec.contracts.Certificate]: an
outward rounded enclosure, a duality gap, an argument backed by an established claim, or a
composition of certified parts. A residual below a tolerance is none of these.

Every metric is read with the computed value first and the target second. Absolute error,
absolute error of log weights and total variation satisfy the
[triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality). Relative error does not:
with target $1$, an intermediate value $100$ and a computed value $101$ it is $100$, against
$0.01 + 99$ through the intermediate value. Kullback-Leibler divergence does not: for Bernoulli
laws with means $0.1$, $0.5$ and $0.9$ it is $1.758$ from the first to the third, against
$0.368 + 0.511$ through the second. Since the logarithm maps the nonnegative reals isomorphically
onto the log weights, a relative error of probabilities is an absolute error of log weights.

## The composition law

[`compose`][quivers.qiec.contracts.compose] gives the attainment of an operation
$g : X_1 \times \dots \times X_n \to Y$ applied to results that each have an attainment. Write
$x_i$ for the targets, $\hat x_i$ for the computed values, $y = g(x_1, \dots, x_n)$, and
$\hat y = \hat g(\hat x_1, \dots, \hat x_n)$ for the computed result, where $\hat g$ is the floating
point implementation of $g$. Let $d$ be the metric of $Y$ and $d_i$ the metric of part $i$.

**Hypotheses.**

1. *Lipschitz constants.* For each argument with a constant $L_i \ge 0$, and all points $z, z'$
   of a product region $R$ that contains $(x_1, \dots, x_n)$ and $(\hat x_1, \dots, \hat x_n)$ and
   that differ only in coordinate $i$,
   $d(g(z), g(z')) \le L_i \, d_i(z_i, z'_i)$. This is
   [Lipschitz continuity](https://en.wikipedia.org/wiki/Lipschitz_continuity) in each argument.
2. *Rounding of the operation.* $d(\hat y, g(\hat x_1, \dots, \hat x_n)) \le r$.
3. *Parts.* $d_i(\hat x_i, x_i) \le e_i + \rho_i$, where $e_i$ is the bound of the part's
   guarantee ($0$ when exact) and $\rho_i$ its rounding bound. For a certified part this holds on
   every run; for a probabilistic part, outside an event $F_i$ with $P(F_i) \le \delta_i$.
4. *Triangle inequality.* $d(a, c) \le d(a, b) + d(b, c)$ on $Y$.

**Theorem.** Under these hypotheses, outside $F_1 \cup \dots \cup F_n$,

$$
d(\hat y, y) \;\le\; \sum_i L_i e_i \;+\; \Big( r + \sum_i L_i \rho_i \Big),
$$

and $P(F_1 \cup \dots \cup F_n) \le \sum_i \delta_i$.

*Proof.* Let $h_0 = (\hat x_1, \dots, \hat x_n)$ and
$h_k = (x_1, \dots, x_k, \hat x_{k+1}, \dots, \hat x_n)$, so $h_n = (x_1, \dots, x_n)$. Each
coordinate of each $h_k$ is a target or a computed value, so every $h_k$ lies in the product region
$R$, and $h_{k-1}$, $h_k$ differ only in coordinate $k$, where they hold $\hat x_k$ and $x_k$.
Applying the triangle inequality $n$ times along $\hat y, g(h_0), g(h_1), \dots, g(h_n) = y$,

$$
d(\hat y, y) \le d(\hat y, g(h_0)) + \sum_{k=1}^{n} d(g(h_{k-1}), g(h_k))
\le r + \sum_{k=1}^{n} L_k \, d_k(\hat x_k, x_k)
$$

by hypotheses 2 and 1. Outside $\bigcup_i F_i$ every part satisfies hypothesis 3, and $L_k \ge 0$,
so the sum is at most $\sum_k L_k (e_k + \rho_k)$, which rearranges to the bound. The probability
of the union is at most the sum of the probabilities by the
[union bound](https://en.wikipedia.org/wiki/Boole%27s_inequality), which needs no independence.
QED

**Corollary (the guarantee of a composite).** The composite takes the weakest guarantee of its
parts, in the order exact, certified, probabilistic, heuristic:

1. every part exact: the composite is exact, with rounding $r + \sum_i L_i \rho_i$ when every term
   is known, and unknown otherwise;
2. every part exact or certified, each certified part with a constant in its own metric: certified
   with bound $\sum_i L_i e_i$, and its certificate is the
   [`Composition`][quivers.qiec.contracts.Composition] of the terms, which can be checked again;
3. some part probabilistic, the rest as in 2: probabilistic with bound $\sum_i L_i e_i$ and
   failure probability $\sum_i \delta_i$, when this is below one; at one or more the statement
   says nothing and the composite is heuristic;
4. some part that is not exact without a constant in its own metric, some heuristic part with a
   constant above zero, or a metric of $Y$ without the triangle inequality while some part is not
   exact: heuristic.

Case 4 is sharp. With $g$ the identity, a heuristic part may be arbitrarily far from its target,
so no finite bound holds; without the triangle inequality the step of the proof fails, as the two
counterexamples above show. A heuristic part whose constant is zero does not move the result and is
left out.

**Local tolerances do not make a global bound.** A run that stops because a residual fell below
its tolerance has no certificate, so
[`attained_guarantee`][quivers.qiec.contracts.attained_guarantee] makes it heuristic, and by case
4 so is every composite that depends on it. The residual can be small while the error is large:
Kleene iteration of a cycle of weight $0.999$, stopped when the log weight changes by less than
$10^{-3}$, returns $6.2158$ after $693$ iterates, while the exact log weight is
$\log 1000 = 6.9078$.

**Derivatives of a composite.** The derivative of a composite is the weakest of the derivatives
of its parts and of the operation itself, component by component (below), with two differences
from the meet. The unrolled loops of all parts are counted together, since each part unrolls its
own loop. An estimate stays unbiased only when every estimated part is unbiased and enters an
argument in which the operation is affine, because an expectation commutes with sums and
deterministic multiples and not with other functions: for a Bernoulli draw $X$ with mean $p$,
$X$ is unbiased for $p$, while $E[X^2] = p$ differs from $p^2$. Otherwise the bias is unknown. The
composite is almost everywhere regular when every part and the operation are: these are
computations whose primitives are analytic and whose decisions are sign tests of computed values,
a class closed under composition, and the zero set of an analytic function that is not
identically zero has measure zero. So the composite is at most as strong as each part.

## Derivative semantics and reports

A [`DerivativeSemantics`][quivers.qiec.contracts.DerivativeSemantics] is a product of components,
each ordered from strong to weak: whether a derivative exists, whether regular points have full
measure, the tie rule of every selection, the hypotheses of implicit derivatives, the iteration
counts of unrolled loops, the estimator of each random site, and the bias of the estimate.
[`at_least`][quivers.qiec.contracts.DerivativeSemantics.at_least] compares them component by
component, so it is a preorder with no derivative at the bottom, and
[`meet`][quivers.qiec.contracts.DerivativeSemantics.meet] is its greatest lower bound. Two tie
rules have no meet: mixing them makes a result depend on bracketing, so there is one rule,
[`FIRST_MAXIMIZER`][quivers.core.laws.FIRST_MAXIMIZER].

The forms of a derivative report are points of this order: exact at regular points, a selected
branch at ties, implicit under conditions, unrolled, an estimator per site with its bias, and none
with its reason. A [`DerivativeReport`][quivers.qiec.contracts.DerivativeReport] holds one
semantics per output and input of a run, its summary is the meet of its entries, and an
[`AccuracyReport`][quivers.qiec.contracts.AccuracyReport] requires the derivative of its attainment
to be that summary. An estimator report is the estimator and bias components of the same record,
so a derivative report of a handled program is an instance of these records, not a second kind.

## Reports and failure

Three rules hold of every report that
[`report_refusals`][quivers.qiec.contracts.report_refusals] admits:

- a resource budget is never an error bound: no guarantee accepts a budget as its bound, and an
  exhausted run claims only the bound of a certificate it computed;
- a residual is never a certificate;
- exhaustion never returns a value labelled exact.

[`settle`][quivers.qiec.contracts.settle] applies the failure policy: the value when the report
meets the contract, [`ContractNotMet`][quivers.qiec.contracts.ContractNotMet] under
`REFUSE`, and an [`Unestablished`][quivers.qiec.contracts.Unestablished] wrapper under
`RETURN_UNESTABLISHED`. The wrapper is not a number, so nothing that expects a weight accepts it.
No policy truncates silently, switches method, or returns zero.

## Admissibility

[`admissibility`][quivers.qiec.contracts.admissibility] returns every reason a query, its
recursion, its contract and the available claims do not fit; `report_refusals` does the same for
a report, and [`complexity_refusals`][quivers.qiec.contracts.complexity_refusals] for a complexity
claim.

| refusal | refused combination |
|---|---|
| `missing-claim` | a law the observable, the recursion or the executor needs has no claim |
| `claim-not-established` | a needed law is only tested or assumed, and the guarantee is not heuristic |
| `exact-without-finite-method` | an exact guarantee for a monotone polynomial or unrolled system |
| `derivative-unavailable` | a derivative the recursion does not give |
| `gradient-without-inputs` | a derivative with respect to nothing |
| `inputs-without-gradient` | inputs to differentiate, and no derivative requested |
| `observable-not-differentiable` | a derivative of support, a sample or the forest |
| `selection-without-tie-rule` | a derivative of a best derivation without the tie rule it accepts |
| `forest-not-retained` | a fold or the forest without the packed forest |
| `infinite-derivations` | a guaranteed nonlinear fold of a system that can have infinitely many derivations |
| `request-with-certificate` | a requested bound that carries a certificate |
| `duplicate-control` | a control set twice |
| `early-stop-labelled-exact` | a run stopped by its budget or a control, reported exact |
| `certified-without-certificate` | an attained bound without its certificate |
| `bound-below-certificate` | an attained bound below what its certificate shows |
| `argument-not-established` | a bound argued from a claim that is only tested or assumed |
| `certificate-metric` | an enclosure outside absolute errors, or a duality gap outside objective gaps |
| `used-claim-not-established` | a guaranteed result from a run that used a claim that is not established |
| `exact-for-inexact-recursion` | an exact result of a monotone polynomial or unrolled system |
| `unmet-misreported` | unmet requirements that differ from the ones the contract and the attainment give |
| `incomplete-claim` | a phase with neither a time and a space bound nor a declaration that it does not apply, or with both |
| `undeclared-parameter` | a bound in a name the claim does not declare |
| `lower-bound-mismatch` | an optimality proof whose lower bound is not the bound |

## Complexity claims and work

A [`ComplexityClaim`][quivers.qiec.contracts.ComplexityClaim] states time and space bounds for
construction, execution, differentiation and refitting in named parameters, in a stated
[`ComputationModel`][quivers.qiec.contracts.ComputationModel]. Each bound has a status:
[`ProvedOptimal`][quivers.qiec.contracts.ProvedOptimal] with the matching lower bound and its
model, [`Conditional`][quivers.qiec.contracts.Conditional] on a hypothesis,
[`BestKnown`][quivers.qiec.contracts.BestKnown], [`Unresolved`][quivers.qiec.contracts.Unresolved]
as a research obligation, or [`Deficient`][quivers.qiec.contracts.Deficient] with a better bound
and a reviewed justification. A [`WorkReport`][quivers.qiec.contracts.WorkReport] counts the work
of one phase in elements inside tensor operations, and a run exceeds its budget when the counts,
added over phases, or the peaks, compared phase by phase, pass a limit.

::: quivers.qiec.contracts
