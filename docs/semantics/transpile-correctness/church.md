# Church

Church is a probabilistic programming language based on Scheme, in which a program is an expression whose
random choices pass values to the rest of the computation, and in which observations and factors change the
weight of an execution [@goodman2008church]. Church's semantics is given by sampling: running a program
produces an execution, a sequence of random choices with their addresses and values, and the joint log
density of an execution is the sum of the log densities of its choices [@goodman2014design]. The probabilistic
choice operator on which such languages rest was given a semantics in a probability monad by Ramsey and
Pfeffer [@ramsey2002stochastic]. The renderer writes programs whose execution under this reading gives the
QVR density ([Transpilation correctness](index.md)).

No external Church implementation is fixed for the tests, and there is no Church image in the shared Docker
matrix, so separate tests check selected emitted programs.

## Change of variables

Church's distributions are defined by their samplers and densities, with no automatic change of variables, so
$\Psi_{\mathsf{Church}} = \mathrm{id}$.

## Family parameterizations

Church uses the usual names of distributions (`gaussian`, `dirichlet`, `categorical`, `bernoulli`, `beta`, and
others). The map is the identity for every family with a Church counterpart, and
$c_{F, \mathsf{Church}} = 0$.

## Emitted constructs

**`sample` and `observe`.** `(define <name> (sample (<family> <args>)))` for a latent variable and
`(observe (<family> <args>) <obs>)` for an observation.

**Plates.** `(map (lambda (m) (sample (<family> <args>))) (iota B))` makes a list of $B$ independent draws,
and an observed plate is `(for-each (lambda (n) (observe ...)) (iota N))`. The density of the $B$ independent
draws is the product of their densities.

**`marginalize`.** The renderer scores the scope once for each value of the latent variable, combines the
weighted alternatives with `log-sum-exp`, and adds the result with `factor`. A grouped block sums the rows of
each group before the reduction, so the latent variable is summed out.

**`score`, `let`, `return`.** `(factor <expr>)` for a score [@goodman2014design], `(define <name> <expr>)` for a
`let`, and the last expression of the `(define (model ...) ...)` body as the returned value.

**Refusals.** The renderer refuses a matrix literal with `arg:matrix-literal`, since Scheme has no standard
matrix form, and wrapper families that need the inner morphism's `~ Family(...)` clause, such as `Truncated`
by rejection sampling.

## Evidence

- **Level 1.** Every emitted model is a `(define (model ...) ...)` with `sample` and `observe` calls, and
  calling the stages directly or composed gives the same output.
- **Level 2.** No external Church parser is used. A check that the parentheses balance catches unmatched
  delimiters, and checking forms and primitive names would need a chosen Church implementation.
- **Level 3.** Church is not in the shared Docker matrix. Tests for Church check selected emitted programs.
