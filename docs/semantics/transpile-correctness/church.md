# Church

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Church}$.

Church expresses a generative computation using Scheme-style functions.
A random choice supplies a value to the remaining computation, and an
observation or factor changes its weight. The source can be read through
those operations directly; the equations below describe their contributions
to the joint density.

## Semantics

Church's denotational semantics is the stochastic lambda calculus
of Goodman, Mansinghka, Roy, Bonawitz, and Tenenbaum
[@goodman2008church]. A program is a Scheme
expression whose [`sample` invocations introduce randomness](https://dippl.org/chapters/03-enumeration.html#exploring-a-random-computation) [@goodman2014design]; the
trace is a sequence of (address, distribution, value) triples. The
joint log-density is the sum of the per-site log-density contributions.

The renderer is currently unpinned to an external Church implementation.
Its tests audit selected emitted programs; the shared Docker numeric matrix
covers the other targets.

## Unconstrained-space change of variables

Identity. Church's distributions are operationally defined and do
not carry an automatic reparameterization layer.

## Family parameterizations

Church uses standard distribution names (`gaussian`, `dirichlet`,
`categorical`, `bernoulli`, `beta`, ...). The QVR ↔ Church mapping
is identity for every family that has a Church counterpart;
$c_{F, \mathsf{Church}} = 0$.

## Per-construct emit

**Sample / observe.** `(define <name> (sample (<family>
<args>)))` for latents; `(observe (<family> <args>) <obs>)` for
observations.

**Plate.** `(map (lambda (m) (sample (<family> <args>))) (iota
B))` produces a length-$B$ list of i.i.d. samples; observed plates
use `(for-each (lambda (n) (observe ...)) (iota N))`. By
the evaluation-history interpretation [@goodman2008church], `map` over `iota` denotes the product
measure of $B$ independent samples.

**Marginalize.** The renderer emits one scored scope per finite support value,
combines the weighted alternatives with `log-sum-exp`, and contributes the
result through `factor`. Grouped scopes sum rows within a group before that
reduction, eliminating the hidden coordinate.

**Score / let / return.** `(factor <expr>)` for score (per
[@goodman2014design]); `(define <name> <expr>)` for let;
the trailing expression in the `(define (model ...) ...)` body
is the return value.

**Limitations.** Church's renderer raises
`UnsupportedConstruct(["arg:matrix-literal"])` on matrix-literal
arguments (no canonical Scheme matrix form). Wrapper families
that need access to an inner morphism's `init_family` (Truncated
via rejection sampling) are also not supported.

## Acceptance

* **Tier 1 structural.** Every emit has `(define (model ...) ...)`
  with `sample` / `observe` invocations.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** No canonical external Church compiler
  is exercised by the shared syntax suite. Its parenthesis-balance check
  catches unmatched delimiters; validating forms and primitive names
  requires a chosen Church implementation.
* **Tier 3 numeric equivalence.** Not part of the shared Docker matrix;
  Church-specific audit tests cover selected emitted programs.
