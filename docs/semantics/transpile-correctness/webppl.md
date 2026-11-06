# WebPPL

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{WebPPL}$.

WebPPL expresses random choices and observations in JavaScript-like syntax.
A continuation is the remaining computation after a choice receives its
answer; transforming code into continuation-passing style lets the runtime
control that continuation. The repository's density probe fixes values and
evaluates distribution scores, yielding a joint log density at each supplied
parameter point.

## Semantics

DIPPL’s [continuation examples](https://dippl.org/chapters/03-enumeration.html#continuations) and [CPS transform in action](https://dippl.org/chapters/03-enumeration.html#cps-transform-in-action) explain the control transformation used by WebPPL [@goodman2014design]. The [QVR handler account](../effects.md#8-qiec-rows-and-lexical-handlers) states how a suspended request is resumed.

WebPPL's denotational semantics is the continuation-passing-style
trace semantics of Goodman and Stuhlmüller
[@goodman2014design]. A program is a JavaScript-syntax
WebPPL function whose `sample` and `observe` calls are reified by
the WebPPL compiler into CPS-transformed continuations; the trace
is the sequence of `(address, distribution, value)` triples
visited. The repository's numeric probe rewrites rendered `sample`
and `observe` sites into calls to each distribution object's
`score(value)` method at clamped parameter and data values.

## Unconstrained-space change of variables

Identity. WebPPL distributions are operationally defined; no
automatic reparameterization layer.

## Family parameterizations

WebPPL families use JavaScript-object keyword arguments:
`Dirichlet({alpha: ...})`, `Categorical({ps: ...})`,
`Gaussian({mu: ..., sigma: ...})`, `Bernoulli({p: ...})`, etc.
The QVR ↔ WebPPL arg mapping requires non-trivial aliases that
live in `FAMILY_META.arg_aliases["webppl"]`:

| QVR arg | WebPPL arg |
|---|---|
| `loc` | `mu` |
| `scale` | `sigma` |
| `concentration` (Dirichlet) | `alpha` |
| `probs` | `ps` |

The renamings are syntactic; $c_{F, \mathsf{WebPPL}} = 0$ for every
family with native WebPPL support.

## Per-construct emit

**Sample / observe.** `var <name> = sample(<Family>({args}));`
for latents; `observe(<Family>({args}), <obs>);` for observations.

**Plate.** `repeat(B, function() { return sample(<dist>); })` for
index-independent batches; `mapIndexed(function(m, _) { return
sample(<dist using m>); }, repeat(B, 0))` for index-dependent.
Under the execution interpretation [@goodman2014design], `repeat` denotes the documented
$B$-fold product measure.

**Marginalize.** The renderer evaluates each finite support value, combines
its prior and scope scores with `_qvr_logsumexp`, and passes the resulting
log weight to `factor`. Grouping is applied before the support reduction.
The resulting density sums over the hidden coordinate.

**Score / let / return.** [`factor(expr)`](https://webppl.readthedocs.io/en/master/inference.html#factor)
for score; `var <name> = <expr>;` for let; native `return`.

## Acceptance

* **Tier 1 structural.** Every emit has `var model = function(...)
  { ... };` whose body uses `sample`, `observe`, `factor`,
  `repeat`, and `mapIndexed`.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** [`node --check /dev/stdin`](https://nodejs.org/api/cli.html#--check)
  checks JavaScript syntax. Checking WebPPL primitives and their execution
  requires a WebPPL runtime.
* **Tier 3 numeric equivalence.** The rewritten `.score(value)`
  probe is compared with the QVR reference on selected fixture grids.
