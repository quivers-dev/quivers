# WebPPL

WebPPL is a probabilistic programming language with the syntax of a subset of JavaScript
[@goodman2014design]. Its compiler rewrites a program into
[continuation-passing style](https://en.wikipedia.org/wiki/Continuation-passing_style), in which each
computation takes the rest of the computation, its *continuation*, as an argument, so that inference
algorithms can control what happens after each random choice. Continuation-passing style is how WebPPL is
implemented, not its semantics: a program's joint log density is the sum of the log densities of the choices
of an execution ([Transpilation correctness](index.md)).

## How the test evaluates a log density

The test probe rewrites each emitted `sample` and `observe` into a call of the distribution's
`score(value)` method at the fixed parameter values and data, and sums the results, which gives the joint log
density at each point.

## Change of variables

WebPPL's distributions are defined by their samplers and densities, with no automatic change of variables, so
$\Psi_{\mathsf{WebPPL}} = \mathrm{id}$.

## Family parameterizations

WebPPL's families take a JavaScript object of named arguments: `Dirichlet({alpha: ...})`,
`Categorical({ps: ...})`, `Gaussian({mu: ..., sigma: ...})`, `Bernoulli({p: ...})`, and so on. The names are
recorded in `FAMILY_META.arg_aliases["webppl"]`:

| QVR argument | WebPPL argument |
|---|---|
| `loc` | `mu` |
| `scale` | `sigma` |
| `concentration` (Dirichlet) | `alpha` |
| `probs` | `ps` |

The renaming changes no value, so $c_{F, \mathsf{WebPPL}} = 0$ for every family WebPPL has.

## Emitted constructs

**`sample` and `observe`.** `var <name> = sample(<Family>({args}));` for a latent variable and
`observe(<Family>({args}), <obs>);` for an observation.

**Plates.** `repeat(B, function() { return sample(<dist>); })` for copies of one distribution, and
`mapIndexed(function(m, _) { return sample(<dist using m>); }, repeat(B, 0))` for copies whose parameters
depend on the index. The density of the $B$ independent draws is the product of their densities.

**`marginalize`.** The renderer evaluates each value of the latent variable, combines its prior and the scores
of the scope with `_qvr_logsumexp`, and passes the log weight to `factor`. Grouping is applied before the
reduction, and the density has the latent variable summed out.

**`score`, `let`, `return`.** [`factor(expr)`](https://webppl.readthedocs.io/en/master/inference.html#factor)
for a score, `var <name> = <expr>;` for a `let`, and JavaScript's `return`. A `let` whose value has no axis is
one expression, whatever plated names it reads at an index. A `let` whose value has a leading axis is built one
row at a time, `mapIndexed(function(i, _) { return <row i>; }, repeat(N, 0))`, where the row is the expression
read at `i`: each operand with the leading axis is read at `i`, and an operand of lower rank passes through. An
operator on an array that a row still holds goes through the elementwise helper `_qvr_bcast`, since a
JavaScript operator would turn the array into `NaN`. A site over a plate reads, at its row, every name bound
over the same leading axis.

## Evidence

- **Level 1.** Every emitted model is a `var model = function(...) { ... };` whose body uses `sample`,
  `observe`, `factor`, `repeat`, and `mapIndexed`, and calling the stages directly or composed gives the same
  output.
- **Level 2.** [`node --check /dev/stdin`](https://nodejs.org/api/cli.html#--check) checks the JavaScript
  syntax. Checking WebPPL's primitives and running the program need a WebPPL runtime.
- **Level 3.** The rewritten `.score(value)` probe is compared with the QVR density on selected fixture grids.
