# Programs as kernels

A `program` declaration

```
program P[static_params](value_params) : τ₁ -> τ₂
    s₁
    ⋮
    sₙ
    return e
```

denotes, at fixed static and runtime arguments, an [s-finite kernel](setting.md#4-finite-and-s-finite-kernels)
$\llbracket P \rrbracket$ from $\llbracket \tau_1 \rrbracket$ to $\llbracket \tau_2 \rrbracket$
[@staton2017commutative]. Each step denotes a kernel from the values bound before it to the
values bound after it, and the body denotes the composite of these kernels. A body with no
`observe` or `score` step denotes a Markov kernel [@kozen1981semantics]. Otherwise the kernel
multiplies the probability of each run by its likelihood, so its total mass need not be one, and
normalizing it, when the mass is finite and positive, gives the posterior. The typing rules for
steps and their elaboration to the Quivers Indexed Effect Core (QIEC) are on
[Typing and checking §3](typing.md#3-program-contexts) and [§8](typing.md#8-program-elaboration).

For the program

```text
sample z <- Normal(loc=0.0, scale=1.0)
observe y <- Normal(loc=z, scale=0.5)
return z
```

with a given value of `y`, the denotation is the measure on the value $z$ returned with density
$\mathcal N(z; 0, 1)\, \mathcal N(y; z, 0.5^2)$ with respect to Lebesgue measure, the prior times
the likelihood. Its total mass is the marginal likelihood of $y$, and dividing by it gives the
posterior of $z$.

## 1. Contexts and the composition of steps {#1-the-giry-monad}

Write $\Gamma = \llbracket \tau_1 \rrbracket$ for the input space. After the steps
$s_1, \dots, s_i$ have bound the variables $v_1, \dots, v_i$ with spaces $X_1, \dots, X_i$, the
*context* is the product space

$$
\Phi_i = \Gamma \times X_1 \times \cdots \times X_i,
\qquad \Phi_0 = \Gamma,
$$

and a point $\phi \in \Phi_i$ assigns a value to each name in scope. This is the environment
$\rho_{\mathrm{var}}$ of [Setting §6](setting.md#5-environments). Each step denotes a kernel
$\mathcal S\llbracket s_i \rrbracket$ from $\Phi_{i-1}$ to $\Phi_i$, and kernels compose by
integrating over the intermediate value, as in
[Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels):

$$
(k ; l)(\phi, C) = \int l(\psi, C)\, k(\phi, \mathrm{d}\psi).
$$

The body denotes

$$
\mathcal{B}\llbracket s_1; \cdots; s_n; \mathsf{return}\ e \rrbracket
= \mathcal{S}\llbracket s_1 \rrbracket ; \mathcal{S}\llbracket s_2 \rrbracket ; \cdots ; \mathcal{S}\llbracket s_n \rrbracket ; \mathsf{ret}_e,
$$

where $\mathsf{ret}_e$ is the deterministic kernel that sends $\phi$ to the value of $e$. A step that
draws a value is a Markov kernel, the Kleisli composite of the [Giry monad](setting.md#3-standard-borel-spaces-and-markov-kernels)
[@giry1982categorical], and Moggi's account of programs as Kleisli morphisms applies to the body
without `observe` and `score` [@moggi1991notions]. A step that conditions multiplies the measure
by a likelihood and is an s-finite kernel. Staton shows that s-finite kernels are closed under this
composition and that it is associative [@staton2017commutative], so the composite is defined for
every body.

### Named distribution values {#named-distribution-values}

A declared kernel can fill a sampleable argument in a measure constructor.
Fixed initializers retain their authored parameter names and values. A learned
kernel reads the original program input, as an ordinary draw from that kernel
does; aliases preserve the same kernel and its parameters.

```qvr
object Input : Real 2
object Output : Real 3
morphism base : Input -> Output [param_source=linear] ~ Normal
define alias = base
program model : Input -> Output
    sample value <- Restrict(base=alias, low=-1.0, high=1.0)
    return value
export model
```

Here `value` is a three-coordinate event at each two-coordinate conditioning
input. Input batch axes remain batch axes when the kernel becomes a distribution
value. The native constructor retains one conditioning block, shares the
registered kernel parameters across references, and evaluates the kernel at the
actual input. It introduces no probing samples or additional random sites.

Checked reference execution uses the same parameter-map telescope as an ordinary
named-kernel draw: this example requires `base_param_weight` and
`base_param_bias` alongside the domain input. These values must be the actual
kernel weights when comparing the reference and native executions. In the native
`Model` interface, those parameters remain owned by the registered kernel and
are excluded from external-data requirements. An optional typed declaration of
a parameter-map input checks its type while retaining that ownership; supplying
it as external data is an error. A lexical binding takes precedence over a
module-level name, and cyclic aliases receive a source-located diagnostic.

A kernel declared with `[weights=latent]` instead reads the program's sampled
network weights. Nested constructors and nominal aliases retain those same
sites: repeated references draw each weight once per run and share its value.
The compiler uses the ordinary latent-network evaluation before constructing
the conditional family, so restrictions and normalization preserve gradients
through every network head. These weights are latent variables, rather than
required external inputs or unused point estimates.

```qvr
object Input : Real 2
object Output : Real 1
morphism base : Input -> Output [weights=latent, prior_scale=0.5] ~ Normal
define alias = base
program model : Input -> Output
    sample value <- Restrict(base=alias, low=-1.0, high=1.0)
    return value
export model
```

Checked reference execution draws the same weight sites, or accepts their
explicit values for a joint-density comparison. Each run has its own drawn
weights; two aliases within that run do not introduce independent copies.

### Independent product events {#independent-product-events}

A distribution retains a **batch/event partition** $(B,E)$: batch positions
index separate measures, while event coordinates belong to one outcome.
For $B=(b_1,\ldots,b_p)$ and $0\leq n\leq p$, `Independent(base, n)` has

$$
B'=(b_1,\ldots,b_{p-n}),\qquad
E'=(b_{p-n+1},\ldots,b_p)\mathbin{+\!+}E.
$$

The case $n=0$ retains $(B,E)$ without reducing a batch axis. At a remaining
batch position $i$, the log density and total log mass are respectively

$$
\log p_{\mathrm{Independent}(d,n)}(y_i)
 = \sum_j \log p_{d_{i,j}}(y_{i,j}),\qquad
\log |\mathrm{Independent}(d,n)_i|
 = \sum_j \log |d_{i,j}|,
$$

where $j$ ranges over the reinterpreted batch positions. An intrinsic event
of $d$ is reduced by its own density, so this sum does not reduce it twice.
Raw restrictions retain their masses through this construction; normalization
at a sampling or scoring boundary divides by the resulting total mass once.

Checked source reserves the trailing base batch axes before consuming the
explicit enclosing observation axes. Thus a parameter declared
`Tensor[Real]([Row, Coordinate])` can supply one vector per `Row`, while
`Tensor[Real]([Coordinate])` supplies a vector shared across rows. Equal extents
leave these roles distinct. The runtime `(...)` telescope declares the values
and their shapes; static `[...]` binders specialize dimensions under the same
rules as other program parameters. Neither declaration creates a random effect.
`Independent` itself is a pure constructor, and its axis count must be a
nonnegative integer literal no larger than the checked base batch rank.
Native execution and the reference interpreter retain this partition through
nested restrictions, normalization, and mixtures. External renderers currently
refuse the nested distribution arguments explicitly.

## 2. Steps

### 2.1 Draws {#21-bind}

A draw

```
sample v <- F(args)
```

extends the context with a value drawn from the family $F$:

$$
\mathcal{S}\llbracket \mathsf{sample}\ v \leftarrow F(\bar a) \rrbracket(\phi, B \times C)
= \mathbf{1}_B(\phi) \int_C p_F\bigl(y ;\, \theta_F(\bar a, \phi)\bigr)\, \nu_F(\mathrm{d}y),
$$

a Markov kernel from $\Phi$ to $\Phi \times \llbracket \mathrm{cod}(F) \rrbracket$. Here
$\theta_F(\bar a, \phi)$ is the parameter of $F$ computed from the arguments $\bar a$, which may
read the names bound in $\phi$, $p_F$ is the family's density, and $\nu_F$ is its reference measure,
Lebesgue measure or counting measure. The kernel keeps $\phi$ and adds the new value.

The arguments of a family are named, and they choose one complete parameterization:
`Normal(loc=m, scale=s)` gives both Normal parameters, while `Categorical(probs=p)` and
`Categorical(logits=l)` choose the probability and the logit parameterization of the categorical.
Positional arguments to a family, an incomplete parameterization, a repeated name, and a mixture of
two parameterizations are compile errors. Calls of computations and morphisms remain positional,
since their signatures fix the order of their arguments.

A draw may name the components of a product value: `sample (v_1, ..., v_m) <- F(args)` requires
$\llbracket \mathrm{cod}(F) \rrbracket = K_1 \times \cdots \times K_m$ and has the same denotation as
the draw of one value $v$, with $v_i$ the $i$-th projection of $v$.

#### Inverse links as value maps

An inverse link maps a real predictor to a response parameter before
the observation kernel is constructed. For a positive parameter, the
squareplus inverse link is

$$
g^{-1}(\eta)=\frac{\eta+\sqrt{\eta^2+4}}{2},
\qquad g(x)=x-\frac1x,\quad x>0.
$$

For a probability parameter, write $L(\eta)=\log(1+\exp\eta)$.
The softit inverse link is

$$
h^{-1}(\eta)=\frac{L(\eta)}{1+L(\eta)},
\qquad h(p)=\log\!\left(\exp\frac{p}{1-p}-1\right),\quad 0<p<1.
$$

These maps follow the released brms definitions of
[squareplus](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/inst/chunks/fun_squareplus.stan)
and [softit](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/inst/chunks/fun_softit.stan).
Their application is a deterministic computation on runtime values:
it introduces neither a static index nor a likelihood factor.
Numerical evaluation must separately respect cancellation and overflow
in these mathematical expressions.

#### Compilation-local distribution contracts

A **distribution catalog** $\mathcal C$ extends the checking environment with
family contracts. Each contract specifies (i) a typed parameter space $\Theta_F$
and dependent event dimensions; (ii) an outcome space $Y_F$ and reference measure;
and (iii) a normalized kernel $K_F:\Theta_F\rightsquigarrow Y_F$ supplied by its
native implementation. The constructor judgment uses the same argument and
shape rules as a built-in family:

$$
\frac{F\in\mathcal C\qquad
      \mathcal C;\Gamma\vdash\theta:\Theta_F}
     {\mathcal C;\Gamma\vdash F(\theta):\operatorname{Distribution}(Y_F)}.
$$

Its denotation is $\llbracket F(\theta)\rrbracket=K_F(\theta)$, with
$K_F(\theta,Y_F)=1$. Sampling and observing use this kernel in the ordinary
step rules. Its density is taken against the declared reference measure, so
integer or mixed laws retain their atomicity rather than acquiring a continuous
interpretation from a host tensor's floating dtype. Recorded-event scoring uses
the same interval and atom operations as built-in laws.

`FamilyCatalog` supplies these contracts explicitly to a compilation; native
factories are held by identity. A catalog extension does not mutate another
compilation's environment. Checked wire metadata records the contract, while
executing it additionally requires the corresponding implementation. A native
factory alone supplies no implementation in another target language. Checking
parameter names, carriers, and dimensions cannot prove that user-written density,
sampling, and moment operations describe the declared kernel; that agreement
remains an obligation of the custom implementation.

The formula adapter is a substitution into this constructor. A mapping
$\rho$ assigns each native parameter either a linked predictor $g^{-1}(\eta)$
or a typed deterministic data value. Thus its response kernel is
$K_F(\rho(\eta,x))$, with no additional latent variable for a data parameter.
Named auxiliary predictors, varying effects, and nonlinear dependencies retain
their existing priors and graph semantics. The
[custom-family API](../api/formulas/custom.md) gives an executable instance;
the same catalog is retained for new-data replay and refitting.

#### Vector count events

For trial count $n\in\mathbb N$ and a probability vector $p$ on $C$
categories, `Multinomial` has counting-measure density

$$
P(Y=y)=\frac{n!}{\prod_{c=1}^{C}y_c!}
       \prod_{c=1}^{C}p_c^{y_c},
\qquad y_c\in\mathbb N,\quad\sum_c y_c=n.
$$

The convention $0^0=1$ admits structural zero categories. With positive
concentrations $\alpha_c$ and $A=\sum_c\alpha_c$, integrating over a
Dirichlet probability vector gives `DirichletMultinomial`:

$$
P(Y=y)=\frac{n!}{\prod_c y_c!}
       \frac{\Gamma(A)}{\Gamma(A+n)}
       \prod_c\frac{\Gamma(\alpha_c+y_c)}{\Gamma(\alpha_c)}.
$$

Both laws put unit mass on the all-zero vector when $n=0$. The category
axis is an event axis, while any observation axes form the batch. Thus
`observe y : Row <- Multinomial(...) [over=Category]` contributes one
log mass per row, rather than independent scores per count coordinate.
Their sum constraint couples the coordinates. Static category and row
extents specify these types; runtime trial and probability or
concentration tensors specify the kernel. Declaring either tensor
introduces no observation or random effect.

#### Multivariate Student events

For positive degrees of freedom $\nu$, location $\mu\in\mathbb R^d$,
and nonsingular lower Cholesky factor $L$, let
$\delta=(y-\mu)^\mathsf T(LL^\mathsf T)^{-1}(y-\mu)$. The
`MultivariateStudentT` kernel has log density

$$
\log\Gamma((\nu+d)/2)-\log\Gamma(\nu/2)
-\frac d2\log(\nu\pi)-\sum_j\log L_{jj}
-\frac{\nu+d}{2}\log(1+\delta/\nu).
$$

Its vector coordinates share one chi-squared radial scale. The event axis
is separate from observation batch axes, and a vector contributes one
marginal log density per observation. No auxiliary scale site enters its
pointwise likelihood. The Gaussian scale $LL^\mathsf T$ is not the Student
response covariance; the latter exists for $\nu>2$ and equals
$\nu LL^\mathsf T/(\nu-2)$.

#### Additive log-ratio Normal events

For $d\geq1$, let $Z\sim\mathcal N_d(\mu,LL^\mathsf T)$, where $L$
is a lower-triangular matrix with positive diagonal. Define

$$
Y_j=\frac{\exp Z_j}{1+\sum_{k=1}^d\exp Z_k}\quad(j\leq d),
\qquad Y_{d+1}=\frac{1}{1+\sum_{k=1}^d\exp Z_k}.
$$

`AdditiveLogRatioNormal` denotes the resulting probability kernel on
the open simplex. In its first $d$ simplex coordinates, its density is

$$
p(y)=\varphi_d\!\left(\left(\log\frac{y_j}{y_{d+1}}\right)_{j=1}^d;
\mu,LL^\mathsf T\right)\prod_{j=1}^{d+1}y_j^{-1}.
$$

An optional reference index $r$ replaces the final component above.
The Gaussian coordinates then follow the increasing component indices
$j\ne r$, with $Z_j=\log(Y_j/Y_r)$. The density has the same product
Jacobian over all components, and outputs keep their original component
order. A reference may vary across batch rows without changing the event
shape or identifying its component axis with a batch axis.

The family signature relates the location and matrix dimensions to the
response dimension by `nat_add(d, 1)`. This is a static equation, solved
alongside the runtime tensors' shape evidence; it introduces no random
variable or likelihood contribution. Row dimensions remain batch axes,
and the simplex dimension is one coupled event. The existing
`LogisticNormal` kernel retains its distinct stick-breaking semantics.

#### Formula response layouts and coordinate maps

A formula response layout separates a row index from a sample event.
A factor response has scalar event shape and an ordered map from codes to
retained labels. A `cbind` response instead has a component event axis in
the authored column order. These layouts persist through the formula lens,
typed runtime input declarations, and posterior output collection.

For a reference component $r$, multinomial and Dirichlet mean predictors
insert a zero at coordinate $r$ into the predictor vector and apply
softmax. In the mean/precision Dirichlet parameterization,
$\alpha_c=\phi p_c$. `dirichlet2` instead sets every
$\alpha_c=\exp(\eta_c)$, without a reference coordinate. A
`logistic_normal` formula supplies Gaussian log-ratio locations against
$r$, coordinate-specific scales, and a correlation factor. References
thus select predictor coordinates; they do not permute the observed event.

Write $z_{ic}=\log(y_{ic}/y_{ir})$ for the log ratio of free component $c$
in row $i$, $L$ for the Cholesky factor of the log-ratio correlation matrix,
and $\sigma_{ic}$ for the scale of component $c$ in row $i$. The row density
is

$$
p(y_i)=\mathcal{N}\!\left(z_i \mid \mu_i,\;
\operatorname{diag}(\sigma_i)\,L L^\top \operatorname{diag}(\sigma_i)\right)
\prod_c y_{ic}^{-1},
$$

where the product over all components is the Jacobian of the log-ratio map.
A `sigma<c>` formula sets $\sigma_{ic}=\exp(\eta^{\sigma_c}_i)$ from its own
linear predictor, a scale without a formula is one positive parameter
$\sigma_{ic}=\sigma_c$, and a fixed scale is a constant. The program stacks
these into the row-by-component matrix `sigma` and scales the rows of $L$ by
it, at cost $O(n\sum_c p_c)$ for $p_c$ columns of the formula of component
$c$. Each predicted scale retains its design, so new rows evaluate
$\sigma_{ic}$ from their own predictors.

For count events, the training row total $n_i=\sum_c y_{ic}$ is runtime
data, not a static category extent or a fitted random variable. New-row
counts require supplied totals independently of their predictors. The
joint log density reduces the event axis once, before any observation
weight or row reduction. Raw pointwise likelihoods retain one score per
observation, so model comparison uses the same event units as fitting.

Posterior expectations average the conditional response mean over posterior
draws. For additive log-ratio responses, an explicitly requested Monte
Carlo average approximates that inner mean; applying softmax to the Normal
location does not compute it. The approximation adds simulation error and
does not change the model's kernel, priors, or observation effects.

#### Normalized polynomial baseline hazards

Let $0=b_0<\cdots<b_s=\tau$ and let $c_{ij}>0$ be Bernstein
coefficients for interval $i$, with polynomial degree $k-1$. These
structural parameters have types $[s+1]$ and $[s,k]$; they do not add
sample-event axes. Define the baseline hazard on each interval by

$$
h_0(t)=\sum_{j=0}^{k-1}c_{ij}{k-1\choose j}
       u^j(1-u)^{k-1-j},\qquad
u=\frac{t-b_i}{b_{i+1}-b_i}.
$$

The normalization constraint is
$\sum_i(b_{i+1}-b_i)\sum_j c_{ij}/k=1$. It identifies baseline
shape separately from a positive proportional rate $\lambda$. For
$t>\tau$, set $h_0(t)=h_0(\tau)$ and
$H_0(t)=1+h_0(\tau)(t-\tau)$. The resulting kernel has

$$
S(t)=\exp[-\lambda H_0(t)],\qquad
f(t)=\lambda h_0(t)\exp[-\lambda H_0(t)],\quad t\geq0.
$$

Since the endpoint hazard is positive, $S(t)\to0$; no probability mass
is discarded beyond the fitted interval. `log_rate` denotes
$\log\lambda$, a runtime value rather than a static extent. Normalized
M-splines with simplex weights supply one reusable construction of the
coefficient table. The spline basis retains its knots, degree, and tail
policy; it introduces no second probability law or likelihood reduction.

The native and reference kernels integrate Bernstein pieces directly.
They invert the cumulative hazard by a bounded safeguarded solve on the
retained interval and an analytic tail inverse. Ordinary floating-point
capacity and convergence limits remain executor obligations, separate
from this normalized denotation. Arbitrary polynomial pieces can have
hazard jumps at knots; a spline construction supplies its own continuity.

The formula translation uses simplex weights $w_j>0$, $\sum_jw_j=1$,
and normalized M-spline basis functions $M_j$ to construct
$h_0(t)=\sum_jw_jM_j(t)$. Its default prior is
$w\sim\operatorname{Dirichlet}(1,\ldots,1)$. The emitted `factor`/`sum`
contraction produces the same Bernstein coefficient table used by the
source family; it introduces no separate likelihood implementation.
The intercept remains an unrestricted log-rate coordinate rather than
an additional baseline-scale parameter.

For group-specific baselines, retain one common basis and introduce a simplex
$w_g$ per grouping identity. Observation row $i$ uses
$h_{0,g_i}(t)=\sum_j w_{g_i,j}M_j(t)$ and its corresponding cumulative hazard.
Each profile integrates to one on the retained interval; a varying rate term
may thus be estimated independently without adding an unidentified baseline
multiplier. The source translation contracts basis coefficients once per
group and pulls them back along the explicit row-to-group map. Group, basis,
and observation axes remain separate even when their cardinalities coincide.

Prediction retains the actual training basis. Donor policies select one fitted
group identity jointly across baseline and coefficient sites; conditional-law
sampling draws a new simplex from its Dirichlet law alongside the existing
Normal/LKJ group effects. Excluding varying predictor effects does not exclude
a structural baseline profile. Knot selection uses exact-event quantiles and
the full included follow-up span, including active interval upper endpoints;
it does not estimate new knots from fresh predictor rows.

A positive constant tail establishes finite first and second moments at each
fixed parameter draw. This statement does not assert that averaging those
moments over an arbitrary parameter prior yields finite posterior moments.
The native law records this fact through its immutable
`finite_moment_orders` certificate, independently of whether a closed
moment formula is implemented. An explicit Monte Carlo expectation may
use this certificate or bounded support, while an available analytic
expectation takes precedence. This numerical request does not change
the response law or introduce another observation effect.

#### Censored extended Beta measures

For `ExtendedBeta(mu, phi, kappa)`, let $a=\mu\phi$,
$b=(1-\mu)\phi$, $s=1+2\kappa$, and
$Y=\min(1,\max(0,sX-\kappa))$ for $X\sim\operatorname{Beta}(a,b)$.
The parameters satisfy $0<\mu<1$, $\phi>0$, and $\kappa\geq0$.
Relative to Lebesgue measure on $(0,1)$ together with counting measure
on the two endpoints, its density is

$$
p_Y(y)=
\begin{cases}
I_{\kappa/s}(a,b), & y=0,\\
s^{-1}p_X((y+\kappa)/s), & 0<y<1,\\
I_{\kappa/s}(b,a), & y=1.
\end{cases}
$$

The endpoint masses are induced by censoring; they are not independent
inflation parameters. At $\kappa=0$ both masses vanish. The observable
mean is the mean of this full mixed measure, whereas $\mu$ remains the
mean of its uncensored Beta component. The implementation retains both
endpoint coordinates to evaluate small tail masses, and uses the shared
censoring left-CDF contract when integrating intervals that include an atom.

#### COM-Poisson normalization

For rate $\lambda\geq0$ and dispersion $\nu>0$, `COMPoisson` has counting-measure density

$$
p(k;\lambda,\nu)=\frac{\lambda^k}{(k!)^\nu Z(\lambda,\nu)},
\qquad Z(\lambda,\nu)=\sum_{j=0}^{\infty}\frac{\lambda^j}{(j!)^\nu},
\quad k\in\mathbb N.
$$

At zero rate the law is $\delta_0$. With $\eta=\log\lambda$,
its score is $k\eta-\nu\log(k!)-\log Z$ and
$\mathbb E[K]=\partial_\eta\log Z$; the rate is not generally this
expectation. `rate` and `log_rate` select equivalent parameterizations,
not separate effects or static indices. The formula frontend uses the
same rate law for fitting and prediction. The released brms
[Stan kernel](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/inst/chunks/fun_com_poisson.stan)
uses this rate, whereas its [R helper](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/R/distributions.R#L1222-L1300)
uses $m$ with $\lambda=m^\nu$.

The infinite series defines the measure independently of an executor's
certified summation budget. Native failure to certify normalization or
moments is an explicit execution refusal, not replacement by a truncated
law. The rejection sampler likewise refuses exhausted proposal or count
capacity instead of changing the target measure.

#### Ordinal probability constructions

Let $F$ be the selected inverse link, $\eta$ a real predictor,
$d>0$ discrimination, and $c_j$ the $K-1$ thresholds, with
$z_j=d(c_j-\eta-e_j)$. The optional $e_j$ is a transition-specific
predictor. For cumulative probabilities, put $F(z_{-1})=0$ and
$F(z_{K-1})=1$. The category masses are

$$
p_j=F(z_j)-F(z_{j-1}),\qquad j=0,\ldots,K-1.
$$

These masses require nondecreasing row contrasts. Ordered base
thresholds with a common predictor and positive discrimination establish
this condition; arbitrary transition-specific predictors do not. For
linear category-specific effects $v_{ij}$, the formula compiler instead
uses the training-domain bound

$$
B_{gj}=\max\left(0,\max_{i:g(i)=g}
           (v_{i,j+1}-v_{ij})\right),\qquad
\delta_{gj}=B_{gj}+\exp(s_{gj}).
$$

Centered cumulative sums of $\delta_{gj}$ define the group's base
cutpoints. Thus $c_{g,j+1}-v_{i,j+1}>c_{gj}-v_{ij}$ on every training
row. A shared profile uses the maximum over all training rows;
equidistant thresholds use the maximum bound across transitions plus
one positive slack. This preserves a direct linear category-specific
predictor, with a proper conditional prior on log slack $s$, rather
than an independent prior on cutpoints. Ordinary derivatives need not
exist where the maximum has tied arguments.

For $K$ cumulative categories, a flexible group profile consists of
$K-2$ centered shape coordinates and one varying location. Equidistant
profiles constrain the shape to a common spacing; flexible and sum-to-zero
profiles use the same centered positive-gap representation. For a grouped
equidistant profile, $\delta_g=\max_j B_{gj}+\exp(s_g)$ is shared
across all adjacent gaps. The compiler reuses or supplies exactly one
ordinary group intercept; binary profiles have only the location coordinate. A duplicate same-group
category-specific intercept is refused. Likewise, a fixed transition
intercept with freely estimated cumulative thresholds and a proportional
intercept repeats free coordinates; that combination is refused even
when the coefficient has an explicit prior. The hurdle construction applies
this representation to its positive categories; a single positive
category has no cutpoint profile.

Profile centering and centered inference coordinates are distinct.
The default hierarchical log-slack law is
$s_{gj}\mid m_j,\tau_j\sim\operatorname{Normal}(m_j,\tau_j)$,
with $m_j\sim\operatorname{Normal}(0,1)$ and
$\tau_j\sim\operatorname{HalfNormal}(0.5)$. Centered inference samples
$s$ directly; noncentered inference samples $z_{gj}\sim\operatorname{Normal}(0,1)$
and sets $s_{gj}=m_j+\tau_j z_{gj}$. The exponential and profile centering
are deterministic maps in both programs. A density relative to the sampled
log-slack coordinates thus introduces no additional exponential Jacobian.
An equidistant group has one such shape coordinate, rather than one per
gap. Together with its one varying location, these are all free coordinates
of that constrained profile; a binary profile has only location.

At prediction, the actual fitted cutpoints are retained per posterior
draw. New covariates and effect selections do not reconstruct the
training bound. Every requested effective profile must remain ordered;
a crossing is an explicit support failure. `posterior_thresholds`
returns these row cutpoints after selected varying and category-specific
shifts, before the proportional population predictor and discrimination.

Adjacent-category probabilities instead set
$q_j=F(-z_j)$ and
$\log(p_{j+1}/p_j)=\log q_j-\log(1-q_j)$, then normalize the
prefix sums of these log odds. For stopping-ratio probabilities, set
$s_j=F(z_j)$; for continuation-ratio probabilities, set
$s_j=1-F(-z_j)$. Both sequential constructions give

$$
p_j=s_j\prod_{i<j}(1-s_i)\quad(j<K-1),\qquad
p_{K-1}=\prod_{i<K-1}(1-s_i).
$$

These noncumulative constructions impose no ordering on thresholds.
Equidistant thresholds have the centered form
$c_j=(j-(K-2)/2)\delta$; $\delta$ must be positive for cumulative
models but may be any real for the other constructions. Threshold
centering states an identification constraint, not an equivalence of
priors induced by distinct latent representations.

For hurdle probability $h$, an ordinal base on categories
$1,\ldots,K-1$ becomes $p_0=h$ and
$p_j=(1-h)p_j^{\mathrm{base}}$ for $j>0$.
It uses $K-2$ base thresholds, including no threshold when there is one
positive category. This is an extra lowest category, not a second atom
added to the base's existing category zero.

A response `y | thres(n, gr = g)` gives each level $g$ of the threshold
group its own number of thresholds $n_g$ and its own unpooled threshold
vector, as brms does. Group $g$ has a shape $u_{g0}=0$,
$u_{gk}=\sum_{s<k}\exp(\lambda_{gs})$ for the cumulative family and
$u_{gk}=o_{gk}$ otherwise, with free coordinates
$\lambda_{gs},o_{gk}\sim\operatorname{Normal}(0,1)$, and thresholds

$$
c_{gk}=u_{gk}-\frac1{n_g}\sum_{j<n_g}u_{gj}+\ell_g,\qquad k=0,\ldots,n_g-1.
$$

The locations $\ell_g$ carry the prior of a population-level coefficient.
With a formula intercept $\alpha$, $\ell_g=0$ for the first group, so the
map from brms's vectors $\tau_{gk}=c_{gk}-\alpha$ to
$(\alpha,\ell,\lambda)$ or $(\alpha,\ell,o)$ is a bijection. Without an
intercept every $\ell_g$ is free. Row $i$ uses the vector of its group
$g(i)$ and has the categories $0,\ldots,n_{g(i)}$.

One row-by-threshold matrix serves every group. With
$K-1=\max_g n_g$, group $g$ is padded with
$c_{gk}=c_{g,n_g-1}+(k-n_g+1)$ for $k\ge n_g$, and the padded masses
$\tilde p_{ij}$ are reduced to the row's own categories. For the cumulative
and sequential constructions, $\tilde p_{ij}=p_{ij}$ for $j<n_g$ and the
padded categories partition the last one, so
$\log p_{i,n_g}=\operatorname{logsumexp}_{j\ge n_g}\log\tilde p_{ij}$. For the
adjacent-category construction the unnormalized log mass of category $j$
involves only $c_{g0},\ldots,c_{g,j-1}$, so the categories $j>n_g$ are
removed before normalization. Both reductions are exact for any finite
padding, and $p_{ij}=0$ for $j>n_g$. Fresh rows are coded against the
fitted groups, and a group without fitted thresholds is refused.
Category-specific effects are refused with threshold groups, as in brms.

Under the hurdle cumulative law the response codes of group $g$ are
$0,\ldots,n_g+1$: code zero is the hurdle with mass $h$, and code $j>0$ has
mass $(1-h)$ times the mass of base category $j-1$. The reduction above
applies to the base categories, which all carry the factor $1-h$, and the
hurdle mass is never padded. A mixture of ordinal laws applies the
construction once per component $m$, with its own coordinates
$(\alpha_m,\ell_m,\lambda_m)$ or $(\alpha_m,\ell_m,o_m)$, and mixes the
reduced category masses, $p_{ij}=\sum_m\theta_m p^{(m)}_{ij}$. With shared
thresholds (`order="mu"`) every component reads the group-by-threshold
matrix and the intercept of one component, so that
$\tau^{(m)}_{gk}=\tau_{gk}$ for all $m$.

`OrdinalSpec` and threshold choices are static compiler configuration;
cutpoints, discrimination, and category effects are runtime values.
Their declarations add no likelihood. The observation contributes the
selected category's log mass, with likelihood weights applied under the
same rules as any other observation.

#### Group effects at prediction time

Selecting group effects restricts the predictor sum: omitted terms have
value zero, rather than being integrated out. Group-specific thresholds
are separate conditional parameters and remain in the response law.
Known grouping identities reuse their posterior vectors. For a new
identity and posterior draw $s$, donor prediction uses one common donor
index $J_s$ across the identity's coefficients and auxiliary predictors.
The `uncertainty` policy samples $J_s$ independently per draw; `old_levels`
uses one $J$ for every chain and draw, while the donor vector still varies
with $s$. Both draw donors from all fitted levels.

The `gaussian` policy samples a new zero-mean group vector from the
Normal/LKJ conditional law at draw $s$. Repeated rows with the same
identity share that vector, including one shared new identity for
repeated missing group values within a factor. This is prediction under
a stated group law; it adds no fitted levels or changes to the posterior.

The same coordinate choice applies to every Gaussian group block.
Let $D=\operatorname{diag}(\sigma)$ and $L$ be its correlation Cholesky
factor. A centered program samples
$b_g\sim\operatorname{Normal}(0,DLL^\mathsf{T}D)$; a noncentered program
samples $z_g\sim\operatorname{Normal}(0,I)$ and sets $b_g=DLz_g$.
The block may concatenate scalar, transition, and response-component
coordinates of different widths. Each named coefficient is a fixed slice
of this single block, so sharing a correlation ID does not introduce a
second coefficient or covariance construction. For $G$ groups, the density
change of variables is
$\log p_B(B)+G\log|DL|=\log p_Z(Z)$, conditional on the same scales
and correlation. Posterior replay uses the retained block layout in either
coordinate system.

A supplied labeled level covariance $A$ changes this iid row law to
$B\sim\operatorname{MatrixNormal}(M,A,\Sigma)$, with
$\Sigma=DLL^\mathsf{T}D$. For declared PSD support, write
$A=QCQ^\mathsf{T}$ with $Q^\mathsf{T}Q=I_r$ and $C$ positive definite.
Centered coordinates sample $U\sim\operatorname{MatrixNormal}(0,C,\Sigma)$
and set $B=M+QU$; noncentered coordinates sample a standard $r\times d$
Normal matrix $Z$ and set $B=M+Q\operatorname{chol}(C)ZL^\mathsf{T}D$.
The mean $M$ need not lie in the deviation subspace. The isometric $Q$
identifies the centered density with conditional Hausdorff measure on that
affine support; rank zero is the deterministic law $B=M$. The noncentered
change of variables on support coordinates contributes
$\frac d2\log|C|+r\log|DL|$.

The equivalent MatrixNormal constructor may specify each covariance or its
lower Cholesky factor independently: `row_covariance` / `row_scale_tril` and
`col_covariance` / `col_scale_tril`. These four parameterizations denote the
same matrix-valued law. They preserve the two intrinsic event axes and introduce
no change-of-variables term, because their parameters differ while the random
matrix carrier remains fixed. Supplying a factor avoids both its covariance
product and another Cholesky factorization. Covariance-aware group emission uses
prepared row factors and the existing learned coefficient factors directly.

For fitted and fresh level partitions $f,n$, covariance-aware Gaussian
prediction uses deviation mean $A_{nf}A_{ff}^{+}(B_f-M_f)$ and row covariance
$A_{nn}-A_{nf}A_{ff}^{+}A_{fn}$, with the same column covariance $\Sigma$.
The pseudoinverse is restricted to the declared positive support. Label order,
coefficient axes and support coordinates are explicit metadata, never inferred
from equal cardinalities. Empirical donor policies instead resample joint
standardized fitted support-coordinate vectors before applying this conditional
mean and Schur factor; they do not discard the supplied covariance. Fitted
principal submatrices must remain unchanged in a covariance extension.

For a constrained category-specific new-group profile, the explicit
`feasible_uncertainty` and `feasible_gaussian` policies condition the
joint donor or Gaussian proposal on ordered effective cutpoints for the
requested rows. Gaussian threshold profiles use the fitted hierarchical
log-slack base law. Known profiles retain their fitted values. Each
posterior draw is retained; exhaustion of 1,024 proposals causes an
execution refusal, without dropping draws or rebuilding thresholds.
Ordinary categorical predictor contrasts remain fixed to their training
levels. The [prediction guide](../guides/analysis-data-and-formulas.md#selecting-effects-and-predicting-for-new-groups)
states the API and the supported policy boundary.

#### Nonlinear predictor dependencies

A nonlinear formula denotes a finite acyclic graph of deterministic
predictors. Its additive leaves are ordinary formula predictors with their
existing coefficient priors and group effects. In a topological ordering,
let $u_j=f_j(u_{<j},x)$ be an intermediate expression and
$\eta_a=f_a(u,x)$ a family-output expression. The runtime family parameter
is $\theta_a=g_a^{-1}(\eta_a)$ for its declared link $g_a$; intermediate
nodes have the identity link. Thus an explicitly exponentiated intermediate
and the final output's inverse link are separate authored maps.

These expressions add no random draws or scores. The model's joint measure
consists of the existing coefficient priors and the response law evaluated
at $\theta$. Component-valued expressions retain nominal coordinate labels,
including the reference-coordinate constraint. Static graph configuration
elaborates to the same runtime telescope and pure `let` constructs as other
formula programs. Cyclic dependencies, unknown inputs, and family-output
backreferences have no accepted elaboration.

#### Scalar response kernels and regression parameters

A regression parameter need not occupy the same slot as a native family
parameter. The formula compiler supplies a deterministic parameter map
$\theta_F$ before the draw or observation; this map adds no random
variable or likelihood factor. The following four scalar kernels use
Lebesgue measure.

For `SkewNormal(loc, scale, skewness)`, write $(\ell,s,\alpha)$
for the parameters and put $z=(y-\ell)/s$. Its density is
$2\varphi(z)\Phi(\alpha z)/s$,
where $\varphi$ and $\Phi$ are the standard Normal density and CDF.
The `skew_normal` formula parameterization instead uses response mean
$\mu$ and response standard deviation $\sigma$. With
$\delta=\alpha/\sqrt{1+\alpha^2}$, its parameter map is

$$
s=\frac{\sigma}{\sqrt{1-2\delta^2/\pi}},
\qquad \ell=\mu-s\delta\sqrt{2/\pi}.
$$

`ShiftedLogNormal(loc, scale, shift)`, with parameters $(\ell,s,d)$, is the pushforward
of $N(\ell,s^2)$ by $u\mapsto d+\exp u$. Its density at $y>d$
includes the Jacobian $1/(y-d)$; outside that support it is zero.
Thus its response mean is $d+\exp(\ell+s^2/2)$, while the formula's
main predictor is the log-location $\ell$.

`Frechet(scale, concentration)`, with parameters $(s,\nu)$, has
$F(y)=\exp(-(y/s)^{-\nu})$ for $y>0$ and positive $s,\nu$.
The mean exists for $\nu>1$ and is $s\Gamma(1-1/\nu)$; the
variance exists for $\nu>2$. A mean-parameterized formula thus
requires $\nu>1$ and sets $s=\mu/\Gamma(1-1/\nu)$.

For `GeneralizedExtremeValue(loc, scale, concentration)`, write
$(\ell,s,\xi)$ for the parameters and let $t=1+\xi(y-\ell)/s$.
For $\xi\ne0$ and $t>0$,

$$
F(y)=\exp(-t^{-1/\xi}),\qquad
f(y)=s^{-1}t^{-1/\xi-1}\exp(-t^{-1/\xi}).
$$

The zero-shape limit is Gumbel, with
$F(y)=\exp(-\exp(-(y-\ell)/s))$. Positive $\xi$ gives a finite
lower endpoint and negative $\xi$ a finite upper endpoint. The main
formula parameter is location, rather than response mean. That mean
exists for $\xi<1$ and equals
$\ell+s(\Gamma(1-\xi)-1)/\xi$, with limit
$\ell+s\gamma$ at zero, where $\gamma$ is Euler's constant.
The variance exists for $\xi<1/2$.

These maps follow the
[released brms likelihood definitions](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/R/stan-likelihood.R).
Matching a kernel does not identify its prior: QVR's direct GEV shape
parameter is distinct from brms's additional scalar shape transformation.

These four kernels have native and checked-reference implementations.
All 11 external renderers currently refuse them explicitly. Their native
CDFs, inverse CDFs, moments, and samplers have separate numerical and
derivative tests; those tests do not assert a uniform error bound over
all tail or shape regimes.

#### Gaussian response-time tails

For `ExGaussian(loc, scale, rate)`, let $z=(y-\mathrm{loc})/\mathrm{scale}$
and $r=\mathrm{rate}\,\mathrm{scale}$. Its cumulative and survival probabilities
are

$$
F(y)=\Phi(z)-e^{r^2/2-rz}\Phi(z-r),\qquad
S(y)=\Phi(-z)+e^{r^2/2-rz}\Phi(z-r).
$$

For `InverseGaussian(mu, shape)`, put
$a=\sqrt{\mathrm{shape}/y}(y/\mathrm{mu}-1)$ and
$b=\sqrt{\mathrm{shape}/y}(y/\mathrm{mu}+1)$ for $y>0$. Then

$$
F(y)=\Phi(a)+e^{2\mathrm{shape}/\mathrm{mu}}\Phi(-b),\qquad
S(y)=\Phi(-a)-e^{2\mathrm{shape}/\mathrm{mu}}\Phi(-b).
$$

These differences need not be evaluated by subtracting ordinary probabilities.
Both kernels use the positive Gaussian divided difference
$D(x,d)=e^{-x^2}(\operatorname{erfcx}(x)-\operatorname{erfcx}(x+d))$.
The ExGaussian CDF is $D(-z/\sqrt2,r/\sqrt2)/2$, and the
InverseGaussian survival probability is
$D(a/\sqrt2,\sqrt{2\mathrm{shape}/y})/2$.
Keeping the separation $d$ explicit avoids losing it when two large
Gaussian arguments round to the same value.

Native and scalar reference evaluation share the continued-fraction and
asymptotic polynomial coefficients used by this calculation. Short
separations use the positive derivative integral, with agreement checked
between fixed eight- and 32-node rules; numerical-capacity failures are
explicit. Interval events use the same paired tails or the shared local
density rule when a bound on log-density variation permits it. Thus a
numerical tail calculation supplies the existing kernel's probability;
it does not introduce a different response law or censoring convention.

#### Truncation as normalized restriction

For a base probability kernel $K_\theta$ and interval $I=[a,b]$,
write $Z_\theta=K_\theta(I)$. When $0<Z_\theta\leq1$,
the truncated kernel is

$$
K^I_\theta(B)=\frac{K_\theta(B\cap I)}{Z_\theta}.
$$

Its log density inside $I$ is $\log f_\theta(y)-\log Z_\theta$.
For a continuous scalar law, $Z_\theta=F_\theta(b)-F_\theta(a)$;
for a discrete law the lower term is the left limit $F_\theta(a^-)$,
so an atom at the lower endpoint remains included. Both the numerator
and normalizer depend on the family parameters. Censoring has a
different denotation: an interval-censored observation contributes
$K_\theta(I)$ without conditioning the whole response law on $I$.

More generally, define the raw lower-tail masses
$F_\mu(t)=\mu((-\infty,t])$ and
$F_\mu^-(t)=\mu((-\infty,t))$. The upper endpoint contributes
$F_\mu(b)$ when closed and $F_\mu^-(b)$ when open; the subtracted
lower term is $F_\mu^-(a)$ when closed and $F_\mu(a)$ when open.
This endpoint convention applies to raw restricted and mixture measures as
well as probability laws. At $a=b$, only a doubly closed interval retains
the atom. Normalization divides the resulting event mass by $\mu(\mathbb R)$;
it does not replace the retained measure before the event is formed.

For a scalar mixed law, densities are interpreted against a common reference
containing Lebesgue measure and the retained atom carrier. At a carrier point,
only atomic component masses contribute. Adding an atomless component's
Lebesgue density there would change the point mass and thus the law.
Integer-valued scalar components instead share counting measure on all
integers. Coupled mixed vector components require their joint stratum
reference; a scalar carrier calculation cannot supply that information.

For a recorded response $y_i$, censoring selects an event
$E_i$: $(-\infty,y_i]$ for a left observation,
$(y_i,\infty)$ for a right observation, and $(y_i,u_i]$ for an
interval observation. If the sampling law is truncated to $T_i$, its
recorded event contributes
$\log K_\theta(E_i\cap T_i)-\log K_\theta(T_i)$.
An exact observation instead contributes its density under that same
conditional law. Likelihood weights multiply this observation contribution
after its event and normalizer are formed; raw pointwise extraction omits
the weight. Censoring metadata introduces no sampling effect of its own.

A finite Normal mixture with weights $w_k$ is the one exception, matching
brms: truncation normalizes each component separately, so a recorded event
contributes $\log\sum_k w_k K_{\theta,k}(E_i\cap T_i)/K_{\theta,k}(T_i)$
and truncated prediction draws a component and then its truncated law.
Without truncation this equals the mixture mass of $E_i$.

The mean of the truncated kernel is the partial first moment over the
normalizer,

$$
\mathbb E_{K^I_\theta}[Y]=\frac{m_\theta(I)}{Z_\theta},
\qquad m_\theta(I)=\int_I y\,K_\theta(dy).
$$

For a nonnegative law with finite mean, $y\,K_\theta(dy)/\mathbb E_\theta[Y]$
is a probability law, the size-biased law $\tilde K_\theta$, so
$m_\theta(I)=\mathbb E_\theta[Y]\,\tilde K_\theta(I)$ and the truncated mean
is the untruncated mean times a ratio of two event masses. The size-biased
law of $\mathrm{Gamma}(k,r)$ is $\mathrm{Gamma}(k+1,r)$, that of
$\mathrm{LogNormal}(\mu,\sigma)$ is $\mathrm{LogNormal}(\mu+\sigma^2,\sigma)$,
that of $\mathrm{Beta}(\alpha,\beta)$ is $\mathrm{Beta}(\alpha+1,\beta)$, and
that of an inverse Gaussian $X$ with mean $\mu$ is the law of $\mu^2/X$. For
a Weibull law with scale $s$ and shape $k$ it is the law of $s\,G^{1/k}$ with
$G\sim\mathrm{Gamma}(1+1/k,1)$, and for a Fréchet law with shape $\alpha>1$
the law of $s\,G^{-1/\alpha}$ with $G\sim\mathrm{Gamma}(1-1/\alpha,1)$. For
counts, the size-biased law of $\mathrm{Poisson}(\lambda)$ is
$1+\mathrm{Poisson}(\lambda)$, that of $\mathrm{Binomial}(n,p)$ is
$1+\mathrm{Binomial}(n-1,p)$, and that of a negative binomial count with
$r$ successes is one plus the same count with $r+1$ successes. An atom at
zero adds to $Z_\theta$ and not to $m_\theta(I)$. The Normal and Student-t
laws use their closed forms, the latter for more than one degree of
freedom. Every other law is summed (counts) or integrated by quadrature
whose own estimate of $Z_\theta$ must agree with the event mass.

For independent scalar coordinates truncated to a rectangle,
$Z_\theta=\prod_j Z_{\theta,j}$, so log normalizers add over the
event coordinates. A coupled vector event requires its joint rectangle
mass; marginal CDF differences cannot supply that mass. An executor
without the needed CDF or rectangle-mass operation must refuse this
kernel, rather than replace its normalizer with a random estimate.

The native `Truncated` wrapper implements this deterministic normalizer
for independent scalar-coordinate distributions supplied by its base.
Normal tails use a stable specialized normalizer; supplied log-tail
methods are preferred over CDF subtraction. A zero or nonfinite
computed mass is an error. Normal sampling uses inverse transformation;
other bases use bounded rejection and explicitly fail when its attempt
limit is exhausted. This rejection path supplies no general pathwise
derivative for parameter-dependent acceptance.

The conditional wrapper constructs the same `Normalize(Restrict(...))`
coordinate laws as the measure interface. Both restrict the score's domain
before evaluating the underlying density: an excluded point contributes zero
mass, and an inactive branch is evaluated only at a feasible support point.
For independent vector events, the restriction mask is reduced over event
coordinates before the scalar event score is selected. This preserves the
batch/event distinction and avoids invalid intermediate derivatives.

The derivative of the mathematical normalizer and an executor's ability
to compute it are separate questions. Native Gamma interval masses use
float64 lower-tail series and upper-tail continued fractions, with
small-shape complement calculations to retain tail probabilities.
Shape and rate derivatives propagate through converged recurrences;
failure to converge within 512 iterations raises explicitly. This
algorithm does not assert uniform numerical accuracy or convergence
over all parameter regimes.

#### Discrete Weibull counts

The discrete Weibull law uses counting measure on the nonnegative integers.
For `DiscreteWeibull(probs=q, concentration=b)`, with $0\leq q<1$
and $b>0$, its mass is

$$
p(k)=q^{k^b}-q^{(k+1)^b},\qquad k=0,1,\ldots.
$$

At $q=0$ this is a point mass at zero; $q=1$ supplies no proper
probability law on finite counts. For $q>0$, the CDF at $y\geq0$
is $1-q^{(\lfloor y\rfloor+1)^b}$, and its strict survival probability
is $q^{(\lfloor y\rfloor+1)^b}$. Its moments satisfy

$$
\mathbb E[Y]=\sum_{k=1}^\infty q^{k^b},\qquad
\mathbb E[Y^2]=\sum_{k=1}^\infty(2k-1)q^{k^b}.
$$

For integer $0\leq a\leq c$, the interval mass is
$q^{a^b}-q^{(c+1)^b}$. Native and reference execution factor this
power difference before taking logarithms, sharing the singleton and interval
kernel. Thus a small concentration does not identify two distinct cumulative
probabilities merely because both round to the same floating-point value.
Unbounded intervals use the corresponding exact tail expression; an early
stop in a generic count sum is not substituted for the remaining tail.

Both sums are finite for the admitted parameters. The formula family's
main parameter `mu` denotes $q$, following the
[pinned brms parameterization](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/inst/chunks/fun_discrete_weibull.stan);
it is not the response mean. A transformed linear predictor and an
expected response thus answer different questions. Numerical moment
evaluation must check its remainder and resource limits rather
than identify a partial sum with this infinite sum.

#### A zero atom with asymmetric Laplace responses

Let $K_{\ell,s,\tau}$ be an asymmetric Laplace law, where $\ell$
is its $\tau$-quantile, $s>0$, and $0<\tau<1$. Its density is
$f(y)=\tau(1-\tau)\exp(-\rho_\tau((y-\ell)/s))/s$,
with $\rho_\tau(u)=u(\tau-\mathbf1_{u<0})$.
The zero-inflated law is the probability kernel

$$
K_{\omega,\ell,s,\tau}=\omega\delta_0+(1-\omega)K_{\ell,s,\tau},
\qquad 0\leq\omega\leq1.
$$

Its density with respect to Lebesgue measure plus $\delta_0$ is
$\omega$ at zero and $(1-\omega)f(y)$ elsewhere. In particular,
the continuous component's density at zero is not added to the atom's
probability. This agrees with the pinned
[brms scoring definition](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/inst/chunks/fun_zero_inflated_asym_laplace.stan).

Writing $F$ for the continuous component's CDF gives

$$
F_{\mathrm{ZI}}(y)=(1-\omega)F(y)+\omega\mathbf1_{y\geq0},
\qquad
\Pr(Y>y)=(1-\omega)(1-F(y))+\omega\mathbf1_{y<0}.
$$

The strict survival function thus excludes the atom at zero. The
released brms chunk's zero-boundary `lccdf` branch includes the atom;
the strict convention here complements the CDF at that boundary.
The response mean is $(1-\omega)m$, where
$m=\ell+s(1-2\tau)/(\tau(1-\tau))$ is the component mean.
The formula's location parameter remains the component quantile;
adding the atom does not make it a quantile of the whole response law.
All distribution parameters remain runtime values, while a static
extent governs only the number and arrangement of responses.

#### Negative-binomial dispersion parameters

An alternate parameterization does not require a different native
kernel. For a negative-binomial response with mean $\mu>0$ and
shape $r>0$, the variance is $\mu+\mu^2/r$. The
`negbinomial2` parameterization sets $\sigma=1/r$, giving
$\operatorname{Var}(Y)=\mu+\sigma\mu^2$. It thus supplies
`total_count=1/sigma` and `probs=mu/(mu + 1/sigma)` to the existing
native `NegativeBinomial` constructor. This is the
[pinned brms transformation](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/R/stan-likelihood.R).

Equivalent conditional laws do not identify the same prior measure.
Transforming a prior on $r$ into one on $\sigma$ requires the
Jacobian $1/\sigma^2$. Choosing a new prior directly on $\sigma$
instead defines a different model, even when both frontends compile
to the same native family. The deterministic likelihood parameter map
itself contributes no Jacobian score.

#### Zero-inflated and hurdle count kernels

These count families use counting measure on the nonnegative integers.
For a component mass function $f$ and structural-zero probability
$\omega\in[0,1]$, the zero-inflated kernel has mass

$$
p_{\mathrm{ZI}}(y)=
\begin{cases}
\omega+(1-\omega)f(0), & y=0,\\
(1-\omega)f(y), & y>0.
\end{cases}
$$

`ZeroInflatedNegativeBinomial`, `ZeroInflatedBinomial`, and
`ZeroInflatedBetaBinomial` choose the corresponding component $f$.
Their `zero_prob` argument is $\omega$, not the complete probability of
observing zero. A hurdle kernel instead has mass

$$
p_{\mathrm{H}}(y)=
\begin{cases}
\omega, & y=0,\\
(1-\omega)\dfrac{f(y)}{1-f(0)}, & y>0.
\end{cases}
$$

`HurdleNegativeBinomial` uses this positive truncation of its
negative-binomial component. At component `probs=0`, conditioning on a
positive count would have a zero denominator. The implemented family
extends the positive component by its limit as `probs` approaches zero,
the point mass at one. This extension is distinct from conditioning on an
event of positive probability.

A binomial or beta-binomial component with zero trials is the point mass
at zero. Its zero-inflated extension is also the point mass at zero for
every $\omega$. Trial counts are supplied integer values; declaring them
does not add a latent variable or an effect. The count samplers are not
reparameterized, though their likelihood parameters remain differentiable.

Native count arithmetic is a separate implementation contract from these
mass functions. Binomial, BetaBinomial, NegativeBinomial, Poisson, and the four mixtures
use float64 parameters, log scores, and moments while retaining gradients
to the original learned heads. Finite trial counts are validated before
conversion with $0\leq n\leq 2^{53}$, so their integer values are represented
exactly in float64. This family-specific bound does not change the signed
64-bit `Int` carrier or the static natural-number index sort. The resulting
scores are floating-point approximations to the mass functions above.
Integer source literals and typed integer inputs retain their integer
values until the family performs this conversion. The native density
kernels use stable deviance, Stirling, and gamma-increment forms where
applicable, rather than relying on subtraction of large log-gamma terms.
This arithmetic policy does not imply a uniform approximation bound for
every count and parameter regime.

Arithmetic-count kernels validate original finite observations before
floating conversion. Observations outside $[-2^{53},2^{53}]$ cause an
explicit native `ValueError` or reference `DistributionError`, preserving
the original integer rather than scoring a rounded value. With argument
validation disabled, fractional or negative observations within that range,
and nonfinite observations, have log mass $-\infty$.
Categorical and ordinal laws instead compare exact original category indices with their finite support; they do not
impose this arithmetic-count ceiling on an out-of-range index.

Poisson, negative-binomial, and their inflated or hurdle samplers have a
separate operational ceiling: a generated intensity or count must be
finite and at most $2^{53}$. An execution that exceeds it fails explicitly,
with `RuntimeError` natively or `DistributionError` in the reference
backend; it does not truncate, renormalize, or redraw from the kernel.
An outcome selected from a structural zero atom skips the unreachable
count branch. This execution limit does not change the mathematical
kernel or the general integer carrier, and does not itself restrict
density or moment evaluation at larger real-valued shapes or rates.

### 2.2 Observations {#22-observe}

An observation

```
observe v <- F(args)
```

reads the given value $v_{\mathrm{obs}}$ of $v$ and multiplies the measure by its likelihood:

$$
\mathcal{S}\llbracket \mathsf{observe}\ v \leftarrow F(\bar a) \rrbracket(\phi, B)
= \mathbf{1}_B(\phi)\, p_F\bigl(v_{\mathrm{obs}} ;\, \theta_F(\bar a, \phi)\bigr),
$$

a kernel from $\Phi$ to $\Phi$ that keeps the context and has total mass
$p_F(v_{\mathrm{obs}} ; \theta_F(\bar a, \phi))$. A density may exceed one, so this kernel is in
general not a subprobability kernel. The observed values are given at run time with the program's
inputs ([observations](../guides/programs.md#the-observations-dict)), and normalization is left to
[inference](#6-inference-and-conditioning).

When an earlier lexical binding defines $v=f(\phi)$, the same rule
uses $f(\phi)$ in place of $v_{\mathrm{obs}}$. The observation contributes
$p_F(f(\phi);\theta_F(\bar a,\phi))$ and preserves that value; it does not
integrate over a new coordinate. In particular, a deterministic response
encoding introduces no guide or MCMC coordinate. A scanned body recomputes
the value at each occurrence, and a plated observation scores its row and
event axes without an additional run axis. This is a likelihood score;
it does not itself introduce the Jacobian of $f$.

#### Power-likelihood weights {#observation-weights}

An observation with `[weights=w]` replaces its log likelihood $\ell$ by
$h(w,\ell)$, where $h(0,\ell)=0$ and $h(w,\ell)=w\ell$ for $w>0$.
Every weight is finite and nonnegative. The resulting kernel is

$$
\mathcal{S}_w(\phi,B)
=\mathbf{1}_B(\phi)\exp h(w(\phi),\ell(\phi)).
$$

This is a *power likelihood*: a positive weight raises the density to that
power, while a zero weight contributes the multiplicative identity, including
where the raw density is zero. The zero branch contributes no score and has
zero gradient with respect to the density's parameters. A runtime must select
that branch without forming an undefined product such as $0\cdot(-\infty)$.

This operation changes only the selected observation factor. Priors and other
score steps retain their original factors. The powered factor need not
integrate to one as a response density; sampling at fixed parameters continues
to use the original distribution. Its fitted posterior, and thus posterior
predictive distribution, may change. An input annotation adds no factor:
`w : Real` or `w : Tensor[Real]([Row])` supplies an admissible value, and the
observation option applies it to a likelihood.

### 2.3 Deterministic bindings

A binding

```
let v = expr
```

extends the context deterministically. The expression denotes a measurable function
$h : \Phi \to T$ of the names in scope, and the step denotes the deterministic kernel

$$
\mathcal{S}\llbracket \mathsf{let}\ v = \mathit{expr} \rrbracket(\phi) = \delta_{(\phi, h(\phi))},
$$

the pushforward along $\phi \mapsto (\phi, h(\phi))$. Arithmetic operators and built-in functions
denote the corresponding measurable functions.

#### 2.3.1 Built-in functions

A call `f(arg, ...)` in an expression resolves first against the closed set of QIEC primitives,
selected by the types of the arguments, and a name with no matching signature is a compile error.
A reduction (`sum`, `mean`, `min`, `max`, `prod`, `logsumexp`) collapses the last axis of its
argument: a vector reduces to a number, and a tensor of shape $(n_1, \ldots, n_k)$ to the tensor of
shape $(n_1, \ldots, n_{k-1})$ of its row reductions. A rowwise operation (`cumsum`, `sort`,
`softmax`, `log_softmax`, `normalize`) acts along the last axis and keeps the shape. Nested
reductions collapse one axis each, so `sum(sum(t))` sums the last two axes. The checker, the
[reference machine](../developer/qiec.md#reference-evaluator) (the library's interpreter for QIEC), the
[compiled runtime](../guides/dsl-overview.md#compilation-pipeline) (its PyTorch implementation), and every transpilation target read the primitives this
way, and a target that cannot express a reduction along the last axis of a given argument refuses the
program. A reduction over a named axis is written as a [`contraction`](composition-rules.md#4-operadic-contractions).
The [let-expression guide](../guides/dsl-programs-and-lets.md#checked-primitive-reference) lists
every primitive. The compiled runtime accepts further functions for programs that use PyTorch
attachments, but a call to one of them makes elaboration to QIEC fail with `qiec-program-gap`.

#### 2.3.2 Collection expressions

For a tensor $x = (x_0, \ldots, x_{n-1})$ with a leading extent $n$ known statically,

$$
\begin{aligned}
\llbracket \mathsf{map}(x, f) \rrbracket
  &= (\llbracket f \rrbracket(x_0), \ldots, \llbracket f \rrbracket(x_{n-1})), \\
\llbracket \mathsf{fold}(x, z, f) \rrbracket
  &= f(\cdots f(f(z, x_0), x_1) \cdots, x_{n-1}), \\
\llbracket \mathsf{length}(x) \rrbracket &= n, \\
\llbracket \mathsf{logsumexp\_over}(x, f) \rrbracket
  &= \log \sum_{i=0}^{n-1} \exp(\llbracket f \rrbracket(x_i)).
\end{aligned}
$$

`map` elaborates to a QIEC comprehension. `fold`, written `fold(xs, init, acc -> item -> body)`, is
a left fold elaborated at its fixed extent. When the body reads the accumulator more than once, or
inside a lambda or a factor expression, each intermediate accumulator is bound once by a QIEC
[`LetValue`](../api/qiec/terms.md) and read through that binding, so the elaborated term grows
linearly in $n$. A source of `map` or `fold` that is not a variable, and the argument of a lambda
whose body reads its parameter more than once, are bound the same way. `filter` has no checked
denotation, because the length of its result depends on values, and only the compiled runtime
evaluates it.

For a computation $k$ from $A$ to $B$ given by a lambda, `traverse(x, k)` runs $k$ on each element
in order and collects the results. Its denotation is the composite of the kernels
$k(x_0), \ldots, k(x_{n-1})$ in that order, each extending the context with its result $b_i$,
followed by the deterministic kernel that returns $(b_0, \ldots, b_{n-1})$. This is the traversal of an applicative
functor over a list [@mcbride2008applicative]. The elaborator expands it into calls and draws in
source order, so the callee's effects join the enclosing row and each call has its own site
addresses. The [collection reference](../reference/qvr/collection-expressions.md) gives the
restrictions on the form.

#### 2.3.3 Calls of morphisms and other declarations

A call in an expression may also name a morphism, an encoder, a decoder, a deduction, or a
signature declared in the module. Built-in primitives take precedence over these names, which take
precedence over the constructors of a declared signature. A `program` with no `sample`, `observe`,
or `score` step denotes a deterministic kernel, one that sends each input to a
[Dirac measure](https://en.wikipedia.org/wiki/Dirac_measure), and an encoder may call it.

The compiler checks the number of positional arguments of each such call: a morphism takes one
argument, a [`MonadicProgram`](../api/continuous/programs.md) with named parameters takes one per
parameter, and another callable takes as many as its positional parameters without defaults (the
check is skipped when it accepts `*args`). A shape mismatch or a `TypeError` inside the callee is
reported as a [`CompileError`](../api/dsl/compiler.md) that names the call.

### 2.4 Plates {#24-indexed-bind-plate}

A [plate](https://en.wikipedia.org/wiki/Plate_notation) repeats a variable over an index set as
conditionally independent copies [@koller2009probabilistic]. A draw over a plate

```
sample v : A <- F(args)
```

draws one value $v(a)$ for each $a$ in the finite set $A$, independently given the context:

$$
\mathcal{S}\llbracket v : A \leftarrow F(\bar a) \rrbracket(\phi, -)
= \delta_\phi \otimes \bigotimes_{a \in A} F\bigl(\theta_F(\bar a, a, \phi)\bigr),
$$

a Markov kernel from $\Phi$ to $\Phi \times K^A$, with $K = \llbracket \mathrm{cod}(F) \rrbracket$ and
density $\prod_{a \in A} p_F(v(a) ; \theta_F(\bar a, a, \phi))$. The family of marginals
$(p_a)_{a \in A}$ does not determine a joint distribution on $K^A$, since other couplings have the
same marginals, and the plate chooses the product. The values may still be dependent in the joint
model when their parameters share a latent variable, such as a common scale, and they are
independent given it. Pyro calls the same construction `plate` [@pyroTensorShapes].

### 2.5 Observations over a plate {#25-plated-observations}

An observation over a plate

```
observe r : N <- F(args) [via = idx]
```

multiplies the measure by the product of the likelihoods of the observed values
$r_{\mathrm{obs}} : N \to \llbracket \mathrm{cod}(F) \rrbracket$:

$$
\mathcal{S}\llbracket \mathsf{observe}\ r : N \leftarrow F(\bar a) \rrbracket(\phi, B)
= \mathbf{1}_B(\phi) \prod_{n \in N} p_F\bigl(r_{\mathrm{obs}}(n) ;\, \theta_F(\bar a, n, \phi)\bigr).
$$

An argument written `theta[N]` reads the value of the plate variable `theta` at $n$. The optional
`via = idx` names an *index map*, a function from the rows to the groups of an enclosing grouped
`marginalize`
([§2.7](#27-grouped-marginalize-with-multi-observe-fibration)).

With `[weights=w]`, each row's log density is replaced by
$h(w(n),\log p_F(r_{\mathrm{obs}}(n);\theta_F(\bar a,n,\phi)))$
before summing over the plate. A scalar broadcasts over its batch positions;
a tensor must have exactly the batch shape, excluding event coordinates.
Subsampling selects the same positions from observations and weights, then
applies the sampling correction once to their weighted log-density sum.

### 2.6 Summing out a discrete latent variable {#26-marginalize}

A block

```
marginalize c : A <- F(args)
    s₁
    ⋮
    sₖ
```

binds $c$ to a draw from $F(\bar a)$, plated over $A$ when $A$ is given, with the steps
$s_1, \ldots, s_k$ as its scope. When $F$ has finite support, the block denotes the measure of its
scope pushed forward along the projection $\pi_\Phi : \Phi \times C \to \Phi$ that forgets $c$:

$$
\nu \mapsto (\pi_\Phi)_* \nu,
\qquad
(\pi_\Phi)_* \nu(B) = \nu(B \times C).
$$

In log space, the scope's weight for each value $c$, added to the log prior of $c$, is reduced
over the support by log-sum-exp, which is the
[marginal likelihood](https://en.wikipedia.org/wiki/Marginal_likelihood) with $c$ summed out. This
is the enumeration of Pyro [@pyroEnumeration] and the marginalization of a mixture in Stan
[@stanFiniteMixtures]. The option `reduction = R` replaces log-sum-exp by `sum` or `mean` of the log
weights, which is not a marginalization. After the block, $c$ is out of scope.

When $F$ does not have finite support, the elaborator draws one value of $c$ and runs the scope
with it. The block then contributes the weight of one sampled configuration, not the integral over
$c$, and the elaborator rejects `reduction` on this path.

A draw, an observation, and a marginalized draw are one kind of step with a mode (draw, condition,
or sum out) and an optional plate. Summing out a discrete latent variable is the usual way to fit a
mixture model by its marginal likelihood [@bishop2006pattern].

#### 2.6.1 A latent variable carried by a scan

A scan cell `cell(x, z') : X * C -> C` that ends with a block over $z \in C$ and returns $z$ does not
close the block at the end of the cell: every later position reads $z$ as its previous state, so the
pushforward is taken only after the last position. For $T$ positions with first state
$z_0 \sim p$, the scan contributes the sum over every sequence of states,

$$
\sum_{z_0, \ldots, z_T} p(z_0) \prod_{t=1}^{T} w_t(z_{t-1}, z_t),
\qquad
w_t(i, j) = p_{F}\bigl(j ;\, \theta(x_t, i)\bigr) \exp \sigma_t(j),
$$

where $F(\theta(x_t, i))$ is the block's family at input $x_t$ and previous state $i$, and
$\sigma_t(j)$ is the log weight of the scope at $z_t = j$. The sum is computed from the left by the
[forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm),
$\alpha_t(j) = \sum_i \alpha_{t-1}(i)\, w_t(i, j)$ with $\alpha_0 = p$, and $\sum_j \alpha_T(j)$ is the
marginal likelihood of the chain [@rabiner1989hmm]. Every $z_t$ is summed out, so the name the scan
binds has no value. The
[marginalized chains reference](../reference/qvr/probabilistic-programs.md#marginalized-chains)
gives the forms the compiler accepts.

### 2.7 Grouped observations sharing one latent variable {#27-grouped-marginalize-with-multi-observe-fibration}

When several rows belong to one group, they share the group's latent class, and the likelihoods of
the rows must be multiplied before the class is summed out [@pyroEnumeration]. Let an *index map*
$\iota : N \to G$ send each row $n$ to its group, and call $\iota^{-1}(g)$, the set of rows of group
$g$, the [fiber](https://en.wikipedia.org/wiki/Fiber_(mathematics)) of $\iota$ over $g$. For the
rows $0, 1, 2$ with groups $\iota = (0, 0, 1)$, the likelihood of group $0$ is

$$
\sum_k \pi_{0k}\, p(y_0 \mid k)\, p(y_1 \mid k),
$$

whereas summing out a class for each row separately gives
$\bigl(\sum_k \pi_{0k} p(y_0 \mid k)\bigr)\bigl(\sum_k \pi_{0k} p(y_1 \mid k)\bigr)$, a model in which the two
rows may have different classes. Transpilation preserves this grouping
([Transpilation correctness §5](transpile-correctness/index.md#5-plates-marginalization-and-via)).

The grouping is declared on the block, and each observation in it names its index map:

```
marginalize c : K <- F(args) [over = G, reduction = R]
    observe r_1 : N_1 <- F_1(...) [via = idx_1]
    observe r_2 : N_2 <- F_2(...) [via = idx_2]
    ⋮
```

`over = G` names the group set (or `over = product(G_1, G_2, ...)`, also written
`over = [G_1, G_2, ...]`, for a product of groups), and `via = idx_m` names the index map
$\iota_m : N_m \to G$ of the $m$-th observation, given as an integer array (role `index_map` in the
API). The block contributes the log weight

$$
\sum_{g \in G} \operatorname{aggr}_{R,\, k}\Bigl[\log \pi(g, k) + \sum_m \sum_{n \,:\, \iota_m(n) = g} \ell_m(n, k)\Bigr],
$$

where $\ell_m(n, k) = \log p_{F_m}(r^{\mathrm{obs}}_m(n) ; \theta_m(n, k, \phi))$ is the log likelihood
of row $n$ of observation $m$ under class $k$, $\pi(g, k)$ is the prior weight of class $k$ in group
$g$, and $\operatorname{aggr}_R$ is the reduction over $k$: log-sum-exp by default, which sums the
class out, or `sum` or `mean`.

The class values, groups, and observation rows have separate index objects. In
`marginalize c : Class <- ... [over=Group]`, the value of `c` is indexed by
`Group`, and each entry selects an element of `Class`. If `mu` is indexed by
`Class` and `x` by `Group`, then `let center = mu[c] + x` is indexed by
`Group`. Each observation with map $\iota_m : N_m \to G$ pulls this value back
to its own rows: its mean at row $n$ is
$\mu(c(\iota_m(n))) + x(\iota_m(n))$. The `via` map thus applies to the
observation's group-indexed arguments, including intermediate values defined
by `let`.

Equal cardinalities do not identify index objects. A value indexed by `Row`
cannot be added implicitly to one indexed by `Group`, even when
$|\mathrm{Row}| = |\mathrm{Group}|$. Such a `let` requires an explicit index
map; the observation's `via` option does not reindex expressions elsewhere
in the block.

A weighted observation replaces $\ell_m(n,k)$ by
$h(w_m(n),\ell_m(n,k))$ before summing over the fiber and applying
$\operatorname{aggr}_R$. The latent prior $\log\pi(g,k)$ retains its
original coefficient. Thus weights act on each component's response likelihood
inside the latent aggregation.

The prior may be the same for every group, $\pi(g, k) = \pi(k)$, or a plate over $G$, as in
`marginalize z : K <- Categorical(probs=theta) [over=G]` with `theta : G`, where group $g$ draws its
one class from its own row $\theta(g, -)$. In both cases the block has one class per group, and the
compiled runtime, the reference machine, and every transpilation target that renders the block
compute the formula above. BUGS refuses every grouped block, and JAGS refuses one whose prior is a
plate over $G$ ([Transpiler support](../transpile-support.md)).

The model in which every row has its own class drawn from its group's row,
$\sum_n \log \sum_k \theta(\iota(n), k)\, p(r_n \mid k)$, is a different block, whose latent is
indexed by the rows: `let theta_row = theta[idx]` gathers each row's prior, and
`marginalize z : K <- Categorical(probs=theta_row) [over=N]` sums out one class per row, its
observation naming the identity map on $N$ with `via`.

A grouped block scores one run of the program at a time
([independent runs](../guides/inference-foundations.md#input-rows-are-independent-runs)). An
observation may carry a leading axis of runs of length one before its rows, which the block drops,
and data for several runs are refused.

For a product of groups $G_1 \times G_2 \times \cdots$ paired with `via = product(idx_1, idx_2, ...)`,
also written `via = [idx_1, idx_2, ...]`, on each observation, the groups are the tuples, ordered with the last factor varying fastest: a row
goes to position $\iota(n) = \sum_i \iota_i(n) \prod_{j > i} |G_j|$.

A grouped block may contain another. Each latent introduces its own class
axis, and reducing the inner latent removes only that axis. Arguments and
intermediate values retain their dependence on every enclosing class. For two
latents over the same group coordinates, the default reductions contribute

$$
\sum_g \log\sum_a \exp\Bigl(
    \log\pi_g(a) + \log\sum_b \exp\bigl(
        \log\rho_g(a,b) +
        \sum_{n:\iota(n)=g}\ell(n,a,b)
    \bigr)
\Bigr).
$$

The class sets need not have equal cardinalities. A third enclosing latent
adds another class argument to the likelihood and another outer reduction;
it is not identified with either existing axis. Observation weights replace
each $\ell$ by $h(w,\ell)$ before these reductions, leaving the priors unchanged.

The inner block contributes one aggregate per inner group to the enclosing
block. Equal checked group extents align positionwise, including when their
object names differ. An inner product of groups may instead refine the outer
group by a projection, as $G\times H\to G$; reducing its additional group
coordinates preserves all enclosing class axes. This group aggregation does
not supply arbitrary missing axes to likelihood parameters. Each parameter
must still satisfy the inner block's checked shape: an outer-group vector
cannot be passed directly where an inner product-group tensor is required.
Such an incompatible use receives a located diagnostic.

A `sample` inside a block that reads nothing the block binds is drawn once
before the block, since every value of the latent shares it. A draw whose
arguments read the latent would be one draw for each class, and the compiler
rejects it.

### 2.8 Effect annotations {#28-effect-signatures}

A program may declare which effects its body performs:

```
program P[static_params](value_params) : τ₁ -> τ₂ [effects = [E₁, E₂, ...]]
    body
```

where each $E_i$ is `Sample`, `Score`, `Marginal`, or `Pure`. This is the effect annotation of an
[effect system](https://en.wikipedia.org/wiki/Effect_system): a static bound on what a computation
may do [@lucassen1988polymorphic]. Each step contributes an effect:

| Step | Effect |
|---|---|
| `sample v <- F(args)`, `sample v : A <- F(args)` | `Sample` |
| `observe v <- F(args)`, `observe r : N <- F(args)`, `score v = expr` | `Score` |
| `marginalize c <- F(args) ...` | `Marginal` |
| `let v = expr`, `return e` | none |

The compiler computes the set $\mathcal E(P)$ of effects the body performs and checks
$\mathcal E(P) \subseteq \mathcal E_{\mathrm{decl}}$. `Pure` requires $\mathcal E(P)$ to be empty and
so rejects every `sample`, `observe`, `score`, and `marginalize`. A program whose body is `Pure`
denotes a deterministic kernel, one with only `Sample` a Markov kernel, and one with `Score` an
s-finite kernel. The QIEC effect row after elaboration is a finer type: it names the built-in
effect instances `random` and `score` and the effects of every reachable call
([Typing and checking §6](typing.md#6-computations-and-rows)).

A program with `over = M` in its options is a program over the posterior draws of the model $M$,
which Stan calls generated quantities. Its parameters include the latent variables of $M$, and its
body must be `Pure`, so it denotes a deterministic kernel $f$ from the latent space $\Theta$ of $M$
to $\llbracket \tau_2 \rrbracket$. Given the posterior kernel $\mathrm{post}$ from the data to
$\Theta$, the distribution of its output is the composite $\mathrm{post} ; f$, the
[pushforward](https://en.wikipedia.org/wiki/Pushforward_measure) of the posterior along $f$: the
posterior kernel comes first.

### 2.9 Reindexing a plate variable

A binding whose right side is `arr[idx]` *reindexes* a plate variable. Let $v : A \to B$ be the
values already drawn for a plate over $A$, and let $\iota : N \to A$ be an index map that assigns
each row to a member of $A$. The binding denotes the precomposition $v \circ \iota : N \to B$, a
deterministic kernel:

$$
\mathcal{S}\llbracket \mathsf{let}\ w = \mathit{arr}[\mathit{idx}] \rrbracket(\phi)
= \delta_{(\phi,\, v \circ \iota)}.
$$

For $v = (10, 20)$ and $\iota = (0, 0, 1)$ it gives $(10, 10, 20)$: rows of one group read the same
value, and no new value is drawn. This is a gather in PyTorch. Several index arrays,
$\mathit{arr}[\iota_1, \ldots, \iota_k]$ for $v : A_1 \times \cdots \times A_k \to B$ and
$\iota_j : N \to A_j$ on one $N$, read $v$ along the map $n \mapsto (\iota_1(n), \ldots, \iota_k(n))$,
so the entry at $n$ is $v(\iota_1(n), \ldots, \iota_k(n))$. A fixed index stands for the constant
map at its value. Index arrays on different row sets have no common $N$, and the compiler refuses
them.

### 2.10 Scores {#210-score-factor}

A score step

```
score v = expr
```

binds the value $h(\phi)$ of `expr` to $v$ and multiplies the measure by $\exp(h(\phi))$:

$$
\mathcal{S}\llbracket \mathsf{score}\ v = \mathit{expr} \rrbracket(\phi, B \times C)
= \mathbf{1}_B(\phi)\, \mathbf{1}_C\bigl(h(\phi)\bigr) \exp\bigl(h(\phi)\bigr),
$$

a kernel from $\Phi$ to $\Phi \times \mathbb R$, where $h : \Phi \to \mathbb R$ is the measurable
function `expr` denotes. `score` generalizes `observe`: where `observe` multiplies by the density
of a given value under a family, `score` multiplies by $\exp(w)$ for any expression $w$, as
`factor` does in WebPPL [@goodman2014design]. Since $h$ need not be bounded, the kernel is s-finite
but in general not finite [@staton2017commutative], and a program with a score need not have a
finite normalizing constant. Its main use is to add a differentiable quantity computed elsewhere,
such as the goal weight of a [deduction chart](grammar.md#9-chart-access-from-program-bodies), to the
program's log joint density. With $\mathit{expr} = \mathit{chart}.\mathsf{goal\_weight}()$ under the
log semiring, the score is the log inside weight of the sentence. A score step contributes the
effect `Score` (§2.8).

### 2.11 Return

`return e` ends the body. For $e = (v_1, \ldots, v_m)$, $\mathsf{ret}_e(\phi) = \delta_{(\phi(v_1), \ldots, \phi(v_m))}$,
the deterministic kernel that keeps the named values in the order written. Composed after the
body, it pushes the measure on the context forward to these values.

## 3. Parameters

A program whose parameters are bare names

```
program P (q₁, ..., qₖ) : τ₁ -> τ₂
    body
```

names the components of its domain: when $\tau_1 = \sigma_1 \times \cdots \times \sigma_k$, $q_i$ is the
$i$-th projection of the input. The denotation is the kernel from $\llbracket \tau_1 \rrbracket$ to
$\llbracket \tau_2 \rrbracket$ of §1.

### 3.1 Static and runtime telescopes

A typed program separates static specialization from values supplied to the
specialized program:

```text
program P[p₁ : P₁, ..., pₖ : Pₖ](x₁ : T₁, ..., xₘ : Tₘ) : A -> B
    body
```

The static telescope $\Delta=(p_1:P_1,\ldots,p_k:P_k)$ binds QIEC types,
effects, and indices, as well as program objects and morphisms used in checking
and specialization. Object kinds are
`FinSet`, `Space`, and `Object`; a morphism kind is `Mor[A, B]`; and a static
`Nat` records an extent. The runtime telescope
$\Gamma_v(\Delta)=(x_1:T_1,\ldots,x_m:T_m)$ binds values at those specialized
types. Scalar hyperparameters of type `Real` or `Nat` belong to this runtime
telescope when they are values read by the body. A runtime `Nat` parameter
denotes the nonnegative subset of the integers, with the discrete measurable
structure inherited from the integer carrier. Lowering records the carrier as
`Int` and retains a nonnegative constraint; native and reference input validation
enforce both integrality and nonnegativity. The corresponding static `Nat`
parameter instead ranges over index valuations in $\Delta$.
Static shape dependencies may also use Nat addition. For instance,
`Tensor[Real]([nat_add(n, 1)])` denotes vectors whose length is exactly
one more than the extent selected by `n`. This relation remains symbolic
until specialization and can supply exact binding evidence. It does not
introduce a runtime count, identify independently named axes, or add a
probabilistic factor. A zero shape-only index is admissible without
creating a categorical finite object.
A family's runtime shape parameter, such as Fréchet `concentration` or
GEV `concentration`, belongs to $\Gamma_v$ even when it controls a tail
or a support endpoint. It does not determine a tensor's index extent.
A static index is not implicitly converted into a runtime integer term.
In particular, a distribution argument `total_count=n` requires a
runtime value, such as a parameter `(n : Nat)`; a static binder
`[n : Nat]` supplies an extent and cannot fill that argument directly.
The source checker rejects this stage mismatch rather than changing the
parameter's role during specialization.

For each static valuation $\delta$, let
$D(\delta)=\llbracket\Gamma_v(\delta)\rrbracket$ be the space of admissible
runtime inputs. The program denotes

$$
\llbracket P \rrbracket \in
\prod_{\delta:\llbracket\Delta\rrbracket}
D(\delta)\to
\mathbf K\bigl(\llbracket A(\delta)\rrbracket,
               \llbracket B(\delta)\rrbracket\bigr),
$$

where $\mathbf K(X,Y)$ is the set of s-finite kernels from $X$ to $Y$.
Runtime values select a kernel over the declared domain; they are not extra
independent executions of that kernel. An input annotation adds no probability
factor. If `y` occurs as a parameter and as the response of `observe y`, its
annotation constrains the response space and the observation step contributes
the likelihood.

A Python factory may partially apply the static valuation without fixing runtime
inputs: `.sub(n=3)` selects $\delta$ while retaining the inputs in
$D(\delta)$. Fixing a scalar runtime argument by named partial application
instead selects a value in that input space and preserves its position in
$\Gamma_v$.

The distinction between stages is independent of explicitness. Elaboration may
infer an input's type from its uses, and binding may recover an omitted static
extent from exact shape evidence. The resulting checked program still has both
telescopes. An ambiguous type must be annotated, and an extent without adequate
evidence must be supplied before execution.

### 3.2 Calls of programs

A call `v <- P[s₁, ..., sₖ](a₁, ..., aₘ)` in another program first substitutes
the static arguments $s_i$ into the program's types and body, then binds the
runtime arguments $a_j$ and inlines the body as steps of the caller. The latent variables of the
callee are renamed under the fresh prefix `v$`, and its return value is bound to $v$. Substitution
commutes with the denotation of the body, because each parameter is bound at the top of the body
and the body's denotation is built from the denotations of its steps, and renaming bound variables
consistently does not change a denotation. Two calls of one program thus contribute separate
factors with their own latent variables, as in a hierarchical model.

Static substitution recurses through categorical object expressions,
including constructor dimensions such as `FinSet n` and `Real n`, products,
coproducts, slash types, and applied effects. It is not restricted to bare
axis names. The source program-call grammar currently admits only names
and signed numeric literals as static actuals; kind checking further
restricts which a binder accepts. This source restriction is separate from
the semantic valuation of a general checked telescope.

Static and runtime arities are checked separately. Runtime arguments must match
the specialized parameter types, including a tensor's element type and shape.
For a runtime `Nat` formal, the current checker admits a nonnegative integer
literal or an input already carrying the nonnegative constraint. It refuses
an arbitrary computed or sampled `Int`, whose nonnegativity it cannot prove.

A program with no runtime parameters may still have static parameters. Here
the expression specializes both the finite domain and the result dimension
at three; its empty `()` supplies no runtime values:

<!-- compile: qiec -->
```qvr
program prior[n : Nat]() : FinSet n -> Real n
    sample z : FinSet n <- Normal(loc=0.0, scale=1.0)
    return z

define fixed = prior[3]()
export fixed
```

### 3.3 Calls of computations

A step `y <- f(a₁, ..., aₖ)` that calls a computation $f$ declared with `define` has the denotation
of the body of $f$ at the arguments, sequenced into the caller. The compiled runtime expresses this
denotation as steps of the caller in four cases.

1. **Pure callees.** A body that makes no request denotes a value $\llbracket f \rrbracket(a)$, so the
   call is the deterministic kernel $\delta_{\llbracket f \rrbracket(a)}$ and its steps are `let`
   bindings. A pure computation that the body calls is substituted in turn. A call that reaches its
   own callee again has no finite unfolding and is refused.
2. **Conditionals between pure branches.** For a Boolean $c$ and pure branches,

    $$
    \llbracket \mathsf{if}\ c\ \mathsf{then}\ e_1\ \mathsf{else}\ e_2 \rrbracket
      = \mathsf{select}(c, \llbracket e_1 \rrbracket, \llbracket e_2 \rrbracket),
    $$

    a value of the same type, differentiable in each branch away from the points where $c$ changes.
    When a branch draws or scores, the set of sites, and with it the space on which the joint
    density is defined, depends on $c$. The compiled runtime fits on a fixed space, so it refuses such
    a conditional.
3. **Callees that request only `Random` and `Score`.** Such a body denotes an s-finite kernel built
   from the caller's own two effects, and inlining it is the substitution of §3.2: its site `s`
   becomes the caller's site `y$s`, and each `Score.add` adds its weight to the caller's joint. A
   callee that handles an effect inside its body is refused, since its denotation is the handler's
   fold over the body, which the caller's steps do not express.
4. **`traverse` over a lambda.** When the runs $k(x_0), \ldots, k(x_{n-1})$ read only their own
   items and names bound before the traverse, the composite of §2.3.2 is the product

    $$
    \llbracket \mathsf{traverse}(x, k) \rrbracket = \bigotimes_{i < n} \llbracket k \rrbracket(x_i),
    $$

    the kernel of a plate over the axis of $x$ (§2.4). A pure $k$ gives a deterministic tensor, and a
    $k$ that draws gives one plated site `y$s` for each site `s` of $k$. A recurrence, in which a run
    reads its predecessor, is not a product and is written with `scan`.

The [reference machine](../reference/qvr/index.md), the library's interpreter for QIEC, runs every
call as written, recursion and handlers included, so a program refused by the compiled runtime
still has its joint density there.

## 4. Composition of programs

Programs $P$ from $X$ to $Y$ and $Q$ from $Y$ to $Z$ compose as kernels:

$$
\llbracket P \mathbin{>\!\!>} Q \rrbracket(x, C) = \int \llbracket Q \rrbracket(y, C)\, \llbracket P \rrbracket(x, \mathrm{d}y).
$$

A composite is named with `define` and made public with `export`:

<!-- compile: false -->
```qvr
define pq = p >> q
export pq
```

`define N = e` extends the morphism environment with $N \mapsto \llbracket e \rrbracket_\rho$, and
`export N` adds $N$ to the module's output. A `where` block may contain nested `define` declarations,
which the compiler processes before the outer expression. For a module $M$ with the declarations
`export E_1`, ..., `export E_k`,
$\llbracket M \rrbracket_{\mathrm{exports}} = (\llbracket E_1 \rrbracket, \ldots, \llbracket E_k \rrbracket)$,
where each $E_i$ names a morphism, a program, a deduction, an encoder, a decoder, or a composite
bound by `define`.

## 5. Equations {#5-algebraic-laws-and-their-scope}

Write $\eta$ for the deterministic kernel $\phi \mapsto \delta_\phi$ and $;$ for the composition of
§1. The denotations satisfy

| Equation | Statement |
|---------|-----------|
| Left unit | $\eta ; k = k$ |
| Right unit | $k ; \eta = k$ |
| Associativity | $(k_1 ; k_2) ; k_3 = k_1 ; (k_2 ; k_3)$ |
| Commutativity | $\mathsf{sample}\ x \leftarrow t;\ \mathsf{sample}\ y \leftarrow u;\ s \;=\; \mathsf{sample}\ y \leftarrow u;\ \mathsf{sample}\ x \leftarrow t;\ s$ when $t$ does not read $y$ and $u$ does not read $x$ |

For the steps `sample`, `let`, and `return`, which denote Markov kernels, the first three equations
are the laws of the Kleisli category of the Giry monad, and commutativity holds because the Giry
monad is commutative, which is Fubini's theorem for probability measures
[@giry1982categorical; @kock1972strong]. The commutation is what lets two independent draws be
reordered without changing the denotation.

For bodies with `observe` and `score`, all four equations hold for s-finite kernels: Staton proves
associativity of their composition and the commutativity equation, including for steps that score
with unbounded weights [@staton2017commutative]. Without s-finiteness, interchanging the order of
integration can fail for kernels into infinite measures, which is why the setting is the s-finite
kernels. Steps that depend on each other, and calls with effects other than `Random` and `Score`,
stay in the order written. A finite `marginalize` is a pushforward, so it respects equality of
measures. Normalizing a scored program is a separate operation, defined when the total mass is
finite and positive.

## 6. Inference {#6-inference-and-conditioning}

A program with observations gives, for each configuration of its latent variables, an unnormalized
weight. When the total weight is finite and positive, dividing by it gives the posterior. Exact
integration, enumeration, Markov chain Monte Carlo, and variational inference approximate this
posterior or its expectations. The module [`quivers.inference`](../api/inference/svi.md) uses the
log joint density of a run to fit a variational family [@blei2017variational; @hoffman2013svi] or to
sample with [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) and the
No-U-Turn Sampler (NUTS) [@neal2011mcmc; @hoffman2014nuts].

### 6.1 Observation likelihoods and posterior analysis

For posterior draw $\theta_s$ and observation unit $i$, the extracted score is
$\ell_{si}=\log p(y_i\mid\theta_s,x_i)$. This is the raw response density;
it excludes the prior and any fitting weight. A fitting weight can change the
posterior distribution of $\theta_s$ without changing this conditional score.
Coordinates of one multivariate event are reduced together, while independent
observation axes remain separate. A marginalized joint density does not determine
pointwise response scores without a corresponding likelihood observer.

`ModelFit.analysis` and `BayesianFit.analysis` retain draws and these raw scores
for reuse. An MCMC analysis retains recorded chains and their sampler statistics.
A variational analysis draws once from the fitted guide and uses one array
pseudo-chain, which supplies no MCMC convergence evidence. Neither operation
changes the program's denotation or introduces a language effect.

For $S\geq2$ pooled draws with finite scores, the empirical WAIC estimator uses

$$
v_i=\frac{1}{S-1}\sum_s(\ell_{si}-\bar\ell_i)^2,
\qquad
\widehat{\operatorname{elpd}}_i
=\log\left(\frac1S\sum_s e^{\ell_{si}}\right)-v_i.
$$

Then `elpd_waic` sums the pointwise estimates, `p_waic` sums $v_i$, and
`waic` is $-2\widehat{\operatorname{elpd}}$. The sample-variance convention
matches [loo](https://github.com/stan-dev/loo/blob/master/R/waic.R).
For $N>1$ observation units, the reported ELPD standard error is
$\sqrt{N\operatorname{Var}_{N-1}(\widehat{\operatorname{elpd}}_i)}$;
it is undefined for one unit. A variance $v_i>0.4$ raises the
[WAIC diagnostic](https://mc-stan.org/loo/reference/waic.html).
These are finite-sample predictive estimates, with uncertainty determined by
the chosen observation units.

PSIS-LOO reweights retained posterior draws and reports importance-weight
diagnostics [@vehtari2017loo]. For variational draws, relative efficiency one
does not repair approximation error. Exact leave-one-out refits and K-fold validation instead evaluate
$\log\int p(y_i\mid\theta,x_i)\,p(d\theta\mid y_{-i})$ or its
explicit fold analogue. The shared [refit interface](../guides/model-refits.md)
retains these original row identities and sums joint response-event scores
within each posterior draw before averaging densities. ArviZ supplies PSIS
and its selected-row refit replacement. Alternative marginalization policies
remain distinct prediction questions.
Multiple response variables require explicit selection before comparison.

### 6.2 Joint reaction-time and decision responses

`Wiener(boundary=a, ndt=t0, bias=z, drift=v)` denotes a joint first-passage
law with $a>0$, $t_0\geq0$, $0<z<1$, and finite $v\in\mathbb R$.
Let $X_u=az+vu+B_u$, where $B$ is standard Brownian motion, and let
$\tau=\inf\{u>0:X_u\notin(0,a)\}$. The decision is $D=1$ at the upper
boundary and $D=0$ at the lower boundary; the measured time is
$T=t_0+\tau$. The law of $(T,D)$ has a joint density $f(t,d)$ with respect
to Lebesgue measure on time and counting measure on decisions. Integrating
over time gives each boundary's probability, rather than a density normalized
separately for each boundary.

The native scalar carrier is the signed time $S=(2D-1)T$. Its density is

$$
g(s)=f\bigl(|s|,\mathbf 1_{s>0}\bigr),\qquad |s|>t_0,
$$

and zero elsewhere. On each decision component, this change of coordinates
has absolute Jacobian one. Thus observing a signed time preserves the joint
choice probability. Observations derived by a preceding `let` follow the
ordinary lexical observation rule; the encoded value need not be supplied as
an additional external input.

The formula response modifier `dec(choice)` retains $T$ and $D$ as separate
checked data columns and emits the deterministic map to $S$ before `observe`.
The shared `ResponseModifiers` value also holds likelihood weights. A weight
$w_i$ contributes $w_i\log f(t_i,d_i)$ to the fitted objective, while raw
pointwise likelihood extraction returns $\log f(t_i,d_i)$; no prior is
weighted. This adapter changes the response representation, without
introducing another sampling effect or conditioning on the choice alone.

The upper-boundary probability and expected measured time are

$$
q=\Pr(D=1)=
\begin{cases}
\dfrac{1-e^{-2vaz}}{1-e^{-2va}}, & v\ne0,\\
z, & v=0,
\end{cases}
\qquad
\mathbb E[T]=
\begin{cases}
t_0+\dfrac{a(q-z)}{v}, & v\ne0,\\
t_0+a^2z(1-z), & v=0.
\end{cases}
$$

The native distribution's `mean` is $\mathbb E[S]$, whereas its
`mean_response_time` and formula `posterior_epred` return $\mathbb E[T]$.
`posterior_choice_probs` returns $(1-q,q)$ for each posterior draw and row.
`posterior_predict` draws $S$; `posterior_response` decodes that same joint
draw into positive time and decision, preserving chain, draw, and row axes.
Prediction samples the replayed conditional response law directly, even when
the observed training response was a deterministic lexical binding.

The density implementation switches between image and spectral expansions.
The sampler proposes a boundary and an exact one-sided hitting time, then
accepts by an interval-certified ratio of two-boundary to one-boundary
densities. Its acceptance probability is at least one half in exact
arithmetic. Finite term, precision, and proposal budgets raise an error when
a decision cannot be certified; no approximate trajectory simulation or
unresolved comparison is substituted. The mathematical and numerical checks
are in `tests/continuous/test_wiener.py`, with independent formula-level
joint-density and coefficient-gradient checks in `tests/test_formula_wiener.py`.
Current external targets report this family as unsupported.

### 6.3 Pathfinder approximations {#pathfinder-approximations}

[Pathfinder](../api/inference/pathfinder.md) [@zhang2022pathfinder]
approximates the posterior of a run on the unconstrained coordinates
$z\in\mathbb{R}^N$ that Hamiltonian Monte Carlo uses. With $T$ the
constraining bijection of the latent registry, the target is

$$
\pi(z)\propto p(T(z),y)\,\lvert\det J_T(z)\rvert .
$$

A path runs [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) on
$-\log\pi$ from $z^{(0)}\sim\operatorname{Uniform}(-r,r)^N$, with $r$
the initial radius, and records the iterates $z^{(0)},\dots,z^{(L)}$. Write
$g^{(l)}=\nabla\log\pi(z^{(l)})$, $s^{(l)}=z^{(l)}-z^{(l-1)}$ and
$\zeta^{(l)}=g^{(l-1)}-g^{(l)}$. A pair with
$s^{(l)\top}\zeta^{(l)}>10^{-12}\lVert\zeta^{(l)}\rVert^2$ updates a
positive diagonal $\alpha^{(l)}$ by the diagonal BFGS formula of
[@gilbert1989numerical]. At iterate $l$, with $S$ and $Z$ the last $J$
accepted pairs as columns, $E$ the upper triangle of $S^\top Z$ and $\eta$
its diagonal, the compact representation of [@byrd1994representations] gives

$$
\Sigma_l=\operatorname{diag}(\alpha^{(l)})+\beta\gamma\beta^\top,
\quad
\beta=\begin{bmatrix}\operatorname{diag}(\alpha^{(l)})Z & S\end{bmatrix},
\quad
\gamma=\begin{bmatrix}0 & -E^{-1}\\ -E^{-\top} &
E^{-\top}(\operatorname{diag}(\eta)+Z^\top\operatorname{diag}(\alpha^{(l)})Z)E^{-1}\end{bmatrix},
$$

the BFGS updates of $\operatorname{diag}(\alpha^{(l)})$ by those pairs, and
the approximation $q_l=\mathcal N(\mu_l,\Sigma_l)$ with
$\mu_l=z^{(l)}+\Sigma_l g^{(l)}$. An iterate whose $\Sigma_l$ is not
positive definite gives no approximation. The path keeps
$l^\ast=\arg\max_l\hat\lambda_l$, where

$$
\hat\lambda_l=\frac1K\sum_{k=1}^K\bigl(\log\pi(\phi_k)-\log q_l(\phi_k)\bigr),
\qquad \phi_k\sim q_l,
$$

estimates the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound)
up to the normalizing constant of $\pi$, and draws $M$ points from
$q_{l^\ast}$. On a Gaussian target, an exact line search makes the pairs
conjugate, so with $J\geq N$ the approximation after $N$ steps equals the
target.

With $I$ paths, the pooled draws come from the mixture
$\tfrac1I\sum_i q^{(i)}$. A draw $\phi$ of path $i$ has importance ratio
$\pi(\phi)/q^{(i)}(\phi)$ (the scheme of the paper, which leaves out the
other components). The ratios are Pareto smoothed [@vehtari2024psis], and $R$
draws are resampled from the pool with replacement with probabilities
proportional to the smoothed weights. A path without any approximation gets
weight zero. The retained draws are $T(\phi)$, read as independent draws
with one pseudo-chain: prediction, likelihood extraction, refits, and
artifacts treat them as any retained draw set. They carry no Markov chain
diagnostics, and the Pareto shape $\hat k$ of the pooled ratios measures how
far the mixture is from the target.

### 6.4 Marginal likelihoods and conditional effects of fits {#marginal-likelihoods-and-conditional-effects}

Let $T$ map the unconstrained coordinates $u\in\mathbb R^D$ of a fit's latent
sites to their supports. The fit's potential is the log joint with the log
Jacobian of $T$,

$$
\log q(u) = \log p\bigl(y, T(u)\bigr) + \log\lvert\det J_T(u)\rvert,
$$

and since every density of a compiled formula program is normalized,
$\int q(u)\,du = p(y)$, the [marginal likelihood](https://en.wikipedia.org/wiki/Marginal_likelihood).
It is defined only when every prior is proper, so improper flat priors and
explicit score potentials of unknown mass are refused. Bridge sampling
[@meng1996simulating; @gronau2017tutorial] splits the posterior draws of each
chain into halves. The first half fixes a normal proposal $g$ with its mean
$m$ and covariance $LL^\top$. The warp-3 proposal [@meng2002warp] instead
compares the symmetrized density $\tfrac12\bigl(q(u)+q(2m-u)\bigr)$ with the
standard normal after $u\mapsto L^{-1}(u-m)$. With $N_1$ draws $u_j$ of the
second half, $N_2$ proposal draws $\tilde u_i$, the ratios
$\ell_{1j}=q(u_j)/g(u_j)$ and $\ell_{2i}=q(\tilde u_i)/g(\tilde u_i)$, and
$s_1=N_\text{eff}/(N_\text{eff}+N_2)$, $s_2=1-s_1$, the estimate is the fixed
point

$$
\hat p(y) = \frac{N_1}{N_2}\,
\frac{\sum_i \ell_{2i}/(s_1\ell_{2i}+s_2\hat p(y))}
     {\sum_j 1/(s_1\ell_{1j}+s_2\hat p(y))}.
$$

It is consistent for $p(y)$ when the draws are posterior draws, which is why
variational and prior draws are refused. Posterior model probabilities are
$p(M_k\mid y)\propto \hat p(y\mid M_k)\,p(M_k)$, and model-averaged draws mix
$\operatorname{round}(w_kS)$ draws of each fit, which samples
$\sum_k w_k\,p(\cdot\mid y,M_k)$ up to the rounding.

A conditional effect of a variable set $E$ holds the remaining predictors at a
condition $c$ and reports, for each grid point $e$, the draws of
$f(e,c;\theta^{(s)})$, where $f$ is the expected response, the linear
predictor, or a response draw, with varying effects excluded unless
`re_formula` includes them. Its summaries are the median and the scaled median
absolute deviation of those draws, with quantile intervals. Nothing is
averaged over the data: an empirical average over reference rows is the
separate `prediction_grid` contract.

## 7. Wrappers that place priors on parameters {#7-bayesian-lifts}

A model whose learnable parameters have no prior still denotes a kernel, with the parameters held
fixed. The module [`quivers.inference.lifts`](../api/inference/lifts.md) turns such a model into a
[`MonadicProgram`](../api/continuous/programs.md) with an explicit prior, so that stochastic variational inference (SVI) and NUTS can
treat the parameters as latent variables, as in a Bayesian neural network [@mackay1992bayesian]. The
library calls these functions *Bayesian lifts*. Below, $\theta \in \mathbb R^D$ is the vector of all
learnable parameters of the wrapped model, $\sigma_\theta > 0$ is a fixed prior scale, and
$\mathbf z$ names optional intermediate latent variables of the wrapped model.

### 7.1 Priors on parameters

[`bayesian_lift_parameters`](../api/inference/lifts.md) takes a PyTorch module `inner_model` that
computes $\log p_{\mathrm{inner}}(\mathbf z, y \mid x, \theta)$ through a method
`log_joint(x, observations)`, and an optional map `additional_latents` from names of latent
variables $\mathbf z$ to their shapes. The returned program draws $\theta_i \sim \mathcal N(0, \sigma_\theta^2)$
for each parameter and $\mathbf z_j \sim \mathcal N(0, \sigma_z^2 I)$ for each additional latent, where
$\sigma_z$ is the argument `latent_placeholder_scale`. Its score step computes

$$
S(\theta, \mathbf{z}) = \log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta) - \sum_{j} \log \mathcal{N}\bigl(\mathbf{z}_{j} ;\, 0, \sigma_{z}^{2} I\bigr),
$$

so the log joint density of the program, the sum of the log priors of its draws and the score, is

$$
\log \pi(\theta, \mathbf z) = \log \mathcal{N}(\theta ;\, 0, \sigma_{\theta}^{2} I_{D}) + \log p_{\mathrm{inner}}(\mathbf{z}, y \mid x, \theta).
$$

The placeholder priors on $\mathbf z$ cancel, so $\sigma_z$ does not change the target density. It
can change the warmup of NUTS, which adapts its mass matrix to the draws. The target is proportional
to the posterior $p(\theta, \mathbf z \mid x, y)$ under the prior $\mathcal N(0, \sigma_\theta^2 I)$ on
$\theta$. To let gradients reach $\theta$, the program temporarily replaces each parameter of the
wrapped module by the drawn value while it evaluates `log_joint`.

### 7.2 Deterministic models with an observation family

[`lift_to_bayesian_program`](../api/inference/lifts.md) takes a deterministic module, a function $\mu$
that reads a location from the module's output, a family $\mathcal F$ (a subclass of
[`torch.distributions.Distribution`](https://docs.pytorch.org/docs/stable/distributions.html)), and the
key $k$ of the observed value in the observations. It wraps the module in a `log_joint` that scores
$y = \mathrm{observations}[k]$ under $\mathcal F(\mu(x) ; \boldsymbol\eta)$, where $\boldsymbol\eta$ holds
the family's other parameters from `observation_kwargs`, and applies §7.1 to the wrapper with no
additional latents. The log joint density is

$$
\log \pi(\theta) = \log \mathcal{N}(\theta ;\, 0, \sigma_{\theta}^{2} I_{D}) + \log p_{\mathcal{F}}\bigl(y \mid \mu(x ; \theta), \boldsymbol{\eta}\bigr),
$$

the log posterior of $\theta$ under this likelihood up to a constant.

### 7.3 Models with a log density

[`lift_from_log_prob`](../api/inference/lifts.md) does the same for a module and a function
$\ell(x, y ; \theta)$ that returns a conditional log density directly, giving
$\log \pi(\theta) = \log \mathcal N(\theta ; 0, \sigma_\theta^2 I_D) + \ell(x, y ; \theta)$. It serves
modules that already have a `log_prob` method, such as a
[`SampledComposition`](../api/continuous/morphisms.md) of Normal kernels.

### 7.4 A Monte Carlo likelihood

[`monte_carlo_log_joint`](../api/inference/lifts.md) wraps a `MonadicProgram` with intermediate
`sample` steps $\mathbf z$ that the caller does not want as latent variables of the sampler. Its
`log_joint(x, observations)` draws each $\mathbf z_j$ from its family $q_j(\mathbf z_j \mid x, \theta)$,
adds the draws to the observations, evaluates the inner `log_joint`, and subtracts the log priors
of the draws, which leaves

$$
\widetilde{\ell}(x, y ;\, \theta) = \log p_{\mathrm{inner}}\bigl(y \mid \mathbf{z}_{*}, x, \theta\bigr),
\qquad
\mathbf{z}_{j*} \sim q_{j}(- \mid x, \theta).
$$

This is a [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimate of
$\mathbb E_{\mathbf z \sim q}[\log p(y \mid \mathbf z, x, \theta)]$, not of the log marginal likelihood
$\log p(y \mid x, \theta) = \log \mathbb E_{\mathbf z \sim q}[p(y \mid \mathbf z, x, \theta)]$. By
[Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) the first is at most the
second, with equality only when $p(y \mid \mathbf z, x, \theta)$ does not depend on $\mathbf z$, so as an
estimate of the log marginal likelihood it is biased downward [@burda2016iwae].

For SVI, with a family that can be reparameterized, $\mathbf z_*$ is a differentiable function of
fixed noise $\xi$, and when differentiation and expectation may be exchanged, one draw gives an
unbiased gradient of $\mathbb E_{\mathbf z}[\log p(y \mid \mathbf z, x, \theta)]$
[@kingma2014autoencoding]. Stochastic gradient ascent then optimizes this lower bound on the log
marginal likelihood, under the usual conditions on step sizes [@robbins1951stochastic].

For NUTS the wrapper is unsuitable. A new $\mathbf z_*$ at every evaluation makes the target density
random, so the Hamiltonian and the acceptance step of NUTS no longer refer to one fixed target
[@neal2011mcmc; @hoffman2014nuts]. Using a random log density inside MCMC is the pseudo-marginal
method, which is valid only for an unbiased estimate of the density itself, not of its logarithm
[@andrieu2009pseudo]. For NUTS, declare $\mathbf z$ as an additional latent with
`bayesian_lift_parameters(..., additional_latents={'<name>': <shape>})` (§7.1), so that
$(\theta, \mathbf z)$ are sampled from one joint target with a deterministic log density.

### 7.5 Weighted deduction systems

[`nuts_program_from_deduction`](../api/stochastic/deduction/bayes.md) applies the construction of §7.1
to a [`DeductionSystem`](../api/stochastic/deduction.md) $D$ with learnable log weights
$\mathbf w \in \mathbb R^{d}$ and a corpus $s_1, \ldots, s_N$. It draws each $w_i \sim \mathcal N(0, \sigma^2)$
and adds the score $\sum_{n=1}^{N} \mathrm{goal}_D(s_n ; \mathbf w)$, where $\mathrm{goal}_D(s ; \mathbf w)$
is the chart's goal weight for the sentence $s$. When $D$ uses the log semiring, the goal weight is the
logarithm $\log Z(s ; \mathbf w)$ of the inside weight $Z$, and the log joint density is

$$
\log \pi(\mathbf{w}) = -\frac{1}{2 \sigma^{2}} \lVert \mathbf{w} \rVert_{2}^{2} + \sum_{n=1}^{N} \log Z(s_{n} ;\, \mathbf{w}) + \mathrm{const}.
$$

Whether this is the posterior $p(\mathbf w \mid s_1, \ldots, s_N)$ depends on what $Z$ is:

- If $Z(s ; \mathbf w)$ is an unnormalized weight of the sentence, the likelihood also contains the
  term $-N \log \sum_{s'} Z(s' ; \mathbf w)$ of the global normalizer, which the function does not
  compute. The target is then the posterior only when that normalizer does not depend on $\mathbf w$,
  or when the caller adds it as a score.
- If the rule weights are the probabilities of a consistent PCFG, normalized for each left-hand side,
  $Z(s ; \mathbf w)$ is the probability of the sentence, the sum over its derivations, and no further
  normalizer is needed [@chi1998estimation]. The function draws unconstrained Normal log weights,
  which do not satisfy these constraints. A caller who wants this reading parameterizes the local rule
  probabilities explicitly, for instance by Dirichlet draws [@johnson2007bayesian] or a softmax for
  each left-hand side, and scores the normalized grammar.

## External inputs and inferred finite objects

The runtime telescope $\Gamma_v(\Delta)$ contains each external symbol at its
checked type. A header parameter `x : T` adds the assumption $x:T$ to that
context. An undeclared external symbol is elaborated into the same context
when its uses determine a type; conflicting or ambiguous uses are rejected.
There is no separate body statement for declaring data, since input formation
does not change the sequence of probabilistic steps.

For fixed external values $d\in\llbracket\Gamma_v(\delta)\rrbracket$, the
program has the kernel denotation of its body evaluated at $d$. Runtime input
typing and likelihood contribution are independent: an observation response
inhabits the declared type, and `observe` contributes its log density.

An object `object A : FinSet _` introduces a natural-number extent variable $a$.
Binding data first solves the extent constraints, then specializes the checked
program at that solution. A tensor with checked dimension $a$ and supplied
length $n$ contributes the equation $a=n$. A directly read external gather index
$i$ into that axis contributes $0\leq i_j<a$ for each supplied entry.

Exact extent equations must agree. When only gather indices constrain a finite
axis and at least one index is supplied, binding uses the least admissible
cardinality, $a=1+\max_j i_j$. This choice is an explicit interpretation of the
underscore: positions absent from the observations need an explicit cardinality
or additional shape evidence. An empty index array supplies no such choice.

For a resulting cardinality $n$, specialization substitutes $n$ into every
occurrence of $a$ in the checked types and executes the ordinary finite program.
It introduces no posterior distribution over cardinalities. Runtime validation
checks supplied tensor shapes and dtypes, and rejects external gather indices
outside their checked axes. The compiler's input contract and the runtime's
checks thus describe the same specialized program.

A leading execution axis supplies one runtime-context valuation per run. The tensor
executor applies the same checked kernel to each valuation, separating runs
before a grouped marginal is evaluated. Observation plates stay inside each
execution.

## Prior-only execution

The observation scoring policy is an execution parameter shared by source and
formula fitting. With scoring enabled, an `observe` contributes its ordinary
recorded-event log density, including any declared likelihood weight. With
scoring disabled, that contribution is zero and the response density is not
evaluated. Latent `sample` scores and authored `score` potentials retain their
ordinary meanings. A compiler-generated factor that integrates a response's
latent assignment is classified as a likelihood factor and follows the same
policy; an authored factor defaults to a potential.

This distinction preserves the same prior-only target before and after exact
response marginalization. It also preserves observation clamping and sampling:
the raw pointwise recorded-event likelihood remains the unweighted event score,
and the generative view retains its original response law, including a declared
truncation. Prediction-law replay omits placeholder response scores, whereas raw
likelihood analysis explicitly enables observation scoring.

Independent prior draws use the dependency closure of the original draw sites,
including upstream deterministic bindings and conditional prior parameters.
They preserve dynamic scan addresses and constrained plate/event axes. A global
potential prevents an ancestral interpretation, so the independent API refuses
it; explicitly selected inference-based prior-only sampling retains that
potential. The `PriorDraws` record distinguishes independent samples from MCMC
chains without introducing a second prediction or likelihood calculus.

### Formula prior selection and sampled shrinkage

A formula prior selector denotes a set of retained parameter roles, rather
than a predicate over generated identifier strings. Resolution produces one
rule per target: strict selector refinement and exact-site precedence must
leave a unique winner. Empty, conflicting, or entirely shadowed rules are
errors. The target retains its nominal coefficient axes; equal cardinalities
do not identify response, group, and coefficient coordinates.

For a regularized horseshoe block with selected coordinate index $j$, the
emitted raw global scale $g$, local scales $\lambda_j$, and slab variable $h$
have proper half-Student-$t$, half-Student-$t$, and inverse-Gamma priors.
Writing $\tau=ag$ and $c^2=s_{\mathrm{slab}}^2h$, the coefficient scale is

$$s_j=\tau\lambda_j\sqrt{\frac{c^2}{c^2+\tau^2\lambda_j^2}}.$$

Here $a$ is a retained scalar residual scale for eligible autoscaled targets,
and one otherwise. The implementation evaluates this expression in the log
domain. For an R2D2 block, $R^2$ has a Beta prior and $\phi$ a Dirichlet prior;
$s_j=a\sqrt{\phi_j R^2/(1-R^2)}$. One selected coordinate has $\phi_1=1$.
Population coefficients use Normal kernels with scale $s_j$. A selected
standard deviation (classes `sd`, `sds`, `sdgp`, `sderr`, `sdcar`) is the
scale $s_j$ itself, as in brms, where R2D2 over coefficients and standard
deviations is the R2D2M2 prior. Rules marked `main` join the shrinkage rules
of one predictor into one block with the main rule's arguments. Existing
design-dependent conditional bounds are retained. A stable dependency order places each hyperparameter and scale
before its consumers without changing latent names.

These are joint prior laws over explicit latent sites, so the same program
supplies fitting, independent prior draws, and prediction replay. Bounds on
any selected prior use normalized measure restriction. For hierarchical
priors, each conditional coefficient density is divided by its interval
mass at the sampled heads; the hyperprior marginals remain unchanged.
This defines a conditional restriction rather than joint conditioning of
the entire hierarchy. The sampled global coordinate is dimensionless
when residual autoscaling is enabled, making that dependency explicit in the
coefficient kernel.

### brms default priors, flat priors, and constants

Under the brms default policy, each target without a user rule receives the
prior of `brms::default_prior`. With $y$ the scalar numeric response, $g$ the
location link (the logarithm for log-scale families), and
$m=\operatorname{round}(\operatorname{mad}(g(y)),1)$, the scale is
$\sigma_0=\max(2.5,m)$ when $m$ is finite and $2.5$ otherwise, and the
centered intercept has the Student-$t$ prior
$t_3(\operatorname{round}(\operatorname{median}(g(y)),1)-\bar o,\ \sigma_0)$
with $\bar o$ the mean offset. Standard deviations and an estimated $\sigma$
have the half Student-$t$ density $2\,t_3(s;0,\sigma_0)$ on $s>0$.
Responses that are zero are moved to $0.1$ before the logarithm or inverse
links, as in brms. The median and mad of an `mi()` response read its recorded
values only. $\operatorname{round}$ is R's: of the two neighbours
$\lfloor 10x\rfloor/10$ and $\lceil 10x\rceil/10$ it keeps the one nearer
to $x$ in double precision, the even one on a tie, so $0.45$ rounds to $0.4$.

The improper flat prior of class `b` (population, `mo()`, `me()` and `mi()`
coefficients), of conditional ARMA coefficients, and of `meanme` is the
[Lebesgue measure](https://en.wikipedia.org/wiki/Lebesgue_measure) $\lambda$
on the target's domain, with log density zero. `sdme` takes it on
$(0,\infty)$. The joint measure
$\lambda(d\beta)\,p(\theta)\,p(y\mid\beta,\theta)$ is not normalized in
$\beta$, so prior draws and marginal likelihoods are undefined and refused.
The posterior is defined whenever the likelihood is integrable in $\beta$.
An inference driver's structure-only recording run reads the site's shape
from a placeholder value and discards it. On a bounded interval $(l,u)$ the
flat prior is the uniform distribution. A constant prior replaces the latent
site $\beta$ by the binding $\beta=c$, so $\beta$ leaves the latent space and
`draws()` reports $c$.

### brms factor levels

A factor with declared levels $L$ of which the rows use $L_0\subseteq L$
contributes the treatment-coded columns of $L_0$ by default and of $L$ under
`drop_unused_levels=False`, in the declared order, so each unused level adds a
column of zeros whose coefficient has its prior as its posterior. A grouping
factor likewise indexes its group effects by $L_0$ or $L$. The retained coding
evaluates new rows at any level of $L$; the likelihood of the training rows is
the same under both, since the extra columns and effects multiply no row.

### Ordered factors, integer categories, and R expressions {#ordered-factors-integer-categories-and-r-expressions}

An ordered factor with $K$ levels and scores $s_k=k$ contributes, where it
is coded by contrasts, the columns of R's `contr.poly(K)`. Let $V$ be the
[Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix) with
entries $V_{kj}=(s_k-\bar s)^{j}$ for $j=0,\dots,K-1$ and $V=QR$ its
[QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition). The
contrast matrix is $C=[\,q_1r_{11},\dots,q_{K-1}r_{K-1,K-1}\,]$ with each
column scaled to unit length, so column $j$ is the orthonormal polynomial of
degree $j$ in the scores, and a row at level $k$ has the design entries
$C_{k\cdot}$. The columns are orthogonal to the intercept and to one
another. The contrast matrix is retained, so a new row at level $k$ is coded
by the same $C_{k\cdot}$. Where R codes the factor by indicators (a design
without an intercept), the indicator columns are kept.

An ordinal response given as positive integers has the categories
$1,\dots,K$ with $K=\max_i y_i$ and $K-1$ thresholds, or $K=n+1$ for a
declared `thres(n)`, and the program observes the position $y_i-1$. With
`thres(gr = g)` and no declared count, group $h$ has
$\max_{i:g_i=h}y_i-1$ thresholds. The retained category values map a fresh
response to its position, and a value outside them is refused.

The right-hand side of a nonlinear formula, each side of a hypothesis
string, and an expression in an addition term are R expressions over
Stan's functions. Each is translated to one QVR value expression: a power
$a$`^`$b$ becomes `pow(a, b)`, numeric literals and integer columns are
reals, and each function is its QVR composition, for instance
`inv_logit(x)` $=1/(1+e^{-x})$ as `sigmoid(x)` and `log1m(x)` as
`log1p(-x)`. The translated expression is checked and evaluated by the
ordinary QVR path, so it adds no language construct.

### brms hypothesis tables {#brms-hypothesis-tables}

For a hypothesis `left ~ right` with relation $\sim\,\in\{=,<,>\}$ and
draws $c_s=\mathrm{left}(\theta_s)-\mathrm{right}(\theta_s)$,
$s=1,\dots,S$, the table reports the mean and standard deviation of the
$c_s$, and the quantiles at $(\alpha,1-\alpha)$ for a one-sided
hypothesis or $(\alpha/2,1-\alpha/2)$ for a two-sided one. The evidence
ratio of $c>0$ is $\#\{c_s>0\}/\#\{c_s\le 0\}$, and likewise for
$c<0$. For $c=0$ it is the ratio $\hat p(0)/\hat\pi(0)$ of
[kernel density estimates](https://en.wikipedia.org/wiki/Kernel_density_estimation)
of the posterior and prior draws of $c$ at zero, with a Gaussian kernel and
the bandwidth $0.9\min(\hat\sigma,\mathrm{IQR}/1.34)\,S^{-1/5}$, as R's
`density` computes it. The posterior probability is
$\mathrm{ER}/(1+\mathrm{ER})$, which assumes equal prior probabilities of
the two hypotheses for a point hypothesis. Cost is $O(S)$ per one-sided
hypothesis and $O(S+m\log m)$ per point hypothesis for a density grid of
$m=4096$ points.

### brms intercept centering and QR coordinates {#brms-intercept-centering-and-qr-coordinates}

Let $X$ be the population design without its intercept column, $\bar x$ its
column means, and $X_c=X-\mathbf 1\bar x^\top$. With centering, the program
samples $(\alpha_c,b)$ and binds $\alpha=\alpha_c-\bar x^\top b$, so

$$\eta=\alpha+Xb=\alpha_c+X_cb.$$

The map $(\alpha_c,b)\mapsto(\alpha,b)$ is a shear with unit Jacobian, so the
posterior of $(\alpha,b)$ is the brms posterior whose intercept prior acts on
$\alpha_c$, the expected linear predictor at the predictor means. With
`decomp="QR"`, the thin [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)
$X_c=QR$ with positive diagonal $R$ gives $X_Q=Q\sqrt{N-1}$ and
$X_R=R/\sqrt{N-1}$. The program samples $b_Q$ with the class `b` priors and
binds $b=X_R^{-1}b_Q$, so $X_cb=X_Qb_Q$. The Jacobian of $b_Q\mapsto b$ is the
constant $|\det X_R^{-1}|$. Without centering, $X$ includes the intercept
column and $\alpha$ is a class `b` coefficient. The training means and
$X_R^{-1}$ are literals of the emitted program, so prediction on new rows
evaluates $\alpha+X_{\text{new}}b$ with the training coordinates, at cost
$O(NK)$ for $N$ rows and $K$ columns.

### Retained contrast expressions

A retained contrast applies one deterministic QVR value expression to each
joint parameter draw. Its input type contains the declared event axes,
represented by distinct static Nat binders; the chain and draw axes belong
to the evaluator and are absent from the expression's environment. Thus an
event reduction cannot pool posterior samples, and two equally sized but
distinct event axes cannot become equal through a shape coincidence.

The contrast uses the existing QVR parser, QIEC checker, and native value
compiler. One checked callable is lifted across the two sample axes. A
formula adapter supplies values from actual parameter-role and Gaussian-block
metadata, retaining authored coefficient, component, and grouping labels.
Centered blocks read their coefficient coordinates; noncentered blocks apply
the retained scales and Cholesky factors once. This same reconstruction
serves ordinary coefficient inspection.

For a posterior contrast $c(\theta)$ and a proper encompassing-prior stream,
the strict-region estimate compares $\{c(\theta)>c_0\}$ against its complement
through posterior log odds minus prior log odds. Both streams evaluate the
same expression jointly, retaining hierarchical dependence. Equality counts
remain descriptive and do not define point-null density evidence. The
[diagnostic contract](../api/diagnostics/hypothesis.md#region-odds-and-bayes-factors)
states the prior-compatibility and finite-sample conditions.

### Penalized Gaussian design terms

A formula smooth `s()` or `t2()` follows brms's construction through mgcv
[@wood2017generalized]. mgcv builds a raw basis $B$ (an $n\times k$ design)
and penalties $S_1,\dots,S_J$. Each penalty is rescaled to
$S_j\,\lVert B\rVert_\infty^2/\lVert S_j\rVert_1$. Unless a numeric `by`
variable varies, the training constraint $c^\mathsf{T}\beta=0$ with $c$ the
column mean of $B$ is absorbed through the Householder reflection $H$ that
maps $c$ to a multiple of the first axis, with $Z=H_{\cdot,2:k}$. For one
penalty, write $Z^\mathsf{T}SZ=V\Lambda V^\mathsf{T}$ with rank $r$, and let
$E$ scale the penalized columns of $BZV$ by $\Lambda_+^{-1/2}$ and its null
columns to the average squared norm of the penalized ones. The emitted
coefficient map is

$$
\beta=ZVE^{-1}\begin{pmatrix}\tau z\\ \gamma\end{pmatrix},
\qquad z\sim\mathcal N(0,I_r).
$$

A `t2` tensor product reparameterizes each margin this way, forms the
row-wise products of marginal penalized and null columns
[@wood2013straightforward], and gives each product block its own identity
penalty, scale $\tau_j$, and innovations $z_j$. A factor `by` variable
repeats the map once per level and sets the rows of other levels to zero.

The null-space coefficients $\gamma$ receive ordinary population priors.
Each positive scale $\tau_j$ receives the configured scale prior, selected by
the common `sds` role, and smooths sharing an mgcv `id` share their scales.
Thus the penalty defines Gaussian coordinates without pretending to
regularize an unidentified null direction. The latent program uses
$(\gamma,z,\tau)$ directly. Its existing inference transforms account for
their supports and Jacobians. No separate smooth density or fitting engine is
introduced.

`GaussianDesignTerm` retains the evaluated Gaussian design, named coordinate
ownership, structural spline knots and degree, coefficient map, training
constraint, and any shared scale. The formulae evaluator retains the raw
basis state: thin-plate eigenvector maps and unique training covariates,
cubic and B-spline knots, and the marginal maps of a tensor product. `FormulaData` preserves it by value. Fresh rows evaluate the same
basis and coefficient map. They do not choose new knots or impose a new
centering constraint. User knots (brms `knots`) replace mgcv's knot
placement in the `cr`, `cs`, `cc`, `ps` and `bs` constructors and are
retained in the same way. A factor-smooth interaction `bs = "fs"` with base
design $B$ (thin QR factorization $B=QR$) and base penalty $S$ of rank $r$
uses the columns $Qu_j/\sqrt{d_j}$ for the nonzero eigenpairs $(d_j,u_j)$ of
$R^{-\top}SR^{-1}$, followed by an orthonormal basis $N$ of the null-space
polynomials taken in increasing degree. For factor levels
$\ell=1,\ldots,L$ the row design is that base row in the block of its level,
with coefficients $b_{\ell j}\sim\mathcal N(0,\sigma_1^2)$ for $j\le r$ and
$b_{\ell,r+i}\sim\mathcal N(0,\sigma_{1+i}^2)$ for null column $i$. The
retained state is the base basis, the coefficient map and the fitted levels,
and a row of an unfitted level is refused. The base is a thin-plate smooth
or the singly penalized basis named by `xt`, and a `by` variable multiplies
the level-expanded rows. A Markov random field smooth `bs = "mrf"` over
regions $1,\ldots,R$ has the indicator design of the region factor and the
penalty $S$ given in `xt`, so its region effects $b$ have the improper
density proportional to $\exp\{-b^\top S b/(2\sigma^2)\}$
[@rue2005gaussian] under the sum-to-zero constraint over the training rows.
With `k < R` and $c_r=\max(n_r,1)$ for the training count $n_r$ of region
$r$, the retained coordinates are the eigenvectors of
$C^{-1/2}SC^{-1/2}$, $C=\operatorname{diag}(c)$, for its `k` smallest
eigenvalues, mapped by $C^{-1/2}$. The retained state is the region levels
and that map, and a row of a region outside the structure is refused. A
spline on the sphere `bs = "sos"` takes a latitude and a longitude in
degrees and has the kernel $R(p,p')=\rho_m(\cos\gamma(p,p'))$ of the angle
$\gamma$ between two locations [@wahba1981spline]. With $P_\nu$ the
[Legendre polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials),
$\rho_0(z)=\frac{1}{4\pi}\sum_{\nu\ge 1}\frac{2\nu+1}{\nu^2(\nu+1)^2}P_\nu(z)
=\frac{1}{4\pi}\{1-\pi^2/6+\operatorname{Li}_2((1+z)/2)\}$ with
$\operatorname{Li}_2$ the
[dilogarithm](https://en.wikipedia.org/wiki/Dilogarithm), and for
$m=1,\ldots,4$
$\rho_m(z)=\frac{1}{2\pi}\{q_m(z)/m!-1/(m+1)!\}$ with
$q_m(z)=\int_0^1(1-h)^m(1-2hz+h^2)^{-1/2}\,dh$. For $m=-1$ and $m=-2$ the
kernel is $r^2\log r/(8\pi)$ and $-r$ of the chord length
$r=2\sin(\gamma/2)$ [@duchon1977splines]. The null space $T$ is the constant,
joined for $m=-1$ by the three coordinates of the unit sphere. With knots
$p_1,\ldots,p_u$, knot kernel matrix $R_u$, and $U D U^\top$ its `k`
eigenpairs of largest magnitude, the raw design is
$[R(\cdot,p_j)UZ,\;T]$ with penalty $Z^\top D Z$, where the columns of $Z$
are an orthonormal basis of the null space of $T_u^\top U$
[@wood2003thin]. Every column except the constant is divided by its training
standard deviation. The retained state is the knots, $UZ$, $m$, and those
column scales, and a row that is not a latitude in $[-90,90]$ and a longitude
in $[-180,360]$ is refused. A Duchon spline `bs = "ds"` over
$x\in\mathbb R^d$ with derivative order $m$ and weight exponent $s$ has the
penalty [@duchon1977splines]

$$
J_{m,s}(f)=\int_{\mathbb R^d}\lVert\tau\rVert^{2s}\sum_{|\alpha|=m}
\frac{m!}{\alpha!}\,\bigl|\widehat{D^\alpha f}(\tau)\bigr|^2\,d\tau ,
$$

with $\widehat{g}$ the Fourier transform of $g$, $-d/2<s<d/2$, and
$m+s>d/2$. Its kernel is $E(x,x')=c\,r^{p}$ for odd $p=2m+2s-d$ and
$c\,r^{p}\log r$ for even $p$, with $r=\lVert x-x'\rVert$ and the sign
$c=(-1)^{\lfloor p/2\rfloor+1}$, and its null space $T$ is the polynomials
of degree below $m$. With centered knots $x_1,\ldots,x_u$, knot kernel
matrix $E_u$, and $UDU^\top$ its `k` eigenpairs of largest magnitude (or
$U=I$ and $D=E_u$ when $u=k$), the raw design is $[E(\cdot,x_j)UZ,\;T]$
with penalty $Z^\top DZ$, where the columns of $Z$ are an orthonormal basis
of the null space of $T_u^\top U$. The retained state is the covariate
means, the knots, $UZ$, $m$, and $s$. User knots of a thin-plate or Duchon
basis replace the unique covariate rows as $x_1,\ldots,x_u$, and every row
is then evaluated as $E(x,x_j)$ against them. Component-valued predictors retain separate Gaussian
innovations and scale coordinates for each free response component. The
[structured predictor guide](../guides/formula-special-predictors.md) states
the implemented constructor and its current frontend limits.

### Approximate Gaussian process terms

A formula term `gp(x_1, ..., x_D, k = k)` follows brms's Hilbert-space
approximation of a zero-mean
[Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process)
[@solin2020hilbert; @riutort2023practical]. For each sub-GP (one, or one per
contrast of a factor `by`) with rows $I$, the training covariates are scaled
by $\delta$, the largest pairwise distance among them ($\delta=1$ when
`scale = FALSE`), and centered by $\mu$, the mean of the scaled distinct rows
(of every row when `gr = FALSE`). With $\tilde x=x/\delta-\mu$, boundary
$L_d=c_d\max(1,\max\tilde x-\min\tilde x)$, and orders
$j\in\{1,\dots,k\}^D$,

$$
\phi_j(x)=\prod_{d=1}^{D}L_d^{-1/2}\sin\Big(\omega_{jd}(\tilde x_d+L_d)\Big),
\qquad \omega_{jd}=\frac{j_d\pi}{2L_d},
$$

and the term adds, to row $n$ with contrast value $C_n$,

$$
C_n\sum_{j}\phi_j(x_n)\exp\Big(\tfrac12\log S(\omega_j;\sigma,\ell)\Big)z_j,
\qquad z_j\sim\mathcal N(0,1),
$$

where $S$ is brms's spectral density of the chosen kernel, with marginal
standard deviation $\sigma$ (`sdgp`) and length scales $\ell$ (`lscale`, one
or one per covariate). $C_n$ is the numeric `by` value, a factor contrast, or
one. $\sigma$ and $\ell$ are latent sites with their typed priors. The
emitted program is the shared Gaussian design block with these coordinate
scales, so the joint density is that of $(\sigma,\ell,z)$ and the response,
and the existing inference transforms account for the positive supports. The
model is identified as the GP prior is: no constraint is imposed on $f$.

`GaussianDesignTerm.gp` retains the frequencies $\omega$, the kernel, the
`coef` labels, and the default length-scale priors. The formulae evaluator
retains $\delta$, $\mu$, $L$, the orders, and the `by` levels, and fresh rows
evaluate $\phi_j$ through them.

### Exact Gaussian process terms

A formula term `gp(x_1, ..., x_D)` without `k` (or with `k = NA`) is brms's
exact GP. For each sub-GP with rows $I$ (rows of nonzero contrast value), the
inputs are the distinct rows of $\{x_n / \delta : n \in I\}$ in order of
first appearance, $X = (x_1, \dots, x_u)$, with $J(n)$ the input of row $n$
(with `gr = FALSE`, every row of $I$ is its own input). The program draws
$z \sim \mathcal N(0, I_u)$ and sets

$$
f = \operatorname{chol}\big(K(X, X;\sigma,\ell) + 10^{-12} I_u\big)\, z,
\qquad
K_{ij} = \sigma^2 \prod_{d} \kappa\big(r_{ij,d} / \ell_d\big),
$$

where $\kappa$ is brms's unit-variance kernel ($e^{-t^2/2}$, $e^{-t}$,
$(1+\sqrt3 t)e^{-\sqrt3 t}$, or $(1+\sqrt5 t+5t^2/3)e^{-\sqrt5 t}$), and the
product has one factor of the Euclidean distance $r_{ij}$ for an isotropic
GP and one factor of $r_{ij,d}=|x_{id}-x_{jd}|$ per covariate otherwise. Row
$n$ receives $C_n f_{J(n)}$. `cholesky` is a checked QVR builtin: its value is
the lower-triangular $L$ with positive diagonal and $L L^\top = A$, and a
matrix that is not numerically positive definite gives a factor whose entries
are not numbers, so the density there is undefined. The joint density is that
of $(\sigma,\ell,z)$ and the response, the non-centered parameterization of
$f \sim \mathcal N(0, K + 10^{-12} I)$.

The block retains $X$, $\delta$, `gr`, the kernel, and the labels. For
$m$ distinct fresh inputs $X_*$ (computed with the retained $\delta$ and
`gr`), prediction draws, for each posterior draw,

$$
f_* \mid f \sim \mathcal N\Big(K_*^\top K^{-1} f,\;
K(X_*, X_*) - K_*^\top K^{-1} K_* + 10^{-8} I_m\Big),
\qquad K_* = K(X, X_*),
$$

with $K$ including its $10^{-12}$ diagonal, as brms's `.predictor_gp_new`
does. The draw is a fresh site of the prediction program that replay draws
from this law under the replayed $(\sigma, \ell, z)$.

### Monotonic effects

A formula term `mo(x)` follows brms and Bürkner and Charpentier
[@burkner2020modelling]. An integer predictor is coded as
$x_n-\min_m x_m$ over the training rows and an ordered factor by the position
of its level among the observed training levels, giving codes in
$\{0,\dots,D\}$ with $D\ge 1$. The term contributes

$$
b\,\operatorname{mo}(x_n,\zeta),\qquad
\operatorname{mo}(x,\zeta)=D\sum_{i=1}^{x}\zeta_i=D\,(M\zeta)_n,
\qquad \zeta\sim\operatorname{Dirichlet}(\alpha),
$$

where $\zeta$ lies on the probability simplex of dimension $D$ and $M$ is the
$n\times D$ matrix with $M_{ni}=1$ when $i\le x_n$. The default concentration is
$\alpha=(1,\dots,1)$. An interaction term multiplies the values of its
`mo()` factors, each with its own simplex, by the product $c_n$ of its other
components, and receives one coefficient $b$. A term whose other components
evaluate to several columns yields one such term per column. Factors with the
same `id` share one simplex. A group-level slope on the same term adds
$r_{j[n]}\,\operatorname{mo}(x_n,\zeta)\,c_n$ with the population simplex,
so the varying coefficient is $b+r_j$, as in brms.

$b$ is identified because $\zeta$ sums to one, so $bD$ is the contribution of
the highest category. When every code occurs in training and $b\ne 0$, the
contributions determine $\zeta$ as well. An integer predictor can skip codes
between its minimum and maximum, and then the data leave the split of the
effect across the skipped steps to the Dirichlet prior. The simplex is a
latent site whose support transform contributes the Jacobian, as for any other
simplex. The program retains $(b,\zeta)$ directly and evaluates
$M\zeta$ as one matrix product. `MonotonicTerm` retains $M$, $c$, the training
coding, and brms's labels. Fresh rows are coded with the training minimum or
levels, and codes outside $\{0,\dots,D\}$ are refused.

### Grouping terms and multi-membership

The right side of a group-level bar follows brms. `gr(g, cor, id, cov)`
denotes the factor `g`. `cor = FALSE` makes each coefficient its own block,
and `id` joins blocks as the bar notation `|id|` does. A block is a matrix
$B$ of $C$ coefficient coordinates over the $J$ levels of its factor, where a
coefficient over response categories or ordinal transitions has one
coordinate per category or transition. Its law is the product of the factors
the term declares. With a $J\times C$ matrix $Z$ of standard-normal
innovations, the rows of $B$ are

$$
b_j=\mu+m_j\,A_{b(j)}\,g_j,\qquad
A_b=\operatorname{diag}(\sigma_{\cdot,b})\,L_b,\qquad
G=F Z .
$$

- `cov = M` takes a known covariance $K=FF^\mathsf{T}$ between levels from
  `data2`. Otherwise $F=I$.
- `by = f` gives $f$'s level $b(j)$ containing $j$ its own standard
  deviations $\sigma_{\cdot,b}$ and correlation factor $L_b$. Otherwise there
  is one of each. Independent coordinates have $L=I$.
- `dist = "student"` draws $\nu\sim\operatorname{Gamma}(2,0.1)$ restricted to
  $\nu>1$ and $u_j\sim\text{inverse-}\chi^2(\nu)$ once per factor, and sets
  $m_j=\sqrt{\nu u_j}$, so the coefficients are multivariate Student-t with
  $\nu$ degrees of freedom. Otherwise $m_j=1$.
- `pw = w` adds $(w_j-1)\log\phi(z_j)$ per level to the log density, so the
  prior of $z_j$ enters with weight $w_j$ as in brms.
- $\mu$ is the mean of grouped threshold log spacings, and zero for
  group-level terms.

Noncentered coordinates sample $Z$. Centered coordinates without a known
covariance sample $b_j\sim\mathcal N(\mu,\,m_j^2A_{b(j)}A_{b(j)}^\mathsf T)$.
Centered coordinates with a known covariance sample the deviations before
mixing in the support of $K$: matrix normal with row covariance $K$ on its
support and column covariance $AA^\mathsf T$, or with a `by` partition and $K$
of full rank the Gaussian vector with covariance blocks
$K_{jl}A_{b(j)}A_{b(l)}^\mathsf T$. Prior weights of a centered block score
$z_j$ recovered by the triangular solves
$Z=F^{-1}G$, $g_j=A_{b(j)}^{-1}(b_j-\mu)/m_j$, which is the noncentered
posterior after the change of variables. The change of variables needs
$F$ invertible, so prior weights are refused for $K$ of deficient rank, and so
are centered coordinates with `by` there, since the support of $B$ then depends
on $A$.

A donor of a new level under an empirical policy is a fitted level with the
same $b(j)$. Under the Gaussian policy a new level draws its sites from the
same law, and with a known covariance from its conditional given the fitted
rows of $G$.

An interaction `g1:g2`
is the factor whose levels are the observed combinations, labeled by pasting
the source labels with `_`, and nesting `g1/g2` denotes the two factors `g1`
and `g1:g2`.

A multi-membership factor `mm(g_1, ..., g_K)` has the union of the member
columns' levels. For a group-level coefficient $r$ with level values
$r_1,\dots,r_L$ from the ordinary Gaussian group layout, row $n$ receives

$$
\sum_{k=1}^{K} W_{kn}\, r_{J_{kn}}\, Z_{kn},
$$

where $J_{kn}$ is the level named by member $k$, $W_{kn}$ its weight, and
$Z_{kn}$ the coefficient's covariate for that member: one, a column or
monotonic value shared by all members, or the $k$-th column of `mmc()`. With
`by = cbind(f_1, ..., f_K)`, column $f_k$ gives the `by` level of the level
that member $k$ names, and every level of the factor has one. Equal weights are $1/K$, and
authored weights are divided by their row sum unless `scale = FALSE`
[@browne2001multiple]. The program evaluates one gather per member and
never forms a rows-by-levels matrix. `Membership` retains the training codes
and weights, and `MemberCovariate` the values $Z_{kn}$. Prediction codes all
members of the fresh rows on one compact level axis, so the new-level
policies of the shared group layout apply to every member alike.

### Joint multivariate formula responses {#joint-multivariate-formula-responses}

A `MultivariateFormula` with responses $r=1,\dots,R$ is one program. Each
response keeps its own family $F_r$, linear predictors $\eta_{r,i}$ for its
location and distributional parameters $\theta_{r,i}$ at row $i$, and its
own priors. A shared group identifier places the coefficients $b_{g}$ of
every response that names it in one block,
$b_g\sim\mathcal N(0,\operatorname{diag}(\tau)\,\Omega\,\operatorname{diag}(\tau))$
for each level $g$, with $\Omega\sim\operatorname{LKJ}(\zeta)$
[@lewandowski2009lkj]. Without residual correlation the joint density is

$$
p(\beta,b,\tau,\Omega)\prod_{r=1}^{R}\prod_{i=1}^{n_r}
F_r\!\left(y_{r,i}\mid\theta_{r,i}\right),
$$

so a row's responses are conditionally independent given the shared
coefficients. With residual correlation every response is Gaussian (or
every response is Student with one shared $\nu$), the rows are common to
all responses, and the row event $y_i=(y_{1,i},\dots,y_{R,i})$ has the law

$$
y_i\sim\mathcal N_R\!\left(\mu_i,\;
\operatorname{diag}(\sigma_i)\,L L^\mathsf T\operatorname{diag}(\sigma_i)\right)
\quad\text{or}\quad
y_i\sim t_{\nu,R}\!\left(\mu_i,\;
\operatorname{diag}(\sigma_i)\,L L^\mathsf T\operatorname{diag}(\sigma_i)\right),
$$

with $\mu_{i}=(\mu_{1,i},\dots,\mu_{R,i})$, $\sigma_i$ the responses'
scales, and $L\sim\operatorname{LKJCholesky}(1)$ the Cholesky factor of the
residual correlation matrix. This is the parameterization brms emits
(`diag_pre_multiply(sigma, Lrescor)` in `multi_normal_cholesky` and
`multi_student_t`) [@burkner2017brms]. The correlation matrix
$LL^\mathsf T$ has unit diagonal, which identifies the scales $\sigma$
separately from the correlations.

Every response is emitted by the single-response compiler and renamed into
the namespace `response_<resp>__`. The program therefore has one prior
catalogue: a `PriorSpec` with `response=` selects one response's targets,
and the residual factor is the target `rescor`. For prediction the fit
retains each response's fitted design evaluators and the joint group
layouts. Fresh rows are bound per response and the specialized joint
program is compiled once per request. A new level of a shared factor is
drawn once for the whole block. Pointwise log-likelihood units are the row
event $y_i$ under residual correlation, a marginal of that event for a
strict subset of responses, and otherwise the sum over selected responses
of $\log F_r(y_{r,i}\mid\theta_{r,i})$, as brms `log_lik` computes.

### Residual dependence structures {#residual-dependence-structures}

A formula autocorrelation term replaces the independent Gaussian or Student
response by one law over all response rows, partitioned into series $g$ by the
grouping factor and ordered by time within each series. Write $\eta$ for the
linear predictor, $r=y-\eta$, and $\sigma$ for the residual scale.

The conditional ARMA formulation defines, within a series,
$e_t=r_t-\sum_{k\le q}\theta_k e_{t-k}$ with $e_t=0$ before the series
starts, and the conditional mean

$$
\mu_t=\eta_t+\sum_{k\le q}\theta_k e_{t-k}+\sum_{k\le p}\phi_k e_{t-k},
\qquad y_t\mid y_{<t}\sim\mathcal N(\mu_t,\sigma_t^2)\ \text{or}\ t_\nu(\mu_t,\sigma_t),
$$

which is brms's Stan loop. Because the truncated autoregressive and
moving-average operators commute, $y-\mu$ equals the innovations of the
filter $A r=M\varepsilon$, and the joint density is the product of these
conditionals. The parameters are unrestricted. Retained for prediction: the
row ordering, the series boundaries, and the response history.

A covariance formulation gives series $g$ of length $n_g$ the law
$y_g\sim\mathcal N(\eta_g,D_gR_gD_g)$, or the multivariate Student law with
the same scale matrix and one mixing variable per series, with
$D_g=\operatorname{diag}(\sigma_g)$. The stationary ARMA(1,1) correlation is
$R_{ii}=(1+\theta^2+2\phi\theta)/(1-\phi^2)$ and
$R_{ij}=(1+\phi\theta)(\phi+\theta)\phi^{|i-j|-1}/(1-\phi^2)$, with
$|\phi|<1$, so $\sigma$ is the innovation scale. Compound symmetry uses
$R=(1-\rho)I+\rho\mathbf 1\mathbf 1^\mathsf T$ with $\rho\in[0,1]$. The
unstructured correlation is $C_{J_g,J_g}$ for a
[Cholesky factor](https://en.wikipedia.org/wiki/Cholesky_decomposition)
$C=LL^\mathsf T$ over the retained time levels $J_g$ of the series. A fixed
covariance uses $R=M$ over all rows. Retained for prediction: the series
boundaries and, for unstructured correlations, the training time levels.

A simultaneous autoregression with weights $W$ and $A=I-\rho W$ has lag form
$Ay=\eta+\varepsilon$ and error form $A(y-\eta)=\varepsilon$, with
independent Gaussian innovations or one multivariate Student vector
$\varepsilon$, and

$$
\log p(y)=\log p_\varepsilon(\varepsilon(y))+\sum_i\log(1-\rho\lambda_i),
$$

where $\lambda_i$ are the eigenvalues of $W$ [@ord1975estimation]. The prior
support of $\rho$ is $(\min_i\lambda_i,\max_i\lambda_i)$, as in brms.
A value of $\rho$ whose factor $1-\rho\lambda_i$ lies within the rounding
error of the computed eigenvalue, $|\rho|\,10^{3}n\epsilon\max(1,\max_i|\lambda_i|)$
for Float64 machine epsilon $\epsilon$ (or $|\rho|$ times the machine epsilon of
the scoring dtype when that is larger), has log density $-\infty$, since $A$
is then singular to working precision.
Retained for prediction: the nonzero entries of $W$ and its eigenvalues.

The pointwise likelihood of a covariance or SAR row is the leave-one-out
predictive density $p(y_i\mid y_{-i},\theta)$ within its series. With
precision $C=\Sigma^{-1}$, $g=C(y-\mathbb E y)$, and $c_i=C_{ii}$, the
Gaussian density has mean $y_i-g_i/c_i$ and variance $1/c_i$. The Student
density has $\nu+n-1$ degrees of freedom and variance
$c_i^{-1}(\nu+\beta_i)/(\nu+n-1)$ with
$\beta_i=(y-\mathbb E y)^\mathsf T g-g_i^2/c_i$ [@burkner2021efficient].

Three combinations follow brms. Known standard errors $s$ with a time-series
covariance give series $g$ the scale matrix
$D_gR_gD_g+\operatorname{diag}(s_g^2)$, which requires the estimated $\sigma$.
Retained for prediction: the authored source of $s$, read again on new rows.
A conditional ARMA term beside a simultaneous autoregression or a fixed
covariance replaces $\eta$ by the conditional mean $\mu$ above in that law,
with the series of the ARMA term coded by its own grouping and time and the
matrix indexed by the rows as given. Retained for prediction: both codings
and the response history. A censored or truncated row of the conditional ARMA
formulation computes $e_t$ from the recorded response (the lower endpoint of
an interval) and scores, under $F_t=\mathcal N(\mu_t,\sigma_t^2)$ or
$t_\nu(\mu_t,\sigma_t)$ and truncation bounds $[a_t,b_t]$,

$$
\log\frac{f_t(y_t)}{F_t(b_t)-F_t(a_t)},\qquad
\log\frac{1-F_t(y_t)}{F_t(b_t)-F_t(a_t)},\qquad
\log\frac{F_t(y_t)}{F_t(b_t)-F_t(a_t)},\qquad
\log\frac{F_t(u_t)-F_t(y_t)}{F_t(b_t)-F_t(a_t)}
$$

for an exact, right-censored, left-censored, and interval-censored row with
upper endpoint $u_t$, each event intersected with the truncation interval.

A conditional autoregressive term adds $r_{\ell(i)}$ to the linear predictor
of row $i$, where $\ell(i)$ is the row's location. With adjacency $W$ (one per
pair of neighbors), neighbor counts $D$, and $\tau=\sigma_{\mathrm{car}}^{-2}$,
the proper law is $r\sim\mathcal N\big(0,[\tau(D-\alpha W)]^{-1}\big)$ with

$$
\log\det\big(\tau(D-\alpha W)\big)=n\log\tau+\sum_i\log d_i+\sum_i\log(1-\alpha\mu_i),
$$

where $\mu_i$ are the eigenvalues of $D^{-1/2}WD^{-1/2}$. The intrinsic law
has precision $\tau L$ with the Laplacian $L=D-W$ on the subspace where $r$
sums to zero within each connected component $C$. Its density of the free
coordinates $z$, with $r=Ez$ filling the last location of each component, is

$$
\log p(z)=\sum_C\tfrac12\big[(m_C-1)\log\tfrac{\tau}{2\pi}+\log\big(m_C\operatorname{pdet}L_C\big)\big]-\tfrac\tau2\sum_{i\sim j}(r_i-r_j)^2,
$$

since $\det(E_C^\mathsf TL_CE_C)=m_C\operatorname{pdet}L_C$ for a component
of $m_C$ locations. BYM2 sets
$r=\sigma_{\mathrm{car}}(\sqrt{1-\rho}\,\theta+\sqrt{\rho/s_C}\,\phi)$, with
$\theta\sim\mathcal N(0,I)$, unit-precision intrinsic $\phi$, and $s_C$ the
geometric mean of $\operatorname{diag}(L_C^{+})$ [@riebler2016intuitive].
Retained for prediction: the location labels, the edge list, and the
scaling factors.

Latent residuals for a family without natural residuals add a vector
$u$ over rows to the linear predictor: $u=(I+\sum_k\phi_kL^k)\varepsilon$
within each series with $\varepsilon\sim\mathcal N(0,\sigma_{\mathrm{err}}^2I)$
for a conditional AR term, and $u_g\sim\mathcal N(0,\sigma_{\mathrm{err}}^2R_g)$
for a covariance term, with $R_g$ as above.

### Missing values and measurement error {#missing-values-and-measurement-error}

An `mi()` response $x$ with family law $F$ and parameters $\theta_i$ has
recorded rows $O$ and missing rows $M$. Its contribution to the joint
density is $\prod_{i\in O}F(x_i\mid\theta_i)\prod_{i\in M}F(x^\ast_i\mid\theta_i)$,
with a latent coordinate $x^\ast_i$ for each missing row. This is brms's
`normal_lpdf(Yl | mu, sigma)` over the completed vector $\tilde x$ with
$\tilde x_O=x_O$ and $\tilde x_M=x^\ast$, since brms gives $x^\ast$ a flat
prior. With known errors $s_i$ (`mi(sdx)`), every row has a latent truth
$x^\ast_i\sim F(\theta_i)$ and each recorded row adds
$\log\varphi\bigl((x_i-x^\ast_i)/s_i\bigr)-\log s_i$. A predictor term
`mi(x)` contributes $\beta\,c_i\prod_k\tilde x^{(k)}_{j_k(i)}$ to the linear
predictor, where $c_i$ is the column with every `mi()` variable set to one
and $j_k$ joins rows by original row identity or by `index()` identities.
As the slope of a group-level term over a grouping factor with level
$\ell(i)$ in row $i$, the same column contributes
$r_{\ell(i)}\,c_i\prod_k\tilde x^{(k)}_{j_k(i)}$ with the group's varying
coefficient $r$, so the row's coefficient of the completed values is
$\beta+r_{\ell(i)}$, as in brms.
The program has no other density for $\tilde x$: no value is filled in.

Under estimated residual correlations the responses of row $i$ form one
vector event, and the completed vector
$\tilde y_i=(\tilde y_{i1},\ldots,\tilde y_{iK})$ is scored by the joint
law, $\mathcal N_K(\tilde y_i\mid\mu_i,L_iL_i^\top)$ or the multivariate
Student law with shared $\nu$, where $L_i=\operatorname{diag}(\sigma_i)L$
and $L$ is the Cholesky factor of the residual correlation. This is brms's
`multi_normal_cholesky_lpdf(Yl | Mu, LSigma)`. Each missing coordinate
$x^\ast_i$ (under `mi(sdx)`, each true value) is a latent site under the
improper flat law, so its density is the joint law's alone. The program
adds the row densities as one log-density factor computed by forward
substitution, $z_{ik}=\bigl(\tilde y_{ik}-\mu_{ik}-\sum_{j<k}L_{i,kj}z_{ij}\bigr)/L_{i,kk}$,
giving $-\tfrac K2\log 2\pi-\sum_k\log L_{i,kk}-\tfrac12\sum_k z_{ik}^2$
for the Normal law and
$\log\Gamma\bigl(\tfrac{\nu+K}2\bigr)-\log\Gamma\bigl(\tfrac\nu2\bigr)-\tfrac K2\log(\nu\pi)-\sum_k\log L_{i,kk}-\tfrac{\nu+K}2\log\bigl(1+\tfrac1\nu\sum_k z_{ik}^2\bigr)$
for the Student law, each multiplied by the row's weight. The locations
$\mu_i$ may use completed values of any response of the same row,
including responses that use each other. For prediction the fit rebuilds
the vector observation with $\mu_i$ evaluated at the replayed completed
values. Pointwise scores are those of the recorded vector and are defined
on rows where every selected response is recorded.

An `me(x, sdx)` predictor has latent values
$X_u=\mu+\sigma\,(Lz)_u$ on its truth axis $u$ (rows, or levels of `gr`),
with $z\sim\mathcal N(0,I)$, $L$ the identity or, under `set_mecor`, an
LKJ Cholesky factor shared by the response's terms with the same grouping,
and each recorded value $x_i\sim\mathcal N(X_{u(i)},s_i)$. The
hyperparameters $\mu$, $\sigma$, and $L$ are the `meanme`, `sdme`, and
`corme` prior targets. Fresh rows use recorded values of $x$ in place of
$\tilde x$ or $X$. Training-row predictions replay $x^\ast$ and $X$ from
each posterior draw. Pointwise log-likelihoods of an `mi()` response are
defined on its recorded rows only.

### Finite mixtures and category-specific predictors {#finite-mixtures-and-category-specific-predictors}

A formula [`mixture`](../api/formulas/index.md#quivers.formulas.mixture)
of component laws $F_1,\ldots,F_K$ observes each row through

$$
p(y_i)=\sum_{k=1}^{K}\theta_{ik}\,F_k\bigl(y_i\mid\mu_{ik},\phi_{ik}\bigr),
$$

where $\mu_{ik}=g_k^{-1}(\eta_{ik})$ and the auxiliary parameters
$\phi_{ik}$ of component $k$ each have their own linear predictor or scalar
latent, exactly as in the ordinary emission of $F_k$. Mixing proportions
are one simplex $\theta\sim\mathrm{Dirichlet}(1,\ldots,1)$ shared by all
rows, fixed values divided by their sum, or
$\theta_{i\cdot}=\operatorname{softmax}(0,\eta^{\theta}_{i2},\ldots,\eta^{\theta}_{iK})$
with the omitted component as reference. Under `order = "mu"` the sampled
location intercepts satisfy $\alpha_k=\alpha_1+\sum_{j<k}\delta_j$ with
$\delta_j=\exp(u_j)>0$, a bijection from
$(\alpha_1,u)\in\mathbb R^K$ onto the ordered cone, so every relabelling
of the components but the identity leaves the parameter space. Truncation
to $[l_i,u_i]$ normalizes each component:
$p(y_i)=\sum_k\theta_{ik}F_k(y_i)/F_k([l_i,u_i])$. A censored row replaces
$F_k(y_i)$ by the component mass of its recorded event. Prediction retains
the same program: $\mathrm E[Y_i]=\sum_k\theta_{ik}\,\mathrm E_k[Y_i]$, and
the membership probability is
$\theta_{ik}F_k(y_i)/\sum_j\theta_{ij}F_j(y_i)$.

For ordinal components each $F_k$ is the category law of its family,
with thresholds $\tau_{k1},\ldots,\tau_{k,C-1}$ and predictor
$\eta_{ik}$, so the mixture has category probabilities
$\Pr(Y_i=c)=\sum_k\theta_{ik}\,\Pr\nolimits_k(Y_i=c\mid\eta_{ik},\tau_k)$.
Every component writes its thresholds as
$\tau_{kj}=\kappa_{kj}-\alpha_k$ with a location intercept $\alpha_k$ and
cutpoints constrained by $\sum_j\kappa_{kj}=0$. Under `order = "none"` each
component has its own $(\alpha_k,\kappa_k)$. Under `order = "mu"` all
components read one pair $(\alpha,\kappa)$, sampled in the coordinates of
the first cumulative component when there is one, where
$\kappa_1<\cdots<\kappa_{C-1}$, and of the first component otherwise.

A categorical or simplex response with free categories $c\neq r$ (the
reference $r$) has linear predictors $\eta_{ic}$ and $\eta_{ir}=0$. With
category-specific formulas each $\eta_{ic}$ has its own design, coefficients
and group effects, and the category probabilities (or mean composition)
are $\operatorname{softmax}(\eta_{i\cdot})$. Each $\eta_{ic}$ is an
ordinary scalar predictor, so a monotonic effect contributes
$b_c\,D\sum_{j\le x_i}\zeta_{cj}$ with its own simplex $\zeta_c$ of length
$D$, a Gaussian process term its own marginal standard deviation, length
scale and coordinates, and a multi-membership factor
$\sum_m w_{im}\,r_{c,\,j_m(i)}$ with its own standard deviation. A main
formula holding one of these terms is the same formula for every free
category. Prediction retains each category's simplex, hyperparameters and
group coefficients under its `dpar_mu<c>_` prefix. A count response with
`rate(d)` uses the law of $F$ at mean $d_i\mu_i$ and, for the negative
binomial, shape $d_i\varphi_i$, so its expectation is $d_i\mu_i$.
