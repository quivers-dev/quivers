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
does not repair approximation error. Exact leave-one-out refits, K-fold
validation, and alternative marginalization policies remain separate workflows.
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
