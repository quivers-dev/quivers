# Point-Estimate Fitting

`adam_fit_deduction` fits a deduction's learnable log weights $\mathbf{w}$ by maximizing the
corpus objective $\sum_n \log Z(s_n; \mathbf{w})$
([charts as differentiable values](../../../guides/deduction.md#charts-as-first-class-differentiable-values)),
optionally with an isotropic Normal [prior](bayes.md) on $\mathbf{w}$. Each $\log Z$ is the
exact fixed point of the chart under the log semiring, and automatic differentiation through it
gives the exact gradient $\nabla_\mathbf{w} \log Z(s; \mathbf{w}) = \mathbb{E}_{d \mid s}[\phi(d)]$,
the expected feature counts of the inside-outside algorithm [@goodman1999semiring]. The objective
is a corpus log likelihood only when the grammar is normalized.

::: quivers.stochastic.deduction.fit
