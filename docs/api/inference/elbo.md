# Objectives

A [variational objective](../../guides/inference-svi.md#elbo-evidence-lower-bound) measures how well
a guide approximates the posterior. The
[evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) is the expected
log joint density minus the expected log density of the guide $q$, and its gap from the log
evidence is the KL divergence $\mathrm{KL}(q \,\|\, p)$ between the guide and the posterior $p$
[@blei2017variational; @hoffman2013svi; @pyroSVI]. The module gives `ELBO`, `IWAEBound` [@burda2016iwae], `RenyiBound` [@li2016renyi],
`VRIWAEBound` [@daudel2023vriwae], `ChiVI` [@dieng2017chi], `RWS` [@bornschein2015rws], and
`DReGsBound` [@tucker2019dreg]. Each pairs with a gradient [estimator](estimators.md).

::: quivers.inference.objectives
