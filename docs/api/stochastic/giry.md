# Giry Monad

The [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) sends a space to the space of probability
distributions on it [@giry1982categorical]. Its unit puts all mass on one point, and its
multiplication averages a distribution of distributions. `GiryMonad` implements it on finite sets,
where distributions are stochastic vectors, and `FinStoch` is its
[Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category), the finite sets with stochastic
matrices [@fritz2020synthetic] ([the Giry monad and FinStoch](../../guides/stochastic.md#the-giry-monad-and-finstoch)).
Scoring by an unnormalized likelihood leaves this monad and needs measure kernels.

::: quivers.stochastic.giry
