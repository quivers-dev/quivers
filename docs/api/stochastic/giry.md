# Giry Monad

The [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) sends a space to the space of probability
distributions on it [@giry1982categorical]. On finite sets its finite-support analogue, the
[distribution monad](https://ncatlab.org/nlab/show/distribution+monad), has stochastic matrices as
the arrows of its [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category), and `FinStoch`
is that category, the finite sets with stochastic matrices [@fritz2020synthetic]. `GiryMonad`
represents these Kleisli arrows by their matrices: its unit is the identity matrix and its Kleisli
composition is the matrix product
([the Giry monad and FinStoch](../../guides/stochastic.md#the-giry-monad-and-finstoch)).
Scoring by an unnormalized likelihood leaves this monad and needs measure kernels.

::: quivers.stochastic.giry
