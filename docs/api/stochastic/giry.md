# Giry Monad

The [Giry monad](../../guides/stochastic.md#the-giry-monad-and-finstoch) treats a probability distribution as a value that can itself be returned
or combined. Its unit concentrates mass at one point; its multiplication averages a
distribution of distributions [@giry1982categorical]. The finite tensor operations here realize restricted
instances of that idea. Scoring by an unnormalized likelihood requires additional
measure structure.

The Giry monad and probability distributions.

::: quivers.stochastic.giry
