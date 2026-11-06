# Giry Monad

The Giry monad treats a probability distribution as a value that can itself be returned
or combined. Its unit concentrates mass at one point; its multiplication averages a
distribution of distributions. The finite tensor operations here realize restricted
instances of that idea. Scoring by an unnormalized likelihood requires additional
measure structure.

For the sequencing pattern that the probability monad instantiates, see Milewski's [Kleisli categories chapter](https://bartoszmilewski.com/2014/12/23/kleisli-categories/) [@milewski2014kleisli].

See [probability distributions and Kleisli composition](../../guides/stochastic.md#the-giry-monad-and-finstoch) and
[a Markov kernel calculation](../../tutorials/python/02-stochastic-relations.md#composition-of-markov-kernels).

For the measure-theoretic construction, see [@giry1982categorical]; for stochastic maps in a categorical setting, see [@fritz2020synthetic].

The Giry monad and probability distributions.

::: quivers.stochastic.giry
