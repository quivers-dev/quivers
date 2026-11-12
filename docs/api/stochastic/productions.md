# Productions

A deduction with a lexicon is a grammar. Its rules and its lexicon entries are productions, each
rewriting the category its conclusion names, its left-hand side. Under the `normalize=lhs`
deduction option, the productions that share a left-hand side form a probability distribution, as
in a [probabilistic context-free grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar)
[@booth1973applying; @chi1998estimation]. The
[deduction guide](../../guides/deduction.md#production-probabilities-per-left-hand-side) defines the
left-hand side of each rule form and the deductions the option refuses.

[`DeductionSystem.production_groups`](agenda.md#quivers.stochastic.agenda.DeductionSystem.production_groups)
returns the groups, and
[`DeductionSystem.production_log_probs`](agenda.md#quivers.stochastic.agenda.DeductionSystem.production_log_probs)
the log-probability of each production.

::: quivers.stochastic.productions
