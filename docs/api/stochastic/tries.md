# Tries

A relation over variables $v_1, \dots, v_k$ is kept as a
[trie](https://en.wikipedia.org/wiki/Trie): a nested dictionary from the value of $v_1$ to a
trie over the rest, whose leaves are what each tuple stands for. [Generic Join](joins.md) binds
the variables of a query one at a time, and the values of a variable are the intersection, over
the tries that contain it, of the values they allow below the variables already bound, found by
iterating the smallest and probing the others.
[`intersect`][quivers.stochastic.tries.intersect] is that step. Rule-body
[joins](joins.md), the [indexed chart](indexed.md) and the [agenda](agenda.md) all use it, so its
cost, at most the AGM bound of the query for each variable, holds for each of them.

::: quivers.stochastic.tries
