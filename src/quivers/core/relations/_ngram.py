"""Deterministic suffix states for sparse contiguous feature sets."""

from collections import deque
from collections.abc import Callable


class _NgramAutomaton:
    def __init__(
        self,
        patterns: tuple[tuple[int, ...], ...],
        charge: Callable[[int], None] | None = None,
    ) -> None:
        edges: list[dict[int, int]] = [{}]
        failure = [0]
        emissions: list[list[int]] = [[]]
        for index, pattern in enumerate(patterns):
            if charge is not None:
                charge(len(pattern))
            state = 0
            for symbol in pattern:
                target = edges[state].get(symbol)
                if target is None:
                    target = len(edges)
                    edges[state][symbol] = target
                    edges.append({})
                    failure.append(0)
                    emissions.append([])
                state = target
            emissions[state].append(index)

        ready = deque(edges[0].values())
        while ready:
            parent = ready.popleft()
            for symbol, child in edges[parent].items():
                fallback = failure[parent]
                while fallback and symbol not in edges[fallback]:
                    if charge is not None:
                        charge(1)
                    fallback = failure[fallback]
                failure[child] = edges[fallback].get(symbol, 0)
                output = emissions[failure[child]]
                if charge is not None:
                    charge(1 + len(output))
                emissions[child].extend(output)
                ready.append(child)
        self._edges = edges
        self._failure = failure
        self.emissions = tuple(map(tuple, emissions))
        self._cache: dict[tuple[int, int], int] = {}

    def step(
        self, state: int, symbol: int, charge: Callable[[int], None] | None = None
    ) -> int:
        key = state, symbol
        target = self._cache.get(key)
        if target is None:
            while state and symbol not in self._edges[state]:
                if charge is not None:
                    charge(1)
                state = self._failure[state]
            target = self._edges[state].get(symbol, 0)
            self._cache[key] = target
        return target
