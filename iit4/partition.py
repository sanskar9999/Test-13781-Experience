"""Partition utilities for IIT 4.0 demo.

Generates directional bipartitions (not full Bell lattice, per user spec
IIT 4.0 restricts to directional bipartitions of purviews).
For big-Phi system cuts, we generate splits of the full system units.
"""
from __future__ import annotations
import itertools


def bipartitions(n: int) -> list[tuple[list[int], list[int]]]:
    """All bipartitions of n units into two non-empty groups (unordered)."""
    units = list(range(n))
    parts = []
    # generate subsets of size 1..n//2 to avoid duplicates, but need all for directional
    seen = set()
    for r in range(1, n):
        for comb in itertools.combinations(units, r):
            other = tuple(u for u in units if u not in comb)
            # canonical ordering to avoid duplicate (A,B) vs (B,A)
            key = tuple(sorted([tuple(sorted(comb)), tuple(sorted(other))]))
            # use frozenset to dedup
            fs = frozenset([frozenset(comb), frozenset(other)])
            if fs in seen:
                continue
            seen.add(fs)
            parts.append((list(comb), list(other)))
    return parts


def all_directional_cuts(n: int) -> list[tuple[list[int], list[int]]]:
    """Directional cuts: ordered partitions (A->B direction matters for some definitions).

    For demo we return both directions as distinct when needed, but big_phi uses unordered.
    """
    und = bipartitions(n)
    directed = []
    for a, b in und:
        directed.append((a, b))
        directed.append((b, a))
    # dedup directional duplicates when symmetric
    # keep unique ordered
    uniq = []
    seen = set()
    for a, b in directed:
        key = (tuple(a), tuple(b))
        if key not in seen:
            seen.add(key)
            uniq.append((a, b))
    return uniq
