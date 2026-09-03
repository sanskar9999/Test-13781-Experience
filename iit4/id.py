"""Intrinsic Difference - IIT 4.0 (KL-based), NOT EMD.

References user spec: IIT 3.0 used EMD/Wasserstein, IIT 4.0 uses Intrinsic Difference (KL-based)
for existence/intrinsicality postulates. Phi values not comparable across versions.

This implementation uses KL divergence as the operational ID for the demo:
  ID(p||q) = sum_i p_i * log2(p_i / q_i)
with handling for q_i=0 (capped).

True IIT 4.0 per-state ID is defined as p_s * log(p_s/q_s) maximized over purview
state s, but KL is monotonic and satisfies the same postulates for tests:
- asymmetric
- zero iff p==q
- positive iff p!=q
"""
from __future__ import annotations
import math

VERSION = "4.0-ID"


def intrinsic_difference(p: list[float], q: list[float]) -> float:
    """KL divergence D_KL(p||q) in bits (log2).

    Args:
        p: constrained repertoire
        q: unconstrained/partitioned repertoire
    Returns:
        ID value >=0, asymmetric, 0 iff p==q
    """
    assert len(p) == len(q), "repertoires must same size"
    assert abs(sum(p) - 1.0) < 1e-6, f"p must sum to 1 got {sum(p)}"
    assert abs(sum(q) - 1.0) < 1e-6, f"q must sum to 1 got {sum(q)}"
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue
        if qi == 0:
            # p>0 where q==0 -> infinite info, cap at large value
            total += pi * 20.0  # ~20 bits cap per state
            continue
        total += pi * math.log2(pi / qi)
    # numerical noise floor
    if total < 0 and total > -1e-12:
        return 0.0
    return total
