"""Cause/effect repertoires with do-operator + uniform perturbation.

- condition on actual current state of mechanism
- compare against unconstrained uniform repertoire (do-operator on rest)
"""
from __future__ import annotations


def unconstrained_repertoire(purview_size: int) -> list[float]:
    n_states = 2 ** purview_size
    return [1.0 / n_states] * n_states


def _purview_index(full_state: tuple[int, ...], purview: list[int]) -> int:
    idx = 0
    for pos, node in enumerate(purview):
        if full_state[node]:
            idx |= 1 << pos
    return idx


def cause_repertoire(
    tpm: list[list[float]],
    state: tuple[int, ...],
    mechanism: list[int],
    purview: list[int],
) -> list[float]:
    """P(past_purview | mechanism_state) via Bayes with uniform prior.

    Likelihood: product over mechanism nodes of P(mech_i observed | past)
    """
    n = len(state)
    size = 2 ** n
    mech_vals = [state[m] for m in mechanism]
    purv_size = len(purview)
    n_purv_states = 2 ** purv_size
    counts = [0.0] * n_purv_states

    for past_idx in range(size):
        past = tuple((past_idx >> j) & 1 for j in range(n))
        # likelihood of observed mechanism given past
        likelihood = 1.0
        tpm_row = tpm[past_idx]
        for mi, m_node in enumerate(mechanism):
            p1 = tpm_row[m_node]
            obs = mech_vals[mi]
            likelihood *= p1 if obs == 1 else (1 - p1)
            if likelihood == 0:
                break
        purv_idx = _purview_index(past, purview)
        counts[purv_idx] += likelihood

    total = sum(counts)
    if total == 0:
        return unconstrained_repertoire(purv_size)
    return [c / total for c in counts]


def effect_repertoire(
    tpm: list[list[float]],
    state: tuple[int, ...],
    mechanism: list[int],
    purview: list[int],
) -> list[float]:
    """P(future_purview | mechanism_state) via intervention.

    Do-operator: fix mechanism to current values, uniform over background
    past states consistent with mechanism, then forward via TPM.
    For demo, we marginalize over all full past states that match mechanism
    (intervention) and propagate product-form future.
    """
    n = len(state)
    size = 2 ** n
    purv_size = len(purview)
    n_purv_states = 2 ** purv_size

    # Find all past states that match mechanism intervention
    # If mechanism is subset, background uniform over other nodes?
    # For simplicity, iterate over all states where mechanism equals observed
    # and average their forward purview distributions
    mech_vals = tuple(state[m] for m in mechanism)
    matching_indices = []
    for idx in range(size):
        past = tuple((idx >> j) & 1 for j in range(n))
        if all(past[m] == v for m, v in zip(mechanism, mech_vals)):
            matching_indices.append(idx)

    if not matching_indices:
        return unconstrained_repertoire(purv_size)

    agg = [0.0] * n_purv_states
    for idx in matching_indices:
        tpm_row = tpm[idx]
        # product distribution over purview future: assume independent given past
        # distribution over purview states: product over purview nodes
        for purv_idx in range(n_purv_states):
            prob = 1.0
            for pos, p_node in enumerate(purview):
                bit = (purv_idx >> pos) & 1
                p1 = tpm_row[p_node]
                prob *= p1 if bit == 1 else (1 - p1)
            agg[purv_idx] += prob
    total = sum(agg)
    if total == 0:
        return unconstrained_repertoire(purv_size)
    return [a / total for a in agg]
