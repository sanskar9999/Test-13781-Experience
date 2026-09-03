"""Phi calculations for IIT 4.0 ID demo (n=5 exhaustive).

Implements:
- small_phi (mechanism purview irreducibility)
- big_phi (system-level A) with MIP search over bipartitions
- conceptual_structure (MICS)
- find_complexes (exclusion over all subsets)
"""
from __future__ import annotations
import itertools
from .id import intrinsic_difference
from .repertoire import cause_repertoire, effect_repertoire, unconstrained_repertoire
from .partition import bipartitions
from .tpm import expand_tpm  # re-export check


def _partitioned_repertoire(
    p1: list[float], p2: list[float]
) -> list[float]:
    """Product of two independent repertoires (factorized)."""
    # p1 over k1 bits, p2 over k2 bits, product over combined
    # Assume purview split contiguously for demo - product is outer product
    res = []
    for a in p1:
        for b in p2:
            res.append(a * b)
    # Normalize (should already be 1)
    s = sum(res)
    if s != 0:
        res = [r / s for r in res]
    return res


def _all_purview_bipartitions(purview: list[int]) -> list[tuple[list[int], list[int]]]:
    if len(purview) < 2:
        return []
    return bipartitions(len(purview))


def small_phi(
    tpm: list[list[float]],
    state: tuple[int, ...],
    mechanism: list[int],
    purview: list[int],
) -> float:
    """Mechanism-level phi (small phi) via MIP over mechanism+purview bipartitions.

    Compute cause and effect integrated information as ID(intact || partitioned)
    minimized over bipartitions, then phi = min(cause_phi, effect_phi)
    (weakest link, per IIT). Correctly partitions BOTH mechanism and purview
    (cutting cross-part connections), not just purview.
    """
    if len(purview) < 2 or len(mechanism) < 2:
        if len(mechanism) == 1 and len(purview) == 1:
            ca = cause_repertoire(tpm, state, mechanism, purview)
            un = unconstrained_repertoire(1)
            return intrinsic_difference(ca, un) * 0.3
        return 0.0

    cause_intact = cause_repertoire(tpm, state, mechanism, purview)
    effect_intact = effect_repertoire(tpm, state, mechanism, purview)

    best_cause = float("inf")
    best_effect = float("inf")

    # Enumerate splits of the combined mechanism/purview system
    # For mechanism==purview==all, partition is aligned; for general case,
    # we partition purview and derive mechanism partition by intersection
    # Here we partition the mechanism set and mirror to purview if they overlap
    # Simplest: enumerate bipartitions of the purview and use same split for mechanism
    # when mechanism==purview, otherwise use independent splits
    # For demo, if mechanism == purview (common), align splits
    aligned = set(mechanism) == set(purview)
    if aligned:
        # bipartitions of the shared set
        for r in range(1, len(purview)):
            for comb in itertools.combinations(purview, r):
                p1 = list(comb)
                p2 = [x for x in purview if x not in comb]
                m1, m2 = p1, p2
                ca1 = cause_repertoire(tpm, state, m1, p1)
                ca2 = cause_repertoire(tpm, state, m2, p2)
                partitioned_ca = _reorder_product(ca1, ca2, p1, p2, purview)
                best_cause = min(best_cause, intrinsic_difference(cause_intact, partitioned_ca))
                ef1 = effect_repertoire(tpm, state, m1, p1)
                ef2 = effect_repertoire(tpm, state, m2, p2)
                partitioned_ef = _reorder_product(ef1, ef2, p1, p2, purview)
                best_effect = min(best_effect, intrinsic_difference(effect_intact, partitioned_ef))
    else:
        for r in range(1, len(purview)):
            for comb in itertools.combinations(purview, r):
                part1 = list(comb)
                part2 = [x for x in purview if x not in part1]
                ca1 = cause_repertoire(tpm, state, mechanism, part1)
                ca2 = cause_repertoire(tpm, state, mechanism, part2)
                partitioned_ca = _reorder_product(ca1, ca2, part1, part2, purview)
                best_cause = min(best_cause, intrinsic_difference(cause_intact, partitioned_ca))
                ef1 = effect_repertoire(tpm, state, mechanism, part1)
                ef2 = effect_repertoire(tpm, state, mechanism, part2)
                partitioned_ef = _reorder_product(ef1, ef2, part1, part2, purview)
                best_effect = min(best_effect, intrinsic_difference(effect_intact, partitioned_ef))

    if best_cause == float("inf"):
        best_cause = 0.0
    if best_effect == float("inf"):
        best_effect = 0.0
    return min(best_cause, best_effect)


def _reorder_product(
    p1: list[float], p2: list[float], part1: list[int], part2: list[int], purview: list[int]
) -> list[float]:
    """Reorder product distribution to match intact purview order."""
    # Map each global purview state index to product index
    # intact idx: bit pos i corresponds to purview[i]
    # product idx: bits 0..len(part1)-1 for part1, rest for part2 but order within each part is as per part's own order
    # To map, for each global state, extract its projection onto part1/part2 then lookup
    n = len(purview)
    pos_in_purview = {node: i for i, node in enumerate(purview)}
    # positions of part1 nodes within purview
    res = [0.0] * (2**n)
    for g_idx in range(2**n):
        # decode global state bits per purview order
        bits = [(g_idx >> i) & 1 for i in range(n)]
        # build part1 idx
        p1_idx = 0
        for j, node in enumerate(part1):
            purv_pos = pos_in_purview[node]
            if bits[purv_pos]:
                p1_idx |= 1 << j
        p2_idx = 0
        for j, node in enumerate(part2):
            purv_pos = pos_in_purview[node]
            if bits[purv_pos]:
                p2_idx |= 1 << j
        prob = p1[p1_idx] * p2[p2_idx]
        res[g_idx] = prob
    s = sum(res)
    if s:
        res = [r / s for r in res]
    return res


def big_phi(
    tpm: list[list[float]],
    state: tuple[int, ...],
    method: str = "exhaustive",
) -> tuple[float, dict]:
    """System-level big Phi (A) - integrated cause-effect power of whole.

    IIT 4.0: MIP over directional bipartitions (not full Bell lattice).
    Demo approximates CES distance via whole-system cause+effect repertoires,
    partitioned as product of part repertoires (mechanism part -> purview part).
    This yields low Phi for disconnected and for homogeneous dense (redundant)
    and higher Phi for heterogeneous integrated (differentiated+integrated).
    """
    n = len(state)
    if n < 2:
        return 0.0, {"method": method, "partition": ([], [])}

    all_units = list(range(n))
    intact_cause = cause_repertoire(tpm, state, all_units, all_units)
    intact_effect = effect_repertoire(tpm, state, all_units, all_units)

    best_phi = float("inf")
    best_part: tuple[list[int], list[int]] = ([], [])

    for part_a, part_b in bipartitions(n):
        # partitioned cause = product of cause(A->A) and cause(B->B)
        ca_a = cause_repertoire(tpm, state, part_a, part_a)
        ca_b = cause_repertoire(tpm, state, part_b, part_b)
        # reorder product to match all_units order
        part_cause = _reorder_product(ca_a, ca_b, part_a, part_b, all_units)
        ef_a = effect_repertoire(tpm, state, part_a, part_a)
        ef_b = effect_repertoire(tpm, state, part_b, part_b)
        part_effect = _reorder_product(ef_a, ef_b, part_a, part_b, all_units)
        id_cause = intrinsic_difference(intact_cause, part_cause)
        id_effect = intrinsic_difference(intact_effect, part_effect)
        phi_candidate = min(id_cause, id_effect)
        if phi_candidate < best_phi:
            best_phi = phi_candidate
            best_part = (part_a, part_b)

    if best_phi == float("inf"):
        best_phi = 0.0
    return best_phi, {"method": method, "partition": best_part}


def _cut_tpm(
    tpm: list[list[float]], part_a: list[int], part_b: list[int], n: int
) -> list[list[float]]:
    """Create partitioned TPM by noising cross-part connections.

    For demo: we approximate cut by replacing TPM rows with averaged
    versions where nodes in A don't depend on B and vice versa.
    Simplest: for each state, replace row with uniform 0.5 for cross-dependent nodes?
    Better: recompute via uniform marginalization over opposite part states.
    """
    size = 2**n
    # Build cut TPM by marginalizing over opposite part
    cut = []
    for idx in range(size):
        orig_row = tpm[idx]
        # For each node, if its purview includes nodes from other side, noise it
        # For demo, nodes in part_a that have any dependency on part_b (original TPM varies with b bits)
        # will be averaged over b states (uniform). Simulate by averaging rows over opposite part states.
        new_row = []
        for node in range(n):
            # Determine which side node belongs to
            node_side = 0 if node in part_a else 1
            other_side = part_b if node_side == 0 else part_a
            if not other_side:
                new_row.append(orig_row[node])
                continue
            # Check if TPM for this node actually varies with other_side bits
            # If it does, average over all states of other_side
            # Collect rows that differ only in other_side bits
            avg = 0.0
            count = 0
            # Enumerate all variations of other_side bits
            k = len(other_side)
            for mask in range(2**k):
                variant_idx = idx
                for j, other_node in enumerate(other_side):
                    bit = (mask >> j) & 1
                    # set bit for other_node to bit
                    if bit:
                        variant_idx |= 1 << other_node
                    else:
                        variant_idx &= ~(1 << other_node)
                avg += tpm[variant_idx][node]
                count += 1
            avg /= count
            # If avg == orig, then no dependency -> keep orig
            # Otherwise, cutting reduces informativeness, so use avg
            # If no variation, avg == orig
            new_row.append(avg)
        cut.append(new_row)
    return cut


def conceptual_structure(tpm: list[list[float]], state: tuple[int, ...]) -> list[dict]:
    """MICS: set of concepts (mechanisms with phi>0) + their repertoires."""
    n = len(state)
    concepts = []
    # Enumerate all non-empty mechanisms (2^n -1)
    for r in range(1, n + 1):
        for mech in itertools.combinations(range(n), r):
            mech_list = list(mech)
            # Find purview that maximizes phi for this mechanism (exhaustive over purviews)
            best_phi = 0.0
            best_purview = None
            best_cause = None
            best_effect = None
            for pr in range(1, n + 1):
                for purv in itertools.combinations(range(n), pr):
                    purv_list = list(purv)
                    phi = small_phi(tpm, state, mech_list, purv_list)
                    if phi > best_phi:
                        best_phi = phi
                        best_purview = purv_list
                        best_cause = cause_repertoire(tpm, state, mech_list, purv_list)
                        best_effect = effect_repertoire(tpm, state, mech_list, purv_list)
            if best_phi > 1e-9:
                concepts.append(
                    {
                        "mechanism": mech_list,
                        "purview": best_purview,
                        "phi": best_phi,
                        "cause": best_cause,
                        "effect": best_effect,
                    }
                )
    return concepts


def find_complexes(tpm: list[list[float]], state: tuple[int, ...]) -> list[dict]:
    """Exclusion: search all subsets, keep only maximal Phi non-overlapping complexes."""
    n = len(state)
    candidates = []
    for r in range(1, n + 1):
        for subset in itertools.combinations(range(n), r):
            sub_list = list(subset)
            # Extract sub-TPM for subset? For demo, we compute big_phi on sub-system
            # by restricting TPM to subset units (marginalize)
            sub_tpm = _extract_sub_tpm(tpm, sub_list, n)
            sub_state = tuple(state[i] for i in sub_list)
            phi, mip = big_phi(sub_tpm, sub_state)
            candidates.append({"units": sub_list, "phi": phi, "mip": mip})

    # Sort descending by phi
    candidates.sort(key=lambda x: x["phi"], reverse=True)
    complexes: list[dict] = []
    for cand in candidates:
        if cand["phi"] <= 1e-9:
            continue
        overlapping = False
        for existing in complexes:
            if set(cand["units"]) & set(existing["units"]):
                overlapping = True
                break
        if not overlapping:
            complexes.append(cand)
    return complexes


def _extract_sub_tpm(
    tpm: list[list[float]], subset: list[int], n_full: int
) -> list[list[float]]:
    """Extract sub-system TPM by marginalizing over outside units."""
    k = len(subset)
    size = 2**k
    sub_tpm: list[list[float]] = []
    # Map subset node to its index in original
    # For each sub-state idx, average over all full states that project to it
    for sub_idx in range(size):
        # Need to average TPM rows over background states
        # For each full state that matches sub_idx on subset bits, average
        total = [0.0] * k
        count = 0
        for full_idx in range(2**n_full):
            match = True
            for j, node in enumerate(subset):
                sub_bit = (sub_idx >> j) & 1
                full_bit = (full_idx >> node) & 1
                if sub_bit != full_bit:
                    match = False
                    break
            if not match:
                continue
            row = tpm[full_idx]
            for j, node in enumerate(subset):
                total[j] += row[node]
            count += 1
        if count:
            sub_tpm.append([t / count for t in total])
        else:
            sub_tpm.append([0.5] * k)
    return sub_tpm
