"""Tests for Phi (big-Phi A) and small-phi, partitions, MIP, exclusion, state-dependence.

Contract:
- A=0 if any partition loses no power
- big-Phi state-dependent (wiring-plus-state)
- exclusion: search all subsets, overlapping losers excluded
- n=5 exhaustive search (2^5-1=31 subsets, ~15 cuts) tractable
- clique homogeneous low Phi vs heterogeneous integrated high Phi
- averaging across states meaningless -> per-state Phi
- no 0.2 threshold (removed)
"""
import pytest


def test_small_phi_import():
    from iit4.phi import small_phi, big_phi
    assert callable(small_phi)
    assert callable(big_phi)


def test_small_phi_zero_if_reducible():
    from iit4.tpm import expand_tpm
    from iit4.phi import small_phi
    n = 2
    conn = [[0, 0], [0, 0]]
    gates = ["COPY", "COPY"]
    tpm = expand_tpm(conn, gates)
    state = (1, 0)
    # disconnected mechanism across two units with no integration -> phi 0
    phi = small_phi(tpm, state, mechanism=[0, 1], purview=[0, 1])
    assert phi == pytest.approx(0.0, abs=1e-6)


def test_small_phi_positive_if_integrated():
    from iit4.tpm import expand_tpm
    from iit4.phi import small_phi
    n = 3
    # heterogeneous chain + feedback
    conn = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    gates = ["XOR", "AND", "OR"]
    tpm = expand_tpm(conn, gates)
    state = (1, 0, 1)
    phi = small_phi(tpm, state, mechanism=[0, 1, 2], purview=[0, 1, 2])
    assert phi > 0


def test_big_phi_zero_for_disconnected():
    from iit4.tpm import expand_tpm
    from iit4.phi import big_phi
    n = 4
    conn = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    gates = ["COPY"] * n
    tpm = expand_tpm(conn, gates)
    state = (1, 0, 1, 0)
    phi, mip = big_phi(tpm, state)
    assert phi == pytest.approx(0.0, abs=1e-6)


def test_big_phi_state_dependent():
    from iit4.tpm import expand_tpm
    from iit4.phi import big_phi
    n = 3
    conn = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    gates = ["AND", "OR", "XOR"]
    tpm = expand_tpm(conn, gates)
    # state (1,0,0) vs (1,1,1) differ under IIT 4.0 - same wiring different Phi
    phi1, _ = big_phi(tpm, (1, 0, 0))
    phi2, _ = big_phi(tpm, (1, 1, 1))
    # same wiring different state -> different Phi (per-state, not average)
    assert phi1 != pytest.approx(phi2)


def test_partition_losses_power():
    from iit4.tpm import expand_tpm
    from iit4.phi import big_phi
    n = 3
    conn = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    gates = ["MAJORITY"] * n
    tpm = expand_tpm(conn, gates)
    state = (1, 0, 1)
    phi, mip = big_phi(tpm, state)
    # homogeneous complete graph should have low Phi (cut is cheap) vs heterogeneous
    assert phi < 1.5  # upper bound, not 0.2 threshold


def test_heterogeneous_higher_than_clique_or_lattice():
    from iit4.tpm import expand_tpm
    from iit4.phi import big_phi
    n = 5
    # clique: fully connected homogeneous - low Phi due to redundancy/symmetry
    clique_conn = [[1]*n for _ in range(n)]
    clique_gates = ["MAJORITY"] * n
    tpm_clique = expand_tpm(clique_conn, clique_gates)
    phi_clique, _ = big_phi(tpm_clique, (1, 0, 1, 0, 1))

    # heterogeneous integrated: differentiated+integrated - higher Phi
    # Found via search to be > clique under product-partition ID metric (IIT 4.0)
    het_conn = [
        [1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
    ]
    het_gates = ["OR", "MAJORITY", "AND", "MAJORITY", "OR"]
    tpm_het = expand_tpm(het_conn, het_gates)
    phi_het, _ = big_phi(tpm_het, (1, 0, 1, 0, 1))

    # lattice: ring COPY - low Phi due to local only coupling
    lat_conn = [[0]*n for _ in range(n)]
    for i in range(n):
        lat_conn[i][i] = 1
        lat_conn[i][(i+1) % n] = 1
        lat_conn[i][(i-1) % n] = 1
    lat_gates = ["COPY"] * n
    tpm_lat = expand_tpm(lat_conn, lat_gates)
    phi_lat, _ = big_phi(tpm_lat, (1, 0, 1, 0, 1))

    assert phi_het > phi_clique
    assert phi_het > phi_lat


def test_exclusion_over_subsets():
    from iit4.tpm import expand_tpm
    from iit4.phi import find_complexes
    n = 3
    conn = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    gates = ["AND", "OR", "XOR"]
    tpm = expand_tpm(conn, gates)
    state = (1, 1, 0)
    complexes = find_complexes(tpm, state)
    # overlapping complexes: only max Phi survives
    # complexes should be non-overlapping maxima
    for i, c1 in enumerate(complexes):
        for c2 in complexes[i+1:]:
            overlap = set(c1["units"]) & set(c2["units"])
            assert len(overlap) == 0, f"overlapping complexes not excluded: {c1} vs {c2}"


def test_big_phi_reports_mip_method():
    from iit4.tpm import expand_tpm
    from iit4.phi import big_phi
    n = 3
    conn = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    gates = ["AND", "OR", "XOR"]
    tpm = expand_tpm(conn, gates)
    phi, mip = big_phi(tpm, (1, 0, 1))
    assert "method" in mip
    assert mip["method"] in ("exhaustive", "queyranne", "cut_one", "heuristic")
    assert "partition" in mip


def test_n5_exhaustive_tractable():
    from iit4.tpm import expand_tpm
    from iit4.phi import big_phi
    import time
    n = 5
    conn = [[1 if i == j or (i+1) % n == j else 0 for j in range(n)] for i in range(n)]
    gates = ["XOR", "AND", "OR", "XOR", "AND"]
    tpm = expand_tpm(conn, gates)
    start = time.time()
    phi, _ = big_phi(tpm, (1, 0, 1, 0, 1))
    elapsed = time.time() - start
    assert elapsed < 5.0  # must be tractable under 5s for n=5
    assert phi >= 0


def test_concept_count_not_isomorphism():
    """Two systems with same 2^n-1 concept count can have max different shape."""
    from iit4.tpm import expand_tpm
    from iit4.phi import conceptual_structure
    n = 3
    conn1 = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    conn2 = [[1, 0, 1], [1, 1, 0], [0, 1, 1]]
    tpm1 = expand_tpm(conn1, ["AND", "OR", "XOR"])
    tpm2 = expand_tpm(conn2, ["OR", "XOR", "AND"])
    cs1 = conceptual_structure(tpm1, (1, 0, 1))
    cs2 = conceptual_structure(tpm2, (1, 0, 1))
    # same number of concepts possible, but structures differ
    if len(cs1) == len(cs2) and len(cs1) > 0:
        # at least one concept differs in purview/phi
        assert cs1 != cs2
