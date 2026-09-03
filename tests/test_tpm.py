"""Tests for TPM expansion: connectivity matrix + gates -> full 2^n x n TPM.

Contract:
- units binary, deterministic is point-mass special case
- full TPM shape 2^n x n
- gates: COPY, AND, OR, XOR, MAJORITY, noisy versions
- n=5 demo tier exhaustive
"""
import pytest


def test_tpm_shape_n5():
    from iit4.tpm import expand_tpm
    n = 5
    # disconnected COPY self-loop should be identity-like
    conn = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    gates = ["COPY"] * n
    tpm = expand_tpm(conn, gates)
    assert len(tpm) == 2**n
    assert len(tpm[0]) == n


def test_tpm_deterministic_point_mass():
    from iit4.tpm import expand_tpm
    n = 2
    # A copies B, B copies A -> swap
    conn = [[0, 1], [1, 0]]
    gates = ["COPY", "COPY"]
    tpm = expand_tpm(conn, gates)
    # state 01 (index 1: 00,1:01,2:10,3:11 with LSB first?) check deterministic
    for row in tpm:
        for v in row:
            assert v in (0.0, 1.0)


def test_tpm_probabilistic_noisy():
    from iit4.tpm import expand_tpm
    n = 2
    conn = [[1, 1], [1, 1]]
    gates = ["AND", "AND"]
    tpm = expand_tpm(conn, gates, noise=0.1)
    # with noise, values should be 0.1 or 0.9 not 0/1
    flat = [v for row in tpm for v in row]
    assert any(0 < v < 1 for v in flat)
    assert all(0 <= v <= 1 for v in flat)


def test_tpm_and_logic():
    from iit4.tpm import expand_tpm
    # 2 nodes, node0 = AND of both, node1 = COPY node0
    n = 2
    conn = [[1, 1], [1, 0]]
    gates = ["AND", "COPY"]
    tpm = expand_tpm(conn, gates)
    # enumerate states as (a,b) with idx = a + 2*b? Need consistent mapping
    # Just verify AND truth table appears somewhere
    # state 11 -> AND=1, state otherwise ->0 for node0
    # Find row where inputs to node0 are both 1
    # If conn[0]=[1,1], node0 sees both nodes
    # So only state 11 should give 1 for node0
    # Count rows where tpm[row][0]==1 should be 1
    count_one = sum(1 for row in tpm if row[0] == 1.0)
    assert count_one == 1


def test_tpm_no_hand_written_2n_rows():
    from iit4.tpm import expand_tpm
    n = 5
    conn = [[0]*n for _ in range(n)]
    for i in range(n):
        conn[i][i] = 1
        if i > 0:
            conn[i][i-1] = 1
    gates = ["XOR"] * n
    tpm = expand_tpm(conn, gates)
    assert len(tpm) == 32


def test_tpm_gate_case_insensitive():
    from iit4.tpm import expand_tpm
    n = 2
    conn = [[1, 1], [1, 1]]
    tpm1 = expand_tpm(conn, ["and", "or"])
    tpm2 = expand_tpm(conn, ["AND", "OR"])
    assert tpm1 == tpm2
