"""Tests for cause/effect repertoires with do-operator + uniform perturbation.

Contract per user:
- condition each mechanism's repertoire on its actual current state
- compare against unconstrained uniform (max-entropy) repertoire via do-operator
- intrinsic info = divergence from uniform
"""
import pytest


def test_repertoire_import():
    from iit4.repertoire import cause_repertoire, effect_repertoire
    assert callable(cause_repertoire)
    assert callable(effect_repertoire)


def test_unconstrained_is_uniform():
    from iit4.repertoire import unconstrained_repertoire
    # purview size 2 -> 4 states uniform = 0.25 each
    p = unconstrained_repertoire(purview_size=2)
    assert p == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_cause_repertoire_sums_to_one():
    from iit4.tpm import expand_tpm
    from iit4.repertoire import cause_repertoire
    n = 3
    conn = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    gates = ["COPY"] * n
    tpm = expand_tpm(conn, gates)
    state = (1, 0, 1)
    # mechanism [0] constraining purview [0]
    p = cause_repertoire(tpm, state, mechanism=[0], purview=[0])
    assert abs(sum(p) - 1.0) < 1e-9


def test_effect_repertoire_sums_to_one():
    from iit4.tpm import expand_tpm
    from iit4.repertoire import effect_repertoire
    n = 3
    conn = [[0]*n for _ in range(n)]
    for i in range(n):
        conn[i][i] = 1
    gates = ["COPY"] * n
    tpm = expand_tpm(conn, gates)
    state = (1, 0, 1)
    p = effect_repertoire(tpm, state, mechanism=[0], purview=[1])
    assert abs(sum(p) - 1.0) < 1e-9


def test_copy_mechanism_is_informative():
    from iit4.tpm import expand_tpm
    from iit4.repertoire import cause_repertoire, unconstrained_repertoire
    from iit4.id import intrinsic_difference
    n = 2
    conn = [[1, 0], [0, 1]]
    gates = ["COPY", "COPY"]
    tpm = expand_tpm(conn, gates)
    state = (1, 0)
    p = cause_repertoire(tpm, state, mechanism=[0], purview=[0])
    q = unconstrained_repertoire(purview_size=1)
    # COPY with state 1 should be highly informative: p=[0,1], q=[0.5,0.5] -> ID=1
    assert intrinsic_difference(p, q) == pytest.approx(1.0, abs=1e-6)


def test_disconnected_mechanism_is_uninformative():
    from iit4.tpm import expand_tpm
    from iit4.repertoire import cause_repertoire, unconstrained_repertoire
    from iit4.id import intrinsic_difference
    n = 2
    # no connections at all -> mechanism tells nothing about purview
    conn = [[0, 0], [0, 0]]
    gates = ["COPY", "COPY"]
    tpm = expand_tpm(conn, gates)
    state = (1, 0)
    # mechanism [0] about purview [1] with no edge -> should be uniform
    p = cause_repertoire(tpm, state, mechanism=[0], purview=[1])
    q = unconstrained_repertoire(purview_size=1)
    assert intrinsic_difference(p, q) == pytest.approx(0.0, abs=1e-6)


def test_state_dependence():
    """Same wiring, different state -> different repertoires -> different phi."""
    from iit4.tpm import expand_tpm
    from iit4.repertoire import cause_repertoire
    n = 2
    conn = [[1, 1], [1, 1]]
    gates = ["AND", "OR"]
    tpm = expand_tpm(conn, gates)
    p1 = cause_repertoire(tpm, (0, 0), mechanism=[0, 1], purview=[0, 1])
    p2 = cause_repertoire(tpm, (1, 1), mechanism=[0, 1], purview=[0, 1])
    assert p1 != pytest.approx(p2)
