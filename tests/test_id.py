"""Tests for Intrinsic Difference (IIT 4.0) - NOT EMD (IIT 3.0).

Contract per user clarifications:
- IIT 4.0 ID is KL-based, asymmetric, satisfies existence/intrinsicality
- Must NOT be EMD/Wasserstein
- Phi values not comparable across versions -> flag version
"""
import math
import pytest


def test_id_import():
    from iit4.id import intrinsic_difference, VERSION
    assert VERSION == "4.0-ID"


def test_id_zero_when_identical():
    from iit4.id import intrinsic_difference
    p = [0.5, 0.5]
    q = [0.5, 0.5]
    assert intrinsic_difference(p, q) == pytest.approx(0.0, abs=1e-9)


def test_id_positive_when_different():
    from iit4.id import intrinsic_difference
    p = [1.0, 0.0]
    q = [0.5, 0.5]
    # KL(p||q) = 1*log2(1/0.5)=1
    assert intrinsic_difference(p, q) == pytest.approx(1.0, abs=1e-6)
    assert intrinsic_difference(p, q) > 0


def test_id_asymmetric():
    from iit4.id import intrinsic_difference
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    assert intrinsic_difference(p, q) != pytest.approx(intrinsic_difference(q, p), rel=1e-3)


def test_id_not_emd():
    """ID must differ from EMD on a case where they diverge."""
    from iit4.id import intrinsic_difference
    # EMD between [1,0] and [0.5,0.5] with unit ground metric = 0.5
    # KL = 1.0, so they must not be equal
    p = [1.0, 0.0]
    q = [0.5, 0.5]
    id_val = intrinsic_difference(p, q)
    emd_val = 0.5  # known EMD for this case
    assert abs(id_val - emd_val) > 0.4


def test_id_satisfies_existence():
    """If p != q, ID >0 ; if p==q, ID==0"""
    from iit4.id import intrinsic_difference
    assert intrinsic_difference([0.25, 0.75], [0.25, 0.75]) == pytest.approx(0.0)
    assert intrinsic_difference([0.25, 0.75], [0.5, 0.5]) > 0


def test_id_handles_zero_in_q():
    from iit4.id import intrinsic_difference
    # p has mass where q=0 -> infinite informativeness, should cap or return large finite
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    val = intrinsic_difference(p, q)
    assert val > 5 or math.isinf(val)
