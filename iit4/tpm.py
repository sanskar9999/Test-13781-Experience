"""TPM expansion: connectivity matrix + gates -> full 2^n x n TPM.

Formal object is 2^n x n state-by-node TPM as in PyPhi, but specified
via connectivity matrix + per-node logic gates expanded algorithmically.
"""
from __future__ import annotations


def _apply_gate(inputs: list[int], gate: str) -> int:
    g = gate.strip().upper()
    if not inputs:
        return 0
    if g == "COPY":
        return inputs[0]
    if g == "AND":
        return 1 if all(v == 1 for v in inputs) else 0
    if g == "OR":
        return 1 if any(v == 1 for v in inputs) else 0
    if g == "XOR":
        s = sum(inputs) % 2
        return s
    if g in ("MAJORITY", "MAJ"):
        return 1 if sum(inputs) > len(inputs) / 2 else 0
    if g == "NAND":
        return 0 if all(v == 1 for v in inputs) else 1
    if g == "NOR":
        return 0 if any(v == 1 for v in inputs) else 1
    raise ValueError(f"unknown gate {gate}")


def expand_tpm(conn: list[list[int]], gates: list[str], noise: float = 0.0) -> list[list[float]]:
    """Expand connectivity + gates to full TPM.

    Args:
        conn: n x n matrix, conn[i][j]==1 means node i receives from j
        gates: length n gate names
        noise: if >0, deterministic 0/1 becomes noise / 1-noise

    Returns:
        tpm: list of 2^n rows, each row length n with P(node=1 | state)
    """
    n = len(conn)
    assert len(gates) == n
    assert 0 <= noise < 0.5
    size = 2 ** n
    tpm: list[list[float]] = []
    for idx in range(size):
        # decode state: bit j is node j value (LSB = node0)
        state = [(idx >> j) & 1 for j in range(n)]
        row: list[float] = []
        for i in range(n):
            inputs = [state[j] for j in range(n) if conn[i][j]]
            det = _apply_gate(inputs, gates[i])
            if noise == 0:
                row.append(float(det))
            else:
                row.append(1 - noise if det == 1 else noise)
        tpm.append(row)
    return tpm
