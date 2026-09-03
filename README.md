# Test-13781-Experience — IIT 4.0 Phi Simulator

Browser simulator for a conscious system (IIT 4.0). Design a network of 5 binary units, measure integrated information Phi (A), and visualize the maximally irreducible cause-effect structure.

**Live demo:** Open `index.html` or enable GitHub Pages on `main` branch.

## What this implements

- **Intrinsic Difference (IIT 4.0, KL-based)** not EMD (IIT 3.0/PyPhi 1.2.0) — flagged via `iit4/id.py:12` VERSION=4.0-ID
- **TPM** `2^n x n` expanded from connectivity matrix + gates (COPY/AND/OR/XOR/MAJORITY) `iit4/tpm.py:23`
- **Cause/effect repertoires** with do-operator + uniform perturbation `iit4/repertoire.py:13`
- **Phi (big Phi A)** exhaustive MIP over directional bipartitions (15 cuts for n=5), state-dependent, per-state not average `iit4/phi.py:144`
- **Small phi** mechanism-level, **MICS** conceptual structure, **exclusion** over 31 subsets `iit4/phi.py:273`
- **n=5 demo tier** — exhaustive tractable (<5s), beyond needs Queyranne/cut_one approximations

## Run locally

```bash
python3 -m pytest tests -v
python3 -m http.server 8000
# open http://127.0.0.1:8000/index.html
```

## Browser simulator

`index.html` is self-contained (no deps). Presets: Heterogeneous Integrated (high Phi 0.41) vs Clique/Lattice/Disconnected (low/0). Tap nodes to flip state, toggle connectivity, change gates, move noise slider. Graph, Phi bar, trajectory (why averaging is wrong), concepts, qualia projection, and complexes update live.

Designed for n=5 exhaustive search — showcases differentiated+integrated topology beats homogeneous clique/lattice, and same wiring different state → different Phi/MIP.

## Verification

- 31 tests passed (TPM, repertoires, ID asymmetric ≠ EMD, Phi state-dependence, exclusion)
- `py_compile` OK, `oxlint` 1.81.0
