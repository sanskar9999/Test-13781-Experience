// Pure helpers for the Touch explorer. No DOM, no imports: testable in node.
export function gridSize(nPatches) {
  const g = Math.round(Math.sqrt(nPatches));
  return g * g === nPatches ? g : -1;
}

// lastHidden: Array of [tokens][dim] (batch already removed).
// Layout: [CLS, ...registers, ...patches]. Returns flat Float32Array + grid.
export function slicePatchTokens(lastHidden, numRegisters) {
  const nTokens = lastHidden.length;
  const dim = lastHidden[0].length;
  const nPatches = nTokens - 1 - numRegisters;
  const grid = gridSize(nPatches);
  if (grid < 0) throw new Error('patch tokens do not form a square grid');
  const out = new Float32Array(nPatches * dim);
  for (let i = 0; i < nPatches; i += 1) {
    const row = lastHidden[1 + numRegisters + i];
    for (let d = 0; d < dim; d += 1) out[i * dim + d] = row[d];
  }
  return { patches: out, grid, dim };
}

export function normalizeRows(patches, nPatches, dim) {
  const out = new Float32Array(patches.length);
  for (let i = 0; i < nPatches; i += 1) {
    let s = 0;
    for (let d = 0; d < dim; d += 1) {
      const v = patches[i * dim + d];
      s += v * v;
    }
    const inv = 1 / (Math.sqrt(s) || 1);
    for (let d = 0; d < dim; d += 1) out[i * dim + d] = patches[i * dim + d] * inv;
  }
  return out;
}

// Cosine similarity of normalized patch `index` against every patch.
export function similarityMap(normed, nPatches, dim, index) {
  const out = new Float32Array(nPatches);
  const off = index * dim;
  for (let i = 0; i < nPatches; i += 1) {
    let s = 0;
    const base = i * dim;
    for (let d = 0; d < dim; d += 1) s += normed[base + d] * normed[off + d];
    out[i] = s;
  }
  return out;
}

// Map a point in a square canvas to a patch cell. Returns -1 outside.
export function pointToCell(x, y, size, grid) {
  if (x < 0 || y < 0 || x >= size || y >= size) return -1;
  const cx = Math.min(grid - 1, Math.floor((x / size) * grid));
  const cy = Math.min(grid - 1, Math.floor((y / size) * grid));
  return cy * grid + cx;
}

// Similarity -> soft display alpha. Matches glow, ignores background noise.
export function simToAlpha(sim) {
  const t = (sim - 0.25) / 0.55;
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  return Math.pow(t, 1.5);
}
