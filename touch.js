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

// Similarity -> soft display alpha. `cut` sets where background melts away.
// Normalized by (1 - cut) so the peak (self-similarity = 1) always stays
// fully visible no matter how strict the cut gets.
export function simToAlpha(sim, cut) {
  const c = cut === undefined ? 0.3 : cut;
  const t = (sim - c) / Math.max(1e-6, 1 - c);
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  return Math.pow(t, 1.5);
}

// First row of the token table: the CLS whole-image vector.
export function sliceCls(lastHidden) {
  return Float32Array.from(lastHidden[0]);
}

// Top-3 PCA of patch features -> per-patch RGB in [0,1]. The dream colors.
// Power iteration on the explicit Gram matrix with deflation.
export function pca3(patches, nPatches, dim) {
  const mean = new Float64Array(dim);
  for (let i = 0; i < nPatches; i += 1) {
    for (let d = 0; d < dim; d += 1) mean[d] += patches[i * dim + d];
  }
  for (let d = 0; d < dim; d += 1) mean[d] /= nPatches;
  const centered = new Float64Array(nPatches * dim);
  for (let i = 0; i < nPatches; i += 1) {
    for (let d = 0; d < dim; d += 1) centered[i * dim + d] = patches[i * dim + d] - mean[d];
  }
  const gram = new Float64Array(nPatches * nPatches);
  for (let i = 0; i < nPatches; i += 1) {
    for (let j = i; j < nPatches; j += 1) {
      let s = 0;
      for (let d = 0; d < dim; d += 1) s += centered[i * dim + d] * centered[j * dim + d];
      gram[i * nPatches + j] = s;
      gram[j * nPatches + i] = s;
    }
  }
  const out = new Float32Array(nPatches * 3);
  const v = new Float64Array(nPatches);
  const w = new Float64Array(nPatches);
  for (let k = 0; k < 3; k += 1) {
    for (let i = 0; i < nPatches; i += 1) v[i] = Math.random() - 0.5;
    let norm = 0;
    for (let i = 0; i < nPatches; i += 1) norm += v[i] * v[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < nPatches; i += 1) v[i] /= norm;
    for (let it = 0; it < 60; it += 1) {
      for (let i = 0; i < nPatches; i += 1) {
        let s = 0;
        const base = i * nPatches;
        for (let j = 0; j < nPatches; j += 1) s += gram[base + j] * v[j];
        w[i] = s;
      }
      norm = 0;
      for (let i = 0; i < nPatches; i += 1) norm += w[i] * w[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      for (let i = 0; i < nPatches; i += 1) v[i] = w[i] / norm;
    }
    let lambda = 0;
    for (let i = 0; i < nPatches; i += 1) {
      let s = 0;
      const base = i * nPatches;
      for (let j = 0; j < nPatches; j += 1) s += gram[base + j] * v[j];
      lambda += v[i] * s;
    }
    if (lambda < 1e-12) {
      for (let i = 0; i < nPatches; i += 1) out[i * 3 + k] = 0.5;
      continue;
    }
    const scale = Math.sqrt(lambda);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < nPatches; i += 1) {
      const s = v[i] * scale;
      if (s < lo) lo = s;
      if (s > hi) hi = s;
    }
    const span = hi - lo || 1;
    for (let i = 0; i < nPatches; i += 1) out[i * 3 + k] = (v[i] * scale - lo) / span;
    for (let i = 0; i < nPatches; i += 1) {
      for (let j = 0; j < nPatches; j += 1) gram[i * nPatches + j] -= lambda * v[i] * v[j];
    }
  }
  return out;
}
