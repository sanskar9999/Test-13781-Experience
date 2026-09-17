# Test-13781-Experience — Touch the Photo

In-browser tactile vision explorer. Tap any part of a photo and the patches
the vision transformer thinks are *the same thing* glow warm orange.

**Live demo:** https://sanskar9999.github.io/Test-13781-Experience/
**Entry point:** `index.html` (served via GitHub Pages from `main`).

No build step. No backend. All inference runs on-device in the browser.

---

## 1. What it feels like

UI is deliberately minimal — one rounded photo, one status strip, two pill docks:

```
        ┌──────────────┐
        │              │
        │    photo     │  <- tap here (canvas#photo)
        │              │
        └──────────────┘
         ● ● ● ● ...    <- canvas#blocks, 12 status dots
      [touch][dream][gaze]   <- .modes
      [<][upload][>]         <- .dock
```

| Gesture | Effect |
|---|---|
| Single tap / click on photo | Switches to `touch` mode, computes cosine similarity of tapped patch vs every patch, glows matches |
| `touch` button | Back to similarity-glow mode |
| `dream` button (orb) | Shows top-3 PCA of patch features as RGB "what the model sees" |
| `gaze` button (eye) | Shows CLS (whole-image) token vs every patch, gently pulsing — "what the model looks at" |
| Two-finger vertical drag on photo | Sensitivity (`cut` 0–0.6, default 0.25). Drag up = stricter, down = looser |
| Mouse wheel over photo | Same sensitivity control on desktop (`cut += deltaY * 0.0006`) |
| `<` / `>` | Cycle built-in samples: `images/dog.jpg` → `images/cats.jpg` → `images/chonk.jpg` |
| Upload (↑) | Pick your own image (`<input type=file accept=image/*>`) |
| `?debug` URL param | e.g. `.../index.html?debug` — shows `#dbg` overlay + console logs (`boot`, `brain try`, `infer ok`, `tap cell=`, `cut=`) |

Orange glow = `rgba(226,160,107,α)`. Sensitivity mapping lives in
`touch.js: simToAlpha()` — `(sim - cut) / 0.55`, pow 1.5 falloff.

## 2. How it works (pipeline)

1. **Crop.** Any image (sample or upload) is center-cropped to a square and
   drawn to an offscreen 448×448 canvas (`index.html: setPhoto()`).
2. **Brain load (once).** `@huggingface/transformers@4.3.0` (ES module from
   jsDelivr) loads
   `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`.
   Tries in order: `q4 + webgpu` → `q4 + wasm` → `q8 + wasm`
   (`ensureBrain()`). Progress feeds the 12 status dots.
3. **Inference (per photo).** `RawImage.read(squareCanvas)` → processor →
   `pixel_values` → `model({pixel_values})` → `last_hidden_state[0]`
   shaped `[tokens][dim]`.
4. **Token slicing (`touch.js: slicePatchTokens()`).** Layout is
   `[CLS, ...registers, ...patches]`. Registers default to 4
   (`model.config.num_register_tokens`). Remaining tokens must form a square
   grid — for 448px / patch-16 that's 28×28 = 784 patches.
5. **Per mode:**
   - *touch:* L2-normalize patches (`normalizeRows`), cosine-similarity of
     tapped cell vs all (`similarityMap`). Result smoothed per-frame
     (`shown += (target - shown) * (1 - e^-6dt)`), upscaled from 28×28 to
     full size with smoothing for the soft glow.
   - *gaze:* raw CLS vector (`sliceCls`) dotted against normalized patches,
     same smoothing, plus a sine pulse in `paintGlow()`.
   - *dream:* top-3 PCA of **raw** (unnormalized) patches → per-patch RGB
     in [0,1] (`pca3`: center → Gram matrix → 60-iter power iteration ×3
     with deflation → min-max per component). Computed once per photo,
     faded in like the others.
6. **Render loop.** `requestAnimationFrame(frame)` always runs: base photo +
   glow layer (clipped to 28px rounded rect) + expanding tap ripples +
   breathing veil while loading / steady dim if model failed (`brainDead`).

If the model fails all three tries, the photo stays usable but dimmed and
taps only make grey ripples + log `tap ignored`.

## 3. Repo map — notes for future self

```
index.html   All UI + CSS + app logic (~600 lines). Only file Pages serves
             alongside assets. Imports transformers from CDN, helpers from touch.js.
touch.js     Pure, DOM-free math helpers (~140 lines). Importable in node for tests.
             Exports: gridSize, slicePatchTokens, normalizeRows, similarityMap,
                      pointToCell, simToAlpha, sliceCls, pca3.
images/      dog.jpg, cats.jpg, chonk.jpg — bundled samples in SAMPLES[].
iit4/ tests/ LEGACY, not used by the live site. See §6.
.gitignore   __pycache__/, .pytest_cache/, *.pyc (leftover from legacy Python part).
```

Key constants in `index.html` (top of module script):

- `MODEL` — HF model id above. Change here to swap brains.
- `N_BLOCKS = 12` — status dots = ViT-S depth. Cosmetic only.
- `SAMPLES` — sample list. Paths are relative, so local `python3 -m http.server` works.
- `S = 448` in `setPhoto()` — inference resolution. Larger = finer grid but
  slower + more RAM. 448 ÷ 16 = 28 is the current grid.
- `cut = 0.25`, clamped 0–0.6 — default sensitivity.

Key mutable state: `feats` (Float32Array n×dim, normalized), `grid`, `dim`,
`target/shown` (touch sim), `clsSims/gazeShown`, `dreamRGB/dreamShown`,
`ready`, `modelLoading`, `brainDead`, `loadFrac`, `sweep`, `ripples`,
`tapped`, `mode`.

## 4. `touch.js` quick reference

| Function | Signature | Notes |
|---|---|---|
| `gridSize` | `(nPatches) → g \| -1` | Validates square grid |
| `slicePatchTokens` | `(lastHidden, numRegisters) → {patches, grid, dim}` | Throws if not square |
| `normalizeRows` | `(patches, n, dim) → Float32Array` | Per-patch L2 norm, `‖v‖‖=0 → ×1` guard |
| `similarityMap` | `(normed, n, dim, index) → Float32Array[n]` | Cosine sim, self = ~1.0 |
| `pointToCell` | `(x, y, size, grid) → cell \| -1` | Clamps edge pixels to last cell |
| `simToAlpha` | `(sim, cut=0.25) → 0..1` | `((sim-cut)/0.55)^1.5` |
| `sliceCls` | `(lastHidden) → Float32Array` | Row 0 |
| `pca3` | `(patches, n, dim) → Float32Array[n*3]` | Random-init power iteration — colors jitter slightly run to run; mid-grey (0.5) fallback if λ≈0 |

## 5. Run / debug locally

```bash
git clone https://github.com/sanskar9999/Test-13781-Experience.git
python3 -m http.server 8000
# open http://127.0.0.1:8000/index.html
# debug: http://127.0.0.1:8000/index.html?debug
```

Requirements: modern Chromium/Edge/Safari/Firefox, internet access on first
load (jsDelivr CDN + HuggingFace model download, ~tens of MB cached by the
browser afterwards). WebGPU gives the fast path; wasm fallback works but
inference takes noticeably longer. Must serve over `http://` (not `file://`)
because ES modules + CDN enforce CORS.

Status dots cheat sheet (`drawStrip()`): filling left→right = model bytes
loading; single sweep = one inference; gentle sine breathing = idle/ready;
flat grey = `brainDead` (all backends failed — open `?debug`).

## 6. History warning — why the old README lied

Commit `7e5c931` added an **IIT 4.0 Phi Simulator** (Python `iit4/` package +
separate dashboard `index.html`, 31 pytest tests). Commits after that
(`8079d8f` onward) replaced the web app with this DINOv3 touch explorer, but
the README was never updated and `iit4/` + `tests/` were left in the tree.

So: `iit4/*.py` + `tests/test_*.py` are **dead code relative to the live
site**. They still pass (`python3 -m pytest tests -v`) but nothing in
`index.html` imports them. Keep or delete on next cleanup — deleting them
does not affect Pages. This README now documents what is actually deployed.

Git log recap: `7e5c931` IIT sim → `821df23` neuron garden →
`8079d8f` touch-only DINOv3 explorer → inference/debug fixes →
`447dfa0` pure similarity map → `ce470d0` dream+gaze+sensitivity (current).

## 7. Maintenance ideas (future self)

- Add `tests/touch.test.js` (node) for the pure helpers — file header already
  promises "testable in node", but no JS tests exist yet.
- Pin `transformers` version deliberately; `@4.3.0` API (`AutoModel`,
  `AutoImageProcessor`, `RawImage.read(canvas)`) breaks across majors.
- `pca3` is O(n²·dim + iters·n²) on 784 patches — fine now, but don't raise
  `S` without moving it off the main thread.
- Uploaded images never leave the device (object URL, revoked on next
  sample switch) — worth stating in any public description.
- Title tag is just `○`; set a real `<title>` + meta description for sharing.
