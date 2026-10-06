# Group-Theme Recoloring (Nguyen et al., Pacific Graphics 2017)

A from-scratch implementation of *"Group-Theme Recoloring for Multi-Image
Color Consistency"* (R. M. H. Nguyen, B. Price, S. Cohen, M. S. Brown).
The paper ships no public code; this reproduces its four-stage pipeline.

## Pipeline → paper sections

| Stage | File | Paper |
|-------|------|-------|
| 1. Individual palette extraction (adaptive-k in the ab plane) | `grouptheme/palette.py` | Sec. 3.2 |
| 2. Group-theme optimization (3-term weighted k-means via EM + medoid) | `grouptheme/grouptheme.py` | Sec. 3.3 |
| 3. External theme (hue matching, ±18°) | `grouptheme/external.py` | Sec. 3.4 |
| 4. Recoloring (inverse-distance weighting in ab) | `grouptheme/recolor.py` | Sec. 3.5 |
| Color-space conversions (sRGB/linear/Lab/hue) | `grouptheme/colorspace.py` | — |
| End-to-end wrapper | `grouptheme/pipeline.py` | — |

## Key design decisions (matching the paper)

- **Only the (a, b) chroma channels are clustered/recolored**; L (lightness) is
  passed through untouched (Sec. 3.2). This is deliberate in the paper.
- **Adaptive k** via the explained-variance ratio: k = 2..7, stop when
  `within_group_distortion / total_distortion < tau` (tau = 0.1).
- **Objective Eq. 1** has three terms: data (weighted k-means), a
  palette-reduction penalty (γ), and an unassociated-colors cost (η).
  - `MODE_KMEANS`           γ=0,    η=1e10
  - `MODE_MIN_REDUCTION`    γ=1e2,  η=1e10
  - `MODE_ALLOW_UNASSIGNED` γ=0,    η=1e3
  - `MODE_BOTH`             γ=1e2,  η=25e3
- **EM**: brute-force per-palette assignment (palettes are tiny) + weighted-mean
  update, iterated to convergence.
- **Medoid** theme colours (nearest real palette colour to each cluster mean),
  not raw means — gives more natural/vivid results.
- **External theme**: cluster brand colours by hue; a group colour adopts a
  brand colour whose hue is within ±18°; ties broken by highest saturation.
- **Recolor**: inverse-distance weighting of per-palette shifts (the paper's
  fast replacement for RBF), applied in the ab plane.

## Usage

```bash
# folder of images -> recolored folder
python cli.py path/to/inputs path/to/outputs --m 5 --mode both

# with an external brand palette and a reference subset
python cli.py inputs outputs --m 5 --external brand.png --include 0,2,3
```

```python
import grouptheme as gt
result = gt.group_theme_recolor(list_of_srgb_images, m=5, mode=gt.MODE_BOTH)
recolored = result["recolored"]      # list of (H,W,3) float arrays
theme_ab  = result["theme_ab"]       # final group theme in ab
group     = result["group"]          # assignments, palettes, etc.
```

Run `python demo.py` for a synthetic smoke test + before/after montage.

## Notes for extending to 3D Gaussian Splatting

The recolor function is defined over *colours*, not pixels, so the same
`recolor_ab(query_ab, src_ab, dst_ab)` recolors Gaussian SH0 colours directly.
To adapt:

1. Convert each Gaussian's **SH0** to colour: `rgb = 0.5 + 0.2820948 * sh0`
   (this is **linear** RGB — use `colorspace.linear_rgb_to_lab`, NOT the sRGB
   path, to reach Lab).
2. Take the ab channels, run stages 1–3 (either on Gaussian colours directly via
   `extract_palette_from_colors`, or extract palettes from rendered views).
3. Recolor the Gaussians' ab with `recolor_ab`, convert Lab→linear RGB→SH0:
   `sh0 = (rgb - 0.5) / 0.2820948`.
4. Decide a per-Gaussian **weight** to replace the pixel histogram (e.g.
   opacity × footprint), and decide whether to also shift SH1+ bands.

Because you edit a per-Gaussian property, the recolor is automatically
multi-view consistent.
