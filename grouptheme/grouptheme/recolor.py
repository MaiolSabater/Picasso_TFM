"""Section 3.5 -- Image Recoloring.

Each palette colour P_j has a target colour Phat_j (its assigned group-theme
colour, possibly modified by the external theme). We propagate these per-palette
SHIFTS to every pixel/point.

The paper replaces Chang et al.'s RBF blending with the cheaper INVERSE-DISTANCE
WEIGHTING (Sec. 3.5): a query colour is shifted by a weighted blend of the
palette shifts, with weight inversely related to distance to each palette colour
(closer palette colours dominate). We operate purely on the (a, b) channels; L
is passed through unchanged.

The exact same function recolors:
  * image pixels  -> recolor_ab(pixels_ab, ...)
  * Gaussian SH0  -> recolor_ab(gaussian_ab, ...)   (identical maths)
"""

import numpy as np


def build_targets(group_result, external_theme_ab=None):
    """For each palette, return (src_ab, dst_ab) arrays of shifts.

    Colours assigned to 0 (unassociated) get dst = src (no change).
    external_theme_ab, if given, replaces the theme colours used as targets.
    """
    theme = group_result.theme_ab if external_theme_ab is None else external_theme_ab
    per_palette = []
    for i, p in enumerate(group_result.palettes):
        labels = group_result.assignments[i]
        src = p.ab.copy()
        dst = p.ab.copy()
        for j, t in enumerate(labels):
            if t > 0:
                dst[j] = theme[t - 1]
        per_palette.append((src, dst))
    return per_palette


def recolor_ab(query_ab, src_ab, dst_ab, power=2.0, eps=1e-6):
    """Inverse-distance-weighted recolor of query colours in the ab plane.

    query_ab : (N, 2) colours to recolor.
    src_ab   : (k, 2) original palette colours.
    dst_ab   : (k, 2) target palette colours.
    power    : IDW exponent; higher = more local influence.

    Returns (N, 2) recolored ab. Each query colour is shifted by an IDW blend of
    the per-palette shifts (dst - src).
    """
    query_ab = np.asarray(query_ab, dtype=np.float64)
    src_ab = np.asarray(src_ab, dtype=np.float64)
    dst_ab = np.asarray(dst_ab, dtype=np.float64)
    shifts = dst_ab - src_ab                                   # (k, 2)

    d2 = np.sum((query_ab[:, None, :] - src_ab[None, :, :]) ** 2, axis=2)  # (N,k)
    d = np.sqrt(d2) + eps
    w = 1.0 / (d ** power)                                      # (N, k)
    w /= w.sum(axis=1, keepdims=True)

    blended_shift = w @ shifts                                 # (N, 2)
    return query_ab + blended_shift


def recolor_image(srgb_image, palette, src_ab, dst_ab, power=2.0):
    """Recolor a full sRGB image (H, W, 3) using one palette's src/dst shift."""
    from . import colorspace as cs
    srgb_image = np.asarray(srgb_image, dtype=np.float64)
    if srgb_image.max() > 1.0 + 1e-6:
        srgb_image = srgb_image / 255.0
    h, w_, _ = srgb_image.shape
    lab = cs.srgb_to_lab(srgb_image.reshape(-1, 3))
    L = lab[:, 0:1]
    ab = lab[:, 1:]
    new_ab = recolor_ab(ab, src_ab, dst_ab, power=power)
    new_lab = np.concatenate([L, new_ab], axis=1)
    out = cs.lab_to_srgb(new_lab).reshape(h, w_, 3)
    return np.clip(out, 0.0, 1.0)

def recolor_ab_L(query_ab, src_ab, dst_ab, src_L, dst_L,
                 power=2.0, l_strength=0.0, eps=1e-6):
    """IDW recolor that also transfers lightness.

    Same inverse-distance weighting as recolor_ab for the (a,b) shift; the SAME
    weights propagate an L-shift (dst_L - src_L), scaled by l_strength
    (0 = keep original L, 1 = fully adopt target L).

    Returns (new_ab (N,2), delta_L (N,)) — add delta_L to each query's original L.
    """
    import numpy as np
    query_ab = np.asarray(query_ab, dtype=np.float64)
    src_ab = np.asarray(src_ab, dtype=np.float64)
    dst_ab = np.asarray(dst_ab, dtype=np.float64)
    src_L = np.asarray(src_L, dtype=np.float64)
    dst_L = np.asarray(dst_L, dtype=np.float64)

    ab_shifts = dst_ab - src_ab
    L_shifts = dst_L - src_L

    d2 = np.sum((query_ab[:, None, :] - src_ab[None, :, :]) ** 2, axis=2)
    d = np.sqrt(d2) + eps
    w = 1.0 / (d ** power)
    w /= w.sum(axis=1, keepdims=True)

    new_ab = query_ab + w @ ab_shifts
    delta_L = l_strength * (w @ L_shifts)
    return new_ab, delta_L