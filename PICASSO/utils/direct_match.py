"""
direct_match.py

Direct scene-palette -> style-palette matching, skipping group-theme entirely.
Each scene palette colour becomes its OWN landmark (src = scene colour, dst =
matched style colour). No centroids, no assignments, so the centroid-vs-average
swap cannot happen. Matching is Hungarian in ab (chroma-aware) so near-neutrals
match to neutrals, not flung by hue noise.

Returns the SAME landmark tuple shape as _theme_landmarks (plus a mapping dict)
so the downstream IDW + margin-pull + plots work unchanged:
    src_all, dst_all, srcL_all, dstL_all, theme_of_landmark, final_theme_ab, mapping
where final_theme_ab == dst_all (each scene colour's matched target), and
mapping is {scene_idx: style_idx} -- the same shape as
grouptheme.external.apply_external_theme's return, so
utils.palette_mapping_viz.save_palette_mapping works for this path too.
"""
import numpy as np


def direct_scene_to_style_landmarks(scene_ab, scene_L, style_ab, style_L,
                                    recolor_space="ab", manual_theme_colors=None,
                                    srgb_to_lab=None, match="unique", debug=False):
    """Match scene palette colours to style palette colours directly.

    scene_ab (K,2), scene_L (K,) : scene palette in Lab.
    style_ab (S,2), style_L (S,) : style palette in Lab.
    match : "unique" (Hungarian one-to-one in ab) or "nearest".
    Returns src_all, dst_all, srcL_all, dstL_all, theme_of_landmark, final_theme_ab, mapping.
    """
    scene_ab = np.asarray(scene_ab, float); scene_L = np.asarray(scene_L, float)
    style_ab = np.asarray(style_ab, float); style_L = np.asarray(style_L, float)
    K = len(scene_ab)

    cost = np.linalg.norm(scene_ab[:, None, :] - style_ab[None, :, :], axis=2)
    matched_ab = scene_ab.copy()
    matched_L = scene_L.copy()
    mapping = {}

    if match == "unique":
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        assigned = set()
        for r, c in zip(rows, cols):
            matched_ab[r] = style_ab[c]; matched_L[r] = style_L[c]; assigned.add(r)
            mapping[int(r)] = int(c)
        # leftovers (scene has more colours than style) -> nearest style colour
        for r in range(K):
            if r not in assigned:
                c = int(np.argmin(cost[r]))
                matched_ab[r] = style_ab[c]; matched_L[r] = style_L[c]
                mapping[int(r)] = c
    else:  # nearest
        for r in range(K):
            c = int(np.argmin(cost[r]))
            matched_ab[r] = style_ab[c]; matched_L[r] = style_L[c]
            mapping[int(r)] = c

    # manual overrides by scene-palette index
    if manual_theme_colors:
        for t, rgb in manual_theme_colors.items():
            t = int(t)
            if not (0 <= t < K):
                if debug:
                    print(f">>> manual idx {t} out of range [0,{K}); skipping")
                continue
            rgb = np.asarray(rgb, float)
            if rgb.max() > 1: rgb /= 255.0
            lab = srgb_to_lab(rgb[None, :])[0]
            matched_ab[t] = lab[1:]; matched_L[t] = lab[0]
            if debug:
                print(f">>> manual override: scene colour {t} -> "
                      f"rgb {tuple((rgb*255).astype(int))} ab={lab[1:].round(1)}")

    src_all = scene_ab
    dst_all = matched_ab
    srcL_all = scene_L
    dstL_all = matched_L if recolor_space == "lab" else scene_L
    theme_of_landmark = np.arange(K)
    final_theme_ab = matched_ab            # margin-pull assigns against these
    return src_all, dst_all, srcL_all, dstL_all, theme_of_landmark, final_theme_ab, mapping