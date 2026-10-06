"""Wire the Group-Theme Recoloring pipeline (../../grouptheme) onto 3DGS Gaussians.

Two front-ends produce the (src -> dst) landmark set that drives the IDW ab-shift:
  * GROUP-THEME (default): scene palette -> group-theme optimisation (m centroids
    + assignments) -> match centroids to the style palette -> landmarks.
  * DIRECT (direct_match=True): scene palette -> match EACH scene colour directly
    to the style palette (Hungarian in ab) -> each scene colour IS a landmark.
    No centroids, no assignments, so the centroid-vs-assignment swap cannot occur.

Both feed the SAME downstream: SH0 -> Lab(ab), IDW shift (recolor_ab_L), adaptive
margin-pull, back to SH0. L (lightness) is preserved: only hue/chroma move.
"""
import os
import sys
import json
import numpy as np
import torch
from utils.sh_utils import SH2RGB
from utils.grouptheme_recolor_viz import save_ab_recolor, save_ab_before_after
# from utils.diagnose_green_shift import diagnose
#from utils.grouptheme_recolor_viz import _srgb_of_ab
from utils.margin_pull import margin_pull_adaptive, assign_nearest_theme
from utils.direct_match import direct_scene_to_style_landmarks
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "grouptheme"))
import grouptheme as gt
from grouptheme.recolor import recolor_ab_L
from utils.palette_mapping_viz import save_palette_mapping
from utils.margin_pull_step import apply_margin_pull

GT_MODES = {
    "kmeans": gt.MODE_KMEANS,
    "min_reduction": gt.MODE_MIN_REDUCTION,
    "allow_unassigned": gt.MODE_ALLOW_UNASSIGNED,
    "both": gt.MODE_BOTH,
}


def force_keep_themes_in_landmarks(src_all, dst_all, srcL_all, dstL_all,
                                   theme_of_landmark, final_theme_ab, theme_L_target,
                                   group_theme_ab, keep_theme_idxs):
    present = set(int(t) for t in theme_of_landmark)
    add_src, add_dst, add_sL, add_dL, add_tol = [], [], [], [], []
    for t in keep_theme_idxs:
        t = int(t)
        if t in present:
            continue
        add_src.append(group_theme_ab[t])
        add_dst.append(final_theme_ab[t])
        add_sL.append(theme_L_target[t])
        add_dL.append(theme_L_target[t])
        add_tol.append(t)
    if add_src:
        src_all = np.vstack([src_all, np.array(add_src)])
        dst_all = np.vstack([dst_all, np.array(add_dst)])
        srcL_all = np.concatenate([srcL_all, np.array(add_sL)])
        dstL_all = np.concatenate([dstL_all, np.array(add_dL)])
        theme_of_landmark = np.concatenate([theme_of_landmark, np.array(add_tol)])
    return src_all, dst_all, srcL_all, dstL_all, theme_of_landmark


def _recolor_ab_batched(query_ab, src_ab, dst_ab, power, batch_size=200_000, n_jobs=1):
    if len(query_ab) <= batch_size:
        return gt.recolor_ab(query_ab, src_ab, dst_ab, power=power)
    bounds = [(start, min(start + batch_size, len(query_ab)))
              for start in range(0, len(query_ab), batch_size)]
    if n_jobs == 1:
        out = np.empty_like(query_ab)
        for start, end in bounds:
            out[start:end] = gt.recolor_ab(query_ab[start:end], src_ab, dst_ab, power=power)
        return out
    from joblib import Parallel, delayed
    chunks = Parallel(n_jobs=n_jobs)(
        delayed(gt.recolor_ab)(query_ab[start:end], src_ab, dst_ab, power=power)
        for start, end in bounds)
    return np.concatenate(chunks)


def _recolor_one_image(img, src_ab, dst_ab, src_L, dst_L, power, out_shape, l_strength):
    lab = gt.colorspace.srgb_to_lab(img.reshape(-1, 3))
    Lc, ab = lab[:, 0], lab[:, 1:]
    new_ab, dL = recolor_ab_L(ab, src_ab, dst_ab, src_L, dst_L,
                              power=power, l_strength=l_strength)
    new_L = Lc + dL
    out = gt.colorspace.lab_to_srgb(np.concatenate([new_L[:, None], new_ab], axis=1))
    return np.clip(out, 0, 1).reshape(out_shape)


def _theme_landmarks(palettes, assignments, indices, theme_ab, theme_L):
    m = len(theme_ab)
    sum_src = np.zeros((m, 2)); sum_srcL = np.zeros(m); sum_w = np.zeros(m)
    unassigned_ab = []; unassigned_L = []
    for i in indices:
        labels = assignments[i]; p = palettes[i]
        for j, t in enumerate(labels):
            if t > 0:
                sum_src[t - 1] += p.weights[j] * p.ab[j]
                sum_srcL[t - 1] += p.weights[j] * p.L[j]
                sum_w[t - 1] += p.weights[j]
            else:
                unassigned_ab.append(p.ab[j]); unassigned_L.append(p.L[j])
    used = sum_w > 0
    # Build one landmark per theme IN THEME ORDER (index t -> row t). For used
    # themes, src = assigned-average; for empty themes, src = centroid (theme_ab)
    # so the row still exists and index t is stable. theme_of_landmark == arange(m).
    src = np.where(used[:, None], np.divide(sum_src, np.maximum(sum_w, 1e-12)[:, None]),
                  theme_ab)
    srcL = np.where(used, sum_srcL / np.maximum(sum_w, 1e-12), theme_L)
    dst = theme_ab.copy()
    dstL = theme_L.copy()
    theme_of_landmark = np.arange(m)
    if unassigned_ab:
        ua = np.array(unassigned_ab); uL = np.array(unassigned_L)
        src = np.vstack([src, ua]); dst = np.vstack([dst, ua])
        srcL = np.concatenate([srcL, uL]); dstL = np.concatenate([dstL, uL])
        theme_of_landmark = np.concatenate([theme_of_landmark, -np.ones(len(ua), int)])
    return src, dst, srcL, dstL, theme_of_landmark


def _extract_palette_from_gaussians(gaussians, max_points=50_000, seed=0,
                                    weight_by="opacity"):
    """Build ONE palette from the Gaussians' SH0 base colours (treated as pixels).

    SH0's RGB is NOT physically linear light: training images are loaded as
    raw sRGB pixel values with no gamma decode (scene/cameras.py just clamps
    to [0,1]), and the reconstruction/style losses compare render() output
    directly against those pixels -- so SH0 is trained to approximate sRGB
    pixel values directly. This goes straight SH0 -> sRGB -> Lab(ab), the
    same space the image path uses (gt.colorspace.srgb_to_lab already does
    the sRGB gamma *decode* internally). Weighted by opacity*size (footprint
    proxy) or uniform.
    """
    C0 = 0.28209479177387814
    with torch.no_grad():
        sh0 = gaussians.get_features_dc.squeeze(1).detach().cpu().numpy().astype(np.float64)
        if weight_by == "opacity":
            opac = torch.sigmoid(gaussians.get_opacity.squeeze(-1)).detach().cpu().numpy()
            # mean exp(scale) as a rough footprint; fall back to 1 if unavailable
            try:
                scale = torch.exp(gaussians.get_scaling).mean(dim=1).detach().cpu().numpy()
            except Exception:
                scale = np.ones(len(sh0))
            w = (opac * scale).astype(np.float64)
        else:
            w = np.ones(len(sh0))

    srgb = np.clip(0.5 + C0 * sh0, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    if len(srgb) > max_points:
        idx = rng.choice(len(srgb), size=max_points, replace=False)
        srgb = srgb[idx]; w = w[idx]

    lab = gt.colorspace.srgb_to_lab(srgb)
    return gt.extract_palette_from_colors(lab[:, 1:], weights=w, L=lab[:, 0])


def _dedupe_palette(theme_ab, theme_L, atol=1e-6):
    """Collapse theme slots that were matched to the same destination color
    (e.g. --gt_match_by hue allows several scene colours to independently
    land on the same style colour -- see palette_mapping.png / mapping dict)
    into a single row. Rows are kept in first-occurrence order; L is averaged
    across whichever slots collapsed together (their ab is identical, but L
    can differ since recolor_space="ab" keeps each slot's own scene lightness).
    """
    kept_ab, kept_L = [], []
    used = np.zeros(len(theme_ab), dtype=bool)
    for i in range(len(theme_ab)):
        if used[i]:
            continue
        dup = np.where(np.all(np.isclose(theme_ab, theme_ab[i], atol=atol), axis=1))[0]
        used[dup] = True
        kept_ab.append(theme_ab[i])
        kept_L.append(theme_L[dup].mean())
    return np.array(kept_ab), np.array(kept_L)


def save_theme_palette(theme_ab, theme_L, path):
    """Save the theme/target AB palette used to recolour the scene (Group-Theme
    or direct-match) so it can be loaded later as the starting palette in
    ColorfulCurvesGS (--palette), instead of ColorfulCurvesGS re-extracting its
    own palette from the recoloured Gaussians via convex-hull simplification.

    Deduplicates theme slots that collapsed onto the same destination colour
    (see _dedupe_palette) -- the saved palette lists distinct colours the
    scene actually gets recoloured toward, not one row per scene-side theme
    slot. This is purely a save-time simplification: the recolor math itself
    (_theme_landmarks/dst_all) still uses one landmark per theme slot, since
    each keeps its own distinct *source* region even when slots share a
    destination.

    Appends a trailing grey [0, 0] entry if the palette doesn't already end in
    one -- ColorfulCurves' own `extract_AB_palette` always does this, and the
    weight/sparsity math (compute_new_luminance_less_grey_weights, the grey
    penalty in color_optimization) assumes the *last* palette row is grey.
    """
    theme_ab = np.asarray(theme_ab, dtype=float)
    theme_L = np.asarray(theme_L, dtype=float)
    theme_ab, theme_L = _dedupe_palette(theme_ab, theme_L)
    if not np.allclose(theme_ab[-1], [0.0, 0.0]):
        theme_ab = np.vstack([theme_ab, [0.0, 0.0]])
        theme_L = np.concatenate([theme_L, [float(theme_L.mean())]])

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "format": "colorfulcurvesgs-palette-v1",
            "ab": theme_ab.tolist(),
            "L": theme_L.tolist(),
            "source": "group_theme_recolor_new.recolor_scene_to_style",
        }, f, indent=2)
    print(">>> saved color-transfer palette to", path)


def recolor_scene_to_style(gaussians, gt_imgs, style_img, m=5, mode=gt.MODE_KMEANS,
                           hue_tol=18.0, power=2.0, include=None,
                           external_strength=1.0, max_pixels=50_000, seed=0,
                           n_jobs=1, l_strength=0.0, debug=False,
                           output_path=None, theme_source="images", recolor_space="ab",
                           margin_pull=True, margin_base_scale=0.5, manual_theme_colors=None,
                           direct_match=False, match_by="unique", unique_by="hue"):
    """Build the theme (or direct match), apply the edit to Gaussian SH0.

    direct_match : if True, skip group-theme and match the scene palette directly
                   to the style palette (Hungarian in ab). No centroids/assignments.
    match_by / unique_by : how each scene theme colour is matched to a style
        palette colour (grouptheme.external.apply_external_theme). Only used
        by the Group-Theme front-end (direct_match already does its own
        Hungarian-in-ab matching regardless of these).
        "unique" : one-to-one Hungarian assignment (cost = unique_by: "ab" or
                   "hue"). Every style colour used at most once until scene
                   colours outnumber style colours.
        "hue"    : per scene colour independently, nearest style colour within
                   hue_tol degrees (falls back to nearest hue if none qualify).
                   Not one-to-one -- multiple scene colours can collapse onto
                   the same style colour, and hue_tol actually applies here
                   (it's otherwise unused).
        "ab"     : per scene colour independently, nearest style colour in ab.
    Returns (new_gt_imgs, new_dc_rgb).
    """
    if isinstance(mode, str):
        mode = GT_MODES[mode]
        print(mode)
    device = gt_imgs.device
    gt_shape = tuple(gt_imgs.shape)
    V = gt_shape[0]
    import time
    def _tic(name):
        if debug:
            now = time.time()
            if _tic.last is not None:
                print(f"  [{_tic.lastname}] {now - _tic.last:.2f}s", flush=True)
            _tic.last = now
            _tic.lastname = name
            print(f"{name}...", flush=True)
    _tic.last = None
    _tic.lastname = None
    if debug:
        print(f"Source palette: {theme_source} | direct_match={direct_match}")
        print(f"Power: {power} | Margin pull: {margin_pull}")
        _tic('Step 1: palette extraction')

    # ---- 1. Per-image (or gaussian) palettes ----
    imgs_np = gt_imgs.detach().cpu().numpy().astype(np.float64)
    if theme_source == "gaussians":
        gpal = _extract_palette_from_gaussians(gaussians, max_points=max_pixels,
                                               seed=seed, weight_by="opacity")
        palettes = [gpal]
        theme_palettes = [gpal]
    else:
        if n_jobs == 1:
            palettes = [gt.extract_palette(imgs_np[v], seed=seed, max_pixels=max_pixels)
                        for v in range(V)]
        else:
            from joblib import Parallel, delayed
            palettes = Parallel(n_jobs=n_jobs)(
                delayed(gt.extract_palette)(imgs_np[v], seed=seed, max_pixels=max_pixels)
                for v in range(V))
        theme_palettes = palettes if include is None else [palettes[i] for i in include]

    # ---- style palette (needed by both paths) ----
    if debug:
        _tic('Step 3: style palette + matching')
    style_np = style_img.detach().cpu().numpy().astype(np.float64)
    style_palette = gt.extract_palette(style_np, seed=seed, max_pixels=max_pixels)
    style_srgb = style_palette.srgb()
    style_lab = gt.colorspace.srgb_to_lab(style_srgb)
    if debug:
        print(f">>> style palette has {style_palette.k} colors")
        print(f">>> style palette RGB:\n{(style_srgb*255).astype(int)}")

    # =====================================================================
    #   FRONT-END: either DIRECT match or GROUP-THEME. Both produce
    #   src_all, dst_all, srcL_all, dstL_all, theme_of_landmark, final_theme_ab,
    #   theme_L_target, eff_l_strength, and (group, mapping) which are None in
    #   direct mode -- guards below skip group-only plots/prints when group is None.
    # =====================================================================
    group = None
    mapping = None
    theme_defining = [0] if theme_source == "gaussians" else (
        list(range(V)) if include is None else list(include))

    if direct_match:
        # ----- DIRECT PATH -----
        if theme_source == "gaussians":
            scene_pal = _extract_palette_from_gaussians(
                gaussians, max_points=max_pixels, seed=seed, weight_by="opacity")
        else:
            scene_pal = gt.extract_palette(imgs_np.reshape(-1, 3), seed=seed,
                                           max_pixels=max_pixels)
        scene_ab, scene_L = scene_pal.ab, scene_pal.L
        # direct_scene_to_style_landmarks only implements two algorithms --
        # "unique" (Hungarian one-to-one) or "nearest" (its else branch,
        # unconditional per-colour nearest-in-ab, no hue-tolerance concept at
        # all) -- so match_by="hue"/"ab" both map to "nearest" here; only
        # "unique" is actually distinguished for this front-end.
        (src_all, dst_all, srcL_all, dstL_all,
         theme_of_landmark, final_theme_ab, mapping) = direct_scene_to_style_landmarks(
            scene_ab, scene_L, style_lab[:, 1:], style_lab[:, 0],
            recolor_space=recolor_space, manual_theme_colors=manual_theme_colors,
            srgb_to_lab=gt.colorspace.srgb_to_lab,
            match=match_by, debug=debug)
        theme_L_target = dstL_all.copy()
        eff_l_strength = 1.0 if recolor_space == "lab" else l_strength
        plot_theme_ab, plot_theme_L = scene_ab, scene_L    # for save_palette_mapping below
        if debug:
            print(">>> DIRECT match: scene", len(scene_ab), "-> style", len(style_lab))
            print(">>> src_all (scene):\n", np.round(src_all, 1))
            print(">>> dst_all (matched style):\n", np.round(dst_all, 1))
            print('Final theme ab:', final_theme_ab)

    else:
        # ----- GROUP-THEME PATH -----
        if debug:
            _tic('Step 2: group-theme optimization')
        group = gt.optimize_group_theme(theme_palettes, m=m, mode=mode, seed=seed)

        matched_theme_ab, mapping = gt.apply_external_theme(
            group.theme_ab, style_srgb, hue_tol=hue_tol,
            match_by=match_by, unique_by=unique_by, return_mapping=True)
        plot_theme_ab, plot_theme_L = group.theme_ab, group.theme_L    # for save_palette_mapping below

        if recolor_space == "lab":
            theme_L_target = np.array([
                style_lab[mapping[t], 0] if (mapping and t in mapping) else group.theme_L[t]
                for t in range(len(group.theme_ab))
            ])
            eff_l_strength = 1.0
        else:
            theme_L_target = group.theme_L.copy()   # copy: override must not mutate group
            eff_l_strength = l_strength
        if debug:
            print(">>> recolor_space:", recolor_space,
                  "| theme L:", np.round(group.theme_L, 0),
                  "| target L:", np.round(theme_L_target, 0))
            print("  before ab:\n", np.round(group.theme_ab, 1))
            print("  after  ab:\n", np.round(matched_theme_ab, 1))

        s = float(np.clip(external_strength, 0.0, 1.0))
        final_theme_ab = (1.0 - s) * group.theme_ab + s * matched_theme_ab
        print("group_theme_ab", group.theme_ab)
        print("matched_theme_ab", matched_theme_ab)
        print('s',s)
        print('Final_theme', final_theme_ab)
        if manual_theme_colors:
            for t, rgb in manual_theme_colors.items():
                t = int(t)
                if not (0 <= t < len(final_theme_ab)):
                    print(f">>> WARNING: manual theme index {t} out of range; skipping")
                    continue
                rgb = np.asarray(rgb, float)
                if rgb.max() > 1: rgb /= 255.0
                lab = gt.colorspace.srgb_to_lab(rgb[None, :])[0]
                final_theme_ab[t] = lab[1:]
                theme_L_target[t] = lab[0]
                if debug:
                    print(f">>> manual override: theme {t} -> rgb "
                          f"{tuple((rgb*255).astype(int))} ab={lab[1:].round(1)} L={lab[0]:.0f}")
            if debug:
                print(">>> AFTER override, final_theme_ab:\n", np.round(final_theme_ab, 1))

        # subset-defined theme: re-assign ALL images against the final theme
        if include is not None:
            from grouptheme.grouptheme import _assign_one_palette, GroupThemeResult
            gamma_v = mode["gamma"]; eta_v = mode["eta"]
            assignments = [
                _assign_one_palette(p.ab, p.weights, final_theme_ab, gamma_v, eta_v)
                for p in palettes]
            group = GroupThemeResult(theme_ab=group.theme_ab, theme_L=group.theme_L,
                                     assignments=assignments, palettes=palettes)

        targets = gt.build_targets(group, external_theme_ab=final_theme_ab)

        # landmarks from the group-theme assignments
        (src_all, dst_all, srcL_all, dstL_all,
         theme_of_landmark) = _theme_landmarks(
            palettes, group.assignments, theme_defining, final_theme_ab, theme_L_target)

        # centroid-vs-assigned-average diagnostic (group-theme only)
        if debug:
            print("\n=== theme centroid vs assigned-average diagnostic ===")
            assigned_count = np.zeros(len(final_theme_ab), int)
            for i in theme_defining:
                for t in group.assignments[i]:
                    if t > 0:
                        assigned_count[t - 1] += 1
            def _rgb_of_ab(ab, L=60.0):
                rgb = gt.colorspace.lab_to_srgb(np.array([[L, ab[0], ab[1]]]))[0]
                return tuple((np.clip(rgb, 0, 1) * 255).astype(int))
            tol = np.asarray(theme_of_landmark)
            for t in range(len(final_theme_ab)):
                centroid = group.theme_ab[t]
                rows = np.where(tol == t)[0]
                if len(rows):
                    avg = src_all[rows].mean(0); dstv = dst_all[rows].mean(0)
                    gap = np.linalg.norm(avg - centroid)
                    print(f" theme {t}: centroid={centroid.round(1)} rgb{_rgb_of_ab(centroid)} | "
                          f"assigned-avg={avg.round(1)} rgb{_rgb_of_ab(avg)} | "
                          f"target={dstv.round(1)} rgb{_rgb_of_ab(dstv)} | "
                          f"#assigned={assigned_count[t]} | gap={gap:.1f}")
                else:
                    print(f" theme {t}: centroid={centroid.round(1)} | NO landmark | "
                          f"#assigned={assigned_count[t]}")
            print(" theme_of_landmark:", tol.tolist())
            print("=== end diagnostic ===\n")

        # force-keep every theme (group-theme can drop empty themes; direct never does)
        print(">>> before force_keep: theme_of_landmark =", theme_of_landmark,
              "dst_all rows =", len(dst_all))
        keep = set(range(len(final_theme_ab)))
        if manual_theme_colors:
            keep |= {int(k) for k in manual_theme_colors.keys()}
        (src_all, dst_all, srcL_all, dstL_all,
         theme_of_landmark) = force_keep_themes_in_landmarks(
            src_all, dst_all, srcL_all, dstL_all, theme_of_landmark,
            final_theme_ab, theme_L_target, group.theme_ab, keep_theme_idxs=sorted(keep))
        print(">>> after force_keep: theme_of_landmark =", theme_of_landmark,
              "dst_all rows =", len(dst_all))

    if output_path:
        save_theme_palette(final_theme_ab, theme_L_target,
                           os.path.join(output_path, "color_transfer_palette.json"))

    if debug:
        # Nearest OTHER landmark to each landmark, measured by source (original
        # scene) ab only -- dst/target is irrelevant here. Landmarks with a
        # very close neighbor compete for the same query points in the IDW
        # shift (recolor_ab_L), which can wash out or blend their intended effect.
        print(">>> Nearest landmark (by source ab) per landmark:")
        d_land = np.linalg.norm(src_all[:, None, :] - src_all[None, :, :], axis=2)
        np.fill_diagonal(d_land, np.inf)
        nearest_land = np.argmin(d_land, axis=1)
        for i in range(len(src_all)):
            j = nearest_land[i]
            print(f"   landmark {i} (theme {theme_of_landmark[i]}, src ab={src_all[i].round(1)}) "
                  f"-> nearest: landmark {j} (theme {theme_of_landmark[j]}, src ab={src_all[j].round(1)}), "
                  f"dist={d_land[i, j]:.2f}")

    # =====================================================================
    #   4a. Recolour the training IMAGES with the SAME landmark set
    # =====================================================================
    if debug:
        _tic('Step 4a: recolor images')
    new_imgs = np.empty_like(imgs_np)
    for v in range(V):
        lab = gt.colorspace.srgb_to_lab(imgs_np[v].reshape(-1, 3))
        Lc, ab = lab[:, 0], lab[:, 1:]
        nab, dL = recolor_ab_L(ab, src_all, dst_all, srcL_all, dstL_all,
                               power=power, l_strength=eff_l_strength)
        nab = apply_margin_pull(ab, nab, final_theme_ab,
                                base_scale=margin_base_scale, enabled=margin_pull)
        out = gt.colorspace.lab_to_srgb(np.concatenate([(Lc + dL)[:, None], nab], axis=1))
        new_imgs[v] = np.clip(out, 0, 1).reshape(gt_shape[1:])
    new_gt_imgs = torch.from_numpy(new_imgs).to(dtype=gt_imgs.dtype, device=device)

    # =====================================================================
    #   4b. Recolour the Gaussian SH0
    # =====================================================================
    # SH0's RGB is trained to approximate sRGB pixel values directly (see the
    # comment on _extract_palette_from_gaussians) -- no gamma-encode step here,
    # unlike an earlier version of this function that incorrectly treated SH0
    # as physically linear light.
    with torch.no_grad():
        dc_srgb = SH2RGB(gaussians.get_features_dc.squeeze(1)).clamp(0.0, 1.0)
        dc_srgb = dc_srgb.detach().cpu().numpy().astype(np.float64)
    dc_lab = gt.colorspace.srgb_to_lab(dc_srgb)
    dc_L, dc_ab = dc_lab[:, 0:1], dc_lab[:, 1:]

    new_dc_ab, dLg = recolor_ab_L(dc_ab, src_all, dst_all, srcL_all, dstL_all,
                                  power=power, l_strength=eff_l_strength)

    # palette_mapping.png -- works for both paths: plot_theme_ab/L is the scene's
    # own palette (group-theme centroids, or the raw scene palette for direct_match)
    # and mapping is {scene_idx: style_idx} either way.
    if mapping is not None and output_path:
        try:
            print("AAAAAAA", style_srgb)
            save_palette_mapping(plot_theme_ab, plot_theme_L, style_srgb, mapping,
                                 os.path.join(output_path, "palette_mapping.png"))
            # Raw data alongside the PNG so multiple scenarios (e.g. different
            # --gt_match_by runs) can later be combined into one multi-panel
            # figure (utils.palette_mapping_viz.save_palette_mapping_multi)
            # without rerunning the whole recolor pipeline.
            map_keys = np.array(sorted(mapping.keys()), dtype=np.int64)
            map_vals = np.array([mapping[k] for k in map_keys], dtype=np.int64)
            np.savez(os.path.join(output_path, "palette_mapping_data.npz"),
                    theme_ab=plot_theme_ab, theme_L=plot_theme_L,
                    style_srgb=style_srgb, map_keys=map_keys, map_vals=map_vals)
        except Exception as e:
            print(">>> palette mapping viz failed:", e)

    # adaptive margin-pull (Fix A: assign on post-IDW position)
    new_dc_ab = apply_margin_pull(dc_ab, new_dc_ab, final_theme_ab,
                                  base_scale=margin_base_scale,
                                  enabled=margin_pull, debug=debug, tag="gaussians")

    new_dc_L = dc_L[:, 0] + dLg
    new_dc_lab = np.concatenate([new_dc_L[:, None], new_dc_ab], axis=1)
    # No gamma-decode on the way back either -- write the sRGB-space result
    # straight back to SH0 (via RGB2SH downstream), matching dc_srgb above.
    new_dc_srgb = gt.colorspace.lab_to_srgb(new_dc_lab)
    new_dc_rgb = np.clip(new_dc_srgb, 0.0, 1.0).astype(np.float32)

    # per-source and per-target Gaussian counts (works for both paths -- uses
    # plot_theme_ab/final_theme_ab). Both are a hard nearest-neighbor
    # classification of every Gaussian's ORIGINAL color against a small set
    # of reference colors -- a coverage diagnostic, not the real (smooth IDW)
    # recolor math. Source uses each theme's own real L; target uses a fixed
    # L=60 swatch preview since final_theme_ab has no single paired L here.
    if debug:
        total = len(dc_ab)
        label = "scene colour" if direct_match else "theme"

        d_src = np.linalg.norm(dc_ab[:, None, :] - plot_theme_ab[None, :, :], axis=2)
        nearest_src = np.argmin(d_src, axis=1)
        counts_src = np.bincount(nearest_src, minlength=len(plot_theme_ab))
        print(f">>> Gaussians per SOURCE {label} cluster (original scene palette, before matching):")
        for t in range(len(plot_theme_ab)):
            rgb = gt.colorspace.lab_to_srgb(
                np.array([[plot_theme_L[t], plot_theme_ab[t, 0], plot_theme_ab[t, 1]]]))[0]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(int)
            print(f"   {label} {t}: {counts_src[t]:>8d} ({100*counts_src[t]/total:5.1f}%)  rgb{tuple(rgb)}")

        d = np.linalg.norm(dc_ab[:, None, :] - final_theme_ab[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        counts = np.bincount(nearest, minlength=len(final_theme_ab))
        print(f">>> Gaussians per TARGET {label} cluster (after matching to style):")
        for t in range(len(final_theme_ab)):
            rgb = gt.colorspace.lab_to_srgb(
                np.array([[60.0, final_theme_ab[t, 0], final_theme_ab[t, 1]]]))[0]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(int)
            print(f"   {label} {t}: {counts[t]:>8d} ({100*counts[t]/total:5.1f}%)  rgb{tuple(rgb)}")

    # ---- plots (both paths; group-only args guarded) ----
    if debug and output_path:
        save_ab_recolor(
            src_all, dst_all, dc_ab, new_dc_ab,
            os.path.join(output_path, "ab_recolor.png"),
            dc_srgb=dc_srgb, style_srgb=style_srgb, mapping=mapping)
        save_ab_before_after(
            dc_ab, new_dc_ab,
            os.path.join(output_path, "ab_before_after.png"),
            dc_srgb=dc_srgb, new_dc_srgb=new_dc_srgb,
            theme_ab=src_all, final_theme_ab=dst_all)

        # two_hop: group-theme draws centroid->avg->style (2 hops); direct has no
        # centroids -> pass src_all as the centroid arg so the first hop is
        # zero-length, rendering a clean 1-hop (triangle->diamond) plot.
        # from utils.two_hop_viz import save_two_hop
        # centroids = group.theme_ab if group is not None else src_all
        # save_two_hop(centroids, src_all, dst_all, theme_of_landmark,
        #              os.path.join(output_path, "two_hop.png"),
        #              dc_ab=dc_ab, new_dc_ab=new_dc_ab, dc_srgb=dc_srgb)

        # diagnose(src_all, dst_all, srcL_all, dstL_all, recolor_ab_L,
        #          probe_center=(-30.0, 35.0), power=power, l_strength=eff_l_strength,
        #          srgb_of_ab=_srgb_of_ab,
        #          out_png=os.path.join(output_path, "green_probe.png"))

        from utils.ab_flow_field import save_ab_flow_compare
        save_ab_flow_compare(
            src_all, dst_all, srcL_all, dstL_all,
            recolor_ab_L, gt.colorspace.lab_to_srgb,
            os.path.join(output_path, "ab_flow.png"),
            final_theme_ab=final_theme_ab,
            assign_nearest_theme=assign_nearest_theme,
            margin_pull_adaptive=margin_pull_adaptive,
            power=power, l_strength=eff_l_strength,
            dc_ab=dc_ab, margin_base_scale=margin_base_scale, n=28)
        print(">>> saved ab_flow.png")
    if debug and output_path:
        from utils.ab_3d_viz import save_lab_3d, save_lab_3d_interactive
        save_lab_3d(
            dc_L, dc_ab, new_dc_ab, gt.colorspace.lab_to_srgb,
            os.path.join(output_path, "lab_3d.png"),
            new_dc_L=new_dc_L,                 # recolored L (from your pipeline)
            landmarks_src=src_all, landmarks_dst=dst_all,
            srcL=srcL_all, dstL=dstL_all)
        save_lab_3d_interactive(
            dc_L, dc_ab, new_dc_ab, gt.colorspace.lab_to_srgb,
            os.path.join(output_path, "lab_3d.html"),
            new_dc_L=new_dc_L,
            dst_ab=dst_all,          # the destination palette colors (targets)
            dstL=dstL_all) 
        print(">>> saved lab_3d.png / lab_3d.html")

    if debug:
        print("final_theme_ab (ab):\n", np.round(final_theme_ab, 1))
        print("theme_L_target (L):\n", np.round(theme_L_target, 1))
        final_lab = np.column_stack([theme_L_target, final_theme_ab])   # (T,3) [L,a,b]
        print("final_theme LAB [L,a,b]:\n", np.round(final_lab, 1))
        final_rgb = (np.clip(gt.colorspace.lab_to_srgb(final_lab), 0, 1) * 255).astype(int)
        print("final_theme RGB:\n", final_rgb)
        print("src_all:\n", np.round(src_all, 1))
        print("dst_all:\n", np.round(dst_all, 1))
        if group is not None:
            print("group.theme_ab:\n", np.round(group.theme_ab, 1))
        _tic('done')

    return new_gt_imgs, new_dc_rgb