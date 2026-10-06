"""
Matching-Criterion Experiment.

Compares three scene-color -> style-color matching criteria for the
direct-match front-end (each scene palette color is its own landmark, no
Group-Theme clustering):
  - hue         : nearest hue angle (circular distance in degrees). NOTE:
                  direct_match.py itself has no real hue-tolerance algorithm
                  (match_by="hue" there silently runs as "nearest" -- see
                  direct_match.py's own docstring and the earlier fix that
                  made this explicit rather than a silent surprise). This
                  experiment needs a REAL hue matcher to have a story to
                  tell, so it implements one directly here: unconditional
                  per-color nearest-hue-angle match, no threshold, no
                  force-match fallback.
  - nearest_ab  : nearest Euclidean ab-distance, per color independently
                  (same as direct_match.py's "nearest").
  - hungarian_unique : one-to-one Hungarian assignment on ab-distance, with
                  leftover reuse if the scene palette outnumbers the style
                  palette (same as direct_match.py's "unique").

Claim being tested: hue-angle matching is unstable specifically for
near-neutral (low-chroma) colors, because hue = atan2(b,a) is nearly
undefined near the origin -- small ab noise causes large hue swings, so a
grey scene color can get matched to an unrelated saturated style color (a
"swap"). Distance-based and Hungarian matching don't have this failure mode
since they operate directly on ab position, not angle.

Efficiency: palette extraction (scene + style) is identical across the three
criteria -- extracted ONCE per scene, then only the matching + IDW + margin-
pull differ per criterion.

Usage (single scene):
    python experiments/matching_criterion_experiment.py \\
        --neutral_scenes truck:output/ckpt_gs/tandt/truck/point_cloud/iteration_30000/point_cloud.ply \\
        --saturated_scenes flower:output/ckpt_gs/llff/flower/point_cloud/iteration_30000/point_cloud.ply \\
        --style /data/.../styles/14.jpg \\
        --output_dir <output_dir> \\
        [--qualitative --qual_scene truck --qual_source_path <data_dir> --qual_model_path <dummy_dir>]
"""
import os
# Must be set before numpy/scipy/sklearn touches OpenBLAS -- see
# margin_pull_sweep.py's identical guard for why (this cluster's high-core
# machines otherwise crash with "tried to allocate too many memory regions").
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)
_STYLIZEDGS_DIR = os.path.dirname(_EXPERIMENTS_DIR)
if _STYLIZEDGS_DIR not in sys.path:
    sys.path.insert(0, _STYLIZEDGS_DIR)
_GROUPTHEME_DIR = os.path.join(_STYLIZEDGS_DIR, "..", "grouptheme")
if _GROUPTHEME_DIR not in sys.path:
    sys.path.insert(0, _GROUPTHEME_DIR)

import grouptheme as gt
from grouptheme.recolor import recolor_ab_L
from scene.gaussian_model import GaussianModel
from utils.group_theme_recolor_new import _extract_palette_from_gaussians
from utils.margin_pull_step import apply_margin_pull

from margin_pull_sweep import gaussian_dc_lab, load_style_srgb
from margin_pull_sweep_multi import parse_labeled_paths, glob_ply_paths, glob_labeled_paths

CRITERIA = ["hue", "nearest_ab", "hungarian_unique"]


# =====================================================================
#   The three matchers
# =====================================================================

def match_by_hue(scene_ab, style_ab):
    """Per scene color independently: nearest hue angle (circular), no
    threshold. This is the failure mode under test."""
    scene_hue = gt.colorspace.ab_hue_deg(scene_ab)
    style_hue = gt.colorspace.ab_hue_deg(style_ab)
    d = gt.colorspace.hue_diff_deg(scene_hue[:, None], style_hue[None, :])
    return np.argmin(d, axis=1)


def match_by_nearest_ab(scene_ab, style_ab):
    """Per scene color independently: nearest Euclidean ab-distance."""
    d = np.linalg.norm(scene_ab[:, None, :] - style_ab[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def match_by_hungarian(scene_ab, style_ab):
    """One-to-one Hungarian assignment on ab-distance; leftover scene colors
    (if the scene palette outnumbers the style palette) reuse their nearest
    style color, same convention as direct_match.py's "unique" mode."""
    from scipy.optimize import linear_sum_assignment
    d = np.linalg.norm(scene_ab[:, None, :] - style_ab[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(d)
    assign = np.empty(len(scene_ab), dtype=int)
    assigned_rows = set()
    for r, c in zip(rows, cols):
        assign[r] = c
        assigned_rows.add(r)
    for r in range(len(scene_ab)):
        if r not in assigned_rows:
            assign[r] = np.argmin(d[r])
    return assign


MATCHERS = {"hue": match_by_hue, "nearest_ab": match_by_nearest_ab, "hungarian_unique": match_by_hungarian}


# =====================================================================
#   Metrics
# =====================================================================

def compute_mismatch(scene_ab, style_ab, assign):
    """mismatch_k = ||s_k - d_k|| - ||s_k - nearest_k||, >= 0. nearest_k is
    the TRUE ab-nearest style color, independent of which criterion produced
    the assignment -- the reference "sensible" target for every criterion."""
    d_assigned = np.linalg.norm(scene_ab - style_ab[assign], axis=1)
    d_all = np.linalg.norm(scene_ab[:, None, :] - style_ab[None, :, :], axis=2)
    d_nearest = d_all.min(axis=1)
    return d_assigned - d_nearest


def summarize_mismatch(scene_ab, mismatch, weights, c_thresh=10.0, swap_thresh=20.0):
    chroma = np.linalg.norm(scene_ab, axis=1)
    neutral_mask = chroma < c_thresh

    def wmean(mask):
        w = weights[mask]
        return float(np.sum(w * mismatch[mask]) / w.sum()) if w.sum() > 0 else float("nan")

    return dict(
        mean_mismatch=wmean(np.ones(len(mismatch), dtype=bool)),
        mismatch_neutral=wmean(neutral_mask),
        mismatch_chromatic=wmean(~neutral_mask),
        max_mismatch=float(mismatch.max()),
        n_swaps=int(np.sum(mismatch > swap_thresh)),
        n_neutral_colors=int(neutral_mask.sum()),
        n_chromatic_colors=int((~neutral_mask).sum()),
    )


def compute_fidelity(dc_ab, scene_ab, style_ab, assign, power, margin_base_scale, l_strength=0.0):
    """Full recolor (IDW + margin-pull) under this criterion's assignment,
    then mean distance of every Gaussian's final chroma to its assigned
    target -- same fidelity definition as the margin-pull sweep."""
    src_all = scene_ab
    dst_all = style_ab[assign]                    # may contain duplicate rows (a collapse) -- harmless here
    zeros = np.zeros(len(scene_ab))
    new_dc_ab_preMP, _ = recolor_ab_L(dc_ab, src_all, dst_all, zeros, zeros, power=power, l_strength=l_strength)
    new_ab, info = apply_margin_pull(dc_ab, new_dc_ab_preMP, dst_all, base_scale=margin_base_scale,
                                     enabled=True, return_info=True)
    assigned_target = dst_all[info["assign"]]
    return float(np.mean(np.linalg.norm(new_ab - assigned_target, axis=1))), new_ab


# =====================================================================
#   Per-scene driver
# =====================================================================

def load_scene_palette(scene_label, scene_ply, seed=0, c_thresh=10.0, max_pixels=50_000):
    """Everything about a scene that's IDENTICAL across every style it's run
    against: load the ply once, extract its palette once. Reused across
    however many styles run against this scene (matters a lot in --style
    folder mode, where one scene can face 100+ styles)."""
    print(f"\n=== scene: {scene_label} ({scene_ply}) ===")
    gaussians = GaussianModel(3)
    gaussians.load_ply(scene_ply)
    dc_L, dc_ab = gaussian_dc_lab(gaussians)
    scene_pal = _extract_palette_from_gaussians(gaussians, max_points=max_pixels, seed=seed, weight_by="opacity")
    scene_ab, scene_weights = scene_pal.ab, scene_pal.weights

    chroma = np.linalg.norm(scene_ab, axis=1)
    print(f"    scene palette: {len(scene_ab)} colors, chroma range [{chroma.min():.1f}, {chroma.max():.1f}], "
         f"{int((chroma < c_thresh).sum())} near-neutral (<{c_thresh})")

    return dict(gaussians=gaussians, dc_ab=dc_ab, scene_ab=scene_ab, scene_weights=scene_weights)


def run_scene_style_pair(scene_state, style_srgb, criteria, palette_num=5, power=2.0,
                         margin_base_scale=0.1, seed=0, c_thresh=10.0, swap_thresh=20.0, max_pixels=50_000):
    """One (scene, style) pair, given an already-loaded scene_state (see
    load_scene_palette) -- only the style side is computed fresh here."""
    scene_ab, scene_weights, dc_ab = scene_state["scene_ab"], scene_state["scene_weights"], scene_state["dc_ab"]

    style_palette = gt.extract_palette(style_srgb, seed=seed, max_pixels=max_pixels,
                                       k_min=palette_num, k_max=palette_num)
    style_ab = gt.colorspace.srgb_to_lab(style_palette.srgb())[:, 1:]

    rows = []
    per_criterion = {}
    for crit_name in criteria:
        assign = MATCHERS[crit_name](scene_ab, style_ab)
        mismatch = compute_mismatch(scene_ab, style_ab, assign)
        summary = summarize_mismatch(scene_ab, mismatch, scene_weights, c_thresh, swap_thresh)
        fidelity, new_ab = compute_fidelity(dc_ab, scene_ab, style_ab, assign, power, margin_base_scale)
        n_distinct = len(set(assign.tolist()))

        row = dict(criterion=crit_name, n_scene_colors=len(scene_ab), n_style_colors=len(style_ab),
                  n_targets_used=n_distinct, collapsed=(n_distinct < len(scene_ab)),
                  fidelity=fidelity, **summary)
        rows.append(row)
        per_criterion[crit_name] = dict(assign=assign, mismatch=mismatch, new_ab=new_ab)

    return rows, dict(style_ab=style_ab, per_criterion=per_criterion)


def run_scene(scene_label, scene_ply, style_srgb, criteria, palette_num=5, power=2.0,
              margin_base_scale=0.1, seed=0, c_thresh=10.0, swap_thresh=20.0, max_pixels=50_000):
    """Single-style convenience wrapper (kept for the original CLI path: one
    scene, one style, full per-pair CSV rows + ab_match plot)."""
    scene_state = load_scene_palette(scene_label, scene_ply, seed=seed, c_thresh=c_thresh, max_pixels=max_pixels)
    rows, pair_state = run_scene_style_pair(scene_state, style_srgb, criteria, palette_num, power,
                                            margin_base_scale, seed, c_thresh, swap_thresh, max_pixels)
    for r in rows:
        r["scene"] = scene_label
        print(f"    [{r['criterion']:>17}] mean_mismatch={r['mean_mismatch']:6.2f}  "
             f"neutral={r['mismatch_neutral']:6.2f}  chromatic={r['mismatch_chromatic']:6.2f}  "
             f"n_swaps={r['n_swaps']}/{r['n_scene_colors']}  targets_used={r['n_targets_used']}/{r['n_scene_colors']}  "
             f"fidelity={r['fidelity']:.2f}")

    state = dict(gaussians=scene_state["gaussians"], scene_ab=scene_state["scene_ab"],
                scene_weights=scene_state["scene_weights"], dc_ab=scene_state["dc_ab"],
                style_ab=pair_state["style_ab"], per_criterion=pair_state["per_criterion"])
    return rows, state


# =====================================================================
#   Outputs
# =====================================================================

def write_csv(rows, path):
    fieldnames = ["group", "scene", "criterion", "n_scene_colors", "n_style_colors", "n_targets_used", "collapsed",
                 "mean_mismatch", "mismatch_neutral", "mismatch_chromatic", "max_mismatch", "n_swaps",
                 "n_neutral_colors", "n_chromatic_colors", "fidelity"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(">>> wrote", path)


def plot_ab_matches(scene_ab, style_ab, per_criterion, criteria, out_path, c_thresh=10.0):
    """Output C (optional): one ab-plane subplot per criterion, scene colors
    (tinted by their real color) with arrows to their assigned style target."""
    scene_lab = np.concatenate([np.full((len(scene_ab), 1), 60.0), scene_ab], axis=1)
    scene_srgb = np.clip(gt.colorspace.lab_to_srgb(scene_lab), 0, 1)
    style_lab = np.concatenate([np.full((len(style_ab), 1), 60.0), style_ab], axis=1)
    style_srgb = np.clip(gt.colorspace.lab_to_srgb(style_lab), 0, 1)
    chroma = np.linalg.norm(scene_ab, axis=1)

    fig, axes = plt.subplots(1, len(criteria), figsize=(5 * len(criteria), 5), sharex=True, sharey=True)
    if len(criteria) == 1:
        axes = [axes]

    for ax, crit_name in zip(axes, criteria):
        assign = per_criterion[crit_name]["assign"]
        for k in range(len(scene_ab)):
            target = style_ab[assign[k]]
            is_neutral = chroma[k] < c_thresh
            ax.annotate("", xy=target, xytext=scene_ab[k],
                       arrowprops=dict(arrowstyle="-|>", color="red" if is_neutral else "black",
                                       alpha=0.8 if is_neutral else 0.4, linewidth=1.5 if is_neutral else 0.8))
        ax.scatter(scene_ab[:, 0], scene_ab[:, 1], c=scene_srgb, s=120, edgecolors="black", zorder=3,
                  marker="o", label="scene")
        ax.scatter(style_ab[:, 0], style_ab[:, 1], c=style_srgb, s=120, edgecolors="black", zorder=3,
                  marker="s", label="style")
        circ = plt.Circle((0, 0), c_thresh, fill=False, linestyle=":", color="grey", alpha=0.7)
        ax.add_patch(circ)
        ax.set_title(crit_name)
        ax.set_xlabel("a")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_aspect("equal")
    axes[0].set_ylabel("b")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Scene palette -> assigned style target, per matching criterion\n"
                "(red arrows = near-neutral source; dotted circle = c_thresh)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(">>> wrote", out_path)


def run_qualitative(scene_label, scene_ply, style_ab, per_criterion, criteria, cam_idx,
                    source_path, model_path, output_dir):
    """Output B: same fixed camera, recolored under each criterion, side by side."""
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams
    from scene import Scene
    from gaussian_renderer import render as gs_render
    from utils.sh_utils import RGB2SH
    import torch

    parser = ArgumentParser()
    # sentinel=True (fill_none) is only correct when merging CLI overrides on
    # top of a saved cfg_args (e.g. render.py's get_combined_args) -- with a
    # plain parse_args() call here it makes unset defaults None instead of
    # their real values (e.g. depths="" -> None), which then makes
    # dataset_readers.py's `if depths != "":` wrongly think depth supervision
    # was requested and demand a depth_params.json that doesn't exist.
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parsed_args = parser.parse_args(["-s", source_path, "-m", model_path])
    dataset = model.extract(parsed_args)
    pipe = pipeline.extract(parsed_args)

    gaussians = GaussianModel(3)
    gaussians.load_ply(scene_ply)
    dc_L, dc_ab = gaussian_dc_lab(gaussians)

    # load_iteration=None re-initializes gaussians from the COLMAP sparse
    # cloud as a side effect of just wanting the camera list -- re-load our
    # ply right after (same fix as margin_pull_sweep.py's run_qualitative).
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    gaussians.load_ply(scene_ply)
    cameras = scene.getTrainCameras()
    cam = cameras[cam_idx]
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    imgs, labels = [], []
    for crit_name in criteria:
        new_ab = per_criterion[crit_name]["new_ab"]
        new_lab = np.concatenate([dc_L[:, None], new_ab], axis=1)
        new_srgb = np.clip(gt.colorspace.lab_to_srgb(new_lab), 0.0, 1.0).astype(np.float32)
        sh0 = RGB2SH(torch.from_numpy(new_srgb).to(gaussians.get_features_dc.device))
        with torch.no_grad():
            gaussians._features_dc.data.copy_(sh0.reshape(gaussians._features_dc.shape))
            rendering = gs_render(cam, gaussians, pipe, bg)["render"].clamp(0, 1)
        imgs.append(rendering.permute(1, 2, 0).detach().cpu().numpy())
        labels.append(crit_name)

    fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
    if len(imgs) == 1:
        axes = [axes]
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Recolor comparison across matching criteria -- scene: {scene_label}")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "match_compare.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(">>> wrote", out_path)


# =====================================================================
#   Batch mode (--style is a folder): run every (scene, style) pair, keep
#   only case-level summaries (group, criterion, mismatch_neutral,
#   n_neutral_colors, n_swaps, fidelity), no individual per-pair outputs.
# =====================================================================

def aggregate_failure_frequency(cases, swap_thresh=20.0):
    """
    cases: list of dicts with group, criterion, mismatch_neutral,
    n_neutral_colors, n_swaps, fidelity (one dict per (scene, style, criterion)
    case). Groups "neutral"/"saturated" are reported separately (that contrast
    is the whole point of the experiment) plus a pooled "all" group.

    A case only counts toward "% cases w/ neutral-swap" if it actually HAD a
    near-neutral scene color to swap (n_neutral_colors > 0) -- a scene/style
    pair with zero near-neutral colors can't demonstrate the failure either
    way, so it's excluded from that denominator rather than counted as "no
    swap" (n_neutral_eligible records how many cases actually counted).
    """
    from collections import defaultdict
    by_group_crit = defaultdict(list)
    for c in cases:
        by_group_crit[(c["group"], c["criterion"])].append(c)
        by_group_crit[("all", c["criterion"])].append(c)

    rows = []
    for (group, crit), items in sorted(by_group_crit.items(), key=lambda kv: (kv[0][0] != "all", kv[0])):
        neutral_eligible = [it for it in items if it["n_neutral_colors"] > 0]
        n_neutral_swap = sum(1 for it in neutral_eligible if it["mismatch_neutral"] > swap_thresh)
        pct_neutral_swap = 100.0 * n_neutral_swap / len(neutral_eligible) if neutral_eligible else float("nan")
        n_any_swap = sum(1 for it in items if it["n_swaps"] > 0)
        pct_any_swap = 100.0 * n_any_swap / len(items) if items else float("nan")
        mean_fidelity = float(np.mean([it["fidelity"] for it in items])) if items else float("nan")
        rows.append(dict(group=group, criterion=crit, n_cases=len(items), n_neutral_eligible=len(neutral_eligible),
                         pct_neutral_swap=pct_neutral_swap, pct_any_swap=pct_any_swap, mean_fidelity=mean_fidelity))
    return rows


def write_frequency_csv(rows, path):
    fieldnames = ["group", "criterion", "n_cases", "n_neutral_eligible",
                 "pct_neutral_swap", "pct_any_swap", "mean_fidelity"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(">>> wrote", path)


def print_frequency_table(rows):
    for group in ["neutral", "saturated", "all"]:
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        print(f"\n=== failure-frequency table: group={group} ===")
        print(f"{'criterion':<20} {'% neutral-swap':>16} {'% any swap':>12} {'mean fidelity':>14}   (n_cases, n_neutral_eligible)")
        for r in group_rows:
            print(f"{r['criterion']:<20} {r['pct_neutral_swap']:>15.1f}% {r['pct_any_swap']:>11.1f}% "
                 f"{r['mean_fidelity']:>14.2f}   ({r['n_cases']}, {r['n_neutral_eligible']})")


def run_batch(neutral_scenes, saturated_scenes, style_paths, criteria, output_dir, palette_num=5,
             power=2.0, margin_base_scale=0.1, seed=0, c_thresh=10.0, swap_thresh=20.0, max_pixels=50_000):
    print(f">>> batch mode: {len(neutral_scenes) + len(saturated_scenes)} scene(s) x "
         f"{len(style_paths)} style(s) = {(len(neutral_scenes) + len(saturated_scenes)) * len(style_paths)} pairs "
         f"-- no per-case files will be saved, only the aggregated frequency table.")

    cases = []
    for group_name, scenes in [("neutral", neutral_scenes), ("saturated", saturated_scenes)]:
        for scene_label, scene_ply in scenes:
            scene_state = load_scene_palette(scene_label, scene_ply, seed=seed, c_thresh=c_thresh, max_pixels=max_pixels)
            for style_idx, (style_label, style_path) in enumerate(style_paths):
                style_srgb = load_style_srgb(style_path)
                rows, _ = run_scene_style_pair(scene_state, style_srgb, criteria, palette_num, power,
                                               margin_base_scale, seed, c_thresh, swap_thresh, max_pixels)
                for r in rows:
                    cases.append(dict(group=group_name, scene=scene_label, style=style_label, **r))
                if (style_idx + 1) % 25 == 0 or style_idx == len(style_paths) - 1:
                    print(f"    ...{scene_label}: {style_idx + 1}/{len(style_paths)} styles done")

    freq_rows = aggregate_failure_frequency(cases, swap_thresh=swap_thresh)
    write_frequency_csv(freq_rows, os.path.join(output_dir, "matching_criterion_frequency.csv"))
    print_frequency_table(freq_rows)
    return cases, freq_rows


def run_multistyle_figure(scene_label, scene_ply, style_specs, criteria, output_dir,
                          scene_img_path=None, palette_num=5, power=2.0,
                          margin_base_scale=0.1, seed=0, c_thresh=10.0, swap_thresh=20.0):
    """scene_img_path: an image of the scene for the reference column (e.g. a
       training view). style_specs: list of (label, style_path)."""
    from matching_multi_style_fig import plot_ab_matches_multistyle
    import imageio.v2 as imageio

    scene_state = load_scene_palette(scene_label, scene_ply, seed=seed, c_thresh=c_thresh)
    scene_img = imageio.imread(scene_img_path) if scene_img_path else None

    entries = []
    for style_label, style_path in style_specs:
        style_srgb = load_style_srgb(style_path)
        rows, pair_state = run_scene_style_pair(
            scene_state, style_srgb, criteria, palette_num, power,
            margin_base_scale, seed, c_thresh, swap_thresh)
        entries.append(dict(
            style_img=imageio.imread(style_path),
            style_ab=pair_state["style_ab"],
            per_criterion=pair_state["per_criterion"],
            label=style_label))

    plot_ab_matches_multistyle(
        scene_state["scene_ab"], scene_img, entries, criteria,
        os.path.join(output_dir, f"ab_match_multistyle_{scene_label}.png"),
        c_thresh=c_thresh, ax_size=4.5, font_scale=1.3)

# =====================================================================
#   CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neutral_scenes", nargs="+", default=[], help="label:path/to/point_cloud.ply")
    parser.add_argument("--neutral_scenes_dir", default=None)
    parser.add_argument("--saturated_scenes", nargs="+", default=[], help="label:path/to/point_cloud.ply")
    parser.add_argument("--saturated_scenes_dir", default=None)
    parser.add_argument("--style", required=True, help="Single style image, fixed across the whole comparison.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--criteria", nargs="+", default=CRITERIA, choices=CRITERIA)
    parser.add_argument("--palette_num", type=int, default=5)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--margin_base_scale", type=float, default=0.1)
    parser.add_argument("--c_thresh", type=float, default=10.0)
    parser.add_argument("--swap_thresh", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--qual_scene", default=None, help="Scene label to render (default: first neutral scene).")
    parser.add_argument("--qual_cam_idx", type=int, default=0)
    parser.add_argument("--qual_source_path", default=None, help="-s for the qualitative scene (required if --qualitative).")
    parser.add_argument("--qual_model_path", default=None, help="Any writable dir (only used for camera loading).")
    args = parser.parse_args()

    neutral_scenes = parse_labeled_paths(args.neutral_scenes)
    if args.neutral_scenes_dir:
        neutral_scenes += glob_ply_paths(args.neutral_scenes_dir)
    saturated_scenes = parse_labeled_paths(args.saturated_scenes)
    if args.saturated_scenes_dir:
        saturated_scenes += glob_ply_paths(args.saturated_scenes_dir)
    if not neutral_scenes and not saturated_scenes:
        raise ValueError("No scenes given -- use --neutral_scenes/--neutral_scenes_dir and/or "
                        "--saturated_scenes/--saturated_scenes_dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    if os.path.isdir(args.style):
        style_paths = glob_labeled_paths(args.style)
        if not style_paths:
            raise ValueError(f"No .jpg/.jpeg/.png images found directly inside {args.style!r}.")
        if args.qualitative:
            raise ValueError("--qualitative needs one fixed style image, not a --style folder (batch mode).")
        run_batch(neutral_scenes, saturated_scenes, style_paths, args.criteria, args.output_dir,
                 palette_num=args.palette_num, power=args.power, margin_base_scale=args.margin_base_scale,
                 seed=args.seed, c_thresh=args.c_thresh, swap_thresh=args.swap_thresh)
        return

    style_srgb = load_style_srgb(args.style)

    all_rows = []
    all_states = {}
    for group_name, scenes in [("neutral", neutral_scenes), ("saturated", saturated_scenes)]:
        for scene_label, scene_ply in scenes:
            rows, state = run_scene(scene_label, scene_ply, style_srgb, args.criteria,
                                    palette_num=args.palette_num, power=args.power,
                                    margin_base_scale=args.margin_base_scale, seed=args.seed,
                                    c_thresh=args.c_thresh, swap_thresh=args.swap_thresh)
            for r in rows:
                r["group"] = group_name
            all_rows.extend(rows)
            all_states[scene_label] = state

            plot_ab_matches(state["scene_ab"], state["style_ab"], state["per_criterion"], args.criteria,
                           os.path.join(args.output_dir, f"ab_match_{scene_label}.png"), c_thresh=args.c_thresh)

    write_csv(all_rows, os.path.join(args.output_dir, "matching_criterion_results.csv"))

    if args.qualitative:
        qual_label = args.qual_scene or (neutral_scenes[0][0] if neutral_scenes else saturated_scenes[0][0])
        qual_ply = dict(neutral_scenes + saturated_scenes)[qual_label]
        if not args.qual_source_path:
            raise ValueError("--qualitative requires --qual_source_path (the -s dataset dir for that scene).")
        run_qualitative(qual_label, qual_ply, all_states[qual_label]["style_ab"],
                       all_states[qual_label]["per_criterion"], args.criteria, args.qual_cam_idx,
                       args.qual_source_path, args.qual_model_path or args.output_dir, args.output_dir)


    style_specs = [
    ("Starry Night", "datasets/styles/14.jpg"),
    ("Palette E",     "datasets/palettes/E.png"),
    ("Sketch",        "datasets/styles/2.jpg"),
    ]
    run_multistyle_figure(
        "family",
        "output/ckpt_gs/tandt/family/point_cloud/iteration_30000/point_cloud.ply",
        style_specs, args.criteria, args.output_dir,
        scene_img_path="/data/storage/users/msabater/datasets/tandt/family/images/0_000009.png",   # reference column
    )

if __name__ == "__main__":
    main()
