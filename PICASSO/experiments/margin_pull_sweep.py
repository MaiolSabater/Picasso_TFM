"""
Margin-Pull Radius Sweep experiment.

Characterizes how the adaptive margin-pull's radius scale (margin_base_scale,
see utils/margin_pull.py) trades target-fidelity against chroma-diversity in
the recoloured Gaussian chroma, and confirms the component's value against
margin-pull disabled entirely (pure IDW).

Efficiency: the palette extraction, style matching, and IDW ab-shift are
IDENTICAL across every margin_base_scale value -- only the margin-pull itself
differs. So all of that is computed ONCE, and each sweep point only re-runs
the (cheap) apply_margin_pull call. This is why the whole sweep is fast even
though it touches every Gaussian in the scene.

This script does NOT recolour or render training images -- everything is
computed directly on the Gaussians' own SH0-derived ab/L, matching
--gt_theme_source gaussians. Use --qualitative to additionally render one
fixed camera at each scale (requires loading the full scene + renderer).

Matching front-end: DIRECT-MATCH (each scene palette colour individually
matched to a style colour), with --match_by hue. Note: direct_match.py (see
utils/direct_match.py) has no real hue-tolerance algorithm -- only "unique"
(Hungarian one-to-one) and "nearest" (unconditional per-colour nearest, no
threshold) are actually implemented there, so match_by="hue" resolves to
"nearest" internally, same as train_style.py's --gt_match_by hue does for the
direct-match front-end. This script prints that mapping explicitly so it's
never a silent surprise.

Usage:
    python experiments/margin_pull_sweep.py \\
        -s <data_dir> -m <dummy_model_dir> \\
        --point_cloud <path/to/point_cloud.ply> \\
        --style <path/to/style_image.jpg> \\
        --output_dir <output_dir> \\
        [--scales 0.0 0.1 0.25 0.5 1.0 inf] [--default_scale 0.1]
        [--palette_num 5] [--power 2.0] [--seed 0]
        [--qualitative --qual_cam_idx 0]
"""
import os
# Must be set before numpy/scipy/sklearn (anything touching OpenBLAS) is
# imported -- on this cluster's high-core-count machines, OpenBLAS otherwise
# tries to spin up a thread pool sized to the core count and crashes
# ("tried to allocate too many memory regions", built for max 128 threads).
# Same guard train_style.py uses, for the same reason.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STYLIZEDGS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STYLIZEDGS_DIR not in sys.path:
    sys.path.insert(0, _STYLIZEDGS_DIR)
_GROUPTHEME_DIR = os.path.join(_STYLIZEDGS_DIR, "..", "grouptheme")
if _GROUPTHEME_DIR not in sys.path:
    sys.path.insert(0, _GROUPTHEME_DIR)

import grouptheme as gt
from grouptheme.recolor import recolor_ab_L
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB, RGB2SH
from utils.direct_match import direct_scene_to_style_landmarks
from utils.margin_pull import assign_nearest_theme
from utils.margin_pull_step import apply_margin_pull
from utils.group_theme_recolor_new import _extract_palette_from_gaussians
from utils.palette_mapping_viz import save_palette_mapping

# match_by -> what direct_match.py's `match` argument actually understands.
# "hue" has no real algorithm there (see module docstring); it maps to "nearest".
_DIRECT_MATCH_BY_MAP = {"unique": "unique", "hue": "nearest", "ab": "nearest", "nearest": "nearest"}


def load_style_srgb(path, max_dim=512):
    """Load a style image as an (H,W,3) float64 sRGB array in [0,1], capped to
    max_dim on its longest side (palette extraction doesn't need full res)."""
    import imageio.v2 as imageio
    img = imageio.imread(path, pilmode="RGB").astype(np.float64) / 255.0
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        import cv2
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def gaussian_dc_lab(gaussians):
    """Every Gaussian's SH0-derived Lab color. Same convention as
    _extract_palette_from_gaussians / recolor_scene_to_style's step 4b: SH0's
    RGB is trained to approximate sRGB directly (no gamma step)."""
    C0 = 0.28209479177387814
    with torch.no_grad():
        sh0 = gaussians.get_features_dc.squeeze(1).detach().cpu().numpy().astype(np.float64)
    srgb = np.clip(0.5 + C0 * sh0, 0.0, 1.0)
    lab = gt.colorspace.srgb_to_lab(srgb)
    return lab[:, 0], lab[:, 1:]   # dc_L, dc_ab


def compute_pre_margin_pull(gaussians, style_srgb, palette_num, power, seed,
                            match_by, l_strength=0.0):
    """Everything that's identical across the whole scale sweep: palette
    extraction, direct-match landmarks, and the one-time IDW ab-shift.
    Returns a dict with dc_ab, dc_L, final_theme_ab, and new_dc_ab_preMP.
    """
    dc_L, dc_ab = gaussian_dc_lab(gaussians)

    scene_pal = _extract_palette_from_gaussians(gaussians, max_points=50_000, seed=seed, weight_by="opacity")
    scene_ab, scene_L = scene_pal.ab, scene_pal.L

    style_palette = gt.extract_palette(style_srgb, seed=seed, max_pixels=50_000,
                                       k_min=palette_num, k_max=palette_num)
    style_lab = gt.colorspace.srgb_to_lab(style_palette.srgb())

    direct_match_arg = _DIRECT_MATCH_BY_MAP[match_by]
    print(f">>> match_by='{match_by}' -> direct_scene_to_style_landmarks(match='{direct_match_arg}')")

    (src_all, dst_all, srcL_all, dstL_all,
     theme_of_landmark, final_theme_ab, mapping) = direct_scene_to_style_landmarks(
        scene_ab, scene_L, style_lab[:, 1:], style_lab[:, 0],
        recolor_space="ab", manual_theme_colors=None,
        srgb_to_lab=gt.colorspace.srgb_to_lab,
        match=direct_match_arg, debug=False)

    new_dc_ab_preMP, dLg = recolor_ab_L(dc_ab, src_all, dst_all, srcL_all, dstL_all,
                                        power=power, l_strength=l_strength)

    return dict(dc_ab=dc_ab, dc_L=dc_L, dLg=dLg, final_theme_ab=final_theme_ab,
                new_dc_ab_preMP=new_dc_ab_preMP, mapping=mapping,
                scene_ab=scene_ab, scene_L=scene_L,
                style_lab=style_lab,
                style_srgb=style_palette.srgb())


def ab_histogram(ab, grid_edges):
    H, _, _ = np.histogram2d(ab[:, 0], ab[:, 1], bins=[grid_edges, grid_edges])
    return H


def entropy_of_hist(H):
    p = H / H.sum()
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def run_sweep(pre, scales, grid_range=(-100, 100), grid_bins=64):
    """scales: list where a float is a real margin_base_scale, and None means
    "off" (margin-pull disabled, pure IDW). Returns a list of row dicts."""
    dc_ab = pre["dc_ab"]
    new_dc_ab_preMP = pre["new_dc_ab_preMP"]
    final_theme_ab = pre["final_theme_ab"]
    n = len(dc_ab)
    grid_edges = np.linspace(grid_range[0], grid_range[1], grid_bins + 1)

    rows = []
    per_scale_ab = {}
    for scale in scales:
        enabled = scale is not None
        if enabled:
            new_ab, info = apply_margin_pull(dc_ab, new_dc_ab_preMP, final_theme_ab,
                                             base_scale=scale, enabled=True, return_info=True)
            revert_pct = 100.0 * info["n_hurt"] / n
            clamp_pct = 100.0 * info["n_clamped"] / n
            keep_pct = 100.0 * info["n_inside"] / n
            assign = info["assign"]
        else:
            new_ab = apply_margin_pull(dc_ab, new_dc_ab_preMP, final_theme_ab,
                                       base_scale=1.0, enabled=False, return_info=False)
            # No pull mechanism runs at all -- by definition nothing is
            # reverted/clamped, every point trivially "keeps" its IDW result.
            revert_pct, clamp_pct, keep_pct = 0.0, 0.0, 100.0
            # Same Fix-A convention (assign on post-IDW position) used when
            # enabled, so fidelity is comparable across every row.
            assign = assign_nearest_theme(new_ab, final_theme_ab)

        target_pts = final_theme_ab[assign]
        fidelity = float(np.mean(np.linalg.norm(new_ab - target_pts, axis=1)))
        H = ab_histogram(new_ab, grid_edges)
        entropy = entropy_of_hist(H)

        label = "inf" if not enabled else scale
        rows.append(dict(scale=label, revert_pct=revert_pct, clamp_pct=clamp_pct,
                         keep_pct=keep_pct, fidelity=fidelity, entropy=entropy))
        per_scale_ab[label] = new_ab
        print(f"scale={label!s:>5}  revert={revert_pct:5.1f}%  clamp={clamp_pct:5.1f}%  "
              f"keep={keep_pct:5.1f}%  fidelity={fidelity:6.2f}  entropy={entropy:.3f}")

    return rows, per_scale_ab


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale", "revert_pct", "clamp_pct", "keep_pct", "fidelity", "entropy"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(">>> wrote", path)


def plot_sweep(rows, path, default_scale=0.1):
    # A log x-axis can't represent 0.0 (undefined) or inf (no fixed position),
    # but both are real, important sweep points (the "full snap" and "off"
    # extremes) -- so both get plotted off-axis, at placeholder positions to
    # the left/right of the real log-scaled data, connected by dashed
    # (not solid) segments to signal they aren't truly at that x position.
    zero_rows = [r for r in rows if r["scale"] == 0.0]
    pos_rows = sorted((r for r in rows if isinstance(r["scale"], float) and r["scale"] > 0),
                      key=lambda r: r["scale"])
    inf_rows = [r for r in rows if r["scale"] == "inf"]

    xs = [r["scale"] for r in pos_rows]
    fidelity = [r["fidelity"] for r in pos_rows]
    entropy = [r["entropy"] for r in pos_rows]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(xs, fidelity, "o-", color="tab:red", label="Fidelity (mean dist. to target)")
    ax2.plot(xs, entropy, "s-", color="tab:blue", label="Entropy (diversity, nats)")

    tick_positions = list(xs)
    tick_labels = [str(x) for x in xs]

    if xs:
        x_min, x_max = min(xs), max(xs)
        x_zero = x_min / 3.0
        x_off = x_max * 3.0

        for r in zero_rows:
            ax1.plot([x_zero, x_min], [r["fidelity"], fidelity[0]], "--", color="tab:red", alpha=0.5, linewidth=1)
            ax2.plot([x_zero, x_min], [r["entropy"], entropy[0]], "--", color="tab:blue", alpha=0.5, linewidth=1)
            ax1.plot([x_zero], [r["fidelity"]], "o", color="tab:red", markerfacecolor="white", markeredgewidth=2)
            ax2.plot([x_zero], [r["entropy"]], "s", color="tab:blue", markerfacecolor="white", markeredgewidth=2)
            tick_positions.insert(0, x_zero); tick_labels.insert(0, "0.0")

        for r in inf_rows:
            ax1.plot([x_max, x_off], [fidelity[-1], r["fidelity"]], "--", color="tab:red", alpha=0.5, linewidth=1)
            ax2.plot([x_max, x_off], [entropy[-1], r["entropy"]], "--", color="tab:blue", alpha=0.5, linewidth=1)
            ax1.plot([x_off], [r["fidelity"]], "o", color="tab:red", markerfacecolor="white", markeredgewidth=2)
            ax2.plot([x_off], [r["entropy"]], "s", color="tab:blue", markerfacecolor="white", markeredgewidth=2)
            tick_positions.append(x_off); tick_labels.append("off (∞)")

        ax1.set_xscale("log")
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)

    if default_scale in xs:
        ax1.axvline(default_scale, color="grey", linestyle="--", alpha=0.7)
        ax1.annotate(f"default={default_scale}", xy=(default_scale, ax1.get_ylim()[1]),
                    xytext=(3, -3), textcoords="offset points", fontsize=8, color="grey")

    ax1.set_xlabel("margin_base_scale  (0.0 and off/∞ plotted off-axis, dashed connectors)")
    ax1.set_ylabel("Fidelity: mean distance to assigned target (ab units)", color="tab:red")
    ax2.set_ylabel("Entropy: diversity of recoloured ab histogram (nats)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Margin-pull radius sweep: fidelity vs. diversity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(">>> wrote", path)


def run_qualitative(gaussians, pres, styles_srgb, scales, cam_idx, source_path,
                    model_path, point_cloud, output_dir, l_strength=0.0):
    """Render one fixed camera at each scale, one ROW per style, with the style
    image shown as a reference column at the left of each row."""
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams
    from scene import Scene
    from gaussian_renderer import render as gs_render

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parsed_args = parser.parse_args(["-s", source_path, "-m", model_path])
    dataset = model.extract(parsed_args)
    pipe = pipeline.extract(parsed_args)

    # Load the scene once (cameras + renderer); re-load the ply Scene discards.
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    gaussians.load_ply(point_cloud)
    cam = scene.getTrainCameras()[cam_idx]
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    original_dc = gaussians.get_features_dc.detach().clone()

    n_styles = len(pres)
    n_cols = 1 + len(scales)          # +1 for the reference style image column
    fig, axes = plt.subplots(n_styles, n_cols,
                             figsize=(4 * n_cols, 4 * n_styles))
    # normalise axes to a 2D array even when n_styles==1 or n_cols==1
    axes = np.atleast_2d(axes)
    if n_styles == 1 and axes.shape[0] != 1:
        axes = axes.reshape(1, -1)

    for r, (pre, style_srgb) in enumerate(zip(pres, styles_srgb)):
        dc_L, dLg = pre["dc_L"], pre["dLg"]
        final_theme_ab = pre["final_theme_ab"]
        dc_ab, new_dc_ab_preMP = pre["dc_ab"], pre["new_dc_ab_preMP"]

        # column 0: the style image itself, as reference
        axes[r, 0].imshow(np.clip(style_srgb, 0, 1))
        axes[r, 0].set_title("style" if r == 0 else "", fontsize=30)
        axes[r, 0].set_ylabel(f"style {r+1}", fontsize=30)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        for c, scale in enumerate(scales):
            enabled = scale is not None
            if enabled:
                new_ab = apply_margin_pull(dc_ab, new_dc_ab_preMP, final_theme_ab,
                                           base_scale=scale, enabled=True)
            else:
                new_ab = new_dc_ab_preMP
            new_L = dc_L + dLg
            new_lab = np.concatenate([new_L[:, None], new_ab], axis=1)
            new_srgb = np.clip(gt.colorspace.lab_to_srgb(new_lab), 0.0, 1.0).astype(np.float32)
            sh0 = RGB2SH(torch.from_numpy(new_srgb).to(gaussians.get_features_dc.device))
            with torch.no_grad():
                gaussians._features_dc.data.copy_(sh0.reshape(gaussians._features_dc.shape))
                rendering = gs_render(cam, gaussians, pipe, bg)["render"].clamp(0, 1)
            img = rendering.permute(1, 2, 0).detach().cpu().numpy()

            ax = axes[r, c + 1]
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"scale={'off' if not enabled else scale}", fontsize=30)
            ax.axis("off")

    with torch.no_grad():
        gaussians._features_dc.data.copy_(original_dc)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "margin_sweep_qualitative.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(">>> wrote", out_path)


def parse_scale(s):
    return None if s.lower() in ("inf", "infinity", "off", "none") else float(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True, help="Any writable dir (only used for the optional --qualitative camera load).")
    parser.add_argument("--point_cloud", required=True)
    parser.add_argument("--style", nargs="+", required=True,
                    help="One or more style images; each becomes a row in the qualitative figure.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scales", nargs="+", type=parse_scale, default=[0.0, 0.1, 0.25, 0.5, 1.0, "inf"],
                        help="Pass 'inf' for margin-pull disabled (pure IDW).")
    parser.add_argument("--default_scale", type=float, default=0.1, help="Annotated as the chosen default on the plot.")
    parser.add_argument("--palette_num", type=int, default=5)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--l_strength", type=float, default=0.0)
    parser.add_argument("--match_by", choices=["unique", "hue", "ab"], default="hue")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_min", type=float, default=-100.0)
    parser.add_argument("--grid_max", type=float, default=100.0)
    parser.add_argument("--grid_bins", type=int, default=64)
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--qual_cam_idx", type=int, default=0)
    args = parser.parse_args()
    scales = [(None if isinstance(s, str) else s) for s in args.scales]
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    gaussians = GaussianModel(3)
    gaussians.load_ply(args.point_cloud)

    # Compute a `pre` (palette + landmarks + IDW) for EACH style.
    styles_srgb = [load_style_srgb(s) for s in args.style]
    pres = [compute_pre_margin_pull(gaussians, ssrgb, args.palette_num, args.power,
                                    args.seed, args.match_by, l_strength=args.l_strength)
            for ssrgb in styles_srgb]

    # Quantitative sweep + CSV + plot + palette-map on the FIRST style only
    # (the sweep characterises the margin trade-off, style-independent enough
    # that one representative style suffices; the qualitative figure shows all).
    pre0 = pres[0]
    rows, per_scale_ab = run_sweep(pre0, scales, grid_range=(args.grid_min, args.grid_max), grid_bins=args.grid_bins)
    write_csv(rows, os.path.join(args.output_dir, "margin_sweep.csv"))
    plot_sweep(rows, os.path.join(args.output_dir, "margin_sweep.png"), default_scale=args.default_scale)
    save_palette_mapping(pre0["scene_ab"], pre0["scene_L"], pre0["style_srgb"],
                         pre0["mapping"], os.path.join(args.output_dir, "palette_mapping.png"))

    if args.qualitative:
        run_qualitative(gaussians, pres, styles_srgb, scales, args.qual_cam_idx,
                        args.source_path, args.model_path, args.point_cloud,
                        args.output_dir, l_strength=args.l_strength)


if __name__ == "__main__":
    main()
