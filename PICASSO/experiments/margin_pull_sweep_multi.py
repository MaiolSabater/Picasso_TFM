"""
Multi-scene x multi-style wrapper around margin_pull_sweep.py.

Runs the margin-pull radius sweep (see margin_pull_sweep.py's own docstring
for the mechanism -- palette extraction, direct-match landmarks, and the IDW
ab-shift are computed once per (scene, style) pair, then only apply_margin_pull
is re-run per scale) independently for every (scene, style) pair, then
aggregates across pairs.

Aggregation: each pair's fidelity/entropy is normalized to that SAME pair's
own margin-pull-DISABLED (inf/off) baseline BEFORE averaging across pairs --
e.g. fidelity_norm(scale) = fidelity(scale) / fidelity(inf). Raw ab-distance
and nats scale with a scene's own color spread and palette size, so averaging
raw numbers would let one wide-gamut pair dominate the trend; normalizing to
each pair's own "no pull" case turns it into a comparable relative statement
("how much does margin-pull help, as a fraction of the off case") that's
meaningful to average across heterogeneous scenes/styles.

Every per-pair result is ALSO written out in full -- its own margin_sweep.csv
/ margin_sweep.png (identical format to the single-pair script), plus one row
per (scene, style, scale) in a combined raw CSV -- so the aggregated trend can
be checked pair-by-pair, not just trusted as an average.

Usage:
    python experiments/margin_pull_sweep_multi.py \\
        --scenes truck:output/ckpt_gs/tandt/truck/point_cloud/iteration_30000/point_cloud.ply \\
                 family:output/ckpt_gs/tandt/family/point_cloud/iteration_30000/point_cloud.ply \\
        --styles warm:/data/.../styles/14.jpg cool:/data/.../styles/9.jpg \\
        --output_dir <output_dir> \\
        [--scales 0.0 0.1 0.25 0.5 1.0 inf] [--default_scale 0.1]
        [--palette_num 5] [--power 2.0] [--seed 0] [--match_by hue]

Each --scenes / --styles entry is "label:path" -- the label is what shows up
in filenames and the aggregate table, so keep it short and filesystem-safe.
"""
import os
# Must be set before numpy/scipy/sklearn (anything touching OpenBLAS) is
# imported -- on this cluster's high-core-count machines, OpenBLAS otherwise
# tries to spin up a thread pool sized to the core count and crashes
# ("tried to allocate too many memory regions", built for max 128 threads).
# Same guard train_style.py uses, for the same reason. This has to be set
# here too (not just in margin_pull_sweep.py) since this file imports numpy
# directly, before it ever reaches the `from margin_pull_sweep import ...`
# line further down -- by then numpy/OpenBLAS would already be initialized.
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

from margin_pull_sweep import (
    compute_pre_margin_pull, run_sweep, write_csv, plot_sweep,
    load_style_srgb, parse_scale,
)
from scene.gaussian_model import GaussianModel


def _check_no_duplicate_labels(pairs):
    # Pairs are keyed by label (output dirs, aggregate rows) -- a repeated
    # label would silently collide (later entries overwrite earlier ones'
    # output dir, and merge in the aggregate) rather than error, so catch
    # it here instead of producing quietly-wrong results.
    seen = {}
    for label, path in pairs:
        if label in seen and seen[label] != path:
            raise ValueError(f"Duplicate label {label!r} used for two different paths "
                            f"({seen[label]!r} and {path!r}) -- each entry needs a unique label.")
        seen[label] = path


def parse_labeled_paths(items):
    out = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Expected 'label:path', got: {item!r}")
        label, path = item.split(":", 1)
        out.append((label, path))
    _check_no_duplicate_labels(out)
    return out


def glob_labeled_paths(dir_path, extensions=(".jpg", ".jpeg", ".png")):
    """Every image file directly inside dir_path (not recursive), labeled by
    its filename stem -- e.g. '14.jpg' -> label '14'."""
    out = []
    for name in sorted(os.listdir(dir_path)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in extensions:
            out.append((stem, os.path.join(dir_path, name)))
    _check_no_duplicate_labels(out)
    return out


def glob_ply_paths(dir_path):
    """Every point_cloud.ply found recursively under dir_path, labeled by
    whatever directory sits two levels above it -- matches this repo's own
    convention, .../<scene_name>/point_cloud/iteration_XXXXX/point_cloud.ply.
    When a scene has multiple iteration checkpoints (e.g. both iteration_7000
    and iteration_30000, as train.py's own default checkpoints do), only the
    highest iteration is kept -- the most-trained one, matching how every
    other script in this repo defaults to the final checkpoint -- rather than
    erroring on the resulting duplicate label."""
    import glob
    import re
    by_label = {}   # label -> (iteration, path)
    for ply_path in sorted(glob.glob(os.path.join(dir_path, "**", "point_cloud.ply"), recursive=True)):
        iter_dir = os.path.basename(os.path.dirname(ply_path))         # "iteration_30000"
        scene_dir = os.path.dirname(os.path.dirname(os.path.dirname(ply_path)))
        label = os.path.basename(scene_dir)
        m = re.match(r"iteration_(\d+)", iter_dir)
        iteration = int(m.group(1)) if m else -1
        if label not in by_label or iteration > by_label[label][0]:
            by_label[label] = (iteration, ply_path)
    return sorted((label, path) for label, (_, path) in by_label.items())


def run_multi_sweep(scenes, styles, scales, output_dir, palette_num=5, power=2.0,
                    l_strength=0.0, match_by="hue", seed=0,
                    grid_range=(-100, 100), grid_bins=64, default_scale=0.1):
    """
    scenes / styles : list of (label, path) tuples.
    scales           : list where a float is a real margin_base_scale and None
                       means "off" (margin-pull disabled) -- same convention
                       as margin_pull_sweep.run_sweep.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_rows = []   # one row per (scene, style, scale)

    for scene_label, scene_ply in scenes:
        print(f"\n=== scene: {scene_label} ({scene_ply}) ===")
        gaussians = GaussianModel(3)
        gaussians.load_ply(scene_ply)

        for style_label, style_path in styles:
            pair_label = f"{scene_label}__{style_label}"
            print(f"--- style: {style_label} ({style_path}) ---")
            style_srgb = load_style_srgb(style_path)

            pre = compute_pre_margin_pull(gaussians, style_srgb, palette_num, power,
                                          seed, match_by, l_strength=l_strength)
            rows, _ = run_sweep(pre, scales, grid_range=grid_range, grid_bins=grid_bins)

            pair_dir = os.path.join(output_dir, pair_label)
            os.makedirs(pair_dir, exist_ok=True)
            write_csv(rows, os.path.join(pair_dir, "margin_sweep.csv"))
            plot_sweep(rows, os.path.join(pair_dir, "margin_sweep.png"), default_scale=default_scale)

            for r in rows:
                raw_rows.append(dict(scene=scene_label, style=style_label, **r))

    write_raw_csv(raw_rows, os.path.join(output_dir, "margin_sweep_all_pairs.csv"))

    scale_labels = ["inf" if s is None else s for s in scales]
    agg_rows = aggregate(raw_rows, scale_labels)
    write_agg_csv(agg_rows, os.path.join(output_dir, "margin_sweep_aggregate.csv"))
    plot_aggregate(agg_rows, os.path.join(output_dir, "margin_sweep_aggregate.png"), default_scale=default_scale)

    return raw_rows, agg_rows


def write_raw_csv(raw_rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "style", "scale", "revert_pct", "clamp_pct",
                                          "keep_pct", "fidelity", "entropy"])
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)
    print(">>> wrote", path)


def aggregate(raw_rows, scale_labels):
    """Normalize each (scene,style) pair's fidelity/entropy to that pair's own
    'inf' (off) row, THEN average the normalized values across pairs, per scale."""
    pairs = {}
    for r in raw_rows:
        pairs.setdefault((r["scene"], r["style"]), {})[r["scale"]] = r

    normed = {label: {"fidelity": [], "entropy": [], "revert_pct": [], "clamp_pct": [], "keep_pct": []}
             for label in scale_labels}

    skipped = []
    for key, by_scale in pairs.items():
        if "inf" not in by_scale:
            skipped.append((key, "no off/inf baseline row"))
            continue
        base_fid = by_scale["inf"]["fidelity"]
        base_ent = by_scale["inf"]["entropy"]
        if base_fid <= 0 or base_ent <= 0:
            skipped.append((key, f"degenerate baseline (fidelity={base_fid}, entropy={base_ent})"))
            continue
        for label in scale_labels:
            if label not in by_scale:
                continue
            row = by_scale[label]
            normed[label]["fidelity"].append(row["fidelity"] / base_fid)
            normed[label]["entropy"].append(row["entropy"] / base_ent)
            normed[label]["revert_pct"].append(row["revert_pct"])
            normed[label]["clamp_pct"].append(row["clamp_pct"])
            normed[label]["keep_pct"].append(row["keep_pct"])

    for key, reason in skipped:
        print(f">>> WARNING: pair {key} excluded from aggregate ({reason})")

    agg_rows = []
    for label in scale_labels:
        vals = normed[label]
        n = len(vals["fidelity"])
        agg_rows.append(dict(
            scale=label, n_pairs=n,
            fidelity_norm_mean=float(np.mean(vals["fidelity"])) if n else float("nan"),
            fidelity_norm_std=float(np.std(vals["fidelity"])) if n else float("nan"),
            entropy_norm_mean=float(np.mean(vals["entropy"])) if n else float("nan"),
            entropy_norm_std=float(np.std(vals["entropy"])) if n else float("nan"),
            revert_pct_mean=float(np.mean(vals["revert_pct"])) if n else float("nan"),
            clamp_pct_mean=float(np.mean(vals["clamp_pct"])) if n else float("nan"),
            keep_pct_mean=float(np.mean(vals["keep_pct"])) if n else float("nan"),
        ))
    return agg_rows


def write_agg_csv(agg_rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale", "n_pairs", "fidelity_norm_mean", "fidelity_norm_std",
                                          "entropy_norm_mean", "entropy_norm_std", "revert_pct_mean",
                                          "clamp_pct_mean", "keep_pct_mean"])
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)
    print(">>> wrote", path)


def plot_aggregate(agg_rows, path, default_scale=0.1):
    """Same off-axis 0.0/inf handling as margin_pull_sweep.plot_sweep (a log
    x-axis can't represent either), but plotting mean +/- std of the
    per-pair-normalized fidelity/entropy, with a y=1.0 reference line marking
    'same as the off/inf baseline'."""
    zero = [r for r in agg_rows if r["scale"] == 0.0]
    pos = sorted((r for r in agg_rows if isinstance(r["scale"], float) and r["scale"] > 0),
                key=lambda r: r["scale"])
    inf_ = [r for r in agg_rows if r["scale"] == "inf"]

    xs = [r["scale"] for r in pos]
    fid_mean = [r["fidelity_norm_mean"] for r in pos]
    fid_std = [r["fidelity_norm_std"] for r in pos]
    ent_mean = [r["entropy_norm_mean"] for r in pos]
    ent_std = [r["entropy_norm_std"] for r in pos]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.errorbar(xs, fid_mean, yerr=fid_std, fmt="o-", color="tab:red", capsize=3,
                label="Fidelity (normalized to off)")
    ax2.errorbar(xs, ent_mean, yerr=ent_std, fmt="s-", color="tab:blue", capsize=3,
                label="Entropy (normalized to off)")
    ax1.axhline(1.0, color="grey", linestyle=":", alpha=0.6, linewidth=1)

    tick_positions = list(xs)
    tick_labels = [str(x) for x in xs]

    if xs:
        x_min, x_max = min(xs), max(xs)
        x_zero = x_min / 3.0
        x_off = x_max * 3.0

        for r in zero:
            ax1.plot([x_zero, x_min], [r["fidelity_norm_mean"], fid_mean[0]], "--", color="tab:red", alpha=0.5, linewidth=1)
            ax2.plot([x_zero, x_min], [r["entropy_norm_mean"], ent_mean[0]], "--", color="tab:blue", alpha=0.5, linewidth=1)
            ax1.errorbar([x_zero], [r["fidelity_norm_mean"]], yerr=[r["fidelity_norm_std"]], fmt="o",
                        color="tab:red", markerfacecolor="white", markeredgewidth=2, capsize=3)
            ax2.errorbar([x_zero], [r["entropy_norm_mean"]], yerr=[r["entropy_norm_std"]], fmt="s",
                        color="tab:blue", markerfacecolor="white", markeredgewidth=2, capsize=3)
            tick_positions.insert(0, x_zero); tick_labels.insert(0, "0.0")

        for r in inf_:
            ax1.plot([x_max, x_off], [fid_mean[-1], r["fidelity_norm_mean"]], "--", color="tab:red", alpha=0.5, linewidth=1)
            ax2.plot([x_max, x_off], [ent_mean[-1], r["entropy_norm_mean"]], "--", color="tab:blue", alpha=0.5, linewidth=1)
            ax1.plot([x_off], [r["fidelity_norm_mean"]], "o", color="tab:red", markerfacecolor="white", markeredgewidth=2)
            ax2.plot([x_off], [r["entropy_norm_mean"]], "s", color="tab:blue", markerfacecolor="white", markeredgewidth=2)
            tick_positions.append(x_off); tick_labels.append("off (∞)")

        ax1.set_xscale("log")
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)

    if default_scale in xs:
        ax1.axvline(default_scale, color="grey", linestyle="--", alpha=0.7)
        ax1.annotate(f"default={default_scale}", xy=(default_scale, ax1.get_ylim()[1]),
                    xytext=(3, -3), textcoords="offset points", fontsize=8, color="grey")

    n_pairs = agg_rows[0]["n_pairs"] if agg_rows else 0
    ax1.set_xlabel("margin_base_scale  (0.0 and off/∞ plotted off-axis, dashed connectors)")
    ax1.set_ylabel("Fidelity relative to off baseline (1.0 = same as off)", color="tab:red")
    ax2.set_ylabel("Entropy relative to off baseline (1.0 = same as off)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(f"Margin-pull radius sweep, aggregated over {n_pairs} scene x style pairs\n"
                  f"(mean +/- std, each pair normalized to its own off/∞ baseline)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(">>> wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=[], help="label:path/to/point_cloud.ply, one or more.")
    parser.add_argument("--scenes_dir", default=None,
                        help="Recursively globs every point_cloud.ply under this dir, labeled by the "
                             "directory two levels above each .ply (this repo's own <scene>/point_cloud/"
                             "iteration_XXXXX/point_cloud.ply convention). Combinable with --scenes.")
    parser.add_argument("--styles", nargs="+", default=[], help="label:path/to/style.jpg, one or more.")
    parser.add_argument("--styles_dir", default=None,
                        help="Every .jpg/.jpeg/.png directly inside this dir, labeled by filename stem "
                             "(e.g. 14.jpg -> label '14'). Combinable with --styles.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scales", nargs="+", type=parse_scale, default=[0.0, 0.1, 0.25, 0.5, 1.0, "inf"],
                        help="Pass 'inf' for margin-pull disabled (pure IDW).")
    parser.add_argument("--default_scale", type=float, default=0.1)
    parser.add_argument("--palette_num", type=int, default=5)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--l_strength", type=float, default=0.0)
    parser.add_argument("--match_by", choices=["unique", "hue", "ab"], default="hue")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_min", type=float, default=-100.0)
    parser.add_argument("--grid_max", type=float, default=100.0)
    parser.add_argument("--grid_bins", type=int, default=64)
    args = parser.parse_args()

    scenes = parse_labeled_paths(args.scenes)
    if args.scenes_dir:
        scenes = scenes + glob_ply_paths(args.scenes_dir)
    styles = parse_labeled_paths(args.styles)
    if args.styles_dir:
        styles = styles + glob_labeled_paths(args.styles_dir)
    _check_no_duplicate_labels(scenes)   # re-check after merging --scenes with --scenes_dir
    _check_no_duplicate_labels(styles)   # re-check after merging --styles with --styles_dir
    if not scenes:
        raise ValueError("No scenes given -- use --scenes and/or --scenes_dir.")
    if not styles:
        raise ValueError("No styles given -- use --styles and/or --styles_dir.")
    print(f">>> {len(scenes)} scene(s) x {len(styles)} style(s) = {len(scenes) * len(styles)} pairs")

    # same argparse default-bypasses-type quirk as margin_pull_sweep.py
    scales = [(None if isinstance(s, str) else s) for s in args.scales]

    np.random.seed(args.seed)

    run_multi_sweep(scenes, styles, scales, args.output_dir,
                    palette_num=args.palette_num, power=args.power, l_strength=args.l_strength,
                    match_by=args.match_by, seed=args.seed,
                    grid_range=(args.grid_min, args.grid_max), grid_bins=args.grid_bins,
                    default_scale=args.default_scale)


if __name__ == "__main__":
    main()
