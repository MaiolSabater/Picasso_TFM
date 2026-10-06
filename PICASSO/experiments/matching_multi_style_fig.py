# New: a single combined figure for ONE scene, rows = styles, columns =
# [scene image][style image][one ab-plane per criterion].
# Drop this into matching_criterion_experiment.py (imports assumed already present:
# numpy as np, matplotlib.pyplot as plt, grouptheme as gt, imageio).

import numpy as np
import matplotlib.pyplot as plt
import grouptheme as gt
import imageio.v2 as imageio


def plot_ab_matches_multistyle(scene_ab, scene_img, style_entries, criteria,
                               out_path, c_thresh=10.0,
                               ax_size=4.5, font_scale=1.3):
    """
    scene_ab      : (K,2) the (shared) scene palette in ab.
    scene_img     : (H,W,3) an image of the scene (RGB, [0,1] or [0,255]) for the
                    reference column, OR None to skip that column.
    style_entries : list of dicts, one per style (row), each with:
                       'style_img'    : (H,W,3) the style image for the row header
                       'style_ab'     : (S,2) that style's palette in ab
                       'per_criterion': {crit: {'assign': ...}} for that style
                       'label'        : short style name (row label)
    criteria      : ordered list of criterion names (columns after the two
                    reference columns).
    ax_size       : inches per subplot (bigger = larger panels).
    font_scale    : multiplies all plot fonts.
    """
    n_rows = len(style_entries)
    has_scene_col = scene_img is not None
    n_ref_cols = (1 if has_scene_col else 0) + 1          # scene? + style
    n_cols = n_ref_cols + len(criteria)

    fs_title = 14 * font_scale
    fs_label = 12 * font_scale
    fs_tick = 10 * font_scale

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(ax_size * n_cols, ax_size * n_rows))
    axes = np.atleast_2d(axes)
    if axes.shape != (n_rows, n_cols):
        axes = axes.reshape(n_rows, n_cols)

    # scene palette colours tinted by their real colour (fixed L for display)
    scene_lab = np.concatenate([np.full((len(scene_ab), 1), 60.0), scene_ab], axis=1)
    scene_srgb = np.clip(gt.colorspace.lab_to_srgb(scene_lab), 0, 1)
    scene_chroma = np.linalg.norm(scene_ab, axis=1)

    def _prep_img(img):
        img = np.asarray(img)
        if img.dtype != np.uint8 and img.max() > 1.5:
            img = img / 255.0
        return np.clip(img, 0, 1) if img.dtype != np.uint8 else img

    for r, entry in enumerate(style_entries):
        style_ab = entry["style_ab"]
        per_criterion = entry["per_criterion"]
        style_lab = np.concatenate([np.full((len(style_ab), 1), 60.0), style_ab], axis=1)
        style_srgb = np.clip(gt.colorspace.lab_to_srgb(style_lab), 0, 1)

        col = 0
        # --- reference column: scene image (same every row, but shown per row
        #     so each row reads left-to-right on its own) ---
        if has_scene_col:
            ax = axes[r, col]
            ax.imshow(_prep_img(scene_img))
            if r == 0:
                ax.set_title("scene", fontsize=fs_title)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(entry.get("label", f"style {r+1}"), fontsize=fs_label)
            col += 1

        # --- reference column: this row's style image ---
        ax = axes[r, col]
        ax.imshow(_prep_img(entry["style_img"]))
        if r == 0:
            ax.set_title("style", fontsize=fs_title)
        ax.set_xticks([]); ax.set_yticks([])
        if not has_scene_col:
            ax.set_ylabel(entry.get("label", f"style {r+1}"), fontsize=fs_label)
        col += 1

        # --- one ab-plane per criterion ---
        for crit_name in criteria:
            ax = axes[r, col]
            assign = per_criterion[crit_name]["assign"]
            for k in range(len(scene_ab)):
                target = style_ab[assign[k]]
                is_neutral = scene_chroma[k] < c_thresh
                ax.annotate("", xy=target, xytext=scene_ab[k],
                            arrowprops=dict(arrowstyle="-|>",
                                            color="red" if is_neutral else "black",
                                            alpha=0.85 if is_neutral else 0.4,
                                            linewidth=2.0 if is_neutral else 1.0))
            ax.scatter(scene_ab[:, 0], scene_ab[:, 1], c=scene_srgb, s=160,
                       edgecolors="black", zorder=3, marker="o")
            ax.scatter(style_ab[:, 0], style_ab[:, 1], c=style_srgb, s=160,
                       edgecolors="black", zorder=3, marker="s")
            ax.add_patch(plt.Circle((0, 0), c_thresh, fill=False, linestyle=":",
                                    color="grey", alpha=0.7))
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.axvline(0, color="grey", linewidth=0.5)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=fs_tick)
            if r == 0:
                ax.set_title(crit_name, fontsize=fs_title)
            if r == n_rows - 1:
                ax.set_xlabel("a", fontsize=fs_label)
            if col == n_ref_cols:            # first ab-plane column
                ax.set_ylabel("b", fontsize=fs_label)
            col += 1

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(">>> wrote", out_path)