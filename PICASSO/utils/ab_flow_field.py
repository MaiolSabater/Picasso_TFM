"""
ab_flow_field.py

"Green probe for the whole ab-plane": sample a dense grid over ab-space, push
every grid point through the SAME recolour the Gaussians get (IDW shift, then
optionally the adaptive margin-pull), and draw the result as a flow field.

Combines:
  #1 quiver -- an arrow at each grid point from old_ab -> new_ab (the displacement)
  #3 destination tint -- each arrow coloured by the sRGB colour it LANDS on

so you can read, across all of ab at once, where each colour goes and how far it
moves. Landmarks (src circles -> dst diamonds) are overlaid, and the real
Gaussian cloud can be shown faintly so you see where points actually live.

Toggle `include_margin_pull`:
  False -> IDW field only (what recolor_ab_L does)
  True  -> IDW + adaptive margin-pull (the TRUE final field the Gaussians get)
Pass both (side by side) via save_ab_flow_compare to see what the pull adds.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _srgb_of_ab(ab, L=65.0, lab_to_srgb=None):
    """ab (N,2) -> sRGB (N,3) at fixed L, for tinting."""
    ab = np.asarray(ab, float)
    lab = np.concatenate([np.full((len(ab), 1), L), ab], axis=1)
    rgb = lab_to_srgb(lab)
    return np.clip(rgb, 0, 1)


def _grid(ab_min, ab_max, n):
    xs = np.linspace(ab_min[0], ab_max[0], n)
    ys = np.linspace(ab_min[1], ab_max[1], n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1), gx.shape


def compute_flow(grid_ab, src_all, dst_all, srcL_all, dstL_all,
                 recolor_ab_L, power=2.0, l_strength=0.0,
                 include_margin_pull=False, final_theme_ab=None,
                 margin_base_scale=0.5, assign_nearest_theme=None,
                 margin_pull_adaptive=None):
    """Push grid points through IDW (+ optional margin-pull). Returns new_ab."""
    new_ab, _dL = recolor_ab_L(grid_ab, src_all, dst_all, srcL_all, dstL_all,
                               power=power, l_strength=l_strength)
    if include_margin_pull:
        if (final_theme_ab is None or assign_nearest_theme is None
                or margin_pull_adaptive is None):
            raise ValueError("margin-pull requires final_theme_ab + the two fns")
        # Fix A: assign on the POST-IDW position so the pull reinforces the field's
        # direction instead of fighting it (a point the IDW pushed toward orange is
        # now nearest orange, not the gray it started near).
        a = assign_nearest_theme(new_ab, final_theme_ab)
        new_ab = margin_pull_adaptive(grid_ab, new_ab, final_theme_ab,
                                      assign=a, base_scale=margin_base_scale)
    return new_ab


def save_ab_flow_field(src_all, dst_all, srcL_all, dstL_all,
                       recolor_ab_L, lab_to_srgb, out_png,
                       power=2.0, l_strength=0.0, n=28,
                       ab_range=None, dc_ab=None,
                       include_margin_pull=False, final_theme_ab=None,
                       margin_base_scale=0.5, assign_nearest_theme=None,
                       margin_pull_adaptive=None,
                       title=None):
    """One flow-field panel. Arrows old->new, tinted by DESTINATION colour.

    ab_range : ((amin,bmin),(amax,bmax)); auto from landmarks+dc_ab if None.
    dc_ab    : optional (N,2) real Gaussian ab, drawn faintly for context.
    """
    src_all = np.asarray(src_all, float); dst_all = np.asarray(dst_all, float)

    # auto range: cover landmarks (src & dst) and the data, with margin
    if ab_range is None:
        pts = [src_all, dst_all]
        if dc_ab is not None and len(dc_ab):
            pts.append(np.asarray(dc_ab, float))
        allp = np.concatenate(pts, 0)
        lo = allp.min(0) - 15
        hi = allp.max(0) + 15
        ab_range = (lo, hi)
    ab_min, ab_max = ab_range

    grid_ab, shape = _grid(ab_min, ab_max, n)
    new_ab = compute_flow(grid_ab, src_all, dst_all, srcL_all, dstL_all,
                          recolor_ab_L, power=power, l_strength=l_strength,
                          include_margin_pull=include_margin_pull,
                          final_theme_ab=final_theme_ab,
                          margin_base_scale=margin_base_scale,
                          assign_nearest_theme=assign_nearest_theme,
                          margin_pull_adaptive=margin_pull_adaptive)

    disp = new_ab - grid_ab
    dest_rgb = _srgb_of_ab(new_ab, lab_to_srgb=lab_to_srgb)   # #3 destination tint

    fig, ax = plt.subplots(figsize=(8, 8))

    # faint real Gaussian cloud for context
    if dc_ab is not None and len(dc_ab):
        dc = np.asarray(dc_ab, float)
        if len(dc) > 20000:
            dc = dc[np.random.default_rng(0).choice(len(dc), 20000, replace=False)]
        ax.scatter(dc[:, 0], dc[:, 1], s=2, c="0.6", alpha=0.12, linewidths=0, zorder=1)

    # #1 quiver, #3 tint: arrow per grid point, coloured by where it lands
    ax.quiver(grid_ab[:, 0], grid_ab[:, 1], disp[:, 0], disp[:, 1],
              color=dest_rgb, angles="xy", scale_units="xy", scale=1.0,
              width=0.003, headwidth=4, headlength=5, alpha=0.9, zorder=2)

    # landmarks: src (circle) -> dst (diamond), connected
    for k in range(len(src_all)):
        s_rgb = _srgb_of_ab(src_all[k:k+1], lab_to_srgb=lab_to_srgb)[0]
        d_rgb = _srgb_of_ab(dst_all[k:k+1], lab_to_srgb=lab_to_srgb)[0]
        ax.plot([src_all[k, 0], dst_all[k, 0]], [src_all[k, 1], dst_all[k, 1]],
                "k-", lw=1.2, zorder=3)
        ax.scatter(*src_all[k], s=140, c=[s_rgb], edgecolors="k",
                   marker="o", zorder=4)
        ax.scatter(*dst_all[k], s=180, c=[d_rgb], edgecolors="k",
                   marker="D", zorder=4)
        ax.annotate(f"L{k}", src_all[k], fontsize=8, zorder=5)

    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlim(ab_min[0], ab_max[0]); ax.set_ylim(ab_min[1], ab_max[1])
    ax.set_xlabel("a (green - red)"); ax.set_ylabel("b (blue - yellow)")
    ax.set_aspect("equal")
    if title is None:
        title = ("ab flow: IDW + margin-pull" if include_margin_pull
                 else "ab flow: IDW only")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def save_ab_flow_compare(src_all, dst_all, srcL_all, dstL_all,
                         recolor_ab_L, lab_to_srgb, out_png,
                         final_theme_ab, assign_nearest_theme, margin_pull_adaptive,
                         power=2.0, l_strength=0.0, n=28,
                         ab_range=None, dc_ab=None, margin_base_scale=0.5):
    """Two panels: IDW-only (left) vs IDW+margin-pull (right), same range."""
    src_all = np.asarray(src_all, float); dst_all = np.asarray(dst_all, float)
    if ab_range is None:
        pts = [src_all, dst_all]
        if dc_ab is not None and len(dc_ab):
            pts.append(np.asarray(dc_ab, float))
        allp = np.concatenate(pts, 0)
        ab_range = (allp.min(0) - 15, allp.max(0) + 15)
    ab_min, ab_max = ab_range
    grid_ab, _ = _grid(ab_min, ab_max, n)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, mp, ttl in [(axes[0], False, "IDW only"),
                        (axes[1], True, "IDW + margin-pull")]:
        new_ab = compute_flow(grid_ab, src_all, dst_all, srcL_all, dstL_all,
                              recolor_ab_L, power=power, l_strength=l_strength,
                              include_margin_pull=mp, final_theme_ab=final_theme_ab,
                              margin_base_scale=margin_base_scale,
                              assign_nearest_theme=assign_nearest_theme,
                              margin_pull_adaptive=margin_pull_adaptive)
        disp = new_ab - grid_ab
        dest_rgb = _srgb_of_ab(new_ab, lab_to_srgb=lab_to_srgb)
        if dc_ab is not None and len(dc_ab):
            dc = np.asarray(dc_ab, float)
            if len(dc) > 20000:
                dc = dc[np.random.default_rng(0).choice(len(dc), 20000, replace=False)]
            ax.scatter(dc[:, 0], dc[:, 1], s=2, c="0.6", alpha=0.12,
                       linewidths=0, zorder=1)
        ax.quiver(grid_ab[:, 0], grid_ab[:, 1], disp[:, 0], disp[:, 1],
                  color=dest_rgb, angles="xy", scale_units="xy", scale=1.0,
                  width=0.003, headwidth=4, headlength=5, alpha=0.9, zorder=2)
        for k in range(len(src_all)):
            s_rgb = _srgb_of_ab(src_all[k:k+1], lab_to_srgb=lab_to_srgb)[0]
            d_rgb = _srgb_of_ab(dst_all[k:k+1], lab_to_srgb=lab_to_srgb)[0]
            ax.plot([src_all[k, 0], dst_all[k, 0]], [src_all[k, 1], dst_all[k, 1]],
                    "k-", lw=1.2, zorder=3)
            ax.scatter(*src_all[k], s=120, c=[s_rgb], edgecolors="k", marker="o", zorder=4)
            ax.scatter(*dst_all[k], s=160, c=[d_rgb], edgecolors="k", marker="D", zorder=4)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlim(ab_min[0], ab_max[0]); ax.set_ylim(ab_min[1], ab_max[1])
        ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_aspect("equal")
        ax.set_title(ttl)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png