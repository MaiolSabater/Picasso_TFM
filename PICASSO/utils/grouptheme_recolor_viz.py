"""
grouptheme_recolor_viz.py

Visualise the ab-shift (group-theme) Gaussian recolour in the space it actually
operates in: the Lab ab-plane. Because the edit is a 2D ab displacement (hue +
chroma), the ab-plane shows the whole story more clearly than a 3D RGB hull:

  * scene theme colours  -> style-aligned theme colours   (the palette mapping)
  * every Gaussian's ab  -> its recoloured ab             (the induced motion)

Two figures:
  save_ab_recolor(...)        : one ab-plane with theme arrows + a subsample of
                                per-Gaussian before->after arrows, coloured by
                                the actual sRGB of each point.
  save_ab_before_after(...)   : two ab-panels side by side (before | after),
                                each Gaussian drawn at its ab position in its
                                own colour -- shows how the cloud redistributes.

All inputs are plain numpy; nothing here imports your pipeline, so it can be
called from recolor_scene_to_style with the arrays you already have.
"""
import os
import numpy as np


def _ab_background(ax, L=70.0, res=120, extent=(-90, 90)):
    """Shade the ab-plane with the sRGB colour of each (a,b) at fixed L, so the
    plane reads as a colour wheel. Falls back silently if colour conversion
    isn't available."""
    try:
        # local, optional: only used for the backdrop
        from grouptheme import colorspace as cs
    except Exception:
        try:
            import grouptheme as _gt
            cs = _gt.colorspace
        except Exception:
            return  # no backdrop, plot still works
    lo, hi = extent
    aa, bb = np.meshgrid(np.linspace(lo, hi, res), np.linspace(lo, hi, res))
    lab = np.stack([np.full_like(aa, L), aa, bb], axis=-1).reshape(-1, 3)
    rgb = np.clip(cs.lab_to_srgb(lab), 0, 1).reshape(res, res, 3)
    ax.imshow(rgb, origin="lower", extent=(lo, hi, lo, hi),
              interpolation="bilinear", alpha=0.35, zorder=0)


def _srgb_of_ab(ab, L=65.0):
    """sRGB colours for an (M,2) array of ab values at a fixed L (for markers)."""
    try:
        from grouptheme import colorspace as cs
    except Exception:
        import grouptheme as _gt
        cs = _gt.colorspace
    lab = np.concatenate([np.full((len(ab), 1), L), np.asarray(ab)], axis=1)
    return np.clip(cs.lab_to_srgb(lab), 0, 1)


def save_ab_recolor(theme_ab, final_theme_ab, dc_ab, new_dc_ab, path,
                    dc_srgb=None, new_dc_srgb=None, style_srgb=None,
                    mapping=None, max_arrows=1500, seed=0, L_bg=70.0,
                    title="ab-shift recolour: theme mapping + Gaussian motion"):
    """One ab-plane: theme arrows (thick) + sampled per-Gaussian arrows (thin).

    Parameters
    ----------
    theme_ab       : (m,2) scene theme colours in ab (before).
    final_theme_ab : (m,2) style-aligned theme colours in ab (after).
    dc_ab          : (N,2) every Gaussian's ab before recolour.
    new_dc_ab      : (N,2) every Gaussian's ab after recolour.
    dc_srgb        : (N,3) optional true sRGB of each Gaussian (before) to colour
                     the arrow tails; if None, colours are derived from ab.
    style_srgb     : (S,3) optional style palette sRGB, plotted as ringed markers
                     so you can see the style targets themselves.
    mapping        : optional dict {theme_idx: style_idx}; if given with
                     style_srgb, theme arrowheads are ringed in their target
                     style colour.
    max_arrows     : cap on per-Gaussian arrows drawn (subsampled).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme_ab = np.asarray(theme_ab, float)
    final_theme_ab = np.asarray(final_theme_ab, float)
    dc_ab = np.asarray(dc_ab, float)
    new_dc_ab = np.asarray(new_dc_ab, float)

    # choose plane extent from data so everything is visible
    allab = np.vstack([theme_ab, final_theme_ab, dc_ab, new_dc_ab])
    m = np.abs(allab).max() * 1.1
    m = max(m, 20.0)

    fig, ax = plt.subplots(figsize=(9, 9))
    _ab_background(ax, L=L_bg, extent=(-m, m))

    # per-Gaussian motion (subsampled, thin, coloured by each point's colour)
    rng = np.random.default_rng(seed)
    N = len(dc_ab)
    idx = rng.choice(N, min(max_arrows, N), replace=False)
    if dc_srgb is not None:
        cols = np.clip(np.asarray(dc_srgb)[idx], 0, 1)
    else:
        cols = _srgb_of_ab(dc_ab[idx])
    for k, i in enumerate(idx):
        ax.annotate("", xy=new_dc_ab[i], xytext=dc_ab[i],
                    arrowprops=dict(arrowstyle="->", color=cols[k],
                                    alpha=0.35, lw=0.6), zorder=2)

    # style palette targets (ringed markers)
    if style_srgb is not None:
        try:
            from grouptheme import colorspace as cs
        except Exception:
            import grouptheme as _gt
            cs = _gt.colorspace
        st_ab = cs.srgb_to_lab(np.asarray(style_srgb))[:, 1:]
        ax.scatter(st_ab[:, 0], st_ab[:, 1], s=260, facecolors="none",
                   edgecolors="k", linewidths=2.0, zorder=4)
        ax.scatter(st_ab[:, 0], st_ab[:, 1], s=140,
                   c=np.clip(style_srgb, 0, 1), edgecolors="k",
                   linewidths=0.5, zorder=5, marker="s")

    # theme mapping arrows (thick, dark) with before/after markers
    for t in range(len(theme_ab)):
        ax.annotate("", xy=final_theme_ab[t], xytext=theme_ab[t],
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=2.2, alpha=0.9), zorder=6)
    ax.scatter(theme_ab[:, 0], theme_ab[:, 1], s=200,
               c=_srgb_of_ab(theme_ab), edgecolors="k", linewidths=1.5,
               zorder=7, label="scene theme (before)")
    ax.scatter(final_theme_ab[:, 0], final_theme_ab[:, 1], s=200,
               c=_srgb_of_ab(final_theme_ab), edgecolors="k", linewidths=1.5,
               marker="D", zorder=7, label="theme -> style (after)")
    for t in range(len(theme_ab)):
        ax.text(theme_ab[t, 0], theme_ab[t, 1], f" T{t}", fontsize=8,
                weight="bold", zorder=8)

    ax.axhline(0, color="0.5", lw=0.6, zorder=1)
    ax.axvline(0, color="0.5", lw=0.6, zorder=1)
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_aspect("equal")
    ax.set_xlabel("a  (green - red)"); ax.set_ylabel("b  (blue - yellow)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_ab_before_after(dc_ab, new_dc_ab, path, dc_srgb=None, new_dc_srgb=None,
                         theme_ab=None, final_theme_ab=None,
                         max_scatter=6000, seed=0, L_bg=70.0,
                         titles=("Before", "After"),
                         suptitle="ab-shift recolour: Gaussian cloud"):
    """Two ab-panels (before | after): every Gaussian at its ab, in its colour.

    Shows how the whole cloud redistributes in hue/chroma. Optionally overlays
    the theme markers on each side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dc_ab = np.asarray(dc_ab, float)
    new_dc_ab = np.asarray(new_dc_ab, float)
    rng = np.random.default_rng(seed)
    N = len(dc_ab)
    idx = rng.choice(N, min(max_scatter, N), replace=False)

    allab = np.vstack([dc_ab[idx], new_dc_ab[idx]])
    if theme_ab is not None:
        allab = np.vstack([allab, theme_ab, final_theme_ab])
    m = max(np.abs(allab).max() * 1.1, 20.0)

    cols_b = np.clip(np.asarray(dc_srgb)[idx], 0, 1) if dc_srgb is not None \
        else _srgb_of_ab(dc_ab[idx])
    cols_a = np.clip(np.asarray(new_dc_srgb)[idx], 0, 1) if new_dc_srgb is not None \
        else _srgb_of_ab(new_dc_ab[idx])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, ab, cols, ttl, th in (
        (axes[0], dc_ab[idx], cols_b, titles[0], theme_ab),
        (axes[1], new_dc_ab[idx], cols_a, titles[1], final_theme_ab)):
        _ab_background(ax, L=L_bg, extent=(-m, m))
        ax.scatter(ab[:, 0], ab[:, 1], c=cols, s=6, alpha=0.5, zorder=2)
        if th is not None:
            ax.scatter(th[:, 0], th[:, 1], s=180, c=_srgb_of_ab(th),
                       edgecolors="k", linewidths=1.5, zorder=5, marker="D")
            for t in range(len(th)):
                ax.text(th[t, 0], th[t, 1], f" T{t}", fontsize=8,
                        weight="bold", zorder=6)
        ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="0.5", lw=0.6)
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_aspect("equal")
        ax.set_xlabel("a  (green - red)"); ax.set_ylabel("b  (blue - yellow)")
        ax.set_title(ttl)
    fig.suptitle(suptitle, fontsize=13, weight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path