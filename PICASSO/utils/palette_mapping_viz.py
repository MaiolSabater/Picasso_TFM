"""Visualize how the scene's Group-Theme palette maps onto the style palette.

Draws a two-row swatch diagram:
  - top:    every scene theme color (theme_ab + theme_L, before matching)
  - bottom: one swatch per STYLE palette color (all of them, even unused ones,
            dimmed if unused -- see `used_style_idxs`). The big square shows
            the color that style slot ACTUALLY produces when applied: source
            L (averaged across every scene theme that maps to it -- see
            `_avg_source_L` below) + that style color's own ab. A small
            square outside (below) it shows the genuine style palette color
            (style's own L + ab), for reference.
Arrows go from each scene theme color to the style color it was matched to;
multiple arrows converge on the same bottom swatch when multiple scene colors
independently matched the same style color (possible with --gt_match_by hue,
which isn't one-to-one).

The "actual applied" reconstruction assumes recolor_space="ab" (L preserved
from the scene, only hue/chroma move -- the default and the common case).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "grouptheme"))
import grouptheme as gt


def _avg_source_L(k, mapping, theme_L):
    '''Mean scene-theme L across every theme slot matched to style color k.'''
    srcs = [t for t, kk in mapping.items() if kk == k]
    return float(np.mean([theme_L[t] for t in srcs]))


def _draw_palette_mapping_panel(ax, theme_ab, theme_L, style_srgb, mapping, title=None):
    """Draw one scene-theme -> style-palette swatch diagram into `ax`. Shared
    by save_palette_mapping (single figure) and save_palette_mapping_multi
    (several side by side, e.g. one panel per --gt_match_by scenario)."""
    theme_ab = np.asarray(theme_ab, dtype=float)
    theme_L = np.asarray(theme_L, dtype=float)
    theme_lab = np.concatenate([theme_L[:, None], theme_ab], axis=1)
    theme_srgb = np.clip(gt.colorspace.lab_to_srgb(theme_lab), 0.0, 1.0)

    style_srgb = np.clip(np.asarray(style_srgb, dtype=float), 0.0, 1.0)
    style_lab = gt.colorspace.srgb_to_lab(style_srgb)
    style_ab = style_lab[:, 1:]

    m, p = len(theme_srgb), len(style_srgb)
    swatch = 0.8
    ref = 0.35
    top_y, bot_y = 1.0, 0.0
    used_style_idxs = set(mapping.values())

    for t in range(m):
        ax.add_patch(Rectangle((t, top_y), swatch, swatch, facecolor=theme_srgb[t],
                                edgecolor='black', linewidth=1.2))

    for k in range(p):
        used = k in used_style_idxs
        alpha = 1.0 if used else 0.3

        if used:
            actual_lab = np.array([[_avg_source_L(k, mapping, theme_L), style_ab[k, 0], style_ab[k, 1]]])
            actual_srgb = np.clip(gt.colorspace.lab_to_srgb(actual_lab)[0], 0.0, 1.0)
        else:
            actual_srgb = style_srgb[k]   # nothing maps here -- nothing to blend with

        ax.add_patch(Rectangle((k, bot_y - swatch), swatch, swatch, facecolor=actual_srgb,
                                edgecolor='black', linewidth=1.2, alpha=alpha))
        # small reference square OUTSIDE (below) the main swatch: genuine style color,
        # always fully opaque -- the used/unused dimming only applies to the big swatch.
        ax.add_patch(Rectangle((k + (swatch - ref) / 2, bot_y - swatch - 0.12 - ref), ref, ref,
                                facecolor=style_srgb[k], edgecolor='black', linewidth=0.8))

    for t, k in sorted(mapping.items()):
        arrow = FancyArrowPatch(
            (t + swatch / 2, top_y), (k + swatch / 2, bot_y),
            connectionstyle="arc3,rad=0.15", arrowstyle='-|>',
            mutation_scale=14, color='black', linewidth=1.2, alpha=0.85,
            shrinkA=2, shrinkB=2)
        ax.add_patch(arrow)

    ax.set_xlim(-0.5, max(m, p) - 1 + swatch + 0.5)
    ax.set_ylim(bot_y - swatch - 0.15 - ref, top_y + swatch + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)


def save_palette_mapping(theme_ab, theme_L, style_srgb, mapping, out_path):
    """
    theme_ab   : (m, 2) scene theme colors, in Lab ab-space, *before* matching
    theme_L    : (m,) scene theme lightness -- the *source* L for each style
                  slot's "actual applied color" swatch (averaged over every
                  theme slot that maps to it -- see module docstring)
    style_srgb : (p, 3) style palette colors in sRGB, [0, 1]
    mapping    : {theme_idx -> style_idx}
    out_path   : PNG path to write
    """
    m, p = len(theme_ab), len(style_srgb)
    fig_w = max(m, p) * 1.1 + 1
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    _draw_palette_mapping_panel(ax, theme_ab, theme_L, style_srgb, mapping)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(">>> saved palette mapping plot to", out_path)


def save_palette_mapping_multi(entries, out_path):
    """Several palette-mapping panels side by side in one figure, e.g. one
    per --gt_match_by scenario for the same scene/style.

    entries : list of (label, theme_ab, theme_L, style_srgb, mapping)
    out_path : PNG path to write
    """
    n = len(entries)
    max_mp = max(max(len(theme_ab), len(style_srgb)) for _, theme_ab, _, style_srgb, _ in entries)
    panel_w = max_mp * 1.1 + 1
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n, 5.3))
    if n == 1:
        axes = [axes]

    for ax, (label, theme_ab, theme_L, style_srgb, mapping) in zip(axes, entries):
        _draw_palette_mapping_panel(ax, theme_ab, theme_L, style_srgb, mapping, title=label)

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(">>> saved combined palette mapping plot to", out_path)
