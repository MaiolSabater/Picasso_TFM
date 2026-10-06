"""
ab_3d_viz.py

3D Lab visualisation of the recolour. The ab-plane plots flatten away L
(lightness), so two Gaussians at the same (a,b) but different L look identical
there while being genuinely different colours. These functions put L on the
third axis so that difference becomes visible.

save_lab_3d          : static matplotlib 3D scatter (before and/or after).
save_lab_3d_interactive : plotly HTML you can rotate/zoom in a browser.

Both tint each point by its OWN true Lab colour (not a fixed-L approximation),
so what you see is the actual colour, lightness included.
"""
import os
import numpy as np


def _lab_to_srgb_safe(lab, lab_to_srgb):
    return np.clip(lab_to_srgb(np.asarray(lab, float)), 0, 1)


def _subsample(n, max_points, seed=0):
    if n <= max_points:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, max_points, replace=False)


def save_lab_3d(dc_L, dc_ab, new_dc_ab, lab_to_srgb, out_png,
                new_dc_L=None, max_points=15000, seed=0,
                landmarks_src=None, landmarks_dst=None,
                srcL=None, dstL=None):
    """Static 3D Lab scatter: BEFORE (left) and AFTER (right) side by side.

    dc_L (N,1) or (N,), dc_ab (N,2) : original Lab.
    new_dc_ab (N,2) : recoloured ab. new_dc_L : recoloured L (defaults to dc_L,
                      since the pipeline preserves L unless lab mode changed it).
    Points are tinted by their own Lab colour. Optional landmark src/dst arrows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    dc_L = np.asarray(dc_L, float).reshape(-1)
    dc_ab = np.asarray(dc_ab, float)
    new_dc_ab = np.asarray(new_dc_ab, float)
    newL = dc_L if new_dc_L is None else np.asarray(new_dc_L, float).reshape(-1)

    idx = _subsample(len(dc_ab), max_points, seed)
    La, ab, nab, nL = dc_L[idx], dc_ab[idx], new_dc_ab[idx], newL[idx]

    old_lab = np.column_stack([La, ab])
    new_lab = np.column_stack([nL, nab])
    old_rgb = _lab_to_srgb_safe(old_lab, lab_to_srgb)
    new_rgb = _lab_to_srgb_safe(new_lab, lab_to_srgb)

    # prepare landmark arrays + their true colours once
    have_lm = landmarks_src is not None and landmarks_dst is not None
    if have_lm:
        ls = np.asarray(landmarks_src, float); ld = np.asarray(landmarks_dst, float)
        sL = np.full(len(ls), 60.) if srcL is None else np.asarray(srcL, float).reshape(-1)
        dL = np.full(len(ld), 60.) if dstL is None else np.asarray(dstL, float).reshape(-1)
        src_rgb = _lab_to_srgb_safe(np.column_stack([sL, ls]), lab_to_srgb)
        dst_rgb = _lab_to_srgb_safe(np.column_stack([dL, ld]), lab_to_srgb)

    fig = plt.figure(figsize=(16, 8))
    for col, (A, RGB, title) in enumerate([
            (old_lab, old_rgb, "Before"), (new_lab, new_rgb, "After (recoloured)")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        # A columns are [L, a, b]; plot a (x), b (y), L (z)
        ax.scatter(A[:, 1], A[:, 2], A[:, 0], c=RGB, s=4, alpha=0.5,
                   linewidths=0, depthshade=True)
        if have_lm:
            # draw the full src->dst arrow in BOTH panels for context, plus a
            # highlighted marker: circle=source (Before), diamond=target (After),
            # each tinted by its real colour with a black edge so it stands out.
            for k in range(len(ls)):
                ax.plot([ls[k, 0], ld[k, 0]], [ls[k, 1], ld[k, 1]], [sL[k], dL[k]],
                        color="0.3", lw=1.2, alpha=0.7, zorder=8)
                # source circle
                ax.scatter([ls[k, 0]], [ls[k, 1]], [sL[k]], c=[src_rgb[k]],
                           marker="o", s=90, edgecolors="k", linewidths=1.2, zorder=11)
                # target diamond
                ax.scatter([ld[k, 0]], [ld[k, 1]], [dL[k]], c=[dst_rgb[k]],
                           marker="D", s=110, edgecolors="k", linewidths=1.2, zorder=11)
                ax.text(ld[k, 0], ld[k, 1], dL[k], f" L{k}", fontsize=7, zorder=12)
        ax.set_xlabel("a (green-red)"); ax.set_ylabel("b (blue-yellow)")
        ax.set_zlabel("L (lightness)")
        ax.set_zlim(0, 100)
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def save_lab_3d_interactive(dc_L, dc_ab, new_dc_ab, lab_to_srgb, out_html,
                            new_dc_L=None, max_points=20000, seed=0,
                            dst_ab=None, dstL=None):
    """Interactive plotly HTML: rotate/zoom the AFTER cloud in 3D Lab.

    dst_ab (T,2), dstL (T,) : the destination PALETTE colours (targets). Drawn as
    large diamond markers tinted by their true colour, so you can see where the
    palette sits inside the recoloured cloud. Falls back silently (returns None)
    if plotly isn't installed.
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        print(">>> plotly not available; skipping interactive 3D. pip install plotly")
        return None

    dc_L = np.asarray(dc_L, float).reshape(-1)
    dc_ab = np.asarray(dc_ab, float)
    new_dc_ab = np.asarray(new_dc_ab, float)
    newL = dc_L if new_dc_L is None else np.asarray(new_dc_L, float).reshape(-1)

    idx = _subsample(len(dc_ab), max_points, seed)
    nL, nab = newL[idx], new_dc_ab[idx]
    oL, oab = dc_L[idx], dc_ab[idx]

    new_rgb = _lab_to_srgb_safe(np.column_stack([nL, nab]), lab_to_srgb)
    old_rgb = _lab_to_srgb_safe(np.column_stack([oL, oab]), lab_to_srgb)

    def _cols(rgb):
        return ["rgb(%d,%d,%d)" % tuple((c * 255).astype(int)) for c in rgb]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=nab[:, 0], y=nab[:, 1], z=nL, mode="markers",
        marker=dict(size=2, color=_cols(new_rgb), opacity=0.7),
        name="After"))
    fig.add_trace(go.Scatter3d(
        x=oab[:, 0], y=oab[:, 1], z=oL, mode="markers",
        marker=dict(size=2, color=_cols(old_rgb), opacity=0.25),
        name="Before", visible="legendonly"))

    # destination palette colours (targets) as big diamonds tinted by real colour
    if dst_ab is not None:
        dst_ab = np.asarray(dst_ab, float)
        dL = np.full(len(dst_ab), 60.) if dstL is None else np.asarray(dstL, float).reshape(-1)
        dst_rgb = _lab_to_srgb_safe(np.column_stack([dL, dst_ab]), lab_to_srgb)
        fig.add_trace(go.Scatter3d(
            x=dst_ab[:, 0], y=dst_ab[:, 1], z=dL, mode="markers+text",
            marker=dict(size=9, color=_cols(dst_rgb), symbol="diamond",
                        line=dict(color="black", width=2)),
            text=[f"P{i}" for i in range(len(dst_ab))],
            textposition="top center",
            name="Palette targets"))

    fig.update_layout(
        scene=dict(xaxis_title="a (green-red)", yaxis_title="b (blue-yellow)",
                   zaxis_title="L (lightness)", zaxis=dict(range=[0, 100])),
        title="Recoloured Gaussians in 3D Lab (toggle traces in legend)",
        margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out_html)
    return out_html