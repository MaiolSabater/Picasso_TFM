"""
margin_pull.py

Post-processing for the ab-shift (group-theme) recolour. After the IDW shift,
pull each Gaussian toward its ASSIGNED theme target, with an ADAPTIVE per-theme
radius, and an "only if the field helped" guard.

Per Gaussian (anchor = its assigned theme's target, option B):
  d_old = ||old_ab - target||        # distance before the IDW shift
  d_new = ||new_ab - target||        # distance after the IDW shift

  1. field HURT (d_new > d_old): revert to old_ab  (don't trust a bad shift).
  2. field helped and already inside R_t (d_new <= R_t): keep new_ab.
  3. field helped but still outside R_t: pull along (target - new_ab) so it
     lands exactly ON the circle of radius R_t around the target.

DEBUG SWITCHES (each rule independently toggleable):
  do_revert : rule 1 -- revert points the field pushed away. OFF -> those points
              keep their (bad) IDW position instead of reverting to old_ab.
  do_clamp  : rule 3 -- clamp still-outside points onto the R_t circle. OFF ->
              those points keep their IDW position (no pull).
  (rule 2 "keep inside" has no code; it's the default when 1 and 3 don't fire.)

Turn BOTH off -> margin_pull is a pure pass-through (returns new_ab unchanged),
equivalent to "IDW only". Turn them on one at a time to isolate each effect.

Adaptive radius (per theme t):
  R_t = base_scale * median_i ||assigned_color_i_ab - target_t||
  base_scale is the single knob (smaller = tighter clamp).
"""
import numpy as np


def assign_nearest_theme(old_ab, theme_ab):
    """(N,) index of the nearest theme (in ab) to each point's ORIGINAL colour."""
    d = np.linalg.norm(old_ab[:, None, :] - theme_ab[None, :, :], axis=2)
    return d.argmin(1)


def adaptive_theme_radii(old_ab, assign, theme_ab, base_scale=0.5,
                         reduce="median", min_radius=1e-3, fixed_radius=None):
    """Per-theme radius R_t = base_scale * spread of theme t's assigned colours.

    fixed_radius : if given, ALL themes use this radius instead of the adaptive
                   one (debug switch: isolate the clamp from the adaptive sizing).
    Returns R : (T,) radius per theme. Empty themes get the global spread.
    """
    T = len(theme_ab)
    if fixed_radius is not None:
        return np.full(T, float(fixed_radius))
    R = np.zeros(T)
    dg = np.linalg.norm(old_ab - theme_ab[assign], axis=1)
    global_spread = np.median(dg) if reduce == "median" else dg.mean()
    for t in range(T):
        m = assign == t
        if m.any():
            dt = np.linalg.norm(old_ab[m] - theme_ab[t], axis=1)
            spread = np.median(dt) if reduce == "median" else dt.mean()
        else:
            spread = global_spread
        R[t] = max(base_scale * spread, min_radius)
    return R


def margin_pull_adaptive(old_ab, new_ab, theme_ab, assign=None,
                         base_scale=0.5, reduce="median", return_info=False,
                         do_revert=True, do_clamp=True, fixed_radius=None):
    """Apply the adaptive margin-pull. See module docstring for the rules.

    old_ab   : (N,2) colours BEFORE the IDW shift.
    new_ab   : (N,2) colours AFTER the IDW shift.
    theme_ab : (T,2) theme target colours (anchors to pull toward).
    assign   : (N,) optional precomputed theme index; if None, nearest to old_ab.
    base_scale : R_t = base_scale * theme spread.

    DEBUG SWITCHES:
      do_revert=False -> skip rule 1 (points the field hurt keep their IDW pos).
      do_clamp=False  -> skip rule 3 (outside points are NOT pulled to the circle).
      do_revert=False and do_clamp=False -> pure pass-through (== IDW only).
      fixed_radius    -> use one radius for all themes (bypass adaptive sizing).

    Returns out_ab (N,2)  (or (out_ab, info) if return_info).
    """

    old_ab = np.asarray(old_ab, float)
    new_ab = np.asarray(new_ab, float)
    theme_ab = np.asarray(theme_ab, float)

    if assign is None:
        assign = assign_nearest_theme(old_ab, theme_ab)

    R = adaptive_theme_radii(old_ab, assign, theme_ab, base_scale=base_scale,
                             reduce=reduce, fixed_radius=fixed_radius)
    target = theme_ab[assign]
    R_pt = R[assign]

    d_old = np.linalg.norm(old_ab - target, axis=1)
    d_new = np.linalg.norm(new_ab - target, axis=1)

    out = new_ab.copy()

    # rule 1: field HURT -> revert to old_ab  (switch: do_revert)
    hurt = d_new > d_old
    if do_revert:
        out[hurt] = old_ab[hurt]
        helped = ~hurt
    else:
        # not reverting: treat every point as eligible for the clamp
        helped = np.ones(len(new_ab), bool)

    # rule 3: still outside R -> clamp onto the circle  (switch: do_clamp)
    outside = helped & (d_new > R_pt)
    if do_clamp and outside.any():
        v = target[outside] - new_ab[outside]
        dv = np.linalg.norm(v, axis=1, keepdims=True)
        unit = v / np.maximum(dv, 1e-9)
        out[outside] = target[outside] - R_pt[outside, None] * unit

    # rule 2 (helped & inside R): already new_ab, nothing to do.

    if return_info:
        info = dict(
            assign=assign, R=R, d_old=d_old, d_new=d_new,
            n_hurt=int(hurt.sum()),
            n_reverted=int(hurt.sum()) if do_revert else 0,
            n_clamped=int(outside.sum()) if do_clamp else 0,
            n_inside=int((helped & (d_new <= R_pt)).sum()),
            frac_hurt=float(hurt.mean()),
            do_revert=do_revert, do_clamp=do_clamp,
        )
        return out, info
    return out