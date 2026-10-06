"""
Small wrapper so the margin-pull can be toggled/commented in one place.

apply_margin_pull(...) does the Fix-A assignment (on the POST-IDW position) plus
the adaptive margin-pull, and returns the possibly-updated ab. If margin_pull is
False it is a no-op that returns new_ab unchanged. Import and call it from both
the image loop and the Gaussian path so there is a single switch.
"""
import numpy as np
from utils.margin_pull import margin_pull_adaptive, assign_nearest_theme


def apply_margin_pull(old_ab, new_ab, final_theme_ab, base_scale=0.5,
                      enabled=True, return_info=False, debug=False, tag=""):
    """Fix-A margin-pull. old_ab = pre-IDW, new_ab = post-IDW.

    enabled=False -> returns new_ab unchanged (so you can flip it off without
    removing the call). Assignment is on new_ab (post-IDW) so the pull follows
    the field's direction (Fix A)."""
    if not enabled:
        return (new_ab, None) if return_info else new_ab

    assign = assign_nearest_theme(new_ab, final_theme_ab)          # Fix A
    out, info = margin_pull_adaptive(
        old_ab, new_ab, final_theme_ab, assign=assign,
        base_scale=base_scale, return_info=True)
    if debug and info is not None:
        print(f">>> margin_pull{(' '+tag) if tag else ''}: "
              f"reverted {info['n_hurt']} ({100*info['frac_hurt']:.1f}%), "
              f"clamped {info['n_clamped']}, kept-inside {info['n_inside']}; "
              f"R per theme={info['R'].round(2)}")
    return (out, info) if return_info else out