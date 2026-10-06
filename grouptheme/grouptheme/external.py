"""Section 3.4 -- External Color Theme Constraints.

Given an external theme T^A (a designer's palette), align the group theme T^G
to it. Several matching strategies are available via `match_by`:

  * "hue"  (paper default): for each theme colour, find external colours within
    +/- hue_tol degrees; among those pick the most SATURATED. Unmatched theme
    colours are left unchanged (or, if force_match, snapped to nearest hue).
  * "ab"   : nearest external colour in the full ab-plane (perceptual nearest).
  * "unique": Hungarian (optimal) one-to-one assignment so NO two theme colours
    collapse onto the same external colour. Cost is hue distance by default
    (set unique_by="ab" for perceptual). If there are more theme colours than
    external colours (m > p), the best p theme colours receive DISTINCT external
    colours and the remaining (m - p) are assigned to their nearest external
    colour (reuse allowed). This avoids the collapse that "hue"/"ab" suffer when
    the scene is low-saturation / narrow-hue.

The user can always override the resulting mapping manually.
"""

import numpy as np

from . import colorspace as cs


def _cost_matrix(theme_ab, ext_ab, by="hue"):
    """(m x p) cost matrix between theme and external colours."""
    if by == "ab":
        return np.sqrt(np.sum(
            (theme_ab[:, None, :] - ext_ab[None, :, :]) ** 2, axis=2))
    theme_hue = cs.ab_hue_deg(theme_ab)
    ext_hue = cs.ab_hue_deg(ext_ab)
    return cs.hue_diff_deg(theme_hue[:, None], ext_hue[None, :])


def _unique_assignment(theme_ab, ext_ab, by="hue"):
    """Return {theme_idx: ext_idx} using optimal (Hungarian) assignment.

    m <= p : every theme colour gets a DISTINCT external colour (optimal).
    m >  p : the best p theme colours get distinct external colours; the rest go
             to their nearest external colour (reuse allowed).
    """
    from scipy.optimize import linear_sum_assignment
    m, p = len(theme_ab), len(ext_ab)
    C = _cost_matrix(theme_ab, ext_ab, by=by)
    mapping = {}
    rows, cols = linear_sum_assignment(C)
    assigned = set()
    for r, c in zip(rows, cols):
        mapping[int(r)] = int(c)
        assigned.add(int(r))
    for r in range(m):
        if r not in assigned:
            mapping[r] = int(np.argmin(C[r]))
    return mapping


def apply_external_theme(theme_ab, external_srgb, hue_tol=18.0,
                         return_mapping=False, force_match=True,
                         match_by="hue", unique_by="hue"):
    """Return a modified copy of theme_ab, aligned to an external palette.

    match_by  : "hue" (paper default), "ab" (nearest perceptual), or
                "unique" (Hungarian one-to-one, no collapse).
    force_match : for match_by="hue", snap otherwise-unmatched colours to the
                  nearest hue instead of leaving them unchanged.
    unique_by : cost used by match_by="unique": "hue" (default) or "ab".
    """
    theme_ab = np.array(theme_ab, dtype=np.float64, copy=True)
    external_srgb = np.asarray(external_srgb, dtype=np.float64)
    if external_srgb.max() > 1.0 + 1e-6:
        external_srgb = external_srgb / 255.0

    ext_lab = cs.srgb_to_lab(external_srgb)
    ext_ab = ext_lab[:, 1:]
    ext_hue = cs.ab_hue_deg(ext_ab)
    ext_sat = cs.ab_saturation(ext_ab)

    theme_hue = cs.ab_hue_deg(theme_ab)
    mapping = {}

    if match_by == "unique":
        mapping = _unique_assignment(theme_ab, ext_ab, by=unique_by)
        m, p = len(theme_ab), len(ext_ab)
        for t, pk in mapping.items():
            print(f"[external] theme[{t}] -> ext[{pk}] "
                  f"(unique/{unique_by} assignment, m={m} theme colours, "
                  f"p={p} external colours{', reuse allowed (m>p)' if m > p else ''})")
            theme_ab[t] = ext_ab[pk]
        if return_mapping:
            return theme_ab, mapping
        return theme_ab

    for t in range(len(theme_ab)):
        if match_by == "ab":
            d = np.sum((ext_ab - theme_ab[t]) ** 2, axis=1)
            pick = int(np.argmin(d))
            print(f"[external] theme[{t}] -> ext[{pick}] "
                  f"(nearest in ab-plane, dist={np.sqrt(d[pick]):.2f})")
            theme_ab[t] = ext_ab[pick]
            mapping[t] = pick
            continue

        dh = cs.hue_diff_deg(theme_hue[t], ext_hue)
        within = np.where(dh <= hue_tol)[0]
        if len(within) == 0:
            if force_match:
                pick = int(np.argmin(dh))
                print(f"[external] theme[{t}] (hue={theme_hue[t]:.1f}deg) -> ext[{pick}] "
                      f"(no external colour within {hue_tol}deg; force_match snapped to "
                      f"nearest hue, off by {dh[pick]:.1f}deg)")
                theme_ab[t] = ext_ab[pick]
                mapping[t] = pick
            else:
                print(f"[external] theme[{t}] (hue={theme_hue[t]:.1f}deg) unchanged "
                      f"(no external colour within {hue_tol}deg and force_match=False)")
            continue
        pick = within[np.argmax(ext_sat[within])]
        print(f"[external] theme[{t}] (hue={theme_hue[t]:.1f}deg) -> ext[{pick}] "
              f"({len(within)} candidate(s) within {hue_tol}deg, picked most saturated: "
              f"sat={ext_sat[pick]:.1f})")
        theme_ab[t] = ext_ab[pick]
        mapping[t] = int(pick)

    if return_mapping:
        return theme_ab, mapping
    return theme_ab