"""End-to-end convenience wrapper for the four-step pipeline."""

import sys
import time

import numpy as np

from .palette import extract_palette
from .grouptheme import optimize_group_theme, MODE_KMEANS
from .external import apply_external_theme
from .recolor import build_targets, recolor_image


def _progress(iterable, total, desc, enabled=True):
    """Yield items with a progress bar. Uses tqdm if present, else a lightweight
    fallback that prints count + elapsed + ETA on one line."""
    if not enabled:
        for x in iterable:
            yield x
        return
    try:
        from tqdm import tqdm
        for x in tqdm(iterable, total=total, desc=desc):
            yield x
        return
    except ImportError:
        pass
    # Fallback: no tqdm.
    start = time.time()
    for i, x in enumerate(iterable, 1):
        yield x
        elapsed = time.time() - start
        rate = elapsed / i
        eta = rate * (total - i)
        sys.stdout.write(
            f"\r  {desc}: {i}/{total}  "
            f"elapsed {elapsed:5.1f}s  eta {eta:5.1f}s  "
            f"({rate:.2f}s/item)")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def group_theme_recolor(images, m=5, mode=MODE_KMEANS, gamma=None, eta=None,
                        external_srgb=None, hue_tol=18.0, power=2.0,
                        use_medoid=True, tau=0.1, seed=0,
                        include=None, progress=True,
                        max_pixels=50_000, n_init=1, minibatch=False,
                        n_jobs=1):
    """Run the whole pipeline and return recolored images + intermediates.

    images        : list of sRGB arrays (H, W, 3), values in [0,1] or [0,255].
    m             : group-theme size.
    mode          : MODE_* preset (gamma/eta override it if given).
    external_srgb : optional (E, 3) external palette; triggers Sec 3.4.
    include       : optional list of indices to USE for building the theme
                    (paper's "select a subset as reference"). All images are
                    still recolored, but only these define T^G.

    Returns dict with: palettes, group, theme_ab (final), recolored (list).
    """
    def _extract(im):
        return extract_palette(im, tau=tau, seed=seed, max_pixels=max_pixels,
                               n_init=n_init, minibatch=minibatch)

    if n_jobs == 1:
        palettes = [_extract(im)
                    for im in _progress(images, len(images),
                                        "Stage 1/4 palettes", progress)]
    else:
        from joblib import Parallel, delayed
        # joblib handles the parallelism; progress is coarse (per-batch).
        if progress:
            print(f"Stage 1/4 palettes: extracting {len(images)} images "
                  f"on {n_jobs} workers...")
        palettes = Parallel(n_jobs=n_jobs)(
            delayed(_extract)(im) for im in images)
        if progress:
            print("Stage 1/4 palettes: done.")

    if progress:
        print("Stage 2/4 group-theme optimization...", flush=True)
    theme_palettes = palettes if include is None else [palettes[i] for i in include]
    group = optimize_group_theme(theme_palettes, m=m, mode=mode,
                                 gamma=gamma, eta=eta, seed=seed,
                                 use_medoid=use_medoid)
    if progress:
        print("Stage 2/4 done.", flush=True)

    final_theme_ab = group.theme_ab
    if external_srgb is not None:
        final_theme_ab = apply_external_theme(group.theme_ab, external_srgb,
                                              hue_tol=hue_tol)

    # If a subset defined the theme, we still need assignments for ALL images.
    # Re-run assignment-only against the (possibly external-adjusted) theme.
    if include is not None:
        from .grouptheme import _assign_one_palette, GroupThemeResult
        gamma_v = mode["gamma"] if gamma is None else gamma
        eta_v = mode["eta"] if eta is None else eta
        assignments = [
            _assign_one_palette(p.ab, p.weights, final_theme_ab, gamma_v, eta_v)
            for p in palettes
        ]
        group = GroupThemeResult(theme_ab=group.theme_ab, theme_L=group.theme_L,
                                 assignments=assignments, palettes=palettes)

    targets = build_targets(group, external_theme_ab=final_theme_ab)

    recolored = []
    for im, (src, dst) in _progress(list(zip(images, targets)), len(images),
                                    "Stage 4/4 recolor", progress):
        recolored.append(recolor_image(im, None, src, dst, power=power))

    return dict(palettes=palettes, group=group, theme_ab=final_theme_ab,
                targets=targets, recolored=recolored)
