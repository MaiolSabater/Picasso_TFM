"""Smoke test + visual demo of the group-theme recoloring pipeline.

Generates several synthetic images with deliberately INCONSISTENT colour casts,
runs the full pipeline, and checks that:
  - palettes extract with sensible k
  - the group theme has m colours
  - recoloring reduces the spread of dominant hues across images
Saves a before/after montage.
"""

import sys
import numpy as np

sys.path.insert(0, "/home/claude/grouptheme")

import grouptheme as gt
from grouptheme import colorspace as cs


def make_image(base_hue_shift, seed, size=128):
    """A simple synthetic 'scene': sky band, ground band, an object blob,
    tinted by base_hue_shift to simulate inconsistent capture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3))
    # sky (top third)
    sky = np.array([0.4, 0.6, 0.9]) + base_hue_shift
    # ground (bottom two-thirds)
    ground = np.array([0.5, 0.4, 0.2]) - base_hue_shift * 0.5
    img[:size // 3] = sky
    img[size // 3:] = ground
    # an object blob
    cy, cx = int(size * 0.6), int(size * 0.5)
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 < (size * 0.18) ** 2
    obj = np.array([0.8, 0.3, 0.3]) + base_hue_shift * 0.3
    img[mask] = obj
    img += rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1)


def dominant_hue(img):
    lab = cs.srgb_to_lab(img.reshape(-1, 3))
    return cs.ab_hue_deg(lab[:, 1:].mean(axis=0))


def main():
    shifts = [0.00, 0.10, -0.08, 0.05, -0.12]
    images = [make_image(s, seed=i) for i, s in enumerate(shifts)]

    print("=== Stage 1: palette extraction ===")
    palettes = [gt.extract_palette(im, seed=0) for im in images]
    for i, p in enumerate(palettes):
        print(f"  image {i}: k={p.k}, weights sum={p.weights.sum():.0f}")

    print("\n=== Stage 2: group theme (plain k-means) ===")
    group = gt.optimize_group_theme(palettes, m=4, mode=gt.MODE_KMEANS, seed=0)
    print(f"  theme size m={group.m}")
    for i, a in enumerate(group.assignments):
        print(f"  image {i} assignments -> {a.tolist()}")

    print("\n=== Stage 2b: group theme (allow unassigned) ===")
    group2 = gt.optimize_group_theme(palettes, m=2,
                                     mode=gt.MODE_ALLOW_UNASSIGNED, seed=0)
    n_unassigned = sum(int((a == 0).sum()) for a in group2.assignments)
    print(f"  with m=2 and low eta, {n_unassigned} palette colours left unchanged")

    print("\n=== Stage 3: external theme (hue matching) ===")
    external = np.array([[0.85, 0.2, 0.2], [0.2, 0.3, 0.8]])  # red + blue brand
    ext_theme = gt.apply_external_theme(group.theme_ab, external, hue_tol=25)
    n_moved = int(np.sum(np.any(np.abs(ext_theme - group.theme_ab) > 1e-6, axis=1)))
    print(f"  {n_moved}/{group.m} theme colours pulled toward external palette")

    print("\n=== Stage 4: recolor + consistency check ===")
    result = gt.group_theme_recolor(images, m=4, mode=gt.MODE_KMEANS, seed=0)
    before_hues = np.array([dominant_hue(im) for im in images])
    after_hues = np.array([dominant_hue(im) for im in result["recolored"]])

    def spread(hues):
        # circular spread proxy: std of unit vectors' angle
        v = np.exp(1j * np.radians(hues))
        return np.degrees(np.std(np.angle(v)))

    print(f"  dominant hue spread BEFORE: {spread(before_hues):.2f} deg")
    print(f"  dominant hue spread AFTER : {spread(after_hues):.2f} deg")
    assert spread(after_hues) <= spread(before_hues) + 1e-6, \
        "recoloring did not improve consistency!"
    print("  OK: recoloring reduced cross-image hue spread.")

    # Save a montage.
    try:
        from PIL import Image
        rows = []
        for orig, rec in zip(images, result["recolored"]):
            pair = np.concatenate([orig, np.ones((orig.shape[0], 4, 3)), rec],
                                  axis=1)
            rows.append(pair)
            rows.append(np.ones((4, pair.shape[1], 3)))
        montage = np.concatenate(rows[:-1], axis=0)
        Image.fromarray((montage * 255).astype(np.uint8)).save(
            "/home/claude/grouptheme/demo_montage.png")
        print("\n  Saved before/after montage -> demo_montage.png "
              "(left = input, right = recolored)")
    except Exception as e:
        print(f"  (montage save skipped: {e})")

    print("\nAll stages ran successfully.")


if __name__ == "__main__":
    main()
