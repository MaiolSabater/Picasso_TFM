"""Command-line runner: recolor a folder of images for colour consistency.

Usage:
  python cli.py INPUT_DIR OUTPUT_DIR [--m 5] [--mode both]
                [--external palette.png] [--hue-tol 18] [--power 2.0]
                [--include 0,2,3]

--mode is one of: kmeans, min_reduction, allow_unassigned, both
--external is an image whose distinct colours are used as the external theme
           (its own palette is extracted with the same k-selection).
--include picks which input images (by index, comma-separated) define the theme.
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grouptheme as gt

MODES = {
    "kmeans": gt.MODE_KMEANS,
    "min_reduction": gt.MODE_MIN_REDUCTION,
    "allow_unassigned": gt.MODE_ALLOW_UNASSIGNED,
    "both": gt.MODE_BOTH,
}


def load(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--mode", choices=MODES.keys(), default="both")
    ap.add_argument("--external", default=None)
    ap.add_argument("--hue-tol", type=float, default=18.0)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--include", default=None,
                    help="comma-separated indices of images defining the theme")
    ap.add_argument("--no-medoid", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=50_000,
                    help="subsample cap per image for palette extraction")
    ap.add_argument("--n-init", type=int, default=1,
                    help="k-means restarts (1 is fast; 4 is more robust)")
    ap.add_argument("--minibatch", action="store_true",
                    help="use MiniBatchKMeans (faster on big images)")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel workers for palette extraction (-1 = all cores)")
    args = ap.parse_args()

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    paths = sorted(sum([glob.glob(os.path.join(args.input_dir, e)) for e in exts], []))
    if not paths:
        print(f"No images found in {args.input_dir}")
        return
    print(f"Loaded {len(paths)} images.")
    images = [load(p) for p in paths]

    external_srgb = None
    if args.external:
        ext_img = load(args.external)
        ext_pal = gt.extract_palette(ext_img)
        external_srgb = ext_pal.srgb()
        print(f"External theme: {ext_pal.k} colours extracted.")

    include = None
    if args.include:
        include = [int(x) for x in args.include.split(",")]

    result = gt.group_theme_recolor(
        images, m=args.m, mode=MODES[args.mode],
        external_srgb=external_srgb, hue_tol=args.hue_tol,
        power=args.power, use_medoid=not args.no_medoid, include=include,
        max_pixels=args.max_pixels, n_init=args.n_init,
        minibatch=args.minibatch, n_jobs=args.n_jobs)

    os.makedirs(args.output_dir, exist_ok=True)
    for path, rec in zip(paths, result["recolored"]):
        name = os.path.splitext(os.path.basename(path))[0] + "_recolored.png"
        out = os.path.join(args.output_dir, name)
        Image.fromarray((rec * 255).astype(np.uint8)).save(out)
    print(f"Wrote {len(paths)} recolored images to {args.output_dir}")


if __name__ == "__main__":
    main()
