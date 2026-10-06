"""Section 3.2 -- Individual Color Palette Extraction.

For each input image we cluster its pixel colors in the (a, b) chroma plane of
Lab space and pick the number of clusters k adaptively using the
"explained variance" criterion described in the paper:

  * total distortion v_t = sum of squared distances of every point to the
    global mean color.
  * for k = 2..7, run k-means; within-group distortion v_w = sum of squared
    distances of each point to its assigned cluster centre.
  * stop at the first k whose ratio v_w / v_t < tau (paper uses tau = 0.1).

Each palette colour also carries a histogram weight w_j = number of pixels
assigned to it (used later as the weighted k-means weight).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from . import colorspace as cs


@dataclass
class Palette:
    """A single image's palette, all in the (a, b) plane.

    ab       : (k, 2) chroma coordinates of the palette colours
    weights  : (k,)   pixel counts per palette colour (the histogram w^i)
    L        : (k,)   mean L (lightness) per cluster, kept so we can rebuild a
                       full Lab colour for display; recoloring never changes L.
    """
    ab: np.ndarray
    weights: np.ndarray
    L: np.ndarray

    @property
    def k(self):
        return self.ab.shape[0]

    def lab(self):
        return np.concatenate([self.L[:, None], self.ab], axis=1)

    def srgb(self):
        return cs.lab_to_srgb(self.lab())


def _kmeans_ab(ab, k, weights=None, seed=0, n_init=1, minibatch=False):
    if minibatch:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, n_init=n_init, random_state=seed,
                             batch_size=4096)
    else:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = km.fit_predict(ab, sample_weight=weights)
    centers = km.cluster_centers_
    return labels, centers


def choose_k_by_explained_variance(ab, weights=None, k_min=5, k_max=5,
                                   tau=0.1, seed=0, n_init=1, minibatch=False):
    """Return (k, labels, centers) for the first k with v_w / v_t < tau."""
    if weights is None:
        weights = np.ones(len(ab))

    mean_color = np.average(ab, axis=0, weights=weights)
    v_t = np.sum(weights * np.sum((ab - mean_color) ** 2, axis=1))
    if v_t <= 1e-12:  # degenerate: a single flat colour
        return 1, np.zeros(len(ab), dtype=int), mean_color[None, :]

    best = None
    for k in range(k_min, k_max + 1):
        if k >= len(np.unique(ab, axis=0)):
            k = max(1, len(np.unique(ab, axis=0)))
        labels, centers = _kmeans_ab(ab, k, weights, seed, n_init, minibatch)
        v_w = np.sum(weights * np.sum((ab - centers[labels]) ** 2, axis=1))
        ratio = v_w / v_t
        best = (k, labels, centers)
        if ratio < tau:
            break
    return best


def extract_palette(srgb_image, tau=0.1, k_min=5, k_max=5,
                    max_pixels=50_000, seed=0, n_init=1, minibatch=False):
    """Extract a Palette from an sRGB image array of shape (H, W, 3), [0, 1].

    max_pixels subsamples very large images for speed; weights still reflect
    the subsample proportionally, which is what the clustering cares about.
    n_init / minibatch trade a little clustering robustness for large speedups
    (n_init=1 is ~4x faster than 4; 50k pixels is plenty for a few clusters).
    """
    srgb_image = np.asarray(srgb_image, dtype=np.float64)
    if srgb_image.max() > 1.0 + 1e-6:
        srgb_image = srgb_image / 255.0
    pixels = srgb_image.reshape(-1, 3)

    if len(pixels) > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pixels), size=max_pixels, replace=False)
        pixels = pixels[idx]

    lab = cs.srgb_to_lab(pixels)
    ab = lab[:, 1:]
    L = lab[:, 0]

    k, labels, centers = choose_k_by_explained_variance(
        ab, k_min=k_min, k_max=k_max, tau=tau, seed=seed,
        n_init=n_init, minibatch=minibatch)

    counts = np.bincount(labels, minlength=len(centers)).astype(np.float64)
    Lmean = np.array([L[labels == j].mean() if counts[j] > 0 else 50.0
                      for j in range(len(centers))])

    keep = counts > 0
    return Palette(ab=centers[keep], weights=counts[keep], L=Lmean[keep])


def extract_palette_from_colors(ab_colors, weights=None, L=None):
    """Build a Palette directly from a set of (a,b) colours + weights.

    Handy for the 3DGS case where the 'pixels' are Gaussian SH0 colours rather
    than an image. L defaults to a neutral 50 if not supplied.
    """
    ab_colors = np.asarray(ab_colors, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(ab_colors))
    if L is None:
        L = np.full(len(ab_colors), 50.0)
    k, labels, centers = choose_k_by_explained_variance(ab_colors, weights)
    counts = np.zeros(len(centers))
    Lmean = np.zeros(len(centers))
    for j in range(len(centers)):
        m = labels == j
        counts[j] = weights[m].sum()
        Lmean[j] = np.average(L[m]) if m.any() else 50.0
    keep = counts > 0
    return Palette(ab=centers[keep], weights=counts[keep], L=Lmean[keep])
