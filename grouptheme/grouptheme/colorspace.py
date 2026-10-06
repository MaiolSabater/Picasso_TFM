"""Color-space conversions used throughout the pipeline.

The paper (Nguyen et al., "Group-Theme Recoloring for Multi-Image Color
Consistency", Pacific Graphics 2017) operates in CIE-Lab and, for clustering
and recoloring, uses ONLY the (a, b) chroma channels -- luminance L is left
untouched on purpose (Sec. 3.2).

All array conventions:
  - sRGB / linear RGB values are float in [0, 1], shape (..., 3)
  - Lab: L in [0, 100], a/b roughly in [-128, 127], shape (..., 3)
  - "ab" arrays are shape (..., 2)

These conversions are self-contained (no skimage dependency) so the same code
can be dropped into a 3DGS project that works in linear light.
"""

import numpy as np

# D65 reference white (sRGB standard illuminant).
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])

# sRGB <-> linear XYZ matrices (IEC 61966-2-1).
_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)


def srgb_to_linear(srgb):
    """Undo the sRGB gamma curve. Input/output float in [0, 1]."""
    srgb = np.asarray(srgb, dtype=np.float64)
    return np.where(srgb <= 0.04045, srgb / 12.92,
                    ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear):
    """Apply the sRGB gamma curve. Input/output float in [0, 1]."""
    linear = np.asarray(linear, dtype=np.float64)
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * (linear ** (1 / 2.4)) - 0.055)


def linear_rgb_to_xyz(linear):
    return linear @ _RGB2XYZ.T


def xyz_to_linear_rgb(xyz):
    return xyz @ _XYZ2RGB.T


def _f_lab(t):
    delta = 6 / 29
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)


def _f_lab_inv(t):
    delta = 6 / 29
    return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))


def xyz_to_lab(xyz):
    xyz = xyz / _WHITE_D65
    f = _f_lab(xyz)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xyz = np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], axis=-1)
    return xyz * _WHITE_D65


def srgb_to_lab(srgb):
    return xyz_to_lab(linear_rgb_to_xyz(srgb_to_linear(srgb)))


def lab_to_srgb(lab):
    return linear_to_srgb(xyz_to_linear_rgb(lab_to_xyz(lab)))


def linear_rgb_to_lab(linear):
    """For 3DGS: SH0 -> RGB is already in LINEAR light, skip the gamma undo."""
    return xyz_to_lab(linear_rgb_to_xyz(linear))


def lab_to_linear_rgb(lab):
    return np.clip(xyz_to_linear_rgb(lab_to_xyz(lab)), 0.0, 1.0)


def ab_hue_deg(ab):
    """Hue angle in degrees [0, 360) from (a, b) chroma coordinates."""
    ab = np.asarray(ab, dtype=np.float64)
    h = np.degrees(np.arctan2(ab[..., 1], ab[..., 0]))
    return np.mod(h, 360.0)


def ab_saturation(ab):
    """Chroma magnitude sqrt(a^2 + b^2) -- used as a saturation proxy."""
    ab = np.asarray(ab, dtype=np.float64)
    return np.sqrt(ab[..., 0] ** 2 + ab[..., 1] ** 2)


def hue_diff_deg(h1, h2):
    """Smallest absolute angular difference between two hues, in degrees."""
    d = np.abs(h1 - h2) % 360.0
    return np.minimum(d, 360.0 - d)
