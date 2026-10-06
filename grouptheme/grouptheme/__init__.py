"""Group-Theme Recoloring for Multi-Image Color Consistency.

Implementation of Nguyen, Price, Cohen & Brown, Pacific Graphics 2017.

Pipeline (see pipeline.py for the convenience wrapper):
  1. extract_palette            (Sec 3.2)
  2. optimize_group_theme       (Sec 3.3)
  3. apply_external_theme        (Sec 3.4, optional)
  4. build_targets + recolor    (Sec 3.5)
"""

from . import colorspace
from .palette import (Palette, extract_palette, extract_palette_from_colors,
                      choose_k_by_explained_variance)
from .grouptheme import (optimize_group_theme, GroupThemeResult,
                         MODE_KMEANS, MODE_MIN_REDUCTION,
                         MODE_ALLOW_UNASSIGNED, MODE_BOTH)
from .external import apply_external_theme
from .recolor import build_targets, recolor_ab, recolor_image
from .pipeline import group_theme_recolor

__all__ = [
    "colorspace",
    "Palette", "extract_palette", "extract_palette_from_colors",
    "choose_k_by_explained_variance",
    "optimize_group_theme", "GroupThemeResult",
    "MODE_KMEANS", "MODE_MIN_REDUCTION", "MODE_ALLOW_UNASSIGNED", "MODE_BOTH",
    "apply_external_theme",
    "build_targets", "recolor_ab", "recolor_image",
    "group_theme_recolor",
]
