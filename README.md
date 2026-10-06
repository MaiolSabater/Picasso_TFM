# PICASSO — Palette-Guided Stylization for 3D Gaussian Splatting

PICASSO is a TFM (Master's thesis) project built on top of [StylizedGS](https://github.com/Kristen-Z/StylizedGS). It splits 3D Gaussian Splatting stylization into two explicit stages instead of leaving everything to the gradient-based style loss:

1. **Global palette recoloring** — match the scene's color palette to the style image's palette and recolor the Gaussians directly (no backprop), using a from-scratch implementation of *Group-Theme Recoloring* ([Nguyen et al., Pacific Graphics 2017](grouptheme/README.md)).
2. **Gradient-based texture stylization** — starting from the recolored scene, optimize the Gaussians with an NNFM style loss (as in StylizedGS/ARF) so brush strokes and texture detail match the style image.

Decoupling color from texture means a style's **palette** and a style's **brushwork** can be controlled, swapped, or taken from two different reference images independently.
