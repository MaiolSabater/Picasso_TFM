import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def side_by_side(paths, titles, out_path, figsize_per=4, dpi=150, suptitle=None):
    """Place images side by side, each with a caption underneath.

    paths   : list of image file paths.
    titles  : list of captions, one per image (same length as paths).
    out_path: where to save the combined figure.
    """
    assert len(paths) == len(titles), "need one title per image"
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per * n, figsize_per))
    if n == 1:
        axes = [axes]
    for ax, path, title in zip(axes, paths, titles):
        img = imageio.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(">>> wrote", out_path)


# --- usage ---
side_by_side(
    ["datasets/styles/14.jpg", "datasets/palettes/E.png", "datasets/styles/2.jpg"],
    ["Starry Night", "Palette E", "Sketch-like image"],
    "combined.png",
)