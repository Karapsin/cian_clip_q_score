"""Formatting shared by the paper's figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def save_figure(fig, folder: Path, number: str, *, dpi: int = 200) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"figure_{number}.pdf", dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def add_basemap(ax, path: Path) -> None:
    """Draw the saved Web Mercator basemap behind the data."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    with Image.open(path) as image:
        extent = json.loads(image.info["extent"])
        ax.imshow(image.convert("RGB"), extent=extent, interpolation="bilinear", zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_categories(data, category: str, xlabel: str, ax, *, title: str, left: bool = False) -> None:
    order = data[category].cat.categories
    groups = data.groupby(category, observed=True)["clip_polarity"].agg(["mean", "median", "count"])
    sns.boxplot(
        data=data, x=category, y="clip_polarity", order=order, ax=ax,
        palette="gist_earth", hue=category, dodge=False, legend=False,
    )
    ax.set(title=title, xlabel=xlabel, ylabel="Average Clip Polarity" if left else "")
    for i, name in enumerate(order):
        if name not in groups.index or not groups.loc[name, "count"]:
            continue
        mean, median, count = groups.loc[name]
        ax.text(i, median + 0.05, f"n={count:,.0f}", ha="center", va="center_baseline", fontsize=7)
        ax.text(i, mean, "x", ha="center", va="center_baseline", color="red", fontsize=7)
