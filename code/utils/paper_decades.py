"""Asking-price premiums by construction decade on matched photo samples."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm

from .paper_data import MARKETS
from .paper_exhibits import FIGURE_TITLES, number_label
from .paper_models import GEO_TERMS, UNIT_TERMS, price_split, remove_unused_categories
from .paper_plots import save_figure
from .paper_variants import VARIANTS, listings_for_variant


DECADES = ("Pre-1950", *(f"{year}s" for year in range(1950, 2030, 10)))
REFERENCE = "2000s"
DECADE_TERM = f'C(build_decade, Treatment(reference="{REFERENCE}"))'
BASE_TERMS = (DECADE_TERM, UNIT_TERMS, "C(material) + C(ceilingHeightCapped)", GEO_TERMS, "C(district)")
SERIES = (
    ("Property and location controls", "#1B598A", "o", -0.12, ()),
    ("Same controls + interior Q", "#B34429", "s", 0.12, VARIANTS["A"]),
)
QUALITY_FEATURES = (*VARIANTS["A"], "exterior_polarity_adj", "plan_polarity_adj")
COMPARISON_SERIES = (
    ("Property and location controls", "#1B598A", "o", -0.18, ()),
    ("+ interior Q", "#B34429", "s", 0.0, VARIANTS["A"]),
    ("+ interior, exterior and floor-plan Q", "#246C58", "^", 0.18, QUALITY_FEATURES),
)


def decade_sample(interior_listings, market, photo_features=VARIANTS["A"]):
    train, _ = price_split(interior_listings, market, photo_features)
    sample = remove_unused_categories(train.dropna(subset=["buildYear"]).copy())
    years = sample["buildYear"].astype(int)
    labels = ((years // 10) * 10).astype(str) + "s"
    sample["build_decade"] = pd.Categorical(labels.mask(years.lt(1950), "Pre-1950"), categories=DECADES)
    return sample


def decade_premiums(sample, photo_features):
    terms = " + ".join([*BASE_TERMS, *photo_features])
    outcome, design = dmatrices(f"log_price ~ {terms}", sample, return_type="dataframe", NA_action="raise")
    fit = sm.OLS(outcome, design).fit(cov_type="HC3")
    intervals = fit.conf_int()
    estimates = pd.DataFrame(0.0, index=DECADES, columns=["premium", "lower", "upper"])
    for decade in DECADES:
        if decade != REFERENCE:
            term = f"{DECADE_TERM}[T.{decade}]"
            estimates.loc[decade] = 100 * np.expm1([fit.params[term], *intervals.loc[term]])
    if not np.isfinite(estimates.to_numpy()).all():
        raise ValueError("Found a nonfinite construction-decade estimate")
    return estimates


def draw_market(axis, sample, title, series):
    positions = np.arange(len(DECADES))
    reference_index = DECADES.index(REFERENCE)
    comparison_positions = np.delete(positions, reference_index)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DCE2E8", linewidth=0.65)
    axis.axhline(0, color="#687585", linewidth=0.9, linestyle=(0, (4, 3)))
    axis.scatter(reference_index, 0, marker="D", color="#687585", s=15, zorder=5)
    for _, color, marker, offset, features in series:
        estimate = decade_premiums(sample, features).drop(index=REFERENCE)
        axis.errorbar(
            comparison_positions + offset, estimate["premium"],
            yerr=[estimate["premium"] - estimate["lower"], estimate["upper"] - estimate["premium"]],
            fmt=marker, color=color, markersize=4.1, elinewidth=1, capsize=2.1, zorder=4,
        )
    axis.set_title(f"{title} (n = {len(sample):,})", loc="left", fontsize=10.3, fontweight="bold", pad=9)
    axis.set_xticks(positions, labels=DECADES)
    axis.set_xlim(-0.55, len(DECADES) - 0.45)
    axis.set_xlabel("Construction decade", labelpad=8)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", length=0, pad=6, labelsize=7.8)


def export_decade_figure(directory, listings, *, include_exterior_and_plan=False):
    if include_exterior_and_plan:
        number, series, features = "A3", COMPARISON_SERIES, QUALITY_FEATURES
        sample_label = "Shared sample with all three Q-scores"
    else:
        number, series, features = "A2", SERIES, VARIANTS["A"]
        sample_label = "Interior-photo sample"
    interior_listings = listings_for_variant(listings, "A")
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
        "text.color": "#172534", "axes.labelcolor": "#172534", "axes.edgecolor": "#91A0AF",
        "xtick.color": "#334658", "ytick.color": "#334658",
    }):
        figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), sharey=True)
        figure.subplots_adjust(left=0.09, right=0.976, bottom=0.16, top=0.71, wspace=0.14)
        title = f"Figure {number_label(number)}. {FIGURE_TITLES[number]}"
        figure.text(0.09, 0.95, title, fontsize=15, fontweight="bold", va="top")
        figure.text(
            0.09, 0.876, f"{sample_label} · Relative to the 2000s · 95% confidence intervals",
            fontsize=8.7, color="#425468", va="top",
        )
        handles = [
            Line2D([], [], marker=marker, color=color, linestyle="none", markersize=4.8, label=label)
            for label, color, marker, _, _ in series
        ]
        figure.legend(handles=handles, loc="center left", bbox_to_anchor=(0.09, 0.79),
                      borderaxespad=0, frameon=False, ncol=len(series), columnspacing=1.6, handlelength=1)
        for axis, (market, title) in zip(axes, MARKETS.items(), strict=True):
            draw_market(axis, decade_sample(interior_listings, market, features), title, series)
        axes[0].set_ylabel("Asking-price difference")
        save_figure(figure, directory, number)
