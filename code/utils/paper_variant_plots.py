"""Figures for the alternate photo-score specifications."""

from contextlib import ExitStack
from shutil import copyfile

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
from sklearn.inspection import partial_dependence

from .paper_data import MARKETS, load_llama_scores
from .paper_models import FEATURE_LABELS, PRICE_SCALE, PRICE_UNITS, SEED, boost_features, boosting_shap, linear_shap, ols_features
from .paper_plots import add_basemap, plot_categories
from .paper_variants import SCORES, VARIANTS, VARIANT_FIGURES, fit_variant, photo_responses, quartile_curves


SCATTER = {"alpha": 0.25, "s": 3, "rasterized": True}


def market_rows(data):
    for market, title in MARKETS.items():
        yield market, title, data.loc[data["deal_type"].eq(market)]


def dimensions(photos, feature):
    score = SCORES[feature]
    data = photos[score.photo_type].dropna(subset=["width", "height"])
    groups = (
        [(data.loc[data["unfinished"].eq(True)], "Flagged unfinished"),
         (data.loc[data["unfinished"].eq(False)], "Other scored interiors")]
        if feature == "mean_p_unfinished" else [(data, score.photo_type.capitalize())]
    )
    fig, axes = plt.subplots(len(groups), 2, figsize=(8, 3.2 * len(groups)), squeeze=False, sharex=True, sharey=True)
    for (group, label), row in zip(groups, axes):
        for ax, (_, market, images) in zip(row, market_rows(group)):
            ax.scatter(images["width"], images["height"], **SCATTER)
            ax.axvline(224, color="red", linewidth=1)
            ax.axhline(224, color="red", linewidth=1)
            ax.set_title(f"{market}\n{label} (n={len(images):,})", fontsize=9)
            ax.set(xlabel="Width (pixels)", ylabel="Height (pixels)")
            ax.grid(linestyle=":", alpha=0.7)
    return fig


def technical_quality(photos, models, feature):
    score = SCORES[feature]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), sharey="row")
    for row, (market, title) in zip(axes, MARKETS.items()):
        data = photo_responses(photos, models, feature, market)
        for ax, measure in zip(row, ("sharpness", "brisque", "brightness")):
            sns.regplot(data=data, x=measure, y=score.photo_value, ax=ax, seed=SEED,
                        scatter_kws=SCATTER, line_kws={"color": "red", "linewidth": 1})
            ax.set(xlabel="BRISQUE" if measure == "brisque" else measure.capitalize(),
                   ylabel=score.raw_label if ax is row[0] else "")
            ax.grid(linestyle=":", alpha=0.7)
        row[1].set_title(f"{title} (n={len(data):,})")
    return fig


def llama_agreement(paired, score):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, (_, title, market) in zip(axes, market_rows(paired)):
        data = market.dropna(subset=[score.raw_listing_value, "average_luxury_score"])
        correlation = data[score.raw_listing_value].corr(data["average_luxury_score"])
        sns.regplot(data=data, x=score.raw_listing_value, y="average_luxury_score", ax=ax,
                    scatter_kws=SCATTER, line_kws={"color": "red"}, seed=SEED)
        ax.set(title=f"{title} (n={len(data):,}, r={correlation:.3f})", xlabel=score.raw_label,
               ylabel="Archived LLaMA4 luxury rating" if ax is axes[0] else "")
        ax.grid(linestyle=":", alpha=0.7)
    return fig


def listing_categories(data, score, attributes, *, show_area=False):
    count = len(attributes) + int(show_area)
    fig, axes = plt.subplots(count, 2, figsize=(8, 3.1 * count), squeeze=False, sharey="row")
    for column, (_, title, market) in enumerate(market_rows(data)):
        if show_area:
            ax = axes[0, column]
            rooms = market.groupby("roomsCount_grouped", observed=True)
            colors = sns.color_palette("gist_earth", n_colors=len(rooms))
            for (room, group), color in zip(rooms, colors):
                sns.regplot(data=group, x="totalArea", y=score.raw_listing_value, ax=ax, ci=None,
                            scatter_kws={**SCATTER, "color": color}, line_kws={"color": color, "label": room})
            ax.set(title=title, xlabel="Total area (m²)", ylabel=score.raw_label, xlim=(0, 200))
            ax.legend(title="Rooms", fontsize="small")
            ax.grid(linestyle=":", alpha=0.7)
        for row, (variable, label) in enumerate(attributes, start=int(show_area)):
            plot_categories(market, variable, label, axes[row, column], title=title, left=column == 0,
                            score=score.raw_listing_value, score_label=score.raw_label)
    return fig


def construction_year(data, score):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, sharey=True)
    for ax, (_, title, market) in zip(axes, market_rows(data)):
        by_year = market.groupby("buildYearCapped", observed=False)[score.raw_listing_value].agg(["mean", "size"])
        x = np.arange(len(by_year))
        ax.plot(x, by_year["mean"], color="blue", marker="o", markersize=2, label="Mean score")
        sns.regplot(x=x, y=by_year["mean"].to_numpy(), ax=ax, lowess=True, scatter=False,
                    line_kws={"color": "blue", "linestyle": "--", "label": "LOWESS"})
        counts = ax.twinx()
        counts.bar(x, by_year["size"], color="brown", alpha=0.25, width=0.7)
        counts.set_ylabel("Scored listings")
        ticks = np.unique(np.r_[x[::4], x[-1]])
        ax.set_xticks(ticks, by_year.index[ticks], rotation=90)
        ax.set(title=title, ylabel=score.raw_label)
        ax.grid(linestyle=":", alpha=0.7)
    axes[-1].set_xlabel("Construction year")
    return fig


def districts_figure(data, score, districts, assets):
    points = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["lng"], data["lat"]), crs=4326).to_crs(districts.crs)
    matched = points.sjoin(districts, predicate="intersects")
    means = matched.groupby(["deal_type", "raion_rus"])[score.raw_listing_value].mean()
    norm = plt.Normalize(vmin=means.min(), vmax=means.max())
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), layout="constrained")
    for ax, (market, title) in zip(axes, MARKETS.items()):
        polygons = districts.join(means.loc[market], on="raion_rus").to_crs(3857)
        polygons.plot(column=score.raw_listing_value, cmap="RdYlGn", norm=norm, ax=ax,
                      edgecolor="0.8", linewidth=0.65, missing_kwds={"color": "lightgrey"})
        add_basemap(ax, assets / "moscow_positron.png")
        ax.set_title(title)
        ax.set_axis_off()
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn"), ax=axes, label=score.raw_label, shrink=0.6)
    fig.text(0.5, 0.02, "© OpenStreetMap contributors · © CARTO", ha="center", size=7)
    return fig


def asking_prices(data, feature):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, (market, title, frame) in zip(axes, market_rows(data)):
        frame = frame.loc[frame["price_last"].gt(0)]
        log_price = np.log(frame["price_last"] / PRICE_SCALE[market])
        sns.regplot(data=frame, x=feature, y=log_price, ax=ax, scatter_kws=SCATTER,
                    line_kws={"color": "red"}, seed=SEED)
        ax.set(title=f"{title} (n={len(frame):,})", xlabel=FEATURE_LABELS[feature],
               ylabel=f"Log asking price ({PRICE_UNITS[market]})")
        ax.grid(linestyle=":", alpha=0.7)
    return fig


def price_dependence(fits, feature):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, (market, title) in zip(axes, MARKETS.items()):
        fit = fits[market]
        model = fit.price
        columns = model["features"]
        test = boost_features(fit.test, columns).to_numpy()
        dependence = partial_dependence(model["hgb"], test, [columns.index(feature)], grid_resolution=100)
        grid = dependence["grid_values"][0]
        design = ols_features(model["design_info"], fit.test)
        ols_mean = model["ols"].predict(design).mean()
        ols_line = ols_mean + model["ols"].params[feature] * (grid - fit.test[feature].mean())
        hgb_line = dependence["average"][0] + fit.train["log_price"].mean()
        ax.plot(grid, ols_line, color="red", linestyle="--", label="OLS")
        ax.plot(grid, hgb_line, color="darkblue", label="Gradient boosting")
        ax.set(title=title, xlabel=FEATURE_LABELS[feature], ylabel=f"Log asking price ({PRICE_UNITS[market]})")
        ax.legend(fontsize="small")
        ax.grid(linestyle=":", alpha=0.7)
    return fig


def shap_figure(fits, features):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for column, (market, title) in enumerate(MARKETS.items()):
        fit = fits[market]
        linear = linear_shap(fit.train, fit.test, market, features)
        boosted = boosting_shap(fit.price["hgb"], fit.test, fit.price["features"])
        for row, (importance, model, color) in enumerate([
            (linear, "Linear regression", "darkorange"), (boosted, "Gradient boosting", "#e90049"),
        ]):
            ax = axes[row, column]
            keep = importance.nlargest(8).index.union(features, sort=False)
            values = importance.loc[keep].sort_values().rename(index=FEATURE_LABELS)
            ax.barh(values.index, values.to_numpy(), color=color)
            ax.set(title=f"{title} · {model}", xlabel="Mean absolute SHAP value")
            ax.grid(axis="x", linestyle=":", alpha=0.7)
            ax.set_axisbelow(True)
    return fig


def kaplan_meier(fits, feature):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, (market, title) in zip(axes, MARKETS.items()):
        for number, group, curve in quartile_curves(fits[market].duration, feature):
            ax.step(np.r_[0, curve.surv_times, group["dur"].max()],
                    np.r_[1, curve.surv_prob, curve.surv_prob[-1]], where="post",
                    label=f"Q{number} (n={len(group):,})")
        ax.set(title=title, xlabel="Days since listing", ylabel="Survival probability", ylim=(0, 1.03))
        ax.legend(title=FEATURE_LABELS[feature], fontsize="small", title_fontsize="small")
        ax.grid(linestyle=":", alpha=0.7)
    return fig


def cox_figure(fits, features):
    fig, axes = plt.subplots(len(features) + 1, 2, figsize=(10, 2.65 * (len(features) + 1)), squeeze=False)
    terms = [*features, "np.log(price_first)", "np.log(totalArea)"]
    labels = [*(FEATURE_LABELS[feature] for feature in features), "Log initial price", "Log area"]
    for column, (market, title) in enumerate(MARKETS.items()):
        fit = fits[market]
        ax = axes[0, column]
        summary = fit.cox.summary.loc[terms]
        ratios = summary["exp(coef)"].to_numpy()
        lower = summary["exp(coef) lower 95%"].to_numpy()
        upper = summary["exp(coef) upper 95%"].to_numpy()
        positions = np.arange(len(terms))[::-1]
        ax.errorbar(ratios, positions, xerr=[ratios - lower, upper - ratios], fmt="s", capsize=3, color="black")
        ax.axvline(1, color="grey", linestyle="--")
        ax.set_yticks(positions, labels)
        ax.set(title=title, xlabel="Hazard ratio per point (95% CI)")
        for row, feature in enumerate(features, start=1):
            ax = axes[row, column]
            levels = np.unique(fit.duration[feature].quantile([0.10, 0.50, 0.90]).to_numpy())
            fit.cox.plot_partial_effects_on_outcome(feature, values=levels, cmap="viridis", ax=ax, plot_baseline=False)
            ax.legend([f"{level:.3g}" for level in levels], title=FEATURE_LABELS[feature], fontsize="small", title_fontsize="small")
            ax.set(xlabel="Days since listing", ylabel="Survival probability")
            ax.grid(linestyle=":", alpha=0.7)
    return fig


def distributions(data, score):
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex="row")
    for column, (_, title, market) in enumerate(market_rows(data)):
        axes[0, column].set_title(title)
        for row, (feature, label) in enumerate([
            ("average_luxury_score", "Archived LLaMA4 luxury rating"), (score.raw_listing_value, score.raw_label),
        ]):
            ax = axes[row, column]
            values = market[feature].dropna()
            sns.histplot(x=values, bins=12, stat="proportion", ax=ax, edgecolor="lightgrey")
            mean, sd = values.mean(), values.std()
            ax.axvline(mean, color="black", linewidth=1, label=f"Mean {mean:.3g}")
            for bound in (mean - 2 * sd, mean + 2 * sd):
                ax.axvline(bound, color="black", linestyle="--", linewidth=1)
            ax.set(xlabel=label, ylabel="Share of listings")
            ax.legend(title=f"n={len(values):,}", fontsize="small", title_fontsize="small")
    return fig


def write_page(pdf, figure, title):
    figure.suptitle(title, fontsize=11)
    if figure.get_layout_engine() is None:
        figure.tight_layout()
    for ax in figure.axes:
        ax.tick_params(labelsize=8)
    pdf.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close(figure)


def score_figure_title(number, variant, feature):
    score = SCORES[feature]
    if number == "05":
        label = (
            "Dimensions by unfinished flag" if feature == "mean_p_unfinished"
            else f"{score.photo_type.capitalize()} photo dimensions"
        )
    else:
        label = FEATURE_LABELS[feature] if number in ("13", "14", "16") else score.raw_label
    return f"Figure {number}{variant} · {label}"


def export_variant_figures(directory, inputs, listings, photos, quality_models):
    for number in VARIANT_FIGURES:
        copyfile(directory / f"figure_{number}.pdf", directory / f"figure_{number}_A.pdf")
    districts = gpd.read_parquet(inputs / "moscow_administrative_districts.parquet", columns=["raion_rus", "geometry"])
    districts = districts.dissolve(by="raion_rus", as_index=False)
    assets = inputs / "figure_assets"
    for variant, features in VARIANTS.items():
        if variant == "A":
            continue
        fits = fit_variant(listings, features)
        data = listings.dropna(subset=list(features))
        paired = load_llama_scores(inputs / "llama4_scores.parquet", data)
        with ExitStack() as stack:
            pages = {
                number: stack.enter_context(PdfPages(directory / f"figure_{number}_{variant}.pdf"))
                for number in VARIANT_FIGURES
            }
            for feature in features:
                score = SCORES[feature]
                builders = {
                    "05": (dimensions, photos, feature),
                    "06": (technical_quality, photos, quality_models, feature),
                    "07": (llama_agreement, paired, score),
                    "08": (listing_categories, data, score,
                           [("roomsCount_grouped", "Rooms"), ("bathrooms_cat", "Bathrooms")]),
                    "09": (listing_categories, data, score,
                           [("floorNumberCapped", "Floor"), ("ceilingHeightCapped", "Ceiling height"), ("repairType", "Repair")]),
                    "10": (construction_year, data, score),
                    "11": (listing_categories, data, score, [("toDemolish", "Scheduled demolition")]),
                    "12": (districts_figure, data, score, districts, assets),
                    "13": (asking_prices, data, feature),
                    "14": (price_dependence, fits, feature),
                    "16": (kaplan_meier, fits, feature),
                    "A1": (distributions, paired, score),
                }
                for number, (build, *arguments) in builders.items():
                    figure = build(*arguments, show_area=True) if number == "08" else build(*arguments)
                    write_page(pages[number], figure, score_figure_title(number, variant, feature))
            write_page(pages["15"], shap_figure(fits, features), f"Figure 15{variant} · Feature importance")
            write_page(pages["17"], cox_figure(fits, features), f"Figure 17{variant} · Proportional hazards")
        print(f"Saved Figures 05–17 and A1 for variant {variant}", flush=True)
