"""Build the alternate paper tables and the cross-variant numeric export."""

import pandas as pd

from .paper_data import MARKETS
from .paper_exhibits import TABLE_VARIANTS
from .paper_models import FEATURE_LABELS
from .paper_variants import (
    SCORES, VARIANTS, fit_shared_cohorts, fit_variant, photo_responses, price_stats,
    shared_duration_table, shared_price_table, variant_numbers,
)


IMAGE_LABELS = {
    "ratio": "Aspect ratio", "resolution": "Resolution (million pixels)", "sharpness": "Sharpness (thousands)",
    "brightness": "Brightness", "brisque": "BRISQUE",
}


def image_descriptives(photos, quality_models, features):
    by_score = {}
    for feature in features:
        score = SCORES[feature]
        labels = {**IMAGE_LABELS, score.photo_value: score.raw_label}
        markets = {}
        for market, title in MARKETS.items():
            data = photo_responses(photos, quality_models, feature, market)
            stats = data[list(labels)].describe().loc[["count", "mean", "std", "min", "50%", "max"]].T
            markets[title] = stats.rename(index=labels, columns={"50%": "median"})
        by_score[score.raw_label] = pd.concat(markets, axis=1)
    return pd.concat(by_score, names=["Score", "Measure"])


def image_regressions(quality_models, features):
    by_score = {}
    for feature in features:
        score = SCORES[feature]
        markets = {}
        for market, title in MARKETS.items():
            model = quality_models[score.quality_model][market]
            rows = {}
            for term, label in IMAGE_LABELS.items():
                rows[label] = coefficient(model, term)
                rows[f"{label} (clustered SE)"] = f"({model.bse[term]:.3f})"
            rows.update({"Observations": f"{model.nobs:,.0f}", "R²": f"{model.rsquared:.3f}",
                         "F-statistic": f"{model.fvalue:.1f}"})
            markets[title] = rows
        by_score[score.raw_label] = pd.DataFrame(markets)
    return pd.concat(by_score, names=["Score", "Measure"])


def coefficient(model, term):
    p_value = model.pvalues[term]
    stars = "***" if p_value < .01 else "**" if p_value < .05 else "*" if p_value < .10 else ""
    return f"{model.params[term]:.3f}{stars}"


def price_tables(fits, features):
    ols_columns, hgb_columns = {}, {}
    for market, fit in fits.items():
        ols = fit.price["ols"]
        ols_rows, hgb_rows = {}, {}
        for feature in features:
            label = FEATURE_LABELS[feature]
            ols_rows[label] = coefficient(ols, feature)
            ols_rows[f"{label} (HC3 SE)"] = f"({ols.bse[feature]:.3f})"
        for split, data in (("Train", fit.train), ("Test", fit.test)):
            stats = price_stats(fit.price, data)
            for rows, kind in ((ols_rows, "ols"), (hgb_rows, "hgb")):
                for label, value in stats[kind].items():
                    rows[f"{split} {label}"] = f"{value:,.0f}" if label == "N" else f"{value:.3f}"
        ols_columns[MARKETS[market]] = ols_rows
        hgb_columns[MARKETS[market]] = hgb_rows
    return pd.DataFrame(ols_columns), pd.DataFrame(hgb_columns)


def duration_table(fits):
    summaries = {}
    for market, fit in fits.items():
        summary = fit.duration.groupby("status")["dur"].agg(["count", "mean", "std", "min", "median", "max"])
        summary = summary.reindex([True, False]).rename(index={True: "Closed", False: "Censored"})
        summary["count"] = summary["count"].map(lambda count: f"{count:,.0f}")
        summaries[MARKETS[market]] = summary.round(1)
    return pd.concat(summaries, names=["Market", "Status"])


def write_table(frame, path, **options):
    path.write_text(frame.to_latex(float_format="{:.3f}".format, escape=True, na_rep="", **options), encoding="utf-8")


def export_variant_tables(directory, listings, photos, quality_models, llama):
    eligible = {}
    for variant, features in VARIANTS.items():
        fits = fit_variant(listings, variant)
        eligible[variant] = fits
        ols, boosting = price_tables(fits, features)
        exhibits = {
            "03": image_descriptives(photos, quality_models, features),
            "04": image_regressions(quality_models, features), "05": ols, "06": boosting, "07": duration_table(fits),
        }
        exported_numbers = [number for number in exhibits if variant in TABLE_VARIANTS[number]]
        for number in exported_numbers:
            write_table(exhibits[number], directory / f"table_{number}_{variant}.tex")
        print(f"Saved Tables {', '.join(exported_numbers)} for variant {variant}", flush=True)

    shared = fit_shared_cohorts(eligible["B"])
    price, duration = shared_price_table(shared), shared_duration_table(shared)
    write_table(price, directory / "table_price_shared.tex", index=False)
    write_table(duration, directory / "table_duration_shared.tex", index=False)
    numbers = variant_numbers(listings, photos, quality_models, llama, eligible, shared)
    numbers.to_csv(directory / "paper_variant_numbers.csv", index=False)
    return price, duration
