"""Score definitions and eligible and shared cohorts for the paper exhibits."""

from dataclasses import dataclass

import pandas as pd
from statsmodels.duration.survfunc import SurvfuncRight

from .paper_data import MARKETS, photo_count_categories
from .paper_models import (
    FEATURE_LABELS, boost_features, fit_price_specification, model_performance, ols_features, price_split,
)
from .paper_offers import PROPERTY_KEY
from .paper_survival import duration_sample, fit_cox_model


@dataclass(frozen=True)
class Score:
    photo_type: str
    photo_value: str
    raw_listing_value: str
    raw_label: str
    quality_model: str


SCORES = {
    "clip_polarity_adj": Score("interior", "clip_polarity", "clip_polarity", "Interior polarity", "interior"),
    "exterior_polarity_adj": Score("exterior", "clip_polarity", "exterior_polarity", "Exterior polarity", "exterior"),
    "plan_polarity_adj": Score("plan", "clip_polarity", "plan_polarity", "Floor-plan polarity", "plan"),
    "mean_p_unfinished": Score(
        "interior", "p_unfinished", "mean_p_unfinished", "Unfinished probability", "unfinished",
    ),
}
ALL_SCORES = tuple(SCORES)
VARIANTS = {
    "A": ("clip_polarity_adj",),
    "B": ALL_SCORES,
}


@dataclass
class MarketFit:
    train: pd.DataFrame
    test: pd.DataFrame
    price: dict
    duration: pd.DataFrame
    cox: object


def listings_for_variant(listings: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "A":
        counts = listings["interior_photo_count"]
        return listings.assign(link_num=counts, link_num_categ=photo_count_categories(counts))
    return listings


def fit_variant(listings: pd.DataFrame, variant: str) -> dict:
    listings = listings_for_variant(listings, variant)
    features = VARIANTS[variant]
    fits = {}
    for market in MARKETS:
        train, test = price_split(listings, market, features)
        duration = duration_sample(listings, market, features)
        fits[market] = MarketFit(
            train, test, fit_price_specification(train, market, features), duration, fit_cox_model(duration, features),
        )
    return fits


def fit_shared_cohorts(combined: dict) -> dict:
    """Refit the interior model on the combined model's existing samples."""
    interior = {}
    features = VARIANTS["A"]
    for market, fit in combined.items():
        cohort = listings_for_variant(pd.concat([fit.train, fit.test]), "A")
        cohort["link_num_categ"] = cohort["link_num_categ"].cat.remove_unused_categories()
        train, test = cohort.loc[fit.train.index], cohort.loc[fit.test.index]
        duration = listings_for_variant(fit.duration, "A")
        interior[market] = MarketFit(
            train, test, fit_price_specification(train, market, features), duration,
            fit_cox_model(duration, features),
        )
    return {"A": interior, "B": combined}


def price_stats(models: dict, data: pd.DataFrame) -> dict:
    target = data["log_price"]
    ols, boosting = models["ols"], models["hgb"]
    design = ols_features(models["design_info"], data)
    values = boost_features(data, models["features"]).to_numpy()
    return {
        "ols": model_performance(target, ols.predict(design), int(ols.df_model)),
        "hgb": model_performance(target, boosting.predict(values), len(models["features"])),
    }


def quartile_curves(frame: pd.DataFrame, feature: str):
    bins = pd.qcut(frame[feature], 4, labels=False, duplicates="drop")
    for number, group in frame.groupby(bins):
        yield int(number) + 1, group, SurvfuncRight(group["dur"], group["status"].astype(int))


def photo_responses(photos: dict, quality_models: dict, feature: str, market: str) -> pd.DataFrame:
    score = SCORES[feature]
    model = quality_models[score.quality_model][market]
    return photos[score.photo_type].loc[model.model.data.row_labels]


def shared_price_table(shared: dict) -> pd.DataFrame:
    rows = []
    for variant, fits in shared.items():
        for market, fit in fits.items():
            metrics = price_stats(fit.price, fit.test)
            rows.append({
                "Market": MARKETS[market], "Variant": variant, "Train N": len(fit.train), "Test N": len(fit.test),
                "OLS adjusted R²": metrics["ols"]["Adjusted R²"], "OLS RMSE": metrics["ols"]["RMSE"],
                "Boosting adjusted R²": metrics["hgb"]["Adjusted R²"], "Boosting RMSE": metrics["hgb"]["RMSE"],
            })
    return pd.DataFrame(rows).sort_values("Market", kind="stable")


def shared_duration_table(shared: dict) -> pd.DataFrame:
    rows = []
    for variant, fits in shared.items():
        for market, fit in fits.items():
            for feature in VARIANTS[variant]:
                row = fit.cox.summary.loc[feature]
                rows.append({
                    "Market": MARKETS[market], "Variant": variant, "Score": FEATURE_LABELS[feature],
                    "N": len(fit.duration), "Closed": int(fit.duration["status"].sum()),
                    "Censored": int(fit.duration["status"].eq(False).sum()), "Concordance": fit.cox.concordance_index_,
                    "HR (+1 point)": row["exp(coef)"],
                    "Lower 95%": row["exp(coef) lower 95%"],
                    "Upper 95%": row["exp(coef) upper 95%"],
                })
    return pd.DataFrame(rows).sort_values("Market", kind="stable")


def variant_numbers(listings, photos, quality_models, llama, eligible, shared) -> pd.DataFrame:
    rows = []
    for cohort, all_fits in (("eligible", eligible), ("shared", shared)):
        for variant, fits in all_fits.items():
            features = VARIANTS[variant]
            for market, fit in fits.items():
                def record(model, values, feature=""):
                    rows.extend((cohort, variant, market, model, feature, key, value) for key, value in values.items())

                for split, data in (("train", fit.train), ("test", fit.test)):
                    for model, stats in price_stats(fit.price, data).items():
                        record(model, {
                            f"{split}_n": stats["N"], f"{split}_adjusted_r2": stats["Adjusted R²"],
                            f"{split}_rmse": stats["RMSE"],
                        })

                events = fit.duration["status"]
                record("cox", {
                    "n": len(events), "closed": events.sum(), "censored": events.eq(False).sum(),
                    "concordance": fit.cox.concordance_index_,
                    "median_closed_days": fit.duration.loc[events, "dur"].median(),
                })
                for feature in features:
                    summary = fit.cox.summary.loc[feature]
                    record("cox", {
                        "hazard_ratio": summary["exp(coef)"],
                        "lower_95": summary["exp(coef) lower 95%"],
                        "upper_95": summary["exp(coef) upper 95%"],
                    }, feature)
                    ols = fit.price["ols"]
                    record("ols", {"coefficient": ols.params[feature], "hc3_se": ols.bse[feature]}, feature)
                    for quartile, group, curve in quartile_curves(fit.duration, feature):
                        record("kaplan_meier", {
                            f"q{quartile}_n": len(group), f"q{quartile}_median_days": curve.quantile(0.5),
                        }, feature)

                if cohort != "eligible":
                    continue
                market_listings = listings.loc[listings["deal_type"].eq(market)]
                scored = market_listings.dropna(subset=list(features))
                record("coverage", {"eligible_listings": len(scored)})
                for feature in features:
                    score = SCORES[feature]
                    data = photos[score.photo_type]
                    model = quality_models[score.quality_model][market]
                    photos_with_score = data.loc[data["deal_type"].eq(market), score.photo_value].dropna()
                    record("coverage", {
                        "scored_photos": len(photos_with_score), "photos_in_quality_regression": model.nobs,
                        "raw_photo_mean": photos_with_score.mean(), "raw_listing_mean": scored[score.raw_listing_value].mean(),
                        "quality_regression_r2": model.rsquared,
                    }, feature)
                    if llama is not None:
                        columns = PROPERTY_KEY + ["average_luxury_score"]
                        pairs = scored.merge(llama[columns], on=PROPERTY_KEY, validate="one_to_one")
                        pairs = pairs.dropna(subset=[score.raw_listing_value, "average_luxury_score"])
                        record("llama4", {
                            "paired_listings": len(pairs),
                            "pearson_r": pairs[score.raw_listing_value].corr(pairs["average_luxury_score"]),
                        }, feature)
    return pd.DataFrame(rows, columns=["cohort", "variant", "deal_type", "model", "feature", "metric", "value"])
