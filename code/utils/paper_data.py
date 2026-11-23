"""Read the saved listing and photo data for the paper notebooks."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import softmax

DATASETS_DIR = Path("inputs") / "datasets"

MARKETS = {
    "long_rent": "Long-term Rentals",
    "sale_secondary": "Secondary Sales",
}
LISTING_FILES = {
    "long_rent": "lr_clean_v4.parquet",
    "sale_secondary": "ss_clean_v4.parquet",
}
LISTING_COLUMNS = [
    "property_id", "author_type", "district", "lat", "lng", "date0", "date1", "status",
    "is_censored", "dur", "price_first", "price_last", "totalArea", "roomsCount",
    "bathrooms_num", "ceilingHeight", "floorNumber", "buildYear", "houseMaterialType",
    "materialType", "repairType", "isApartments", "demolishedInMoscowProgramm",
]
DISTANCES = {
    "distance_to_center_meters": "distance_center",
    "nearest_subway_distance_meters": "distance_subway",
    "nearest_mcd_distance_meters": "distance_mcd",
    "closest_waste_distance_meters": "distance_waste",
    "closest_energy_distance_meters": "distance_energy",
    "closest_industrial_area_distance_meters": "distance_industrial",
    "closest_water_distance_meters": "distance_water",
    "closest_green_distance_meters": "distance_green",
}

PROPERTY_KEY = ["deal_type", "property_id"]
PHOTO_KEY = PROPERTY_KEY + ["offer_id", "photo_name"]
PHOTO_BINS = ["0", "1-10", "11-20", "21-30", "31-40", "41+"]
AUTHOR_TYPES = {"owner": "Owner", "specialist": "Realtor", "": "Realtor",
                "agency": "Agency", "managementCompany": "Agency", "rentDepartment": "Agency"}

# openai/clip-vit-base-patch32, saved checkpoint logit_scale.exp().
CLIP_SCALE = np.exp(4.605170249938965)
UNFINISHED_CUTOFF = 0.90
INTERIOR_SCORES = [
    "recommended_pos_similarity", "recommended_mid_similarity",
    "recommended_neg_similarity", "recommended_unfinished_similarity",
]
TIER_SCORES = ["low_similarity", "mid_similarity", "high_similarity"]
QUALITY_COLUMNS = ["ratio", "resolution", "brightness", "brisque", "sharpness"]
QUALITY_FORMULA = "clip_polarity ~ ratio + resolution + brightness + brisque + sharpness"


def load_listings(inputs_dir: Path) -> pd.DataFrame:
    listings = pd.concat(
        [pd.read_parquet(inputs_dir / name, columns=LISTING_COLUMNS).assign(deal_type=market)
         for market, name in LISTING_FILES.items()],
        ignore_index=True,
    )
    geo = pd.read_parquet(inputs_dir / "geo_features.parquet", columns=["lat", "lng", *DISTANCES])
    geo = geo.rename(columns=DISTANCES)
    geo[list(DISTANCES.values())] /= 100
    geo["distance_ugly"] = geo[["distance_waste", "distance_energy", "distance_industrial"]].min(axis=1)
    geo["distance_nice"] = geo[["distance_water", "distance_green"]].min(axis=1)
    geo["distance_metro"] = geo[["distance_subway", "distance_mcd"]].min(axis=1)
    listings = listings.merge(geo, on=["lat", "lng"], how="left", validate="many_to_one")
    return recode_listings(listings)


def recode_listings(listings: pd.DataFrame) -> pd.DataFrame:
    # The original notebook corrects one rental reported at 1 square metre.
    area_error = listings["deal_type"].eq("long_rent") & listings["totalArea"].eq(1)
    listings.loc[area_error, "totalArea"] = 15
    rooms = listings["roomsCount"].astype("Int64")
    room_labels = rooms.astype("string").mask(rooms.ge(4).fillna(False), "4+")
    listings["roomsCount_grouped"] = pd.Categorical(
        room_labels, categories=["1", "2", "3", "4+"], ordered=True,
    )
    listings["bathrooms_cat"] = pd.Categorical(
        np.where(listings["bathrooms_num"].ge(2).fillna(False), "2+", "1"),
        categories=["1", "2+"], ordered=True,
    )
    for source, target in [("demolishedInMoscowProgramm", "toDemolish"), ("isApartments", "apartment")]:
        yes = listings[source].astype("string").str.lower().eq("true").fillna(False)
        listings[target] = pd.Categorical(np.where(yes, "yes", "no"), categories=["no", "yes"])

    repair = listings["repairType"].astype("string").map(
        {"design": "Designer", "euro": "Euro", "cosmetic": "Cosmetic", "no": "None"},
    )
    listings["repairType"] = pd.Categorical(
        repair.fillna("NR"), categories=["Designer", "Euro", "Cosmetic", "None", "NR"], ordered=True,
    )
    listings["floorNumberCapped"] = pd.cut(
        pd.to_numeric(listings["floorNumber"]), [-np.inf, 1, 9, 19, 29, np.inf],
        labels=["1st", "2-9", "10-19", "20-29", "30+"],
    )
    height = pd.to_numeric(listings["ceilingHeight"]).astype(float).clip(lower=2.4)
    centimetres = height.ge(100)
    decimetres = height.gt(20) & height.lt(100)
    height.loc[centimetres] /= 100
    height.loc[decimetres] /= 10
    height_bins = pd.cut(height, [2.4, 2.7, 3, np.inf], right=False, labels=["2.4-2.7", "2.7-3.0", "3.0+"])
    listings["ceilingHeightCapped"] = height_bins.where(height.le(20)).cat.add_categories("NR").fillna("NR")

    year = listings["buildYear"].astype("Int64")
    year_labels = year.clip(upper=2025).astype("string").mask(year.le(1955).fillna(False), "pre-55")
    years = ["pre-55", *map(str, range(1956, 2026)), "NR"]
    listings["buildYearCapped"] = pd.Categorical(year_labels.fillna("NR"), categories=years, ordered=True)
    listings["age"] = (2025 - year).clip(lower=0, upper=100).fillna(0).astype(int)
    listings["ageNA"] = year.isna().astype(int)

    material = listings["houseMaterialType"].astype("string").replace(
        {"stalin": "brick", "monolith": "concrete", "monolithBrick": "concrete",
         "": "NR", "none": "NR", "wood": "NR"},
    ).fillna("NR")
    inferred_material = listings["materialType"].astype("string").replace(
        {"aerocreteBlock": "block", "foamConcreteBlock": "block", "monolith": "concrete",
         "monolithBrick": "concrete", "old": "brick", "stalin": "brick", "wood": "NR", "": "NR"},
    )
    materials = ["brick", "block", "panel", "concrete", "NR"]
    material = material.mask(material.eq("NR"), inferred_material)
    listings["material"] = pd.Categorical(material.where(material.isin(materials), "NR"), categories=materials)
    listings["author_cat"] = listings["author_type"].astype("string").map(AUTHOR_TYPES)
    listings["district"] = listings["district"].astype("category")
    return listings


def load_interiors(photo_dir: Path, interior_keys: pd.DataFrame) -> pd.DataFrame:
    photos = pd.read_parquet(photo_dir / "interior_photo_scores.parquet", columns=PHOTO_KEY + INTERIOR_SCORES)
    stats = pd.read_parquet(photo_dir / "interior_photo_stats.parquet")
    photos = photos.merge(interior_keys, on=PHOTO_KEY, validate="one_to_one")
    photos = photos.merge(stats, on=PHOTO_KEY, how="left", validate="one_to_one")
    scores = photos[INTERIOR_SCORES].to_numpy(dtype=float)
    scored = np.isfinite(scores).all(axis=1)
    photos["p_unfinished"] = softmax(CLIP_SCALE * scores, axis=1)[:, 3]
    photos["unfinished"] = photos["p_unfinished"].ge(UNFINISHED_CUTOFF).astype("boolean").where(scored)
    # Conditioning the four-way probabilities on the three finished classes
    # gives the same result as a softmax over their three logits.
    finished_probs = softmax(CLIP_SCALE * scores[:, :3], axis=1)
    finished = scored & photos["p_unfinished"].lt(UNFINISHED_CUTOFF)
    photos["clip_polarity"] = pd.Series(finished_probs[:, 0] - finished_probs[:, 2], index=photos.index).where(finished)

    photos["ratio"] = photos["width"].astype(float) / photos["height"].where(photos["height"].gt(0)).astype(float)
    photos["resolution"] = photos["resolution"].astype(float) / 1e6
    for _, market in photos.groupby("deal_type"):
        for column in ["sharpness", "brightness"]:
            photos.loc[market.index, column] = market[column].fillna(
                market.loc[market["clip_polarity"].notna(), column].mean(),
            )
    photos["sharpness"] = (photos["sharpness"] / 1e3).clip(upper=10)
    photos["brightness"] /= 256
    photos["brisque"] = photos["brisque"].fillna(0).clip(0, 100) / 100
    return photos


def adjust_photo_quality(photos: pd.DataFrame) -> dict:
    photos["clip_polarity_adj"] = np.nan
    models = {}
    for market, group in photos.groupby("deal_type"):
        group = group.dropna(subset=["clip_polarity", *QUALITY_COLUMNS])
        model = smf.ols(QUALITY_FORMULA, data=group).fit(
            cov_type="cluster", cov_kwds={"groups": group["property_id"]},
        )
        photos.loc[group.index, "clip_polarity_adj"] = model.resid
        models[market] = model
    return models


def property_polarity(photo_dir: Path, filename: str, column: str, photo_keys: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_parquet(photo_dir / filename, columns=PHOTO_KEY + TIER_SCORES)
    scores = scores.merge(photo_keys, on=PHOTO_KEY, validate="one_to_one")
    scores = scores.dropna(subset=TIER_SCORES)
    probabilities = softmax(CLIP_SCALE * scores[TIER_SCORES].to_numpy(dtype=float), axis=1)
    scores[column] = probabilities[:, 2] - probabilities[:, 0]
    return scores.groupby(PROPERTY_KEY, as_index=False)[column].mean()


def load_paper_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    listings = load_listings(project_root / DATASETS_DIR)
    properties = listings[PROPERTY_KEY]
    photo_dir = project_root / "outputs" / "clip_and_photo_stats"
    inventory = pd.read_parquet(photo_dir / "photo_classifications.parquet", columns=PHOTO_KEY + ["label"])
    inventory = inventory.merge(properties, on=PROPERTY_KEY, validate="many_to_one")
    counts = inventory.dropna(subset=["label"]).groupby(PROPERTY_KEY).size().rename("link_num")
    listings = listings.merge(counts, on=PROPERTY_KEY, how="left", validate="one_to_one")
    listings["link_num"] = listings["link_num"].fillna(0).astype(int)
    listings["link_num_categ"] = pd.cut(
        listings["link_num"], [-1, 0, 10, 20, 30, 40, np.inf], labels=PHOTO_BINS,
    )

    interiors = load_interiors(photo_dir, inventory.loc[inventory["label"].eq("interior"), PHOTO_KEY])
    quality_models = adjust_photo_quality(interiors)
    interior_features = interiors.groupby(PROPERTY_KEY, as_index=False).agg(
        clip_polarity=("clip_polarity", "mean"),
        clip_polarity_adj=("clip_polarity_adj", "mean"),
        share_unfinished=("unfinished", "mean"),
    )
    interior_features["share_unfinished"] = interior_features["share_unfinished"].astype(float)
    exterior_keys = inventory.loc[inventory["label"].isin(["street", "view", "common"]), PHOTO_KEY]
    plan_keys = inventory.loc[inventory["label"].eq("floor_plan"), PHOTO_KEY]
    # Use the property key because the listing exports store a different offer identifier.
    for features in (
        interior_features,
        property_polarity(photo_dir, "exterior_photo_scores.parquet", "exterior_polarity", exterior_keys),
        property_polarity(photo_dir, "plan_photo_scores.parquet", "plan_polarity", plan_keys),
    ):
        listings = listings.merge(features, on=PROPERTY_KEY, how="left", validate="one_to_one")
    return listings, interiors, quality_models


def load_llama_scores(path: Path, listings: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_parquet(path, columns=PROPERTY_KEY + ["average_luxury_score"])
    scored = scores["average_luxury_score"].dropna()
    if not scored.between(0, 10).all():
        raise ValueError(f"{path} contains ratings outside the 0–10 scale")
    return listings.merge(scores, on=PROPERTY_KEY, how="left", validate="one_to_one")
