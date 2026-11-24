"""Select the photo offer recorded for each listing in the source analysis."""

from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd


LISTING_FILES = {
    "long_rent": "lr_clean_v4.parquet",
    "sale_secondary": "ss_clean_v4.parquet",
}
PROPERTY_KEY = ["deal_type", "property_id"]
OFFER_KEY = PROPERTY_KEY + ["offer_id"]
PHOTO_KEY = OFFER_KEY + ["photo_name"]
DATASETS_DIR = Path("inputs") / "datasets"
PHOTO_OUTPUTS = Path("outputs") / "clip_and_photo_stats"
OFFER_FILE = "listing_photo_offers.parquet"
PHOTO_TYPES = {
    "interior": "interior",
    "view": "exterior",
    "street": "exterior",
    "common": "exterior",
    "floor_plan": "plan",
}


def source_photo_offer(links, market: str, property_id: str) -> str | None:
    if links is None or len(links) == 0:
        return None
    offers = set()
    for link in links:
        parts = unquote(urlsplit(link).path).rsplit("/", 4)
        if len(parts) != 5 or parts[1:3] != [market, property_id] or not parts[3].isdigit():
            raise ValueError(f"Invalid archived photo URL for {market}/{property_id}: {link}")
        offers.add(parts[3])
    if len(offers) != 1:
        raise ValueError(f"Multiple archived photo offers for {market}/{property_id}: {sorted(offers)}")
    return offers.pop()


def build_source_offers(source_dir: Path) -> pd.DataFrame:
    frames = []
    for market, filename in LISTING_FILES.items():
        source = pd.read_parquet(source_dir / filename, columns=["property_id", "link"])
        offers = [
            source_photo_offer(links, market, property_id)
            for property_id, links in source.itertuples(index=False, name=None)
        ]
        frames.append(source.assign(deal_type=market, offer_id=pd.array(offers, dtype="string"))[OFFER_KEY])
    mapping = pd.concat(frames, ignore_index=True)
    if mapping[PROPERTY_KEY].isna().any().any() or mapping.duplicated(PROPERTY_KEY).any():
        raise ValueError("The source offer mapping must have one row per market and property")
    return mapping


def load_source_offers(project_root: Path) -> pd.DataFrame:
    path = project_root / DATASETS_DIR / OFFER_FILE
    offers = pd.read_parquet(path, columns=OFFER_KEY)
    if offers[PROPERTY_KEY].isna().any().any() or offers.duplicated(PROPERTY_KEY).any():
        raise ValueError(f"Invalid property keys in {path}")
    return offers


def load_selected_inventory(project_root: Path) -> pd.DataFrame:
    offers = load_source_offers(project_root).dropna(subset=["offer_id"])
    path = project_root / PHOTO_OUTPUTS / "photo_classifications.parquet"
    photos = pd.read_parquet(path, columns=PHOTO_KEY + ["label"]).dropna(subset=["label"])
    selected = photos.merge(offers, on=OFFER_KEY, validate="many_to_one")
    if selected.duplicated(PHOTO_KEY).any():
        raise ValueError("Duplicate keys in the selected photo inventory")
    selected["photo_type"] = selected["label"].map(PHOTO_TYPES)
    if selected["photo_type"].isna().any():
        raise ValueError("Unknown category in the selected photo inventory")
    return selected
