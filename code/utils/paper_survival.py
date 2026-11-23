"""Time-on-market sample and the proportional hazards model with adjusted interior Q."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from patsy import dmatrices

from .paper_models import LISTING_FEATURES, PHOTO_FEATURES, remove_unused_categories

COX_FORMULA = (
    "dur ~ np.log(totalArea) + C(roomsCount_grouped) + C(link_num_categ) + C(bathrooms_cat)"
    " + C(material) + I(age / 10) + C(ageNA) + C(toDemolish) + C(apartment)"
    " + distance_center + distance_metro + distance_nice + distance_ugly + C(repairType)"
    " + np.log(price_first) + C(floorNumberCapped) + C(ceilingHeightCapped) + C(district)"
    " + " + " + ".join(PHOTO_FEATURES)
)


def duration_sample(listings: pd.DataFrame, market: str) -> pd.DataFrame:
    frame = listings.loc[listings["deal_type"].eq(market) & listings["link_num"].gt(0)]
    frame = frame.dropna(subset=LISTING_FEATURES + PHOTO_FEATURES + ["price_first", "status", "dur"])
    frame = frame.loc[frame["price_first"].gt(0) & frame["dur"].gt(0)]
    frame = frame.loc[frame["price_first"].lt(frame["price_first"].quantile(0.99))]
    return frame.loc[frame["status"] | frame["dur"].gt(60)]


def fit_cox_model(frame: pd.DataFrame) -> CoxPHFitter:
    data = remove_unused_categories(frame.copy())
    _, design = dmatrices(COX_FORMULA, data=data, return_type="dataframe", NA_action="raise")
    fit_data = data[["dur", "status"]].join(design.drop(columns="Intercept"))
    return CoxPHFitter().fit(fit_data, duration_col="dur", event_col="status", robust=True)
