"""Price models using the paper's listing controls and adjusted interior Q-score."""

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

SEED = 108
PRICE_SCALE = {"long_rent": 1e3, "sale_secondary": 1e6}
PRICE_UNITS = {"long_rent": "thousand rubles", "sale_secondary": "million rubles"}

CATEGORICAL_FEATURES = [
    "roomsCount_grouped", "bathrooms_cat", "material", "ageNA", "toDemolish",
    "link_num_categ", "apartment", "repairType", "floorNumberCapped", "ceilingHeightCapped", "district",
]
NUMERICAL_FEATURES = ["totalArea", "age", "distance_center", "distance_metro", "distance_nice", "distance_ugly"]
LISTING_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
PHOTO_FEATURES = ["clip_polarity_adj"]

FEATURE_LABELS = {
    "roomsCount_grouped": "Rooms", "bathrooms_cat": "Bathrooms", "material": "Material",
    "ageNA": "Missing year built", "toDemolish": "Scheduled demolition",
    "link_num_categ": "Photo count", "apartment": "Apartment status", "repairType": "State of repair",
    "floorNumberCapped": "Floor number", "ceilingHeightCapped": "Ceiling height", "district": "District",
    "totalArea": "Total area", "age": "Building age", "distance_center": "Distance to center",
    "distance_metro": "Distance to metro", "distance_nice": "Distance to green area or water",
    "distance_ugly": "Distance to industrial zone",
    "clip_polarity_adj": "Adjusted interior Q-score",
}

UNIT_TERMS = (
    "I(totalArea / 10) + I(totalArea ** 2 / 100) + C(roomsCount_grouped) + C(bathrooms_cat)"
    " + C(floorNumberCapped) + C(repairType) + C(link_num_categ)"
)
BUILDING_TERMS = "C(material) + I(age / 10) + I(age ** 2 / 100) + C(ageNA) + C(ceilingHeightCapped)"
GEO_TERMS = "distance_center + distance_metro + distance_nice + distance_ugly"
LISTING_TERMS = " + ".join([UNIT_TERMS, BUILDING_TERMS, GEO_TERMS, "C(district)"])

# Randomized-search selections saved in hedonic_analysis.ipynb, cells 86, 87, 89, and 90.
BOOST_SETTINGS = {
    "long_rent": {
        "listing": dict(max_iter=1100, learning_rate=0.016054385683402672, max_depth=9,
                        min_samples_leaf=17, l2_regularization=0.9151434157189245, max_features=0.5354574314579672),
        "clip": dict(max_iter=950, learning_rate=0.018156218061225138, max_depth=7,
                     min_samples_leaf=16, l2_regularization=0.05807641153320586, max_features=0.3617962026323815),
    },
    "sale_secondary": {
        "listing": dict(max_iter=1350, learning_rate=0.041186272759856535, max_depth=11,
                        min_samples_leaf=16, l2_regularization=0.4707699863845225, max_features=0.3623196725604967),
        "clip": dict(max_iter=1100, learning_rate=0.033424695464408294, max_depth=11,
                     min_samples_leaf=22, l2_regularization=0.8069301642299439, max_features=0.35739414119192897),
    },
}


def remove_unused_categories(frame: pd.DataFrame) -> pd.DataFrame:
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("category").cat.remove_unused_categories()
    return frame


def price_split(listings: pd.DataFrame, market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = listings.loc[listings["deal_type"].eq(market) & listings["link_num"].gt(0)]
    frame = frame.dropna(subset=LISTING_FEATURES + PHOTO_FEATURES + ["price_last", "price_first"])
    frame = frame.loc[frame["price_last"].gt(0) & frame["price_last"].lt(frame["price_last"].quantile(0.99))]
    frame = remove_unused_categories(frame.copy())
    frame["log_price"] = np.log(frame["price_last"] / PRICE_SCALE[market])
    return train_test_split(frame, test_size=0.2, random_state=SEED, stratify=frame["district"])


def boost_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    features = frame[columns].copy()
    for column in columns:
        if column in CATEGORICAL_FEATURES:
            features[column] = frame[column].astype("category").cat.codes
    return features


def fit_price_models(listings: pd.DataFrame, market: str, specifications=(1, 2)):
    train, test = price_split(listings, market)
    feature_sets = {
        1: LISTING_FEATURES,
        2: LISTING_FEATURES + PHOTO_FEATURES,
    }
    formulas = {
        1: LISTING_TERMS,
        2: " + ".join([LISTING_TERMS, *PHOTO_FEATURES]),
    }
    models = {}
    for number in specifications:
        y_train, X_train = dmatrices(
            "log_price ~ " + formulas[number], data=train, return_type="dataframe", NA_action="raise",
        )
        ols = sm.OLS(y_train, X_train).fit(cov_type="HC3")
        columns = feature_sets[number]
        settings = BOOST_SETTINGS[market]["listing" if number == 1 else "clip"]
        hgb = HistGradientBoostingRegressor(
            random_state=SEED, early_stopping=True, categorical_features="from_dtype",
            scoring="neg_mean_squared_error", **settings,
        )
        hgb.fit(boost_features(train, columns).to_numpy(), train["log_price"])
        models[number] = {"ols": ols, "design_info": X_train.design_info, "hgb": hgb, "features": columns}
    return train, test, models


def model_performance(observed: pd.Series, predicted, predictors: int) -> dict:
    r2 = r2_score(observed, predicted)
    n = len(observed)
    return {
        "N": n,
        "Adjusted R²": 1 - (1 - r2) * (n - 1) / (n - predictors - 1),
        "RMSE": root_mean_squared_error(observed, predicted),
    }


def ols_features(design_info, data: pd.DataFrame) -> pd.DataFrame:
    return build_design_matrices([design_info], data, return_type="dataframe", NA_action="raise")[0]


def squares_only(values):
    return np.hstack([values, values ** 2])


def polynomial_feature_names(_transformer, input_features):
    return [f"{name}_{power}" for power in (1, 2) for name in input_features]


def linear_shap(train: pd.DataFrame, test: pd.DataFrame, market: str) -> pd.Series:
    # Figure 15 fits its own linear model using the same controls as boosting.
    columns = LISTING_FEATURES + PHOTO_FEATURES
    polynomial = ["totalArea", "age"]
    linear = [name for name in NUMERICAL_FEATURES + PHOTO_FEATURES if name not in polynomial]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("poly", FunctionTransformer(squares_only, feature_names_out=polynomial_feature_names), polynomial),
        ("linear", "passthrough", linear),
    ])
    train_design = preprocessor.fit_transform(boost_features(train, columns))
    test_design = preprocessor.transform(boost_features(test, columns))
    model = LinearRegression().fit(train_design, train["log_price"])
    background_size = {"long_rent": 12708, "sale_secondary": 8812}[market]
    masker = shap.maskers.Independent(train_design, max_samples=background_size)
    effects = shap.LinearExplainer(model, masker=masker)(test_design).values

    contributions = {}
    for index, name in enumerate(preprocessor.get_feature_names_out()):
        transformer, feature = name.split("__", 1)
        if transformer in ("cat", "poly"):
            feature = feature.rsplit("_", 1)[0]
        contributions[feature] = contributions.get(feature, 0) + effects[:, index]
    return pd.DataFrame(contributions).abs().mean()


def boosting_shap(model, test: pd.DataFrame, columns: list[str]) -> pd.Series:
    observed = boost_features(test, columns).to_numpy()
    values = shap.TreeExplainer(model=model)(observed).values
    return pd.Series(np.abs(values).mean(axis=0), index=columns)
