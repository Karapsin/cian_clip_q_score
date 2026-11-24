"""Titles and available variants for the paper tables and figures."""

VARIANT_TITLES = {
    "A": "Interior photos only",
    "B": "All CLIP features",
}
FIGURE_TITLES = {
    "02": "Listing creation dates",
    "03": "Images per listing by author type",
    "04": "Property location and number of images",
    "05": "Image dimensions",
    "06": "Image quality and CLIP scores",
    "07": "LLaMA4 ratings and CLIP scores",
    "08": "Property size and CLIP scores",
    "09": "Property characteristics and CLIP scores",
    "10": "Construction year and CLIP scores",
    "11": "Moscow demolition programme",
    "12": "CLIP scores by Moscow district",
    "13": "Asking prices and CLIP scores",
    "14": "Partial dependence of asking price on CLIP scores",
    "15": "SHAP feature importance",
    "16": "Kaplan–Meier curves by score quartile",
    "17": "Proportional hazards model",
    "A1": "LLaMA4 and CLIP score distributions",
}
TABLE_TITLES = {
    "02": "Listing characteristics",
    "03": "Image quality descriptives",
    "04": "Regression of CLIP scores on image quality",
    "05": "Hedonic regression",
    "06": "Histogram gradient boosting",
    "07": "Time on the market",
}
FIGURE_VARIANTS = {number: tuple(VARIANT_TITLES) for number in FIGURE_TITLES if number != "02"}
TABLE_VARIANTS = {number: tuple(VARIANT_TITLES) for number in TABLE_TITLES}
TABLE_VARIANTS["04"] = ("B",)


def number_label(number):
    return "A.1" if number == "A1" else str(int(number))
