"""Refresh the existing arXiv comparison with the generated figures and tables."""

import argparse
import base64
import csv
from html import escape, unescape
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from utils.paper_data import MARKETS, load_llama_scores, load_paper_data
from utils.paper_variants import VARIANTS, VARIANT_FIGURES, VARIANT_TABLES, VARIANT_TITLES


TABLE_CELLS = {"01": (5, 0), "02": (7, 0), "03": (9, 0), "04": (11, 0), "05": (16, 0),
               "06": (16, 1), "07": (18, 0), "A1": (21, 0)}
INTERIOR = VARIANTS["A"][0]
TABLE = r"<table\b[^>]*>(.*?)</table>"
VARIANT_LINKS = r'<nav class="variant-links"[^>]*>.*?</nav>'


def displays(cell, mime):
    return ["".join(output["data"][mime]) for output in cell.get("outputs", []) if mime in output.get("data", {})]


def figure_cells(cells):
    found = {}
    for index, cell in enumerate(cells[:-1]):
        if cell["cell_type"] != "markdown":
            continue
        number = re.match(r"^### Figure (\d+|A\.1)\.", "".join(cell["source"]))
        if number:
            found[number[1].replace(".", "").zfill(2)] = cells[index + 1]
    return found


def link(path, report):
    return escape(quote(os.path.relpath(path, report.parent), safe="/"), quote=True)


def number_format(value, style):
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    return format(value * 100, "+.2f") + " pp" if style == "pp" else format(value, style)


def read_values(root, figures):
    folder = root / "outputs" / "tables"
    with (folder / "paper_numbers.csv").open() as file:
        paper = {(row["metric"], row["deal_type"]): row["value"] for row in csv.DictReader(file)}
    with (folder / "paper_variant_numbers.csv").open() as file:
        variants = {
            (row["model"], row["metric"], row["feature"], row["deal_type"]): float(row["value"] or "nan")
            for row in csv.DictReader(file) if row["cohort"] == "eligible" and row["variant"] == "A"
        }

    def old(metric, style=",.0f"):
        return tuple(number_format(float(paper[metric, market]), style) for market in MARKETS)

    def current(model, metric, style=".3f", feature=""):
        return tuple(number_format(variants[model, metric, feature, market], style) for market in MARKETS)

    listings, _, _ = load_paper_data(root)
    ratings = load_llama_scores(root / "inputs/datasets/llama4_scores.parquet", listings)
    rated = ratings.loc[ratings["link_num"].gt(0)].groupby("deal_type")["average_luxury_score"].mean()
    without = tuple(f"{len(group) - int(float(paper['listings_with_usable_photos', market])):,}"
                    for market, group in listings.groupby("deal_type"))
    shap_tables = displays(figures["15"], "text/html")
    rank = []
    for table in shap_tables:
        rows = [
            (unescape(name.strip()), float(ols), float(hgb))
            for name, ols, hgb in re.findall(r"<th>([^<]+)</th>\s*<td>([\d.]+)</td>\s*<td>([\d.]+)</td>", table)
        ]
        target = next(row for row in rows if row[0] == "Adjusted interior Q-score")
        rank.append(tuple(str(1 + sum(row[column] > target[column] for row in rows)) for column in (1, 2)))
    hazard_ci = tuple(f"{low}–{high}" for low, high in zip(
        current("cox", "lower_95", ".2f", INTERIOR), current("cox", "upper_95", ".2f", INTERIOR),
    ))
    counts = {
        "All study listings": "listings", "Listings with usable photos": "listings_with_usable_photos",
        "Photos in the collected inventory": "usable_photos", "Grayscale images": "grayscale_interior_photos",
        "Qualified interior photos": "interior_photos", "JPEG interiors": "jpeg_interior_photos",
        "Unfinished interiors": "unfinished_interior_photos",
        "Listings with all scored interiors unfinished": "listings_with_all_scored_interiors_unfinished",
        "Listings with adjusted interior Q": "listings_with_adjusted_interior_score",
        "Listings with an exterior polarity score": "listings_with_exterior_score",
        "Listings with a floor-plan polarity score": "listings_with_floor_plan_score",
        "Images in the technical-quality regression": "finished_interiors_with_quality_metrics",
        "Archived LLaMA4 ratings": "llama_scored_listings", "Matched CLIP and LLaMA4 pairs": "clip_llama_comparable_listings",
        "Training listings in the source fit": "price_train_listings", "Test listings in the source fit": "price_test_listings",
        "Listings in duration sample": "duration_listings",
    }
    values = {label: old(metric) for label, metric in counts.items()}
    values.update({
        "Mean new listings per day": old("mean_new_listings_per_day", ".1f"),
        "Listings without usable photos": without, "Share without usable photos": old("fraction_without_usable_photos", ".1%"),
        "Landscape share of interiors with dimensions": old("landscape_share_of_interiors_with_dimensions", ".1%"),
        "Mean image-level CLIP polarity": current("coverage", "raw_photo_mean", feature=INTERIOR),
        "Polarity variance unexplained by image metrics": old("fraction_polarity_variance_unexplained", ".1%"),
        "LLaMA4 coverage among photo-eligible listings": old("llama_scored_share", ".1%"),
        "CLIP and LLaMA4 Pearson correlation": old("clip_llama_pearson_correlation", ".3f"),
        "Mean LLaMA4 property rating": tuple(f"{rated.loc[market]:.2f}" for market in MARKETS),
        "Mean raw CLIP polarity per listing": current("coverage", "raw_listing_mean", ".2f", INTERIOR),
        "OLS interior Q slope with listing controls": current("ols", "coefficient", feature=INTERIOR),
        "OLS asking-price change at Q +0.10": tuple(f"{value}%" for value in old("ols_interior_price_percent_change_per_point_one", ".2f")),
        "OLS test adjusted R², listing baseline": current("ols_baseline", "test_adjusted_r2"),
        "OLS test adjusted R², adding interior Q": current("ols", "test_adjusted_r2"),
        "OLS test RMSE, adding interior Q": current("ols", "test_rmse"),
        "OLS test adjusted R² gain from interior Q": old("ols_test_adj_r2_gain_interior", "pp"),
        "Boosting test adjusted R², listing baseline": current("hgb_baseline", "test_adjusted_r2"),
        "Boosting test adjusted R², adding interior Q": current("hgb", "test_adjusted_r2"),
        "Boosting test RMSE, adding interior Q": current("hgb", "test_rmse"),
        "Boosting test adjusted R² gain from interior Q": old("hgb_test_adj_r2_gain_interior", "pp"),
        "Interior Q importance rank, OLS": tuple(row[0] for row in rank),
        "Interior Q importance rank, boosting": tuple(row[1] for row in rank),
        "Closed listings": current("cox", "closed", ",.0f"), "Right-censored listings": current("cox", "censored", ",.0f"),
        "Right-censored share": old("right_censored_share", ".1%"),
        "Median days for observed closures": current("cox", "median_closed_days", ".1f"),
        "Kaplan–Meier median days, lowest Q quartile": current("kaplan_meier", "q1_median_days", ".1f", INTERIOR),
        "Kaplan–Meier median days, highest Q quartile": current("kaplan_meier", "q4_median_days", ".1f", INTERIOR),
        "Cox hazard ratio per 1.0 point higher adjusted Q": current("cox", "hazard_ratio", ".2f", INTERIOR),
        "Updated 95% interval for the Cox Q ratio": hazard_ci,
    })
    return values, shap_tables


def refresh_numeric_section(page, values):
    start = page.index('<section id="numbers"')
    end = page.index('<section id="figures"')
    section = page[start:end]
    seen = set()

    def replace(match):
        label = unescape(match[1])
        if label not in values:
            raise ValueError(f"Unrecognized comparison metric: {label}")
        seen.add(label)
        current_cell = r'(<td class="current-number">)[^<]*(</td>)'
        cell = current_cell if 'class="current-number"' in match[2] else r"(<td>)[^<]*(</td>)"
        cells = iter(values[label])
        body, count = re.subn(cell, lambda item: item[1] + escape(next(cells)) + item[2], match[2])
        if count != 2:
            raise ValueError(f"Expected two market values for {label}")
        return match[0].replace(match[2], body, 1)

    section = re.sub(r'<tr><th scope="row">(.*?)</th>(.*?)</tr>', replace, section)
    if seen != set(values):
        raise ValueError("The comparison is missing known metrics")
    section = section.replace(
        "the current labeled inventory, respectively.",
        "the currently classified photos from the source-selected offers, respectively.",
    )
    return page[:start] + section + page[end:]


def overview(page, values):
    usable = values["Listings with usable photos"]
    boost = values["Boosting test adjusted R² gain from interior Q"]
    hazards = values["Cox hazard ratio per 1.0 point higher adjusted Q"]
    body = f"""
<section id="overview" aria-labelledby="overview-title">
  <div class="section-head"><h2 id="overview-title">Overview</h2></div>
  <p class="intro">The report compares the arXiv v1 results with new analyses of long-term rentals and secondary sales in Moscow. The updated figures and tables use the photo offer recorded in the source links for each property.</p>
  <div class="key-results">
    <article class="key-card"><h3>Listings with usable photos</h3><span class="key-figure">{usable[0]} rentals · {usable[1]} sales</span>
      <p>Listings without a recorded source photo offer remain in the study totals.</p>
      <a href="#table-2">Listing characteristics</a></article>
    <article class="key-card"><h3>Price prediction</h3><span class="key-figure">{boost[0]} · {boost[1]}</span>
      <p>Boosting gains in test adjusted R² from adjusted interior Q on the variant A price samples.</p>
      <a href="#table-6">Gradient boosting results</a></article>
    <article class="key-card"><h3>Ad closure and interior Q</h3><span class="key-figure">{hazards[0]} · {hazards[1]}</span>
      <p>Variant A hazard ratios per point of adjusted interior Q, for rentals and sales.</p>
      <a href="#figure-17">Proportional hazards estimates</a></article>
  </div>
  <p class="method-note">The side-by-side exhibits show the interior specification, A. The <a href="#variants">shared-sample comparisons</a> include all six specifications and the listing-controls baseline.</p>
</section>
""".strip()
    return re.sub(r'<section id="overview"[^>]*>.*?</section>', lambda _: body, page, count=1, flags=re.S)


def table_markup(display, opening):
    body = re.search(TABLE, display, re.S)[1].replace(' style="text-align: right;"', "")
    return opening + body + "</table>"


def refresh_exhibits(page, figures, tables, shap, project, report):
    findings = {
        "figure-3": "The bins count classified photos from the single source-selected offer for each property.",
        "figure-4": "The dots use thin gray outlines and are colored by the number of photos in the source-selected offer. The basemap uses a locally saved Humanitarian tile snapshot.",
        "figure-5": "Both panels mark the 224-pixel CLIP preprocessing target. Variant A uses selected interior photos with dimensions.",
        "figure-6": "The red lines are univariate fits for finished interiors. Table 4 shows the joint regression on the five technical measures.",
        "figure-7": "The comparison uses the archived LLaMA4 luxury ratings and the selected offers’ mean finished-interior polarity.",
        "figure-8": "Each panel shows the relationship of mean interior polarity to area, room count, or bathroom count.",
        "figure-9": "Each box uses listings with a finished-interior score in the displayed category. Repair labels are supplied by listing authors.",
        "figure-11": "The demolition flag comes from the study listings and the score uses finished interiors from the source offers.",
        "figure-12": "District means use the original boundaries. The updated color scale covers the observed score range across both markets.",
        "figure-13": "The plot relates the last quoted asking price to the listing mean of residuals from the finished-interior image-quality regression.",
        "figure-15": "The linear model uses grouped feature attributions. The boosting model uses TreeExplainer on the full test split.",
        "figure-16": "Each market’s quartiles use the adjusted interior scores on the duration cohort.",
        "figure-17": "The hazard ratios condition on initial asking price and listing controls. The survival profiles illustrate changes in adjusted interior Q.",
        "figure-a1": "The top row uses archived LLaMA4 ratings. The bottom row uses listing means of finished-interior polarity.",
        "table-2": "The updated sample includes every listing with a classified photo on its source-selected offer. Geographic distances are in units of 100 metres.",
        "table-3": "The updated table covers finished interior photos used in the technical-quality regression. Brightness is HSL lightness divided by 256.",
        "table-4": "Each market’s regression clusters standard errors by listing. The mean residual for each listing is its adjusted Q.",
        "table-5": "Variant A uses the original OLS formula and a shared district-stratified split for the listing baseline and the model with interior Q.",
        "table-6": "The updated boosting fits use the archived scikit-learn 1.6.1 settings with the source-selected photos and the updated split.",
        "table-7": "The duration summary uses the adjusted interior score, initial-price and censoring filters, and source-selected photo offers.",
    }

    def replace(exhibit):
        kind, label, article = exhibit.groups()
        number = label.upper().zfill(2)
        key = f"{kind}-{label}"
        if key in findings:
            article = re.sub(r'(<p class="finding">).*?(</p>)',
                             lambda text: text[1] + escape(findings[key]) + text[2], article, count=1, flags=re.S)
        if kind == "figure":
            png, = displays(figures[number], "image/png")
            width, height = struct.unpack(">II", base64.b64decode(png[:32])[16:24])
            article, count = re.subn(
                r'<img src="data:image/png;base64,[^"]+" width="\d+" height="\d+"',
                lambda _: f'<img src="data:image/png;base64,{png}" width="{width}" height="{height}"', article, count=1,
            )
        else:
            cell, position = TABLE_CELLS[number]
            rendered = displays(tables[cell], "text/html")[position]
            article, count = re.subn(r'(<table [^>]+aria-label="Updated table [^"]+"[^>]*>).*?</table>',
                                     lambda old: table_markup(rendered, old[1]), article, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f"Missing updated panel in {key}")
        if key == "figure-15":
            parts = iter(shap)
            article = re.sub(r'(<table [^>]+aria-label="[^"]+feature contributions"[^>]*>).*?</table>',
                             lambda old: table_markup(next(parts), old[1]), article, flags=re.S)
        article = re.sub(VARIANT_LINKS, "", article, flags=re.S)
        supported = VARIANT_FIGURES if kind == "figure" else VARIANT_TABLES
        if number in supported:
            extension = "pdf" if kind == "figure" else "tex"
            folder = project / "outputs" / f"{kind}s"
            items = []
            for variant in VARIANTS:
                target = folder / f"{kind}_{number}_{variant}.{extension}"
                if not target.is_file():
                    raise FileNotFoundError(target)
                items.append(f'<a href="{link(target, report)}" title="{VARIANT_TITLES[variant]}">{variant}</a>')
            article += (f'<nav class="variant-links" aria-label="{kind.capitalize()} {number} variants">'
                        f'<span>Variants</span>{"".join(items)}</nav>')
        return article + "</article>"

    pattern = r'(?=<article class="comparison" id="(figure|table)-([^"]+)")(<article .*?)</article>'
    page, count = re.subn(pattern, replace, page, flags=re.S)
    if count != len(figures) + len(TABLE_CELLS):
        raise ValueError(f"Expected {len(figures) + len(TABLE_CELLS)} comparison exhibits, found {count}")
    return page


def variant_section(tables, project, report):
    shared = displays(next(cell for cell in tables if cell.get("id") == "export-variant-tables"), "text/html")
    rows = "".join(f"<li><strong>{variant}</strong> {title}</li>" for variant, title in VARIANT_TITLES.items())
    output = project / "outputs/tables"
    comparisons = []
    for title, filename, display in zip(("Asking price", "Ad closure"),
                                         ("table_price_shared.tex", "table_duration_shared.tex"), shared, strict=True):
        path = output / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        table = table_markup(display, f'<table class="data-table" aria-label="{title} comparison">')
        table = re.sub(r"(<tr>\s*)<th>\d*</th>", r"\1", table)
        comparisons.append(f'<h3>{title}</h3><div class="scroll-table" tabindex="0" role="region" '
                           f'aria-label="{title} comparison">{table}</div><p><a href="{link(path, report)}">LaTeX table</a></p>')
    return f"""
<section id="variants" aria-labelledby="variants-title">
  <div class="section-head"><h2 id="variants-title">Score variants</h2>
    <a href="{link(output / 'paper_variant_numbers.csv', report)}">Variant CSV</a></div>
  <ul class="variant-key">{rows}</ul>
  <p>Each exhibit uses the listings eligible for its scores. The two comparisons below use the same complete-case listings across the controls-only baseline and A–F within each market and outcome. The price comparison also shares a test split. Unfinished probability is the mean CLIP estimate across scored interior photos.</p>
  {"".join(comparisons)}
</section>
""".strip()


def update_report(project, report):
    code = project / "code/figures_and_tables"
    figures = figure_cells(json.loads((code / "figures.ipynb").read_text())["cells"])
    tables = json.loads((code / "tables.ipynb").read_text())["cells"]
    values, shap = read_values(project, figures)
    page = report.read_text(encoding="utf-8")
    page = re.sub(r'<section id="variants"[^>]*>.*?</section>\s*', "", page, flags=re.S)
    page = overview(page, values)
    page = refresh_numeric_section(page, values)
    page = refresh_exhibits(page, figures, tables, shap, project, report)
    page = page.replace('<section id="figures"', variant_section(tables, project, report) + '\n<section id="figures"', 1)
    if '<a href="#variants">Variants</a>' not in page:
        page = page.replace('<a href="#numbers">Numbers</a>', '<a href="#numbers">Numbers</a><a href="#variants">Variants</a>', 1)
    if ".variant-links {" not in page:
        page = page.replace("</style>", """
    .variant-links { display: flex; flex-wrap: wrap; gap: .7rem; align-items: center; margin-top: .7rem; }
    .variant-links span { font-size: .9rem; font-weight: 600; }
    .variant-links a { padding: .15rem .38rem; }
    .variant-key { display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; padding: 0; list-style: none; }
    .variant-key strong { margin-right: .25rem; }
  </style>""", 1)
    page = page.replace('<a href="final_project/README.md">Project README</a> · ', "")
    with NamedTemporaryFile(mode="w", dir=report.parent, encoding="utf-8", delete=False) as temporary:
        temporary.write(page)
    Path(temporary.name).replace(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=ROOT.parent / "comparison_report.html")
    args = parser.parse_args()
    update_report(args.project.resolve(), args.report.resolve())
