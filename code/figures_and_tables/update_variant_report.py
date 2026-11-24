"""Build the report for the exterior-only and all-CLIP comparisons."""

import argparse
from html import escape
import json
import os
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from utils.paper_variants import FIGURE_TITLES, TABLE_TITLES, VARIANTS, VARIANT_FIGURES, VARIANT_TABLES, VARIANT_TITLES


STYLE = """
:root {
  color-scheme: light;
  color: #242721;
  background: #faf9f5;
  font: 16px/1.55 system-ui, sans-serif;
  --border: #d9ded3;
  --ink: #234831;
}
body { margin: 0; }
header { background: #e9eee3; border-bottom: 1px solid var(--border); }
.wrap { width: min(1050px, calc(100% - 2rem)); margin-inline: auto; }
header .wrap { padding-block: 2rem; }
h1, h2 { font-family: Georgia, serif; font-weight: 600; line-height: 1.2; }
h1 { font-size: clamp(2rem, 4vw, 3rem); margin: 0 0 .65rem; }
h2 { font-size: 1.65rem; }
h3 { font-size: 1.05rem; }
a { color: var(--ink); text-underline-offset: .15em; }
header nav { display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; }
main { padding-block: 1.25rem 3rem; }
section + section { margin-top: 2rem; }
dl { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: .35rem 1rem; }
dt { font-weight: 650; }
dd { margin: 0; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; background: white; font-variant-numeric: tabular-nums; }
th, td { padding: .42rem .62rem; text-align: left; border-bottom: 1px solid var(--border); }
thead th { background: #f0f3ea; font-size: .9rem; }
tbody tr:nth-child(even) { background: #fcfcf9; }
.index th[scope=row] { width: 4.5em; font-weight: 550; }
.index td:not(:first-of-type) { white-space: nowrap; }
.numbers th, .numbers td { font-size: .9rem; white-space: nowrap; }
@media (max-width: 580px) {
  dl { display: block; }
  dd { margin-bottom: .45rem; }
  th, td { padding-inline: .4rem; }
}
"""


def link(path, report, label):
    if not path.is_file():
        raise FileNotFoundError(path)
    target = quote(os.path.relpath(path, report.parent), safe="/")
    return f'<a href="{escape(target, quote=True)}">{escape(label)}</a>'


def number_label(number):
    return "A.1" if number == "A1" else str(int(number))


def exhibit_index(kind, titles, variants, project, report):
    folder = project / "outputs" / f"{kind}s"
    extension, download = ("pdf", "PDF") if kind == "figure" else ("tex", "LaTeX")
    headers = "".join(f"<th scope='col'>{variant}. {escape(VARIANT_TITLES[variant])}</th>" for variant in VARIANTS)
    rows = []
    for number, title in titles.items():
        if number in variants:
            cells = "".join(
                f"<td>{link(folder / f'{kind}_{number}_{variant}.{extension}', report, download)}</td>"
                for variant in VARIANTS
            )
        else:
            cells = f"<td colspan='2'>{link(folder / f'{kind}_{number}.{extension}', report, download)}</td>"
        rows.append(
            f"<tr><th scope='row'>{number_label(number)}</th><td>{escape(title)}</td>{cells}</tr>"
        )
    return (
        f"<div class='scroll'><table class='index' aria-label='{kind.capitalize()} files'>"
        f"<thead><tr><th scope='col'>{kind.capitalize()}</th><th scope='col'>Title</th>"
        f"{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def shared_tables(project):
    path = project / "code/figures_and_tables/tables.ipynb"
    notebook = json.loads(path.read_text())
    cell = next(item for item in notebook["cells"] if item.get("id") == "export-variant-tables")
    html = ["".join(item["data"]["text/html"]) for item in cell["outputs"] if "text/html" in item.get("data", {})]
    if len(html) != 2:
        raise ValueError("Run the tables notebook to generate the two shared-sample comparisons")
    for title, display in zip(("Asking price", "Ad closure"), html, strict=True):
        body = re.search(r"<table\b[^>]*>(.*?)</table>", display, flags=re.S)[1]
        body = body.replace(' style="text-align: right;"', "")
        body = re.sub(r"(<tr>\s*)<th>\d*</th>", r"\1", body)
        yield (f'<h3>{title}</h3><div class="scroll" role="region" tabindex="0" aria-label="{title} comparison">'
               f'<table class="numbers">{body}</table></div>')


def update_report(project, report):
    pdf = link(project / "tables_and_figures.pdf", report, "Tables and figures PDF")
    csv = link(project / "outputs/tables/paper_variant_numbers.csv", report, "Results CSV")
    figures = exhibit_index("figure", FIGURE_TITLES, VARIANT_FIGURES, project, report)
    tables = exhibit_index("table", TABLE_TITLES, VARIANT_TABLES, project, report)
    comparisons = "\n".join(shared_tables(project))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLIP score comparison</title>
<style>{STYLE}</style>
</head>
<body>
<header><div class="wrap">
  <h1>CLIP score comparison</h1>
  <nav aria-label="Report"><a href="#shared">Shared samples</a><a href="#figures">Figures</a>
    <a href="#tables">Tables</a>{pdf}{csv}</nav>
</div></header>
<main class="wrap">
<section aria-label="Specifications">
  <dl>
    <dt>A. Exterior Q only</dt><dd>Exterior Q and listing controls.</dd>
    <dt>B. All CLIP features</dt><dd>Variant A plus interior Q, floor-plan Q, and mean unfinished probability.</dd>
  </dl>
  <p>Both analyses use the photo offer recorded in the source links for each property.</p>
</section>
<section id="shared">
  <h2>Shared-sample results</h2>
  <p>The models use the same listings within each market and outcome. The price models share the train and test split.</p>
  {comparisons}
</section>
<section id="figures">
  <h2>Figures</h2>
  {figures}
</section>
<section id="tables">
  <h2>Tables</h2>
  {tables}
</section>
</main>
</body>
</html>
"""
    with NamedTemporaryFile(mode="w", dir=report.parent, encoding="utf-8", delete=False) as temporary:
        temporary.write(page)
    Path(temporary.name).replace(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=ROOT.parent / "comparison_report.html")
    args = parser.parse_args()
    update_report(args.project.resolve(), args.report.resolve())
