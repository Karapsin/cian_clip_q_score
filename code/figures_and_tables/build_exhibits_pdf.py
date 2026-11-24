"""Build the tables-and-figures PDF from the notebook exports.

Requires XeLaTeX with geometry, booktabs, and fontspec, plus fontconfig and
DejaVu Serif. Run the figures and tables notebooks to refresh the exports.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import pymupdf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from utils.paper_exhibits import (
    FIGURE_TITLES, FIGURE_VARIANTS, TABLE_TITLES, TABLE_VARIANTS, VARIANT_TITLES, number_label,
)


A4 = (595.28, 841.89)
MARGIN = 28
TABLE_GAP = 18
MIN_PAIR_SCALE = 0.90

TABLE_PREAMBLE = r"""
\documentclass[11pt]{article}
\usepackage[paperwidth=360mm,paperheight=400mm,margin=13mm]{geometry}
\usepackage{booktabs,fontspec}
\setmainfont{DejaVu Serif}
% The notebook exports put each multirow label on the first row of its group.
\newcommand{\multirow}[4][]{#4}
\pagestyle{empty}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\newcommand{\exhibittable}[2]{%
  \begin{center}
    {\large\bfseries #1\par}
    \vspace{8pt}
    {\small\input{#2}\par}
  \end{center}
  \clearpage
}
\begin{document}
"""


@dataclass(frozen=True)
class RenderedTable:
    title: str
    page_index: int
    bounds: pymupdf.Rect


@dataclass(frozen=True)
class TableLayout:
    width: float
    height: float
    scale: float
    side_by_side: bool


def table_groups(directory):
    groups = []
    for number, name in TABLE_TITLES.items():
        group = []
        for variant in TABLE_VARIANTS[number]:
            title = f"Table {number_label(number)}{variant}. {name} — {VARIANT_TITLES[variant]}"
            group.append((title, directory / f"table_{number}_{variant}.tex"))
        groups.append(group)
    groups.append([("Table S.1. Shared-sample asking-price models", directory / "table_price_shared.tex")])
    groups.append([("Table S.2. Shared-sample ad-closure models", directory / "table_duration_shared.tex")])
    return groups


def typeset_tables(groups, workspace):
    source = [TABLE_PREAMBLE]
    for group in groups:
        for title, path in group:
            shutil.copyfile(path, workspace / path.name)
            source.append(r"\exhibittable{" + title + "}{" + path.name + "}")
    source.append(r"\end{document}")
    (workspace / "tables.tex").write_text("\n".join(source), encoding="utf-8")
    result = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "tables.tex"],
        cwd=workspace, env={**os.environ, "TEXMFVAR": str(workspace / "texmf-var")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    warnings = [line for line in result.stdout.splitlines() if line.startswith(("Missing character:", "Overfull "))]
    if result.returncode:
        raise RuntimeError(result.stdout)
    if warnings:
        raise RuntimeError("\n".join(warnings))
    return workspace / "tables.pdf"


def table_bounds(page):
    bounds = pymupdf.Rect()
    for kind, coordinates in page.get_bboxlog():
        if kind != "ignore-text":
            bounds |= pymupdf.Rect(coordinates)
    if bounds.is_empty:
        raise ValueError(f"Empty table on typeset page {page.number + 1}")
    return (bounds + (-1, -1, 1, 1)) & page.rect


def fit_table_sheet(tables):
    widths = [table.bounds.width for table in tables]
    heights = [table.bounds.height for table in tables]
    gap = TABLE_GAP * (len(tables) - 1)
    layouts = []
    for width, height in (A4, A4[::-1]):
        usable_width = width - 2 * MARGIN
        usable_height = height - 2 * MARGIN
        stacked = min(1, usable_width / max(widths), (usable_height - gap) / sum(heights))
        layouts.append(TableLayout(width, height, stacked, False))
        if len(tables) == 2:
            side_by_side = min(1, (usable_width - gap) / sum(widths), usable_height / max(heights))
            layouts.append(TableLayout(width, height, side_by_side, True))
    return max(layouts, key=lambda layout: layout.scale)


def append_table_sheet(document, typeset, tables, layout, outline):
    page = document.new_page(width=layout.width, height=layout.height)
    if layout.side_by_side:
        total_width = sum(table.bounds.width for table in tables) * layout.scale + TABLE_GAP
        position = (layout.width - total_width) / 2
    else:
        position = MARGIN
    for table in tables:
        width = table.bounds.width * layout.scale
        height = table.bounds.height * layout.scale
        if layout.side_by_side:
            x, y = position, MARGIN
            position += width + TABLE_GAP
        else:
            x, y = (layout.width - width) / 2, position
            position += height + TABLE_GAP
        frame = pymupdf.Rect(x, y, x + width, y + height)
        page.show_pdf_page(frame, typeset, table.page_index, clip=table.bounds)
        outline.append([2, table.title, page.number + 1])


def append_tables(document, typeset, groups, outline):
    if len(typeset) != sum(len(group) for group in groups):
        raise ValueError("Expected one typeset page per table")
    page_index = 0
    for group in groups:
        tables = []
        for title, _ in group:
            tables.append(RenderedTable(title, page_index, table_bounds(typeset[page_index])))
            page_index += 1
        layout = fit_table_sheet(tables)
        if len(tables) == 2 and layout.scale < MIN_PAIR_SCALE:
            for table in tables:
                single = [table]
                append_table_sheet(document, typeset, single, fit_table_sheet(single), outline)
        else:
            append_table_sheet(document, typeset, tables, layout, outline)


def append_untitled_figure(document, figure, title):
    image = figure[0].rect
    width, height = A4[::-1] if image.width >= image.height else A4
    page = document.new_page(width=width, height=height)
    font_file = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "DejaVu Serif:style=Bold"], text=True,
    ).strip()
    font = pymupdf.Font(fontfile=font_file)
    size = min(13.5, (width - 58) / font.text_length(title, fontsize=1))
    extent = font.text_length(title, fontsize=size)
    page.insert_font(fontname="ExhibitTitle", fontfile=font_file)
    page.insert_text(((width - extent) / 2, 34), title, fontname="ExhibitTitle", fontsize=size)
    page.show_pdf_page(pymupdf.Rect(25, 53, width - 25, height - 25), figure, 0)


def append_figures(document, directory, outline):
    for number, name in FIGURE_TITLES.items():
        for variant in FIGURE_VARIANTS.get(number, ("",)):
            suffix = f"_{variant}" if variant else ""
            qualifier = f" · {VARIANT_TITLES[variant]}" if variant else ""
            title = f"Figure {number_label(number)}{variant}. {name}{qualifier}"
            path = directory / f"figure_{number}{suffix}.pdf"
            with pymupdf.open(path) as figure:
                if len(figure) != 1:
                    raise ValueError(f"Expected one page in {path}")
                outline.append([2, title, len(document) + 1])
                if variant:
                    document.insert_pdf(figure)
                else:
                    append_untitled_figure(document, figure, title)


def build_pdf(project, output):
    groups = table_groups(project / "outputs" / "tables")
    with TemporaryDirectory(prefix=".exhibits-", dir=output.parent) as temporary, pymupdf.open() as document:
        workspace = Path(temporary)
        outline = [[1, "Tables", 1]]
        with pymupdf.open(typeset_tables(groups, workspace)) as typeset:
            append_tables(document, typeset, groups, outline)
        table_sheets = len(document)
        outline.append([1, "Figures", table_sheets + 1])
        append_figures(document, project / "outputs" / "figures", outline)
        document.set_toc(outline)
        document.set_metadata({"title": "Tables and figures"})
        rebuilt = workspace / "exhibits.pdf"
        document.save(rebuilt, deflate=True, garbage=4)
        rebuilt.replace(output)
        return table_sheets, len(document) - table_sheets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output if args.output is not None else args.project / "tables_and_figures.pdf"
    tables, figures = build_pdf(args.project, output)
    print(f"Wrote {tables} table sheets and {figures} figure pages to {output}")
