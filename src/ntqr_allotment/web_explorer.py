from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

COVER_FIGURE_NAME = "ntqr_cover.png"


@dataclass(frozen=True)
class ExplorerDataset:
    dataset_id: str
    title: str
    source: str
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ExplorerFigure:
    filename: str
    title: str
    source: str
    statistic: str
    caveat: str
    category: str
    tier: str


@dataclass(frozen=True)
class ExplorerModel:
    metadata: dict[str, object]
    datasets: tuple[ExplorerDataset, ...]
    figures: tuple[ExplorerFigure, ...]


def build_explorer_model(repo_root: Path, *, max_rows: int = 500) -> ExplorerModel:
    root = Path(repo_root)
    data_dir = root / "output" / "data"
    figure_dir = root / "output" / "figures"
    sweep = _read_json(data_dir / "sweep_results.json")
    postdoc = _read_optional_json(data_dir / "postdoc_panel_results.json")
    postdoc_alignment = _read_optional_json(data_dir / "postdoc_panel_alignment.json")
    cover_manifest = _read_optional_json(data_dir / "cover_manifest.json")
    analytical = _read_optional_json(data_dir / "analytical_predictions.json")
    config_metadata = _read_config_metadata(root / "manuscript" / "config.yaml")
    datasets = (
        ExplorerDataset(
            "sweep_rows",
            "Synthetic sweep rows",
            "output/data/sweep_results.json",
            tuple(_limit_rows(sweep.get("rows", []), max_rows)),
        ),
        ExplorerDataset(
            "sweep_aggregates",
            "Synthetic sweep aggregates",
            "output/data/sweep_aggregated.csv",
            tuple(_read_csv(data_dir / "sweep_aggregated.csv", max_rows=max_rows)),
        ),
        ExplorerDataset(
            "power_diagnostics",
            "Power diagnostics",
            "output/data/power_analysis.csv",
            tuple(_read_csv(data_dir / "power_analysis.csv", max_rows=max_rows)),
        ),
        ExplorerDataset(
            "analytical_predictions",
            "Analytical prediction cells",
            "output/data/analytical_predictions.json",
            tuple(_limit_rows(analytical.get("rep_vs_ideo_cells", []), max_rows)),
        ),
        ExplorerDataset(
            "pre_post_cells",
            "Pre/post NTQR cells",
            "output/data/analytical_predictions.json",
            tuple(_limit_rows(analytical.get("pre_post_cells", []), max_rows)),
        ),
        ExplorerDataset(
            "postdoc_panel_aggregates",
            "Gemma postdoc panel aggregates",
            "output/data/postdoc_panel_results.json",
            tuple(_limit_rows(postdoc.get("aggregates", []), max_rows)),
        ),
        ExplorerDataset(
            "postdoc_alignment",
            "Analytical-vs-Gemma postdoc alignment",
            "output/data/postdoc_panel_alignment.json",
            tuple(_limit_rows(postdoc_alignment.get("cells", []), max_rows)),
        ),
    )
    figures = tuple(
        _figure_record(path)
        for path in sorted(figure_dir.glob("*.png"))
        if path.is_file()
        and path.name != COVER_FIGURE_NAME
        and not path.stem.startswith("cross_family")
    )
    metadata = {
        **config_metadata,
        "sweep_profile": (sweep.get("metadata") or {}).get("profile_name"),
        "sweep_config_hash": (sweep.get("metadata") or {}).get("config_hash"),
        "postdoc_config_hash": postdoc.get("config_hash"),
        "postdoc_model": postdoc.get("model"),
        "postdoc_live": postdoc.get("live_ollama"),
        "postdoc_runs": len(((postdoc.get("config") or {}).get("seed_list") or [])),
        "postdoc_alignment_rate": postdoc_alignment.get("agreement_rate_resolved"),
        "analytical_prediction_schema": analytical.get("schema_version"),
        "figure_count": len(figures),
        "cover_image": cover_manifest.get("cover_image", f"output/figures/{COVER_FIGURE_NAME}"),
        "cover_manifest": cover_manifest.get("manifest", "output/data/cover_manifest.json"),
        "cover_role": cover_manifest.get("semantic_role", "schematic front-matter cover"),
        "cover_caveat": cover_manifest.get(
            "caveat",
            "Deterministic schematic only; not an empirical result.",
        ),
        "claim_boundary": (
            "Local explorer only; PDF/manuscript claims remain bounded by "
            "generated static artifacts."
        ),
    }
    return ExplorerModel(metadata=metadata, datasets=datasets, figures=figures)


def render_explorer_html(model: ExplorerModel) -> str:
    payload = {
        "metadata": model.metadata,
        "datasets": [dataset.__dict__ for dataset in model.datasets],
        "figures": [figure.__dict__ for figure in model.figures],
    }
    state_json = json.dumps(payload, sort_keys=True)
    state_json_html = _script_json(state_json)
    dataset_options = "\n".join(
        f'<option value="{html.escape(dataset.dataset_id)}">'
        f"{html.escape(dataset.title)}</option>"
        for dataset in model.datasets
    )
    figure_tiles = "\n".join(_render_figure_tile(figure) for figure in model.figures)
    figure_category_options = "\n".join(
        f'<option value="{html.escape(category)}">{html.escape(category)}</option>'
        for category in sorted({figure.category for figure in model.figures})
    )
    metadata_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(model.metadata.items())
    )
    masthead = _render_masthead(model.metadata)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NTQR Local Artifact Explorer</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5c6973;
      --line: #d8dee4;
      --panel: #f7f9fb;
      --accent: #0b6b74;
      --accent-2: #8b4d18;
      --wash: #edf6f2;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    header {{ border-bottom: 1px solid var(--line); padding-bottom: 16px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; letter-spacing: 0; }}
    p {{ color: var(--muted); max-width: 860px; }}
    .masthead {{
      display: grid; grid-template-columns: minmax(260px, 0.8fr) 1.2fr;
      gap: 20px; align-items: center; margin: 18px 0 8px;
      padding: 16px; border: 1px solid var(--line); background: #fbf7ef;
    }}
    .masthead img {{
      width: 100%; height: auto; display: block; border: 1px solid var(--line);
      background: white;
    }}
    .metadata-list {{ display: grid; gap: 7px; margin-top: 12px; }}
    .metadata-list div {{ color: var(--muted); }}
    .metadata-list strong {{ color: var(--ink); }}
    .controls {{
      display: flex; gap: 12px; flex-wrap: wrap; align-items: end;
      margin: 16px 0; padding: 12px; background: var(--panel);
      border: 1px solid var(--line);
    }}
    label {{ display: grid; gap: 4px; font-weight: 600; }}
    select, input {{
      min-height: 34px; border: 1px solid var(--line); padding: 6px 8px;
      font: inherit; background: white;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; }}
    th {{ background: var(--panel); position: sticky; top: 0; }}
    .table-wrap {{ max-height: 520px; overflow: auto; border: 1px solid var(--line); }}
    .summary-line {{ color: var(--muted); margin: 0 0 12px; }}
    .figures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; border: 1px solid var(--line); background: white; }}
    figure img {{ display: block; width: 100%; height: auto; background: var(--panel); }}
    figcaption {{ padding: 10px 12px; color: var(--muted); }}
    .stat {{ color: var(--accent); font-weight: 650; }}
    .caveat {{ color: var(--accent-2); }}
    .figure-meta {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 8px; }}
    .pill {{
      display: inline-flex; align-items: center; min-height: 22px;
      padding: 2px 7px; border: 1px solid var(--line); background: var(--wash);
      color: var(--ink); font-size: 12px; font-weight: 650;
    }}
    .figure-actions {{ margin-top: 8px; }}
    .figure-actions a {{ color: var(--accent); font-weight: 650; text-decoration: none; }}
    .figure-hidden {{ display: none; }}
    @media (max-width: 720px) {{
      main {{ padding: 16px; }}
      .masthead {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>NTQR Local Artifact Explorer</h1>
    <p>{html.escape(str(model.metadata.get("claim_boundary")))}</p>
  </header>
  {masthead}
  <h2>Run Metadata</h2>
  <table>{metadata_rows}</table>
  <h2>Source Tables</h2>
  <div class="controls">
    <label>Dataset<select id="dataset-select">{dataset_options}</select></label>
    <label>Filter<input id="table-filter" type="search" placeholder="type to filter rows"></label>
  </div>
  <div id="table-source"></div>
  <div class="table-wrap"><table id="data-table"></table></div>
  <h2>Figure Contracts</h2>
  <div class="controls">
    <label>Category<select id="figure-category-select"><option value="all">All categories</option>{figure_category_options}</select></label>
    <label>Figure filter<input id="figure-search" type="search" placeholder="source, statistic, caveat"></label>
  </div>
  <div id="figure-summary" class="summary-line"></div>
  <div id="figure-gallery" class="figures">{figure_tiles}</div>
</main>
<script id="explorer-state" type="application/json">{state_json_html}</script>
<script>
const state = JSON.parse(document.getElementById("explorer-state").textContent);
const select = document.getElementById("dataset-select");
const filter = document.getElementById("table-filter");
const table = document.getElementById("data-table");
const source = document.getElementById("table-source");
const figureCategory = document.getElementById("figure-category-select");
const figureSearch = document.getElementById("figure-search");
const figureSummary = document.getElementById("figure-summary");
const figureCards = Array.from(document.querySelectorAll("#figure-gallery figure"));
function renderTable() {{
  const dataset = state.datasets.find(d => d.dataset_id === select.value);
  const needle = filter.value.toLowerCase();
  const rows = dataset.rows.filter(row => JSON.stringify(row).toLowerCase().includes(needle));
  const columns = Array.from(new Set(rows.flatMap(row => Object.keys(row)))).slice(0, 24);
  source.innerHTML = `<p><strong>${{dataset.title}}</strong> from <code>${{dataset.source}}</code>; showing ${{rows.length}} row(s).</p>`;
  table.innerHTML = "";
  const thead = table.createTHead();
  const tr = thead.insertRow();
  columns.forEach(col => tr.insertCell().outerHTML = `<th>${{col}}</th>`);
  const tbody = table.createTBody();
  rows.forEach(row => {{
    const bodyRow = tbody.insertRow();
    columns.forEach(col => bodyRow.insertCell().textContent = row[col] ?? "");
  }});
}}
function renderFigures() {{
  const category = figureCategory.value;
  const needle = figureSearch.value.toLowerCase();
  let visible = 0;
  figureCards.forEach(card => {{
    const categoryMatch = category === "all" || card.dataset.category === category;
    const textMatch = card.dataset.search.includes(needle);
    const show = categoryMatch && textMatch;
    card.classList.toggle("figure-hidden", !show);
    if (show) visible += 1;
  }});
  figureSummary.textContent = `Showing ${{visible}} of ${{figureCards.length}} figure contracts.`;
}}
select.addEventListener("change", renderTable);
filter.addEventListener("input", renderTable);
figureCategory.addEventListener("change", renderFigures);
figureSearch.addEventListener("input", renderFigures);
renderTable();
renderFigures();
</script>
</body>
</html>
"""


def write_explorer(repo_root: Path, output_path: Path | None = None) -> Path:
    root = Path(repo_root)
    out = output_path or root / "output" / "web" / "ntqr_explorer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_explorer_html(build_explorer_model(root)), encoding="utf-8")
    return out


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _script_json(payload: str) -> str:
    return payload.replace("</", "<\\/")


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


def _read_config_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    paper = data.get("paper") if isinstance(data.get("paper"), dict) else {}
    publication = data.get("publication") if isinstance(data.get("publication"), dict) else {}
    authors = data.get("authors") if isinstance(data.get("authors"), list) else []
    author = next((item for item in authors if isinstance(item, dict)), {})
    doi = publication.get("doi") or publication.get("doi_status")
    return {
        "paper_title": paper.get("title"),
        "paper_subtitle": paper.get("subtitle"),
        "paper_date": paper.get("date"),
        "author_name": author.get("name"),
        "author_affiliation": author.get("affiliation"),
        "author_orcid": author.get("orcid"),
        "author_email": author.get("email"),
        "publication_doi": doi,
    }


def _read_csv(path: Path, *, max_rows: int) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for _, row in zip(range(max_rows), csv.DictReader(handle), strict=False)]


def _limit_rows(rows: object, max_rows: int) -> list[dict[str, object]]:
    if not isinstance(rows, list):
        return []
    limited: list[dict[str, object]] = []
    for row in rows[:max_rows]:
        if isinstance(row, dict):
            limited.append(_flatten_row(row))
    return limited


def _flatten_row(row: dict[str, object]) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened[key] = value
        else:
            flattened[key] = json.dumps(value, sort_keys=True)
    return flattened


def _figure_record(path: Path) -> ExplorerFigure:
    name = path.stem.replace("_", " ")
    return ExplorerFigure(
        filename=path.name,
        title=name.title(),
        source=f"output/figures/{path.name}",
        statistic=_figure_statistic(path.stem),
        caveat=_figure_caveat(path.stem),
        category=_figure_category(path.stem),
        tier=_figure_tier(path.stem),
    )


def _figure_category(stem: str) -> str:
    if "postdoc" in stem:
        return "Live companion"
    if "power" in stem:
        return "Power and design"
    if any(token in stem for token in ("correlation", "bloc", "concentration")):
        return "Dependence mechanism"
    if "heatmap" in stem or "alignment" in stem:
        return "Regime grid"
    if "ranking" in stem:
        return "Strategy ranking"
    if "alarm" in stem or "fairness" in stem:
        return "Structural diagnostics"
    return "Figure contract"


def _figure_tier(stem: str) -> str:
    if "postdoc" in stem:
        return "single-model live companion"
    if "power" in stem:
        return "design diagnostic"
    if "pipeline" in stem or "schematic" in stem:
        return "schematic front matter"
    if "heatmap" in stem or "alignment" in stem:
        return "profile-bounded synthetic result"
    return "generated local artifact"


def _figure_statistic(stem: str) -> str:
    if "postdoc_strategy" in stem:
        return "analytical and Gemma EIE ranking with descriptive intervals"
    if "postdoc_age_bias" in stem:
        return "older-minus-younger recommendation-rate disparity by strategy and panel size"
    if "postdoc_empirical" in stem:
        return "cell-level directional alignment between analytical and Gemma observations"
    if "power" in stem:
        return "observed effect, MDE, power, or analytic curve"
    if "heatmap" in stem or "alignment" in stem:
        return "regime-cell contrast, pre/post delta, or analytical alignment"
    if "ranking" in stem:
        return "weighted mean EIE error with 95% CI"
    return "artifact-derived descriptive statistic"


def _figure_caveat(stem: str) -> str:
    if "postdoc" in stem:
        return "single-model Gemma reviewer-panel companion; synthetic applicants and age metadata"
    if "power" in stem:
        return "design diagnostic, not retrospective evidence of truth"
    if "heatmap" in stem or "alignment" in stem:
        return "directional and profile-bounded; live evidence is a separate bounded companion"
    return "source-bounded to generated local artifacts"


def _render_figure_tile(figure: ExplorerFigure) -> str:
    filename = html.escape(figure.filename)
    category = html.escape(figure.category, quote=True)
    search_text = " ".join(
        [
            figure.title,
            figure.source,
            figure.statistic,
            figure.caveat,
            figure.category,
            figure.tier,
        ]
    ).lower()
    return (
        f'<figure data-category="{category}" data-search="{html.escape(search_text, quote=True)}">'
        f'<img src="../figures/{filename}" alt="{html.escape(figure.title)}">'
        "<figcaption>"
        '<div class="figure-meta">'
        f'<span class="pill">{html.escape(figure.category)}</span>'
        f'<span class="pill">{html.escape(figure.tier)}</span>'
        "</div>"
        f"<strong>{html.escape(figure.title)}</strong><br>"
        f"source <code>{html.escape(figure.source)}</code><br>"
        f'<span class="stat">{html.escape(figure.statistic)}</span><br>'
        f'<span class="caveat">{html.escape(figure.caveat)}</span>'
        f'<div class="figure-actions"><a href="../figures/{filename}">Full-size PNG</a></div>'
        "</figcaption></figure>"
    )


def _render_masthead(metadata: dict[str, object]) -> str:
    cover_image = str(metadata.get("cover_image") or "")
    image_html = ""
    if cover_image:
        rel = cover_image.removeprefix("output/")
        image_html = (
            f'<img src="../{html.escape(rel)}" '
            'alt="Deterministic schematic cover for the sortition-to-NTQR pipeline">'
        )
    orcid = str(metadata.get("author_orcid") or "")
    orcid_html = (
        f'<a href="https://orcid.org/{html.escape(orcid)}">ORCID: {html.escape(orcid)}</a>'
        if orcid
        else ""
    )
    return (
        '<section class="masthead" aria-label="Manuscript front matter">'
        f"<div>{image_html}</div>"
        "<div>"
        f"<h2>{html.escape(str(metadata.get('paper_title') or 'NTQR manuscript'))}</h2>"
        f"<p>{html.escape(str(metadata.get('paper_subtitle') or ''))}</p>"
        '<div class="metadata-list">'
        f"<div><strong>Author</strong> {html.escape(str(metadata.get('author_name') or ''))}</div>"
        f"<div><strong>Affiliation</strong> {html.escape(str(metadata.get('author_affiliation') or ''))}</div>"
        f"<div><strong>DOI</strong> {html.escape(str(metadata.get('publication_doi') or ''))}</div>"
        f"<div>{orcid_html}</div>"
        f"<div><strong>Cover</strong> {html.escape(str(metadata.get('cover_role') or ''))}</div>"
        f'<div class="caveat">{html.escape(str(metadata.get("cover_caveat") or ""))}</div>'
        f'<div><code>{html.escape(str(metadata.get("cover_manifest") or ""))}</code></div>'
        "</div></div></section>"
    )


__all__ = [
    "ExplorerDataset",
    "ExplorerFigure",
    "ExplorerModel",
    "build_explorer_model",
    "render_explorer_html",
    "write_explorer",
]
