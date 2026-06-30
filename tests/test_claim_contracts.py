from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (_REPO / path).read_text(encoding="utf-8")


def _compact(text: str) -> str:
    return " ".join(text.split())


_TEXT_SURFACES = (
    "README.md",
    "ISA.md",
    "audits/claim_method_audit.md",
    "manuscript/00_abstract.md",
    "manuscript/01_introduction.md",
    "manuscript/02_methods.md",
    "manuscript/03_results.md",
    "manuscript/04_discussion.md",
    "manuscript/99_references.md",
    "src/ntqr_allotment/__init__.py",
    "src/ntqr_allotment/contrast_analysis.py",
    "src/ntqr_allotment/experts.py",
    "src/ntqr_allotment/ntqr_eval.py",
    "src/ntqr_allotment/pipeline.py",
    "output/manuscript/00_abstract.md",
    "output/manuscript/01_introduction.md",
    "output/manuscript/02_methods.md",
    "output/manuscript/99_references.md",
    "output/pdf/_combined_manuscript.md",
    "output/pdf/_combined_manuscript.tex",
    "output/web/index.html",
    "output/web/_combined_manuscript.md",
    "output/web/manuscript__00_abstract.html",
    "output/web/manuscript__01_introduction.html",
    "output/web/manuscript__99_references.html",
    "output/slides/00_abstract_slides.tex",
    "output/slides/01_introduction_slides.tex",
    "output/slides/99_references_slides.tex",
)


def test_ntqr_is_not_expanded_as_quantified_reliability() -> None:
    forbidden = (
        "Network " + "Theory of " + "Quantified " + "Reliability",
        "Quantified " + "Reliability",
        "NTQR " + "stands for",
    )

    offenders: list[str] = []
    for path in _TEXT_SURFACES:
        text = _read(path)
        for phrase in forbidden:
            if phrase in text:
                offenders.append(f"{path}: {phrase}")

    assert offenders == []


def test_ntqr_first_definition_is_source_backed() -> None:
    abstract = _compact(_read("manuscript/00_abstract.md"))
    readme = _compact(_read("README.md"))
    introduction = _compact(_read("manuscript/01_introduction.md"))

    assert "error-independent (EIE) evaluator" in abstract
    assert "logically consistent estimates" in abstract
    assert "without an answer key" in abstract
    assert "with no labels at all" in abstract
    assert "network" not in abstract.lower().split("`ntqr` package", 1)[0]

    assert "algebraic logic tools for unsupervised evaluation" in readme
    assert "unlabeled decision data" in readme
    assert "possible logically consistent" in readme

    assert "algebraic logic for unsupervised evaluation" in introduction
    assert "unlabeled decision data" in introduction
    assert "possible logically consistent combinations" in introduction


def test_ntqr_method_claims_include_upstream_caveats() -> None:
    methods = _compact(_read("manuscript/02_methods.md"))
    audit = _read("audits/claim_method_audit.md")

    for phrase in [
        "exactly three",
        "binary judges",
        "error-independent algebraic system",
        "up to two real solutions",
        "closest to the oracle",
        "crowd-right and crowd-wrong",
        "ground-truth-free\" is project shorthand",
    ]:
        assert phrase in methods

    for phrase in [
        "https://pypi.org/project/ntqr/",
        "https://ntqr.readthedocs.io/en/latest/",
        "ErrorIndependentEvaluation",
        "MajorityVotingEvaluation",
        "SupervisedEvaluation",
        "not as network theory",
    ]:
        assert phrase in audit

    assert "recovers the " + "competence" not in methods
    assert "estimates each judge's " + "accuracy" not in methods


def test_bibliography_and_citation_contract() -> None:
    bib = _read("manuscript/references.bib")
    prose_manuscript = "\n".join(
        _read(path)
        for path in [
            "manuscript/01_introduction.md",
            "manuscript/02_methods.md",
            "manuscript/03_results.md",
            "manuscript/04_discussion.md",
        ]
    )
    manuscript = "\n".join(
        _read(path)
        for path in [
            "manuscript/01_introduction.md",
            "manuscript/02_methods.md",
            "manuscript/03_results.md",
            "manuscript/04_discussion.md",
            "manuscript/99_references.md",
        ]
    )
    config = _read("manuscript/config.yaml")

    pandoc_keys = {
        key
        for key in re.findall(r"@([A-Za-z0-9_:-]+)", manuscript)
        if not key.startswith(("fig:", "tbl:", "sec:", "eq:")) and key != "pytest"
    }
    latex_keys = set(re.findall(r"\\citep\{([^}]+)\}", manuscript))
    nocite_keys = set(re.findall(r"\\nocite\{([^}]+)\}", manuscript))
    cited_keys = pandoc_keys | {
        key.strip()
        for group in latex_keys | nocite_keys
        for key in group.split(",")
        if key.strip()
    }
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert {
        "dawid1979",
        "ntqr08",
        "stone2011",
        "flanigan2021",
        "cohen1988",
        "efron1993",
        "holm1979",
        "raykar2010",
        "karger2014",
        "aristotle_politics",
        "aristotle_athenian",
        "aquinas_lots",
        "contarini1599",
        "montesquieu1748",
        "rousseau1762",
        "borda1781",
        "condorcet1785",
        "lamont2009",
        "peters1982",
        "cole1981",
        "lee2013",
        "pier2018",
        "wenneras1997",
        "ginther2011",
        "kaplan2008",
        "johnson2008",
        "fang2016elife",
        "merton1968",
        "neumark2019",
        "north2013",
        "grofman1983",
        "hong2004",
        "fishkin2009",
        "bollen2014",
        "fang2016",
        "avin2019",
        "arvan2025",
        "zheng2023",
        "shi2025",
        "tomkins2017",
        "helmer2017",
        "hardt2016",
        "bender2021",
    } <= cited_keys
    assert cited_keys <= bib_keys
    assert "link-citations: true" in config
    assert "1. Kerman" not in _read("manuscript/99_references.md")
    assert "The bibliography below is generated" in _read("manuscript/99_references.md")
    assert "<a href=" not in prose_manuscript
    assert r"</a>\citep" not in manuscript
    for raw_key in ["[@dawid1979]", "[@stone2011]", r"\citep{stone2011}"]:
        assert raw_key not in manuscript
    for linked_text in [
        "[Dawid and Skene (1979)](https://doi.org/10.2307/2346806)",
        "[Stone (2011)](https://doi.org/10.1093/acprof:oso/9780199756100.001.0001)",
        "[Flanigan et al. (2021)](https://doi.org/10.1038/s41586-021-03788-6)",
        "[Raykar et al. (2010)](https://jmlr.org/papers/v11/raykar10a.html)",
        "[Karger et al. (2014)](https://doi.org/10.1287/opre.2013.1235)",
        "[Aristotle (Politics)](https://topostext.org/work/100)",
        "[Aristotle (Athenian Constitution)](https://topostext.org/work/99)",
        "[Aquinas (Summa Theologiae II-II q.95 a.8)](https://www.newadvent.org/summa/3095.htm#article8)",
        "[Contarini (1599)](https://onlinebooks.library.upenn.edu/webbin/book/lookupid?key=olbp14764)",
        "[Montesquieu (1748)](https://oll.libertyfund.org/titles/montesquieu-complete-works-vol-1-the-spirit-of-laws)",
        "[Rousseau (1762)](https://oll.libertyfund.org/titles/cole-the-social-contract-and-discourses)",
        "[Borda (1781)](https://bibbase.org/network/publication/denbspborda-mmoiresurleslectionsauscrutin-1781)",
        "[Condorcet (1785)](https://archive.org/details/bub_gb_RzAVAAAAQAAJ)",
        "[Lamont (2009)](https://doi.org/10.4159/9780674054158)",
        "[Peters and Ceci (1982)](https://doi.org/10.1017/S0140525X00011183)",
        "[Cole et al. (1981)](https://doi.org/10.1126/science.7302566)",
        "[Lee et al. (2013)](https://doi.org/10.1002/asi.22784)",
        "[Pier et al. (2018)](https://doi.org/10.1073/pnas.1714379115)",
        "[Wennerås and Wold (1997)](https://doi.org/10.1038/387341a0)",
        "[Ginther et al. (2011)](https://doi.org/10.1126/science.1196783)",
        "[Kaplan et al. (2008)](https://doi.org/10.1371/journal.pone.0002761)",
        "[Johnson (2008)](https://doi.org/10.1073/pnas.0804538105)",
        "[Fang et al. (2016)](https://doi.org/10.7554/eLife.13323)",
        "[Merton (1968)](https://doi.org/10.1126/science.159.3810.56)",
        "[North and Fiske (2013)](https://doi.org/10.1177/0146167213480043)",
        "[Neumark et al. (2019)](https://doi.org/10.1086/701029)",
        "[Grofman et al. (1983)](https://doi.org/10.1007/BF00125672)",
        "[Hong and Page (2004)](https://doi.org/10.1073/pnas.0403723101)",
        "[Fishkin (2009)](https://global.oup.com/academic/product/when-the-people-speak-9780199604432)",
        "[Bollen et al. (2014)](https://doi.org/10.1002/embr.201338068)",
        "[Fang and Casadevall (2016)](https://doi.org/10.1128/mBio.00422-16)",
        "[Avin (2019)](https://doi.org/10.1016/j.shpsa.2018.11.006)",
        "[Arvan et al. (2025)](https://doi.org/10.1086/719117)",
        "[Zheng et al. (2023)](https://arxiv.org/abs/2306.05685)",
        "[Shi et al. (2025)](https://doi.org/10.18653/v1/2025.ijcnlp-long.18)",
        "[Tomkins et al. (2017)](https://doi.org/10.1073/pnas.1707323114)",
        "[Helmer et al. (2017)](https://doi.org/10.7554/eLife.21718)",
        "[Hardt et al. (2016)](https://papers.nips.cc/paper_files/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html)",
        "[Bender et al. (2021)](https://doi.org/10.1145/3442188.3445922)",
    ]:
        assert linked_text in manuscript
    for reference_id in [
        "10.2307/2346806",
        "https://pypi.org/project/ntqr/",
        "10.1093/acprof:oso/9780199756100.001.0001",
        "10.1038/s41586-021-03788-6",
        "10.1201/9780429246593",
        "https://www.jstor.org/stable/4615733",
        "https://jmlr.org/papers/v11/raykar10a.html",
        "10.1287/opre.2013.1235",
        "https://topostext.org/work/100",
        "https://topostext.org/work/99",
        "https://www.newadvent.org/summa/3095.htm#article8",
        "https://onlinebooks.library.upenn.edu/webbin/book/lookupid?key=olbp14764",
        "https://oll.libertyfund.org/titles/montesquieu-complete-works-vol-1-the-spirit-of-laws",
        "https://oll.libertyfund.org/titles/cole-the-social-contract-and-discourses",
        "https://bibbase.org/network/publication/denbspborda-mmoiresurleslectionsauscrutin-1781",
        "https://archive.org/details/bub_gb_RzAVAAAAQAAJ",
        "10.4159/9780674054158",
        "10.1017/S0140525X00011183",
        "10.1126/science.7302566",
        "10.1002/asi.22784",
        "10.1073/pnas.1714379115",
        "10.1038/387341a0",
        "10.1126/science.1196783",
        "10.1371/journal.pone.0002761",
        "10.1073/pnas.0804538105",
        "10.7554/eLife.13323",
        "10.1126/science.159.3810.56",
        "10.1086/701029",
        "10.1177/0146167213480043",
        "10.1007/BF00125672",
        "10.1073/pnas.0403723101",
        "https://global.oup.com/academic/product/when-the-people-speak-9780199604432",
        "10.1002/embr.201338068",
        "10.1128/mBio.00422-16",
        "10.1016/j.shpsa.2018.11.006",
        "10.1086/719117",
        "https://arxiv.org/abs/2306.05685",
        "10.18653/v1/2025.ijcnlp-long.18",
        "10.1073/pnas.1707323114",
        "10.7554/eLife.21718",
        "https://papers.nips.cc/paper_files/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html",
        "10.1145/3442188.3445922",
    ]:
        assert reference_id in bib


def test_manuscript_injector_copies_bibliography_to_render_surface() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/z_generate_manuscript_variables.py", "--check"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "orphans: 0" in result.stdout

    source = _REPO / "scripts" / "z_generate_manuscript_variables.py"
    assert 'glob("*.bib")' in source.read_text(encoding="utf-8")


def test_notation_table_is_labeled_and_has_symbol_column() -> None:
    methods = _read("manuscript/02_methods.md")

    assert "Table @tbl:notation" in methods
    assert "| Symbol | Surface / estimator | Unit | Aggregation and uncertainty | Source artifact |" in methods
    assert ": Notation and inferential units for the manuscript's reported statistics. {#tbl:notation}" in methods
    for symbol in [
        "$E, Q, \\pi$",
        "$\\widehat{\\theta}_{\\mathrm{EIE}}$",
        "$\\rho_{\\mathrm{NTQR}}$",
        "$d, n, \\mathrm{MDE}$",
    ]:
        assert symbol in methods
    assert "per-strategy EIE observations at fixed panel size" in methods
    assert "per-group observation budget" in methods
    assert "seeds-for-80" not in methods


def test_research_question_and_falsification_ledger_match_active_design() -> None:
    intro = _read("manuscript/01_introduction.md")
    methods = _read("manuscript/02_methods.md")
    audit = _read("audits/claim_method_audit.md")
    readme = _read("README.md")

    assert "Holding the expert population and the panel size fixed" not in intro
    assert "strategy" in intro
    assert "size of the panel" in intro
    assert "Size is a sampling knob, not guaranteed improvement" in intro

    assert "Table @tbl:falsification" in methods
    assert "| Claim family | Load-bearing assumption | Negative-control or falsification check | Current interpretation |" in methods
    for phrase in [
        "refutes the error-correlation explanation for it.",
        "The diagnostic works; the recovery slope is unresolved under global injection",
        "n-limited empirical companion evidence",
        "Nulls are split into resolved, underpowered, and well-powered design statements.",
    ]:
        assert phrase in methods
    assert "Main claim families have explicit falsification checks" in audit
    assert "over 96 seeds" in audit
    assert "over 24 seeds" not in audit
    assert "best->worst" not in readme
    assert "bunched bottom cluster" in readme


def test_math_notation_is_not_plaintext_in_manuscript_source() -> None:
    source = "\n".join(
        _read(path)
        for path in [
            "manuscript/02_methods.md",
            "manuscript/03_results.md",
            "manuscript/04_discussion.md",
        ]
    )

    assert "O(Q^3)" not in source.replace("$O(Q^3)$", "")
    assert "Q≤" not in source
    assert "R=3" not in source.replace("$R=3$", "")
    assert "$O(Q^3)$" in source
    assert "$Q\\leq" in source


def test_tolerance_and_power_figure_contracts_are_explicit() -> None:
    results = _read("manuscript/03_results.md")

    assert "TOLERANCE_SLOSS" not in results
    tolerance_caption = re.search(
        r"!\[(?P<caption>Oracle-referenced EIE error .+?)\]"
        r"\(\.\./output/figures/error_vs_correlation\.png\)",
        results,
    )
    assert tolerance_caption is not None
    for phrase in [
        "y-axis",
        "x-axis",
        "$\\rho_{\\mathrm{NTQR}}$",
        "OLS error-vs-correlation slope",
        "95% bootstrap CI",
        "not a resolved recovery-effect law",
        "output/data/independence_sweep.csv",
    ]:
        assert phrase in tolerance_caption.group("caption")

    power_caption = re.search(
        r"!\[(?P<caption>Analytic two-sample power .+?)\]"
        r"\(\.\./output/figures/power_vs_n\.png\)",
        results,
    )
    assert power_caption is not None
    for phrase in [
        "$1-\\beta$",
        "samples-per-group $n$",
        "Cohen's $d$",
        "chosen $\\alpha$",
        "MDE visual",
        "not retrospective observed power",
    ]:
        assert phrase in power_caption.group("caption")
    assert "standardized effect $d$" in results
    assert "per-group observation count $n$" in results
    assert "seeded trials across the active profile cells" in results
    assert "target power $1-\\beta$" in results
    assert "seed budgets" not in results
    assert "seeds-per-group" not in results


def test_section_titles_are_claim_oriented_and_current() -> None:
    introduction = _read("manuscript/01_introduction.md")
    results = _read("manuscript/03_results.md")
    methods = _read("manuscript/02_methods.md")
    discussion = _read("manuscript/04_discussion.md")
    source = "\n".join([introduction, methods, results, discussion])
    source_lines = {line.strip() for line in source.splitlines()}

    for heading in [
        "# Introduction: panel formation before blind evaluation",
        "## Blind evaluation begins before the estimator",
        "## Application review as the empirical stress test",
        "## Sortition from civic lotteries to evaluator sampling",
        "## Falsifiable claims and negative controls",
        "## Synthetic deterministic track: seeded panels, blind estimates, oracle scoring",
        "## Real-Ollama reviewer-panel track: single-model live companion",
        "### Pipeline: panel formation precedes no-answer-key estimation",
        "### Postdoctoral corpus: protected-attribute stress test",
        "### Reviewer profiles: expertise and age-bias prompts",
        "### Postdoc aggregation: analytical-vs-Gemma alignment",
        "## Synthetic deterministic results: controlled spine for H1-H4",
        "## Real-Ollama postdoctoral panel results: live H5 companion",
        "### Formation strategy sets the recovery floor",
        "### Sortition only separates when the confound rides on the balanced axis",
        "### NTQR beats majority voting only in selected regimes",
        "### Larger panels are a neutral sampling knob here",
        "### Global injected correlation is measurable but recovery-limited",
        "### Power budgets distinguish ranking from resolved contrasts",
        "### Companion diagnostics bound cost, correlation, fairness, and consistency",
        "### Gemma ranking asks the same sampling question under prompt labels",
        "### Same-bias panels expose age-conditioned recommendations",
        "### Analytical and Gemma cells stay juxtaposed, not pooled",
        "### Synthetic strategy ranking does not transfer to the live track",
        "## Formation strategy is the measured lever",
        "## Design-limited nulls remain results",
        "## Synthetic and live tracks operate at different inference levels",
        "## Data, code, and generated-artifact availability",
        "## Ethics, protected attributes, and competing interests",
    ]:
        assert heading in source_lines

    method_lines = methods.splitlines()
    assert method_lines.index(
        "## Synthetic deterministic track: seeded panels, blind estimates, oracle scoring"
    ) < method_lines.index(
        "## Real-Ollama reviewer-panel track: single-model live companion"
    )

    result_lines = results.splitlines()
    assert result_lines.index(
        "## Synthetic deterministic results: controlled spine for H1-H4"
    ) < result_lines.index(
        "## Real-Ollama postdoctoral panel results: live H5 companion"
    )
    assert results.index("\\clearpage\n\n## Real-Ollama postdoctoral panel results") > results.index(
        "### Companion diagnostics bound cost, correlation, fairness, and consistency"
    )

    for stale_heading in [
        "### Synthetic and live tracks do not pool evidence",
        "### Pre/post NTQR contrasts depend on regime",
        "### The correlation diagnostic resolves before the recovery slope",
        "### Diagnostics separate fairness and consistency from recovery",
        "### Alarm cost: an $O(Q^3)$ scaling limit",
        "### Local web explorer (non-publishing QA surface)",
        "## Panel formation sets the recovery ceiling",
        "## The sortition contrast is regime-dependent",
        "## Pre/post NTQR contrasts depend on regime",
        "## Larger panels help only conditionally",
        "## The correlation diagnostic resolves before the recovery slope",
        "## Power budgets separate ranking from resolved contrasts",
        "## Diagnostics separate fairness and consistency from recovery",
        "## Pipeline: form panels before estimating",
        "## Synthetic and live tracks do not pool evidence",
        "## Cross-family decorrelation (live, n-limited)",
        "## Cross-family decorrelation (live empirical)",
        "## Real-Ollama LLM track",
        "## Real-Ollama LLM results",
        "## Strategy ranking: the dominant lever",
        "## Representative vs ideological: where the contrast lives",
        "## Pre- and post-NTQR: separating baseline voting from recovery",
        "## Power curve: size is not a uniform knob",
        "## Error-correlation tolerance",
        "## Companion-track diagnostics",
        "## What the instrument shows",
        "## The nulls, stated explicitly",
        "## The two-track design",
        "## Synthetic deterministic track: a seeded, oracle-scored instrument",
        "## Real-Ollama single-model reviewer-panel track: the live empirical companion",
        "### Pipeline: form panels before estimating",
        "### Postdoctoral application corpus and protected-attribute stress test",
        "### Reviewer profiles and Gemma prompting",
        "### Postdoc panel aggregation and alignment",
        "## Synthetic deterministic results: the controlled spine (H1–H4)",
        "## Real-Ollama postdoctoral review panel results: the live companion (H5)",
        "### Panel formation sets the recovery ceiling",
        "### The sortition contrast is regime-dependent",
        "### NTQR improves on majority voting only in some regimes",
        "### Larger panels help only conditionally",
        "### Injected correlation is measurable; its recovery cost is not",
        "### Power budgets separate ranking from resolved contrasts",
        "### Companion diagnostics: alarm cost, correlation, fairness, consistency",
        "### Gemma reviewer-panel ranking mirrors the sampling question",
        "### Same-bias sampling exposes age-conditioned recommendations",
        "### Analytical and Gemma cells are juxtaposed rather than pooled",
        "### Strategy ranking does not transfer between tracks",
        "## Formation strategy is the main measured lever",
        "## Design-limited nulls are part of the result",
        "## Synthetic and live validation are reported at different inference levels",
        "## Data and code availability",
        "## Ethics and competing interests",
        "# Introduction",
        "## Ground-truth-free evaluation and panel formation",
        "## Why application review panels",
        "## Why sortition",
        "## Hypotheses",
    ]:
        assert stale_heading not in source_lines


def test_manuscript_live_track_is_gemma_postdoc_not_qwen_family_comparison() -> None:
    manuscript = "\n".join(
        _read(path)
        for path in [
            "manuscript/00_abstract.md",
            "manuscript/02_methods.md",
            "manuscript/03_results.md",
            "manuscript/04_discussion.md",
        ]
    )

    assert "qwen" not in manuscript.lower()
    assert "Qwen" not in manuscript
    assert "cross-family" not in manuscript
    assert "same-family" not in manuscript
    assert "model-family comparison" in manuscript
    assert "postdoctoral-review" in manuscript


def test_statistical_citations_and_power_grain_are_contract_bound() -> None:
    methods = _read("manuscript/02_methods.md")
    results = _read("manuscript/03_results.md")
    audit = _read("audits/claim_method_audit.md")

    assert "[Cohen (1988)](https://archive.org/details/statisticalpower0000cohe)" in methods
    assert "[Efron and Tibshirani (1993)](https://doi.org/10.1201/9780429246593)" in methods
    assert "[Holm (1979)](https://www.jstor.org/stable/4615733)" in methods
    assert "per-group observations at the analyzed trial/cell grain" in methods
    assert "per-group observation budgets" in results
    assert "per-group seed count" not in methods + results
    assert "seeds-per-group" not in methods + results
    assert "Bootstrap intervals are descriptive uncertainty summaries" in audit
    assert "Holm-adjusted significance counts" in audit


def test_alarm_power_claim_matches_saturated_curve() -> None:
    methods = _read("manuscript/02_methods.md")
    results = _read("manuscript/03_results.md")
    figure_source = _read("src/ntqr_allotment/figure_parts/power.py")

    joined = "\n".join([methods, results, figure_source])
    assert "strengthens as the panel grows" not in joined
    assert "monotone growth law" in results
    assert "saturated" in methods
    assert "Alarm firing rate" in figure_source


def test_discussion_reports_synthetic_and_live_tracks_at_distinct_levels() -> None:
    discussion = _compact(_read("manuscript/04_discussion.md"))

    assert "Synthetic and live tracks operate at different inference levels" in discussion
    assert "deterministic synthetic track is the controlled Results spine" in discussion
    assert "real-Ollama track is reported as separate live artifacts and empirical companion evidence" in discussion
    assert "required-live `{{POSTDOC_MODEL}}`" in discussion
    assert "Gemma substitutes for human reviewers" in discussion
    assert "does not validate the full synthetic regime grid" in discussion


def test_rendered_strategy_ranking_table_matches_numeric_order() -> None:
    text = _read("output/manuscript/03_results.md")
    table_match = re.search(
        r"\| Strategy \| Mean EIE error \| 95% CI \|\n"
        r"\| -------- \| -------------- \| ------ \|\n"
        r"(?P<rows>(?:\| .+\n){4})",
        text,
    )
    assert table_match is not None
    rows = []
    for line in table_match.group("rows").strip().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append((cells[0], float(cells[1])))

    assert rows[0][0] == "expertise threshold (far best)"
    assert {strategy for strategy, _ in rows[1:]} == {
        "representative sortition",
        "random selection",
        "single-bloc ideological selection",
    }
    assert rows[0][1] < min(mean for _, mean in rows[1:])
    bottom_three = [mean for _, mean in rows[1:]]
    assert max(bottom_three) - min(bottom_three) <= 0.002
    compact = _compact(text)
    assert "statistically indistinguishable from one another" in compact
    assert "| Rank | Strategy |" not in text


def test_rendered_size_direction_column_matches_its_own_cells() -> None:
    """NEGATIVE CONTROL for the size-direction inversion defect.

    The Direction word in the 'Larger panels help only conditionally' table must
    agree with the row's own Size 3 -> Size 6 cells. A hardcoded/stale direction
    (the original bug: 3 of 4 labels contradicted the numbers) fails here.
    """
    text = _read("output/manuscript/03_results.md")
    table_match = re.search(
        r"\| Strategy \| Size 3 \| Size 6 \| Pooled direction \|\n"
        r"\| -+ \| -+ \| -+ \| -+ \|\n"
        r"(?P<rows>(?:\| .+\n){4})",
        text,
    )
    assert table_match is not None
    checked = 0
    for line in table_match.group("rows").strip().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        size3, size6, direction = float(cells[1]), float(cells[2]), cells[3]
        if abs(size6 - size3) < 0.01:
            expected = "roughly flat"
        else:
            expected = "error rises" if size6 > size3 else "error falls"
        assert direction == expected, f"{cells[0]}: {size3}->{size6} should be {expected!r}, got {direction!r}"
        checked += 1
    assert checked == 4


def test_postdoc_item_count_is_tokenized_not_stale_literal() -> None:
    source = _read("manuscript/03_results.md")
    rendered = _read("output/manuscript/03_results.md")
    n_items = json.loads((_REPO / "output/data/postdoc_panel_results.json").read_text())[
        "config"
    ]["n_applications"]

    assert "400-item" not in source
    assert "400-item" not in rendered
    assert "{{POSTDOC_N_APPLICATIONS}} applications" in source
    assert f"{n_items} applications" in rendered


def test_discussion_size_effect_claim_matches_power_table() -> None:
    source = _read("manuscript/04_discussion.md")
    rendered = _read("output/manuscript/04_discussion.md")

    assert "two weaker strategies" not in source
    assert "two weaker strategies" not in rendered
    # The size claim must be the paired-test framing, not a direction read off the
    # noisy pooled curve. The earlier inverted claim and the over-resolved
    # per-strategy-direction claim are both forbidden.
    assert "Only single-bloc ideological selection improves" not in _compact(rendered)
    assert (
        "recovery error falls for representative sortition and is essentially flat"
        not in _compact(rendered)
    )
    assert "for most strategies modestly hurt" not in _compact(rendered)
    assert "degrade materially" not in _compact(rendered)
    assert "at most very slightly hurt" in _compact(rendered)
    assert "essentially neutral" in _compact(rendered)
    assert "paired" in _compact(rendered).lower()


def test_postdoc_alignment_statistic_is_reported() -> None:
    rendered = _read("output/manuscript/03_results.md")
    variables = json.loads((_REPO / "output/manuscript_variables.json").read_text())

    assert "POSTDOC_ALIGNMENT_RATE" in variables
    assert "resolved-cell sign-agreement rate" in _compact(rendered)
    assert variables["POSTDOC_ALIGNMENT_RATE"] in rendered


def test_manuscript_captions_include_source_statistic_and_caveat() -> None:
    source = _read("manuscript/03_results.md")

    required_caption_phrases = [
        "source `output/data/sweep_aggregated.csv`",
        "output/data/analytical_predictions.json",
        "stars mark descriptive 95% intervals excluding zero",
        "metric is `eie_mean - mv_mean`",
        "source `output/data/postdoc_panel_results.json`",
        "source `output/data/postdoc_panel_alignment.json`",
        "single-model live evidence is descriptive and n-limited",
        "not a human-review validation",
        "source `output/data/power_analysis.csv`",
        "Holm",
        # Cross-track ranking inversion (fig:trackinversion): source + statistic + caveat.
        "Cross-track strategy-ranking inversion at the matched three-seat grain",
        "ranks compared, magnitudes not pooled",
        "does not transfer to the live single-model panel",
    ]
    for phrase in required_caption_phrases:
        assert phrase in source
    assert "Claim: both intervals cross zero" not in source


def test_isa_records_latest_gate_and_no_stale_pending_headings() -> None:
    isa = _read("ISA.md")

    assert "Session 10" in isa
    assert "--require-live" in isa
    assert "num_predict=1" in isa
    assert "gemma3:4b" in isa
    assert "postdoctoral reviewer-panel" in isa
    assert "Session 16" in isa
    assert "10,584 current-hash live prompt evaluations" in isa
    assert "manuscript_contrast" in isa
    assert "analytical_predictions.json" in isa
    assert "rep_vs_ideo_heatmap.png" in isa
    assert "output/web/ntqr_explorer.html" in isa
    assert "output/figures/ntqr_cover.png" in isa
    assert "phase: complete" in isa
    assert "pending — workflow" not in isa
    assert "full default Gemma postdoctoral reviewer-panel profile (6 seeds, 48 reviewers, 72 applications)" not in isa
    assert "bounded 2-seed companion evidence" not in isa
    assert "Upcoming scoped work" in isa


def test_rendered_tex_has_title_page_then_toc_then_abstract() -> None:
    tex = _read("output/pdf/_combined_manuscript.tex")
    document_body = tex.split(r"\begin{document}", 1)[1]

    title_idx = document_body.index(r"\begin{titlepage}")
    toc_idx = document_body.index(r"\tableofcontents")
    abstract_idx = document_body.index(r"\section{Abstract}")

    assert title_idx < toc_idx < abstract_idx
    assert r"\maketitle" not in document_body
    assert "ORCID: 0000-0001-6232-9096" in tex
    assert "DOI: " in tex
    assert "10.5281/zenodo.21083779" in tex
    assert "Sortition Upstream of NTQR" in tex
    assert "How Panel Formation and Size Shape Ground-Truth-Free Evaluation" in tex
    assert "ntqr_cover.png" in tex


def test_project_margin_override_reaches_rendered_tex() -> None:
    preamble = _read("manuscript/preamble.md")
    tex = _read("output/pdf/_combined_manuscript.tex")

    assert r"\geometry{left=0.55in,right=0.55in,top=0.70in,bottom=0.70in}" in preamble
    assert r"\geometry{left=0.55in,right=0.55in,top=0.70in,bottom=0.70in}" in tex
    assert tex.index(r"\usepackage[margin=0.75in]{geometry}") < tex.index(
        r"\geometry{left=0.55in,right=0.55in,top=0.70in,bottom=0.70in}"
    )


def test_stego_workflow_documents_visual_equivalence_and_verifier() -> None:
    readme = _read("README.md")

    assert "The stego PDF is expected to look the same as the normal PDF" in readme
    assert "uv run python scripts/make_stego_pdf.py" in readme
    assert "uv run python scripts/verify_stego.py" in readme
    assert "source_pdf_sha256" in readme
    assert "not a visible watermark" in readme


def test_combined_html_has_linked_citations_not_plaintext_keys() -> None:
    html = _read("output/web/index.html")

    for key in [
        "dawid1979",
        "ntqr08",
        "stone2011",
        "flanigan2021",
        "cohen1988",
        "efron1993",
        "holm1979",
        "raykar2010",
        "lamont2009",
        "peters1982",
        "cole1981",
        "lee2013",
        "pier2018",
        "wenneras1997",
        "ginther2011",
        "kaplan2008",
        "merton1968",
        "neumark2019",
        "north2013",
        "grofman1983",
        "hong2004",
        "fishkin2009",
        "bollen2014",
        "fang2016",
        "arvan2025",
        "zheng2023",
        "tomkins2017",
        "helmer2017",
        "hardt2016",
        "bender2021",
    ]:
        assert f"[{key}]" not in html
        assert f"@{key}" not in html
    for href in [
        "https://doi.org/10.2307/2346806",
        "https://pypi.org/project/ntqr/",
        "https://doi.org/10.1201/9780429246593",
        "https://www.jstor.org/stable/4615733",
        "https://doi.org/10.1126/science.7302566",
        "https://doi.org/10.1073/pnas.1714379115",
        "https://doi.org/10.1371/journal.pone.0002761",
        "https://doi.org/10.1002/embr.201338068",
        "https://doi.org/10.1128/mBio.00422-16",
    ]:
        assert href in html


def test_cover_is_embedded_separately_from_manuscript_data_figures() -> None:
    aux = _read("output/pdf/_combined_manuscript.aux")
    tex = _read("output/pdf/_combined_manuscript.tex")

    assert aux.count(r"\newlabel{fig:") == 20  # +2: fig:blocphase, fig:dial (bloc-confound study)
    assert "ntqr_cover.png" in tex
    assert r"\caption" not in tex.split("ntqr_cover.png", 1)[0].rsplit(r"\begin{titlepage}", 1)[-1]


def test_size_penalty_mechanism_is_measured_not_asserted() -> None:
    source = _read("manuscript/03_results.md")
    rendered = _read("output/manuscript/03_results.md")

    assert "worse-conditioned trios faster than it adds" not in source
    assert "worse-conditioned trios faster than it adds" not in rendered
    assert "size-growing error-correlation mechanism" in source
    assert "{{MECH_CORR_VERDICT}}" in source
    assert "{{MECH_N_TRIO_RECORDS}}" in source
    assert "fig:triocond" in source
    assert "trio_conditioning.py" in rendered
    assert "does not identify a positive mechanism" in _compact(rendered)
    assert "not as an affirmative aggregation effect" in _compact(rendered)
