from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from ntqr_allotment.stego import (
    build_stego_payload,
    capacity_bytes,
    embed_png_payload,
    extract_pdf_payload,
    extract_png_payload,
    render_stego_artifacts,
    verify_stego_artifacts,
    write_stego_pdf,
)


def test_png_payload_round_trips_losslessly(tmp_path: Path) -> None:
    source = tmp_path / "cover.png"
    output = tmp_path / "cover_stego.png"
    Image.new("RGB", (80, 80), "#d9ece8").save(source)
    payload = b'{"kind":"ntqr-provenance","ok":true}'

    written = embed_png_payload(source, output, payload)

    assert written == output
    assert extract_png_payload(output) == payload
    assert capacity_bytes((80, 80)) > len(payload)


def test_pdf_payload_round_trips_without_pdf_dependency(tmp_path: Path) -> None:
    source = tmp_path / "plain.pdf"
    output = tmp_path / "stego.pdf"
    payload = b'{"source":"ntqr","claim":"provenance-only"}'
    source.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nstartxref\n0\n%%EOF\n")

    write_stego_pdf(source, output, payload)

    pdf_bytes = output.read_bytes()
    assert pdf_bytes.endswith(b"%%EOF\n")
    assert pdf_bytes.rfind(b"% NTQR_STEGO_PAYLOAD_END") < pdf_bytes.rfind(b"startxref")
    assert extract_pdf_payload(output) == payload


def test_render_stego_artifacts_writes_manifest_and_extractable_payload(
    tmp_path: Path,
) -> None:
    _write_minimal_repo_outputs(tmp_path)

    paths = render_stego_artifacts(tmp_path)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    payload_from_png = extract_png_payload(paths.cover)
    payload_from_pdf = extract_pdf_payload(paths.pdf)

    assert payload_from_png == payload_from_pdf
    assert hashlib.sha256(payload_from_png).hexdigest() == manifest["payload_sha256"]
    assert manifest["stego_cover"] == "output/figures/ntqr_cover_stego.png"
    assert manifest["stego_pdf"] == "output/pdf/NTQR_allotment_combined_stego.pdf"
    assert manifest["payload"]["payload_caveat"].startswith("Plain JSON provenance")
    assert manifest["payload"]["manuscript_data_figure_count"] == 1


def test_verify_stego_artifacts_checks_both_carriers_and_current_pdf(
    tmp_path: Path,
) -> None:
    _write_minimal_repo_outputs(tmp_path)
    render_stego_artifacts(tmp_path)

    verification = verify_stego_artifacts(tmp_path)

    assert verification.png_payload_matches_manifest is True
    assert verification.pdf_payload_matches_manifest is True
    assert verification.source_pdf_hash_is_current is True
    assert "visually equivalent by design" in verification.visual_equivalence_note
    assert verification.to_dict()["source_pdf_hash_is_current"] is True


def test_verify_stego_artifacts_rejects_stale_source_pdf(tmp_path: Path) -> None:
    _write_minimal_repo_outputs(tmp_path)
    render_stego_artifacts(tmp_path)
    (tmp_path / "output/pdf/NTQR_allotment_combined.pdf").write_bytes(
        b"%PDF-1.4\nchanged\nstartxref\n0\n%%EOF\n"
    )

    with pytest.raises(ValueError, match="source_pdf_sha256 is stale"):
        verify_stego_artifacts(tmp_path)


def test_stego_payload_pins_source_hashes(tmp_path: Path) -> None:
    _write_minimal_repo_outputs(tmp_path)

    payload = build_stego_payload(
        tmp_path,
        tmp_path / "output/pdf/NTQR_allotment_combined.pdf",
        tmp_path / "output/figures/ntqr_cover.png",
    )

    assert payload["source_pdf_sha256"] == hashlib.sha256(
        (tmp_path / "output/pdf/NTQR_allotment_combined.pdf").read_bytes()
    ).hexdigest()
    assert payload["source_cover_sha256"] == hashlib.sha256(
        (tmp_path / "output/figures/ntqr_cover.png").read_bytes()
    ).hexdigest()
    assert payload["semantic_role"] == "local steganographic provenance variant"


def test_stego_payload_counts_rendered_manuscript_figures(tmp_path: Path) -> None:
    _write_minimal_repo_outputs(tmp_path)
    Image.new("RGB", (20, 20), "#663399").save(tmp_path / "output/figures/unused.png")
    (tmp_path / "output/pdf/_combined_manuscript.tex").write_text(
        "\\includegraphics{../figures/ntqr_cover.png}\n"
        "\\includegraphics{../figures/strategy_ranking.png}\n"
    )

    payload = build_stego_payload(
        tmp_path,
        tmp_path / "output/pdf/NTQR_allotment_combined.pdf",
        tmp_path / "output/figures/ntqr_cover.png",
    )

    assert payload["manuscript_data_figure_count"] == 1


def _write_minimal_repo_outputs(root: Path) -> None:
    (root / "output/figures").mkdir(parents=True)
    (root / "output/pdf").mkdir(parents=True)
    (root / "output/data").mkdir(parents=True)
    (root / "manuscript").mkdir()
    Image.new("RGB", (220, 220), "#fbf7ef").save(root / "output/figures/ntqr_cover.png")
    Image.new("RGB", (20, 20), "#336699").save(root / "output/figures/strategy_ranking.png")
    (root / "output/pdf/NTQR_allotment_combined.pdf").write_bytes(
        b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nstartxref\n0\n%%EOF\n"
    )
    (root / "output/data/cover_manifest.json").write_text('{"schema_version":1}\n')
    (root / "output/manuscript_variables.json").write_text('{"N_SEEDS":"12"}\n')
    (root / "manuscript/config.yaml").write_text("paper:\n  title: NTQR\n")
