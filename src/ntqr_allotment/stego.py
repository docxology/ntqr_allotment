from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image


def _generated_at() -> str:
    """Provenance timestamp, byte-deterministic when ``SOURCE_DATE_EPOCH`` is set.

    The embedded stego payload is hashed, so a wall-clock ``datetime.now`` made the
    cover PNG and PDF differ on every run, breaking the byte-determinism contract.
    Honoring the reproducible-builds ``SOURCE_DATE_EPOCH`` env var makes the payload
    reproducible; provenance otherwise rests on the embedded source SHA256 fields.
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        return datetime.fromtimestamp(int(epoch), UTC).isoformat(timespec="seconds")
    return datetime.now(UTC).isoformat(timespec="seconds")


HEADER_BYTES = 4
HEADER_FORMAT = ">I"
BITS_PER_PIXEL = 3
PDF_BEGIN = "% NTQR_STEGO_PAYLOAD_BEGIN"
PDF_END = "% NTQR_STEGO_PAYLOAD_END"
STEGO_COVER_FILENAME = "ntqr_cover_stego.png"
STEGO_PDF_FILENAME = "NTQR_allotment_combined_stego.pdf"
STEGO_MANIFEST_FILENAME = "stego_manifest.json"


@dataclass(frozen=True)
class StegoPaths:
    cover: Path
    pdf: Path
    manifest: Path


@dataclass(frozen=True)
class StegoVerification:
    cover: Path
    pdf: Path
    manifest: Path
    payload_sha256: str
    source_pdf_sha256: str
    png_payload_matches_manifest: bool
    pdf_payload_matches_manifest: bool
    source_pdf_hash_is_current: bool
    visual_equivalence_note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cover": str(self.cover),
            "pdf": str(self.pdf),
            "manifest": str(self.manifest),
            "payload_sha256": self.payload_sha256,
            "source_pdf_sha256": self.source_pdf_sha256,
            "png_payload_matches_manifest": self.png_payload_matches_manifest,
            "pdf_payload_matches_manifest": self.pdf_payload_matches_manifest,
            "source_pdf_hash_is_current": self.source_pdf_hash_is_current,
            "visual_equivalence_note": self.visual_equivalence_note,
        }


def capacity_bytes(size: tuple[int, int]) -> int:
    width, height = size
    return (width * height * BITS_PER_PIXEL) // 8 - HEADER_BYTES


def embed_png_payload(source: Path, output: Path, payload: bytes) -> Path:
    image = Image.open(source).convert("RGB")
    capacity = capacity_bytes(image.size)
    if len(payload) > capacity:
        raise ValueError(f"payload of {len(payload)} bytes exceeds capacity {capacity}")
    message = struct.pack(HEADER_FORMAT, len(payload)) + payload
    bits = _iter_bits(message)
    out = image.copy()
    pixels = out.load()
    if pixels is None:
        raise ValueError("image has no addressable pixel data")
    done = False
    for y in range(out.height):
        if done:
            break
        for x in range(out.width):
            r, g, b = pixels[x, y]
            channels = [r, g, b]
            for index in range(BITS_PER_PIXEL):
                bit = next(bits, None)
                if bit is None:
                    done = True
                    break
                channels[index] = (channels[index] & ~1) | bit
            pixels[x, y] = tuple(channels)
            if done:
                break
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output, format="PNG")
    return output


def extract_png_payload(path: Path) -> bytes:
    image = Image.open(path).convert("RGB")
    pixels = image.load()
    if pixels is None:
        raise ValueError("image has no addressable pixel data")
    header = _read_bytes(pixels, image.size, HEADER_BYTES)
    if len(header) != HEADER_BYTES:
        raise ValueError("image is too small to contain a payload header")
    length = struct.unpack(HEADER_FORMAT, header)[0]
    capacity = capacity_bytes(image.size)
    if length > capacity:
        raise ValueError(f"declared payload length {length} exceeds capacity {capacity}")
    return _read_bytes(pixels, image.size, HEADER_BYTES + length)[HEADER_BYTES:]


def write_stego_pdf(source_pdf: Path, output_pdf: Path, payload: bytes) -> Path:
    source_bytes = source_pdf.read_bytes()
    eof_at = source_bytes.rfind(b"%%EOF")
    if eof_at < 0:
        raise ValueError(f"{source_pdf} is missing %%EOF")
    startxref_at = source_bytes.rfind(b"startxref", 0, eof_at)
    if startxref_at < 0:
        raise ValueError(f"{source_pdf} is missing startxref")
    encoded = base64.b64encode(payload).decode("ascii")
    block = "\n".join(
        [PDF_BEGIN, *[f"% {encoded[i:i + 96]}" for i in range(0, len(encoded), 96)], PDF_END, ""]
    ).encode("ascii")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.write_bytes(source_bytes[:startxref_at] + block + source_bytes[startxref_at:])
    return output_pdf


def extract_pdf_payload(path: Path) -> bytes:
    text = path.read_bytes().decode("latin-1", errors="ignore")
    match = re.search(
        re.escape(PDF_BEGIN) + r"\n(?P<body>(?:% [A-Za-z0-9+/=]+\n)+)" + re.escape(PDF_END),
        text,
    )
    if not match:
        raise ValueError(f"{path} has no NTQR stego payload")
    encoded = "".join(line[2:] for line in match.group("body").splitlines())
    return base64.b64decode(encoded)


def build_stego_payload(repo_root: Path, source_pdf: Path, source_cover: Path) -> dict[str, Any]:
    root = Path(repo_root)
    data_dir = root / "output" / "data"
    figures_dir = root / "output" / "figures"
    manuscript_dir = root / "manuscript"
    return {
        "schema_version": 1,
        "semantic_role": "local steganographic provenance variant",
        "generated_at": _generated_at(),
        "source_pdf": _rel(root, source_pdf),
        "source_pdf_sha256": _file_hash(source_pdf),
        "source_cover": _rel(root, source_cover),
        "source_cover_sha256": _file_hash(source_cover),
        "manuscript_config_sha256": _file_hash(manuscript_dir / "config.yaml"),
        "manuscript_variables_sha256": _file_hash(data_dir.parent / "manuscript_variables.json"),
        "cover_manifest_sha256": _file_hash(data_dir / "cover_manifest.json"),
        "manuscript_data_figure_count": _manuscript_data_figure_count(root, figures_dir),
        "payload_caveat": (
            "Plain JSON provenance embedded in a lossless PNG LSB layer and mirrored "
            "inside the PDF as comments; not encryption, not a scientific result, and "
            "not intended to carry confidential data."
        ),
    }


def render_stego_artifacts(repo_root: Path) -> StegoPaths:
    root = Path(repo_root)
    source_cover = root / "output" / "figures" / "ntqr_cover.png"
    source_pdf = root / "output" / "pdf" / "NTQR_allotment_combined.pdf"
    stego_cover = root / "output" / "figures" / STEGO_COVER_FILENAME
    stego_pdf = root / "output" / "pdf" / STEGO_PDF_FILENAME
    manifest = root / "output" / "data" / STEGO_MANIFEST_FILENAME
    payload = build_stego_payload(root, source_pdf, source_cover)
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    embed_png_payload(source_cover, stego_cover, payload_bytes)
    write_stego_pdf(source_pdf, stego_pdf, payload_bytes)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest": _rel(root, manifest),
                "stego_cover": _rel(root, stego_cover),
                "stego_pdf": _rel(root, stego_pdf),
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "payload": payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return StegoPaths(cover=stego_cover, pdf=stego_pdf, manifest=manifest)


def verify_stego_artifacts(repo_root: Path) -> StegoVerification:
    root = Path(repo_root)
    manifest = root / "output" / "data" / STEGO_MANIFEST_FILENAME
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    cover = root / str(data["stego_cover"])
    pdf = root / str(data["stego_pdf"])
    normal_pdf = root / str(data["payload"]["source_pdf"])
    expected_hash = str(data["payload_sha256"])
    png_payload = extract_png_payload(cover)
    pdf_payload = extract_pdf_payload(pdf)
    png_hash = hashlib.sha256(png_payload).hexdigest()
    pdf_hash = hashlib.sha256(pdf_payload).hexdigest()
    if png_payload != pdf_payload:
        raise ValueError("stego PNG and PDF payloads differ")
    if png_hash != expected_hash or pdf_hash != expected_hash:
        raise ValueError("stego payload hash does not match manifest")
    current_pdf_hash = _file_hash(normal_pdf)
    source_pdf_sha256 = str(data["payload"]["source_pdf_sha256"])
    if current_pdf_hash != source_pdf_sha256:
        raise ValueError("stego payload source_pdf_sha256 is stale")
    return StegoVerification(
        cover=cover,
        pdf=pdf,
        manifest=manifest,
        payload_sha256=expected_hash,
        source_pdf_sha256=source_pdf_sha256,
        png_payload_matches_manifest=True,
        pdf_payload_matches_manifest=True,
        source_pdf_hash_is_current=True,
        visual_equivalence_note=(
            "The stego PDF is visually equivalent by design; provenance is "
            "stored as extractable PDF comments and mirrored in the PNG LSB layer."
        ),
    )


def _iter_bits(data: bytes):
    for byte in data:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1


def _read_bytes(pixels: Any, size: tuple[int, int], count: int) -> bytes:
    width, height = size
    out = bytearray()
    current = 0
    filled = 0
    needed = count * 8
    seen = 0
    for y in range(height):
        if seen >= needed:
            break
        for x in range(width):
            for channel in pixels[x, y]:
                current = (current << 1) | (channel & 1)
                filled += 1
                seen += 1
                if filled == 8:
                    out.append(current)
                    current = 0
                    filled = 0
                if seen >= needed:
                    break
            if seen >= needed:
                break
    return bytes(out)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manuscript_data_figure_count(root: Path, figures_dir: Path) -> int:
    excluded = {"ntqr_cover.png", STEGO_COVER_FILENAME}
    tex_path = root / "output" / "pdf" / "_combined_manuscript.tex"
    if tex_path.exists():
        tex = tex_path.read_text(encoding="utf-8", errors="ignore")
        names = set(re.findall(r"figures/([^}]+\.png)", tex))
        return len(names - excluded)
    return sum(1 for path in figures_dir.glob("*.png") if path.name not in excluded)


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


__all__ = [
    "StegoPaths",
    "build_stego_payload",
    "capacity_bytes",
    "embed_png_payload",
    "extract_pdf_payload",
    "extract_png_payload",
    "render_stego_artifacts",
    "StegoVerification",
    "verify_stego_artifacts",
    "write_stego_pdf",
]
