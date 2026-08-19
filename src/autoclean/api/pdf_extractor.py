"""PDF extraction module for AutoCleanEEG report files.

Extracts structured data (text tables, rendered page images) from
pipeline-generated PDFs. Each PDF type has its own extraction function.
Layout assumptions are isolated here — if the PDF format changes,
update these functions without touching API routes or frontend.

Dependencies: PyMuPDF (fitz)
"""

import base64
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Known ICA component types from ICLabel classifier
_KNOWN_TYPES = {"brain", "eog", "muscle", "ecg", "ch_noise", "line_noise", "other"}


# ── ICA Components PDF ─────────────────────────────────────────────


def extract_ica_summary(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract the ICA component summary table from summary pages.

    Scans ALL pages that contain the summary table pattern (not just the
    first two). The generator uses ceil(n_components / 20) summary pages.

    Returns a list of dicts:
    [{"component": "IC1", "type": "eog", "confidence": 0.96, "rejected": True}, ...]

    This is pure text extraction — layout-independent.
    """
    import fitz  # noqa: PLC0415 — lazy import

    components: list[dict[str, Any]] = []
    seen: set[str] = set()  # Deduplicate across pages
    doc = fitz.open(str(pdf_path))
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text()

            # Only scan pages that look like summary tables
            if "Component" not in text or "Confidence" not in text:
                continue

            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith("IC") and i + 3 < len(lines):
                    comp = line
                    candidate_type = lines[i + 1]
                    if candidate_type in _KNOWN_TYPES and comp not in seen:
                        try:
                            conf = float(lines[i + 2])
                        except ValueError:
                            conf = 0.0
                        rejected = lines[i + 3].strip().lower() == "yes"
                        components.append(
                            {
                                "component": comp,
                                "type": candidate_type,
                                "confidence": round(conf, 3),
                                "rejected": rejected,
                            }
                        )
                        seen.add(comp)
                        i += 4
                        continue
                i += 1
    finally:
        doc.close()
    return components


def render_ica_page(pdf_path: Path, page_num: int, dpi: int = 120) -> str:
    """Render a single page of the ICA PDF as a base64-encoded PNG.

    Args:
        pdf_path: Path to the ICA components PDF
        page_num: 0-based page index
        dpi: Resolution (120 is good for detail, 80 for thumbnails)

    Returns base64-encoded PNG string, or "" if page doesn't exist.
    """
    import fitz  # noqa: PLC0415

    doc = fitz.open(str(pdf_path))
    try:
        if page_num >= len(doc):
            return ""
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        png_data = pix.tobytes("png")
        return base64.b64encode(png_data).decode("utf-8")
    finally:
        doc.close()


def get_ica_page_count(pdf_path: Path) -> int:
    """Return the total number of pages in the ICA PDF."""
    import fitz  # noqa: PLC0415

    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def get_ica_structure(pdf_path: Path) -> dict[str, Any]:
    """Detect the structural layout of the ICA PDF by inspecting page text.

    Instead of hardcoding page indices, this scans each page's text to
    classify it as summary, topo grid, or per-component detail. This
    handles variable summary page counts (ceil(n_components / 20)),
    variable topo grid page counts, and legacy 1-indexed summary
    component names (IC1, IC2, ...) from PDFs generated before the
    generator was fixed to use 0-indexed names everywhere (issue #294).

    UPDATE HERE if the PDF generator changes its page layout or naming.
    """
    import fitz  # noqa: PLC0415

    doc = fitz.open(str(pdf_path))
    try:
        n_pages = len(doc)
        summary_pages: list[int] = []
        topo_grid_pages: list[int] = []
        detail_page_map: dict[str, int] = {}  # "IC1" -> page_num

        for page_idx in range(n_pages):
            page = doc[page_idx]
            text = page.get_text(sort=True)[:500]  # First 500 chars is enough

            # Summary pages contain "Component" + "Confidence" table headers
            if "Component" in text and "Confidence" in text and "Rejected" in text:
                summary_pages.append(page_idx)
                continue

            # Topo grid pages contain "Topographies Overview" or "Topograph...Batch"
            if "Topographies Overview" in text or (
                "Topograph" in text and "Batch" in text
            ):
                topo_grid_pages.append(page_idx)
                continue

            # Per-component detail pages contain "IC<N> Topography" or
            # "ICA Component IC<N>"
            ic_match = re.search(
                r"(?:ICA Component |^)\s*(IC\d+)\s+(?:Topography|Analysis)", text
            )
            if ic_match:
                comp_name = ic_match.group(1)
                detail_page_map[comp_name] = page_idx
                continue

            # Fallback: if text starts with "IC" followed by a digit and "Topography"
            ic_fallback = re.search(r"(IC\d+)\s*Topography", text)
            if ic_fallback:
                detail_page_map[ic_fallback.group(1)] = page_idx

        return {
            "total_pages": n_pages,
            "summary_pages": summary_pages,
            "topo_grid_page": topo_grid_pages[0] if topo_grid_pages else None,
            "topo_grid_pages": topo_grid_pages,
            "detail_start_page": (
                min(detail_page_map.values()) if detail_page_map else None
            ),
            "n_detail_pages": len(detail_page_map),
            "detail_page_map": detail_page_map,
        }
    finally:
        doc.close()


def extract_ica_full(pdf_path: Path) -> dict[str, Any]:
    """Combined extraction: summary + structure in a single PDF open.

    Opens the PDF once and does both text-based summary extraction and
    page classification in one pass. Use this from API endpoints instead
    of calling extract_ica_summary + get_ica_structure separately.

    Returns {"components": [...], "structure": {...}}.
    """
    import fitz  # noqa: PLC0415

    components: list[dict[str, Any]] = []
    seen: set[str] = set()
    summary_pages: list[int] = []
    topo_grid_pages: list[int] = []
    detail_page_map: dict[str, int] = {}

    doc = fitz.open(str(pdf_path))
    try:
        n_pages = len(doc)
        for page_idx in range(n_pages):
            page = doc[page_idx]
            text = page.get_text(sort=True)[:500]

            # Summary pages
            if "Component" in text and "Confidence" in text and "Rejected" in text:
                summary_pages.append(page_idx)
                # Extract component rows from this page
                full_text = page.get_text()
                lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
                i = 0
                while i < len(lines):
                    line = lines[i]
                    if line.startswith("IC") and i + 3 < len(lines):
                        candidate_type = lines[i + 1]
                        if candidate_type in _KNOWN_TYPES and line not in seen:
                            try:
                                conf = float(lines[i + 2])
                            except ValueError:
                                conf = 0.0
                            rejected = lines[i + 3].strip().lower() == "yes"
                            components.append(
                                {
                                    "component": line,
                                    "type": candidate_type,
                                    "confidence": round(conf, 3),
                                    "rejected": rejected,
                                }
                            )
                            seen.add(line)
                            i += 4
                            continue
                    i += 1
                continue

            # Topo grid pages
            if "Topographies Overview" in text or (
                "Topograph" in text and "Batch" in text
            ):
                topo_grid_pages.append(page_idx)
                continue

            # Per-component detail pages
            ic_match = re.search(
                r"(?:ICA Component |^)\s*(IC\d+)\s+(?:Topography|Analysis)", text
            )
            if ic_match:
                detail_page_map[ic_match.group(1)] = page_idx
                continue
            ic_fallback = re.search(r"(IC\d+)\s*Topography", text)
            if ic_fallback:
                detail_page_map[ic_fallback.group(1)] = page_idx

        # Build a mapping from summary component names to detail page numbers.
        # Current-generation PDFs use 0-indexed names on both summary and
        # detail pages (offset 0). Legacy PDFs generated before issue #294
        # was fixed have 1-indexed summary names (IC1, IC2, ...) against
        # 0-indexed detail names (IC0, IC1, ...) - a constant offset of 1.
        #
        # The offset is determined once for the whole document, not per row:
        # matching each summary name against detail_page_map directly is
        # unsafe when the two label spaces are one apart, because
        # "IC1"-as-1-indexed-summary and "IC1"-as-0-indexed-detail refer to
        # *different* components but are the same string - a per-row direct
        # match can silently pair the wrong pages together in that case.
        #
        # The offset is picked by whichever candidate (0 or 1) makes more of
        # the summary numbers overlap with the detail numbers, rather than
        # just comparing the two minimums: a single missing page can shift
        # the minimum on one side without changing the offset, and overlap
        # over the whole set is more resilient to that than one data point.
        # It can still tie (and falls back to offset 0) in the narrow case
        # where the missing page is specifically the lowest-indexed
        # component's *detail* page on a legacy (1-indexed-summary) PDF -
        # the label text alone doesn't carry enough information to
        # disambiguate that from a current-generation PDF in that case.
        summary_names = [c["component"] for c in components]

        def _numeric_suffix(label: str) -> Optional[int]:
            try:
                return int(label[2:])
            except (ValueError, IndexError):
                return None

        summary_num_set = {
            n
            for n in (_numeric_suffix(name) for name in summary_names)
            if n is not None
        }
        detail_num_set = {
            n
            for n in (_numeric_suffix(name) for name in detail_page_map)
            if n is not None
        }

        offset = 0
        if summary_num_set and detail_num_set:
            overlap_no_shift = len(summary_num_set & detail_num_set)
            overlap_shifted = len({n - 1 for n in summary_num_set} & detail_num_set)
            if overlap_shifted > overlap_no_shift:
                offset = 1

        component_page_map: dict[str, int] = {}
        for summary_name in summary_names:
            num = _numeric_suffix(summary_name)
            if num is None:
                continue
            detail_name = f"IC{num - offset}"
            if detail_name in detail_page_map:
                component_page_map[summary_name] = detail_page_map[detail_name]

        structure = {
            "total_pages": n_pages,
            "summary_pages": summary_pages,
            "topo_grid_page": topo_grid_pages[0] if topo_grid_pages else None,
            "topo_grid_pages": topo_grid_pages,
            "detail_start_page": (
                min(detail_page_map.values()) if detail_page_map else None
            ),
            "n_detail_pages": len(detail_page_map),
            "detail_page_map": component_page_map,  # Keyed by SUMMARY names
        }
    finally:
        doc.close()

    return {"components": components, "structure": structure}


# ── Run Report PDF (future) ────────────────────────────────────────


def render_report_page(pdf_path: Path, page_num: int, dpi: int = 150) -> str:
    """Render a page from the autoclean_report.pdf.

    Same logic as render_ica_page but at higher DPI since run reports
    have more text.
    """
    return render_ica_page(pdf_path, page_num, dpi=dpi)


def get_report_page_count(pdf_path: Path) -> int:
    """Return page count for the run report PDF."""
    return get_ica_page_count(pdf_path)
