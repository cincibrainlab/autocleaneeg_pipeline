"""Unit tests for pdf_extractor.py's summary-to-detail-page label matching.

Regression coverage for issue #294 (see also tests/unit/mixins/test_ica.py):
current-generation PDFs label both the summary table and detail pages with
the same 0-based IC index, so extract_ica_full's offset resolves to 0 and
every summary row matches its detail page directly. Older PDFs generated
before that fix have a 1-indexed summary table against 0-indexed detail
pages; extract_ica_full detects that constant +1 offset once for the whole
document and applies it uniformly, so those legacy reports keep working.

These tests also guard against a bug found while writing this coverage: an
earlier "direct match per row, else positional fallback" strategy could
silently pair a summary row with the *wrong* component's detail page,
because a 1-indexed summary label and a 0-indexed detail label can be the
same string (e.g. "IC1") while referring to different components.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

import pytest

from autoclean.api import pdf_extractor


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self, sort: bool = False) -> str:
        return self._text


class _FakeDoc:
    def __init__(self, pages: list[_FakePage]):
        self._pages = pages

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self._pages[index]

    def close(self) -> None:
        pass


@pytest.fixture
def fake_fitz(monkeypatch):
    """Install a stand-in `fitz` module so pdf_extractor's lazy `import fitz`
    resolves without requiring PyMuPDF to be installed, and let the test
    hand back whichever fake document `fitz.open` should return."""
    box: dict[str, Optional[_FakeDoc]] = {"doc": None}

    fake_module = types.ModuleType("fitz")
    fake_module.open = lambda _path: box["doc"]
    monkeypatch.setitem(sys.modules, "fitz", fake_module)

    return box


def _summary_page(rows: list[tuple[str, str, str, str]]) -> _FakePage:
    lines = ["ICA Components Summary", "Component", "Type", "Confidence", "Rejected"]
    for label, ic_type, confidence, rejected in rows:
        lines += [label, ic_type, confidence, rejected]
    return _FakePage("\n".join(lines))


def _detail_page(label: str) -> _FakePage:
    return _FakePage(f"{label} Topography\nSome plot content")


def test_current_generation_pdf_matches_directly(fake_fitz):
    """Summary and detail pages both 0-indexed: every row maps to its own
    detail page, with no offset applied."""
    pages = [
        _summary_page([("IC0", "brain", "0.95", "No"), ("IC1", "eog", "0.80", "Yes")]),
        _detail_page("IC0"),
        _detail_page("IC1"),
    ]
    fake_fitz["doc"] = _FakeDoc(pages)

    result = pdf_extractor.extract_ica_full(Path("/fake.pdf"))

    assert [c["component"] for c in result["components"]] == ["IC0", "IC1"]
    assert result["structure"]["detail_page_map"] == {"IC0": 1, "IC1": 2}


def test_legacy_pdf_applies_document_wide_offset(fake_fitz):
    """Legacy PDFs: 1-indexed summary ("IC1", "IC2") vs 0-indexed detail
    ("IC0", "IC1") for the same two components. Each summary row must
    resolve to *its own* component's detail page, not get scrambled by the
    "IC1" string existing in both label spaces for different components."""
    pages = [
        _summary_page([("IC1", "brain", "0.95", "No"), ("IC2", "eog", "0.80", "Yes")]),
        _detail_page("IC0"),
        _detail_page("IC1"),
    ]
    fake_fitz["doc"] = _FakeDoc(pages)

    result = pdf_extractor.extract_ica_full(Path("/fake.pdf"))

    assert [c["component"] for c in result["components"]] == ["IC1", "IC2"]
    page_map = result["structure"]["detail_page_map"]
    # "IC1" (1st summary row, real component 0) -> its own detail page (IC0).
    assert page_map["IC1"] == 1
    # "IC2" (2nd summary row, real component 1) -> its own detail page (IC1).
    assert page_map["IC2"] == 2


def test_legacy_pdf_five_components_stay_aligned(fake_fitz):
    """A larger legacy report should keep every row aligned to its own
    detail page, not just the first/last one."""
    rows = [(f"IC{i + 1}", "brain", "0.90", "No") for i in range(5)]
    pages = [_summary_page(rows)] + [_detail_page(f"IC{i}") for i in range(5)]
    fake_fitz["doc"] = _FakeDoc(pages)

    result = pdf_extractor.extract_ica_full(Path("/fake.pdf"))

    page_map = result["structure"]["detail_page_map"]
    assert page_map == {f"IC{i + 1}": i + 1 for i in range(5)}


def test_current_generation_pdf_tolerates_a_missing_detail_page(fake_fitz):
    """If one component's detail page failed to render, the rest should
    still match directly rather than shifting out of alignment."""
    pages = [
        _summary_page(
            [
                ("IC0", "brain", "0.95", "No"),
                ("IC1", "eog", "0.80", "Yes"),
                ("IC2", "other", "0.70", "No"),
            ]
        ),
        _detail_page("IC0"),
        # IC1's detail page is missing (render failure).
        _detail_page("IC2"),
    ]
    fake_fitz["doc"] = _FakeDoc(pages)

    result = pdf_extractor.extract_ica_full(Path("/fake.pdf"))

    page_map = result["structure"]["detail_page_map"]
    assert page_map == {"IC0": 1, "IC2": 2}
    assert "IC1" not in page_map


def test_legacy_pdf_missing_first_summary_row_still_detects_offset(fake_fitz):
    """If the summary row for the lowest-indexed component failed to parse
    (e.g. a malformed row), comparing only the two minimums would wrongly
    conclude offset=0 (min(summary)=2, min(detail)+1=1, mismatch). Overlap
    across the whole set should still detect the legacy 1-indexed offset
    from the remaining rows."""
    pages = [
        _summary_page(
            [
                # IC1's row (real component 0) is missing from the summary.
                ("IC2", "eog", "0.80", "Yes"),
                ("IC3", "other", "0.70", "No"),
                ("IC4", "brain", "0.60", "No"),
            ]
        ),
        _detail_page("IC0"),
        _detail_page("IC1"),
        _detail_page("IC2"),
        _detail_page("IC3"),
    ]
    fake_fitz["doc"] = _FakeDoc(pages)

    result = pdf_extractor.extract_ica_full(Path("/fake.pdf"))

    page_map = result["structure"]["detail_page_map"]
    assert page_map == {"IC2": 2, "IC3": 3, "IC4": 4}
