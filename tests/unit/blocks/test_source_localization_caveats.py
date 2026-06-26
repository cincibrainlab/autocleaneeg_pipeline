"""Tests for source-localization caveat documentation helpers."""

from pathlib import Path

from autoclean.blocks.analysis.source_localization.caveats import (
    SOURCE_LOCALIZATION_SHORT_NOTE,
    SOURCE_LOCALIZATION_UNITS,
    source_localization_readme_text,
    source_localization_report_notes,
    write_source_localization_readme,
)


def test_source_localization_report_note_is_concise_and_unit_aware():
    notes = source_localization_report_notes()

    assert notes == [SOURCE_LOCALIZATION_SHORT_NOTE]
    assert "template-based" in notes[0]
    assert "native source units" in notes[0]
    assert "scalp microvolts" in notes[0]


def test_source_localization_readme_records_method_params_and_caveats(tmp_path: Path):
    params = {
        "method": "MNE",
        "lambda2": 0.111,
        "montage": "GSN-HydroCel-129",
        "units": SOURCE_LOCALIZATION_UNITS,
    }

    readme_path = write_source_localization_readme(tmp_path, params)
    text = readme_path.read_text(encoding="utf-8")

    assert readme_path.name == "source_localization_README.md"
    assert "Method: `MNE`" in text
    assert "Lambda2: `0.111`" in text
    assert "Montage: `GSN-HydroCel-129`" in text
    assert f"Output units: `{SOURCE_LOCALIZATION_UNITS}`" in text
    assert "inverse estimate" in text
    assert "ROI polarity/sign" in text


def test_source_localization_readme_text_has_practical_guidance():
    text = source_localization_readme_text({})

    assert "template-based ROI source estimates" in text
    assert "Avoid describing them as precise anatomical localization" in text
