"""Regression tests for issue #226: ecg/heart IC flag mismatch."""

from unittest.mock import MagicMock, patch

import pandas as pd

from autoclean.configkit.schema import _ic_flags_valid
from autoclean.functions.ica.ica_processing import (
    _icalabel_to_dataframe,
    apply_ica_component_rejection,
    normalize_ic_type,
)


def test_normalize_ic_type_aliases():
    assert normalize_ic_type("heart") == "ecg"
    assert normalize_ic_type("cardiac") == "ecg"
    assert normalize_ic_type("Ecg") == "ecg"
    assert normalize_ic_type("ECG") == "ecg"
    assert normalize_ic_type("eye") == "eog"
    assert normalize_ic_type("channel_noise") == "ch_noise"
    assert normalize_ic_type("brain") == "brain"
    assert normalize_ic_type(None) is None


def test_ic_flag_schema_accepts_case_variants():
    assert _ic_flags_valid(["Heart", "ECG", "CARDIAC", " channel_noise "])


def test_icalabel_to_dataframe_normalizes_at_source():
    ica = MagicMock()
    ica._ica_names = ["ICA000", "ICA001"]
    ica.n_components_ = 2
    ica.labels_ = {"Ecg": [0], "Brain": [1]}
    ica.labels_scores_ = pd.DataFrame([[0.95, 0.05], [0.01, 0.99]])

    labels_df = _icalabel_to_dataframe(ica)

    assert labels_df.loc[0, "ic_type"] == "ecg"
    assert labels_df.loc[1, "ic_type"] == "brain"


def test_heart_config_rejects_ecg_labeled_component():
    """A component labeled 'ecg' by the classifier must be rejected when
    the task config uses the historically-documented 'heart' flag."""
    labels_df = pd.DataFrame({"ic_type": ["ecg", "brain"], "confidence": [0.95, 0.99]})
    raw = MagicMock()
    ica = MagicMock()
    ica.exclude = []
    ica.copy.return_value = ica
    ica.apply.return_value = raw

    _, rejected = apply_ica_component_rejection(
        raw=raw,
        ica=ica,
        labels_df=labels_df,
        ic_flags_to_reject=["heart", "muscle"],
        ic_rejection_threshold=0.8,
    )

    assert rejected == [0]


def test_capitalized_ecg_label_still_matches_lowercase_config():
    labels_df = pd.DataFrame({"ic_type": ["Ecg"], "confidence": [0.9]})
    raw = MagicMock()
    ica = MagicMock()
    ica.exclude = []
    ica.copy.return_value = ica
    ica.apply.return_value = raw

    _, rejected = apply_ica_component_rejection(
        raw=raw,
        ica=ica,
        labels_df=labels_df,
        ic_flags_to_reject=["ecg"],
        ic_rejection_threshold=0.8,
    )

    assert rejected == [0]


def test_heart_override_applies_to_ecg_labeled_component():
    labels_df = pd.DataFrame({"ic_type": ["ecg"], "confidence": [0.7]})
    raw = MagicMock()
    ica = MagicMock()
    ica.exclude = []
    ica.copy.return_value = ica
    ica.apply.return_value = raw

    _, rejected = apply_ica_component_rejection(
        raw=raw,
        ica=ica,
        labels_df=labels_df,
        ic_flags_to_reject=["ecg"],
        ic_rejection_threshold=0.8,
        ic_rejection_overrides={"heart": 0.5},
    )

    assert rejected == [0]


def test_duplicate_normalized_overrides_emit_warning():
    labels_df = pd.DataFrame({"ic_type": ["ecg"], "confidence": [0.7]})
    raw = MagicMock()
    ica = MagicMock()
    ica.exclude = []
    ica.copy.return_value = ica
    ica.apply.return_value = raw

    with patch("autoclean.functions.ica.ica_processing.message") as mock_message:
        _, rejected = apply_ica_component_rejection(
            raw=raw,
            ica=ica,
            labels_df=labels_df,
            ic_flags_to_reject=["heart"],
            ic_rejection_threshold=0.8,
            ic_rejection_overrides={"heart": 0.6, "ECG": 0.5},
        )

    assert rejected == [0]
    mock_message.assert_any_call(
        "warning",
        "Multiple ICA rejection threshold overrides normalize to "
        "'ecg'. Using the last configured value.",
    )
