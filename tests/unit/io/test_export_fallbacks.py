"""Tests for export fallbacks used by large EEGLAB outputs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from autoclean.io.export import save_epochs_to_set, save_raw_to_set


def _autoclean_dict(tmp_path: Path) -> dict:
    return {
        "run_id": "run-001",
        "unprocessed_file": str(tmp_path / "sample.set"),
        "stage_dir": tmp_path / "stage",
        "move_flagged_files": True,
    }


@patch("autoclean.io.export.manage_database_conditionally")
def test_save_raw_to_set_falls_back_to_fif_for_eeglab_size_error(
    mock_manage_database, tmp_path: Path
) -> None:
    """Raw export should save FIF when EEGLAB rejects MATLAB v5 matrix size."""
    raw = MagicMock()
    raw.n_times = 100
    raw.ch_names = ["Cz", "Pz"]
    raw.info = {"sfreq": 100.0}
    raw.times = [0.0, 0.99]
    raw.export.side_effect = RuntimeError("Matlab 5 file format cannot store matrix")

    result = save_raw_to_set(raw, _autoclean_dict(tmp_path), stage="post_import")

    assert result.suffix == ".fif"
    raw.save.assert_called_once_with(result, overwrite=True, fmt="single")
    raw.export.assert_called_once()
    assert mock_manage_database.call_count == 2


@patch("autoclean.io.export.manage_database_conditionally")
@patch("autoclean.io.export.sio.savemat")
@patch("autoclean.io.export.sio.loadmat", return_value={})
@patch("autoclean.io.export.save_epochs_to_set_chunked")
def test_save_epochs_to_set_falls_back_to_chunked_for_eeglab_size_error(
    mock_chunked,
    mock_loadmat,
    mock_savemat,
    mock_manage_database,
    tmp_path: Path,
) -> None:
    """Epoch export should chunk when EEGLAB rejects MATLAB v5 matrix size."""
    chunk_path = tmp_path / "stage" / "01_clean_epochs" / "sample_clean_epochs_epo.set"
    mock_chunked.return_value = [chunk_path]

    epochs = MagicMock()
    epochs.__len__.return_value = 2
    epochs.ch_names = ["Cz", "Pz"]
    epochs.times = [0.0, 0.1, 0.2]
    epochs.info = {"sfreq": 100.0}
    epochs.metadata = None
    epochs.events = []
    epochs.export.side_effect = RuntimeError("Matrix too large for Matlab 5")

    result = save_epochs_to_set(
        epochs, _autoclean_dict(tmp_path), stage="post_clean_epochs"
    )

    assert result.suffix == ".set"
    mock_chunked.assert_called_once()
    mock_loadmat.assert_called_once_with(chunk_path)
    mock_savemat.assert_called_once()
    assert mock_manage_database.call_count == 2
