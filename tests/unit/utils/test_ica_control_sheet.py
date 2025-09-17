import pandas as pd

from autoclean.utils.ica_control_sheet import update_pending_rows


def _make_row(**kwargs):
    defaults = {
        "original_file": "raw.fif",
        "ica_fif": "ica.fif",
        "auto_initial": "0,3,7",
        "final_removed": "0,3,7",
        "manual_add": "",
        "manual_drop": "",
        "status": "auto",
        "last_run_iso": "",
    }
    defaults.update(kwargs)
    return defaults


def test_add_components_updates_row():
    df = pd.DataFrame([_make_row(manual_add="2", status="pending")])
    updated = update_pending_rows(df.copy())
    row = updated.iloc[0]
    assert row.final_removed == "0,2,3,7"
    assert row.manual_add == ""
    assert row.manual_drop == ""
    assert row.status == "applied"
    assert row.last_run_iso


def test_drop_components_updates_row():
    df = pd.DataFrame([
        _make_row(final_removed="0,2,3,7", manual_drop="2", status="pending")
    ])
    updated = update_pending_rows(df.copy())
    row = updated.iloc[0]
    assert row.final_removed == "0,3,7"
    assert row.status == "applied"
    assert row.manual_drop == ""


def test_no_changes_sets_auto_status():
    df = pd.DataFrame([_make_row(status="pending")])
    updated = update_pending_rows(df.copy())
    row = updated.iloc[0]
    assert row.final_removed == "0,3,7"
    assert row.status == "auto"
    assert row.manual_add == ""
    assert row.manual_drop == ""
