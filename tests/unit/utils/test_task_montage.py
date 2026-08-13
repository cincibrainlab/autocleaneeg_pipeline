from autoclean.utils.task_montage import (
    extract_montage_value,
    locate_montage_block,
    update_task_montage_source,
)


def test_update_task_montage_source_changes_only_value() -> None:
    source = """config = {
    "dataset_name": "Demo",
    "montage": {"enabled": True, "value": "GSN-HydroCel-128"},
    "filtering": {"enabled": True, "value": {"l_freq": 1}},
}
"""

    updated = update_task_montage_source(source, "GSN-HydroCel-129")

    assert '"value": "GSN-HydroCel-129"' in updated
    assert '"dataset_name": "Demo"' in updated
    assert '"filtering": {"enabled": True, "value": {"l_freq": 1}}' in updated


def test_locate_montage_block_handles_nested_formatting() -> None:
    source = """config = {
    "montage": {
        "enabled": True,
        "value": "GSN-HydroCel-128",
        "metadata": {"note": "literal } brace"},
    },
}
"""

    block = locate_montage_block(source)

    assert block is not None
    assert extract_montage_value(block.text) == "GSN-HydroCel-128"


def test_update_task_montage_source_replaces_none_value() -> None:
    source = """config = {
    "montage": {"enabled": False, "value": None},
}
"""

    updated = update_task_montage_source(source, "GSN-HydroCel-129")

    assert '"value": "GSN-HydroCel-129"' in updated
