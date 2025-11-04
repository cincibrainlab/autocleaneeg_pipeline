#!/usr/bin/env python3
"""
Standalone BDF import diagnostic script.

Tests BioSemi BDF file import with channel name normalization and montage application.
Run with: uv run test_bdf_import.py

This script:
1. Loads the BDF file
2. Shows original channel names
3. Applies channel renaming logic
4. Applies biosemi64 montage
5. Checks channel positions
6. Validates montage application success
"""

import sys
from pathlib import Path

try:
    import mne
    import numpy as np
except ImportError:
    print("ERROR: MNE-Python not installed")
    print("Install with: uv pip install mne")
    sys.exit(1)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def test_bdf_import(bdf_file: str) -> None:
    """Test BDF import with detailed diagnostics."""

    bdf_path = Path(bdf_file)
    if not bdf_path.exists():
        print(f"ERROR: File not found: {bdf_file}")
        sys.exit(1)

    print_section("BDF FILE INFORMATION")
    print(f"File: {bdf_path}")
    print(f"Size: {bdf_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Step 1: Load BDF file
    print_section("STEP 1: LOADING BDF FILE")
    print("Loading with mne.io.read_raw_bdf()...")

    try:
        raw = mne.io.read_raw_bdf(
            input_fname=str(bdf_path),
            preload=True,
            stim_channel="auto",
            exclude=[],
            verbose=True,
        )
        print(f"✅ Successfully loaded BDF file")
    except Exception as e:
        print(f"❌ Failed to load BDF file: {e}")
        sys.exit(1)

    # Step 2: Analyze original channel names
    print_section("STEP 2: ORIGINAL CHANNEL NAMES")
    print(f"Total channels: {len(raw.ch_names)}")
    print(f"Channel types: {raw.get_channel_types()[:20]}...")
    print(f"\nFirst 20 channels:")
    for i, ch in enumerate(raw.ch_names[:20], 1):
        print(f"  {i:2d}. {ch}")

    # Check for EXG and Status channels
    eeg_channels = [ch for ch in raw.ch_names if ch not in ['Status'] and not ch.startswith('EXG')]
    exg_channels = [ch for ch in raw.ch_names if ch.startswith('EXG')]
    status_channels = [ch for ch in raw.ch_names if ch == 'Status']

    print(f"\nChannel breakdown:")
    print(f"  EEG channels: {len(eeg_channels)}")
    print(f"  EXG channels: {len(exg_channels)} {exg_channels}")
    print(f"  Status channels: {len(status_channels)} {status_channels}")

    # Check for prefix pattern
    prefixed = [ch for ch in eeg_channels if '_' in ch]
    print(f"\nChannels with prefix pattern: {len(prefixed)}")
    if prefixed:
        print(f"  Examples: {prefixed[:5]}")

    # Step 3: Build channel renaming map
    print_section("STEP 3: CHANNEL RENAMING LOGIC")

    rename_mapping = {}
    for ch_name in raw.ch_names:
        # Skip Status channel
        if ch_name == "Status":
            continue
        # BioSemi channels may have prefix like A1_, B5_, etc.
        if "_" in ch_name:
            prefix, standard_name = ch_name.split("_", 1)
            # Fix known case issues (e.g., Afz -> AFz)
            if standard_name == "Afz":
                standard_name = "AFz"
            rename_mapping[ch_name] = standard_name

    if rename_mapping:
        print(f"Found {len(rename_mapping)} channels to rename")
        print("\nRename mapping (first 10):")
        for i, (old, new) in enumerate(list(rename_mapping.items())[:10], 1):
            print(f"  {i:2d}. {old:15s} → {new}")

        # Check for case issues
        case_fixes = {old: new for old, new in rename_mapping.items() if old.split('_')[1] != new}
        if case_fixes:
            print(f"\nCase corrections: {len(case_fixes)}")
            for old, new in case_fixes.items():
                print(f"  {old} → {new}")
    else:
        print("⚠️  No channels need renaming (no prefix pattern found)")

    # Step 4: Apply channel renaming
    print_section("STEP 4: APPLYING CHANNEL RENAMING")

    if rename_mapping:
        try:
            raw.rename_channels(rename_mapping)
            print(f"✅ Renamed {len(rename_mapping)} channels successfully")

            print("\nNew channel names (first 20):")
            for i, ch in enumerate(raw.ch_names[:20], 1):
                print(f"  {i:2d}. {ch}")
        except Exception as e:
            print(f"❌ Failed to rename channels: {e}")
            sys.exit(1)
    else:
        print("⏭️  Skipping renaming (no changes needed)")

    # Step 5: Check montage availability
    print_section("STEP 5: CHECKING MONTAGE AVAILABILITY")

    montage_name = "biosemi64"
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        print(f"✅ Montage '{montage_name}' available")
        print(f"   Positions: {len(montage.get_positions()['ch_pos'])} channels")

        # Show montage channel names
        montage_channels = sorted(montage.get_positions()['ch_pos'].keys())
        print(f"\n   Montage channel names (first 20):")
        for i, ch in enumerate(montage_channels[:20], 1):
            print(f"     {i:2d}. {ch}")
    except Exception as e:
        print(f"❌ Failed to load montage: {e}")
        sys.exit(1)

    # Step 6: Check channel name matching
    print_section("STEP 6: CHECKING CHANNEL NAME MATCHING")

    eeg_channels_now = [ch for ch in raw.ch_names if ch != 'Status' and not ch.startswith('EXG')]
    montage_channels = set(montage.get_positions()['ch_pos'].keys())

    matching = [ch for ch in eeg_channels_now if ch in montage_channels]
    missing = [ch for ch in eeg_channels_now if ch not in montage_channels]

    print(f"EEG channels in file: {len(eeg_channels_now)}")
    print(f"Channels in montage: {len(montage_channels)}")
    print(f"Matching channels: {len(matching)}")
    print(f"Missing from montage: {len(missing)}")

    if missing:
        print(f"\nChannels not in montage:")
        for ch in missing[:10]:
            print(f"  - {ch}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if matching:
        print(f"\n✅ Good! {len(matching)} channels will get positions from montage")
    else:
        print(f"\n❌ ERROR: No matching channels between file and montage!")
        print(f"\nThis means montage application will fail.")

    # Step 7: Apply montage
    print_section("STEP 7: APPLYING MONTAGE")

    try:
        raw.set_montage(montage, match_case=False, on_missing="warn")
        print(f"✅ Montage application completed")

        # Check if montage was actually applied
        applied_montage = raw.get_montage()
        if applied_montage is None:
            print(f"❌ WARNING: Montage is None after application!")
        else:
            print(f"✅ Montage successfully applied")
    except Exception as e:
        print(f"❌ Failed to apply montage: {e}")
        sys.exit(1)

    # Step 8: Check channel positions
    print_section("STEP 8: CHECKING CHANNEL POSITIONS")

    info = raw.info
    positions = []
    nan_positions = []

    for ch_name in raw.ch_names:
        ch_idx = raw.ch_names.index(ch_name)
        ch_type = info['chs'][ch_idx]['kind']
        loc = info['chs'][ch_idx]['loc'][:3]  # Get x, y, z position

        if ch_type == mne.io.constants.FIFF.FIFFV_EEG_CH:
            if np.any(np.isnan(loc)):
                nan_positions.append(ch_name)
            else:
                positions.append((ch_name, loc))

    print(f"EEG channels with valid positions: {len(positions)}")
    print(f"EEG channels with NaN positions: {len(nan_positions)}")

    if positions:
        print(f"\nValid position examples (first 10):")
        for ch_name, loc in positions[:10]:
            print(f"  {ch_name:10s}: x={loc[0]:7.4f}, y={loc[1]:7.4f}, z={loc[2]:7.4f}")

    if nan_positions:
        print(f"\n❌ ERROR: Found {len(nan_positions)} channels with NaN positions:")
        for ch in nan_positions[:10]:
            print(f"  - {ch}")
        if len(nan_positions) > 10:
            print(f"  ... and {len(nan_positions) - 10} more")
        print(f"\n⚠️  This will cause bad channel detection to fail!")
    else:
        print(f"\n✅ All EEG channels have valid positions!")

    # Step 9: Drop EXG channels and filter to EEG
    print_section("STEP 9: DROPPING EXG CHANNELS AND FILTERING")

    channels_before = len(raw.ch_names)
    print(f"Channels before filtering: {channels_before}")

    # Drop EXG channels that don't have montage positions
    exg_to_exclude = [ch for ch in raw.ch_names if ch in [
        'LM', 'RM', 'LVE', 'RVE', 'LHE', 'RHE', 'EXG7', 'EXG8',
        'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6'
    ]]

    if exg_to_exclude:
        print(f"\nDropping {len(exg_to_exclude)} EXG channels:")
        for ch in exg_to_exclude:
            print(f"  - {ch}")
        raw.drop_channels(exg_to_exclude)
        print(f"✅ Dropped {len(exg_to_exclude)} EXG channels")

    try:
        raw.pick_types(eeg=True, stim=True, exclude=[])
        print(f"✅ Filtered to EEG + STIM channels")
        print(f"Channels after filtering: {len(raw.ch_names)}")
        print(f"Removed: {channels_before - len(raw.ch_names)} channels total")
    except Exception as e:
        print(f"❌ Failed to filter channels: {e}")

    # Step 10: Final validation
    print_section("STEP 10: FINAL VALIDATION")

    # Check montage
    final_montage = raw.get_montage()
    if final_montage is None:
        print(f"❌ FAIL: Montage is None")
        success = False
    else:
        print(f"✅ PASS: Montage is applied")
        success = True

    # Check for NaN positions in remaining channels
    nan_count = 0
    for ch_idx in range(len(raw.ch_names)):
        ch_type = raw.info['chs'][ch_idx]['kind']
        if ch_type == mne.io.constants.FIFF.FIFFV_EEG_CH:
            loc = raw.info['chs'][ch_idx]['loc'][:3]
            if np.any(np.isnan(loc)):
                nan_count += 1

    if nan_count > 0:
        print(f"❌ FAIL: {nan_count} EEG channels still have NaN positions")
        success = False
    else:
        print(f"✅ PASS: All EEG channels have valid positions")

    # Check channel count
    print(f"\n✅ Final channel count: {len(raw.ch_names)}")
    print(f"✅ Sample rate: {raw.info['sfreq']} Hz")
    print(f"✅ Duration: {raw.times[-1]:.2f} seconds")

    # Overall result
    print_section("OVERALL RESULT")

    if success:
        print("✅ ✅ ✅ SUCCESS! ✅ ✅ ✅")
        print("\nBDF import process completed successfully:")
        print("  ✓ File loaded")
        print("  ✓ Channels renamed")
        print("  ✓ Montage applied")
        print("  ✓ Positions validated")
        print("  ✓ Ready for pipeline processing")
        print("\nThis BDF file should now work with bad channel detection!")
    else:
        print("❌ ❌ ❌ FAILURE ❌ ❌ ❌")
        print("\nBDF import process failed. Review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    # BDF file to test
    bdf_file = "/Users/ernie/Downloads/Example EEGs/st101as.bdf"

    if len(sys.argv) > 1:
        bdf_file = sys.argv[1]

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BDF Import Diagnostic Script                             ║
║                                                                              ║
║  Tests BioSemi BDF import with channel name normalization                   ║
║  and montage application logic from the BDF plugins                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    try:
        test_bdf_import(bdf_file)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
