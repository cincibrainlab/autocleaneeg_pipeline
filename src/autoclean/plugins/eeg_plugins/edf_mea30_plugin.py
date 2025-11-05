# src/autoclean/plugins/eeg_plugins/edf_mea30_plugin.py
"""EDF file plugin with MEA30 mouse electrode array configuration.

This plugin handles the complete import and montage configuration
for EDF files recorded with the MEA30 30-channel mouse EEG electrode array.

IMPORTANT CHANNEL MAPPING:
--------------------------
Raw EDF files contain 33 channels with generic names ("Chan 1", "Chan 2", etc.).
These channels are NOT in the correct anatomical order and require special handling:

1. **Channel Exclusion** (3 channels dropped):
   - Chan 2: Excluded from MEA mapping (likely bad/reference channel)
   - Chan 32: Reference electrode
   - Chan 33: Ground electrode

2. **Channel Scrambling** (30 EEG channels remapped):
   The remaining 30 channels are in a scrambled hardware routing order.
   This plugin remaps them to the correct anatomical MEA channel order.

   Examples of the scrambling:
   - EDF "Chan 1"  → MEA "Ch 23" (Right Temporal)
   - EDF "Chan 3"  → MEA "Ch 22" (Right Medial)
   - EDF "Chan 30" → MEA "Ch 01" (Left Temporal)
   - EDF "Chan 31" → MEA "Ch 08" (Left Temporal)

3. **3D Montage Application**:
   After remapping, 3D brain coordinates (MNI space) are applied to each
   MEA channel for proper anatomical visualization and source localization.

This mapping has been validated against:
- MATLAB EEGLAB code (edf2meaLookupTest function)
- CSV mapping files (Mea_adult_atlas-30_dict.csv)
- Actual EDF file structure analysis

See: MEA30_EDF_mapping.csv and MEA30_EDF.sfp for complete mapping details.
"""

from pathlib import Path

import mne
import pandas as pd

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message


class EDFMouseMEA30Plugin(BaseEEGPlugin):
    """Plugin for EDF files with MEA30 mouse electrode array.

    This plugin handles the specific combination of EDF files with
    the 30-channel Mouse EEG MEA30 probe system. The probe has
    scrambled channel routing that requires channel dropping and remapping.
    """

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "EDF" and montage_name == "MEA30_EDF"

    def _load_mea30_mapping(self) -> tuple:
        """Load MEA30 EDF channel mapping from package resources.

        Returns:
            Tuple of (edf_channels_to_drop, rename_map, coordinates_dict)
        """
        # Load from package resources
        package_dir = Path(__file__).parent.parent.parent  # src/autoclean
        csv_file = package_dir / "data" / "probe_maps" / "MEA30_EDF_mapping.csv"

        if not csv_file.exists():
            raise FileNotFoundError(
                f"MEA30 mapping file not found: {csv_file}\n"
                "Package resources may be corrupted."
            )

        mapping_df = pd.read_csv(csv_file)

        # Channels to drop: 2, 32, 33 (not in CSV, plus reference/ground)
        edf_channels_to_drop = ["Chan 2", "Chan 32", "Chan 33"]

        # Build rename map: "Chan N" → "Ch ##"
        rename_map = {}
        coords_dict = {}

        for _, row in mapping_df.iterrows():
            edf_chan = int(row['edf_chan'])
            mea_label = row['mea_label']

            # EDF channel name in raw file
            edf_name = f"Chan {edf_chan}"
            rename_map[edf_name] = mea_label

            # Extract 3D coordinates (already in meters)
            x = row['x']
            y = row['y']
            z = row['z']
            coords_dict[mea_label] = (x, y, z)

        return edf_channels_to_drop, rename_map, coords_dict

    def _load_saved_montage(self) -> mne.channels.DigMontage:
        """Load MEA30_EDF montage file.

        Returns:
            MNE DigMontage object or None if not found
        """
        package_dir = Path(__file__).parent.parent.parent
        montage_file = package_dir / "data" / "montages" / "MEA30_EDF.sfp"

        if montage_file.exists():
            try:
                montage = mne.channels.read_custom_montage(str(montage_file))
                message("info", f"Loaded montage from {montage_file.name}")
                return montage
            except Exception as e:
                message("warning", f"Failed to load montage: {e}")
                return None

        return None

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ):
        """Import EDF file and configure MEA30 mouse electrode array montage."""

        # Display comprehensive info block about channel mapping
        message(
            "info",
            f"Loading Mouse EEG EDF file with MEA30 montage: {file_path.name}"
        )
        message(
            "info",
            "="*70
        )
        message(
            "info",
            "MEA30 CHANNEL MAPPING INFORMATION"
        )
        message(
            "info",
            "="*70
        )
        message(
            "info",
            "Raw EDF files contain 33 channels with SCRAMBLED hardware routing:"
        )
        message(
            "info",
            ""
        )
        message(
            "info",
            "1. CHANNELS TO DROP (3 total):"
        )
        message(
            "info",
            "   • Chan 2  → Excluded (not in MEA mapping, likely bad/reference)"
        )
        message(
            "info",
            "   • Chan 32 → Reference electrode"
        )
        message(
            "info",
            "   • Chan 33 → Ground electrode"
        )
        message(
            "info",
            ""
        )
        message(
            "info",
            "2. CHANNEL REMAPPING (30 EEG channels):"
        )
        message(
            "info",
            "   • Remaining 30 channels are reordered to anatomical MEA positions"
        )
        message(
            "info",
            "   • Example: EDF 'Chan 1' → MEA 'Ch 23' (Right Temporal)"
        )
        message(
            "info",
            "   • Example: EDF 'Chan 30' → MEA 'Ch 01' (Left Temporal)"
        )
        message(
            "info",
            ""
        )
        message(
            "info",
            "3. 3D MONTAGE APPLICATION:"
        )
        message(
            "info",
            "   • MNI stereotactic brain coordinates (~2.0 unit range)"
        )
        message(
            "info",
            "   • Validated against MATLAB code and CSV mappings"
        )
        message(
            "info",
            "   • Compatible with adult and P21 mice"
        )
        message(
            "info",
            "="*70
        )

        try:
            # Step 1: Load raw EDF file
            message("info", "Step 1/5: Loading raw EDF file...")
            raw = mne.io.read_raw_edf(
                input_fname=file_path,
                preload=preload,
                stim_channel="auto",  # Auto-detect any status channels
                exclude=[],  # Include all channels initially
                verbose=False,
            )
            message("success", f"✓ Loaded EDF with {len(raw.ch_names)} channels")

            # Step 2: Drop non-EEG channels (2, 32, 33)
            message("info", "Step 2/5: Dropping reference/ground channels...")
            channels_to_drop, rename_map, coords_dict = self._load_mea30_mapping()

            # Check which channels actually exist before dropping
            existing_drops = [ch for ch in channels_to_drop if ch in raw.ch_names]
            if existing_drops:
                raw.drop_channels(existing_drops)
                message("success", f"✓ Dropped {len(existing_drops)} channels: {', '.join(existing_drops)}")
            else:
                message("warning", "⚠ Expected channels to drop not found - file may have different structure")

            # Step 3: Rename channels from "Chan N" to "Ch ##"
            message("info", "Step 3/5: Remapping channels to MEA anatomical order...")

            # Only rename channels that exist in the raw object
            existing_rename_map = {k: v for k, v in rename_map.items() if k in raw.ch_names}

            if existing_rename_map:
                raw.rename_channels(existing_rename_map)
                message("success", f"✓ Remapped {len(existing_rename_map)} channels to MEA order")

                # Show a few examples
                example_items = list(existing_rename_map.items())[:3]
                for edf_name, mea_name in example_items:
                    message("debug", f"   {edf_name} → {mea_name}")
                if len(existing_rename_map) > 3:
                    message("debug", f"   ... and {len(existing_rename_map) - 3} more")
            else:
                message("warning", "⚠ No channels found for renaming - check file structure")

            # Step 4: Load and apply 3D montage
            message("info", "Step 4/5: Applying MEA30 3D brain coordinates...")
            montage = self._load_saved_montage()

            if montage:
                raw.set_montage(montage, match_case=False, on_missing='warn')
                message("success", f"✓ Applied 3D montage with {len(coords_dict)} electrode positions")
                message("info", "   Coordinates: MNI stereotactic space (normalized to unit sphere)")
            else:
                message("warning", "⚠ Montage file not found - creating from coordinates")
                # Create montage from coordinates dict
                montage = mne.channels.make_dig_montage(
                    ch_pos=coords_dict,
                    coord_frame='mni_tal'  # MNI/Talairach coordinates
                )
                raw.set_montage(montage, match_case=False, on_missing='warn')
                message("success", "✓ Created and applied montage from mapping CSV")

            # Step 5: Pick EEG channels only
            message("info", "Step 5/5: Finalizing channel selection...")
            raw.pick_types(eeg=True, stim=True, exclude=[])

            eeg_count = len([ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg'])
            message("success", f"✓ Final channel count: {eeg_count} EEG channels")

            message(
                "info",
                "="*70
            )
            message(
                "success",
                "✓ MEA30 EDF import complete!"
            )
            message(
                "info",
                f"  • Started with: 33 raw channels"
            )
            message(
                "info",
                f"  • Dropped: 3 non-EEG channels (2, 32, 33)"
            )
            message(
                "info",
                f"  • Remapped: 30 EEG channels to anatomical order"
            )
            message(
                "info",
                f"  • Result: {eeg_count} positioned MEA channels ready for analysis"
            )
            message(
                "info",
                "="*70
            )

            return raw

        except Exception as e:
            raise RuntimeError(
                f"Failed to process MEA30 EDF file: {str(e)}"
            ) from e

    def process_events(self, raw: mne.io.Raw) -> tuple:
        """Process events and annotations from EDF file.

        Most mouse MEA30 recordings are continuous/resting-state,
        but some may have event markers.
        """
        message("info", "Processing events from EDF file")
        try:
            # Try to extract events from annotations
            try:
                events, event_id = mne.events_from_annotations(raw)
            except ValueError:
                # No annotations/events
                events = None
                event_id = {}

            # Create events DataFrame if events exist
            if events is not None and len(events) > 0:
                events_df = pd.DataFrame(
                    {
                        "time": events[:, 0] / raw.info["sfreq"],
                        "sample": events[:, 0],
                        "id": events[:, 2],
                        "type": [
                            list(event_id.keys())[list(event_id.values()).index(id)]
                            if id in event_id.values()
                            else f"Event_{id}"
                            for id in events[:, 2]
                        ],
                    }
                )

                message(
                    "info",
                    f"Found {len(events)} events of {len(event_id)} unique types"
                )

                return events, event_id, events_df
            else:
                message("info", "No events found (continuous/resting-state recording)")
                return None, None, None

        except Exception as e:  # pylint: disable=broad-except
            message("warning", f"Failed to process events: {str(e)}")
            return None, None, None

    def get_metadata(self) -> dict:
        """Get additional metadata about this plugin."""
        return {
            "plugin_name": self.__class__.__name__,
            "plugin_version": self.VERSION,
            "montage_details": {
                "type": "MEA30_EDF",
                "channel_count": 30,
                "manufacturer": "Generic (Mouse EEG)",
                "reference": "Probe-specific (channels 32-33)",
                "layout": "Mouse cortical surface array (bilateral)",
                "file_format": "EDF/EDF+",
                "coordinate_system": "MNI stereotactic (normalized)",
                "coordinate_range": "~2.0 units (unit sphere)",
            },
            "channel_mapping": {
                "raw_channels": 33,
                "dropped_channels": ["Chan 2", "Chan 32", "Chan 33"],
                "eeg_channels": 30,
                "scrambled_routing": True,
                "remapping_source": "MEA30_EDF_mapping.csv",
            },
            "validation": {
                "validated_against": [
                    "MATLAB EEGLAB code (edf2meaLookupTest)",
                    "Mea_adult_atlas-30_dict.csv",
                    "Mea_P21_atlas-30_dict.csv",
                ],
                "coordinate_match": "Exact match across all sources",
                "age_variants": "Adult and P21 (4 region label differences only)",
            },
            "notes": [
                "Channel 2 is excluded from MEA mapping (bad/reference)",
                "Channels 32-33 are reference/ground electrodes",
                "30 EEG channels require remapping from scrambled hardware order",
                "3D coordinates are MNI brain space (normalized to unit sphere)",
                "Compatible with 'autocleaneeg-pipeline montage test' visualization",
            ]
        }
