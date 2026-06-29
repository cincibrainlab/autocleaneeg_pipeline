# src/autoclean/plugins/eeg_plugins/xdat_h32_plugin.py
"""NeuroNexus XDAT file plugin with MouseEEGv2 H32 montage configuration.

This plugin handles the complete import and montage configuration
for NeuroNexus XDAT files with the MouseEEGv2 H32 probe system.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message

# Optional Neo support
try:
    import neo
    import neo.io

    NEO_AVAILABLE = True
except ImportError:
    NEO_AVAILABLE = False


class XDATMouseH32Plugin(BaseEEGPlugin):
    """Plugin for NeuroNexus XDAT files with MouseEEGv2 H32 montage.

    This plugin handles the specific combination of NeuroNexus XDAT files
    with the 30-channel Mouse EEG v2 H32 probe system. The probe has
    scrambled pin routing that requires channel remapping.
    """

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "NEURONEXUS_XDAT" and montage_name == "MouseEEGv2_H32"

    def _load_saved_montage(self) -> tuple:
        """Load previously saved H32 montage file.

        Returns:
            Tuple of (montage, coords_dict) or None if not found
        """
        package_dir = Path(__file__).parent.parent.parent
        montage_file = package_dir / "data" / "montages" / "MouseEEGv2_H32.sfp"

        if montage_file.exists():
            try:
                # Read the montage file
                montage = mne.channels.read_custom_montage(str(montage_file))

                # Extract coords_dict from montage
                coords_dict = montage.get_positions()["ch_pos"]

                message("info", f"Loaded saved montage from {montage_file.name}")
                return montage, coords_dict

            except Exception as e:
                message("warning", f"Failed to load saved montage: {e}")
                return None

        return None

    def _load_h32_mapping(self) -> tuple:
        """Load H32 channel mapping from package resources.

        Returns:
            Tuple of (rename_map, coordinates_dict)
        """
        # Load from package resources
        package_dir = Path(__file__).parent.parent.parent  # src/autoclean
        csv_file = (
            package_dir / "data" / "probe_maps" / "MouseEEGv2H32_Import_Stage2.csv"
        )

        if not csv_file.exists():
            raise FileNotFoundError(
                f"H32 mapping file not found: {csv_file}\n"
                "Package resources may be corrupted."
            )

        stage2_df = pd.read_csv(csv_file)

        rename_map = {}
        coords_dict = {}

        for _, row in stage2_df.iterrows():
            chan_name = row["chan_name"]
            mea_name = row.get("mea_formatted_name", None)

            # Only process valid EEG channels
            if pd.notna(mea_name) and chan_name.startswith("pri_"):
                rename_map[chan_name] = mea_name

                # Extract coordinates (micrometers → meters)
                x = row["site_ctr_x"] / 1e6
                y = row["site_ctr_y"] / 1e6
                z = row["site_ctr_z"] / 1e6
                coords_dict[mea_name] = np.array([x, y, z])

        return rename_map, coords_dict

    def _load_raw_xdat(self, file_path: Path, preload: bool) -> mne.io.Raw:
        """Load raw XDAT data via Neo.

        Args:
            file_path: Path to XDAT file
            preload: Whether to preload data

        Returns:
            MNE Raw object
        """
        if not NEO_AVAILABLE:
            raise ImportError(
                "Neo package is required for XDAT files. "
                "Install with: pip install neo"
            )

        # Handle NeuroNexus file naming patterns
        stem = file_path.stem
        if stem.endswith("_data"):
            base_stem = stem[:-5]
            json_file = file_path.parent / f"{base_stem}.xdat.json"
        elif stem.endswith("_timestamp"):
            base_stem = stem[:-10]
            json_file = file_path.parent / f"{base_stem}.xdat.json"
        else:
            json_file = file_path.with_suffix(".xdat.json")

        reader_file = str(json_file) if json_file.exists() else str(file_path)

        # Load via Neo
        reader = neo.io.NeuroNexusIO(filename=reader_file)
        block = reader.read_block()
        segment = block.segments[0]

        # Extract all analog signals
        all_data = []
        ch_names = []
        ch_types_list = []
        sfreq = None

        for analog_signal in segment.analogsignals:
            sig_data = analog_signal.magnitude.T
            all_data.append(sig_data)

            if sfreq is None:
                sfreq = float(analog_signal.sampling_rate.magnitude)

            for ch_name in analog_signal.array_annotations.get("channel_names", []):
                ch_names.append(str(ch_name))

                # Determine channel type
                ch_name_lower = ch_name.lower()
                if "din" in ch_name_lower or "dout" in ch_name_lower:
                    ch_types_list.append("stim")
                elif "aux" in ch_name_lower:
                    ch_types_list.append("misc")
                else:
                    ch_types_list.append("eeg")

        # Stack channel data
        data = np.vstack(all_data)

        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types_list)
        raw = mne.io.RawArray(data, info)
        # MNE has no public setter for RawArray source filenames. Downstream
        # export/provenance code reads this private attribute, so keep this
        # assignment narrow and revisit it if MNE changes Raw._filenames.
        raw._filenames = [file_path]

        return raw

    def _save_montage(self, montage: mne.channels.DigMontage, coords_dict: dict):
        """Save montage to package resources for future use.

        Args:
            montage: MNE DigMontage object
            coords_dict: Dictionary of channel positions
        """
        # Save montage in package data directory
        package_dir = Path(__file__).parent.parent.parent  # src/autoclean
        montage_dir = package_dir / "data" / "montages"
        montage_dir.mkdir(parents=True, exist_ok=True)

        # Save in .sfp format (sensor positions file)
        montage_file = montage_dir / "MouseEEGv2_H32.sfp"

        try:
            # Write sensor positions file
            with open(montage_file, "w") as f:
                # Write header
                f.write("# NeuroNexus MouseEEGv2 H32 Probe Montage\n")
                f.write("# 30-channel mouse EEG probe with scrambled routing\n")
                f.write("# Coordinates in meters (probe dimensions: 0.64 x 0.83 mm)\n")
                f.write("# Generated by AutoClean XDAT H32 Plugin\n")

                # Write channel positions
                for ch_name, pos in coords_dict.items():
                    # .sfp format: label x y z
                    f.write(f"{ch_name}\t{pos[0]:.8f}\t{pos[1]:.8f}\t{pos[2]:.8f}\n")

            message(
                "success",
                f"Saved montage to {montage_file.relative_to(package_dir.parent)}",
            )

        except Exception as e:
            message("warning", f"Failed to save montage file: {e}")

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ):
        """Import NeuroNexus XDAT file and configure MouseEEGv2 H32 montage."""
        message(
            "info",
            f"Loading NeuroNexus XDAT file with MouseEEGv2 H32 montage: {file_path}",
        )

        try:
            # Step 1: Load raw XDAT file via Neo
            raw = self._load_raw_xdat(file_path, preload)
            message(
                "success",
                f"Successfully loaded XDAT file with {len(raw.ch_names)} channels",
            )

            # Step 2: Try to load saved montage first, fall back to CSV if not found
            saved_result = self._load_saved_montage()

            if saved_result:
                # Use saved montage
                montage, coords_dict = saved_result

                # Still need rename_map from CSV for channel renaming
                message("info", "Loading channel name mapping")
                rename_map, _ = self._load_h32_mapping()

                message("success", "Using cached montage file")
            else:
                # Generate from CSV
                message("info", "Generating montage from H32 mapping CSV")
                rename_map, coords_dict = self._load_h32_mapping()

                # Create montage from coordinates
                montage = mne.channels.make_dig_montage(
                    ch_pos=coords_dict,
                    coord_frame="unknown",  # Mouse probe uses local coordinate system
                )

                # Save for future use
                self._save_montage(montage, coords_dict)

            # Step 3: Rename channels (pri_N → EN)
            raw.rename_channels(rename_map)
            message("success", f"Renamed {len(rename_map)} channels (pri_N → EN)")

            # Step 4: Apply montage to raw object
            message("info", "Applying MouseEEGv2 H32 electrode positions")
            raw.set_montage(montage, on_missing="ignore")
            message(
                "success",
                f"Applied montage with {len(coords_dict)} electrode positions",
            )

            # Step 5: Demote EEG channels with no montage position to misc.
            # The montage covers only the 30 mouse electrodes; ref/ground/aux
            # channels mislabeled "eeg" keep NaN coords and break spatial steps
            # like bad-channel detection. Reclassify them so they're preserved
            # but excluded from EEG analyses.
            positionless = [
                ch["ch_name"]
                for ch in raw.info["chs"]
                if raw.get_channel_types([ch["ch_name"]])[0] == "eeg"
                and np.any(np.isnan(ch["loc"][:3]))
            ]
            if positionless:
                raw.set_channel_types({ch: "misc" for ch in positionless})
                message(
                    "warning",
                    f"Reclassified {len(positionless)} position-less EEG channel(s) "
                    f"to misc (not in montage): {positionless}",
                )

            # Step 6: Pick EEG and stimulus channels
            # Keep aux and digital I/O for complete data preservation
            raw.pick_types(eeg=True, stim=True, misc=True, exclude=[])

            eeg_count = len(
                [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == "eeg"]
            )
            message(
                "info", f"Selected {eeg_count} EEG channels + auxiliary/digital I/O"
            )

            # Note: Mouse EEG probe coordinates are in micrometers (< 1mm scale)
            # The montage validation will auto-detect this and scale visualization
            message(
                "info",
                "Mouse-scale coordinates detected (probe dimensions: ~0.64 × 0.83 mm). "
                "Visualizations will auto-scale for visibility.",
            )

            return raw

        except Exception as e:
            raise RuntimeError(
                f"Failed to process XDAT file with MouseEEGv2 H32 montage: {str(e)}"
            ) from e

    def process_events(self, raw: mne.io.Raw) -> tuple:
        """Process events from XDAT digital I/O channels.

        NeuroNexus XDAT files can have events in:
        - Digital input channels (din_0, din_1)
        - Digital output channels (dout_0, dout_1)
        """
        message("info", "Processing events from XDAT digital channels")
        try:
            # Try to extract events from annotations
            try:
                events, event_id = mne.events_from_annotations(raw)
            except ValueError:
                # No annotations, try digital channels directly
                events = None
                event_id = {}

            # If no events in annotations, check digital channels
            if events is None or len(events) == 0:
                stim_channels = [
                    ch
                    for ch in raw.ch_names
                    if raw.get_channel_types([ch])[0] == "stim"
                ]

                if stim_channels:
                    # Try first stimulus channel
                    events = mne.find_events(
                        raw, stim_channel=stim_channels[0], verbose=False
                    )

                    if events is not None and len(events) > 0:
                        # Create event_id from unique event codes
                        unique_codes = np.unique(events[:, 2])
                        event_id = {f"Event_{code}": code for code in unique_codes}
                    else:
                        message("warning", "No events found in digital channels")
                        return None, None, None
                else:
                    message("warning", "No stimulus/digital channels found")
                    return None, None, None

            # Create events DataFrame
            if events is not None and len(events) > 0:
                events_df = pd.DataFrame(
                    {
                        "time": events[:, 0] / raw.info["sfreq"],
                        "sample": events[:, 0],
                        "id": events[:, 2],
                        "type": [
                            (
                                list(event_id.keys())[list(event_id.values()).index(id)]
                                if id in event_id.values()
                                else f"Unknown-{id}"
                            )
                            for id in events[:, 2]
                        ],
                    }
                )

                message(
                    "info",
                    f"Found {len(events)} events of {len(event_id)} unique types",
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
                "type": "MouseEEGv2_H32",
                "channel_count": 30,
                "manufacturer": "NeuroNexus",
                "reference": "Probe-specific routing",
                "layout": "Mouse cortical surface array",
                "file_format": "NeuroNexus XDAT",
                "probe_dimensions": "0.64 × 0.83 mm",
                "coordinate_units": "micrometers (converted to meters)",
            },
            "notes": [
                "H32 probe has scrambled pin routing (e.g., E1 → pri_29)",
                "Channels pri_1 and pri_31 are unconnected",
                "Coordinate scale is mouse-specific (~1mm total spread)",
            ],
        }
