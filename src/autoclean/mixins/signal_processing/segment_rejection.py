"""Segment rejection mixin

This mixin provides functionality for rejecting segments of data based on user-defined criteria.

"""

import mne
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr
from scipy.spatial import distance_matrix

from autoclean.utils.logging import message


class SegmentRejectionMixin:
    """Mixin for segment rejection."""

    def annotate_noisy_epochs(
        self,
        raw: mne.io.Raw = None,
        epoch_duration: float = 2.0,
        epoch_overlap: float = 0.0,
        picks: list[str] = None,
        quantile_k: float = 3.0,
        quantile_flag_crit: float = 0.2,
        annotation_description: str = "BAD_noisy_epoch",
    ):
        """
        Identifies noisy epochs in continuous EEG data and annotates them.

        This function temporarily epochs the continuous data, calculates channel-wise
        standard deviations for each epoch, and then identifies epochs where a
        significant proportion of channels exhibit outlier standard deviations.
        The outlier detection is based on the interquartile range (IQR) method,
        similar to what's used in the pylossless pipeline.

        Parameters
        ----------
        raw : mne.io.Raw
            The continuous MNE Raw object to process.
        epoch_duration : float, default 2.0
            Duration of epochs in seconds for noise detection.
        epoch_overlap : float, default 0.0
            Overlap between epochs in seconds.
        picks : list of str | str | None, default None
            Channels to include. If None, defaults to 'eeg'. Can be channel
            names or types (e.g., ['EEG 001', 'EEG 002'] or 'grad').
        quantile_k : float, default 3.0
            Multiplier for the IQR when defining outlier thresholds for channel
            standard deviations. A channel's std in an epoch is an outlier if it's
            `k` IQRs above Q3 or below Q1 relative to its own distribution of stds
            across all epochs.
        quantile_flag_crit : float, default 0.2
            Proportion threshold. If more than this proportion of picked channels
            are marked as outliers (having outlier std) within an epoch,
            that epoch is flagged as noisy.
        annotation_description : str, default "BAD_noisy_epoch"
            The description to use for MNE annotations marking noisy epochs.

        Returns
        -------
        mne.io.Raw
            The input Raw object with added annotations for noisy epochs.

        Notes
        -----
        The noise detection criteria are:
        1. For each channel, its standard deviation is calculated within each epoch.
        2. For each channel, the distribution of its standard deviations across all
        epochs is considered. Outlier thresholds (lower and upper bounds) are
        determined using `quantile_k` times the IQR of this distribution.
        3. An epoch is marked as noisy if the proportion of channels whose standard
        deviation in that epoch falls outside their respective outlier bounds
        exceeds `quantile_flag_crit`.
        This implementation adapts concepts from `flag_noisy_epochs` and related
        helper functions in `pylossless.pipeline`.
        """
        if picks is None:
            picks = "eeg"  # Default to EEG channels

        # 1. Create fixed-length epochs
        # MNE epoching tmax is inclusive, adjust for exact duration
        # No, mne.make_fixed_length_epochs `duration` is the length.
        # mne.Epochs tmax is where the adjustment is typically needed if using events.

        message("header", "Annotating noisy epochs")

        if raw is None:
            raw = self.raw

        message(
            "info",
            f"Epoching data with duration {epoch_duration} and overlap {epoch_overlap}.",
        )

        events = mne.make_fixed_length_events(
            raw, duration=epoch_duration, overlap=epoch_overlap
        )

        # Ensure events are within data boundaries
        if events.shape[0] == 0:
            message("error", "No epochs could be created with the given parameters.")
            return raw.copy()

        max_event_time = events[-1, 0] + int(epoch_duration * raw.info["sfreq"])
        if max_event_time > len(raw.times):
            # Prune events that would lead to epochs exceeding data length
            valid_events_mask = events[:, 0] + int(
                epoch_duration * raw.info["sfreq"]
            ) <= len(raw.times)
            events = events[valid_events_mask]
            if events.shape[0] == 0:
                message("error", "No valid epochs after boundary check.")
                return raw.copy()

        epochs = mne.Epochs(
            raw,
            events,
            tmin=0.0,
            tmax=epoch_duration - 1.0 / raw.info["sfreq"],  # tmax is exclusive endpoint
            picks=picks,
            preload=True,
            baseline=None,  # No baseline correction for std calculation
            reject=None,  # We are detecting bads, not rejecting yet
        )

        if len(epochs) == 0:
            message(
                "error",
                f"No epochs left after picking channels: {picks}. Cannot proceed.",
            )
            return raw.copy()

        # 2. Convert epochs to xarray DataArray (channels, epochs, time)
        epochs_xr = self._epochs_to_xr(epochs)

        # 3. Calculate standard deviation for each channel within each epoch
        data_sd = epochs_xr.std("time")  # Shape: (channels, epochs)

        # 4. Detect noisy epochs using the adapted outlier detection logic
        outliers_kwargs_config = {
            "k": quantile_k
        }  # Corresponds to 'k' in _get_outliers_quantile

        bad_epoch_indices = self._detect_outliers(
            data_sd,
            flag_dim="epoch",  # We want to flag epochs
            outlier_method="quantile",
            flag_crit=quantile_flag_crit,
            init_dir="pos",  # Typically interested in high std_dev for noise
            outliers_kwargs=outliers_kwargs_config,
        )

        if len(bad_epoch_indices) == 0:
            message("info", "No noisy epochs found.")
            return raw.copy()

        # 5. Add annotations to the original raw object
        # Adapted from pylossless.pipeline.LosslessPipeline.add_pylossless_annotations
        message("debug", "Adding annotations to the original raw object.")
        relative_onsets = epochs.events[bad_epoch_indices, 0] / raw.info["sfreq"]

        onsets = relative_onsets - raw.first_samp / raw.info["sfreq"]

        # Duration of each epoch (pylossless uses n_samples - 1, let's use full epoch duration for simplicity here)
        # The duration of the annotation should match the epoch_duration.
        # Using epochs.times can be tricky if there was any cropping/shifting not accounted for.
        # Safest is to use the intended epoch_duration.
        annotation_durations = np.full_like(onsets, fill_value=epoch_duration)

        descriptions = [annotation_description] * len(bad_epoch_indices)

        # Create new annotations
        new_annotations = mne.Annotations(
            onset=onsets,
            duration=annotation_durations,
            description=descriptions,
            orig_time=raw.annotations.orig_time,  # Preserve original time reference
        )

        # Make a copy of the raw object to modify annotations
        raw_annotated = raw.copy()
        raw_annotated.set_annotations(raw_annotated.annotations + new_annotations)

        message(
            "info",
            f"Added {len(bad_epoch_indices)} '{annotation_description}' annotations.",
        )

        message("debug", "Reporting flagged epochs.")
        self._report_flagged_epochs(raw_annotated, annotation_description)

        self._update_instance_data(raw, raw_annotated, use_epochs=False)

        metadata = {
            "flagged_epochs": bad_epoch_indices,
            "flagged_epochs_description": annotation_description,
        }

        self._update_metadata("step_annotate_noisy_epochs", metadata)

        return raw_annotated

    def annotate_uncorrelated_epochs(
        self,
        raw: mne.io.Raw = None,
        epoch_duration: float = 2.0,
        epoch_overlap: float = 0.0,
        picks: list[str] = None,
        n_nearest_neighbors: int = 5,
        corr_method: str = "max",
        corr_trim_percent: float = 10.0,
        outlier_k: float = 4.0,
        outlier_flag_crit: float = 0.2,
        annotation_description: str = "BAD_uncorrelated_epoch",
    ):
        """
        Identifies epochs with low channel-neighbor correlations and annotates them.

        This function temporarily epochs data, calculates correlations between each
        channel and its spatial neighbors for each epoch, and then flags epochs
        where a significant proportion of channels show unusually low correlations.
        Outlier detection for low correlations is based on the IQR method.

        Parameters
        ----------
        raw : mne.io.Raw
            The continuous MNE Raw object to process. Must have a montage set.
        epoch_duration : float, default 2.0
            Duration of epochs in seconds.
        epoch_overlap : float, default 0.0
            Overlap between epochs in seconds.
        picks : list of str | str | None, default None
            Channels to include. If None, defaults to 'eeg'.
        n_nearest_neighbors : int, default 5
            Number of nearest spatial neighbors to consider for correlation.
        corr_method : str, default "max"
            Method to aggregate correlations with neighbors: "max", "mean", or "trimmean".
            "max" takes the maximum absolute correlation.
        corr_trim_percent : float, default 10.0
            If `corr_method` is "trimmean", the percentage to trim from each end
            of the distribution of neighbor correlations before averaging.
        outlier_k : float, default 3.0
            Multiplier for the IQR when defining outlier thresholds for low correlations.
            A channel's aggregated neighbor correlation in an epoch is an outlier if it's
            `k` IQRs below Q1 of its own distribution of correlations across all epochs.
        outlier_flag_crit : float, default 0.2
            Proportion threshold. If more than this proportion of picked channels
            are marked as outliers (having low neighbor correlation) within an epoch,
            that epoch is flagged.
        annotation_description : str, default "BAD_uncorrelated_epoch"
            Description for MNE annotations marking these epochs.

        Returns
        -------
        mne.io.Raw
            The input Raw object with added annotations for uncorrelated epochs.

        Notes
        -----
        - Requires `scipy` for distance matrix and `xarray` for data handling.
        - The `raw` object *must* have a montage set for channel locations.
        - The `first_samp` correction for annotations assumes that
          `raw.annotations.orig_time` is the true start of the original recording
          (e.g., `raw.info['meas_date']`). If your `orig_time` is different or `None`,
          you may need to adjust the onset calculation.
        """

        message("header", "Annotating uncorrelated epochs")

        if raw is None:
            raw = self.raw

        if picks is None:
            picks = "eeg"

        # 1. Create fixed-length epochs
        message(
            "info",
            f"Epoching data with duration {epoch_duration} and overlap {epoch_overlap}.",
        )
        events = mne.make_fixed_length_events(
            raw, duration=epoch_duration, overlap=epoch_overlap
        )
        if events.shape[0] == 0:
            message("error", "No epochs could be created with the given parameters.")
            return raw.copy()

        max_event_time = events[-1, 0] + int(epoch_duration * raw.info["sfreq"])
        if max_event_time > len(raw.times):
            valid_events_mask = events[:, 0] + int(
                epoch_duration * raw.info["sfreq"]
            ) <= len(raw.times)
            events = events[valid_events_mask]
            if events.shape[0] == 0:
                message("error", "No valid epochs after boundary check.")
                return raw.copy()

        epochs = mne.Epochs(
            raw,
            events,
            tmin=0.0,
            tmax=epoch_duration - 1.0 / raw.info["sfreq"],
            picks=picks,
            preload=True,
            baseline=None,
            reject=None,
        )

        if len(epochs) == 0:
            message(
                "error",
                f"No epochs left after picking channels: {picks}. Cannot proceed.",
            )
            return raw.copy()

        if epochs.get_montage() is None:
            raise ValueError(
                "The raw object (and thus epochs) must have a montage set. Use raw.set_montage()."
            )

        # 2. Calculate nearest neighbor correlations for channels within epochs
        # data_r_ch has shape (channels, epochs)
        data_r_ch = self._chan_neighbour_r(
            epochs,
            n_nearest_neighbors=n_nearest_neighbors,
            corr_method=corr_method,
            corr_trim_percent=corr_trim_percent,
        )

        # 3. Detect epochs with too many uncorrelated channels
        # We are looking for *low* correlations, so init_dir="neg"
        outliers_kwargs_config = {"k": outlier_k}
        bad_epoch_indices = self._detect_outliers(
            data_r_ch,
            flag_dim="epoch",
            outlier_method="quantile",
            flag_crit=outlier_flag_crit,
            init_dir="neg",  # Flagging based on *low* correlation values
            outliers_kwargs=outliers_kwargs_config,
        )

        if len(bad_epoch_indices) == 0:
            message("info", "No uncorrelated epochs found.")
            return raw.copy()

        # 4. Add annotations to the original raw object
        # Correctly calculate onsets relative to raw.annotations.orig_time
        # This assumes raw.annotations.orig_time is the original measurement start.
        # (event_sample_in_current_raw + raw.first_samp) / sfreq
        absolute_onsets = (
            epochs.events[bad_epoch_indices, 0] - raw.first_samp
        ) / raw.info["sfreq"]

        annotation_durations = np.full_like(absolute_onsets, fill_value=epoch_duration)
        descriptions = [annotation_description] * len(bad_epoch_indices)

        new_annotations = mne.Annotations(
            onset=absolute_onsets,
            duration=annotation_durations,
            description=descriptions,
            orig_time=raw.annotations.orig_time,
        )

        raw_annotated = raw.copy()
        raw_annotated.set_annotations(raw_annotated.annotations + new_annotations)

        message(
            "info",
            f"Added {len(bad_epoch_indices)} '{annotation_description}' annotations.",
        )
        self._report_flagged_epochs(raw_annotated, annotation_description)

        self._update_instance_data(raw, raw_annotated, use_epochs=False)

        metadata = {
            "flagged_epochs": bad_epoch_indices,
            "flagged_epochs_description": annotation_description,
        }

        self._update_metadata("step_annotate_uncorrelated_epochs", metadata)

        return raw_annotated

    def _epochs_to_xr(self, epochs):
        """
        Create an Xarray DataArray from an instance of mne.Epochs.
        Adapted from pylossless.pipeline.epochs_to_xr.
        """
        data = epochs.get_data()  # n_epochs, n_channels, n_times
        ch_names = epochs.ch_names
        # Transpose to (n_channels, n_epochs, n_times) for consistency with pylossless internal processing
        data_transposed = data.transpose(1, 0, 2)
        return xr.DataArray(
            data_transposed,
            coords={
                "ch": ch_names,
                "epoch": np.arange(data_transposed.shape[1]),
                "time": epochs.times,
            },
            dims=("ch", "epoch", "time"),
        )

    def _get_outliers_quantile(
        self, array, dim, lower=0.25, upper=0.75, mid=0.5, k=3.0
    ):
        """
        Calculate outliers based on the IQR.
        Adapted from pylossless.pipeline._get_outliers_quantile.
        `array` is expected to be (channels, epochs).
        `dim` is 'epoch' (to calculate quantiles across epochs for each channel).
        """
        lower_val, mid_val, upper_val = array.quantile([lower, mid, upper], dim=dim)

        lower_dist = mid_val - lower_val
        upper_dist = upper_val - mid_val
        return mid_val - lower_dist * k, mid_val + upper_dist * k

    def _detect_outliers(
        self,
        array,
        flag_dim,
        outlier_method="quantile",
        flag_crit=0.2,
        init_dir="pos",
        outliers_kwargs=None,
    ):
        """
        Mark items along flag_dim as flagged for artifact.
        Adapted from pylossless.pipeline._detect_outliers.
        `array` is (channels, epochs).
        `flag_dim` is 'epoch'.
        `operate_dim` will be 'ch'.
        """
        if outliers_kwargs is None:
            outliers_kwargs = {}

        # Determine the dimension to operate across (the one NOT being flagged)
        dims = list(array.dims)
        if flag_dim not in dims:
            raise ValueError(f"flag_dim '{flag_dim}' not in array dimensions: {dims}")
        dims.remove(flag_dim)
        if not dims:
            raise ValueError("Array must have at least two dimensions.")
        operate_dim = dims[
            0
        ]  # Should be 'ch' if array is (ch, epoch) and flag_dim is 'epoch'

        if outlier_method == "quantile":
            l_out, u_out = self._get_outliers_quantile(
                array, dim=flag_dim, **outliers_kwargs
            )
        # Add other methods like 'trimmed' or 'fixed' here if needed, similar to pylossless
        else:
            raise ValueError(
                f"outlier_method '{outlier_method}' not supported. Use 'quantile'."
            )

        outlier_mask = xr.zeros_like(array, dtype=bool)

        if init_dir == "pos" or init_dir == "both":
            outlier_mask = outlier_mask | (array > u_out)
        if init_dir == "neg" or init_dir == "both":
            outlier_mask = outlier_mask | (array < l_out)

        # Calculate proportion of outliers along operate_dim (e.g., channels)
        # For each epoch, what proportion of channels are outliers?
        prop_outliers = outlier_mask.astype(float).mean(operate_dim)

        if "quantile" in list(
            prop_outliers.coords.keys()
        ):  # A coordinate that might be introduced by xarray's quantile
            prop_outliers = prop_outliers.drop_vars("quantile")

        flagged_indices = (
            prop_outliers[prop_outliers > flag_crit].coords[flag_dim].values
        )
        return flagged_indices

    def _report_flagged_epochs(self, raw, desc):
        """Helper to report total duration of flagged epochs for a given description."""
        total_duration = 0
        for annot in raw.annotations:
            if annot["description"] == desc:
                total_duration += annot["duration"]
        if total_duration > 0:
            message(
                "info", f"Total duration for '{desc}': {total_duration:.2f} seconds."
            )

    def _chan_neighbour_r(
        self, epochs, n_nearest_neighbors, corr_method="max", corr_trim_percent=10.0
    ):
        """
        Compute nearest neighbor correlations for channels within epochs.
        Adapted from pylossless.pipeline.chan_neighbour_r.

        Parameters
        ----------
        epochs : mne.Epochs
            The epoched data. Must have a montage with channel positions.
        n_nearest_neighbors : int
            Number of nearest neighbors to consider for correlation.
        corr_method : str, default "max"
            Method to aggregate correlations with neighbors: "max", "mean", or "trimmean".
        corr_trim_percent : float, default 10.0
            Percentage to trim from each end if `corr_method` is "trimmean".
            E.g., 10.0 means 10% from lower and 10% from upper end.

        Returns
        -------
        xr.DataArray
            An xarray DataArray of shape (channels, epochs) containing the
            aggregated correlation of each channel with its neighbors for each epoch.
            The 'channels' dimension here refers to the reference channels.
        """
        montage = epochs.get_montage()
        if montage is None:
            raise ValueError(
                "Epochs object must have a montage set to calculate neighbor correlations. "
                "Use `epochs.set_montage()`."
            )

        ch_positions = montage.get_positions()["ch_pos"]
        valid_chs = [
            ch for ch in epochs.ch_names if ch in ch_positions
        ]  # Channels present in both data and montage

        if len(valid_chs) < len(epochs.ch_names):
            print(
                f"Warning: Could not find positions for all channels in epochs. "
                f"Using {len(valid_chs)} out of {len(epochs.ch_names)} channels that have positions."
            )
        if not valid_chs:
            raise ValueError(
                "No channel positions found for any channels in the epochs object."
            )
        if len(valid_chs) <= n_nearest_neighbors:
            print(
                f"Warning: Number of valid channels with positions ({len(valid_chs)}) "
                f"is less than or equal to n_nearest_neighbors ({n_nearest_neighbors}). "
                "Each channel will be correlated with all other available valid channels."
            )
            actual_n_neighbors = max(0, len(valid_chs) - 1)  # Max possible neighbors
        else:
            actual_n_neighbors = n_nearest_neighbors

        ch_locs_df = pd.DataFrame(ch_positions).T.loc[valid_chs]

        dist_matrix_val = distance_matrix(ch_locs_df.values, ch_locs_df.values)
        chan_dist_df = pd.DataFrame(
            dist_matrix_val, columns=ch_locs_df.index, index=ch_locs_df.index
        )

        rank = chan_dist_df.rank(axis="columns", method="first", ascending=True) - 1
        rank[rank == 0] = np.nan

        nearest_neighbor_df = pd.DataFrame(
            index=ch_locs_df.index, columns=range(actual_n_neighbors), dtype=object
        )
        for ch_name_iter in ch_locs_df.index:
            sorted_neighbors = rank.loc[ch_name_iter].dropna().sort_values()
            nearest_neighbor_df.loc[ch_name_iter] = sorted_neighbors.index[
                :actual_n_neighbors
            ].values

        # Pick only valid channels for epochs_xr to avoid issues if some channels in epochs had no positions
        epochs_xr = self._epochs_to_xr(epochs.copy().pick(valid_chs))

        all_channel_corrs = []

        print(
            f"Calculating neighbor correlations for {len(valid_chs)} channels using {actual_n_neighbors} nearest neighbors..."
        )
        for _, ch_name in enumerate(valid_chs):  # ch_name is the reference channel
            neighbor_names_for_ch = [
                n
                for n in nearest_neighbor_df.loc[ch_name].values.tolist()
                if pd.notna(n) and n != ch_name
            ]

            if not neighbor_names_for_ch:
                # Handle case with no valid neighbors (e.g. only 1 channel, or actual_n_neighbors is 0)
                ch_neighbor_corr_aggregated = xr.DataArray(
                    np.full(
                        epochs_xr.sizes["epoch"], np.nan
                    ),  # NaN correlation if no neighbors
                    coords={"epoch": epochs_xr.coords["epoch"]},
                    dims=["epoch"],
                )
            else:
                # Data for the current reference channel
                this_ch_data = epochs_xr.sel(
                    ch=ch_name
                )  # xr.DataArray with dims (epoch, time)

                # Data for its neighbors
                # neighbor_chs_data will have dims (ch, epoch, time) where 'ch' are the neighbor channels
                neighbor_chs_data = epochs_xr.sel(ch=neighbor_names_for_ch)

                # Calculate Pearson correlation along the 'time' dimension.
                # this_ch_data (epoch, time) is broadcast against neighbor_chs_data (ch_neighbor, epoch, time).
                # The result ch_to_neighbors_corr will have dims ('ch', 'epoch'),
                # where 'ch' dimension contains coordinates from neighbor_names_for_ch.
                ch_to_neighbors_corr = xr.corr(
                    this_ch_data, neighbor_chs_data, dim="time"
                )

                # Aggregate correlations based on corr_method, reducing the 'ch' (neighbor) dimension
                if corr_method == "max":
                    ch_neighbor_corr_aggregated = np.abs(ch_to_neighbors_corr).max(
                        dim="ch"
                    )
                elif corr_method == "mean":
                    ch_neighbor_corr_aggregated = np.abs(ch_to_neighbors_corr).mean(
                        dim="ch"
                    )
                elif corr_method == "trimmean":
                    proportion_to_cut = corr_trim_percent / 100.0
                    # np_data should be (epoch, ch_neighbors) for scipy.stats.trim_mean
                    # Transpose ch_to_neighbors_corr (dims: ch, epoch) to (epoch, ch) for trim_mean input
                    np_data = np.abs(ch_to_neighbors_corr).transpose("epoch", "ch").data

                    trimmed_means_per_epoch = [
                        (
                            scipy.stats.trim_mean(
                                epoch_data_for_trim, proportiontocut=proportion_to_cut
                            )
                            if not np.all(np.isnan(epoch_data_for_trim))
                            and len(epoch_data_for_trim)
                            > 0  # Check for non-empty and non-all-NaN
                            else np.nan
                        )
                        for epoch_data_for_trim in np_data
                    ]
                    ch_neighbor_corr_aggregated = xr.DataArray(
                        trimmed_means_per_epoch,
                        coords={
                            "epoch": ch_to_neighbors_corr.coords["epoch"]
                        },  # Use original epoch coords
                        dims=["epoch"],
                    )
                else:
                    raise ValueError(f"Unknown corr_method: {corr_method}")

            # At this point, ch_neighbor_corr_aggregated has only the 'epoch' dimension.
            # Now, expand it to add the reference channel's name as a new 'ch' dimension.
            expanded_corr = ch_neighbor_corr_aggregated.expand_dims(
                dim={"ch": [ch_name]}
            )
            all_channel_corrs.append(expanded_corr)

        if not all_channel_corrs:  # Should not happen if valid_chs is not empty
            print("Warning: No channel correlations were computed.")
            # Return an empty or appropriate DataArray to avoid errors downstream
            return xr.DataArray(
                np.empty((0, epochs_xr.sizes.get("epoch", 0))),  # (channels, epochs)
                coords={"ch": [], "epoch": epochs_xr.coords.get("epoch", [])},
                dims=("ch", "epoch"),
            )

        # Concatenate results for all reference channels along the new 'ch' dimension
        concatenated_corrs = xr.concat(all_channel_corrs, dim="ch")
        # The 'epoch' dimension name should be correct from ch_neighbor_corr_aggregated.
        # No rename like 'epoch_dim_temp' should be needed if handled carefully above.

        return concatenated_corrs  # Shape: (ch_reference, epochs)
