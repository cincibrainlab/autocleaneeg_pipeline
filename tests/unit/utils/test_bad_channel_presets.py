"""Unit tests for montage/channel-count-aware bad-channel detection presets."""

import pytest

from autoclean.utils.bad_channel_presets import (
    BASE_DEFAULTS,
    DEFAULT_CHANNEL_COUNT_BINS,
    LEGACY_PRESET,
    merge_channel_count_bins,
    resolve_bad_channel_settings,
    select_density_bin,
)


class TestSelectDensityBin:
    def test_boundary_channel_counts_map_to_expected_bins(self):
        assert select_density_bin(19, DEFAULT_CHANNEL_COUNT_BINS) == "low_density"
        assert select_density_bin(32, DEFAULT_CHANNEL_COUNT_BINS) == "low_density"
        assert select_density_bin(33, DEFAULT_CHANNEL_COUNT_BINS) == "mid_density"
        assert select_density_bin(64, DEFAULT_CHANNEL_COUNT_BINS) == "mid_density"
        assert select_density_bin(65, DEFAULT_CHANNEL_COUNT_BINS) == "high_density"
        assert select_density_bin(128, DEFAULT_CHANNEL_COUNT_BINS) == "high_density"

    def test_falls_back_to_nearest_bin_on_gap(self):
        bins = {
            "low": {"max_channels": 10},
            "high": {"min_channels": 20},
        }
        # 17 is in the gap between the two bins; 20 is closer than 10.
        assert select_density_bin(17, bins) == "high"

    def test_raises_on_empty_bins(self):
        with pytest.raises(ValueError):
            select_density_bin(32, {})


class TestMergeChannelCountBins:
    def test_none_returns_defaults(self):
        merged = merge_channel_count_bins(None)
        assert merged == DEFAULT_CHANNEL_COUNT_BINS

    def test_partial_override_preserves_other_fields(self):
        merged = merge_channel_count_bins({"low_density": {"correlation_thresh": 0.15}})
        assert merged["low_density"]["correlation_thresh"] == 0.15
        # Untouched fields still present from the default bin.
        assert merged["low_density"]["max_channels"] == 32
        assert merged["low_density"]["ransac_enabled"] is False

    def test_custom_bin_name_is_added(self):
        merged = merge_channel_count_bins(
            {"ultra_low": {"max_channels": 16, "correlation_thresh": 0.1}}
        )
        assert "ultra_low" in merged
        assert "low_density" in merged  # defaults untouched

    def test_does_not_mutate_module_defaults(self):
        merge_channel_count_bins({"low_density": {"correlation_thresh": 0.01}})
        assert DEFAULT_CHANNEL_COUNT_BINS["low_density"]["correlation_thresh"] == 0.20


class TestResolveBadChannelSettings:
    def test_legacy_preset_matches_historical_defaults(self):
        resolved = resolve_bad_channel_settings(channel_count=32, preset="legacy")
        assert resolved.density_bin is None
        assert resolved.correlation_thresh == LEGACY_PRESET["correlation_thresh"]
        assert resolved.deviation_thresh == LEGACY_PRESET["deviation_thresh"]
        assert resolved.ransac_corr_thresh == LEGACY_PRESET["ransac_corr_thresh"]
        assert resolved.ransac_enabled is True
        assert resolved.max_bad_fraction == LEGACY_PRESET["max_bad_fraction"]

    def test_auto_high_density_matches_legacy_thresholds(self):
        """High-density auto resolution should not change existing behavior."""
        resolved = resolve_bad_channel_settings(channel_count=128, preset="auto")
        assert resolved.density_bin == "high_density"
        assert resolved.correlation_thresh == BASE_DEFAULTS["correlation_thresh"]
        assert resolved.deviation_thresh == BASE_DEFAULTS["deviation_thresh"]
        assert resolved.ransac_corr_thresh == BASE_DEFAULTS["ransac_corr_thresh"]
        assert resolved.ransac_enabled is True

    def test_auto_low_density_disables_ransac_and_is_more_conservative(self):
        resolved = resolve_bad_channel_settings(channel_count=19, preset="auto")
        assert resolved.density_bin == "low_density"
        assert resolved.ransac_enabled is False
        assert resolved.detector_options()["ransac_corr_thresh"] == 0.0
        assert resolved.correlation_thresh == 0.20
        assert resolved.deviation_thresh == 4.0
        assert resolved.max_bad_fraction == 0.10

    def test_auto_mid_density(self):
        resolved = resolve_bad_channel_settings(channel_count=40, preset="auto")
        assert resolved.density_bin == "mid_density"
        assert resolved.correlation_thresh == 0.30
        assert resolved.deviation_thresh == 3.0
        assert resolved.max_bad_fraction == 0.15
        assert resolved.ransac_enabled is True

    def test_explicit_density_preset_ignores_channel_count(self):
        resolved = resolve_bad_channel_settings(channel_count=128, preset="low_density")
        assert resolved.density_bin == "low_density"
        assert resolved.ransac_enabled is False

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            resolve_bad_channel_settings(channel_count=32, preset="not_a_preset")

    def test_config_overrides_apply_over_preset(self):
        resolved = resolve_bad_channel_settings(
            channel_count=19,
            preset="auto",
            config_overrides={"correlation_thresh": 0.5},
        )
        assert resolved.correlation_thresh == 0.5
        assert resolved.density_bin == "low_density"

    def test_explicit_overrides_win_over_config_overrides(self):
        resolved = resolve_bad_channel_settings(
            channel_count=19,
            preset="auto",
            config_overrides={"correlation_thresh": 0.5},
            explicit_overrides={"correlation_thresh": 0.9},
        )
        assert resolved.correlation_thresh == 0.9

    def test_none_values_in_overrides_do_not_clobber_resolved_value(self):
        resolved = resolve_bad_channel_settings(
            channel_count=19,
            preset="auto",
            explicit_overrides={"correlation_thresh": None},
        )
        assert resolved.correlation_thresh == 0.20

    def test_ransac_enabled_explicit_override_forces_disable(self):
        resolved = resolve_bad_channel_settings(
            channel_count=128,
            preset="auto",
            explicit_overrides={"ransac_enabled": False},
        )
        assert resolved.ransac_enabled is False
        assert resolved.detector_options()["ransac_corr_thresh"] == 0.0

    def test_custom_channel_count_bins_are_honored(self):
        custom_bins = {
            "tiny": {"max_channels": 8, "correlation_thresh": 0.1, "ransac_enabled": False},
            "big": {"min_channels": 9, "correlation_thresh": 0.4},
        }
        resolved = resolve_bad_channel_settings(
            channel_count=5, preset="auto", channel_count_bins=custom_bins
        )
        assert resolved.density_bin == "tiny"
        assert resolved.correlation_thresh == 0.1

    def test_detector_options_keys_match_detect_bad_channels_kwargs(self):
        resolved = resolve_bad_channel_settings(channel_count=64, preset="auto")
        options = resolved.detector_options()
        assert set(options) == {
            "correlation_thresh",
            "deviation_thresh",
            "ransac_sample_prop",
            "ransac_corr_thresh",
            "ransac_frac_bad",
            "ransac_channel_wise",
        }

    def test_as_metadata_is_json_serializable_shape(self):
        resolved = resolve_bad_channel_settings(channel_count=19, preset="auto")
        metadata = resolved.as_metadata()
        assert metadata["preset"] == "auto"
        assert metadata["density_bin"] == "low_density"
        assert metadata["channel_count"] == 19
