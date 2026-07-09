"""Montage/channel-count-aware presets for bad-channel detection.

Correlation and RANSAC-based bad-channel detection rely on spatial
redundancy between neighboring channels. A threshold tuned for a
128-channel net can be too aggressive (or numerically unstable) on a
19-32 channel montage, where losing even a handful of channels is a much
larger fraction of the total montage. This module resolves a single set
of detection thresholds from a named preset (or channel-count-based
"auto" selection) plus any user overrides, so that
:meth:`~autoclean.mixins.signal_processing.channels.ChannelsMixin.clean_bad_channels`
can apply montage-appropriate defaults while keeping the historical
behavior available as an explicit "legacy" opt-in.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# Historical hardcoded defaults (pre-dating montage-aware presets). These
# double as the fallback for any threshold not defined by a selected
# density bin, and as the full threshold set for preset="legacy".
BASE_DEFAULTS: Dict[str, Any] = {
    "correlation_thresh": 0.35,
    "deviation_thresh": 2.5,
    "ransac_sample_prop": 0.35,
    "ransac_corr_thresh": 0.65,
    "ransac_frac_bad": 0.25,
    "ransac_channel_wise": False,
    "ransac_enabled": True,
    "cleaning_method": "interpolate",
    "max_bad_fraction": 0.15,
}

LEGACY_PRESET: Dict[str, Any] = dict(BASE_DEFAULTS)

# Suggested channel-count bins for preset="auto". Low-density montages get
# a more conservative correlation/deviation threshold and RANSAC disabled
# outright (few neighboring channels make RANSAC's resampling unstable).
DEFAULT_CHANNEL_COUNT_BINS: Dict[str, Dict[str, Any]] = {
    "low_density": {
        "max_channels": 32,
        # Lower = more aggressive. Suggested low-density default: 0.20
        "correlation_thresh": 0.20,
        # Lower = more aggressive. Suggested low-density default: 4.0
        "deviation_thresh": 4.0,
        # RANSAC is unstable with limited spatial redundancy; off by default.
        "ransac_enabled": False,
        # Lower = stricter file flagging. Suggested low-density default: 0.10
        "max_bad_fraction": 0.10,
    },
    "mid_density": {
        "min_channels": 33,
        "max_channels": 64,
        "correlation_thresh": 0.30,
        "deviation_thresh": 3.0,
        # Higher = more aggressive. Suggested mid-density default: 0.65
        "ransac_corr_thresh": 0.65,
        "max_bad_fraction": 0.15,
    },
    "high_density": {
        "min_channels": 65,
        "correlation_thresh": 0.35,
        "deviation_thresh": 2.5,
        "ransac_corr_thresh": 0.65,
        "max_bad_fraction": 0.20,
    },
}

DENSITY_PRESET_NAMES = {"low_density", "mid_density", "high_density"}
VALID_PRESETS = DENSITY_PRESET_NAMES | {"auto", "legacy"}

# Keys resolve_bad_channel_settings will accept from config/explicit overrides.
_OVERRIDABLE_KEYS = set(BASE_DEFAULTS)


@dataclass(frozen=True)
class ResolvedBadChannelSettings:
    """Fully resolved bad-channel detection thresholds for one recording."""

    preset: str
    density_bin: Optional[str]
    channel_count: int
    correlation_thresh: float
    deviation_thresh: float
    ransac_sample_prop: float
    ransac_corr_thresh: float
    ransac_frac_bad: float
    ransac_channel_wise: bool
    ransac_enabled: bool
    cleaning_method: str
    max_bad_fraction: float

    def detector_options(self) -> Dict[str, Any]:
        """Options accepted by ``detect_bad_channels`` / ``NoisyChannels``."""

        return {
            "correlation_thresh": self.correlation_thresh,
            "deviation_thresh": self.deviation_thresh,
            "ransac_sample_prop": self.ransac_sample_prop,
            # A corr_thresh of 0 disables RANSAC in detect_bad_channels().
            "ransac_corr_thresh": (
                self.ransac_corr_thresh if self.ransac_enabled else 0.0
            ),
            "ransac_frac_bad": self.ransac_frac_bad,
            "ransac_channel_wise": self.ransac_channel_wise,
        }

    def as_metadata(self) -> Dict[str, Any]:
        return asdict(self)


def merge_channel_count_bins(
    overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Merge user-supplied bin overrides onto :data:`DEFAULT_CHANNEL_COUNT_BINS`.

    Bins matching a default name (``low_density``/``mid_density``/
    ``high_density``) are shallow-merged so a user can override a single
    field (e.g. ``correlation_thresh``) without repeating the rest.
    Unrecognized bin names are added as-is, allowing fully custom bins.
    """

    merged: Dict[str, Dict[str, Any]] = {
        name: dict(cfg) for name, cfg in DEFAULT_CHANNEL_COUNT_BINS.items()
    }
    if not overrides:
        return merged

    for name, cfg in overrides.items():
        if name in merged:
            merged[name].update(cfg or {})
        else:
            merged[name] = dict(cfg or {})

    return merged


def select_density_bin(
    channel_count: int, channel_count_bins: Dict[str, Dict[str, Any]]
) -> str:
    """Pick the bin whose [min_channels, max_channels] range contains ``channel_count``.

    Falls back to the closest bin by boundary distance if no bin's range
    contains the count (e.g. gaps in a custom bin configuration).
    """

    if not channel_count_bins:
        raise ValueError("channel_count_bins must contain at least one bin")

    def bounds(cfg: Dict[str, Any]) -> tuple:
        return cfg.get("min_channels", 0), cfg.get("max_channels", math.inf)

    exact_matches = [
        name
        for name, cfg in channel_count_bins.items()
        if bounds(cfg)[0] <= channel_count <= bounds(cfg)[1]
    ]
    if exact_matches:
        # Prefer the narrowest matching range if bins overlap.
        return min(
            exact_matches,
            key=lambda name: bounds(channel_count_bins[name])[1]
            - bounds(channel_count_bins[name])[0],
        )

    def distance(cfg: Dict[str, Any]) -> float:
        min_ch, max_ch = bounds(cfg)
        if channel_count < min_ch:
            return min_ch - channel_count
        return channel_count - max_ch

    return min(channel_count_bins, key=lambda name: distance(channel_count_bins[name]))


def resolve_bad_channel_settings(
    channel_count: int,
    preset: str = "auto",
    channel_count_bins: Optional[Dict[str, Dict[str, Any]]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    explicit_overrides: Optional[Dict[str, Any]] = None,
) -> ResolvedBadChannelSettings:
    """Resolve preset + channel-count-bin + override layers into one threshold set.

    Priority (highest wins): ``explicit_overrides`` (direct method call
    arguments) > ``config_overrides`` (flat keys under the task's
    ``bad_channel_detection.value``) > the selected preset/density bin >
    :data:`BASE_DEFAULTS`.
    """

    normalized_preset = (preset or "auto").strip().lower()
    if normalized_preset not in VALID_PRESETS:
        raise ValueError(
            f"Unknown bad_channel_detection preset {preset!r}; expected one of "
            f"{sorted(VALID_PRESETS)}"
        )

    bins = (
        channel_count_bins
        if channel_count_bins is not None
        else DEFAULT_CHANNEL_COUNT_BINS
    )

    resolved: Dict[str, Any] = dict(BASE_DEFAULTS)
    density_bin: Optional[str] = None

    if normalized_preset == "legacy":
        resolved.update(LEGACY_PRESET)
    else:
        if normalized_preset == "auto":
            density_bin = select_density_bin(channel_count, bins)
        else:
            if normalized_preset not in bins:
                raise ValueError(
                    f"Preset {normalized_preset!r} has no matching entry in "
                    f"channel_count_bins (available: {sorted(bins)})"
                )
            density_bin = normalized_preset

        bin_settings = {
            key: value
            for key, value in bins[density_bin].items()
            if key not in ("min_channels", "max_channels")
        }
        resolved.update(bin_settings)

    for source in (config_overrides, explicit_overrides):
        if not source:
            continue
        resolved.update(
            {
                key: value
                for key, value in source.items()
                if key in _OVERRIDABLE_KEYS and value is not None
            }
        )

    return ResolvedBadChannelSettings(
        preset=normalized_preset,
        density_bin=density_bin,
        channel_count=channel_count,
        correlation_thresh=float(resolved["correlation_thresh"]),
        deviation_thresh=float(resolved["deviation_thresh"]),
        ransac_sample_prop=float(resolved["ransac_sample_prop"]),
        ransac_corr_thresh=float(resolved["ransac_corr_thresh"]),
        ransac_frac_bad=float(resolved["ransac_frac_bad"]),
        ransac_channel_wise=bool(resolved["ransac_channel_wise"]),
        ransac_enabled=bool(resolved["ransac_enabled"]),
        cleaning_method=resolved["cleaning_method"],
        max_bad_fraction=float(resolved["max_bad_fraction"]),
    )
