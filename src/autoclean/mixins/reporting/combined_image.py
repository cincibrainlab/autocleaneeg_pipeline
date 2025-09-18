"""Combined EEG report image generation mixin."""

from __future__ import annotations

import ast
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from autoclean.utils.logging import message

try:  # Pillow is required for final compositing
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - dependency optional at runtime
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

if Image is not None:
    if hasattr(Image, "Resampling"):
        _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    else:
        _RESAMPLE_LANCZOS = Image.LANCZOS
else:  # pragma: no cover - dependency optional at runtime
    _RESAMPLE_LANCZOS = None

_DEFAULT_CANVAS_SIZE = (2600, 1400)
_DEFAULT_GAP_SECONDS = 0.2
_DEFAULT_SPACING = 1.2
_DEFAULT_TARGET_HZ = 60.0


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely."""
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(round(value))
        text = str(value).strip()
        if not text:
            return default
        return int(round(float(text)))
    except (ValueError, TypeError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (ValueError, TypeError):
        return default


def _ensure_list(value: Any) -> list[Any]:
    """Return a list representation of the value."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass
        return [item.strip() for item in text.split(",") if item.strip()]
    return [value]


def _trim_whitespace(image: Image.Image, margin: int = 10) -> Image.Image:
    """Crop pure white borders while keeping a margin."""
    if Image is None:
        return image

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return image

    white = np.array([255, 255, 255], dtype=np.uint8)
    mask = ~(arr == white).all(axis=-1)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return image

    top = max(int(rows[0]) - margin, 0)
    bottom = min(int(rows[-1]) + margin + 1, arr.shape[0])
    left = max(int(cols[0]) - margin, 0)
    right = min(int(cols[-1]) + margin + 1, arr.shape[1])

    trimmed = arr[top:bottom, left:right]
    return Image.fromarray(trimmed)

def _decimate_to_target(
    data: np.ndarray,
    times: np.ndarray,
    srate: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decimate data to approximately target_hz, returning new arrays and rate."""
    if target_hz <= 0 or srate <= target_hz:
        return data, times, srate

    try:
        from scipy.signal import decimate  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - optional dependency
        message(
            "warning",
            f"Fastplot summary decimation skipped: SciPy unavailable ({exc}).",
        )
        return data, times, srate

    factor = int(round(srate / target_hz))
    factor = max(factor, 1)
    if factor <= 1:
        return data, times, srate

    new_rate = srate / factor

    decimated = decimate(data, factor, axis=-1, ftype="fir", zero_phase=True)
    decimated = decimated.astype(np.float32, copy=False)

    new_times = times[::factor]
    if new_times.size != decimated.shape[-1]:
        new_times = times[0] + np.arange(decimated.shape[-1], dtype=np.float32) / new_rate

    return decimated, new_times.astype(np.float32, copy=False), new_rate

def _build_time_series(
    data: np.ndarray,
    times: np.ndarray,
    gap_seconds: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten epochs into a continuous stack suitable for fastplotlib."""
    n_epochs, n_channels, n_times = data.shape
    epoch_times = times - times[0]
    sample_step = float(np.mean(np.diff(epoch_times))) if n_times > 1 else 0.0
    epoch_duration = float(epoch_times[-1] + sample_step)

    offsets = np.arange(n_epochs, dtype=np.float32) * (epoch_duration + gap_seconds)
    timeline = (offsets[:, None] + epoch_times[None, :]).reshape(-1).astype(np.float32)

    traces = data.transpose(1, 0, 2).reshape(n_channels, -1)
    traces -= traces.mean(axis=1, keepdims=True)

    scale = float(np.percentile(np.abs(traces), 98))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    traces /= scale
    traces += np.arange(n_channels, dtype=np.float32)[:, None] * spacing

    x_data = np.tile(timeline, (n_channels, 1))
    lines = np.stack((x_data, traces.astype(np.float32, copy=False)), axis=-1)

    return lines[::-1], timeline

def _compose_plot_image(plot_path: Path, title: str) -> Image.Image:
    """Return a PIL image with the plot and title banner."""
    if Image is None:  # pragma: no cover - dependency optional at runtime
        raise RuntimeError("Pillow is required to compose fastplot summary images")

    plot_img = Image.open(plot_path).convert("RGB")
    return _compose_plot_banner(plot_img, title)


def _compose_plot_banner(plot_img: Image.Image, title: str) -> Image.Image:
    """Add a title banner to an existing plot image."""
    if Image is None:  # pragma: no cover - dependency optional at runtime
        raise RuntimeError("Pillow is required to compose fastplot summary images")

    plot_width, plot_height = plot_img.size
    title_height = 80

    final_img = Image.new("RGB", (plot_width, plot_height + title_height), "white")
    final_img.paste(plot_img, (0, title_height))

    draw = ImageDraw.Draw(final_img)
    font = None
    if ImageFont is not None:
        for font_path in (
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttf",
            "arial.ttf",
        ):
            try:
                font = ImageFont.truetype(font_path, 72)
                break
            except (OSError, IOError):
                continue
    if font is None:
        font = ImageFont.load_default() if ImageFont is not None else None

    if font is not None:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width = len(title) * 6
        text_height = 10

    x = (plot_width - text_width) // 2
    y = (title_height - text_height) // 2
    shadow_offset = 1
    draw.text((x + shadow_offset, y + shadow_offset), title, fill="#666666", font=font)
    draw.text((x, y), title, fill="#222222", font=font)
    return final_img

def _create_processing_summary_image(
    csv_path: Optional[Path] = None,
    summary_dict: Optional[dict[str, Any]] = None,
) -> Optional[Image.Image]:
    """Create the processing summary pie chart image from available data."""
    if Image is None:
        return None

    metrics = None

    if summary_dict:
        metrics = _metrics_from_summary(summary_dict)

    if metrics is None and csv_path and csv_path.exists():
        metrics = _metrics_from_csv(csv_path)

    if metrics is None:
        return None

    channels, duration, trials, components = metrics
    return _create_pie_chart_image(
        channels=channels,
        duration=duration,
        trials=trials,
        components=components,
    )


def _metrics_from_summary(summary: dict[str, Any]) -> Optional[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Extract channel, duration, trial, and component metrics from summary dict."""
    if not summary:
        return None

    import_details = summary.get("import_details") or {}
    export_details = summary.get("export_details") or {}
    channel_dict = summary.get("channel_dict") or {}
    ica_details = summary.get("ica_details") or {}

    total_channels = _safe_int(import_details.get("net_nbchan_orig"), 0)
    post_channels = _safe_int(export_details.get("net_nbchan_post"), total_channels)
    removed_channels = _ensure_list(channel_dict.get("removed_channels"))
    bad_channels = len(removed_channels)
    if bad_channels <= 0 and total_channels and post_channels < total_channels:
        bad_channels = max(total_channels - post_channels, 0)
    if post_channels <= 0 and total_channels:
        post_channels = max(total_channels - bad_channels, 0)

    raw_duration = _safe_float(import_details.get("duration"), 0.0)
    final_duration = _safe_float(export_details.get("final_duration"), 0.0)
    if final_duration <= 0.0:
        final_duration = _safe_float(export_details.get("initial_duration"), 0.0)
    if final_duration <= 0.0:
        final_duration = raw_duration
    removed_duration = max(raw_duration - final_duration, 0.0)

    initial_epochs = _safe_int(export_details.get("initial_n_epochs"), 0)
    final_epochs = _safe_int(export_details.get("final_n_epochs"), 0)
    if initial_epochs <= 0 and final_epochs > 0:
        initial_epochs = final_epochs
    good_trials = max(final_epochs, 0)
    bad_trials = max(initial_epochs - final_epochs, 0)
    total_trials = max(initial_epochs, good_trials + bad_trials)

    total_components = _safe_int(ica_details.get("proc_nComps"), 0)
    removed_components_list = _ensure_list(ica_details.get("proc_removeComps"))
    removed_components = len([c for c in removed_components_list if str(c).strip()])
    if removed_components > total_components and total_components > 0:
        removed_components = total_components
    retained_components = max(total_components - removed_components, 0)

    channels = {
        "good": max(good_channels := max(total_channels - bad_channels, 0), 0),
        "bad": max(bad_channels, 0),
        "total": max(total_channels, 0),
        "post": max(post_channels, 0),
    }
    duration = {
        "raw": max(raw_duration, 0.0),
        "kept": max(final_duration, 0.0),
        "removed": max(removed_duration, 0.0),
    }
    trials = {
        "good": max(good_trials, 0),
        "bad": max(bad_trials, 0),
        "total": max(total_trials, 0),
    }
    components = {
        "retained": max(retained_components, 0),
        "removed": max(removed_components, 0),
        "total": max(total_components, 0),
    }

    return channels, duration, trials, components


def _metrics_from_csv(csv_path: Path) -> Optional[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Extract metrics from a processing log CSV."""
    try:
        import pandas as pd  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - optional dependency
        message("warning", f"Fastplot summary: pandas unavailable ({exc}).")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - runtime file issues
        message("warning", f"Fastplot summary: could not read {csv_path.name} ({exc}).")
        return None

    if df.empty:
        return None

    row = df.iloc[0]

    total_channels = _safe_int(row.get("net_nbchan_orig"), 0)
    post_channels = _safe_int(row.get("net_nbchan_post"), total_channels)
    bad_channels = len(_ensure_list(row.get("proc_badchans", [])))
    if bad_channels <= 0 and total_channels and post_channels < total_channels:
        bad_channels = max(total_channels - post_channels, 0)
    good_channels = max(total_channels - bad_channels, 0)

    raw_duration = _safe_float(row.get("proc_xmax_raw"), 0.0)
    post_duration = _safe_float(row.get("proc_xmax_post"), 0.0)
    removed_duration = max(raw_duration - post_duration, 0.0)

    total_trials = _safe_int(row.get("epoch_trials"), 0)
    bad_trials = _safe_int(row.get("epoch_badtrials"), 0)
    if bad_trials > total_trials:
        bad_trials = total_trials
    good_trials = max(total_trials - bad_trials, 0)

    total_components = _safe_int(row.get("proc_nComps"), 0)
    removed_components = len(_ensure_list(row.get("proc_removeComps", [])))
    if removed_components > total_components and total_components > 0:
        removed_components = total_components
    retained_components = max(total_components - removed_components, 0)

    channels = {
        "good": good_channels,
        "bad": bad_channels,
        "total": total_channels,
        "post": post_channels,
    }
    duration = {
        "raw": raw_duration,
        "kept": post_duration,
        "removed": removed_duration,
    }
    trials = {
        "good": good_trials,
        "bad": bad_trials,
        "total": total_trials,
    }
    components = {
        "retained": retained_components,
        "removed": removed_components,
        "total": total_components,
    }

    return channels, duration, trials, components


def _create_pie_chart_image(
    *,
    channels: Optional[dict[str, Any]] = None,
    duration: Optional[dict[str, Any]] = None,
    trials: Optional[dict[str, Any]] = None,
    components: Optional[dict[str, Any]] = None,
) -> Optional[Image.Image]:
    """Render pie chart summary image from provided metric dictionaries."""
    if Image is None:
        return None

    try:
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - optional dependency
        message("warning", f"Fastplot summary: matplotlib unavailable ({exc}).")
        return None

    sections: list[dict[str, Any]] = []

    if channels:
        total_channels = max(int(channels.get("total", 0)), 0)
        post_channels = max(int(channels.get("post", 0)), 0)
        good_channels = max(int(channels.get("good", 0)), 0)
        bad_channels = max(int(channels.get("bad", 0)), 0)
        if good_channels + bad_channels <= 0 and total_channels > 0:
            good_channels = min(total_channels, max(total_channels - bad_channels, 0))
        if good_channels + bad_channels > 0:
            sections.append(
                {
                    "title": "Channels: Good vs Bad",
                    "values": [good_channels, bad_channels],
                    "labels": [
                        f"Good ({good_channels})",
                        f"Bad ({bad_channels})",
                    ],
                    "footer": f"Original: {total_channels}\nPost-cleaning: {post_channels}",
                    "value_fmt": "int",
                    "value_suffix": "",
                }
            )

    if duration:
        kept = max(float(duration.get("kept", 0.0)), 0.0)
        removed = max(float(duration.get("removed", 0.0)), 0.0)
        raw = float(duration.get("raw", kept + removed))
        if raw <= 0.0:
            raw = kept + removed
        if kept + removed > 0:
            sections.append(
                {
                    "title": "Recording Duration (Xmax)",
                    "values": [kept, removed],
                    "labels": [
                        f"Kept ({kept:.1f}s)",
                        f"Removed ({removed:.1f}s)",
                    ],
                    "footer": f"Raw duration: {raw:.1f}s",
                    "value_fmt": "float",
                    "value_suffix": "s",
                }
            )

    if trials:
        good_trials = max(int(trials.get("good", 0)), 0)
        bad_trials = max(int(trials.get("bad", 0)), 0)
        total_trials = max(int(trials.get("total", good_trials + bad_trials)), 0)
        if good_trials + bad_trials <= 0 and total_trials > 0:
            good_trials = total_trials
            bad_trials = 0
        if good_trials + bad_trials > 0:
            sections.append(
                {
                    "title": "Epoch Trials",
                    "values": [good_trials, bad_trials],
                    "labels": [
                        f"Good ({good_trials})",
                        f"Bad ({bad_trials})",
                    ],
                    "footer": f"Total Trials: {total_trials}",
                    "value_fmt": "int",
                    "value_suffix": "",
                }
            )

    if components:
        retained = max(int(components.get("retained", 0)), 0)
        removed_comps = max(int(components.get("removed", 0)), 0)
        total_components = max(int(components.get("total", retained + removed_comps)), 0)
        if retained + removed_comps <= 0 and total_components > 0:
            retained = total_components - removed_comps
        if retained + removed_comps > 0:
            sections.append(
                {
                    "title": "ICA Components",
                    "values": [retained, removed_comps],
                    "labels": [
                        f"Retained ({retained})",
                        f"Removed ({removed_comps})",
                    ],
                    "footer": f"Total Components: {total_components}",
                    "value_fmt": "int",
                    "value_suffix": "",
                }
            )

    if not sections:
        return None

    good_color = "#1f77b4"
    bad_color = "#d62728"

    fig, axes = plt.subplots(1, len(sections), figsize=(5 * len(sections), 4))
    if len(sections) == 1:
        axes_iter = [axes]
    else:
        axes_iter = axes

    for ax, section in zip(axes_iter, sections):
        values = section["values"]
        total = float(sum(values))
        if total <= 0:
            ax.axis("off")
            ax.set_title(section["title"])
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        value_fmt = section.get("value_fmt", "int")
        value_suffix = section.get("value_suffix", "")

        def autopct_fmt(pct: float) -> str:
            val = pct / 100.0 * total
            if value_fmt == "float":
                val_text = f"{val:.1f}"
            else:
                val_text = f"{val:.0f}"
            return f"{pct:.0f}%\n({val_text}{value_suffix})"

        wedges, _texts, autotexts = ax.pie(
            values,
            labels=section["labels"],
            autopct=autopct_fmt,
            startangle=140,
            colors=[good_color, bad_color],
            pctdistance=0.65,
            labeldistance=1.1,
            wedgeprops=dict(linewidth=1.5, edgecolor="white"),
            textprops=dict(color="black", fontsize=11),
        )

        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontweight("bold")
            autotext.set_size(13)

        ax.set_title(section["title"])
        ax.axis("equal")
        ax.text(0, -1.3, section["footer"], ha="center")

    plt.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200)
    plt.close(fig)
    buffer.seek(0)

    summary_img = Image.open(buffer).convert("RGB")
    summary_img.load()
    buffer.close()
    return summary_img


def _render_lines_fastplotlib(
    lines: np.ndarray,
    canvas_size: Tuple[int, int],
    title: str,
    temp_path: Path,
) -> Image.Image:
    """Render stacked traces using fastplotlib."""
    try:
        import imageio  # pylint: disable=import-outside-toplevel

        _ = imageio.__version__  # Avoid unused import warnings
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"ImageIO unavailable ({exc})") from exc

    try:
        import fastplotlib as fpl  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"fastplotlib unavailable ({exc})") from exc

    fig = fpl.Figure(size=canvas_size, canvas="offscreen")

    try:
        subplot = fig[0, 0]
        subplot.background_color = ("#ffffff", "#ffffff")
        subplot.add_line_collection(data=lines, colors="#111111", thickness=0.6)

        axes = subplot.axes
        if axes is not None:
            axes.visible = False
            if getattr(axes, "x", None) is not None:
                axes.x.label = ""
                if getattr(axes.x, "ticks", None) is not None:
                    axes.x.ticks.visible = False
            if getattr(axes, "y", None) is not None:
                axes.y.label = ""
                if getattr(axes.y, "ticks", None) is not None:
                    axes.y.ticks.visible = False

        try:
            subplot.auto_scale(maintain_aspect=False, zoom=1.05)
        except Exception:  # pragma: no cover - defensive
            pass

        try:
            fig._render(draw=False)  # pylint: disable=protected-access
        except Exception:  # pragma: no cover - defensive
            pass

        fig.export(str(temp_path))
    except Exception as exc:
        raise RuntimeError(f"fastplotlib export failed ({exc})") from exc
    finally:
        try:
            fig.close()
        except Exception:  # pragma: no cover - cleanup best effort
            pass

    return _compose_plot_image(temp_path, title)


def _render_lines_matplotlib(
    lines: np.ndarray,
    canvas_size: Tuple[int, int],
    title: str,
) -> Image.Image:
    """Render stacked traces using matplotlib as a fallback."""
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Matplotlib unavailable for fallback ({exc})") from exc

    dpi = 100
    fig_width = max(canvas_size[0] / dpi, 1)
    fig_height = max(canvas_size[1] / dpi, 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for line in lines:
        ax.plot(line[:, 0], line[:, 1], color="#111111", linewidth=0.4, alpha=0.9)

    ax.set_xlim(float(lines[..., 0].min()), float(lines[..., 0].max()))
    ax.set_ylim(float(lines[..., 1].min()), float(lines[..., 1].max()))
    ax.margins(0)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white", edgecolor="white")
    plt.close(fig)

    buffer.seek(0)
    plot_img = Image.open(buffer).convert("RGB")
    plot_img.load()
    buffer.close()

    return _compose_plot_banner(plot_img, title)


def _plot_to_tiff(
    lines: np.ndarray,
    timeline: np.ndarray,
    output_path: Path,
    canvas_size: Tuple[int, int],
    title: str,
    summary_image: Optional[Image.Image],
) -> None:
    """Render the stacked traces, append summary panel, and export as TIFF/PNG."""
    if Image is None:  # pragma: no cover - dependency optional at runtime
        raise RuntimeError("Pillow is required to compose fastplot summary images")

    # timeline is currently unused but kept for API compatibility
    _ = timeline

    plot_image: Optional[Image.Image] = None
    temp_path = output_path.with_suffix(".temp.tiff")

    try:
        plot_image = _render_lines_fastplotlib(lines, canvas_size, title, temp_path)
    except RuntimeError as exc:
        message(
            "warning",
            f"Fastplot summary: fastplotlib export unavailable ({exc}). Falling back to matplotlib.",
        )
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except TypeError:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

    if plot_image is None:
        try:
            plot_image = _render_lines_matplotlib(lines, canvas_size, title)
        except RuntimeError as exc:
            raise RuntimeError(f"Fastplot summary fallback failed: {exc}") from exc

    if plot_image is None:
        raise RuntimeError("Fastplot summary failed: no renderer available")

    plot_image = _trim_whitespace(plot_image, margin=10)

    if summary_image is not None:
        summary = _trim_whitespace(summary_image.convert("RGB"), margin=5)
        if summary.width != plot_image.width:
            ratio = plot_image.width / max(summary.width, 1)
            new_height = max(int(summary.height * ratio), 1)
            if _RESAMPLE_LANCZOS is not None:
                summary = summary.resize((plot_image.width, new_height), _RESAMPLE_LANCZOS)
            else:
                summary = summary.resize((plot_image.width, new_height))
        combined = Image.new(
            "RGB",
            (plot_image.width, plot_image.height + summary.height),
            "white",
        )
        combined.paste(plot_image, (0, 0))
        combined.paste(summary, (0, plot_image.height))
    else:
        combined = plot_image

    combined = _trim_whitespace(combined, margin=5)
    combined.save(output_path, format="TIFF")

    png_path = output_path.with_suffix(".png")
    combined.save(png_path, format="PNG")

def _find_processing_log_csv(
    basename: str,
    reports_root: Optional[Path],
    metadata_dir: Optional[Path],
    final_files_dir: Optional[Path],
    derivatives_dir: Optional[Path],
) -> Optional[Path]:
    """Return the first matching per-file processing log CSV if it exists."""
    candidate_names = []
    candidate_names.append(f"{basename}_processing_log.csv")

    base_subject = basename.split("_comp")[0]
    candidate_names.append(f"{base_subject}_processing_log.csv")

    suffix = ""
    for part in basename.split("_"):
        if part.isdigit():
            suffix = part
    if suffix:
        candidate_names.append(f"{base_subject}_processing_log_{suffix}.csv")

    unique_names = []
    for name in candidate_names:
        if name not in unique_names:
            unique_names.append(name)

    candidate_dirs: list[Path] = []
    if reports_root:
        candidate_dirs.append(reports_root / "run_reports")
        candidate_dirs.append(reports_root)
    if metadata_dir:
        candidate_dirs.append(metadata_dir)
        candidate_dirs.append(metadata_dir.parent / "reports" / "run_reports")
    if final_files_dir:
        candidate_dirs.append(final_files_dir)
    if derivatives_dir:
        candidate_dirs.append(derivatives_dir)

    for directory in candidate_dirs:
        try:
            for name in unique_names:
                path = directory / name
                if path.exists():
                    return path
        except TypeError:
            continue

    return None

class FastPlotReportMixin:
    """Provide fastplotlib-based combined EEG trace + summary visualization."""

    FASTPLOT_STEP_KEY = "fastplot_summary"

    def _create_runtime_summary_image(self) -> Optional[Image.Image]:
        """Build a summary image using in-memory task state when CSV is unavailable."""
        original_raw = getattr(self, "original_raw", None)
        cleaned_raw = getattr(self, "raw", None)

        def _duration(raw_obj: Any) -> float:
            try:
                if raw_obj is None or getattr(raw_obj, "n_times", 0) == 0:
                    return 0.0
                times = getattr(raw_obj, "times", None)
                if times is not None and len(times) > 1:
                    return float(times[-1] - times[0])
                sfreq = float(raw_obj.info.get("sfreq", 0.0))
                n_times = int(getattr(raw_obj, "n_times", 0))
                return float(n_times / sfreq) if sfreq > 0 and n_times > 0 else 0.0
            except Exception:
                return 0.0

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, str):
                lower = value.strip().lower()
                if lower in {"", "0", "false", "no", "none", "nan"}:
                    return False
                return True
            try:
                return bool(int(value))
            except Exception:
                return bool(value)

        channels: Optional[dict[str, Any]] = None
        if original_raw is not None:
            total_channels = len(original_raw.ch_names)
            post_channels = len(cleaned_raw.ch_names) if cleaned_raw is not None else total_channels
            bad_candidates = set(original_raw.info.get("bads", []))
            if cleaned_raw is not None:
                bad_candidates.update(cleaned_raw.info.get("bads", []))
            bad_chans = len(bad_candidates)
            if bad_chans <= 0 and total_channels >= post_channels:
                bad_chans = max(total_channels - post_channels, 0)
            good_channels = max(total_channels - bad_chans, 0)
            channels = {
                "good": good_channels,
                "bad": bad_chans,
                "total": total_channels,
                "post": post_channels,
            }

        duration: Optional[dict[str, Any]] = None
        if original_raw is not None:
            raw_duration = _duration(original_raw)
            post_duration = _duration(cleaned_raw) if cleaned_raw is not None else raw_duration
            removed_duration = max(raw_duration - post_duration, 0.0)
            duration = {
                "raw": raw_duration,
                "kept": post_duration,
                "removed": removed_duration,
            }

        trials: Optional[dict[str, Any]] = None
        epochs_obj = getattr(self, "epochs", None)
        if epochs_obj is not None:
            try:
                drop_log = getattr(epochs_obj, "drop_log", [])
                total_trials = len(drop_log) if drop_log is not None else len(epochs_obj)
                bad_trials = sum(1 for log in drop_log if log) if drop_log else 0
            except Exception:
                total_trials = len(epochs_obj)
                bad_trials = 0
            if total_trials == 0:
                total_trials = len(epochs_obj)
                bad_trials = 0
            good_trials = max(total_trials - bad_trials, 0)
            trials = {
                "good": good_trials,
                "bad": bad_trials,
                "total": total_trials,
            }

        components: Optional[dict[str, Any]] = None
        final_ica = getattr(self, "final_ica", None)
        if final_ica is not None and hasattr(final_ica, "n_components_"):
            total_components = int(getattr(final_ica, "n_components_", 0) or 0)
            excluded = getattr(final_ica, "exclude", []) or []
            removed_components = len(excluded)
            retained_components = max(total_components - removed_components, 0)
            components = {
                "retained": retained_components,
                "removed": removed_components,
                "total": total_components,
            }
        else:
            ica_flags = getattr(self, "ica_flags", None)
            if ica_flags is not None:
                try:
                    total_components = int(len(ica_flags))
                except Exception:
                    total_components = 0
                removed_components = 0
                try:
                    if hasattr(ica_flags, "columns") and "reject" in ica_flags.columns:
                        reject_col = ica_flags["reject"]
                        try:
                            removed_components = int(sum(_coerce_bool(x) for x in reject_col))
                        except Exception:
                            removed_components = 0
                    elif hasattr(ica_flags, "reject"):
                        reject_attr = getattr(ica_flags, "reject")
                        removed_components = int(sum(_coerce_bool(x) for x in reject_attr))
                except Exception:
                    pass
                retained_components = max(total_components - removed_components, 0)
                components = {
                    "retained": retained_components,
                    "removed": removed_components,
                    "total": total_components,
                }

        if not any([channels, duration, trials, components]):
            return None

        return _create_pie_chart_image(
            channels=channels,
            duration=duration,
            trials=trials,
            components=components,
        )

    def generate_fastplot_summary(
        self,
        epochs: Optional[Any] = None,
        *,
        gap_seconds: Optional[float] = None,
        spacing: Optional[float] = None,
        target_hz: Optional[float] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
        report_key: str = "fastplot_summary",
    ) -> Optional[Path]:
        """Generate and save the combined fastplot summary image.

        Parameters
        ----------
        epochs
            Optional MNE Epochs instance to use. Defaults to ``self.epochs``.
        gap_seconds
            Override the inter-epoch gap in seconds.
        spacing
            Override vertical spacing between channels.
        target_hz
            Override target sampling rate after decimation.
        canvas_size
            Override fastplotlib canvas size as ``(width, height)``.
        report_key
            Reports subdirectory key for storing the artifact.

        Returns
        -------
        Path or None
            Path to the generated TIFF file, or ``None`` if generation was skipped.
        """
        if Image is None:
            message("warning", "Fastplot summary skipped: Pillow is not installed.")
            return None

        step_settings: dict[str, Any] = {}
        if isinstance(getattr(self, "settings", None), dict):
            step_settings = self.settings.get(self.FASTPLOT_STEP_KEY, {}) or {}

        enabled = step_settings.get("enabled", True)
        if not enabled:
            message("info", "✗ Fastplot summary disabled in task settings")
            return None

        value_cfg = step_settings.get("value", {}) if isinstance(step_settings.get("value"), dict) else {}

        gap = float(
            gap_seconds
            if gap_seconds is not None
            else value_cfg.get("gap_seconds", value_cfg.get("gap", _DEFAULT_GAP_SECONDS))
        )
        vert_spacing = float(
            spacing
            if spacing is not None
            else value_cfg.get("spacing", _DEFAULT_SPACING)
        )
        target_rate = float(
            target_hz
            if target_hz is not None
            else value_cfg.get("target_hz", _DEFAULT_TARGET_HZ)
        )

        if canvas_size is not None:
            canvas = (int(canvas_size[0]), int(canvas_size[1]))
        else:
            width = value_cfg.get("width") or value_cfg.get("canvas_width")
            height = value_cfg.get("height") or value_cfg.get("canvas_height")
            if width and height:
                canvas = (int(width), int(height))
            else:
                canvas = _DEFAULT_CANVAS_SIZE

        epochs_obj = epochs if epochs is not None else getattr(self, "epochs", None)
        if epochs_obj is None:
            message("info", "Fastplot summary skipped: no epochs available.")
            return None

        try:
            data = epochs_obj.get_data()
            times = np.asarray(epochs_obj.times, dtype=np.float32)
            srate = float(epochs_obj.info.get("sfreq", 0.0))
        except Exception as exc:  # pragma: no cover - defensive
            message("error", f"Fastplot summary failed to access epoch data: {exc}")
            return None

        if data.ndim != 3 or data.size == 0:
            message("info", "Fastplot summary skipped: unexpected epoch data shape.")
            return None

        data = data.astype(np.float32, copy=False)

        data, times, new_rate = _decimate_to_target(data, times, srate, target_rate)
        lines, timeline = _build_time_series(data, times, gap, vert_spacing)

        if lines.size == 0 or timeline.size == 0:
            message("info", "Fastplot summary skipped: no samples after formatting.")
            return None

        cfg = getattr(self, "config", {}) or {}
        unprocessed = cfg.get("unprocessed_file")
        unprocessed_path = Path(unprocessed) if unprocessed else None
        basename = unprocessed_path.stem if unprocessed_path else "fastplot"
        title = unprocessed_path.name if unprocessed_path else "Fastplot Summary"

        try:
            output_path = self._resolve_report_path(
                report_key,
                f"{basename}_fastplot_summary.tiff",
            )
        except Exception as exc:  # pragma: no cover - missing directories
            message("error", f"Fastplot summary could not resolve report path: {exc}")
            return None

        reports_root = None
        try:
            reports_root = self._get_reports_root()
        except Exception:
            reports_dir_cfg = cfg.get("reports_dir")
            reports_root = Path(reports_dir_cfg) if reports_dir_cfg else None

        metadata_dir = Path(cfg["metadata_dir"]) if cfg.get("metadata_dir") else None
        final_files_dir = Path(cfg["final_files_dir"]) if cfg.get("final_files_dir") else None
        derivatives_dir = Path(cfg["derivatives_dir"]) if cfg.get("derivatives_dir") else None

        csv_path = _find_processing_log_csv(
            basename,
            reports_root,
            metadata_dir,
            final_files_dir,
            derivatives_dir,
        )

        summary_dict: Optional[dict[str, Any]] = None
        run_id = cfg.get("run_id")
        if run_id:
            try:
                from autoclean.step_functions.reports import (  # pylint: disable=import-outside-toplevel
                    create_json_summary,
                )

                flagged_reasons = getattr(self, "flagged_reasons", [])
                summary_candidate = create_json_summary(str(run_id), flagged_reasons)
                if summary_candidate:
                    summary_dict = summary_candidate
            except Exception as exc:  # pragma: no cover - defensive
                message("debug", f"Fastplot summary: create_json_summary failed: {exc}")

        summary_image: Optional[Image.Image] = None
        if summary_dict:
            summary_image = _create_processing_summary_image(summary_dict=summary_dict)

        if summary_image is None and csv_path is not None:
            try:
                summary_image = _create_processing_summary_image(csv_path=csv_path)
            except Exception as exc:  # pragma: no cover - defensive
                message(
                    "warning",
                    f"Fastplot summary: failed to build processing summary from {csv_path.name}: {exc}",
                )

        if summary_image is None:
            runtime_summary = self._create_runtime_summary_image()
            if runtime_summary is not None:
                summary_image = runtime_summary
                message("info", "Fastplot summary pie charts derived from runtime metrics.")
            elif csv_path is None and summary_dict is None:
                message("info", "Fastplot summary metrics unavailable; pie charts skipped.")

        try:
            _plot_to_tiff(lines, timeline, output_path, canvas, title, summary_image)
        except RuntimeError as exc:  # pragma: no cover - optional dependency missing
            message("warning", f"Fastplot summary skipped: {exc}")
            return None
        except Exception as exc:  # pragma: no cover - defensive
            message("error", f"Fastplot summary failed during rendering: {exc}")
            return None

        rel_png = self._report_relative_path(output_path.with_suffix(".png"))
        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "fastplot_summary": str(rel_png),
                "decimated_rate_hz": new_rate,
                "channels": lines.shape[0],
            }
        }
        self._update_metadata("fastplot_summary", metadata)

        message("success", f"Fastplot summary image saved to {output_path}")
        if csv_path is not None and summary_image is not None:
            message("info", f"Fastplot summary appended processing stats from {csv_path.name}")
        elif csv_path is None and summary_image is not None:
            message("info", "Fastplot summary includes runtime-derived processing stats.")

        return output_path

__all__ = ["FastPlotReportMixin"]
