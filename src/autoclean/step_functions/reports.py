# src/autoclean/step_functions/reports.py # pylint: disable=too-many-lines
"""Visualization and reporting functions.



The reporting mixins provide the same functionality with improved integration
with the Task class and better configuration handling.

This module provides functions for generating visualizations and reports
from EEG processing results. It includes:
- Run summary reports
- Data quality visualizations
- Artifact detection plots
- Processing stage comparisons

The functions generate clear, publication-ready figures and detailed
HTML reports documenting the processing pipeline results.
"""

import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import ast

import matplotlib
import pandas as pd

# ReportLab imports for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.platypus import Table as ReportLabTable
from reportlab.platypus import TableStyle

from autoclean.utils.database import (
    get_run_record,
)
from autoclean.utils.logging import message

__all__ = [
    "create_run_report",
    "update_task_processing_log",
    "create_json_summary",
    "generate_bad_channels_tsv",
]

# Force matplotlib to use non-interactive backend for async operations
matplotlib.use("Agg")


def create_run_report(
    run_id: str, autoclean_dict: dict = None, json_summary: dict = None
) -> None:
    """
    Creates a pdf report summarizing the run.

    Parameters
    ----------
    run_id : str
        The run ID to generate a report for
    autoclean_dict : dict
        The autoclean dictionary
    json_summary : dict, optional
        Pre-computed JSON summary to use instead of looking up from database
    """
    if not run_id:
        message("error", "No run ID provided")
        return

    run_record = get_run_record(run_id)
    if not run_record or "metadata" not in run_record:
        message("error", "No metadata found for run ID")
        return

    # Early validation of required metadata sections
    required_sections = ["step_prepare_directories"]
    missing_sections = [
        section
        for section in required_sections
        if section not in run_record["metadata"]
    ]
    if missing_sections:
        message(
            "error",
            f"Missing required metadata sections: {', '.join(missing_sections)}",
        )
        return

    # Use provided JSON summary or check if it exists in metadata
    if json_summary is None:
        json_summary = None
        if "json_summary" in run_record["metadata"]:
            json_summary = run_record["metadata"]["json_summary"]
            message("info", "Using JSON summary from database for report generation")
    else:
        message("info", "Using provided JSON summary for report generation")

    # If no JSON summary, create it
    if not json_summary:
        message(
            "warning", "No json summary found, run report may be missing or incomplete"
        )
        json_summary = {}

    # Set up BIDS path
    derivatives_path = None
    try:
        if autoclean_dict:
            derivatives_path = autoclean_dict["derivatives_dir"]
    except Exception as e:  # pylint: disable=broad-except
        message(
            "warning",
            f"Failed to get derivatives path: {str(e)} : Saving only to metadata directory",
        )
        derivatives_path = None

    # Get metadata directory from step_prepare_directories
    metadata_dir = Path(run_record["metadata"]["step_prepare_directories"]["metadata"])
    if not metadata_dir.exists():
        metadata_dir.mkdir(parents=True, exist_ok=True)

    # Create PDF filename
    pdf_path = metadata_dir / f"{run_record['report_file']}"

    # Initialize the PDF document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    # Get styles
    styles = getSampleStyleSheet()

    # Custom styles for better visual hierarchy
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=14,
        spaceAfter=6,
        textColor=colors.HexColor("#2C3E50"),
        alignment=1,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading1"],
        fontSize=10,
        spaceAfter=4,
        textColor=colors.HexColor("#34495E"),
        alignment=1,
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=7,
        spaceAfter=2,
        textColor=colors.HexColor("#2C3E50"),
    )

    # steps_style = ParagraphStyle(
    #     "Steps",
    #     parent=normal_style,
    #     fontSize=7,
    #     leading=10,
    #     spaceBefore=1,
    #     spaceAfter=1,
    # )

    # Define frame style for main content
    frame_style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ECF0F1")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]
    )

    # Common table style
    table_style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6FA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
    )

    # Create story (content) for the PDF
    story = []

    # Title and Basic Info
    title = "EEG Processing Report"
    story.append(Paragraph(title, title_style))

    # Add status-colored subtitle
    status_color = (
        colors.HexColor("#2ECC71")
        if run_record.get("success", False)
        else colors.HexColor("#E74C3C")
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=heading_style,
        textColor=status_color,
        spaceAfter=2,
    )
    status_text = "SUCCESS" if run_record.get("success", False) else "FAILED"
    subtitle = f"Run ID: {run_id} - {status_text}"
    story.append(Paragraph(subtitle, subtitle_style))

    # Add timestamp
    timestamp_style = ParagraphStyle(
        "Timestamp",
        parent=normal_style,
        textColor=colors.HexColor("#7F8C8D"),
        alignment=1,
        spaceAfter=8,
    )
    timestamp = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(timestamp, timestamp_style))

    # Tables layout with better styling - NOW 2 COLUMNS
    data = [
        [
            Paragraph("Import Information", heading_style),
            Paragraph(
                "Processing Details", heading_style
            ),  # Changed from "Preprocessing Parameters"
        ]
    ]

    # Left column: Import info with colored background
    try:
        import_info = []

        if json_summary and "import_details" in json_summary:
            # Use data from JSON summary
            import_details = json_summary["import_details"]

            # Get values and format them safely
            duration = import_details.get("duration")
            duration_str = (
                f"{duration:.1f} sec" if isinstance(duration, (int, float)) else "N/A"
            )

            sample_rate = import_details.get("sample_rate")
            sample_rate_str = (
                f"{sample_rate} Hz" if isinstance(sample_rate, (int, float)) else "N/A"
            )

            import_info.extend(
                [
                    ["File", import_details.get("basename", "N/A")],
                    ["Duration", duration_str],
                    ["Sample Rate", sample_rate_str],
                    ["Channels", str(import_details.get("net_nbchan_orig", "N/A"))],
                ]
            )
        else:
            # Fall back to direct metadata access
            raw_info = run_record["metadata"].get("import_eeg", {})
            if not raw_info:
                raw_info = {"message": "Step import metadata not available"}

            # Get values and format them safely
            duration = raw_info.get("durationSec")
            duration_str = (
                f"{duration:.1f} sec" if isinstance(duration, (int, float)) else "N/A"
            )

            sample_rate = raw_info.get("sampleRate")
            sample_rate_str = (
                f"{sample_rate} Hz" if isinstance(sample_rate, (int, float)) else "N/A"
            )

            import_info.extend(
                [
                    ["File", raw_info.get("unprocessedFile", "N/A")],
                    ["Duration", duration_str],
                    ["Sample Rate", sample_rate_str],
                    ["Channels", str(raw_info.get("channelCount", "N/A"))],
                ]
            )

        if not import_info:
            import_info = [["No import data available", "N/A"]]

    except Exception as e:  # pylint: disable=broad-except
        message("warning", f"Error processing import information: {str(e)}")
        import_info = [["Error processing import data", "N/A"]]

    import_table = ReportLabTable(import_info, colWidths=[0.7 * inch, 1.3 * inch])
    import_table.setStyle(
        TableStyle(
            [
                *table_style._cmds,
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8F9F9"),
                ),
            ]
        )
    )

    # Middle column: Processing Details
    processing_details_info = []
    try:
        # Check if JSON summary and processing_details exist
        if json_summary and "processing_details" in json_summary:
            current_processing_details = json_summary["processing_details"]

            # Ensure current_processing_details is a dictionary
            if isinstance(current_processing_details, dict):
                # Filter
                l_freq = current_processing_details.get("l_freq", "N/A")
                h_freq = current_processing_details.get("h_freq", "N/A")
                filter_display = "N/A"
                if l_freq != "N/A" or h_freq != "N/A":
                    filter_display = f"{l_freq if l_freq is not None else 'N/A'}-{h_freq if h_freq is not None else 'N/A'} Hz"
                processing_details_info.append(["Filter", filter_display])

                # Notch
                notch_freqs_list = current_processing_details.get("notch_freqs", [])
                notch_freq_display = "N/A"
                actual_notch_freqs = []
                if isinstance(notch_freqs_list, list):
                    actual_notch_freqs = [
                        str(f) for f in notch_freqs_list if f is not None
                    ]
                elif isinstance(
                    notch_freqs_list, (int, float, str)
                ) and notch_freqs_list not in [
                    None,
                    "",
                ]:  # Handle scalar if somehow it's not a list
                    actual_notch_freqs = [str(notch_freqs_list)]

                if actual_notch_freqs:
                    notch_freq_display = f"{', '.join(actual_notch_freqs)} Hz"
                processing_details_info.append(["Notch", notch_freq_display])

                # Resample Rate
                resample_rate = current_processing_details.get("target_sfreq")
                resample_display = (
                    f"{resample_rate} Hz" if resample_rate is not None else "N/A"
                )
                processing_details_info.append(["Resample Rate", resample_display])

                # Trim Duration
                trim_duration = current_processing_details.get("trim_duration")
                if trim_duration is not None:
                    processing_details_info.append(
                        ["Trim Duration", f"{trim_duration} sec"]
                    )

                # Crop Info
                crop_s = current_processing_details.get("crop_start")
                crop_e = current_processing_details.get("crop_end")
                crop_d = current_processing_details.get(
                    "crop_duration"
                )  # from create_json_summary

                crop_display_val = None
                if crop_s is not None and crop_e is not None:
                    crop_display_val = f"{crop_s:.2f}s to {crop_e:.2f}s"
                    processing_details_info.append(["Crop Window", crop_display_val])
                elif (
                    crop_d is not None
                ):  # if specific start/end aren't there, maybe overall duration is
                    crop_display_val = f"{crop_d:.2f} sec"
                    processing_details_info.append(["Crop Duration", crop_display_val])

                # EOG Channels
                eog_channels = current_processing_details.get("eog_channels", [])
                if (
                    isinstance(eog_channels, list) and eog_channels
                ):  # Only show if present and not empty
                    processing_details_info.append(
                        ["EOG Channels", ", ".join(eog_channels)]
                    )

                # Dropped Outer Layer Channels
                dropped_ch_list = current_processing_details.get("dropped_channels", [])
                # Ensure dropped_ch_list is a list before calling len()
                num_dropped = (
                    len(dropped_ch_list) if isinstance(dropped_ch_list, list) else 0
                )
                if num_dropped > 0:  # Only show if channels were actually dropped
                    processing_details_info.append(
                        ["Outer Chans Dropped", str(num_dropped)]
                    )

                # Reference Type
                ref_type = current_processing_details.get("ref_type")
                if ref_type:  # Only show if present
                    processing_details_info.append(["Reference", str(ref_type)])

            else:
                message(
                    "debug",
                    f"json_summary['processing_details'] is not a dictionary: {type(current_processing_details)}",
                )
        else:
            # This case means json_summary or json_summary["processing_details"] is missing.
            # The 'if not processing_details_info:' check below will handle it.
            message(
                "debug",
                "processing_details not found in json_summary. 'Processing Details' section will be sparse or N/A.",
            )

    except Exception as e:  # pylint: disable=broad-except
        message("warning", f"Error populating 'Processing Details' section: {str(e)}")
        processing_details_info = [
            ["Error processing details", str(e)]
        ]  # Show error in report

    if not processing_details_info:  # If after all attempts, it's still empty
        processing_details_info = [["Processing data N/A", ""]]

    processing_details_table = ReportLabTable(
        processing_details_info, colWidths=[1.2 * inch, 2.1 * inch]
    )
    processing_details_table.setStyle(
        TableStyle(
            [
                *table_style._cmds,
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#EFF8F9"
                    ),  # Light blue background for this section
                ),
            ]
        )
    )

    # Add tables to main layout with spacing
    data.append([import_table, processing_details_table])
    main_table = ReportLabTable(
        data, colWidths=[2.5 * inch, 3.5 * inch]
    )  # Adjusted for 2 columns
    main_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ]
        )
    )

    # Add main content in a frame
    frame_data = [[main_table]]
    frame = ReportLabTable(frame_data, colWidths=[6.5 * inch])
    frame.setStyle(frame_style)
    story.append(frame)
    story.append(Spacer(1, 0.2 * inch))

    # Task Parameters Section
    story.append(Paragraph("Task Parameters", heading_style))

    task_params_data = []
    try:
        def _flatten_dict(d, parent_key=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else str(k)
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key))
                else:
                    items.append((new_key, v))
            return items

        task_file_info = run_record.get("task_file_info", {})
        file_content = task_file_info.get("file_content")
        if file_content:
            try:
                module_ast = ast.parse(file_content)
                for node in module_ast.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "config":
                                config_dict = ast.literal_eval(node.value)
                                if isinstance(config_dict, dict):
                                    for key, value in _flatten_dict(config_dict):
                                        task_params_data.append([str(key), str(value)])
                                break
            except Exception as e:  # pylint: disable=broad-except
                message("warning", f"Error extracting task parameters: {str(e)}")
        if not task_params_data:
            task_params_data = [["No task parameters available", ""]]
    except Exception as e:  # pylint: disable=broad-except
        message("warning", f"Error processing task parameters: {str(e)}")
        task_params_data = [["Error processing task parameters", ""]]

    task_params_table = ReportLabTable(
        [[Paragraph("Parameter", heading_style), Paragraph("Value", heading_style)]] + task_params_data,
        colWidths=[3 * inch, 3 * inch],
    )
    task_params_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6FA")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(task_params_table)
    story.append(Spacer(1, 0.2 * inch))

    # Processing Steps Section
    story.append(Paragraph("Processing Steps", heading_style))

    # Get processing steps from metadata
    steps_data = []
    try:
        # Fall back to metadata for steps
        for step_name, step_data in run_record["metadata"].items():
            if step_name.startswith("step_") and step_name not in [
                "step_prepare_directories",
            ]:
                # Format step name for display
                display_name = step_name.replace("step_", "").replace("_", " ").title()
                steps_data.append([display_name])
    except Exception as e:  # pylint: disable=broad-except
        message("warning", f"Error processing steps data: {str(e)}")
        steps_data = [["Error processing steps"]]

    if not steps_data:
        steps_data = [["No processing steps data available"]]

    # Create steps table with background styling
    steps_table = ReportLabTable(
        [[Paragraph("Processing Step", heading_style)]] + steps_data,
        colWidths=[6 * inch],
    )
    steps_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6FA")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8F9F9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(steps_table)
    story.append(Spacer(1, 0.2 * inch))

    # Bad Channels Section
    story.append(Paragraph("Bad Channels", heading_style))

    # Get bad channels from metadata
    bad_channels_data = []
    try:
        # First try to get bad channels from JSON summary
        if json_summary and "channel_dict" in json_summary:
            channel_dict = json_summary["channel_dict"]

            # Add each category of bad channels
            for category, channels in channel_dict.items():
                if (
                    category != "removed_channels" and channels
                ):  # Skip the combined list
                    display_category = (
                        category.replace("step_", "").replace("_", " ").title()
                    )
                    bad_channels_data.append([display_category, ", ".join(channels)])

            # Add total count
            if "removed_channels" in channel_dict:
                total_removed = len(channel_dict["removed_channels"])
                if (
                    "import_details" in json_summary
                    and "net_nbchan_orig" in json_summary["import_details"]
                ):
                    total_channels = json_summary["import_details"]["net_nbchan_orig"]
                    percentage = (
                        (total_removed / total_channels) * 100 if total_channels else 0
                    )
                    bad_channels_data.append(
                        [
                            "Total Removed",
                            f"{total_removed} / {total_channels} ({percentage:.1f}%)",
                        ]
                    )
                else:
                    bad_channels_data.append(["Total Removed", str(total_removed)])
        else:
            # Fall back to metadata
            # Look for bad channels in various metadata sections
            for step_name, step_data in run_record["metadata"].items():
                if isinstance(step_data, dict) and "bads" in step_data:
                    display_name = (
                        step_name.replace("step_", "").replace("_", " ").title()
                    )
                    if isinstance(step_data["bads"], list) and step_data["bads"]:
                        bad_channels_data.append(
                            [display_name, ", ".join(step_data["bads"])]
                        )
    except Exception as e:  # pylint: disable=broad-except
        message("warning", f"Error processing bad channels data: {str(e)}")
        bad_channels_data = [["Error processing bad channels", "N/A"]]

    if not bad_channels_data:
        bad_channels_data = [["No bad channels data available", "N/A"]]

    # Create bad channels table with background styling
    bad_channels_table = ReportLabTable(
        [[Paragraph("Source", heading_style), Paragraph("Bad Channels", heading_style)]]
        + bad_channels_data,
        colWidths=[3 * inch, 3 * inch],
    )
    bad_channels_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6FA")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EFF8F9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(bad_channels_table)
    story.append(Spacer(1, 0.2 * inch))

    # Results Summary Section
    story.append(Paragraph("Results Summary", heading_style))

    # Get results summary from metadata
    results_data = []
    try:
        # First try to get results from JSON summary
        if json_summary:
            # Add processing state
            if "proc_state" in json_summary:
                results_data.append(["Processing State", json_summary["proc_state"]])

            # Add exclusion category if any
            if "exclude_category" in json_summary and json_summary["exclude_category"]:
                results_data.append(
                    ["Exclusion Category", json_summary["exclude_category"]]
                )

            # Add export details
            if "export_details" in json_summary:
                export_details = json_summary["export_details"]

                if (
                    "initial_n_epochs" in export_details
                    and "final_n_epochs" in export_details
                ):
                    initial = export_details["initial_n_epochs"]
                    final = export_details["final_n_epochs"]
                    percentage = (final / initial) * 100 if initial else 0
                    results_data.append(
                        ["Epochs Retained", f"{final} / {initial} ({percentage:.1f}%)"]
                    )

                # For duration, use the actual epoch duration values
                if (
                    "initial_duration" in export_details
                    and "final_duration" in export_details
                ):
                    initial = export_details["initial_duration"]
                    final = export_details["final_duration"]

                    # Calculate the actual duration based on epochs and epoch length
                    if "epoch_length" in export_details:
                        epoch_length = export_details["epoch_length"]
                        if (
                            "initial_n_epochs" in export_details
                            and "final_n_epochs" in export_details
                        ):
                            initial_epochs = export_details["initial_n_epochs"]
                            final_epochs = export_details["final_n_epochs"]

                            # Recalculate durations based on epoch count and length
                            initial_duration = initial_epochs * epoch_length
                            final_duration = final_epochs * epoch_length

                            percentage = (
                                (final_duration / initial_duration) * 100
                                if initial_duration
                                else 0
                            )
                            results_data.append(
                                [
                                    "Duration Retained",
                                    f"{final_duration:.1f}s / {initial_duration:.1f}s ({percentage:.1f}%)",  # pylint: disable=line-too-long
                                ]
                            )
                    else:
                        # Use the values directly from export_details if epoch_length is not available # pylint: disable=line-too-long
                        percentage = (final / initial) * 100 if initial else 0
                        results_data.append(
                            [
                                "Duration Retained",
                                f"{final:.1f}s / {initial:.1f}s ({percentage:.1f}%)",
                            ]
                        )

            # Add ICA details
            if "ica_details" in json_summary:
                ica_details = json_summary["ica_details"]
                if "proc_removeComps" in ica_details:
                    removed_comps = ica_details["proc_removeComps"]
                    if isinstance(removed_comps, list):
                        results_data.append(
                            [
                                "Removed ICA Components",
                                ", ".join(map(str, removed_comps)),
                            ]
                        )
        else:
            # Fall back to metadata
            pass

    except Exception as e:  # pylint: disable=broad-except
        message("warning", f"Error processing results data: {str(e)}")
        results_data = [["Error processing results", "N/A"]]

    if not results_data:
        results_data = [["No results data available", "N/A"]]

    # Create results table with background styling
    results_table = ReportLabTable(
        [[Paragraph("Metric", heading_style), Paragraph("Value", heading_style)]]
        + results_data,
        colWidths=[3 * inch, 3 * inch],
    )
    results_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6FA")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5EEF8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(results_table)
    story.append(Spacer(1, 0.2 * inch))

    # Flagged Reasons Section (if any flags exist)
    flagged_reasons_data = []
    try:
        # Check if flagged reasons exist in the JSON summary or processing log
        if json_summary and "flagged_reasons" in json_summary:
            flagged_reasons = json_summary["flagged_reasons"]
            if isinstance(flagged_reasons, list) and flagged_reasons:
                for reason in flagged_reasons:
                    flagged_reasons_data.append([reason])

        # Also check metadata for flags from the task processing log update
        elif "processing_log" in run_record.get("metadata", {}):
            # Try to get flags from processing log metadata if available
            pass  # This would require reading the CSV file, keeping it simple for now

    except Exception as e:  # pylint: disable=broad-except
        message("warning", f"Error processing flagged reasons: {str(e)}")
        flagged_reasons_data = [["Error processing flagged reasons"]]

    # Only add flagged reasons section if there are any flags
    if flagged_reasons_data:
        story.append(Paragraph("Quality Control Flags", heading_style))

        # Create flagged reasons table with background styling
        flagged_reasons_table = ReportLabTable(
            [[Paragraph("Flagged Reason", heading_style)]] + flagged_reasons_data,
            colWidths=[6 * inch],
        )
        flagged_reasons_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6FA")),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.HexColor("#FDF2F2"),
                    ),  # Light red background for flags
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                    (
                        "TEXTCOLOR",
                        (0, 1),
                        (-1, -1),
                        colors.HexColor("#E74C3C"),
                    ),  # Red text for flags
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(flagged_reasons_table)
        story.append(Spacer(1, 0.2 * inch))

    # Output Files Section
    story.append(Paragraph("Output Files", heading_style))

    # Get output files from JSON summary
    output_files_data = []
    try:
        if json_summary and "outputs" in json_summary:
            outputs = json_summary["outputs"]
            for output_file in outputs:
                output_files_data.append([output_file])
        elif derivatives_path and Path(derivatives_path).exists():
            # If no JSON summary, try to get files directly from derivatives directory
            files = list(Path(derivatives_path).glob("*"))
            for file in files:
                if file.is_file():
                    output_files_data.append([file.name])
    except Exception as e:  # pylint: disable=broad-except
        message("warning", f"Error processing output files: {str(e)}")
        output_files_data = [["Error processing output files"]]

    if not output_files_data:
        output_files_data = [["No output files available"]]

    # Create output files table with background styling
    output_files_table = ReportLabTable(
        [[Paragraph("File Name", heading_style)]] + output_files_data,
        colWidths=[6 * inch],
    )
    output_files_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6FA")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EFF8F9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(output_files_table)
    story.append(Spacer(1, 0.2 * inch))

    # Add footer with run information
    footer_style = ParagraphStyle(
        "Footer",
        parent=normal_style,
        fontSize=6,
        textColor=colors.HexColor("#7F8C8D"),
        alignment=1,
        spaceBefore=12,
    )
    footer_text = (
        f"Run ID: {run_id} | "
        f"Task: {run_record.get('task', 'N/A')} | "
        f"Timestamp: {run_record.get('timestamp', 'N/A')}"
    )
    story.append(Paragraph(footer_text, footer_style))

    # Build the PDF
    doc.build(story)

    message("success", f"Report saved to {pdf_path}")

    # If derivatives path is available, also save there
    if derivatives_path:
        try:
            shutil.copy(pdf_path, derivatives_path)
            message("success", f"Report saved to {derivatives_path}")
        except Exception as e:  # pylint: disable=broad-except
            message("warning", f"Could not save to derivatives: {str(e)}")

    return pdf_path


def update_task_processing_log(
    summary_dict: Dict[str, Any], flagged_reasons: list[str] = []
):
    """Update the task-specific processing log CSV file with processing details.


    This function is called by the Pipeline upon exiting the run.


    Parameters
    ----------
    summary_dict : dict
        The summary dictionary containing processing details
    flagged_reasons : list
        Any flags found during the run. Flags are stored in the task instance.

    See Also
    --------
    autoclean.step_functions.reports.create_json_summary : Create a JSON summary of the run metadata

    Notes
    -----
    Although there are safeguards to ensure updates even upon run failure, it may still fail.
    """
    try:
        # Validate required top-level keys
        required_keys = [
            "output_dir",
            "task",
            "timestamp",
            "run_id",
            "proc_state",
            "basename",
            "bids_subject",
        ]
        for key in required_keys:
            if key not in summary_dict:
                message("error", f"Missing required key in summary_dict: {key}")
                return

        # Define CSV path
        csv_path = (
            Path(summary_dict["output_dir"])
            / f"{summary_dict['task']}_processing_log.csv"
        )

        # Safe dictionary access function
        def safe_get(d, *keys, default=""):
            """Safely access nested dictionary keys"""
            current = d
            for key in keys:
                if not isinstance(current, dict):
                    return default
                current = current.get(key, {})
            # Handle case where the value is a dict - return default instead
            if isinstance(current, dict):
                return default
            return current if current is not None else default

        # Function to calculate bad trials safely
        def calculate_bad_trials():
            try:
                initial_epochs = safe_get(
                    summary_dict, "export_details", "initial_n_epochs", default=0
                )
                final_epochs = safe_get(
                    summary_dict, "export_details", "final_n_epochs", default=0
                )

                # Convert to integers safely
                if isinstance(initial_epochs, (int, float, str)):
                    initial_epochs = int(float(initial_epochs)) if initial_epochs else 0
                else:
                    initial_epochs = 0

                if isinstance(final_epochs, (int, float, str)):
                    final_epochs = int(float(final_epochs)) if final_epochs else 0
                else:
                    final_epochs = 0

                # Calculate bad trials
                return initial_epochs - final_epochs
            except Exception:  # pylint: disable=broad-except
                return 0  # Default to 0 if calculation fails

        # Calculate percentages safely with fixed precision
        def safe_percentage(numerator, denominator, default=""):
            try:
                num = float(numerator)
                denom = float(denominator)
                return f"{num / denom:.3f}" if denom != 0 else default
            except (ValueError, TypeError):
                return default

        # Format integers safely
        def safe_int(value, default=""):
            try:
                return str(int(round(float(value))))
            except (ValueError, TypeError):
                return default

        # Format floats safely with configurable precision
        def safe_float(value, decimals=1, default=""):
            try:
                return f"{float(value):.{decimals}f}"
            except (ValueError, TypeError):
                return default

        # Combine flags into a single string
        flags = "; ".join(flagged_reasons) if flagged_reasons else ""

        # Extract details from summary_dict with safe access
        details = {
            "timestamp": summary_dict.get("timestamp", ""),
            "study_user": os.getenv("USERNAME", "unknown"),
            "run_id": summary_dict.get("run_id", ""),
            "proc_state": summary_dict.get("proc_state", ""),
            "subj_basename": Path(summary_dict.get("basename", "")).stem,
            "bids_subject": summary_dict.get("bids_subject", ""),
            "task": summary_dict.get("task", ""),
            "flags": flags,  # Add the new flagged column
            "net_nbchan_orig": str(
                safe_get(summary_dict, "import_details", "net_nbchan_orig", default="")
            ),
            "net_nbchan_post": str(
                safe_get(summary_dict, "export_details", "net_nbchan_post", default="")
            ),
            "proc_badchans": str(
                safe_get(summary_dict, "channel_dict", "removed_channels", default="")
            ),
            "proc_filt_lowcutoff": str(
                safe_get(summary_dict, "processing_details", "l_freq", default="")
            ),
            "proc_filt_highcutoff": str(
                safe_get(summary_dict, "processing_details", "h_freq", default="")
            ),
            "proc_filt_notch": str(
                safe_get(summary_dict, "processing_details", "notch_freqs", default="")
            ),
            "proc_filt_notch_width": str(
                safe_get(summary_dict, "processing_details", "notch_widths", default="")
            ),
            "proc_sRate_raw": safe_int(
                safe_get(summary_dict, "import_details", "sample_rate", default="")
            ),
            "proc_sRate1": str(
                safe_get(summary_dict, "export_details", "srate_post", default="")
            ),
            "proc_xmax_raw": safe_float(
                safe_get(summary_dict, "import_details", "duration", default="")
            ),
            "proc_xmax_post": safe_float(
                safe_get(summary_dict, "export_details", "final_duration", default="")
            ),
        }

        # Add calculated fields
        details.update(
            {
                "proc_xmax_percent": safe_percentage(
                    safe_get(
                        summary_dict, "export_details", "final_duration", default=""
                    ),
                    safe_get(summary_dict, "import_details", "duration", default=""),
                ),
                "epoch_length": str(
                    safe_get(summary_dict, "export_details", "epoch_length", default="")
                ),
                "epoch_limits": str(
                    safe_get(summary_dict, "export_details", "epoch_limits", default="")
                ),
                "epoch_trials": str(
                    safe_get(
                        summary_dict, "export_details", "initial_n_epochs", default=""
                    )
                ),
                "epoch_badtrials": str(calculate_bad_trials()),
                "epoch_percent": safe_percentage(
                    safe_get(
                        summary_dict, "export_details", "final_n_epochs", default=""
                    ),
                    safe_get(
                        summary_dict, "export_details", "initial_n_epochs", default=""
                    ),
                ),
            }
        )

        details.update(
            {
                "proc_nComps": str(
                    safe_get(summary_dict, "ica_details", "proc_nComps", default="")
                ),
                "proc_removeComps": str(
                    safe_get(
                        summary_dict, "ica_details", "proc_removeComps", default=""
                    )
                ),
                "exclude_category": summary_dict.get("exclude_category", ""),
            }
        )

        # Handle CSV operations with appropriate error handling
        if csv_path.exists():
            try:
                # Read existing CSV
                df = pd.read_csv(
                    csv_path, dtype=str
                )  # Force all columns to be string type

                # Ensure all columns exist in DataFrame
                for col in details.keys():
                    if col not in df.columns:
                        df[col] = ""

                # Update or append entry
                subj_basename = details.get("subj_basename", "")
                if subj_basename and subj_basename in df["subj_basename"].values:
                    # Update existing row
                    df.loc[
                        df["subj_basename"] == subj_basename,
                        list(details.keys()),
                    ] = list(
                        details.values()
                    )  # Use list of values instead of pd.Series which can cause index mismatch
                else:
                    # Append new entry
                    df = pd.concat([df, pd.DataFrame([details])], ignore_index=True)
            except Exception as csv_err:  # pylint: disable=broad-except
                message("error", f"Error processing existing CSV: {str(csv_err)}")
                # Create new DataFrame as fallback
                df = pd.DataFrame([details], dtype=str)
        else:
            # Create new DataFrame with all columns as string type
            df = pd.DataFrame([details], dtype=str)

        # Save updated CSV with error handling
        try:
            # Ensure directory exists
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            message(
                "success",
                f"Updated processing log for {details['subj_basename']} in {csv_path}",
            )
        except Exception as save_err:  # pylint: disable=broad-except
            message("error", f"Error saving CSV: {str(save_err)}")
            return

        # -------------------------------------------------------------
        # NEW: Save a *per-file* one-row CSV into the derivatives folder
        # -------------------------------------------------------------
        try:
            # Determine derivatives directory from summary (fallback to output_dir)
            derivatives_root = Path(
                summary_dict.get("derivatives_dir", summary_dict["output_dir"])
            )
            per_file_csv = (
                derivatives_root / f"{details['subj_basename']}_processing_log.csv"
            )

            # Create the directory if it doesn't exist
            per_file_csv.parent.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([details], dtype=str).to_csv(per_file_csv, index=False)
            message("success", f"Saved per-file processing log to {per_file_csv}")

            # Also drop a copy into the `final_files` folder for easy collation
            final_files_dir = (
                Path(summary_dict.get("output_dir", "")) / "bids" / "final_files"
            )
            try:
                final_files_dir.mkdir(parents=True, exist_ok=True)
                final_file_csv = final_files_dir / per_file_csv.name
                pd.DataFrame([details], dtype=str).to_csv(final_file_csv, index=False)
                message("success", f"Saved per-file processing log to {final_file_csv}")
            except Exception as ff_err:  # pylint: disable=broad-except
                message(
                    "error", f"Error saving CSV to final_files directory: {str(ff_err)}"
                )

        except Exception as perfile_err:  # pylint: disable=broad-except
            message("error", f"Error saving per-file CSV: {str(perfile_err)}")

        # Note: Database update removed to avoid audit record conflicts
        # Processing log metadata would be included in completion update if needed
        # The CSV file has been successfully created above

    except Exception as e:  # pylint: disable=broad-except
        message(
            "error",
            f"Error updating processing log: {str(e)}\n{traceback.format_exc()}",
        )


def create_json_summary(run_id: str, flagged_reasons: list[str] = []) -> dict:
    """
    Creates a JSON summary of the run metadata.
    The main purpose of this is to create a summary of the run for the autoclean report.

    Parameters
    ----------
    run_id : str
        The run ID to create a JSON summary for

    Returns
    -------
    summary_dict : dict
        The JSON summary of the run metadata
    """
    run_record = get_run_record(run_id)
    if not run_record:
        message("error", f"No run record found for run ID: {run_id}")
        return

    metadata = run_record.get("metadata", {})

    # Create a JSON summary of the metadata
    if "step_create_bids_path" in run_record["metadata"]:
        bids_info = run_record["metadata"]["step_create_bids_path"]
        derivatives_dir = Path(bids_info["derivatives_dir"])
    else:
        message(
            "warning",
            "Failed to create json summary -> Could not find bids info in metadata.",
        )
        return {}

    outputs = [file.name for file in derivatives_dir.iterdir() if file.is_file()]

    # Determine processing state and exclusion category
    proc_state = "completed_files"
    exclude_category = ""
    if not run_record.get("success", False):
        error_msg = run_record.get("error") or ""
        if "line noise" in error_msg:
            proc_state = "LINE NOISE"
            exclude_category = "Excessive Line Noise"
        elif "insufficient data" in error_msg:
            proc_state = "INSUFFICIENT_DATA"
            exclude_category = "Insufficient Data"
        else:
            proc_state = "ERROR"
            exclude_category = f"Processing Error: {error_msg[:100]}"

    # FIND BAD CHANNELS
    channel_dict = {}
    if "step_clean_bad_channels" in metadata:
        channel_dict["step_clean_bad_channels"] = metadata["step_clean_bad_channels"][
            "bads"
        ]
        channel_dict["uncorrelated_channels"] = metadata["step_clean_bad_channels"][
            "uncorrelated_channels"
        ]
        channel_dict["deviation_channels"] = metadata["step_clean_bad_channels"][
            "deviation_channels"
        ]
        channel_dict["ransac_channels"] = metadata["step_clean_bad_channels"][
            "ransac_channels"
        ]

    flagged_chs_file = None
    for file_name in outputs:
        if file_name.endswith("FlaggedChs.tsv"):
            flagged_chs_file = file_name
            break

    if flagged_chs_file:
        with open(derivatives_dir / flagged_chs_file, "r", encoding="utf8") as f:
            # Skip the header line
            next(f)
            # Read each line and extract the label and channel name
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    label, channel = parts
                    if label not in channel_dict:
                        channel_dict[label] = []
                    channel_dict[label].append(channel)

    # Get all bad channels
    bad_channels = [
        channel for channels in channel_dict.values() for channel in channels
    ]
    # Remove duplicates while preserving order
    unique_bad_channels = []
    for channel in bad_channels:
        if channel not in unique_bad_channels:
            unique_bad_channels.append(channel)
    channel_dict["removed_channels"] = unique_bad_channels

    output_dir = Path(metadata["step_prepare_directories"]["bids"]).parent

    # FIND IMPORT DETAILS
    import_details = {}
    dropped_channels = []
    if "step_drop_outerlayer" in metadata:
        try:
            dropped_channels = metadata["step_drop_outerlayer"][
                "dropped_outer_layer_channels"
            ]
            if dropped_channels is None:
                dropped_channels = []
            import_details["dropped_channels"] = dropped_channels
        except Exception as e:  # pylint: disable=broad-except
            message("error", f"Failed to load dropped channels: {str(e)}")

    if "import_eeg" in metadata:
        import_details["sample_rate"] = metadata["import_eeg"]["sampleRate"]
        import_details["net_nbchan_orig"] = metadata["import_eeg"]["channelCount"]
        import_details["duration"] = metadata["import_eeg"]["durationSec"]
        import_details["basename"] = metadata["import_eeg"]["unprocessedFile"]
        original_channel_count = int(metadata["import_eeg"]["channelCount"]) - int(
            len(dropped_channels)
        )
    else:
        message("error", "No import details found")
        return {}

    # FIND PROCESSING DETAILS - use verified applied values, not requested parameters
    processing_details = {}
    if "step_filter_data" in metadata:
        # Use actual applied values, not filter_args (which are requested values)
        filter_metadata = metadata["step_filter_data"]
        processing_details["h_freq"] = filter_metadata.get("applied_h_freq")
        processing_details["l_freq"] = filter_metadata.get("applied_l_freq")
        processing_details["notch_freqs"] = filter_metadata.get("applied_notch_freqs")
        processing_details["notch_widths"] = filter_metadata.get("applied_notch_widths")
        # Also record verification of what was actually achieved
        processing_details["filtered_sfreq"] = filter_metadata.get("filtered_sfreq")
        processing_details["filtered_n_channels"] = filter_metadata.get(
            "filtered_n_channels"
        )
    if "step_resample_data" in metadata:
        # Use actual achieved sample rate, not target (which is requested value)
        resample_metadata = metadata["step_resample_data"]
        processing_details["target_sfreq"] = resample_metadata.get("target_sfreq")
        processing_details["actual_sfreq"] = resample_metadata.get("actual_sfreq")
        # Record verification metrics
        processing_details["resampled_n_samples"] = resample_metadata.get(
            "resampled_n_samples"
        )
    if "step_trim_edges" in metadata:
        processing_details["trim_duration"] = metadata["step_trim_edges"][
            "trim_duration"
        ]
    if "step_crop_duration" in metadata:
        processing_details["crop_duration"] = metadata["step_crop_duration"][
            "crop_duration"
        ]
        processing_details["crop_start"] = metadata["step_crop_duration"]["crop_start"]
        processing_details["crop_end"] = metadata["step_crop_duration"]["crop_end"]
    if "step_assign_eog_channels" in metadata:
        processing_details["eog_channels"] = metadata["step_assign_eog_channels"][
            "assigned_eog_channels"
        ]
    if "step_drop_outerlayer" in metadata:
        processing_details["dropped_channels"] = metadata["step_drop_outerlayer"][
            "dropped_outer_layer_channels"
        ]
        processing_details["original_channel_count"] = metadata["step_drop_outerlayer"][
            "original_channel_count"
        ]
        processing_details["new_channel_count"] = metadata["step_drop_outerlayer"][
            "new_channel_count"
        ]
    if "step_rereference_data" in metadata:
        processing_details["ref_type"] = metadata["step_rereference_data"][
            "new_ref_type"
        ]

    # FIND EXPORT DETAILS - Quality Control Integrity
    # Priority order: 1) Actual measured values from exported files (best QC)
    #                2) Verified values from processing steps
    #                3) Calculated values (flagged as such for transparency)
    export_details = {}
    if "save_epochs_to_set" in metadata:
        save_epochs_to_set = metadata["save_epochs_to_set"]
        epoch_length = save_epochs_to_set["tmax"] - save_epochs_to_set["tmin"]
        export_details["epoch_length"] = epoch_length
        export_details["final_n_epochs"] = save_epochs_to_set["n_epochs"]

        # Use actual duration from final exported data (true QC verification)
        if "actual_duration" in save_epochs_to_set:
            export_details["final_duration"] = save_epochs_to_set["actual_duration"]
            export_details["final_duration_verified"] = True
        else:
            # Fallback: calculate expected duration (mark as calculated, not measured)
            export_details["final_duration"] = (
                epoch_length * save_epochs_to_set["n_epochs"]
            )
            export_details["final_duration_calculated"] = True
        # Use actual channel count from the final exported data, not calculations
        if (
            "save_epochs_to_set" in metadata
            and "n_channels" in metadata["save_epochs_to_set"]
        ):
            # Use actual channel count from exported file (true QC verification)
            export_details["net_nbchan_post"] = metadata["save_epochs_to_set"][
                "n_channels"
            ]
        elif original_channel_count and unique_bad_channels:
            # Fallback: calculate based on removed channels (mark as calculated)
            export_details["net_nbchan_post"] = original_channel_count - len(
                unique_bad_channels
            )
            export_details["net_nbchan_post_calculated"] = True
        else:
            # Last resort: use original count (mark as unverified)
            export_details["net_nbchan_post"] = original_channel_count
            export_details["net_nbchan_post_unverified"] = True

    if "step_create_regular_epochs" in metadata:
        epoch_metadata = metadata["step_create_regular_epochs"]
    elif "step_create_eventid_epochs" in metadata:
        epoch_metadata = metadata["step_create_eventid_epochs"]
    elif "step_create_sl_epochs" in metadata:
        epoch_metadata = metadata["step_create_sl_epochs"]
    else:
        message(
            "warning",
            "No epoch creation details found. Processing details may be missing",
        )
        epoch_metadata = None

    if epoch_metadata is not None:
        export_details["initial_n_epochs"] = epoch_metadata["initial_epoch_count"]
        export_details["initial_duration"] = epoch_metadata["initial_duration"]
        # Use actual sample rate from final exported data (best QC verification)
        if (
            "save_epochs_to_set" in metadata
            and "actual_sfreq" in metadata["save_epochs_to_set"]
        ):
            export_details["srate_post"] = metadata["save_epochs_to_set"][
                "actual_sfreq"
            ]
        elif (
            "step_resample_data" in metadata
            and "actual_sfreq" in metadata["step_resample_data"]
        ):
            # Second choice: actual sample rate from resampling step
            export_details["srate_post"] = metadata["step_resample_data"][
                "actual_sfreq"
            ]
        else:
            # Fallback: calculate from epoch metadata (mark as calculated)
            export_details["srate_post"] = (
                epoch_metadata["single_epoch_samples"]
            ) / epoch_metadata["single_epoch_duration"]
            export_details["srate_post_calculated"] = True
        export_details["epoch_limits"] = [
            epoch_metadata["tmin"],
            epoch_metadata["tmax"],
        ]

    # FIND ICA DETAILS - collect from multiple step names
    ica_details = {}

    # Get total component count from ICA fitting step
    if "step_run_ica" in metadata:
        ica_run_metadata = metadata["step_run_ica"]
        message(
            "debug", f"step_run_ica metadata structure: {list(ica_run_metadata.keys())}"
        )
        # Access nested 'ica' metadata structure
        if "ica" in ica_run_metadata:
            ica_nested = ica_run_metadata["ica"]
            ica_details["proc_nComps"] = ica_nested.get("ica_components", "")
            ica_details["ica_method"] = ica_nested.get("ica_method", "")
            message(
                "debug",
                f"Found ICA components from nested structure: {ica_details['proc_nComps']}",
            )
        else:
            # Fallback to direct access for legacy metadata
            ica_details["proc_nComps"] = ica_run_metadata.get("ica_components", "")
            ica_details["ica_method"] = ica_run_metadata.get("ica_method", "")
            message(
                "debug",
                f"Found ICA components from legacy structure: {ica_details['proc_nComps']}",
            )

    # Get classification method from classification step
    if "classify_ica_components" in metadata:
        ica_classify_metadata = metadata["classify_ica_components"]
        message(
            "debug",
            f"classify_ica_components metadata structure: {list(ica_classify_metadata.keys())}",
        )
        # Access nested 'ica' metadata structure
        if "ica" in ica_classify_metadata:
            ica_nested = ica_classify_metadata["ica"]
            ica_details["classification_method"] = ica_nested.get(
                "classification_method", ""
            )
            # Also get component count from here if not found above
            if "proc_nComps" not in ica_details:
                ica_details["proc_nComps"] = ica_nested.get("ica_components", "")
            message(
                "debug",
                f"Found ICA classification method from nested structure: {ica_details.get('classification_method', 'N/A')}",
            )
        else:
            # Fallback to direct access for legacy metadata
            ica_details["classification_method"] = ica_classify_metadata.get(
                "classification_method", ""
            )
            if "proc_nComps" not in ica_details:
                ica_details["proc_nComps"] = ica_classify_metadata.get(
                    "ica_components", ""
                )
            message(
                "debug",
                f"Found ICA classification method from legacy structure: {ica_details.get('classification_method', 'N/A')}",
            )

    # Get rejected components from rejection step
    if "step_apply_ica_component_rejection" in metadata:
        ica_rejection_metadata = metadata["step_apply_ica_component_rejection"]
        message(
            "debug",
            f"step_apply_ica_component_rejection metadata structure: {list(ica_rejection_metadata.keys())}",
        )
        # Access nested 'ica' metadata structure (same as other ICA steps)
        if "ica" in ica_rejection_metadata:
            ica_nested = ica_rejection_metadata["ica"]
            rejected_components = ica_nested.get("final_excluded_indices", [])
            message(
                "debug",
                f"Found rejected components from nested structure: {rejected_components}",
            )
        else:
            # Fallback to direct access for legacy metadata
            rejected_components = ica_rejection_metadata.get(
                "final_excluded_indices", []
            )
            message(
                "debug",
                f"Found rejected components from legacy structure: {rejected_components}",
            )

        ica_details["proc_removeComps"] = rejected_components
        # Also record number of rejected components
        if isinstance(rejected_components, list):
            ica_details["num_rejected_components"] = len(rejected_components)

    # Debug: Show final ICA details collected
    message("debug", f"Final ICA details collected: {ica_details}")

    if "step_detect_dense_oscillatory_artifacts" in metadata:
        ref_artifacts = metadata["step_detect_dense_oscillatory_artifacts"][
            "artifacts_detected"
        ]
        processing_details["ref_artifacts"] = ref_artifacts

    summary_dict = {
        "run_id": run_id,
        "task": run_record["task"],
        "bids_subject": f"sub-{bids_info['bids_subject']}",
        "timestamp": run_record["created_at"],
        "basename": import_details["basename"],
        "proc_state": proc_state,
        "exclude_category": exclude_category,
        "flagged_reasons": flagged_reasons,  # Add flagged reasons to the summary
        "import_details": import_details,
        "processing_details": processing_details,
        "export_details": export_details,
        "ica_details": ica_details,
        "channel_dict": channel_dict,
        "outputs": outputs,
        "output_dir": str(output_dir),
        "derivatives_dir": str(derivatives_dir),
    }

    message("success", f"Created JSON summary for run {run_id}")

    # Note: Database update moved to pipeline completion to avoid audit record conflicts
    # The JSON summary will be saved when the run is marked as completed

    return summary_dict


def generate_bad_channels_tsv(summary_dict: Dict[str, Any]) -> None:
    """
    Generates a tsv file containing the bad channels and reasons for flagging for the run.

    Parameters
    ----------
    summary_dict : dict
        The summary dictionary containing the run metadata
    """
    try:
        channel_dict = summary_dict["channel_dict"]
    except Exception as e:  # pylint: disable=broad-except
        message(
            "warning",
            f"Could not generate bad channels tsv -> No channel dict found in summary dict: {str(e)}",  # pylint: disable=line-too-long
        )
        return

    try:
        noisy_channels = channel_dict.get("noisy_channels", [])
        uncorrelated_channels = channel_dict.get("uncorrelated_channels", [])
        deviation_channels = channel_dict.get("deviation_channels", [])
        bridged_channels = channel_dict.get("bridged_channels", [])
        rank_channels = channel_dict.get("rank_channels", [])
        ransac_channels = channel_dict.get("ransac_channels", [])
    except Exception as e:  # pylint: disable=broad-except
        message(
            "warning",
            f"Could not generate bad channels tsv -> Failed to fetch bad channels: {str(e)}",
        )
        return

    with open(
        f"{summary_dict['derivatives_dir']}/FlaggedChs.tsv", "w", encoding="utf8"
    ) as f:
        f.write("label\tchannel\n")
        for channel in noisy_channels:
            f.write("Noisy\t" + channel + "\n")
        for channel in uncorrelated_channels:
            f.write("Uncorrelated\t" + channel + "\n")
        for channel in deviation_channels:
            f.write("Deviation\t" + channel + "\n")
        for channel in ransac_channels:
            f.write("Ransac\t" + channel + "\n")
        for channel in bridged_channels:
            f.write("Bridged\t" + channel + "\n")
        for channel in rank_channels:
            f.write("Rank\t" + channel + "\n")

    message("success", f"Bad channels tsv generated for {summary_dict['run_id']}")
