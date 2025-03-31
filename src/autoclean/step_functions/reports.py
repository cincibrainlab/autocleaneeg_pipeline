# src/autoclean/step_functions/reports.py
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

import matplotlib
import mne
import numpy as np
import pandas as pd
import pylossless as ll
from matplotlib.gridspec import GridSpec
from mne_bids import BIDSPath

__all__ = [
    "create_run_report",
    "update_task_processing_log",
    "create_json_summary",
    "generate_bad_channels_tsv",
]

# Force matplotlib to use non-interactive backend for async operations
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages

# ReportLab imports for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)
from reportlab.platypus import Table as ReportLabTable
from reportlab.platypus import (
    TableStyle,
)

from autoclean.utils.database import get_run_record, manage_database
from autoclean.utils.logging import message
from autoclean.utils.montage import get_standard_set_in_montage, validate_channel_set


def create_run_report(run_id: str, autoclean_dict: dict = None) -> None:
    """
    Creates a scientific report in PDF format using ReportLab based on the run metadata.

    Args:
        run_id (str): The run ID to generate a report for
        Optional: autoclean_dict (dict): The autoclean dictionary
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

    # Check if JSON summary exists and use it if available
    json_summary = None
    if "json_summary" in run_record["metadata"]:
        json_summary = run_record["metadata"]["json_summary"]
        message("info", "Using JSON summary for report generation")
    
    # If no JSON summary, create it
    if not json_summary:
        message("warning", "No json summary found, run report may be missing or incomplete")
        json_summary = {}
    
    # Set up BIDS path
    bids_path = None
    try:
        if autoclean_dict:
            try:
                bids_path = autoclean_dict["bids_path"]
            except Exception:
                message(
                    "warning",
                    "Failed to get BIDS path from autoclean_dict: Trying metadata",
                )
        
        if not bids_path:
            if json_summary and "bids_subject" in json_summary:
                # Try to reconstruct from JSON summary
                if "step_convert_to_bids" in run_record["metadata"]:
                    bids_info = run_record["metadata"]["step_convert_to_bids"]
                    if bids_info:
                        # Reconstruct BIDSPath object
                        bids_path = BIDSPath(
                            subject=bids_info["bids_subject"],
                            session=bids_info["bids_session"],
                            task=bids_info["bids_task"],
                            run=bids_info["bids_run"],
                            datatype=bids_info["bids_datatype"],
                            root=bids_info["bids_root"],
                            suffix=bids_info["bids_suffix"],
                            extension=bids_info["bids_extension"],
                        )

        task = run_record["task"]
        config_path = run_record["metadata"]["entrypoint"]["tasks"][task][
            "lossless_config"
        ]
        derivative_name = "pylossless"
        pipeline = ll.LosslessPipeline(config_path)
        derivatives_path = pipeline.get_derivative_path(bids_path, derivative_name)
        derivatives_dir = Path(derivatives_path.directory)
        derivatives_path = str(
            derivatives_path.copy().update(suffix="report", extension=".pdf")
        )
    except Exception as e:
        message(
            "warning",
            f"Failed to get BIDS path: {str(e)} : Saving only to metadata directory",
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

    steps_style = ParagraphStyle(
        "Steps",
        parent=normal_style,
        fontSize=7,
        leading=10,
        spaceBefore=1,
        spaceAfter=1,
    )

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

    # Tables layout with better styling
    data = [
        [
            Paragraph("Import Information", heading_style),
            Paragraph("Preprocessing Parameters", heading_style),
            Paragraph("Lossless Configuration", heading_style),
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

    except Exception as e:
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

    # Middle column: Preprocessing parameters
    preproc_info = []
    try:
        if json_summary and "processing_details" in json_summary:
            # Use data from JSON summary
            processing_details = json_summary["processing_details"]
            
            preproc_info.extend(
                [
                    [
                        "Filter",
                        f"{processing_details.get('l_freq', 'N/A')}-{processing_details.get('h_freq', 'N/A')} Hz",
                    ],
                    [
                        "Notch",
                        f"{processing_details.get('notch_freqs', ['N/A'])[0]} Hz",
                    ],
                ]
            )
            
            # Add more preprocessing info if available in JSON summary
            if "export_details" in json_summary:
                export_details = json_summary["export_details"]
                if "srate_post" in export_details:
                    preproc_info.append(["Resampled", f"{export_details['srate_post']} Hz"])
        else:
            # Fall back to direct metadata access
            if "entrypoint" in run_record["metadata"]:
                task_config = run_record["metadata"]["entrypoint"]["tasks"][
                    run_record["metadata"]["entrypoint"]["task"]
                ]["settings"]
                preproc_info.extend(
                    [
                        [
                            "Resample",
                            (
                                f"{task_config['resample_step']['value']} Hz"
                                if task_config["resample_step"]["enabled"]
                                else "Disabled"
                            ),
                        ],
                        [
                            "Trim",
                            (
                                f"{task_config['trim_step']['value']} sec"
                                if task_config["trim_step"]["enabled"]
                                else "Disabled"
                            ),
                        ],
                        [
                            "Reference",
                            (
                                str(task_config["reference_step"]["value"])
                                if isinstance(task_config["reference_step"]["value"], str)
                                else (
                                    ", ".join(task_config["reference_step"]["value"])
                                    if isinstance(
                                        task_config["reference_step"]["value"], list
                                    )
                                    else (
                                        "Disabled"
                                        if task_config["reference_step"]["enabled"]
                                        else "Disabled"
                                    )
                                )
                            ),
                        ],
                    ]
                )
    except Exception as e:
        message("warning", f"Error processing preprocessing parameters: {str(e)}")
        preproc_info = [["Error processing parameters", "N/A"]]

    if not preproc_info:
        preproc_info = [["No preprocessing data available", "N/A"]]

    preproc_table = ReportLabTable(preproc_info, colWidths=[0.7 * inch, 1.3 * inch])
    preproc_table.setStyle(
        TableStyle(
            [
                *table_style._cmds,
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#EFF8F9"),
                ),
            ]
        )
    )

    # Right column: Lossless settings
    lossless_info = []
    try:
        if json_summary and "processing_details" in json_summary and "ica_details" in json_summary:
            # Use data from JSON summary
            processing_details = json_summary["processing_details"]
            ica_details = json_summary["ica_details"]
            
            lossless_info.extend(
                [
                    [
                        "Filter",
                        f"{processing_details.get('l_freq', 'N/A')}-{processing_details.get('h_freq', 'N/A')} Hz",
                    ],
                    [
                        "Notch",
                        f"{processing_details.get('notch_freqs', ['N/A'])[0]} Hz",
                    ],
                    ["ICA Method", ica_details.get("proc_method", "N/A")],
                    [
                        "Components",
                        str(ica_details.get("proc_nComps", "N/A")),
                    ],
                ]
            )
        else:
            # Fall back to direct metadata access
            if "step_run_pylossless" in run_record["metadata"]:
                lossless_config = run_record["metadata"]["step_run_pylossless"].get(
                    "pylossless_config", {}
                )
                filter_args = lossless_config.get("filtering", {}).get("filter_args", {})
                ica_args = lossless_config.get("ica", {}).get("ica_args", {})
                lossless_info.extend(
                    [
                        [
                            "Filter",
                            f"{filter_args.get('l_freq', 'N/A')}-{filter_args.get('h_freq', 'N/A')} Hz",
                        ],
                        [
                            "Notch",
                            f"{lossless_config.get('filtering', {}).get('notch_filter_args', {}).get('freqs', ['N/A'])[0]} Hz",
                        ],
                        ["ICA", ica_args.get("run2", {}).get("method", "N/A")],
                        [
                            "Components",
                            str(ica_args.get("run2", {}).get("n_components", "N/A")),
                        ],
                    ]
                )
    except Exception as e:
        message("warning", f"Error processing lossless settings: {str(e)}")
        lossless_info = [["Error processing lossless data", "N/A"]]

    if not lossless_info:
        lossless_info = [["No lossless data available", "N/A"]]

    lossless_table = ReportLabTable(lossless_info, colWidths=[0.7 * inch, 1.3 * inch])
    lossless_table.setStyle(
        TableStyle(
            [
                *table_style._cmds,
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F5EEF8"),
                ),
            ]
        )
    )

    # Add tables to main layout with spacing
    data.append([import_table, preproc_table, lossless_table])
    main_table = ReportLabTable(data, colWidths=[2 * inch, 2 * inch, 2 * inch])
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
    except Exception as e:
        message("warning", f"Error processing steps data: {str(e)}")
        steps_data = [["Error processing steps"]]

    if not steps_data:
        steps_data = [["No processing steps data available"]]

    # Create steps table with background styling
    steps_table = ReportLabTable(
        [
            [Paragraph("Processing Step", heading_style)]
        ] + steps_data, 
        colWidths=[6 * inch]
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
                if category != "removed_channels" and channels:  # Skip the combined list
                    display_category = category.replace("step_", "").replace("_", " ").title()
                    bad_channels_data.append([display_category, ", ".join(channels)])
            
            # Add total count
            if "removed_channels" in channel_dict:
                total_removed = len(channel_dict["removed_channels"])
                if "import_details" in json_summary and "net_nbchan_orig" in json_summary["import_details"]:
                    total_channels = json_summary["import_details"]["net_nbchan_orig"]
                    percentage = (total_removed / total_channels) * 100 if total_channels else 0
                    bad_channels_data.append(
                        ["Total Removed", f"{total_removed} / {total_channels} ({percentage:.1f}%)"]
                    )
                else:
                    bad_channels_data.append(["Total Removed", str(total_removed)])
        else:
            # Fall back to metadata
            # Look for bad channels in various metadata sections
            for step_name, step_data in run_record["metadata"].items():
                if isinstance(step_data, dict) and "bads" in step_data:
                    display_name = step_name.replace("step_", "").replace("_", " ").title()
                    if isinstance(step_data["bads"], list) and step_data["bads"]:
                        bad_channels_data.append(
                            [display_name, ", ".join(step_data["bads"])]
                        )
    except Exception as e:
        message("warning", f"Error processing bad channels data: {str(e)}")
        bad_channels_data = [["Error processing bad channels", "N/A"]]

    if not bad_channels_data:
        bad_channels_data = [["No bad channels data available", "N/A"]]

    # Create bad channels table with background styling
    bad_channels_table = ReportLabTable(
        [
            [Paragraph("Source", heading_style), Paragraph("Bad Channels", heading_style)]
        ] + bad_channels_data, 
        colWidths=[3 * inch, 3 * inch]
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
                results_data.append(["Exclusion Category", json_summary["exclude_category"]])
            
            # Add export details
            if "export_details" in json_summary:
                export_details = json_summary["export_details"]
                
                if "initial_n_epochs" in export_details and "final_n_epochs" in export_details:
                    initial = export_details["initial_n_epochs"]
                    final = export_details["final_n_epochs"]
                    percentage = (final / initial) * 100 if initial else 0
                    results_data.append(
                        ["Epochs Retained", f"{final} / {initial} ({percentage:.1f}%)"]
                    )
                
                # For duration, use the actual epoch duration values
                if "initial_duration" in export_details and "final_duration" in export_details:
                    initial = export_details["initial_duration"]
                    final = export_details["final_duration"]
                    
                    # Calculate the actual duration based on epochs and epoch length
                    if "epoch_length" in export_details:
                        epoch_length = export_details["epoch_length"]
                        if "initial_n_epochs" in export_details and "final_n_epochs" in export_details:
                            initial_epochs = export_details["initial_n_epochs"]
                            final_epochs = export_details["final_n_epochs"]
                            
                            # Recalculate durations based on epoch count and length
                            initial_duration = initial_epochs * epoch_length
                            final_duration = final_epochs * epoch_length
                            
                            percentage = (final_duration / initial_duration) * 100 if initial_duration else 0
                            results_data.append(
                                ["Duration Retained", f"{final_duration:.1f}s / {initial_duration:.1f}s ({percentage:.1f}%)"]
                            )
                    else:
                        # Use the values directly from export_details if epoch_length is not available
                        percentage = (final / initial) * 100 if initial else 0
                        results_data.append(
                            ["Duration Retained", f"{final:.1f}s / {initial:.1f}s ({percentage:.1f}%)"]
                        )
            
            # Add ICA details
            if "ica_details" in json_summary:
                ica_details = json_summary["ica_details"]
                if "proc_removeComps" in ica_details:
                    removed_comps = ica_details["proc_removeComps"]
                    if isinstance(removed_comps, list):
                        results_data.append(
                            ["Removed ICA Components", ", ".join(map(str, removed_comps))]
                        )
        else:
            # Fall back to metadata
            # Add any available results data from metadata
            if "step_run_ll_rejection_policy" in run_record["metadata"]:
                rejection_data = run_record["metadata"]["step_run_ll_rejection_policy"]
                if "ica_components" in rejection_data:
                    components = rejection_data["ica_components"]
                    if isinstance(components, list):
                        results_data.append(
                            ["Removed ICA Components", ", ".join(map(str, components))]
                        )
    except Exception as e:
        message("warning", f"Error processing results data: {str(e)}")
        results_data = [["Error processing results", "N/A"]]

    if not results_data:
        results_data = [["No results data available", "N/A"]]

    # Create results table with background styling
    results_table = ReportLabTable(
        [
            [Paragraph("Metric", heading_style), Paragraph("Value", heading_style)]
        ] + results_data, 
        colWidths=[3 * inch, 3 * inch]
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
    
    # Output Files Section
    story.append(Paragraph("Output Files", heading_style))
    
    # Get output files from JSON summary
    output_files_data = []
    try:
        if json_summary and "outputs" in json_summary:
            outputs = json_summary["outputs"]
            for output_file in outputs:
                output_files_data.append([output_file])
        elif derivatives_dir and derivatives_dir.exists():
            # If no JSON summary, try to get files directly from derivatives directory
            files = list(derivatives_dir.glob("*"))
            for file in files:
                if file.is_file():
                    output_files_data.append([file.name])
    except Exception as e:
        message("warning", f"Error processing output files: {str(e)}")
        output_files_data = [["Error processing output files"]]
    
    if not output_files_data:
        output_files_data = [["No output files available"]]
    
    # Create output files table with background styling
    output_files_table = ReportLabTable(
        [
            [Paragraph("File Name", heading_style)]
        ] + output_files_data, 
        colWidths=[6 * inch]
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
            message("success", f"Report also saved to {derivatives_path}")
        except Exception as e:
            message("warning", f"Could not save to derivatives: {str(e)}")

    return pdf_path

def update_task_processing_log(summary_dict: Dict[str, Any], flagged_reasons: list[str] = []) -> None:
    """Update the task-specific processing log CSV file with details about the current file.

    Args:
        summary_dict: The summary dictionary containing processing details
        flagged: Whether the task is flagged
        flagged_reasons: The reasons for flagging
    """
    try:
        # Validate required top-level keys
        required_keys = ["output_dir", "task", "timestamp", "run_id", "proc_state", 
                        "basename", "bids_subject"]
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
            return current if current is not None else default
        
        # Calculate percentages safely
        def safe_percentage(numerator, denominator, default=""):
            try:
                num = float(numerator)
                denom = float(denominator)
                return str(num / denom) if denom != 0 else default
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
            "net_nbchan_orig": str(safe_get(summary_dict, "import_details", "net_nbchan_orig", default="")),
            "net_nbchan_post": str(safe_get(summary_dict, "export_details", "net_nbchan_post", default="")),
            "proc_badchans": str(safe_get(summary_dict, "channel_dict", "removed_channels", default="")),
            "proc_filt_lowcutoff": str(safe_get(summary_dict, "processing_details", "l_freq", default="")),
            "proc_filt_highcutoff": str(safe_get(summary_dict, "processing_details", "h_freq", default="")),
            "proc_filt_notch": str(safe_get(summary_dict, "processing_details", "notch_freqs", default="")),
            "proc_filt_notch_width": str(safe_get(summary_dict, "processing_details", "notch_widths", default="")),
            "proc_sRate_raw": str(safe_get(summary_dict, "import_details", "sample_rate", default="")),
            "proc_sRate1": str(safe_get(summary_dict, "export_details", "srate_post", default="")),
            "proc_xmax_raw": str(safe_get(summary_dict, "import_details", "duration", default="")),
            "proc_xmax_post": str(safe_get(summary_dict, "export_details", "final_duration", default="")),
        }
        
        # Add calculated fields
        details.update({
            "proc_xmax_percent": safe_percentage(
                safe_get(summary_dict, "export_details", "final_duration", default=""),
                safe_get(summary_dict, "import_details", "duration", default=""),
            ),
            "epoch_length": str(safe_get(summary_dict, "export_details", "epoch_length", default="")),
            "epoch_limits": str(safe_get(summary_dict, "export_details", "epoch_limits", default="")),
            "epoch_trials": str(safe_get(summary_dict, "export_details", "initial_n_epochs", default="")),
            "epoch_badtrials": str(
                int(safe_get(summary_dict, "export_details", "initial_n_epochs", default=0)) - 
                int(safe_get(summary_dict, "export_details", "final_n_epochs", default=0))
            ),
            "epoch_percent": safe_percentage(
                safe_get(summary_dict, "export_details", "final_n_epochs", default=""),
                safe_get(summary_dict, "export_details", "initial_n_epochs", default=""),
            ),
        })

        details.update({
            "proc_nComps": str(safe_get(summary_dict, "ica_details", "proc_nComps", default="")),
            "proc_removeComps": str(safe_get(summary_dict, "ica_details", "proc_removeComps", default="")),
            "exclude_category": summary_dict.get("exclude_category", ""),
        })
        
        # Handle CSV operations with appropriate error handling
        if csv_path.exists():
            try:
                # Read existing CSV
                df = pd.read_csv(csv_path, dtype=str)  # Force all columns to be string type

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
                    ] = list(details.values())  # Use list of values instead of pd.Series which can cause index mismatch
                else:
                    # Append new entry
                    df = pd.concat([df, pd.DataFrame([details])], ignore_index=True)
            except Exception as csv_err:
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
        except Exception as save_err:
            message("error", f"Error saving CSV: {str(save_err)}")
            return

        # Update run record with CSV path
        try:
            metadata = {
                "processing_log": {
                    "creationDateTime": datetime.now().isoformat(),
                    "csv_path": str(csv_path),
                }
            }
            manage_database(
                operation="update",
                update_record={"run_id": summary_dict.get("run_id", ""), "metadata": metadata},
            )

        except Exception as db_err:
            message("error", f"Error updating database: {str(db_err)}")

    except Exception as e:
        message("error", f"Error updating processing log: {str(e)}\n{traceback.format_exc()}")


def create_json_summary(run_id: str) -> None:
    run_record = get_run_record(run_id)
    if not run_record:
        message("error", f"No run record found for run ID: {run_id}")
        return

    metadata = run_record.get("metadata", {})

    # Create a JSON summary of the metadata
    try:
        if "step_convert_to_bids" in run_record["metadata"]:
            bids_info = run_record["metadata"]["step_convert_to_bids"]
            if bids_info:
                # Reconstruct BIDSPath object
                bids_path = BIDSPath(
                    subject=bids_info["bids_subject"],
                    session=bids_info["bids_session"],
                    task=bids_info["bids_task"],
                    run=bids_info["bids_run"],
                    datatype=bids_info["bids_datatype"],
                    root=bids_info["bids_root"],
                    suffix=bids_info["bids_suffix"],
                    extension=bids_info["bids_extension"],
                )

                config_path = run_record["lossless_config"]
                derivative_name = "pylossless"
                pipeline = ll.LosslessPipeline(config_path)
                derivatives_path = pipeline.get_derivative_path(bids_path, derivative_name)
                derivatives_dir = Path(derivatives_path.directory)
        else:
            message("warning", "Failed to create json summary -> Could not find bids info in metadata.")
            return {}
        
    except Exception as e:
        message("error", f"Failed to get derivatives path: {str(e)}")
        return {}

    outputs = [file.name for file in derivatives_dir.iterdir() if file.is_file()]

    # Determine processing state and exclusion category
    proc_state = "postcomps"
    exclude_category = ""
    if not run_record.get("success", False):
        error_msg = run_record.get("error", "").lower()
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
        

    if "step_custom_pylossless_pipeline" in metadata:
        channel_dict["step_custom_pylossless_pipeline"] = metadata["step_custom_pylossless_pipeline"][
            "bads"
        ]
        channel_dict["noisy_channels"] = metadata["step_custom_pylossless_pipeline"][
            "noisy_channels"
        ]
        channel_dict["uncorrelated_channels"] = metadata["step_custom_pylossless_pipeline"][
            "uncorrelated_channels"
        ]
        channel_dict["bridged_channels"] = metadata["step_custom_pylossless_pipeline"][
            "bridged_channels"
        ]
        channel_dict["rank_channels"] = metadata["step_custom_pylossless_pipeline"][
            "rank_channels"
        ]

    flagged_chs_file = None
    for file_name in outputs:
        if file_name.endswith("FlaggedChs.tsv"):
            flagged_chs_file = file_name
            break

    if flagged_chs_file:
        with open(derivatives_dir / flagged_chs_file, "r") as f:
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

    if "step_prepare_directories" in metadata:
        output_dir = Path(metadata["step_prepare_directories"]["bids"]).parent



    # FIND IMPORT DETAILS
    import_details = {}
    dropped_channels = []
    if "pre_pipeline_processing" in metadata:
        try:
            dropped_channels = metadata["pre_pipeline_processing"]["OuterLayerChannels"]
            if dropped_channels is None:
                dropped_channels = []
            import_details["dropped_channels"] = dropped_channels
        except:
            pass


    if "import_eeg" in metadata:
        import_details["sample_rate"] = metadata["import_eeg"]["sampleRate"]
        import_details["net_nbchan_orig"] = metadata["import_eeg"]["channelCount"]
        import_details["duration"] = metadata["import_eeg"]["durationSec"]
        import_details["basename"] = metadata["import_eeg"]["unprocessedFile"]
        original_channel_count = int(metadata["import_eeg"]["channelCount"]) - int(len(dropped_channels))
    else:
        message("error", "No import details found")
        return {}
    


    processing_details = {}
    if "step_run_pylossless" in metadata:
        pylossless_info = metadata["step_run_pylossless"]["pylossless_config"]
    elif "step_custom_pylossless_pipeline" in metadata:
        pylossless_info = metadata["step_get_pylossless_pipeline"]["pylossless_config"]
    else:
        message("warning", "No pylossless info found. Processing details may be missing")
        pylossless_info = None
    
    if pylossless_info is not None:
        processing_details["h_freq"] = pylossless_info["filtering"]["filter_args"][
            "h_freq"
        ]
        processing_details["l_freq"] = pylossless_info["filtering"]["filter_args"][
            "l_freq"
        ]
        processing_details["notch_freqs"] = pylossless_info["filtering"][
            "notch_filter_args"
        ]["freqs"]
        if "notch_widths" in pylossless_info["filtering"]["notch_filter_args"]:
            processing_details["notch_widths"] = pylossless_info["filtering"][
                "notch_filter_args"
            ]["notch_widths"]
        else:
            processing_details["notch_widths"] = "notch_freqs/200"
    

    # FIND EXPORT DETAILS
    export_details = {}
    if "save_epochs_to_set" in metadata:
        save_epochs_to_set = metadata["save_epochs_to_set"]
        epoch_length = save_epochs_to_set["tmax"] - save_epochs_to_set["tmin"]
        export_details["epoch_length"] = epoch_length
        export_details["final_n_epochs"] = save_epochs_to_set["n_epochs"]
        export_details["final_duration"] = epoch_length * save_epochs_to_set["n_epochs"]
        if original_channel_count and unique_bad_channels:
            export_details["net_nbchan_post"] = original_channel_count - len(
                unique_bad_channels
            )
        else:
            export_details["net_nbchan_post"] = original_channel_count

    if "step_create_regular_epochs" in metadata:
        epoch_metadata = metadata["step_create_regular_epochs"]
    elif "step_create_eventid_epochs" in metadata:
        epoch_metadata = metadata["step_create_eventid_epochs"]
    else:
        message("warning", "No epoch creation details found. Processing details may be missing")
        epoch_metadata = None
    
    if epoch_metadata is not None:
        export_details["initial_n_epochs"] = epoch_metadata[
            "initial_epoch_count"
        ]
        export_details["initial_duration"] = epoch_metadata["initial_duration"]
        export_details["srate_post"] = (
            (epoch_metadata["single_epoch_samples"] -1)
            // epoch_metadata["single_epoch_duration"]
        )
        export_details["epoch_limits"] = [
            epoch_metadata["tmin"],
            epoch_metadata["tmax"],
        ]

    ica_details = {}
    if "step_run_ll_rejection_policy" in metadata:
        ll_rejection_policy = metadata["step_run_ll_rejection_policy"]
        ica_details["proc_removeComps"] = ll_rejection_policy["ica_components"]
        ica_details["proc_nComps"] = ll_rejection_policy["n_components"]

    if "step_detect_dense_oscillatory_artifacts" in metadata:
        ref_artifacts = metadata["step_detect_dense_oscillatory_artifacts"]["artifacts_detected"]
        processing_details["ref_artifacts"] = ref_artifacts

    summary_dict = {
        "run_id": run_id,
        "task": run_record["task"],
        "bids_subject": f"sub-{bids_path.subject}",
        "timestamp": run_record["timestamp"],
        "basename": import_details["basename"],
        "proc_state": proc_state,
        "exclude_category": exclude_category,
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
    
    # Add metadata to database
    manage_database(
        operation="update",
        update_record={
            "run_id": run_id, 
            "metadata": {"json_summary": summary_dict}
        },
    )
    
    return summary_dict

def generate_bad_channels_tsv(summary_dict: Dict[str, Any])->None:
    try: 
        channel_dict = summary_dict["channel_dict"]
    except:
        message("warning", "Could not generate bad channels tsv -> No channel dict found in summary dict")
        return
    
    try:
        noisy_channels = channel_dict["noisy_channels"]
        uncorrelated_channels = channel_dict["uncorrelated_channels"]
        deviation_channels = channel_dict["deviation_channels"]
        bridged_channels = channel_dict["bridged_channels"]
        rank_channels = channel_dict["rank_channels"]
        ransac_channels = channel_dict["ransac_channels"]
    except:
        message("warning", "Could not generate bad channels tsv -> Failed to fetch bad channels")
        return
    
    with open(f"{summary_dict['derivatives_dir']}/FlaggedChs.tsv", "w") as f:
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
