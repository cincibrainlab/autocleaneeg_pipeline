"""Report generation mixin for autoclean tasks.

This module provides specialized report generation functionality for the AutoClean
pipeline. It defines methods for creating comprehensive reports that document the
processing pipeline and results, including:

- Run summary reports in PDF format
- Processing log updates for task-specific tracking
- JSON summary generation for machine-readable outputs
- Metadata capture and organization

These reports help users document the preprocessing steps applied to their data and
provide transparent traceability of the processing pipeline.

Example:
    ```python
    from autoclean.core.task import Task
    
    class MyEEGTask(Task):
        def process(self, raw, pipeline, autoclean_dict):
            # Process the data
            raw_cleaned = self.apply_preprocessing(raw)
            
            # Generate reports
            self.generate_report(raw, raw_cleaned, pipeline, autoclean_dict)
            self.update_processing_log(pipeline, autoclean_dict, status="complete")
            self.generate_json_summary(pipeline, autoclean_dict)
    ```
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as ReportLabTable,
    TableStyle,
)

from autoclean.utils.logging import message
from autoclean.utils.database import get_run_record, manage_database
from autoclean.mixins.reporting.base import ReportingMixin


class ReportGenerationMixin(object):
    """Mixin providing report generation functionality for EEG data.
    
    This mixin extends the base ReportingMixin with specialized methods for
    generating comprehensive reports documenting the EEG processing pipeline
    and results. It provides tools for creating PDF reports, updating processing
    logs, and generating structured summaries of processing outcomes.
    
    All reporting methods respect configuration toggles from `autoclean_config.yaml`,
    checking if their corresponding step is enabled before execution. Each method
    can be individually enabled or disabled via configuration.
    
    Available report generation methods include:
    
    - `generate_report`: Create a comprehensive PDF report of the processing pipeline
    - `update_processing_log`: Update a log file with processing status and metadata
    - `generate_json_summary`: Create a structured JSON summary of processing results
    
    Configuration options for these methods include:
    
    ```yaml
    report_generation_step:
      enabled: true
      value: {}
    log_update_step:
      enabled: true
      value: {}
    json_summary_step:
      enabled: true
      value: {}
    ```
    """
    
    def create_run_report(self, run_id: str, autoclean_dict: dict = None) -> None:
        """Create a scientific report in PDF format using ReportLab based on the run metadata.
        
        This method generates a comprehensive PDF report that documents all aspects
        of the EEG processing pipeline for a specific run, including:
        - Run metadata (subject, task, etc.)
        - Processing steps applied
        - Data quality metrics before and after processing
        - Artifact detection and removal statistics
        - References to visualization files
        
        Parameters:
        -----------
        run_id : str
            The run ID to generate a report for
        autoclean_dict : dict, optional
            The autoclean dictionary with configuration information
        """
        # Check if configuration checking is enabled and the step is enabled
        if hasattr(self, '_check_step_enabled'):
            is_enabled, _ = self._check_step_enabled("report_generation_step")
            if not is_enabled:
                message("info", "✗ Report generation is disabled in configuration")
                return
        
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

        # Prepare PDF output directory and filename
        try:
            if "step_prepare_directories" in run_record["metadata"]:
                out_dir = Path(run_record["metadata"]["step_prepare_directories"]["bids"]).parent
                pdf_dir = out_dir / "reports"
                pdf_dir.mkdir(exist_ok=True, parents=True)
                pdf_path = pdf_dir / f"{run_id}_processing_report.pdf"

                # If autoclean_dict is provided, add more metadata to the PDF name
                if autoclean_dict and "bids_path" in autoclean_dict:
                    bp = autoclean_dict["bids_path"]
                    subject = bp.subject if hasattr(bp, "subject") else "unknown"
                    task = bp.task if hasattr(bp, "task") else "unknown"
                    pdf_path = pdf_dir / f"{subject}_{task}_{run_id}_report.pdf"

        except Exception as e:
            message("error", f"Error setting up report output path: {str(e)}")
            return

        message("info", f"Generating PDF report: {pdf_path}")

        # Initialize PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        # Define styles for the document
        styles = getSampleStyleSheet()

        # Custom styles for better presentation
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=16,
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

        # Add processing steps and timeline
        if "metadata" in run_record:
            # Extract processing steps from metadata
            metadata = run_record["metadata"]
            step_keys = [k for k in metadata.keys() if k.startswith("step_")]

            if step_keys:
                story.append(Paragraph("Processing Timeline", heading_style))
                story.append(Spacer(1, 4))

                # Create processing steps table
                step_data = [("Step", "Status", "Details")]
                for step in sorted(step_keys):
                    step_name = step.replace("step_", "").replace("_", " ").title()
                    step_status = "Completed" if metadata[step] else "Failed"
                    step_details = ""

                    # Extract key details (adapt this based on your step metadata structure)
                    if isinstance(metadata[step], dict):
                        # Just display a few key details to keep it simple
                        details = []
                        for k, v in metadata[step].items():
                            if k in ["duration", "timestamp", "message"] and v:
                                if k == "duration" and isinstance(v, (int, float)):
                                    details.append(f"{k}: {v:.2f}s")
                                elif k == "timestamp" and isinstance(v, str):
                                    # Format timestamp
                                    try:
                                        dt = datetime.fromisoformat(v)
                                        details.append(f"{dt.strftime('%H:%M:%S')}")
                                    except:
                                        details.append(f"{v}")
                                else:
                                    details.append(f"{k}: {str(v)[:30]}")
                        step_details = ", ".join(details)

                    step_data.append((step_name, step_status, step_details))

                # Add steps table
                steps_table = ReportLabTable(step_data, colWidths=[1.5*inch, 1.0*inch, 3.5*inch])
                steps_table.setStyle(table_style)
                story.append(steps_table)
                story.append(Spacer(1, 12))

        # Add error message if processing failed
        if not run_record.get("success", False) and "error" in run_record:
            error_style = ParagraphStyle(
                "Error",
                parent=normal_style,
                textColor=colors.HexColor("#E74C3C"),
                backColor=colors.HexColor("#FADBD8"),
                borderWidth=1,
                borderColor=colors.HexColor("#E74C3C"),
                borderPadding=(4, 4, 4, 4),
                spaceBefore=8,
                spaceAfter=8,
            )
            error_text = f"<b>Processing Error:</b> {run_record['error']}"
            story.append(Paragraph(error_text, error_style))
            story.append(Spacer(1, 8))

        # Add processing outputs and files
        if "step_prepare_directories" in run_record["metadata"]:
            try:
                # Find derivatives directory
                out_dir = Path(run_record["metadata"]["step_prepare_directories"]["bids"]).parent
                derivatives_dir = out_dir / "derivatives"
                
                if derivatives_dir.exists() and derivatives_dir.is_dir():
                    story.append(Paragraph("Processing Outputs", heading_style))
                    story.append(Spacer(1, 4))
                    
                    # List files in derivatives directory
                    files_data = [("Type", "Filename", "Size")]
                    
                    for file in derivatives_dir.glob("**/*"):
                        if not file.is_file():
                            continue
                            
                        # Categorize file by extension
                        file_type = file.suffix.lower().replace(".", "").upper()
                        
                        # Get file size
                        size_bytes = file.stat().st_size
                        size_str = (
                            f"{size_bytes} B"
                            if size_bytes < 1024
                            else (
                                f"{size_bytes/1024:.1f} KB"
                                if size_bytes < 1024 * 1024
                                else f"{size_bytes/(1024*1024):.1f} MB"
                            )
                        )
                        
                        files_data.append([file_type, file.name, size_str])
                    
                    # Create table if files were found
                    if len(files_data) > 1:
                        story.append(
                            Paragraph(f"Directory: {derivatives_dir}", normal_style)
                        )
                        story.append(Spacer(1, 4))
                        
                        files_table = ReportLabTable(
                            files_data, colWidths=[1.0 * inch, 4.0 * inch, 1.5 * inch]
                        )
                        files_table.setStyle(
                            TableStyle(
                                [
                                    (
                                        "GRID",
                                        (0, 0),
                                        (-1, -1),
                                        0.5,
                                        colors.HexColor("#BDC3C7"),
                                    ),
                                    (
                                        "BACKGROUND",
                                        (0, 0),
                                        (-1, 0),
                                        colors.HexColor("#F5F6FA"),
                                    ),
                                    (
                                        "BACKGROUND",
                                        (0, 1),
                                        (-1, -1),
                                        colors.HexColor("#F8F9F9"),
                                    ),
                                    (
                                        "TEXTCOLOR",
                                        (0, 0),
                                        (-1, 0),
                                        colors.HexColor("#2C3E50"),
                                    ),
                                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ]
                            )
                        )
                        story.append(files_table)
                        story.append(Spacer(1, 12))
                    
            except Exception as e:
                message("warning", f"Error listing derivatives: {str(e)}")

        # Build the PDF document
        try:
            doc.build(story)
            message("success", f"PDF report generated: {pdf_path}")
            
            # Update database with report information
            manage_database(
                operation="update",
                update_record={
                    "run_id": run_id, 
                    "metadata": {
                        "create_run_report": {
                            "timestamp": datetime.now().isoformat(),
                            "pdf_path": str(pdf_path),
                        }
                    },
                },
            )
            
        except Exception as e:
            message("error", f"Error building PDF report: {str(e)}")
        
    def update_task_processing_log(self, summary_dict: Dict[str, Any]) -> None:
        """Update the task-specific processing log CSV file with details about the current file.
        
        This method maintains a CSV file that tracks processing outcomes across all files
        processed with the same task, enabling batch analysis of processing metrics.
        
        Parameters:
        -----------
        summary_dict : dict
            The summary dictionary containing processing details
        """
        # Check if configuration checking is enabled and the step is enabled
        if hasattr(self, '_check_step_enabled'):
            is_enabled, _ = self._check_step_enabled("log_update_step")
            if not is_enabled:
                message("info", "✗ Processing log update is disabled in configuration")
                return
        
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
            
            # Extract details from summary_dict with safe access
            details = {
                "timestamp": summary_dict.get("timestamp", ""),
                "study_user": os.getenv("USERNAME", "unknown"),
                "run_id": summary_dict.get("run_id", ""),
                "proc_state": summary_dict.get("proc_state", ""),
                "subj_basename": Path(summary_dict.get("basename", "")).stem,
                "bids_subject": summary_dict.get("bids_subject", ""),
                "task": summary_dict.get("task", ""),
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
            
            # Calculate percentages safely
            raw_duration = safe_get(summary_dict, "import_details", "duration", default="0")
            final_duration = safe_get(summary_dict, "export_details", "final_duration", default="0")
            initial_epochs = safe_get(summary_dict, "export_details", "initial_n_epochs", default="0")
            final_epochs = safe_get(summary_dict, "export_details", "final_n_epochs", default="0")
            
            # Add calculated fields
            details.update({
                "proc_xmax_percent": safe_percentage(final_duration, raw_duration),
                "epoch_length": str(safe_get(summary_dict, "export_details", "epoch_length", default="")),
                "epoch_limits": str(safe_get(summary_dict, "export_details", "epoch_limits", default="")),
                "epoch_trials": str(initial_epochs),
                "epoch_badtrials": safe_percentage(
                    float(initial_epochs) - float(final_epochs) if initial_epochs and final_epochs else "0", 
                    "1"
                ),
                "epoch_percent": safe_percentage(final_epochs, initial_epochs),
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
                
        except Exception as e:
            message("error", f"Error updating task processing log: {str(e)}")
            
    def create_json_summary(self, run_id: str) -> Dict[str, Any]:
        """Create a comprehensive JSON summary of the processing run.
        
        This method generates a structured dictionary that summarizes all aspects of the
        processing run, suitable for machine-readable outputs and database storage.
        
        Parameters:
        -----------
        run_id : str
            The run ID to generate the summary for
            
        Returns:
        --------
        dict
            A comprehensive summary dictionary of the processing run
        """
        # Check if configuration checking is enabled and the step is enabled
        if hasattr(self, '_check_step_enabled'):
            is_enabled, _ = self._check_step_enabled("json_summary_step")
            if not is_enabled:
                message("info", "✗ JSON summary generation is disabled in configuration")
                return {}
        
        run_record = get_run_record(run_id)
        if not run_record:
            message("error", f"No run record found for run ID: {run_id}")
            return {}

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
        except Exception as e:
            message("error", f"Could not get derivatives path: {str(e)}")
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
        
        if "step_custom_pylossless_pipeline" in metadata:
            channel_dict["step_custom_pylossless_pipeline"] = metadata["step_custom_pylossless_pipeline"][
                "bads"
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
        channel_dict["removed_channels"] = bad_channels

        if "step_prepare_directories" in metadata:
            output_dir = Path(metadata["step_prepare_directories"]["bids"]).parent

        # FIND IMPORT DETAILS
        import_details = {}
        if "import_eeg" in metadata:
            import_details["sample_rate"] = metadata["import_eeg"]["sampleRate"]
            import_details["net_nbchan_orig"] = metadata["import_eeg"]["channelCount"]
            import_details["duration"] = metadata["import_eeg"]["durationSec"]
            import_details["basename"] = metadata["import_eeg"]["unprocessedFile"]
            original_channel_count = int(metadata["import_eeg"]["channelCount"])
        else:
            message("error", "No import details found")
            return {}

        processing_details = {}
        if "step_run_pylossless" in metadata:
            pylossless_info = metadata["step_run_pylossless"]["pylossless_config"]
        elif "step_custom_pylossless_pipeline" in metadata:
            pylossless_info = metadata["step_custom_pylossless_pipeline"]["pylossless_config"]
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
            if original_channel_count and bad_channels:
                export_details["net_nbchan_post"] = original_channel_count - len(
                    bad_channels
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
        self._update_metadata("json_summary", summary_dict)
        manage_database(
            operation="update",
            update_record={
                "run_id": run_id, 
                "metadata": {"json_summary": {"timestamp": datetime.now().isoformat()}}
            },
        )
        
        return summary_dict
