"""Event processor for resting state EEG data."""

from typing import Optional

import mne
import numpy as np
import pandas as pd

from autoclean.io.import_ import BaseEventProcessor
from autoclean.utils.logging import message


class RestingStateEventProcessor(BaseEventProcessor):
    """Event processor for resting state tasks.

    This processor handles various resting state paradigms, including
    eyes open and eyes closed conditions.
    """

    VERSION = "1.0.0"

    @classmethod
    def supports_task(cls, task_name: str) -> bool:
        """Check if this processor supports the given task."""
        return "resting" in task_name.lower() or "rest" in task_name.lower()

    def process_events(
        self,
        raw: mne.io.Raw,
        events: Optional[np.ndarray],
        events_df: Optional[pd.DataFrame],
        autoclean_dict: dict,
    ) -> mne.io.Raw:
        """Process resting state task-specific annotations.

        This method respects configuration settings and can be disabled via
        the 'resting_state_event_processing' configuration parameter.

        Args:
            raw: Raw EEG data
            events: Event array from MNE
            events_df: DataFrame containing event information
            autoclean_dict: Configuration dictionary

        Returns:
            mne.io.Raw: Raw data with processed events/annotations
        """
        # Check if this specific processor is enabled
        if not self._check_config_enabled(
            "resting_state_event_processing", autoclean_dict
        ):
            message(
                "info", "✗ Resting state event processing disabled in configuration"
            )
            return raw

        message("info", "✓ Processing resting state task-specific annotations...")

        # For resting state, we may not have structured events/annotations
        # But we can add metadata about the resting state condition if available

        # Get condition from config with fallback to legacy parameter name
        condition = autoclean_dict.get(
            "resting_state_condition", autoclean_dict.get("condition", "")
        )

        # Get description format from config or use default
        description_format = autoclean_dict.get(
            "resting_state_description_format", "resting_state/{condition}"
        )

        if condition and (
            condition.lower() == "eyes_open" or condition.lower() == "eyes_closed"
        ):
            try:
                # Add a task annotation at the beginning of the recording
                new_annotation = mne.Annotations(
                    onset=[0],
                    duration=[raw.times[-1]],  # Duration is the entire recording
                    description=[
                        description_format.format(condition=condition.lower())
                    ],
                )

                # Append to existing annotations
                if len(raw.annotations) > 0:
                    raw.annotations = raw.annotations + new_annotation
                else:
                    raw.set_annotations(new_annotation)

                message(
                    "success",
                    f"Added resting state annotation with condition: {condition}",
                )
            except Exception as e:  # pylint: disable=broad-except
                message("error", f"Error creating resting state annotation: {e}")
        else:
            message("info", "No specific condition found for resting state task")

        # Create segment annotations if configured
        if self._check_config_enabled(
            "resting_state_create_segments", autoclean_dict, False
        ):
            # Calculate total duration
            duration = raw.times[-1]

            # Get segment duration from config or use default
            segment_duration = autoclean_dict.get(
                "resting_state_segment_duration", 5.0
            )  # seconds

            # Get segment description from config or use default
            segment_description = autoclean_dict.get(
                "resting_state_segment_description", "segment"
            )

            # Create segments at regular intervals
            segment_times = np.arange(0, duration, segment_duration)

            try:
                # Create new annotations
                segment_annotations = mne.Annotations(
                    onset=segment_times,
                    duration=np.zeros(len(segment_times)),
                    description=[segment_description] * len(segment_times),
                )

                # Add to raw data
                if len(raw.annotations) > 0:
                    raw.annotations = raw.annotations + segment_annotations
                else:
                    raw.set_annotations(segment_annotations)

                message(
                    "success",
                    f"Created {len(segment_times)} segment annotations for resting state data",
                )
            except Exception as e:  # pylint: disable=broad-except
                message("error", f"Error creating segment annotations: {e}")

        return raw
