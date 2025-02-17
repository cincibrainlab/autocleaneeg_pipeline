"""Processing step functions."""

import inspect
from typing import List

from . import continuous, epochs, io, reports

# Collect all step functions from each module
def _collect_step_functions(module) -> List[str]:
    return [
        name
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj) and name.startswith("step_")
    ]

# Create explicit imports for step functions
__all__ = (
    _collect_step_functions(continuous) +
    _collect_step_functions(epochs) +
    _collect_step_functions(io) +
    _collect_step_functions(reports)
)

# Import specific functions from each module
from .continuous import (  # noqa: E402
    step_pre_pipeline_processing,
    step_create_bids_path,
    step_clean_bad_channels,
    step_run_pylossless,
    step_run_ll_rejection_policy,
    step_detect_dense_oscillatory_artifacts,
    step_reject_bad_segments,
)
from .epochs import (  # noqa: E402
    step_create_epochs,
    step_autoreject_epochs,
    step_detect_bad_epochs,
)
from .io import (  # noqa: E402
    step_load_raw,
    step_save_raw,
    step_save_epochs,
    step_load_epochs,
    save_raw_to_set,
)
from .reports import (  # noqa: E402
    step_create_report,
    step_plot_raw,
    step_plot_epochs,
    plot_bad_channels_with_topography,
)
