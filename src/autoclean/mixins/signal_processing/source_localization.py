"""
Source-localisation mixin leveraging the external `autoclean-eeg2source` package.

Add `self.run_source_localization()` as the final step in any Task.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from autoclean.io.export import save_epochs_to_set, save_raw_to_set
from autoclean.utils.database import manage_database_conditionally
from autoclean.utils.logging import message

try:
    from autoclean_eeg2source import MemoryManager, SequentialProcessor
except ImportError as exc:  # pragma: no cover
    SequentialProcessor = None                # type: ignore
    MemoryManager = None                       # type: ignore
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


class SourceLocalizationMixin:
    """Mixin that attaches a template-based source localisation step."""

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def run_source_localization(          # type: ignore[override]
        self,
        use_epochs: bool = True,
        montage: Optional[str] = None,
        output_subdir: str = "source_localization",
        **processor_kwargs: Any,
    ) -> Path:
        """
        Run AutoClean-EEG2Source as the ultimate stage.

        Parameters
        ----------
        use_epochs
            If True (default) export `self.epochs`; otherwise fall back to `self.raw`.
        montage
            Electrode montage; falls back to `self.config["eeg_system"]`.
        output_subdir
            Folder inside `stage_dir` where results are written.
        **processor_kwargs
            Extra kwargs passed straight to ``SequentialProcessor``.
        """
        if _IMPORT_ERR:
            raise RuntimeError(
                "autoclean-eeg2source is not installed; install via "
                "`uv pip install autoclean-eeg2source`."
            ) from _IMPORT_ERR

        # 1.  Decide input & export as a *pre_source_localization* stage
        if use_epochs and getattr(self, "epochs", None) is not None:
            input_path = save_epochs_to_set(
                epochs=self.epochs,
                autoclean_dict=self.config,
                stage="pre_source_localization",
            )
        elif getattr(self, "raw", None) is not None:
            input_path = save_raw_to_set(
                raw=self.raw,
                autoclean_dict=self.config,
                stage="pre_source_localization",
            )
        else:
            raise RuntimeError("No epochs or raw data available.")

        # 2.  Prepare output directory
        out_dir = Path(self.config["stage_dir"]) / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3.  Call the processor
        montage = montage or self.config.get("eeg_system", "standard_1020")
        mem_mgr = MemoryManager(max_memory_gb=processor_kwargs.pop("max_memory_gb", 4))

        message("info", f"Source localisation → {out_dir} (montage={montage})")

        processor = SequentialProcessor(memory_manager=mem_mgr, montage=montage, **processor_kwargs)
        result = processor.process_file(str(input_path), str(out_dir))

        # 4.  Persist metadata
        manage_database_conditionally(
            operation="update",
            update_record={
                "run_id": self.config["run_id"],
                "metadata": {
                    "source_localization": {
                        "creationDateTime": datetime.now().isoformat(),
                        "input_file": str(input_path),
                        "output_dir": str(out_dir),
                        "result": result,
                        "montage": montage,
                    }
                },
            },
        )

        message("success", f"✓ Source localisation complete – results in {out_dir}")
        return out_dir 