from email.mime import message
from typing import Union
import numpy as np
import mne

from autoclean.functions.analysis import compute_itc, plot_itc, export_itc_csv
from autoclean.utils.logging import message

class InterTrialCoherenceMixin:
    def run_sl_itc_analysis(self, 
            epochs: Union[mne.BaseEpochs, None] = None, 
            roi: list[str] | None = [f"E{idx + 1}" for idx in [27,19,11,4,117,116,28,12,5,111,110,35,29,6,105,104,103,30,79,54,36,86,40,102]], 
            fmin: float = 0.6, 
            fmax: float = 5.0, 
            df: float = 0.01, 
            tmin: float | None = None, 
            tmax: float | None = None, 
            syllable_interval: float = 0.3,
            word_length_syllables: int = 3,
            ):
        """
        Run inter-trial coherence analysis for statistical learning epochs.
        """

        # Determine which data to use
        epochs = self._get_data_object(epochs, use_epochs=True)

        # Type checking
        if not isinstance(
            epochs, mne.Epochs
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Epochs object for GFP cleaning")
        
        # Target frequencies in statistical learning
        syllable_frequency = 1.0 / syllable_interval
        word_frequency = syllable_frequency / float(word_length_syllables)
        
        derivatives_dir = self.config.get("derivatives_dir", {})
        basename = self.config["unprocessed_file"].stem
        csv_path = f"{derivatives_dir}/{basename}_itc.csv"

        try:
            message("header", "Running inter-trial coherence analysis")
            freqs, plv_roi, power_roi, info = compute_itc(epochs, roi, fmin, fmax, df, tmin, tmax, [word_frequency, syllable_frequency])
            
            message("info", f"Exporting inter-trial coherence analysis to {csv_path}")
            export_itc_csv(csv_path, basename, freqs, plv_roi, power_roi)
            
            message("info", f"Plotting inter-trial coherence analysis to {derivatives_dir}/{basename}_itc_plot.png")
            plot_itc(freqs, plv_roi, target_frequencies=info["target_frequencies"], output_path=f"{derivatives_dir}/{basename}_itc_plot.png")

            metadata = {
                "num_epochs": len(epochs),
                "roi": roi,
                "fmin": fmin,
                "fmax": fmax,
                "df": df,
                "word_frequency": word_frequency,
                "syllable_frequency": syllable_frequency,
                "itc_word_frequency": float(plv_roi[np.argmin(np.abs(freqs - word_frequency))]),
                "itc_syllable_frequency": float(plv_roi[np.argmin(np.abs(freqs - syllable_frequency))]),
            }

            self._update_metadata("step_itc_analysis", metadata)


            message("success", "Inter-trial coherence analysis completed successfully")

            return freqs, plv_roi, power_roi, info
        
        except Exception as e:
            message("error", f"Error running inter-trial coherence analysis: {e}")
            return None
        
