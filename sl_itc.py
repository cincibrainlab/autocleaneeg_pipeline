import mne
import numpy as np
import matplotlib.pyplot as plt

epochs = mne.read_epochs_eeglab("C:/Users/Gam9LG/Documents/Autoclean-EEG/output/Statistical_Learning/bids/derivatives/autoclean-v2.1.0/intermediate/FLAGGED_08_drop_bad_epochs/1037_slstructured_drop_bad_epochs_epo.set")
montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
epochs.set_montage(montage, match_case=False)

syll_frequency = 1.0 / 0.3
word_frequency = syll_frequency / 3.0

data = epochs.get_data(picks="eeg")
sfreq = epochs.info["sfreq"]
n_epochs, n_ch, n_times = data.shape
t = np.arange(n_times) / sfreq

roi = [f"E{idx + 1}" for idx in [27,19,11,4,117,116,28,12,5,111,110,35,29,6,105,104,103,30,79,54,36,86,40,102]]
roi_idx = mne.pick_channels(epochs.ch_names, roi, ordered=False)
if len(roi_idx) == 0:
    roi_idx = np.arange(n_ch)

freqs_fft = np.unique(np.sort(np.r_[np.arange(0.6, 5.0, 0.01), word_frequency, syll_frequency]))

tmin, tmax = 0, 9.0
tmask = (t >= tmin) & (t <= tmax)
data_win = data[..., tmask]
t_win = t[tmask]

exps = np.exp(-2j * np.pi * np.outer(freqs_fft, t_win))
Z = np.tensordot(data_win, exps, axes=([-1], [-1]))

phases = Z / np.maximum(np.abs(Z), 1e-12)
plv = np.abs(phases.mean(axis=0))
plv_roi = plv[roi_idx].mean(axis=0)

plt.figure(figsize=(8, 4))
plt.plot(freqs_fft, plv_roi, "k-", lw=2)
plt.axvline(word_frequency, color="b", ls="--", label="Word ~1.11 Hz")
plt.axvline(syll_frequency, color="r", ls="--", label="Syllable ~3.33 Hz")
plt.xlim(0.6, 5.0)
yl = plv_roi.min(), plv_roi.max()
pad = max(1e-3, 0.1 * (yl[1] - yl[0]))
plt.ylim(yl[0] - pad, yl[1] + pad)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PLV")
plt.title("FFT-ITC Spectrum (ROI)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()


plt.show()