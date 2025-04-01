import mne
from mne.preprocessing import ICA


raw = mne.io.read_raw_eeglab('C:/Users/Gam9LG/Documents/Autoclean/RestingEyesOpen/bids/derivatives/pylossless/sub-141278/eeg/0101_rest_pre_ica.set', preload=True)
ica = mne.preprocessing.read_ica('C:/Users/Gam9LG/Documents/Autoclean/RestingEyesOpen/bids/derivatives/pylossless/sub-141278/eeg/0101_restica2-ica.fif')

ica.plot_components(picks=range(ica.n_components_))  # User inspects and closes plot
exclusions = input("Enter components to exclude (e.g., '0 1 3'): ")
ica.exclude = [int(i) for i in exclusions.split()]