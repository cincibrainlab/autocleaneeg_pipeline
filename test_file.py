# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne"
# ]
# ///

import mne
from pathlib import Path
import matplotlib.pyplot as plt

def plot_sensors(raw: mne.io.Raw) -> None:
    print(raw.ch_names[:len(raw.ch_names)//2])
    ch_groups = [
        [raw.ch_names.index(ch) for ch in ['E20', 'E21', 'E22', 'E23']],
        [raw.ch_names.index(ch) for ch in ['E10', 'E17']]
    ]
    colors = 'RdYlBu_r'  # Use colormap name as string with _r suffix to reverse
    
    fig = plt.figure(figsize=(10, 8))
    mne.viz.plot_sensors(
        raw.info,
        kind='topomap', 
        to_sphere=True,
        ch_type='all',  # Adjust if using a different channel type
        title='Sensor Topography: Good vs Bad Channels',
        show_names=True,
        ch_groups=ch_groups,
        pointsize=75,
        linewidth=0,
        cmap=colors,
        show=False,
        axes=fig.gca()
    )
    
    plt.savefig('sensor_topography.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
        # Load the SET file
    file_path = Path('/Users/ernie/Downloads/autoclean/rest_eyesopen/stage/_postrejection/0006_rest_postrejection_raw.set')
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    plot_sensors(raw)
    plt.show()


if __name__ == "__main__":
    main()
