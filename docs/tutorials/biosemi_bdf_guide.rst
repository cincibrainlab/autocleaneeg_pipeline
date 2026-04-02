BioSemi BDF File Support
========================

AutoClean Pipeline provides full support for BioSemi BDF (BioSemi Data Format) files through dedicated plugins for all common BioSemi montage configurations.

Supported BioSemi Montages
---------------------------

The following BioSemi montages are supported:

* **biosemi32** - 32-channel BioSemi system
* **biosemi64** - 64-channel BioSemi system
* **biosemi128** - 128-channel BioSemi system
* **biosemi256** - 256-channel BioSemi system

Additional montages (biosemi16, biosemi160) are configured but may require testing with actual data files.

Quick Start
-----------

Processing a single BDF file::

    autocleaneeg-pipeline process --file /path/to/data.bdf

Processing a directory of BDF files::

    autocleaneeg-pipeline process --dir /path/to/data/ --format "*.bdf"

Setting Your Montage
---------------------

Before processing, set your BioSemi montage using one of these methods:

**Interactive Selection:**
::

    autocleaneeg-pipeline montage set

**Direct Selection:**
::

    autocleaneeg-pipeline montage set biosemi64

**Via Setup Wizard:**
::

    autocleaneeg-pipeline wizard

The wizard will guide you through workspace setup, task selection, and montage configuration.

BioSemi-Specific Features
--------------------------

Status Channel
~~~~~~~~~~~~~~

BDF files contain a status channel that encodes trigger information in a 16-bit format. The AutoClean plugins automatically:

* Detect the status channel using ``stim_channel='auto'``
* Extract trigger codes from the lower 16 bits
* Preserve the status channel for event processing
* Log system status information (battery, CMS range) when available

CMS/DRL Referencing
~~~~~~~~~~~~~~~~~~~

BioSemi systems use **active electrodes** with CMS (Common Mode Sense) and DRL (Driven Right Leg) referencing during acquisition. This is different from passive electrode systems.

**Important:** The BDF plugins preserve the CMS/DRL referencing from acquisition. If you need to rereference your data:

1. Apply rereferencing in your task configuration
2. Common choices for BioSemi data:

   * Average reference (recommended for most analyses)
   * Specific electrode reference (e.g., mastoids)
   * REST (Reference Electrode Standardization Technique)

Example task configuration for average referencing::

    config = {
        "montage": {"enabled": True, "value": "biosemi64"},
        "rereferencing": {
            "enabled": True,
            "value": {"ref_type": "average"}
        },
        # ... other settings
    }

Event/Trigger Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~

BioSemi trigger codes are automatically extracted from the status channel during import. The plugins:

* Use MNE's automatic event detection from annotations
* Create detailed event DataFrames with timing and type information
* Log all detected event types and counts
* Support standard BioSemi trigger encoding (16-bit values)

File Format Details
-------------------

BDF (BioSemi Data Format) specifications:

* **Data Format:** 24-bit integers (converted to 32-bit by MNE)
* **Sample Rate:** Configurable (typically 256, 512, 1024, 2048 Hz)
* **Channel Layout:** A1-A32, B1-B32, etc. (channel count dependent)
* **Status Channel:** Always present, contains triggers and system codes
* **File Extension:** ``.bdf``

Integration with Pipeline
--------------------------

The BDF plugins integrate seamlessly with all pipeline features:

Format Detection
~~~~~~~~~~~~~~~~

File format is automatically detected from the ``.bdf`` extension. No manual format specification needed.

BIDS Compatibility
~~~~~~~~~~~~~~~~~~

BDF files can be processed into BIDS-compatible derivatives:

* Montage information stored in BIDS sidecar JSON
* Format metadata preserved in processing logs
* Standard BIDS derivative structure maintained

Quality Control
~~~~~~~~~~~~~~~

All standard QC metrics apply to BDF data:

* Bad channel detection
* Epoch rejection statistics
* Signal quality metrics
* ICA component classification

Metadata Tracking
~~~~~~~~~~~~~~~~~

The pipeline automatically stores:

* Plugin name and version
* File format (``BIOSEMI_BDF``)
* Montage name (e.g., ``biosemi64``)
* Channel count and sample rate
* Processing timestamps

Troubleshooting
---------------

Plugin Not Found
~~~~~~~~~~~~~~~~

If you see an error about no plugin found for BDF files:

1. Verify your montage is set correctly::

    autocleaneeg-pipeline montage list

2. Ensure the montage matches your BDF file's channel count:

   * 32 channels → use ``biosemi32``
   * 64 channels → use ``biosemi64``
   * etc.

3. Check that the BDF file is valid and not corrupted

Montage Application Warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see warnings about missing channels during montage application:

* This usually means channel names in your BDF file don't match standard BioSemi naming
* The ``on_missing='warn'`` parameter allows processing to continue
* Check your acquisition software's channel naming conventions
* Consider channel name mapping if needed

Status Channel Not Detected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If triggers/events are not detected:

* Verify your BDF file contains a status channel
* Check that triggers were actually sent during acquisition
* Examine the status channel manually if needed
* Contact your acquisition software support

Referencing Issues
~~~~~~~~~~~~~~~~~~

If you see unexpected referencing behavior:

* Remember: BioSemi data retains CMS/DRL referencing by default
* Always inspect your data before and after rereferencing
* For most analyses, average reference is recommended
* Some analyses may benefit from specific electrode references

Example Workflow
----------------

Complete workflow for processing BioSemi 64-channel data::

    # 1. Run setup wizard (first time only)
    autocleaneeg-pipeline wizard
    # Select workspace, task, biosemi64 montage

    # 2. Test with single file
    autocleaneeg-pipeline process --file /data/subject001.bdf

    # 3. Review outputs in workspace/subject001/

    # 4. If satisfied, process entire dataset
    autocleaneeg-pipeline process --dir /data/ --format "*.bdf"

    # 5. Results appear in workspace/ with BIDS structure

Advanced: Custom Task for BDF Data
-----------------------------------

Create a custom task with BioSemi-specific processing::

    from autoclean.core.task import Task

    class BioSemiRestingState(Task):
        """Custom task for BioSemi resting state data."""

        config = {
            "schema_version": "2025.09",
            "montage": {"enabled": True, "value": "biosemi64"},
            "rereferencing": {
                "enabled": True,
                "value": {"ref_type": "average"}
            },
            "filtering": {
                "enabled": True,
                "value": {"l_freq": 1, "h_freq": 50}
            },
            # ... additional settings
        }

        def run(self):
            # Import BDF with automatic plugin selection
            self.import_raw()

            # Apply average reference (recommended for BioSemi)
            self.apply_rereferencing()

            # Continue with standard processing
            self.apply_filtering()
            self.detect_bad_channels()
            # ... etc.

Technical Details
-----------------

Plugin Architecture
~~~~~~~~~~~~~~~~~~~

Each BioSemi montage has a dedicated plugin class:

* ``BDFBiosemi32Plugin`` - Handles 32-channel files
* ``BDFBiosemi64Plugin`` - Handles 64-channel files
* ``BDFBiosemi128Plugin`` - Handles 128-channel files
* ``BDFBiosemi256Plugin`` - Handles 256-channel files

All plugins inherit from ``BaseEEGPlugin`` and follow the standard plugin interface.

MNE Integration
~~~~~~~~~~~~~~~

BDF support uses MNE-Python's native BDF reader::

    mne.io.read_raw_bdf(
        input_fname=file_path,
        preload=preload,
        stim_channel='auto',
        exclude=[]
    )

Montage application uses MNE's standard montages::

    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, match_case=False, on_missing='warn')

Further Reading
---------------

* MNE-Python BDF documentation: https://mne.tools/stable/generated/mne.io.read_raw_bdf.html
* BioSemi system documentation: https://www.biosemi.com/
* AutoClean Pipeline general documentation: https://cincibrainlab.github.io/autoclean_pipeline/

Support
-------

If you encounter issues with BDF file processing:

1. Check this guide's troubleshooting section
2. Review the main troubleshooting documentation
3. Open an issue at: https://github.com/cincibrainlab/autoclean_pipeline/issues

Include in your issue:

* BDF file details (channel count, sample rate)
* Montage you're using
* Full error message
* Pipeline version (``autocleaneeg-pipeline --version``)
