.. _api_montage_constants:

=================
Montage Constants
=================

This section documents the standard montage mappings and constants used in AutoClean for EEG channel locations and naming.

.. currentmodule:: autoclean.utils.montage

Constants
---------

.. data:: VALID_MONTAGES
   :type: list
   
   List of valid montages loaded from montages.yaml in the configs directory. These montages are verified to work with AutoClean's processing pipeline.

.. data:: GSN_TO_1020_MAPPING
   :type: dict

   Standard 10-20 to GSN-HydroCel mapping based on official EGI GSN-HydroCel channel maps.
   This dictionary maps standard 10-20 system channel names to their corresponding GSN-HydroCel electrode numbers.

   The mapping includes:

   * Frontal midline electrodes (Fz, FCz, Cz)
   * Left/Right frontal electrodes (F3/F4, F7/F8, FC3/FC4)
   * Left/Right central/temporal electrodes (C3/C4, T7/T8, CP3/CP4)
   * Parietal midline electrodes (Pz, POz)
   * Left/Right parietal/occipital electrodes (P3/P4, P7/P8, O1/O2)

.. data:: _1020_TO_GSN_MAPPING
   :type: dict

   Reverse mapping from GSN-HydroCel to 10-20 system. This is automatically generated from GSN_TO_1020_MAPPING
   and provides the inverse lookup (GSN electrode number to 10-20 name). 