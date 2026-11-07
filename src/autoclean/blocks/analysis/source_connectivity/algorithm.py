"""Source-level functional connectivity and graph theory algorithms.

This block module delegates to :mod:`autoclean.calc.source_connectivity`, the
single implementation shared with the sensor-level path. It keeps the public
names that ``mixin.py`` loads from this file.

References
----------
Vinck M, et al. (2011). The pairwise phase consistency: a bias-free measure of
rhythmic neuronal synchronization. NeuroImage, 51(1), 112-122.

Rubinov M & Sporns O (2010). Complex network measures of brain connectivity: Uses
and interpretations. NeuroImage, 52(3), 1059-1069.
"""

from autoclean.calc.source_connectivity import (  # noqa: F401
    BCTPY_AVAILABLE,
    NETWORK_ANALYSIS_AVAILABLE,
    calculate_source_connectivity,
)

__all__ = [
    "BCTPY_AVAILABLE",
    "NETWORK_ANALYSIS_AVAILABLE",
    "calculate_source_connectivity",
]
