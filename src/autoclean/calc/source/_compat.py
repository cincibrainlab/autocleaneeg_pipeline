"""Compatibility helpers and optional dependency flags for source analysis."""
from __future__ import annotations

# Optional imports with availability flags
try:  # pragma: no cover - optional dependency
    import networkx as nx  # type: ignore
    from networkx.algorithms.community import (  # type: ignore
        louvain_communities,
        modularity,
    )

    NETWORK_ANALYSIS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    nx = None  # type: ignore
    louvain_communities = None  # type: ignore
    modularity = None  # type: ignore
    NETWORK_ANALYSIS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from bctpy import charpath, clustering_coef_wu, efficiency_wei  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    charpath = None  # type: ignore
    clustering_coef_wu = None  # type: ignore
    efficiency_wei = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from fooof import FOOOF, FOOOFGroup  # type: ignore
    from fooof.analysis import get_band_peak_fm  # type: ignore

    FOOOF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FOOOF = None  # type: ignore
    FOOOFGroup = None  # type: ignore
    get_band_peak_fm = None  # type: ignore
    FOOOF_AVAILABLE = False

__all__ = [
    "NETWORK_ANALYSIS_AVAILABLE",
    "FOOOF_AVAILABLE",
    "nx",
    "louvain_communities",
    "modularity",
    "charpath",
    "clustering_coef_wu",
    "efficiency_wei",
    "FOOOF",
    "FOOOFGroup",
    "get_band_peak_fm",
]
