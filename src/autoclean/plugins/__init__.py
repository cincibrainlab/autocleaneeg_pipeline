# src/autoclean/plugins/__init__.py
"""AutoClean plugins package.

This package contains all plugins for the AutoClean package.
"""

# Initialize subpackages
from . import formats
from . import eeg_plugins

# Import all plugins to ensure they are registered
from autoclean.step_functions.io import register_plugin

# Import built-in plugins
try:
    from .eeg_plugins.eeglab_gsn129_plugin import EEGLABSetGSN129Plugin
    from .eeg_plugins.eeglab_gsn124_plugin import EEGLABSetGSN124Plugin
    from .eeg_plugins.eeglab_standard1020_plugin import EEGLABSetStandard1020Plugin
    from .eeg_plugins.eeglab_mea30_plugin import EEGLABSetMEA30Plugin
    from .eeg_plugins.egi_raw_gsn129_plugin import EGIRawGSN129Plugin
    
    # Register built-in plugins
    register_plugin(EEGLABSetGSN129Plugin)
    register_plugin(EEGLABSetGSN124Plugin)
    register_plugin(EEGLABSetStandard1020Plugin)
    register_plugin(EEGLABSetMEA30Plugin)
    register_plugin(EGIRawGSN129Plugin)
    
except ImportError as e:
    # This will happen during initial package setup before plugins are created
    pass