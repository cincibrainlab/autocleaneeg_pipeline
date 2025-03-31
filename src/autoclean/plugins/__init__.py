# src/autoclean/plugins/__init__.py
"""AutoClean plugins package.

This package contains plugins for extending the AutoClean functionality.
The plugin architecture includes:

1. Format Plugins: For registering new EEG file formats
2. EEG Plugins: For handling specific combinations of file formats and montages
3. Event Processor Plugins: For processing task-specific event annotations

Plugins are automatically discovered and registered at runtime.
"""

# Initialize subpackages
from . import formats
from . import eeg_plugins
from . import event_processors

# Import all plugins to ensure they are registered
from autoclean.io.import_ import register_plugin, register_event_processor

# Import built-in plugins

# EEG plugins
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

# Event processor plugins
try:
    from .event_processors.p300 import P300EventProcessor
    from .event_processors.hbcd_mmn import HBCDMMNEventProcessor
    from .event_processors.resting_state import RestingStateEventProcessor
    
    # Register built-in event processors
    register_event_processor(P300EventProcessor)
    register_event_processor(HBCDMMNEventProcessor)
    register_event_processor(RestingStateEventProcessor)
    
except ImportError as e:
    # This will happen during initial package setup before event processors are created
    pass