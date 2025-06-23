# src/autoclean/plugins/__init__.py
"""AutoClean plugins package.

This package contains plugins for extending the AutoClean functionality.
The plugin architecture includes:

1. Format Plugins: For registering new EEG file formats
2. EEG Plugins: For handling specific combinations of file formats and montages
3. Event Processor Plugins: For processing task-specific event annotations

Plugins are automatically discovered and registered at runtime.
"""

# All plugins will be automatically discovered and registered during runtime discovery
# No module-level imports or registrations needed here - the discovery system handles it
