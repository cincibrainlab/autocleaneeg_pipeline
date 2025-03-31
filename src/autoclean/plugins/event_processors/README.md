# Event Processor Plugin System

This directory contains event processor plugins for the AutoClean pipeline. These plugins handle task-specific event processing for different experimental paradigms.

## Overview

The event processor plugin system provides a flexible way to handle event processing for different EEG experiment types (tasks) without modifying the core code. Each plugin is responsible for processing events for a specific task type.

## How It Works

1. The `BaseEventProcessor` abstract class defines the interface for all event processors
2. Specific processors (like `P300EventProcessor` or `HBCDMMNEventProcessor`) implement the interface
3. Processors register themselves with the system using the `register_event_processor` function
4. The system automatically discovers and loads processors at runtime
5. When processing data for a specific task, the appropriate processor is selected automatically

## Creating a New Event Processor

To create a new event processor:

1. Create a new Python file in this directory
2. Define a class that inherits from `BaseEventProcessor`
3. Implement the required methods:
   - `supports_task(task_name)` - Returns True if this processor handles the task
   - `process_events(raw, events, events_df, autoclean_dict)` - Processes events for the task

Example:

```python
from autoclean.io.import_ import BaseEventProcessor

class MyTaskEventProcessor(BaseEventProcessor):
    """Event processor for My custom task."""
    
    @classmethod
    def supports_task(cls, task_name: str) -> bool:
        return task_name == "my_custom_task"
    
    def process_events(self, raw, events, events_df, autoclean_dict):
        # Custom event processing logic
        return raw  # Return the modified raw object
```

The processor will be automatically discovered and registered when the AutoClean pipeline runs.

## Integration with Configuration

Event processors can respect configuration parameters from `autoclean_config.yaml`. For example:

```python
def process_events(self, raw, events, events_df, autoclean_dict):
    # Check if event processing is enabled
    if not autoclean_dict.get("event_processing_enabled", True):
        return raw  # Skip processing if disabled
        
    # Get custom parameters from config
    mapping = autoclean_dict.get("event_mapping", {})
    if not mapping:
        mapping = {"default": "default_value"}  # Fallback
        
    # Process with config parameters
    # ...
    
    return raw
```

This follows the same pattern used in the `SignalProcessingMixin` which respects configuration toggles.

## Built-in Processors

The system includes several built-in processors:

- `P300EventProcessor` - Handles P300 oddball paradigm data
- `HBCDMMNEventProcessor` - Handles HBCD MMN (Mismatch Negativity) data
- `RestingStateEventProcessor` - Handles resting state recordings
