"""Example: Using Event Discovery Helper for EEG Epoching Configuration.

This example demonstrates how to use the print_discovered_events() method
to understand what events are available in your EEG data and configure
your epoching parameters correctly.

The event discovery feature helps solve the common problem where users
don't know what event codes are in their data, leading to empty epochs
or configuration errors.
"""

import mne
from autoclean.tasks import YourTaskClass  # Replace with actual task class


def example_1_basic_discovery():
    """Example 1: Basic event discovery on a dataset."""
    # Load your EEG data
    raw = mne.io.read_raw_fif("your_data.fif", preload=True)
    
    # Create your processor instance
    processor = YourTaskClass(raw=raw)
    
    # Discover what events are available
    # This will print a formatted table and provide config examples
    available_events = processor.print_discovered_events()
    
    # Output will show:
    # - Event names (e.g., 'DIN4', 'DIN5')
    # - Event codes (e.g., 4, 5)
    # - Counts of each event
    # - Example JSON configurations
    
    print("\nDiscovered events:", available_events)


def example_2_programmatic_config():
    """Example 2: Use discovery results to build config programmatically."""
    raw = mne.io.read_raw_fif("your_data.fif", preload=True)
    processor = YourTaskClass(raw=raw)
    
    # Get available events without printing config examples
    available_events = processor.print_discovered_events(show_config_example=False)
    
    if available_events:
        # Programmatically select specific events
        # For example, select all events except artifacts
        event_id = {
            name: code
            for name, code in available_events.items()
            if not name.upper().startswith("BAD")
        }
        
        print(f"Selected events for epoching: {event_id}")
        
        # Now create epochs with the discovered events
        epochs = processor.create_eventid_epochs(
            event_id=event_id,
            tmin=-0.2,
            tmax=0.8,
        )


def example_3_troubleshooting_empty_epochs():
    """Example 3: Troubleshooting when epoching returns no epochs.
    
    When create_eventid_epochs() returns None or empty results,
    the discovery helper is automatically called to show what's available.
    """
    raw = mne.io.read_raw_fif("your_data.fif", preload=True)
    processor = YourTaskClass(raw=raw, config="config.json")
    
    # If config has event_id set incorrectly, this will:
    # 1. Detect the mismatch
    # 2. Automatically call print_discovered_events()
    # 3. Show what events are actually in the data
    # 4. Provide corrected config examples
    epochs = processor.create_eventid_epochs()
    
    # If epochs is None, check the output - it will show:
    # - What event codes you tried to use
    # - What event codes are actually available
    # - Example configurations that will work


def example_4_interactive_workflow():
    """Example 4: Interactive workflow for first-time setup.
    
    When working with new data for the first time:
    """
    raw = mne.io.read_raw_fif("new_dataset.fif", preload=True)
    processor = YourTaskClass(raw=raw)
    
    # Step 1: Discover events
    print("Step 1: Discovering events in your data...")
    available = processor.print_discovered_events()
    
    # Step 2: User manually updates config.json based on the output
    # The printed examples can be copy-pasted directly
    
    # Step 3: Test the configuration
    print("\nStep 2: Testing configuration...")
    epochs = processor.create_eventid_epochs()
    
    if epochs is not None:
        print(f"Success! Created {len(epochs)} epochs")
        print(f"Event types: {epochs.event_id}")
    else:
        print("Configuration needs adjustment - check the output above")


# Typical use cases:

# Use Case 1: "I don't know what events are in my data"
# → Call print_discovered_events() first

# Use Case 2: "My epoching returns empty/None"
# → It will automatically call print_discovered_events() and show the mismatch

# Use Case 3: "I want to see events without config examples"
# → Use print_discovered_events(show_config_example=False)

# Use Case 4: "Building a batch processing script"
# → Use the programmatic approach (example_2)


if __name__ == "__main__":
    # Run the examples (uncomment as needed)
    # example_1_basic_discovery()
    # example_2_programmatic_config()
    # example_3_troubleshooting_empty_epochs()
    # example_4_interactive_workflow()
    pass

