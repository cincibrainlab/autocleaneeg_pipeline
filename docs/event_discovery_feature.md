# Event Discovery and Configuration Helper

## Overview

This document describes the event discovery and configuration coaching system implemented in the `EventIDEpochsMixin` to help users correctly configure event-based epoching for EEG data analysis.

## Problem Solved

When working with EEG data, users often encounter these issues:
1. **Unknown event codes**: Don't know what event markers exist in their data files
2. **Empty epochs**: Configured `event_id` doesn't match actual data, resulting in no epochs
3. **Configuration errors**: Mismatch between config codes and data codes (e.g., using `"DIN4"` instead of `4`)
4. **Poor documentation**: Hard to understand the mapping between event descriptions and numeric codes

## Solution: `print_discovered_events()` Method

A standalone helper method that:
- Automatically extracts all events from raw EEG data annotations
- Displays them in an easy-to-understand formatted table
- Provides ready-to-use JSON configuration snippets
- Offers coaching tips for proper event configuration

### Features

#### 1. **Automatic Event Extraction**
```python
processor.print_discovered_events()
```
Extracts and displays all events found in the data with:
- Event names (descriptions)
- Numeric event codes
- Occurrence counts
- Percentage of total events

#### 2. **Smart Filtering**
- Automatically filters out artifact markers (BAD_* events)
- Excludes rare events (< 5 occurrences by default)
- Sorts by frequency (most common first)

#### 3. **Configuration Examples**
Generates two types of example configurations:
- **Basic**: Uses all discovered event codes as-is
- **Custom**: Demonstrates mapping to meaningful names (recommended)

Example output:
```json
{
  "epoch_settings": {
    "enabled": true,
    "value": {"tmin": -0.5, "tmax": 2.0},
    "event_id": {
      "target": 4,
      "standard": 5
    }
  }
}
```

#### 4. **Automatic Coaching**
Automatically called in these scenarios:

**Always (on every epoch creation):**
- Displays full event summary table with occurrence counts and percentages
- Shows complete JSON configuration examples for copy-paste
- Provides pro tips on proper event configuration
- Helps users verify their event_id mapping is working correctly
- Gives transparency into what events exist in the data
- Allows users to see what additional events they could use

**Additionally on errors:**
- User provides no `event_id` configuration
- Configured `event_id` doesn't match any events in data
- Empty epochs result from event matching

This provides **continuous coaching** - users always see how to configure events properly, not just when errors occur.

## Usage Examples

### Basic Discovery
```python
# See what events are in your data
processor.print_discovered_events()
```

### Programmatic Use
```python
# Get events without printing examples
available = processor.print_discovered_events(show_config_example=False)

# Select specific events
event_id = {
    "target": available["DIN4"],
    "standard": available["DIN5"]
}

# Create epochs
epochs = processor.create_eventid_epochs(event_id=event_id)
```

### Troubleshooting
```python
# If this returns None, print_discovered_events() is automatically called
epochs = processor.create_eventid_epochs()
# Output will show:
# - What codes you tried to use
# - What codes are actually available
# - Example configurations that will work
```

## Output Format

### Sample Output
```
================================================================================
Discovering Events in EEG Data
================================================================================
Extracting events from annotations...

Found 4 unique event types:
================================================================================
Event Name                Code       Count        % of Total     
--------------------------------------------------------------------------------
DIN4                      4          150          45.5%
DIN5                      5          120          36.4%
DIN8                      8          50           15.2%
BAD_EOG                   999        10           3.0%
================================================================================
Total events: 330

================================================================================
Configuration Guide
================================================================================

To use these events for epoching, add to your config.json:

Basic Example (uses all available events):
{
  "epoch_settings": {
    "enabled": true,
    "value": {
      "tmin": -0.2,
      "tmax": 0.8
    },
    "event_id": {
      "DIN4": 4,
      "DIN5": 5,
      "DIN8": 8
    }
  }
}

Custom Naming Example (recommended for clarity):
{
  "epoch_settings": {
    "enabled": true,
    "value": {
      "tmin": -0.5,
      "tmax": 2.0
    },
    "event_id": {
      "target": 4,
      "standard": 5
    }
  }
}

Note: Replace 'target' and 'standard' with meaningful names for your experiment

================================================================================
Pro Tips:
  • Event codes must match EXACTLY (integers)
  • Use descriptive names instead of 'DIN4' for better readability
  • Adjust tmin/tmax based on your experimental design
  • If epochs are empty, verify codes match what's shown above
================================================================================
```

## Implementation Details

### Method Signature
```python
def print_discovered_events(
    self,
    data: Union[mne.io.Raw, None] = None,
    show_config_example: bool = True,
) -> Optional[Dict[str, int]]
```

### Parameters
- `data`: Raw EEG data (uses `self.raw` if None)
- `show_config_example`: Whether to print configuration examples

### Returns
- Dictionary mapping event descriptions to codes
- `None` if no events found or error occurs

### Error Handling
- Catches annotation parsing errors
- Provides helpful error messages
- Returns `None` gracefully on failure

### Automatic Integration
The method is automatically called in these scenarios:

1. **Every epoch creation** (always-on coaching):
   ```python
   # Always show discovered events with full configuration coaching
   discovered_events = self.print_discovered_events(data=data, show_config_example=True)
   ```
   - Displays complete event table with counts and percentages
   - Shows JSON configuration examples
   - Provides pro tips and best practices
   - Helps users verify and improve their configuration
   - Shows what additional events are available

2. **No event_id configured** (full coaching mode):
   ```python
   if event_id is None or len(event_id) == 0:
       self.print_discovered_events(data=data)  # Full coaching with examples
   ```

3. **No matching events found** (full coaching mode):
   ```python
   if len(event_patterns) == 0:
       self.print_discovered_events(data=data)  # Full coaching with examples
   ```

## Design Principles

### 1. **Zero Dependencies**
- Uses only stdlib (`json`) and existing deps (MNE, numpy)
- No need for `tabulate`, `rich`, or `prompt_toolkit`
- Works in any environment

### 2. **Non-Intrusive**
- Standalone method - can be called independently
- Doesn't modify any state
- Returns data for programmatic use

### 3. **Educational**
- Provides context, not just data
- Includes examples and tips
- Explains common pitfalls

### 4. **Practical**
- Copy-paste ready JSON
- Handles edge cases (no events, only artifacts)
- Filters noise automatically

## Future Enhancements

The current implementation provides a solid foundation. Potential future additions:

### 1. **Interactive Mode** (as discussed in the design doc)
- CLI flag: `--interactive`
- Prompts for event selection
- Autocomplete event names
- Direct config file update

### 2. **Advanced Filtering**
- Custom thresholds for rare events
- Pattern-based grouping (e.g., "response_*")
- Temporal analysis (event timing patterns)

### 3. **Validation**
- Check epoch duration vs. event timing
- Warn about insufficient epoch counts
- Suggest alternative configurations

### 4. **Export Options**
- Save to file (JSON, YAML)
- Generate complete config templates
- Create backup before overwriting

### 5. **Enhanced Coaching**
- Experiment type detection (ERP, resting state, etc.)
- Common paradigm templates (oddball, N-back, etc.)
- Link to relevant tutorials

## Testing

### Unit Tests
```python
def test_print_discovered_events():
    """Test event discovery on synthetic data."""
    raw = create_synthetic_raw_with_events()
    processor = YourTaskClass(raw=raw)
    
    events = processor.print_discovered_events(show_config_example=False)
    
    assert events is not None
    assert len(events) > 0
    assert all(isinstance(code, int) for code in events.values())
```

### Integration Tests
- Test with real EEG data files
- Verify formatted output
- Test automatic calling on error conditions
- Validate JSON generation

## Related Files

- **Implementation**: `src/autoclean/mixins/signal_processing/eventid_epochs.py`
- **Examples**: `examples/event_discovery_example.py`
- **Tests**: `tests/unit/test_eventid_epochs.py` (to be added)

## Benefits

### For Users
- ✅ **Always-on verification**: See what events exist every time epochs are created
- ✅ **Confidence**: Confirm your configuration is working as expected
- ✅ **Transparency**: Understand what's happening with your data
- ✅ **Reduces trial-and-error**: Immediate feedback on configuration issues
- ✅ **Clear, actionable error messages**: Full coaching when problems occur
- ✅ **Learn correct patterns**: Examples provided when needed
- ✅ **Faster debugging**: Know immediately if event codes don't match

### For Developers
- ✅ **Reduces support burden**: Common config errors explained automatically
- ✅ **Self-documenting**: Code shows what it's doing through coaching messages
- ✅ **Easier debugging**: Users can share event discovery output
- ✅ **Extensible foundation**: Easy to add more coaching features
- ✅ **Follows MNE best practices**: Uses standard MNE event handling

## See Also

- MNE-Python Events Tutorial: https://mne.tools/stable/auto_tutorials/raw/20_event_arrays.html
- EEG Epoching Guide: https://mne.tools/stable/auto_tutorials/epochs/
- AutoCleanEEG Configuration: `docs/tutorials/first_time_processing.rst`

