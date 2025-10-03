# GUI Enhancements: Channel Removal Tracking

**Date**: 2025-10-02
**Feature**: Enhanced display of channel removals with color-coded reasons
**File**: `src/autoclean/tools/autoclean_exclude.py`

---

## Overview

The `autoclean_exclude` GUI has been enhanced to leverage the new unified `channel_removals` metadata, providing reviewers with detailed context about **why** each channel was removed during preprocessing.

---

## Visual Changes

### Before (Legacy Display)
```
Related Files & Metadata
─────────────────
Bad Channels: E1, E8, E14, E17, E21, E25, E32, E125, E126, E127, E128, E34, E55, E58, E117
```

**Problem**: No context about removal reasons. All channels displayed as a flat list.

---

### After (Enhanced Display)
```
Related Files & Metadata
─────────────────
Bad Channels (15):
  [EOG] E1, E8, E14, E17, E21, E25, E32, E125, E126, E127, E128
  [Uncorrelated] E34, E55, E58, E117
```

**Improvement**:
- ✅ Total count displayed
- ✅ Grouped by removal reason
- ✅ Color-coded for visual distinction
- ✅ Clear labeling of removal type

---

## Color Scheme

| Removal Reason | Label | Color | Hex Code |
|----------------|-------|-------|----------|
| `EOG_DROPPED` | EOG | Purple | `#9b59b6` |
| `OUTER_LAYER` | Outer Layer | Blue | `#3498db` |
| `UNCORRELATED` | Uncorrelated | Orange | `#e67e22` |
| `DEVIATION` | Deviation | Red | `#e74c3c` |
| `RANSAC` | RANSAC | Dark Orange | `#d35400` |
| `BRIDGED` | Bridged | Dark Red | `#c0392b` |
| `RANK` | Rank | Purple | `#8e44ad` |
| `MANUAL_EXCLUDE` | Manual | Gray | `#7f8c8d` |
| `TEMPLATE_EXCLUDE` | Template | Light Gray | `#95a5a6` |
| `NOISY` | Noisy | Red | `#e74c3c` |

---

## Technical Implementation

### 1. Helper Functions Added

**`_group_channel_removals(channel_removals)`** (line 191)
- Groups channel removal entries by reason code
- Returns dictionary: `{"EOG_DROPPED": ["E1", "E8"], ...}`

**`_get_removal_reason_display(reason_code)`** (line 217)
- Maps reason codes to human-readable labels and colors
- Returns tuple: `("EOG", "#9b59b6")`

### 2. Metadata Extraction Enhanced

**`_extract_metadata_info()`** (line 3995)
```python
# Extract unified channel removals (preferred)
channel_removals = metadata_section.get("channel_removals", [])
if channel_removals:
    result["channel_removals"] = channel_removals
    # Build bad_channels list for backward compatibility
    result["bad_channels"] = [r["channel"] for r in channel_removals]
else:
    # Fallback to legacy extraction
    result["bad_channels"] = metadata_section.get("step_clean_bad_channels", {}).get("bads", [])
```

**`_prepare_editor_metadata()`** (line 2954)
```python
# Extract unified channel removals (preferred)
channel_removals = metadata_section.get("channel_removals", [])
if channel_removals:
    bad_channels = [r["channel"] for r in channel_removals]
else:
    # Fallback to legacy
    bad_channels = metadata_section.get("step_clean_bad_channels", {}).get("bads", [])
```

### 3. Display Logic Updated

**`_refresh_related_list()`** (line 4122)
```python
# Display bad channels with removal reasons (enhanced)
channel_removals = metadata.get("channel_removals", [])
if channel_removals:
    # Enhanced display: group by removal reason
    grouped = _group_channel_removals(channel_removals)
    total_count = len(metadata.get("bad_channels", []))

    # Header showing total
    header_item = QListWidgetItem(f"Bad Channels ({total_count}):")
    header_item.setForeground(QColor("#e67e22"))
    self.related_list.addItem(header_item)

    # Display each group with color-coded reason
    for reason, channels in grouped.items():
        label, color = _get_removal_reason_display(reason)
        channels_str = ", ".join(channels)
        reason_item = QListWidgetItem(f"  [{label}] {channels_str}")
        reason_item.setForeground(QColor(color))
        self.related_list.addItem(reason_item)
elif metadata["bad_channels"]:
    # Fallback: legacy flat display
    ...
```

---

## Backward Compatibility

### Three-Tier Fallback System

1. **Preferred**: Use `metadata["channel_removals"]` for enhanced display
2. **Fallback**: Use `metadata["bad_channels"]` for legacy flat display
3. **Empty**: Show "Bad Channels: None" in gray

### Legacy Run Support

Runs processed with older pipeline versions (before unified tracking) will:
- ✅ Display correctly using legacy flat format
- ✅ Show all bad channels without reasons
- ✅ Maintain full functionality in editor widget

### New Run Benefits

Runs processed with unified tracking will:
- ✅ Display grouped by removal reason
- ✅ Show color-coded categories
- ✅ Provide complete removal context
- ✅ Work seamlessly in editor widget

---

## User Benefits

### For Reviewers
1. **Quick Assessment**: Instantly see why channels were removed
2. **Pattern Recognition**: Easily spot systematic issues (e.g., many EOG channels)
3. **Quality Control**: Verify expected removal categories
4. **Visual Clarity**: Color-coding reduces cognitive load

### For Researchers
1. **Transparency**: Complete audit trail of preprocessing decisions
2. **Reproducibility**: Understand exact preprocessing steps
3. **Debugging**: Identify unexpected channel removals
4. **Reporting**: Export detailed removal information for methods sections

---

## Example Use Cases

### Use Case 1: EOG Channel Verification
**Scenario**: Verify that EOG channels were properly identified and removed

**Legacy Display**:
```
Bad Channels: E1, E8, E14, E125, E126, E127, E128, E34
```
❌ Cannot distinguish EOG from artifact channels

**Enhanced Display**:
```
Bad Channels (8):
  [EOG] E1, E8, E14, E125, E126, E127, E128
  [Uncorrelated] E34
```
✅ Clear separation: 7 EOG channels + 1 artifact

---

### Use Case 2: Excessive Artifact Detection
**Scenario**: Investigate run flagged for too many bad channels

**Legacy Display**:
```
Bad Channels: E1, E8, E10, E15, E20, E25, E30, E34, E40, E45, E50
```
❌ No context for why so many channels flagged

**Enhanced Display**:
```
Bad Channels (11):
  [EOG] E1, E8
  [Outer Layer] E10, E15, E20, E25, E30
  [Uncorrelated] E34, E40, E45, E50
```
✅ Clear: 2 EOG + 5 outer layer + 4 artifacts = expected pattern

---

### Use Case 3: Debugging Unexpected Removals
**Scenario**: User reports unexpected channel count

**Legacy Display**:
```
Bad Channels: E5, E10, E15, E20
```
❌ No indication of removal source

**Enhanced Display**:
```
Bad Channels (4):
  [Manual] E5, E10, E15, E20
```
✅ Immediately identifies manual exclusion in config

---

## Testing

### Manual Testing Checklist

- [ ] Load run with `channel_removals` metadata
  - [ ] Verify grouped display appears
  - [ ] Check color-coding matches reason codes
  - [ ] Confirm total count is correct

- [ ] Load legacy run without `channel_removals`
  - [ ] Verify fallback to flat display
  - [ ] Confirm all channels shown
  - [ ] Check no errors occur

- [ ] Load run with no bad channels
  - [ ] Verify "Bad Channels: None" appears
  - [ ] Check gray color used

- [ ] Test editor widget
  - [ ] Verify channels populate correctly
  - [ ] Check editing functionality works
  - [ ] Confirm export preserves metadata

### Integration Testing

```python
# Test with your Chirp run
run_id = "128_SteadyState_D3158"
# Expected: 15 channels grouped as 11 EOG + 4 Uncorrelated
```

---

## Future Enhancements (Optional)

### Phase 2: Interactive Features
- Click reason label to highlight channels in browser
- Toggle visibility of removal categories
- Filter editor list by removal reason
- Export removal report to CSV

### Phase 3: Advanced Analytics
- Histogram of removal reasons across dataset
- Trend analysis for systematic artifacts
- Automated flagging of unusual patterns
- Integration with quality metrics dashboard

---

## Files Modified

1. **`src/autoclean/tools/autoclean_exclude.py`**:
   - Added `_group_channel_removals()` helper (line 191)
   - Added `_get_removal_reason_display()` helper (line 217)
   - Updated `_extract_metadata_info()` (line 3998-4008)
   - Updated `_prepare_editor_metadata()` (line 2954-2982)
   - Enhanced `_refresh_related_list()` display (line 4122-4155)

---

## Validation

### Verify Enhancement Working

1. Open `autoclean_exclude` GUI
2. Load a .set file from recent processing run
3. Check "Related Files & Metadata" panel
4. Confirm you see grouped display with reasons

### Expected Output Format

```
Related Files & Metadata
├── Report (HTML): ✓ opens
├── Metadata (JSON): ✓ opens
├── ICA File: ✓ opens
├── PSD Plot: ✓ opens
├── Topomap: ✓ opens
├── TSV Channels: ✓ opens
─────────────────
Bad Channels (15):
  [EOG] E1, E8, E14, E17, E21, E25, E32, E125, E126, E127, E128
  [Uncorrelated] E34, E55, E58, E117
Rejected ICA: [0, 3, 7, 12]
```

---

## Documentation Complete ✅

All GUI enhancements are backward compatible and ready for use!
