# Plugin-Specific Montage Report Interface - Proposal

## Current State

### How `montage test` Works Now

The `autocleaneeg-pipeline montage test` command (in `cli.py:cmd_montage_test()`):

1. **Loads raw file** using standard MNE methods (not through plugin system)
2. **Loads montage** (.sfp file)
3. **Validates** channel name matching
4. **Generates** generic HTML report with 3D plots
5. **No plugin involvement** - plugins only run during actual pipeline processing

**Problem**: For MEA30_EDF, the report shows **0% match** because:
- Raw EDF has "Chan 1", "Chan 2", etc. (33 channels)
- Montage expects "Ch01", "Ch02", etc. (30 channels)
- Plugin's channel dropping & remapping never runs

### BaseEEGPlugin Current Interface

```python
class BaseEEGPlugin(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if plugin supports format/montage combo."""

    @abc.abstractmethod
    def import_and_configure(self, file_path, autoclean_dict, preload=True):
        """Import and configure in one step."""

    def process_events(self, raw):
        """Process events (optional override)."""

    def get_metadata(self) -> dict:
        """Get plugin metadata (optional override)."""
```

**No report generation method currently exists.**

---

## Proposed Solution

### Option 1: Conditional Report Enhancement (Quick Fix)

Add special handling in `cmd_montage_test` for MEA30_EDF:

```python
def cmd_montage_test(args):
    # ... existing code ...

    # After loading raw and montage
    if montage_name == "MEA30_EDF" and suffix == ".edf":
        # Special handling for MEA30 EDF files
        console.print()
        console.print("[bold yellow]⚠ MEA30 EDF SPECIAL HANDLING[/bold yellow]")
        console.print("[info]This montage requires channel remapping via plugin.[/info]")
        console.print("[info]Raw file: 33 channels → Plugin output: 30 channels[/info]")
        console.print()
        console.print("[info]Channels dropped:[/info] Chan 2, 32, 33")
        console.print("[info]Channel remapping:[/info] Chan 1→Ch23, Chan 3→Ch22, etc.")
        console.print()
        console.print("[yellow]To see actual result, run pipeline processing.[/yellow]")
```

**Pros:**
- Quick to implement
- No architecture changes
- Works immediately

**Cons:**
- Hard-coded logic
- Not extensible to other plugins
- Doesn't scale

### Option 2: Plugin Report Interface (Unified Architecture) ⭐ RECOMMENDED

Add optional `generate_montage_report()` method to `BaseEEGPlugin`:

```python
class BaseEEGPlugin(abc.ABC):
    # ... existing abstract methods ...

    def generate_montage_report(
        self,
        raw_before: mne.io.Raw,
        raw_after: mne.io.Raw,
        montage: mne.channels.DigMontage,
        report_data: dict
    ) -> dict:
        """Generate plugin-specific montage validation report data.

        This method is called by `montage test` to generate custom
        report sections for plugins that perform complex transformations.

        Args:
            raw_before: Raw data BEFORE plugin processing
            raw_after: Raw data AFTER plugin processing
            montage: The montage that was applied
            report_data: Standard report data dict to augment

        Returns:
            dict: Additional report sections with keys:
                - 'html_sections': List of HTML strings to insert
                - 'summary_stats': Dict of additional statistics
                - 'warnings': List of warning messages
                - 'visualizations': Dict of plot objects

        If not overridden, returns empty dict (no custom report).
        """
        return {}
```

**Update `cmd_montage_test` to use plugins:**

```python
def cmd_montage_test(args):
    # ... existing code to load raw file ...

    # Check if a plugin exists for this format/montage
    plugin = find_plugin_for_format_montage(file_format, montage_name)

    if plugin:
        # Use plugin to process file
        console.print(f"[info]Using plugin: {plugin.__class__.__name__}[/info]")

        # Save raw_before for comparison
        raw_before = raw.copy()

        # Process through plugin
        raw_after = plugin.import_and_configure(input_file, {}, preload=True)

        # Generate custom report if plugin supports it
        if hasattr(plugin, 'generate_montage_report'):
            custom_report = plugin.generate_montage_report(
                raw_before, raw_after, montage, report_data
            )
            # Merge custom_report into standard report
            report_data.update(custom_report)
    else:
        # Use standard MNE loading (current behavior)
        raw_after = raw

    # Generate report using raw_after
    # ...
```

**Example plugin implementation (MEA30_EDF):**

```python
class EDFMouseMEA30Plugin(BaseEEGPlugin):
    # ... existing methods ...

    def generate_montage_report(self, raw_before, raw_after, montage, report_data):
        """Generate custom report showing channel transformations."""

        html_sections = []

        # Transformation summary
        html_sections.append(f"""
        <div class="plugin-report">
            <h3>🔄 MEA30 EDF Channel Transformation</h3>
            <div class="transformation-info">
                <h4>Input (Raw EDF):</h4>
                <ul>
                    <li>33 channels with generic names ("Chan 1" - "Chan 33")</li>
                    <li>Scrambled hardware routing order</li>
                    <li>No embedded coordinates</li>
                </ul>

                <h4>Transformation Pipeline:</h4>
                <ol>
                    <li><strong>Drop channels:</strong> Chan 2, 32, 33 (reference/ground)</li>
                    <li><strong>Remap channels:</strong> 30 channels to anatomical MEA order</li>
                    <li><strong>Apply coordinates:</strong> 3D MNI brain positions</li>
                </ol>

                <h4>Output (Processed):</h4>
                <ul>
                    <li>30 EEG channels ("Ch01" - "Ch30")</li>
                    <li>Anatomically ordered MEA positions</li>
                    <li>Full 3D coordinate information</li>
                </ul>
            </div>
        </div>
        """)

        # Channel mapping table
        mapping_table = self._generate_mapping_table()
        html_sections.append(mapping_table)

        # Validation info
        html_sections.append(f"""
        <div class="validation-info">
            <h4>✓ Validated Against:</h4>
            <ul>
                <li>MATLAB edf2meaLookupTest function</li>
                <li>Mea_adult_atlas-30_dict.csv</li>
                <li>Mea_P21_atlas-30_dict.csv</li>
            </ul>
            <p><em>All coordinates match exactly (< 0.001m tolerance)</em></p>
        </div>
        """)

        return {
            'html_sections': html_sections,
            'summary_stats': {
                'channels_before': len(raw_before.ch_names),
                'channels_after': len(raw_after.ch_names),
                'channels_dropped': 3,
                'channels_remapped': 30,
                'transformation': 'MEA30 EDF scramble correction'
            },
            'warnings': []
        }

    def _generate_mapping_table(self):
        """Generate HTML table showing channel mappings."""
        # ... implementation ...
```

**Pros:**
- ✅ **Extensible**: Any plugin can add custom reports
- ✅ **Clean architecture**: Follows plugin pattern
- ✅ **Optional**: Plugins don't need reports if not useful
- ✅ **Powerful**: Full access to before/after data
- ✅ **Standardized interface**: All plugins use same method signature

**Cons:**
- More complex implementation
- Requires updating `cmd_montage_test` logic
- Need to handle case where plugin fails/errors

### Option 3: Separate Report Plugin System

Create a parallel plugin system just for reports:

```python
class MontageReportPlugin(abc.ABC):
    @abstractmethod
    def should_handle(self, format_id: str, montage_name: str) -> bool:
        """Check if this report plugin applies."""

    @abstractmethod
    def generate_report_sections(self, raw, montage, metadata) -> dict:
        """Generate custom report sections."""
```

**Pros:**
- Decouples report generation from data import
- Can have multiple report plugins per montage

**Cons:**
- More complex architecture
- Duplication of format/montage detection logic
- Overkill for current needs

---

## Recommendation

**Go with Option 2: Plugin Report Interface**

### Implementation Plan

1. **Phase 1: Add optional method to BaseEEGPlugin** (backward compatible)
   ```python
   def generate_montage_report(self, raw_before, raw_after, montage, report_data):
       return {}  # Default: no custom report
   ```

2. **Phase 2: Update cmd_montage_test to detect and use plugins**
   - Detect if plugin exists for format/montage combo
   - Run plugin.import_and_configure() to get processed data
   - Call plugin.generate_montage_report() if it exists
   - Merge custom report sections into HTML output

3. **Phase 3: Implement for MEA30_EDF plugin**
   - Add generate_montage_report() method
   - Create HTML showing transformation pipeline
   - Include channel mapping table
   - Add validation information

4. **Phase 4: Update HTML report template**
   - Add section for plugin-specific content
   - Style custom report sections
   - Handle optional nature gracefully

### Benefits

1. **Immediate value**: MEA30_EDF gets clear explanation of transformations
2. **Future-proof**: Other complex plugins can add reports (e.g., XDAT with scrambling)
3. **Clean**: Follows existing plugin architecture
4. **Optional**: Simple plugins don't need extra work
5. **Powerful**: Full control over report content and styling

### Example Use Cases

- **MEA30_EDF**: Show channel dropping and remapping
- **XDAT_H32**: Explain scrambled pin routing corrections
- **BDF with odd naming**: Show renaming transformations
- **Custom arrays**: Explain custom coordinate transformations

---

## Code Locations

**Files to modify:**

1. `src/autoclean/io/import_.py`
   - Add `generate_montage_report()` to BaseEEGPlugin
   - Add helper to find plugin for format/montage

2. `src/autoclean/cli.py`
   - Update `cmd_montage_test()` to use plugins
   - Add plugin report section to HTML generation

3. `src/autoclean/plugins/eeg_plugins/edf_mea30_plugin.py`
   - Implement `generate_montage_report()`
   - Create channel mapping visualization

**New files needed:**

- None (uses existing architecture)

---

## Example Output

**Current montage test for MEA30_EDF:**
```
Match: 0.0% (0/33 channels)
Positioned: 0/33
⚠ No channels matched
```

**With plugin report:**
```
Match: 100% (30/30 channels)
Positioned: 30/30

🔄 MEA30 EDF Transformation
  Input:  33 channels (Chan 1-33, scrambled order)
  Dropped: 3 channels (2, 32, 33)
  Remapped: 30 channels to anatomical order
  Output: 30 positioned MEA channels (Ch01-Ch30)

[Detailed HTML report shows:]
- Visual diagram of transformation pipeline
- Interactive channel mapping table
- Before/after comparison
- Validation information
```

---

## Timeline Estimate

- **Option 1 (Conditional)**: 30 minutes
- **Option 2 (Plugin Interface)**: 3-4 hours
  - 1 hour: Update BaseEEGPlugin and find_plugin logic
  - 1 hour: Update cmd_montage_test to use plugins
  - 1-2 hours: Implement MEA30_EDF report method
- **Option 3 (Separate System)**: 6-8 hours

---

## Decision

Should we:
1. **Quick fix**: Add conditional MEA30_EDF handling in montage test?
2. **Unified interface**: Implement plugin report interface (Option 2)?
3. **Something else**: Alternative approach?

My recommendation is **Option 2** for long-term architecture, but **Option 1** could be done first as a temporary solution if you need something working immediately.
