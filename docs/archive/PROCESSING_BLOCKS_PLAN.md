# Processing Blocks Architecture Plan

**Date**: 2025-09-29
**Goal**: Create co-located "plugin packet" processing blocks for Lego-style modularity

## Executive Summary

Transform processing methods into self-contained "processing blocks" that bundle:
- Mixin interface (task-level API)
- Core algorithm implementation
- Documentation and examples
- Registry metadata (manifest.json)
- Schema definitions

**Primary Candidate**: Wavelet thresholding (902 lines, well-isolated, scientifically complex)

---

## Current Architecture Analysis

### Wavelet Thresholding is Currently Scattered

**File locations:**
- **Mixin**: `mixins/signal_processing/wavelet_threshold.py` (249 lines)
  - Task-level interface
  - Config validation
  - Metadata tracking
  - Report path resolution

- **Implementation**: `functions/preprocessing/wavelet_thresholding.py` (902 lines)
  - Core algorithm (lines 71-294)
  - Report generation (lines 295-902)
  - Visualization helpers
  - PDF creation

- **Schema**: `configkit/schema.py`
  - `_wavelet_descriptor()` function (lines 82-96)
  - Validation rules

- **Documentation**: Scattered in docstrings, no central guide

**Issues with current organization:**
- Related code spread across 3+ locations
- Hard to discover what processing methods exist
- No standardized metadata format
- Difficult to share blocks between projects
- No version tracking per processing method

---

## Proposed Architecture

### Option A: In-Repo Processing Blocks

```
src/autoclean/plugins/processing_blocks/
├── __init__.py                    # Discovery system
├── README.md                      # Architecture documentation
└── wavelet_threshold/             # Self-contained block
    ├── __init__.py                # Public API exports
    ├── mixin.py                   # WaveletThresholdMixin
    ├── algorithm.py               # wavelet_threshold() core
    ├── reporting.py               # generate_wavelet_report()
    ├── schema.py                  # Config schema
    ├── manifest.json              # Metadata (see below)
    └── README.md                  # User docs + examples
```

**Pros:**
- Easy to maintain in single repo
- Simple imports
- No external dependency management
- Works offline

**Cons:**
- Blocks tied to pipeline version
- Harder to share blocks independently
- Can't install just one block

### Option B: Separate Registry Repository (Recommended)

```
autocleaneeg-processing-blocks/ (separate repo)
├── blocks/
│   ├── wavelet_threshold/
│   │   ├── __init__.py
│   │   ├── mixin.py
│   │   ├── algorithm.py
│   │   ├── reporting.py
│   │   ├── schema.py
│   │   ├── manifest.json
│   │   └── README.md
│   │
│   ├── advanced_ica/
│   │   └── ...
│   │
│   └── connectivity_measures/
│       └── ...
│
├── registry.json                  # Central block registry
└── README.md                      # Registry overview
```

**Integrate with existing task registry:**
```
autocleaneeg-task-registry/ (existing repo)
├── tasks/                         # Existing task files
│   ├── resting/
│   ├── auditory/
│   └── ...
│
├── blocks/                        # NEW: Processing blocks
│   ├── wavelet_threshold/
│   ├── advanced_ica/
│   └── ...
│
├── registry.json                  # Tasks registry
└── blocks_registry.json           # NEW: Blocks registry
```

**Pros:**
- Independent versioning per block
- Can share blocks across projects
- Install only needed blocks
- Mirrors task registry pattern
- Community can contribute blocks
- Can have different update cadence

**Cons:**
- More complex dependency management
- Need installation mechanism
- Requires network for remote blocks

---

## Manifest Format

```json
{
  "name": "wavelet_threshold",
  "version": "1.0.0",
  "description": "Wavelet-based artifact removal using discrete wavelet transform with universal thresholding",
  "author": "AutoCleanEEG Team",
  "maintainer": "ernie.pedapati@cchmc.org",
  "license": "MIT",
  "category": "signal_processing",

  "api": {
    "mixin_class": "WaveletThresholdMixin",
    "mixin_method": "apply_wavelet_threshold",
    "config_key": "wavelet_threshold",
    "schema_function": "wavelet_descriptor"
  },

  "dependencies": {
    "python": ">=3.10",
    "packages": {
      "pywt": ">=1.9.0",
      "numpy": ">=2.0.0",
      "matplotlib": ">=3.10.0",
      "reportlab": ">=4.0.0"
    },
    "mne": ">=1.10.0",
    "autocleaneeg-pipeline": ">=3.0.0"
  },

  "compatibility": {
    "data_types": ["raw", "epochs"],
    "processing_stages": ["post_filtering", "pre_epoching", "post_ica"],
    "min_channels": 1,
    "requires_montage": false
  },

  "references": [
    {
      "name": "HAPPE Pipeline",
      "url": "https://github.com/PINE-Lab/HAPPE",
      "citation": "Gabard-Durnam et al. (2018)"
    },
    {
      "name": "Wavelet thresholding theory",
      "citation": "Donoho & Johnstone (1994). Ideal spatial adaptation by wavelet shrinkage",
      "doi": "10.1093/biomet/81.3.425"
    }
  ],

  "tags": ["artifact-removal", "denoising", "wavelet", "preprocessing", "happe"],

  "registry": {
    "url": "https://registry.autocleaneeg.org/blocks/wavelet_threshold",
    "documentation_url": "https://docs.autocleaneeg.org/blocks/wavelet_threshold",
    "source_url": "https://github.com/cincibrainlab/autocleaneeg-task-registry/tree/main/blocks/wavelet_threshold"
  },

  "created_at": "2025-09-29T00:00:00Z",
  "updated_at": "2025-09-29T00:00:00Z"
}
```

---

## Implementation Phases

### Phase 1: Infrastructure Setup (Week 1)

**A. Repository Decision**
- [ ] Decide: In-repo vs separate repo vs extend task registry
- [ ] If separate: Create `autocleaneeg-processing-blocks` repo
- [X] If task registry: Add `blocks/` directory

**B. Create Discovery System**
- [] Create `processing_blocks/__init__.py` with auto-discovery
- [ ] Mirror mixin discovery pattern
- [ ] Add manifest validation
- [ ] Create central block registry class

**C. Schema System Integration**
- [ ] Extend schema export to include blocks
- [ ] Add block schema validation
- [ ] Update `configkit` to discover block schemas

### Phase 2: Wavelet Block Migration (Week 1-2)

**A. Create Block Structure**
- [ ] Create `blocks/wavelet_threshold/` directory
- [ ] Write `manifest.json` with full metadata
- [ ] Create `README.md` with comprehensive docs

**B. Refactor Code**
- [ ] `mixin.py`: Extract from `mixins/signal_processing/wavelet_threshold.py`
- [ ] `algorithm.py`: Core functions from `wavelet_thresholding.py` (lines 71-294)
- [ ] `reporting.py`: Report generation (lines 295-902)
- [ ] `schema.py`: Extract `_wavelet_descriptor()` from configkit
- [ ] `__init__.py`: Clean public API

**C. Backward Compatibility**
- [ ] Keep old file locations as thin import wrappers
- [ ] Add deprecation warnings (DeprecationWarning)
- [ ] Update internal imports
- [ ] Test all existing tasks still work

**D. Documentation**
- [ ] Scientific background (DWT theory)
- [ ] Usage examples from tasks
- [ ] Parameter tuning guide
- [ ] Comparison with HAPPE MATLAB implementation
- [ ] Literature references

### Phase 3: CLI Integration (Week 2)

**A. Block Management Commands**
```bash
autocleaneeg-pipeline blocks list
autocleaneeg-pipeline blocks info <block_name>
autocleaneeg-pipeline blocks search <keyword>
autocleaneeg-pipeline blocks validate
autocleaneeg-pipeline blocks install <block_name>  # If remote
autocleaneeg-pipeline blocks update
```

**B. Schema Export Enhancement**
```bash
autocleaneeg-pipeline schema export --blocks-only
autocleaneeg-pipeline schema export --include-blocks
autocleaneeg-pipeline blocks schema <block_name>
```

### Phase 4: Additional Block Candidates (Week 3-4)

See "Candidate Analysis" section below for full list.

**Priority candidates:**
1. Advanced ICA methods (ICALabel integration)
2. Autoreject (already well-isolated)
3. Connectivity measures (complex, valuable)
4. Source localization (advanced users)

### Phase 5: Registry Integration (Week 4+)

**A. Web Registry**
- [ ] Add blocks to registry.autocleaneeg.org
- [ ] Block search/browse interface
- [ ] Download statistics
- [ ] Community ratings/comments

**B. TaskWizard Integration**
- [ ] Query manifest for available blocks
- [ ] Display block descriptions in UI
- [ ] Show parameter ranges from schema
- [ ] Link to documentation

---

## Integration with Task Registry

### Current Task Registry Structure

```
autocleaneeg-task-registry/
├── tasks/
│   ├── resting/
│   │   ├── RestingEyesOpen.py
│   │   └── RestingEyesClosed.py
│   ├── auditory/
│   │   ├── ASSR_40Hz.py
│   │   └── MMN_Standard.py
│   └── ...
│
├── registry.json
├── README.md
└── .github/workflows/
```

### Proposed Extension

```
autocleaneeg-task-registry/
├── tasks/                         # Existing
│   └── ...
│
├── blocks/                        # NEW: Processing blocks
│   ├── signal_processing/
│   │   ├── wavelet_threshold/
│   │   ├── advanced_ica/
│   │   └── autoreject/
│   ├── analysis/
│   │   ├── connectivity/
│   │   └── source_localization/
│   └── visualization/
│       └── topographic_plots/
│
├── registry.json                  # Existing task registry
├── blocks_registry.json           # NEW: Block registry
├── README.md
└── .github/workflows/
    ├── validate_tasks.yml         # Existing
    └── validate_blocks.yml        # NEW
```

### Block Registry Format (`blocks_registry.json`)

```json
{
  "version": 1,
  "commit": "sha256hash",
  "updated_at": "2025-09-29T00:00:00Z",
  "blocks": [
    {
      "name": "wavelet_threshold",
      "category": "signal_processing",
      "path": "blocks/signal_processing/wavelet_threshold",
      "version": "1.0.0",
      "tags": ["artifact-removal", "denoising", "wavelet"],
      "downloads": 0,
      "rating": 0.0
    },
    {
      "name": "advanced_ica",
      "category": "signal_processing",
      "path": "blocks/signal_processing/advanced_ica",
      "version": "1.0.0",
      "tags": ["ica", "artifact-removal", "icalabel"],
      "downloads": 0,
      "rating": 0.0
    }
  ]
}
```

### CLI Integration with Remote Registry

```bash
# List available blocks from registry
autocleaneeg-pipeline blocks list --source=registry

# Install block from registry
autocleaneeg-pipeline blocks install wavelet_threshold

# Update all installed blocks
autocleaneeg-pipeline blocks update

# Show block details from registry
autocleaneeg-pipeline blocks info wavelet_threshold --source=registry
```

---

## Benefits Analysis

### For Developers

**Discoverability**
- One place to find all processing methods
- Clear API boundaries
- Self-documenting structure

**Maintainability**
- Related code co-located
- Easier to test in isolation
- Version tracking per block
- Clear dependencies

**Extensibility**
- Easy to add new blocks
- Standard template to follow
- Can contribute without touching core

### For Users

**Discovery**
- `blocks list` shows available methods
- Search by tags/category
- Read documentation in one place

**Usage**
- Examples in block README
- Parameter guides
- Scientific references co-located

**Customization**
- Can fork individual blocks
- Easier to understand implementation
- Can share custom blocks

### For Research Reproducibility

**Version Control**
- Each block has version number
- Dependencies clearly stated
- Can pin block versions in tasks

**Documentation**
- Scientific references with DOIs
- Method comparison with literature
- Parameter justification

**Sharing**
- Can bundle blocks with publications
- Others can install exact versions
- Clear licensing per block

### For TaskWizard

**UI Generation**
- Query manifest for descriptions
- Show compatible data types
- Display parameter schemas
- Link to documentation

**Smart Suggestions**
- Recommend blocks based on task type
- Show popularity/ratings
- Warn about incompatibilities

---

## Candidate Analysis

### Complete Function Inventory

**Current `functions/` structure:**
```
functions/
├── advanced/
│   └── autoreject.py                      (315 lines)
├── analysis/
│   └── statistical_learning.py            (812 lines)
├── artifacts/
│   └── channels.py                        (362 lines)
├── epoching/
│   ├── eventid.py
│   ├── quality.py
│   ├── regular.py
│   ├── statistical_randomized.py
│   └── statistical.py
├── ica/
│   └── ica_processing.py                  (740 lines)
├── preprocessing/
│   ├── basic_ops.py
│   ├── filtering.py
│   ├── referencing.py
│   ├── resampling.py
│   └── wavelet_thresholding.py            (901 lines) ⭐
├── segment_rejection/
│   ├── dense_oscillatory.py               (201 lines)
│   └── segment_rejection.py               (735 lines)
└── visualization/
    ├── icvision_layouts.py
    ├── plotting.py
    └── reports.py
```

---

### Priority 1: Ideal Candidates (Implement First)

#### 1. **Wavelet Thresholding** ⭐⭐⭐⭐⭐
**Location**: `functions/preprocessing/wavelet_thresholding.py` (901 lines)

**Why ideal:**
- ✅ Large, well-isolated codebase
- ✅ Complex algorithm with scientific depth
- ✅ Has extensive reporting (PDF generation)
- ✅ Already has clear separation (algorithm + reporting)
- ✅ References established method (HAPPE)
- ✅ Unique enough to be a standalone tool

**Block structure:**
```
wavelet_threshold/
├── mixin.py          # Task interface
├── algorithm.py      # Core DWT + thresholding
├── reporting.py      # PDF generation + visualization
├── schema.py         # Config validation
├── manifest.json     # Metadata + references
└── README.md         # Theory + usage guide
```

**Scientific value:**
- DWT theory and universal thresholding
- Comparison with HAPPE MATLAB implementation
- Parameter tuning guide (wavelet family, threshold scale)
- ERP-preserving mode explanation

**Estimated effort**: 2-3 days

---

#### 2. **Advanced ICA Processing** ⭐⭐⭐⭐⭐
**Location**: `functions/ica/ica_processing.py` (740 lines)

**Why ideal:**
- ✅ Integrates ICALabel and ICVision
- ✅ Multiple classification methods
- ✅ Complex enough to benefit from documentation
- ✅ Active research area
- ✅ Users need guidance on parameter choices

**Block structure:**
```
advanced_ica/
├── mixin.py          # Task interface
├── fitting.py        # fit_ica()
├── classification.py # label_ica_components()
├── rejection.py      # apply_ica()
├── control_sheet.py  # Control sheet management
├── schema.py         # Config validation
├── manifest.json     # Metadata
└── README.md         # ICA theory + comparison guide
```

**Scientific value:**
- ICA algorithm comparison (FastICA vs Infomax vs Picard)
- ICALabel deep learning classifier explanation
- ICVision integration
- Component interpretation guide
- References to seminal ICA papers

**Dependencies:**
- mne-icalabel
- icvision (optional)

**Estimated effort**: 3-4 days

---

#### 3. **AutoReject Epoch Cleaning** ⭐⭐⭐⭐
**Location**: `functions/advanced/autoreject.py` (315 lines)

**Why good candidate:**
- ✅ Well-isolated (315 lines)
- ✅ Machine learning method
- ✅ Clear parameter space
- ✅ Research paper backing
- ✅ Alternative to manual rejection

**Block structure:**
```
autoreject/
├── mixin.py          # Task interface
├── algorithm.py      # autoreject_epochs()
├── visualization.py  # Diagnostic plots
├── schema.py         # Config validation
├── manifest.json     # Metadata
└── README.md         # Cross-validation explanation
```

**Scientific value:**
- AutoReject algorithm explanation
- Cross-validation methodology
- Parameter selection guidance
- Comparison with manual rejection
- Reference to Jas et al. (2017) paper

**Dependencies:**
- autoreject

**Estimated effort**: 1-2 days

---

#### 4. **Statistical Learning Analysis** ⭐⭐⭐⭐
**Location**: `functions/analysis/statistical_learning.py` (812 lines)

**Why good candidate:**
- ✅ Large, complex module (812 lines)
- ✅ Specialized research application
- ✅ Inter-trial coherence (ITC) analysis
- ✅ Word Learning Index calculation
- ✅ Domain-specific knowledge required

**Block structure:**
```
statistical_learning_itc/
├── mixin.py          # Task interface
├── itc_analysis.py   # compute_statistical_learning_itc()
├── wli_calculation.py # calculate_word_learning_index()
├── significance.py   # Rayleigh test
├── visualization.py  # ITC topoplots
├── schema.py         # Config validation
├── manifest.json     # Metadata
└── README.md         # Neural entrainment explanation
```

**Scientific value:**
- Neural entrainment theory
- ITC calculation methodology
- Word Learning Index (WLI) metric
- Syllable vs word frequency analysis (3.33 Hz vs 1.11 Hz)
- Statistical significance testing
- References to statistical learning research

**Dependencies:**
- Scipy for Rayleigh test

**Estimated effort**: 3-4 days

---

#### 5. **Segment Rejection** ⭐⭐⭐⭐
**Location**: `functions/segment_rejection/segment_rejection.py` (735 lines)

**Why good candidate:**
- ✅ Large module (735 lines)
- ✅ Multiple rejection methods
- ✅ IQR-based outlier detection
- ✅ Correlation-based rejection
- ✅ Pylossless-inspired

**Block structure:**
```
segment_rejection/
├── mixin.py                # Task interface
├── noisy_segments.py       # annotate_noisy_segments()
├── correlation_rejection.py # annotate_bad_epochs_correlation()
├── dense_oscillatory.py    # Dense oscillation detection
├── visualization.py        # Segment quality plots
├── schema.py               # Config validation
├── manifest.json           # Metadata
└── README.md               # Rejection strategies guide
```

**Scientific value:**
- IQR-based outlier detection theory
- Correlation-based quality metrics
- Comparison with FASTER, pylossless
- Parameter tuning for different noise types
- Annotation strategy

**Estimated effort**: 2-3 days

---

### Priority 2: Good Candidates (Implement After Priority 1)

#### 6. **Bad Channel Detection** ⭐⭐⭐
**Location**: `functions/artifacts/channels.py` (362 lines)

**Why consider:**
- Well-isolated functionality
- RANSAC-based detection
- Multiple detection methods
- Important preprocessing step

**Block potential:**
```
bad_channel_detection/
├── mixin.py
├── ransac_detection.py
├── correlation_detection.py
├── visualization.py
├── manifest.json
└── README.md
```

**Estimated effort**: 1-2 days

---

#### 7. **Dense Oscillatory Rejection** ⭐⭐⭐
**Location**: `functions/segment_rejection/dense_oscillatory.py` (201 lines)

**Why consider:**
- Specialized rejection method
- Targets specific artifact type
- Could be combined with main segment_rejection block
- Or separate block for specialists

**Block potential:**
```
dense_oscillatory_detection/
├── mixin.py
├── algorithm.py
├── visualization.py
├── manifest.json
└── README.md
```

**Estimated effort**: 1 day

---

### Priority 3: Consider for Later

#### 8. **Source Localization** ⭐⭐
**Location**: `mixins/signal_processing/source_localization.py`

**Why lower priority:**
- Advanced users only
- Requires head models
- Complex setup
- Better as separate advanced package

**Recommendation**: Create as advanced block after core blocks are established

---

#### 9. **Connectivity Measures** ⭐⭐
**Location**: Various analysis functions

**Why lower priority:**
- Multiple methods (coherence, PLV, etc.)
- Analysis-focused rather than preprocessing
- Could be multiple blocks

**Recommendation**: Group related connectivity methods into themed blocks

---

### Priority 4: Not Recommended for Blocks

#### Small/Simple Functions (Keep as utilities)
- `preprocessing/basic_ops.py` - Too simple
- `preprocessing/filtering.py` - Core functionality, not block-worthy
- `preprocessing/referencing.py` - Core functionality
- `preprocessing/resampling.py` - Core functionality
- `epoching/*.py` - Core functionality

**Reason**: These are fundamental operations that should stay as core mixins, not optional blocks.

---

## Processing Block Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. ✅ Wavelet Thresholding
2. ✅ AutoReject

**Goal**: Prove the pattern works, establish tooling

### Phase 2: Core Methods (Weeks 3-4)
3. ✅ Advanced ICA Processing
4. ✅ Segment Rejection

**Goal**: Cover most common advanced preprocessing

### Phase 3: Specialized Analysis (Weeks 5-6)
5. ✅ Statistical Learning ITC
6. ✅ Bad Channel Detection
7. ✅ Dense Oscillatory Detection

**Goal**: Support specialized research protocols

### Phase 4: Advanced (Weeks 7+)
8. Source Localization (separate advanced block)
9. Connectivity Measures (multiple themed blocks)
10. Custom visualization blocks

**Goal**: Build ecosystem for advanced users

---

## Task Registry Integration Strategy

### Recommended Approach: Extend Task Registry

**Why this makes sense:**
1. **Single source of truth**: Tasks and blocks in one place
2. **Shared infrastructure**: Same CI/CD, validation, hosting
3. **Consistent user experience**: One registry, one CLI
4. **Logical organization**: Blocks are building blocks for tasks

**Repository structure:**
```
autocleaneeg-task-registry/
├── tasks/                    # Existing
│   ├── resting/
│   ├── auditory/
│   └── ...
│
├── blocks/                   # NEW
│   ├── signal_processing/
│   │   ├── wavelet_threshold/
│   │   ├── advanced_ica/
│   │   ├── autoreject/
│   │   └── segment_rejection/
│   ├── analysis/
│   │   └── statistical_learning_itc/
│   └── artifacts/
│       └── bad_channel_detection/
│
├── registry.json             # Tasks (existing)
├── blocks_registry.json      # Blocks (new)
├── README.md
├── BLOCKS_README.md          # NEW: Block contribution guide
└── .github/workflows/
    ├── validate_tasks.yml    # Existing
    └── validate_blocks.yml   # NEW
```

### Pipeline Integration

**In main pipeline repo:**
```python
# src/autoclean/plugins/processing_blocks/__init__.py
# Discovery system that can:
# 1. Auto-discover local blocks (bundled with pipeline)
# 2. Fetch blocks from task registry (remote)
# 3. Install blocks to workspace
# 4. Validate block manifests
```

**Workflow:**
1. User runs: `autocleaneeg-pipeline blocks install wavelet_threshold`
2. CLI fetches from task registry
3. Downloads block to `~/.config/autocleaneeg/blocks/`
4. Discovery system auto-imports on next run
5. Block mixin available to all tasks

### Advantages of Task Registry Integration

1. **Unified ecosystem**: One place for all AutoCleanEEG extensions
2. **Easy discovery**: Users browse one site for tasks + blocks
3. **Shared tooling**: Same validation, CI, deployment
4. **Natural fit**: Blocks are building blocks for tasks
5. **Community**: Single community forum for contributions

---

## Next Steps

1. **Review this plan** - Discuss priorities and approach
2. **Choose repository strategy** - In-repo, separate, or task registry?
3. **Create proof-of-concept** - Implement wavelet_threshold block
4. **Develop CLI commands** - `blocks list/info/install/validate`
5. **Document contribution process** - Guide for creating new blocks
6. **Plan registry integration** - Web interface for block discovery

---

## Questions for Discussion

1. **Repository location**: Extend task registry vs separate repo?
2. **Installation mechanism**: How do users install remote blocks?
3. **Version compatibility**: How to handle block vs pipeline versions?
4. **Community contributions**: Process for accepting new blocks?
5. **Documentation hosting**: Mintlify vs separate docs?
6. **Testing strategy**: How to test blocks in isolation?

---

## Summary

**Total candidates identified**: 9 processing blocks
**Priority 1 (ideal)**: 5 blocks
**Priority 2 (good)**: 2 blocks
**Priority 3 (later)**: 2 blocks

**Recommended first implementation**: Wavelet Thresholding
**Estimated timeline**: 6-8 weeks for all Priority 1 + 2 blocks
**Repository recommendation**: Extend autocleaneeg-task-registry