# Plugin Blocks Architecture Plan

**Status:** Phase 1 In Progress - Discovery System Implemented ✅
**Date:** 2025-09-30
**Task-Registry Location:** `/Volumes/braindata/cbl_github/autocleaneeg-task-registry/`
**Implementation:** See `PLUGIN_DISCOVERY_IMPLEMENTATION.md` for technical details

---

## Summary

A comprehensive architecture plan has been created for transitioning AutoClean EEG blocks from duplicated multi-file structures to single-file plugins that import from the pipeline.

## Key Documents

### 1. Full Architecture Specification
**Location:** `task-registry/PLUGIN_BLOCK_ARCHITECTURE.md` (50+ pages)

**Contents:**
- Complete technical specification
- Current state analysis
- Proposed architecture
- Migration path
- Implementation roadmap (16-week plan)
- Code examples
- Testing strategy
- Distribution model

### 2. Prototype Implementation
**Location:** `task-registry/blocks/source_localization_plugin.py`

**Features:**
- Single-file block (vs. 6-file structure)
- Imports from pipeline (zero duplication)
- Comprehensive documentation
- Working example of new pattern
- 58% smaller than multi-file version

### 3. Quick Summary
**Location:** `task-registry/PLUGIN_ARCHITECTURE_SUMMARY.md`

**Contents:**
- Quick reference
- Key decisions
- Comparison tables
- FAQ
- Next steps

## Core Concept

**Align blocks with the existing task file pattern:**
- Blocks become single-file plugins
- Import from pipeline instead of duplicating code
- Use same discovery mechanism as tasks
- Zero synchronization overhead

## The Problem We're Solving

**Current State:**
```
Pipeline:         Task-Registry:
calc/source.py ←→ blocks/.../algorithm.py (DUPLICATE!)
mixins/.../   ←→ blocks/.../mixin.py (DUPLICATE!)

Result: Manual synchronization required for every change
```

**Proposed State:**
```
Pipeline:         Task-Registry:
calc/source.py ←─ blocks/source_localization_plugin.py (IMPORTS!)

Result: Pipeline is single source of truth
```

## Benefits

| Benefit | Impact |
|---------|--------|
| **Zero Duplication** | No more manual synchronization |
| **Single File** | Easy to understand, copy, share |
| **Familiar Pattern** | Works like task files (users already know) |
| **Auto-Discovery** | Pipeline finds blocks automatically |
| **Distributable** | Can pip install or copy file |
| **58% Smaller** | 20KB vs. 48KB per block |

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [x] Architecture document
- [x] Prototype block
- [x] Discovery system in pipeline (**IMPLEMENTED**)
- [ ] Migration script
- [ ] Testing framework

**Phase 1 Update (2025-09-30):**
- ✅ External block discovery system implemented in `src/autoclean/mixins/__init__.py`
- ✅ Searches three locations: `~/.autoclean/blocks/`, `./blocks/`, and task-registry via env var
- ✅ Loads Python files, extracts Mixin classes, adds to Task inheritance
- ✅ Prototype plugin copied to `~/.autoclean/blocks/source_localization_plugin.py`
- ⏳ Full testing requires complete dependency installation

### Phase 2: Core Blocks (Weeks 3-6)
- [ ] Migrate 5 analysis blocks
- [ ] Migrate 2 signal processing blocks
- [ ] Update documentation
- [ ] Deprecate old structure

### Phase 3: Distribution (Weeks 7-10)
- [ ] PyPI packages
- [ ] CI/CD workflows
- [ ] Installation guide
- [ ] Block registry

### Phase 4: Ecosystem (Weeks 11-16)
- [ ] CLI commands (`blocks list`, `blocks install`)
- [ ] Developer tools
- [ ] Community guidelines
- [ ] Block marketplace

## Changes Required in Pipeline

### 1. Enhanced Block Discovery ✅ **IMPLEMENTED**
**File:** `src/autoclean/mixins/__init__.py` (lines 251-327)

**Status:** Complete - External block discovery is now active

**Implementation Details:**
- Scans three locations: `~/.autoclean/blocks/`, `./blocks/`, task-registry (via env var)
- Uses `importlib.util.spec_from_file_location()` for direct file loading
- Extracts classes ending with "Mixin"
- Adds to Task inheritance after internal mixins
- Graceful error handling (bad blocks don't crash pipeline)
- Prints confirmation when blocks load: `✓ Loaded external block: {name}`

**See:** `PLUGIN_DISCOVERY_IMPLEMENTATION.md` for full technical details

### 2. Block Validation
**Status:** Future enhancement
**Optional:** Add validation for block metadata, dependencies, versions

### 3. CLI Commands (Future)
**Status:** Phase 4 (Weeks 11-16)
**Optional:** Add `autocleaneeg-pipeline blocks` subcommands

## Backwards Compatibility

**No breaking changes:**
- Internal mixins continue to work
- Existing tasks continue to work
- External blocks are additive

**Deprecation timeline:**
- v2.4.0: Introduce plugin system, mark old structure deprecated
- v2.5.0-v2.9.0: Support both formats
- v3.0.0: Remove old multi-file blocks

## Testing the Prototype

**1. Copy prototype to test location:**
```bash
cp /path/to/task-registry/blocks/source_localization_plugin.py \
   ~/.autoclean/blocks/
```

**2. Create test task:**
```python
from autoclean.core.task import Task

config = {
    "apply_source_localization": {
        "enabled": True,
        "value": {"method": "MNE"}
    }
}

class TestPluginBlock(Task):
    def run(self):
        self.import_raw()
        self.apply_source_localization()  # Method from plugin!
```

**3. Expected output:**
```
✓ Source localization plugin loaded (MNE v1.6.0)
✓ Algorithms available from autoclean.calc.source
Applying MNE source localization to Raw data...
```

## Next Steps

### Immediate (Week 1)
1. Review architecture document
2. Test prototype with pipeline discovery
3. Validate zero duplication
4. Get stakeholder approval

### Short-term (Weeks 2-4)
1. Implement discovery system in pipeline
2. Create migration script
3. Migrate first block (source_localization)
4. Document process

### Medium-term (Weeks 5-10)
1. Migrate remaining blocks
2. Set up PyPI distribution
3. Create installation guide
4. Update all documentation

### Long-term (Weeks 11-16+)
1. Build block ecosystem
2. Enable community contributions
3. Create block marketplace
4. Continuous improvement

## Resources

- **Architecture Doc:** `task-registry/PLUGIN_BLOCK_ARCHITECTURE.md`
- **Prototype:** `task-registry/blocks/source_localization_plugin.py`
- **Summary:** `task-registry/PLUGIN_ARCHITECTURE_SUMMARY.md`
- **Review Checklist:** `REVIEW_CHECKLIST.html` (interactive HTML guide)

## Questions & Feedback

**GitHub Issues:**
- Pipeline: https://github.com/cincibrainlab/autocleaneeg_pipeline/issues
- Task-Registry: https://github.com/cincibrainlab/autocleaneeg-task-registry/issues

**Contact:** ernest.pedapati@cchmc.org

---

**Status:** Architecture planning complete. Ready for stakeholder review and Phase 1 implementation.