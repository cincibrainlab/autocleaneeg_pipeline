# AutoClean Exclude File Searching Strategies

## Overview

The `autoclean_exclude.py` tool implements sophisticated file searching strategies to match base filenames with different asset types (processing logs, PSD overviews, run reports, and ICA reports). This document serves as ground truth for understanding how these strategies work and their potential inconsistencies.

## Core Architecture

### File Matching Methods

The tool implements four main file matching methods, each targeting a specific asset type:

1. **`_find_processing_log_for_file()`** - Matches `.csv` processing logs
2. **`_find_psd_overview_for_file()`** - Matches `.png` PSD topography figures  
3. **`_find_run_report_for_file()`** - Matches `.pdf` run reports
4. **`_find_ica_overview_for_file()`** - Matches `.pdf` ICA component reports

### Directory Resolution Strategy

Each method follows a consistent directory resolution pattern:

```python
def _find_[asset_type]_for_file(self, file_path: Path) -> Optional[Path]:
    # 1. Get asset-specific directory
    asset_dir = self._[asset_type]_reports_dir()
    if asset_dir is None:
        return None
    
    # 2. Find candidates using glob patterns
    candidates = list(asset_dir.glob("[pattern]"))
    
    # 3. Generate filename variants
    variants = self._generate_variants(file_path.stem)
    
    # 4. Score and select best match
    return self._select_best_match(candidates, variants)
```

## File Searching Strategies

### Strategy 1: Glob Pattern Matching

**Used by:** All four methods

**Patterns:**
- Processing logs: `*_processing_log.csv`
- PSD overviews: `*_psd_topo_figure.png`
- Run reports: `*_autoclean_report.pdf` (fallback to `*.pdf`)
- ICA reports: `*.pdf`

**Implementation:**
```python
candidates = list(asset_dir.glob("*_processing_log.csv"))
```

**Concerns:**
- Different glob patterns across asset types
- Run reports have fallback pattern (`*.pdf`) while others don't
- No wildcard flexibility for different naming conventions

### Strategy 2: Filename Variant Generation

**Used by:** All four methods

**Common Suffix Removal:**
```python
suffixes = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]
for suffix in suffixes:
    if stem.endswith(suffix):
        variants.add(stem[:-len(suffix)])
```

**Part-based Variants:**
```python
parts = stem.split("_")
# Different strategies across methods:
# Processing logs: Fixed lengths (3, 2, 1 parts)
# PSD/Run/ICA: Dynamic lengths (all lengths down to 1)
```

**Subject ID Variants:**
```python
if parts:
    variants.add(parts[0])                    # Raw subject ID
    variants.add(f"sub-{parts[0]}")          # BIDS format with dash
    variants.add(f"sub_{parts[0]}")          # BIDS format with underscore
```

**Inconsistencies Identified:**

1. **Part Length Strategy:**
   - **Processing logs:** Fixed approach - checks for 3, 2, and 1 parts
   - **PSD/Run/ICA:** Dynamic approach - iterates from full length down to 1

2. **Variant Generation Order:**
   - Processing logs generate variants in a specific order
   - Other methods use different iteration patterns

### Strategy 3: Scoring Algorithm

**Core Function:** `_normalized_prefix_score(a: str, b: str) -> int`

**Normalization Process:**
```python
def normalize(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum()).lower()
```

**Scoring Logic:**
```python
def _longest_common_prefix_length(a: str, b: str) -> int:
    length = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        length += 1
    return length
```

**Selection Process:**
```python
best_score = -1
best_path: Optional[Path] = None
for candidate in candidates:
    score = max(_normalized_prefix_score(candidate_stem, variant) for variant in variants)
    if score > best_score:
        best_score = score
        best_path = candidate
```

**Concerns:**
- Score threshold is `> 0` - any match is accepted
- No tie-breaking mechanism for equal scores
- Normalization removes all non-alphanumeric characters, potentially losing important separators

### Strategy 4: Asset-Specific Stem Processing

**Processing Logs:**
```python
if "_processing_log" in log_stem:
    log_prefix = log_stem.rsplit("_processing_log", 1)[0]
else:
    log_prefix = log_stem
```

**PSD Overviews:**
```python
if "_psd_topo" in c_stem:
    c_stem = c_stem.split("_psd_topo", 1)[0]
```

**Run Reports:**
```python
for needle in ("_autoclean_report", "_report"):
    if needle in c_stem:
        c_stem = c_stem.split(needle, 1)[0]
        break
```

**ICA Reports:**
```python
for needle in ("_ica_components", "_components", "_report"):
    if needle in c_stem:
        c_stem = c_stem.split(needle, 1)[0]
        break
```

**Inconsistencies Identified:**

1. **Needle Processing:**
   - Processing logs: Single needle with fallback
   - PSD: Single needle only
   - Run reports: Multiple needles with priority order
   - ICA reports: Multiple needles with priority order

2. **Split Strategy:**
   - Processing logs: `rsplit()` (right split)
   - Others: `split()` (left split)

## Directory Resolution Strategies

### Asset Directory Resolution

**Pattern:**
```python
def _[asset_type]_reports_dir(self) -> Optional[Path]:
    # Primary: task_root/reports/[asset_type]
    if self.task_root and (self.task_root / "reports" / "[asset_type]").exists():
        return self.task_root / "reports" / "[asset_type]"
    
    # Fallback: exports_dir.parent/reports/[asset_type]
    if self.exports_dir:
        candidate = self.exports_dir.parent / "reports" / "[asset_type]"
        if candidate.exists():
            return candidate
    return None
```

**Asset-Specific Directories:**
- Processing logs: Same directory as source file
- PSD overviews: `reports/psd_topo`
- Run reports: `reports/run_reports`
- ICA reports: `reports/ica_components`

**Concerns:**
- Processing logs use different resolution strategy (same directory)
- No fallback mechanisms for missing directories
- Hard-coded directory names

## File Discovery Strategy

### Related Files Discovery

**Method:** `_find_related_files(file_path: Path) -> List[Path]`

**Strategy:**
```python
base_stem = file_path.stem

# 1. Same directory siblings
for sibling in sorted(file_path.parent.iterdir()):
    if sibling.name.startswith(base_stem):
        results.append(sibling)

# 2. Reports directory recursive search
if self.task_root and self.task_root.exists():
    reports_root = self.task_root / "reports"
    if reports_root.exists():
        for report in sorted(reports_root.rglob("*")):
            if report.is_file() and base_stem in report.stem:
                results.append(report)
```

**Concerns:**
- Uses `startswith()` for siblings but `in` for reports
- No scoring mechanism for related files
- Potential for false positives with substring matching

## Identified Inconsistencies and Concerns

### 1. Variant Generation Inconsistencies

- **Processing logs:** Fixed part lengths (3, 2, 1)
- **Other methods:** Dynamic part lengths (all lengths down to 1)
- **Impact:** Different matching behavior for complex filenames

### 2. Stem Processing Inconsistencies

- **Split methods:** `rsplit()` vs `split()`
- **Needle handling:** Single vs multiple needles
- **Fallback strategies:** Different approaches across methods

### 3. Directory Resolution Inconsistencies

- **Processing logs:** Same directory as source file
- **Other assets:** Dedicated reports subdirectories
- **Impact:** Different file organization expectations

### 4. Scoring Threshold Issues

- **Threshold:** `> 0` accepts any match
- **No tie-breaking:** Equal scores not handled consistently
- **Normalization:** May lose important filename structure

### 5. Pattern Matching Inconsistencies

- **Glob patterns:** Different specificity levels
- **Fallback patterns:** Only run reports have fallback
- **Wildcard usage:** Inconsistent across asset types

## Recommendations for Standardization

### 1. Unify Variant Generation

```python
def _generate_standard_variants(self, stem: str) -> Set[str]:
    variants = {stem}
    
    # Standard suffix removal
    suffixes = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            variants.add(stem[:-len(suffix)])
    
    # Standard part-based variants (dynamic approach)
    parts = stem.split("_")
    for length in range(len(parts), 1, -1):
        variants.add("_".join(parts[:length]))
    
    # Standard subject ID variants
    if parts:
        variants.add(parts[0])
        variants.add(f"sub-{parts[0]}")
        variants.add(f"sub_{parts[0]}")
    
    return variants
```

### 2. Standardize Stem Processing

```python
def _extract_base_stem(self, candidate_stem: str, asset_type: str) -> str:
    needles = {
        'processing_log': ["_processing_log"],
        'psd_topo': ["_psd_topo"],
        'run_report': ["_autoclean_report", "_report"],
        'ica_report': ["_ica_components", "_components", "_report"]
    }
    
    for needle in needles.get(asset_type, []):
        if needle in candidate_stem:
            return candidate_stem.split(needle, 1)[0]
    
    return candidate_stem
```

### 3. Implement Consistent Scoring

```python
def _score_match(self, candidate_stem: str, variants: Set[str], asset_type: str) -> Tuple[int, str]:
    base_stem = self._extract_base_stem(candidate_stem, asset_type)
    
    best_score = -1
    best_variant = ""
    
    for variant in variants:
        score = _normalized_prefix_score(base_stem, variant)
        if score > best_score:
            best_score = score
            best_variant = variant
    
    return best_score, best_variant
```

### 4. Add Tie-Breaking Logic

```python
def _select_best_match_with_tiebreak(self, candidates: List[Path], variants: Set[str], asset_type: str) -> Optional[Path]:
    scored_candidates = []
    
    for candidate in candidates:
        score, variant = self._score_match(candidate.stem, variants, asset_type)
        if score > 0:
            scored_candidates.append((score, candidate, variant))
    
    if not scored_candidates:
        return None
    
    # Sort by score (descending), then by variant length (descending), then by filename
    scored_candidates.sort(key=lambda x: (-x[0], -len(x[2]), x[1].name))
    
    return scored_candidates[0][1]
```

## Testing Recommendations

### 1. Test Cases for Variant Generation

```python
test_cases = [
    "sub-001_task-rest_run-01_comp_epo",
    "sub-001_task-rest_run-01_clean",
    "sub-001_task-rest_run-01_raw",
    "sub-001_task-rest_run-01",
    "sub-001_task-rest",
    "sub-001"
]
```

### 2. Test Cases for Scoring

```python
scoring_tests = [
    ("sub001_task_rest_run01", "sub-001_task-rest_run-01", 8),  # Should match well
    ("sub001", "sub-001_task-rest_run-01", 6),                 # Partial match
    ("different", "sub-001_task-rest_run-01", 0),              # No match
]
```

### 3. Edge Cases

- Empty directories
- Files with special characters
- Very long filenames
- Unicode characters
- Case sensitivity
- Multiple files with same score

## Conclusion

The current file searching strategies in `autoclean_exclude.py` show significant inconsistencies across different asset types. While the core scoring algorithm is consistent, the variant generation, stem processing, and directory resolution strategies differ substantially. These inconsistencies could lead to:

1. **Unpredictable matching behavior** across different asset types
2. **Maintenance challenges** when adding new asset types
3. **User confusion** when files aren't found as expected
4. **Potential bugs** in edge cases

Standardizing these strategies would improve reliability, maintainability, and user experience while reducing the risk of matching failures.