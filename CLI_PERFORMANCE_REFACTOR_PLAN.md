# AutoClean CLI Performance Refactor Plan

## Problem Analysis

### Current Performance Issues
- **CLI startup time**: 4-5 seconds for any command (including `--help`, `--version`)
- **Heavy imports on every command**: Even lightweight commands load the entire EEG processing stack
- **Circular dependencies**: Creates import loops that force loading of unnecessary modules

### Root Causes Identified

#### 1. Monolithic Import Architecture
```
Every CLI command → Imports everything
```
- `user_config` is imported at module level for ALL commands
- Task discovery system loads ALL task modules on every command
- Rich console, database, auth systems loaded unconditionally

#### 2. Circular Dependencies
```
CLI → user_config → autoclean package → core pipeline → utils → user_config
```
- `user_config.py` imports the entire `autoclean` package just to get version
- Creates recursive import chain that loads everything

#### 3. Unnecessary Heavy Dependencies
- **PyTorch** (1GB+): Imported just for GPU detection in system info display
- **psutil**: Imported for system resource detection (rarely needed)
- **Rich**: Imported for console formatting (not needed for simple commands)
- **Task Discovery**: Scans and imports ALL task modules even for non-task commands

#### 4. Poor Separation of Concerns
Commands are tightly coupled to:
- Workspace configuration
- Active task/input status display
- Logging systems
- System information gathering

## Refactor Architecture Plan

### Phase 1: Break Circular Dependencies

#### 1.1 Remove autoclean package import from user_config
**Problem**: `user_config.py` imports entire autoclean package just for version info
```python
# Current (BAD)
import autoclean
from autoclean import __version__

# Solution
def get_version():
    """Get version without importing entire package."""
    try:
        from importlib.metadata import version
        return version("autoclean")
    except ImportError:
        return "unknown"
```

#### 1.2 Create lightweight config module
**Problem**: `user_config` is too heavy for simple commands
```python
# New: lightweight_config.py
class LightweightConfig:
    """Minimal config for commands that don't need workspace."""
    @staticmethod
    def get_version():
        # Version detection without heavy imports
    
    @staticmethod  
    def get_basic_paths():
        # Just paths, no validation or heavy operations
```

### Phase 2: Command Categorization

#### 2.1 Lightweight Commands (No workspace needed)
```python
LIGHTWEIGHT_COMMANDS = {
    "help", "version", "auth", "login", "logout", "whoami"
}
```
**Requirements**:
- No user_config import
- No task discovery
- No workspace validation
- Minimal console formatting

#### 2.2 Workspace Commands (Need config, no processing)
```python
WORKSPACE_COMMANDS = {
    "workspace", "config", "input", "source"  
}
```
**Requirements**:
- Import user_config only
- No task discovery
- No processing modules

#### 2.3 Processing Commands (Full imports)
```python
PROCESSING_COMMANDS = {
    "process", "task", "list-tasks", "review", "view"
}
```
**Requirements**:
- Full imports allowed
- Task discovery when needed
- Processing pipeline access

### Phase 3: Lazy Import Strategy

#### 3.1 Conditional Import Router
```python
def main(argv=None):
    """Smart CLI entry point with conditional imports."""
    # Parse command FIRST, import SECOND
    command = _extract_command_fast(argv)
    
    if command in LIGHTWEIGHT_COMMANDS:
        return _handle_lightweight_command(command, argv)
    elif command in WORKSPACE_COMMANDS:
        return _handle_workspace_command(command, argv)
    else:
        return _handle_processing_command(command, argv)

def _handle_lightweight_command(command, argv):
    """Handle commands without any heavy imports."""
    # Only import what's absolutely needed
    pass

def _handle_workspace_command(command, argv):
    """Handle workspace commands with minimal imports."""
    from autoclean.utils.lightweight_config import LightweightConfig
    # Import user_config only here
    pass

def _handle_processing_command(command, argv):
    """Handle processing commands with full imports."""
    # Import everything here
    pass
```

#### 3.2 Defer Heavy Operations
```python
# Instead of module-level task discovery
def get_tasks_when_needed():
    """Only discover tasks when actually needed."""
    # Cache results, only scan when files change
    
# Instead of module-level GPU detection  
def get_system_info_when_requested():
    """Only detect system info when explicitly requested."""
    # Remove torch dependency entirely
```

### Phase 4: Module Restructuring

#### 4.1 Split user_config.py
```
user_config.py (current 1300+ lines)
├── lightweight_config.py (paths, version, basic settings)
├── workspace_config.py (workspace setup, validation) 
├── system_info.py (GPU detection, system resources)
└── user_preferences.py (active task/input, UI preferences)
```

#### 4.2 Create CLI Module Hierarchy
```
cli/
├── __init__.py (minimal entry point)
├── lightweight.py (help, version, auth commands)
├── workspace.py (workspace, config commands)
├── processing.py (process, task commands)
└── shared.py (common utilities)
```

### Phase 5: Task Discovery Optimization

#### 5.1 Lazy Task Discovery
```python
class TaskDiscoveryCache:
    """Cache task discovery results with file watching."""
    
    def __init__(self):
        self._cache = None
        self._cache_time = None
        self._file_mtimes = {}
    
    def get_tasks(self):
        """Get tasks with intelligent caching."""
        if self._needs_refresh():
            self._refresh_cache()
        return self._cache
```

#### 5.2 Targeted Task Loading
```python
def find_single_task(task_name):
    """Find one task without loading all tasks."""
    # Check built-ins first (fast)
    # Check workspace tasks only if needed
    # No full discovery unless absolutely necessary
```

## Implementation Priority

### Immediate Wins (Low Risk, High Impact)
1. **Remove torch import** from user_config.py
2. **Add command categorization** in main() function  
3. **Create lightweight entry points** for help/version commands
4. **Expected improvement**: 80% faster for lightweight commands

### Medium Term (Moderate Risk, High Impact)
1. **Break circular dependency** between user_config and autoclean package
2. **Split user_config.py** into focused modules
3. **Implement lazy task discovery**
4. **Expected improvement**: 60% faster for all commands

### Long Term (High Risk, Highest Impact)  
1. **Restructure CLI module hierarchy**
2. **Implement conditional import router**
3. **Add intelligent caching throughout**
4. **Expected improvement**: 90% faster CLI, sub-second response times

## Performance Targets

### Current State
- Help/Version: 4-5 seconds
- Workspace commands: 4-5 seconds  
- Processing commands: 4-5 seconds + processing time

### Target State  
- Help/Version: <0.1 seconds
- Workspace commands: <0.5 seconds
- Processing commands: <1 second + processing time

## Testing Strategy

### Performance Benchmarks
```python
# Before/after measurements
def benchmark_command(command):
    start = time.time()
    subprocess.run([sys.executable, "-m", "autoclean.cli", command])
    return time.time() - start

# Target commands to benchmark
BENCHMARK_COMMANDS = [
    "help", "version", "workspace", "list-tasks", "process --help"
]
```

### Regression Testing
- Ensure all existing functionality works
- Verify workspace detection still works
- Confirm task discovery accuracy
- Test error handling paths

## Migration Strategy

### Backward Compatibility
- Keep existing CLI interface unchanged
- Maintain all current command syntax
- Preserve configuration file formats
- Support existing custom tasks

### Gradual Rollout
1. **Phase 1**: Internal refactoring only
2. **Phase 2**: Add new lightweight entry points
3. **Phase 3**: Migrate commands gradually  
4. **Phase 4**: Remove legacy code

## Risk Mitigation

### High Risk Areas
- **Task discovery changes**: Could break custom task loading
- **Config file changes**: Could break existing setups
- **Import changes**: Could break plugin systems

### Mitigation Strategies
- Extensive testing with real workspaces
- Maintain backward compatibility layers
- Gradual migration with feature flags
- Clear upgrade documentation

## Success Metrics

### Performance Metrics
- CLI startup time < 0.5 seconds for 90% of commands
- Memory usage reduced by 50%+ for lightweight commands
- Import time reduced by 80%+ for non-processing commands

### User Experience Metrics  
- Faster help/documentation access
- More responsive workspace management
- Improved development workflow (faster testing)

---

**Next Steps**: 
1. Get stakeholder approval for refactor scope
2. Create feature branch for Phase 1 implementation  
3. Set up performance benchmarking infrastructure
4. Begin with immediate wins (torch removal, command categorization)
