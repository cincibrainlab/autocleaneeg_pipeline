# AutoClean EEG CLI Reference

Quick reference for all AutoClean EEG command line interface commands.

## Getting Started

```bash
# Show main interface (no arguments)
autocleaneeg-pipeline

# Show version
autocleaneeg-pipeline version

# Run tutorial
autocleaneeg-pipeline tutorial
```

## Workspace Management

```bash
# Setup workspace (first time)
autocleaneeg-pipeline workspace

# Workspace commands
autocleaneeg-pipeline workspace explore    # Open in file manager
autocleaneeg-pipeline workspace size       # Show size
autocleaneeg-pipeline workspace set PATH   # Change location  
autocleaneeg-pipeline workspace default    # Reset to default
autocleaneeg-pipeline workspace cd         # Navigate to workspace
```

## Processing Data

```bash
# Basic processing
autocleaneeg-pipeline process TASK FILE
autocleaneeg-pipeline process RestingEyesOpen data.raw

# Directory processing
autocleaneeg-pipeline process TASK --dir PATH
autocleaneeg-pipeline process TASK --dir PATH --format "*.raw"
autocleaneeg-pipeline process TASK --dir PATH --recursive

# Advanced options
autocleaneeg-pipeline process TASK FILE --output OUTPUT_DIR
autocleaneeg-pipeline process TASK FILE --parallel 4
autocleaneeg-pipeline process TASK FILE --dry-run
autocleaneeg-pipeline process TASK FILE --verbose
```

## Task Management

```bash
# List tasks
autocleaneeg-pipeline list-tasks
autocleaneeg-pipeline task list
autocleaneeg-pipeline list-tasks --verbose

# Add/manage custom tasks  
autocleaneeg-pipeline task add FILE.py
autocleaneeg-pipeline task add FILE.py --name TASKNAME
autocleaneeg-pipeline task remove TASKNAME
autocleaneeg-pipeline task delete TASKNAME

# Edit/copy tasks
autocleaneeg-pipeline task edit TASKNAME
autocleaneeg-pipeline task edit BUILTIN_TASK --name NEW_NAME
autocleaneeg-pipeline task copy SOURCE --name DEST_NAME
autocleaneeg-pipeline task import FILE.py --name TASKNAME

# Explore tasks folder
autocleaneeg-pipeline task explore
```

## Configuration

```bash
# Show config location
autocleaneeg-pipeline config show

# Configuration management
autocleaneeg-pipeline config setup         # Reconfigure workspace
autocleaneeg-pipeline config reset --confirm
autocleaneeg-pipeline config export PATH
autocleaneeg-pipeline config import PATH
```

## Results & Review

```bash
# Review processed data
autocleaneeg-pipeline review
autocleaneeg-pipeline review --output PATH

# View EEG files
autocleaneeg-pipeline view FILE
autocleaneeg-pipeline view FILE --no-view  # Validate only
```

## Maintenance

```bash
# Clean task outputs
autocleaneeg-pipeline clean-task TASKNAME
autocleaneeg-pipeline clean-task TASKNAME --dry-run
autocleaneeg-pipeline clean-task TASKNAME --force
```

## Compliance & Audit

```bash
# Export audit logs
autocleaneeg-pipeline export-access-log --output FILE
autocleaneeg-pipeline export-access-log --format csv --output FILE
autocleaneeg-pipeline export-access-log --start-date YYYY-MM-DD --end-date YYYY-MM-DD
autocleaneeg-pipeline export-access-log --verify-only
```

## Authentication (Advanced)

```bash
# User authentication
autocleaneeg-pipeline login
autocleaneeg-pipeline logout  
autocleaneeg-pipeline whoami

# Diagnostics
autocleaneeg-pipeline auth0-diagnostics --verbose
```

## Help & Information

```bash
# Get help
autocleaneeg-pipeline help
autocleaneeg-pipeline COMMAND --help

# Show version info
autocleaneeg-pipeline version

# Interactive tutorial
autocleaneeg-pipeline tutorial
```

## Common Workflows

### First Time Setup
```bash
autocleaneeg-pipeline                 # Check installation
autocleaneeg-pipeline workspace      # Setup workspace
autocleaneeg-pipeline list-tasks     # See available tasks
```

### Single File Processing
```bash
cd /path/to/data
autocleaneeg-pipeline process RestingEyesOpen subject01.raw
autocleaneeg-pipeline review
```

### Batch Processing
```bash
autocleaneeg-pipeline process RestingEyesOpen --dir /data --format "*.raw" --dry-run
autocleaneeg-pipeline process RestingEyesOpen --dir /data --format "*.raw" --parallel 4
```

### Custom Task Development  
```bash
autocleaneeg-pipeline task copy RestingEyesOpen --name MyCustomTask
autocleaneeg-pipeline task edit MyCustomTask
autocleaneeg-pipeline task list
```

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Command not found | `pip install autocleaneeg-pipeline` |
| Workspace not configured | `autocleaneeg-pipeline workspace` |
| Task not found | `autocleaneeg-pipeline list-tasks` |
| File not found | Check path with `ls` (Mac/Linux) or `dir` (Windows) |
| Processing stuck | Wait or use Ctrl+C to cancel |

## File Formats Supported

- `.raw` (MNE-Python raw format)
- `.set` (EEGLAB format) 
- `.edf` (European Data Format)
- `.bdf` (BioSemi Data Format)
- And more via MNE-Python plugins

## Common Task Names

- `RestingEyesOpen` - Resting state, eyes open
- `RestingEyesClosed` - Resting state, eyes closed
- `MMN` - Mismatch negativity
- `ASSR` - Auditory steady-state response  
- `Chirp` - Chirp stimulus paradigm