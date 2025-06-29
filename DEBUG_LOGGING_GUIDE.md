# How to Enable DEBUG Level Logging in AutoClean

AutoClean provides multiple ways to control logging verbosity. Here's how to enable DEBUG level logging:

## Method 1: Environment Variable (Recommended for Development)

Set the `MNE_LOGGING_LEVEL` environment variable before running AutoClean:

```bash
# Linux/macOS
export MNE_LOGGING_LEVEL=DEBUG
autoclean-eeg process YourTask data.raw

# Windows (PowerShell)
$env:MNE_LOGGING_LEVEL="DEBUG"
autoclean-eeg process YourTask data.raw

# Windows (CMD)
set MNE_LOGGING_LEVEL=DEBUG
autoclean-eeg process YourTask data.raw
```

## Method 2: Python API - Pipeline Initialization

When using the Python API, pass `verbose` parameter to the Pipeline:

```python
from autoclean import Pipeline

# Using string
pipeline = Pipeline(output_dir="results", verbose="debug")

# Using integer (10 = DEBUG level)
pipeline = Pipeline(output_dir="results", verbose=10)

# Using logging constant
import logging
pipeline = Pipeline(output_dir="results", verbose=logging.DEBUG)
```

## Method 3: Command Line (If Implemented)

If the CLI supports verbose flags:

```bash
# Multiple -v flags increase verbosity
autoclean-eeg -vv process YourTask data.raw  # DEBUG level
```

## Method 4: In Your Task Code

You can temporarily set logging level within your task:

```python
from autoclean.utils.logging import configure_logger

class MyTask(Task):
    def run(self):
        # Temporarily enable DEBUG logging
        configure_logger(verbose="debug")
        
        # Your processing steps...
        self.import_raw()
        # etc.
```

## Method 5: Configuration File

Add verbose setting to your task configuration:

```python
config = {
    # ... other config ...
    'logging': {
        'verbose': 'debug'  # or 10, or logging.DEBUG
    }
}
```

## Logging Levels

AutoClean supports these logging levels:

| Level | String | Integer | Description |
|-------|--------|---------|-------------|
| DEBUG | "debug" | 10 | Detailed information for debugging |
| INFO | "info" | 20 | General informational messages |
| WARNING | "warning" | 30 | Warning messages |
| ERROR | "error" | 40 | Error messages |
| CRITICAL | "critical" | 50 | Critical error messages |

## Log Output Locations

When DEBUG logging is enabled, logs are written to:

1. **Console**: Color-coded output to stderr
2. **File**: `{output_dir}/{task}/logs/autoclean_{timestamp}.log`

Example log file location:
```
results/
└── RestingEyesOpen/
    └── logs/
        └── autoclean_2024-06-27.log
```

## What DEBUG Mode Shows

With DEBUG logging enabled, you'll see:

- Detailed processing steps
- MNE library debug output
- Variable values in tracebacks
- Timing information
- Memory usage
- Detailed error diagnostics
- ICA component classification details
- Filter parameters
- Channel rejection reasons

## Example DEBUG Output

```
12:34:56 | DEBUG    | Loading raw data from /path/to/data.raw
12:34:57 | DEBUG    | Raw data shape: (129, 500000)
12:34:57 | DEBUG    | Sampling frequency: 250.0 Hz
12:34:58 | DEBUG    | Applying bandpass filter: 0.1-50.0 Hz
12:34:59 | DEBUG    | Filter parameters: {'l_freq': 0.1, 'h_freq': 50.0, 'method': 'fir'}
12:35:00 | DEBUG    | Running ICA with 25 components
12:35:01 | DEBUG    | ICA convergence: 0.0001 after 123 iterations
```

## Performance Note

DEBUG logging can significantly increase processing time and disk usage due to:
- Extensive console output
- Large log files
- Additional diagnostic calculations

Use DEBUG mode for development and troubleshooting, but switch to INFO for production runs.

## Troubleshooting

If DEBUG logging isn't working:

1. Check environment variable: `echo $MNE_LOGGING_LEVEL`
2. Verify Pipeline initialization: `print(pipeline.verbose)`
3. Check log file permissions in output directory
4. Ensure loguru is installed: `pip install loguru`

## Advanced: Custom Log Filtering

For specific debug output:

```python
from loguru import logger

# Only show DEBUG from specific modules
logger.add(
    sys.stderr,
    filter=lambda record: record["name"].startswith("autoclean.mixins"),
    level="DEBUG"
)
```