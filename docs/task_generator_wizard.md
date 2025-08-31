# Web Task Generator Wizard — Quick Start

Create custom EEG preprocessing tasks for the AutoClean EEG Pipeline in minutes. This wizard outputs a Python task file you can import into your local workspace and run on your data.

## How It Works
- You configure options in the web wizard → it generates a Python task file (for example: `resting_state.py`).
- You install the AutoClean EEG Pipeline locally.
- You import the generated task into your workspace and run it by task name.

## 1) Install the Pipeline
```bash
pip install autoclean-eeg
```

Optional (uv):
```bash
uv pip install autoclean-eeg
```

Verify install:
```bash
autocleaneeg-pipeline -h
```

## 2) Configure Your Workspace (one‑time)
Pick or create a folder where AutoClean stores config, tasks, and outputs.

- Quick default in your Documents folder:
```bash
autocleaneeg-pipeline workspace default
```

- Guided wizard (choose a folder):
```bash
autocleaneeg-pipeline workspace
```

- Open the workspace folder in Finder/Explorer:
```bash
autocleaneeg-pipeline workspace explore
```

## 3) Import Your Generated Task
Assuming you downloaded `resting_state.py` to `~/Downloads`:
```bash
autocleaneeg-pipeline task import ~/Downloads/resting_state.py --name RestingState
```
Notes:
- `--name` controls the filename in your workspace (we append `.py` if omitted).
- If a file already exists, you’ll be prompted to overwrite or rename (use `--force` to skip prompts).

See all tasks (and the exact Task class names) with:
```bash
autocleaneeg-pipeline task list
```

## 4) (Optional) Edit the Task
Open in your shell editor (`$VISUAL`, `$EDITOR`, else nano/vim/notepad):
```bash
autocleaneeg-pipeline task edit RestingState
```
If you choose a built‑in task name instead of a workspace file, the CLI will offer to copy it into your workspace so you can safely customize it.

## 5) Run the Pipeline
Use the Task class name shown in `task list` (for the example above it’s likely `RestingState`).

- Single file:
```bash
autocleaneeg-pipeline process RestingState /path/to/data.raw
```

- Directory of files (example pattern for EEGLAB `.set`):
```bash
autocleaneeg-pipeline process RestingState /path/to/dir --format "*.set" --recursive
```

- Start the review GUI for your workspace output:
```bash
autocleaneeg-pipeline review
```

## Python API (programmatic)
```python
from autoclean.core.pipeline import Pipeline

pipeline = Pipeline(output_dir=None)  # default to workspace/output
pipeline.process_file("/path/to/data.raw", task="RestingState")
# Or for a directory
# pipeline.process_directory(directory="/path/to/dir", task="RestingState", pattern="*.set", recursive=True)
```

## Helpful Workspace Shortcuts
- Print the workspace folder (for cd in your shell):
```bash
autocleaneeg-pipeline workspace cd
```
- Spawn a subshell in the workspace:
```bash
autocleaneeg-pipeline workspace cd --spawn
```
- Show total size of the workspace:
```bash
autocleaneeg-pipeline workspace size
```

## Troubleshooting
- Task not found:
  - Run `autocleaneeg-pipeline task list` and use the Task class name exactly as shown.
  - Re‑import the file: `autocleaneeg-pipeline task import /path/to/task.py --name MyTask`.
- Editor didn’t open:
  - Set `$VISUAL` or `$EDITOR`, or install `nano`/`vim`. On Windows, `notepad` is used.
- Workspace not set:
  - Run `autocleaneeg-pipeline workspace default` or `autocleaneeg-pipeline workspace` to choose a folder.
- Compliance/Part‑11 mode:
  - See `autocleaneeg-pipeline -h auth` and use `autocleaneeg-pipeline auth setup|enable|disable`.

## Learn More
- Docs: https://docs.autocleaneeg.org
- GitHub: https://github.com/cincibrainlab/autoclean_pipeline

