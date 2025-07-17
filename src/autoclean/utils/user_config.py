"""
Simple workspace management for AutoClean.

Handles workspace setup and basic configuration without complex JSON tracking.
Task discovery is done directly from filesystem scanning.
"""

import ast
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import platformdirs

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import autoclean
    from autoclean import __version__
    AUTOCLEAN_AVAILABLE = True
except ImportError:
    AUTOCLEAN_AVAILABLE = False

from autoclean.utils.branding import AutoCleanBranding
from autoclean.utils.logging import message


class UserConfigManager:
    """Simple workspace manager for AutoClean."""

    def __init__(self):
        """Initialize workspace manager."""
        # Get workspace directory (without auto-creating)
        self.config_dir = self._get_workspace_path()
        self.tasks_dir = self.config_dir / "tasks"

        # Only create directories if workspace is valid
        if self._is_workspace_valid():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _get_workspace_path(self) -> Path:
        """Get configured workspace path or default."""
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )

        base_path = None
        if global_config.exists():
            try:
                with open(global_config, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    base_path = Path(config["config_directory"])
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass

        # Default location if no config
        if base_path is None:
            base_path = Path(platformdirs.user_documents_dir()) / "Autoclean-EEG"

        return base_path

    def _is_workspace_valid(self) -> bool:
        """Check if workspace exists and has expected structure."""
        return self.config_dir.exists() and (self.config_dir / "tasks").exists()

    def get_default_output_dir(self) -> Path:
        """Get default output directory."""
        return self.config_dir / "output"

    def list_custom_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List custom tasks by scanning tasks directory."""
        custom_tasks = {}

        if not self.tasks_dir.exists():
            return custom_tasks

        # Scan for Python files
        for task_file in self.tasks_dir.glob("*.py"):
            if task_file.name.startswith("_"):
                continue

            try:
                class_name, description = self._extract_task_info(task_file)

                # Handle duplicates by using newest file
                if class_name in custom_tasks:
                    existing_file = Path(custom_tasks[class_name]["file_path"])
                    if task_file.stat().st_mtime <= existing_file.stat().st_mtime:
                        continue

                custom_tasks[class_name] = {
                    "file_path": str(task_file),
                    "description": description,
                    "class_name": class_name,
                    "modified_time": task_file.stat().st_mtime,
                }

            except Exception as e:
                message("warning", f"Could not parse {task_file.name}: {e}")
                continue

        return custom_tasks

    def get_custom_task_path(self, task_name: str) -> Optional[Path]:
        """Get path to a custom task file."""
        custom_tasks = self.list_custom_tasks()
        if task_name in custom_tasks:
            return Path(custom_tasks[task_name]["file_path"])
        return None

    def _display_system_info(self, console) -> None:
        """Display system information and status."""
        if not RICH_AVAILABLE:
            return

        # Get AutoClean version
        if AUTOCLEAN_AVAILABLE:
            version = __version__
        else:
            version = "unknown"

        # Create system info table
        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column(style="dim")
        info_table.add_column()

        info_table.add_row("Version:", f"AutoClean EEG v{version}")
        info_table.add_row(
            "Python:", f"{sys.version.split()[0]} ({platform.python_implementation()})"
        )
        info_table.add_row("Platform:", f"{platform.system()} {platform.release()}")
        info_table.add_row("Architecture:", platform.machine())

        # System resources (if psutil is available)
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                cpu_count = psutil.cpu_count(logical=True)
                cpu_physical = psutil.cpu_count(logical=False)

                memory_gb = memory.total / (1024**3)
                memory_available_gb = memory.available / (1024**3)
                info_table.add_row(
                    "Memory:",
                    f"{memory_gb:.1f} GB total, {memory_available_gb:.1f} GB available",
                )

                if cpu_physical and cpu_physical != cpu_count:
                    info_table.add_row(
                        "CPU:", f"{cpu_physical} cores ({cpu_count} threads)"
                    )
                else:
                    info_table.add_row("CPU:", f"{cpu_count} cores")
            except Exception:
                info_table.add_row("Memory:", "Unable to detect")
                info_table.add_row("CPU:", "Unable to detect")
        else:
            info_table.add_row("Memory:", "psutil not available")
            info_table.add_row("CPU:", "psutil not available")

        # GPU information
        gpu_info = self._get_gpu_info()
        info_table.add_row("GPU:", gpu_info)

        console.print(info_table)

    def _get_gpu_info(self) -> str:
        """Get GPU information for system display."""
        try:
            # Try to detect NVIDIA GPU first
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_names = result.stdout.strip().split("\n")
                if len(gpu_names) == 1:
                    return f"✓ NVIDIA {gpu_names[0].strip()}"
                else:
                    return f"✓ {len(gpu_names)}× NVIDIA GPUs"
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        try:
            # Try to detect other GPUs via system info
            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "Apple" in result.stdout and (
                    "M1" in result.stdout
                    or "M2" in result.stdout
                    or "M3" in result.stdout
                ):
                    return "✓ Apple Silicon GPU"
                elif "AMD" in result.stdout or "Radeon" in result.stdout:
                    return "✓ AMD GPU detected"
                elif "Intel" in result.stdout:
                    return "✓ Intel GPU detected"
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        try:
            # Try PyTorch GPU detection as fallback
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count == 1:
                    gpu_name = torch.cuda.get_device_name(0)
                    return f"✓ CUDA {gpu_name}"
                else:
                    return f"✓ {gpu_count}× CUDA GPUs"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "✓ Apple Metal GPU"
        except ImportError:
            pass

        return "None detected"

    def setup_workspace(self) -> Path:
        """Smart workspace setup."""
        if RICH_AVAILABLE:
            console = Console()
        else:
            console = None
        workspace_status = self._check_workspace_status()

        if workspace_status == "first_time":
            return self._run_setup_wizard(is_first_time=True)

        elif workspace_status == "missing":
            # Professional header for missing workspace
            AutoCleanBranding.get_professional_header(console)
            console.print(f"\n{AutoCleanBranding.get_simple_divider()}")
            console.print("\n[yellow]⚠[/yellow] [bold]Workspace Missing[/bold]")
            console.print("[dim]Previous workspace location no longer exists[/dim]")
            return self._run_setup_wizard(is_first_time=False)

        elif workspace_status == "valid":
            # Professional header with clear hierarchy
            AutoCleanBranding.get_professional_header(console)

            # Clean section separator
            console.print(f"\n{AutoCleanBranding.get_simple_divider()}")

            # Configuration section without redundant brain icon
            console.print("\n[bold blue]Workspace Configuration[/bold blue]")
            console.print(
                "[green]✓[/green] [dim]Workspace is properly configured[/dim]"
            )

            # System information in organized layout
            console.print("\n[bold]Current Workspace:[/bold]")
            console.print(f"  📁 [dim]{self.config_dir}[/dim]")

            # Compact system info
            console.print("\n[bold]System Information:[/bold]")
            self._display_system_info(console)

            try:
                response = input("\nChange workspace location? (y/N): ").strip().lower()
                if response not in ["y", "yes"]:
                    console.print("[green]✓[/green] Keeping current location")
                    return self.config_dir
            except (EOFError, KeyboardInterrupt):
                console.print("[green]✓[/green] Keeping current location")
                return self.config_dir

            # User wants to change
            new_workspace = self._run_setup_wizard(is_first_time=False)

            # Handle migration if different location
            if new_workspace != self.config_dir:
                self._offer_migration(self.config_dir, new_workspace)

            return new_workspace

        else:
            return self._run_setup_wizard(is_first_time=True)

    def _check_workspace_status(self) -> str:
        """Check workspace status: first_time, missing, or valid."""
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )

        if not global_config.exists():
            return "first_time"

        try:
            with open(global_config, "r", encoding="utf-8") as f:
                json.load(f)  # Just check if valid JSON

            # Check if current workspace is valid
            if self._is_workspace_valid():
                return "valid"
            else:
                return "missing"

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return "first_time"

    def _run_setup_wizard(self, is_first_time: bool = True) -> Path:
        """Run setup wizard."""
        if RICH_AVAILABLE:
            console = Console()
        else:
            console = None

        # Professional header with system info
        if is_first_time:
            # Main header panel with consistent branding
            console.print(AutoCleanBranding.get_welcome_panel(console))

            # Status message
            console.print(
                "\n[bold]System Status:[/bold] [green]✓ Ready for initialization[/green]"
            )

            # System information
            self._display_system_info(console)

            # Welcome message
            console.print(
                "\n[bold]Welcome to AutoClean![/bold] Let's set up your workspace for EEG processing.\n"
                "This workspace will contain your custom tasks, configuration, and results."
            )
        else:
            # Professional header for reconfigure case
            AutoCleanBranding.get_professional_header(console)
            console.print(f"\n{AutoCleanBranding.get_simple_divider()}")
            console.print(
                "\n[blue]⚙️[/blue] [bold blue]Workspace Reconfiguration[/bold blue]"
            )
            console.print("[dim]Setting up new workspace location[/dim]")

        # Get location
        default_dir = Path(platformdirs.user_documents_dir()) / "Autoclean-EEG"
        console.print(f"\n[bold]Workspace location:[/bold] [dim]{default_dir}[/dim]")
        console.print(
            "[dim]• Custom tasks  • Configuration  • Results  • Easy backup[/dim]"
        )

        try:
            response = input(
                "\nPress Enter for default, or type a custom path: "
            ).strip()
            if response:
                chosen_dir = Path(response).expanduser()
                console.print(f"[green]✓[/green] Using: [bold]{chosen_dir}[/bold]")
            else:
                chosen_dir = default_dir
                console.print("[green]✓[/green] Using default location")
        except (EOFError, KeyboardInterrupt):
            chosen_dir = default_dir
            console.print("[yellow]⚠[/yellow] Using default location")

        # Save config and create workspace
        self._save_global_config(chosen_dir)
        self._create_workspace_structure(chosen_dir)

        console.print(f"\n[green]✅ Setup complete![/green] [dim]{chosen_dir}[/dim]")
        self._create_example_script(chosen_dir)

        # Update instance
        self.config_dir = chosen_dir
        self.tasks_dir = chosen_dir / "tasks"

        return chosen_dir

    def _save_global_config(self, workspace_dir: Path) -> None:
        """Save workspace location to global config."""
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )
        global_config.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "config_directory": str(workspace_dir),
            "setup_date": self._current_timestamp(),
            "version": "1.0",
        }

        try:
            with open(global_config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            message("warning", f"Could not save global config: {e}")

    def setup_part11_workspace(self) -> Path:
        """
        Setup Part-11 compliance workspace with -part11 suffix.
        This ensures Part-11 users get an isolated workspace.
        """
        # Determine Part-11 workspace path
        current_workspace = (
            self._get_base_workspace_path()
        )  # Get without Part-11 suffix
        part11_workspace = current_workspace.parent / f"{current_workspace.name}-part11"

        # Check if Part-11 workspace already exists
        if part11_workspace.exists() and (part11_workspace / "tasks").exists():
            message("info", f"Part-11 workspace already exists: {part11_workspace}")
            self._save_global_config(part11_workspace)
            self.config_dir = part11_workspace
            self.tasks_dir = part11_workspace / "tasks"
            return part11_workspace

        # Create Part-11 workspace
        message("info", f"Creating Part-11 compliance workspace: {part11_workspace}")
        self._create_workspace_structure(part11_workspace)
        self._save_global_config(part11_workspace)

        # Update instance
        self.config_dir = part11_workspace
        self.tasks_dir = part11_workspace / "tasks"

        message("info", f"✓ Part-11 workspace created: {part11_workspace}")
        return part11_workspace

    def _get_base_workspace_path(self) -> Path:
        """Get workspace path without Part-11 suffix."""
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )

        if global_config.exists():
            try:
                with open(global_config, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    base_path = Path(config["config_directory"])
                    # Remove -part11 suffix if present
                    if base_path.name.endswith("-part11"):
                        base_path = (
                            base_path.parent / base_path.name[:-7]
                        )  # Remove "-part11"
                    return base_path
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass

        # Default location
        return Path(platformdirs.user_documents_dir()) / "Autoclean-EEG"

    def _create_workspace_structure(self, workspace_dir: Path) -> None:
        """Create workspace directories and copy template files."""
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "tasks").mkdir(exist_ok=True)
        (workspace_dir / "tasks" / "builtin").mkdir(exist_ok=True)
        (workspace_dir / "output").mkdir(exist_ok=True)

        # Copy template task file to tasks directory
        self._create_template_task(workspace_dir / "tasks")
        
        # Copy built-in tasks to builtin examples directory
        self._copy_builtin_tasks(workspace_dir / "tasks" / "builtin")

    def _copy_builtin_tasks(self, builtin_dir: Path) -> None:
        """Copy built-in task files to workspace for easy access and customization."""
        try:
            builtin_dir.mkdir(parents=True, exist_ok=True)
            
            # Get built-in tasks directory from autoclean package
            if AUTOCLEAN_AVAILABLE:
                try:
                    package_dir = Path(autoclean.__file__).parent
                    builtin_tasks_dir = package_dir / "tasks"
                    
                    if not builtin_tasks_dir.exists():
                        message("warning", "Built-in tasks directory not found in package")
                        return
                    
                    copied_count = 0
                    skipped_count = 0
                    
                    # Copy each built-in task file
                    for task_file in builtin_tasks_dir.glob("*.py"):
                        # Skip private files and __init__.py
                        if task_file.name.startswith("_") or task_file.name == "__init__.py":
                            continue
                            
                        dest_file = builtin_dir / task_file.name
                        
                        # Skip if file already exists (avoid overwrites)
                        if dest_file.exists():
                            skipped_count += 1
                            continue
                            
                        # Copy the file with header comment
                        self._copy_builtin_task_with_header(task_file, dest_file)
                        copied_count += 1
                    
                    # Provide feedback about copied tasks
                    if RICH_AVAILABLE:
                        console = Console()
                        if copied_count > 0:
                            console.print(f"[green]📋[/green] Copied {copied_count} built-in task examples to [dim]{builtin_dir}[/dim]")
                        if skipped_count > 0:
                            console.print(f"[yellow]ℹ[/yellow] Skipped {skipped_count} existing built-in task files")
                    
                except Exception as e:
                    message("warning", f"Could not copy built-in tasks: {e}")
            else:
                message("warning", "AutoClean package not available for copying built-in tasks")
                
        except Exception as e:
            message("warning", f"Failed to create built-in tasks directory: {e}")

    def _copy_builtin_task_with_header(self, source_file: Path, dest_file: Path) -> None:
        """Copy a built-in task file with informative header comment."""
        try:
            # Read the source file
            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Create header comment
            header = f"""# =============================================================================
#                          BUILT-IN TASK: {source_file.name}
# =============================================================================
# This is a built-in task from the AutoClean package.
# 
# ✨ CUSTOMIZE THIS FILE:
# - Copy this file to the parent tasks/ directory to customize it
# - Rename the file and class to match your experiment
# - Modify the configuration and run() method as needed
# - The original built-in task remains unchanged in the package
#
# 📖 USAGE:
# - This file serves as a reference and starting point
# - Built-in tasks are automatically available via Pipeline.process_file()
# - Custom tasks override built-in tasks when placed in tasks/ directory
#
# 🔄 UPDATES:
# - This file may be updated when AutoClean is upgraded
# - Your custom tasks in tasks/ directory are never overwritten
# =============================================================================

"""
            
            # Write the file with header
            with open(dest_file, "w", encoding="utf-8") as f:
                f.write(header + content)
                
        except Exception as e:
            # Fallback to simple copy if header addition fails
            shutil.copy2(source_file, dest_file)
            message("warning", f"Could not add header to {dest_file.name}: {e}")

    def _offer_migration(self, old_dir: Path, new_dir: Path) -> None:
        """Offer to migrate workspace."""
        if RICH_AVAILABLE:
            console = Console()
        else:
            console = None

        try:
            response = input("\nMigrate existing tasks? (y/N): ").strip().lower()
            if response in ["yes", "y"] and old_dir.exists():
                shutil.copytree(
                    old_dir / "tasks", new_dir / "tasks", dirs_exist_ok=True
                )
                console.print("[green]✓[/green] Tasks migrated")
            else:
                console.print("[green]✓[/green] Starting fresh")
        except (EOFError, KeyboardInterrupt):
            console.print("[yellow]⚠[/yellow] Starting fresh")

        # Update instance
        self.config_dir = new_dir
        self.tasks_dir = new_dir / "tasks"

    def _create_example_script(self, workspace_dir: Path) -> None:
        """Create example script in workspace."""
        try:
            dest_file = workspace_dir / "example_basic_usage.py"

            # Try to copy from package
            if AUTOCLEAN_AVAILABLE:
                try:
                    package_dir = Path(autoclean.__file__).parent.parent.parent
                    source_file = package_dir / "examples" / "basic_usage.py"

                    if source_file.exists():
                        shutil.copy2(source_file, dest_file)
                    else:
                        self._create_fallback_example(dest_file)
                except Exception:
                    self._create_fallback_example(dest_file)
            else:
                self._create_fallback_example(dest_file)

            if RICH_AVAILABLE:
                console = Console()
                console.print(f"[green]📄[/green] Example script: [dim]{dest_file}[/dim]")

        except Exception as e:
            message("warning", f"Could not create example script: {e}")

    def _create_fallback_example(self, dest_file: Path) -> None:
        """Create fallback example script."""
        content = """import asyncio
from pathlib import Path

from autoclean import Pipeline

# Example usage of AutoClean Pipeline
def main():
    # Create pipeline (uses your workspace output by default)
    pipeline = Pipeline()
    
    # Process a single file
    pipeline.process_file("path/to/your/data.raw", "RestingEyesOpen")
    
    # Process multiple files
    asyncio.run(pipeline.process_directory_async(
        directory_path="path/to/your/data/",
        task="RestingEyesOpen",
        pattern="*.raw"
    ))

if __name__ == "__main__":
    main()
"""

        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(content)

    def _create_template_task(self, tasks_dir: Path) -> None:
        """Create template task file in tasks directory."""
        try:
            dest_file = tasks_dir / "custom_task_template.py"

            # Try to copy from package templates
            if AUTOCLEAN_AVAILABLE:
                try:
                    package_dir = Path(autoclean.__file__).parent
                    source_file = package_dir / "templates" / "custom_task_template.py"

                    if source_file.exists():
                        shutil.copy2(source_file, dest_file)
                    else:
                        self._create_fallback_template(dest_file)
                except Exception:
                    self._create_fallback_template(dest_file)
            else:
                self._create_fallback_template(dest_file)

            if RICH_AVAILABLE:
                console = Console()
                console.print(f"[green]📋[/green] Template task: [dim]{dest_file}[/dim]")

        except Exception as e:
            message("warning", f"Could not create template task: {e}")

    def _create_fallback_template(self, dest_file: Path) -> None:
        """Create fallback template task file."""
        content = '''from autoclean.core.task import Task

# =============================================================================
#                     CUSTOM EEG PREPROCESSING TASK TEMPLATE
# =============================================================================
# This is a template for creating custom EEG preprocessing tasks.
# Customize the configuration below to match your specific EEG paradigm.
# 
# Instructions:
# 1. Rename this file to match your task (e.g., my_experiment.py)
# 2. Update the class name below (e.g., MyExperiment)
# 3. Modify the config dictionary to match your data requirements
# 4. Customize the run() method to define your processing pipeline
#
# 🟢 enabled: True  = Apply this processing step
# 🔴 enabled: False = Skip this processing step
#
# 💡 TIP: Use the AutoClean configuration wizard to generate settings
#         automatically, or copy settings from existing tasks!
# =============================================================================

config = {
    'resample_step': {
        'enabled': True,
        'value': 250  # Resample to 250 Hz
    },
    'filtering': {
        'enabled': True,
        'value': {
            'l_freq': 1,      # High-pass filter (Hz)
            'h_freq': 100,    # Low-pass filter (Hz)
            'notch_freqs': [60, 120],  # Notch filter frequencies
            'notch_widths': 5          # Notch filter width
        }
    },
    'drop_outerlayer': {
        'enabled': False,
        'value': []  # Channel indices to drop
    },
    'eog_step': {
        'enabled': False,
        'value': []  # EOG channel indices
    },
    'trim_step': {
        'enabled': True,
        'value': 4  # Trim seconds from start/end
    },
    'crop_step': {
        'enabled': False,
        'value': {
            'start': 0,   # Start time (seconds)
            'end': 60     # End time (seconds)
        }
    },
    'reference_step': {
        'enabled': True,
        'value': 'average'  # Reference type
    },
    'montage': {
        'enabled': True,
        'value': 'GSN-HydroCel-129'  # EEG montage
    },
    'ICA': {
        'enabled': True,
        'value': {
            'method': 'fastica',
            'n_components': None,
            'fit_params': {}
        }
    },
    'component_rejection': {
        'enabled': True,
        'method': 'iclabel',
        'value': {
            'ic_flags_to_reject': ['muscle', 'heart', 'eog', 'ch_noise', 'line_noise'],
            'ic_rejection_threshold': 0.3
        }
    },
    'epoch_settings': {
        'enabled': True,
        'value': {
            'tmin': -1,  # Epoch start (seconds)
            'tmax': 1    # Epoch end (seconds)
        },
        'event_id': None,
        'remove_baseline': {
            'enabled': False,
            'window': [None, 0]
        },
        'threshold_rejection': {
            'enabled': False,
            'volt_threshold': {
                'eeg': 0.000125
            }
        }
    }
}


class CustomTask(Task):
    """
    Custom EEG preprocessing task template.
    
    Modify this class to create your own EEG preprocessing pipeline.
    """

    def run(self) -> None:
        """Define your custom EEG preprocessing pipeline."""
        # Import raw EEG data
        self.import_raw()

        # Basic preprocessing steps
        self.resample_data()
        self.filter_data()
        self.drop_outer_layer()
        self.assign_eog_channels()
        self.trim_edges()
        self.crop_duration()

        # Store original data for comparison
        self.original_raw = self.raw.copy()
        
        # Create BIDS-compliant paths and filenames
        self.create_bids_path()
        
        # Channel cleaning
        self.clean_bad_channels()
        
        # Re-referencing
        self.rereference_data()
        
        # Artifact detection
        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()
        
        # ICA processing
        self.run_ica()
        self.classify_ica_components()
        
        # Epoching
        self.create_regular_epochs()
        
        # Outlier detection
        self.detect_outlier_epochs()
        
        # Clean epochs
        self.gfp_clean_epochs()

        # Generate reports
        self.generate_reports()

    def generate_reports(self) -> None:
        """Generate quality control visualizations and reports."""
        if self.raw is None or self.original_raw is None:
            return
            
        # Plot raw vs cleaned overlay
        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        
        # Plot power spectral density topography
        self.step_psd_topo_figure(self.original_raw, self.raw)
'''

        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(content)

    def _extract_task_info(self, task_file: Path) -> tuple[str, str]:
        """Extract class name and description from task file."""
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if (isinstance(base, ast.Name) and base.id == "Task") or (
                            isinstance(base, ast.Attribute) and base.attr == "Task"
                        ):
                            class_name = node.name
                            description = ast.get_docstring(node)
                            if description:
                                description = description.split("\n")[0]
                            else:
                                description = f"Custom task: {class_name}"
                            return class_name, description

            return task_file.stem, f"Custom task from {task_file.name}"

        except Exception:
            return task_file.stem, f"Custom task from {task_file.name}"

    def _current_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()


# Global instance
user_config = UserConfigManager()
