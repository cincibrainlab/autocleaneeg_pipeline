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

# Simple branding constants
PRODUCT_NAME = "AutoClean EEG"
TAGLINE = "Professional EEG Processing & Analysis Platform"
LOGO_ICON = "🧠"
DIVIDER = "═════════════════════════════════════════════════"


class UserConfigManager:
    """Simple workspace manager for AutoClean."""

    def __init__(self):
        """Initialize workspace manager."""
        # Get workspace directory (without auto-creating)
        self.config_dir = self._get_workspace_path()
        self.tasks_dir = self.config_dir / "tasks"

        # Always create basic directory structure to avoid confusion
        # This eliminates "workspace not configured" messages on fresh installs
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
        """Check if workspace is properly configured.
        
        Returns True if the workspace has been set up through the configuration wizard
        (indicated by setup.json existing) and has the expected directory structure.
        """
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )
        return (
            global_config.exists() and
            self.config_dir.exists() and 
            (self.config_dir / "tasks").exists()
        )

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
                # Use print for internal warnings since this is a utility function
                # not part of the main CLI setup experience
                print(f"Warning: Could not parse {task_file.name}: {e}")
                continue

        return custom_tasks

    def get_custom_task_path(self, task_name: str) -> Optional[Path]:
        """Get path to a custom task file."""
        custom_tasks = self.list_custom_tasks()
        if task_name in custom_tasks:
            return Path(custom_tasks[task_name]["file_path"])
        return None

    def get_active_task(self) -> Optional[str]:
        """Get the currently active task name."""
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )
        
        if not global_config.exists():
            return None
            
        try:
            with open(global_config, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("active_task")
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return None

    def get_active_source(self) -> Optional[str]:
        """Get the currently active source path."""
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )
        
        if not global_config.exists():
            return None
            
        try:
            with open(global_config, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("active_source")
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return None

    def set_active_task(self, task_name: Optional[str]) -> bool:
        """Set the active task name. Use None to unset."""
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )
        
        # Load existing config or create new one
        config = {
            "version": "1.0",
            "setup_date": self._current_timestamp(),
        }
        
        if global_config.exists():
            try:
                with open(global_config, "r", encoding="utf-8") as f:
                    config.update(json.load(f))
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Update active task
        if task_name is None:
            config.pop("active_task", None)
        else:
            config["active_task"] = task_name
        
        # Save config
        global_config.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(global_config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Warning: Could not save active task config: {e}")
            return False

    def set_active_source(self, source_path: Optional[str]) -> bool:
        """Set the active source path. Use None to unset."""
        global_config = (
            Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        )
        
        # Load existing config or create new one
        config = {
            "version": "1.0",
            "setup_date": self._current_timestamp(),
        }
        
        if global_config.exists():
            try:
                with open(global_config, "r", encoding="utf-8") as f:
                    config.update(json.load(f))
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Update active source
        if source_path is None:
            config.pop("active_source", None)
        else:
            config["active_source"] = str(source_path)
        
        # Save config
        global_config.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(global_config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Warning: Could not save active source config: {e}")
            return False

    def select_active_task_interactive(self) -> Optional[str]:
        """Interactive selection of active task from available custom tasks."""
        custom_tasks = self.list_custom_tasks()
        
        if not custom_tasks:
            print("No custom tasks found in workspace.")
            print("Add tasks to your workspace with: autocleaneeg-pipeline task add <file>")
            return None
        
        # Try to use rich for better display, fallback to basic if not available
        if RICH_AVAILABLE:
            return self._select_task_rich(custom_tasks)
        else:
            return self._select_task_basic(custom_tasks)
    
    def _select_task_rich(self, custom_tasks: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Rich-based interactive task selection."""
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.table import Table
        
        console = Console()
        
        console.print("\n[bold]Available Custom Tasks:[/bold]")
        
        # Create table of tasks
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Task Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Modified", style="dim")
        
        task_list = list(custom_tasks.items())
        for i, (task_name, task_info) in enumerate(task_list, 1):
            description = task_info.get("description", "No description")
            # Format modified time
            try:
                from datetime import datetime
                mod_time = datetime.fromtimestamp(task_info["modified_time"])
                mod_str = mod_time.strftime("%Y-%m-%d %H:%M")
            except (KeyError, ValueError, OSError):
                mod_str = "Unknown"
            
            table.add_row(str(i), task_name, description, mod_str)
        
        console.print(table)
        
        # Get current active task to show in prompt
        current_active = self.get_active_task()
        current_msg = f" (current: {current_active})" if current_active else ""
        
        console.print(f"\nSelect a task by number{current_msg}, or press Enter to cancel:")
        
        try:
            choice = Prompt.ask("Choice", default="", show_default=False)
            if not choice.strip():
                return None
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(task_list):
                selected_task = task_list[choice_num - 1][0]
                return selected_task
            else:
                console.print("[red]Invalid selection.[/red]")
                return None
        except (ValueError, KeyboardInterrupt):
            return None
    
    def _select_task_basic(self, custom_tasks: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Basic console-based interactive task selection."""
        print("\nAvailable Custom Tasks:")
        
        task_list = list(custom_tasks.items())
        for i, (task_name, task_info) in enumerate(task_list, 1):
            description = task_info.get("description", "No description")
            print(f"  {i}. {task_name} - {description}")
        
        current_active = self.get_active_task()
        current_msg = f" (current: {current_active})" if current_active else ""
        
        try:
            choice = input(f"\nSelect a task by number{current_msg}, or press Enter to cancel: ").strip()
            if not choice:
                return None
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(task_list):
                selected_task = task_list[choice_num - 1][0]
                return selected_task
            else:
                print("Invalid selection.")
                return None
        except (ValueError, KeyboardInterrupt):
            return None

    def select_active_source_interactive(self) -> Optional[str]:
        """Interactive selection of active source path."""
        # Try to use rich for better display, fallback to basic if not available
        if RICH_AVAILABLE:
            return self._select_source_rich()
        else:
            return self._select_source_basic()

    @staticmethod
    def _strip_wrapping_quotes(text: Optional[str]) -> Optional[str]:
        """Remove matching wrapping quotes (single or double), up to two layers.

        Handles common paste cases like "'/path with spaces'" and '\"/path\"'.
        Returns input unchanged if not quoted or None.
        """
        if text is None:
            return None
        s = text.strip()
        for _ in range(2):
            if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                s = s[1:-1].strip()
            else:
                break
        return s

    def _select_source_rich(self) -> Optional[str]:
        """Rich-based interactive source selection."""
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.table import Table
        
        console = Console()
        
        console.print("\n[bold]Select Active Source:[/bold]")
        console.print("[muted]Choose how to handle input files for processing[/muted]\n")
        
        # Create options table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="green")
        
        table.add_row("1", "Select File", "Choose a specific EEG file")
        table.add_row("2", "Select Directory", "Choose a directory containing EEG files")
        table.add_row("3", "No Default", "Always prompt for input (current behavior)")
        
        current_source = self.get_active_source()
        if current_source:
            table.add_row("4", "Keep Current", f"Keep: {current_source}")
        
        console.print(table)
        
        try:
            max_choice = 4 if current_source else 3
            choice = Prompt.ask("Choice", default="", show_default=False)
            if not choice.strip():
                return None
                
            choice_num = int(choice)
            
            if choice_num == 1:
                # File selection
                from rich.prompt import Prompt
                file_path = Prompt.ask("Enter file path (quotes OK)")
                file_path = self._strip_wrapping_quotes(file_path)
                if file_path and Path(file_path).exists():
                    return str(Path(file_path).resolve())
                else:
                    console.print("[red]File not found[/red]")
                    return None
                    
            elif choice_num == 2:
                # Directory selection
                from rich.prompt import Prompt
                dir_path = Prompt.ask("Enter directory path (quotes OK)") 
                dir_path = self._strip_wrapping_quotes(dir_path)
                if dir_path and Path(dir_path).exists() and Path(dir_path).is_dir():
                    return str(Path(dir_path).resolve())
                else:
                    console.print("[red]Directory not found[/red]")
                    return None
                    
            elif choice_num == 3:
                # No default
                return "NONE"
                
            elif choice_num == 4 and current_source:
                # Keep current
                return current_source
                
            else:
                console.print("[red]Invalid selection.[/red]")
                return None
                
        except (ValueError, KeyboardInterrupt):
            return None
    
    def _select_source_basic(self) -> Optional[str]:
        """Basic console-based interactive source selection."""
        print("\nSelect Active Source:")
        print("Choose how to handle input files for processing\n")
        
        print("  1. Select File - Choose a specific EEG file")
        print("  2. Select Directory - Choose a directory containing EEG files") 
        print("  3. No Default - Always prompt for input (current behavior)")
        
        current_source = self.get_active_source()
        if current_source:
            print(f"  4. Keep Current - Keep: {current_source}")
        
        try:
            max_choice = 4 if current_source else 3
            choice = input("\nSelect option (1-{}), or press Enter to cancel: ".format(max_choice)).strip()
            if not choice:
                return None
                
            choice_num = int(choice)
            
            if choice_num == 1:
                # File selection
                file_path = input("Enter file path (quotes OK): ").strip()
                file_path = self._strip_wrapping_quotes(file_path)
                if file_path and Path(file_path).exists():
                    return str(Path(file_path).resolve())
                else:
                    print("File not found")
                    return None
                    
            elif choice_num == 2:
                # Directory selection
                dir_path = input("Enter directory path (quotes OK): ").strip()
                dir_path = self._strip_wrapping_quotes(dir_path)
                if dir_path and Path(dir_path).exists() and Path(dir_path).is_dir():
                    return str(Path(dir_path).resolve())
                else:
                    print("Directory not found")
                    return None
                    
            elif choice_num == 3:
                # No default
                return "NONE"
                
            elif choice_num == 4 and current_source:
                # Keep current  
                return current_source
                
            else:
                print("Invalid selection.")
                return None
                
        except (ValueError, KeyboardInterrupt):
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

    def _get_system_info_dict(self) -> Dict[str, str]:
        """Get system information as dictionary for table display."""
        # Get AutoClean version
        if AUTOCLEAN_AVAILABLE:
            version = __version__
        else:
            version = "unknown"

        info = {
            "Version": f"AutoClean EEG v{version}",
            "Python": f"{sys.version.split()[0]} ({platform.python_implementation()})",
            "Platform": f"{platform.system()} {platform.release()}",
            "Architecture": platform.machine(),
        }

        # System resources (if psutil is available)
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                cpu_count = psutil.cpu_count(logical=True)
                cpu_physical = psutil.cpu_count(logical=False)

                memory_gb = memory.total / (1024**3)
                memory_available_gb = memory.available / (1024**3)
                info["Memory"] = (
                    f"{memory_gb:.1f} GB total, {memory_available_gb:.1f} GB available"
                )

                if cpu_physical and cpu_physical != cpu_count:
                    info["CPU"] = f"{cpu_physical} cores ({cpu_count} threads)"
                else:
                    info["CPU"] = f"{cpu_count} cores"
            except Exception:
                info["Memory"] = "Unable to detect"
                info["CPU"] = "Unable to detect"
        else:
            info["Memory"] = "psutil not available"
            info["CPU"] = "psutil not available"

        # GPU information
        info["GPU"] = self._get_gpu_info()

        return info

    def setup_workspace(self, show_branding: bool = True) -> Path:
        """Smart workspace setup."""
        from autoclean.utils.cli_display import setup_display

        workspace_status = self._check_workspace_status()

        if workspace_status == "first_time":
            return self._run_setup_wizard(
                is_first_time=True, show_branding=show_branding
            )

        elif workspace_status == "missing":
            if show_branding:
                setup_display.console.print(
                    f"[bold green]{LOGO_ICON} {PRODUCT_NAME}[/bold green]"
                )
                setup_display.console.print(f"[muted]{DIVIDER}[/muted]")
                setup_display.blank_line()
            setup_display.warning(
                "Workspace Missing", "Previous workspace location no longer exists"
            )
            return self._run_setup_wizard(
                is_first_time=False, show_branding=show_branding
            )

        elif workspace_status == "valid":
            # Display workspace status cleanly with boxed header (only if showing branding)
            if show_branding:
                setup_display.boxed_header(
                    f"{LOGO_ICON} Welcome to AutoClean",
                    TAGLINE,
                    title="[bold green]✓ Workspace Ready[/bold green]",
                )
                setup_display.blank_line()
            else:
                # Just show the workspace info without branding
                setup_display.console.print("[header]Workspace Configuration[/header]")

            # Workspace information (no duplicate header)
            setup_display.workspace_info(self.config_dir, is_valid=True)

            # Suppress system information to keep directory selection focused

            # Prompt for changes
            try:
                change_location = setup_display.prompt_yes_no(
                    "Change workspace location?", default=False
                )
                if not change_location:
                    setup_display.success("Keeping current location")
                    return self.config_dir
            except (EOFError, KeyboardInterrupt):
                setup_display.success("Keeping current location")
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

    def _run_setup_wizard(
        self, is_first_time: bool = True, show_branding: bool = True
    ) -> Path:
        """Run setup wizard."""
        from autoclean.utils.cli_display import setup_display

        # Display header based on setup type (only if showing branding)
        if show_branding:
            setup_display.welcome_header(is_first_time)

        if is_first_time:
            # Keep first-time flow minimalist: no system information section
            pass

        # Get workspace location
        default_dir = Path(platformdirs.user_documents_dir()) / "Autoclean-EEG"
        chosen_dir = setup_display.workspace_location_prompt(default_dir, current_dir=self.config_dir)

        # Save config and create workspace
        self._save_global_config(chosen_dir)
        self._create_workspace_structure(chosen_dir)

        # Setup completion
        setup_display.setup_complete_summary(chosen_dir)
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

        # Load existing config to preserve active_task and other settings
        existing_config = {}
        if global_config.exists():
            try:
                with open(global_config, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        config = {
            "config_directory": str(workspace_dir),
            "setup_date": self._current_timestamp(),
            "version": "1.0",
        }
        
        # Preserve active_task if it exists
        if "active_task" in existing_config:
            config["active_task"] = existing_config["active_task"]

        try:
            with open(global_config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save global config: {e}")

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
            print(f"Part-11 workspace already exists: {part11_workspace}")
            self._save_global_config(part11_workspace)
            self.config_dir = part11_workspace
            self.tasks_dir = part11_workspace / "tasks"
            return part11_workspace

        # Create Part-11 workspace
        print(f"Creating Part-11 compliance workspace: {part11_workspace}")
        self._create_workspace_structure(part11_workspace)
        self._save_global_config(part11_workspace)

        # Update instance
        self.config_dir = part11_workspace
        self.tasks_dir = part11_workspace / "tasks"

        print(f"✓ Part-11 workspace created: {part11_workspace}")
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
                        print("Warning: Built-in tasks directory not found in package")
                        return

                    copied_count = 0
                    skipped_count = 0

                    # Copy each built-in task file
                    for task_file in builtin_tasks_dir.glob("*.py"):
                        # Skip private files and __init__.py
                        if (
                            task_file.name.startswith("_")
                            or task_file.name == "__init__.py"
                        ):
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
                        from autoclean.utils.cli_display import setup_display

                        if copied_count > 0:
                            setup_display.success(
                                f"Copied {copied_count} built-in task examples",
                                str(builtin_dir),
                            )
                        if skipped_count > 0:
                            setup_display.info(
                                f"Skipped {skipped_count} existing built-in task files"
                            )

                except Exception as e:
                    print(f"Warning: Could not copy built-in tasks: {e}")
            else:
                print(
                    "Warning: AutoClean package not available for copying built-in tasks"
                )

        except Exception as e:
            print(f"Warning: Failed to create built-in tasks directory: {e}")

    def _copy_builtin_task_with_header(
        self, source_file: Path, dest_file: Path
    ) -> None:
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
# 🔄 TASK OVERRIDE BEHAVIOR:
# - Tasks in workspace/tasks/ automatically override built-in tasks with same name
# - Move/copy this file to tasks/ directory to activate override
# - Your workspace task will take precedence over the package version
# - Use 'autocleaneeg-pipeline list-tasks --overrides' to see active overrides
#
# 📖 USAGE:
# - This file serves as a reference and starting point
# - Built-in tasks remain available until overridden
# - Workspace tasks are never overwritten during upgrades
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
            print(f"Warning: Could not add header to {dest_file.name}: {e}")

    def _offer_migration(self, old_dir: Path, new_dir: Path) -> None:
        """Offer to migrate workspace."""
        from autoclean.utils.cli_display import setup_display

        migrate = setup_display.migration_prompt(old_dir, new_dir)

        if migrate and old_dir.exists():
            try:
                shutil.copytree(
                    old_dir / "tasks", new_dir / "tasks", dirs_exist_ok=True
                )
                setup_display.success("Tasks migrated successfully")
            except Exception as e:
                setup_display.error("Migration failed", str(e))
                setup_display.info("Starting with fresh workspace")
        else:
            setup_display.success("Starting with fresh workspace")

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
                from autoclean.utils.cli_display import setup_display

                setup_display.success("Example script created", str(dest_file))

        except Exception as e:
            print(f"Warning: Could not create example script: {e}")

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
                from autoclean.utils.cli_display import setup_display

                setup_display.success("Template task created", str(dest_file))

        except Exception as e:
            print(f"Warning: Could not create template task: {e}")

    def _create_fallback_template(self, dest_file: Path) -> None:
        """Create fallback template task file."""
        content = '''from autoclean.core.task import Task
from autoclean.task_config_schema import CONFIG_VERSION

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
    'version': CONFIG_VERSION,
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
