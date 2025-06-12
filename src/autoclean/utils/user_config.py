"""
Simple workspace management for AutoClean.

Handles workspace setup and basic configuration without complex JSON tracking.
Task discovery is done directly from filesystem scanning.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
import platformdirs

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
        global_config = Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        
        if global_config.exists():
            try:
                with open(global_config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return Path(config["config_directory"])
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass
        
        # Default location
        return Path(platformdirs.user_documents_dir()) / "Autoclean-EEG"
    
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
                    "modified_time": task_file.stat().st_mtime
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
    
    def setup_workspace(self) -> Path:
        """Smart workspace setup."""
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        workspace_status = self._check_workspace_status()
        
        if workspace_status == "first_time":
            return self._run_setup_wizard(is_first_time=True)
            
        elif workspace_status == "missing":
            console.print("[yellow]⚠[/yellow] Previous workspace no longer exists")
            return self._run_setup_wizard(is_first_time=False)
            
        elif workspace_status == "valid":
            console.print(Panel("🔧 [bold blue]Workspace Configuration[/bold blue]", style="blue"))
            console.print(f"\n[bold]Current workspace:[/bold] [dim]{self.config_dir}[/dim]")
            
            try:
                response = input("\nChange workspace location? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
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
        global_config = Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        
        if not global_config.exists():
            return "first_time"
            
        try:
            with open(global_config, 'r', encoding='utf-8') as f:
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
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        # Header
        if is_first_time:
            console.print(Panel("🧠 [bold green]Welcome to AutoClean![/bold green]", style="green"))
        else:
            console.print(Panel("🔧 [bold blue]Workspace Setup[/bold blue]", style="blue"))
        
        # Get location
        default_dir = Path(platformdirs.user_documents_dir()) / "Autoclean-EEG"
        console.print(f"\n[bold]Workspace location:[/bold] [dim]{default_dir}[/dim]")
        console.print("[dim]• Custom tasks  • Configuration  • Results  • Easy backup[/dim]")
        
        try:
            response = input(f"\nPress Enter for default, or type a custom path: ").strip()
            if response:
                chosen_dir = Path(response).expanduser()
                console.print(f"[green]✓[/green] Using: [bold]{chosen_dir}[/bold]")
            else:
                chosen_dir = default_dir
                console.print(f"[green]✓[/green] Using default location")
        except (EOFError, KeyboardInterrupt):
            chosen_dir = default_dir
            console.print(f"[yellow]⚠[/yellow] Using default location")
        
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
        global_config = Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        global_config.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "config_directory": str(workspace_dir),
            "setup_date": self._current_timestamp(),
            "version": "1.0"
        }
        
        try:
            with open(global_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            message("warning", f"Could not save global config: {e}")
    
    def _create_workspace_structure(self, workspace_dir: Path) -> None:
        """Create workspace directories."""
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "tasks").mkdir(exist_ok=True)
        (workspace_dir / "output").mkdir(exist_ok=True)
    
    def _offer_migration(self, old_dir: Path, new_dir: Path) -> None:
        """Offer to migrate workspace."""
        from rich.console import Console
        console = Console()
        
        try:
            response = input(f"\nMigrate existing tasks? (y/N): ").strip().lower()
            if response in ['yes', 'y'] and old_dir.exists():
                shutil.copytree(old_dir / "tasks", new_dir / "tasks", dirs_exist_ok=True)
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
            try:
                import autoclean
                package_dir = Path(autoclean.__file__).parent.parent.parent
                source_file = package_dir / "examples" / "basic_usage.py"
                
                if source_file.exists():
                    shutil.copy2(source_file, dest_file)
                else:
                    self._create_fallback_example(dest_file)
            except Exception:
                self._create_fallback_example(dest_file)
                
            from rich.console import Console
            console = Console()
            console.print(f"[green]📄[/green] Example script: [dim]{dest_file}[/dim]")
            
        except Exception as e:
            message("warning", f"Could not create example script: {e}")
    
    def _create_fallback_example(self, dest_file: Path) -> None:
        """Create fallback example script."""
        content = '''import asyncio
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
'''
        
        with open(dest_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _extract_task_info(self, task_file: Path) -> tuple[str, str]:
        """Extract class name and description from task file."""
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if (isinstance(base, ast.Name) and base.id == 'Task') or \
                           (isinstance(base, ast.Attribute) and base.attr == 'Task'):
                            class_name = node.name
                            description = ast.get_docstring(node)
                            if description:
                                description = description.split('\n')[0]
                            else:
                                description = f"Custom task: {class_name}"
                            return class_name, description
            
            return task_file.stem, f"Custom task from {task_file.name}"
            
        except Exception:
            return task_file.stem, f"Custom task from {task_file.name}"
    
    def _current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# Global instance
user_config = UserConfigManager()