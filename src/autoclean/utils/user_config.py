"""
User configuration management for AutoClean.

This module handles persistent user configuration including custom tasks,
preferences, and settings that persist across AutoClean sessions.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import platformdirs

from autoclean.utils.logging import message


class UserConfigManager:
    """Manages persistent user configuration for AutoClean."""
    
    def __init__(self):
        """Initialize the user configuration manager."""
        # Check if this is first time setup
        self.config_dir = self._get_or_setup_config_directory()
        self.config_file = self.config_dir / "user_config.json"
        self.tasks_dir = self.config_dir / "tasks"
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing config
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load user configuration from disk."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                message("warning", f"Failed to load user config: {e}")
                return self._default_config()
        else:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default user configuration."""
        return {
            "version": "1.0",
            "custom_tasks": {},
            "preferences": {
                "default_output_dir": str(self.config_dir / "output"),
                "auto_save_tasks": True,
                "confirm_overwrite": True
            },
            "recent_files": [],
            "recent_outputs": []
        }
    
    def _save_config(self) -> bool:
        """Save user configuration to disk."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            message("error", f"Failed to save user config: {e}")
            return False
    
    def add_custom_task(self, task_file: Path, task_name: Optional[str] = None) -> str:
        """
        Add a custom task to the user's persistent collection.
        
        Args:
            task_file: Path to the Python task file
            task_name: Optional custom name (defaults to file stem)
            
        Returns:
            The name of the saved task
        """
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        
        # Extract class name and description
        class_name, description = self._extract_task_info(task_file)
        
        # Use provided name or class name
        if task_name is None:
            task_name = class_name
        
        # Ensure unique task name
        base_name = task_name
        counter = 1
        while task_name in self._config["custom_tasks"]:
            task_name = f"{base_name}_{counter}"
            counter += 1
        
        # Copy task file to user's tasks directory
        dest_file = self.tasks_dir / f"{task_name}.py"
        shutil.copy2(task_file, dest_file)
        
        # Update config
        self._config["custom_tasks"][task_name] = {
            "file_path": str(dest_file),
            "original_path": str(task_file),
            "added_date": self._current_timestamp(),
            "description": description,
            "class_name": class_name
        }
        
        # Save config
        if self._save_config():
            message("info", f"Custom task '{task_name}' saved to user configuration")
            return task_name
        else:
            raise RuntimeError("Failed to save user configuration")
    
    def remove_custom_task(self, task_name: str) -> bool:
        """
        Remove a custom task from the user's collection.
        
        Args:
            task_name: Name of the task to remove
            
        Returns:
            True if task was removed, False if not found
        """
        if task_name not in self._config["custom_tasks"]:
            return False
        
        # Remove task file
        task_info = self._config["custom_tasks"][task_name]
        task_file = Path(task_info["file_path"])
        if task_file.exists():
            task_file.unlink()
        
        # Remove from config
        del self._config["custom_tasks"][task_name]
        
        # Save config
        self._save_config()
        message("info", f"Custom task '{task_name}' removed")
        return True
    
    def list_custom_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        List all custom tasks in the user's collection.
        
        Returns:
            Dictionary of task names and their metadata
        """
        # Auto-discover new tasks before listing
        self._auto_discover_tasks()
        return self._config["custom_tasks"].copy()
    
    def get_custom_task_path(self, task_name: str) -> Optional[Path]:
        """
        Get the path to a custom task file.
        
        Args:
            task_name: Name of the custom task
            
        Returns:
            Path to the task file, or None if not found
        """
        # Auto-discover new tasks before looking up
        self._auto_discover_tasks()
        
        if task_name in self._config["custom_tasks"]:
            task_path = Path(self._config["custom_tasks"][task_name]["file_path"])
            if task_path.exists():
                return task_path
        return None
    
    def update_preference(self, key: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            key: Preference key
            value: New value
            
        Returns:
            True if updated successfully
        """
        self._config["preferences"][key] = value
        return self._save_config()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference value.
        
        Args:
            key: Preference key
            default: Default value if key not found
            
        Returns:
            Preference value or default
        """
        return self._config["preferences"].get(key, default)
    
    def add_recent_file(self, file_path: Path, max_recent: int = 10) -> None:
        """Add a file to the recent files list."""
        file_str = str(file_path.absolute())
        recent = self._config["recent_files"]
        
        # Remove if already exists
        if file_str in recent:
            recent.remove(file_str)
        
        # Add to front
        recent.insert(0, file_str)
        
        # Limit length
        self._config["recent_files"] = recent[:max_recent]
        self._save_config()
    
    def add_recent_output(self, output_path: Path, max_recent: int = 10) -> None:
        """Add an output directory to the recent outputs list."""
        output_str = str(output_path.absolute())
        recent = self._config["recent_outputs"]
        
        # Remove if already exists
        if output_str in recent:
            recent.remove(output_str)
        
        # Add to front
        recent.insert(0, output_str)
        
        # Limit length
        self._config["recent_outputs"] = recent[:max_recent]
        self._save_config()
    
    def get_recent_files(self) -> List[str]:
        """Get list of recent files."""
        return self._config["recent_files"].copy()
    
    def get_recent_outputs(self) -> List[str]:
        """Get list of recent output directories."""
        return self._config["recent_outputs"].copy()
    
    def export_config(self, export_path: Path) -> bool:
        """
        Export user configuration to a file.
        
        Args:
            export_path: Path to export the config to
            
        Returns:
            True if exported successfully
        """
        try:
            shutil.copytree(self.config_dir, export_path, dirs_exist_ok=True)
            message("info", f"User configuration exported to: {export_path}")
            return True
        except OSError as e:
            message("error", f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: Path) -> bool:
        """
        Import user configuration from a file.
        
        Args:
            import_path: Path to import the config from
            
        Returns:
            True if imported successfully
        """
        try:
            if (import_path / "user_config.json").exists():
                shutil.copytree(import_path, self.config_dir, dirs_exist_ok=True)
                self._config = self._load_config()
                message("info", f"User configuration imported from: {import_path}")
                return True
            else:
                message("error", "Invalid configuration directory")
                return False
        except OSError as e:
            message("error", f"Failed to import configuration: {e}")
            return False
    
    def reset_config(self) -> bool:
        """Reset user configuration to defaults."""
        try:
            # Remove config directory
            if self.config_dir.exists():
                shutil.rmtree(self.config_dir)
            
            # Recreate with defaults
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.tasks_dir.mkdir(parents=True, exist_ok=True)
            self._config = self._default_config()
            self._save_config()
            
            message("info", "User configuration reset to defaults")
            return True
        except OSError as e:
            message("error", f"Failed to reset configuration: {e}")
            return False
    
    def get_config_dir(self) -> Path:
        """Get the user configuration directory path."""
        return self.config_dir
    
    def get_default_output_dir(self) -> Path:
        """Get the default output directory for pipeline processing."""
        return self.config_dir / "output"
    
    def reconfigure_workspace(self) -> Path:
        """Allow user to reconfigure their workspace location."""
        # Check if this is a completely fresh installation
        global_config_file = Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        
        if not global_config_file.exists():
            # True first-time setup
            return self._run_first_time_setup(is_reconfigure=False)
        
        # This is a reconfiguration
        print("\n" + "="*60)
        print("🔧 Reconfigure AutoClean Workspace")
        print("="*60)
        print(f"Current workspace: {self.config_dir}")
        print()
        
        # Run setup again
        new_config_dir = self._run_first_time_setup(is_reconfigure=True)
        
        # If user chose a different directory, offer to migrate
        if new_config_dir != self.config_dir:
            print("Would you like to migrate your existing tasks and configuration?")
            try:
                migrate = input("Type 'yes' to migrate, or 'no' to start fresh: ").strip().lower()
                if migrate in ['yes', 'y']:
                    self._migrate_workspace(self.config_dir, new_config_dir)
            except (EOFError, KeyboardInterrupt):
                print("Skipping migration.")
            
            # Update current instance
            self.config_dir = new_config_dir
            self.config_file = self.config_dir / "user_config.json"
            self.tasks_dir = self.config_dir / "tasks"
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.tasks_dir.mkdir(parents=True, exist_ok=True)
            self._config = self._load_config()
        
        return new_config_dir
    
    def _migrate_workspace(self, old_dir: Path, new_dir: Path) -> None:
        """Migrate workspace from old to new location."""
        try:
            if not old_dir.exists():
                print("No existing workspace to migrate.")
                return
                
            import shutil
            
            # Ensure new directory exists
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all contents
            items_copied = 0
            for item in old_dir.iterdir():
                dest = new_dir / item.name
                try:
                    if item.is_file():
                        shutil.copy2(item, dest)
                        items_copied += 1
                    elif item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                        items_copied += 1
                except Exception as e:
                    print(f"⚠️  Could not copy {item.name}: {e}")
            
            if items_copied > 0:
                print(f"✅ Successfully copied {items_copied} items to {new_dir}")
                
                # Always offer to clean up old directory
                print(f"\nKeep the old workspace directory at {old_dir}?")
                try:
                    import sys
                    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
                        # In interactive/Jupyter - default to keeping
                        print("Keeping old workspace (you can delete it manually if desired)")
                    else:
                        # Command line - can prompt
                        cleanup = input("Type 'no' to remove it, or press Enter to keep: ").strip().lower()
                        if cleanup in ['no', 'n']:
                            shutil.rmtree(old_dir)
                            print("✅ Old workspace removed")
                        else:
                            print("Old workspace kept - you can delete it manually if desired")
                except (EOFError, KeyboardInterrupt):
                    print("Old workspace kept")
            else:
                print("No items found to migrate.")
                    
        except Exception as e:
            print(f"⚠️  Migration failed: {e}")
            print("You may need to manually copy your files.")
    
    def _extract_task_info(self, task_file: Path) -> tuple[str, str]:
        """Extract class name and description from task file."""
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for Task class
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it inherits from Task
                    for base in node.bases:
                        if (isinstance(base, ast.Name) and base.id == 'Task') or \
                           (isinstance(base, ast.Attribute) and base.attr == 'Task'):
                            class_name = node.name
                            description = ast.get_docstring(node)
                            if description:
                                description = description.split('\n')[0]  # First line
                            else:
                                description = f"Custom task: {class_name}"
                            return class_name, description
                        
            # Fallback if no Task class found
            return task_file.stem, f"Custom task from {task_file.name}"
            
        except Exception:
            return task_file.stem, f"Custom task from {task_file.name}"
    
    def _auto_discover_tasks(self) -> None:
        """Auto-discover and register new task files in the tasks directory."""
        if not self.tasks_dir.exists():
            return
            
        discovered_tasks = []
        
        # Scan for Python files in tasks directory
        for task_file in self.tasks_dir.glob("*.py"):
            if task_file.name.startswith("_"):  # Skip private files
                continue
                
            # Check if already registered
            file_path_str = str(task_file)
            already_registered = any(
                task_info.get("file_path") == file_path_str 
                for task_info in self._config["custom_tasks"].values()
            )
            
            if not already_registered:
                try:
                    # Extract task info
                    class_name, description = self._extract_task_info(task_file)
                    
                    # Generate unique task name
                    task_name = class_name
                    base_name = task_name
                    counter = 1
                    while task_name in self._config["custom_tasks"]:
                        task_name = f"{base_name}_{counter}"
                        counter += 1
                    
                    # Register the task
                    self._config["custom_tasks"][task_name] = {
                        "file_path": str(task_file),
                        "original_path": str(task_file),  # Same as file_path for discovered tasks
                        "added_date": self._current_timestamp(),
                        "description": description,
                        "class_name": class_name,
                        "auto_discovered": True  # Mark as auto-discovered
                    }
                    
                    discovered_tasks.append(task_name)
                    
                except Exception as e:
                    # Skip files that can't be parsed
                    message("warning", f"Could not parse task file {task_file.name}: {e}")
                    continue
        
        # Save config if new tasks were discovered
        if discovered_tasks:
            self._save_config()
            for task_name in discovered_tasks:
                message("info", f"Auto-discovered custom task: {task_name}")
    
    def _get_or_setup_config_directory(self) -> Path:
        """Get the config directory, setting it up if this is the first time."""
        # Check if user has already set up a config directory
        global_config_file = Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        
        if global_config_file.exists():
            try:
                with open(global_config_file, 'r', encoding='utf-8') as f:
                    setup_config = json.load(f)
                    return Path(setup_config["config_directory"])
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass  # Fall through to setup
        
        # First time setup - guide user through configuration
        return self._run_first_time_setup()
    
    def _run_first_time_setup(self, is_reconfigure: bool = False) -> Path:
        """Guide user through configuration setup."""
        print("\n" + "="*60)
        print("🧠 Welcome to AutoClean EEG Pipeline!")
        print("="*60)
        if is_reconfigure:
            print("Workspace setup: Let's configure your workspace location.")
        else:
            print("First-time setup: Let's configure your workspace.")
        print()
        
        # Default to Documents folder
        default_dir = Path(platformdirs.user_documents_dir()) / "Autoclean-EEG"
        
        print(f"Where would you like to store your custom tasks and configuration?")
        print(f"Default: {default_dir}")
        print()
        print("This folder will contain:")
        print("  • Your custom EEG processing tasks")
        print("  • Configuration settings")
        print("  • Default output directory for processing results")
        print("  • Easy access for editing and backup")
        print()
        
        # In Jupyter/interactive environments, we can't easily prompt, so use default
        try:
            # Check if we're in an interactive environment
            import sys
            if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
                # Interactive Python or Jupyter - use default
                chosen_dir = default_dir
                print(f"Using default location: {chosen_dir}")
            else:
                # Command line - can prompt
                response = input(f"Press Enter for default, or type a custom path: ").strip()
                if response:
                    chosen_dir = Path(response).expanduser()
                else:
                    chosen_dir = default_dir
        except (EOFError, KeyboardInterrupt):
            # Fallback to default if input fails
            chosen_dir = default_dir
            print(f"Using default location: {chosen_dir}")
        
        # Save the setup configuration
        global_config_file = Path(platformdirs.user_config_dir("autoclean", "autoclean")) / "setup.json"
        global_config_file.parent.mkdir(parents=True, exist_ok=True)
        
        setup_config = {
            "config_directory": str(chosen_dir),
            "setup_date": self._current_timestamp(),
            "version": "1.0"
        }
        
        try:
            with open(global_config_file, 'w', encoding='utf-8') as f:
                json.dump(setup_config, f, indent=2)
        except Exception as e:
            message("warning", f"Could not save setup configuration: {e}")
        
        print()
        if is_reconfigure:
            print("✅ Workspace configuration updated!")
        else:
            print("✅ Setup complete!")
        print(f"📁 Your AutoClean workspace: {chosen_dir}")
        print()
        if not is_reconfigure:
            print("Next steps:")
            print("1. Visit the web UI configuration wizard to create custom tasks")
            print("2. Save task files to your tasks folder")
            print("3. Use them in Jupyter notebooks or scripts!")
            print()
        print("="*60)
        
        # Create example script in the workspace
        self._create_example_script(chosen_dir)
        
        return chosen_dir
    
    def _current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _create_example_script(self, workspace_dir: Path) -> None:
        """Create an example script in the workspace directory."""
        try:
            # Path to the example file in the package
            import autoclean
            package_dir = Path(autoclean.__file__).parent.parent.parent  # Go up to autoclean_pipeline
            examples_dir = package_dir / "examples"
            source_file = examples_dir / "basic_usage.py"
            
            # Destination in workspace
            dest_file = workspace_dir / "example_basic_usage.py"
            
            # Copy the example file if it exists
            if source_file.exists():
                import shutil
                shutil.copy2(source_file, dest_file)
                print(f"📄 Example script created: {dest_file}")
                print("   Edit this file to customize for your data paths and processing needs!")
            else:
                # Fallback: create a simple example inline
                self._create_fallback_example(dest_file)
                
        except Exception as e:
            message("warning", f"Could not create example script: {e}")
            # Try fallback method
            try:
                self._create_fallback_example(workspace_dir / "example_basic_usage.py")
            except Exception:
                pass  # Silently fail if fallback also fails
    
    def _create_fallback_example(self, dest_file: Path) -> None:
        """Create a simple example script as fallback."""
        example_content = '''import asyncio
from pathlib import Path

from autoclean import Pipeline

EXAMPLE_OUTPUT_DIR = Path("path/to/output/directory")  # Where processed data will be stored

async def batch_run():
    """Example of batch processing multiple EEG files asynchronously."""
    # Create pipeline instance
    pipeline = Pipeline(
        output_dir=EXAMPLE_OUTPUT_DIR,
        verbose='HEADER'
    )
    # Example INPUT directory path - modify this to point to your EEG files
    directory = Path("path/to/input/directory")

    # Process all files in directory
    await pipeline.process_directory_async(
        directory=directory,
        task="RestingEyesOpen",  # Choose appropriate task
        sub_directories=False, # Optional: process files in subfolders
        pattern="*.set", # Optional: specify a pattern to filter files (use "*.extention" for all files of that extension)
        max_concurrent=3 # Optional: specify the maximum number of concurrent files to process
    )

def single_file_run():
    pipeline = Pipeline(
        output_dir=EXAMPLE_OUTPUT_DIR,
        verbose='HEADER'
    )
    file_path = Path("path/to/input/file")

    pipeline.process_file(
        file_path=file_path,
        task="RestingEyesOpen",  # Choose appropriate task
    )
    
if __name__ == "__main__":
    #Batch run example
    asyncio.run(batch_run())

    #Single file run example
    single_file_run()
'''
        
        with open(dest_file, 'w', encoding='utf-8') as f:
            f.write(example_content)
        print(f"📄 Example script created: {dest_file}")
        print("   Edit this file to customize for your data paths and processing needs!")


# Global instance
user_config = UserConfigManager()