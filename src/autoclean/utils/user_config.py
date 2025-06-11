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
        # Use platformdirs for cross-platform config directory
        self.config_dir = Path(platformdirs.user_config_dir("autoclean", "autoclean"))
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
                "default_output_dir": "autoclean_output",
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
        return self._config["custom_tasks"].copy()
    
    def get_custom_task_path(self, task_name: str) -> Optional[Path]:
        """
        Get the path to a custom task file.
        
        Args:
            task_name: Name of the custom task
            
        Returns:
            Path to the task file, or None if not found
        """
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
    
    def _current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()


# Global instance
user_config = UserConfigManager()