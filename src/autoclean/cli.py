#!/usr/bin/env python3
"""
AutoClean EEG Pipeline - Command Line Interface

This module provides a flexible CLI for AutoClean that works both as a
standalone tool (via uv tool) and within development environments.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from autoclean.utils.logging import message
from autoclean.utils.user_config import user_config


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for AutoClean CLI."""
    parser = argparse.ArgumentParser(
        description="AutoClean EEG Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage (recommended)
  autoclean process RestingEyesOpen data.raw
  autoclean process RestingEyesOpen data_directory/
  
  # Advanced usage with options
  autoclean process --task RestingEyesOpen --file data.raw --output results/
  autoclean process --task RestingEyesOpen --dir data/ --output results/
  
  # Use Python task file
  autoclean process --task-file my_task.py --file data.raw
  
  # List available tasks
  autoclean list-tasks --include-custom
  
  # Start review GUI
  autoclean review --output results/
  
  # Add a custom task (saves to user config)
  autoclean task add my_task.py --name MyCustomTask
  
  # List custom tasks
  autoclean task list
  
  # Remove a custom task
  autoclean task remove MyCustomTask
  
  # Run setup wizard
  autoclean setup
  
  # Show user config location
  autoclean config show
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process EEG data")

    # Positional arguments for simple usage: autoclean process TaskName FilePath
    process_parser.add_argument(
        "task_name", nargs="?", type=str, help="Task name (e.g., RestingEyesOpen)"
    )
    process_parser.add_argument(
        "input_path", nargs="?", type=Path, help="EEG file or directory to process"
    )

    # Optional named arguments (for advanced usage)
    process_parser.add_argument(
        "--task", type=str, help="Task name (alternative to positional)"
    )
    process_parser.add_argument(
        "--task-file", type=Path, help="Python task file to use"
    )

    # Input options (for advanced usage)
    process_parser.add_argument(
        "--file",
        type=Path,
        help="Single EEG file to process (alternative to positional)",
    )
    process_parser.add_argument(
        "--dir",
        "--directory",
        type=Path,
        dest="directory",
        help="Directory containing EEG files to process (alternative to positional)",
    )

    process_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: workspace/output)",
    )
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )

    # List tasks command
    list_parser = subparsers.add_parser("list-tasks", help="List available tasks")
    list_parser.add_argument(
        "--include-custom",
        action="store_true",
        help="Include custom tasks from user configuration",
    )

    # Review command
    review_parser = subparsers.add_parser("review", help="Start review GUI")
    review_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="AutoClean output directory to review",
    )

    # Task management commands
    task_parser = subparsers.add_parser("task", help="Manage custom tasks")
    task_subparsers = task_parser.add_subparsers(
        dest="task_action", help="Task actions"
    )

    # Add task
    add_task_parser = task_subparsers.add_parser("add", help="Add a custom task")
    add_task_parser.add_argument("task_file", type=Path, help="Python task file to add")
    add_task_parser.add_argument(
        "--name", type=str, help="Custom name for the task (default: filename)"
    )
    add_task_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing task with same name"
    )

    # Remove task
    remove_task_parser = task_subparsers.add_parser(
        "remove", help="Remove a custom task"
    )
    remove_task_parser.add_argument(
        "task_name", type=str, help="Name of the task to remove"
    )

    # List custom tasks
    list_custom_parser = task_subparsers.add_parser("list", help="List custom tasks")
    list_custom_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )

    # Show config location
    config_parser = subparsers.add_parser("config", help="Manage user configuration")
    config_subparsers = config_parser.add_subparsers(
        dest="config_action", help="Config actions"
    )

    # Show config location
    config_subparsers.add_parser("show", help="Show configuration directory location")

    # Setup/reconfigure workspace
    config_subparsers.add_parser("setup", help="Reconfigure workspace location")

    # Reset config
    reset_parser = config_subparsers.add_parser(
        "reset", help="Reset configuration to defaults"
    )
    reset_parser.add_argument(
        "--confirm", action="store_true", help="Confirm the reset action"
    )

    # Export/import config
    export_parser = config_subparsers.add_parser("export", help="Export configuration")
    export_parser.add_argument(
        "export_path", type=Path, help="Directory to export configuration to"
    )

    import_parser = config_subparsers.add_parser("import", help="Import configuration")
    import_parser.add_argument(
        "import_path", type=Path, help="Directory to import configuration from"
    )

    # Setup command (same as config setup for simplicity)
    subparsers.add_parser("setup", help="Setup or reconfigure workspace")

    # Export access log command
    export_log_parser = subparsers.add_parser(
        "export-access-log", 
        help="Export database access log with integrity verification"
    )
    export_log_parser.add_argument(
        "--output", 
        type=Path, 
        help="Output file path (default: access-log-{timestamp}.json)"
    )
    export_log_parser.add_argument(
        "--format", 
        choices=["json", "csv", "human"], 
        default="json",
        help="Output format (default: json)"
    )
    export_log_parser.add_argument(
        "--start-date", 
        type=str, 
        help="Start date filter (YYYY-MM-DD format)"
    )
    export_log_parser.add_argument(
        "--end-date", 
        type=str, 
        help="End date filter (YYYY-MM-DD format)"
    )
    export_log_parser.add_argument(
        "--operation", 
        type=str, 
        help="Filter by operation type"
    )
    export_log_parser.add_argument(
        "--verify-only", 
        action="store_true",
        help="Only verify integrity, don't export data"
    )
    export_log_parser.add_argument(
        "--database", 
        type=Path, 
        help="Path to database file (default: auto-detect from workspace)"
    )

    # Version command
    subparsers.add_parser("version", help="Show version information")

    return parser


def validate_args(args) -> bool:
    """Validate command line arguments."""
    if args.command == "process":
        # Normalize positional vs named arguments
        task_name = args.task_name or args.task
        input_path = args.input_path or args.file or args.directory

        # Check that either task or task-file is provided
        if not task_name and not args.task_file:
            message("error", "Either task name or --task-file must be specified")
            return False

        if task_name and args.task_file:
            message("error", "Cannot specify both task name and --task-file")
            return False

        # Check input exists
        if input_path and not input_path.exists():
            message("error", f"Input path does not exist: {input_path}")
            return False
        elif not input_path:
            message("error", "Input file or directory must be specified")
            return False

        # Store normalized values back to args
        args.final_task = task_name
        args.final_input = input_path

        # Check task file exists if provided
        if args.task_file and not args.task_file.exists():
            message("error", f"Task file does not exist: {args.task_file}")
            return False

    elif args.command == "review":
        if not args.output.exists():
            message("error", f"Output directory does not exist: {args.output}")
            return False

    return True


def cmd_process(args) -> int:
    """Execute the process command."""
    try:
        # Lazy import Pipeline only when needed
        from autoclean.core.pipeline import Pipeline

        # Initialize pipeline
        pipeline_kwargs = {"output_dir": args.output}

        pipeline = Pipeline(**pipeline_kwargs)

        # Add Python task file if provided
        if args.task_file:
            task_name = pipeline.add_task(args.task_file)
            message("info", f"Loaded Python task: {task_name}")
        else:
            task_name = args.final_task

            # Check if this is a custom task from user config
            custom_task_path = user_config.get_custom_task_path(task_name)
            if custom_task_path:
                task_name = pipeline.add_task(custom_task_path)
                message(
                    "info",
                    f"Loaded custom task '{args.final_task}' from user configuration",
                )

        if args.dry_run:
            message("info", "DRY RUN - No processing will be performed")
            message("info", f"Would process: {args.final_input}")
            message("info", f"Task: {task_name}")
            message("info", f"Output: {args.output}")
            return 0

        # Process files
        if args.final_input.is_file():
            message("info", f"Processing single file: {args.final_input}")
            pipeline.process_file(file_path=args.final_input, task=task_name)
        else:
            message("info", f"Processing directory: {args.final_input}")
            pipeline.process_directory(directory=args.final_input, task=task_name)

        message("info", "Processing completed successfully!")
        return 0

    except Exception as e:
        message("error", f"Processing failed: {str(e)}")
        return 1


def cmd_list_tasks(args) -> int:
    """Execute the list-tasks command."""
    try:
        # Lazy import Pipeline only when needed
        from autoclean.core.pipeline import Pipeline

        pipeline_kwargs = {
            "output_dir": Path("./temp_autoclean")  # Temporary dir for listing tasks
        }

        pipeline = Pipeline(**pipeline_kwargs)

        # List built-in tasks
        message("info", "Built-in tasks:")
        tasks = pipeline.list_tasks()

        if not tasks:
            print("  No built-in tasks found")
        else:
            for task in tasks:
                print(f"  • {task}")

        # List custom tasks if requested
        if args.include_custom:
            custom_tasks = user_config.list_custom_tasks()
            if custom_tasks:
                print()
                message("info", "Custom tasks:")
                for task_name, task_info in custom_tasks.items():
                    desc = task_info.get("description", "No description")
                    print(f"  • {task_name} - {desc}")
            else:
                print()
                message("info", "No custom tasks found")
                print("  Add custom tasks with: autoclean task add <file>")
        else:
            custom_tasks = user_config.list_custom_tasks()
            if custom_tasks:
                print()
                print(
                    f"  ({len(custom_tasks)} custom tasks available - use --include-custom to show)"
                )

        return 0

    except Exception as e:
        message("error", f"Failed to list tasks: {str(e)}")
        return 1


def cmd_review(args) -> int:
    """Execute the review command."""
    try:
        # Lazy import Pipeline only when needed
        from autoclean.core.pipeline import Pipeline

        pipeline = Pipeline(output_dir=args.output)

        message("info", f"Starting review GUI for: {args.output}")
        pipeline.start_autoclean_review()

        return 0

    except Exception as e:
        message("error", f"Failed to start review GUI: {str(e)}")
        return 1


def cmd_setup(args) -> int:
    """Run the setup wizard."""
    try:
        user_config.setup_workspace()
        return 0
    except Exception as e:
        message("error", f"Setup failed: {str(e)}")
        return 1


def cmd_version(args) -> int:
    """Show version information."""
    try:
        from autoclean import __version__

        print(f"AutoClean EEG Pipeline v{__version__}")
        return 0
    except ImportError:
        print("AutoClean EEG Pipeline (version unknown)")
        return 0


def cmd_task(args) -> int:
    """Execute task management commands."""
    if args.task_action == "add":
        return cmd_task_add(args)
    elif args.task_action == "remove":
        return cmd_task_remove(args)
    elif args.task_action == "list":
        return cmd_task_list(args)
    else:
        message("error", "No task action specified")
        return 1


def cmd_task_add(args) -> int:
    """Add a custom task by copying to workspace tasks folder."""
    try:
        if not args.task_file.exists():
            message("error", f"Task file not found: {args.task_file}")
            return 1

        # Ensure workspace exists
        if not user_config.tasks_dir.exists():
            user_config.tasks_dir.mkdir(parents=True, exist_ok=True)

        # Determine destination name
        if args.name:
            dest_name = f"{args.name}.py"
        else:
            dest_name = args.task_file.name

        dest_file = user_config.tasks_dir / dest_name

        # Check if task already exists
        if dest_file.exists() and not args.force:
            message(
                "error", f"Task '{dest_name}' already exists. Use --force to overwrite."
            )
            return 1

        # Copy the task file
        import shutil

        shutil.copy2(args.task_file, dest_file)

        # Extract class name for usage message
        try:
            class_name, _ = user_config._extract_task_info(dest_file)
            task_name = class_name
        except Exception:
            task_name = dest_file.stem

        message("info", f"Task '{task_name}' added to workspace!")
        print(f"📁 Copied to: {dest_file}")
        print("\nUse your custom task with:")
        print(f"  autoclean process {task_name} <data_file>")

        return 0

    except Exception as e:
        message("error", f"Failed to add custom task: {str(e)}")
        return 1


def cmd_task_remove(args) -> int:
    """Remove a custom task by deleting from workspace tasks folder."""
    try:
        # Find task file by class name or filename
        custom_tasks = user_config.list_custom_tasks()

        task_file = None
        if args.task_name in custom_tasks:
            # Found by class name
            task_file = Path(custom_tasks[args.task_name]["file_path"])
        else:
            # Try by filename
            potential_file = user_config.tasks_dir / f"{args.task_name}.py"
            if potential_file.exists():
                task_file = potential_file

        if not task_file or not task_file.exists():
            message("error", f"Task '{args.task_name}' not found")
            return 1

        # Remove the file
        task_file.unlink()
        message("info", f"Task '{args.task_name}' removed from workspace!")
        return 0

    except Exception as e:
        message("error", f"Failed to remove custom task: {str(e)}")
        return 1


def cmd_task_list(args) -> int:
    """List custom tasks."""
    try:
        custom_tasks = user_config.list_custom_tasks()

        if not custom_tasks:
            message("info", "No custom tasks found")
            print("  Add custom tasks with: autoclean task add <file>")
            return 0

        message("info", f"Custom tasks ({len(custom_tasks)} found):")

        for task_name, task_info in custom_tasks.items():
            print(f"\n  📝 {task_name}")
            print(f"     Description: {task_info.get('description', 'No description')}")

            if args.verbose:
                print(f"     File: {task_info['file_path']}")
                print(f"     Added: {task_info.get('added_date', 'Unknown')}")
                if task_info.get("original_path"):
                    print(f"     Original: {task_info['original_path']}")

        print(
            f"\nUse any task with: autoclean process {list(custom_tasks.keys())[0]} <data_file>"
        )
        return 0

    except Exception as e:
        message("error", f"Failed to list custom tasks: {str(e)}")
        return 1


def cmd_config(args) -> int:
    """Execute configuration management commands."""
    if args.config_action == "show":
        return cmd_config_show(args)
    elif args.config_action == "setup":
        return cmd_config_setup(args)
    elif args.config_action == "reset":
        return cmd_config_reset(args)
    elif args.config_action == "export":
        return cmd_config_export(args)
    elif args.config_action == "import":
        return cmd_config_import(args)
    else:
        message("error", "No config action specified")
        return 1


def cmd_config_show(args) -> int:
    """Show user configuration directory."""
    config_dir = user_config.get_config_dir()
    message("info", f"User configuration directory: {config_dir}")

    custom_tasks = user_config.list_custom_tasks()
    print(f"  • Custom tasks: {len(custom_tasks)}")
    print(f"  • Tasks directory: {config_dir / 'tasks'}")
    print(f"  • Config file: {config_dir / 'user_config.json'}")

    return 0


def cmd_config_setup(args) -> int:
    """Reconfigure workspace location."""
    try:
        user_config.setup_workspace()
        return 0
    except Exception as e:
        message("error", f"Failed to reconfigure workspace: {str(e)}")
        return 1


def cmd_config_reset(args) -> int:
    """Reset user configuration to defaults."""
    if not args.confirm:
        message("error", "This will delete all custom tasks and reset configuration.")
        print("Use --confirm to proceed with reset.")
        return 1

    try:
        user_config.reset_config()
        message("info", "User configuration reset to defaults")
        return 0
    except Exception as e:
        message("error", f"Failed to reset configuration: {str(e)}")
        return 1


def cmd_config_export(args) -> int:
    """Export user configuration."""
    try:
        if user_config.export_config(args.export_path):
            return 0
        else:
            return 1
    except Exception as e:
        message("error", f"Failed to export configuration: {str(e)}")
        return 1


def cmd_config_import(args) -> int:
    """Import user configuration."""
    try:
        if user_config.import_config(args.import_path):
            return 0
        else:
            return 1
    except Exception as e:
        message("error", f"Failed to import configuration: {str(e)}")
        return 1


def cmd_export_access_log(args) -> int:
    """Export database access log with integrity verification."""
    try:
        from datetime import datetime
        from autoclean.utils.audit import verify_access_log_integrity
        from autoclean.utils.database import DB_PATH
        from autoclean.utils.user_config import user_config
        import sqlite3
        import json
        import csv
        
        # Get workspace directory for database discovery and default output location
        workspace_dir = user_config._get_workspace_path()
        
        # Determine database path
        if args.database:
            db_path = Path(args.database)
        elif DB_PATH:
            db_path = DB_PATH / "pipeline.db"
        else:
            # Try to find database in workspace
            if workspace_dir:
                # Check for database directly in workspace directory 
                workspace_db = workspace_dir / "pipeline.db"
                if workspace_db.exists():
                    db_path = workspace_db
                else:
                    # Fall back to checking workspace/output/ directory (most common location)
                    output_db = workspace_dir / "output" / "pipeline.db"
                    if output_db.exists():
                        db_path = output_db
                    else:
                        # Finally, look in output subdirectories (for multiple runs)
                        potential_outputs = workspace_dir / "output"
                        if potential_outputs.exists():
                            # Look for most recent output directory with database
                            for output_dir in sorted(potential_outputs.iterdir(), reverse=True):
                                if output_dir.is_dir():
                                    potential_db = output_dir / "pipeline.db"
                                    if potential_db.exists():
                                        db_path = potential_db
                                        break
                            else:
                                message("error", "No database found in workspace directory, output directory, or output subdirectories")
                                return 1
                        else:
                            message("error", "No database found in workspace directory and no output directory exists")
                            return 1
            else:
                message("error", "No workspace configured and no database path provided")
                return 1
        
        if not db_path.exists():
            message("error", f"Database file not found: {db_path}")
            return 1
        
        message("info", f"Using database: {db_path}")
        
        # Verify integrity first (need to temporarily set DB_PATH for verification)
        from autoclean.utils import database
        original_db_path = database.DB_PATH
        database.DB_PATH = db_path.parent
        
        try:
            integrity_result = verify_access_log_integrity()
        finally:
            database.DB_PATH = original_db_path
        
        if args.verify_only:
            if integrity_result["status"] == "valid":
                message("success", f"✓ {integrity_result['message']}")
                return 0
            elif integrity_result["status"] == "compromised":
                message("error", f"✗ {integrity_result['message']}")
                if "issues" in integrity_result:
                    for issue in integrity_result["issues"]:
                        message("error", f"  - {issue}")
                return 1
            else:
                message("error", f"✗ {integrity_result['message']}")
                return 1
        
        # Report integrity status
        if integrity_result["status"] == "valid":
            message("success", f"✓ {integrity_result['message']}")
        elif integrity_result["status"] == "compromised":
            message("warning", f"⚠ {integrity_result['message']}")
            if "issues" in integrity_result:
                for issue in integrity_result["issues"]:
                    message("warning", f"  - {issue}")
        else:
            message("warning", f"⚠ {integrity_result['message']}")
        
        # Query access log with filters
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM database_access_log WHERE 1=1"
        params = []
        
        # Apply date filters
        if args.start_date:
            query += " AND date(timestamp) >= ?"
            params.append(args.start_date)
        
        if args.end_date:
            query += " AND date(timestamp) <= ?"
            params.append(args.end_date)
        
        # Apply operation filter
        if args.operation:
            query += " AND operation LIKE ?"
            params.append(f"%{args.operation}%")
        
        query += " ORDER BY log_id ASC"
        
        cursor.execute(query, params)
        entries = cursor.fetchall()
        conn.close()
        
        if not entries:
            message("warning", "No access log entries found matching filters")
            return 0
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use .jsonl extension for JSON Lines format
            if args.format == "json":
                extension = "jsonl"
            else:
                extension = args.format
            filename = f"access-log-{timestamp}.{extension}"
            # Default to workspace directory, not current working directory
            output_file = workspace_dir / filename if workspace_dir else Path(filename)
        
        # Export data
        export_data = []
        for entry in entries:
            entry_dict = dict(entry)
            # Parse JSON fields for export
            if entry_dict.get("user_context"):
                try:
                    entry_dict["user_context"] = json.loads(entry_dict["user_context"])
                except json.JSONDecodeError:
                    pass
            if entry_dict.get("details"):
                try:
                    entry_dict["details"] = json.loads(entry_dict["details"])
                except json.JSONDecodeError:
                    pass
            export_data.append(entry_dict)
        
        # Write export file
        if args.format == "json":
            # Write as JSONL (JSON Lines) format - more compact and easier to process
            with open(output_file, "w") as f:
                # First line: metadata
                metadata = {
                    "type": "metadata",
                    "export_timestamp": datetime.now().isoformat(),
                    "database_path": str(db_path),
                    "total_entries": len(export_data),
                    "integrity_status": integrity_result["status"],
                    "integrity_message": integrity_result["message"],
                    "filters_applied": {
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                        "operation": args.operation
                    }
                }
                f.write(json.dumps(metadata) + "\n")
                
                # Subsequent lines: one JSON object per access log entry
                for entry in export_data:
                    entry["type"] = "access_log"
                    f.write(json.dumps(entry) + "\n")
                
        elif args.format == "csv":
            with open(output_file, "w", newline="") as f:
                if export_data:
                    fieldnames = export_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in export_data:
                        # Flatten JSON fields for CSV
                        csv_entry = {}
                        for key, value in entry.items():
                            if isinstance(value, (dict, list)):
                                csv_entry[key] = json.dumps(value)
                            else:
                                csv_entry[key] = value
                        writer.writerow(csv_entry)
                        
        elif args.format == "human":
            with open(output_file, "w") as f:
                f.write("AutoClean Database Access Log Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Export Date: {datetime.now().isoformat()}\n")
                f.write(f"Database: {db_path}\n")
                f.write(f"Total Entries: {len(export_data)}\n")
                f.write(f"Integrity Status: {integrity_result['status']}\n")
                f.write(f"Integrity Message: {integrity_result['message']}\n\n")
                
                if args.start_date or args.end_date or args.operation:
                    f.write("Filters Applied:\n")
                    if args.start_date:
                        f.write(f"  Start Date: {args.start_date}\n")
                    if args.end_date:
                        f.write(f"  End Date: {args.end_date}\n")
                    if args.operation:
                        f.write(f"  Operation: {args.operation}\n")
                    f.write("\n")
                
                f.write("Access Log Entries:\n")
                f.write("-" * 30 + "\n\n")
                
                for i, entry in enumerate(export_data, 1):
                    f.write(f"Entry {i} (ID: {entry['log_id']})\n")
                    f.write(f"  Timestamp: {entry['timestamp']}\n")
                    f.write(f"  Operation: {entry['operation']}\n")
                    
                    if entry.get("user_context"):
                        user_ctx = entry["user_context"]
                        if isinstance(user_ctx, dict):
                            # Handle both old and new format
                            user = user_ctx.get('user', user_ctx.get('username', 'unknown'))
                            host = user_ctx.get('host', user_ctx.get('hostname', 'unknown'))
                            f.write(f"  User: {user}\n")
                            f.write(f"  Host: {host}\n")
                    
                    if entry.get("details") and entry["details"]:
                        f.write(f"  Details: {json.dumps(entry['details'], indent=4)}\n")
                    
                    f.write(f"  Hash: {entry['log_hash'][:16]}...\n")
                    f.write("\n")
        
        message("success", f"✓ Access log exported to: {output_file}")
        message("info", f"Format: {args.format}, Entries: {len(export_data)}")
        
        return 0
        
    except Exception as e:
        message("error", f"Failed to export access log: {e}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the AutoClean CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Validate arguments
    if not validate_args(args):
        return 1

    # Execute command
    if args.command == "process":
        return cmd_process(args)
    elif args.command == "list-tasks":
        return cmd_list_tasks(args)
    elif args.command == "review":
        return cmd_review(args)
    elif args.command == "task":
        return cmd_task(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "setup":
        return cmd_setup(args)
    elif args.command == "export-access-log":
        return cmd_export_access_log(args)
    elif args.command == "version":
        return cmd_version(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
