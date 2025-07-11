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
    from autoclean.utils.branding import AutoCleanBranding

    parser = argparse.ArgumentParser(
        description=f"{AutoCleanBranding.PRODUCT_NAME}\n{AutoCleanBranding.TAGLINE}\n\nGitHub: https://github.com/cincibrainlab/autoclean_pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing:
  autoclean-eeg setup                           # Interactive setup wizard
  autoclean-eeg process TaskName data.raw       # Process single file
  autoclean-eeg process TaskName data_dir/      # Process directory
  autoclean-eeg list-tasks                      # Show available tasks
  autoclean-eeg review --output results/       # Start review GUI

Authentication (Compliance Mode):
  autoclean-eeg setup --compliance-mode        # Enable FDA 21 CFR Part 11 compliance
  autoclean-eeg login                          # Authenticate with Auth0
  autoclean-eeg logout                         # Clear authentication
  autoclean-eeg whoami                         # Show current user status
  autoclean-eeg auth0 show                     # Show current Auth0 configuration
  autoclean-eeg auth0 set                     # Set custom Auth0 application (interactive)
  autoclean-eeg auth0 reset                    # Reset to default Auth0 config
  autoclean-eeg auth0 test                     # Test Auth0 connectivity

Custom Tasks:
  autoclean-eeg task add my_task.py            # Add custom task
  autoclean-eeg task list                      # List custom tasks
  autoclean-eeg task remove TaskName           # Remove custom task

Configuration:
  autoclean-eeg config show                    # Show config location
  autoclean-eeg config setup                   # Reconfigure workspace
  autoclean-eeg config reset                   # Reset to defaults

Audit & Export:
  autoclean-eeg export-access-log              # Export compliance audit log
  autoclean-eeg version                        # Show version info

Examples:
  # Simple usage (recommended)
  autoclean-eeg process RestingEyesOpen data.raw
  autoclean-eeg process RestingEyesOpen data_directory/
  
  # Advanced usage with options
  autoclean-eeg process --task RestingEyesOpen --file data.raw --output results/
  autoclean-eeg process --task RestingEyesOpen --dir data/ --output results/ --format "*.raw"
  
  # Use Python task file
  autoclean-eeg process --task-file my_task.py --file data.raw
  autoclean-eeg process --task-file custom.py --file data.raw
  
  # List available tasks
  autoclean-eeg task list
  
  # Start review GUI
  autoclean-eeg review --output results/
  
  # Add a custom task (saves to user config)
  autoclean-eeg task add my_task.py --name MyCustomTask
  
  # List all tasks (built-in and custom)
  autoclean-eeg task list
  
  # Remove a custom task
  autoclean-eeg task remove MyCustomTask
  
  # Run setup wizard
  autoclean-eeg setup
  
  # Reset all configuration and start fresh
  autoclean-eeg setup --reset
  
  # Show user config location
  autoclean-eeg config show
  
  # Compliance and audit features
  autoclean-eeg export-access-log --format csv --start-date 2025-01-01
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process EEG data")

    # Positional arguments for simple usage: autoclean-eeg process TaskName FilePath
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
        "--format",
        type=str,
        default="*.set",
        help="File format glob pattern for directory processing (default: *.set). Examples: '*.raw', '*.edf', '*.set'. Note: '.raw' will be auto-corrected to '*.raw'",
    )
    process_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )
    process_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug output",
    )

    # List tasks command (alias for 'task list')
    list_tasks_parser = subparsers.add_parser(
        "list-tasks", help="List all available tasks"
    )
    list_tasks_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
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

    # List all tasks (replaces old list-tasks command)
    list_all_parser = task_subparsers.add_parser(
        "list", help="List all available tasks"
    )
    list_all_parser.add_argument(
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
    setup_parser = subparsers.add_parser("setup", help="Setup or reconfigure workspace")
    setup_parser.add_argument(
        "--compliance-mode",
        action="store_true",
        help="Enable FDA 21 CFR Part 11 compliance mode with Auth0 authentication",
    )
    setup_parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear all previous configuration and start completely fresh",
    )

    # Export access log command
    export_log_parser = subparsers.add_parser(
        "export-access-log",
        help="Export database access log with integrity verification",
    )
    export_log_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: access-log-{timestamp}.json)",
    )
    export_log_parser.add_argument(
        "--format",
        choices=["json", "csv", "human"],
        default="json",
        help="Output format (default: json)",
    )
    export_log_parser.add_argument(
        "--start-date", type=str, help="Start date filter (YYYY-MM-DD format)"
    )
    export_log_parser.add_argument(
        "--end-date", type=str, help="End date filter (YYYY-MM-DD format)"
    )
    export_log_parser.add_argument(
        "--operation", type=str, help="Filter by operation type"
    )
    export_log_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify integrity, don't export data",
    )
    export_log_parser.add_argument(
        "--database",
        type=Path,
        help="Path to database file (default: auto-detect from workspace)",
    )

    # Export encrypted outputs command
    export_outputs_parser = subparsers.add_parser(
        "export",
        help="Export encrypted outputs from compliance mode (requires authentication)",
    )
    export_outputs_parser.add_argument(
        "task",
        type=str,
        help="Task name to export outputs for (e.g., 'RestingEyesOpen')",
    )
    export_outputs_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory or file path (default: ./exported_{task})",
    )
    export_outputs_parser.add_argument(
        "--format",
        choices=["directory", "zip"],
        default="directory",
        help="Export format: directory structure or ZIP archive (default: directory)",
    )
    export_outputs_parser.add_argument(
        "--type",
        "-t",
        choices=["all", "reports", "plots", "metadata", "logs"],
        default="all",
        help="Output types to export (default: all)",
    )
    export_outputs_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data integrity during export",
    )
    export_outputs_parser.add_argument(
        "--database",
        type=Path,
        help="Path to database file (default: auto-detect from workspace)",
    )

    # Authentication commands (for compliance mode)
    subparsers.add_parser("login", help="Login to Auth0 for compliance mode")

    subparsers.add_parser("logout", help="Logout and clear authentication tokens")

    subparsers.add_parser("whoami", help="Show current authenticated user")

    # Auth0 configuration command
    auth0_parser = subparsers.add_parser(
        "auth0", help="Configure custom Auth0 application settings"
    )
    auth0_subparsers = auth0_parser.add_subparsers(
        dest="auth0_action", help="Auth0 configuration actions"
    )
    
    # Set custom Auth0 credentials (interactive wizard)
    auth0_subparsers.add_parser(
        "set", help="Set custom Auth0 application credentials (interactive)"
    )
    
    # Show current Auth0 configuration
    auth0_subparsers.add_parser(
        "show", help="Show current Auth0 configuration"
    )
    
    # Reset to default Auth0 configuration
    auth0_subparsers.add_parser(
        "reset", help="Reset to default AutoClean Auth0 configuration"
    )
    
    # Test Auth0 configuration
    test_auth0_parser = auth0_subparsers.add_parser(
        "test", help="Test Auth0 configuration and connectivity"
    )
    test_auth0_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed test information"
    )

    auth_diag_parser = subparsers.add_parser(
        "auth0-diagnostics", help="Diagnose Auth0 configuration and connectivity issues"
    )
    auth_diag_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed diagnostic information",
    )

    # Version command
    subparsers.add_parser("version", help="Show version information")

    # Help command (for consistency)
    subparsers.add_parser("help", help="Show detailed help information")

    # Tutorial command
    subparsers.add_parser(
        "tutorial", help="Show a helpful tutorial for first-time users"
    )

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

        # Check input exists - with fallback to task config
        if input_path and not input_path.exists():
            message("error", f"Input path does not exist: {input_path}")
            return False
        elif not input_path:
            # Try to get input_path from task config as fallback
            task_input_path = None
            if task_name:
                from autoclean.utils.task_discovery import extract_config_from_task

                task_input_path = extract_config_from_task(task_name, "input_path")

            if task_input_path:
                input_path = Path(task_input_path)
                if not input_path.exists():
                    message(
                        "error",
                        f"Input path from task config does not exist: {input_path}",
                    )
                    return False
                message("info", f"Using input path from task config: {input_path}")
            else:
                message(
                    "error",
                    "Input file or directory must be specified (via CLI or task config)",
                )
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

        # Initialize pipeline with verbose logging if requested
        pipeline_kwargs = {"output_dir": args.output}
        if args.verbose:
            pipeline_kwargs["verbose"] = "debug"

        pipeline = Pipeline(**pipeline_kwargs)

        # Add Python task file if provided
        if args.task_file:
            task_name = pipeline.add_task(args.task_file)
            message("info", f"Loaded Python task: {task_name}")
        else:
            task_name = args.final_task

            # Check if this is a custom task using the new discovery system
            from autoclean.utils.task_discovery import get_task_by_name

            task_class = get_task_by_name(task_name)
            if task_class:
                # Task found via discovery system
                message("info", f"Loaded task: {task_name}")
            else:
                # Fall back to old method for compatibility
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
            if args.final_input.is_dir():
                message("info", f"File format: {args.format}")
                if args.recursive:
                    message("info", "Recursive search: enabled")
            return 0

        # Process files
        if args.final_input.is_file():
            message("info", f"Processing single file: {args.final_input}")
            pipeline.process_file(file_path=args.final_input, task=task_name)
        else:
            message("info", f"Processing directory: {args.final_input}")
            message("info", f"Using file format: {args.format}")
            if args.recursive:
                message("info", "Recursive search: enabled")
            pipeline.process_directory(
                directory=args.final_input,
                task=task_name,
                pattern=args.format,
                recursive=args.recursive,
            )

        message("info", "Processing completed successfully!")
        return 0

    except Exception as e:
        message("error", f"Processing failed: {str(e)}")
        return 1


def cmd_list_tasks(_args) -> int:
    """Execute the list-tasks command."""
    try:
        from pathlib import Path

        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        from autoclean.utils.task_discovery import safe_discover_tasks

        console = Console()

        valid_tasks, invalid_files = safe_discover_tasks()

        console.print("\n[bold]Available Processing Tasks[/bold]\n")

        # --- Built-in Tasks ---
        built_in_tasks = [
            task for task in valid_tasks if "autoclean/tasks" in task.source
        ]
        if built_in_tasks:
            built_in_table = Table(
                show_header=True, header_style="bold blue", box=None, padding=(0, 1)
            )
            built_in_table.add_column("Task Name", style="cyan", no_wrap=True)
            built_in_table.add_column("Module", style="dim")
            built_in_table.add_column("Description", style="dim", max_width=50)

            for task in sorted(built_in_tasks, key=lambda x: x.name):
                # Extract just the module name from the full path
                module_name = Path(task.source).stem
                built_in_table.add_row(
                    task.name, module_name + ".py", task.description or "No description"
                )

            built_in_panel = Panel(
                built_in_table,
                title="[bold]Built-in Tasks[/bold]",
                border_style="blue",
                padding=(1, 1),
            )
            console.print(built_in_panel)
        else:
            console.print(
                Panel(
                    "[dim]No built-in tasks found[/dim]",
                    title="[bold]Built-in Tasks[/bold]",
                    border_style="blue",
                    padding=(1, 1),
                )
            )

        # --- Custom Tasks ---
        custom_tasks = [
            task for task in valid_tasks if "autoclean/tasks" not in task.source
        ]
        if custom_tasks:
            custom_table = Table(
                show_header=True, header_style="bold magenta", box=None, padding=(0, 1)
            )
            custom_table.add_column("Task Name", style="magenta", no_wrap=True)
            custom_table.add_column("File", style="dim")
            custom_table.add_column("Description", style="dim", max_width=50)

            for task in sorted(custom_tasks, key=lambda x: x.name):
                # Show just the filename for custom tasks
                file_name = Path(task.source).name
                custom_table.add_row(
                    task.name, file_name, task.description or "No description"
                )

            custom_panel = Panel(
                custom_table,
                title="[bold]Custom Tasks[/bold]",
                border_style="magenta",
                padding=(1, 1),
            )
            console.print(custom_panel)
        else:
            console.print(
                Panel(
                    "[dim]No custom tasks found.\n"
                    "Use [yellow]autoclean-eeg task add <file.py>[/yellow] to add one.[/dim]",
                    title="[bold]Custom Tasks[/bold]",
                    border_style="magenta",
                    padding=(1, 1),
                )
            )

        # --- Invalid Task Files ---
        if invalid_files:
            invalid_table = Table(
                show_header=True, header_style="bold red", box=None, padding=(0, 1)
            )
            invalid_table.add_column("File", style="red", no_wrap=True)
            invalid_table.add_column("Error", style="yellow", max_width=70)

            for file in invalid_files:
                # Show relative path if in workspace, otherwise just filename
                file_path = Path(file.source)
                if file_path.is_absolute():
                    display_name = file_path.name
                else:
                    display_name = file.source

                invalid_table.add_row(display_name, file.error)

            invalid_panel = Panel(
                invalid_table,
                title="[bold]Invalid Task Files[/bold]",
                border_style="red",
                padding=(1, 1),
            )
            console.print(invalid_panel)

        # Summary statistics
        console.print(
            f"\n[dim]Found {len(valid_tasks)} valid tasks "
            f"({len(built_in_tasks)} built-in, {len(custom_tasks)} custom) "
            f"and {len(invalid_files)} invalid files[/dim]"
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
    """Run the interactive setup wizard."""
    try:
        # Check if reset flag was passed
        if hasattr(args, "reset") and args.reset:
            return _reset_configuration()
        # Check if compliance mode flag was passed
        elif hasattr(args, "compliance_mode") and args.compliance_mode:
            return _setup_compliance_mode()
        else:
            return _run_interactive_setup()
    except Exception as e:
        message("error", f"Setup failed: {str(e)}")
        return 1


def _run_interactive_setup() -> int:
    """Run interactive setup wizard with arrow key navigation."""
    try:
        import inquirer
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        from autoclean.utils.config import (
            get_compliance_status,
        )
        from autoclean.utils.branding import AutoCleanBranding

        console = Console()
        
        # Professional header
        AutoCleanBranding.get_professional_header(console)
        console.print(f"\n{AutoCleanBranding.get_simple_divider()}")
        
        # Setup wizard panel
        wizard_text = Text()
        wizard_text.append("🧠 AutoClean EEG Setup Wizard\n\n", style="bold blue")
        wizard_text.append("Use arrow keys to navigate, Enter to select", style="dim")
        
        console.print(Panel(wizard_text, border_style="blue", padding=(1, 2)))

        # Get current compliance status
        compliance_status = get_compliance_status()
        is_enabled = compliance_status["enabled"]
        is_permanent = compliance_status["permanent"]

        # Show current status
        status_msg = "enabled" if is_enabled else "disabled"
        console.print(f"\n[bold]Current compliance mode:[/bold] [{'green' if is_enabled else 'yellow'}]{status_msg}[/{'green' if is_enabled else 'yellow'}]")

        # Check if compliance mode is permanently enabled
        if is_permanent:
            console.print("[yellow]⚠[/yellow] FDA 21 CFR Part 11 compliance mode is permanently enabled.")
            console.print("[dim]You can only configure workspace location in compliance mode.[/dim]")

            # Only allow workspace configuration
            questions = [
                inquirer.List(
                    "setup_type",
                    message="What would you like to configure?",
                    choices=[
                        ("Configure workspace location", "workspace_only"),
                        ("Exit setup", "exit"),
                    ],
                    default="workspace_only",
                )
            ]
        else:
            # Build setup options based on current state
            choices = [
                ("Basic setup (standard research use)", "basic"),
                ("Just configure workspace location", "workspace_only"),
            ]

            if is_enabled:
                choices.insert(
                    1,
                    (
                        "Disable FDA 21 CFR Part 11 compliance mode",
                        "disable_compliance",
                    ),
                )
            else:
                choices.insert(
                    1,
                    ("Enable FDA 21 CFR Part 11 compliance mode", "enable_compliance"),
                )

            questions = [
                inquirer.List(
                    "setup_type",
                    message="What would you like to configure?",
                    choices=choices,
                    default="workspace_only",
                )
            ]

        answers = inquirer.prompt(questions)
        if not answers:  # User canceled
            message("info", "Setup canceled.")
            return 0

        setup_type = answers["setup_type"]

        if setup_type == "exit":
            message("info", "Setup canceled.")
            return 0
        elif setup_type == "workspace_only":
            user_config.setup_workspace()
            message("success", "✓ Workspace setup complete!")
            return 0
        elif setup_type == "basic":
            # Standard setup
            return _setup_basic_mode()
        elif setup_type == "enable_compliance":
            # Enable compliance mode
            return _enable_compliance_mode()
        elif setup_type == "disable_compliance":
            # Disable compliance mode
            return _disable_compliance_mode()
        elif setup_type == "compliance":
            # Legacy compliance setup (permanent)
            return _setup_compliance_mode()

    except ImportError:
        # Fall back to basic setup if inquirer not available
        message("warning", "Interactive prompts not available. Running basic setup...")
        user_config.setup_workspace()
        return 0
    except KeyboardInterrupt:
        message("info", "\nSetup canceled by user.")
        return 0
    except Exception as e:
        message("error", f"Interactive setup failed: {e}")
        return 1


def _setup_basic_mode() -> int:
    """Setup basic (non-compliance) mode."""
    try:
        import inquirer
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        from autoclean.utils.config import load_user_config, save_user_config

        console = Console()
        
        # Basic setup panel
        basic_text = Text()
        basic_text.append("📋 Basic Setup Configuration\n\n", style="bold blue")
        basic_text.append("Standard research use without compliance requirements", style="blue")
        
        console.print(Panel(basic_text, border_style="blue", padding=(1, 2)))

        console.print("\n[bold]Configuring workspace...[/bold]")

        # Setup workspace first
        try:
            user_config.setup_workspace()
            console.print("[green]✓[/green] Workspace configured")
        except Exception as e:
            console.print(f"[red]✗[/red] Workspace setup failed: {e}")
            return 1

        # Ask about workspace preferences
        questions = [
            inquirer.Confirm(
                "auto_backup",
                message="Enable automatic database backups?",
                default=True,
            ),
        ]

        answers = inquirer.prompt(questions)
        if not answers:
            return 0

        # Update user configuration
        user_config_data = load_user_config()

        # Ensure compliance and workspace are dictionaries
        if not isinstance(user_config_data.get("compliance"), dict):
            user_config_data["compliance"] = {}
        if not isinstance(user_config_data.get("workspace"), dict):
            user_config_data["workspace"] = {}

        user_config_data["compliance"]["enabled"] = False
        user_config_data["workspace"]["auto_backup"] = answers["auto_backup"]

        save_user_config(user_config_data)

        console.print("[green]✓[/green] Basic configuration saved")

        # Success panel with next steps
        success_text = Text()
        success_text.append("✓ Basic setup complete!\n\n", style="bold green")
        success_text.append("Configured:\n", style="green")
        success_text.append("• Workspace directory\n", style="dim")
        success_text.append("• Standard processing mode\n", style="dim")
        backup_status = "enabled" if answers["auto_backup"] else "disabled"
        success_text.append(f"• Database backups ({backup_status})\n\n", style="dim")
        success_text.append("Next steps:\n", style="green")
        success_text.append("Run 'autoclean-eeg process TaskName file.raw' to get started.", style="dim")
        
        console.print(Panel(success_text, border_style="green", padding=(1, 2)))

        return 0

    except ImportError:
        # Fall back without inquirer
        user_config.setup_workspace()
        return 0


def _setup_compliance_mode() -> int:
    """Setup FDA 21 CFR Part 11 compliance mode with developer-managed Auth0."""
    try:
        import inquirer

        from autoclean.utils.auth import get_auth0_manager
        from autoclean.utils.config import load_user_config, save_user_config

        message("info", "\n🔐 FDA 21 CFR Part 11 Compliance Setup")
        message("warning", "⚠️  Once enabled, compliance mode cannot be disabled.")
        message("info", "This mode provides:")
        message("info", "• Mandatory user authentication")
        message("info", "• Tamper-proof audit trails")
        message("info", "• Encrypted data storage")
        message("info", "• Electronic signature support")

        # Confirm user understands permanent nature
        confirm_question = [
            inquirer.Confirm(
                "confirm_permanent",
                message="Do you understand that compliance mode cannot be disabled once enabled?",
                default=False,
            )
        ]

        confirm_answer = inquirer.prompt(confirm_question)
        if not confirm_answer or not confirm_answer["confirm_permanent"]:
            message("info", "Compliance mode setup canceled.")
            return 0

        # Setup Part-11 workspace with suffix
        user_config.setup_part11_workspace()

        # Ask about electronic signatures
        signature_question = [
            inquirer.Confirm(
                "require_signatures",
                message="Require electronic signatures for processing runs?",
                default=True,
            )
        ]

        signature_answer = inquirer.prompt(signature_question)
        if not signature_answer:
            return 0

        # Configure Auth0 manager with developer credentials
        auth_manager = get_auth0_manager()
        auth_manager.configure_developer_auth0()

        # Update user configuration
        user_config_data = load_user_config()

        # Ensure compliance and workspace are dictionaries
        if not isinstance(user_config_data.get("compliance"), dict):
            user_config_data["compliance"] = {}
        if not isinstance(user_config_data.get("workspace"), dict):
            user_config_data["workspace"] = {}

        user_config_data["compliance"]["enabled"] = True
        user_config_data["compliance"]["permanent"] = True  # Cannot be disabled
        user_config_data["compliance"]["auth_provider"] = "auth0"
        user_config_data["compliance"]["require_electronic_signatures"] = (
            signature_answer["require_signatures"]
        )
        user_config_data["workspace"][
            "auto_backup"
        ] = True  # Always enabled for compliance

        save_user_config(user_config_data)

        message("success", "✓ Compliance mode setup complete!")
        message("info", "\nNext steps:")
        message("info", "1. Run 'autoclean-eeg login' to authenticate")
        message("info", "2. Use 'autoclean-eeg whoami' to check authentication status")
        message(
            "info",
            "3. All processing will now include audit trails and user authentication",
        )

        return 0

    except ImportError:
        message("error", "Interactive setup requires 'inquirer' package.")
        message("info", "Install with: pip install inquirer")
        return 1
    except Exception as e:
        message("error", f"Compliance setup failed: {e}")
        return 1


def _enable_compliance_mode() -> int:
    """Enable FDA 21 CFR Part 11 compliance mode (non-permanent)."""
    try:
        import inquirer
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        from autoclean.utils.auth import get_auth0_manager
        from autoclean.utils.config import enable_compliance_mode

        console = Console()
        
        # Compliance mode info panel
        compliance_text = Text()
        compliance_text.append("🔐 Enable FDA 21 CFR Part 11 Compliance Mode\n\n", style="bold blue")
        compliance_text.append("This mode provides:\n", style="blue")
        compliance_text.append("• User authentication (when processing)\n", style="dim")
        compliance_text.append("• Audit trails\n", style="dim")
        compliance_text.append("• Electronic signature support\n", style="dim")
        compliance_text.append("• Can be disabled later", style="dim")
        
        console.print(Panel(compliance_text, border_style="blue", padding=(1, 2)))

        # Confirm enabling
        confirm_question = [
            inquirer.Confirm(
                "confirm_enable", message="Enable compliance mode?", default=True
            )
        ]

        confirm_answer = inquirer.prompt(confirm_question)
        if not confirm_answer or not confirm_answer["confirm_enable"]:
            console.print("[yellow]ℹ[/yellow] Compliance mode not enabled.")
            return 0

        console.print("\n[bold]Configuring compliance mode...[/bold]")

        # Enable compliance mode first (so workspace gets Part 11 suffix)
        if enable_compliance_mode(permanent=False):
            console.print("[green]✓[/green] Compliance mode enabled")
        else:
            console.print("[red]✗[/red] Failed to enable compliance mode")
            return 1

        # Setup Part 11 workspace (non-interactive, direct setup)
        try:
            part11_workspace = user_config.setup_part11_workspace()
            console.print(f"[green]✓[/green] Part 11 workspace configured: [dim]{part11_workspace}[/dim]")
        except Exception as e:
            console.print(f"[red]✗[/red] Part 11 workspace setup failed: {e}")
            return 1

        # Configure Auth0 manager with developer credentials
        try:
            auth_manager = get_auth0_manager()
            auth_manager.configure_developer_auth0()
            console.print("[green]✓[/green] Auth0 configured")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Auth0 configuration failed: {e}")
            console.print("[dim]You can configure Auth0 later for authentication[/dim]")

        # All configuration complete - show success panel
        success_text = Text()
        success_text.append("✓ Compliance mode setup complete!\n\n", style="bold green")
        success_text.append("Configured:\n", style="green")
        success_text.append("• Part 11 workspace directory\n", style="dim")
        success_text.append("• FDA 21 CFR Part 11 compliance mode\n", style="dim")
        success_text.append("• Auth0 authentication\n\n", style="dim")
        success_text.append("Next steps:\n", style="green")
        success_text.append("1. Run 'autoclean-eeg login' to authenticate (when needed)\n", style="dim")
        success_text.append("2. Run 'autoclean-eeg setup' again to disable if needed", style="dim")
        
        console.print(Panel(success_text, border_style="green", padding=(1, 2)))
        return 0

    except ImportError:
        message("error", "Interactive setup requires 'inquirer' package.")
        return 1
    except Exception as e:
        message("error", f"Failed to enable compliance mode: {e}")
        return 1


def _disable_compliance_mode() -> int:
    """Disable FDA 21 CFR Part 11 compliance mode."""
    try:
        import inquirer
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        from autoclean.utils.config import (
            disable_compliance_mode,
            get_compliance_status,
        )

        console = Console()

        # Disable compliance mode panel
        disable_text = Text()
        disable_text.append("🔓 Disable FDA 21 CFR Part 11 Compliance Mode\n\n", style="bold yellow")
        disable_text.append("This will:\n", style="yellow")
        disable_text.append("• Switch to regular workspace (without Part 11 suffix)\n", style="dim")
        disable_text.append("• Disable required authentication\n", style="dim")
        disable_text.append("• Disable audit trail logging\n", style="dim")
        disable_text.append("• Disable electronic signatures", style="dim")
        
        console.print(Panel(disable_text, border_style="yellow", padding=(1, 2)))

        # Check if permanent
        compliance_status = get_compliance_status()
        if compliance_status["permanent"]:
            console.print("[red]✗[/red] Cannot disable permanently enabled compliance mode")
            return 1

        # Confirm disabling
        confirm_question = [
            inquirer.Confirm(
                "confirm_disable",
                message="Are you sure you want to disable compliance mode?",
                default=False,
            )
        ]

        confirm_answer = inquirer.prompt(confirm_question)
        if not confirm_answer or not confirm_answer["confirm_disable"]:
            console.print("[green]ℹ[/green] Compliance mode remains enabled.")
            return 0

        console.print("\n[bold]Disabling compliance mode...[/bold]")

        # Disable compliance mode first
        if disable_compliance_mode():
            console.print("[green]✓[/green] Compliance mode disabled")
        else:
            console.print("[red]✗[/red] Failed to disable compliance mode")
            return 1

        # Switch back to regular workspace (without Part 11 suffix)
        try:
            # Get the regular workspace path (without Part 11 suffix)
            regular_workspace = user_config._get_base_workspace_path()
            
            # Check if regular workspace exists, if not create it
            if not regular_workspace.exists() or not (regular_workspace / "tasks").exists():
                console.print("[yellow]ℹ[/yellow] Setting up regular workspace...")
                user_config._create_workspace_structure(regular_workspace)
                console.print(f"[green]✓[/green] Regular workspace created: [dim]{regular_workspace}[/dim]")
            else:
                console.print(f"[green]✓[/green] Switched to regular workspace: [dim]{regular_workspace}[/dim]")
            
            # Update global config to point to regular workspace
            user_config._save_global_config(regular_workspace)
            
            # Update user_config instance
            user_config.config_dir = regular_workspace
            user_config.tasks_dir = regular_workspace / "tasks"
            
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Could not switch workspace: {e}")
            console.print("[dim]You may need to run 'autoclean-eeg setup' to reconfigure workspace[/dim]")

        # Success panel
        success_text = Text()
        success_text.append("✓ Compliance mode disabled!\n\n", style="bold green")
        success_text.append("Changes made:\n", style="green")
        success_text.append("• Switched to regular workspace (without Part 11 suffix)\n", style="dim")
        success_text.append("• Disabled authentication requirements\n", style="dim")
        success_text.append("• Disabled audit trail logging\n", style="dim")
        success_text.append("• Disabled electronic signatures\n\n", style="dim")
        success_text.append("AutoClean will now operate in standard mode", style="green")
        
        console.print(Panel(success_text, border_style="green", padding=(1, 2)))
        return 0

    except ImportError:
        message("error", "Interactive setup requires 'inquirer' package.")
        return 1
    except Exception as e:
        message("error", f"Failed to disable compliance mode: {e}")
        return 1


# FUTURE FEATURE: User-managed Auth0 setup (commented out for now)
# def _setup_compliance_mode_user_managed() -> int:
#     """Setup FDA 21 CFR Part 11 compliance mode with user-managed Auth0."""
#     try:
#         import inquirer
#         from autoclean.utils.config import load_user_config, save_user_config
#         from autoclean.utils.auth import get_auth0_manager, validate_auth0_config
#
#         message("info", "\n🔐 FDA 21 CFR Part 11 Compliance Setup")
#         message("warning", "This mode requires Auth0 account and application setup.")
#
#         # Setup workspace first
#         user_config.setup_workspace()
#
#         # Explain Auth0 requirements
#         message("info", "\nAuth0 Application Setup Instructions:")
#         message("info", "1. Create an Auth0 account at https://auth0.com")
#         message("info", "2. Go to Applications > Create Application")
#         message("info", "3. Choose 'Native' as the application type (for CLI apps)")
#         message("info", "4. In your application settings, configure:")
#         message("info", "   - Allowed Callback URLs: http://localhost:8080/callback")
#         message("info", "   - Allowed Logout URLs: http://localhost:8080/logout")
#         message("info", "   - Grant Types: Authorization Code, Refresh Token (default for Native)")
#         message("info", "5. Copy your Domain, Client ID, and Client Secret")
#         message("info", "6. Your domain will be something like: your-tenant.us.auth0.com\n")
#
#         # Confirm user is ready
#         ready_question = [
#             inquirer.Confirm(
#                 'auth0_ready',
#                 message="Do you have your Auth0 application configured and credentials ready?",
#                 default=False
#             )
#         ]
#
#         ready_answer = inquirer.prompt(ready_question)
#         if not ready_answer or not ready_answer['auth0_ready']:
#             message("info", "Please set up your Auth0 application first, then run:")
#             message("info", "autoclean setup --compliance-mode")
#             return 0
#
#         # Get Auth0 configuration
#         auth_questions = [
#             inquirer.Text(
#                 'domain',
#                 message="Auth0 Domain (e.g., your-tenant.auth0.com)",
#                 validate=lambda _, x: len(x) > 0 and '.auth0.com' in x
#             ),
#             inquirer.Text(
#                 'client_id',
#                 message="Auth0 Client ID",
#                 validate=lambda _, x: len(x) > 0
#             ),
#             inquirer.Password(
#                 'client_secret',
#                 message="Auth0 Client Secret",
#                 validate=lambda _, x: len(x) > 0
#             ),
#             inquirer.Confirm(
#                 'require_signatures',
#                 message="Require electronic signatures for processing runs?",
#                 default=True
#             )
#         ]
#
#         auth_answers = inquirer.prompt(auth_questions)
#         if not auth_answers:
#             return 0
#
#         # Validate Auth0 configuration
#         message("info", "Validating Auth0 configuration...")
#
#         is_valid, error_msg = validate_auth0_config(
#             auth_answers['domain'],
#             auth_answers['client_id'],
#             auth_answers['client_secret']
#         )
#
#         if not is_valid:
#             message("error", f"Auth0 configuration invalid: {error_msg}")
#             return 1
#
#         message("success", "✓ Auth0 configuration validated!")
#
#         # Configure Auth0 manager
#         auth_manager = get_auth0_manager()
#         auth_manager.configure_auth0(
#             auth_answers['domain'],
#             auth_answers['client_id'],
#             auth_answers['client_secret']
#         )
#
#         # Update user configuration
#         user_config_data = load_user_config()
#
#         # Ensure compliance and workspace are dictionaries
#         if not isinstance(user_config_data.get('compliance'), dict):
#             user_config_data['compliance'] = {}
#         if not isinstance(user_config_data.get('workspace'), dict):
#             user_config_data['workspace'] = {}
#
#         user_config_data['compliance']['enabled'] = True
#         user_config_data['compliance']['auth_provider'] = 'auth0'
#         user_config_data['compliance']['require_electronic_signatures'] = auth_answers['require_signatures']
#         user_config_data['workspace']['auto_backup'] = True  # Always enabled for compliance
#
#         save_user_config(user_config_data)
#
#         message("success", "✓ Compliance mode setup complete!")
#         message("info", "\nNext steps:")
#         message("info", "1. Run 'autoclean login' to authenticate")
#         message("info", "2. Use 'autoclean whoami' to check authentication status")
#         message("info", "3. All processing will now include audit trails and user authentication")
#
#         return 0
#
#     except ImportError:
#         message("error", "Interactive setup requires 'inquirer' package.")
#         message("info", "Install with: pip install inquirer")
#         return 1
#     except Exception as e:
#         message("error", f"Compliance setup failed: {e}")
#         return 1


def _reset_configuration() -> int:
    """Reset all AutoClean configuration and start completely fresh."""
    try:
        from pathlib import Path
        import shutil
        from platformdirs import user_config_dir
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        
        from autoclean.utils.config import load_user_config
        from autoclean.utils.branding import AutoCleanBranding
        
        console = Console()
        
        # Professional header
        AutoCleanBranding.get_professional_header(console)
        console.print(f"\n{AutoCleanBranding.get_simple_divider()}")
        
        # Reset warning panel
        warning_text = Text()
        warning_text.append("🔄 Configuration Reset\n\n", style="bold yellow")
        warning_text.append("This will completely clear all AutoClean configuration and workspace settings.\n", style="yellow")
        warning_text.append("You will start fresh as if AutoClean was just installed.", style="yellow")
        
        console.print(Panel(warning_text, border_style="yellow", padding=(1, 2)))
        
        # Get current configuration paths
        global_config_dir = Path(user_config_dir("autoclean", "autoclean"))
        workspace_config = load_user_config()
        current_workspace = None
        
        # Try to get current workspace path
        global_config_file = global_config_dir / "setup.json"
        if global_config_file.exists():
            try:
                import json
                with open(global_config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    current_workspace = Path(config.get("config_directory", ""))
            except Exception:
                pass
        
        # Show what will be reset (excluding workspace)
        console.print("\n[bold]The following will be cleared:[/bold]")
        console.print(f"  📁 Global config: [dim]{global_config_dir}[/dim]")
        console.print(f"  🔑 Authentication tokens and settings")
        console.print(f"  ⚙️  All user preferences and configuration")
        
        # Confirm reset
        try:
            console.print()
            response = input("Type 'yes' to confirm complete reset: ").strip().lower()
            if response != "yes":
                console.print("[green]✓[/green] Reset canceled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            console.print("\n[green]✓[/green] Reset canceled.")
            return 0
        
        reset_successful = True
        
        console.print("\n[bold]Clearing configuration...[/bold]")
        
        # 1. Clear global AutoClean config directory
        if global_config_dir.exists():
            try:
                shutil.rmtree(global_config_dir)
                console.print(f"[green]✓[/green] Cleared global configuration directory")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to clear global config: {e}")
                reset_successful = False
        
        # 2. Clear any Auth0 tokens (in case they're stored separately)
        try:
            from autoclean.utils.auth import get_auth0_manager
            auth_manager = get_auth0_manager()
            auth_manager.logout()  # This clears tokens
            console.print(f"[green]✓[/green] Cleared authentication tokens")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Note: Could not clear auth tokens: {e}")
        
        # 3. Handle workspace directory (if it exists and user confirms)
        if current_workspace and current_workspace.exists():
            try:
                console.print(f"\n[yellow]Workspace directory found:[/yellow] [dim]{current_workspace}[/dim]")
                console.print("[dim]This contains your custom tasks, output, and configuration.[/dim]")
                workspace_response = input("Also delete workspace directory? (y/N): ").strip().lower()
                if workspace_response in ["y", "yes"]:
                    shutil.rmtree(current_workspace)
                    console.print(f"[green]✓[/green] Deleted workspace directory")
                else:
                    console.print(f"[green]✓[/green] Kept workspace directory")
            except (EOFError, KeyboardInterrupt):
                console.print(f"[green]✓[/green] Kept workspace directory")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to delete workspace: {e}")
                reset_successful = False
        
        # Final status
        console.print()
        if reset_successful:
            success_text = Text()
            success_text.append("🎉 Configuration reset complete!\n\n", style="bold green")
            success_text.append("AutoClean is now in a fresh state.\n", style="green")
            success_text.append("Run 'autoclean-eeg setup' to begin initial configuration.", style="dim")
            
            console.print(Panel(success_text, border_style="green", padding=(1, 2)))
            return 0
        else:
            error_text = Text()
            error_text.append("❌ Reset completed with some errors.\n\n", style="bold red")
            error_text.append("You may need to manually clean up remaining files.", style="red")
            
            console.print(Panel(error_text, border_style="red", padding=(1, 2)))
            return 1
            
    except Exception as e:
        message("error", f"Reset failed: {e}")
        return 1


def cmd_version(args) -> int:
    """Show version information."""
    try:
        from rich.console import Console

        from autoclean import __version__
        from autoclean.utils.branding import AutoCleanBranding
        from autoclean.utils.user_config import UserConfigManager

        console = Console()

        # Professional header consistent with setup
        AutoCleanBranding.get_professional_header(console)
        console.print(f"\n{AutoCleanBranding.get_simple_divider()}")

        console.print("\n[bold]Version Information:[/bold]")
        console.print(f"  🏷️  [bold]{__version__}[/bold]")

        # Include system information for troubleshooting
        console.print("\n[bold]System Information:[/bold]")
        temp_config = UserConfigManager()
        temp_config._display_system_info(console)

        console.print(f"\n[dim]{AutoCleanBranding.TAGLINE}[/dim]")

        # GitHub and support info
        console.print("\n[bold]GitHub Repository:[/bold]")
        console.print(
            "  [blue]https://github.com/cincibrainlab/autoclean_pipeline[/blue]"
        )
        console.print("  [dim]Report issues, contribute, or get help[/dim]")

        return 0
    except ImportError:
        print("AutoClean EEG (version unknown)")
        return 0


def cmd_task(args) -> int:
    """Execute task management commands."""
    if args.task_action == "add":
        return cmd_task_add(args)
    elif args.task_action == "remove":
        return cmd_task_remove(args)
    elif args.task_action == "list":
        return cmd_list_tasks(args)
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
        print(f"  autoclean-eeg process {task_name} <data_file>")

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


def cmd_config_show(_args) -> int:
    """Show user configuration directory."""
    config_dir = user_config.config_dir
    message("info", f"User configuration directory: {config_dir}")

    custom_tasks = user_config.list_custom_tasks()
    print(f"  • Custom tasks: {len(custom_tasks)}")
    print(f"  • Tasks directory: {config_dir / 'tasks'}")
    print(f"  • Config file: {config_dir / 'user_config.json'}")

    return 0


def cmd_config_setup(_args) -> int:
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


def cmd_help(_args) -> int:
    """Show elegant, user-friendly help information."""
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from autoclean.utils.branding import AutoCleanBranding

    console = Console()

    # Professional header with branding
    AutoCleanBranding.get_professional_header(console)
    # console.print(f"\n{AutoCleanBranding.get_simple_divider()}")

    # Main help sections organized for new users
    console.print("\n[bold bright_green]🚀 Getting Started[/bold bright_green]")
    console.print(
        "  [bright_yellow]autoclean-eeg setup[/bright_yellow]     [dim]→[/dim] Configure your workspace (run this first!)"
    )
    console.print(
        "  [bright_yellow]autoclean-eeg version[/bright_yellow]   [dim]→[/dim] Check system information"
    )

    # Core workflow - Processing
    console.print("\n[bold bright_blue]⚡ Process EEG Data[/bold bright_blue]")

    # Simple usage examples
    simple_panel = Panel(
        "[green]autoclean-eeg process RestingEyesOpen data.raw[/green]\n"
        "[green]autoclean-eeg process MMN data_folder/[/green]\n"
        "[green]autoclean-eeg process ASSR experiment.edf[/green]\n\n"
        "[dim]Built-in tasks: RestingEyesOpen, RestingEyesClosed, MMN, ASSR, Chirp[/dim]",
        title="[bold]Simple Processing[/bold]",
        border_style="green",
        padding=(0, 1),
    )

    # Advanced usage examples
    advanced_panel = Panel(
        "[yellow]autoclean-eeg process --task RestingEyesOpen \\[/yellow]\n"
        "[yellow]  --file data.raw --output results/[/yellow]\n\n"
        "[yellow]autoclean-eeg process --task-file my_task.py \\[/yellow]\n"
        '[yellow]  --dir data/ --format "*.raw"[/yellow]\n\n'
        "[dim]Specify custom output directories and file formats[/dim]",
        title="[bold]Advanced Options[/bold]",
        border_style="yellow",
        padding=(0, 1),
    )

    console.print(Columns([simple_panel, advanced_panel], equal=True, expand=True))

    # Part11 compliance section
    console.print(
        "\n[bold bright_red]🔒 FDA 21 CFR Part 11 Compliance[/bold bright_red]"
    )

    compliance_panel = Panel(
        "[red]setup --compliance-mode[/red]  [dim]Enable regulatory compliance mode[/dim]\n"
        "[red]login[/red]                    [dim]Authenticate with Auth0[/dim]\n"
        "[red]logout[/red]                   [dim]Clear authentication[/dim]\n"
        "[red]whoami[/red]                   [dim]Show current user status[/dim]\n"
        "[red]export[/red]                  [dim]Export encrypted outputs (compliance mode)[/dim]",
        "[red]export-access-log[/red]        [dim]Export audit trail[/dim]",
        title="[bold]Regulated Environments Only[/bold]",
        border_style="red",
        padding=(0, 1),
    )

    console.print(compliance_panel)

    # Task management workflow
    console.print("\n[bold bright_magenta]📋 Task Management[/bold bright_magenta]")

    task_manage_panel = Panel(
        "[cyan]task add my_task.py[/cyan]  [dim]Add custom task file[/dim]\n"
        "[cyan]task remove MyTask[/cyan]   [dim]Remove custom task[/dim]\n"
        "[cyan]task list[/cyan]          [dim]Show available tasks[/dim]",
        title="[bold]Tasks Commands[/bold]",
        border_style="cyan",
        padding=(0, 1),
    )

    console.print(task_manage_panel)

    # Quick reference table
    console.print("\n[bold]📖 Quick Reference[/bold]")

    ref_table = Table(
        show_header=True, header_style="bold blue", box=None, padding=(0, 1)
    )
    ref_table.add_column("Command", style="cyan", no_wrap=True)
    ref_table.add_column("Purpose", style="dim")
    ref_table.add_column("Example", style="green")

    ref_table.add_row("process", "Process EEG data", "process RestingEyesOpen data.raw")
    ref_table.add_row("task", "Manage custom tasks", "task add my_task.py")
    ref_table.add_row("list-tasks", "Show available tasks", "list-tasks")
    ref_table.add_row("config", "Manage settings", "config show")
    ref_table.add_row("review", "Review results", "review --output results/")
    ref_table.add_row("setup", "Configure workspace", "setup")
    ref_table.add_row("version", "System information", "version")

    console.print(ref_table)

    # Help tips
    console.print("\n[bold]💡 Pro Tips[/bold]")
    console.print(
        "  • Get command-specific help: [bright_white]autoclean-eeg <command> --help[/bright_white]"
    )
    console.print(
        "  • Process entire directories: [bright_white]autoclean-eeg process TaskName folder/[/bright_white]"
    )
    console.print(
        "  • Create custom tasks: Save Python task files and add with [bright_white]task add[/bright_white]"
    )
    console.print(
        "  • Run [bright_white]setup[/bright_white] first to configure your workspace"
    )

    # Support section
    console.print("\n[bold]🤝 Support & Community[/bold]")
    console.print("  [blue]https://github.com/cincibrainlab/autoclean_pipeline[/blue]")
    console.print("  [dim]Report issues • Documentation • Contribute • Get help[/dim]")

    return 0


def cmd_tutorial(_args) -> int:
    """Show a helpful tutorial for first-time users."""
    from rich.console import Console

    from autoclean.utils.branding import AutoCleanBranding

    console = Console()

    # Use the tutorial header for consistent branding
    AutoCleanBranding.print_tutorial_header(console)

    console.print(
        "\n[bold bright_green]🚀 Welcome to the AutoClean EEG Tutorial![/bold bright_green]"
    )
    console.print(
        "This tutorial will walk you through the basics of using AutoClean EEG."
    )
    console.print(
        "\n[bold bright_yellow]Step 1: Setup your workspace[/bold bright_yellow]"
    )
    console.print(
        "The first step is to set up your workspace. This is where AutoClean EEG will store its configuration and any custom tasks you create."
    )
    console.print("To do this, run the following command:")
    console.print("\n[green]autoclean-eeg setup[/green]\n")

    console.print(
        "\n[bold bright_yellow]Step 2: List available tasks[/bold bright_yellow]"
    )
    console.print(
        "Once your workspace is set up, you can see the built-in processing tasks that are available."
    )
    console.print("To do this, run the following command:")
    console.print("\n[green]autoclean-eeg task list[/green]\n")

    console.print("\n[bold bright_yellow]Step 3: Process a file[/bold bright_yellow]")
    console.print(
        "Now you are ready to process a file. You will need to specify the task you want to use and the path to the file you want to process."
    )
    console.print(
        "For example, to process a file called 'data.raw' with the 'RestingEyesOpen' task, you would run the following command:"
    )
    console.print("\n[green]autoclean-eeg process RestingEyesOpen data.raw[/green]\n")

    return 0


def cmd_export_access_log(args) -> int:
    """Export database access log with integrity verification."""
    try:
        import csv
        import json
        import sqlite3
        from datetime import datetime

        from autoclean.utils.audit import verify_access_log_integrity
        from autoclean.utils.database import DB_PATH
        from autoclean.utils.user_config import user_config

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
                            for output_dir in sorted(
                                potential_outputs.iterdir(), reverse=True
                            ):
                                if output_dir.is_dir():
                                    potential_db = output_dir / "pipeline.db"
                                    if potential_db.exists():
                                        db_path = potential_db
                                        break
                            else:
                                message(
                                    "error",
                                    "No database found in workspace directory, output directory, or output subdirectories",
                                )
                                return 1
                        else:
                            message(
                                "error",
                                "No database found in workspace directory and no output directory exists",
                            )
                            return 1
            else:
                message(
                    "error", "No workspace configured and no database path provided"
                )
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
                        "operation": args.operation,
                    },
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
                            user = user_ctx.get(
                                "user", user_ctx.get("username", "unknown")
                            )
                            host = user_ctx.get(
                                "host", user_ctx.get("hostname", "unknown")
                            )
                            f.write(f"  User: {user}\n")
                            f.write(f"  Host: {host}\n")

                    if entry.get("details") and entry["details"]:
                        f.write(
                            f"  Details: {json.dumps(entry['details'], indent=4)}\n"
                        )

                    f.write(f"  Hash: {entry['log_hash'][:16]}...\n")
                    f.write("\n")

        message("success", f"✓ Access log exported to: {output_file}")
        message("info", f"Format: {args.format}, Entries: {len(export_data)}")

        return 0

    except Exception as e:
        message("error", f"Failed to export access log: {e}")
        return 1


def cmd_export_outputs(args) -> int:
    """Export encrypted outputs by task name from compliance mode (requires authentication)."""
    try:
        import json
        import zipfile
        from datetime import datetime
        from pathlib import Path

        from autoclean.utils.auth import get_auth0_manager, is_compliance_mode_enabled
        from autoclean.utils.database import get_encrypted_outputs, get_encrypted_output_data, get_run_record, set_database_path
        from autoclean.utils.encryption import get_encryption_manager
        from autoclean.utils.logging import message
        from autoclean.utils.user_config import user_config

        # Check compliance mode
        if not is_compliance_mode_enabled():
            message("error", "Export command only available in compliance mode")
            message("info", "Run 'autoclean-eeg setup --compliance-mode' to enable compliance mode")
            return 1

        # Require authentication
        auth_manager = get_auth0_manager()
        if not auth_manager.is_authenticated():
            message("error", "Authentication required to export encrypted outputs")
            message("info", "Run 'autoclean-eeg login' to authenticate")
            return 1

        # Get workspace directory and initialize database path
        workspace_dir = user_config._get_workspace_path()
        output_dir = user_config.get_default_output_dir()
        set_database_path(output_dir)

        # Find runs by task name
        from autoclean.utils.database import get_runs_by_task
        
        runs = get_runs_by_task(args.task)
        if not runs:
            message("error", f"No runs found for task '{args.task}'")
            return 1

        # Get all encrypted outputs for all runs of this task
        all_outputs = []
        run_ids = []
        for run in runs:
            run_id = run["run_id"]
            run_ids.append(run_id)
            outputs = get_encrypted_outputs(run_id)
            if outputs:
                # Add run info to each output for context
                for output in outputs:
                    output["run_info"] = {
                        "run_id": run_id,
                        "created_at": run.get("created_at"),
                        "success": run.get("success", False)
                    }
                all_outputs.extend(outputs)

        if not all_outputs:
            message("warning", f"No encrypted outputs found for task '{args.task}'")
            return 0
        
        outputs = all_outputs

        # Filter outputs by type if specified
        if args.type != "all":
            type_mapping = {
                "reports": ["report_pdf"],
                "plots": ["plot_png", "plot_jpg"],
                "metadata": ["metadata_json"],
                "logs": ["processing_log"]
            }
            allowed_types = type_mapping.get(args.type, [])
            outputs = [o for o in outputs if o["output_type"] in allowed_types]

        if not outputs:
            message("warning", f"No outputs of type '{args.type}' found for run {args.run_id}")
            return 0

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"exported_{args.task}")

        # Create export directory or prepare ZIP
        if args.format == "directory":
            output_path.mkdir(parents=True, exist_ok=True)
            export_dir = output_path
        else:  # ZIP format
            if not output_path.suffix:
                output_path = output_path.with_suffix(".zip")
            export_dir = output_path.parent / f"temp_{args.task}"
            export_dir.mkdir(parents=True, exist_ok=True)

        # Get encryption manager
        encryption_manager = get_encryption_manager()

        # Export outputs
        exported_files = []
        for output in outputs:
            try:
                # Get and decrypt the data
                output_data = get_encrypted_output_data(output["id"])
                if not output_data:
                    message("warning", f"Could not retrieve data for output {output['file_name']}")
                    continue

                encrypted_data = output_data["encrypted_data"]
                decrypted_data = encryption_manager.decrypt_output(encrypted_data, "bytes")
                if decrypted_data is None:
                    message("warning", f"Could not decrypt output {output['file_name']}")
                    continue

                # Create organized directory structure by run
                run_id = output["run_info"]["run_id"]
                run_dir = export_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = run_dir / output["file_name"]

                if output["file_name"].endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                    # Binary files
                    with open(file_path, "wb") as f:
                        f.write(decrypted_data)
                else:
                    # Text files (JSON, CSV, etc.)
                    with open(file_path, "w", encoding="utf-8") as f:
                        if isinstance(decrypted_data, bytes):
                            f.write(decrypted_data.decode("utf-8"))
                        else:
                            f.write(str(decrypted_data))

                exported_files.append(output["file_name"])

                # Verify integrity if requested
                if args.verify:
                    import hashlib
                    file_hash = hashlib.sha256(decrypted_data).hexdigest()
                    if file_hash != output.get("content_hash"):
                        message("warning", f"Integrity check failed for {output['file_name']}")

            except Exception as e:
                message("error", f"Failed to export {output['file_name']}: {e}")

        # Create ZIP if requested
        if args.format == "zip":
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(export_dir)
                        zipf.write(file_path, arcname)

            # Clean up temporary directory
            import shutil
            shutil.rmtree(export_dir)

        # Log export operation for audit trail
        try:
            from autoclean.utils.database import manage_database_conditionally
            audit_details = {
                "task": args.task,
                "run_ids": run_ids,
                "run_count": len(run_ids),
                "output_type": args.type,
                "format": args.format,
                "exported_files": exported_files,
                "file_count": len(exported_files),
                "export_path": str(output_path)
            }
            
            manage_database_conditionally(
                operation="add_access_log",
                run_record={
                    "operation": "export_outputs",
                    "details": audit_details,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            message("warning", f"Failed to log export operation: {e}")

        # Success message
        if args.format == "zip":
            message("success", f"✓ Exported {len(exported_files)} files from task '{args.task}' ({len(run_ids)} runs) to ZIP: {output_path}")
        else:
            message("success", f"✓ Exported {len(exported_files)} files from task '{args.task}' ({len(run_ids)} runs) to directory: {output_path}")

        return 0

    except Exception as e:
        message("error", f"Failed to export outputs: {e}")
        return 1


def cmd_login(args) -> int:
    """Execute the login command."""
    try:
        from autoclean.utils.auth import get_auth0_manager, is_compliance_mode_enabled

        if not is_compliance_mode_enabled():
            message("error", "Compliance mode is not enabled.")
            message(
                "info",
                "Run 'autoclean-eeg setup --compliance-mode' to enable compliance mode and configure Auth0.",
            )
            return 1

        auth_manager = get_auth0_manager()

        # Always refresh configuration with latest credentials from environment/.env
        try:
            message("debug", "Loading Auth0 credentials from environment/.env files...")
            auth_manager.configure_developer_auth0()
        except Exception as e:
            message("error", f"Failed to configure Auth0: {e}")
            message(
                "info",
                "Check your .env file or environment variables for Auth0 credentials.",
            )
            return 1

        # Double-check configuration after loading
        if not auth_manager.is_configured():
            message("error", "Auth0 not configured.")
            message(
                "info",
                "Check your .env file or environment variables for Auth0 credentials.",
            )
            return 1

        if auth_manager.is_authenticated():
            user_info = auth_manager.get_current_user()
            user_email = user_info.get("email", "Unknown") if user_info else "Unknown"
            message("info", f"Already logged in as: {user_email}")
            return 0

        message("info", "Starting Auth0 login process...")

        if auth_manager.login():
            user_info = auth_manager.get_current_user()
            user_email = user_info.get("email", "Unknown") if user_info else "Unknown"
            message("success", f"✓ Login successful! Welcome, {user_email}")

            # Store user in database
            if user_info:
                from autoclean.utils.database import (
                    manage_database_conditionally,
                    set_database_path,
                )
                from autoclean.utils.user_config import user_config

                # Set database path for the operation
                output_dir = user_config.get_default_output_dir()
                output_dir.mkdir(parents=True, exist_ok=True)
                set_database_path(output_dir)

                # Initialize database with all tables (including new auth tables)
                manage_database_conditionally("create_collection")

                user_record = {
                    "auth0_user_id": user_info.get("sub"),
                    "email": user_info.get("email"),
                    "name": user_info.get("name"),
                    "user_metadata": user_info,
                }
                manage_database_conditionally(
                    "store_authenticated_user", user_record
                )

            return 0
        else:
            message("error", "Login failed. Please try again.")
            return 1

    except Exception as e:
        message("error", f"Login error: {e}")
        return 1


def cmd_logout(args) -> int:
    """Execute the logout command."""
    try:
        from autoclean.utils.auth import get_auth0_manager, is_compliance_mode_enabled

        if not is_compliance_mode_enabled():
            message(
                "info", "Compliance mode is not enabled. No authentication to clear."
            )
            return 0

        auth_manager = get_auth0_manager()

        if not auth_manager.is_authenticated():
            message("info", "Not currently logged in.")
            return 0

        user_info = auth_manager.get_current_user()
        user_email = user_info.get("email", "Unknown") if user_info else "Unknown"

        auth_manager.logout()
        message("success", f"✓ Logged out successfully. Goodbye, {user_email}!")

        return 0

    except Exception as e:
        message("error", f"Logout error: {e}")
        return 1


def cmd_whoami(args) -> int:
    """Execute the whoami command."""
    try:
        from autoclean.utils.auth import get_auth0_manager, is_compliance_mode_enabled

        if not is_compliance_mode_enabled():
            message("info", "Compliance mode: Disabled")
            message("info", "Authentication: Not required")
            return 0

        auth_manager = get_auth0_manager()

        if not auth_manager.is_configured():
            message("info", "Compliance mode: Enabled")
            message("info", "Authentication: Not configured")
            message("info", "")
            message("info", "Auth0 Configuration Required:")
            print(auth_manager.get_callback_urls_help())
            message(
                "info",
                "Run 'autoclean-eeg setup --compliance-mode' to configure Auth0.",
            )
            return 0

        if not auth_manager.is_authenticated():
            message("info", "Compliance mode: Enabled")
            message("info", "Authentication: Not logged in")
            message("info", "Run 'autoclean-eeg login' to authenticate.")
            return 0

        user_info = auth_manager.get_current_user()
        if user_info:
            message("info", "Compliance mode: Enabled")
            message("info", "Authentication: Logged in")
            message("info", f"Email: {user_info.get('email', 'Unknown')}")
            message("info", f"Name: {user_info.get('name', 'Unknown')}")
            message("info", f"User ID: {user_info.get('sub', 'Unknown')}")

            # Check token expiration
            if (
                hasattr(auth_manager, "token_expires_at")
                and auth_manager.token_expires_at
            ):

                expires_str = auth_manager.token_expires_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                message("info", f"Token expires: {expires_str}")
        else:
            message("warning", "User information unavailable")

        return 0

    except Exception as e:
        message("error", f"Error checking authentication status: {e}")
        return 1


def cmd_auth0(args) -> int:
    """Execute the auth0 command."""
    try:
        from autoclean.utils.auth import get_auth0_manager, validate_auth0_config
        
        auth_manager = get_auth0_manager()
        
        if not hasattr(args, "auth0_action") or args.auth0_action is None:
            message("error", "No action specified. Use: set, show, reset, or test")
            return 1
            
        if args.auth0_action == "set":
            return _cmd_auth0_set(args, auth_manager)
        elif args.auth0_action == "show":
            return _cmd_auth0_show(args, auth_manager)
        elif args.auth0_action == "reset":
            return _cmd_auth0_reset(args, auth_manager)
        elif args.auth0_action == "test":
            return _cmd_auth0_test(args, auth_manager)
        else:
            message("error", f"Unknown action: {args.auth0_action}")
            return 1
            
    except Exception as e:
        message("error", f"Auth0 command failed: {e}")
        return 1


def _cmd_auth0_set(args, auth_manager) -> int:
    """Set custom Auth0 configuration using interactive wizard."""
    try:
        message("info", "🔧 Auth0 Application Setup Wizard")
        message("info", "")
        message("info", "This will configure AutoClean to use your custom Auth0 application.")
        message("info", "You'll need your Auth0 application credentials from your Auth0 dashboard.")
        message("info", "")
        
        # Get Auth0 domain
        while True:
            domain = input("Auth0 Domain (e.g., your-tenant.us.auth0.com): ").strip()
            if not domain:
                message("error", "Domain is required. Please try again.")
                continue
            if not domain.endswith('.auth0.com'):
                message("error", "Domain should end with '.auth0.com'. Please try again.")
                continue
            break
        
        # Get Client ID
        while True:
            client_id = input("Client ID: ").strip()
            if not client_id:
                message("error", "Client ID is required. Please try again.")
                continue
            if len(client_id) < 16:
                message("error", "Client ID seems too short. Please verify and try again.")
                continue
            break
        
        # Get Client Secret
        while True:
            client_secret = input("Client Secret: ").strip()
            if not client_secret:
                message("error", "Client Secret is required. Please try again.")
                continue
            if len(client_secret) < 32:
                message("error", "Client Secret seems too short. Please verify and try again.")
                continue
            break
        
        # Optional audience
        audience = input(f"Audience (optional, press Enter for default): ").strip()
        if not audience:
            audience = f"https://{domain}/api/v2/"
        
        message("info", "")
        message("info", "Validating configuration...")
        
        # Validate the provided credentials
        from autoclean.utils.auth import validate_auth0_config
        is_valid, error_msg = validate_auth0_config(domain, client_id, client_secret)
        
        if not is_valid:
            message("error", f"Configuration validation failed: {error_msg}")
            return 1
        
        # Configure Auth0 with custom credentials
        auth_manager.configure_auth0(
            domain=domain,
            client_id=client_id,
            client_secret=client_secret,
            audience=audience
        )
        
        message("success", "✓ Custom Auth0 configuration saved!")
        message("info", f"Domain: {domain}")
        message("info", f"Client ID: {client_id}")
        message("info", f"Audience: {audience}")
        message("info", "")
        message("info", "📋 Required callback URLs for your Auth0 application:")
        print(auth_manager.get_callback_urls_help())
        message("info", "")
        message("info", "Copy the callback URLs above and add them to your Auth0 application settings.")
        message("info", "Then run 'autoclean-eeg auth0 test' to verify the setup.")
        
        return 0
        
    except KeyboardInterrupt:
        message("info", "\nAuth0 setup canceled.")
        return 0
    except Exception as e:
        message("error", f"Failed to set Auth0 configuration: {e}")
        return 1


def _cmd_auth0_show(args, auth_manager) -> int:
    """Show current Auth0 configuration."""
    try:
        if not auth_manager.is_configured():
            message("warning", "Auth0 not configured")
            return 1
            
        is_default = auth_manager.is_using_default_config()
        config_source = "Default (embedded)" if is_default else "Custom"
        
        message("info", f"Auth0 Configuration ({config_source}):")
        message("info", f"Domain: {auth_manager.domain}")
        message("info", f"Client ID: {auth_manager.client_id}")
        message("info", f"Client Secret: {auth_manager.client_secret[:8]}..." if auth_manager.client_secret else "Not set")
        message("info", f"Audience: {auth_manager.audience}")
        
        if not is_default:
            message("info", "")
            message("info", "Required callback URLs for your Auth0 application:")
            print(auth_manager.get_callback_urls_help())
        
        return 0
        
    except Exception as e:
        message("error", f"Failed to show Auth0 configuration: {e}")
        return 1


def _cmd_auth0_reset(args, auth_manager) -> int:
    """Reset to default Auth0 configuration."""
    try:
        auth_manager.use_default_auth0_config()
        message("success", "✓ Reset to default AutoClean Auth0 configuration")
        message("info", "You can now use compliance mode with the default settings")
        return 0
        
    except Exception as e:
        message("error", f"Failed to reset Auth0 configuration: {e}")
        return 1


def _cmd_auth0_test(args, auth_manager) -> int:
    """Test Auth0 configuration and connectivity."""
    try:
        import requests
        
        if not auth_manager.is_configured():
            message("error", "Auth0 not configured")
            return 1
            
        message("info", f"Testing Auth0 configuration for domain: {auth_manager.domain}")
        
        # Test 1: Domain connectivity
        try:
            domain_url = f"https://{auth_manager.domain}/.well-known/openid_configuration"
            response = requests.get(domain_url, timeout=10)
            response.raise_for_status()
            
            config_data = response.json()
            message("success", "✓ Auth0 domain is accessible")
            
            if args.verbose:
                message("info", f"Authorization endpoint: {config_data.get('authorization_endpoint', 'Unknown')}")
                message("info", f"Token endpoint: {config_data.get('token_endpoint', 'Unknown')}")
                
        except requests.RequestException as e:
            message("error", f"✗ Auth0 domain connectivity failed: {e}")
            return 1
        
        # Test 2: Client ID format validation
        if len(auth_manager.client_id) >= 16 and auth_manager.client_id.replace("_", "").replace("-", "").isalnum():
            message("success", "✓ Client ID format appears valid")
        else:
            message("warning", "⚠ Client ID format may be invalid")
            
        # Test 3: Client secret format validation  
        if len(auth_manager.client_secret) >= 32:
            message("success", "✓ Client secret length appears valid")
        else:
            message("warning", "⚠ Client secret appears too short")
            
        # Test 4: Port availability for callback
        available_port = auth_manager._find_available_port()
        if available_port:
            message("success", f"✓ Callback port {available_port} is available")
        else:
            start, end = auth_manager.callback_port_range
            message("warning", f"⚠ No available ports in range {start}-{end}")
            
        message("info", "")
        message("info", "Auth0 configuration test completed")
        message("info", f"Use 'autoclean-eeg login' to test full authentication flow")
        
        return 0
        
    except Exception as e:
        message("error", f"Auth0 configuration test failed: {e}")
        return 1


def cmd_auth0_diagnostics(args) -> int:
    """Execute the auth0-diagnostics command."""
    try:
        import os
        from pathlib import Path

        import requests
        from rich.console import Console
        from rich.table import Table

        from autoclean.utils.auth import get_auth0_manager, is_compliance_mode_enabled

        console = Console()

        # Header
        console.print("\n🔍 [bold]Auth0 Configuration Diagnostics[/bold]", style="blue")
        console.print("[dim]Checking Auth0 setup and connectivity...[/dim]\n")

        # 1. Check compliance mode
        compliance_enabled = is_compliance_mode_enabled()
        console.print(
            f"✓ Compliance mode: {'[green]Enabled[/green]' if compliance_enabled else '[yellow]Disabled[/yellow]'}"
        )

        if not compliance_enabled:
            console.print(
                "[yellow]ℹ[/yellow] Auth0 is only used in compliance mode. Run 'autoclean-eeg setup --compliance-mode' to enable."
            )
            return 0

        # 2. Check environment variables
        console.print("\n📋 [bold]Environment Variables[/bold]")
        env_table = Table(show_header=True, header_style="bold blue")
        env_table.add_column("Variable", style="cyan", no_wrap=True)
        env_table.add_column("Status", style="green")
        env_table.add_column("Value Preview", style="dim")

        env_vars = [
            ("AUTOCLEAN_AUTH0_DOMAIN", os.getenv("AUTOCLEAN_AUTH0_DOMAIN")),
            ("AUTOCLEAN_AUTH0_CLIENT_ID", os.getenv("AUTOCLEAN_AUTH0_CLIENT_ID")),
            (
                "AUTOCLEAN_AUTH0_CLIENT_SECRET",
                os.getenv("AUTOCLEAN_AUTH0_CLIENT_SECRET"),
            ),
            ("AUTOCLEAN_AUTH0_AUDIENCE", os.getenv("AUTOCLEAN_AUTH0_AUDIENCE")),
            ("AUTOCLEAN_DEVELOPMENT_MODE", os.getenv("AUTOCLEAN_DEVELOPMENT_MODE")),
        ]

        for var_name, var_value in env_vars:
            if var_value:
                if "SECRET" in var_name:
                    preview = f"{var_value[:8]}..." if len(var_value) > 8 else "***"
                elif len(var_value) > 30:
                    preview = f"{var_value[:30]}..."
                else:
                    preview = var_value
                env_table.add_row(var_name, "✓ Set", preview)
            else:
                env_table.add_row(var_name, "[red]✗ Not Set[/red]", "")

        console.print(env_table)

        # 3. Check .env file
        console.print("\n📄 [bold].env File Detection[/bold]")
        env_paths = [
            Path(".env"),
            Path(".env.local"),
            Path("../.env"),
            Path("../../.env"),
        ]
        env_found = False
        for env_path in env_paths:
            if env_path.exists():
                console.print(f"✓ Found .env file: [cyan]{env_path.absolute()}[/cyan]")
                env_found = True
                if args.verbose:
                    try:
                        with open(env_path, "r") as f:
                            content = f.read()
                            auth_lines = [
                                line
                                for line in content.split("\n")
                                if "AUTOCLEAN_AUTH0" in line
                                and not line.strip().startswith("#")
                            ]
                            if auth_lines:
                                console.print("[dim]  Auth0 variables in file:[/dim]")
                                for line in auth_lines:
                                    # Mask secrets
                                    if "SECRET" in line and "=" in line:
                                        key, value = line.split("=", 1)
                                        masked_value = (
                                            f"{value[:8]}..."
                                            if len(value) > 8
                                            else "***"
                                        )
                                        console.print(
                                            f"[dim]    {key}={masked_value}[/dim]"
                                        )
                                    else:
                                        console.print(f"[dim]    {line}[/dim]")
                    except Exception as e:
                        console.print(f"[red]  Error reading file: {e}[/red]")
                break

        if not env_found:
            console.print(
                "[yellow]⚠[/yellow] No .env file found in current or parent directories"
            )

        # 4. Test credential loading
        console.print("\n🔧 [bold]Credential Loading Test[/bold]")
        try:
            auth_manager = get_auth0_manager()
            credentials = auth_manager._load_developer_credentials()

            if credentials:
                console.print("✓ Credentials loaded successfully")
                console.print(
                    f"  Source: [cyan]{credentials.get('source', 'unknown')}[/cyan]"
                )
                console.print(
                    f"  Domain: [cyan]{credentials.get('domain', 'NOT FOUND')}[/cyan]"
                )
                client_id = credentials.get("client_id", "NOT FOUND")
                if client_id != "NOT FOUND":
                    console.print(f"  Client ID: [cyan]{client_id[:8]}...[/cyan]")
                else:
                    console.print(f"  Client ID: [red]{client_id}[/red]")
            else:
                console.print("[red]✗ Failed to load credentials[/red]")
                console.print(
                    "[yellow]  Try setting environment variables or checking .env file[/yellow]"
                )
        except Exception as e:
            console.print(f"[red]✗ Error loading credentials: {e}[/red]")

        # 5. Test Auth0 domain connectivity
        console.print("\n🌐 [bold]Domain Connectivity Test[/bold]")
        openid_accessible = False
        connectivity_error = None

        try:
            # Get fresh credentials to ensure we test the correct domain
            auth_manager = get_auth0_manager()
            credentials = auth_manager._load_developer_credentials()

            if credentials and credentials.get("domain"):
                domain = credentials["domain"]
                console.print(f"Testing connection to: [cyan]{domain}[/cyan]")

                # Test basic connectivity
                try:
                    response = requests.get(f"https://{domain}", timeout=10)
                    if response.status_code in [
                        200,
                        404,
                        403,
                    ]:  # Any of these indicates domain exists
                        console.print("✓ Domain is reachable")
                        if args.verbose:
                            console.print(f"  HTTP Status: {response.status_code}")
                            console.print(
                                f"  Response time: {response.elapsed.total_seconds():.2f}s"
                            )
                    else:
                        connectivity_error = (
                            f"Unexpected status code: {response.status_code}"
                        )
                        console.print(f"[yellow]⚠[/yellow] {connectivity_error}")
                except requests.Timeout:
                    connectivity_error = "Connection timeout"
                    console.print(f"[red]✗ {connectivity_error}[/red]")
                except requests.ConnectionError:
                    connectivity_error = "Connection failed - check domain name"
                    console.print(f"[red]✗ {connectivity_error}[/red]")
                except Exception as e:
                    connectivity_error = f"Connection error: {e}"
                    console.print(f"[red]✗ {connectivity_error}[/red]")

                # Test Auth0 authorization endpoint (what login actually uses)
                try:
                    auth_url = f"https://{domain}/authorize"
                    response = requests.get(auth_url, timeout=10, allow_redirects=False)
                    # Auth0 authorize endpoint should return 400 (missing parameters) or redirect, not 404
                    if response.status_code in [400, 302, 301]:
                        openid_accessible = True
                        console.print("✓ Auth0 authorization endpoint accessible")
                        if args.verbose:
                            console.print(f"  Authorization URL: {auth_url}")
                            console.print(f"  Response status: {response.status_code}")
                    else:
                        console.print(
                            f"[yellow]⚠[/yellow] Auth0 authorization endpoint unexpected status: {response.status_code}"
                        )

                    # Also test the well-known endpoint (optional)
                    well_known_url = (
                        f"https://{domain}/.well-known/openid_configuration"
                    )
                    response = requests.get(well_known_url, timeout=5)
                    if response.status_code == 200:
                        console.print("✓ OpenID configuration also accessible")
                        if args.verbose:
                            config = response.json()
                            console.print(
                                f"  Issuer: {config.get('issuer', 'unknown')}"
                            )
                    else:
                        console.print(
                            f"[dim]ℹ OpenID config not available (status: {response.status_code}) - this is optional[/dim]"
                        )

                except Exception as e:
                    console.print(
                        f"[yellow]⚠[/yellow] Could not test Auth0 endpoints: {e}"
                    )
            else:
                connectivity_error = "No domain configured"
                console.print("[red]✗ No domain configured[/red]")
        except Exception as e:
            connectivity_error = f"Error testing connectivity: {e}"
            console.print(f"[red]✗ {connectivity_error}[/red]")

        # 6. Configuration summary
        console.print("\n📊 [bold]Configuration Summary[/bold]")
        try:
            auth_manager = get_auth0_manager()
            # Ensure configuration is loaded from environment/credentials
            credentials = auth_manager._load_developer_credentials()
            if credentials and not auth_manager.is_configured():
                auth_manager.configure_developer_auth0()

            summary_table = Table(show_header=True, header_style="bold blue")
            summary_table.add_column("Component", style="cyan")
            summary_table.add_column("Status", style="green")
            summary_table.add_column("Details", style="dim")

            # Check if configured
            is_configured = auth_manager.is_configured()
            summary_table.add_row(
                "Auth0 Configuration",
                "✓ Valid" if is_configured else "[red]✗ Invalid[/red]",
                "Ready for login" if is_configured else "Missing required credentials",
            )

            # Check authentication status
            is_authenticated = auth_manager.is_authenticated()
            summary_table.add_row(
                "Authentication",
                "✓ Logged in" if is_authenticated else "[yellow]Not logged in[/yellow]",
                "Valid session" if is_authenticated else "Run 'autoclean-eeg login'",
            )

            # Check config file
            config_file = auth_manager.config_file
            config_exists = config_file.exists()
            summary_table.add_row(
                "Config File",
                "✓ Exists" if config_exists else "[yellow]Not found[/yellow]",
                str(config_file) if config_exists else "Will be created on first setup",
            )

            console.print(summary_table)

        except Exception as e:
            console.print(f"[red]Error generating summary: {e}[/red]")

        # 7. Recommendations
        console.print("\n💡 [bold]Recommendations[/bold]")
        try:
            auth_manager = get_auth0_manager()
            credentials = auth_manager._load_developer_credentials()

            if not credentials:
                console.print("1. Set Auth0 environment variables:")
                console.print(
                    '   [cyan]export AUTOCLEAN_AUTH0_DOMAIN="your-tenant.us.auth0.com"[/cyan]'
                )
                console.print(
                    '   [cyan]export AUTOCLEAN_AUTH0_CLIENT_ID="your_client_id"[/cyan]'
                )
                console.print(
                    '   [cyan]export AUTOCLEAN_AUTH0_CLIENT_SECRET="your_client_secret"[/cyan]'
                )
                console.print("2. Or create a .env file with these variables")
                console.print(
                    "3. Install python-dotenv if using .env: [cyan]pip install python-dotenv[/cyan]"
                )
            elif connectivity_error:
                # Don't recommend login if there are connectivity issues
                console.print(
                    f"[red]⚠ Auth0 connectivity issue detected:[/red] {connectivity_error}"
                )
                console.print("")
                console.print("Please check:")
                console.print("1. Verify your Auth0 domain is correct in .env file")
                console.print("2. Ensure your Auth0 application is properly configured")
                console.print(
                    "3. Check that your Auth0 application type is 'Native' (for CLI apps)"
                )
                console.print("4. Verify your Auth0 tenant is active and accessible")
                console.print("")
                console.print(
                    "Once connectivity is fixed, run [cyan]autoclean-eeg login[/cyan] to authenticate"
                )
            elif not openid_accessible:
                console.print(
                    "[yellow]⚠ Auth0 OpenID configuration not accessible[/yellow]"
                )
                console.print("")
                console.print("This may indicate:")
                console.print("1. Auth0 domain is incorrect")
                console.print("2. Auth0 application is not properly configured")
                console.print("3. Network or firewall issues")
                console.print("")
                console.print(
                    "You can try [cyan]autoclean-eeg login[/cyan] but it may fail"
                )
            elif not auth_manager.is_authenticated():
                console.print("✓ Configuration looks good!")
                console.print("1. Run [cyan]autoclean-eeg login[/cyan] to authenticate")
            else:
                console.print(
                    "✓ Configuration looks good! You're ready to use Auth0 authentication."
                )

        except Exception as e:
            console.print(f"[red]Error generating recommendations: {e}[/red]")

        return 0

    except Exception as e:
        message("error", f"Diagnostics error: {e}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the AutoClean CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        # Show our custom 80s-style main interface instead of default help
        from rich.console import Console

        from autoclean.utils.branding import AutoCleanBranding

        console = Console()
        AutoCleanBranding.print_main_interface(console)
        return 0

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
    elif args.command == "export":
        return cmd_export_outputs(args)
    elif args.command == "login":
        return cmd_login(args)
    elif args.command == "logout":
        return cmd_logout(args)
    elif args.command == "whoami":
        return cmd_whoami(args)
    elif args.command == "auth0":
        return cmd_auth0(args)
    elif args.command == "auth0-diagnostics":
        return cmd_auth0_diagnostics(args)
    elif args.command == "version":
        return cmd_version(args)
    elif args.command == "help":
        return cmd_help(args)
    elif args.command == "tutorial":
        return cmd_tutorial(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
