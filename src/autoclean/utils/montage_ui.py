"""Textual UI for EEG montage selection and display."""

from pathlib import Path
from typing import Dict, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

# Handle importlib.resources compatibility
try:
    from importlib import resources

    IMPORTLIB_RESOURCES_AVAILABLE = True
except ImportError:
    try:
        import importlib_resources as resources

        IMPORTLIB_RESOURCES_AVAILABLE = True
    except ImportError:
        IMPORTLIB_RESOURCES_AVAILABLE = False
        resources = None

try:
    import autoclean.data.montages

    DATA_MONTAGES_AVAILABLE = True
except ImportError:
    DATA_MONTAGES_AVAILABLE = False
    autoclean.data.montages = None


class MontageDetailPanel(Static):
    """Display detailed information about a selected montage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.montage_id: Optional[str] = None
        self.description: Optional[str] = None

    def update_montage(self, montage_id: str, description: str) -> None:
        """Update the panel with new montage information.

        Parameters
        ----------
        montage_id : str
            The montage identifier
        description : str
            The montage description
        """
        self.montage_id = montage_id
        self.description = description

        # Try to find an associated .sfp file
        sfp_content = self._get_sfp_content(montage_id)

        # Build the detail view
        content_parts = [
            f"[bold cyan]Montage ID:[/bold cyan] {montage_id}",
            f"[bold cyan]Description:[/bold cyan] {description}",
            "",
        ]

        # Check if this is a standard MNE montage or custom
        if sfp_content:
            content_parts.extend(
                [
                    "[bold cyan]Source:[/bold cyan] Custom .sfp file",
                    "[bold cyan]File Path:[/bold cyan] "
                    + str(self._get_sfp_path(montage_id)),
                    "",
                    "[bold yellow]═" * 50 + "[/bold yellow]",
                    "[bold yellow]Montage File Contents:[/bold yellow]",
                    "[bold yellow]═" * 50 + "[/bold yellow]",
                    "",
                    f"[dim]{sfp_content}[/dim]",
                ]
            )
        else:
            content_parts.extend(
                [
                    "[bold cyan]Source:[/bold cyan] MNE-Python standard montage",
                    "",
                    "[dim]This is a standard montage provided by MNE-Python.",
                    "No custom electrode file is available for display.[/dim]",
                ]
            )

        self.update("\n".join(content_parts))

    def _get_sfp_path(self, montage_id: str) -> Optional[Path]:
        """Get the path to the .sfp file for a montage.

        Parameters
        ----------
        montage_id : str
            The montage identifier

        Returns
        -------
        Optional[Path]
            Path to .sfp file if it exists, None otherwise
        """
        # Try package resources first
        if IMPORTLIB_RESOURCES_AVAILABLE and DATA_MONTAGES_AVAILABLE:
            try:
                sfp_file = f"{montage_id}.sfp"
                if (
                    resources.files(autoclean.data.montages)
                    .joinpath(sfp_file)
                    .is_file()
                ):
                    return resources.files(autoclean.data.montages).joinpath(sfp_file)
            except Exception:
                pass

        # Try relative path for development
        src_path = (
            Path(__file__).parent.parent / "data" / "montages" / f"{montage_id}.sfp"
        )
        if src_path.exists():
            return src_path

        return None

    def _get_sfp_content(self, montage_id: str) -> Optional[str]:
        """Get the contents of the .sfp file for a montage.

        Parameters
        ----------
        montage_id : str
            The montage identifier

        Returns
        -------
        Optional[str]
            Contents of .sfp file if it exists, None otherwise
        """
        sfp_path = self._get_sfp_path(montage_id)
        if not sfp_path:
            return None

        try:
            # For package resources
            if hasattr(sfp_path, "read_text"):
                return sfp_path.read_text(encoding="utf-8")
            # For regular Path objects
            else:
                return sfp_path.read_text(encoding="utf-8")
        except Exception:
            return None


class MontageListView(App):
    """A beautiful TUI for browsing and selecting EEG montages.

    This application provides a two-pane interface:
    - Left: List of available montages
    - Right: Detailed information about the selected montage

    Parameters
    ----------
    montages : Dict[str, str]
        Dictionary mapping montage IDs to descriptions
    selectable : bool
        If True, pressing Enter will select a montage and exit.
        If False, the app is in view-only mode.
    """

    CSS = """
    Screen {
        background: $background;
    }

    Header {
        background: $primary;
        color: $text;
    }

    Footer {
        background: $panel;
    }

    #main-container {
        height: 100%;
        width: 100%;
    }

    #left-panel {
        width: 40%;
        border-right: thick $primary;
        background: $surface;
    }

    #right-panel {
        width: 60%;
        background: $background;
        padding: 1 2;
    }

    ListView {
        background: $surface;
        height: 100%;
    }

    ListView > ListItem {
        padding: 1 2;
    }

    ListView > ListItem.--highlight {
        background: $accent;
    }

    MontageDetailPanel {
        height: 100%;
        background: $background;
        color: $text;
    }

    .montage-list-item {
        padding: 0 1;
    }

    Label.montage-id {
        color: $success;
        text-style: bold;
    }

    Label.montage-desc {
        color: $text-muted;
        margin-left: 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("escape", "quit", "Quit", priority=True),
        Binding("enter", "select", "Select", show=True, priority=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    def __init__(
        self,
        montages: Dict[str, str],
        selectable: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the montage list view.

        Parameters
        ----------
        montages : Dict[str, str]
            Dictionary mapping montage IDs to descriptions
        selectable : bool
            If True, pressing Enter will select a montage and exit
        """
        super().__init__(*args, **kwargs)
        self.montages = montages
        self.selectable = selectable
        self.selected_montage: Optional[str] = None

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        if self.selectable:
            yield Header(show_clock=False)
        else:
            yield Header(show_clock=False)

        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                # Create ListView and populate it directly with ListItems
                with ListView(id="montage-list"):
                    for montage_id, description in sorted(self.montages.items()):
                        list_item = ListItem(
                            Vertical(
                                Label(montage_id, classes="montage-id"),
                                Label(description, classes="montage-desc"),
                                classes="montage-list-item",
                            )
                        )
                        # Store montage info as attributes
                        list_item.montage_id = montage_id
                        list_item.description = description
                        yield list_item

            with VerticalScroll(id="right-panel"):
                yield MontageDetailPanel(id="detail-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Handle mount event - select first item."""
        list_view = self.query_one("#montage-list", ListView)
        if len(list_view) > 0:
            list_view.index = 0
            # Trigger the highlight event manually
            first_item = list_view.children[0]
            if hasattr(first_item, "montage_id"):
                detail_panel = self.query_one("#detail-panel", MontageDetailPanel)
                detail_panel.update_montage(
                    first_item.montage_id, first_item.description
                )

    @on(ListView.Highlighted)
    def on_list_highlight(self, event: ListView.Highlighted) -> None:
        """Update detail panel when a montage is highlighted."""
        if event.item and hasattr(event.item, "montage_id"):
            detail_panel = self.query_one("#detail-panel", MontageDetailPanel)
            detail_panel.update_montage(event.item.montage_id, event.item.description)

    def action_select(self) -> None:
        """Handle selection action."""
        if not self.selectable:
            return

        list_view = self.query_one("#montage-list", ListView)
        if list_view.highlighted_child:
            item = list_view.highlighted_child
            if hasattr(item, "montage_id"):
                self.selected_montage = item.montage_id
                self.exit(self.selected_montage)

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit(None)


def display_montage_browser(
    montages: Dict[str, str], selectable: bool = False
) -> Optional[str]:
    """Display an interactive montage browser.

    Parameters
    ----------
    montages : Dict[str, str]
        Dictionary mapping montage IDs to descriptions
    selectable : bool
        If True, user can select a montage with Enter.
        If False, view-only mode.

    Returns
    -------
    Optional[str]
        Selected montage ID if selectable=True and user selected one,
        None otherwise
    """
    app = MontageListView(montages=montages, selectable=selectable)
    return app.run()
