#!/usr/bin/env python3
"""
Textual TUI for live cost function tuning while running the hexapod simulation.
"""

import asyncio
import threading
import time
from typing import Dict, Any
import queue

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Input, Button, RichLog, Label
from textual.reactive import reactive
from rich.text import Text
from rich.console import Console
from rich.logging import RichHandler
import logging

from hexapod_mdp import HexapodMDP

# Global shared MDP for live tuning
current_mdp = None

# Global logging queue for TUI
log_queue = queue.Queue()

# Global flag to track if TUI logging is set up
tui_logging_initialized = False

def setup_early_logging():
    """Set up logging capture as early as possible"""
    global tui_logging_initialized
    if not tui_logging_initialized:
        # Create handler immediately
        handler = TUILogHandler()
        
        # Add to root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        # Also add to specific loggers
        for logger_name in ['__main__', 'run_viewer', 'hexapod_mdp']:
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
        
        tui_logging_initialized = True

class TUILogHandler(logging.Handler):
    """Custom logging handler that sends logs to the TUI with Rich formatting"""
    
    def __init__(self):
        super().__init__()
        # Create a console to capture Rich-formatted output
        self.console = Console(file=None, width=80)
    
    def emit(self, record):
        try:
            # Create a Rich-formatted log entry using markup
            level_colors = {
                'DEBUG': 'cyan',
                'INFO': 'green', 
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold red'
            }
            
            level_color = level_colors.get(record.levelname, 'white')
            timestamp = self.formatTime(record, '%H:%M:%S')
            
            # Create Rich markup string
            formatted_text = (
                f"[dim]{timestamp}[/dim] "
                f"[{level_color}]{record.levelname:8}[/{level_color}] "
                f"[blue]{record.name}[/blue]: {record.getMessage()}"
            )
            
            # Put the formatted text in the queue
            log_queue.put(formatted_text)
        except Exception:
            # Avoid infinite recursion if logging fails
            pass


class CostCoefficientControl(Container):
    """Widget for controlling a single cost coefficient"""
    
    def __init__(self, cost_name: str, initial_value: float, callback):
        super().__init__()
        self.cost_name = cost_name
        self.value = initial_value
        self.callback = callback
        
    def compose(self) -> ComposeResult:
        yield Label(f"{self.cost_name}:", classes="cost-label")
        yield Input(
            value=str(self.value), 
            placeholder="0.0",
            id=f"input-{self.cost_name}",
            classes="cost-input"
        )
        yield Button("Update", id=f"btn-{self.cost_name}", classes="cost-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == f"btn-{self.cost_name}":
            input_widget = self.query_one(f"#input-{self.cost_name}", Input)
            try:
                new_value = float(input_widget.value)
                self.value = new_value
                self.callback(self.cost_name, new_value)
                self.notify(f"Updated {self.cost_name} to {new_value}")
            except ValueError:
                self.notify(f"Invalid value for {self.cost_name}", severity="error")


class CostTunerTUI(App):
    """TUI for live cost function tuning"""
    
    CSS = """
    Screen {
        layout: horizontal;
    }
    
    .left-panel {
        width: 60%;
        height: 100%;
        border: solid green;
        margin: 1;
    }
    
    .right-panel {
        width: 40%;
        height: 100%;
        border: solid blue;
        margin: 1;
    }
    
    .cost-control {
        layout: horizontal;
        height: 3;
        margin: 1 0;
    }
    
    .cost-label {
        width: 30%;
        content-align: right middle;
    }
    
    .cost-input {
        width: 50%;
        margin: 0 1;
    }
    
    .cost-button {
        width: 20%;
    }
    
    RichLog {
        height: 1fr;
        margin: 1;
    }
    
    #cost-info {
        height: auto;
        margin: 1;
        border: solid white;
    }
    """
    
    TITLE = "Hexapod Cost Function Tuner"
    SUB_TITLE = "Live tuning of cost coefficients"
    
    def __init__(self, mdp: HexapodMDP):
        super().__init__()
        self.mdp = mdp
        self.cost_controls: Dict[str, CostCoefficientControl] = {}
        self.running = True
        
        # Initialize global MDP
        global current_mdp
        current_mdp = mdp
        
        # Set up logging to capture in RichLog
        self.log_console = Console()
        self.rich_log = None
        
        # Set up TUI logging handler
        self.setup_logging_handler()
        
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(classes="left-panel"):
                yield Static("Simulation Log", id="log-header")
                yield RichLog(id="simulation-log", markup=True)
            
            with Vertical(classes="right-panel"):
                yield Static("Cost Function Coefficients", id="cost-header")
                yield Static("", id="cost-info")
                
                # Create controls for each cost function
                for name, value in self.mdp.cost_coefficients.items():
                    cost_control = CostCoefficientControl(
                        name, value, self.update_cost_coefficient
                    )
                    cost_control.add_class("cost-control")
                    yield cost_control
                    self.cost_controls[name] = cost_control
                
                yield Button("Reset to Defaults", id="reset-btn", classes="reset-button")
                
        yield Footer()
    
    def setup_logging_handler(self):
        """Set up custom logging handler to capture logs for TUI display"""
        # Ensure early logging is set up (may already be done)
        setup_early_logging()
    
    def on_mount(self) -> None:
        """Called when the app is mounted"""
        self.rich_log = self.query_one("#simulation-log", RichLog)
        self.update_cost_info()
        
        # Add an initial test message to verify the log display works
        self.rich_log.write(Text.from_markup("[green]TUI[/green]: Logging system initialized"))
        
        # Start background task to capture logs and update display
        self.set_timer(0.1, self.update_display)
        
        # Start dummy logging coroutine
        self.set_timer(1.0, self.dummy_log_message)
    
    def update_cost_coefficient(self, name: str, value: float) -> None:
        """Update a cost coefficient using JAX-compatible immutable updates"""
        global current_mdp
        
        # Create new MDP with updated coefficient
        coefficient_update = {name: value}
        new_mdp = self.mdp.with_updated_coefficients(**coefficient_update)
        
        # Update both local and global references explicitly
        self.mdp = new_mdp
        current_mdp = new_mdp
        
        self.log_message(f"[green]Updated {name} = {value}[/green]")
        self.update_cost_info()
    
    def update_cost_info(self) -> None:
        """Update the cost information display"""
        info_text = "Current Coefficients:\n"
        for name, value in self.mdp.cost_coefficients.items():
            info_text += f"  {name}: {value:.3f}\n"
        
        cost_info = self.query_one("#cost-info", Static)
        cost_info.update(info_text)
    
    def log_message(self, message: str) -> None:
        """Add a message to the simulation log"""
        if self.rich_log:
            self.rich_log.write(Text.from_markup(message))
    
    def update_display(self) -> None:
        """Periodic update of the display"""
        if self.running:
            # Check for new log messages and display them
            self.process_log_queue()
            # Schedule next update
            self.set_timer(0.1, self.update_display)
    
    def process_log_queue(self) -> None:
        """Process any new log messages from the queue"""
        if not self.rich_log:
            return
            
        # Process all available log messages
        messages_processed = 0
        while True:
            try:
                log_message = log_queue.get_nowait()
                # Convert Rich markup to Text object and write it
                rich_text = Text.from_markup(log_message)
                self.rich_log.write(rich_text)
                messages_processed += 1
            except queue.Empty:
                break
        
        # Messages are now displayed in the log panel
    
    def dummy_log_message(self) -> None:
        """Send a dummy log message every second to test the logging system"""
        # Remove this in production - just for testing the log display
        pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "reset-btn":
            self.reset_to_defaults()
    
    def reset_to_defaults(self) -> None:
        """Reset all coefficients to default values"""
        global current_mdp
        
        defaults = {
            'forward_velocity': 10.0,
            'height': 5.0,
            'action_penalty': 0.1,
            'stability': 5.0,
            'fall_penalty': 100.0,
        }
        
        # Update all coefficients at once using the single interface
        new_mdp = self.mdp.with_updated_coefficients(**defaults)
        
        # Update both local and global references
        self.mdp = new_mdp
        current_mdp = new_mdp
        
        # Update input widgets
        for name, value in defaults.items():
            if name in self.cost_controls:
                input_widget = self.cost_controls[name].query_one(f"#input-{name}", Input)
                input_widget.value = str(value)
                self.cost_controls[name].value = value
        
        self.update_cost_info()
        self.log_message("[yellow]Reset all coefficients to defaults[/yellow]")
    
    def action_quit(self) -> None:
        """Quit the application"""
        self.running = False
        # Note: We don't remove the logging handler since it might be shared
        # and other parts of the application might still need it
        self.exit()
    
    def on_key(self, event) -> None:
        """Handle key presses"""
        # Ctrl+C to quit
        if event.key == "ctrl+c":
            self.action_quit()


def create_tui_with_mdp(mdp: HexapodMDP) -> CostTunerTUI:
    """Create and return a TUI instance with the given MDP"""
    return CostTunerTUI(mdp)

async def run_tui_interface(mdp: HexapodMDP):
    """Run the TUI interface as an async coroutine"""
    app = CostTunerTUI(mdp)
    await app.run_async()


def main():
    """Test the TUI standalone"""
    from hexapod_mdp import HexapodMDP
    
    mdp = HexapodMDP.create('hexagonal_robot.xml')
    app = CostTunerTUI(mdp)
    app.run()


if __name__ == "__main__":
    main()