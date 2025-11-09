"""Main entry point for AI-OS GUI application."""
import sys
import tkinter as tk
from pathlib import Path


def main():
    """Launch the AI-OS GUI application."""
    # Import here to avoid circular imports and speed up CLI commands
    from aios.gui.app import AiosTkApp
    
    # Create and run the application
    app = AiosTkApp()
    app.mainloop()


if __name__ == "__main__":
    main()
