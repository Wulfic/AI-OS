"""AI-OS GUI Application Entry Point.

This module provides a simplified entry point for running the AI-OS GUI.
It creates the AiosTkApp instance which handles all initialization internally.
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from multiprocessing import freeze_support

# Add src to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aios.gui.app import AiosTkApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for AI-OS GUI."""
    # Required for Windows multiprocessing
    freeze_support()
    
    try:
        # Create app instance
        # Note: AiosTkApp.__init__ already calls run_app(self) internally,
        # which initializes all components and starts the main loop
        logger.info("Creating AiosTkApp instance...")
        app = AiosTkApp()
        
        # No need to call run_app(app) here - it's already called in __init__
        # The main loop has already run and exited when we reach this point
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested via keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
