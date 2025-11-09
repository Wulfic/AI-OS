"""AI-OS GUI Application Entry Point.

This module provides a simplified entry point for running the AI-OS GUI.
It creates the AiosTkApp instance and delegates to app_main.run_app().
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from multiprocessing import freeze_support

# Add src to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aios.gui.app import AiosTkApp, run_app

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
        logger.info("Creating AiosTkApp instance...")
        app = AiosTkApp()
        
        # Run app (orchestrates all initialization)
        run_app(app)
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested via keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
