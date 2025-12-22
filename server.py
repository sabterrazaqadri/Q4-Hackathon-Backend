#!/usr/bin/env python
"""
Production server entry point for Render deployment.
This includes error handling and logging to help diagnose startup issues.
"""
import os
import sys
import logging
import traceback
from src.main import app

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting application initialization...")
        
        # Get port from environment variable (provided by Render)
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0"
        
        logger.info(f"Configured to run on {host}:{port}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")
        
        # Import uvicorn here to catch any import errors
        import uvicorn
        
        logger.info("Starting uvicorn server...")
        # Run the server binding to 0.0.0.0 for external access (required by Render)
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()