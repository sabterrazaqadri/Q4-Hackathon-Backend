"""
Main entry point for the RAG Chatbot application.
This file serves dual purposes:
1. As an entry point for local development and ingestion
2. As a reference for the modular application structure in src/
"""

import os
import sys
import uvicorn

def main():
    """
    Main function to run the application based on command-line arguments.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        # Import and run the modular app for serving
        from src.main import app

        # Get port from environment variable (for Render deployment)
        port = int(os.environ.get("PORT", 8000))

        # Run the server binding to 0.0.0.0 for external access (required by Render)
        uvicorn.run(app, host="0.0.0.0", port=port)
    elif len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        # Run ingestion pipeline
        from populate_db import ingest_book
        print("Starting ingestion pipeline...")
        ingest_book()
        print("Ingestion completed!")
    else:
        # Default behavior - show usage
        print("Usage:")
        print("  python main.py --serve     : Run the web server")
        print("  python main.py --ingest    : Run the ingestion pipeline")
        print("  python main.py             : Show this help message")


if __name__ == "__main__":
    main()