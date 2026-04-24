#!/bin/bash
# Startup script for agentic_RAG app

# Exit on error
set -e

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please set up .venv first."
    exit 1
fi

# Run the main app
python app.py
