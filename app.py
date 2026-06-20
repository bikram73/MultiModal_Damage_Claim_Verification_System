#!/usr/bin/env python3
"""
Vercel entry point for the MultiModal Damage Claim Verification System.
This file serves as the main entry point for Vercel deployment.
"""

import sys
import os

# Add the code directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

# Import the FastAPI application from dashboard_server
from dashboard_server import app

# Vercel expects the app to be available at module level
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))