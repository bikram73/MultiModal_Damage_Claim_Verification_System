#!/usr/bin/env python3
"""
Vercel entry point for the MultiModal Damage Claim Verification System.
"""

import sys
import os

# Add the code directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

# Import the FastAPI application from dashboard_server
from dashboard_server import app

# Export the app for Vercel
# This is the standard pattern for Vercel Python deployments
def handler(request):
    return app(request)

# Also make the app available directly
application = app