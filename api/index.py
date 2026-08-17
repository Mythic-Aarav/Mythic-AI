"""
Vercel serverless entry point.

Vercel's Python runtime looks for a WSGI-compatible `app` object inside
files under the /api directory. This file just imports the real Flask
app (which lives at the project root as ai_chat.py) and re-exports it.

No logic lives here on purpose — ai_chat.py already detects Vercel via
the VERCEL / VERCEL_ENV environment variables (see IS_SERVERLESS in that
file) and adjusts its behavior automatically: /tmp-only file writes,
non-streaming chat responses, and skips the always-on background thread.
"""

import sys
import os

# Make the project root (one level up from /api) importable so
# `import ai_chat` finds ai_chat.py sitting next to this /api folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_chat import app  # noqa: E402  (Flask app object Vercel will serve)
