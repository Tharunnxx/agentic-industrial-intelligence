"""ASGI entry point:  python -m uvicorn agentic_app.main:app"""
from .api.routes import create_app

app = create_app()
