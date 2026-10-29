"""
API Module - REST & WebSocket Endpoints

FastAPI routes for ticket submission, workflow management, and real-time updates.
"""

from ticket_to_code.api.routes import router

__all__ = [
    'router',
]
