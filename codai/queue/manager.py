"""Queue manager module - manages request queues for model loading notifications."""

from typing import Dict, Optional
import asyncio
import time


class QueueManager:
    """
    Manages request queue for model loading notifications.
    When clients are waiting for a model to load, sends them progress updates.
    """
    
    def __init__(self):
        self.waiting_requests: Dict[str, float] = {}  # request_id -> start_time
        self.current_request_id: Optional[str] = None
        self.model_loading: bool = False
        self.model_name: Optional[str] = None
        self.lock = asyncio.Lock()
    
    async def add_waiting(self, request_id: str) -> None:
        """Add a request to the waiting queue."""
        async with self.lock:
            self.waiting_requests[request_id] = time.time()
    
    async def remove_waiting(self, request_id: str) -> None:
        """Remove a request from the waiting queue."""
        async with self.lock:
            self.waiting_requests.pop(request_id, None)
    
    async def start_processing(self, request_id: str, model_name: str = None) -> None:
        """Mark a request as now processing (model loaded)."""
        async with self.lock:
            self.waiting_requests.pop(request_id, None)
            self.current_request_id = request_id
            self.model_name = model_name
    
    async def finish_processing(self) -> None:
        """Mark current request as finished."""
        async with self.lock:
            self.current_request_id = None
    
    async def is_waiting(self, request_id: str) -> bool:
        """Check if a request is in the waiting queue."""
        async with self.lock:
            return request_id in self.waiting_requests
    
    async def get_wait_time(self, request_id: str) -> float:
        """Get how long a request has been waiting in seconds."""
        async with self.lock:
            if request_id in self.waiting_requests:
                return time.time() - self.waiting_requests[request_id]
            return 0.0
    
    async def get_queue_position(self, request_id: str) -> int:
        """Get the position of a request in the queue (1-based)."""
        async with self.lock:
            keys = list(self.waiting_requests.keys())
            try:
                return keys.index(request_id) + 1
            except ValueError:
                return 0


# Global queue manager instance
queue_manager = QueueManager()
