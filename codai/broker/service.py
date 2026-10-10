"""Broker client lifecycle wrapper."""

from __future__ import annotations

import asyncio

from codai.broker.client import BrokerClient, being_cancelled
from codai.broker.dispatcher import execute_broker_request


class BrokerService:
    def __init__(self, client: BrokerClient, app=None):
        self.client = client
        if app is not None:
            async def dispatch(message):
                envelope = self.client.message_to_envelope(message)
                return await execute_broker_request(app, envelope)

            self.client.dispatcher = dispatch
        self.task: asyncio.Task | None = None
        self._started = False

    def start(self):
        if not self.client.runtime.enabled or self.task is not None or self._started:
            return
        self._started = True
        self.task = asyncio.create_task(self.client.run_forever())

    async def stop(self):
        if self.task is None:
            self._started = False
            return
        task = self.task
        self.task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # The client reporting that it stopped is the expected outcome. A
            # cancellation of THIS coroutine — stop() awaited from a task that is
            # itself being torn down — is not, and swallowing it tells the caller
            # the shutdown completed normally.
            if being_cancelled():
                raise
        finally:
            self._started = False
