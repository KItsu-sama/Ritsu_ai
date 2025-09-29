from __future__ import annotations

"""
core/event_manager.py

EventManager — central async dispatcher for events
- Async pub/sub system using asyncio.Queue
- Registers listeners, emits events, processes queue
- Routes input events → system commands, tools, core AI
- Extensible for new event types and handlers
"""


import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

EventHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class EventManager:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        queue: Optional[asyncio.Queue[Dict[str, Any]]] = None,
        ritsu_core: Optional[Any] = None,
        tools: Optional[Dict[str, Any]] = None,
        output_manager: Optional[Any] = None,
    ):
        self.config = config or {}
        self.queue: asyncio.Queue[Dict[str, Any]] = queue or asyncio.Queue()
        self.ritsu_core = ritsu_core
        self.tools = tools or {}
        self.output_manager = output_manager

        # Mapping of event_type → list of async handlers
        self._listeners: Dict[str, List[EventHandler]] = {}

        # Register default handler for 'input' events
        self.on("input", self._handle_input_event)

    # ------------------------- Registration -------------------------
    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register an async listener for a given event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)
        log.debug(f"Listener registered for event_type={event_type} handler={handler.__name__}")

    # ------------------------- Event Emission -------------------------
    async def emit(self, event: Dict[str, Any]) -> None:
        """Push event into queue for processing."""
        if not isinstance(event, dict):
            raise ValueError("Event must be a dict")
        if "type" not in event:
            raise ValueError("Event dict must contain 'type' key")

        await self.queue.put(event)
        log.debug(f"Event emitted: {event}")

    # ------------------------- Processing -------------------------
    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Continuously consume and dispatch events until shutdown."""
        try:
            while not shutdown_event.is_set():
                event = await self.queue.get()
                try:
                    await self._dispatch(event)
                except Exception:
                    log.exception("Error dispatching event", extra={"event": event})
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Event loop crashed")
            raise

    async def _dispatch(self, event: Dict[str, Any]) -> None:
        """Send event to all registered listeners of its type."""
        event_type = event.get("type")
        if not event_type:
            log.warning("Event missing 'type' key", extra={"event": event})
            return

        listeners = self._listeners.get(event_type, [])
        if not listeners:
            log.debug(f"No listeners registered for event type '{event_type}'")
            return

        for handler in listeners:
            try:
                await handler(event)
            except Exception:
                log.exception(f"Listener {handler.__name__} failed", extra={"event": event})

    # ------------------------- Default Input Event Handler -------------------------
    async def _handle_input_event(self, event: Dict[str, Any]) -> None:
        """
        Default handler for 'input' events.
        Routes commands, tool calls, or sends to core AI.
        """
        content = event.get("content", "")
        source = event.get("source", "unknown")
        self.logger = log  # for convenience

        self.logger.info(f"[Input Event] From {source}: {content}")

        # 1. System command (e.g., "!help")
        if content.startswith("!"):
            await self._handle_command(content)
            return

        # 2. Tool call (e.g., "calc 2+2")
        if await self._check_tool(content):
            return

        # 3. Default: send to core AI
        if self.ritsu_core:
            response = await self.ritsu_core.process(content, source=source)
            if self.output_manager:
                await self.output_manager.send(response, target=source)

    async def _handle_command(self, content: str) -> None:
        """Handle system commands prefixed with '!'."""
        parts = content[1:].split()
        cmd = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        self.logger.debug(f"System command received: {cmd} args={args}")

        if cmd == "help":
            msg = "Available commands: !help, !tools, !quit"
        elif cmd == "tools":
            msg = f"Available tools: {', '.join(self.tools.keys())}"
        elif cmd == "quit":
            msg = "Shutting down..."
            if self.output_manager:
                await self.output_manager.send(msg)
            raise SystemExit
        else:
            msg = f"Unknown command: {cmd}"

        if self.output_manager:
            await self.output_manager.send(msg)

    async def _check_tool(self, content: str) -> bool:
        """Check if content matches a registered tool call and run it."""
        for name, tool in self.tools.items():
            if content.startswith(name):
                args = content[len(name):].strip()
                result = await tool.run(args)
                if self.output_manager:
                    await self.output_manager.send(result)
                return True
        return False

    # ------------------------- Utilities -------------------------
    def listener_count(self, event_type: Optional[str] = None) -> int:
        if event_type:
            return len(self._listeners.get(event_type, []))
        return sum(len(lst) for lst in self._listeners.values())

    def close(self) -> None:
        self._listeners.clear()
        log.info("EventManager closed")
