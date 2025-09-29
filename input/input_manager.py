from __future__ import annotations

"""
input/input_manager.py

InputManager — decides: mic, chat, file
- Unified input handling across multiple sources
- Async event-driven input processing
- Source-specific preprocessing and validation
- Event emission to central event queue
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class InputManager:
    """Manages input from multiple sources and emits events."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mic=None,
        chat=None,
        command_parser=None,
    ):
        self.config = config or {}
        
        # Component references
        self.mic = mic
        self.chat = chat
        self.command_parser = command_parser
        
        # Input sources configuration
        self.enable_cli = True  # Always enabled
        self.enable_mic = self.config.get("enable_mic", False) and mic is not None
        self.enable_chat = self.config.get("enable_chat", False) and chat is not None
        self.enable_ui = self.config.get("enable_ui", False)
        self.enable_api = self.config.get("enable_api", False)
        
        # State tracking
        self.active_sources = []
        self._setup_active_sources()
    
    def _setup_active_sources(self) -> None:
        """Setup list of active input sources."""
        self.active_sources = []
        
        if self.enable_cli:
            self.active_sources.append(("cli", self._handle_cli_input))
        
        if self.enable_mic:
            self.active_sources.append(("mic", self._handle_mic_input))
        
        if self.enable_chat:
            self.active_sources.append(("chat", self._handle_chat_input))
        
        if self.enable_ui:
            self.active_sources.append(("ui", self._handle_ui_input))
        
        if self.enable_api:
            self.active_sources.append(("api", self._handle_api_input))
        
        log.info("Active input sources", extra={
            "sources": [name for name, _ in self.active_sources]
        })
    
    async def run(
        self,
        queue: "asyncio.Queue[Dict[str, Any]]",
        shutdown_event: asyncio.Event
    ) -> None:
        """Main input processing loop.
        
        Args:
            queue: Event queue to emit input events to
            shutdown_event: Event to signal shutdown
        """
        log.info("Input manager starting", extra={
            "active_sources": len(self.active_sources)
        })
        
        try:
            # Create tasks for each active input source
            tasks = []
            for source_name, handler in self.active_sources:
                task = asyncio.create_task(
                    self._run_input_source(source_name, handler, queue, shutdown_event),
                    name=f"input_{source_name}"
                )
                tasks.append(task)
            
            # Wait for shutdown or all tasks to complete
            if tasks:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            else:
                log.warning("No input sources active, waiting for shutdown")
                await shutdown_event.wait()
                
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Input manager crashed")
            raise
        finally:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            log.info("Input manager stopped")
    
    async def _run_input_source(
        self,
        source_name: str,
        handler,
        queue: "asyncio.Queue[Dict[str, Any]]",
        shutdown_event: asyncio.Event
    ) -> None:
        """Run a single input source."""
        try:
            while not shutdown_event.is_set():
                try:
                    # Get input from source
                    input_data = await handler()
                    
                    if input_data:
                        # Create event
                        event = await self._create_event(source_name, input_data)
                        
                        if event:
                            # Emit to queue
                            await queue.put(event)
                            
                            log.debug("Input event emitted", extra={
                                "source": source_name,
                                "content_preview": input_data.get("content", "")[:50]
                            })
                    
                    # Small delay to prevent tight loops
                    await asyncio.sleep(0.1)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.error("Input source error", extra={
                        "source": source_name,
                        "error": str(e)
                    })
                    # Continue running other sources
                    await asyncio.sleep(1.0)  # Longer delay after error
                    
        except asyncio.CancelledError:
            log.debug("Input source cancelled", extra={"source": source_name})
            raise
    
    async def _create_event(self, source: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a standardized event from input data.
        
        Args:
            source: Name of the input source
            input_data: Raw input data from source
            
        Returns:
            Standardized event dictionary or None if invalid
        """
        try:
            content = input_data.get("content", "")
            if not content or not content.strip():
                return None
            
            # Parse command if parser is available
            parsed_command = None
            if self.command_parser:
                try:
                    parsed_command = self.command_parser.parse(content)
                except Exception as e:
                    log.warning("Command parsing failed", extra={"error": str(e)})
            
            # Create standardized event
            event = {
                "type": "user_input",
                "source": source,
                "content": content.strip(),
                "timestamp": asyncio.get_event_loop().time(),
                "metadata": input_data.get("metadata", {}),
            }
            
            # Add parsed command if available
            if parsed_command:
                event["parsed_command"] = parsed_command
            
            return event
            
        except Exception as e:
            log.error("Failed to create event", extra={
                "source": source,
                "input_data": input_data,
                "error": str(e)
            })
            return None
    
    # Input source handlers
    
    async def _handle_cli_input(self) -> Optional[Dict[str, Any]]:
        """Handle command line input."""
        try:
            # Use asyncio to run input() in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, input, "Ritsu> ")
            
            if user_input.strip():
                return {
                    "content": user_input.strip(),
                    "metadata": {"input_method": "keyboard"}
                }
            
        except EOFError:
            # Handle Ctrl+D or EOF
            return {
                "content": "!quit",
                "metadata": {"input_method": "keyboard", "eof": True}
            }
        except KeyboardInterrupt:
            # Handle Ctrl+C
            return {
                "content": "!quit",
                "metadata": {"input_method": "keyboard", "interrupt": True}
            }
        except Exception as e:
            log.error("CLI input error", extra={"error": str(e)})
        
        return None
    
    async def _handle_mic_input(self) -> Optional[Dict[str, Any]]:
        """Handle microphone input via STT."""
        if not self.mic:
            return None
        
        try:
            # This would interface with the mic listener component
            audio_data = await self.mic.listen()
            
            if audio_data:
                return {
                    "content": audio_data.get("transcription", ""),
                    "metadata": {
                        "input_method": "microphone",
                        "confidence": audio_data.get("confidence", 0.0),
                        "duration": audio_data.get("duration", 0.0)
                    }
                }
        except Exception as e:
            log.error("Microphone input error", extra={"error": str(e)})
        
        return None
    
    async def _handle_chat_input(self) -> Optional[Dict[str, Any]]:
        """Handle chat input from external platforms."""
        if not self.chat:
            return None
        
        try:
            # This would interface with chat listener components
            chat_message = await self.chat.get_message()
            
            if chat_message:
                return {
                    "content": chat_message.get("content", ""),
                    "metadata": {
                        "input_method": "chat",
                        "platform": chat_message.get("platform", "unknown"),
                        "user_id": chat_message.get("user_id", "unknown"),
                        "channel": chat_message.get("channel", "unknown")
                    }
                }
        except Exception as e:
            log.error("Chat input error", extra={"error": str(e)})
        
        return None
    
    async def _handle_ui_input(self) -> Optional[Dict[str, Any]]:
        """Handle UI input from C# interface."""
        try:
            # This would interface with UI via IPC/sockets
            # Placeholder for now
            await asyncio.sleep(1.0)  # Prevent tight loop
            return None
        except Exception as e:
            log.error("UI input error", extra={"error": str(e)})
        
        return None
    
    async def _handle_api_input(self) -> Optional[Dict[str, Any]]:
        """Handle API input from external services."""
        try:
            # This would interface with external APIs
            # Placeholder for now
            await asyncio.sleep(1.0)  # Prevent tight loop
            return None
        except Exception as e:
            log.error("API input error", extra={"error": str(e)})
        
        return None
    
    def get_active_sources(self) -> List[str]:
        """Get list of currently active input sources."""
        return [name for name, _ in self.active_sources]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get input manager statistics."""
        return {
            "active_sources": len(self.active_sources),
            "source_names": self.get_active_sources(),
            "config": {
                "cli_enabled": self.enable_cli,
                "mic_enabled": self.enable_mic,
                "chat_enabled": self.enable_chat,
                "ui_enabled": self.enable_ui,
                "api_enabled": self.enable_api,
            }
        }
