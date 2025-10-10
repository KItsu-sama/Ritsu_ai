from __future__ import annotations

"""
output/output_manager.py

OutputManager — central output handler
- Coordinates output across multiple channels
- Format-specific rendering and delivery
- Source routing and filtering
- Performance monitoring and queuing
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class OutputManager:
    """Manages output delivery across multiple channels and formats."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tts=None,
        avatar=None,
        stream=None,
    ):
        self.config = config or {}
        
        # Component references
        self.tts = tts
        self.avatar = avatar
        self.stream = stream
        
        # Output channels configuration
        self.enable_console = self.config.get("enable_console", True)
        self.enable_tts = self.config.get("enable_tts", False) and tts is not None
        self.enable_avatar = self.config.get("enable_avatar", False) and avatar is not None
        self.enable_stream = self.config.get("enable_stream", False) and stream is not None
        
        # Output queue for async processing
        self.output_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "errors": 0,
            "by_destination": {},
            "by_format": {},
        }
        
        log.info("Output manager initialized", extra={
            "console_enabled": self.enable_console,
            "tts_enabled": self.enable_tts,
            "avatar_enabled": self.enable_avatar,
            "stream_enabled": self.enable_stream,
        })
    
    async def start(self) -> None:
        """Start the output processing task."""
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(
                self._process_output_queue(),
                name="output_processor"
            )
            log.info("Output processing started")
    
    async def stop(self) -> None:
        """Stop the output processing task."""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            log.info("Output processing stopped")
    
    async def emit(self, output_data: Dict[str, Any]) -> None:
        """Emit output data for processing.
        
        Args:
            output_data: Dictionary containing output information:
                - content: The content to output
                - destination: Target destination (cli, stream, etc.)
                - format: Output format (text, audio, visual)
                - metadata: Additional metadata
        """
        try:
            # Validate output data
            if not self._validate_output_data(output_data):
                log.warning("Invalid output data", extra={"data": output_data})
                return
            
            # Add timestamp and ID
            output_data["timestamp"] = asyncio.get_event_loop().time()
            output_data["output_id"] = f"out_{int(output_data['timestamp'] * 1000)}"
            
            # Queue for processing
            await self.output_queue.put(output_data)
            
            log.debug("Output queued", extra={
                "destination": output_data.get("destination"),
                "format": output_data.get("format"),
                "content_preview": str(output_data.get("content", ""))[:50]
            })
            
        except Exception as e:
            log.error("Failed to emit output", extra={
                "output_data": output_data,
                "error": str(e)
            })
            self.stats["errors"] += 1
    
    def _validate_output_data(self, output_data: Dict[str, Any]) -> bool:
        """Validate output data structure."""
        required_fields = ["content"]
        
        for field in required_fields:
            if field not in output_data:
                log.warning("Missing required field", extra={"field": field})
                return False
        
        content = output_data.get("content")
        if not content or (isinstance(content, str) and not content.strip()):
            return False
        
        return True
    
    async def _process_output_queue(self) -> None:
        """Process queued output messages."""
        try:
            while True:
                try:
                    # Get next output from queue
                    output_data = await self.output_queue.get()
                    
                    # Process the output
                    await self._process_output(output_data)
                    
                    # Mark task as done
                    self.output_queue.task_done()
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.error("Error processing output", extra={"error": str(e)})
                    self.stats["errors"] += 1
                    
        except asyncio.CancelledError:
            log.debug("Output processor cancelled")
            raise
    
    async def _process_output(self, output_data: Dict[str, Any]) -> None:
        """Process a single output message."""
        try:
            destination = output_data.get("destination", "cli")
            format_type = output_data.get("format", "text")
            content = output_data.get("content")
            is_error = output_data.get("is_error", False)
            
            # Update statistics
            self.stats["messages_sent"] += 1
            self.stats["by_destination"][destination] = self.stats["by_destination"].get(destination, 0) + 1
            self.stats["by_format"][format_type] = self.stats["by_format"].get(format_type, 0) + 1
            
            # Route to appropriate output channels
            await self._route_output(destination, format_type, content, is_error, output_data)
            
            log.debug("Output processed", extra={
                "destination": destination,
                "format": format_type,
                "is_error": is_error
            })
            
        except Exception as e:
            log.error("Failed to process output", extra={
                "output_data": output_data,
                "error": str(e)
            })
            self.stats["errors"] += 1
    
    async def _route_output(
        self,
        destination: str,
        format_type: str,
        content: Any,
        is_error: bool,
        metadata: Dict[str, Any]
    ) -> None:
        """Route output to appropriate channels."""
        # Always output to console if enabled
        if self.enable_console and destination in ["cli", "console", "unknown"]:
            await self._output_to_console(content, is_error)
        
        # Route to specific channels based on destination
        if destination == "stream" and self.enable_stream:
            await self._output_to_stream(content, format_type, metadata)
        
        # Handle TTS for audio format
        if format_type == "audio" or (format_type == "text" and self.enable_tts):
            await self._output_to_tts(content, metadata)
        
        # Handle avatar animations
        if self.enable_avatar and not is_error:
            await self._output_to_avatar(content, format_type, metadata)
    
    async def emit_simple(self, message: str, destination: str = "cli") -> None:
        """Simple emit method for quick messages."""
        await self.emit({
            "content": message,
            "destination": destination,
            "format": "text"
        })
    
    async def _output_to_console(self, content: Any, is_error: bool = False) -> None:
        """Output to console/terminal."""
        try:
            # Convert content to string
            if isinstance(content, str):
                text = content
            else:
                text = str(content)
            
            # Add prefix for errors
            if is_error:
                text = f"[ERROR] {text}"
            else:
                text = f"Ritsu: {text}"
            
            # Print to console (using print for simplicity)
            print(text)
            
        except Exception as e:
            log.error("Console output failed", extra={"error": str(e)})
    
    async def _output_to_tts(self, content: Any, metadata: Dict[str, Any]) -> None:
        """Output to text-to-speech."""
        if not self.tts:
            return
        
        try:
            # Convert content to speech-friendly text
            if isinstance(content, str):
                text = content
            else:
                text = str(content)
            
            # Remove special characters that might affect TTS
            text = text.replace("[ERROR]", "Error:")
            text = text.replace("Ritsu:", "")
            text = text.strip()
            
            if text:
                await self.tts.speak(text, metadata)
            
        except Exception as e:
            log.error("TTS output failed", extra={"error": str(e)})
    
    async def _output_to_avatar(self, content: Any, format_type: str, metadata: Dict[str, Any]) -> None:
        """Output to avatar animation system."""
        if not self.avatar:
            return
        
        try:
            # Determine appropriate animation/expression
            animation_type = "speaking"
            
            if isinstance(content, str):
                content_lower = content.lower()
                if any(word in content_lower for word in ["error", "sorry", "apologize"]):
                    animation_type = "confused"
                elif any(word in content_lower for word in ["happy", "great", "excellent"]):
                    animation_type = "happy"
                elif "?" in content:
                    animation_type = "curious"
            
            await self.avatar.animate(animation_type, metadata)
            
        except Exception as e:
            log.error("Avatar output failed", extra={"error": str(e)})
    
    async def _output_to_stream(self, content: Any, format_type: str, metadata: Dict[str, Any]) -> None:
        """Output to streaming overlay/interface."""
        if not self.stream:
            return
        
        try:
            stream_data = {
                "type": "message",
                "content": content,
                "format": format_type,
                "timestamp": metadata.get("timestamp"),
                "metadata": metadata,
            }
            
            await self.stream.send(stream_data)
            
        except Exception as e:
            log.error("Stream output failed", extra={"error": str(e)})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get output statistics."""
        return {
            "messages_sent": self.stats["messages_sent"],
            "errors": self.stats["errors"],
            "by_destination": self.stats["by_destination"].copy(),
            "by_format": self.stats["by_format"].copy(),
            "queue_size": self.output_queue.qsize(),
            "channels_enabled": {
                "console": self.enable_console,
                "tts": self.enable_tts,
                "avatar": self.enable_avatar,
                "stream": self.enable_stream,
            },
        }
    
    def clear_stats(self) -> None:
        """Clear output statistics."""
        self.stats = {
            "messages_sent": 0,
            "errors": 0,
            "by_destination": {},
            "by_format": {},
        }
        log.info("Output statistics cleared")
    
    async def close(self) -> None:
        """Close output manager and cleanup resources."""
        await self.stop()
        
        # Close component connections
        if self.tts and hasattr(self.tts, "close"):
            await self.tts.close()
        
        if self.avatar and hasattr(self.avatar, "close"):
            await self.avatar.close()
        
        if self.stream and hasattr(self.stream, "close"):
            await self.stream.close()
        
        log.info("Output manager closed")