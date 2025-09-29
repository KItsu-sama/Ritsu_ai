from __future__ import annotations

"""
output/stream_adapter.py

StreamAdapter — connects to stream overlay/UI
- Streaming platform integration
- Overlay management and updates
- Real-time viewer interaction
- Chat and donation handling
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class StreamAdapter:
    """Manages streaming platform integration and overlays."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Stream configuration
        self.enabled = self.config.get("enabled", False)
        self.platforms = self.config.get("platforms", ["twitch", "youtube"])
        self.overlay_enabled = self.config.get("overlay_enabled", True)
        
        # Connection state
        self.connected_platforms = []
        self.overlay_connected = False
        
        # Message queue for streaming
        self.stream_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "overlay_updates": 0,
            "errors": 0,
            "by_platform": {},
        }
        
        log.info("Stream adapter initialized", extra={
            "enabled": self.enabled,
            "platforms": self.platforms,
            "overlay_enabled": self.overlay_enabled
        })
    
    async def start(self) -> None:
        """Start the stream processing task."""
        if not self.enabled:
            log.info("Stream adapter disabled, not starting")
            return
        
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(
                self._process_stream_queue(),
                name="stream_processor"
            )
            log.info("Stream processing started")
            
            # Initialize platform connections
            await self._initialize_connections()
    
    async def stop(self) -> None:
        """Stop the stream processing task."""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            log.info("Stream processing stopped")
            
            # Cleanup connections
            await self._cleanup_connections()
    
    async def send(self, data: Dict[str, Any]) -> None:
        """Send data to streaming platforms.
        
        Args:
            data: Stream data containing content and metadata
        """
        if not self.enabled:
            return
        
        try:
            # Add timestamp
            data["timestamp"] = asyncio.get_event_loop().time()
            
            # Queue for processing
            await self.stream_queue.put(data)
            
            log.debug("Stream data queued", extra={
                "data_type": data.get("type"),
                "content_preview": str(data.get("content", ""))[:50]
            })
            
        except Exception as e:
            log.error("Failed to queue stream data", extra={
                "data": data,
                "error": str(e)
            })
            self.stats["errors"] += 1
    
    async def _process_stream_queue(self) -> None:
        """Process queued stream messages."""
        try:
            while True:
                try:
                    # Get next stream data
                    data = await self.stream_queue.get()
                    
                    # Process the data
                    await self._process_stream_data(data)
                    
                    # Mark task as done
                    self.stream_queue.task_done()
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.error("Error processing stream data", extra={"error": str(e)})
                    self.stats["errors"] += 1
                    
        except asyncio.CancelledError:
            log.debug("Stream processor cancelled")
            raise
    
    async def _process_stream_data(self, data: Dict[str, Any]) -> None:
        """Process a single stream data item."""
        try:
            data_type = data.get("type", "message")
            content = data.get("content")
            
            # Update statistics
            self.stats["messages_sent"] += 1
            
            # Route to appropriate handlers
            if data_type == "message":
                await self._send_message(content, data)
            elif data_type == "overlay_update":
                await self._update_overlay(content, data)
            elif data_type == "notification":
                await self._send_notification(content, data)
            
            log.debug("Stream data processed", extra={
                "type": data_type,
                "platforms": len(self.connected_platforms)
            })
            
        except Exception as e:
            log.error("Failed to process stream data", extra={
                "data": data,
                "error": str(e)
            })
            self.stats["errors"] += 1
    
    async def _send_message(self, content: Any, metadata: Dict[str, Any]) -> None:
        """Send message to streaming platforms."""
        if not content:
            return
        
        # Convert content to string
        message = str(content) if not isinstance(content, str) else content
        
        # Send to connected platforms
        for platform in self.connected_platforms:
            try:
                await self._send_to_platform(platform, message, metadata)
                
                # Update platform statistics
                platform_stats = self.stats["by_platform"].setdefault(platform, 0)
                self.stats["by_platform"][platform] = platform_stats + 1
                
            except Exception as e:
                log.error("Failed to send to platform", extra={
                    "platform": platform,
                    "message": message,
                    "error": str(e)
                })
    
    async def _update_overlay(self, content: Any, metadata: Dict[str, Any]) -> None:
        """Update streaming overlay."""
        if not self.overlay_enabled or not self.overlay_connected:
            return
        
        try:
            # This would update the actual overlay
            # In a real implementation, this might use:
            # - OBS WebSocket API
            # - Browser source updates
            # - Direct overlay application communication
            
            self.stats["overlay_updates"] += 1
            
            log.info(f"[Stream] Overlay update: {content}")
            
        except Exception as e:
            log.error("Overlay update failed", extra={
                "content": content,
                "error": str(e)
            })
    
    async def _send_notification(self, content: Any, metadata: Dict[str, Any]) -> None:
        """Send notification to streaming platforms."""
        try:
            # This would send platform-specific notifications
            # Such as chat messages, alerts, or announcements
            
            log.info(f"[Stream] Notification: {content}")
            
        except Exception as e:
            log.error("Notification failed", extra={
                "content": content,
                "error": str(e)
            })
    
    async def _send_to_platform(self, platform: str, message: str, metadata: Dict[str, Any]) -> None:
        """Send message to specific platform."""
        # This would integrate with actual platform APIs:
        # 
        # For Twitch:
        # - Use Twitch IRC/API
        # - Handle chat rate limits
        # - Manage authentication
        # 
        # For YouTube:
        # - Use YouTube Live Chat API
        # - Handle authentication
        # - Manage message formatting
        # 
        # For Discord:
        # - Use Discord bot API
        # - Handle webhook integration
        
        # Placeholder implementation
        log.info(f"[Stream:{platform}] {message}")
        
        # Simulate network delay
        await asyncio.sleep(0.1)
    
    async def _initialize_connections(self) -> None:
        """Initialize connections to streaming platforms."""
        self.connected_platforms = []
        
        for platform in self.platforms:
            try:
                # This would establish actual platform connections
                success = await self._connect_to_platform(platform)
                
                if success:
                    self.connected_platforms.append(platform)
                    log.info(f"Connected to {platform}")
                else:
                    log.warning(f"Failed to connect to {platform}")
                    
            except Exception as e:
                log.error("Platform connection failed", extra={
                    "platform": platform,
                    "error": str(e)
                })
        
        # Initialize overlay connection
        if self.overlay_enabled:
            self.overlay_connected = await self._connect_overlay()
            if self.overlay_connected:
                log.info("Overlay connected")
            else:
                log.warning("Overlay connection failed")
    
    async def _connect_to_platform(self, platform: str) -> bool:
        """Connect to a specific streaming platform."""
        # Placeholder implementation
        log.info(f"Connecting to {platform} (placeholder)")
        await asyncio.sleep(0.5)  # Simulate connection time
        return True  # Always succeed in placeholder
    
    async def _connect_overlay(self) -> bool:
        """Connect to streaming overlay system."""
        # Placeholder implementation
        log.info("Connecting to overlay (placeholder)")
        await asyncio.sleep(0.3)  # Simulate connection time
        return True  # Always succeed in placeholder
    
    async def _cleanup_connections(self) -> None:
        """Cleanup platform and overlay connections."""
        for platform in self.connected_platforms:
            try:
                await self._disconnect_from_platform(platform)
                log.info(f"Disconnected from {platform}")
            except Exception as e:
                log.error("Platform disconnect failed", extra={
                    "platform": platform,
                    "error": str(e)
                })
        
        self.connected_platforms = []
        
        if self.overlay_connected:
            try:
                await self._disconnect_overlay()
                self.overlay_connected = False
                log.info("Overlay disconnected")
            except Exception as e:
                log.error("Overlay disconnect failed", extra={"error": str(e)})
    
    async def _disconnect_from_platform(self, platform: str) -> None:
        """Disconnect from a specific platform."""
        # Placeholder implementation
        log.debug(f"Disconnecting from {platform}")
        await asyncio.sleep(0.1)
    
    async def _disconnect_overlay(self) -> None:
        """Disconnect from overlay system."""
        # Placeholder implementation
        log.debug("Disconnecting from overlay")
        await asyncio.sleep(0.1)
    
    def get_connected_platforms(self) -> List[str]:
        """Get list of currently connected platforms."""
        return self.connected_platforms.copy()
    
    def is_overlay_connected(self) -> bool:
        """Check if overlay is connected."""
        return self.overlay_connected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "enabled": self.enabled,
            "connected_platforms": len(self.connected_platforms),
            "platform_names": self.connected_platforms,
            "overlay_connected": self.overlay_connected,
            "messages_sent": self.stats["messages_sent"],
            "overlay_updates": self.stats["overlay_updates"],
            "errors": self.stats["errors"],
            "by_platform": self.stats["by_platform"].copy(),
            "queue_size": self.stream_queue.qsize(),
        }
    
    def clear_stats(self) -> None:
        """Clear streaming statistics."""
        self.stats = {
            "messages_sent": 0,
            "overlay_updates": 0,
            "errors": 0,
            "by_platform": {},
        }
        log.info("Stream statistics cleared")
    
    async def close(self) -> None:
        """Close stream adapter and cleanup resources."""
        await self.stop()
        log.info("Stream adapter closed")