from __future__ import annotations

"""
input/chat_listener.py

ChatListener - External chat platform integration
- Placeholder for Discord, Twitch, Slack, etc. integration
- Message queue handling
- User authentication and permissions
"""

import asyncio
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a message from a chat platform."""
    content: str
    user_id: str
    username: str
    platform: str
    channel: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp == 0.0:
            self.timestamp = asyncio.get_event_loop().time()


class ChatListener:
    """Handles input from various chat platforms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.platforms = self.config.get("platforms", ["discord", "twitch"])
        self.enabled = self.config.get("enabled", False)
        
        # Message queues for each platform
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.running = False
        
        # Initialize queues
        for platform in self.platforms:
            self.message_queues[platform] = asyncio.Queue()
        
        log.info(f"ChatListener initialized with platforms: {self.platforms}")
    
    async def start(self):
        """Start the chat listener."""
        if not self.enabled:
            log.info("ChatListener disabled in config")
            return
            
        self.running = True
        log.info("ChatListener started")
        
        # Start listeners for each platform
        tasks = []
        for platform in self.platforms:
            if platform == "discord":
                task = asyncio.create_task(self._discord_listener())
                tasks.append(task)
            elif platform == "twitch":
                task = asyncio.create_task(self._twitch_listener())
                tasks.append(task)
            # Add more platforms as needed
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop the chat listener."""
        self.running = False
        log.info("ChatListener stopped")
    
    async def get_message(self) -> Optional[Dict[str, Any]]:
        """Get the next message from any platform."""
        if not self.running:
            return None
        
        # Check all platform queues for messages
        for platform, queue in self.message_queues.items():
            try:
                # Non-blocking get
                if not queue.empty():
                    message = await asyncio.wait_for(queue.get(), timeout=0.1)
                    return {
                        "content": message.content,
                        "platform": message.platform,
                        "user_id": message.user_id,
                        "username": message.username,
                        "channel": message.channel,
                        "timestamp": message.timestamp,
                        "metadata": message.metadata
                    }
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Error getting message from {platform}: {e}")
        
        return None
    
    async def _discord_listener(self):
        """Listen for Discord messages."""
        log.info("Discord listener started (placeholder)")
        
        while self.running:
            try:
                # Placeholder - would integrate with discord.py
                await asyncio.sleep(5)
                
                # Simulate a test message occasionally
                if self.config.get("test_mode", False):
                    test_message = ChatMessage(
                        content="Hello from Discord!",
                        user_id="test_user_123",
                        username="TestUser",
                        platform="discord",
                        channel="general"
                    )
                    await self.message_queues["discord"].put(test_message)
                    
            except Exception as e:
                log.error(f"Discord listener error: {e}")
                await asyncio.sleep(10)
    
    async def _twitch_listener(self):
        """Listen for Twitch chat messages."""
        log.info("Twitch listener started (placeholder)")
        
        while self.running:
            try:
                # Placeholder - would integrate with Twitch IRC or API
                await asyncio.sleep(7)
                
                # Simulate a test message occasionally
                if self.config.get("test_mode", False):
                    test_message = ChatMessage(
                        content="PogChamp Nice stream!",
                        user_id="twitch_user_456",
                        username="TwitchViewer",
                        platform="twitch",
                        channel="ritsu_stream"
                    )
                    await self.message_queues["twitch"].put(test_message)
                    
            except Exception as e:
                log.error(f"Twitch listener error: {e}")
                await asyncio.sleep(10)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat listener statistics."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "platforms": self.platforms,
            "queue_sizes": {
                platform: queue.qsize() 
                for platform, queue in self.message_queues.items()
            }
        }
    
    async def send_message(self, platform: str, channel: str, message: str) -> bool:
        """Send a message to a chat platform (placeholder)."""
        log.info(f"Would send message to {platform}:{channel}: {message}")
        
        # Placeholder - would actually send message
        return True
    
    def close(self):
        """Close the chat listener."""
        self.running = False
        log.info("ChatListener closed")