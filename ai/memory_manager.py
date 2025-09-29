from __future__ import annotations

"""
ai/memory_manager.py

MemoryManager — short + long-term memory
- Short-term conversation context
- Long-term interaction patterns
- Memory consolidation and retrieval
- Context-aware memory management
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque

log = logging.getLogger(__name__)


class MemoryManager:
    """Manages short-term and long-term memory for conversations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory_limit = self.config.get("memory_limit", 1000)
        self.memory_path = Path("data/memory.json")
        
        # Short-term memory (recent interactions)
        self.short_term: deque = deque(maxlen=100)
        
        # Long-term memory (persistent)
        self.long_term: List[Dict[str, Any]] = []
        
        # Context tracking
        self.current_context: Dict[str, Any] = {}
        
        # Load existing memory
        self._load_memory()
    
    def _load_memory(self) -> None:
        """Load memory from persistent storage."""
        if not self.memory_path.exists():
            log.info("Memory file not found, starting fresh")
            return
        
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load long-term memory
            self.long_term = data.get("long_term", [])
            
            # Load recent short-term memory
            recent_interactions = data.get("recent", [])
            for interaction in recent_interactions[-50:]:  # Last 50 interactions
                self.short_term.append(interaction)
            
            log.info("Loaded memory", extra={
                "long_term_count": len(self.long_term),
                "short_term_count": len(self.short_term)
            })
            
        except Exception as e:
            log.error("Failed to load memory", extra={"error": str(e)})
    
    def _save_memory(self) -> None:
        """Save memory to persistent storage."""
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "long_term": self.long_term,
                "recent": list(self.short_term),
                "context": self.current_context,
            }
            
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            log.debug("Saved memory", extra={"path": str(self.memory_path)})
            
        except Exception as e:
            log.error("Failed to save memory", extra={"error": str(e)})
    
    async def store_interaction(
        self, 
        user_message: Dict[str, Any], 
        assistant_message: Dict[str, Any]
    ) -> None:
        """Store a user-assistant interaction pair.
        
        Args:
            user_message: User's message with metadata
            assistant_message: Assistant's response with metadata
        """
        interaction = {
            "timestamp": asyncio.get_event_loop().time(),
            "user": user_message,
            "assistant": assistant_message,
        }
        
        # Add to short-term memory
        self.short_term.append(interaction)
        
        # Update current context
        self._update_context(interaction)
        
        # Check if should be moved to long-term memory
        await self._consolidate_memory()
        
        # Save to disk periodically
        if len(self.short_term) % 10 == 0:
            self._save_memory()
    
    def _update_context(self, interaction: Dict[str, Any]) -> None:
        """Update current conversation context."""
        user_content = interaction["user"].get("content", "")
        source = interaction["user"].get("source", "unknown")
        
        # Track conversation patterns
        self.current_context.update({
            "last_user_input": user_content,
            "last_source": source,
            "interaction_count": self.current_context.get("interaction_count", 0) + 1,
            "topics": self._extract_topics(user_content),
        })
    
    def _extract_topics(self, text: str) -> List[str]:
        """Simple topic extraction from text."""
        # Basic keyword-based topic extraction
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            "programming": ["code", "python", "programming", "development", "function", "class"],
            "help": ["help", "assist", "support", "how", "what", "explain"],
            "creative": ["create", "write", "story", "poem", "generate", "make"],
            "analysis": ["analyze", "compare", "evaluate", "review", "assess"],
            "conversation": ["chat", "talk", "discuss", "conversation", "tell me"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    async def _consolidate_memory(self) -> None:
        """Move important short-term memories to long-term storage."""
        if len(self.short_term) < 20:  # Not enough for consolidation
            return
        
        # Simple consolidation: move older interactions to long-term
        # In a more advanced system, this would use importance scoring
        
        while len(self.long_term) >= self.memory_limit:
            # Remove oldest long-term memories
            self.long_term.pop(0)
        
        # Move some short-term to long-term
        consolidation_count = min(5, len(self.short_term) - 10)
        for _ in range(consolidation_count):
            if self.short_term:
                interaction = self.short_term.popleft()
                # Add importance score (simplified)
                interaction["importance"] = self._calculate_importance(interaction)
                self.long_term.append(interaction)
    
    def _calculate_importance(self, interaction: Dict[str, Any]) -> float:
        """Calculate importance score for memory consolidation."""
        score = 0.0
        
        user_content = interaction["user"].get("content", "")
        assistant_content = interaction["assistant"].get("content", "")
        
        # Longer interactions are more important
        score += min(len(user_content.split()), 20) / 20.0 * 0.3
        score += min(len(assistant_content.split()), 50) / 50.0 * 0.3
        
        # Certain topics are more important
        topics = self._extract_topics(user_content)
        important_topics = ["programming", "help", "analysis"]
        if any(topic in important_topics for topic in topics):
            score += 0.4
        
        return min(1.0, score)
    
    def get_recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context.
        
        Args:
            limit: Number of recent interactions to return
            
        Returns:
            List of recent interactions
        """
        return list(self.short_term)[-limit:] if limit > 0 else list(self.short_term)
    
    def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search through memory for relevant interactions.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant interactions
        """
        results = []
        query_lower = query.lower()
        
        # Search short-term memory
        for interaction in self.short_term:
            if self._interaction_matches_query(interaction, query_lower):
                results.append(interaction)
        
        # Search long-term memory
        for interaction in self.long_term:
            if self._interaction_matches_query(interaction, query_lower):
                results.append(interaction)
        
        # Sort by relevance (timestamp for now, could be improved)
        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return results[:limit]
    
    def _interaction_matches_query(self, interaction: Dict[str, Any], query: str) -> bool:
        """Check if an interaction matches a search query."""
        searchable_text = ""
        
        if "user" in interaction and "content" in interaction["user"]:
            searchable_text += interaction["user"]["content"].lower()
        
        if "assistant" in interaction and "content" in interaction["assistant"]:
            searchable_text += " " + interaction["assistant"]["content"].lower()
        
        return query in searchable_text
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state."""
        total_interactions = len(self.short_term) + len(self.long_term)
        
        # Aggregate topics from recent interactions
        all_topics = []
        for interaction in list(self.short_term)[-10:]:
            user_content = interaction.get("user", {}).get("content", "")
            all_topics.extend(self._extract_topics(user_content))
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_interactions": total_interactions,
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "current_context": self.current_context.copy(),
            "recent_topics": topic_counts,
        }
    
    def clear_context(self) -> None:
        """Clear current conversation context."""
        self.current_context.clear()
        log.info("Conversation context cleared")
    
    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term.clear()
        self.clear_context()
        log.info("Short-term memory cleared")
    
    async def close(self) -> None:
        """Save memory and cleanup."""
        self._save_memory()
        log.info("Memory manager closed")