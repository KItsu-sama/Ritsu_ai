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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque

log = logging.getLogger(__name__)

# Lazy import psutil to avoid hanging on Windows
_psutil = None
def _get_psutil():
    global _psutil
    if _psutil is None:
        try:
            import psutil
            _psutil = psutil
        except Exception:
            _psutil = False
    return _psutil if _psutil else None


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

        # Backwards-compatible memory_log and logger
        self.memory_log = self.short_term
        self.log = log

        # Load existing memory (with timeout protection)
        try:
            self._load_memory()
        except Exception as e:
            log.warning(f"Failed to load memory during init: {e}")
            # Continue anyway - memory will be empty but system will work

    async def get_user_patterns(self):
        """Return detected user patterns from topics and context."""
        # get_conversation_summary is async
        summary = await self.get_conversation_summary()
        patterns = []
        for topic, count in summary.get("recent_topics", {}).items():
            if count > 1:
                patterns.append(f"Frequent {topic} queries (x{count})")
        return patterns + [f"Source: {summary['current_context'].get('last_source', 'unknown')}"]
    
    def _load_memory(self) -> None:
        """Load memory from persistent storage."""
        if not self.memory_path.exists():
            log.info("Loaded memory")
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

            log.info("Loaded memory")

        except Exception as e:
            log.warning(f"Memory load skipped: {type(e).__name__}")
    
    def _save_memory(self) -> None:
        """Save memory to persistent storage."""
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "long_term": self.long_term,
                "recent": [item for item in self.short_term if item],  # Prune empties
                "context": {k: v for k, v in self.current_context.items() if v},  # Prune empties
            }
            
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.log.debug("Saved memory", extra={"path": str(self.memory_path)})
            
        except Exception as e:
            self.log.error("Failed to save memory", extra={"error": str(e)}, exc_info=True)
    
    async def store_interaction(self, user_message, assistant_message):
        try:
            # Single normalization: Extract content (support dict or str)
            def _extract_content(msg):
                if isinstance(msg, dict):
                    return msg.get('content') or msg.get('message') or msg.get('text') or str(msg)
                return str(msg) if msg is not None else ''

            user_content = _extract_content(user_message)
            assistant_content = _extract_content(assistant_message)

            # Skip trivial greetings/duplicates in one check
            trivial_greetings = [
                "hello how can i assist you today?", "hello how can i assist you?", "hello",
                "hi how can i assist you today?", "hi"
            ]
            if (isinstance(assistant_content, str) and 
                assistant_content.strip().lower() in trivial_greetings):
                # Still update context for flow
                interaction = {"user": {"content": user_content}, "assistant": {"content": assistant_content}}
                self._update_context(interaction)
                return

            # Check for duplicate assistant (last in short_term)
            last_assistant = None
            if self.short_term:
                last_item = self.short_term[-1]
                last_assistant = _extract_content(last_item.get("assistant", {}))
            if last_assistant == assistant_content:
                self._update_context({"user": {"content": user_content}, "assistant": {"content": assistant_content}})
                return

            # Build and store interaction
            interaction = {
                "user": {"role": "user", "content": user_content},
                "assistant": {"role": "assistant", "content": assistant_content},
                "timestamp": time.time(),
            }
            self.memory_log.append(interaction)
            
            # Update context
            self._update_context(interaction)
            
            # Consolidate and save periodically
            await self._consolidate_memory()
            if len(self.memory_log) % 10 == 0:
                self._save_memory()
        except Exception as e:
            self.log.exception("Failed to store interaction", exc_info=True)

    def _update_context(self, interaction: Dict[str, Any]) -> None:
        """Update current conversation context."""
        user_content = interaction["user"].get("content", "")
        source = interaction.get("source", "unknown")  # From event if present
        
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
            "trivia": ["fact", "surprise", "random", "info"],  # Ritsu-specific: Playful topics
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    async def _consolidate_memory(self) -> None:
        """Move important short-term memories to long-term storage."""
        # Trigger consolidation either normally or when system is under stress
        if len(self.short_term) < 20 and not self._system_under_load():
            return
        # Prune long-term if over limit
        while len(self.long_term) >= self.memory_limit:
            self.long_term.pop(0)
        
        # Move high-importance items (threshold 0.5); summarize for efficiency
        consolidation_count = 0
        to_move = []
        for interaction in list(self.short_term)[:5]:  # Check recent 5
            importance = self._calculate_importance(interaction)
            if importance > 0.5:
                # Summarize: Keep core, drop redundancy
                summary = {
                    "user": interaction["user"],
                    "assistant": interaction["assistant"],
                    "summary": f"Outcome: {interaction.get('outcome', 'success')}; Topics: {', '.join(self._extract_topics(interaction['user']['content']))}",
                    "importance": importance,
                    "timestamp": interaction["timestamp"],
                }
                to_move.append(summary)
                consolidation_count += 1
        
        for summary in to_move:
            self.short_term.remove(next(item for item in self.short_term if item["timestamp"] == summary["timestamp"]))  # Remove original
            self.long_term.append(summary)
        
        if consolidation_count > 0:
            self.log.debug(f"Consolidated {consolidation_count} memories")
    
    def _calculate_importance(self, interaction: dict) -> float:
        user_content = interaction.get("user", {}).get("content", "")
        assistant_content = interaction.get("assistant", {}).get("content", "")
        
        score = 0.0
        
        # Length-based (user: 30%, assistant: 20%)
        score += min(len(user_content.split()), 20) / 20.0 * 0.3
        score += min(len(assistant_content.split()), 50) / 50.0 * 0.2
        
        # Topic boosts (40%)
        topics = self._extract_topics(user_content)
        important_topics = ["programming", "help", "analysis", "trivia"]  # Ritsu-aligned
        if any(topic in important_topics for topic in topics):
            score += 0.4
        
        # Bonus for non-trivial (e.g., not greetings)
        if len(user_content.split()) > 5:
            score += 0.1
        
        return min(1.0, score)
    
    async def save_event(self, event: Dict[str, Any]) -> None:
        """Save an event to memory (compatibility method). Summarize for efficiency."""
        try:
            timestamp = event.get("timestamp", time.time())
            
            # Summarize event (avoid full Planner bloat)
            if "plan" in event:
                plan_type = event["plan"].get("type", "unknown")
                outcome = "success" if not event.get("result", {}).get("errors") else "error"
                summary = f"Plan: {plan_type}; Outcome: {outcome}"
                user_content = event.get("content", "")  # From original event
            else:
                summary = "System event"
                user_content = ""
            
            interaction = {
                "timestamp": timestamp,
                "event": {"type": event.get("type", "system_event"), "summary": summary},
                "user": {"content": user_content} if user_content else None,  # Optional
                "outcome": outcome if "outcome" in locals() else "unknown"
            }
            
            # Prune if no content (e.g., pure system)
            if not user_content and event.get("type") != "critical":
                return  # Skip trivial events
            
            self.short_term.append(interaction)
            
            # Save periodically
            if len(self.short_term) % 5 == 0:
                self._save_memory()
                
            self.log.debug("Event saved to memory", extra={"event_type": event.get("type", "unknown")})
            
        except Exception as e:
            self.log.error(f"Failed to save event: {e}", exc_info=True)
    
    def get_recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context."""
        return list(self.short_term)[-limit:] if limit > 0 else list(self.short_term)
    
    def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search through memory for relevant interactions."""
        results = []
        query_lower = query.lower()
        query_topics = self._extract_topics(query)
        
        def relevance(interaction) -> Any:
            text = (interaction.get("user", {}).get("content", "") + " " + 
                    interaction.get("assistant", {}).get("content", "")).lower()
            score = sum(1 for topic in query_topics if topic in self._extract_topics(text))
            score += 1 if query_lower in text else 0
            return score / max(1, len(query_topics))  # Normalize 0-1
        
        # Search short + long
        all_mem = list(self.short_term) + list(self.long_term)

        # Score each interaction by simple relevance heuristics
        scored: List[Dict[str, Any]] = []
        for interaction in all_mem:
            try:
                text = (
                    (interaction.get("user") or {}).get("content", "") + " " +
                    (interaction.get("assistant") or {}).get("content", "")
                ).lower()
            except Exception:
                # Fallback to string representation
                text = str(interaction).lower()

            # Basic relevance: matches query text + topic overlap
            score = 0.0
            if query_lower and query_lower in text:
                score += 1.0

            if query_topics:
                # count topic overlap
                overlap = sum(1 for t in query_topics if t in self._extract_topics(text))
                score += overlap * 0.5

            if score > 0:
                scored.append({"score": score, "interaction": interaction})

        # Sort by score desc then timestamp desc
        scored.sort(key=lambda x: (x["score"], x["interaction"].get("timestamp", 0)), reverse=True)

        return [s["interaction"] for s in scored[:limit]]

    def _system_under_load(self, cpu_threshold: int = 85, ram_threshold: int = 85) -> bool:
        """
        Check if the system is under heavy load.
        Returns True if CPU or RAM usage exceeds thresholds.

        If the system is under load, schedule an async summarization task (non-blocking)
        when possible. If no event loop is running, run a synchronous fallback.
        """
        # Simple recursion/re-entrancy guard
        if getattr(self, "_under_load_guard", False):
            # Already handling under-load; avoid re-entrancy
            return True
        try:
            psutil = _get_psutil()
            if not psutil:
                # psutil not available, assume not under load
                return False

            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().percent
            under = cpu > cpu_threshold or ram > ram_threshold

            if under:
                self.log.info(
                    "System under load detected (cpu=%s%%, ram=%s%%) — scheduling summarization.",
                    cpu, ram
                )

                # Set guard to avoid concurrent summarizations
                self._under_load_guard = True

                try:
                    # If running inside an asyncio loop, schedule an async summarization
                    loop = None
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                    if loop and loop.is_running():
                        # schedule and don't await (fire-and-forget)
                        loop.create_task(self._async_summarize_under_load())
                    else:
                        # No running loop (maybe called from sync context) — run a safe sync fallback
                        try:
                            summary = asyncio.run(self.get_conversation_summary())
                        except Exception as e:
                            self.log.warning(f"Sync summary fallback failed: {e}")
                            summary = {"summary": "Unavailable", "recent_topics": {}, "current_context": {}}

                        self.long_term.append({
                            "timestamp": time.time(),
                            "summary": summary.get("summary", "Quick summary under load."),
                            "recent_topics": summary.get("recent_topics", {}),
                            "context_snapshot": summary.get("current_context", {}),
                            "note": "Auto-summarized due to system stress (sync fallback)."
                        })
                        # Clear short-term to free memory, then persist
                        self.short_term.clear()
                        try:
                            self._save_memory()
                        except Exception as e:
                            self.log.warning(f"Failed to save memory during sync fallback: {e}")

                finally:
                    # Release the guard after scheduling (guard persists until summarization clears it)
                    # _async_summarize_under_load will clear the guard when done; but to avoid a stuck state
                    # if the async path wasn't used, clear it here.
                    if not (hasattr(self, "_async_summarize_running") and self._async_summarize_running):
                        # no async summarize in progress; clear guard
                        self._under_load_guard = False

            return under

        except Exception as e:
            self.log.warning(f"System load check failed: {e}", exc_info=True)
            # On error, assume not under load to avoid aggressive behavior
            return False


    async def _async_summarize_under_load(self) -> None:
        """
        Async helper that performs conversation summarization and persistence.
        This is scheduled by _system_under_load when an event loop is running.
        """
        # prevent multiple concurrent async summarizations
        if getattr(self, "_async_summarize_running", False):
            return

        self._async_summarize_running = True
        try:
            try:
                summary = await self.get_conversation_summary()
            except Exception as e:
                self.log.warning(f"Async summarization failed to get summary: {e}", exc_info=True)
                summary = {"summary": "Unavailable", "recent_topics": {}, "current_context": {}}

            self.long_term.append({
                "timestamp": time.time(),
                "summary": summary.get("summary", "Quick summary under load."),
                "recent_topics": summary.get("recent_topics", {}),
                "context_snapshot": summary.get("current_context", {}),
                "note": "Auto-summarized due to system stress (async)."
            })

            # Free short-term memory and persist
            self.short_term.clear()
            try:
                # run save in thread to avoid blocking event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._save_memory)
            except Exception as e:
                # If run_in_executor isn't available for some reason, try direct save
                try:
                    self._save_memory()
                except Exception as ee:
                    self.log.warning(f"Failed to save memory during async summarization: {ee}", exc_info=True)

            self.log.info("Async summarization under load completed and memory persisted.")
        finally:
            # clear guards
            self._async_summarize_running = False
            self._under_load_guard = False


    async def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Generate a structured summary of recent conversations.
        Safely handles missing data. Returns dict for future compatibility.
        """
        try:
            # 1 Gather last few short-term interactions
            recent = list(self.short_term)[-10:]
            if not recent:
                return {
                    "summary": "No recent conversation data available.",
                    "recent_topics": {},
                    "current_context": self.current_context
                }

            # 2 Extract content and topics
            combined_text = []
            topic_counts = {}

            for interaction in recent:
                user_text = interaction.get("user", {}).get("content", "")
                assistant_text = interaction.get("assistant", {}).get("content", "")
                combined_text.append(f"User: {user_text}")
                combined_text.append(f"Ritsu: {assistant_text}")

                # Topic extraction
                for topic in self._extract_topics(user_text + " " + assistant_text):
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

            # 3 Summarize into a short string
            joined = " ".join(combined_text)
            summary_text = (joined[:300] + "...") if len(joined) > 300 else joined

            # 4 Return structured summary (so get_user_patterns can use it)
            return {
                "summary": summary_text or "Conversation summary empty.",
                "recent_topics": topic_counts,
                "current_context": self.current_context or {},
            }

        except Exception as e:
            self.log.warning(f"get_conversation_summary failed: {e}", exc_info=True)
            return {
                "summary": "Summary unavailable due to error.",
                "recent_topics": {},
                "current_context": self.current_context or {},
            }