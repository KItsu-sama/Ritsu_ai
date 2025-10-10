from __future__ import annotations

"""
core/Ritsu_self.py 

Ritsu Core (aka Ritsu_self)
Acts as the brain of Ritsu. Wraps the LLM, memory, and planning logic.
Supports both normal + streaming responses.

RitsuSelf — evolving metadata + self-reflection
- Stores traits, goals, and reflections
- Reads/writes memory.json (under data/)
- Modular: no direct I/O except JSON persistence
- Used by planning.py & self_improvement.py
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class RitsuSelf:
    def __init__(self, config: Optional[Dict[str, Any]] = None, path: Optional[Path] = None):
        self.config = config or {}
        self.memory_path: Path = path or Path("data/memory.json")

        # Metadata structure
        self.traits: Dict[str, Any] = {
            "baseline": {"curiosity": 0.7, "stability": 0.8, "adaptability": 0.6},
            "evolving": {},
        }
        self.goals: List[Dict[str, Any]] = []
        self.reflections: List[str] = []

        self._load()

    # ---------------------------- Persistence ----------------------------
    def _load(self) -> None:
        if not self.memory_path.exists():
            log.info("memory.json not found, starting fresh")
            return
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.traits = data.get("traits", self.traits)
            self.goals = data.get("goals", [])
            self.reflections = data.get("reflections", [])
            log.debug("Loaded RitsuSelf metadata", extra={"path": str(self.memory_path)})
        except Exception:
            log.exception("Failed to load memory.json")

    def _save(self) -> None:
        data = {
            "traits": self.traits,
            "goals": self.goals,
            "reflections": self.reflections,
        }
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            log.debug("Saved RitsuSelf metadata", extra={"path": str(self.memory_path)})
        except Exception:
            log.exception("Failed to save memory.json")

    # ------------------------- Public API -------------------------
    def analyze_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze one interaction (conversation turn, event, etc.).
        Returns a dict with extracted signals useful for planning.
        """
        signals = {
            "positivity": 0.0,
            "negativity": 0.0,
            "engagement": 0.0,
        }

        text = interaction.get("text") or ""
        if not text:
            return signals

        lowered = text.lower()
        if any(word in lowered for word in ["good", "great", "love", "thanks"]):
            signals["positivity"] += 0.5
        if any(word in lowered for word in ["bad", "hate", "annoy", "problem"]):
            signals["negativity"] += 0.5
        if len(text.split()) > 12:
            signals["engagement"] += 0.3

        log.debug("Analyzed interaction", extra={"signals": signals})
        return signals

    def update_metadata(self, signals: Dict[str, Any]) -> None:
        """Update evolving traits/goals based on signals."""
        evolving = self.traits.setdefault("evolving", {})

        # Adjust adaptability or stability based on signals
        if signals.get("positivity", 0) > 0:
            evolving["adaptability"] = evolving.get("adaptability", 0.5) + 0.01
        if signals.get("negativity", 0) > 0:
            evolving["stability"] = evolving.get("stability", 0.5) + 0.01

        # Keep values bounded
        for k, v in evolving.items():
            evolving[k] = max(0.0, min(1.0, v))

        log.debug("Updated metadata", extra={"traits": self.traits})
        self._save()

    def reflect(self, context: Optional[str] = None) -> str:
        """Generate a reflection string and save it."""
        reflection = ""
        if context:
            reflection = f"Reflecting on: {context}"
        else:
            reflection = "I am evolving steadily, learning from interactions."

        self.reflections.append(reflection)
        self._save()

        log.debug("Added reflection", extra={"reflection": reflection})
        return reflection

    # ------------------------- Convenience -------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "traits": self.traits,
            "goals": self.goals,
            "reflections": self.reflections,
        }

    def close(self) -> None:
        self._save()
