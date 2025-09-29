from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class Planner:
    """Plans actions based on incoming events and current context."""

    TOOL_KEYWORDS = ["run", "execute", "calculate", "search", "file", "save", "load"]
    MEMORY_KEYWORDS = ["remember", "recall", "previous", "earlier", "before"]
    CREATIVE_KEYWORDS = ["create", "write", "generate", "make", "invent", "compose"]
    ANALYSIS_KEYWORDS = ["analyze", "compare", "evaluate", "assess", "review"]
    URGENT_KEYWORDS = ["urgent", "quickly", "asap", "immediately", "now"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.current_goals: List[Dict[str, Any]] = []
        self.active_plans: Dict[str, Dict[str, Any]] = {}

        self.strategies = {
            "direct_response": self._plan_direct_response,
            "tool_execution": self._plan_tool_execution,
            "information_gathering": self._plan_information_gathering,
            "creative_task": self._plan_creative_task,
            "analysis_task": self._plan_analysis_task,
        }

    def decide(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on the best plan of action for an event.

        Args:
            event: Input event to plan for.

        Returns:
            Plan dictionary with actions to execute.
        """
        try:
            analysis = self._analyze_event(event)
            strategy = self._choose_strategy(analysis)
            plan = self.strategies[strategy](event, analysis)

            plan.update(
                {
                    "strategy": strategy,
                    "event_id": event.get("id", "unknown"),
                    "priority": self._calculate_priority(analysis),
                    "estimated_duration": self._estimate_duration(plan),
                }
            )

            log.debug(
                "Generated plan",
                extra={
                    "strategy": strategy,
                    "priority": plan["priority"],
                    "actions_count": len(plan.get("actions", [])),
                },
            )
            return plan

        except Exception as e:
            log.error(
                "Planning failed",
                extra={"event": event, "error": str(e)},
                exc_info=True,
            )
            return self._fallback_plan(event)

    def _analyze_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze event to understand context and requirements."""
        event_type = event.get("type", "unknown")
        content = event.get("content", "")
        source = event.get("source", "unknown")

        analysis = {
            "type": event_type,
            "content": content,
            "source": source,
            "complexity": "simple",
            "requires_tools": False,
            "requires_memory": False,
            "requires_creativity": False,
            "requires_analysis": False,
            "urgency": "normal",
        }

        if content:
            content_lower = content.lower()
            word_count = len(content.split())

            if word_count > 50 or "complex" in content_lower:
                analysis["complexity"] = "complex"
            elif word_count > 20:
                analysis["complexity"] = "medium"

            if any(k in content_lower for k in self.TOOL_KEYWORDS):
                analysis["requires_tools"] = True

            if any(k in content_lower for k in self.MEMORY_KEYWORDS):
                analysis["requires_memory"] = True

            if any(k in content_lower for k in self.CREATIVE_KEYWORDS):
                analysis["requires_creativity"] = True

            if any(k in content_lower for k in self.ANALYSIS_KEYWORDS):
                analysis["requires_analysis"] = True

            if any(k in content_lower for k in self.URGENT_KEYWORDS):
                analysis["urgency"] = "high"

        return analysis

    def _choose_strategy(self, analysis: Dict[str, Any]) -> str:
        """Choose the best planning strategy based on event analysis."""
        if analysis["requires_tools"]:
            return "tool_execution"
        if analysis["requires_creativity"]:
            return "creative_task"
        if analysis["requires_analysis"]:
            return "analysis_task"
        if analysis["requires_memory"] or analysis["complexity"] == "complex":
            return "information_gathering"
        return "direct_response"

    def _plan_direct_response(self, event: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a direct conversational response."""
        return {
            "type": "direct_response",
            "actions": [
                {
                    "type": "ai_response",
                    "target": "ai_assistant",
                    "parameters": {
                        "input_text": analysis["content"],
                        "source": analysis["source"],
                        "mode": "direct",
                    },
                },
                {
                    "type": "output",
                    "target": "output_manager",
                    "parameters": {"destination": analysis["source"], "format": "text"},
                },
            ],
        }

    def _plan_tool_execution(self, event: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan tool execution and response."""
        return {
            "type": "tool_execution",
            "actions": [
                {
                    "type": "tool_call",
                    "target": "toolbelt",
                    "parameters": {"command": analysis["content"], "source": analysis["source"]},
                },
                {
                    "type": "ai_response",
                    "target": "ai_assistant",
                    "parameters": {
                        "input_text": analysis["content"],
                        "source": analysis["source"],
                        "mode": "tool_use",
                        "context": "tool_execution",
                    },
                },
                {
                    "type": "output",
                    "target": "output_manager",
                    "parameters": {"destination": analysis["source"], "format": "text"},
                },
            ],
        }

    def _plan_information_gathering(self, event: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan information gathering and contextualized response."""
        return {
            "type": "information_gathering",
            "actions": [
                {
                    "type": "memory_search",
                    "target": "memory_manager",
                    "parameters": {"query": analysis["content"], "limit": 5},
                },
                {
                    "type": "knowledge_search",
                    "target": "knowledge_base",
                    "parameters": {"query": analysis["content"], "limit": 3},
                },
                {
                    "type": "ai_response",
                    "target": "ai_assistant",
                    "parameters": {
                        "input_text": analysis["content"],
                        "source": analysis["source"],
                        "mode": "analytical",
                        "use_context": True,
                    },
                },
                {
                    "type": "output",
                    "target": "output_manager",
                    "parameters": {"destination": analysis["source"], "format": "text"},
                },
            ],
        }

    def _plan_creative_task(self, event: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan creative task execution."""
        return {
            "type": "creative_task",
            "actions": [
                {
                    "type": "ai_response",
                    "target": "ai_assistant",
                    "parameters": {
                        "input_text": analysis["content"],
                        "source": analysis["source"],
                        "mode": "creative",
                    },
                },
                {
                    "type": "output",
                    "target": "output_manager",
                    "parameters": {"destination": analysis["source"], "format": "text"},
                },
            ],
        }

    def _plan_analysis_task(self, event: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan analytical task execution."""
        return {
            "type": "analysis_task",
            "actions": [
                {
                    "type": "nlp_analysis",
                    "target": "nlp_engine",
                    "parameters": {"text": analysis["content"]},
                },
                {
                    "type": "ai_response",
                    "target": "ai_assistant",
                    "parameters": {
                        "input_text": analysis["content"],
                        "source": analysis["source"],
                        "mode": "analytical",
                    },
                },
                {
                    "type": "output",
                    "target": "output_manager",
                    "parameters": {"destination": analysis["source"], "format": "text"},
                },
            ],
        }

    def _calculate_priority(self, analysis: Dict[str, Any]) -> int:
        """Calculate priority score for the plan (1-10)."""
        priority = 5  # default

        if analysis["urgency"] == "high":
            priority += 3

        if analysis["complexity"] == "complex":
            priority += 2
        elif analysis["complexity"] == "medium":
            priority += 1

        if analysis["requires_tools"]:
            priority += 1

        return min(10, max(1, priority))

    def _estimate_duration(self, plan: Dict[str, Any]) -> float:
        """Estimate execution duration in seconds."""
        base_duration = 1.0  # base response time
        actions = plan.get("actions", [])

        for action in actions:
            action_type = action.get("type", "")

            if action_type == "ai_response":
                base_duration += 2.0
            elif action_type == "tool_call":
                base_duration += 3.0
            elif action_type in ["memory_search", "knowledge_search"]:
                base_duration += 0.5
            else:
                base_duration += 0.1

        return base_duration

    def _fallback_plan(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a safe fallback plan when normal planning fails."""
        log.error("Fallback plan triggered", extra={"event": event})
        return {
            "type": "fallback",
            "strategy": "direct_response",
            "priority": 1,
            "estimated_duration": 1.0,
            "actions": [
                {
                    "type": "error_response",
                    "target": "output_manager",
                    "parameters": {
                        "message": "I'm having trouble processing your request. Could you please try again?",
                        "destination": event.get("source", "unknown"),
                    },
                }
            ],
        }

    def add_goal(self, goal: Dict[str, Any]) -> None:
        """Add a new goal to the planner."""
        self.current_goals.append(goal)
        log.debug("Added goal", extra={"goal": goal})

    def get_active_plans(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active plans."""
        return self.active_plans.copy()

    def clear_completed_plans(self) -> None:
        """Remove completed plans from active list."""
        completed = [
            plan_id for plan_id, plan in self.active_plans.items() if plan.get("status") == "completed"
        ]

        for plan_id in completed:
            del self.active_plans[plan_id]

        if completed:
            log.debug("Cleared completed plans", extra={"count": len(completed)})
