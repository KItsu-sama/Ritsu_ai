from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from collections import deque

log = logging.getLogger(__name__)

class Planner:
    """Unified Planner — handles event analysis, routing, policy, simulation, and plan generation."""

    # ---- Keyword-based heuristics ----
    TOOL_KEYWORDS = ["run", "execute", "calculate", "search", "file", "save", "load"]
    MEMORY_KEYWORDS = ["remember", "recall", "previous", "before"]
    MEMORY_KEYWORDS = ["remember", "recall", "previous", "before", "earlier"]  
    CREATIVE_KEYWORDS = ["create", "write", "generate", "make", "compose", "invent"] 
    ANALYSIS_KEYWORDS = ["analyze", "compare", "evaluate", "assess", "review"] 
    URGENT_KEYWORDS = ["urgent", "immediately", "asap", "critical", "now"]

    

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.strategies = {
                            "direct_response": self._plan_direct_response,
                            "tool_execution": self._plan_tool_execution,
                            "information_gathering": self._plan_information_gathering,
                            "creative_task": self._plan_creative_task,
                            "analysis_task": self._plan_analysis_task,
                          }
        self.current_goals: List[Dict[str, Any]] = []
        self.active_plans: Dict[str, Dict[str, Any]] = {}
        self.context_memory = deque(maxlen=10)

        self.enable_memory = self.config.get("enable_memory", True)

        # ---- Internal subsystems ----
        self.router_rules = self._default_router_rules()
        self.policy = self._default_policy_rules()
        self.simulator_enabled = True

    # =====================================================
    # 🧩 Core Decision Loop
    # =====================================================
    def decide(self, event: Dict[str, Any], system_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main planning entrypoint.
        Accepts optional system_status dict (telemetry) to enrich analysis.
        
        Main planning entrypoint.
        Args:
            event: Input event dictionary (e.g., {"type": "user", "content": "...", "id": "123"}).
        Returns:
            Plan dictionary with actions, strategy, priority, etc. May include simulation_result or policy_violation.
        """
        try:
            # Analyze once, then merge telemetry immediately so routing/strategy can consider it
            analysis = self._analyze_event(event)
            if system_status:
                # Attach system_status under a stable key for downstream decisions
                analysis.setdefault("system_status", system_status)

            strategy = self._choose_strategy(analysis)

            # Internal router
            expert = self._route_to_expert(analysis)

            # Build plan
            plan = self._generate_plan(strategy, event, analysis, expert)

            # Policy check
            if not self._validate_policy(plan):
                return self._policy_violation(event)

            # Optional dry-run simulation
            if self.simulator_enabled:
                plan["simulation_result"] = self._simulate(plan)

            # Save memory
            if self.enable_memory:
                self.context_memory.append({"event": event, "plan": plan})
                self.active_plans[event.get("id", "unknown")] = plan
            self.active_plans[event.get("id", "unknown")] = plan

            log.debug(
                "Generated plan",
                extra={
                    "event_id": event.get("id", "unknown"),
                    "strategy": strategy,
                    "expert": expert,
                    "priority": plan["priority"],
                    "actions_count": len(plan.get("actions", [])),
                    "simulation": "enabled" if self.simulator_enabled else "disabled",  # Tie to A's feature
                    },
                )
            
            # analysis already computed and merged with system_status above
            return plan

        except Exception as e:
            log.error("Planner failed", extra={"event": event, "error": str(e)}, exc_info=True)
            return self._fallback_plan(event)

    # =====================================================
    # 🔍 Event Analysis
    # =====================================================
    def _analyze_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        content = event.get("content", "")
        lower = content.lower() if content else ""  # B's defensiveness: Handle empty
        word_count = len(content.split()) if content else 0
        analysis = {  #Initialize to safe defaults
            "type": event.get("type", "unknown"),
            "content": content,
            "source": event.get("source", "unknown"),
            "complexity": "simple",
            "requires_tools": False,
            "requires_memory": False,
            "requires_creativity": False,
            "requires_analysis": False,
            "urgency": "normal",
        }
        if content:  # Skip if empty 
            analysis["requires_tools"] = any(k in lower for k in self.TOOL_KEYWORDS)
            analysis["requires_memory"] = any(k in lower for k in self.MEMORY_KEYWORDS)
            analysis["requires_creativity"] = any(k in lower for k in self.CREATIVE_KEYWORDS)
            analysis["requires_analysis"] = any(k in lower for k in self.ANALYSIS_KEYWORDS)
            analysis["urgency"] = "high" if any(k in lower for k in self.URGENT_KEYWORDS) else "normal"
            if word_count > 50 or "complex" in lower:
                analysis["complexity"] = "complex"
            elif word_count > 20:
                analysis["complexity"] = "medium"
        return analysis

    # =====================================================
    # 🎯 Strategy Selection + Routing
    # =====================================================
    def _choose_strategy(self, analysis: Dict[str, Any]) -> str:
        if analysis["requires_tools"]:
            return "tool_execution"
        if analysis["requires_creativity"]:
            return "creative_task"
        if analysis["requires_analysis"]:
            return "analysis_task"
        if analysis["requires_memory"] or analysis["complexity"] == "complex":
            return "information_gathering"
        return "direct_response"

    def _route_to_expert(self, analysis: Dict[str, Any]) -> str:
        """Internal router - rule-based, can be replaced by ML later."""
        for key, expert in self.router_rules.items():
            if key in analysis["content"].lower():
                return expert
        # fallback expert
        return "general_reasoning"

    def _default_router_rules(self) -> Dict[str, str]:
        return {
            "code": "expert_code_debug",
            "bug": "expert_code_debug",
            "cpu": "expert_system_perf",
            "lag": "expert_system_perf",
            "network": "expert_network",
            "wifi": "expert_network",
            "fan": "expert_hardware",
            "rgb": "expert_hardware",
        }

    # =====================================================
    # Policy & Safety System
    # =====================================================
    def _default_policy_rules(self) -> Dict[str, bool]:
        return {
            "allow_kill_process": False,
            "allow_file_delete": False,
            "allow_driver_update": False,
            "allow_fan_control": True,
        }

    def _validate_policy(self, plan: Dict[str, Any]) -> bool:
        for action in plan.get("actions", []):
            if action["type"] in ["tool_call", "hardware_control"]:
                params = action.get("parameters", {})
                if "kill" in params.get("command", "") and not self.policy["allow_kill_process"]:
                    return False
        return True

    def _policy_violation(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "policy_violation",
            "actions": [
                {
                    "type": "output",
                    "target": "output_manager",
                    "parameters": {
                        "message": "⚠️ Action blocked by safety policy.",
                        "destination": event.get("source", "unknown"),
                    },
                }
            ],
        }

    # =====================================================
    # Simulation Layer (Dry-Run)
    # =====================================================
    def _simulate(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Pretend to run the plan safely and estimate impact."""
        summary = []
        for action in plan.get("actions", []):
            summary.append({
                "action": action["type"],
                "impact": "low" if action["type"] != "tool_call" else "medium",
                "status": "ok"
            })

        return {"dry_run": True, "result": summary}

    # =====================================================
    # Plan Builders
    # =====================================================
    def _generate_plan(self, strategy, event, analysis, expert):
        # Use registry instead of getattr
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")  # Defensive, like B's simplicity
        base = self.strategies[strategy](event, analysis)
        base["expert"] = expert  # Keep A's routing
        base["event_id"] = event.get("id", "unknown")  # From B: Add for logging/tracking
        base["priority"] = self._calculate_priority(analysis)
        base["estimated_duration"] = self._estimate_duration(base)
        return base

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

    # =====================================================
    # Utility
    # =====================================================
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

