"""
Planner Manager - Intelligent routing between planning systems

Routes requests to appropriate planner based on complexity:
- Simple/conversational → planning.py (fast response, event-driven)
- Complex/multi-step → module_planning.py (thorough decomposition, dependencies)

Provides unified interface, fallback handling, and graceful degradation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PlannerType(Enum):
    """Available planner types"""
    EVENT_DRIVEN = "event_driven"      # planning.py - fast, conversational
    MODULE_BASED = "module_based"      # module_planning.py - complex, structured
    HYBRID = "hybrid"                   # Uses both in sequence


class ComplexityLevel(Enum):
    """Request complexity classification"""
    SIMPLE = 1          # Single action, direct response
    MODERATE = 2        # Few steps, simple dependencies
    COMPLEX = 3         # Multi-step, parallel execution
    CRITICAL = 4        # High risk, requires full analysis


@dataclass
class RoutingDecision:
    """Decision on which planner to use"""
    planner_type: PlannerType
    complexity: ComplexityLevel
    reasoning: str
    confidence: float
    fallback_planner: Optional[PlannerType] = None


class PlannerManager:
    """
    Orchestrates between event-driven and module-based planners.
    
    Features:
    - Intelligent routing based on request analysis
    - Graceful fallback if primary planner fails
    - Unified interface for both planning systems (compatible with existing code)
    - Performance monitoring and adaptive routing
    
    Compatible with existing main.py code that expects:
    - planner.decide(event, system_status) method
    """
    
    def __init__(
        self,
        event_planner=None,
        module_planner=None,
        config: Optional[Dict[str, Any]] = None,
        llm=None  # For compatibility with safe_init
    ):
        """
        Initialize planner manager.
        
        Args:
            event_planner: Instance of Planner from planning.py
            module_planner: Instance of PlannerOrchestrator from module_planning.py
            config: Configuration dict
            llm: LLM instance (for compatibility, passed to planners if needed)
        """
        self.config = config or {}
        self.llm = llm
        
        # Initialize planners if not provided
        if event_planner is None:
            event_planner = self._init_event_planner()
        if module_planner is None:
            module_planner = self._init_module_planner()
            
        self.event_planner = event_planner
        self.module_planner = module_planner
        
        # Routing thresholds
        self.complexity_threshold = self.config.get("complexity_threshold", ComplexityLevel.MODERATE)
        self.auto_fallback = self.config.get("auto_fallback", True)
        self.prefer_fast = self.config.get("prefer_fast", True)  # Prefer event planner when uncertain
        
        # Performance tracking
        self.performance_stats = {
            PlannerType.EVENT_DRIVEN: {"calls": 0, "failures": 0, "avg_time": 0.0},
            PlannerType.MODULE_BASED: {"calls": 0, "failures": 0, "avg_time": 0.0},
        }
        
        logger.info(
            "PlannerManager initialized",
            extra={
                "event_planner": "available" if event_planner else "missing",
                "module_planner": "available" if module_planner else "missing",
                "auto_fallback": self.auto_fallback,
            }
        )
    
    def _init_event_planner(self):
        """Initialize event planner if not provided."""
        try:
            from core.planning import Planner
            return Planner(config=self.config.get("event_planner_config"))
        except Exception as e:
            logger.warning(f"Failed to initialize event planner: {e}")
            return None
    
    def _init_module_planner(self):
        """Initialize module planner if not provided."""
        try:
            from core.module_planning import PlannerOrchestrator
            return PlannerOrchestrator()
        except Exception as e:
            logger.warning(f"Failed to initialize module planner: {e}")
            return None
    
    # ==========================================
    # MAIN INTERFACE - Compatible with existing code
    # ==========================================
    
    def decide(
        self,
        event: Dict[str, Any],
        system_status: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - routes to appropriate planner.
        
        SYNCHRONOUS version for compatibility with existing main.py code.
        For async usage, use plan() method instead.
        
        Args:
            event: Input event dict (e.g., {"type": "user", "content": "...", "id": "123"})
            system_status: Optional system telemetry
            
        Returns:
            Plan dict with actions, metadata, and execution strategy
        """
        # Run async plan() in sync context
        logger.debug("PlannerManager.decide: received event", extra={"event_id": event.get("id", "unknown"), "preview": event.get("content", "")[:120]})
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create task
                return asyncio.create_task(self.plan(event, system_status))
            else:
                # Run in new event loop
                return loop.run_until_complete(self.plan(event, system_status))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.plan(event, system_status))
    
    async def plan(
        self,
        event: Dict[str, Any],
        system_status: Optional[Dict[str, Any]] = None,
        force_planner: Optional[PlannerType] = None
    ) -> Dict[str, Any]:
        """
        Main entry point (async version) - routes to appropriate planner.
        """
        logger.debug(
            "PlannerManager.plan: start",
            extra={
                "event_id": event.get("id", "unknown"),
                "preview": event.get("content", "")[:120],
            },
        )

        try:
            # 1. Analyze request complexity
            routing = (
                self._analyze_and_route(event, system_status)
                if not force_planner
                else RoutingDecision(force_planner, ComplexityLevel.MODERATE, "forced", 1.0)
            )

            logger.debug(
                "Routing decision",
                extra={
                    "planner": routing.planner_type.value,
                    "complexity": routing.complexity.value,
                    "reasoning": routing.reasoning,
                    "confidence": routing.confidence,
                },
            )

            # 2. Execute with chosen planner
            plan = await self._execute_planner(routing, event, system_status)

            # 3. Enrich plan with metadata
            plan["planner_used"] = routing.planner_type.value
            plan["complexity_level"] = routing.complexity.value
            plan["routing_confidence"] = routing.confidence

            return plan

        except Exception as e:
            logger.error(
                "PlannerManager failed",
                extra={"event": event, "error": str(e)},
                exc_info=True,
            )
            return self._emergency_fallback(event)

    
    def _analyze_and_route(
        self,
        event: Dict[str, Any],
        system_status: Optional[Dict[str, Any]]
    ) -> RoutingDecision:
        """
        Analyze request and decide which planner to use.
        
        Factors considered:
        - Word count and linguistic complexity
        - Presence of multi-step indicators
        - Risk/safety requirements
        - System load and resource availability
        """
        content = event.get("content", "").lower()
        word_count = len(content.split())
        
        # Initialize scores
        complexity_score = 0
        reasoning_parts = []
        
        # === Complexity Indicators ===
        
        # 1. Multi-step keywords
        multi_step_keywords = [
            "then", "after", "first", "next", "finally", "step",
            "process", "workflow", "sequence", "order", "pipeline"
        ]
        if any(kw in content for kw in multi_step_keywords):
            complexity_score += 2
            reasoning_parts.append("multi-step indicators")
        
        # 2. Dependency keywords
        dependency_keywords = [
            "depends on", "requires", "needs", "before", "after",
            "prerequisite", "conditional", "if.*then"
        ]
        if any(kw in content for kw in dependency_keywords):
            complexity_score += 2
            reasoning_parts.append("dependencies detected")
        
        # 3. Destructive/risky operations
        risky_keywords = [
            "delete", "remove", "shutdown", "kill", "modify",
            "update", "install", "uninstall", "format"
        ]
        if any(kw in content for kw in risky_keywords):
            complexity_score += 3
            reasoning_parts.append("risky operation")
        
        # 4. Automation/scheduling
        automation_keywords = [
            "automate", "schedule", "whenever", "recurring",
            "automatically", "trigger", "monitor"
        ]
        if any(kw in content for kw in automation_keywords):
            complexity_score += 2
            reasoning_parts.append("automation request")
        
        # 5. Long/complex requests
        if word_count > 50:
            complexity_score += 2
            reasoning_parts.append("lengthy request")
        elif word_count > 100:
            complexity_score += 3
            reasoning_parts.append("very complex request")
        
        # 6. System status considerations
        if system_status:
            cpu_usage = system_status.get("cpu_percent", 0)
            memory_usage = system_status.get("memory_percent", 0)
            
            # High load → prefer lightweight event planner
            if cpu_usage > 80 or memory_usage > 80:
                complexity_score -= 1
                reasoning_parts.append("high system load")
        
        # 7. Simple conversational patterns
        simple_keywords = [
            "hello", "hi", "thanks", "thank you", "okay",
            "what is", "who is", "explain", "tell me"
        ]
        if any(kw in content for kw in simple_keywords) and word_count < 20:
            complexity_score -= 2
            reasoning_parts.append("simple conversation")
        
        # === Decision Logic ===
        
        if complexity_score <= 1:
            complexity_level = ComplexityLevel.SIMPLE
            planner_type = PlannerType.EVENT_DRIVEN
            confidence = 0.9
        elif complexity_score <= 3:
            complexity_level = ComplexityLevel.MODERATE
            planner_type = PlannerType.EVENT_DRIVEN if self.prefer_fast else PlannerType.MODULE_BASED
            confidence = 0.7
        elif complexity_score <= 5:
            complexity_level = ComplexityLevel.COMPLEX
            planner_type = PlannerType.MODULE_BASED
            confidence = 0.8
        else:
            complexity_level = ComplexityLevel.CRITICAL
            planner_type = PlannerType.MODULE_BASED
            confidence = 0.95
        
        # Fallback logic
        fallback = PlannerType.EVENT_DRIVEN if planner_type == PlannerType.MODULE_BASED else None
        
        reasoning = f"Score: {complexity_score}, " + ", ".join(reasoning_parts) if reasoning_parts else "default routing"
        
        return RoutingDecision(
            planner_type=planner_type,
            complexity=complexity_level,
            reasoning=reasoning,
            confidence=confidence,
            fallback_planner=fallback
        )
    
    async def _execute_planner(
        self,
        routing: RoutingDecision,
        event: Dict[str, Any],
        system_status: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute the chosen planner with fallback support."""
        import time
        start_time = time.time()
        
        try:
            if routing.planner_type == PlannerType.EVENT_DRIVEN:
                plan = await self._execute_event_planner(event, system_status)
            elif routing.planner_type == PlannerType.MODULE_BASED:
                plan = await self._execute_module_planner(event, system_status)
            else:
                raise ValueError(f"Unknown planner type: {routing.planner_type}")
            
            # Update stats
            elapsed = time.time() - start_time
            self._update_stats(routing.planner_type, success=True, duration=elapsed)
            
            return plan
            
        except Exception as e:
            logger.error(
                f"{routing.planner_type.value} planner failed",
                extra={"error": str(e), "event": event},
                exc_info=True
            )
            
            # Update failure stats
            self._update_stats(routing.planner_type, success=False, duration=0)
            
            # Try fallback if enabled
            if self.auto_fallback and routing.fallback_planner:
                logger.warning(f"Attempting fallback to {routing.fallback_planner.value}")
                fallback_routing = RoutingDecision(
                    planner_type=routing.fallback_planner,
                    complexity=routing.complexity,
                    reasoning=f"fallback from {routing.planner_type.value}",
                    confidence=0.5
                )
                return await self._execute_planner(fallback_routing, event, system_status)
            
            raise
    
    async def _execute_event_planner(
        self,
        event: Dict[str, Any],
        system_status: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute planning.py (event-driven planner)."""
        if not self.event_planner:
            raise RuntimeError("Event planner not initialized")
        
        # Event planner may be sync or async
        if asyncio.iscoroutinefunction(self.event_planner.decide):
            plan = await self.event_planner.decide(event, system_status)
        else:
            # Wrap sync call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            plan = await loop.run_in_executor(
                None, 
                lambda: self.event_planner.decide(event, system_status)
            )
        
        return plan
    
    async def _execute_module_planner(
        self,
        event: Dict[str, Any],
        system_status: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute module_planning.py (advanced planner)."""
        if not self.module_planner:
            raise RuntimeError("Module planner not initialized")
        
        # Module planner is async
        plan = await self.module_planner.plan(event)
        return plan
    
    def _emergency_fallback(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Last resort fallback when all planners fail."""
        logger.critical("Emergency fallback activated")
        return {
            "type": "emergency_fallback",
            "strategy": "error_recovery",
            "priority": 10,
            "actions": [
                {
                    "type": "error_response",
                    "target": "output_manager",
                    "parameters": {
                        "message": "I'm experiencing technical difficulties. Please try again in a moment.",
                        "destination": event.get("source", "unknown"),
                    },
                }
            ],
            "planner_used": "emergency",
            "errors": ["All planners failed"],
        }
    
    def _update_stats(self, planner_type: PlannerType, success: bool, duration: float):
        """Update performance statistics."""
        stats = self.performance_stats[planner_type]
        stats["calls"] += 1
        
        if not success:
            stats["failures"] += 1
        
        if duration > 0:
            # Running average
            n = stats["calls"]
            stats["avg_time"] = (stats["avg_time"] * (n - 1) + duration) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "performance": self.performance_stats.copy(),
            "config": {
                "auto_fallback": self.auto_fallback,
                "prefer_fast": self.prefer_fast,
                "complexity_threshold": self.complexity_threshold.value,
            }
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        for planner_type in self.performance_stats:
            self.performance_stats[planner_type] = {
                "calls": 0,
                "failures": 0,
                "avg_time": 0.0
            }
        logger.info("Performance statistics reset")
