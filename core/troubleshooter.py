from __future__ import annotations

"""
core/advanced_troubleshooting_engine.py

AdvancedTroubleshootingEngine — Comprehensive error handling, recovery, and system diagnostics
- Combines error recovery strategies with proactive problem detection
- Learns from failures and improves over time
- Provides detailed diagnostics and actionable solutions
- Integrated with system monitoring and log analysis
"""

import asyncio
import re
import json
import logging
import time
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from core.executor import ActionExecutor
from config.config import _model_
from dataclasses import dataclass


log = logging.getLogger(__name__)

@dataclass
class Problem:
    """Represents a detected problem"""
    id: str
    severity: str  # critical, high, medium, low
    category: str  # system, code, network, etc.
    title: str
    description: str
    symptoms: List[str]
    detected_at: str
    context: Dict[str, Any]
    suggested_solutions: List[str]
    status: str = "detected"  # detected, analyzing, fixing, resolved

@dataclass
class Solution:
    """Represents a potential solution"""
    id: str
    problem_pattern: str
    steps: List[str]
    success_rate: float
    prerequisites: List[str]
    risks: List[str]

class TroubleshootingEngine:
    """Advanced troubleshooting and problem-solving engine for Ritsu and Kitsu"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_retries = self.config.get("max_retries", 3)
        self.base_retry_delay = self.config.get("base_retry_delay", 1.0)
        self.enable_learning = self.config.get("enable_learning", True)
        
        # Error tracking and recovery
        self.error_history: List[Dict[str, Any]] = []
        self.retry_counts: Dict[str, int] = {}
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Problem detection and diagnostics
        self.problems: Dict[str, Problem] = {}
        self.solutions_db: Dict[str, Solution] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.knowledge_base_file = Path("troubleshooting_kb.json")

        self.action_executor = ActionExecutor(self)
        
        self.load_knowledge_base()
    
    def _initialize_recovery_strategies(self) -> Dict[str, Any]:
        """Initialize all recovery strategies"""
        return {
            "connection_error": self._handle_connection_error,
            "timeout_error": self._handle_timeout_error,
            "permission_error": self._handle_permission_error,
            "resource_error": self._handle_resource_error,
            "parse_error": self._handle_parse_error,
            "general_error": self._handle_general_error,
            "unicode_error": self._handle_unicode_error,
            "high_cpu_usage": self._handle_performance_issue,
            "high_memory_usage": self._handle_performance_issue,
            "ollama_error": self._handle_llm_error
        }
    
    # === Error Recovery System ===
    
    async def attempt_recovery(
        self, 
        plan: Dict[str, Any], 
        error: Any,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Attempt to recover from a failed plan execution."""
        try:
            error_analysis = self._analyze_error(error, plan, execution_context)
            self._record_error(error_analysis)
            
            if not self._should_attempt_recovery(error_analysis):
                log.info("Recovery not attempted", extra={"error_type": error_analysis["type"]})
                return None
            
            strategy_name = self._select_recovery_strategy(error_analysis)
            if strategy_name not in self.recovery_strategies:
                log.warning("No recovery strategy available", extra={"error_type": error_analysis["type"]})
                return None
            
            recovery_plan = await self.recovery_strategies[strategy_name](error_analysis, plan)
            
            if recovery_plan:
                log.info("Recovery plan generated", extra={
                    "strategy": strategy_name,
                    "original_plan_type": plan.get("type"),
                    "recovery_actions": len(recovery_plan.get("actions", []))
                })
            
            return recovery_plan
            
        except Exception as e:
            log.exception("Troubleshooter failed", extra={"error": str(e)})
            return None
    
    def _analyze_error(self, error: Any, plan: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced error analysis with pattern matching."""
        error_str = str(error)
        error_type = type(error).__name__
        
        # Extended error classification
        error_classification = "general_error"
        error_patterns = {
            "connection_error": ["connection", "network", "unreachable", "refused"],
            "timeout_error": ["timeout", "timed out"],
            "permission_error": ["permission", "access denied", "forbidden"],
            "resource_error": ["memory", "disk space", "resource", "out of memory"],
            "parse_error": ["parse", "json", "syntax", "format", "unicode"],
            "unicode_error": ["unicode", "encoding", "charmap"],
            "ollama_error": ["ollama", "model", "llm", "ai service"],
            "performance_error": ["cpu", "performance", "slow", "bottleneck"]
        }
        
        for classification, keywords in error_patterns.items():
            if any(keyword in error_str.lower() for keyword in keywords):
                error_classification = classification
                break
        
        # Determine severity
        severity = "medium"
        if any(keyword in error_str.lower() for keyword in ["critical", "fatal", "emergency"]):
            severity = "high"
        elif any(keyword in error_str.lower() for keyword in ["warning", "minor"]):
            severity = "low"
        
        # Enhanced recoverability assessment
        recoverable = error_classification in [
            "connection_error", "timeout_error", "resource_error", 
            "parse_error", "unicode_error", "performance_error"
        ]
        
        return {
            "type": error_classification,
            "original_type": error_type,
            "message": error_str,
            "severity": severity,
            "recoverable": recoverable,
            "plan_type": plan.get("type"),
            "timestamp": time.time(),
            "context": context or {},
        }
    
    def _record_error(self, error_analysis: Dict[str, Any]) -> None:
        """Record error for learning and statistics."""
        self.error_history.append(error_analysis)
        
        # Keep only recent errors (last 100)
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        # Track retry counts
        error_key = f"{error_analysis['type']}_{error_analysis['plan_type']}"
        self.retry_counts[error_key] = self.retry_counts.get(error_key, 0) + 1
        
        # Also record as a problem for diagnostics
        self._create_problem_from_error(error_analysis)
    
    def _should_attempt_recovery(self, error_analysis: Dict[str, Any]) -> bool:
        """Determine if recovery should be attempted."""
        if not error_analysis["recoverable"]:
            return False
        
        error_key = f"{error_analysis['type']}_{error_analysis['plan_type']}"
        retry_count = self.retry_counts.get(error_key, 0)
        
        if retry_count >= self.max_retries:
            log.warning("Max retries exceeded", extra={
                "error_key": error_key,
                "retry_count": retry_count
            })
            return False
        
        if error_analysis["severity"] == "high" and error_analysis.get("plan_type") in ["system", "critical"]:
            return False
        
        return True
    
    def _select_recovery_strategy(self, error_analysis: Dict[str, Any]) -> str:
        """Select the appropriate recovery strategy."""
        return error_analysis["type"]
    
    # === Recovery Strategy Implementations ===
    
    async def _handle_connection_error(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        retry_delay = self._calculate_retry_delay(error_analysis)
        return {
            "type": "recovery_retry",
            "strategy": "connection_error",
            "actions": [
                {"type": "delay", "parameters": {"seconds": retry_delay}},
                {"type": "retry_original", "parameters": {"original_plan": original_plan}}
            ],
            "metadata": {
                "retry_attempt": self.retry_counts.get(f"connection_error_{original_plan.get('type')}", 0),
                "delay": retry_delay
            }
        }
    
    async def _handle_timeout_error(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        retry_delay = self._calculate_retry_delay(error_analysis)
        return {
            "type": "recovery_retry",
            "strategy": "timeout_error",
            "actions": [
                {"type": "delay", "parameters": {"seconds": retry_delay}},
                {"type": "retry_with_timeout", "parameters": {
                    "original_plan": original_plan,
                    "increased_timeout": True
                }}
            ]
        }
    
    async def _handle_unicode_error(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {
            "type": "recovery_fix",
            "strategy": "unicode_error",
            "actions": [
                {"type": "apply_encoding_fix", "parameters": {
                    "encoding": "utf-8",
                    "errors": "replace"
                }},
                {"type": "retry_original", "parameters": {"original_plan": original_plan}}
            ]
        }
    
    async def _handle_performance_issue(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {
            "type": "recovery_optimize",
            "strategy": "performance_issue",
            "actions": [
                {"type": "resource_cleanup", "parameters": {"cleanup_type": "memory"}},
                {"type": "optimize_process", "parameters": {"priority": "low"}},
                {"type": "retry_original", "parameters": {"original_plan": original_plan}}
            ]
        }
    
    async def _handle_llm_error(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {
            "type": "recovery_fallback",
            "strategy": "llm_error",
            "actions": [
                {"type": "switch_to_fallback_model", "parameters": {}},
                {"type": "retry_original", "parameters": {"original_plan": original_plan}}
            ]
        }
    
    async def _handle_permission_error(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {
            "type": "recovery_fallback",
            "strategy": "permission_error",
            "actions": [
                {"type": "error_response", "target": "output_manager", "parameters": {
                    "message": "Permission error encountered. Please check access rights.",
                    "destination": original_plan.get("source", "unknown")
                }}
            ]
        }
    
    async def _handle_resource_error(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        retry_delay = self._calculate_retry_delay(error_analysis) * 2
        return {
            "type": "recovery_retry",
            "strategy": "resource_error",
            "actions": [
                {"type": "resource_cleanup", "parameters": {"cleanup_type": "memory"}},
                {"type": "delay", "parameters": {"seconds": retry_delay}},
                {"type": "retry_original", "parameters": {"original_plan": original_plan}}
            ]
        }
    
    async def _handle_parse_error(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {
            "type": "recovery_fallback",
            "strategy": "parse_error",
            "actions": [
                {"type": "error_response", "target": "output_manager", "parameters": {
                    "message": "Format parsing error. Please rephrase your request.",
                    "destination": original_plan.get("source", "unknown")
                }}
            ]
        }
    
    async def _handle_general_error(self, error_analysis: Dict[str, Any], original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if error_analysis["severity"] == "high":
            return {
                "type": "recovery_fallback",
                "strategy": "general_error",
                "actions": [
                    {"type": "error_response", "target": "output_manager", "parameters": {
                        "message": "Serious error encountered. Please try again later.",
                        "destination": original_plan.get("source", "unknown")
                    }}
                ]
            }
        
        retry_delay = self._calculate_retry_delay(error_analysis)
        return {
            "type": "recovery_retry",
            "strategy": "general_error",
            "actions": [
                {"type": "delay", "parameters": {"seconds": retry_delay}},
                {"type": "retry_original", "parameters": {"original_plan": original_plan}}
            ]
        }
    
    def _calculate_retry_delay(self, error_analysis: Dict[str, Any]) -> float:
        """Calculate exponential backoff delay for retries."""
        error_key = f"{error_analysis['type']}_{error_analysis['plan_type']}"
        retry_count = self.retry_counts.get(error_key, 0)
        delay = self.base_retry_delay * (2 ** retry_count)
        max_delay = self.config.get("max_retry_delay", 30.0)
        return min(delay, max_delay)
    
    # === Problem Detection and Diagnostics ===
    
    def load_knowledge_base(self):
        """Load existing knowledge base or create default one."""
        if self.knowledge_base_file.exists():
            try:
                with open(self.knowledge_base_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert dict back to Solution objects
                    self.solutions_db = {
                        key: Solution(**value) for key, value in data.get('solutions', {}).items()
                    }
            except Exception as e:
                logging.error(f"Failed to load knowledge base: {e}")
                self._create_default_knowledge_base()
        else:
            self._create_default_knowledge_base()
    
    def _create_default_knowledge_base(self):
        """Create default knowledge base with executable solutions."""
        default_solutions = {
            "unicode_decode_error": Solution(
                id="unicode_fix",
                problem_pattern=r"UnicodeDecodeError.*charmap.*can't decode",
                steps=[
                    {
                        "action": "modify_code",
                        "target": "subprocess_calls", 
                        "parameters": {"encoding": "utf-8", "errors": "replace"},
                        "description": "Add encoding parameters to subprocess calls"
                    },
                    {
                        "action": "restart_service", 
                        "target": "text_processing",
                        "parameters": {"delay": 2},
                        "description": "Restart text processing service"
                    }
                ],
                success_rate=0.95,
                prerequisites=["file system access"],
                risks=["May mask other encoding issues"]
            ),
            "high_cpu_usage": Solution(
                id="cpu_optimization",
                problem_pattern=r"CPU usage.*high|CPU.*100%",
                steps=[
                    {
                        "action": "analyze_processes",
                        "target": "system",
                        "parameters": {"count": 5},
                        "description": "Identify top 5 CPU-consuming processes"
                    },
                    {
                        "action": "adjust_monitoring",
                        "target": "frequency", 
                        "parameters": {"interval": 5.0},
                        "description": "Reduce monitoring frequency to 5 seconds"
                    },
                    {
                        "action": "optimize_code",
                        "target": "loops",
                        "parameters": {"sleep_time": 0.1},
                        "description": "Add sleep intervals in CPU-intensive loops"
                    }
                ],
                success_rate=0.85,
                prerequisites=["system monitoring access"],
                risks=["May affect performance monitoring accuracy"]
            ),
            "ollama_connection_error": Solution(
                id="ollama_fallback",
                problem_pattern=r"ollama.*not.*responding|ollama.*error|model.*not.*exist",
                steps=[
                    {
                        "action": "check_service",
                        "target": "ollama",
                        "parameters": {"timeout": 10},
                        "description": "Verify Ollama server is running"
                    },
                    {
                        "action": "switch_model",
                        "target": "fallback",
                        "parameters": {"model": _model_},
                        "description": "Switch to fallback model"
                    },
                    {
                        "action": "update_config",
                        "target": "timeouts",
                        "parameters": {"timeout": 30},
                        "description": "Increase request timeout to 30 seconds"
                    }
                ],
                success_rate=0.80,
                prerequisites=["ollama CLI access"],
                risks=["May degrade response quality in fallback mode"]
            )
        }
    
        self.solutions_db = default_solutions
        self.save_knowledge_base()
    
    def save_knowledge_base(self):
        """Save knowledge base to file."""
        try:
            # Convert Solution objects to dict for JSON serialization
            solutions_dict = {
                key: {
                    "id": sol.id,
                    "problem_pattern": sol.problem_pattern,
                    "steps": sol.steps,
                    "success_rate": sol.success_rate,
                    "prerequisites": sol.prerequisites,
                    "risks": sol.risks
                } for key, sol in self.solutions_db.items()
            }
            
            data = {
                "solutions": solutions_dict,
                "updated_at": datetime.now().isoformat()
            }
            with open(self.knowledge_base_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save knowledge base: {e}")
    
    def analyze_logs(self, log_file: Path, hours: int = 1) -> List[Problem]:
        """Analyze log files for problems."""
        problems = []
        
        if not log_file.exists():
            return problems
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            error_patterns = {
                "unicode_error": (r"UnicodeDecodeError.*", "critical", "encoding"),
                "timeout_error": (r"TimeoutExpired|timeout.*expired", "high", "network"),
                "permission_error": (r"PermissionError|Access.*denied", "medium", "system"),
                "connection_error": (r"Connection.*refused|Connection.*failed", "high", "network"),
                "memory_error": (r"MemoryError|Out of memory", "critical", "system"),
                "ollama_error": (r"OLLAMA_ERROR|ollama.*failed", "medium", "llm")
            }
            
            for error_type, (pattern, severity, category) in error_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    problem = Problem(
                        id=f"{error_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        severity=severity,
                        category=category,
                        title=f"Detected {error_type.replace('_', ' ').title()}",
                        description=f"Found {len(matches)} instances of {error_type}",
                        symptoms=matches[:5],
                        detected_at=datetime.now().isoformat(),
                        context={"log_file": str(log_file), "pattern": pattern},
                        suggested_solutions=self._get_solutions_for_problem(error_type, matches)
                    )
                    problems.append(problem)
                    self.problems[problem.id] = problem
                    
        except Exception as e:
            logging.error(f"Failed to analyze logs: {e}")
        
        return problems
    
    def analyze_system_metrics(self, monitor) -> List[Problem]:
        """Analyze system metrics for potential issues."""
        problems = []
        
        if not monitor or not monitor.latest:
            return problems
        
        metrics = monitor.latest
        cpu = metrics.get('cpu_percent', 0)
        memory = metrics.get('memory', {})
        mem_percent = memory.get('percent', 0)
        
        if cpu > 90:
            problem = Problem(
                id=f"high_cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="high" if cpu > 95 else "medium",
                category="performance",
                title="High CPU Usage Detected",
                description=f"CPU usage is at {cpu:.1f}%",
                symptoms=[f"CPU at {cpu:.1f}%"],
                detected_at=datetime.now().isoformat(),
                context={"cpu_percent": cpu},
                suggested_solutions=self._get_solutions_for_problem("high_cpu_usage", [f"CPU: {cpu}%"])
            )
            problems.append(problem)
            self.problems[problem.id] = problem
        
        if mem_percent > 85:
            problem = Problem(
                id=f"high_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity="high" if mem_percent > 90 else "medium", 
                category="performance",
                title="High Memory Usage Detected",
                description=f"Memory usage is at {mem_percent:.1f}%",
                symptoms=[f"Memory at {mem_percent:.1f}%"],
                detected_at=datetime.now().isoformat(),
                context={"memory_percent": mem_percent},
                suggested_solutions=self._get_solutions_for_problem("high_memory_usage", [f"Memory: {mem_percent}%"])
            )
            problems.append(problem)
            self.problems[problem.id] = problem
        
        return problems
    
    def _create_problem_from_error(self, error_analysis: Dict[str, Any]) -> None:
        """Create a Problem instance from error analysis."""
        problem = Problem(
            id=f"error_{error_analysis['type']}_{int(time.time())}",
            severity=error_analysis["severity"],
            category=error_analysis["type"],
            title=f"{error_analysis['type'].replace('_', ' ').title()} Error",
            description=error_analysis["message"],
            symptoms=[error_analysis["message"]],
            detected_at=datetime.now().isoformat(),
            context=error_analysis["context"],
            suggested_solutions=self._get_solutions_for_problem(error_analysis["type"], [error_analysis["message"]])
        )
        self.problems[problem.id] = problem
    
    def _get_solutions_for_problem(self, problem_type: str, evidence: List[str]) -> List[str]:
        """Get suggested solutions for a specific problem type."""
        solutions = []
        
        if problem_type in self.solutions_db:
            sol = self.solutions_db[problem_type]
            solutions.extend(sol.steps)
        
        for key, solution in self.solutions_db.items():
            pattern = solution.problem_pattern
            if pattern and any(re.search(pattern, str(ev), re.IGNORECASE) for ev in evidence):
                solutions.extend(solution.steps)
        
        if not solutions:
            solutions = [
                "Review recent system changes",
                "Check additional logs for context",
                "Monitor the issue for patterns",
                "Consider service restart if appropriate"
            ]
        
        return list(set(solutions))
    
    def generate_diagnostic_report(self, monitor=None) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": {},
            "detected_problems": [],
            "recovery_statistics": self.get_error_statistics(),
            "recommendations": []
        }
        
        # Analyze logs and system metrics
        log_files = [Path("monitor.log"), Path("kitsu.log"), Path("ritsu.log")]
        for log_file in log_files:
            if log_file.exists():
                problems = self.analyze_logs(log_file)
                report["detected_problems"].extend([
                    {
                        "id": p.id,
                        "severity": p.severity,
                        "title": p.title,
                        "description": p.description,
                        "solutions": p.suggested_solutions
                    } for p in problems
                ])
        
        if monitor:
            system_problems = self.analyze_system_metrics(monitor)
            report["detected_problems"].extend([
                {
                    "id": p.id,
                    "severity": p.severity, 
                    "title": p.title,
                    "description": p.description,
                    "solutions": p.suggested_solutions
                } for p in system_problems
            ])
            
            if monitor.latest:
                report["system_status"] = {
                    "cpu_percent": monitor.latest.get("cpu_percent", 0),
                    "memory_percent": monitor.latest.get("memory", {}).get("percent", 0),
                    "disk_usage": monitor.latest.get("disk", {}),
                    "active_processes": len(monitor.latest.get("top_procs", []))
                }
        
        report["recommendations"] = self._generate_recommendations(report["detected_problems"])
        return report
    
    def _generate_recommendations(self, problems: List[Dict]) -> List[str]:
        """Generate high-level recommendations."""
        recommendations = []
        severity_count = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for problem in problems:
            severity_count[problem["severity"]] += 1
        
        if severity_count["critical"] > 0:
            recommendations.append(f"🚨 {severity_count['critical']} critical issues require immediate attention")
        if severity_count["high"] > 2:
            recommendations.append(f"⚠️ {severity_count['high']} high-priority issues detected")
        
        if len(problems) > 5:
            recommendations.append("🔍 High number of issues - consider systematic review")
        if not recommendations:
            recommendations.append("✅ System appears stable")
        
        return recommendations
    
    def create_action_plan(self, problems: List[Problem]) -> Dict[str, Any]:
        """Create prioritized action plan."""
        action_plan = {
            "created_at": datetime.now().isoformat(),
            "priority_order": [],
            "estimated_time": "0-2 hours",
            "risk_level": "low"
        }
        
        severity_priority = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_problems = sorted(problems, key=lambda p: severity_priority.get(p.severity, 3))
        
        for i, problem in enumerate(sorted_problems, 1):
            action_item = {
                "step": i,
                "problem_id": problem.id,
                "title": problem.title,
                "actions": problem.suggested_solutions,
                "estimated_minutes": len(problem.suggested_solutions) * 10,
                "dependencies": []
            }
            action_plan["priority_order"].append(action_item)
        
        return action_plan
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_counts = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        recoverable_count = 0
        
        for error in self.error_history:
            error_type = error["type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            severity_counts[error["severity"]] += 1
            if error["recoverable"]:
                recoverable_count += 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_counts,
            "severity_distribution": severity_counts,
            "recoverable_errors": recoverable_count,
            "retry_counts": self.retry_counts.copy(),
            "recent_errors": self.error_history[-5:] if len(self.error_history) >= 5 else self.error_history,
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.error_history.clear()
        self.retry_counts.clear()
        self.problems.clear()
        log.info("AdvancedTroubleshootingEngine statistics reset")
    
    async def proactive_health_check(self, monitor) -> Dict[str, Any]:
        """Perform proactive system health check."""
        report = self.generate_diagnostic_report(monitor)
        
        # Check if any critical issues need immediate attention
        critical_issues = [p for p in report["detected_problems"] if p["severity"] in ["critical", "high"]]
        
        return {
            "health_status": "healthy" if not critical_issues else "degraded",
            "critical_issues_count": len(critical_issues),
            "report": report,
            "suggested_maintenance": self._suggest_maintenance_actions(report)
        }
    
    def _suggest_maintenance_actions(self, report: Dict[str, Any]) -> List[str]:
        """Suggest maintenance actions based on diagnostic report."""
        suggestions = []
        
        if report["recovery_statistics"]["total_errors"] > 10:
            suggestions.append("Review error patterns and implement preventive measures")
        
        if any(p["severity"] == "critical" for p in report["detected_problems"]):
            suggestions.append("Address critical issues immediately to prevent system instability")
        
        if report["system_status"].get("memory_percent", 0) > 80:
            suggestions.append("Consider memory optimization or cleanup procedures")
        
        return suggestions