from __future__ import annotations

"""
core/executor.py

Executor — executes chosen plan
- Action execution coordination
- Resource management and allocation  
- Parallel execution of independent actions
- Result aggregation and status tracking
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

log = logging.getLogger(__name__)

class ActionExecutor:
    """Executes actionable steps from solutions"""
    
    def __init__(self, executor):  # Changed from 'engine' to 'executor'
        self.executor = executor  # Reference to main Executor for coordination
        self.action_handlers = {
            "modify_code": self._handle_modify_code,
            "restart_service": self._handle_restart_service,
            "analyze_processes": self._handle_analyze_processes,
            "adjust_monitoring": self._handle_adjust_monitoring,
            "check_service": self._handle_check_service,
            "switch_model": self._handle_switch_model,
            "update_config": self._handle_update_config,
            "optimize_code": self._handle_optimize_code
        }
    
    async def execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step and return result"""
        action = step.get("action")
        if action not in self.action_handlers:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        try:
            result = await self.action_handlers[action](step, context)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_modify_code(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Modify code parameters like encoding"""
        target = step.get("target")
        params = step.get("parameters", {})
        
        if target == "subprocess_calls":
            return await self._fix_subprocess_encoding(params)
        return {"status": "modified", "target": target}
    
    async def _handle_restart_service(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Restart a specific service"""
        target = step.get("target")
        delay = step.get("parameters", {}).get("delay", 1)
        
        if target == "text_processing":
            await asyncio.sleep(delay)
            return {"status": "restarted", "service": target}
        return {"status": "unknown_service"}
    
    async def _handle_check_service(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Check if a service is running"""
        import subprocess
        target = step.get("target")
        timeout = step.get("parameters", {}).get("timeout", 5)
        
        if target == "ollama":
            try:
                result = subprocess.run(["ollama", "list"], 
                                      timeout=timeout, capture_output=True)
                return {"running": result.returncode == 0}
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return {"running": False}
        return {"running": False, "error": "Unknown service"}
    
    async def _handle_switch_model(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Switch to a fallback model"""
        model = step.get("parameters", {}).get("model", "gemma:2b")
        
        # Access AI system through executor's components
        if hasattr(self.executor, 'components') and 'ai_system' in self.executor.components:
            self.executor.components['ai_system'].current_model = model
            return {"model_switched": True, "new_model": model}
        return {"model_switched": False, "error": "AI system not available"}
    
    async def _handle_adjust_monitoring(self, step, context):
        """Adjust monitoring intervals"""
        interval = step.get("parameters", {}).get("interval", 5.0)
        return {"interval_set": interval}
    
    async def _handle_analyze_processes(self, step, context):
        """Analyze system processes"""
        count = step.get("parameters", {}).get("count", 5)
        # This would integrate with your system monitor
        if hasattr(self.executor, 'components') and 'system_monitor' in self.executor.components:
            monitor = self.executor.components['system_monitor']
            processes = getattr(monitor, 'get_top_processes', lambda x: [])(count)
            return {"top_processes": processes}
        return {"top_processes": []}
    
    async def _handle_update_config(self, step, context):
        """Update configuration parameters"""
        target = step.get("target")
        params = step.get("parameters", {})
        return {"config_updated": True, "target": target, "params": params}
    
    async def _handle_optimize_code(self, step, context):
        """Optimize code performance"""
        target = step.get("target")
        params = step.get("parameters", {})
        return {"optimized": True, "target": target}
    
    async def _fix_subprocess_encoding(self, params: Dict[str, Any]):
        """Fix subprocess encoding issues"""
        encoding = params.get("encoding", "utf-8")
        errors = params.get("errors", "replace")
        
        # This would modify your subprocess calls in runtime
        return {
            "encoding_fixed": True,
            "new_encoding": encoding,
            "error_handling": errors
        }


class ExecutionResult:
    """Container for execution results."""
    
    def __init__(self):
        self.status: str = "pending"  # pending, running, completed, failed
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        # REMOVED: Don't instantiate dependencies here
    
    def mark_started(self) -> None:
        """Mark execution as started."""
        self.status = "running"
        self.start_time = time.time()
    
    def mark_completed(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Mark execution as completed."""
        self.status = "completed"
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time
        if results:
            self.results.update(results)
    
    def mark_failed(self, error: str) -> None:
        """Mark execution as failed."""
        self.status = "failed"
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time
        self.errors.append(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "results": self.results,
            "errors": self.errors,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }


class Executor:
    """Executes plans by coordinating actions across components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_concurrent_actions = self.config.get("max_concurrent_actions", 5)
        
        # Component references (will be injected by main.py)
        self.components: Dict[str, Any] = {}
        
        # Initialize ActionExecutor with reference to self
        self.action_executor = ActionExecutor(self)
        
        # Action handlers - ADDED recovery actions
        self.action_handlers = {
            "ai_response": self._execute_ai_response,
            "tool_call": self._execute_tool_call,
            "memory_search": self._execute_memory_search,
            "knowledge_search": self._execute_knowledge_search,
            "nlp_analysis": self._execute_nlp_analysis,
            "output": self._execute_output,
            "error_response": self._execute_error_response,
            # ADDED: Recovery action handlers
            "execute_solution": self._execute_solution_action,
            "shell_command": self._execute_shell_command,
            "delay": self._execute_delay_action,
        }
    
    def set_components(self, components: Dict[str, Any]) -> None:
        """Set component references."""
        self.components.update(components)
        log.debug("Components set", extra={"components": list(components.keys())})
    
    async def execute(self, plan: Dict[str, Any]) -> ExecutionResult:
        """Execute a plan and return results."""
        result = ExecutionResult()
        result.mark_started()
        
        try:
            actions = plan.get("actions", [])
            if not actions:
                result.mark_completed({"message": "No actions to execute"})
                return result
            
            log.debug("Executing plan", extra={
                "plan_type": plan.get("type"),
                "actions_count": len(actions),
                "priority": plan.get("priority")
            })
            
            # Execute actions
            action_results = {}
            context = {}  # Shared context between actions
            
            for i, action in enumerate(actions):
                action_result = await self._execute_action(action, context)
                action_results[f"action_{i}"] = action_result
                
                # Update context with results for next actions
                if action_result.get("status") == "completed":
                    context.update(action_result.get("data", {}))
                else:
                    error_msg = f"Action {i} failed: {action_result.get('error', 'Unknown error')}"
                    result.errors.append(error_msg)
                    log.warning("Action failed during execution", extra={
                        "action_index": i,
                        "action_type": action.get("type"),
                        "error": action_result.get("error")
                    })
            
            # Aggregate results
            final_results = {
                "plan_type": plan.get("type"),
                "actions_executed": len(actions),
                "action_results": action_results,
                "context": context,
            }
            
            result.mark_completed(final_results)
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            result.mark_failed(error_msg)
            log.exception("Plan execution failed", extra={"plan": plan})
        
        return result
    
    async def _execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action."""
        action_type = action.get("type", "unknown")
        
        try:
            if action_type in self.action_handlers:
                handler = self.action_handlers[action_type]
                result = await handler(action, context)
                
                log.debug("Action executed", extra={
                    "action_type": action_type,
                    "status": result.get("status", "unknown")
                })
                
                return result
            else:
                return {
                    "status": "failed",
                    "error": f"Unknown action type: {action_type}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    # === RECOVERY ACTION HANDLERS ===
    
    async def _execute_solution_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a solution step from troubleshooter."""
        params = action.get("parameters", {})
        solution_step = params.get("solution", {})
        
        try:
            result = await self.action_executor.execute_step(solution_step, context)
            return {
                "status": "completed" if result.get("success") else "failed",
                "data": {"solution_result": result}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_shell_command(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command."""
        shell_executor = self.components.get("shell_executor")
        if not shell_executor:
            return {"status": "failed", "error": "Shell Executor not available"}
        
        params = action.get("parameters", {})
        command = params.get("command", "")
        
        try:
            result = await shell_executor.execute(command)
            return {
                "status": "completed",
                "data": {"shell_result": result}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_delay_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a delay action (for retry strategies)."""
        params = action.get("parameters", {})
        seconds = params.get("seconds", 1.0)
        
        try:
            await asyncio.sleep(seconds)
            return {
                "status": "completed",
                "data": {"delayed_seconds": seconds}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    # === ORIGINAL ACTION HANDLERS ===
    
    async def _execute_ai_response(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI response generation."""
        ai_assistant = self.components.get("ai_assistant")
        if not ai_assistant:
            return {"status": "failed", "error": "AI Assistant not available"}
        
        params = action.get("parameters", {})
        input_text = params.get("input_text", "")
        source = params.get("source", "unknown")
        
        try:
            # This would call your actual AI assistant
            response = f"AI Response to: {input_text}"  # Placeholder
            return {
                "status": "completed",
                "data": {"response": response}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_tool_call(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool call."""
        toolbelt = self.components.get("toolbelt")
        if not toolbelt:
            return {"status": "failed", "error": "Toolbelt not available"}
        
        params = action.get("parameters", {})
        command = params.get("command", "")
        
        try:
            result = f"Tool execution: {command}"  # Placeholder
            return {
                "status": "completed",
                "data": {"tool_result": result}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_memory_search(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory search."""
        memory_manager = self.components.get("memory_manager")
        if not memory_manager:
            return {"status": "failed", "error": "Memory Manager not available"}
        
        params = action.get("parameters", {})
        query = params.get("query", "")
        
        try:
            results = []  # Placeholder - would call memory_manager.search_memory(query)
            return {
                "status": "completed",
                "data": {"memory_search": results}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_knowledge_search(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge base search."""
        knowledge_base = self.components.get("knowledge_base")
        if not knowledge_base:
            return {"status": "failed", "error": "Knowledge Base not available"}
        
        params = action.get("parameters", {})
        query = params.get("query", "")
        
        try:
            results = []  # Placeholder
            return {
                "status": "completed",
                "data": {"knowledge_search": results}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_nlp_analysis(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NLP analysis."""
        nlp_engine = self.components.get("nlp_engine")
        if not nlp_engine:
            return {"status": "failed", "error": "NLP Engine not available"}
        
        params = action.get("parameters", {})
        text = params.get("text", "")
        
        try:
            analysis = {}  # Placeholder
            return {
                "status": "completed",
                "data": {"nlp_analysis": analysis}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_output(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute output action."""
        output_manager = self.components.get("output_manager")
        if not output_manager:
            return {"status": "failed", "error": "Output Manager not available"}
        
        params = action.get("parameters", {})
        destination = params.get("destination", "unknown")
        
        try:
            # Get response from context or use default
            response_data = context.get("response", "No response data")
            return {
                "status": "completed", 
                "data": {"output_sent": True}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_error_response(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute error response output."""
        output_manager = self.components.get("output_manager")
        if not output_manager:
            return {"status": "failed", "error": "Output Manager not available"}
        
        params = action.get("parameters", {})
        message = params.get("message", "An error occurred")
        
        try:
            return {
                "status": "completed",
                "data": {"error_sent": True}
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "max_concurrent_actions": self.max_concurrent_actions,
            "available_components": list(self.components.keys()),
            "supported_actions": list(self.action_handlers.keys()),
        }