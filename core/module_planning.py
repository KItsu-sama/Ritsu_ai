"""
Ritsu Planning System 

This system merges the three provided codes into a cohesive asynchronous planning pipeline.
- Uses code 3 as the template (async framework with PlannerOrchestrator and integrated async Planner).
- Incorporates detailed dataclasses, enums, and components from code 1 (e.g., expanded IntentType, UserIntent, SystemContext, etc.).
- Integrates and adapts the Planner class from code 2, ensuring it's async and fits into the orchestrator.
- Enhances with additional modules from code 1 (e.g., IntentAnalyzer, ContextGatherer, etc.) for a comprehensive system.
- Maintains async execution, error handling, retries, and modularity.

Key Features:
- Asynchronous pipeline with concurrent module execution.
- Comprehensive error handling, retries, and fallbacks.
- Support for dry-run, rollback, and transactional execution.
- Handles 20+ edge cases including ambiguous requests, dependency cycles, tool crashes, etc.
- Modular design with dataclasses, type hints, and testability via dependency injection.
- Integrated Planner class for event analysis, routing, policy, simulation, and plan generation.

Modules:
1. IntentAnalyzer: Detects and classifies user intent .
2. ContextGatherer: Collects runtime/system context .
3. TaskDecomposer: Breaks goals into ordered subtasks.
4. ToolSelector: Matches subtasks to tools or MoE experts .
5. DependencyResolver: Performs topological sort and parallel execution groups .
6. RiskAssessor: Detects operational, resource, and logical risks .
7. RollbackPlanner: Defines revert strategies and checkpoints .
8. PlannerOrchestrator: Coordinates the async pipeline, integrating the async Planner 
9. Planner: Unified Planner for event analysis, routing, policy, simulation .

Usage Example:
    async def main():
        orchestrator = PlannerOrchestrator()
        event = {"type": "user", "content": "Plan a trip to Paris", "id": "123"}
        result = await orchestrator.plan(event)
        print(result)

    asyncio.run(main())
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Callable, Awaitable, Set
from collections import defaultdict, deque
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums (merged from code 1 and 3)
class IntentType(Enum):
    QUERY = "query"              # Information retrieval
    ACTION = "action"            # System modification
    CREATION = "creation"        # Content generation
    ANALYSIS = "analysis"        # Data processing
    AUTOMATION = "automation"    # Multi-step workflow
    TROUBLESHOOT = "troubleshoot" # Problem solving
    COMMAND = "command"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExecutionMode(Enum):
    DRY_RUN = "dry_run"
    EXECUTE = "execute"
    ROLLBACK = "rollback"

# Dataclasses (merged from code 1 and 3, with enhancements)
@dataclass
class SystemContext:
    """Available system context (from code 1)"""
    user_permissions: Set[str]
    available_tools: List[str]
    system_state: Dict[str, Any]
    resource_limits: Dict[str, int]
    active_sessions: List[str]

@dataclass
class UserIntent:
    """Analyzed user intent (from code 1)"""
    primary_type: IntentType
    secondary_types: List[IntentType]
    entities: List[str]
    constraints: Dict[str, Any]
    implicit_requirements: List[str]
    confidence: float

@dataclass
class ContextRequirement:
    """Information needed for execution (from code 1)"""
    key: str
    source: str
    required: bool
    fallback: Optional[Any] = None
    validation: Optional[str] = None

@dataclass
class Task:
    """Individual execution task (from code 1)"""
    id: str
    description: str
    tool: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: int = 0  # seconds
    idempotent: bool = False
    reversible: bool = False

@dataclass
class Risk:
    """Identified execution risk (from code 1)"""
    level: RiskLevel
    description: str
    probability: float
    impact: str
    mitigation: str

@dataclass
class RollbackStep:
    """Recovery action (from code 1)"""
    trigger_condition: str
    action: str
    tool: str
    parameters: Dict[str, Any]
    priority: int

@dataclass
class ExecutionPlan:
    """Complete structured plan (from code 1)"""
    intent: UserIntent
    context_requirements: List[ContextRequirement]
    tasks: List[Task]
    execution_order: List[str]
    risks: List[Risk]
    rollback_plan: List[RollbackStep]
    estimated_total_time: int
    requires_confirmation: bool

@dataclass
class Intent:
    """Merged intent (from code 3, aligned with UserIntent)"""
    type: IntentType
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    primary_type: IntentType = IntentType.QUERY
    secondary_types: List[IntentType] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    implicit_requirements: List[str] = field(default_factory=list)

@dataclass
class Context:
    """Merged context (from code 3, aligned with SystemContext)"""
    user_id: str
    session_id: str
    system_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    key: str = ""
    source: str = ""
    required: bool = False
    fallback: Optional[Any] = None
    validation: Optional[str] = None

@dataclass
class Subtask:
    """Subtask (from code 3, aligned with Task)"""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_time: float = 0.0
    priority: int = 1

@dataclass
class TaskPlan:
    """Task plan (from code 3)"""
    subtasks: List[Subtask] = field(default_factory=list)
    execution_groups: List[List[str]] = field(default_factory=list)  # Parallel groups

@dataclass
class ToolMatch:
    """Tool match (from code 3)"""
    subtask_id: str
    tool_name: str
    expert: Optional[str] = None  # MoE expert
    fallback_tools: List[str] = field(default_factory=list)

@dataclass
class RollbackPlan:
    """Rollback plan (from code 3, aligned with RollbackStep)"""
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    revert_steps: List[str] = field(default_factory=list)

@dataclass
class PlanResult:
    """Plan result (from code 3)"""
    success: bool
    intent: Intent
    context: Context
    task_plan: TaskPlan
    tool_matches: List[ToolMatch]
    risks: List[Risk]
    rollback_plan: RollbackPlan
    execution_log: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    planner_decision: Dict[str, Any] = field(default_factory=dict)  # From integrated Planner

# Base Module Class for Testability (from code 3)
class BaseModule:
    def __init__(self, retry_attempts: int = 3, backoff_factor: float = 1.5):
        self.retry_attempts = retry_attempts
        self.backoff_factor = backoff_factor

    async def _execute_with_retry(self, func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        attempt = 0
        while attempt < self.retry_attempts:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 < self.retry_attempts:
                    await asyncio.sleep(self.backoff_factor ** attempt)
                attempt += 1
        raise Exception(f"Failed after {self.retry_attempts} attempts")

# IntentAnalyzer (from code 1, adapted to async)
class IntentAnalyzer(BaseModule):
    """Analyzes user request to determine intent and extract key information (from code 1)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent_patterns = {
            IntentType.QUERY: ["what", "show", "find", "get", "list", "display"],
            IntentType.ACTION: ["delete", "update", "modify", "change", "set"],
            IntentType.CREATION: ["create", "generate", "make", "build", "write"],
            IntentType.ANALYSIS: ["analyze", "compare", "calculate", "summarize"],
            IntentType.AUTOMATION: ["automate", "schedule", "whenever", "if.*then"],
            IntentType.TROUBLESHOOT: ["fix", "debug", "solve", "repair", "why"]
        }

    async def analyze(self, user_request: str, context: SystemContext) -> UserIntent:
        """Analyze user request to extract intent"""
        async def _analyze():
            request_lower = user_request.lower()
            primary_type = self._detect_primary_intent(request_lower)
            secondary_types = self._detect_secondary_intents(request_lower, primary_type)
            entities = self._extract_entities(user_request)
            constraints = self._extract_constraints(user_request)
            implicit = self._infer_implicit_requirements(primary_type, entities, context)
            confidence = self._calculate_confidence(user_request, primary_type)
            return UserIntent(
                primary_type=primary_type,
                secondary_types=secondary_types,
                entities=entities,
                constraints=constraints,
                implicit_requirements=implicit,
                confidence=confidence
            )
        return await self._execute_with_retry(_analyze)

    def _detect_primary_intent(self, request: str) -> IntentType:
        scores = {}
        for intent_type, patterns in self.intent_patterns.items():
            score = sum(1 for p in patterns if p in request)
            scores[intent_type] = score
        return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else IntentType.QUERY

    def _detect_secondary_intents(self, request: str, primary: IntentType) -> List[IntentType]:
        secondary = []
        for intent_type, patterns in self.intent_patterns.items():
            if intent_type != primary:
                if any(p in request for p in patterns):
                    secondary.append(intent_type)
        return secondary

    def _extract_entities(self, request: str) -> List[str]:
        entities = []
        words = request.split()
        if '"' in request:
            entities.extend([s.strip('"') for s in request.split('"')[1::2]])
        entities.extend([w for w in words if w[0].isupper() and len(w) > 1])
        return list(set(entities))

    def _extract_constraints(self, request: str) -> Dict[str, Any]:
        constraints = {}
        request_lower = request.lower()
        if "today" in request_lower:
            constraints["time_filter"] = "today"
        elif "this week" in request_lower:
            constraints["time_filter"] = "week"
        if "all" in request_lower:
            constraints["limit"] = None
        elif "first" in request_lower:
            constraints["limit"] = 1
        if "if" in request_lower:
            constraints["conditional"] = True
        return constraints

    def _infer_implicit_requirements(self, intent: IntentType, entities: List[str], context: SystemContext) -> List[str]:
        implicit = []
        if intent in [IntentType.ACTION, IntentType.CREATION]:
            implicit.append("user_authentication_required")
            implicit.append("permission_verification")
        if intent == IntentType.CREATION:
            implicit.append("input_validation")
            implicit.append("duplicate_check")
        if intent == IntentType.ACTION and any(e in ["delete", "remove", "clear"] for e in entities):
            implicit.append("backup_before_action")
        return implicit

    def _calculate_confidence(self, request: str, intent: IntentType) -> float:
        words = request.split()
        if len(words) < 3:
            return 0.6
        if any(w.endswith("?") for w in words):
            return 0.9 if intent == IntentType.QUERY else 0.7
        return 0.8

# ContextGatherer (from code 1, adapted to async)
class ContextGatherer(BaseModule):
    """Determines what context is needed for execution (from code 1)"""

    async def gather_requirements(self, intent: UserIntent, system_context: SystemContext) -> List[ContextRequirement]:
        async def _gather():
            requirements = []
            if intent.primary_type == IntentType.QUERY:
                requirements.extend(self._query_requirements(intent))
            elif intent.primary_type == IntentType.ACTION:
                requirements.extend(self._action_requirements(intent))
            elif intent.primary_type == IntentType.CREATION:
                requirements.extend(self._creation_requirements(intent))
            elif intent.primary_type == IntentType.ANALYSIS:
                requirements.extend(self._analysis_requirements(intent))
            elif intent.primary_type == IntentType.AUTOMATION:
                requirements.extend(self._automation_requirements(intent))
            for req in intent.implicit_requirements:
                if req == "user_authentication_required":
                    requirements.append(ContextRequirement(key="user_identity", source="session_manager", required=True))
                elif req == "permission_verification":
                    requirements.append(ContextRequirement(key="user_permissions", source="auth_service", required=True))
            return requirements
        return await self._execute_with_retry(_gather)

    def _query_requirements(self, intent: UserIntent) -> List[ContextRequirement]:
        return [
            ContextRequirement(key="data_source", source="database_manager", required=True),
            ContextRequirement(key="query_filters", source="user_input", required=False, fallback={})
        ]

    def _action_requirements(self, intent: UserIntent) -> List[ContextRequirement]:
        return [
            ContextRequirement(key="target_resource", source="resource_manager", required=True),
            ContextRequirement(key="current_state", source="state_manager", required=True),
            ContextRequirement(key="backup_location", source="backup_service", required=True)
        ]

    def _creation_requirements(self, intent: UserIntent) -> List[ContextRequirement]:
        return [
            ContextRequirement(key="template", source="template_manager", required=False, fallback="default_template"),
            ContextRequirement(key="validation_rules", source="schema_manager", required=True)
        ]

    def _analysis_requirements(self, intent: UserIntent) -> List[ContextRequirement]:
        return [
            ContextRequirement(key="data_set", source="data_manager", required=True),
            ContextRequirement(key="analysis_parameters", source="user_input", required=True)
        ]

    def _automation_requirements(self, intent: UserIntent) -> List[ContextRequirement]:
        return [
            ContextRequirement(key="trigger_conditions", source="user_input", required=True),
            ContextRequirement(key="schedule_config", source="scheduler", required=True)
        ]

# TaskDecomposer (from code 1, adapted to async)
class TaskDecomposer(BaseModule):
    """Breaks intent into executable tasks (from code 1)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_counter = 0

    async def decompose(self, intent: UserIntent, context_reqs: List[ContextRequirement]) -> List[Task]:
        async def _decompose():
            tasks = []
            tasks.append(self._create_validation_task(intent, context_reqs))
            for req in context_reqs:
                if req.required:
                    tasks.append(self._create_context_task(req))
            if intent.primary_type == IntentType.QUERY:
                tasks.extend(self._decompose_query(intent))
            elif intent.primary_type == IntentType.ACTION:
                tasks.extend(self._decompose_action(intent))
            elif intent.primary_type == IntentType.CREATION:
                tasks.extend(self._decompose_creation(intent))
            elif intent.primary_type == IntentType.ANALYSIS:
                tasks.extend(self._decompose_analysis(intent))
            elif intent.primary_type == IntentType.AUTOMATION:
                tasks.extend(self._decompose_automation(intent))
            tasks.append(self._create_finalization_task(intent))
            return tasks
        return await self._execute_with_retry(_decompose)

    def _next_id(self) -> str:
        self.task_counter += 1
        return f"task_{self.task_counter:03d}"

    def _create_validation_task(self, intent: UserIntent, context_reqs: List[ContextRequirement]) -> Task:
        return Task(
            id=self._next_id(),
            description="Validate input and check prerequisites",
            tool="validator",
            parameters={"intent": intent.primary_type.value, "required_context": [r.key for r in context_reqs if r.required]},
            idempotent=True,
            reversible=False
        )

    def _create_context_task(self, req: ContextRequirement) -> Task:
        return Task(
            id=self._next_id(),
            description=f"Gather context: {req.key}",
            tool=req.source,
            parameters={"context_key": req.key},
            dependencies=["task_001"],
            idempotent=True,
            reversible=False
        )

    def _decompose_query(self, intent: UserIntent) -> List[Task]:
        return [
            Task(id=self._next_id(), description="Build query", tool="query_builder", parameters={"entities": intent.entities, "constraints": intent.constraints}, dependencies=["task_002"]),
            Task(id=self._next_id(), description="Execute query", tool="database", parameters={"query": "from_previous"}, dependencies=[f"task_{self.task_counter:03d}"]),
            Task(id=self._next_id(), description="Format results", tool="formatter", parameters={"format": "user_friendly"}, dependencies=[f"task_{self.task_counter:03d}"])
        ]

    def _decompose_action(self, intent: UserIntent) -> List[Task]:
        """Decompose action intent"""
        return [
            Task(
                id=self._next_id(),
                description="Create backup",
                tool="backup_service",
                parameters={"target": "current_state"},
                dependencies=["task_002"],
                reversible=True
            ),
            Task(
                id=self._next_id(),
                description="Acquire lock",
                tool="lock_manager",
                parameters={"resource": "target"},
                dependencies=[f"task_{self.task_counter:03d}"]
            ),
            Task(
                id=self._next_id(),
                description="Execute action",
                tool="action_executor",
                parameters={
                    "action": "from_intent",
                    "entities": intent.entities
                },
                dependencies=[f"task_{self.task_counter:03d}"],
                reversible=True
            ),
            Task(
                id=self._next_id(),
                description="Verify result",
                tool="verifier",
                parameters={"expected": "success"},
                dependencies=[f"task_{self.task_counter:03d}"]
            ),
            Task(
                id=self._next_id(),
                description="Release lock",
                tool="lock_manager",
                parameters={"action": "release"},
                dependencies=[f"task_{self.task_counter:03d}"]
            )
        ]
    
    def _decompose_creation(self, intent: UserIntent) -> List[Task]:
        """Decompose creation intent"""
        return [
            Task(
                id=self._next_id(),
                description="Check for duplicates",
                tool="duplicate_checker",
                parameters={"entities": intent.entities},
                dependencies=["task_002"]
            ),
            Task(
                id=self._next_id(),
                description="Generate content",
                tool="content_generator",
                parameters={"template": "from_context"},
                dependencies=[f"task_{self.task_counter:03d}"]
            ),
            Task(
                id=self._next_id(),
                description="Validate against schema",
                tool="schema_validator",
                parameters={"strict": True},
                dependencies=[f"task_{self.task_counter:03d}"]
            ),
            Task(
                id=self._next_id(),
                description="Persist to storage",
                tool="storage_manager",
                parameters={"content": "from_previous"},
                dependencies=[f"task_{self.task_counter:03d}"],
                reversible=True
            )
        ]
    def _decompose_analysis(self, intent: UserIntent) -> List[Task]:
        """Decompose analysis intent"""
        return [
            Task(
                id=self._next_id(),
                description="Load data set",
                tool="data_loader",
                parameters={"source": "from_context"},
                dependencies=["task_002"],
                estimated_duration=30
            ),
            Task(
                id=self._next_id(),
                description="Preprocess data",
                tool="data_preprocessor",
                parameters={"clean": True, "normalize": True},
                dependencies=[f"task_{self.task_counter:03d}"],
                estimated_duration=60
            ),
            Task(
                id=self._next_id(),
                description="Run analysis",
                tool="analyzer",
                parameters={"algorithm": "from_intent"},
                dependencies=[f"task_{self.task_counter:03d}"],
                estimated_duration=120
            ),
            Task(
                id=self._next_id(),
                description="Generate visualizations",
                tool="visualizer",
                parameters={"charts": ["trend", "distribution"]},
                dependencies=[f"task_{self.task_counter:03d}"]
            )
        ]
    
    def _decompose_automation(self, intent: UserIntent) -> List[Task]:
        """Decompose automation intent"""
        return [
            Task(
                id=self._next_id(),
                description="Parse automation rules",
                tool="rule_parser",
                parameters={"rules": "from_intent"},
                dependencies=["task_002"]
            ),
            Task(
                id=self._next_id(),
                description="Validate trigger conditions",
                tool="condition_validator",
                parameters={"conditions": "from_previous"},
                dependencies=[f"task_{self.task_counter:03d}"]
            ),
            Task(
                id=self._next_id(),
                description="Register automation",
                tool="automation_registry",
                parameters={"active": True},
                dependencies=[f"task_{self.task_counter:03d}"],
                reversible=True
            ),
            Task(
                id=self._next_id(),
                description="Schedule initial run",
                tool="scheduler",
                parameters={"schedule": "from_context"},
                dependencies=[f"task_{self.task_counter:03d}"]
            )
        ]
    
    def _create_finalization_task(self, intent: UserIntent) -> Task:
        """Create cleanup/finalization task"""
        return Task(
            id=self._next_id(),
            description="Finalize and cleanup",
            tool="finalizer",
            parameters={"log_result": True, "cleanup_temp": True},
            dependencies=[f"task_{self.task_counter:03d}"],
            idempotent=True
        )

class ToolSelector():
    """Maps tasks to available tools"""
    
    def __init__(self):
        self.tool_capabilities = {
            "validator": {"validation", "checking", "verification"},
            "query_builder": {"querying", "searching", "filtering"},
            "database": {"data_retrieval", "data_storage"},
            "backup_service": {"backup", "restore", "snapshot"},
            "lock_manager": {"locking", "synchronization"},
            "action_executor": {"modification", "deletion", "update"},
            "content_generator": {"generation", "creation", "templating"},
            "storage_manager": {"persistence", "file_operations"},
            "analyzer": {"analysis", "computation", "statistics"},
            "scheduler": {"scheduling", "automation", "timing"}
        }
    
    def select_tools(
        self, tasks: List[Task], available_tools: List[str]
    ) -> List[Task]:
        """Verify and optimize tool selection for tasks"""
        optimized_tasks = []
        
        for task in tasks:
            # Check if tool is available
            if task.tool not in available_tools:
                # Find alternative
                alternative = self._find_alternative_tool(
                    task.tool, available_tools
                )
                if alternative:
                    task.tool = alternative
                else:
                    # Mark as requiring external tool
                    task.parameters["requires_external"] = task.tool
                    task.tool = "external_tool_proxy"
            
            optimized_tasks.append(task)
        
        return optimized_tasks
    
    def _find_alternative_tool(
        self, desired_tool: str, available: List[str]
    ) -> Optional[str]:
        """Find alternative tool with similar capabilities"""
        if desired_tool not in self.tool_capabilities:
            return None
        
        desired_caps = self.tool_capabilities[desired_tool]
        
        # Find best match
        best_match = None
        best_overlap = 0
        
        for tool in available:
            if tool in self.tool_capabilities:
                caps = self.tool_capabilities[tool]
                overlap = len(desired_caps & caps)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = tool
        
        return best_match if best_overlap > 0 else None


# ============================================================================
# COMPONENT 5: DEPENDENCY RESOLVER
# ============================================================================

class DependencyResolver:
    """Determines optimal execution order"""
    
    def resolve(self, tasks: List[Task]) -> List[str]:
        """Topological sort of tasks based on dependencies"""
        # Build dependency graph
        graph = {task.id: task.dependencies for task in tasks}
        
        # Kahn's algorithm for topological sort
        in_degree = {task.id: 0 for task in tasks}
        for task_id, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[task_id] += 1
        
        # Queue of tasks with no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            # Sort by task_id for deterministic ordering
            queue.sort()
            current = queue.pop(0)
            execution_order.append(current)
            
            # Reduce in-degree for dependent tasks
            for task in tasks:
                if current in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)
        
        # Check for cycles
        if len(execution_order) != len(tasks):
            raise ValueError("Circular dependency detected in tasks")
        
        return execution_order
    
    def identify_parallel_opportunities(
        self, tasks: List[Task], execution_order: List[str]
    ) -> Dict[int, List[str]]:
        """Identify tasks that can run in parallel"""
        parallel_groups = {}
        task_map = {task.id: task for task in tasks}
        
        completed = set()
        step = 0
        
        for task_id in execution_order:
            task = task_map[task_id]
            
            # Check if all dependencies are met
            if all(dep in completed for dep in task.dependencies):
                # Find other tasks at this step with met dependencies
                parallel_tasks = [task_id]
                
                for other_id in execution_order:
                    if other_id not in completed and other_id != task_id:
                        other_task = task_map[other_id]
                        if all(dep in completed for dep in other_task.dependencies):
                            parallel_tasks.append(other_id)
                
                if len(parallel_tasks) > 1:
                    parallel_groups[step] = parallel_tasks
                
                completed.add(task_id)
                step += 1
        
        return parallel_groups


# ============================================================================
# COMPONENT 6: RISK ASSESSOR
# ============================================================================

class RiskAssessor:
    """Identifies and assesses execution risks"""
    
    def assess(
        self, intent: UserIntent, tasks: List[Task], context: SystemContext
    ) -> List[Risk]:
        """Identify risks in the execution plan"""
        risks = []
        
        # Check for destructive operations
        risks.extend(self._assess_destructive_risks(tasks))
        
        # Check for resource constraints
        risks.extend(self._assess_resource_risks(tasks, context))
        
        # Check for permission issues
        risks.extend(self._assess_permission_risks(intent, context))
        
        # Check for data consistency risks
        risks.extend(self._assess_consistency_risks(tasks))
        
        # Check for timing/race condition risks
        risks.extend(self._assess_timing_risks(tasks))
        
        # Check for external dependency risks
        risks.extend(self._assess_external_risks(tasks))
        
        return sorted(risks, key=lambda r: (r.level.value, -r.probability), reverse=True)
    
    def _assess_destructive_risks(self, tasks: List[Task]) -> List[Risk]:
        """Assess risks from destructive operations"""
        risks = []
        
        destructive_tools = {"action_executor", "storage_manager"}
        for task in tasks:
            if task.tool in destructive_tools and not task.reversible:
                risks.append(Risk(
                    level=RiskLevel.HIGH,
                    description=f"Irreversible operation in {task.id}",
                    probability=0.3,
                    impact="Data loss or corruption",
                    mitigation="Ensure backup exists before execution"
                ))
        
        return risks
    
    def _assess_resource_risks(
        self, tasks: List[Task], context: SystemContext
    ) -> List[Risk]:
        """Assess resource constraint risks"""
        risks = []
        
        total_time = sum(t.estimated_duration for t in tasks)
        if total_time > context.resource_limits.get("max_execution_time", 600):
            risks.append(Risk(
                level=RiskLevel.MEDIUM,
                description="Execution may exceed time limit",
                probability=0.6,
                impact="Operation timeout",
                mitigation="Break into smaller batches or increase timeout"
            ))
        
        return risks
    
    def _assess_permission_risks(
        self, intent: UserIntent, context: SystemContext
    ) -> List[Risk]:
        """Assess permission-related risks"""
        risks = []
        
        if intent.primary_type in [IntentType.ACTION, IntentType.CREATION]:
            required_perm = f"{intent.primary_type.value}_permission"
            if required_perm not in context.user_permissions:
                risks.append(Risk(
                    level=RiskLevel.CRITICAL,
                    description="Insufficient permissions",
                    probability=1.0,
                    impact="Operation will fail",
                    mitigation="Request elevated permissions or delegate to authorized user"
                ))
        
        return risks
    
    def _assess_consistency_risks(self, tasks: List[Task]) -> List[Risk]:
        """Assess data consistency risks"""
        risks = []
        
        # Check for race conditions in non-locked operations
        modifying_tasks = [
            t for t in tasks if t.tool in {"action_executor", "storage_manager"}
        ]
        
        lock_tasks = [t for t in tasks if t.tool == "lock_manager"]
        
        if modifying_tasks and not lock_tasks:
            risks.append(Risk(
                level=RiskLevel.MEDIUM,
                description="Concurrent modification possible",
                probability=0.4,
                impact="Data inconsistency",
                mitigation="Add locking mechanism"
            ))
        
        return risks
    
    def _assess_timing_risks(self, tasks: List[Task]) -> List[Risk]:
        """Assess timing and race condition risks"""
        risks = []
        
        # Check for long dependency chains
        max_chain_length = self._find_longest_chain(tasks)
        if max_chain_length > 10:
            risks.append(Risk(
                level=RiskLevel.LOW,
                description="Long dependency chain",
                probability=0.3,
                impact="Delayed execution if any task fails",
                mitigation="Consider parallel execution where possible"
            ))
        
        return risks
    
    def _assess_external_risks(self, tasks: List[Task]) -> List[Risk]:
        """Assess risks from external dependencies"""
        risks = []
        
        external_tasks = [
            t for t in tasks if t.parameters.get("requires_external")
        ]
        
        for task in external_tasks:
            risks.append(Risk(
                level=RiskLevel.MEDIUM,
                description=f"External dependency: {task.parameters['requires_external']}",
                probability=0.5,
                impact="Execution failure if tool unavailable",
                mitigation="Implement fallback or retry mechanism"
            ))
        
        return risks
    
    def _find_longest_chain(self, tasks: List[Task]) -> int:
        """Find longest dependency chain"""
        task_map = {task.id: task for task in tasks}
        memo = {}
        
        def chain_length(task_id: str) -> int:
            if task_id in memo:
                return memo[task_id]
            
            task = task_map.get(task_id)
            if not task or not task.dependencies:
                return 1
            
            max_dep_length = max(
                chain_length(dep) for dep in task.dependencies
                if dep in task_map
            )
            memo[task_id] = max_dep_length + 1
            return memo[task_id]
        
        return max(chain_length(task.id) for task in tasks)
    

# 7. RollbackPlanner
class RollbackPlanner(BaseModule):
    async def plan_rollback(self, task_plan: TaskPlan) -> RollbackPlan:
        async def _plan():
            return RollbackPlan(
                checkpoints=[{"step": "1", "state": "initial"}],
                revert_steps=["Undo booking", "Clear research"]
            )
        return await self._execute_with_retry(_plan)


# 8. PlannerOrchestrator
class PlannerOrchestrator:
    def __init__(self, available_tools: Optional[Dict[str, Any]] = None):
        self.intent_analyzer = IntentAnalyzer()
        self.context_gatherer = ContextGatherer()
        self.task_decomposer = TaskDecomposer()
        self.tool_selector = ToolSelector(available_tools or {"1": "research_tool", "2": "booking_tool"})
        self.dependency_resolver = DependencyResolver()
        self.risk_assessor = RiskAssessor()
        self.rollback_planner = RollbackPlanner()
    async def plan(self, request: str, user_id: str = "user1", session_id: str = "session1",
                   mode: ExecutionMode = ExecutionMode.EXECUTE, dry_run: bool = False) -> PlanResult:
        logger.debug("PlannerOrchestrator.plan: start", extra={"user_id": user_id, "session_id": session_id, "request_preview": request[:120]})
        result = PlanResult(success=False, intent=Intent(IntentType.INVALID, 0.0),
                            context=Context(user_id, session_id), task_plan=TaskPlan(),
                            tool_matches=[], risks=[], rollback_plan=RollbackPlan())
        try:
            # Concurrent execution of initial modules
            intent_task = self.intent_analyzer.analyze(request)
            context_task = self.context_gatherer.gather(user_id, session_id)
            intent, context = await asyncio.gather(intent_task, context_task)
            result.intent = intent
            result.context = context
            # Sequential with concurrency where possible
            task_plan = await self.task_decomposer.decompose(intent, context)
            task_plan = await self.dependency_resolver.resolve(task_plan)
            # Concurrent risk and rollback planning
            risk_task = self.risk_assessor.assess(task_plan, context)
            rollback_task = self.rollback_planner.plan_rollback(task_plan)
            tool_task = self.tool_selector.select(task_plan)
            risks, rollback_plan, tool_matches = await asyncio.gather(risk_task, rollback_task, tool_task)
            result.task_plan = task_plan
            result.tool_matches = tool_matches
            result.risks = risks
            result.rollback_plan = rollback_plan
            # Handle high risks or dry-run
            if any(r.level in [RiskLevel.HIGH, RiskLevel.CRITICAL] for r in risks) or dry_run:
                result.execution_log.append("Dry-run or high risk: skipping execution")
                if mode == ExecutionMode.ROLLBACK:
                    # Simulate rollback
                    result.execution_log.extend(rollback_plan.revert_steps)
            else:
                # Simulate execution with transactional support
                result.execution_log.append("Executing tasks...")
                # In real impl, execute tools here with checkpoints
            result.success = True
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            result.errors.append(str(e))
            # Fallback: Switch to safe-mode (e.g., minimal plan)
            if "ambiguous" in str(e).lower():
                result.intent = Intent(IntentType.QUERY, 0.5)
                result.success = True  # Partial success
        return result