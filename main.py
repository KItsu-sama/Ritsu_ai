from __future__ import annotations

"""
Ritsu Main Entrypoint
Initializes core systems, input/output, and starts event loop.
"""

from config.config import _model_
import inspect
import importlib
import asyncio
import traceback
import contextlib
import logging
import signal
import sys
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Callable
from config.config import version
from enum import Enum
try:
    from aiohttp import web
except Exception:
    web = None
from enum import Enum



# By default, silence root logging during module import so startup is quiet.
# It will be restored when the user types a wake phrase.
try:
    logging.disable(logging.CRITICAL)
    _root_logging_silenced = True
except Exception:
    _root_logging_silenced = False

    # If the user passed a debug flag on the CLI (common patterns), restore logging
    # immediately so startup import-time logs are visible when running with --debug or -d.
    try:
        _debug_tokens = set(arg.lower() for arg in sys.argv[1:])
        if '--debug' in _debug_tokens or '-d' in _debug_tokens or 'debug' in _debug_tokens:
            logging.disable(logging.NOTSET)
            # On early import we cannot call restore_root_logging (defined later),
            # so set root logger to DEBUG directly as a best-effort fallback.
            try:
                logging.getLogger().setLevel(logging.DEBUG)
            except Exception:
                pass
            _root_logging_silenced = False
    except Exception:
        pass

# ------------------------- Optional uvloop (best-effort) -------------------------
try:  # pragma: no cover
    import uvloop  # type: ignore

    uvloop.install()
except Exception:
    pass

# ------------------------------ Structured logging ------------------------------
try:
    from system.logger import get_logger, setup_logging
except Exception:
    import logging

    def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None, json: bool = True) -> None:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)

    def get_logger(name = None) -> logging.Logger:
        import logging
        return logging.getLogger(name or __name__)


log = get_logger("ritsu.main")


def log_with_context(logger: logging.Logger, level: int, msg: str, **kwargs: Any) -> None:
    """
    Helper to log with structured context.
    """
    logger.log(level, msg, extra=kwargs)


def set_system_mode(ctx: AppContext, mode: str, persist: bool = True) -> bool:
    """Set system mode on the context and optionally persist to disk.

    Returns True on success, False on invalid mode.
    """
    mode_map = {"background": "background", "chat": "chat", "follow_plan": "follow_plan", "follow": "follow_plan"}
    desired = mode_map.get(mode, None)
    if not desired:
        return False
    try:
        ctx.system_mode = desired
        # update config representation when present
        if getattr(ctx, "config", None) is not None and isinstance(ctx.config, dict):
            ctx.config.setdefault("system", {})["mode"] = desired
        # persist using config_manager if available
        if persist and getattr(ctx, "config_manager", None) is not None:
            try:
                ctx.config_manager._config = ctx.config  # sync underlying
                ctx.config_manager.save()
            except Exception:
                log.exception("Failed to persist mode change to disk")
        return True
    except Exception:
        log.exception("Failed to set system mode")
        return False


# ---------------------------------- Config ----------------------------------
try:
    from system.config_manager import ConfigManager
except Exception as e:
    log.warning(f"Failed to import ConfigManager: {e}")

    class ConfigManager:  # minimal shim for early dev
        def __init__(self, path: Optional[Path] = None):
            self._path = Path(path) if path else None
            self._config: Dict[str, Any] = {
                "app": {
                    "name": "Ritsu",
                    "env": "dev",
                    "safe_mode": False,
                    "restart_on_crash": True,
                },
                "logging": {"level": "INFO", "dir": "data/logs", "json": True},
                "system": {"mode": "background"},
                "io": {"enable_mic": False, "enable_chat": True},
                "ui": {"enabled": False},
                "rust_editor": {"enabled": False},
            }

        def load(self) -> Dict[str, Any]:
            return self._config
        
        def save(self, path: Optional[Path] = None) -> None:
            """Persist current config to YAML if available, otherwise JSON.

            Writes to the provided path or the path passed to the ConfigManager or
            defaults to 'system/config.yaml'. This is a best-effort helper used when
            runtime changes (like system.mode) should be persisted.
            """
            out_path = Path(path) if path else (self._path if self._path else Path("system/config.yaml"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import yaml  # type: ignore
                with out_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(self._config, f)
            except Exception:
                # fallback to JSON
                try:
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(self._config, f, indent=2)
                except Exception:
                    log.exception("Failed to persist config to %s", out_path)


# ---------------------------- Core / Input / Output ----------------------------
log = logging.getLogger(__name__)

def import_optional(path: str, package: str = None):
    try:
        if ":" in path:
            mod_name, attr = path.split(":")
            mod = importlib.import_module(mod_name, package=package)
            return getattr(mod, attr, None)
        else:
            mod = importlib.import_module(path, package=package)
            # Try auto-extract class with same name as module (capitalized)
            class_name = path.split(".")[-1].capitalize()
            return getattr(mod, class_name, mod)  # fallback to module
    except Exception as e:
        log.warning(f"Optional import failed '{path}': {e}")
        return None

# Import the basic tools
try:
    from core.Tool_Library.math.calculator import Calculator
    from core.Tool_Library.file_reader import FileReader
    from core.Tool_Library.process_monitor import ProcessMonitor
    from llm.ritsu_llm import RitsuLLM
except ImportError as e:
    log.warning(f"Failed to import basic tools: {e}")
    Calculator = FileReader = ProcessMonitor = RitsuLLM = None

from core.planning import Planner
from core.planner_manager import PlannerManager
from core.planner_manager import PlannerType


EventManager = import_optional("core.event_manager:EventManager")
Planner = import_optional("core.planning:Planner")
Executor = import_optional("core.executor")
Troubleshooter = import_optional("core.troubleshooter:Troubleshooter")
SelfImprovement = import_optional("core.self_improvement:SelfImprovement")
RitsuSelf = import_optional("core.Ritsu_self:RitsuSelf")
Toolbelt = import_optional("core.tools:Tool")
CodeAnalyzer = import_optional("core.code_analyzer:CodeAnalyzer")
CodeGenerator = import_optional("core.code_generator:CodeGenerator")
CodeDB = import_optional("core.codedb:CodeDB")

InputManager = import_optional("input.input_manager:InputManager")
STT = import_optional("input.stt")
ChatListener = import_optional("input.chat_listener:ChatListener")
CommandParser = import_optional("input.command_parser:CommandParser")

OutputManager = import_optional("output.output_manager:OutputManager")
TTS = import_optional("output.tts")
AvatarAnimator = import_optional("output.avatar_animator:AvatarAnimator")
StreamAdapter = import_optional("output.stream_adapter:StreamAdapter")

PromptTemplates = import_optional("llm.prompt_templates:character_context")

NLPEngine = import_optional("ai.nlp_engine:NLPEngine")
KnowledgeBase = import_optional("ai.knowledge_base:KnowledgeBase")
MemoryManager = import_optional("ai.memory_manager:MemoryManager")
AIAssistant = import_optional("ai.ai_assistant:AIAssistant")

RustEditor = import_optional("system.bindings_rust:RustEditor")
UIClient = import_optional("system.bindings_ui:UIClient")

#====Needed to be worked on latter====

PerformanceMonitor = import_optional("core.performance_monitor:PerformanceMonitor")
SecurityManager = import_optional("core.security_manager:SecurityManager")
AutoUpdater = import_optional("core.auto_updater:AutoUpdater")
PluginManager = import_optional("core.plugin_manager:PluginManager")
TaskScheduler = import_optional("core.task_scheduler:TaskScheduler")

HardwareMonitor = import_optional("core.hardware_monitor:HardwareMonitor")
SystemAnalyzer = import_optional("core.system_analyzer:SystemAnalyzer")
NetworkMonitor = import_optional("core.network_monitor:NetworkMonitor")

CodeReviewer = import_optional("core.code_reviewer:CodeReviewer")
TestGenerator = import_optional("core.test_generator:TestGenerator")
DocumentationGenerator = import_optional("core.documentation_generator:DocumentationGenerator")


# --------------------------------- App types ---------------------------------
try:
    @dataclass
    class AppContext:
        config: Dict[str, Any]
        event_queue: "asyncio.Queue[Dict[str, Any]]" = field(default_factory=asyncio.Queue)
        shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)

        # Core
        event_manager: Optional[Any] = None
        planner: Optional[Any] = None
        executor: Optional[Any] = None
        troubleshooter: Optional[Any] = None
        self_improvement: Optional[Any] = None
        ritsu_self: Optional[Any] = None
        tools: Optional[Any] = None
        code_analyzer: Optional[Any] = None
        code_generator: Optional[Any] = None
        codedb: Optional[Any] = None

        # Basic tools
        calculator: Optional[Any] = None
        file_reader: Optional[Any] = None
        process_monitor: Optional[Any] = None

        # IO
        input_manager: Optional[Any] = None
        mic: Optional[Any] = None
        chat: Optional[Any] = None
        command_parser: Optional[Any] = None

        output_manager: Optional[Any] = None
        tts: Optional[Any] = None
        avatar: Optional[Any] = None
        stream: Optional[Any] = None

        # AI/LLM
        llm: Optional[Any] = None
        prompts: Optional[Any] = None
        nlp: Optional[Any] = None
        kb: Optional[Any] = None
        memory: Optional[Any] = None
        ai_assistant: Optional[Any] = None

        # Integrations
        rust_editor: Optional[Any] = None
        ui: Optional[Any] = None

        # Enhanced core systems
        performance_monitor: Optional[Any] = None
        security_manager: Optional[Any] = None
        auto_updater: Optional[Any] = None
        plugin_manager: Optional[Any] = None
        task_scheduler: Optional[Any] = None

        # Hardware monitoring
        hardware_monitor: Optional[Any] = None

        # Advanced troubleshooting
        system_analyzer: Optional[Any] = None
        network_monitor: Optional[Any] = None

        # Code intelligence
        code_reviewer: Optional[Any] = None
        test_generator: Optional[Any] = None
        documentation_generator: Optional[Any] = None
        # System mode controls terminal/interaction flow
        # defaults to background (light, shell-first behavior)
        system_mode: str = "background"
except Exception:
    # Fallback plain class if dataclass processing fails at import time
    class AppContext:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.event_queue = asyncio.Queue()
            self.shutdown_event = asyncio.Event()

            # Core
            self.event_manager = None
            self.planner = None
            self.executor = None
            self.troubleshooter = None
            self.self_improvement = None
            self.ritsu_self = None
            self.tools = None
            self.code_analyzer = None
            self.code_generator = None
            self.codedb = None

            # Basic tools
            self.calculator = None
            self.file_reader = None
            self.process_monitor = None

            # IO
            self.input_manager = None
            self.mic = None
            self.chat = None
            self.command_parser = None

            self.output_manager = None
            self.tts = None
            self.avatar = None
            self.stream = None

            # AI/LLM
            self.llm = None
            self.prompts = None
            self.nlp = None
            self.kb = None
            self.memory = None
            self.ai_assistant = None

            # Integrations
            self.rust_editor = None
            self.ui = None

            # Enhanced core systems
            self.performance_monitor = None
            self.security_manager = None
            self.auto_updater = None
            self.plugin_manager = None
            self.task_scheduler = None

            # Hardware monitoring
            self.hardware_monitor = None

            # Advanced troubleshooting
            self.system_analyzer = None
            self.network_monitor = None

            # Code intelligence
            self.code_reviewer = None
            self.test_generator = None
            self.documentation_generator = None
            # System mode controls terminal/interaction flow
            self.system_mode = "background"
    auto_updater: Optional[Any] = None
    plugin_manager: Optional[Any] = None
    task_scheduler: Optional[Any] = None
    
    # Hardware monitoring
    hardware_monitor: Optional[Any] = None
    
    # Advanced troubleshooting
    system_analyzer: Optional[Any] = None
    network_monitor: Optional[Any] = None
    
    # Code intelligence
    code_reviewer: Optional[Any] = None
    test_generator: Optional[Any] = None
    documentation_generator: Optional[Any] = None


# --------------------------------- Bootstrap ---------------------------------
def safe_init(cls, config=None, *args, **kwargs):
    """Initialize component safely, return None if fails"""
    logger = get_logger(__name__)

    # 1. Handle invalid or missing class
    if cls is None or not callable(cls):
        logger.error(f"safe_init received invalid cls: {cls!r}")
        return None

    try:
        if not inspect.isclass(cls):
            log.error(f"safe_init received invalid cls: {repr(cls)}")
            return None
        # 2. Try with config keyword
        if config is not None:
            try:
                return cls(config=config, *args, **kwargs)
            except TypeError:
                pass

        # 3. Try without config
        try:
            return cls(*args, **kwargs)
        except TypeError:
            pass

        # 4. Try with config as positional
        if config is not None:
            try:
                return cls(config, *args, **kwargs)
            except TypeError:
                pass

        # 5. Last resort: call with no arguments
        return cls()

    except Exception as e:
        name = getattr(cls, "__name__", str(cls))  # handles strings safely
        logger.warning(f"Failed to initialize {name}: {e}")
        return None

    
def add_enhanced_components(ctx: "AppContext", config: dict) -> None:
    """Attach optional/enhanced components to the context."""

    # Helper to safely build using the imported class (if available)
    def safe_build(cls, cfg_key: str):
        if cls:
            try:
                cfg = config.get(cfg_key, {})
                return safe_init(cls, config=cfg)
            except Exception:
                return safe_init(cls)
        return None

    ctx.performance_monitor = safe_build(PerformanceMonitor, "performance")
    ctx.security_manager = safe_build(SecurityManager, "security")
    ctx.auto_updater = safe_build(AutoUpdater, "auto_update")
    ctx.plugin_manager = safe_build(PluginManager, "plugins")
    ctx.task_scheduler = safe_build(TaskScheduler, "scheduler")

    ctx.hardware_monitor = safe_build(HardwareMonitor, "hardware")
    ctx.system_analyzer = safe_build(SystemAnalyzer, "system_analyzer")
    ctx.network_monitor = safe_build(NetworkMonitor, "network")

    ctx.code_reviewer = safe_build(CodeReviewer, "code_review")
    ctx.test_generator = safe_build(TestGenerator, "test_gen")
    ctx.documentation_generator = safe_build(DocumentationGenerator, "docs")


async def bootstrap(config_path: Optional[Path]) -> "AppContext":
    cfg_mgr = ConfigManager(config_path)
    config = cfg_mgr.load()

    # --- Logging setup ---
    log_dir = Path(config.get("logging", {}).get("dir", "data/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        level=str(config.get("logging", {}).get("level", "INFO")),
        log_dir=log_dir,
        json=bool(config.get("logging", {}).get("json", True)),
    )

    ctx = AppContext(config=config)
    # Keep a reference to the config manager so runtime changes can be saved
    try:
        ctx.config_manager = cfg_mgr
    except Exception:
        # best-effort; ignore if ctx doesn't support attr assignment
        pass

    # --- LLM base ---
    ctx.llm = safe_init(RitsuLLM, model=_model_)  # Very lightweight model for low-end PC
    # If the LLM exposes an async router start, ensure it's started so worker tasks are running
    try:
        if ctx.llm and hasattr(ctx.llm, "start"):
            maybe = ctx.llm.start()
            # start may be coroutine; schedule or await accordingly
            if asyncio.iscoroutine(maybe):
                # schedule background start without blocking bootstrap
                asyncio.create_task(maybe)
    except Exception:
        log.exception("Failed to start LLM router; continuing without async start")

    # --- Core logic ---
    # Prefer the new Planner implementation; fall back to legacy planner shim if necessary
    # Initialize PlannerManager (handles both planners automatically)
    ctx.planner = safe_init(
        PlannerManager,
        config={
                "auto_fallback": True,
                "prefer_fast": True,
                "event_planner_config": {"enable_memory": True}
        },
        llm=ctx.llm  # Passed for compatibility
    )
    if not ctx.planner:
        log.error("Failed to initialize PlannerManager")
        ctx.planner = None
    
    ctx.code_analyzer = safe_init(CodeAnalyzer, llm=ctx.llm)
    ctx.code_generator = safe_init(CodeGenerator, llm=ctx.llm)
    ctx.troubleshooter = safe_init(Troubleshooter)
    ctx.event_manager = safe_init(EventManager)
    ctx.executor = safe_init(Executor)

    # --- I/O management ---
    ctx.input_manager = safe_init(InputManager, config.get("input", {}))
    ctx.output_manager = safe_init(OutputManager, config.get("output", {}))

    # --- AI and memory ---
    ctx.memory = safe_init(MemoryManager)
    ctx.knowledge = safe_init(KnowledgeBase)

    # --- Tools ---
    ctx.calculator = safe_init(Calculator)
    ctx.file_reader = safe_init(FileReader)
    ctx.process_monitor = safe_init(ProcessMonitor)

    # --- Prompt templates ---
    ctx.prompts = safe_init(PromptTemplates)

    # --- Optional / advanced ---
    ctx.tools = safe_init(Toolbelt, config=config.get("tools", {}))
    ctx.event_manager = safe_init(EventManager, config=config.get("events", {}), queue=ctx.event_queue)
    ctx.self_improvement = safe_init(SelfImprovement, config=config.get("self_improvement", {}))
    ctx.ritsu_self = safe_init(RitsuSelf, config=config.get("ritsu_self", {}))
    ctx.codedb = safe_init(CodeDB, path=Path(config.get("codedb", {}).get("path", "data/codedb")))

    # --- Input stack ---
    ctx.command_parser = safe_init(CommandParser, config=config.get("command_parser", {}))
    ctx.mic = safe_init(STT, config=config.get("mic", {}))
    ctx.chat = safe_init(ChatListener, config=config.get("chat", {}))
    # Optional command classifier - rule-based/ML fallback
    try:
        from core.command_classifier import CommandClassifier
        ctx.command_classifier = safe_init(CommandClassifier)
    except Exception:
        ctx.command_classifier = None
    ctx.input_manager = safe_init(
        InputManager,
        config=config.get("input", {}),
        mic=ctx.mic,
        chat=ctx.chat,
        command_parser=ctx.command_parser,
        command_classifier=ctx.command_classifier,
    )

    # --- Output stack ---
    ctx.tts = safe_init(TTS, config=config.get("tts", {}))
    ctx.avatar = safe_init(AvatarAnimator, config=config.get("avatar", {}))
    ctx.stream = safe_init(StreamAdapter, config=config.get("stream", {}))
    ctx.output_manager = safe_init(
        OutputManager,
        config=config.get("output", {}),
        tts=ctx.tts,
        avatar=ctx.avatar,
        stream=ctx.stream,
    )

    # --- AI stack ---
    ctx.nlp = safe_init(NLPEngine, config=config.get("nlp", {}))
    ctx.kb = safe_init(KnowledgeBase, config=config.get("knowledge_base", {}))
    ctx.memory = safe_init(MemoryManager, config=config.get("memory", {}))
    ctx.ai_assistant = safe_init(
        AIAssistant,
        config=config.get("ai", {}),
        nlp_engine=ctx.nlp,
        knowledge_base=ctx.kb,
        memory_manager=ctx.memory,
        llm_engine=ctx.llm,
    )

    # --- Optional externals ---
    if config.get("rust_editor", {}).get("enabled", False) and RustEditor:
        ctx.rust_editor = safe_init(RustEditor, config=config.get("rust_editor", {}))
    if config.get("ui", {}).get("enabled", False) and UIClient:
        ctx.ui = safe_init(UIClient, config=config.get("ui", {}))

    # --- Cross-link executor dependencies ---
    if ctx.executor:
        try:
            executor_components = {
                "ai_assistant": ctx.ai_assistant,
                "memory_manager": ctx.memory,
                "knowledge_base": ctx.kb,
                "nlp_engine": ctx.nlp,
                "toolbelt": ctx.tools,
                "output_manager": ctx.output_manager,
                "llm_engine": ctx.llm,  # Add LLM for fallback responses
                "calculator": ctx.calculator,
                "file_reader": ctx.file_reader,
                "process_monitor": ctx.process_monitor,
            }
            ctx.executor.set_components(executor_components)
        except Exception as e:
            log.warning(f"Failed to set executor components: {e}")
            log.debug("".join(traceback.format_exc()))

    # --- Finalize setup ---
    add_enhanced_components(ctx, config)
    log.info("Bootstrapped components", extra={"env": config.get("app", {}).get("env", "dev")})
    return ctx

# ------------------------------ Application Loops ------------------------------
async def monitoring_loop(ctx: AppContext) -> None:
    """Continuous system monitoring"""
    if not ctx.performance_monitor:
        return
        
    interval = int(ctx.config.get("monitoring", {}).get("interval_sec", 30))
    try:
        while not ctx.shutdown_event.is_set():
            await asyncio.sleep(interval)
            with contextlib.suppress(Exception):
                # Monitor system performance
                metrics = await ctx.performance_monitor.collect_metrics()
                if metrics.get("critical"):
                    await ctx.event_queue.put({
                        "type": "system_alert",
                        "data": metrics,
                        "priority": "high"
                    })
                
                # Check security
                if ctx.security_manager:
                    threats = await ctx.security_manager.scan()
                    if threats:
                        await ctx.event_queue.put({
                            "type": "security_alert", 
                            "data": threats,
                            "priority": "critical"
                        })
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Monitoring loop crashed")

async def maintenance_loop(ctx: AppContext) -> None:
    """Automated maintenance tasks"""
    interval = int(ctx.config.get("maintenance", {}).get("interval_sec", 3600))
    try:
        while not ctx.shutdown_event.is_set():
            await asyncio.sleep(interval)
            
            # Cleanup temporary files
            if ctx.tools:
                await ctx.tools.cleanup_temp_files()
            
            # Update knowledge base (if method exists)
            if ctx.kb and hasattr(ctx.kb, 'auto_update'):
                try:
                    await ctx.kb.auto_update()
                except Exception as e:
                    log.warning(f"Knowledge base auto_update failed: {e}")
            
            # Optimize databases
            if ctx.codedb:
                ctx.codedb.optimize()
                
    except asyncio.CancelledError:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"Maintenance loop crashed: {e}\n{tb}")
        await asyncio.sleep(5)  # prevent tight crash loop


async def input_loop(ctx: AppContext) -> None:
    """
    Runs the input manager loop to capture user input and enqueue events.
    """
    if not ctx.input_manager:
        log.warning("InputManager unavailable; input loop disabled")
        return

    try:
        await ctx.input_manager.run(queue=ctx.event_queue, shutdown_event=ctx.shutdown_event)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Input loop crashed")
        raise


async def core_loop(ctx: AppContext) -> None:
    """Enhanced core loop with advanced capabilities"""
    if not (ctx.planner and ctx.executor and ctx.output_manager):
        log.error("Core components missing; core loop cannot start")
        return

    try:
        while not ctx.shutdown_event.is_set():
            event: Dict[str, Any] = await ctx.event_queue.get()
            try:
                # 1) Enhanced event processing with context
                enriched_event = await enrich_event(ctx, event)
                log.debug(f"Processing event type: {enriched_event.get('type')}")

                # 2) Plan with system context
                system_status = await get_system_status(ctx)
                # Special-case handling for follow_plan events: force module-based planner
                if enriched_event.get("type") == "follow_plan":
                    if ctx.planner and hasattr(ctx.planner, 'plan'):
                        try:
                            plan = await ctx.planner.plan(enriched_event, system_status, force_planner=PlannerType.MODULE_BASED)
                        except Exception:
                            log.exception("follow_plan: PlannerManager.plan failed; falling back to normal planning")
                            plan = None
                    else:
                        plan = None
                elif ctx.planner:
                    # Use async plan() if available, otherwise use decide()
                    try:
                        if hasattr(ctx.planner, 'plan') and asyncio.iscoroutinefunction(ctx.planner.plan):
                            plan = await ctx.planner.plan(enriched_event, system_status)
                        else:
                            plan = ctx.planner.decide(enriched_event, system_status)
                    except TypeError:
                        if hasattr(ctx.planner, 'plan') and asyncio.iscoroutinefunction(ctx.planner.plan):
                            plan = await ctx.planner.plan(enriched_event)
                        else:
                            plan = ctx.planner.decide(enriched_event)
                else:
                    plan = None

                if not plan:
                    log.warning("Planner returned no plan; skipping execution", extra={"event": enriched_event})
                    await ctx.output_manager.emit({"status": "failed", "error": "No plan generated", "event": enriched_event})
                    continue

                # 3) Execute with monitoring
                execution_context = {
                    "system_metrics": await ctx.performance_monitor.get_current_metrics() if ctx.performance_monitor else {},
                    "security_level": ctx.security_manager.get_threat_level() if ctx.security_manager else "normal",
                    "plan_type": plan.get("type"),
                    "plan_id": plan.get("event_id", plan.get("plan_id"))
                }
                exec_result = await ctx.executor.execute(plan, context=execution_context)

                # Normalize result to dict
                if hasattr(exec_result, "to_dict"):
                    result = exec_result.to_dict()
                    exec_status = getattr(exec_result, "status", result.get("status"))
                elif isinstance(exec_result, dict):
                    result = exec_result
                    exec_status = result.get("status")
                else:
                    result = {"status": getattr(exec_result, "status", "unknown"), "raw": exec_result}
                    exec_status = result["status"]

                # 4) Troubleshooting & learning for failures
                if exec_status == "failed":
                    log.warning("Plan execution failed, invoking troubleshooter", extra={"plan": plan})
                    if ctx.self_improvement:
                        await ctx.self_improvement.learn_from_failure(plan, result)
                    if ctx.troubleshooter:
                        fix = await ctx.troubleshooter.attempt(plan=plan, error=result.get("error"))
                        if fix:
                            fix_exec = await ctx.executor.execute(fix)
                            # normalize
                            if hasattr(fix_exec, "to_dict"):
                                fix_res = fix_exec.to_dict()
                            else:
                                fix_res = fix_exec if isinstance(fix_exec, dict) else {"status": getattr(fix_exec, "status", "unknown")}
                            if fix_res.get("status") in ("completed", "success"):
                                if ctx.self_improvement:
                                    await ctx.self_improvement.learn_from_fix(plan, fix, fix_res)

                # 5) Persist result to memory & emit output
                try:
                    if ctx.memory:
                        await ctx.memory.save_event({"plan": plan, "result": result, "timestamp": time.time()})
                except Exception:
                    log.exception("Failed to save to memory")

                await ctx.output_manager.emit(result)


            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Core loop iteration error", extra={"event_type": safe_repr(event.get('type', 'unknown'))})
            finally:
                ctx.event_queue.task_done()
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Core loop crashed")
        raise

async def enrich_event(ctx: AppContext, event: Dict[str, Any]) -> Dict[str, Any]:
    """Add contextual information to events"""
    enriched = event.copy()
    
    # Add system context
    if ctx.performance_monitor:
        enriched["system_metrics"] = await ctx.performance_monitor.get_current_metrics()
    
    # Add security context
    if ctx.security_manager:
        enriched["security_context"] = {
            "threat_level": ctx.security_manager.get_threat_level(),
            "recent_activity": ctx.security_manager.get_recent_activity()
        }
    
    # Add user behavior patterns
    if ctx.memory and hasattr(ctx.memory, "get_user_patterns"):
        enriched["user_patterns"] = await ctx.memory.get_user_patterns()
    return enriched

async def get_system_status(ctx: AppContext) -> Dict[str, Any]:
    """Get comprehensive system status"""
    status = {}
    
    if ctx.performance_monitor:
        status["performance"] = await ctx.performance_monitor.get_system_status()
    
    if ctx.hardware_monitor:
        status["hardware"] = await ctx.hardware_monitor.get_status()
    
    if ctx.network_monitor:
        status["network"] = await ctx.network_monitor.get_status()
    
    # Add basic process monitoring if available
    if ctx.process_monitor:
        status["process_info"] = ctx.process_monitor.get_status()
    
    return status


async def improvement_maintenance_loop(ctx: AppContext) -> None:
    """
    Periodic self-improvement heartbeat loop.
    """
    if not ctx.self_improvement:
        return
    interval = int(ctx.config.get("self_improvement", {}).get("interval_sec", 900))
    try:
        while not ctx.shutdown_event.is_set():
            await asyncio.sleep(interval)
            with contextlib.suppress(Exception):
                await ctx.self_improvement.heartbeat()
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Self-improvement loop crashed")
        raise


# --------------------------------- Watchdog ---------------------------------
async def watchdog(
    task_factory: Callable[[], asyncio.Task],
    name: str,
    restart: bool,
    backoff: Sequence[float],
    shutdown_event: asyncio.Event,
) -> None:
    """
    Watches a task, restarts on failure with backoff if enabled.
    """
    restart_attempts = 0
    while not shutdown_event.is_set():
        task = task_factory()
        try:
            await task
            return  # completed normally
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Task crashed", extra={"task": name})
            if not restart:
                return
            delay = backoff[min(restart_attempts, len(backoff) - 1)]
            restart_attempts += 1
            log.warning("Restarting task", extra={"task": name, "delay": delay})
            await asyncio.sleep(delay)


# ---------------------------------- Signals ----------------------------------
def install_signal_handlers(shutdown_event: asyncio.Event) -> None:
    """
    Install SIGINT and SIGTERM handlers to trigger shutdown event.
    """
    loop = asyncio.get_running_loop()

    def _signal_handler(sig: signal.Signals) -> None:
        log.info("Signal received; initiating shutdown", extra={"signal": int(sig)})
        shutdown_event.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(s, _signal_handler, s)


# ----------------------------------- Utils -----------------------------------
def safe_repr(obj: Any, limit: int = 2048) -> str:
    """
    Safe repr with length limit and exception handling.
    """
    try:
        r = repr(obj)
        return (r[: limit - 3] + "...") if len(r) > limit else r
    except Exception:
        return "<unrepr-able>"


# Module-level flag to track logging silence
_root_logging_silenced = False


def silence_root_logging() -> None:
    """Silence the root logger so nothing is emitted to stderr/stdout.

    Use restore_root_logging() to re-enable logging.
    """
    global _root_logging_silenced
    try:
        logging.disable(logging.CRITICAL)
        _root_logging_silenced = True
    except Exception:
        # Best-effort; ignore failures
        pass


def restore_root_logging(level: int = logging.INFO) -> None:
    """Restore logging to normal and set the root level.

    level: a logging level like logging.INFO or logging.DEBUG
    """
    global _root_logging_silenced
    try:
        logging.disable(logging.NOTSET)
        # Set root logger level
        logging.getLogger().setLevel(level)

        # If requested level is DEBUG, aggressively enable DEBUG for all
        # existing loggers and handlers so the runtime is as verbose as
        # possible. This ensures --debug or a wake-phrase with -d produces
        # maximal diagnostic output.
        if level == logging.DEBUG:
            # Ensure basic configuration exists so handlers write to the console
            if not logging.getLogger().handlers:
                logging.basicConfig(level=logging.DEBUG)

            # Set all known loggers to DEBUG
            try:
                for name, logger_obj in list(logging.root.manager.loggerDict.items()):
                    if isinstance(logger_obj, logging.Logger):
                        logger_obj.setLevel(logging.DEBUG)
            except Exception:
                # best-effort; continue even if we cannot touch some loggers
                pass

            # Also ensure any handlers on the root logger are at DEBUG
            try:
                for h in logging.getLogger().handlers:
                    try:
                        h.setLevel(logging.DEBUG)
                    except Exception:
                        pass
            except Exception:
                pass

            # Additionally, ensure specific modules we commonly debug are set
            # to DEBUG so you get maximum detail from those subsystems.
            try:
                DEFAULT_DEBUG_MODULES = [
                    "core.executor",
                    "core.module_planning",
                    "core.planner_manager",
                    "core.planning",
                    "core.Ritsu_self",
                    "input.input_manager",
                    "ai.ai_assistant",
                ]
                for mod_name in DEFAULT_DEBUG_MODULES:
                    try:
                        logging.getLogger(mod_name).setLevel(logging.DEBUG)
                    except Exception:
                        # ignore per-module failures
                        pass
            except Exception:
                pass
        _root_logging_silenced = False
    except Exception:
        pass


def dump_loggers() -> str:
    """Return a human-readable dump of known loggers and their effective levels/handlers.

    This is intended for quick runtime inspection when debugging is enabled.
    """
    out_lines = []
    try:
        root = logging.getLogger()
        out_lines.append(f"Root level: {logging.getLevelName(root.getEffectiveLevel())}")
        out_lines.append(f"Root handlers: {len(root.handlers)}")

        try:
            for name, obj in sorted(logging.root.manager.loggerDict.items()):
                try:
                    if isinstance(obj, logging.Logger):
                        lvl = logging.getLevelName(obj.getEffectiveLevel())
                        handlers = getattr(obj, 'handlers', [])
                        out_lines.append(f"{name}: level={lvl} handlers={len(handlers)}")
                    else:
                        out_lines.append(f"{name}: <non-logger-entry>")
                except Exception:
                    out_lines.append(f"{name}: <error-inspecting>")
        except Exception:
            out_lines.append("<failed to enumerate loggerDict>")
    except Exception as e:
        out_lines.append(f"Failed to dump loggers: {e}")

    return "\n".join(out_lines)


async def terminal_wake_listener(ctx: AppContext) -> None:
    """Background task that reads terminal input and looks for wake phrases.

    When a wake phrase is detected (e.g. "ritsu", "hey ritsu", "ritsu on"),
    this will restore logging (INFO or DEBUG if --debug is present) and
    enqueue a 'terminal_wake' event onto ctx.event_queue.
    """
    import re

    wake_pattern = re.compile(r'^(hey\s+ritsu|ritsu)(?:\s|$)', re.IGNORECASE)

    # Use a thread for blocking stdin.readline so we don't block the loop
    while not ctx.shutdown_event.is_set():
        try:
            line = await asyncio.to_thread(sys.stdin.readline)
            if line is None:
                await asyncio.sleep(0.1)
                continue
            line = line.strip()
            if not line:
                continue

            m = wake_pattern.match(line)
            if m:
                # detect debug flag (case-insensitive)
                level = logging.INFO
                try:
                    if re.search(r"(?i)(?:--debug|-d)\b", line or ""):
                        level = logging.DEBUG
                except Exception:
                    # best-effort; leave level as INFO on error
                    pass

                restore_root_logging(level=level)

                # Extract remainder after wake phrase
                tail = line[m.end():].strip()

                # If user requests shell execution (prefix with '!' or --shell), run it locally
                if tail.startswith('!') or tail.startswith('--shell') or tail.startswith('-s'):
                    # normalize command
                    if tail.startswith('!'):
                        cmd = tail[1:].strip()
                    else:
                        # --shell <cmd> or -s <cmd>
                        parts = tail.split(None, 1)
                        cmd = parts[1].strip() if len(parts) > 1 else ''

                    if cmd:
                        try:
                            await run_shell_command(cmd)
                        except Exception:
                            log.exception("Shell command execution failed")
                    else:
                        print("No shell command provided.")

                else:
                    # If no command followed the wake phrase, read the next line as command
                    if not tail:
                        # Prompt user and read their follow-up input (non-blocking)
                        print("Ritsu: listening...", end=" \n")
                        try:
                            next_line = await asyncio.to_thread(sys.stdin.readline)
                            if next_line is None:
                                continue
                            next_line = next_line.strip()
                        except Exception:
                            log.exception("Failed to read follow-up command after wake")
                            next_line = ""

                        # Detect and remove debug flags from the user's input so the
                        # actual message passed to the LLM doesn't include them.
                        try:
                            # case-insensitive flag detection
                            if re.search(r"(?i)(?:--debug|-d)\b", next_line or ""):
                                level = logging.DEBUG
                                restore_root_logging(level=level)
                            # Remove any occurrences of -d or --debug from the input
                            cleaned = re.sub(r"(?i)(?:--debug|-d)\b", "", next_line or "").strip()
                        except Exception:
                            cleaned = next_line or ""

                        # Ask the LLM to generate a small greeting/ack now that we have
                        # the user's input. If LLM isn't available, skip.
                        greet = ""
                        try:
                            # Always pass the cleaned user input into the LLM greeting
                            # call (even if empty). Debug flags have already been removed
                            # from 'cleaned' above.
                            try:
                                if getattr(ctx, "llm", None) and hasattr(ctx.llm, "generate_response"):
                                    greet = await ctx.llm.generate_response({"User": cleaned}, mode=1)
                            except Exception:
                                log.exception("LLM generate_response failed")
                        except Exception:
                            log.exception("LLM generate_response failed")

                        if greet:
                            print(f"Ritsu: {greet}", end=" \n")

                        text_to_send = cleaned
                    else:
                        # tail already contains the command
                        text_to_send = tail

                    # If the text_to_send requests shell execution, handle that here
                    if text_to_send and (text_to_send.startswith('!') or text_to_send.startswith('--shell') or text_to_send.startswith('-s')):
                        if text_to_send.startswith('!'):
                            cmd = text_to_send[1:].strip()
                        else:
                            parts = text_to_send.split(None, 1)
                            cmd = parts[1].strip() if len(parts) > 1 else ''

                        if cmd:
                            try:
                                await run_shell_command(cmd)
                            except Exception:
                                log.exception("Shell command execution failed")
                        else:
                            print("No shell command provided.")
                    else:
                        # enqueue a wake event so core loop can handle it as input
                        try:
                            # Special runtime commands handled directly
                            cmd_lower = (text_to_send or line or "").strip()
                            cmd_lower_l = cmd_lower.lower()

                            # Handle dump/loggers command
                            if cmd_lower_l in ("dump-loggers", "dump loggers", "dump_loggers"):
                                try:
                                    out = dump_loggers()
                                    print(out)
                                except Exception:
                                    log.exception("Failed to dump loggers via terminal command")
                                continue

                            # Mode change commands: e.g. 'mode chat', 'set mode background', 'enter follow_plan'
                            try:
                                parts = cmd_lower_l.replace(':', ' ').replace('-', '_').split()
                                # last token may be the desired mode
                                # detect 'mode status' requests
                                if len(parts) >= 2 and parts[0] in ("mode",) and parts[1] in ("status", "status?"):
                                    print(f"System mode: {getattr(ctx, 'system_mode', 'background')}")
                                    desired = None
                                elif len(parts) >= 2 and parts[0] in ("mode", "set", "enter"):
                                    desired = parts[-1].replace('-', '_')
                                elif parts and parts[0] in ("chat", "background", "follow_plan", "follow", "followplan"):
                                    desired = parts[0].replace('-', '_')
                                else:
                                    desired = None
                            except Exception:
                                desired = None

                            if desired in ("background", "chat", "follow_plan", "follow"):
                                # normalize follow -> follow_plan
                                if desired == "follow":
                                    desired = "follow_plan"
                                ctx.system_mode = desired
                                print(f"System mode set to: {ctx.system_mode}")
                                continue

                            # Route based on current system mode
                            try:
                                mode = getattr(ctx, "system_mode", "background")
                            except Exception:
                                mode = "background"

                            if mode == "background":
                                # In background mode, treat most inputs as shell commands (light-weight)
                                if cmd_lower:
                                    try:
                                        await run_shell_command(cmd_lower)
                                    except Exception:
                                        log.exception("Background shell execution failed")
                                else:
                                    print("No command provided.")
                            elif mode == "chat":
                                # Chat mode: send text to the core as a terminal_wake/chat event
                                await ctx.event_queue.put({
                                    "type": "terminal_wake",
                                    "text": text_to_send or line,
                                    "debug": level == logging.DEBUG,
                                })
                            elif mode == "follow_plan":
                                # Heavy planning mode: enqueue specialized follow_plan event
                                await ctx.event_queue.put({
                                    "type": "follow_plan",
                                    "text": text_to_send or line,
                                    "debug": level == logging.DEBUG,
                                })
                                print("Enqueued follow_plan event to planner")
                            else:
                                # Default: enqueue as terminal_wake
                                await ctx.event_queue.put({
                                    "type": "terminal_wake",
                                    "text": text_to_send or line,
                                    "debug": level == logging.DEBUG,
                                })
                        except Exception:
                            log.exception("Failed to enqueue terminal wake event")

        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Terminal wake listener crashed")
            await asyncio.sleep(1)


async def run_shell_command(cmd: str) -> None:
    """Run a shell command and stream output to the terminal.

    This is a synchronous command run wrapper using asyncio subprocess.
    """
    if not cmd:
        print("No command to run")
        return

    print(f"$ {cmd}")
    try:
        # Use shell on Windows via powershell.exe for consistent behavior
        if sys.platform == 'win32':
            proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL, shell=True)
        else:
            proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL)

        # Read both stdout and stderr
        async def stream_reader(stream, write_fn):
            while True:
                line = await stream.readline()
                if not line:
                    break
                write_fn(line.decode(errors='replace'))

        await asyncio.gather(
            stream_reader(proc.stdout, lambda s: print(s, end='')),
            stream_reader(proc.stderr, lambda s: print(s, end='')),
        )

        rc = await proc.wait()
        print(f"[exit code: {rc}]")
    except Exception as e:
        print(f"Error running command: {e}")
        log.exception("Error running shell command")


async def shutdown(ctx: AppContext, tasks: Sequence[asyncio.Task], timeout: float = 10.0) -> None:
    """
    Gracefully shutdown all tasks and components.
    """
    log.info("Shutting down...")
    ctx.shutdown_event.set()

    # Cancel running tasks
    for t in tasks:
        t.cancel()

    # Give tasks a chance to finish
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("Timeout waiting for tasks to finish during shutdown")

    # Component-level cleanup
    for name in [
        "input_manager",
        "event_manager",
        "planner",
        "executor",
        "output_manager",
        "troubleshooter",
        "self_improvement",
        "ritsu_self",
        "nlp",
        "kb",
        "memory",
        "tts",
        "avatar",
        "stream",
        "rust_editor",
        "ui",
        # Basic tools
        "calculator",
        "file_reader",
        "process_monitor",
    ]:
        comp = getattr(ctx, name, None)
        if comp and hasattr(comp, "close"):
            try:
                maybe = comp.close()
                if asyncio.iscoroutine(maybe):
                    await maybe
                    # Prefer coroutine shutdown() if present
                    if hasattr(comp, "shutdown"):
                        try:
                            maybe = comp.shutdown()
                            if asyncio.iscoroutine(maybe):
                                await asyncio.wait_for(maybe, timeout=5.0)
                            else:
                                # non-async shutdown
                                comp.shutdown()
                        except Exception:
                            log.exception("Error during component.shutdown() for %s", name)
                    else:
                        # Fallback to close()
                        comp.close()
            except Exception as e:
                log.warning(f"Error closing component {name}: {e}")

    log.info("Shutdown complete")


# ------------------------------------ CLI ------------------------------------
def parse_args(argv: Sequence[str]) -> Dict[str, Any]:
    """
    Parse CLI arguments.
    """
    import argparse

    p = argparse.ArgumentParser(prog="ritsu", description="Ritsu — {des}")
    p.add_argument("--config", type=Path, default=Path("system/config.yaml"), help="Path to config file")
    p.add_argument("--log-level", type=str, default=None, help="Override log level (e.g., INFO, DEBUG)")
    p.add_argument("--headless", action="store_true", help="Disable UI integration")
    p.add_argument("--safe-mode", action="store_true", help="Restrict risky operations/tools")
    p.add_argument("--no-restart", action="store_true", help="Disable auto-restart on crash")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    p.add_argument("--debug", "-d", action="store_true", help="Enable debug logging at startup")
    p.add_argument("--dump-loggers", action="store_true", help="Dump logger states at startup and exit")
    p.add_argument("--mode", type=str, choices=["background", "chat", "follow_plan"], default=None, help="Initial system interaction mode")
    p.add_argument("-m", "--mode-report", action="store_true", dest="mode_report", help="Print current system mode and exit")
    p.add_argument("--sys-mode", type=int, choices=[1,2,3], default=None, help="Numeric shorthand to set system mode: 1=background,2=chat,3=follow_plan")

# later add --debug, --dry-run, --test, --profile --view-logs, --reset-config, --list-tools, --review-logs, --review-self-log, |-acsess-invoke ....
# |--enable-plugin <name>, --disable-plugin <name>, --list-plugins, --update-plugins

    ns = p.parse_args(argv)
    if ns.version:
        print(f"Ritsu main.py v{version}")
        sys.exit(0)

    return vars(ns)


# ----------------------------------- Runner -----------------------------------
async def run(argv: Sequence[str]) -> int:
    """
    Main async runner.
    """
    args = parse_args(argv)
    # At startup, silence structured logging by default and only print a simple startup line.
    # If --debug/-d was provided, restore debug logging immediately so startup logs are visible.
    if not args.get("debug"):
        silence_root_logging()
    else:
        restore_root_logging(level=logging.DEBUG)
        # Inform the user clearly that debug mode is active so logs from
        # other modules will be visible. This is helpful when debugging
        # across the codebase.
        print("Starting Ritsu in DEBUG mode: verbose logging enabled.")
    print(f"Starting Ritsu v{version}...")
    # Load and merge config
    ctx = await bootstrap(args.get("config"))
    # Apply CLI provided mode override, or read from config if present
    desired_mode = args.get("mode") if isinstance(args, dict) else None
    if not desired_mode:
        try:
            desired_mode = ctx.config.get("system", {}).get("mode") if isinstance(ctx.config, dict) else getattr(ctx.config, "system", {}).get("mode")
        except Exception:
            desired_mode = None
    if desired_mode:
        set_system_mode(ctx, desired_mode, persist=False)

    # If numeric sys-mode CLI provided, apply mapping and persist
    sys_mode_num = args.get("sys_mode") if isinstance(args, dict) else None
    if sys_mode_num:
        num_map = {1: "background", 2: "chat", 3: "follow_plan"}
        mapped = num_map.get(sys_mode_num)
        if mapped:
            set_system_mode(ctx, mapped, persist=True)

    # If mode-report flag set, print and exit
    if args.get("mode_report"):
        print(f"System mode: {getattr(ctx, 'system_mode', 'background')}")
        return 0

    # If requested, dump logger states immediately for troubleshooting and exit
    if args.get("dump_loggers"):
        try:
            dump = dump_loggers()
            print(dump)
        except Exception:
            log.exception("Failed to dump loggers on startup")
        return 0

    # CLI overrides
    if args.get("log_level"):
        setup_logging(level=args["log_level"], log_dir=Path(ctx.config.get("logging", {}).get("dir", "data/logs")))
        try:
            lvl = getattr(logging, args["log_level"].upper(), logging.INFO)
        except Exception:
            lvl = logging.INFO
        restore_root_logging(level=lvl)
    if args.get("headless") and ctx.ui:
        log.info("Headless mode: disabling UI client")
        ctx.ui = None
        if "ui" in ctx.config:
            ctx.config["ui"]["enabled"] = False
    if args.get("safe_mode"):
        ctx.config.setdefault("app", {})["safe_mode"] = True

    install_signal_handlers(ctx.shutdown_event)
    
    # Start output manager processing
    if ctx.output_manager and hasattr(ctx.output_manager, 'start'):
        await ctx.output_manager.start()
        log.info("Output manager started")

    tasks: Dict[str, asyncio.Task] = {}

    # Terminal wake listener: only start it when InputManager isn't handling CLI
    terminal_wake_task: Optional[asyncio.Task] = None
    try:
        input_manager_handles_cli = getattr(ctx, 'input_manager', None) is not None
    except Exception:
        input_manager_handles_cli = False

    if sys.stdin and sys.stdin.isatty() and not input_manager_handles_cli:
        terminal_wake_task = asyncio.create_task(terminal_wake_listener(ctx), name="terminal_wake")
        tasks["terminal_wake"] = terminal_wake_task

    # --- PRIORITY STARTUP ORDER ---
    # Core brain first
    if ctx.event_manager and ctx.planner and ctx.executor and ctx.output_manager:
        tasks["core"] = asyncio.create_task(core_loop(ctx), name="core_loop")
    else:
        missing = []
        if not ctx.event_manager: missing.append("event_manager")
        if not ctx.planner: missing.append("planner")
        if not ctx.executor: missing.append("executor")
        if not ctx.output_manager: missing.append("output_manager")
        log.error(f"Core components missing: {', '.join(missing)}; cannot start core loop")
        return 1

    # Input pipeline (only after core)
    if ctx.input_manager:
        tasks["input"] = asyncio.create_task(input_loop(ctx), name="input_loop")
    else:
        log.warning("InputManager not available; skipping input loop")

    # Monitoring (optional)
    if ctx.performance_monitor:
        tasks["monitoring"] = asyncio.create_task(monitoring_loop(ctx), name="monitoring_loop")
    else:
        log.info("PerformanceMonitor not available; skipping monitoring loop")

    # Maintenance (always safe to run)
    tasks["maintenance"] = asyncio.create_task(maintenance_loop(ctx), name="maintenance_loop")

    # Self-improvement (optional background)
    if ctx.self_improvement:
        tasks["improve"] = asyncio.create_task(improvement_maintenance_loop(ctx), name="improve_loop")
    else:
        log.info("SelfImprovement not available; skipping improvement loop")

    # Restart logic
    restart_on_crash = bool(ctx.config.get("app", {}).get("restart_on_crash", True)) and not args.get("no_restart")
    backoff = [1.0, 2.0, 5.0, 10.0, 15.0]

    if ctx.event_manager:
        # If core loop is running it consumes the main event_queue. Running EventManager.run
        # against the same queue would cause duplicate consumption of events (each consumer
        # would receive different events) and can lead to duplicate outputs. Only start the
        # EventManager loop when core loop is NOT started and the event manager should act
        # as the primary consumer.
        if "core" in tasks:
            log.info("EventManager available but core loop is active; not starting event_manager.run to avoid duplicate event consumption")
        else:
            tasks["events"] = asyncio.create_task(
                ctx.event_manager.run(ctx.shutdown_event),
                name="event_manager_loop"
            )
    else:
        log.warning("EventManager not available; skipping event manager loop")

    try:
        while not ctx.shutdown_event.is_set():
            if not tasks:
                log.error("No tasks to run; exiting.")
                break

            done, pending = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)

            for t in done:
                # Determine the task name from the tasks mapping (if present)
                name = next((k for k, v in tasks.items() if v is t), t.get_name())
                try:
                    _ = t.result()
                    # Use DEBUG here to avoid noisy INFO-level spam when background
                    # tasks complete frequently; the user-visible startup line is
                    # already printed and normal runtime logs will appear when
                    # logging is restored (e.g. via wake or --debug).
                    log.debug("Task completed", extra={"task": name})

                    if name == "core":
                        log.warning("Core loop finished; shutting down everything")
                        ctx.shutdown_event.set()

                except asyncio.CancelledError:
                    # Task was cancelled during shutdown; ignore
                    pass
                except Exception:
                    log.exception("Task error", extra={"task": name})
                    if name == "core":
                        log.error("Core loop crashed; shutting down")
                        ctx.shutdown_event.set()
                    elif restart_on_crash and name == "input" and not ctx.shutdown_event.is_set():
                        delay = backoff[min(len(backoff) - 1, int(time.time()) % len(backoff))]
                        log.warning("Restarting input loop after backoff", extra={"delay": delay})
                        await asyncio.sleep(delay)
                        if ctx.input_manager:
                            # recreate task and replace mapping
                            tasks[name] = asyncio.create_task(input_loop(ctx), name="input_loop")
                        else:
                            log.error("InputManager still missing; cannot restart input loop.")
                    else:
                        ctx.shutdown_event.set()

                # Mark completed task for removal from the dict so it won't be considered again
                # (prevents busy-loop where wait() keeps returning already-done tasks)
                try:
                    # only remove the mapping if the mapping still points to this completed task
                    if tasks.get(name) is t:
                        tasks.pop(name, None)
                except Exception:
                    # continue even if removal fails for some reason
                    pass

            if all(t.done() for t in tasks.values()):
                break

        await shutdown(ctx, list(tasks.values()))
        return 0

    except Exception:
        log.exception("Fatal error in run loop")
        with contextlib.suppress(Exception):
            await shutdown(ctx, list(tasks.values()))
        return 1

def main() -> None:
    try:
        exit_code = asyncio.run(run(sys.argv[1:]))
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        log.exception("Unhandled exception at top-level")
        exit_code = 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()  