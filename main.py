from __future__ import annotations

"""
Ritsu Main Entrypoint
Initializes core systems, input/output, and starts event loop.
"""

import asyncio
import contextlib
import logging
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Callable
from config import version

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

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


log = get_logger("ritsu.main")


def log_with_context(logger: logging.Logger, level: int, msg: str, **kwargs: Any) -> None:
    """
    Helper to log with structured context.
    """
    logger.log(level, msg, extra=kwargs)


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
                "io": {"enable_mic": False, "enable_chat": True},
                "ui": {"enabled": False},
                "rust_editor": {"enabled": False},
            }

        def load(self) -> Dict[str, Any]:
            return self._config


# ---------------------------- Core / Input / Output ----------------------------
def import_optional(module_name: str) -> Optional[Any]:
    try:
        module = __import__(module_name, fromlist=["*"])
        return module
    except Exception as e:
        log.warning(f"Optional module '{module_name}' failed to import: {e}")
        return None

# Import the basic tools you created
try:
    from core.Tools.math.calculator import Calculator
    from core.Tools.file_reader import FileReader
    from core.Tools.process_monitor import ProcessMonitor
    from llm.ritsu_llm import RitsuLLM
except ImportError as e:
    log.warning(f"Failed to import basic tools: {e}")
    Calculator = FileReader = ProcessMonitor = RitsuLLM = None

EventManager = import_optional("core.event_manager")
Planner = import_optional("core.planning")
Executor = import_optional("core.executor")
Troubleshooter = import_optional("core.troubleshooter")
SelfImprovement = import_optional("core.self_improvement")
RitsuSelf = import_optional("core.Ritsu_self")
Toolbelt = import_optional("core.tools")
CodeAnalyzer = import_optional("core.code_analyzer")
CodeGenerator = import_optional("core.code_generator")
CodeDB = import_optional("core.codedb")

InputManager = import_optional("input.input_manager")
STT = import_optional("input.stt")
ChatListener = import_optional("input.chat_listener")
CommandParser = import_optional("input.command_parser")

OutputManager = import_optional("output.output_manager")
TTS = import_optional("output.tts")
AvatarAnimator = import_optional("output.avatar_animator")
StreamAdapter = import_optional("output.stream_adapter")

PromptTemplates = import_optional("llm.prompt_templates")

NLPEngine = import_optional("ai.nlp_engine")
KnowledgeBase = import_optional("ai.knowledge_base")
MemoryManager = import_optional("ai.memory_manager")
AIAssistant = import_optional("ai.ai_assistant")

RustEditor = import_optional("system.bindings_rust")
UIClient = import_optional("system.bindings_ui")

#====Needed to be worked on latter====

PerformanceMonitor = import_optional("core.performance_monitor")
SecurityManager = import_optional("core.security_manager")
AutoUpdater = import_optional("core.auto_updater")
PluginManager = import_optional("core.plugin_manager")
TaskScheduler = import_optional("core.task_scheduler")

HardwareMonitor = import_optional("core.hardware_monitor")
SystemAnalyzer = import_optional("core.system_analyzer")
NetworkMonitor = import_optional("core.network_monitor")

CodeReviewer = import_optional("core.code_reviewer")
TestGenerator = import_optional("core.test_generator")
DocumentationGenerator = import_optional("core.documentation_generator")


# --------------------------------- App types ---------------------------------
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

    # Basic tools (your new additions)
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


# --------------------------------- Bootstrap ---------------------------------
def add_enhanced_components(ctx: "AppContext", config: dict) -> None:
    """Attach optional/enhanced components to the context."""
    with contextlib.suppress(Exception):
        ctx.performance_monitor = PerformanceMonitor(config=config.get("performance", {})) if PerformanceMonitor else None
    with contextlib.suppress(Exception):
        ctx.security_manager = SecurityManager(config=config.get("security", {})) if SecurityManager else None
    with contextlib.suppress(Exception):
        ctx.auto_updater = AutoUpdater(config=config.get("auto_update", {})) if AutoUpdater else None
    with contextlib.suppress(Exception):
        ctx.plugin_manager = PluginManager(config=config.get("plugins", {})) if PluginManager else None
    with contextlib.suppress(Exception):
        ctx.task_scheduler = TaskScheduler(config=config.get("scheduler", {})) if TaskScheduler else None

    # Hardware and system monitoring 
    with contextlib.suppress(Exception):
        ctx.hardware_monitor = HardwareMonitor(config=config.get("hardware", {})) if HardwareMonitor else None
    with contextlib.suppress(Exception):
        ctx.system_analyzer = SystemAnalyzer(config=config.get("system_analyzer", {})) if SystemAnalyzer else None
    with contextlib.suppress(Exception):
        ctx.network_monitor = NetworkMonitor(config=config.get("network", {})) if NetworkMonitor else None

    # Advanced code intelligence
    with contextlib.suppress(Exception):
        ctx.code_reviewer = CodeReviewer(config=config.get("code_review", {})) if CodeReviewer else None
    with contextlib.suppress(Exception):
        ctx.test_generator = TestGenerator(config=config.get("test_gen", {})) if TestGenerator else None
    with contextlib.suppress(Exception):
        ctx.documentation_generator = DocumentationGenerator(config=config.get("docs", {})) if DocumentationGenerator else None


async def bootstrap(config_path: Optional[Path]) -> "AppContext":
    """
    Initialize and configure all subsystems.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Initialized AppContext with all components.
    """
    cfg_mgr = ConfigManager(config_path)
    config = cfg_mgr.load()

    log_dir = Path(config.get("logging", {}).get("dir", "data/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        level=str(config.get("logging", {}).get("level", "INFO")),
        log_dir=log_dir,
        json=bool(config.get("logging", {}).get("json", True)),
    )

    ctx = AppContext(config=config)

    # Instantiate basic tools first
    with contextlib.suppress(Exception):
        ctx.calculator = Calculator() if Calculator else None
    with contextlib.suppress(Exception):
        ctx.file_reader = FileReader() if FileReader else None
    with contextlib.suppress(Exception):
        ctx.process_monitor = ProcessMonitor() if ProcessMonitor else None

    # Instantiate core components defensively
    with contextlib.suppress(Exception):
        ctx.prompts = PromptTemplates() if PromptTemplates else None
    with contextlib.suppress(Exception):
        ctx.llm = RitsuLLM(config=config.get("llm", {})) if RitsuLLM else None
    with contextlib.suppress(Exception):
        ctx.tools = Toolbelt(config=config.get("tools", {})) if Toolbelt else None
    with contextlib.suppress(Exception):
        ctx.event_manager = EventManager(config=config.get("events", {}), queue=ctx.event_queue) if EventManager else None
    with contextlib.suppress(Exception):
        ctx.planner = Planner(config=config.get("planning", {})) if Planner else None
    with contextlib.suppress(Exception):
        ctx.executor = Executor(config=config.get("executor", {})) if Executor else None
    with contextlib.suppress(Exception):
        ctx.troubleshooter = Troubleshooter(config=config.get("troubleshooter", {})) if Troubleshooter else None
    with contextlib.suppress(Exception):
        ctx.self_improvement = SelfImprovement(config=config.get("self_improvement", {})) if SelfImprovement else None
    with contextlib.suppress(Exception):
        ctx.ritsu_self = RitsuSelf(config=config.get("ritsu_self", {})) if RitsuSelf else None
    with contextlib.suppress(Exception):
        ctx.code_analyzer = CodeAnalyzer(config=config.get("code_analyzer", {})) if CodeAnalyzer else None
    with contextlib.suppress(Exception):
        ctx.code_generator = CodeGenerator(config=config.get("code_generator", {})) if CodeGenerator else None
    with contextlib.suppress(Exception):
        ctx.codedb = CodeDB(path=Path(config.get("codedb", {}).get("path", "data/codedb"))) if CodeDB else None

    # Input stack
    with contextlib.suppress(Exception):
        ctx.command_parser = CommandParser(config=config.get("command_parser", {})) if CommandParser else None
    with contextlib.suppress(Exception):
        ctx.mic = STT(config=config.get("mic", {})) if STT else None
    with contextlib.suppress(Exception):
        ctx.chat = ChatListener(config=config.get("chat", {})) if ChatListener else None
    with contextlib.suppress(Exception):
        ctx.input_manager = InputManager(
            config=config.get("input", {}),
            mic=ctx.mic,
            chat=ctx.chat,
            command_parser=ctx.command_parser,
        ) if InputManager else None

    # Output stack
    with contextlib.suppress(Exception):
        ctx.tts = TTS(config=config.get("tts", {})) if TTS else None
    with contextlib.suppress(Exception):
        ctx.avatar = AvatarAnimator(config=config.get("avatar", {})) if AvatarAnimator else None
    with contextlib.suppress(Exception):
        ctx.stream = StreamAdapter(config=config.get("stream", {})) if StreamAdapter else None
    with contextlib.suppress(Exception):
        ctx.output_manager = OutputManager(
            config=config.get("output", {}), tts=ctx.tts, avatar=ctx.avatar, stream=ctx.stream
        ) if OutputManager else None

    # AI components
    with contextlib.suppress(Exception):
        ctx.nlp = NLPEngine(config=config.get("nlp", {})) if NLPEngine else None
    with contextlib.suppress(Exception):
        ctx.kb = KnowledgeBase(config=config.get("knowledge_base", {})) if KnowledgeBase else None
    with contextlib.suppress(Exception):
        ctx.memory = MemoryManager(config=config.get("memory", {})) if MemoryManager else None
    with contextlib.suppress(Exception):
        ctx.ai_assistant = AIAssistant(
            config=config.get("ai", {}),
            nlp_engine=ctx.nlp,
            knowledge_base=ctx.kb,
            memory_manager=ctx.memory,
            llm_engine=ctx.llm,
        ) if AIAssistant else None

    # Optional external tools
    with contextlib.suppress(Exception):
        if config.get("rust_editor", {}).get("enabled", False) and RustEditor:
            ctx.rust_editor = RustEditor(config.get("rust_editor", {}))
    with contextlib.suppress(Exception):
        if config.get("ui", {}).get("enabled", False) and UIClient:
            ctx.ui = UIClient(config=config.get("ui", {}))

    # Executor cross-references - include basic tools
    with contextlib.suppress(Exception):
        if ctx.executor:
            executor_components = {
                "ai_assistant": ctx.ai_assistant,
                "memory_manager": ctx.memory,
                "knowledge_base": ctx.kb,
                "nlp_engine": ctx.nlp,
                "toolbelt": ctx.tools,
                "output_manager": ctx.output_manager,
                # Add basic tools
                "calculator": ctx.calculator,
                "file_reader": ctx.file_reader,
                "process_monitor": ctx.process_monitor,
            }
            ctx.executor.set_components(executor_components)

    # Attach enhanced components
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
            
            # Update knowledge base
            if ctx.kb:
                await ctx.kb.auto_update()
            
            # Optimize databases
            if ctx.codedb:
                ctx.codedb.optimize()
                
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Maintenance loop crashed")



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
    if not (ctx.event_manager and ctx.planner and ctx.executor and ctx.output_manager):
        log.error("Core components missing; core loop cannot start")
        return

    try:
        while not ctx.shutdown_event.is_set():
            event: Dict[str, Any] = await ctx.event_queue.get()
            try:
                # 1) Enhanced event processing with context
                enriched_event = await enrich_event(ctx, event)
                ctx.event_manager.handle(enriched_event)

                # 2) Plan with system context
                system_status = await get_system_status(ctx)
                plan = ctx.planner.decide(enriched_event, system_context=system_status)

                # 3) Execute with monitoring
                execution_context = {
                    "system_metrics": await ctx.performance_monitor.get_current_metrics() if ctx.performance_monitor else {},
                    "security_level": ctx.security_manager.get_threat_level() if ctx.security_manager else "normal"
                }
                result = await ctx.executor.execute(plan, context=execution_context)

                # 4) Advanced troubleshooting and learning
                if result and result.get("status") == "failed":
                    # Log failure for learning
                    if ctx.self_improvement:
                        await ctx.self_improvement.learn_from_failure(plan, result)
                    
                    # Advanced troubleshooting
                    if ctx.troubleshooter:
                        fix = await ctx.troubleshooter.attempt(plan=plan, error=result.get("error"))
                        if fix:
                            result = await ctx.executor.execute(fix)
                            
                            # Learn from successful fix
                            if result.get("status") == "success" and ctx.self_improvement:
                                await ctx.self_improvement.learn_from_fix(plan, fix, result)

                # 5) Enhanced output with analytics
                if result and result.get("status") == "success" and ctx.performance_monitor:
                    # Update performance benchmarks
                    await ctx.performance_monitor.record_success(plan.get("action"))

                await ctx.output_manager.emit(result)

            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Core loop iteration error", extra={"event": safe_repr(event)})
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
    if ctx.memory:
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

    ns = p.parse_args(argv)
    if ns.version:
        print("Ritsu main.py v{version}")
        sys.exit(0)

    return vars(ns)


# ----------------------------------- Runner -----------------------------------
async def run(argv: Sequence[str]) -> int:
    """
    Main async runner.
    """
    args = parse_args(argv)

    # Load and merge config
    ctx = await bootstrap(args.get("config"))

    # CLI overrides
    if args.get("log_level"):
        setup_logging(level=args["log_level"], log_dir=Path(ctx.config.get("logging", {}).get("dir", "data/logs")))
    if args.get("headless") and ctx.ui:
        log.info("Headless mode: disabling UI client")
        ctx.ui = None
        if "ui" in ctx.config:
            ctx.config["ui"]["enabled"] = False
    if args.get("safe_mode"):
        ctx.config.setdefault("app", {})["safe_mode"] = True

    install_signal_handlers(ctx.shutdown_event)

    tasks: Dict[str, asyncio.Task] = {}

    # --- PRIORITY STARTUP ORDER ---
    # Core brain first
    if ctx.event_manager and ctx.planner and ctx.executor and ctx.output_manager:
        tasks["core"] = asyncio.create_task(core_loop(ctx), name="core_loop")
    else:
        log.warning("Core components missing; cannot start core loop")
        return 1  # critical failure → exit immediately

    # Input pipeline (only after core)
    if ctx.input_manager:
        tasks["input"] = asyncio.create_task(input_loop(ctx), name="input_loop")
    else:
        log.warning("InputManager not available; skipping input loop")

    # Monitoring (optional)
    if ctx.performance_monitor:
        tasks["monitoring"] = asyncio.create_task(monitoring_loop(ctx), name="monitoring_loop")

    # Maintenance (always safe to run)
    tasks["maintenance"] = asyncio.create_task(maintenance_loop(ctx), name="maintenance_loop")

    # Self-improvement (optional background)
    if ctx.self_improvement:
        tasks["improve"] = asyncio.create_task(improvement_maintenance_loop(ctx), name="improve_loop")

    # Restart logic
    restart_on_crash = bool(ctx.config.get("app", {}).get("restart_on_crash", True)) and not args.get("no_restart")
    backoff = [1.0, 2.0, 5.0, 10.0, 15.0]

    try:
        while not ctx.shutdown_event.is_set():
            done, pending = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)

            for t in done:
                name = next((k for k, v in tasks.items() if v is t), t.get_name())
                try:
                    _ = t.result()
                    log.info("Task completed", extra={"task": name})

                    if name == "core":
                        log.warning("Core loop finished; shutting down everything")
                        ctx.shutdown_event.set()

                except asyncio.CancelledError:
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
                        tasks[name] = asyncio.create_task(input_loop(ctx), name="input_loop")
                    else:
                        ctx.shutdown_event.set()

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