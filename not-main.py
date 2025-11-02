# main.py
# Ritsu — Robust async entrypoint for the autonomous assistant
# Target: Python 3.11+
#
# Features implemented:
# - asyncio-based startup/shutdown with dependency injection
# - Config loading (YAML) with pydantic validation and env overrides
# - Hot-reload for non-critical settings (file watcher)
# - Logging (console + rotating file) with structured messages
# - Health-check HTTP endpoint (aiohttp) and metrics endpoint
# - Service-mode awareness (systemd notify optional, Windows Service optional)
# - Signal handling (SIGINT, SIGTERM) and graceful shutdown
# - Component lifecycle hooks (startup/shutdown)
# - Watchdog-style restart/backoff for critical loops
# - Basic system requirement checks (python version, disk space)
# - LLM / memory / background loops scaffolding (with health checks)
# - Comprehensive exception handling and degraded fallback modes
#
# NOTE: Many components (LLM adapters, MemoryManager, Planner, Executor...) are
# expected to exist in the project and expose standard lifecycle methods:
#   async def startup(self) -> None
#   async def healthcheck(self) -> dict | bool
#   async def shutdown(self) -> None
# If such methods aren't present, safe wrappers attempt best-effort usage.
#
# This file is intentionally dependency-light: optional imports are guarded
# so the runtime can still start on low-spec systems. Add real implementations
# in project modules and they will be auto-detected.
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import logging.handlers
import os
import platform
import signal
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

# Optional heavy deps: guarded imports
try:
    import yaml
except Exception:
    yaml = None  # fallback; ConfigManager will error if yaml not present

try:
    from pydantic import BaseModel, Field, validator
except Exception:
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore

# aiohttp used only for health/metrics server; optional but recommended
try:
    from aiohttp import web
except Exception:
    web = None  # health endpoint will be disabled if aiohttp missing

# Optional systemd notifier
try:
    import sdnotify  # type: ignore
except Exception:
    sdnotify = None

# Optional Windows service support
try:
    import win32serviceutil  # type: ignore
    import win32service  # type: ignore
    import win32event  # type: ignore
except Exception:
    win32serviceutil = win32service = win32event = None

__all__ = ("main",)

# ---------------------------
# Logging utilities
# ---------------------------


def setup_logging(
    *,
    level: str = "INFO",
    log_dir: Path | str = "logs",
    filename: str = "ritsu.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    fmt: str | None = None,
) -> None:
    """
    Configure root logger with rotating file handler and console handler.
    Safe to call multiple times (reconfigures).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    levelno = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(levelno)

    fmt = fmt or "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"
    formatter = logging.Formatter(fmt)

    fh = logging.handlers.RotatingFileHandler(
        log_dir / filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    fh.setLevel(levelno)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(levelno)

    root.addHandler(fh)
    root.addHandler(ch)


LOG = logging.getLogger("ritsu.main")


# ---------------------------
# Config Model & Manager
# ---------------------------

if BaseModel is not object:

    class AppConfig(BaseModel):
        """
        Pydantic model for application configuration.
        Add or extend fields as project modules grow.
        Environment variables may override fields via ConfigManager.
        """

        app_name: str = Field("Ritsu", description="Application name")
        env: str = Field("dev", description="Environment: dev | staging | prod")
        safe_mode: bool = Field(False, description="Restrict risky features")
        restart_on_crash: bool = Field(True, description="Auto restart loops on crash")

        logging: dict = Field(default_factory=lambda: {"level": "INFO", "dir": "data/logs", "json": False})
        http: dict = Field(default_factory=lambda: {"host": "127.0.0.1", "port": 8765, "metrics": True})
        monitoring: dict = Field(default_factory=lambda: {"interval_sec": 30})
        memory: dict = Field(default_factory=dict)
        llm: dict = Field(default_factory=dict)
        input: dict = Field(default_factory=dict)
        output: dict = Field(default_factory=dict)
        plugins: dict = Field(default_factory=dict)
        hot_reload: dict = Field(default_factory=lambda: {"enabled": True, "watch_paths": ["system/config.yaml"], "debounce_sec": 1.0})

        @validator("logging")
        def ensure_logging(cls, v):
            if not isinstance(v, dict):
                raise ValueError("logging must be a dict")
            v.setdefault("dir", "data/logs")
            v.setdefault("level", "INFO")
            return v

        class Config:
            extra = "allow"

else:
    AppConfig = dict  # type: ignore


class ConfigManager:
    """
    Loads YAML config with environment variable overrides.
    Supports hot-reload via file watching.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else Path("system/config.yaml")
        self._raw: dict = {}
        self._model: AppConfig | dict | None = None

    def load(self) -> AppConfig | dict:
        """Load config from YAML and apply env overrides and defaults."""
        if yaml is None:
            raise RuntimeError("PyYAML (yaml) is required to load config.yaml")

        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        # merge with env vars: RITSU__SECTION__KEY style
        for k, v in os.environ.items():
            if not k.startswith("RITSU__"):
                continue
            _, *parts = k.split("__")
            cur = data
            for p in parts[:-1]:
                cur = cur.setdefault(p.lower(), {})
            cur[parts[-1].lower()] = _coerce_env_value(v)

        self._raw = data

        if BaseModel is not object:
            self._model = AppConfig(**data)
            return self._model
        else:
            self._model = data
            return data


def _coerce_env_value(v: str) -> Any:
    """Coerce env var strings to bool/int/float where reasonable."""
    lower = v.lower()
    if lower in {"true", "yes", "1"}:
        return True
    if lower in {"false", "no", "0"}:
        return False
    # ints
    try:
        if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            return int(v)
    except Exception:
        pass
    # float
    try:
        return float(v)
    except Exception:
        pass
    return v


# ---------------------------
# AppContext & Lifecycle
# ---------------------------


@dataclass
class AppContext:
    """
    Central dependency container for components and runtime state.
    Components should be assigned (llm, memory, planner, executor, input_manager, output_manager...).
    """

    config: AppConfig | dict
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=lambda: {"uptime": time.time(), "events_processed": 0})
    # components (optional)
    llm: Any | None = None
    memory: Any | None = None
    planner: Any | None = None
    executor: Any | None = None
    input_manager: Any | None = None
    output_manager: Any | None = None
    performance_monitor: Any | None = None
    security_manager: Any | None = None
    event_manager: Any | None = None
    # control
    degraded: bool = False


# ---------------------------
# Safe component helpers
# ---------------------------


def safe_call_sync(obj: Any, method: str, *args, **kwargs):
    """
    Call obj.method(*args, **kwargs) safely for sync methods.
    """
    if obj is None:
        return None
    fn = getattr(obj, method, None)
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        LOG.exception("Exception calling %s.%s", obj.__class__.__name__, method)
        return None


async def safe_call_async(obj: Any, method: str, *args, **kwargs):
    """
    Call obj.method(*args, **kwargs) safely for sync or async methods.
    """
    if obj is None:
        return None
    fn = getattr(obj, method, None)
    if fn is None:
        return None
    try:
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            return await res
        return res
    except Exception:
        LOG.exception("Async exception calling %s.%s", obj.__class__.__name__, method)
        return None


# ---------------------------
# System requirement checks
# ---------------------------


def check_python_version(min_major: int = 3, min_minor: int = 11) -> None:
    if sys.version_info < (min_major, min_minor):
        raise RuntimeError(f"Python {min_major}.{min_minor}+ is required, found {platform.python_version()}")


def check_disk_space(path: Path | str = ".", required_mb: int = 200) -> None:
    total, used, free = shutil.disk_usage(path)
    free_mb = free // (1024 * 1024)
    if free_mb < required_mb:
        raise RuntimeError(f"Insufficient disk space at {path}: {free_mb} MB free (required {required_mb} MB)")


# ---------------------------
# Health & Metrics HTTP Server
# ---------------------------


class HealthServer:
    """
    Lightweight aiohttp-based health and metrics server.
    Exposes:
      GET /health -> {status: "ok"/"degraded"/"down", components: {...}}
      GET /metrics -> JSON metrics
      POST /shutdown -> accept shutdown (optional secret in config)
    """

    def __init__(self, ctx: AppContext):
        if web is None:
            raise RuntimeError("aiohttp required for HealthServer")
        self.ctx = ctx
        self._app = web.Application()
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._secret = None
        self._routes_setup()

    def _routes_setup(self):
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/metrics", self._handle_metrics)
        self._app.router.add_post("/shutdown", self._handle_shutdown)

    async def _handle_health(self, request: web.Request) -> web.Response:
        comp_status: Dict[str, Any] = {}
        # perform quick healthchecks
        components = {
            "llm": self.ctx.llm,
            "memory": self.ctx.memory,
            "planner": self.ctx.planner,
            "executor": self.ctx.executor,
            "performance_monitor": self.ctx.performance_monitor,
        }
        overall = "ok"
        for name, comp in components.items():
            if comp is None:
                comp_status[name] = {"status": "missing"}
                overall = "degraded"
                continue
            try:
                res = await safe_call_async(comp, "healthcheck")
                if isinstance(res, dict):
                    comp_status[name] = {"status": "ok", "detail": res}
                elif res is True or res is None:
                    comp_status[name] = {"status": "ok"}
                else:
                    comp_status[name] = {"status": "degraded", "detail": res}
                    overall = "degraded"
            except Exception:
                comp_status[name] = {"status": "down"}
                overall = "degraded"
        payload = {"status": overall, "components": comp_status, "uptime": time.time() - self.ctx.metrics["uptime"]}
        return web.json_response(payload)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        return web.json_response(self.ctx.metrics)

    async def _handle_shutdown(self, request: web.Request) -> web.Response:
        data = await request.json() if request.can_read_body else {}
        secret = data.get("secret")
        # optional secret check
        if self._secret and secret != self._secret:
            raise web.HTTPUnauthorized()
        # trigger shutdown
        LOG.info("HealthServer requested shutdown via HTTP")
        self.ctx.shutdown_event.set()
        return web.json_response({"ok": True, "reason": "shutdown requested"})

    async def startup(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host=host, port=port)
        await self._site.start()
        LOG.info("Health server started on %s:%s", host, port)

    async def shutdown(self) -> None:
        if self._runner:
            with contextlib.suppress(Exception):
                await self._runner.cleanup()
                LOG.info("Health server stopped")


# ---------------------------
# Hot-reload watcher (simple)
# ---------------------------


class HotReloader:
    """
    Watch configured files for changes and call a callback.
    simple asyncio-based polling watcher (low dependency).
    """

    def __init__(self, files: Iterable[Path], callback: Callable[[], Awaitable[None]], debounce: float = 1.0):
        self.files = [Path(p) for p in files]
        self.callback = callback
        self.debounce = float(debounce)
        self._last_mtimes: Dict[Path, float] = {}
        self._task: Optional[asyncio.Task] = None
        for p in self.files:
            try:
                self._last_mtimes[p] = p.stat().st_mtime
            except Exception:
                self._last_mtimes[p] = 0.0

    async def _loop(self, shutdown_event: asyncio.Event):
        while not shutdown_event.is_set():
            for p in self.files:
                try:
                    m = p.stat().st_mtime
                except Exception:
                    m = 0.0
                if m != self._last_mtimes.get(p, 0.0):
                    LOG.info("Hot-reload detected change in %s", p)
                    self._last_mtimes[p] = m
                    try:
                        await self.callback()
                    except Exception:
                        LOG.exception("Error running hot-reload callback")
                    await asyncio.sleep(self.debounce)
                    # break out early to avoid multiple reloads back-to-back
                    break
            await asyncio.sleep(0.5)

    def start(self, shutdown_event: asyncio.Event) -> asyncio.Task:
        self._task = asyncio.create_task(self._loop(shutdown_event), name="hot_reloader")
        return self._task


# ---------------------------
# Watchdog restart helper
# ---------------------------


async def watchdog_loop(
    factory: Callable[[], Awaitable[None]],
    shutdown_event: asyncio.Event,
    *,
    name: str = "watchdog_task",
    restart: bool = True,
    backoff: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
):
    """
    Run factory() continuously, restart on unhandled exceptions using backoff.
    factory is an async callable that runs until it returns or raises.
    """
    attempt = 0
    while not shutdown_event.is_set():
        try:
            LOG.info("Starting task %s (attempt %d)", name, attempt + 1)
            await factory()
            LOG.info("Task %s completed normally", name)
            return
        except asyncio.CancelledError:
            LOG.info("Task %s was cancelled", name)
            return
        except Exception:
            LOG.exception("Task %s crashed", name)
            if not restart:
                LOG.warning("Not configured to restart task %s", name)
                return
            delay = backoff[min(attempt, len(backoff) - 1)]
            LOG.info("Restarting %s after %.1fs", name, delay)
            attempt += 1
            # allow shutdown to short-circuit sleep
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                continue


# ---------------------------
# Bootstrap logic
# ---------------------------


async def initialize_components(ctx: AppContext) -> None:
    """
    Initialize & start components that implement startup/healthcheck APIs.
    This function is intentionally resilient: failures are logged and components
    may be left as None (degraded mode).
    """
    # Example pattern: try to import and initialize components from project modules
    # (these imports are project-specific and optional)
    # Use a safe_init pattern: try constructors with config injection, otherwise fallback
    def safe_init_module(path: str, **kwargs):
        try:
            # dynamic import
            mod_name, _, attr = path.partition(":")
            module = __import__(mod_name, fromlist=["*"])
            cls = getattr(module, attr or "Component", None) or getattr(module, "Component", None)
            if cls is None:
                return None
            try:
                return cls(config=ctx.config, **kwargs)
            except TypeError:
                return cls(**kwargs)
        except Exception:
            LOG.exception("Failed to import/init %s", path)
            return None

    # Examples (user should adapt to real module names in project)
    # Try to load LLM adapter
    if ctx.config and isinstance(ctx.config, dict):
        llm_cfg = ctx.config.get("llm", {})
    else:
        llm_cfg = getattr(ctx.config, "llm", {}) if hasattr(ctx.config, "llm") else {}

    # Try multiple known adapter paths (project-specific)
    for candidate in ("llm.ritsu_llm:RitsuLLM", "llm.adapter:LLMAdapter"):
        comp = safe_init_module(candidate, config=llm_cfg)
        if comp:
            ctx.llm = comp
            LOG.info("LLM adapter %s initialized", candidate)
            break

    # Memory manager
    for candidate in ("ai.memory_manager:MemoryManager", "memory.vector_db:VectorMemory"):
        comp = safe_init_module(candidate, config=getattr(ctx.config, "memory", {}) if not isinstance(ctx.config, dict) else ctx.config.get("memory", {}))
        if comp:
            ctx.memory = comp
            LOG.info("Memory manager %s initialized", candidate)
            break

    # Planner & Executor
    for candidate in ("core.planner:Planner", "core.planner_manager:PlannerManager"):
        comp = safe_init_module(candidate, config={})
        if comp:
            ctx.planner = comp
            LOG.info("Planner %s initialized", candidate)
            break

    for candidate in ("core.executor:Executor", "core.simple_executor:Executor"):
        comp = safe_init_module(candidate, config={})
        if comp:
            ctx.executor = comp
            LOG.info("Executor %s initialized", candidate)
            break

    # Input & Output managers
    for candidate in ("input.input_manager:InputManager",):
        comp = safe_init_module(candidate, config=getattr(ctx.config, "input", {}) if not isinstance(ctx.config, dict) else ctx.config.get("input", {}))
        if comp:
            ctx.input_manager = comp
            LOG.info("InputManager %s initialized", candidate)
            break

    for candidate in ("output.output_manager:OutputManager",):
        comp = safe_init_module(candidate, config=getattr(ctx.config, "output", {}) if not isinstance(ctx.config, dict) else ctx.config.get("output", {}))
        if comp:
            ctx.output_manager = comp
            LOG.info("OutputManager %s initialized", candidate)
            break

    # Try calling async startup hooks where available
    for name in ("llm", "memory", "planner", "executor", "input_manager", "output_manager"):
        comp = getattr(ctx, name, None)
        if comp is None:
            LOG.debug("Component %s missing", name)
            continue
        try:
            # prefer startup()
            res = safe_call_sync(comp, "startup")
            if asyncio.iscoroutine(res):
                await res
            else:
                # maybe comp.start() async
                maybe = getattr(comp, "start", None)
                if maybe:
                    res2 = maybe()
                    if asyncio.iscoroutine(res2):
                        await res2
        except Exception:
            LOG.exception("Failed starting %s; component will be left as-is", name)
            # degraded mode but continue


# ---------------------------
# Background loops (examples)
# ---------------------------


async def monitoring_loop(ctx: AppContext) -> None:
    """
    Periodic performance and health monitoring; pushes alerts/events to event_queue.
    """
    interval = 30
    try:
        interval = int(ctx.config.get("monitoring", {}).get("interval_sec", 30)) if isinstance(ctx.config, dict) else getattr(ctx.config, "monitoring", {}).get("interval_sec", 30)
    except Exception:
        pass

    while not ctx.shutdown_event.is_set():
        try:
            # collect health from components that support it
            if ctx.performance_monitor:
                await safe_call_async(ctx.performance_monitor, "collect_metrics")
            # simple LLM ping
            if ctx.llm:
                try:
                    _ = await safe_call_async(ctx.llm, "healthcheck")
                except Exception:
                    LOG.warning("LLM healthcheck failed; marking degraded")
                    ctx.degraded = True
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        except Exception:
            LOG.exception("Monitoring loop error")
            await asyncio.sleep(5)


async def maintenance_loop(ctx: AppContext) -> None:
    """Periodic maintenance: cleanup, DB compaction, etc."""
    interval = 3600
    try:
        interval = int(ctx.config.get("maintenance", {}).get("interval_sec", 3600)) if isinstance(ctx.config, dict) else getattr(ctx.config, "maintenance", {}).get("interval_sec", 3600)
    except Exception:
        pass

    while not ctx.shutdown_event.is_set():
        try:
            if ctx.memory and hasattr(ctx.memory, "maintenance"):
                await safe_call_async(ctx.memory, "maintenance")
            if ctx.executor and hasattr(ctx.executor, "prune"):
                await safe_call_async(ctx.executor, "prune")
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        except Exception:
            LOG.exception("Maintenance loop crashed")
            await asyncio.sleep(5)


async def input_loop(ctx: AppContext) -> None:
    """Drive input manager to produce events onto ctx.event_queue"""
    if not ctx.input_manager:
        LOG.warning("No InputManager configured; input loop disabled")
        return
    if hasattr(ctx.input_manager, "run"):
        try:
            res = ctx.input_manager.run(queue=ctx.event_queue, shutdown_event=ctx.shutdown_event)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            LOG.exception("Input loop failed; shutting down input loop")
            raise
    else:
        LOG.warning("InputManager has no run() method; input loop disabled")


async def core_loop(ctx: AppContext) -> None:
    """
    Core loop: consume events from queue, plan, execute, persist results, emit outputs.
    Expects planner and executor to implement decide/execute semantics.
    """
    if not (ctx.planner and ctx.executor):
        LOG.error("Planner or Executor missing; core loop cannot run")
        return

    while not ctx.shutdown_event.is_set():
        try:
            event = await ctx.event_queue.get()
            ctx.metrics["events_processed"] = ctx.metrics.get("events_processed", 0) + 1
            try:
                plan = await safe_call_async(ctx.planner, "decide", event)
                if not plan:
                    LOG.warning("Planner returned no plan for event: %s", event)
                    continue
                result = await safe_call_async(ctx.executor, "execute", plan, context={"event": event})
                # persist memory
                if ctx.memory:
                    await safe_call_async(ctx.memory, "save_event", {"event": event, "plan": plan, "result": result})
                # emit output
                if ctx.output_manager:
                    await safe_call_async(ctx.output_manager, "emit", result)
            except Exception:
                LOG.exception("Error processing event")
            finally:
                ctx.event_queue.task_done()
        except asyncio.CancelledError:
            return
        except Exception:
            LOG.exception("Core loop top-level failure; sleeping then retrying")
            await asyncio.sleep(1)


# ---------------------------
# Startup & Shutdown orchestration
# ---------------------------


async def startup_sequence(ctx: AppContext) -> None:
    """
    Run system checks, initialize components, start servers and background loops.
    """
    LOG.info("Running startup sequence")

    # 1) system requirements
    try:
        check_python_version(3, 11)
        check_disk_space(".", required_mb=100)
    except Exception as e:
        LOG.exception("System requirement check failed: %s", e)
        # Critical failure: exit
        raise

    # 2) init components
    try:
        await initialize_components(ctx)
    except Exception:
        LOG.exception("Component initialization failure (non-fatal)")
        # allow degraded but continue

    # 3) health server
    health_server: Optional[HealthServer] = None
    if web is not None:
        http_cfg = ctx.config.get("http", {}) if isinstance(ctx.config, dict) else getattr(ctx.config, "http", {})
        try:
            host = http_cfg.get("host", "127.0.0.1")
            port = int(http_cfg.get("port", 8765))
        except Exception:
            host, port = "127.0.0.1", 8765
        try:
            health_server = HealthServer(ctx)
            await health_server.startup(host=host, port=port)
            ctx.tasks["health_server"] = health_server
        except Exception:
            LOG.exception("Failed to start health server; continuing without it")
            health_server = None

    # 4) start background loops via watchdogs
    ctx.tasks["core_loop"] = asyncio.create_task(watchdog_loop(lambda: core_loop(ctx), ctx.shutdown_event, name="core_loop", restart=False))
    ctx.tasks["input_loop"] = asyncio.create_task(watchdog_loop(lambda: input_loop(ctx), ctx.shutdown_event, name="input_loop", restart=True))
    ctx.tasks["monitoring"] = asyncio.create_task(watchdog_loop(lambda: monitoring_loop(ctx), ctx.shutdown_event, name="monitoring", restart=True))
    ctx.tasks["maintenance"] = asyncio.create_task(watchdog_loop(lambda: maintenance_loop(ctx), ctx.shutdown_event, name="maintenance", restart=True))

    # 5) hot reload (non-blocking)
    try:
        hr_cfg = ctx.config.get("hot_reload", {}) if isinstance(ctx.config, dict) else getattr(ctx.config, "hot_reload", {})
        if hr_cfg.get("enabled", True):
            watch_paths = hr_cfg.get("watch_paths", ["system/config.yaml"])
            debounce = float(hr_cfg.get("debounce_sec", 1.0))
            async def _reload_callback():
                LOG.info("Hot-reload callback: reloading config")
                # reload config & update components that support update_config()
                cm = ConfigManager(Path("system/config.yaml"))
                new_cfg = cm.load()
                ctx.config = new_cfg
                for name in ("llm", "memory", "planner", "executor", "input_manager", "output_manager"):
                    comp = getattr(ctx, name, None)
                    if comp and hasattr(comp, "update_config"):
                        try:
                            await safe_call_async(comp, "update_config", ctx.config)
                        except Exception:
                            LOG.exception("Failed to update_config on %s", name)
                LOG.info("Hot-reload complete")

            hr = HotReloader([Path(p) for p in watch_paths], _reload_callback, debounce=debounce)
            ctx.tasks["hot_reloader"] = hr.start(ctx.shutdown_event)
    except Exception:
        LOG.exception("Failed to start hot-reloader; continuing")

    LOG.info("Startup sequence complete")


async def shutdown_sequence(ctx: AppContext, *, timeout: float = 10.0) -> None:
    """
    Gracefully shutdown all tasks and components. Attempt to call common lifecycle hooks.
    """
    LOG.info("Shutdown sequence started")
    ctx.shutdown_event.set()

    # cancel all tasks we've created
    for name, task in list(ctx.tasks.items()):
        try:
            if isinstance(task, asyncio.Task):
                task.cancel()
        except Exception:
            LOG.exception("Error cancelling task %s", name)

    # wait for tasks to finish
    tasks_to_wait = [t for t in ctx.tasks.values() if isinstance(t, asyncio.Task)]
    if tasks_to_wait:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks_to_wait, return_exceptions=True), timeout=timeout)
        except Exception:
            LOG.warning("Timeout waiting for tasks to finish")

    # try component-level shutdown
    for name in ("input_manager", "event_manager", "planner", "executor", "output_manager", "memory", "llm"):
        comp = getattr(ctx, name, None)
        if comp is None:
            continue
        try:
            # prefer shutdown()
            res = safe_call_sync(comp, "shutdown")
            if asyncio.iscoroutine(res):
                await res
            else:
                maybe = getattr(comp, "close", None)
                if maybe:
                    r2 = maybe()
                    if asyncio.iscoroutine(r2):
                        await r2
        except Exception:
            LOG.exception("Exception during shutdown of component %s", name)

    # if health server present, call its shutdown
    hs = ctx.tasks.get("health_server")
    if isinstance(hs, HealthServer):
        try:
            await hs.shutdown()
        except Exception:
            LOG.exception("HealthServer shutdown failed")

    LOG.info("Shutdown complete")


# ---------------------------
# Service integration helpers
# ---------------------------


def is_running_under_systemd() -> bool:
    return bool(os.environ.get("NOTIFY_SOCKET")) or "SYSTEMD_EXEC_PID" in os.environ


def sd_notify_ready() -> None:
    if sdnotify is None:
        return
    try:
        n = sdnotify.SystemdNotifier()
        n.notify("READY=1")
    except Exception:
        LOG.debug("sdnotify failed")


# -------------
# CLI / Runner
# -------------


def parse_args(argv: Sequence[str]) -> dict:
    import argparse

    p = argparse.ArgumentParser("ritsu", description="Ritsu autonomous assistant")
    p.add_argument("--config", "-c", type=str, default="system/config.yaml", help="Path to config.yaml")
    p.add_argument("--log-level", "-l", type=str, default=None, help="Override log level")
    p.add_argument("--service-systemd", action="store_true", help="Run as systemd service (optional)")
    p.add_argument("--service-windows", action="store_true", help="Run as Windows service (optional)")
    p.add_argument("--no-hot-reload", action="store_true", help="Disable hot-reload")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    return vars(p.parse_args(list(argv)))


async def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    # Load config
    cm = ConfigManager(args.get("config"))
    cfg = cm.load()

    # Configure logging
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else getattr(cfg, "logging", {})
    log_dir = Path(log_cfg.get("dir", "data/logs"))
    setup_logging(level=args.get("log_level") or log_cfg.get("level", "INFO"), log_dir=log_dir)

    LOG.info("Starting Ritsu (pid=%d) environment=%s", os.getpid(), cfg.get("env") if isinstance(cfg, dict) else getattr(cfg, "env", "dev"))

    # AppContext
    ctx = AppContext(config=cfg)

    # disable hot reload if CLI says so
    if args.get("no_hot_reload", False):
        if isinstance(ctx.config, dict):
            ctx.config.setdefault("hot_reload", {})["enabled"] = False
        else:
            setattr(ctx.config, "hot_reload", getattr(ctx.config, "hot_reload", {}))
            ctx.config.hot_reload["enabled"] = False

    # Register signal handlers
    loop = asyncio.get_running_loop()

    def _sig_handler(signum, frame):
        LOG.info("Signal %s received: scheduling shutdown", signum)
        ctx.shutdown_event.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, functools.partial(_sig_handler, s), None)
        except NotImplementedError:
            # e.g., on Windows asyncio loop may not support add_signal_handler
            signal.signal(s, _sig_handler)

    # Start up sequence
    try:
        await startup_sequence(ctx)
    except Exception:
        LOG.exception("Startup failed; shutting down")
        await shutdown_sequence(ctx)
        return 1

    # Notify systemd if present
    if is_running_under_systemd():
        sd_notify_ready()

    # Main wait loop: wait until shutdown_event is set
    try:
        await ctx.shutdown_event.wait()
        LOG.info("Shutdown flag observed; proceeding to shutdown_sequence")
        await shutdown_sequence(ctx)
    except Exception:
        LOG.exception("Exception while running main loop")
        await shutdown_sequence(ctx)
        return 1
    return 0


def main() -> None:
    try:
        exit_code = asyncio.run(run(sys.argv[1:]))
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        LOG.exception("Unhandled exception at top-level")
        exit_code = 1
    finally:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()


# ---------------------------
# Example config.yaml template
# ---------------------------
#
# Save this as system/config.yaml and edit as needed.
#
# example_config_yaml = """
# app_name: Ritsu
# env: dev
# safe_mode: false
# restart_on_crash: true
#
# logging:
#   level: INFO
#   dir: data/logs
#   json: false
#
# http:
#   host: 127.0.0.1
#   port: 8765
#   metrics: true
#
# monitoring:
#   interval_sec: 30
#
# memory:
#   type: vectordb
#   path: data/memory
#
# llm:
#   provider: local
#   model: ritsu-small
#   max_tokens: 1024
#
# input:
#   mic: false
#   chat_listener: true
#
# output:
#   tts: false
#   avatar: false
#
# hot_reload:
#   enabled: true
#   watch_paths:
#     - system/config.yaml
#   debounce_sec: 1.0
#
# plugins:
#   enabled: []
# """
#
