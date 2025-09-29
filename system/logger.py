from __future__ import annotations

"""
system/logger.py

Logging + debugging utilities
- Structured JSON logging
- Multiple output formats
- Log rotation and management
- Performance tracking
"""

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if they exist
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class PerformanceFilter(logging.Filter):
    """Filter to add performance timing information."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add process time to all records
        record.process_time = time.process_time()
        return True


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    json: bool = True,
    console: bool = True,
    file_rotation: bool = True,
) -> None:
    """Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None = no file logging)
        json: Use JSON formatting instead of plain text
        console: Enable console output
        file_rotation: Enable log file rotation
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(numeric_level)
    
    # Setup formatters
    if json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(PerformanceFilter())
        root_logger.addHandler(console_handler)
    
    # File handlers
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        if file_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "ritsu.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
        else:
            file_handler = logging.FileHandler(
                log_dir / "ritsu.log",
                encoding="utf-8"
            )
        
        file_handler.setFormatter(formatter)
        file_handler.addFilter(PerformanceFilter())
        root_logger.addHandler(file_handler)
        
        # Error-only log file
        error_handler = logging.FileHandler(
            log_dir / "errors.log",
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Custom adapter that supports structured extra data."""
    
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        # Extract 'extra' dict and store it as 'extra_data' for the formatter
        extra = kwargs.get("extra", {})
        if extra:
            # Create a new LogRecord attribute for our JSON formatter
            kwargs.setdefault("extra", {})["extra_data"] = extra
        
        return msg, kwargs


def get_structured_logger(name: str, **context) -> LoggerAdapter:
    """Get a structured logger that automatically includes context.
    
    Args:
        name: Logger name
        **context: Additional context to include in all log messages
        
    Returns:
        Logger adapter with structured logging support
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context)


class PerformanceTracker:
    """Context manager for tracking operation performance."""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> "PerformanceTracker":
        self.start_time = time.perf_counter()
        self.logger.debug(
            f"Starting {self.operation}",
            extra={"operation": self.operation, "phase": "start", **self.context}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            
            if exc_type is None:
                self.logger.info(
                    f"Completed {self.operation}",
                    extra={
                        "operation": self.operation,
                        "phase": "complete",
                        "duration_ms": round(duration * 1000, 2),
                        **self.context
                    }
                )
            else:
                self.logger.error(
                    f"Failed {self.operation}",
                    extra={
                        "operation": self.operation,
                        "phase": "error",
                        "duration_ms": round(duration * 1000, 2),
                        "error_type": exc_type.__name__ if exc_type else None,
                        **self.context
                    }
                )


def track_performance(logger: logging.Logger, operation: str, **context) -> PerformanceTracker:
    """Create a performance tracking context manager.
    
    Usage:
        with track_performance(log, "database_query", query_id="123"):
            # ... do work ...
            pass
    """
    return PerformanceTracker(logger, operation, **context)