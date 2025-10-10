from __future__ import annotations
"""
command_parser.py
---------------------------------
Central command parser for Ritsu.
- Normalizes raw input (text, API calls, system events).
- Tokenizes and parses into structured commands.
- Validates arguments and flags.
- Supports aliases, subcommands, and dynamic command registration.
- Separates system/meta commands from natural language.
- Sends parsed command objects to event_manager or executor.
"""

import re
import shlex
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field


# -----------------------------
# Data Structures
# -----------------------------
@dataclass
class Command:
    """Represents a fully parsed command"""
    name: str
    args: List[str] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""                # Original input for logging/debugging
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandSpec:
    """Specifies how a command should be parsed"""
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    usage: str = ""
    handler: Optional[Callable[[Command], Any]] = None
    allow_unknown_flags: bool = False
    subcommands: Dict[str, "CommandSpec"] = field(default_factory=dict)
    flags: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Example flags: {"--verbose": {"type": bool, "default": False, "help": "Enable debug"}}


@dataclass
class CommandResult:
    """
    Standardized representation of parsed input.
    This object is returned to input_manager.py and routed forward.
    """
    raw_text: str                # original input string
    type: str                   # "system", "meta", "natural", "empty"
    command: Optional[str] = None  # parsed command keyword (e.g. "restart", "config")
    args: Optional[List[str]] = None  # arguments split from command
    flags: Optional[Dict[str, Any]] = None  # parsed flags if any
    metadata: Optional[Dict[str, Any]] = None  # any extra info (confidence, source, etc.)


# -----------------------------
# Command Registry
# -----------------------------
class CommandRegistry:
    """Holds all known command specs"""
    def __init__(self):
        self.commands: Dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec):
        """Register a command and its aliases"""
        self.commands[spec.name] = spec
        for alias in spec.aliases:
            self.commands[alias] = spec

    def get(self, name: str) -> Optional[CommandSpec]:
        return self.commands.get(name)


# -----------------------------
# Main Command Parser
# -----------------------------
class CommandParser:
    """
    CommandParser separates system-level commands from natural text input.
    It supports:
      - System commands with prefixes (/ ! >)
      - Meta commands with prefixes (# @)
      - Natural language fallback
      - Spec-aware parsing with flags and subcommands
    """

    def __init__(self, registry: Optional[CommandRegistry] = None):
        self.registry = registry or CommandRegistry()

        # Recognized prefixes for commands
        self.system_prefixes = ["/", "!", ">"]  # shell or admin commands
        self.meta_prefixes = ["#", "@"]          # config, mode switching

        # Regex patterns for system and meta commands
        self.patterns = {
            "system": re.compile(r"^(?:/|!|>)(\w+)(?:\s+(.*))?$"),
            "meta": re.compile(r"^(?:#|@)(\w+)(?:\s+(.*))?$"),
        }

        # Predefined supported commands (can be extended or loaded dynamically)
        self.supported_system_commands = [
            "restart", "shutdown", "status", "exec", "clear",
            "reload", "memory", "config", "help"
        ]
        self.supported_meta_commands = [
            "mode", "persona", "voice", "lang", "debug"
        ]

    def parse(self, text: str, source: str = "chat") -> CommandResult:
        """
        Main entrypoint: parse text and classify.

        Args:
            text: user input string
            source: source of input (mic, chat, file, etc.)

        Returns:
            CommandResult object with parsed info.
        """
        raw = text.strip()
        if not raw:
            return CommandResult(raw_text=text, type="empty")

        # 1. Try system commands (/restart, !status, >exec ...)
        match = self.patterns["system"].match(raw)
        if match:
            cmd, args = match.groups()
            cmd = cmd.lower()
            if cmd in self.supported_system_commands:
                args_list, flags = self._parse_args_and_flags(args)
                return CommandResult(
                    raw_text=text,
                    type="system",
                    command=cmd,
                    args=args_list,
                    flags=flags,
                    metadata={"source": source}
                )

        # 2. Try meta commands (#mode chatty, @voice male ...)
        match = self.patterns["meta"].match(raw)
        if match:
            cmd, args = match.groups()
            cmd = cmd.lower()
            if cmd in self.supported_meta_commands:
                args_list, flags = self._parse_args_and_flags(args)
                return CommandResult(
                    raw_text=text,
                    type="meta",
                    command=cmd,
                    args=args_list,
                    flags=flags,
                    metadata={"source": source}
                )

        # 3. Otherwise → treat as natural input
        return CommandResult(
            raw_text=text,
            type="natural",
            metadata={"source": source, "tokens": raw.split()}
        )

    def _parse_args_and_flags(self, arg_str: Optional[str]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Parses arguments and flags from a string.

        Supports:
          - Flags: --flag or --flag=value
          - Positional args

        Returns:
            Tuple of (args list, flags dict)
        """
        if not arg_str:
            return [], {}

        try:
            tokens = shlex.split(arg_str)
        except ValueError:
            tokens = arg_str.split()

        args = []
        flags = {}

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("--"):
                if "=" in token:
                    flag, val = token.split("=", 1)
                    flags[flag] = self._cast_value(val)
                    i += 1
                else:
                    # Check if next token is a value or another flag
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                        flags[token] = self._cast_value(tokens[i + 1])
                        i += 2
                    else:
                        flags[token] = True
                        i += 1
            elif token.startswith("-") and len(token) > 1:
                # Short flags like -v or -abc (multiple flags combined)
                for ch in token[1:]:
                    flags[f"-{ch}"] = True
                i += 1
            else:
                args.append(token)
                i += 1

        return args, flags

    def _cast_value(self, val: str) -> Union[str, int, float, bool]:
        """
        Attempts to cast a string value to int, float, or bool if applicable.
        Falls back to string.
        """
        val_lower = val.lower()
        if val_lower in ("true", "yes", "on"):
            return True
        if val_lower in ("false", "no", "off"):
            return False
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return val

    def is_system_command(self, text: str) -> bool:
        return bool(self.patterns["system"].match(text.strip()))

    def is_meta_command(self, text: str) -> bool:
        return bool(self.patterns["meta"].match(text.strip()))

    def suggest_help(self) -> Dict[str, List[str]]:
        """Return available commands for help menus."""
        return {
            "system": self.supported_system_commands,
            "meta": self.supported_meta_commands
        }
