from __future__ import annotations

"""
system/config_manager.py

ConfigManager — loads configs
- Supports YAML, JSON, and TOML formats
- Environment variable overrides
- Default fallback configurations
- Hot-reload capability (optional)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import tomli
    HAS_TOML = True
except ImportError:
    HAS_TOML = False


class ConfigManager:
    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else Path("system/config.yaml")
        self._config: Dict[str, Any] = self._get_default_config()
        self._env_prefix = "RITSU_"
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration structure."""
        return {
            "app": {
                "name": "Ritsu",
                "env": "dev",
                "safe_mode": False,
                "restart_on_crash": True,
            },
            "logging": {
                "level": "INFO",
                "dir": "data/logs",
                "json": True,
            },
            "events": {
                "queue_size": 1000,
                "batch_size": 10,
            },
            "input": {
                "enable_mic": False,
                "enable_chat": True,
                "enable_ui": False,
                "enable_api": False,
            },
            "output": {
                "enable_tts": False,
                "enable_avatar": False,
                "enable_stream": False,
            },
            "llm": {
                "provider": "ollama",
                "model": "llama3.2",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            "ai": {
                "memory_limit": 1000,
                "knowledge_base_path": "data/knowledge_base.json",
            },
            "rust_editor": {
                "enabled": False,
                "gpu": False,
            },
            "ui": {
                "enabled": False,
                "port": 8080,
            },
        }
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file with fallback to defaults."""
        if self._path.exists():
            try:
                content = self._path.read_text(encoding="utf-8")
                
                if self._path.suffix.lower() in (".yaml", ".yml") and HAS_YAML:
                    file_config = yaml.safe_load(content)
                elif self._path.suffix.lower() == ".toml" and HAS_TOML:
                    file_config = tomli.loads(content)
                elif self._path.suffix.lower() == ".json":
                    file_config = json.loads(content)
                else:
                    # Try JSON as fallback
                    file_config = json.loads(content)
                
                # Deep merge with defaults
                self._config = self._deep_merge(self._config, file_config)
            except Exception as e:
                print(f"Warning: Failed to load config from {self._path}: {e}")
                print("Using default configuration")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        return self._config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Simple flat override for common settings
        env_mappings = {
            f"{self._env_prefix}LOG_LEVEL": ("logging", "level"),
            f"{self._env_prefix}LOG_DIR": ("logging", "dir"),
            f"{self._env_prefix}LLM_MODEL": ("llm", "model"),
            f"{self._env_prefix}LLM_BASE_URL": ("llm", "base_url"),
            f"{self._env_prefix}SAFE_MODE": ("app", "safe_mode"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert boolean strings
                if value.lower() in ("true", "1", "yes", "on"):
                    value = True
                elif value.lower() in ("false", "0", "no", "off"):
                    value = False
                
                if section in self._config:
                    self._config[section][key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self._path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix.lower() in (".yaml", ".yml") and HAS_YAML:
            content = yaml.dump(self._config, default_flow_style=False, sort_keys=True)
        else:
            content = json.dumps(self._config, indent=2, sort_keys=True)
        
        save_path.write_text(content, encoding="utf-8")

import tempfile
import os
import json
import logging
from pathlib import Path

# Optional dependency
try:
    import yaml
except ImportError:
    yaml = None

# === PATHS ===
APP_DIR = Path.home() / ".ritsu"
APP_DIR.mkdir(exist_ok=True)
LOG_FILE = APP_DIR / "audit.log"
CONFIG_FILE = APP_DIR / "config.json"
POLICY_FILE = APP_DIR / "policy.yaml"
SNIPPETS_FILE = APP_DIR / "code_snippets.json"
MEMORY_FILE = APP_DIR / "ritsu_memory.json"
SHORT_TERM_FILE = APP_DIR / "short_term_mem.json"

# === LOGGING ===
logging.basicConfig(level=logging.INFO, filename=str(LOG_FILE), filemode="a",
                    format="%(asctime)s | %(levelname)s | %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# === DEFAULTS ===
DEFAULT_CONFIG = {
    "mic_enabled": False,
    "tts_enabled": True,
    "auto_kill": False,
    "safe_delete_dirs": [tempfile.gettempdir()],
    "http_api_port": 8765,
    "llm": {
        "provider": "local_ollama",
        "model": "llama3.1:8b-q5_K_M",
        "openai_api_key": None
    }
}

DEFAULT_POLICY_YAML = f"""
version: 1
auto_allow:
  - notify
  - collect_metrics
ask_to_run:
  - kill_process
  - close_window
  - clear_temp
forbid:
  - delete_file_outside_scope
scopes:
  clear_temp:
    allow_paths:
      - "{tempfile.gettempdir()}"
"""

DEFAULT_SNIPPETS = {
    "open_file_windows": "import os\nos.startfile('C:\\path\\to\\file')",
    "list_dir": "import os\nprint(os.listdir('.'))",
    "kill_process_by_name": (
        "import psutil\n"
        "for p in psutil.process_iter(['pid','name']):\n"
        "    if 'chrome' in (p.info.get('name') or '').lower():\n"
        "        p.terminate()"
    ),
}

# === SAFE BOOT ===
def ensure_files():
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")

    if not POLICY_FILE.exists():
        if yaml:
            POLICY_FILE.write_text(DEFAULT_POLICY_YAML, encoding="utf-8")
        else:
            POLICY_FILE.write_text("# Install pyyaml to parse policy\n" + DEFAULT_POLICY_YAML, encoding="utf-8")

    if not SNIPPETS_FILE.exists():
        SNIPPETS_FILE.write_text(json.dumps(DEFAULT_SNIPPETS, indent=2), encoding="utf-8")

    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(json.dumps({"long_term": [], "short_term": []}, indent=2), encoding="utf-8")

    if not SHORT_TERM_FILE.exists():
        SHORT_TERM_FILE.write_text(json.dumps([], indent=2), encoding="utf-8")

# === CONFIG CLASS ===
class Config:
    def __init__(self):
        ensure_files()
        self.load()

    def load(self):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception:
            self._data = DEFAULT_CONFIG.copy()
            self.save()

    def save(self):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self.save()

# === POLICY CLASS ===
class PolicyManager:
    def __init__(self):
        ensure_files()
        self.raw = None
        self.load()

    def load(self):
        try:
            if yaml:
                with open(POLICY_FILE, "r", encoding="utf-8") as f:
                    self.raw = yaml.safe_load(f)
            else:
                with open(POLICY_FILE, "r", encoding="utf-8") as f:
                    self.raw = {"raw_text": f.read()}
        except Exception as e:
            logging.error("Failed to load policy: %s", e)
            self.raw = {}

    def allows_auto(self, action_type):
        if not self.raw:
            return False
        return action_type in self.raw.get("auto_allow", [])

    def requires_ask(self, action_type):
        if not self.raw:
            return True
        return action_type in self.raw.get("ask_to_run", [])

SETTINGS = {
    "model": "llama3.1:8b-q5_K_M",
    "temperature": 0.7,
    "memory_file": "ritsu_memory.json",
    "short_term_file": "short_term_mem.json",
    "cpu_threshold": 95,  # % before warning (increased to prevent spam)
    "ram_threshold": 90,  # % before warning
    "check_interval": 10,  # seconds (increased to reduce frequency)
    "alert_cooldown": 60   # seconds between same alert type
}



