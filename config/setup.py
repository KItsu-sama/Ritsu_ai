# config/setup.py
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