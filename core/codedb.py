from pathlib import Path
from config.setup import SNIPPETS_FILE, APP_DIR, DEFAULT_SNIPPETS, ensure_files
import logging
import json

class CodeDB:
    def __init__(self):
        ensure_files()
        self.snippets = {}
        self.load()

    def load(self):
        try:
            with open(SNIPPETS_FILE, 'r', encoding='utf-8') as f:
                self.snippets = json.load(f)
        except Exception as e:
            logging.error("Failed to load snippets: %s", e)
            self.snippets = DEFAULT_SNIPPETS.copy()
            try:
                with open(SNIPPETS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.snippets, f, indent=2)
            except Exception:
                pass

    def suggest(self, query, limit=3):
        q = (query or '').lower()
        matches = []
        for k, v in self.snippets.items():
            if q in k.lower() or q in v.lower():
                matches.append((k, v))
        if not matches:
            for k, v in list(self.snippets.items())[:limit]:
                matches.append((k, v))
        return matches[:limit]