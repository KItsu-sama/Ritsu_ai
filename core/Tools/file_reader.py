from pathlib import Path

class FileReader:
    """Simple file reader tool for Ritsu."""

    def read_file(self, file_path: str, max_bytes: int = 4096) -> str:
        """Read up to max_bytes from a .txt or .md file."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return "File not found."
        if not path.suffix.lower() in [".txt", ".md"]:
            return "Unsupported file type."
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read(max_bytes)
        except Exception as e:
            return f"Error reading file: {e}"