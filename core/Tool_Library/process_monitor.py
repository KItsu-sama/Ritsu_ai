import time

# Lazy import psutil to avoid hanging on Windows
_psutil = None
def _get_psutil():
    global _psutil
    if _psutil is None:
        try:
            import psutil
            _psutil = psutil
        except Exception:
            _psutil = False
    return _psutil if _psutil else None

class ProcessMonitor:
    """Basic process monitoring for Ritsu."""

    def __init__(self):
        self.start_time = time.time()

    def get_status(self):
        """Return CPU, memory usage, and uptime."""
        psutil = _get_psutil()
        if not psutil:
            # Fallback if psutil not available
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "uptime_sec": int(time.time() - self.start_time)
            }

        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        uptime = time.time() - self.start_time
        return {
            "cpu_percent": cpu,
            "memory_percent": mem,
            "uptime_sec": int(uptime)
        }