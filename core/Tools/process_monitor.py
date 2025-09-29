import psutil
import time

class ProcessMonitor:
    """Basic process monitoring for Ritsu."""

    def __init__(self):
        self.start_time = time.time()

    def get_status(self):
        """Return CPU, memory usage, and uptime."""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        uptime = time.time() - self.start_time
        return {
            "cpu_percent": cpu,
            "memory_percent": mem,
            "uptime_sec": int(uptime)
        }