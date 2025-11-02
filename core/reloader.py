# core/reloader.py
from multiprocessing import Process, Queue
import importlib, time, sys, traceback


def _reload_ (mod_names, q: Queue):
    status = {}
    for name in mod_names:
        try:
            mod = importlib.import_module(name)
            importlib.reload(mod)
            status[name] = "reloaded"
        except Exception as e:
            status[name] = f"failed: {e}"
    q.put({"status": status, "timestamp": time.time()})

def reload_modules_safe(module_names, timeout=30):
    q = Queue()
    p = Process(target=_reload_worker, args=(module_names, q), daemon=False)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return {"error": "reload timed out", "status": {}}
    try:
        return q.get_nowait()
    except Exception:
        return {"error": "failed to get reload result", "status": {}}
