from __future__ import annotations

# ============================================================
#  RITSU ASYNC LLM INTERFACE
#  Handles Ollama interaction, async queue, workers, and auto restart
#  Includes built-in debug flag and structured comments for maintainability
# ============================================================

from config.config import _model_
import asyncio
import concurrent.futures
import json
import logging
import time
import aiohttp
import subprocess
from typing import Any, Dict, Optional, List, AsyncIterator

log = logging.getLogger(__name__)

# Lazy import Ollama Python client to avoid hanging on import
_ollama = None
def _get_ollama():
    global _ollama
    if _ollama is None:
        try:
            import ollama
            _ollama = ollama
        except Exception as e:
            log.warning("Failed to import ollama Python client: %s. LLM features will be disabled.", e)
            _ollama = None  # explicit None indicates missing client
    return _ollama if _ollama else None


# ============================================================
#  OllamaManager - Health check + restart manager
# ============================================================

class OllamaManager:
    """Handles Ollama health check and restart if needed."""

    def __init__(self, host: str = "https://good-doors-rest.loca.lt", debug: bool = False):
        # Allow override via environment variable (or fallback to localhost)
        import os
        # Prefer explicit host param -> OLLAMA_BASE_URL -> local default
        self.host = host or os.getenv("OLLAMA_BASE_URL", None) or "http://127.0.0.1:11434"
        self._restart_lock = asyncio.Lock()
        self.debug = debug

        log.info(f"OllamaManager using host: {self.host}")

    async def is_running(self) -> bool:
        """Ping Ollama API to confirm it's up and responsive."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/api/version", timeout=2) as resp:
                    if self.debug:
                        log.debug(f"Ollama health check → status={resp.status}")
                    return resp.status == 200
        except Exception as e:
            if self.debug:
                log.debug(f"Ollama health check failed: {e}")
            return False

    async def restart(self) -> bool:
        """Restart Ollama if not responding."""
        async with self._restart_lock:
            # Check again inside the lock to avoid multiple restarts
            if await self.is_running():
                return True
            log.warning("⚠️ Ollama seems offline — restarting...")
            try:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Wait up to 10s for Ollama to come online
                for i in range(20): # ~20 × 0.5s = 10 seconds max
                    if await self.is_running():
                        log.info("✅ Ollama restarted successfully.")
                        return True
                    await asyncio.sleep(0.5)
                    if self.debug:
                        log.debug(f"Waiting for Ollama to boot... ({i})")
                log.error("❌ Ollama failed to start within timeout.")
                return False
            except Exception as e:
                log.exception(f"Failed to restart Ollama: {e}")
                return False


# ============================================================
#  AsyncRequest - request container used by the async worker queue
# ============================================================

class AsyncRequest:
    """Internal representation of a request to the LLM worker."""
    def __init__(self, user_input: Dict[str, str], mode: int, stream: bool = False):
        self.user_input = user_input
        self.mode = mode
        self.stream = stream
        self.created_at = time.time()
        self.response_future: asyncio.Future = asyncio.get_event_loop().create_future()
        # for streaming: bounded queue for chunks to avoid unbounded buffering
        # Use a modest maxsize so the producer (blocking client) feels backpressure
        self.stream_q: Optional[asyncio.Queue] = asyncio.Queue(maxsize=32) if stream else None


# ============================================================
#  RitsuLLMAsync - Core asynchronous Ollama router
# ============================================================

class RitsuLLM:
    def __init__(
        self,
        model: str = _model_,
        ollama_client=None,
        ollama_manager: Optional[OllamaManager] = None,
        max_workers: int = 1,
        queue_size: int = 4,
        history_limit: int = 12,
        max_response_chars: int = 3000,
        request_timeout: float = 180.0,
        ollama_options: Optional[Dict] = None,
        debug: bool = False, 
    ):
        # ------------------------------
        # General configuration
        # ------------------------------
        self.debug = debug
        self.model = model
        # Use lazy-loaded ollama if not provided
        if ollama_client is None:
            ollama_client = _get_ollama()
        self.client = ollama_client
        self.ollama_manager = ollama_manager or OllamaManager(debug=debug)  # Enhanced from second version
        self.ollama_options = ollama_options or {"temperature": 0.7, "top_p": 0.9, "num_predict": 200}
        self.history_limit = history_limit
        self.max_workers = max_workers
        self._request_timeout = request_timeout
        self._max_response_chars = max_response_chars  # Protect against large responses
        # ------------------------------
        # Async management
        # ------------------------------
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.worker_tasks: List[asyncio.Task] = []
        self._running = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.health_check_task = None
        # ------------------------------
        # Context and memory load
        # ------------------------------
        self.conversation_history: List[Dict[str, str]] = []
        self.context, self.core_memory, self.ai_router = self._load_static_data()
        # Configure logging based on debug flag
        if self.debug:
            log.setLevel(logging.DEBUG)  # Set to DEBUG mode if enabled
            log.debug("RitsuLLM initialized in DEBUG mode.", extra={"model": model})
        else:
            log.info("RitsuLLM initialized", extra={"model": model})

    # ============================================================
    #  Static file loaders
    # ============================================================
    def _load_static_data(self):
        """Load Ritsu's character context, core memory, and router configs."""
        try:
            with open("sample_Ritsu.txt", "r", encoding="utf-8") as f:
                character_context = f.read().strip()
        except FileNotFoundError:
            character_context = "You are Ritsu, a technical AI assistant. Be helpful, concise, and accurate."
            log.warning("sample_Ritsu.txt not found; using default context")

        core_memory_str = ""
        try:
            with open("core_memory.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                essential_rules = [
                    item for item in data.get("ritsu_identity_v1", [])
                    if item.get("category") in ["core_behavior", "communication"]
                ]
                if essential_rules:
                    core_memory_str = "\n" + "\n".join([f"- {item['data']}" for item in essential_rules])
        except Exception:
            log.warning("core_memory.json not found or invalid; skipping core memory")

        ai_router_config = {}
        try:
            with open("meta_data_router.json", "r", encoding="utf-8") as f:
                ai_router_config = json.load(f)
        except Exception:
            log.warning("meta_data_router.json not found or invalid; using empty router")

        return character_context, core_memory_str, ai_router_config

    # ============================================================
    #  Lifecycle management
    # ============================================================
    async def start(self):
        """Start async workers and background health monitor."""
        if self._running:
            return
        self._running = True

        # Launch workers
        for i in range(self.max_workers):
            t = asyncio.create_task(self._worker_loop(i))
            self.worker_tasks.append(t)

        # Launch health monitor
        self.health_check_task = asyncio.create_task(self._background_health_check_loop())
        log.info(f"RitsuLLMAsync started with {self.max_workers} worker(s).")

    async def shutdown(self):
        """Gracefully stop all workers and tasks."""
        if not self._running:
            return
        self._running = False

        # Stop health check
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None

        # Send sentinel None to stop each worker
        for _ in range(len(self.worker_tasks)):
            await self.request_queue.put(None)
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        # Shutdown executor
        self._executor.shutdown(wait=True)
        log.info("RitsuLLMAsync router stopped.")

    async def _background_health_check_loop(self):
        """Background watchdog — restarts Ollama if it goes down."""
        while self._running:
            try:
                if not await self.ollama_manager.is_running():
                    log.warning("Background health check: Ollama is down, restarting...")
                    await self.ollama_manager.restart()
                await asyncio.sleep(30) # check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("Background health check error: %s", e)
                await asyncio.sleep(30)

    # ============================================================
    #  Queue / request management
    # ============================================================
    async def enqueue(self, user_input: Dict[str, str], mode: int = 0, stream: bool = False) -> Any:
        """Queue a user request for generation."""
        self._append_history_user(user_input)
        req = AsyncRequest(user_input, mode, stream)
        await self.request_queue.put(req) 
        return (self._create_stream_generator(req) if stream else await req.response_future) ## If streaming, return an async generator that yields from req.stream_q | wait for completion and return the final text

    # ============================================================
    #  Worker loop — main generation logic
    # ============================================================
    def _call_ollama_generate(self, prompt: str) -> Any:
        """Synchronous Ollama generation call (thread-safe, stable version)."""
        result_container = {}

        def target():
            try:
                result_container["result"] = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options=self.ollama_options
                )
            except Exception as e:
                result_container["error"] = e

        future = self._executor.submit(target)
        try:
            # Extend timeout to 3 minutes, enough for slower models on CPU
            future.result(timeout=self._request_timeout or 180)
        except concurrent.futures.TimeoutError:
            log.warning("⚠️ Ollama generation exceeded timeout. Checking system load...")
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            if cpu > 90:
                log.warning(f"System under heavy load (CPU {cpu}%). Skipping restart.")
            else:
                log.warning("Attempting soft restart of Ollama...")
                subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(5)
            raise TimeoutError("Generation timeout — Ollama may be overloaded.")
        except Exception as e:
            log.exception("Ollama generation crashed.")
            raise e

        if "error" in result_container:
            raise result_container["error"]

        return result_container.get("result", "[No response]")

    # ============================================================
    #  Blocking bridge for Ollama
    # ============================================================
    def _call_ollama_generate(self, prompt: str) -> Any:
        """Synchronous Ollama generation call (wrapped by worker)."""
        result_container = {}

        def target():
            try:
                result_container["result"] = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options=self.ollama_options
                )
            except Exception as e:
                # Handle ollama ResponseError or any other exception
                ollama = _get_ollama()
                is_response_error = (ollama and hasattr(ollama, '_types') and
                                    isinstance(e, ollama._types.ResponseError))
                if is_response_error or "not found" in str(e).lower():
                    print(f"[⚠️] Model '{self.model}' not found.")
                    user_input = input(f"Would you like to pull '{self.model}' from Ollama Hub? (y/n): ").strip().lower()
                    if user_input == "y":
                        print(f"➡️ Pulling model '{self.model}'... this may take a few minutes.")
                        subprocess.run(["ollama", "pull", self.model], check=False)
                        print(f"✅ Model '{self.model}' downloaded. Retrying request...")
                        result_container["result"] = self.client.generate(
                            model=self.model,
                            prompt=prompt,
                            options=self.ollama_options
                        )
                    else:
                        print("❌ Model not pulled. Aborting request.")
                        result_container["error"] = e
                else:
                    raise

        thread = concurrent.futures.ThreadPoolExecutor().submit(target)
        try:
            thread.result(timeout=120.0)  # 2-minute timeout for blocking call
        except Exception:
            log.error("Ollama unresponsive — restarting.")
            subprocess.Popen(["taskkill", "/f", "/im", "ollama.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raise TimeoutError("Ollama hung; restarted.")

        if "error" in result_container:
            raise result_container["error"]
        return result_container.get("result", "[No response]")

    async def _run_blocking_stream(self, prompt: str, req: AsyncRequest):
        """Push streamed chunks from blocking generator into asyncio queue."""
        loop = asyncio.get_event_loop()
        q = req.stream_q

        def blocking_reader():
            try:
                for chunk in self.client.generate(model=self.model, prompt=prompt, stream=True, options=self.ollama_options):
                    text = self._extract_chunk_text(chunk)
                    if text:
                        asyncio.run_coroutine_threadsafe(q.put(text), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(q.put(f"[Stream Error: {e}]"), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)

        loop.run_in_executor(self._executor, blocking_reader)

    # ============================================================
    #  Utility / history methods
    # ============================================================
    async def _create_stream_generator(self, req: AsyncRequest) -> AsyncIterator[str]:
        """Async generator that yields streamed text chunks."""
        q = req.stream_q
        if q is None:
            yield "[Error: Stream queue not initialized]"
            return

        collected_text = ""
        while True:
            chunk = await q.get()
            if chunk is None:
                break
            collected_text += chunk
            if len(collected_text) > self._max_response_chars:
                collected_text = collected_text[:self._max_response_chars] + " [Truncated]"
                yield collected_text
                break
            yield collected_text
            q.task_done()

        self._append_history_assistant(collected_text)
        if not req.response_future.done():
            req.response_future.set_result(collected_text)

    def _extract_chunk_text(self, chunk: Any) -> str:
        """Helper to safely extract the text from a streamed chunk dictionary (from B)."""
        # Based on Ollama documentation, streaming generate chunks should have a 'response' key.
        if isinstance(chunk, dict) and "response" in chunk:
            return chunk["response"]
        
        # Fallback for unexpected chunk format (shouldn't happen with official client)
        return ""

    def _build_prompt(self, user_input_dict: Dict[str, str], mode: int) -> str:
        """Construct a structured prompt from memory + user turn."""
        MODE_INSTRUCTION = (
            "Emotion Protocol: Conversational (MODE 1). Playful, sarcastic, or teasing when appropriate."
            if mode == 1 else
            "Emotion Protocol: Professional (MODE 0). Concise, neutral, and logical."
        )

        dialogue = "\n".join([f"{k}: {v}" for k, v in user_input_dict.items()])
        history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-self.history_limit:]
        ]) or "(No recent history)"

        prompt = f"""
{self.context}
{self.core_memory}
{MODE_INSTRUCTION}
{history}
{dialogue}
Ritsu:"""
        return prompt.strip()

    def _append_history_assistant(self, text: str):
        self.conversation_history.append({"role": "assistant", "content": text})
        if len(self.conversation_history) > self.history_limit:
            self.conversation_history = self.conversation_history[-self.history_limit:]

    def _append_history_user(self, user_input_dict: Dict[str, str]):
        """Helper to append the full user turn *before* request processing (from B)."""
        # Use JSON dump to preserve the multi-user structure for the history block logic.
        content = json.dumps(user_input_dict)
        self.conversation_history.append({"role": "user_turn", "content": content})
        if len(self.conversation_history) > self.history_limit:
            self.conversation_history = self.conversation_history[-self.history_limit:]

    def _extract_response(self, response: Any) -> str:
        try:
            if isinstance(response, dict) and 'response' in response:
                return response['response'].strip()
            else:
                return str(response).strip()
        except Exception as e:
            log.error("Failed to extract response: %s", e, exc_info=True)
            return "[Error: Invalid LLM response]"

    # ---------- Integrated Ollama Health-Checked Generate Method ----------
    async def generate(self, prompt: str) -> str:
        """Safe generate with auto Ollama health check and restart."""
        if not await self.ollama_manager.is_running():
            success = await self.ollama_manager.restart()
            if not success:
                raise RuntimeError("Failed to restart Ollama; cannot generate response")
        
        try:
            # Run blocking generate in thread to avoid blocking event loop
            result = await asyncio.to_thread(
                self.client.generate,
                model=self.model,
                prompt=prompt,
                options=self.ollama_options
            )
            return self._extract_response(result)
        except Exception as e:
            log.exception(f"Ollama generation error: {e}")
            # Attempt restart on error
            await self.ollama_manager.restart()
            raise

    # Convenience helper that mirrors previous synchronous API 
    async def generate_response(self, user_input_dict: Dict[str, str], mode: int = 0, stream: bool = False):
        """Simple wrapper that starts the router if needed and enqueues the request."""
        if not self._running:
            await self.start()
        result = await self.enqueue(user_input_dict, mode, stream)
        return result