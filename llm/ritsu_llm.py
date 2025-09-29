import httpx

class RitsuLLM:
    """Basic LLM caller for Ritsu (Ollama local example)."""

    def __init__(self, endpoint="http://localhost:11434/api/generate"):
        self.endpoint = endpoint

    async def generate(self, prompt: str, model: str = "llama2") -> str:
        payload = {"model": model, "prompt": prompt}
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")