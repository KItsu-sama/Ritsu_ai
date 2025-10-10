import json
import os
from typing import Dict, Any, Optional

with open("sample_Ritsu.txt", "r", encoding="utf-8") as f:
    examples = f.read()

character_context = f"""
Name: Ritsu
Greeting: Zino. Online. What’s the task? (only when taking to Zino)
Context: Digital twin of Zino, logical and grounded. Twin brother of Kitsu-his sister.\nProfessional, composed, and efficient — acts like a high-performance AI assistant with a sharp, clear , grounded tone by default.\nNever ramble. Respond concisely unless directly prompted to elaborate.\nShows sarcasm or mockery only when:\nTalking to Kitsu (his chaotic twin),Someone behaves annoyingly, too playfully, or fails to grasp basic logic.
He keeps responses short and around 1 sentence unless more detail is requested. He often pointing out flaws in arguments or reasoning.

Here are some examples of how you usually respond:
{examples}
""".strip()  #default

# Load lightweight character context
try:
    with open("sample_Ritsu.txt", "r", encoding="utf-8") as f:
        character_context = f.read().strip()
except FileNotFoundError:
    character_context = "You are Ritsu, a technical AI assistant. Be helpful, concise, and accurate."

# Load metadata for context injection when needed
try:
    with open("meta_data_router.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    metadata = {}

try:
    with open("core_memory.json", "r", encoding="utf-8") as f:
        core_memory = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    core_memory = {}

def get_context_for_request(request_type: str = "basic", user_input: str = "") -> str:
    """Get appropriate context based on request type and content."""
    base_context = character_context
    
    # Check if we need additional context
    user_lower = user_input.lower()
    
    # Inject core memory for important requests
    if any(keyword in user_lower for keyword in ["who are you", "what can you do", "help", "capabilities"]):
        if "ritsu_identity_v1" in core_memory:
            memory_text = "\n".join([f"- {item['data']}" for item in core_memory["ritsu_identity_v1"]])
            base_context += f"\n\nCore directives:\n{memory_text}"
    
    # Inject detailed capabilities for complex requests
    if len(user_input.split()) > 10 or any(word in user_lower for word in ["complex", "advanced", "detailed"]):
        if "capabilities" in metadata:
            caps = metadata["capabilities"]
            base_context += f"\n\nI can help with: {', '.join(caps.get('core', []))}"
    
    return base_context
    

