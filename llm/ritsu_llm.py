"""
Ritsu's LLM Interface using Ollama - Modified for Router-Ready Architecture
"""
import ollama
from typing import Optional, Dict, Any, List
import json

class RitsuLLM:
    """Handles all LLM interactions for Ritsu"""
    
    def __init__(self, model: str = "llama3.2:3b", **kwargs):
        self.model = model
        self.client = ollama
        
        # Load static persona and configuration data
        self.context, self.core_memory, self.ai_router = self._load_static_data()
        
        # History now stores messages as role/content pairs, not user_input strings
        self.conversation_history: List[Dict[str, str]] = []
        
    def _load_static_data(self) -> tuple[str, str, Dict]:
        """Load Ritsu's character definition, Core Memory, and Router Config."""
        
        # 1. Character Context (Persona)
        try:
            with open("sample_Ritsu.txt", "r", encoding="utf-8") as f:
                character_context = f.read()
        except FileNotFoundError:
            character_context = "You are Ritsu, a technical AI assistant."
            
        # 2. Core Memory (Rules) - Formatted for prompt injection
        core_memory_str = ""
        try:
            with open("core_memory.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                core_memory_str = "### CORE MEMORY DIRECTIVES ###\n"
                for item in data.get("ritsu_identity_v1", []):
                    core_memory_str += f"- [{item['category']}]: {item['data']}\n"
        except (FileNotFoundError, json.JSONDecodeError):
            core_memory_str = "No core memory loaded. Adhere to basic assistant rules."

        # 3. AI Router Config (Placeholder for now)
        ai_router_config = {}
        try:
             with open("meta_data_router.json", "r", encoding="utf-8") as f:
                 ai_router_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
             pass

        return character_context, core_memory_str, ai_router_config

    def generate_response(
        self, 
        user_input_dict: Dict[str, str],  # Now accepts a dictionary {User: Input}
        mode: int,                        # 0 or 1
        stream: bool = False
    ) -> str:
        """Generate a response to user input"""
        
        # Build prompt with all context
        prompt = self._build_prompt(user_input_dict, mode)
        
        # NOTE: For multi-user input, we only add the *current turn's dialogue* # to the history as a single block, not individual messages.
        self.conversation_history.append({
            "role": "user_turn",
            "content": json.dumps(user_input_dict) # Store raw dict for context
        })
        
        try:
            if stream:
                return self._stream_response(prompt)
            else:
                return self._complete_response(prompt)
        except Exception as e:
            return f"[Error generating response: {e}]"
    
    def _build_prompt(self, user_input_dict: Dict[str, str], mode: int) -> str:
        """Build the complete, compact, mode-aware prompt."""
        
        # 1. Mode Instruction
        if mode == 1:
            MODE_INSTRUCTION = "Emotion Protocol: Conversational (MODE 1). Allows sarcasm, jokes, and mockery (only when appropriate)."
        else:
            MODE_INSTRUCTION = "Emotion Protocol: Professional (MODE 0). Strict professionalism, mid or no emotional tone, concise and grounded logic only."

        # 2. Dialogue Block
        dialogue_block = "--- CURRENT TURN DIALOGUE ---\n"
        for user, text in user_input_dict.items():
            dialogue_block += f"{user}: {text}\n"
        dialogue_block += "-----------------------------"
        
        # 3. History (Keep the last 5 turns)
        history_block = "### RECENT CONVERSATION HISTORY ###\n"
        for msg in self.conversation_history[-5:]:
            if msg['role'] == 'assistant':
                 # Ritsu's past response
                 history_block += f"Ritsu: {msg['content']}\n"
            elif msg['role'] == 'user_turn':
                # Reconstruct multi-user turn for clarity
                recent_turn = json.loads(msg['content'])
                for user, text in recent_turn.items():
                    history_block += f"|--> {user}: {text}\n"
            
        
        # 4. Final Compact Prompt Assembly
        Full_Prompt = f"""
{self.context}

{self.core_memory}

{MODE_INSTRUCTION}

{history_block}

{dialogue_block}

Ritsu's Response:
"""
        return Full_Prompt.strip() # Clean up leading/trailing whitespace

    # The _complete_response, _stream_response, analyze_code, solve_math, 
    # and plan_action methods remain largely the same, but the prompt passed 
    # to them will now contain the full Ritsu context and history.

    def _complete_response(self, prompt: str) -> str:
        """Get complete response from Ollama (Same as original, but ensures 
           history update includes only the raw text response)"""
        # ... (rest of the original _complete_response logic) ...
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 500
            }
        )
        
        result = response['response']
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": result
        })
        
        # Trim history if too long (20 turns = 10 user_turns + 10 assistant responses)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return result

    # ... (Other methods like _stream_response, analyze_code, solve_math, 
    # plan_action, and clear_history would be carried over from your original class) ...