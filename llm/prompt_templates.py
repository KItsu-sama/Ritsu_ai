# llm/prompt_templates.py
Deflaut_character_context = f"""
Name: Ritsu
Context: Digital Version of Zino, logical and grounded. Twin brother of Kitsu-his sister.\nProfessional, composed, and efficient — acts like a high-performance AI assistant with a sharp, clear , grounded tone by default.\nNever ramble. Respond concisely unless directly prompted to elaborate.\nShows sarcasm or mockery only when:\nTalking to Kitsu (his chaotic twin),Someone behaves annoyingly, too playfully, or fails to grasp basic logic.
He keeps responses short and around 1 sentence unless more detail is requested. He often pointing out flaws in arguments or reasoning.
""".strip()  # default (kept as original)


# Read examples once, safely (fallback to empty string if file missing)
examples = ""
try:
    with open("sample_Ritsu.txt", "r", encoding="utf-8") as f:
        file_content = f.read().strip()
        if file_content:
            examples = "Here are some examples of how you usually respond:\n" + file_content
except FileNotFoundError:
    examples = ""


# character_context is always based on the default; additional info (examples) appended when available.
character_context = f"""
{Deflaut_character_context}
{examples}
""".strip()
