from __future__ import annotations
from input.command_parser import CommandParser, CommandResult
"""
from core import 

event_manager = EventManager(
    config=config,
    ritsu_core=ritsu_core_instance,
    tools=tools_dict,
    output_manager=output_manager_instance,
)

# Register additional listeners if needed
async def on_custom_event(event):
    print("Custom event received:", event)

event_manager.on("custom_event", on_custom_event)

# Start event loop
asyncio.create_task(event_manager.run(shutdown_event))

# Emit events
await event_manager.emit({"type": "input", "content": "!help", "source": "chat"})
"""

if __name__ == "__main__":
    parser = CommandParser()

    examples = [
        "/restart now --force",
        "!status --verbose",
        ">exec python script.py --debug",
        "#mode chatty",
        "@voice male",
        "just some natural language input",
        "",
        "/unknowncmd arg1 arg2",
        "/help --topic=commands",
        "#debug on",
        "/clear",
        "/reload --all",
        "/memory --max=1024",
        "/config --set=value",
    ]

    for example in examples:
        result = parser.parse(example, source="test")
        print(f"Input: {example!r}\nParsed: {result}\n")
        print(120/2.5)
