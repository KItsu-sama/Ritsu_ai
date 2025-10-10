#!/usr/bin/env python3
"""
Simple interactive test for Ritsu - basic input/output functionality
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm.ritsu_llm import RitsuLLM
from core.planning import Planner
from core.executor import Executor
from input.command_parser import CommandParser


async def simple_chat_test():
    """Simple chat test with Ritsu."""
    print("🤖 Ritsu Simple Chat Test")
    print("=" * 40)
    print("Type 'quit' to exit")
    print()
    
    # Initialize components
    print("Initializing Ritsu...")
    try:
        llm = RitsuLLM(model="qwen2:0.5b")
        parser = CommandParser()
        planner = Planner()
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    print("\n🎉 Ritsu is ready! Start chatting:")
    print("-" * 40)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Parse input
            parsed = parser.parse(user_input, "terminal")
            
            # Create event
            event = {
                "type": parsed.type,
                "content": user_input,
                "source": "terminal",
                "command": parsed.command,
                "args": parsed.args
            }
            
            # Plan response
            plan = planner.decide(event)
            
            # Generate response using LLM
            user_dict = {"User": user_input}
            response = llm.generate_response(user_dict, mode=0)
            
            print(f"Ritsu> {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    try:
        asyncio.run(simple_chat_test())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)