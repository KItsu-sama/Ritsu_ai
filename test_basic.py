#!/usr/bin/env python3
"""
Simple test script to verify basic Ritsu functionality
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm.ritsu_llm import RitsuLLM
from core.planning import Planner
from input.command_parser import CommandParser


async def test_basic_functionality():
    """Test basic components."""
    print("🧪 Testing Ritsu Basic Functionality...")
    
    # Test 1: LLM Initialization
    print("\n1. Testing LLM Initialization...")
    try:
        llm = RitsuLLM(model="gemma:2b")
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        return False
    
    # Test 2: Command Parser
    print("\n2. Testing Command Parser...")
    try:
        parser = CommandParser()
        
        # Test system command
        result = parser.parse("/help", "test")
        print(f"✅ System command parsed: {result.type}")
        
        # Test natural language
        result = parser.parse("Hello, how are you?", "test")
        print(f"✅ Natural language parsed: {result.type}")
        
    except Exception as e:
        print(f"❌ Command parser failed: {e}")
        return False
    
    # Test 3: Planner
    print("\n3. Testing Planner...")
    try:
        planner = Planner()
        
        # Test event planning
        test_event = {
            "type": "user_input",
            "content": "Hello Ritsu",
            "source": "test"
        }
        
        plan = planner.decide(test_event)
        print(f"✅ Plan generated: {plan.get('type', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Planner failed: {e}")
        return False
    
    # Test 4: Simple LLM Response
    print("\n4. Testing LLM Response...")
    try:
        user_input = {"TestUser": "Hello, who are you?"}
        response = llm.generate_response(user_input, mode=0)
        print(f"✅ LLM Response: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ LLM response failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed! Ritsu is ready for basic operation.")
    return True


if __name__ == "__main__":
    result = asyncio.run(test_basic_functionality())
    sys.exit(0 if result else 1)