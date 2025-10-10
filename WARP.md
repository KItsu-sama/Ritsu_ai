# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

The_Ritsu is an advanced AI assistant system with a modular, event-driven architecture. It's written in Python 3.13+ and integrates with Ollama for LLM operations. The system features autonomous planning, execution, memory management, and self-improvement capabilities.

## Development Commands

### Running the System
```powershell
# Start Ritsu with default configuration
python main.py

# Show command line options
python main.py --help

# Run with specific configuration
python main.py --config path/to/config.yaml

# Run in safe mode (restricts risky operations)
python main.py --safe-mode

# Run headless (no UI integration)
python main.py --headless

# Enable debug logging
python main.py --log-level DEBUG

# Check version
python main.py --version
```

### Development Setup
```powershell
# Install Ollama for LLM support (required dependency)
# Download from https://ollama.ai and install locally

# Install Python dependencies
pip install ollama

# Optional: Install YAML support for policy configuration
pip install pyyaml

# Optional: Install uvloop for better async performance (Linux/Mac)
pip install uvloop
```

### Testing and Debugging
```powershell
# Currently no test suite exists - this is a development opportunity
# Tests would typically be run with:
# python -m pytest tests/

# Debug mode with verbose logging
python main.py --log-level DEBUG

# Check if Ollama is running (required for LLM functionality)
ollama list

# View system logs (created in data/logs/ directory)
# Logs are structured JSON by default
```

## Architecture Overview

### Core Event-Driven System
The system operates on an event-driven architecture with these key components:

1. **Event Loop** (`main.py`): Central orchestrator that manages multiple async loops
   - Core loop: Processes events and coordinates planning/execution
   - Input loop: Handles user input from various sources
   - Monitoring loop: System health and performance tracking
   - Maintenance loop: Automated cleanup and optimization

2. **Planning System** (`core/planning.py`): 
   - Analyzes incoming events and determines appropriate strategies
   - Routes requests to specialized "experts" based on content analysis
   - Implements policy validation and safety checks
   - Supports simulation mode for plan validation

3. **Execution Engine** (`core/executor.py`):
   - Coordinates action execution across components
   - Manages parallel execution with resource allocation
   - Handles recovery actions and troubleshooting
   - Aggregates results and tracks execution status

4. **AI Assistant** (`ai/ai_assistant.py`):
   - Processes natural language input
   - Coordinates with NLP engine and knowledge base
   - Maintains conversation context and history
   - Supports multiple response modes (direct, analytical, creative, tool-use)

### Component Interactions

```
Input Sources → Command Parser → Event Manager → Planner → Executor → Output
     ↓              ↓               ↓            ↓          ↓         ↓
   (mic,         (system/        (event       (strategy  (action   (TTS,
   chat,         natural         queue)       selection) execution) avatar,
   API)          language)                                          stream)
                                                   ↕
                              AI Assistant ←→ LLM Engine (Ollama)
                                    ↕              ↕
                              Memory Manager   Knowledge Base
```

### Key Modules

- **`core/`**: Core system logic (planning, execution, tools, self-improvement)
- **`input/`**: Input processing (command parsing, STT, chat listeners)  
- **`output/`**: Output generation (TTS, avatar animation, streaming)
- **`ai/`**: AI capabilities (NLP, knowledge base, memory management)
- **`llm/`**: LLM interface and prompt templates (Ollama integration)
- **`config/`**: Configuration management and setup utilities
- **`data/`**: Runtime data storage (logs, memory, knowledge base)

### Self-Improvement System
The system includes autonomous learning capabilities:
- **`core/Ritsu_self.py`**: Metadata evolution and self-reflection
- **`core/self_improvement.py`**: Learning from failures and successful fixes
- **Memory persistence**: Stores interaction patterns and learns from user behavior

### Configuration System
- **Default config location**: `~/.ritsu/config.json`
- **Policy management**: YAML-based safety policies in `~/.ritsu/policy.yaml`
- **Memory persistence**: JSON-based storage in `data/` directory
- **Logging**: Structured JSON logs in `data/logs/`

## Important Development Notes

### Dependencies
- **Python 3.13+** required
- **Ollama**: Must be installed and running locally for LLM functionality
- **Optional**: PyYAML for policy configuration, uvloop for performance

### Missing Components
The system is designed with many optional components that are not yet implemented:
- Performance monitoring, security management, auto-updater
- Hardware monitoring, network monitoring, system analysis  
- Code review, test generation, documentation generation
- Plugin management, task scheduling

### Error Handling
The system uses graceful degradation - missing optional components are logged as warnings but don't prevent startup. The `safe_init()` function handles component initialization failures.

### Data Storage
- Configuration: `~/.ritsu/` directory
- Runtime data: `data/` directory in project root
- Logs: `data/logs/` with structured JSON format
- Memory: `data/ritsu_memory.json` and `data/short_term_mem.json`

### Development Patterns
- **Async-first**: Heavy use of asyncio throughout the system
- **Component injection**: Dependencies injected via `AppContext` in `main.py`
- **Graceful degradation**: Optional components fail safely
- **Event-driven**: All interactions flow through the event system
- **Safe execution**: Policy-based safety checks on all actions

## Getting Started for Development

1. Ensure Python 3.13+ and Ollama are installed
2. Run `python main.py --version` to verify setup
3. Check `data/logs/` for system logs after first run
4. The system creates configuration files in `~/.ritsu/` on first run
5. Monitor the event flow by enabling debug logging: `python main.py --log-level DEBUG`

The system is designed to be modular and extensible - new components can be added by following the existing patterns in `main.py` for component initialization and the event-driven architecture for integration.