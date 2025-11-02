# Ritsu - Autonomous AI Framework

## Quick Start Guide

Ritsu is a local, modular AI framework that operates entirely offline with autonomous system management capabilities.

### Prerequisites

- Python 3.8+
- Ollama (for local LLM inference)
- Dependencies: `pip install -r requirements.txt`

### Installation

```bash
# Clone/setup the project
cd The_Ritsu

# Install dependencies
pip install ollama aiohttp psutil

# Ensure Ollama is running
ollama serve  # In a separate terminal
```

### Running Ritsu

#### 1. Basic Startup

```bash
python main.py
```

This starts Ritsu with default configuration. You'll see:
```
Starting Ritsu v1.0...
Ritsu: listening...
```

#### 2. Wake Phrase Activation

Type the wake phrase to activate Ritsu:
```
ritsu hello
```

Or with debug mode:
```
ritsu --debug what is 2 + 2?
```

#### 3. Shell Commands

Execute shell commands directly:
```
ritsu ! ls -la
ritsu --shell pip list
```

### Core Features

#### 1. **Calculator Tool**
```python
from core.Tool_Library.math.calculator import Calculator
calc = Calculator()
result = calc.calculate("2 + 2 * 3")  # Returns 8
```

#### 2. **File Operations**
```python
from core.Tool_Library.file_reader import FileReader
reader = FileReader()
content = reader.read_file("data/sample.txt")
```

#### 3. **Process Monitoring**
```python
from core.Tool_Library.process_monitor import ProcessMonitor
monitor = ProcessMonitor()
status = monitor.get_status()  # CPU, memory, uptime
```

#### 4. **Memory Management**
```python
from ai.memory_manager import MemoryManager
memory = MemoryManager()
await memory.save_event({"user": "input", "response": "output"})
summary = await memory.get_conversation_summary()
```

#### 5. **NLP Analysis**
```python
from ai.nlp_engine import OptimizedNLPEngine
nlp = OptimizedNLPEngine()
result = await nlp.analyze("What is the weather?")
# Returns: intent, entities, sentiment, confidence, etc.
```

#### 6. **Planning & Execution**
```python
from core.planning import Planner
from core.executor import Executor

planner = Planner()
executor = Executor()

event = {"type": "user_input", "content": "Calculate 5 * 5"}
plan = planner.decide(event)
result = await executor.execute(plan)
```

### Configuration

Edit `system/config.yaml` to customize:

```yaml
app:
  name: Ritsu
  env: dev
  safe_mode: false
  restart_on_crash: true

logging:
  level: INFO
  dir: data/logs
  json: true

io:
  enable_mic: false
  enable_chat: false

llm:
  model: mistral
  host: http://127.0.0.1:11434
```

### Testing

Run the test suite:

```bash
# Basic functionality tests
python test_ritsu_complete.py

# Integration tests
python test_ritsu_integration.py
```

Expected output:
```
RESULTS: 9/9 tests passed
INTEGRATION TEST SUMMARY: 2/2 passed
```

### Architecture Overview

```
Input Sources (CLI, Mic, Chat)
    ↓
InputManager
    ↓
EventManager
    ↓
Planner (generates plan)
    ↓
Executor (executes actions)
    ↓
AI/Tools/Memory
    ↓
OutputManager
    ↓
Output Destinations (Console, TTS, Stream)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Planner** | Analyzes events and creates execution plans |
| **Executor** | Executes plans with action coordination |
| **EventManager** | Central event dispatcher |
| **MemoryManager** | Short/long-term memory management |
| **NLPEngine** | Intent detection and text analysis |
| **KnowledgeBase** | Structured facts and skills storage |
| **AIAssistant** | Main AI logic coordination |
| **OutputManager** | Multi-channel output handling |

### Common Tasks

#### Ask a Question
```
ritsu what is machine learning?
```

#### Perform Calculation
```
ritsu calculate 15 * 3 + 7
```

#### Get System Status
```
ritsu system status
```

#### Clear Memory
```
ritsu clear memory
```

### Troubleshooting

**Ollama not responding:**
```bash
# Restart Ollama
ollama serve
```

**Import errors:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Memory issues:**
- Check `data/memory.json` size
- Clear old logs in `data/logs/`

### Next Steps

1. **Customize prompts** in `llm/prompt_templates.py`
2. **Add new tools** in `core/Tool_Library/`
3. **Extend NLP** in `ai/nlp_engine.py`
4. **Configure I/O** in `system/config.yaml`

### Support

For issues or questions:
- Check logs in `data/logs/`
- Review `layout.rb` for architecture details
- Run tests to verify functionality

---

**Ritsu v1.0** - Your autonomous AI assistant

