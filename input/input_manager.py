from __future__ import annotations

"""
input/input_manager.py

InputManager — decides: mic, chat, file
- Unified input handling across multiple sources
- Async event-driven input processing
- Source-specific preprocessing and validation
- Event emission to central event queue
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import subprocess
import re

log = logging.getLogger(__name__)

# Initialize logging flags
_root_logging_silenced = False

Ritsu_turn = False

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
shell_color = YELLOW

# Color handling for prompt
try:
    import colorama
    colorama.init()
    YELLOW = colorama.Fore.YELLOW
    RESET = colorama.Style.RESET_ALL
except Exception:
    # Fallback ANSI
    YELLOW = "\x1b[33m"
    RESET = "\x1b[0m"

class InputManager:
    """Manages input from multiple sources and emits events."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mic=None,
        chat=None,
        command_parser=None,
        command_classifier=None,
    ):
        self.config = config or {}
        
        # Component references
        self.mic = mic
        self.chat = chat
        self.command_parser = command_parser
        # Optional fallback classifier (rule-based/ML)
        self.command_classifier = command_classifier
        
        # Input sources configuration
        self.enable_cli = True  # Always enabled
        self.enable_mic = False  # Disable mic by default
        self.enable_chat = False  # Disable chat by default
        self.enable_ui = False  # Disable UI by default
        self.enable_api = False  # Disable API by default
        
        # State tracking
        self.active_sources = []
        self._setup_active_sources()

        global _root_logging_silenced
    
    def _setup_active_sources(self) -> None:
        """Setup list of active input sources."""
        self.active_sources = []
        
        if self.enable_cli:
            self.active_sources.append(("cli", self._handle_cli_input))
        
        if self.enable_mic:
            self.active_sources.append(("mic", self._handle_mic_input))
        
        if self.enable_chat:
            self.active_sources.append(("chat", self._handle_chat_input))
        
        if self.enable_ui:
            self.active_sources.append(("ui", self._handle_ui_input))
        
        if self.enable_api:
            self.active_sources.append(("api", self._handle_api_input))
        
        log.info("Active input sources", extra={
            "sources": [name for name, _ in self.active_sources]
        })
    
    async def get_input(self) -> str:
        """Get input from the user asynchronously."""
        try:
            # Use asyncio.to_thread for blocking input operations
            result = await asyncio.to_thread(input, f"{shell_color}$ {RESET}")
            return result.strip()
        except EOFError:
            log.debug("EOF received")
            return ""
        except KeyboardInterrupt:
            log.debug("KeyboardInterrupt received")
            return ""
        except Exception as e:
            log.error(f"Error getting input: {e}")
            return ""

    async def run(
        self,
        queue: "asyncio.Queue[Dict[str, Any]]",
        shutdown_event: asyncio.Event
    ) -> None:
        """Main input processing loop.
        
        Args:
            queue: Event queue to emit input events to
            shutdown_event: Event to signal shutdown
        """
        log.info("Input manager starting", extra={
            "active_sources": len(self.active_sources)
        })
        
        try:
            # Create tasks for each active input source
            tasks = []
            for source_name, handler in self.active_sources:
                task = asyncio.create_task(
                    self._run_input_source(source_name, handler, queue, shutdown_event),
                    name=f"input_{source_name}"
                )
                tasks.append(task)
            
            # Wait for shutdown or all tasks to complete
            if tasks:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            else:
                log.warning("No input sources active, waiting for shutdown")
                await shutdown_event.wait()
                
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Input manager crashed")
            raise
        finally:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            log.info("Input manager stopped")
    
    async def _run_input_source(
        self,
        source_name: str,
        handler,
        queue: "asyncio.Queue[Dict[str, Any]]",
        shutdown_event: asyncio.Event
    ) -> None:
        """Run a single input source."""
        log.debug(f"Starting input source: {source_name}")
        try:
            while not shutdown_event.is_set():
                try:
                    # Get input from source
                    log.debug(f"Awaiting input from {source_name}")
                    input_data = await handler()
                    log.debug(f"Received input from {source_name}: {input_data}")
                    
                    if input_data:
                        # Create event
                        event = await self._create_event(source_name, input_data)
                        
                        if event:
                            # Emit to queue
                            await queue.put(event)
                            
                            log.debug("Input event emitted", extra={
                                "source": source_name,
                                "content_preview": input_data.get("content", "")[:50]
                            })
                    
                    # Small delay to prevent tight loops
                    await asyncio.sleep(0.1)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if str(e) == "EOF when reading a line":
                        # Handle EOF gracefully
                        log.debug(f"EOF received on {source_name}, stopping")
                        break
                    log.error(f"Input source error on {source_name}: {e}", extra={
                        "source": source_name,
                        "error": str(e)
                    })
                    # Continue running other sources
                    await asyncio.sleep(1.0)  # Longer delay after error
                    
        except asyncio.CancelledError:
            log.debug("Input source cancelled", extra={"source": source_name})
            raise
    
    async def _handle_cli_input(self) -> Optional[Dict[str, Any]]:
        """Handle CLI input."""
        try:
            user_input = await self.get_input()
            if not user_input:
                return None
                
            log.debug(f"CLI input received: {user_input}")
            
            # Check for wake words
            wake_pattern = re.compile(r'^(hey\s+ritsu|ritsu)(?:\s|$)', re.IGNORECASE)
            if wake_pattern.match(user_input):
                log.debug("Wake word detected")
                # If command follows wake word, extract it
                command = user_input[wake_pattern.match(user_input).end():].strip()
                if not command:
                    print("Ritsu: listening...")
                    # Get follow-up input
                    command = await self.get_input()
                    if not command:
                        return None
                
                return {
                    "content": command,
                    "metadata": {
                        "source": "cli",
                        "wake_word": True
                    }
                }
            
            # Non-wake word input
            return {
                "content": user_input,
                "metadata": {
                    "source": "cli",
                    "wake_word": False
                }
            }
        except Exception as e:
            log.error(f"CLI input error: {e}")
            return None

    async def _create_event(self, source: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a standardized event from input data.
        
        Args:
            source: Name of the input source
            input_data: Raw input data from source
            
        Returns:
            Standardized event dictionary or None if invalid
        """
        try:
            content = input_data.get("content", "")
            if not content or not content.strip():
                return None
            
            # Parse command if parser is available
            parsed_command = None
            if self.command_parser:
                try:
                    parsed_command = self.command_parser.parse(content)
                except Exception as e:
                    log.warning("Command parsing failed", extra={"error": str(e)})

            # Fallback: use command_classifier to infer system-like inputs if parser returned nothing
            if not parsed_command and self.command_classifier:
                try:
                    cls_result = self.command_classifier.classify(content)
                    # If classifier identifies a shell_command or high-confidence system intent, synthesize
                    if cls_result and cls_result.confidence >= 0.6 and cls_result.category in ("executor", "monitor"):
                        # Create a minimal CommandResult-like object (simple namespace)
                        class _Synth:
                            pass
                        synth = _Synth()
                        synth.raw_text = content
                        synth.type = "system"
                        synth.command = cls_result.metadata.get("match") or (content.split()[0] if content else "")
                        synth.args = cls_result.metadata.get("args", []) if isinstance(cls_result.metadata.get("args", []), list) else []
                        synth.flags = {}
                        synth.metadata = {"source": source, "confidence": cls_result.confidence}
                        parsed_command = synth
                except Exception as e:
                    log.debug("Command classifier failed", extra={"error": str(e)})
            
            # Create standardized event
            event = {
                "type": "user_input",
                "source": source,
                "content": content.strip(),
                "timestamp": asyncio.get_event_loop().time(),
                "metadata": input_data.get("metadata", {}),
            }
            
            # Add parsed command if available
            if parsed_command:
                event["parsed_command"] = parsed_command
            
            return event
            
        except Exception as e:
            log.error("Failed to create event", extra={
                "source": source,
                "input_data": input_data,
                "error": str(e)
            })
            return None
    
    # Input source handlers
    
    async def _handle_cli_input(self) -> Optional[Dict[str, Any]]:
        """Handle command line input."""
        global Ritsu_turn
        global _root_logging_silenced

        
        if _root_logging_silenced:
            # Run shell command synchronously but without blocking the event loop
            print(f"{shell_color}Terminal/User>{RESET}", end=" ", flush=True)
            loop = asyncio.get_event_loop()
            try:
                Command = await loop.run_in_executor(None, input)
                Command = Command.strip()
            except Exception:
                return None

            if not Command:
                return None

            try:
                # Run the subprocess in a thread to avoid blocking
                log.debug("CLI: executing shell command", extra={"cmd_preview": Command[:200]})
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(Command, shell=True, text=True, capture_output=True)
                )

                if result.returncode == 0:
                    # Command succeeded
                    print(f"{GREEN}{result.stdout}{RESET}")
                    globals()['shell_color'] = GREEN
                    log.debug("CLI: shell command succeeded", extra={"returncode": result.returncode})
                else:
                    # Command ran but failed (non-zero exit)
                    print(f"{RED}{result.stderr or 'Command failed.'}{RESET}")
                    globals()['shell_color'] = RED
                    log.debug("CLI: shell command failed", extra={"returncode": result.returncode})

            except Exception as e:
                # Command couldn’t even run
                print(f"{RED}Error executing command: {e}{RESET}")
                globals()['shell_color'] = RED
                log.exception("CLI: shell execution error")

            # After running a shell command, don't produce an InputManager event
            return None
        else:
            if not Ritsu_turn:
                try:
                    # Use asyncio to run input() in a thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    prompt = f"{YELLOW}User> {RESET}"
                    user_input = await loop.run_in_executor(None, input, prompt)

                    if user_input.strip():
                        Ritsu_turn = True  # Set flag before returning
                        log.debug("CLI: user input captured", extra={"preview": user_input.strip()[:120]})
                        return {
                            "content": user_input.strip(),
                            "metadata": {"input_method": "keyboard"}
                        }
                    
                except EOFError:
                    # Handle Ctrl+D or EOF
                    return {
                        "content": "!quit",
                        "metadata": {"input_method": "keyboard", "eof": True}
                    }
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    return {
                        "content": "!quit",
                        "metadata": {"input_method": "keyboard", "interrupt": True}
                    }
                except Exception as e:
                    log.error("CLI input error", extra={"error": str(e)})
                
                return None
            else:
                # Wait for turn to be reset (happens after response is generated)
                await asyncio.sleep(0.5)  # Prevent tight loop
                return None
        
    async def _handle_mic_input(self) -> Optional[Dict[str, Any]]:
        """Handle microphone input via STT."""
        if not self.mic:
            return None
            
        try:
            # This would interface with the mic listener component
            audio_data = await self.mic.listen()
                
            if audio_data:
                return {
                        "content": audio_data.get("transcription", ""),
                        "metadata": {
                            "input_method": "microphone",
                            "confidence": audio_data.get("confidence", 0.0),
                            "duration": audio_data.get("duration", 0.0)
                        }
                    }
        except Exception as e:
            log.error("Microphone input error", extra={"error": str(e)})
            
        return None
        
    async def _handle_chat_input(self) -> Optional[Dict[str, Any]]:
        """Handle chat input from external platforms."""
        if not self.chat:
            return None
            
        try:
                # This would interface with chat listener components
            chat_message = await self.chat.get_message()
                
            if chat_message:
                return {
                        "content": chat_message.get("content", ""),
                        "metadata": {
                            "input_method": "chat",
                            "platform": chat_message.get("platform", "unknown"),
                            "user_id": chat_message.get("user_id", "unknown"),
                            "channel": chat_message.get("channel", "unknown")
                        }
                    }
        except Exception as e:
            log.error("Chat input error", extra={"error": str(e)})
            return None
    
    async def _handle_ui_input(self) -> Optional[Dict[str, Any]]:
        """Handle UI input from C# interface."""
        try:
            # This would interface with UI via IPC/sockets
            # Placeholder for now
            await asyncio.sleep(1.0)  # Prevent tight loop
            return None
        except Exception as e:
            log.error("UI input error", extra={"error": str(e)})
        
        return None
    
    async def _handle_api_input(self) -> Optional[Dict[str, Any]]:
        """Handle API input from external services."""
        try:
            # This would interface with external APIs
            # Placeholder for now
            await asyncio.sleep(1.0)  # Prevent tight loop
            return None
        except Exception as e:
            log.error("API input error", extra={"error": str(e)})
        
        return None
    
    def get_active_sources(self) -> List[str]:
        """Get list of currently active input sources."""
        return [name for name, _ in self.active_sources]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get input manager statistics."""
        return {
            "active_sources": len(self.active_sources),
            "source_names": self.get_active_sources(),
            "config": {
                "cli_enabled": self.enable_cli,
                "mic_enabled": self.enable_mic,
                "chat_enabled": self.enable_chat,
                "ui_enabled": self.enable_ui,
                "api_enabled": self.enable_api,
            }
        }
