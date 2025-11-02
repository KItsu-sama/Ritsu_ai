from __future__ import annotations

"""
ai/ai_assistant.py

AIAssistant — main AI logic coordination
- Processes natural language input
- Coordinates with NLP engine and knowledge base
- Generates responses and actions
- Handles conversation context
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import asyncio
import inspect

log = logging.getLogger(__name__)


class AIAssistant:
    """Main AI assistant that coordinates all AI components."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        nlp_engine=None,
        knowledge_base=None,
        memory_manager=None,
        llm_engine=None,
    ):
        self.config = config or {}
        self.nlp = nlp_engine
        self.kb = knowledge_base
        self.memory = memory_manager
        self.llm = llm_engine
        
        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}
        
        # Response modes
        self.response_modes = {
            "direct": self._direct_response,
            "analytical": self._analytical_response,
            "creative": self._creative_response,
            "tool_use": self._tool_response,
        }
    
    async def process_input(
        self,
        input_text: str,
        source: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Process user input and generate appropriate response.
        
        Args:
            input_text: The user's input
            source: Source of input (cli, mic, chat, etc.)
            context: Additional context information
            stream: Whether to stream the response
            
        Returns:
            Response string or async generator for streaming
        """
        try:
            log.debug("AIAssistant.process_input: start", extra={"source": source, "preview": input_text[:160]})
            # Update conversation history
            user_message = {
                "role": "user",
                "content": input_text,
                "source": source,
                "timestamp": asyncio.get_event_loop().time(),
                "context": context or {},
            }
            self.conversation_history.append(user_message)
            
            # Analyze input intent and extract key information
            analysis = await self._analyze_input(input_text, context)
            log.debug("AIAssistant.process_input: analysis", extra={"intent": analysis.get("intent"), "requires_tools": analysis.get("requires_tools")})
            
            # Determine response mode
            response_mode = self._determine_response_mode(analysis)
            
            # Generate response
            response_generator = self.response_modes[response_mode]
            
            if stream:
                return self._stream_response(response_generator, analysis)
            else:
                response = await self._generate_response(response_generator, analysis)
                
                # Update conversation history
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "mode": response_mode,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                self.conversation_history.append(assistant_message)
                
                # Store in memory if memory manager is available
                if self.memory:
                    await self.memory.store_interaction(user_message, assistant_message)
                
                return response
                
        except Exception as e:
            log.exception("Error processing input", extra={"input": input_text, "source": source})
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    async def _analyze_input(
        self, input_text: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze input to understand intent and extract key information."""
        analysis = {
            "text": input_text,
            "intent": "general",
            "entities": [],
            "sentiment": "neutral",
            "complexity": "simple",
            "requires_tools": False,
            "context": context or {},
        }
        
        # Use NLP engine if available
        if self.nlp:
            try:
                nlp_result = await self.nlp.analyze(input_text)
                analysis.update(nlp_result)
            except Exception as e:
                log.warning("NLP analysis failed", extra={"error": str(e)})
        else:
            # Basic fallback analysis
            analysis.update(self._basic_analysis(input_text))
        
        return analysis
    
    def _basic_analysis(self, text: str) -> Dict[str, Any]:
        """Basic text analysis when NLP engine is unavailable."""
        text_lower = text.lower()
        
        # Simple intent detection
        intent = "general"
        if any(word in text_lower for word in ["help", "how", "what", "explain"]):
            intent = "help"
        elif any(word in text_lower for word in ["create", "make", "generate", "write"]):
            intent = "creative"
        elif any(word in text_lower for word in ["analyze", "compare", "evaluate"]):
            intent = "analytical"
        elif any(word in text_lower for word in ["run", "execute", "calculate", "search"]):
            intent = "tool_use"
        
        # Simple sentiment analysis
        sentiment = "neutral"
        positive_words = ["good", "great", "excellent", "love", "like", "amazing"]
        negative_words = ["bad", "terrible", "hate", "dislike", "awful", "horrible"]
        
        if any(word in text_lower for word in positive_words):
            sentiment = "positive"
        elif any(word in text_lower for word in negative_words):
            sentiment = "negative"
        
        return {
            "intent": intent,
            "sentiment": sentiment,
            "complexity": "complex" if len(text.split()) > 20 else "simple",
            "requires_tools": intent == "tool_use",
        }
    
    def _determine_response_mode(self, analysis: Dict[str, Any]) -> str:
        """Determine the appropriate response mode based on analysis."""
        intent = analysis.get("intent", "general")
        
        if analysis.get("requires_tools", False):
            return "tool_use"
        elif intent == "analytical":
            return "analytical"
        elif intent == "creative":
            return "creative"
        else:
            return "direct"
    
    async def _generate_response(self, response_generator, analysis: Dict[str, Any]) -> str:
        """Generate a complete response."""
        try:
            log.debug("AIAssistant._generate_response: invoking generator", extra={"mode_preview": analysis.get("complexity", "?")})
            return await response_generator(analysis)
        except Exception as e:
            log.error("Response generation failed", extra={"error": str(e), "analysis": analysis})
            return "I'm sorry, I had trouble generating a response. Could you try rephrasing your question?"
    
    async def _stream_response(
        self, response_generator, analysis: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens."""
        try:
            # For now, generate full response then yield in chunks
            # TODO: Implement true streaming when LLM supports it
            response = await response_generator(analysis)
            
            # Yield response in chunks to simulate streaming
            chunk_size = 20  # words per chunk
            words = response.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                yield chunk + (" " if i + chunk_size < len(words) else "")
                await asyncio.sleep(0.1)  # Small delay between chunks
                
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    # Response mode implementations
    async def _direct_response(self, analysis: Dict[str, Any]) -> str:
        """Generate a direct, conversational response."""
        if self.llm:
            try:
                # Use LLM for response generation - RitsuLLM expects a dict
                text = analysis["text"]
                source = analysis["context"].get("source", "user")
                user_dict = {source: text}
                maybe = self.llm.generate_response(user_dict, mode=0)
                # Support both sync and async LLM interfaces
                if inspect.isawaitable(maybe):
                    response = await maybe
                else:
                    response = maybe
                return (response or "").strip()
            except Exception as e:
                log.warning("LLM generation failed, using fallback", extra={"error": str(e)})
        
        # Fallback response
        return self._fallback_response(analysis)
    
    async def _analytical_response(self, analysis: Dict[str, Any]) -> str:
        """Generate an analytical, detailed response."""
        if self.llm:
            try:
                text = analysis["text"]
                source = analysis["context"].get("source", "user")
                user_dict = {source: text}
                maybe = self.llm.generate_response(user_dict, mode=0)  # Professional mode
                if inspect.isawaitable(maybe):
                    response = await maybe
                else:
                    response = maybe
                return (response or "").strip()
            except Exception as e:
                log.warning("LLM generation failed, using fallback", extra={"error": str(e)})
        
        return "I'd be happy to help analyze that. However, I need my full AI capabilities to provide a detailed analysis. Please ensure the LLM is properly configured."
    
    async def _creative_response(self, analysis: Dict[str, Any]) -> str:
        """Generate a creative response."""
        if self.llm:
            try:
                text = analysis["text"]
                source = analysis["context"].get("source", "user")
                user_dict = {source: text}
                maybe = self.llm.generate_response(user_dict, mode=1)  # Creative mode
                if inspect.isawaitable(maybe):
                    response = await maybe
                else:
                    response = maybe
                return (response or "").strip()
            except Exception as e:
                log.warning("LLM generation failed, using fallback", extra={"error": str(e)})
        
        return "I'd love to help with creative tasks! However, I need my full AI capabilities for the best results. Please ensure the LLM is properly configured."
    
    async def _tool_response(self, analysis: Dict[str, Any]) -> str:
        """Handle tool use and command execution."""
        # This would integrate with the tools system
        return "Tool integration is not yet fully implemented. This would execute the requested action."
    
    def _build_prompt(self, analysis: Dict[str, Any], mode: str) -> str:
        """Build a prompt for the LLM based on analysis and mode."""
        base_prompt = f"You are Ritsu, a helpful AI assistant. "
        
        if mode == "analytical":
            base_prompt += "Provide a detailed, analytical response. "
        elif mode == "creative":
            base_prompt += "Be creative and engaging in your response. "
        elif mode == "direct":
            base_prompt += "Provide a clear, direct response. "
        
        # Add conversation history context
        if self.conversation_history:
            recent_history = self.conversation_history[-5:]  # Last 5 messages
            context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_history
            ])
            base_prompt += f"\n\nConversation context:\n{context}\n\n"
        
        base_prompt += f"User: {analysis['text']}\nAssistant:"
        
        return base_prompt
    
    def _fallback_response(self, analysis: Dict[str, Any]) -> str:
        """Generate a simple fallback response when LLM is unavailable."""
        intent = analysis.get("intent", "general")
        text = analysis["text"]
        
        responses = {
            "help": f"I understand you're asking for help with something. While I can see you mentioned '{text}', I need my full AI capabilities to provide detailed assistance.",
            "creative": f"That sounds like an interesting creative request! I'd love to help, but I need my full AI capabilities to generate creative content effectively.",
            "analytical": f"I can see you want me to analyze something. For detailed analysis, I need my full AI capabilities to be properly configured.",
            "general": f"I received your message: '{text}'. I'm functioning in basic mode right now. For full conversation capabilities, please ensure all AI components are properly configured.",
        }
        
        return responses.get(intent, responses["general"])
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:] if limit > 0 else self.conversation_history
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.current_context.clear()
        log.info("Conversation history cleared")
    
    async def close(self) -> None:
        """Cleanup resources."""
        if self.memory:
            await self.memory.close()
        log.info("AIAssistant closed")