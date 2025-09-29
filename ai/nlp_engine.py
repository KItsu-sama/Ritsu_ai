from __future__ import annotations

"""
ai/nlp_engine.py

NLPEngine — intent detection, embeddings
- Text preprocessing and tokenization
- Intent classification
- Entity extraction
- Sentiment analysis
- Semantic similarity
"""

import re
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
import functools
from dataclasses import dataclass
import math

log = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Optimized data structure for NLP results."""
    intent: str
    entities: List[Dict[str, Any]]
    sentiment: str
    confidence: float
    complexity: str
    keywords: List[str]
    requires_tools: bool
    processing_time: float

class OptimizedNLPEngine:
    """Highly optimized NLP engine with performance enhancements."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._setup_patterns()
        self._setup_sentiment_lexicon()
        self._cache = {}  # Simple cache for frequent queries
        self._cache_max_size = 1000
        
    def _setup_patterns(self):
        """Pre-compile all regex patterns for maximum performance."""
        # Intent patterns with optimized regex
        self.intent_patterns = {
            "greeting": re.compile(
                r'\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b',
                re.IGNORECASE
            ),
            "goodbye": re.compile(
                r'\b(bye|goodbye|farewell|see\s+you|talk\s+to\s+you\s+later)\b',
                re.IGNORECASE
            ),
            "question": re.compile(
                r'\b(what|how|when|where|why|who|which)\b.*\?|\?$',
                re.IGNORECASE
            ),
            "help": re.compile(
                r'\b(help|assist|support|can\s+you|how\s+do\s+I)\b',
                re.IGNORECASE
            ),
            "creative": re.compile(
                r'\b(create|make|generate|write|compose).*\b(story|poem|song|article|code)\b',
                re.IGNORECASE
            ),
            "analytical": re.compile(
                r'\b(analyze|compare|evaluate|assess|review|pros\s+and\s+cons)\b',
                re.IGNORECASE
            ),
            "command": re.compile(
                r'^(run|execute|start|stop|restart|install|download|update)\b',
                re.IGNORECASE
            ),
        }
        
        # Combined entity patterns for single pass extraction
        self.entity_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://[^\s]+',
            "phone": r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            "number": r'\b\d+(?:\.\d+)?\b',
        }
        
        # Single compiled pattern for all entities
        entity_patterns_combined = '|'.join(
            f'(?P<{name}>{pattern})' for name, pattern in self.entity_patterns.items()
        )
        self.combined_entity_pattern = re.compile(entity_patterns_combined, re.IGNORECASE)
        
        # Pre-compiled preprocessing patterns
        self.whitespace_pattern = re.compile(r'\s+')
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s.,!?\'-]')
        self.math_pattern = re.compile(r'\d+\s*[+\-*/]\s*\d+')
        
    def _setup_sentiment_lexicon(self):
        """Optimize sentiment analysis with bloom-filter like approach."""
        self.positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "enjoy", "happy", "pleased", "satisfied",
            "perfect", "awesome", "brilliant", "outstanding", "superb"
        }
        
        self.negative_words = {
            "bad", "terrible", "awful", "horrible", "hate", "dislike",
            "sad", "angry", "frustrated", "disappointed", "annoyed",
            "worst", "disgusting", "pathetic", "useless", "broken"
        }
        
        # Create word masks for faster lookup
        self._word_masks = self._create_word_masks()
        
        # Common words for keyword extraction
        self.common_words = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through",
            "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }
        
        # Tool indicators as frozenset for O(1) lookups
        self.tool_indicators = frozenset({
            "run", "execute", "start", "stop", "install", "download",
            "calculate", "compute", "search", "find", "lookup",
            "save", "load", "open", "close", "delete"
        })
    
    def _create_word_masks(self):
        """Create hash masks for faster word lookups."""
        return {
            'positive': frozenset(self.positive_words),
            'negative': frozenset(self.negative_words)
        }
    
    @functools.lru_cache(maxsize=1000)
    def _preprocess(self, text: str) -> str:
        """Cached text preprocessing."""
        if not text:
            return ""
            
        # Single pass preprocessing
        text = text.lower()
        text = self.whitespace_pattern.sub(' ', text)
        text = self.clean_pattern.sub('', text)
        return text.strip()
    
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Highly optimized NLP analysis."""
        start_time = time.perf_counter()
        
        # Cache lookup
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Single preprocessing pass
            clean_text = self._preprocess(text)
            
            # Parallelizable operations
            intent_task = self._detect_intent_optimized(clean_text)
            entities_task = self._extract_entities_optimized(text)
            sentiment_task = self._analyze_sentiment_optimized(clean_text)
            features_task = self._extract_features_optimized(clean_text)
            
            # Combine results
            intent = intent_task
            entities = entities_task
            sentiment = sentiment_task
            features = features_task
            
            result = AnalysisResult(
                intent=intent,
                entities=entities,
                sentiment=sentiment,
                confidence=features["confidence"],
                complexity=features["complexity"],
                keywords=features["keywords"],
                requires_tools=self._requires_tools_optimized(clean_text, intent),
                processing_time=time.perf_counter() - start_time
            )
            
            # Cache result
            self._manage_cache(cache_key, result)
            
            return self._result_to_dict(result)
            
        except Exception as e:
            log.error("NLP analysis failed", extra={"text": text, "error": str(e)})
            return self._get_fallback_result()
    
    def _detect_intent_optimized(self, text: str) -> str:
        """Optimized intent detection using single-pass pattern matching."""
        if not text:
            return "general"
            
        # Quick checks for common patterns
        if '?' in text:
            return "question"
        
        # Single pass through all patterns
        best_intent = "general"
        max_score = 0
        
        for intent, pattern in self.intent_patterns.items():
            matches = pattern.findall(text)
            score = len(matches)
            
            if score > max_score:
                max_score = score
                best_intent = intent
        
        return best_intent
    
    def _extract_entities_optimized(self, text: str) -> List[Dict[str, Any]]:
        """Single-pass entity extraction."""
        entities = []
        
        for match in self.combined_entity_pattern.finditer(text):
            for name, value in match.groupdict().items():
                if value:
                    entities.append({
                        "type": name,
                        "value": value,
                        "start": match.start(),
                        "end": match.end(),
                    })
                    break  # Only one group matches per position
        
        return entities
    
    def _analyze_sentiment_optimized(self, text: str) -> str:
        """Optimized sentiment analysis using set operations."""
        words = set(text.split())
        
        positive_count = len(words & self._word_masks['positive'])
        negative_count = len(words & self._word_masks['negative'])
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_features_optimized(self, text: str) -> Dict[str, Any]:
        """Optimized feature extraction."""
        if not text:
            return self._get_empty_features()
            
        words = text.split()
        word_count = len(words)
        
        # Complexity classification
        if word_count > 50:
            complexity = "complex"
        elif word_count > 20:
            complexity = "medium"
        else:
            complexity = "simple"
        
        # Confidence scoring with smoothing
        confidence = min(1.0, math.log(word_count + 1) / 3.0)
        
        # Efficient keyword extraction
        keywords = [
            word for word in words
            if len(word) > 3 and word not in self.common_words
        ][:10]
        
        return {
            "word_count": word_count,
            "complexity": complexity,
            "confidence": round(confidence, 2),
            "keywords": keywords,
        }
    
    def _requires_tools_optimized(self, text: str, intent: str) -> bool:
        """Optimized tool requirement detection."""
        words = set(text.split())
        
        # Fast set intersection
        if words & self.tool_indicators:
            return True
            
        if intent == "command":
            return True
            
        return bool(self.math_pattern.search(text))
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """Optimized similarity calculation using Jaccard index."""
        words1 = set(self._preprocess(text1).split())
        words2 = set(self._preprocess(text2).split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union else 0.0
    
    def _manage_cache(self, key: int, value: Any):
        """Simple cache management."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value
    
    def _result_to_dict(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            "intent": result.intent,
            "entities": result.entities,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "complexity": result.complexity,
            "keywords": result.keywords,
            "requires_tools": result.requires_tools,
            "processing_time_ms": round(result.processing_time * 1000, 2)
        }
    
    def _get_fallback_result(self) -> Dict[str, Any]:
        """Return fallback result on error."""
        return {
            "intent": "general",
            "entities": [],
            "sentiment": "neutral",
            "confidence": 0.0,
            "complexity": "simple",
            "keywords": [],
            "requires_tools": False,
            "processing_time_ms": 0.0
        }
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty feature set."""
        return {
            "word_count": 0,
            "complexity": "simple",
            "confidence": 0.0,
            "keywords": [],
        }

# Performance benchmarking utility
class NLPBenchmark:
    """Utility for benchmarking NLP engine performance."""
    
    @staticmethod
    def benchmark_engine(engine: OptimizedNLPEngine, texts: List[str], iterations: int = 1000):
        """Benchmark the NLP engine performance."""
        import asyncio
        import time
        
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            for text in texts:
                asyncio.run(engine.analyze(text))
        
        total_time = time.perf_counter() - start_time
        avg_time = (total_time / (len(texts) * iterations)) * 1000
        
        print(f"Average processing time: {avg_time:.2f}ms")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Processed {len(texts) * iterations} texts")