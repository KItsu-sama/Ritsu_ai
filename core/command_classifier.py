"""
command_classifier.py

Purpose:
---------
Classifies raw user input (commands, queries, events) into structured categories
that Ritsu (or any AI core) can understand and handle.

This file is designed to be highly modular and extensible, similar to Warp's
command classification system. It provides:
  - Rule-based intent detection
  - Regex and keyword matching with word boundaries
  - Optional ML/NLP-based classification hooks
  - Confidence scoring with thresholds
  - Logging for debugging
  - Error handling and fallback classification
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable


class ClassificationResult:
    """
    Standardized result for command classification.
    """
    def __init__(self,
                 intent: str,
                 category: str,
                 confidence: float,
                 metadata: Optional[Dict[str, Any]] = None):
        self.intent = intent              # e.g., "run_code", "query_ai", "system_event"
        self.category = category          # e.g., "executor", "planning", "troubleshooter"
        self.confidence = confidence      # 0.0 - 1.0
        self.metadata = metadata or {}    # extra context (regex matches, arguments, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "category": self.category,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

    def __repr__(self):
        return (f"<ClassificationResult intent={self.intent} "
                f"category={self.category} confidence={self.confidence:.2f}>")


class CommandClassifier:
    """
    Main classifier for detecting user intent and routing commands.

    Priority order:
      1. Hard-coded rules (regex, keywords)
      2. Heuristic scoring
      3. Optional ML/NLP model (hookable)
      4. Fallback (unknown command)
    """

    MIN_CONFIDENCE = 0.2  # Minimum confidence threshold for keyword classification

    def __init__(self, ml_classifier: Optional[Callable[[str], Optional[ClassificationResult]]] = None):
        # --- Define intent dictionaries ---
        self.keyword_map = {
            "run_code": ["run", "execute", "compile", "python", "script"],
            "system_event": ["cpu", "ram", "status", "system", "monitor"],
            "query_ai": ["explain", "what is", "tell me", "?", "why"],
            "planning": ["plan", "next step", "goal", "objective"],
            "troubleshooter": ["error", "fix", "problem", "bug", "debug", "issue"],
            "tool_call": ["search", "open", "calculate", "fetch"]
        }

        # Regex patterns (for more structured inputs like shell commands)
        self.regex_map = {
            "shell_command": re.compile(r"^(?P<cmd>[a-zA-Z0-9_\-\.\/]+)(\s+(?P<args>.*))?$"),
            "assignment": re.compile(r"^(?P<var>\w+)\s*=\s*(?P<value>.+)$"),
            "function_call": re.compile(r"^(?P<func>\w+)\((?P<args>.*)\)$"),
        }

        # Fallback category
        self.default_intent = "unknown"
        self.default_category = "unclassified"

        # ML/NLP classifier hook
        self.ml_classifier = ml_classifier

        # Logging
        self.logger = logging.getLogger("CommandClassifier")
        if not self.logger.hasHandlers():
            # Configure default logging if not already configured
            logging.basicConfig(level=logging.DEBUG)

    # -------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify raw text into a structured command.

        Args:
            text: The raw user input string.

        Returns:
            ClassificationResult with intent, category, confidence, and metadata.
        """
        text = text.strip()
        if not text:
            self.logger.debug("Input text is empty.")
            return self._make_result("empty", "unclassified", 0.0)

        self.logger.debug(f"Classifying input: {text}")

        # 1. Regex classification
        regex_result = self._check_regex(text)
        if regex_result:
            self.logger.debug(f"Regex matched: {regex_result}")
            return regex_result

        # 2. Keyword classification
        keyword_result = self._check_keywords(text)
        if keyword_result:
            self.logger.debug(f"Keyword matched: {keyword_result}")
            return keyword_result

        # 3. Optional ML/NLP hook
        ml_result = self._ml_classify(text)
        if ml_result:
            self.logger.debug(f"ML classified: {ml_result}")
            return ml_result

        # 4. Fallback
        self.logger.debug("Fallback classification")
        return self._make_result(self.default_intent, self.default_category, 0.1, {"raw": text})

    # -------------------------------------------------------
    # INTERNAL METHODS
    # -------------------------------------------------------

    def _check_regex(self, text: str) -> Optional[ClassificationResult]:
        """
        Detect structured patterns (e.g., shell commands, assignments).

        Returns:
            ClassificationResult if matched, else None.
        """
        for intent, pattern in self.regex_map.items():
            match = pattern.match(text)
            if match:
                metadata = match.groupdict()
                self.logger.debug(f"Regex '{intent}' matched with groups: {metadata}")
                return self._make_result(intent, "executor", 0.9, {"match": match.group(), **metadata})
        return None

    def _check_keywords(self, text: str) -> Optional[ClassificationResult]:
        """
        Simple keyword-based intent detection using word boundaries.

        Returns:
            ClassificationResult if confident match found, else None.
        """
        scores = {}
        lower_text = text.lower()

        for intent, keywords in self.keyword_map.items():
            match_count = 0
            for kw in keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, lower_text):
                    match_count += 1
            if match_count > 0:
                scores[intent] = match_count / len(keywords)

        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = scores[best_intent]

            if confidence < self.MIN_CONFIDENCE:
                self.logger.debug(f"Keyword matches below confidence threshold: {confidence:.2f}")
                return None  # too weak to classify

            # Handle ties by deterministic choice (alphabetical)
            best_intents = [k for k, v in scores.items() if v == confidence]
            chosen_intent = sorted(best_intents)[0]

            category = self._map_intent_to_category(chosen_intent)
            self.logger.debug(f"Keyword classification chosen intent: {chosen_intent} with confidence {confidence:.2f}")
            return self._make_result(chosen_intent, category, confidence, {"matches": scores})

        return None

    def _ml_classify(self, text: str) -> Optional[ClassificationResult]:
        """
        Placeholder for ML/NLP classification (e.g., HuggingFace model).
        In Warp-like systems, this can handle natural language queries.

        Returns:
            ClassificationResult if ML classifier is set and returns a result, else None.
        """
        if self.ml_classifier:
            try:
                result = self.ml_classifier(text)
                if result:
                    self.logger.debug(f"ML classifier returned: {result}")
                return result
            except Exception as e:
                self.logger.error(f"ML classifier error: {e}")
        return None

    def _map_intent_to_category(self, intent: str) -> str:
        """
        Maps intent to a broader category for routing.

        Args:
            intent: Intent string.

        Returns:
            Category string.
        """
        mapping = {
            "run_code": "executor",
            "system_event": "monitor",
            "query_ai": "assistant",
            "planning": "planner",
            "troubleshooter": "troubleshooter",
            "tool_call": "tools",
            "shell_command": "executor",
            "assignment": "executor",
            "function_call": "executor",
            "empty": "unclassified",
            "unknown": "unclassified"
        }
        return mapping.get(intent, self.default_category)

    def _make_result(self,
                     intent: str,
                     category: str,
                     confidence: float,
                     metadata: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Helper to create a standardized result.

        Args:
            intent: Intent string.
            category: Category string.
            confidence: Confidence score (0.0 - 1.0).
            metadata: Optional metadata dictionary.

        Returns:
            ClassificationResult instance.
        """
        return ClassificationResult(intent, category, confidence, metadata)


# Example usage (remove or comment out in production, keep for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    classifier = CommandClassifier()

    examples = [
        "run this python script",
        "cpu status check",
        "what is quantum computing?",
        "plan next step",
        "fix error in code",
        "x = 5",
        "ls -la",
        "",
        "running tests",
        "open file",
        "calculate 2 + 2"
    ]

    for e in examples:
        result = classifier.classify(e)
        print(f"Input: {e!r} -> {result}")
