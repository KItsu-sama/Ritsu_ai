from __future__ import annotations

"""
ai/knowledge_base.py

KnowledgeBase — structured facts/skills
- Store and retrieve factual information
- Skill and capability tracking
- Context-aware information retrieval
- Knowledge graph relationships
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)


class KnowledgeBase:
    """Knowledge base for storing and retrieving structured information."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.kb_path = Path(self.config.get("knowledge_base_path", "data/knowledge_base.json"))
        
        # Knowledge storage
        self.facts: Dict[str, Dict[str, Any]] = {}
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, List[str]] = {}
        self.tags: Dict[str, Set[str]] = {}
        
        # Load existing knowledge
        self._load_knowledge()
    
    def _load_knowledge(self) -> None:
        """Load knowledge from persistent storage."""
        if not self.kb_path.exists():
            log.info("Knowledge base file not found, starting fresh")
            self._initialize_default_knowledge()
            return
        
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.facts = data.get("facts", {})
            self.skills = data.get("skills", {})
            self.relationships = data.get("relationships", {})
            # Convert tag sets back from lists
            self.tags = {
                tag: set(items) for tag, items in data.get("tags", {}).items()
            }
            
            log.info("Loaded knowledge base", extra={
                "facts_count": len(self.facts),
                "skills_count": len(self.skills),
                "relationships_count": len(self.relationships)
            })
            
        except Exception as e:
            log.error("Failed to load knowledge base", extra={"error": str(e)})
            self._initialize_default_knowledge()
    
    def _save_knowledge(self) -> None:
        """Save knowledge to persistent storage."""
        try:
            self.kb_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert tag sets to lists for JSON serialization
            tags_serializable = {
                tag: list(items) for tag, items in self.tags.items()
            }
            
            data = {
                "facts": self.facts,
                "skills": self.skills,
                "relationships": self.relationships,
                "tags": tags_serializable,
            }
            
            with open(self.kb_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            log.debug("Saved knowledge base", extra={"path": str(self.kb_path)})
            
        except Exception as e:
            log.error("Failed to save knowledge base", extra={"error": str(e)})
    
    def _initialize_default_knowledge(self) -> None:
        """Initialize with default knowledge and skills."""
        # Basic facts about Ritsu
        self.add_fact(
            "ritsu_identity",
            {
                "name": "Ritsu",
                "type": "AI Assistant",
                "capabilities": ["conversation", "text_analysis", "code_assistance", "creative_writing"],
                "personality": "helpful, curious, adaptive",
                "version": "0.1.0"
            },
            tags=["identity", "self"]
        )
        
        # Basic skills
        self.add_skill(
            "text_analysis",
            {
                "description": "Analyze text for intent, sentiment, and entities",
                "confidence": 0.8,
                "prerequisites": [],
                "outputs": ["intent", "sentiment", "entities", "keywords"]
            },
            tags=["nlp", "analysis"]
        )
        
        self.add_skill(
            "conversation",
            {
                "description": "Engage in natural conversation with users",
                "confidence": 0.9,
                "prerequisites": ["text_analysis"],
                "outputs": ["response", "questions", "suggestions"]
            },
            tags=["communication", "interaction"]
        )
        
        # Programming knowledge
        self.add_fact(
            "python_programming",
            {
                "language": "Python",
                "paradigms": ["object-oriented", "functional", "procedural"],
                "common_libraries": ["asyncio", "pathlib", "json", "logging"],
                "best_practices": ["PEP 8", "type hints", "error handling"]
            },
            tags=["programming", "python", "development"]
        )
        
        self._save_knowledge()
    
    def add_fact(self, fact_id: str, data: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        """Add a new fact to the knowledge base.
        
        Args:
            fact_id: Unique identifier for the fact
            data: Fact data dictionary
            tags: Optional tags for categorization
        """
        self.facts[fact_id] = {
            "id": fact_id,
            "data": data,
            "type": "fact",
            "created_at": self._get_timestamp(),
            "updated_at": self._get_timestamp()
        }
        
        if tags:
            for tag in tags:
                if tag not in self.tags:
                    self.tags[tag] = set()
                self.tags[tag].add(fact_id)
        
        log.debug("Added fact", extra={"fact_id": fact_id, "tags": tags})
        self._save_knowledge()
    
    def add_skill(self, skill_id: str, data: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        """Add a new skill to the knowledge base.
        
        Args:
            skill_id: Unique identifier for the skill
            data: Skill data dictionary
            tags: Optional tags for categorization
        """
        self.skills[skill_id] = {
            "id": skill_id,
            "data": data,
            "type": "skill",
            "created_at": self._get_timestamp(),
            "updated_at": self._get_timestamp()
        }
        
        if tags:
            for tag in tags:
                if tag not in self.tags:
                    self.tags[tag] = set()
                self.tags[tag].add(skill_id)
        
        log.debug("Added skill", extra={"skill_id": skill_id, "tags": tags})
        self._save_knowledge()
    
    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific fact.
        
        Args:
            fact_id: ID of the fact to retrieve
            
        Returns:
            Fact data or None if not found
        """
        return self.facts.get(fact_id)
    
    def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific skill.
        
        Args:
            skill_id: ID of the skill to retrieve
            
        Returns:
            Skill data or None if not found
        """
        return self.skills.get(skill_id)
    
    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Search for facts and skills by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching items
        """
        results = []
        
        if tag in self.tags:
            for item_id in self.tags[tag]:
                if item_id in self.facts:
                    results.append(self.facts[item_id])
                elif item_id in self.skills:
                    results.append(self.skills[item_id])
        
        return results
    
    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search for facts and skills containing a keyword.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of matching items
        """
        results = []
        keyword_lower = keyword.lower()
        
        # Search in facts
        for fact in self.facts.values():
            if self._contains_keyword(fact, keyword_lower):
                results.append(fact)
        
        # Search in skills
        for skill in self.skills.values():
            if self._contains_keyword(skill, keyword_lower):
                results.append(skill)
        
        return results
    
    def _contains_keyword(self, item: Dict[str, Any], keyword: str) -> bool:
        """Check if an item contains a keyword."""
        # Convert item to string and search
        item_str = json.dumps(item, default=str).lower()
        return keyword in item_str
    
    def add_relationship(self, item1: str, item2: str, relationship_type: str = "related") -> None:
        """Add a relationship between two items.
        
        Args:
            item1: First item ID
            item2: Second item ID
            relationship_type: Type of relationship
        """
        if item1 not in self.relationships:
            self.relationships[item1] = []
        if item2 not in self.relationships:
            self.relationships[item2] = []
        
        # Add bidirectional relationship
        relationship1 = f"{relationship_type}:{item2}"
        relationship2 = f"{relationship_type}:{item1}"
        
        if relationship1 not in self.relationships[item1]:
            self.relationships[item1].append(relationship1)
        if relationship2 not in self.relationships[item2]:
            self.relationships[item2].append(relationship2)
        
        log.debug("Added relationship", extra={
            "item1": item1, "item2": item2, "type": relationship_type
        })
        self._save_knowledge()
    
    def get_related_items(self, item_id: str) -> List[str]:
        """Get items related to the given item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            List of related item IDs
        """
        if item_id not in self.relationships:
            return []
        
        # Extract item IDs from relationship strings
        related = []
        for rel in self.relationships[item_id]:
            if ":" in rel:
                _, related_id = rel.split(":", 1)
                related.append(related_id)
        
        return related
    
    def get_capabilities(self) -> List[str]:
        """Get list of known capabilities."""
        capabilities = []
        
        # Extract capabilities from skills
        for skill in self.skills.values():
            skill_data = skill.get("data", {})
            if "description" in skill_data:
                capabilities.append(skill_data["description"])
        
        # Extract capabilities from identity fact
        identity = self.get_fact("ritsu_identity")
        if identity and "data" in identity:
            caps = identity["data"].get("capabilities", [])
            capabilities.extend(caps)
        
        return capabilities
    
    def update_skill_confidence(self, skill_id: str, confidence: float) -> None:
        """Update the confidence level of a skill.
        
        Args:
            skill_id: ID of the skill
            confidence: confidence level (0.0 to 1.0)
        """
        if skill_id in self.skills:
            self.skills[skill_id]["data"]["confidence"] = max(0.0, min(1.0, confidence))
            self.skills[skill_id]["updated_at"] = self._get_timestamp()
            self._save_knowledge()
            log.debug("Updated skill confidence", extra={
                "skill_id": skill_id, "confidence": confidence
            })
    
    def get_all_tags(self) -> List[str]:
        """Get all available tags."""
        return list(self.tags.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "facts_count": len(self.facts),
            "skills_count": len(self.skills),
            "tags_count": len(self.tags),
            "relationships_count": len(self.relationships),
            "total_items": len(self.facts) + len(self.skills)
        }
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    async def close(self) -> None:
        """Cleanup and save knowledge base."""
        self._save_knowledge()
        log.info("Knowledge base closed")