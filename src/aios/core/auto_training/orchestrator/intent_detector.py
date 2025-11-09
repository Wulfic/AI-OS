"""
Intent detection for learning requests.

Detects when users express learning intent and extracts domain/categories.
"""

import re
from typing import List, Optional

from .models import LearningIntent


class IntentDetector:
    """Detects learning intent from user messages.
    
    Uses pattern matching and keyword analysis to identify when a user
    wants the system to learn something new. Extracts domain and categories
    from the message.
    
    Example:
        >>> detector = IntentDetector()
        >>> intent = detector.detect("Learn Python programming")
        >>> print(intent.domain)  # "coding"
        >>> print(intent.categories)  # ["python", "programming"]
    """
    
    # Patterns that indicate learning intent
    LEARNING_PATTERNS = [
        r"learn (?:about )?(.+)",
        r"teach (?:me |yourself )?(?:about )?(.+)",
        r"study (.+)",
        r"I want to (?:learn|understand) (.+)",
        r"help me (?:learn|understand) (.+)",
        r"get better at (.+)",
        r"master (.+)",
        r"understand (.+)",
        r"train (?:on|yourself on) (.+)",
        r"practice (.+)",
    ]
    
    # Domain keywords for classification
    DOMAIN_KEYWORDS = {
        "coding": [
            "python", "javascript", "java", "c++", "code", "programming",
            "software", "algorithm", "data structure", "web development",
            "api", "backend", "frontend", "database", "sql", "git",
        ],
        "math": [
            "math", "calculus", "algebra", "geometry", "statistics",
            "trigonometry", "linear algebra", "differential", "integral",
            "probability", "number theory", "arithmetic",
        ],
        "writing": [
            "writing", "creative writing", "story", "essay", "literature",
            "poetry", "narrative", "fiction", "non-fiction", "journalism",
            "technical writing", "copywriting",
        ],
        "science": [
            "physics", "chemistry", "biology", "science", "astronomy",
            "ecology", "geology", "botany", "zoology", "genetics",
            "molecular", "quantum", "thermodynamics",
        ],
        "general": [
            "knowledge", "encyclopedia", "wiki", "facts", "trivia",
            "general knowledge", "world history", "geography",
        ],
    }
    
    # Category extraction keywords
    CATEGORY_KEYWORDS = {
        # Coding
        "python": ["python", "py"],
        "javascript": ["javascript", "js", "node"],
        "java": ["java"],
        "programming": ["programming", "code", "coding", "software"],
        
        # Math
        "calculus": ["calculus", "derivative", "integral"],
        "algebra": ["algebra", "equation"],
        "geometry": ["geometry", "shape", "triangle"],
        "statistics": ["statistics", "stats", "probability"],
        
        # Writing
        "creative_writing": ["creative writing", "story", "fiction"],
        "essay": ["essay", "article"],
        "poetry": ["poetry", "poem"],
        
        # Science
        "physics": ["physics", "mechanics", "quantum"],
        "chemistry": ["chemistry", "chemical", "molecule"],
        "biology": ["biology", "cell", "organism"],
    }
    
    def detect(self, message: str) -> Optional[LearningIntent]:
        """Detect learning intent from a message.
        
        Args:
            message: User message to analyze
        
        Returns:
            LearningIntent if detected, None otherwise
        """
        message_lower = message.lower().strip()
        
        # Try each pattern
        for pattern in self.LEARNING_PATTERNS:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                
                # Extract domain and categories
                domain = self._extract_domain(topic)
                categories = self._extract_categories(topic, domain)
                
                # Generate description
                description = f"Learn {topic}"
                
                # Calculate confidence
                confidence = self._calculate_confidence(message_lower, topic, domain)
                
                return LearningIntent(
                    domain=domain,
                    categories=categories,
                    description=description,
                    confidence=confidence,
                    raw_message=message,
                    extracted_topic=topic,
                )
        
        return None
    
    def _extract_domain(self, topic: str) -> str:
        """Extract domain from topic.
        
        Args:
            topic: Topic string
        
        Returns:
            Domain name (coding, math, writing, science, general)
        """
        topic_lower = topic.lower()
        
        # Count keyword matches per domain
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in topic_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def _extract_categories(self, topic: str, domain: str) -> List[str]:
        """Extract categories from topic.
        
        Args:
            topic: Topic string
            domain: Detected domain
        
        Returns:
            List of category names
        """
        topic_lower = topic.lower()
        categories = []
        
        # Find matching categories
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in topic_lower for keyword in keywords):
                categories.append(category)
        
        # If no categories found, use domain as category
        if not categories:
            categories = [domain]
        
        return categories
    
    def _calculate_confidence(self, message: str, topic: str, domain: str) -> float:
        """Calculate confidence score for intent detection.
        
        Args:
            message: Original message
            topic: Extracted topic
            domain: Detected domain
        
        Returns:
            Confidence score 0.0-1.0
        """
        confidence = 0.5  # Base confidence
        
        # Boost if domain keywords present
        domain_keywords = self.DOMAIN_KEYWORDS.get(domain, [])
        if any(keyword in message for keyword in domain_keywords):
            confidence += 0.2
        
        # Boost if topic is substantial (>2 words)
        if len(topic.split()) > 2:
            confidence += 0.1
        
        # Boost if strong learning verbs present
        strong_verbs = ["learn", "master", "study", "understand"]
        if any(verb in message for verb in strong_verbs):
            confidence += 0.1
        
        return min(confidence, 1.0)
