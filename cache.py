"""Conversation cache for storing and retrieving medical Q&A pairs.

This module provides persistent storage for question-answer pairs, allowing
the chatbot to reuse previous responses for similar questions without
needing to scrape the web or run the AI model again.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

CACHE_FILE = Path("conversation_cache.json")
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score to consider questions similar


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and converting to lowercase."""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _calculate_similarity(query1: str, query2: str) -> float:
    """Calculate similarity between two queries using word overlap.
    
    Returns a score between 0 and 1, where 1 means identical questions.
    """
    words1 = set(_normalize_text(query1).split())
    words2 = set(_normalize_text(query2).split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    if union == 0:
        return 0.0
    
    similarity = intersection / union
    return similarity


def _extract_keywords(text: str) -> List[str]:
    """Extract important keywords from a query."""
    normalized = _normalize_text(text)
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'i', 'my', 'me', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they', 'them', 'this', 'that', 'these', 'those'}
    words = normalized.split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords


def _load_cache() -> List[Dict[str, str]]:
    """Load conversation cache from JSON file."""
    if not CACHE_FILE.exists():
        return []
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        return cache if isinstance(cache, list) else []
    except (json.JSONDecodeError, IOError) as exc:
        LOGGER.warning("Failed to load cache: %s", exc)
        return []


def _save_cache(cache: List[Dict[str, str]]) -> None:
    """Save conversation cache to JSON file."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        LOGGER.info("Cache saved with %d entries", len(cache))
    except IOError as exc:
        LOGGER.error("Failed to save cache: %s", exc)


def find_similar_question(user_query: str) -> Optional[Tuple[str, float]]:
    """Find a similar question in the cache.
    
    Returns:
        Tuple of (cached_answer, similarity_score) if a similar question is found,
        None otherwise.
    """
    cache = _load_cache()
    if not cache:
        return None
    
    best_match = None
    best_score = 0.0
    
    for entry in cache:
        cached_query = entry.get('question', '')
        similarity = _calculate_similarity(user_query, cached_query)
        
        if similarity > best_score and similarity >= SIMILARITY_THRESHOLD:
            best_score = similarity
            best_match = (entry.get('answer', ''), similarity)
    
    if best_match:
        LOGGER.info("Found similar question in cache (similarity: %.2f)", best_score)
        return best_match
    
    return None


def save_conversation(question: str, answer: str) -> None:
    """Save a question-answer pair to the cache."""
    cache = _load_cache()
    
    # Check if this exact question already exists
    normalized_question = _normalize_text(question)
    for entry in cache:
        if _normalize_text(entry.get('question', '')) == normalized_question:
            # Update existing entry
            entry['answer'] = answer
            entry['timestamp'] = datetime.now().isoformat()
            entry['count'] = entry.get('count', 0) + 1
            _save_cache(cache)
            LOGGER.info("Updated existing cache entry")
            return
    
    # Add new entry
    new_entry = {
        'question': question,
        'answer': answer,
        'timestamp': datetime.now().isoformat(),
        'count': 1,
        'keywords': _extract_keywords(question)
    }
    cache.append(new_entry)
    
    # Limit cache size to prevent unbounded growth (keep last 1000 entries)
    if len(cache) > 1000:
        # Sort by timestamp and keep most recent
        cache.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        cache = cache[:1000]
    
    _save_cache(cache)
    LOGGER.info("Saved new conversation to cache")


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about the cache."""
    cache = _load_cache()
    return {
        'total_entries': len(cache),
        'total_queries': sum(entry.get('count', 1) for entry in cache)
    }


def clear_cache() -> None:
    """Clear the entire conversation cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        LOGGER.info("Cache cleared")
