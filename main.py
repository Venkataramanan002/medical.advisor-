"""Entry-point for the terminal medical chatbot."""

from __future__ import annotations

import logging
import textwrap

from ai_model import summarize_medical_pages
from cache import find_similar_question, get_cache_stats, save_conversation
from scraper import available_sources, scrape_medical_info

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("medical_bot")


def medical_bot(user_input: str, use_cache: bool = True) -> str:
    """Process medical query with caching support.
    
    Parameters
    ----------
    user_input:
        The user's medical question or symptom description.
    use_cache:
        Whether to check cache before scraping (default: True).
    
    Returns
    -------
    str:
        The bot's response to the user's query.
    """
    # Check cache first if enabled
    if use_cache:
        cached_result = find_similar_question(user_input)
        if cached_result:
            cached_answer, similarity = cached_result
            LOGGER.info("Using cached answer (similarity: %.2f%%)", similarity * 100)
            return cached_answer
    
    # Cache miss or cache disabled - scrape and generate new answer
    LOGGER.info("Cache miss - scraping web and generating new answer")
    pages_content = scrape_medical_info(user_input)

    if not pages_content:
        error_message = (
            "I couldn't retrieve enough reliable information just now. "
            "Please try a different description of your symptoms or consult a "
            "licensed healthcare professional."
        )
        return error_message

    answer = summarize_medical_pages(pages_content, user_input)
    
    # Save to cache for future use
    if use_cache and answer:
        save_conversation(user_input, answer)
    
    return answer


def _print_welcome_message() -> None:
    cache_stats = get_cache_stats()
    intro = textwrap.dedent(
        f"""
        === Local Medical Assistant ===
        Ask about common symptoms and receive concise, practical guidance sourced
        from:
        - WebMD
        - Mayo Clinic

        Cache: {cache_stats['total_entries']} saved conversations
        Commands: type 'exit' or 'quit' to stop.
        """
    ).strip()
    print(intro)


def _print_sources() -> None:
    print("\nTrusted sources:")
    for url in available_sources():
        print(f"  - {url}")
    print()


def main() -> None:
    _print_welcome_message()
    _print_sources()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - user interaction
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Take care!")
            break

        if not user_input:
            print("Bot: Please describe your symptoms or question.")
            continue

        LOGGER.info("Received query: %s", user_input)
        answer = medical_bot(user_input)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
