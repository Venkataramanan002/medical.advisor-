"""Entry-point for the terminal medical chatbot."""

from __future__ import annotations

import logging
import textwrap

from ai_model import summarize_medical_pages
from scraper import available_sources, scrape_medical_info

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("medical_bot")


def medical_bot(user_input: str) -> str:
    pages_content = scrape_medical_info(user_input)

    if not pages_content:
        return (
            "I couldn't retrieve enough reliable information just now. "
            "Please try a different description of your symptoms or consult a "
            "licensed healthcare professional."
        )

    return summarize_medical_pages(pages_content, user_input)


def _print_welcome_message() -> None:
    intro = textwrap.dedent(
        """
        === Local Medical Assistant ===
        Ask about common symptoms and receive concise, practical guidance sourced
        from:
        - WebMD
        - Mayo Clinic

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
