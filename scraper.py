"""Utilities for retrieving trusted medical content locally.

The scraper fetches articles from a small curated list of URLs rather than
performing arbitrary web searches. This keeps the behaviour deterministic and
avoids violating the terms of service of major search providers.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, Iterable, List

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

# Curated list of trusted URLs grouped by common symptom keywords.
TRUSTED_SOURCES: Dict[str, List[str]] = {
    "cold": [
        "https://www.webmd.com/cold-and-flu/cold-guide/common_cold_overview",
        "https://www.mayoclinic.org/diseases-conditions/common-cold/symptoms-causes/syc-20351605",
    ],
    "flu": [
        "https://www.webmd.com/cold-and-flu/flu-guide/what-is-flu",
        "https://www.mayoclinic.org/diseases-conditions/flu/symptoms-causes/syc-20351719",
    ],
    "covid": [
        "https://www.webmd.com/covid/covid-19-symptoms-and-treatment",
        "https://www.mayoclinic.org/diseases-conditions/coronavirus/in-depth/coronavirus-symptoms/art-20483365",
    ],
    "anxiety": [
        "https://www.mayoclinic.org/diseases-conditions/anxiety/symptoms-causes/syc-20350961",
        "https://www.webmd.com/anxiety-panic/guide/anxiety-disorders",
    ],
    "diabetes": [
        "https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193",
        "https://www.webmd.com/diabetes/default.htm",
    ],
}

DEFAULT_URLS: List[str] = [url for urls in TRUSTED_SOURCES.values() for url in urls]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def _match_symptom_urls(user_query: str) -> List[str]:
    """Return relevant URLs based on keyword matches in the user query."""

    normalized = user_query.lower()
    matched_urls: List[str] = []
    for keyword, urls in TRUSTED_SOURCES.items():
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            matched_urls.extend(urls)

    # Fallback to the complete list if nothing matched.
    return matched_urls or DEFAULT_URLS


@lru_cache(maxsize=32)
def fetch_page_content(url: str) -> str:
    """Download and clean a medical article.

    Results are cached to avoid repeated network calls within a single session.
    """

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network variability
        LOGGER.warning("Failed to fetch %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    cleaned = " ".join(
        normalize_whitespace(p.get_text(separator=" ")) for p in paragraphs
    )
    # Limit the payload so the summariser remains quick.
    return cleaned[:4000]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def scrape_medical_info(user_query: str, max_pages: int = 3) -> List[str]:
    """Retrieve content for the query from trusted sources.

    Parameters
    ----------
    user_query:
        Free-form question provided by the user.
    max_pages:
        Safety limit for the number of pages to fetch.
    """

    urls_to_fetch = _match_symptom_urls(user_query)[:max_pages]
    pages: List[str] = []

    for url in urls_to_fetch:
        content = fetch_page_content(url)
        if content:
            pages.append(content)

    return pages


def available_sources() -> Iterable[str]:
    """Expose the list of URLs so the UI can surface them."""

    seen = set()
    for url in DEFAULT_URLS:
        if url not in seen:
            seen.add(url)
            yield url
