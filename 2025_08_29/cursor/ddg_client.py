"""
DuckDuckGo Instant Answer API client.

Provides a typed function to query the Instant Answer endpoint and return a
normalized result object suitable for application consumption or CLI output.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict, Mapping

import requests


class InstantAnswerResult(TypedDict, total=False):
    """Normalized fields extracted from a DuckDuckGo Instant Answer response."""
    query: str
    abstract_text: Optional[str]
    abstract_source: Optional[str]
    abstract_url: Optional[str]
    results: List[Dict[str, Any]]
    related_topics: List[Dict[str, Any]]
    redirect: Optional[str]
    heading: Optional[str]
    raw: Dict[str, Any]


DDG_ENDPOINT = "https://api.duckduckgo.com/"

__all__ = [
    "InstantAnswerResult",
    "search_instant_answer",
    "search_first_text",
    "pretty_print",
]


class ResultItem(TypedDict, total=False):
    """Structure of an item in the Results array returned by the API."""
    Text: str
    Result: str


class RelatedTopicItem(TypedDict, total=False):
    """Structure of an item in RelatedTopics, possibly with nested Topics."""
    Text: str
    Topics: List[Dict[str, Any]]


def search_instant_answer(
    query: str,
    *,
    no_html: bool = True,
    skip_disambig: bool = True,
    timeout_seconds: float = 10.0,
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
) -> InstantAnswerResult:
    """
    Query DuckDuckGo Instant Answer API and return normalized result data.

    Parameters
    ----------
    query: str
        The search query.
    no_html: bool
        If True, the API response will exclude HTML in text fields.
    skip_disambig: bool
        If True, the API will skip disambiguation pages when possible.
    timeout_seconds: float
        HTTP request timeout in seconds.
    user_agent: str
        User-Agent header to send with the request.

    Returns
    -------
    InstantAnswerResult
        A dictionary with normalized fields and the raw response.
    """

    params = {
        "q": query,
        "format": "json",
        "no_html": "1" if no_html else "0",
        "skip_disambig": "1" if skip_disambig else "0",
    }

    headers = {"User-Agent": user_agent, "Accept": "application/json"}

    response = requests.get(DDG_ENDPOINT, params=params, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()

    data: Dict[str, Any] = response.json()

    result: InstantAnswerResult = {
        "query": query,
        "abstract_text": data.get("AbstractText") or None,
        "abstract_source": data.get("AbstractSource") or None,
        "abstract_url": data.get("AbstractURL") or None,
        "results": data.get("Results") or [],
        "related_topics": data.get("RelatedTopics") or [],
        "redirect": data.get("Redirect") or None,
        "heading": data.get("Heading") or None,
        "raw": data,
    }

    return result


def pretty_print(result: InstantAnswerResult) -> str:
    """
    Return a human-readable JSON string for display.
    """
    return json.dumps(result, indent=2, ensure_ascii=False)


def _normalize_text(value: Any) -> Optional[str]:
    """Return a stripped string if value is a non-empty string; otherwise None.

    Returns
    -------
    Optional[str]
        The stripped string when non-empty; otherwise None.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _first_results_text(raw: Mapping[str, Any]) -> Optional[str]:
    """Return text from the first item in Results, if present.

    Parameters
    ----------
    raw: Mapping[str, Any]
        The raw API response mapping.
    """
    results_list = raw.get("Results") or []
    if not isinstance(results_list, list) or not results_list:
        return None
    first_item = results_list[0]
    if not isinstance(first_item, dict):
        return None
    return _normalize_text(first_item.get("Text") or first_item.get("Result"))


def _first_related_topic_text(raw: Mapping[str, Any]) -> Optional[str]:
    """Return text from the first RelatedTopics item (including nested Topics).

    Parameters
    ----------
    raw: Mapping[str, Any]
        The raw API response mapping.
    """
    related = raw.get("RelatedTopics") or []
    if not isinstance(related, list):
        return None
    for item in related:
        if not isinstance(item, dict):
            continue
        direct_text = _normalize_text(item.get("Text"))
        if direct_text:
            return direct_text
        topics = item.get("Topics")
        if not isinstance(topics, list):
            continue
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            topic_text = _normalize_text(topic.get("Text"))
            if topic_text:
                return topic_text
    return None


def _extract_first_text(result: InstantAnswerResult) -> Optional[str]:
    """Extract the most relevant single text snippet from the Instant Answer.

    Preference order:
    1) Answer
    2) AbstractText
    3) Definition
    4) First item in Results[].Text
    5) First item in RelatedTopics[].Text (including nested Topics)
    """
    raw = result.get("raw", {}) or {}

    candidates: List[Optional[str]] = [
        _normalize_text(raw.get("Answer")),
        _normalize_text(result.get("abstract_text")),
        _normalize_text(raw.get("Definition")),
        _first_results_text(raw),
        _first_related_topic_text(raw),
    ]

    for text in candidates:
        if text:
            return text
    return None


def search_first_text(
    query: str,
    *,
    no_html: bool = True,
    skip_disambig: bool = True,
    timeout_seconds: float = 10.0,
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
) -> Optional[str]:
    """Return just the main text snippet for a query, or None if unavailable."""
    result = search_instant_answer(
        query,
        no_html=no_html,
        skip_disambig=skip_disambig,
        timeout_seconds=timeout_seconds,
        user_agent=user_agent,
    )
    return _extract_first_text(result)
