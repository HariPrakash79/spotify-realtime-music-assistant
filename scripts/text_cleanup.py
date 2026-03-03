#!/usr/bin/env python3
"""
Shared text cleanup helpers for music metadata.

Goal:
- Normalize null-like values.
- Repair common UTF-8 mojibake artifacts (e.g. "NaÃ¯ve" -> "Naïve").
"""

from __future__ import annotations

import re
from typing import Optional


NULL_LIKE = {"nan", "none", "null"}

MOJIBAKE_TOKENS = (
    "Ã",
    "Â",
    "â€™",
    "â€˜",
    "â€œ",
    "â€\x9d",
    "â€“",
    "â€”",
    "â€¦",
    "â€¢",
    "\ufffd",
)

COMMON_REPLACEMENTS = {
    "â€™": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€\x9d": "”",
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "â€¢": "•",
}

_CONTRACTION_SUFFIXES = {"s", "re", "ve", "ll", "d", "t", "m"}
_CONTRACTION_PATTERN = re.compile(r"\b([A-Za-z]+)'([A-Za-z]{1,3})\b")


def _badness_score(text: str) -> int:
    score = text.count("\ufffd")
    for token in MOJIBAKE_TOKENS:
        score += text.count(token)
    return score


def _looks_mojibake(text: str) -> bool:
    return any(token in text for token in MOJIBAKE_TOKENS)


def _try_utf8_redecode(text: str) -> str:
    """
    Try common "decoded as latin-1/cp1252 instead of utf-8" recovery.
    """
    best = text
    best_score = _badness_score(text)
    for enc in ("latin1", "cp1252"):
        try:
            candidate = text.encode(enc, errors="strict").decode("utf-8", errors="strict")
        except Exception:
            continue
        candidate_score = _badness_score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


def fix_mojibake(text: str) -> str:
    fixed = text
    # First pass: common direct token replacements.
    for src, dst in COMMON_REPLACEMENTS.items():
        fixed = fixed.replace(src, dst)

    # "Â" is usually a broken non-breaking-space marker.
    fixed = fixed.replace("Â ", " ").replace("Â", "")

    # Second pass: re-decode if it still looks broken.
    if _looks_mojibake(fixed):
        # Two iterations handles nested cases without being too aggressive.
        for _ in range(2):
            redone = _try_utf8_redecode(fixed)
            if redone == fixed:
                break
            fixed = redone
            if not _looks_mojibake(fixed):
                break
    return fixed


def clean_text(value: object, *, repair_mojibake: bool = True) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.lower() in NULL_LIKE:
        return None

    if repair_mojibake:
        text = fix_mojibake(text).strip()
        if not text:
            return None

    return text


def normalize_apostrophe_case(text: str) -> str:
    """
    Normalize common contraction casing:
      You'Re -> You're
      She'S  -> She's
      I'M    -> I'm
    """

    def _replace(match: re.Match[str]) -> str:
        stem = match.group(1)
        suffix = match.group(2)
        suffix_l = suffix.lower()
        if suffix_l in _CONTRACTION_SUFFIXES:
            return f"{stem}'{suffix_l}"
        return match.group(0)

    return _CONTRACTION_PATTERN.sub(_replace, text)


def clean_display_text(value: object, *, repair_mojibake: bool = True) -> Optional[str]:
    text = clean_text(value, repair_mojibake=repair_mojibake)
    if text is None:
        return None
    return normalize_apostrophe_case(text)
